import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmseg.apis import init_model

############################################
# feature -> heatmap
############################################
def feature_to_heatmap(feat):
    """
    将特征 tensor 转换为归一化的 numpy 数组 (0-1)
    """
    if isinstance(feat, torch.Tensor):
        feat = feat.detach().cpu().numpy()
    
    # 如果是 batch (B, C, H, W)，取第一个样本并沿通道平均
    if feat.ndim == 4:
        feat = feat[0]
        feat = np.mean(feat, axis=0)
    elif feat.ndim == 3:
        feat = np.mean(feat, axis=0)

    # 归一化到 0-1
    min_val = feat.min()
    max_val = feat.max()
    if max_val - min_val > 1e-6:
        feat = (feat - min_val) / (max_val - min_val)
    else:
        feat = np.zeros_like(feat)

    return feat


############################################
# save heatmap (上采样至指定尺寸)
############################################
def save_heatmap(feat, save_path, target_size=None):
    """
    保存 heatmap。
    :param feat: 2D numpy array
    :param save_path: 保存路径
    :param target_size: tuple (width, height)。如果指定，则将 heatmap resize 到此尺寸。
    """
    if feat is None:
        return
        
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)

    # 如果需要上采样/下采样到目标尺寸
    if target_size is not None:
        h, w = feat.shape
        target_w, target_h = target_size
        
        # 只有当当前尺寸不等于目标尺寸时才 resize
        if h != target_h or w != target_w:
            # 使用 cv2.INTER_CUBIC 进行高质量重采样
            feat = cv2.resize(feat, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    plt.figure(figsize=(8, 6)) # 稍微调大画布以适应可能的大图
    plt.imshow(feat, cmap="viridis", aspect="auto")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


############################################
# feature storage
############################################
features = {}

############################################
# hooks
############################################
def hook_metaformer_input(module, input):
    features["meta_input"] = input[0]

def hook_local(module, input, output):
    features["local"] = output

def hook_global(module, input, output):
    features["global"] = output

def hook_spatial(module, input, output):
    features["spatial"] = output

def hook_metaformer_output(module, input, output):
    features["meta_output"] = output


############################################
# register hooks
############################################
def register_hooks(model):
    if hasattr(model.decode_head, 'metaformer'):
        meta = model.decode_head.metaformer
        
        meta.register_forward_pre_hook(hook_metaformer_input)
        
        if hasattr(meta, 'dwconv'):
            meta.dwconv.register_forward_hook(hook_local)
        if hasattr(meta, 'lck'):
            meta.lck.register_forward_hook(hook_global)
        if hasattr(meta, 'spatial_interaction'):
            meta.spatial_interaction.register_forward_hook(hook_spatial)
            
        meta.register_forward_hook(hook_metaformer_output)
    else:
        print("Warning: Could not find 'metaformer' in decode_head.")


############################################
# visualize one image
############################################
def visualize_one(model, image_path, save_dir):
    # 1. 读取原始图像
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Failed to load image: {image_path}")
        return

    # 【关键步骤 A】获取数据集原始图片的尺寸 (Width, Height)
    original_h, original_w = img_bgr.shape[:2]
    original_size = (original_w, original_h)
    
    # 2. 预处理：Resize 到模型需要的输入尺寸
    # 这里保留您原有的逻辑，模型必须吃固定尺寸的输入
    MODEL_INPUT_W, MODEL_INPUT_H = 512, 512 
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (MODEL_INPUT_W, MODEL_INPUT_H))

    # 3. 转为 Tensor
    img_tensor = torch.from_numpy(img_resized).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.cuda()

    # 清空之前的特征
    features.clear()

    # 4. 前向传播
    with torch.no_grad():
        model(img_tensor)

    # 5. 转换特征为 heatmap
    # 此时这些特征图的尺寸是模型内部尺寸 (如 1/4, 1/8 等)
    meta_input = feature_to_heatmap(features.get("meta_input"))
    local = feature_to_heatmap(features.get("local"))
    global_ = feature_to_heatmap(features.get("global"))
    spatial = feature_to_heatmap(features.get("spatial"))
    meta_output = feature_to_heatmap(features.get("meta_output"))

    os.makedirs(save_dir, exist_ok=True)

    # 【关键步骤 B】保存时，强制上采样回 original_size
    # 这样输出的图片尺寸将与 data/gqyyz/image/test 中的原图完全一致
    print(f"  -> Upsampling features from {meta_input.shape} to Original Size: {original_size}")
    
    save_heatmap(meta_input, os.path.join(save_dir, "metaformer_input.png"), target_size=original_size)
    save_heatmap(local, os.path.join(save_dir, "local_feature.png"), target_size=original_size)
    save_heatmap(global_, os.path.join(save_dir, "global_feature.png"), target_size=original_size)
    save_heatmap(spatial, os.path.join(save_dir, "spatial_attention.png"), target_size=original_size)
    save_heatmap(meta_output, os.path.join(save_dir, "metaformer_output.png"), target_size=original_size)
    
    # 可选：保存一份原图的副本在该目录下，方便您直接对比查看
    cv2.imwrite(os.path.join(save_dir, "original_image_reference.jpg"), img_bgr)


############################################
# main
############################################
if __name__ == "__main__":
    config = "./configs_l/lformer/lformer-gqy-256x256.py"
    checkpoint = "./mmseg_log/lformer/yz/train_pretrained/best_mIoU_iter_68000.pth"

    image_folder = "./data/gqyyz/image/test2"
    output_folder = "./mmseg_log/lformer/feature_output_pretrained" 

    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found at {checkpoint}")
        exit()
        
    model = init_model(config, checkpoint, device="cuda:0")
    model.eval()

    register_hooks(model)

    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        exit()

    img_list = sorted(os.listdir(image_folder))

    count = 0
    for img_name in img_list:
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            continue

        img_path = os.path.join(image_folder, img_name)
        save_dir = os.path.join(output_folder, os.path.splitext(img_name)[0])

        print(f"Processing ({count+1}): {img_name}")
        try:
            visualize_one(model, img_path, save_dir)
            count += 1
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll done. Processed {count} images.")
    print(f"All heatmaps are resized to match their corresponding ORIGINAL image dimensions.")
    print(f"Results saved to: {os.path.abspath(output_folder)}")