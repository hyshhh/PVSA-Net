import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmseg.apis import init_model


############################################
# feature -> heatmap
############################################
def feature_to_heatmap(feat, mode="l2"):
    if isinstance(feat, torch.Tensor):
        feat = feat.detach().cpu().numpy()

    if feat.ndim == 4:
        feat = feat[0]

    if mode == "mean":
        feat = np.mean(feat, axis=0)
    elif mode == "max":
        feat = np.max(feat, axis=0)
    elif mode == "l2":
        feat = np.linalg.norm(feat, axis=0)

    # 去极值
    high = np.percentile(feat, 99)
    low = np.percentile(feat, 1)
    feat = np.clip(feat, low, high)

    # 归一化
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)

    # 平滑
    feat = cv2.GaussianBlur(feat, (7, 7), 0)

    return feat


############################################
# 🔥 标准上采样（替代 cv2.resize）
############################################
def upsample_to_original(feat, target_size):
    """
    feat: numpy [H,W]
    target_size: (H, W)
    """
    feat = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()  # 1,1,H,W

    feat = F.interpolate(
        feat,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )

    feat = feat.squeeze().cpu().numpy()
    return feat


############################################
# save heatmap
############################################
def save_heatmap(feat, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(feat, cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


############################################
# overlay
############################################
def save_overlay(feat, img_bgr, save_path):
    heatmap = (feat * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)

    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)


############################################
# 拼接图
############################################
def save_concat_image(img_bgr, feats, save_path):
    h, w = img_bgr.shape[:2]
    vis_list = [img_bgr]

    for feat in feats:
        feat = (feat * 255).astype(np.uint8)
        feat = cv2.applyColorMap(feat, cv2.COLORMAP_VIRIDIS)
        vis_list.append(feat)

    concat = np.concatenate(vis_list, axis=1)
    cv2.imwrite(save_path, concat)


############################################
# backbone feature
############################################
def extract_backbone_features(model, img_tensor):
    backbone = model.backbone

    if hasattr(backbone, "forward_features"):
        feats = backbone.forward_features(img_tensor)
    else:
        feats = backbone(img_tensor)

    return feats


############################################
# visualize one
############################################
def visualize_one(model, image_path, save_dir):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Failed: {image_path}")
        return

    original_h, original_w = img_bgr.shape[:2]

    # ⚠️ PyTorch size = (H, W)
    target_size = (original_h, original_w)

    # 模型输入
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))

    img_tensor = torch.from_numpy(img_resized).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.cuda()

    # forward
    with torch.no_grad():
        feats = extract_backbone_features(model, img_tensor)

    stage1, stage2, stage3, stage4 = feats

    # 转 heatmap
    stage1 = feature_to_heatmap(stage1)
    stage2 = feature_to_heatmap(stage2)
    stage3 = feature_to_heatmap(stage3)
    stage4 = feature_to_heatmap(stage4)

    # 🔥 上采样到原图尺寸
    stage1 = upsample_to_original(stage1, target_size)
    stage2 = upsample_to_original(stage2, target_size)
    stage3 = upsample_to_original(stage3, target_size)
    stage4 = upsample_to_original(stage4, target_size)

    os.makedirs(save_dir, exist_ok=True)

    # 保存 heatmap
    save_heatmap(stage1, os.path.join(save_dir, "stage1.png"))
    save_heatmap(stage2, os.path.join(save_dir, "stage2.png"))
    save_heatmap(stage3, os.path.join(save_dir, "stage3.png"))
    save_heatmap(stage4, os.path.join(save_dir, "stage4.png"))

    # overlay
    save_overlay(stage1, img_bgr, os.path.join(save_dir, "stage1_overlay.png"))
    save_overlay(stage2, img_bgr, os.path.join(save_dir, "stage2_overlay.png"))
    save_overlay(stage3, img_bgr, os.path.join(save_dir, "stage3_overlay.png"))
    save_overlay(stage4, img_bgr, os.path.join(save_dir, "stage4_overlay.png"))

    # 拼接
    save_concat_image(img_bgr, [stage1, stage2, stage3, stage4],
                      os.path.join(save_dir, "concat.png"))

    # 原图
    cv2.imwrite(os.path.join(save_dir, "original.jpg"), img_bgr)

    print(f"Saved → {save_dir}")


############################################
# main
############################################
if __name__ == "__main__":
    config = "./configs_l/lformer/lformer-gqy-256x256.py"
    checkpoint = "./mmseg_log/lformer/yz/train_pretrained/best_mIoU_iter_68000.pth"

    image_folder = "./data/gqyyz/image/test2"
    output_folder = "./mmseg_log/lformer/feature_backbone"

    model = init_model(config, checkpoint, device="cuda:0")
    model.eval()

    img_list = sorted(os.listdir(image_folder))

    for i, img_name in enumerate(img_list):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            continue

        print(f"[{i+1}] {img_name}")

        img_path = os.path.join(image_folder, img_name)
        save_dir = os.path.join(output_folder, os.path.splitext(img_name)[0])

        try:
            visualize_one(model, img_path, save_dir)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")