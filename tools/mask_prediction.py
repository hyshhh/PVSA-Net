import os
import cv2
import numpy as np
from tqdm import tqdm
from mmseg.apis import init_model, inference_model

# ================= 配置区域 =================

# 1. 模型路径
config = "./configs_l/lformer/lformer-masTR-512x512.py"
checkpoint = "./mmseg_log/lformer/masTR1325/train_aug_2/best_mIoU_iter_46000.pth"
device = 'cuda:0'

# 2. 数据路径
mods_root = "./data/MODS/sequences"
# 直接保存到 RGB 文件夹，不再经过灰度中间层
save_root = "./mmseg_log/lformer/MODS/predictions_3/lformer" 

# 3. 【核心】颜色映射定义 (与您之前的代码完全一致)
# 顺序必须与模型训练时的类别顺序一致 (0: obstacle, 1: water, 2: sky)
# 格式：[R, G, B]
PALETTE_DICT = {
    0: [247, 195,  37],  # obstacle (Yellow-ish)
    1: [ 41, 167, 224],  # water (Blue)
    2: [ 90,  75, 164],  # sky (Purple-ish)
}

# 为了加速计算，我们将字典转换为 numpy 数组查找表 (Lookup Table)
# 假设最大类别 ID 是 2，我们创建一个大小为 (max_id + 1, 3) 的数组
max_cls_id = max(PALETTE_DICT.keys())
PALETTE_LUT = np.zeros((max_cls_id + 1, 3), dtype=np.uint8)
for cls_id, color in PALETTE_DICT.items():
    PALETTE_LUT[cls_id] = color

# ===========================================

def index_to_rgb_fast(mask_index, palette_lut):
    """
    利用 numpy 索引快速将单通道 mask 转为 RGB
    """
    mask_index = mask_index.astype(np.int32)
    
    # 防止模型预测出超出定义的类别 ID (例如背景噪声预测为 255)
    # 将其裁剪到最大合法类别 ID
    mask_index = np.clip(mask_index, 0, palette_lut.shape[0] - 1)
    
    # 核心操作：直接用 mask 的值作为索引去取颜色
    # 输入 (H, W) -> 输出 (H, W, 3)
    return palette_lut[mask_index]

def main():
    print(f"Initializing model: {os.path.basename(checkpoint)}...")
    model = init_model(config, checkpoint, device=device)
    
    sequences = sorted(os.listdir(mods_root))
    print(f"Found {len(sequences)} sequences.")

    for seq in tqdm(sequences, desc="Processing Sequences"):
        frame_dir = os.path.join(mods_root, seq, "frames")
        save_seq_dir = os.path.join(save_root, seq)

        if not os.path.exists(frame_dir):
            continue
            
        os.makedirs(save_seq_dir, exist_ok=True)

        # 获取图片列表
        img_list = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

        for img_name in tqdm(img_list, desc=seq, leave=False):
            img_path = os.path.join(frame_dir, img_name)
            
            # 读取原图获取尺寸
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            # 1. 推理
            result = inference_model(model, img)
            
            # 2. 提取 Mask
            if hasattr(result, 'pred_sem_seg'):
                mask_index = result.pred_sem_seg.data[0].cpu().numpy()
            elif hasattr(result, 'seg'):
                mask_index = result.seg[0].cpu().numpy()
            else:
                raise ValueError("Cannot extract mask from result")

            # 3. 尺寸对齐
            if mask_index.shape != (h, w):
                mask_index = cv2.resize(mask_index.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

            # 4. 转换为 RGB
            rgb_mask = index_to_rgb_fast(mask_index, PALETTE_LUT)

            # 5. 保存 (🔴 修复此处：将 name 改为 img_name)
            # 使用 os.path.splitext 更安全地处理后缀名
            file_name_no_ext = os.path.splitext(img_name)[0]
            save_filename = f"{file_name_no_ext}.png"
            
            save_path = os.path.join(save_seq_dir, save_filename)
            
            cv2.imwrite(save_path, cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))

    print(f"\nDone! RGB masks saved to: {save_root}")

if __name__ == "__main__":
    main()