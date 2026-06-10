import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

from mmseg.apis import init_model, inference_model

try:
    from mmseg.structures import SegDataSample
except ImportError:
    SegDataSample = None


# ======================== 配置 ========================
CONFIG_FILE = "./configs_l/lformer/lformer-masTR-512x512.py"
CHECKPOINT_FILE = "./mmseg_log/lformer/masTR1325/train_aug_2/best_mIoU_iter_46000.pth"

DATASET_ROOT = "../data/data/modd2_video/video_data"
OUTPUT_ROOT = "./mmseg_log/lformer/MODD2_2"
METHOD_NAME = "lformer"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# =====================================================


# ======================== 颜色定义 ========================
# 模型输出: 0=Obstacle, 1=Water, 2=Sky
MODEL_PALETTE = {
    0: [247, 195, 37],   # Obstacle
    1: [41, 167, 224],   # Water
    2: [90, 75, 164]     # Sky
}

# MODD2标准: [Sky; Obstacle; Water]
STANDARD_COLOR_MATRIX = np.array([
    MODEL_PALETTE[2],
    MODEL_PALETTE[0],
    MODEL_PALETTE[1]
], dtype=np.uint8)
# =====================================================


def extract_prediction(result):
    """兼容所有 MMSeg 输出格式"""

    pred = None

    if SegDataSample and isinstance(result, SegDataSample):
        pred = result.pred_sem_seg.data

    elif isinstance(result, list):
        item = result[0]
        if SegDataSample and isinstance(item, SegDataSample):
            pred = item.pred_sem_seg.data
        elif isinstance(item, dict):
            pred = item.get("pred_sem_seg", item.get("seg"))

    elif isinstance(result, dict):
        pred = result.get("pred_sem_seg", result.get("seg"))

    elif torch.is_tensor(result):
        pred = result

    if pred is None:
        raise ValueError(f"无法解析输出类型: {type(result)}")

    pred = pred.cpu().numpy() if hasattr(pred, "cpu") else np.array(pred)

    # shape 处理
    if pred.ndim == 3:
        if pred.shape[0] == 1:
            pred = pred[0]
        else:
            pred = np.argmax(pred, axis=0)

    return pred.astype(np.uint8)


def convert_to_rgb_mask(pred_label):
    """标签 -> MODD2 RGB 掩码"""

    h, w = pred_label.shape
    new_map = np.zeros((h, w), dtype=np.uint8)

    # label mapping
    new_map[pred_label == 2] = 0  # Sky
    new_map[pred_label == 0] = 1  # Obstacle
    new_map[pred_label == 1] = 2  # Water

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(3):
        rgb[new_map == i] = STANDARD_COLOR_MATRIX[i]

    return rgb


def process_sequence(model, seq_name):
    """处理单个序列"""

    seq_path = os.path.join(DATASET_ROOT, seq_name)
    img_dir = os.path.join(seq_path, "framesRectified")

    if not os.path.exists(img_dir):
        return

    save_dir = os.path.join(OUTPUT_ROOT, seq_name, METHOD_NAME)
    os.makedirs(save_dir, exist_ok=True)

    # 更稳健的筛选（只要左目图像）
    images = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg")) and "L" in f
    ])

    if len(images) == 0:
        return

    for img_name in tqdm(images, leave=False, desc=seq_name):
        img_path = os.path.join(img_dir, img_name)

        try:
            result = inference_model(model, img_path)

            pred = extract_prediction(result)

            rgb_mask = convert_to_rgb_mask(pred)

            save_path = os.path.join(
                save_dir,
                os.path.splitext(img_name)[0] + ".png"
            )

            # RGB -> BGR
            cv2.imwrite(save_path, cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"❌ {img_name} 失败: {e}")


def main():
    print("🚀 初始化模型...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

    sequences = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    ])

    print(f"📂 共 {len(sequences)} 个序列")

    for seq in tqdm(sequences, desc="Processing"):
        process_sequence(model, seq)

    print("\n✅ 全部完成！")
    print(f"📁 输出目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()