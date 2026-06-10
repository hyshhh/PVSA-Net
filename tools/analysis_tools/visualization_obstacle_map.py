import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

# ========= 0. 配置区域 =========
# 修改为包含图片的文件夹路径
INPUT_FOLDER = 'data/gqyyz/image/test2' 
# 结果保存根目录
OUTPUT_ROOT = 'mmseg_log/lformer/obstacle_maps_2'
# 支持的图片格式
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# ========= 1. 初始化 (逻辑不变) =========
register_all_modules()

cfg = Config.fromfile(
    'configs_l/lformer/lformer-gqy-256x256.py'
)
cfg.model.pretrained = None
cfg.model.train_cfg = None
cfg.model.test_cfg = dict(mode='whole')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MODELS.build(cfg.model)
checkpoint = 'mmseg_log/lformer/yz/train_onlyQ/best_mIoU_iter_150000.pth'
load_checkpoint(model, checkpoint, map_location='cpu')
model.to(device)
model.eval()

print(f"[INFO] Model loaded on {device}. Starting batch inference...")

# 创建输出目录
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 获取所有图片文件
input_path = Path(INPUT_FOLDER)
image_files = [f for f in input_path.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
image_files.sort() # 排序以保证处理顺序一致

if not image_files:
    print(f"[ERROR] No images found in {INPUT_FOLDER}")
    exit(0)

print(f"[INFO] Found {len(image_files)} images.")

# ========= 2. 批量推理循环 =========
for idx, img_path_obj in enumerate(image_files):
    img_path = str(img_path_obj)
    file_name = img_path_obj.stem  # 不含扩展名的文件名
    
    print(f"[{idx+1}/{len(image_files)}] Processing: {img_path}")

    try:
        # ========= 2. 读取图片 (逻辑不变) =========
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [WARN] Failed to read image, skipping: {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ========= 3. 简单预处理 (逻辑不变) =========
        # 注意：这里保留了你的原始逻辑 .float()，这能确保是 float32，避免类型报错
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        # ========= 4. 前向推理 (逻辑不变) =========
        with torch.no_grad():
            feats = model.backbone(img_tensor)
            if model.with_neck:
                feats = model.neck(feats)

            feat = model.decode_head.forward_seghead(feats)
            obstacle_logits = model.decode_head.obstacle_head(feat)
            obstacle_prob = torch.sigmoid(obstacle_logits)[0, 0].cpu().numpy()

        # ========= 5. resize 到原图 (逻辑不变) =========
        obstacle_prob = cv2.resize(
            obstacle_prob,
            (img_rgb.shape[1], img_rgb.shape[0])
        )

        # ========= 6. 可视化 (逻辑不变) =========
        heatmap = cv2.applyColorMap(
            (obstacle_prob * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

        # ========= 7. 显示 & 保存 (修改为自动命名) =========
        # 为每张图生成独立的文件名
        save_name = f"obstacle_map_{file_name}.png"
        save_path = os.path.join(OUTPUT_ROOT, save_name)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img_rgb)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Obstacle Probability Map')
        plt.imshow(obstacle_prob, cmap='jet')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Overlay')
        plt.imshow(overlay)
        plt.axis('off')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # 重要：关闭图形以释放内存

        print(f"  [OK] Saved to {save_path}")

    except Exception as e:
        print(f"  [ERROR] Failed processing {img_path}: {e}")
        continue
    finally:
        # 如果是 GPU，每张图处理后清理缓存，防止显存溢出
        if device == 'cuda':
            torch.cuda.empty_cache()

print(f"\n[SUCCESS] All done! Results saved in '{OUTPUT_ROOT}'")