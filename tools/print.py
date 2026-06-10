# from mmseg.apis import init_model
# from mmseg.utils import register_all_modules

# # 注册模块（很关键，不然有的自定义层找不到）
# register_all_modules()

# # 初始化模型
# model = init_model(
#     'configs/biformer/biformer_mm-20k_chase_db1-512x512.py',
#     'mmseg_convnexyz/train/iter_4000.pth',
#     device='cuda:0'
# )

# # 打印 backbone 主结构
# print("===== BACKBONE STRUCTURE =====")
# print(model.backbone)

# # 打印所有可访问的子模块名称（CAM 需要）
# print("\n===== ALL SUBMODULES IN BACKBONE =====")
# for name, module in model.backbone.named_modules():
#     print(name)

import cv2
import numpy as np

label_path = "/media/ddc/新加卷/hys/ljf/mmsegmentation-main/mmsegmentation-main/data/cityscapes_cleaned/annotations/train/zurich_000029_000019.png"

# 读取label
label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

# 获取所有唯一像素值
unique_values = np.unique(label)

print("Label中包含的类别ID:")
print(unique_values)