import os
import cv2
from tqdm import tqdm


############################################
# 路径配置
############################################

image_dir = "data/gqyyz/image/test2"
feature_root = "mmseg_log/lformer/feature_output"
output_root = "mmseg_log/lformer/feature_output_upto"

os.makedirs(output_root, exist_ok=True)


############################################
# 获取图片列表
############################################

image_list = sorted(os.listdir(image_dir))

print("Total images:", len(image_list))


############################################
# 遍历图片
############################################

for img_name in tqdm(image_list):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        continue

    name = os.path.splitext(img_name)[0]

    img_path = os.path.join(image_dir, img_name)
    feature_dir = os.path.join(feature_root, name)

    # 检查feature文件夹
    if not os.path.isdir(feature_dir):
        print("Missing feature folder:", name)
        continue

    ############################################
    # 读取原始图像尺寸
    ############################################

    img = cv2.imread(img_path)

    if img is None:
        print("Image read failed:", img_path)
        continue

    h, w = img.shape[:2]

    ############################################
    # 创建输出目录
    ############################################

    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    ############################################
    # 遍历特征图
    ############################################

    feat_list = sorted(os.listdir(feature_dir))

    for feat_name in feat_list:

        if not feat_name.lower().endswith(".png"):
            continue

        feat_path = os.path.join(feature_dir, feat_name)

        feat = cv2.imread(feat_path)

        if feat is None:
            print("Feature read failed:", feat_path)
            continue

        ############################################
        # 上采样
        ############################################

        feat_resized = cv2.resize(
            feat,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        save_path = os.path.join(out_dir, feat_name)

        cv2.imwrite(save_path, feat_resized)


print("\nFinished resizing all feature maps.")