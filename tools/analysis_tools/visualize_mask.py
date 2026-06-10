import os
import cv2
import numpy as np
from pathlib import Path


# =====================================================
# Cityscapes 19类颜色映射
# =====================================================
CITYSCAPES_COLORS = {
    0: (128, 64, 128),     # road
    1: (244, 35, 232),     # sidewalk
    2: (70, 70, 70),       # building
    3: (102, 102, 156),    # wall
    4: (190, 153, 153),    # fence
    5: (153, 153, 153),    # pole
    6: (250, 170, 30),     # traffic light
    7: (220, 220, 0),      # traffic sign
    8: (107, 142, 35),     # vegetation
    9: (152, 251, 152),    # terrain
    10: (70, 130, 180),    # sky
    11: (220, 20, 60),     # person
    12: (255, 0, 0),       # rider
    13: (0, 0, 142),       # car
    14: (0, 0, 70),        # truck
    15: (0, 60, 100),      # bus
    16: (0, 80, 100),      # train
    17: (0, 0, 230),       # motorcycle
    18: (119, 11, 32)      # bicycle
}


# =====================================================
# CamVid 11类颜色映射
# =====================================================
CAMVID_COLORS = {
    0: (128, 128, 128),    # Sky
    1: (128, 0, 0),        # Building
    2: (192, 192, 128),    # Pole
    3: (128, 64, 128),     # Road
    4: (60, 40, 222),      # Pavement
    5: (128, 128, 0),      # Tree
    6: (192, 128, 128),    # SignSymbol
    7: (64, 64, 128),      # Fence
    8: (64, 0, 128),       # Car
    9: (64, 64, 0),        # Pedestrian
    10: (0, 128, 192)      # Bicyclist
}


def mask_to_color(mask, dataset="camvid"):
    """
    将单通道标签图转换为RGB彩色图
    """

    if dataset.lower() == "cityscapes":
        color_map = CITYSCAPES_COLORS
    else:
        color_map = CAMVID_COLORS

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in color_map.items():
        color_mask[mask == label] = color

    return color_mask


def visualize_sample(
        image_path,
        mask_path,
        save_dir,
        dataset="camvid",
        alpha=0.5):

    os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, 0)

    print("=" * 50)
    print(mask_path)
    print("shape:", mask.shape)
    print("dtype:", mask.dtype)
    print("unique values:", np.unique(mask)[:50])
    print("max value:", np.max(mask))
    print("=" * 50)

    color_mask = mask_to_color(mask, dataset)

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        color_mask,
        alpha,
        0
    )

    stem = Path(image_path).stem

    # cv2.imwrite(
    #     os.path.join(save_dir, f"{stem}_image.png"),
    #     cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # )

    cv2.imwrite(
        os.path.join(save_dir, f"{stem}_mask.png"),
        cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    )

    # cv2.imwrite(
    #     os.path.join(save_dir, f"{stem}_overlay.png"),
    #     cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    # )

    print(f"Saved: {stem}")


if __name__ == "__main__":

    # ==================================
    # 修改为自己的路径
    # ==================================

    image_dir = "./data/cityscapes_cleaned/images/val"
    mask_dir = "./data/cityscapes_cleaned/annotations/val"

    save_dir = "./data/cityscapes_cleaned/visualization"

    for img_name in os.listdir(image_dir):

        image_path = os.path.join(image_dir, img_name)

        mask_name = img_name

        mask_path = os.path.join(mask_dir, mask_name)

        visualize_sample(
            image_path,
            mask_path,
            save_dir,
            dataset="cityscapes"
        )