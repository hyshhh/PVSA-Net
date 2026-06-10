import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# 原 ADE20K 风格的颜色 mask 目录
ANN_ROOT = "data/ade/ADEChallengeData2016/annotations"
OUT_ROOT = "data/ade/ADEChallengeData2016/annotations_converted"

os.makedirs(OUT_ROOT, exist_ok=True)

# 全局颜色 → 类别索引映射
color2id = {}
next_id = 0

def convert_mask(mask_path, save_path):
    global next_id

    mask = np.array(Image.open(mask_path))
    h, w, _ = mask.shape

    # 输出灰度 mask
    out = np.zeros((h, w), dtype=np.uint8)

    # 找到所有颜色组合
    colors = np.unique(mask.reshape(-1, 3), axis=0)

    for color in colors:
        c_tuple = tuple(color.tolist())
        if c_tuple not in color2id:
            color2id[c_tuple] = next_id
            next_id += 1

        cls_id = color2id[c_tuple]
        # 将对应颜色的位置赋值类别 ID
        out[(mask == color).all(axis=2)] = cls_id

    Image.fromarray(out).save(save_path)


def process_split(split):
    in_dir = os.path.join(ANN_ROOT, split)
    out_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.endswith((".png", ".jpg"))]

    print(f"Processing {split} ...")

    for name in tqdm(files):
        in_path = os.path.join(in_dir, name)
        out_path = os.path.join(out_dir, name.replace(".jpg", ".png"))
        convert_mask(in_path, out_path)


if __name__ == "__main__":
    process_split("training")
    process_split("validation")

    # 保存颜色映射文件（用于复现 & debug）
    with open(os.path.join(OUT_ROOT, "color2id.json"), "w") as f:
        json.dump({str(k): v for k, v in color2id.items()}, f, indent=2)

    print("转换完成！输出目录:", OUT_ROOT)
    print("共生成类别数:", len(color2id))
