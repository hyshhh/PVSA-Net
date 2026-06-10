import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple

# 可配置项
images_dir = r"../../data/yz_segmentation/yz_segmentation_images"         # 源图像目录（所有图像）
anns_dir = r"../../data/yz_segmentation/yz_segmentation_annotation"       # 源标签目录（与图像同名或可对应）
out_root = r"./YZ_Seg"                        # 输出根目录（会创建 images/ annotations/ 子目录）
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
seed = 42
copy_not_move = True   # True=copy (保留原始文件)，False=move (移动文件)

# 支持的图片扩展名（会按这些后缀检索）
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def list_files(folder: str, exts=IMG_EXTS) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"路径不存在: {folder}")
    files = [f for f in sorted(p.iterdir()) if f.suffix.lower() in exts and f.is_file()]
    return files


def basename_no_ext(p: Path) -> str:
    return p.stem


def match_images_and_anns(img_files: List[Path], ann_files: List[Path]) -> List[Tuple[Path, Path]]:
    # 通过不带后缀的 basename 进行匹配
    ann_map = {basename_no_ext(a): a for a in ann_files}
    pairs = []
    missed_imgs = []
    for img in img_files:
        key = basename_no_ext(img)
        ann = ann_map.get(key)
        if ann:
            pairs.append((img, ann))
        else:
            missed_imgs.append(img)
    return pairs, missed_imgs


def ensure_dirs(root: str):
    images_out = Path(root) / "images"
    anns_out = Path(root) / "annotations"
    for sub in ["training", "validation", "test"]:
        (images_out / sub).mkdir(parents=True, exist_ok=True)
        (anns_out / sub).mkdir(parents=True, exist_ok=True)


def split_list(lst: List, train_ratio, val_ratio, test_ratio, seed=42):
    random.seed(seed)
    items = lst.copy()
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # 剩下给 test（避免因四舍五入导致总和不等）
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def transfer_file(src: Path, dst: Path, copy=True):
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def main():
    print("读取文件列表...")
    img_files = list_files(images_dir)
    ann_files = list_files(anns_dir)

    print(f"找到图像: {len(img_files)} 张，标签: {len(ann_files)} 张")
    pairs, missed = match_images_and_anns(img_files, ann_files)

    print(f"匹配成功对数: {len(pairs)}，未匹配的图像: {len(missed)}")
    if missed:
        print("未找到对应标签的图像（将被忽略）示例：")
        for m in missed[:10]:
            print("  ", m.name)

    if len(pairs) == 0:
        print("错误: 没有匹配到任何图像-标签对，请检查文件名是否一致（去掉后缀后应相同）。")
        return

    ensure_dirs(out_root)

    # 划分
    train_pairs, val_pairs, test_pairs = split_list(pairs, train_ratio, val_ratio, test_ratio, seed=seed)
    print(f"划分结果 -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    # 执行复制/移动
    for subset_name, subset_pairs in [('training', train_pairs), ('validation', val_pairs), ('test', test_pairs)]:
        print(f"处理 {subset_name} ({len(subset_pairs)}) ...")
        for img_path, ann_path in subset_pairs:
            img_dst = Path(out_root) / "images" / subset_name / img_path.name
            ann_dst = Path(out_root) / "annotations" / subset_name / ann_path.name
            transfer_file(img_path, img_dst, copy=copy_not_move)
            transfer_file(ann_path, ann_dst, copy=copy_not_move)

    print("完成！输出结构：")
    print(Path(out_root).resolve())
    print("images/ ->", sorted([p.name for p in (Path(out_root) / "images").iterdir()]))
    print("annotations/ ->", sorted([p.name for p in (Path(out_root) / "annotations").iterdir()]))

    # 报告未匹配的标签（标签多但图像缺失）
    img_map = {basename_no_ext(p) for p in img_files}
    orphan_anns = [a for a in ann_files if basename_no_ext(a) not in img_map]
    if orphan_anns:
        print(f"警告: 存在 {len(orphan_anns)} 个没有对应图像的标签（未处理）：")
        for a in orphan_anns[:10]:
            print("  ", a.name)


if __name__ == "__main__":
    main()