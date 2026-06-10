import os
import cv2
import sys
import json
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 确保这些模块在 Python 路径中
from utils import read_gt_file, code_mask_to_labels, code_labels_to_colors
from configs import get_cfg

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def get_arguments():
    parser = argparse.ArgumentParser(description='MODS Debug Visualization')
    parser.add_argument("--methods", type=str, nargs='+', required=True, help="Method names")
    parser.add_argument("--sequence", type=int, required=True, help="Sequence ID (1-based)")
    parser.add_argument("--frame", type=int, required=True, help="Frame ID (1-based)")
    parser.add_argument("--config-file", type=str, default=None, help="Config file path")
    return parser.parse_args()

def main():
    args = get_arguments()
    cfg = get_cfg(args)

    print("\n" + "="*60)
    print(f"🔍 调试模式: Sequence={args.sequence}, Frame={args.frame}")
    print("="*60)

    # 1. 加载 GT JSON
    gt_path = os.path.join(cfg.PATHS.DATASET, 'modb.json')
    if not os.path.exists(gt_path):
        print(f"❌ GT 文件不存在: {gt_path}")
        return
    gt_data = read_gt_file(gt_path)

    seq_idx = args.sequence - 1
    if seq_idx >= len(gt_data['dataset']['sequences']):
        print(f"❌ 序列索引越界: {seq_idx}")
        return
    
    seq_info = gt_data['dataset']['sequences'][seq_idx]
    frames_list = seq_info['frames']

    # 确定帧索引 (原始代码直接用 args.frame 作为列表索引)
    frame_idx = args.frame
    if frame_idx >= len(frames_list):
        print(f"⚠️ 警告: 帧索引 {frame_idx} 超出 JSON 记录范围 (最大 {len(frames_list)-1})")
        # 尝试不崩溃，继续执行看效果
    
    # 2. 获取原图路径 (来自 JSON)
    try:
        gt_frame_info = frames_list[frame_idx]
        gt_filename = gt_frame_info['image_file_name']
        seq_rel_path = seq_info['path']
        # raw_img_full_path = os.path.join(cfg.PATHS.DATASET, seq_rel_path, gt_filename)
        
        # 🔧 修复核心：去除 path 开头的 '/' 或 '\'，防止 os.path.join 将其视为绝对路径
        # 如果 JSON 里的 path 是 "/kope102...", 这行代码会把它变成 "kope102..."
        seq_rel_path_clean = seq_rel_path.lstrip('/').lstrip('\\')
        
        # 现在拼接路径就会正确了：/your/dataset/root + kope102... + filename
        raw_img_full_path = os.path.join(cfg.PATHS.DATASET, seq_rel_path_clean, gt_filename)

        print(f"\n📷 [原图信息]")
        print(f"   JSON 文件名: {gt_filename}")
        print(f"   完整路径:    {raw_img_full_path}")
        
        if not os.path.exists(raw_img_full_path):
            print(f"   ❌ 文件不存在!")
            raw_img = None
        else:
            raw_img = cv2.imread(raw_img_full_path)
            if raw_img is None:
                print(f"   ❌ 读取失败!")
            else:
                raw_img = cv2.resize(raw_img, (cfg.DATASET.IMG_WIDTH, cfg.DATASET.IMG_HEIGHT))
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                print(f"   ✅ 加载成功, 形状: {raw_img.shape}")
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        raw_img = None

    # 3. 准备画布
    num_methods = len(args.methods)
    fig, axes = plt.subplots(1, num_methods + 1, figsize=(5 * (num_methods + 1), 5))
    if num_methods == 0:
        axes = [axes]
    elif not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 绘制原图
    ax_raw = axes[0]
    if raw_img is not None:
        ax_raw.imshow(raw_img)
        ax_raw.set_title(f"Raw Image (JSON)\nFile: {gt_filename}", fontsize=10)
    else:
        ax_raw.text(0.5, 0.5, "Raw Image\nLoad Failed", ha='center', va='center', color='red')
        ax_raw.set_title("Raw Image (Error)", fontsize=10)
    ax_raw.axis('off')

    # 4. 循环处理每个方法
    for i, method in enumerate(args.methods):
        ax = axes[i+1]
        print(f"\n🎨 [方法 {i+1}] {method}")
        
        # --- 核心调试逻辑：构建预测图路径 ---
        # 假设你的重命名规则: Frame 1 -> 0010.png, Frame 2 -> 0011.png
        START_ID = 10
        calc_file_id = START_ID + (args.frame - 1)
        pred_filename = f"{calc_file_id:04d}.png"
        seq_folder = f"seq{args.sequence:02d}"
        
        if cfg.SEGMENTATIONS.SEQ_FIRST:
            pred_path = os.path.join(cfg.PATHS.SEGMENTATIONS, seq_folder, method, pred_filename)
        else:
            pred_path = os.path.join(cfg.PATHS.SEGMENTATIONS, method, seq_folder, pred_filename)
            
        print(f"   计算逻辑: {START_ID} + ({args.frame} - 1) = {calc_file_id}")
        print(f"   目标文件: {pred_filename}")
        print(f"   完整路径: {pred_path}")
        
        seg_overlay = None
        if not os.path.exists(pred_path):
            print(f"   ❌ 预测图文件不存在!")
            ax.text(0.5, 0.5, f"Mask Not Found\n{pred_filename}", ha='center', va='center', color='red', fontsize=12)
        else:
            seg_mask = cv2.imread(pred_path)
            if seg_mask is None:
                print(f"   ❌ 读取失败!")
                ax.text(0.5, 0.5, "Read Error", ha='center', va='center', color='red')
            else:
                print(f"   ✅ 加载成功, 形状: {seg_mask.shape}")
                try:
                    # 处理掩码
                    seg_labels = code_mask_to_labels(seg_mask, cfg.SEGMENTATIONS.INPUT_COLORS)
                    seg_colors = code_labels_to_colors(seg_labels, cfg)
                    seg_resized = cv2.resize(seg_colors, (cfg.DATASET.IMG_WIDTH, cfg.DATASET.IMG_HEIGHT))
                    
                    # 融合 (如果原图加载成功)
                    if raw_img is not None:
                        seg_overlay = cv2.addWeighted(raw_img, 0.4, seg_resized, 0.6, 0)
                        ax.imshow(seg_overlay)
                    else:
                        ax.imshow(cv2.cvtColor(seg_resized, cv2.COLOR_BGR2RGB))
                    
                    # 在图上打印文件名以便视觉确认
                    ax.text(10, 30, f"Pred: {pred_filename}", color='yellow', fontsize=12, 
                            bbox=dict(facecolor='black', alpha=0.7))
                    if raw_img is not None:
                        ax.text(10, 60, f"Raw: {gt_filename}", color='cyan', fontsize=10,
                                bbox=dict(facecolor='black', alpha=0.5))
                        
                except Exception as e:
                    print(f"   ⚠️ 处理出错: {e}")
                    ax.text(0.5, 0.5, f"Process Error\n{str(e)}", ha='center', va='center', color='red')
        
        ax.set_title(f"Method: {method}", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    out_name = f"debug_seq{args.sequence}_frame{args.frame}.png"
    plt.savefig(out_name, dpi=150)
    print(f"\n💾 调试图已保存: {out_name}")
    plt.show()

if __name__ == '__main__':
    main()