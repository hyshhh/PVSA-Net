import os
import cv2
import sys
import json
import shutil
import argparse
import matplotlib
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import read_gt_file, code_mask_to_labels, code_labels_to_colors, resize_image
from visualization import visualize_single_image, visualize_image_for_video
from configs import get_cfg

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42


def get_arguments():
    """ Parse all the arguments provided from the CLI
    Returns: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark.')
    parser.add_argument("--methods", type=str, nargs='+', required=True,
                        help="<Required> First method name. This should be equal to the folder name in which the "
                             "segmentation masks are located.")
    parser.add_argument("--show-overlap-perc", type=bool, default=False,
                        help="Show overlapping percentage next to each detection")
    parser.add_argument("--sequence", type=int,
                        help="Sequence for visualization.")
    parser.add_argument("--frame", type=int,
                        help="Frame for visualization")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Config file to use. If not specified, the default config is used.")

    return parser.parse_args()


def main():
    args = get_arguments()
    cfg  = get_cfg(args)

    # Get number of methods
    num_methods = len(args.methods)

    # Load ground truth
    gt = read_gt_file(os.path.join(cfg.PATHS.DATASET, 'modb.json'))

    # Load image
    seq_path = gt['dataset']['sequences'][args.sequence - 1]['path']
    img = cv2.imread(os.path.join(cfg.PATHS.DATASET + seq_path +
                     gt['dataset']['sequences'][args.sequence - 1]['frames'][args.frame]['image_file_name']))
    img = cv2.resize(img, (cfg.DATASET.IMG_WIDTH, cfg.DATASET.IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load segmentation output for each method to compare
    methods_seg_masks = []
    results_seg = []
    for i in range(num_methods):
        tmp_method_name_string = 'method_%01d' % i

        # Get and append results
        # Load results
        with open(os.path.join(cfg.PATHS.RESULTS, 'results_%s.json' % args.methods[i])) as f:
            tmp_results = json.load(f)
            results_seg.append({tmp_method_name_string: tmp_results})

        # Get and append segmentation masks
        tmp_seg_mask = load_segmentation_mask(cfg.PATHS.SEGMENTATIONS, cfg.SEGMENTATIONS.INPUT_COLORS, args.sequence,
                                              args.methods[i], args.frame, img, cfg)
        methods_seg_masks.append({tmp_method_name_string: tmp_seg_mask})

    fig = plt.figure(1, figsize=(10, 5))
    fig.clf()
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.05)

    # Plot raw image
    ax = fig.add_subplot(1, num_methods+1, 1)
    plt.title("Raw image")
    ax.imshow(img)
    ax.axis('off')

    # Update of gt parameter for easier access
    gt = gt['dataset']['sequences'][args.sequence - 1]['frames'][args.frame]

    for i in range(num_methods):
        # Plot segmentation mask
        ax = fig.add_subplot(1, num_methods+1, i+2)
        plt.title(args.methods[i])
        ax.imshow(methods_seg_masks[i]['method_%01d' % i])
        ax.axis('off')

        # Get number of water edge lines
        num_danger_lines = len(gt['water_edges'])

        # Plot water-edge danger lines
        for j in range(num_danger_lines):
            tmp_danger_line_x = gt['water_edges'][j]['x_axis']
            tmp_danger_line_y = gt['water_edges'][j]['y_axis']
            #ax.plot(tmp_danger_line_x, tmp_danger_line_y, marker='', color='black', linewidth=3, linestyle='solid')
            ax.plot(tmp_danger_line_x, tmp_danger_line_y, marker='', color='purple', linewidth=1, linestyle='dashed')
            # plt.text(tmp_danger_line_x[0], tmp_danger_line_y[0] - 2, 'water_edge-%d' % i, fontsize=6)

        # Plot detection rectangles
        plot_detection_rectangles(ax, results_seg[i]['method_%01d' % i], 'tp_list', args.sequence - 1, args.frame, args.show_overlap_perc)  # Plot TPs
        plot_detection_rectangles(ax, results_seg[i]['method_%01d' % i], 'fp_list', args.sequence - 1, args.frame, args.show_overlap_perc)  # Plot FPs
        plot_detection_rectangles(ax, results_seg[i]['method_%01d' % i], 'fn_list', args.sequence - 1, args.frame, args.show_overlap_perc)  # Plot FNs

    plt.show()
    save_path = f"qualitative_seq{args.sequence}_frame{args.frame}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Saved to:", save_path)

# Plot detection rectangles
def plot_detection_rectangles(ax, results, list_name, sequence, frame, show_overlap_percentages):
    if list_name == 'tp_list':
        edge_color = 'green'
    elif list_name == 'fn_list':
        edge_color = 'red'
    else:
        edge_color = 'yellow'

    detection_type = 'obstacles'

    results_detection = results['sequences'][sequence]['frames'][frame]

    num_dets = len(results_detection[detection_type][list_name])
    for i in range(num_dets):
        tmp_bbox = results_detection[detection_type][list_name][i]['bbox']
        if edge_color is not 'yellow':
            rect_fg = patches.Rectangle((tmp_bbox[0], tmp_bbox[1]), tmp_bbox[2] - tmp_bbox[0],
                                        tmp_bbox[3] - tmp_bbox[1],
                                        linewidth=1, edgecolor='black', facecolor=edge_color, alpha=0.45)

            if show_overlap_percentages:
                ax.text(tmp_bbox[0], tmp_bbox[1], results_detection[detection_type][list_name][i]['type'] +
                        '-%d%%' % results_detection[detection_type][list_name][i]['coverage'], fontsize=6)
        else:
            rect_fg = patches.Rectangle((tmp_bbox[0], tmp_bbox[1]), tmp_bbox[2] - tmp_bbox[0],
                                        tmp_bbox[3] - tmp_bbox[1],
                                        linewidth=1, edgecolor='black', facecolor=edge_color, alpha=0.45)

            plt.text(tmp_bbox[0], tmp_bbox[1],
                     'FP (%d)' % results_detection[detection_type][list_name][i]['num_triggers'], fontsize=6)

        ax.add_patch(rect_fg)

    return ax


def load_segmentation_mask(segmentation_path, segmentation_colors, seq_id, method, frame, img, cfg):
    # 计算文件名对应的帧号 (根据你的重命名逻辑，frame 1 -> 0010)
    # 注意：如果传入的 frame 已经是 10 (对应 0010.png)，这里就不需要 *10
    # 假设命令行传入 --frame 1，且你的文件是 0010.png，则保留 *10
    file_frame_id = frame * 10 
    
    # ✅ 修复：先构建包含序列号的文件夹名称
    seq_folder_name = "seq{:02d}".format(seq_id)  # 将 21 变成 "seq21"
    file_name = "{:04d}.png".format(file_frame_id) # 将 10 变成 "0010.png"

    if cfg.SEGMENTATIONS.SEQ_FIRST:
        # 结构：.../seq21/lformer/0010.png
        full_path = os.path.join(segmentation_path, seq_folder_name, method, file_name)
    else:
        # 结构：.../lformer/seq21/0010.png
        full_path = os.path.join(segmentation_path, method, seq_folder_name, file_name)

    # 调试打印：确认路径是否正确
    # print(f"Trying to load: {full_path}") 

    seg = cv2.imread(full_path)

    if seg is None:
        print(f"❌ 错误：无法读取图片，请检查路径是否存在：{full_path}")
        # 可以选择返回 None 或者抛出更明确的错误，防止后续 shape 报错
        return None 

    # Code mask to labels
    # 只有当 seg 不为 None 时才执行 shape 操作
    try:
        print(f"成功加载图片，形状：{seg.shape}")
        seg = code_mask_to_labels(seg, segmentation_colors)
        # Code labels to colors
        seg = code_labels_to_colors(seg, cfg)
        seg = cv2.resize(seg, (cfg.DATASET.IMG_WIDTH, cfg.DATASET.IMG_HEIGHT))
        added_image = cv2.addWeighted(img, 0.4, seg, 0.6, 0)
        return added_image
    except AttributeError:
        print("⚠️ 图片加载失败，跳过后续处理。")
        return None


if __name__ == '__main__':
    main()
