from mmengine.config import Config
from mmseg.registry import TRANSFORMS
from mmseg.utils import register_all_modules
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import copy


def visualize_pipeline_steps():
    """可视化流水线的每个步骤"""

    # 注册所有模块
    register_all_modules()

    # 确保输出目录存在
    os.makedirs('../../mmseg_log/visualization_pipeline', exist_ok=True)

    # 加载配置
    cfg = Config.fromfile('../../configs/_base_/datasets/yz_seg.py')

    # 准备数据
    data = {
        'img_path': '../../data/YZ_Seg/images/training/00000.jpg',
        'seg_map_path': '../../data/YZ_Seg/annotations/training/00000.png',
        'seg_fields': [],
    }

    # 存储每个步骤的结果
    steps_data = []
    step_names = []

    # 逐步执行每个转换
    current_data = copy.deepcopy(data)
    steps_data.append(copy.deepcopy(current_data))
    step_names.append("Original")

    # 手动加载图像和标注，跳过有问题的 LoadAnnotations
    print("手动加载图像和标注...")
    try:
        # 手动加载图像
        img = cv2.imread(data['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        current_data['img'] = img
        current_data['img_shape'] = img.shape[:2]
        current_data['ori_shape'] = img.shape[:2]

        # 手动加载标注
        if os.path.exists(data['seg_map_path']):
            gt_seg_map = cv2.imread(data['seg_map_path'], cv2.IMREAD_GRAYSCALE)
            current_data['gt_seg_map'] = gt_seg_map
            print(f"手动加载成功 - 图像: {img.shape}, 标注: {gt_seg_map.shape}")
        else:
            print("标注文件不存在")

        steps_data.append(copy.deepcopy(current_data))
        step_names.append("ManualLoad")

    except Exception as e:
        print(f"手动加载失败: {e}")

    # 逐个应用剩余的转换
    for transform_cfg in cfg.train_pipeline:
        transform_name = transform_cfg['type']

        # 跳过已经手动处理的转换
        if transform_name in ['LoadImageFromFile', 'LoadAnnotations']:
            continue

        print(f"执行转换: {transform_name}")

        try:
            # 构建转换
            transform = TRANSFORMS.build(transform_cfg)

            # 应用转换
            current_data = transform(current_data)

            # 保存结果
            steps_data.append(copy.deepcopy(current_data))
            step_names.append(transform_name)

            print(f"  成功 - 数据键: {list(current_data.keys())}")

        except Exception as e:
            print(f"  转换失败: {e}")
            # 跳过这个转换，继续下一个
            continue

    # 可视化每个步骤
    visualize_results(steps_data, step_names)


def visualize_results(steps_data, step_names):
    """可视化结果"""
    if len(steps_data) <= 1:
        print("没有足够的步骤数据进行可视化")
        return

    fig, axes = plt.subplots(2, len(steps_data), figsize=(4 * len(steps_data), 8))

    # 如果只有一个步骤，调整 axes 的形状
    if len(steps_data) == 1:
        axes = axes.reshape(2, 1)

    for i, (step_data, step_name) in enumerate(zip(steps_data, step_names)):
        # 可视化图像
        visualize_image(axes[0, i], step_data, step_name, i)

        # 可视化标注
        visualize_annotation(axes[1, i], step_data, step_name, i)

    plt.tight_layout()
    plt.savefig('../../mmseg_log/visualization_pipeline/pipeline_steps.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("可视化完成！")


def visualize_image(ax, step_data, step_name, step_index):
    """可视化图像"""
    if step_index == 0:  # 原始数据
        if os.path.exists(step_data['img_path']):
            img = cv2.imread(step_data['img_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f'{step_name}\n{img.shape}')
        else:
            ax.set_title(f'{step_name}\nFile not found')
    elif 'img' in step_data:
        img = step_data['img']
        if len(img.shape) == 3 and img.shape[0] == 3:  # CHW -> HWC
            img = img.transpose(1, 2, 0)

        # 反归一化
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        ax.imshow(img)
        ax.set_title(f'{step_name}\n{img.shape}')
    else:
        ax.set_title(f'{step_name}\nNo Image')

    ax.axis('off')


def visualize_annotation(ax, step_data, step_name, step_index):
    """可视化标注"""
    if step_index == 0:  # 原始数据
        if os.path.exists(step_data['seg_map_path']):
            gt = cv2.imread(step_data['seg_map_path'], cv2.IMREAD_GRAYSCALE)
            ax.imshow(gt, cmap='tab20')
            ax.set_title(f'{step_name}\n{gt.shape}')
        else:
            ax.set_title(f'{step_name}\nNo GT')
    elif 'gt_seg_map' in step_data:
        gt = step_data['gt_seg_map']
        ax.imshow(gt, cmap='tab20')
        ax.set_title(f'{step_name}\n{gt.shape}')
    elif 'gt_semantic_seg' in step_data:
        gt = step_data['gt_semantic_seg']
        ax.imshow(gt, cmap='tab20')
        ax.set_title(f'{step_name}\n{gt.shape}')
    else:
        ax.set_title(f'{step_name}\nNo GT')

    ax.axis('off')


if __name__ == '__main__':
    visualize_pipeline_steps()