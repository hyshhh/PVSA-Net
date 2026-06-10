import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class YZSegDataset_copy(BaseSegDataset):
    """YZ Segmentation Dataset for water, ground and object classes."""

    METAINFO = dict(
        # 三类：水、地面、物体
        classes=('water', 'ground', 'object'),
        # 为每个类别分配颜色：水-蓝色，地面-绿色，物体-红色
        palette=[[255, 255, 255],  # ground - green
                 [0, 127, 135],  # water - blue
                 [136, 11, 86]]  # object - red
    )

    def __init__(self,
                 img_suffix='.jpg',  # 根据您的实际图像格式调整
                 seg_map_suffix='.png',  # 根据您的实际标注格式调整
                 reduce_zero_label=False,  # 重要：新数据集可能不需要减少零标签
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)