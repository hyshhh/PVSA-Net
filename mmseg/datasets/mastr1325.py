import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MasDataset(BaseSegDataset):
    """YZ Segmentation Dataset for water, ground and object classes."""

    METAINFO = dict(
        # 三类：水、地面、物体
        classes=('obstacle', 'water', 'sky'),
        # 为每个类别分配颜色：水-蓝色，地面-绿色，物体-红色
        palette=[[246, 193,  59],  # water - blue
                 [ 51, 168, 222],  # ground - green
                 [ 89,  78, 161],  # object - red
]
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
            ignore_index=255,
            **kwargs)