import mmengine.fileio as fileio
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class camvid(BaseSegDataset):
    """YZ Segmentation Dataset (CamVid 12 classes)."""

    METAINFO = dict(
        classes=(
            'Sky', 'Building', 'Pole', 'Road', 'Sidewalk',
            'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian',
            'Bicyclist', 'unlabelled'
        ),

        # Palette 顺序必须与 classes 完全一致
        palette=[
            [128, 128, 128],   # Sky
            [128, 0, 0],       # Building
            [192, 192, 128],   # Pole
            [128, 64, 128],    # Road
            [0, 0, 192],       # Sidewalk
            [128, 128, 0],     # Tree
            [192, 128, 128],   # SignSymbol
            [64, 64, 128],     # Fence
            [64, 0, 128],      # Car
            [64, 64, 0],       # Pedestrian
            [0, 128, 192],     # Bicyclist
            [0, 0, 0],         # unlabelled
        ]
    )

    def __init__(self,
                 img_suffix='.png',      # CamVid 是 PNG
                 seg_map_suffix='.png',  # CamVid 标签也是 PNG
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
