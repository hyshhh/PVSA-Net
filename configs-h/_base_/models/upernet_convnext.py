norm_cfg = dict(type='SyncBN', requires_grad=True)

custom_imports = dict(imports='mmpretrain.models', allow_failed_imports=False)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,

    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
    ),

    decode_head=dict(
        type='SegformerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,          # SegFormer decoder embedding dim
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)