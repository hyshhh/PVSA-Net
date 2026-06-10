checkpoint_config = dict(by_epoch=True, interval=10)
crop_size = (
    224,
    224,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        224,
        224,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/gqyyz'
dataset_type = 'YZSegDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    1280,
    720,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        auto_pad=True,
        before_attn_dwconv=3,
        depth=[
            3,
            4,
            6,
            3,
        ],
        diff_routing=False,
        embed_dim=[
            64,
            128,
            256,
            512,
        ],
        head_dim=32,
        kv_downsample_mode='identity',
        kv_per_wins=[
            -1,
            -1,
            -1,
            -1,
        ],
        layer_scale_init_value=-1,
        mlp_ratios=[
            3,
            3,
            3,
            3,
        ],
        n_win=7,
        param_routing=False,
        pe=None,
        pre_norm=True,
        qk_dims=[
            64,
            128,
            256,
            512,
        ],
        side_dwconv=5,
        soft_routing=False,
        topks=[
            1,
            4,
            16,
            -2,
        ],
        type='BiFormer_NexConv1'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            224,
            224,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            256,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=3,
        type='SegformerHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0006, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(begin=0, by_epoch=True, end=10, start_factor=0.001, type='LinearLR'),
    dict(
        begin=10,
        by_epoch=True,
        end=200,
        eta_min=1e-06,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='image/test', seg_map_path='annotation/test'),
        data_root='data/gqyyz',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='YZSegDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    classwise=True,
    ignore_index=255,
    iou_metrics=[
        'mIoU',
        'mDice',
    ],
    type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_epochs=200, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='image/train', seg_map_path='annotation/train'),
        data_root='data/gqyyz',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=False,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    224,
                    224,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    224,
                    224,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='YZSegDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=False,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            224,
            224,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        224,
        224,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='image/val', seg_map_path='annotation/val'),
        data_root='data/gqyyz',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='YZSegDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    classwise=True,
    ignore_index=255,
    iou_metrics=[
        'mIoU',
        'mDice',
    ],
    type='IoUMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'mmseg_convnexyz/yzgqy_Nexconv1'
