_base_ = [
    '../_base_/default_runtime.py', # '../_base_/schedules/schedule_160k.py',
    '../_base_/datasets/yz_seg.py'
]
# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (224, 224)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='VWHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        short_cut=True,
        nheads=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
train_dataloader = dict(batch_size=32)

# 训练配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,
    val_interval=4000
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]

train_dataloader = dict(
    dataset=dict(
        # 只使用部分数据
        indices=6400),
    batch_size=32, num_workers=12
    )
val_dataloader = dict(batch_size=16, num_workers=8)
# 覆盖测试配置
test_dataloader = dict(batch_size=16, num_workers=8)

# 随机种子
randomness = dict(seed=42)

# 验证集评估器
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    ignore_index=255,  # 避免把填充值算进指标
)

# 测试集评估器
test_evaluator = dict(
        type='IoUMetric',
        iou_metrics=['mIoU', 'mDice'],
        keep_results=True
    )

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,  # 按迭代次数保存
        interval=4000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3,
        save_optimizer=False,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
