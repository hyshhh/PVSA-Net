_base_ = [
    '../_base_/datasets/gqy.py',
    '../_base_/default_runtime.py',
]
crop_size = (256, 256)
norm_cfg = dict(type='BN', requires_grad=True)
class_weight = [
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
]

# 数据预处理
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=64,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)),
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDHead',
        in_channels=256,
        channels=128,
        num_classes=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# 训练配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=150000,
    val_interval=4000
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=150000,  # 直接使用数字，不要用max_iters变量
        by_epoch=False,
    )
]

train_dataloader = dict(
    # dataset=dict(
    #     # 只使用部分数据
    #     indices=6400),
    batch_size=16, num_workers=8
    )
val_dataloader = dict(batch_size=4, num_workers=8)
# 覆盖测试配置
test_dataloader = dict(batch_size=4, num_workers=8)

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
        max_keep_ckpts=5,
        save_optimizer=False,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))