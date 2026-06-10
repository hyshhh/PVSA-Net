_base_ = [
    '../_base_/models/biformer_mm.py',
    '../_base_/datasets/yz_seg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# --------------------------
# 数据预处理配置
# --------------------------
crop_size = (720, 720)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

# --------------------------
# 模型额外配置（仅修改部分）
# --------------------------
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole')
)

# --------------------------
# 优化器与学习率
# --------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0)
        })
)

# 学习率调度
param_scheduler = [
    #warm-up（预热）阶段，避免训练初期梯度过大导致模型不稳定。
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500), #start_factor=1e-6：初始学习率是 原始学习率 × start_factor。假设基础学习率是 0.001，则训练开始时实际学习率是 0.001 × 1e-6。
    #训练主阶段使用多项式衰减学习率，保证收敛平稳
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=5000,
        by_epoch=False,
    )
]
# --------------------------
# 数据加载
# --------------------------
train_dataloader = dict(batch_size=8, num_workers=12, sampler=dict(type='InfiniteSampler', shuffle=True))
val_dataloader = dict(batch_size=8, num_workers=12)
test_dataloader = val_dataloader

# --------------------------
# 运行时设置
# --------------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000, val_interval=1000)   #执行 5000 个 batch 后结束，每训练 1000 次迭代进行一次验证。
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# # 评估配置
# val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# test_evaluator = val_evaluator

# randomness = dict(seed=42)