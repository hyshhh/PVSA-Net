_base_ = [
    '../_base_/models/biformer_mm.py',  # 这里缺少了逗号
    '../_base_/datasets/chase_db1.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# 数据预处理配置 - 覆盖基础配置中的设置
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

# 模型配置 - 只覆盖需要修改的部分
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole')
)

# 训练配置
max_iters = 20000
val_interval = 1000

# 优化器配置
optim_wrapper = dict(
    _delete_=True,
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
        end=max_iters,
        by_epoch=False,
    )
]

# 数据加载配置
train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# 随机种子
randomness = dict(seed=42)