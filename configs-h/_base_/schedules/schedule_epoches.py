

# ----------------------------
# 1. 优化器设置
# ----------------------------
optimizer = dict(
    type='AdamW',
    lr=0.0001,  # 初始学习率，可按数据集调整
    betas=(0.9, 0.999),
    weight_decay=0.01)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None  # 如果显存紧张或梯度爆炸，可改为 dict(max_norm=1.0)
)

# ----------------------------
# 2. 学习率调度策略 (param_scheduler)
# ----------------------------
# 这里采用两阶段策略：
# - 前 5 个 epoch 线性预热
# - 后续 45 个 epoch 使用多项式衰减 (poly decay)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,  # 预热阶段
        convert_to_iter_based=False
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5,
        end=50,
        by_epoch=True,
        convert_to_iter_based=False
    )
]

# ----------------------------
# 3. 训练/验证/测试配置
# ----------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,      # 总共 50 个 epoch
    val_interval=5      # 每 5 个 epoch 进行验证
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ----------------------------
# 4. 默认钩子设置
# ----------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=True  # 日志以 epoch 方式显示
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=5,          # 每 5 个 epoch 保存一次模型
        save_best='auto',    # 自动保存最佳模型
        max_keep_ckpts=3     # 最多保留 3 个权重文件
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

