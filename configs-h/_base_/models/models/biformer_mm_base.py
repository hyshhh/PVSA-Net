# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='l_BiFormer_mm',
        # embed_dim=[64, 128, 320, 512],     # BiFormer的通道配置
        # depth=[3, 4, 6, 3],               # 每个stage的block数量
        embed_dim=[96,192,384,768],     # BiFormer的通道配置
        depth=[4,4,18,4],               # 每个stage的block数量

        mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[96,192,384,768],   # 改这里
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=256,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,                    # 与 decode_head 一致
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    # ),
    # 模型训练与推理配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)