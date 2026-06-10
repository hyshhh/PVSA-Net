norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    # pretrained="EFT_T.pth",
    backbone=dict(
        type='EFT_T',
        style='pytorch',
        reduction_ratios=[1, 1, 1, 1]), #  if ISR is applied, adjust this "reduction_ratios". (ex) reduction_ratios=[2, 2, 1, 1]
    decode_head=dict(
        type='EDAFormerHead',
        in_channels=[64, 128, 256],
        in_index=[1, 2, 3],
        reduction_ratios=[1, 1, 1], # reduction_ratio_ratios=[2, 2, 2] if ISR is applied
        mlp_ratio=2,
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        embed_dim=128,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))