# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    # pretrained="EFT_B.pth",
    backbone=dict(
        type='l_EFT_T',
        style='pytorch',
        reduction_ratios=[1, 1, 1, 1]), #  if ISR is applied, adjust this "reduction_ratios". (ex) reduction_ratios=[2, 2, 1, 1]
   
    decode_head=dict(
        type='L_light_head_obstacle',
        in_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        obstacle_id = 0,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))