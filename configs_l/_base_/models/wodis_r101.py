model = dict(
    type='EncoderDecoder',

    backbone=dict(
        type='WODISBackbone',
        pretrained=True
    ),

    decode_head=dict(
        type='WODISHead',

        in_channels=[256,512,1024,2048],
        in_index=[0, 1, 2, 3],

        channels=64,

        num_classes=3,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)