# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='OTEMetaNet',
        version=4,
        frozen_stages=-1,
        out_indices=(5,),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        init_cfg=dict(
            type='Kaiming',
            a=2.23606,
            mode='fan_out',
            nonlinearity='relu',
            distribution='uniform',
        ),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
