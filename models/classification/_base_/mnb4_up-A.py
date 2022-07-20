# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='OTEMetaNet',
        version=4,
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
