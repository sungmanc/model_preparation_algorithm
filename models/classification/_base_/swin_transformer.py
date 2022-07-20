# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='OTESwinTransformer',
        version="tiny",
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
