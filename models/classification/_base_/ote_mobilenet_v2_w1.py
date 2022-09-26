# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mobilenetv2_w1',
        pretrained=True,
        out_indices= [5,],
        frozen_stages=-1,
        norm_eval=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='NonLinearClsHead',
        num_classes=1000,
        in_channels=576,
        hid_channels=1024,
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
