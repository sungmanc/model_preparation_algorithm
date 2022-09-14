__cls_img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
__cls_data_root='./data/cifar100/'
__cls_resize_target_size = 224

__cls_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type="Resize", size=__cls_resize_target_size),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="RandomRotate", p=0.35, angle=(-10, 10)),
    dict(type="ToNumpy"),
    dict(type='Normalize', **__cls_img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
__cls_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=__cls_resize_target_size),
    dict(type='Normalize', **__cls_img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

__samples_per_gpu = 8

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type="ClsDirDataset",
        data_dir=__cls_data_root + 'train',
        pipeline=__cls_train_pipeline
    ),
    val=dict(
        type="ClsDirDataset",
        data_dir=__cls_data_root + 'test',
        pipeline=__cls_test_pipeline
    ),
    test=dict(
        type="ClsDirDataset",
        data_dir=__cls_data_root + 'test',
        pipeline=__cls_test_pipeline
    ),
)