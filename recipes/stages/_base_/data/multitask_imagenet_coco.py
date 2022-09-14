#### General setting for multitask dataset
__cls_img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
__det_img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

#### Pipeline for classification
#__cls_data_root='./data/imagenet/'
__cls_data_root='./data/cifar100_cls_per_img_6_1/'
#__cls_resize_target_size = (992, 736)
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

#### Pipeline for detection
__det_data_root='./data/coco/'
#__det_data_root='./data/pothole/'
__det_img_size = (992, 736)
__det_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672), (992, 800)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **__det_img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
__det_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=__det_img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **__det_img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

#### 
__cls_samples_per_gpu = 12
__det_samples_per_gpu = 12

data = dict(
    samples_per_gpu=[__cls_samples_per_gpu, __det_samples_per_gpu],
    workers_per_gpu=2,
    train=[
        # Classification Dataset
        dict(
            type="ClsDirDataset",
            data_dir=__cls_data_root + 'train',
            pipeline=__cls_train_pipeline),
        # Detection Dataset
        dict(
            type="CocoDataset",
            ann_file=__det_data_root + 'annotations/instances_train2017.json',
            #ann_file=__det_data_root + 'annotations/instances_train.json',
            img_prefix=__det_data_root + 'train2017/',
            #img_prefix=__det_data_root + 'images/train/',
            pipeline=__det_train_pipeline,
            #classes=['pothole']
            )
    ],
    val=[
        # Classification Dataset
        dict(
            type="ClsDirDataset",
            #data_dir=__cls_data_root + 'val',
            data_dir=__cls_data_root + 'test',
            pipeline=__cls_test_pipeline),
        # Detection Dataset
        dict(
            type="CocoDataset",
            ann_file=__det_data_root + 'annotations/instances_val2017.json',
            #ann_file=__det_data_root + 'annotations/instances_val.json',
            img_prefix=__det_data_root + 'val2017/',
            #img_prefix=__det_data_root + 'images/val/',
            test_mode=True,
            pipeline=__det_test_pipeline,
            #classes=['pothole']
            )
    ],
    test=[
        # Classification Dataset
        dict(
            type="ClsDirDataset",
            #data_dir=__cls_data_root + 'val',
            data_dir=__cls_data_root + 'test',
            pipeline=__cls_test_pipeline),
        # Detection Dataset
        dict(
            type="CocoDataset",
            ann_file=__det_data_root + 'annotations/instances_val2017.json',
            #ann_file=__det_data_root + 'annotations/instances_val.json',
            img_prefix=__det_data_root + 'val2017/',
            #img_prefix=__det_data_root + 'images/val/',
            test_mode=True,
            pipeline=__det_test_pipeline,
            #classes=['pothole']
            )
    ]
)