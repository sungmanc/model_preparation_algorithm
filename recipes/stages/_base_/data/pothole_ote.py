_base_ = [
    './data.py',
    './pipelines/coco_ote_pipeline.py'
]

__dataset_type = 'CocoDataset'
__data_root = 'data/pothole/'

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

__samples_per_gpu = 2

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_train.json',
        img_prefix=__data_root + 'images/train/',
        pipeline=__train_pipeline,
        classes=['pothole']),
    val=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val.json',
        img_prefix=__data_root + 'images/val/',
        test_mode=True,
        pipeline=__test_pipeline,
        classes=['pothole']),
    test=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val.json',
        img_prefix=__data_root + 'images/val/',
        test_mode=True,
        pipeline=__test_pipeline,
        classes=['pothole'])
)
