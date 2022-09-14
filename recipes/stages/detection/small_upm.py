_base_ = [
    '../_base_/data/pothole_ote.py',
    '../_base_/logs/tensorboard_logger.py',
    '../_base_/optimizers/sgd.py',
    '../_base_/runners/iter_runner.py',
    '../_base_/schedules/cos_anneal.py',
    '../_base_/models/detectors/detector.py'
]

task_adapt = dict(
    type='mpa',
    op='REPLACE',
    efficient_mode=False
)

runner = dict(
    type='IterBasedRunner',
    max_iters=62500
)
optimizer = dict(
    lr= 0.01,
    momentum= 0.9,
    weight_decay= 0.0005,
)

optimizer_config = dict(
    type= 'SAMOptimizerHook'
)

evaluation = dict(
    interval= 6250,
    metric= 'bbox',
    save_best= 'bbox',
)

ignore = True


custom_hooks = [
    dict(type= 'NoBiasDecayHook'),
    dict(type= 'ModelEmaV2Hook'),
]

log_config = dict(
    interval= 6250,
    by_epoch= False,
)

checkpoint_config = dict(
    interval = 6250,
    by_epoch = False,
)