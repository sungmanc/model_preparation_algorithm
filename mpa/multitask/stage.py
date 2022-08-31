import numpy as np

from mmcv import ConfigDict
from mpa.stage import Stage
from mpa.utils.config_utils import update_or_add_custom_hook
from mpa.utils.logger import get_logger

logger = get_logger()


class MultitaskStage(Stage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs
        """
        logger.info(f'configure!: training={training}')

        # Recipe + model
        cfg = self.cfg
        if model_cfg:
            if hasattr(model_cfg, 'model'):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                raise ValueError("Unexpected config was passed through 'model_cfg'. "
                                 "it should have 'model' attribute in the config")
            model_task = cfg.model.pop('task', 'multitask')
            if model_task != 'multitask':
                raise ValueError(
                    f'Given model_cfg ({model_cfg.filename}) is not supported by multitask recipe'
                )

        # Checkpoint
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)
        pretrained = kwargs.get('pretrained', None)
        if pretrained and isinstance(pretrained, str):
            logger.info(f'Overriding cfg.load_from -> {pretrained}')
            cfg.load_from = pretrained  # Overriding by stage input

        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)

        # Task
        if 'task_adapt' in cfg:
            self.configure_task(cfg, training, **kwargs)

        # Other hyper-parameters
        if 'hyperparams' in cfg:
            self.configure_hyperparams(cfg, training, **kwargs)

        # Hooks
        self.configure_hook(cfg)

        return cfg

    def configure_task(self, cfg, training, **kwargs):
        """Adjust settings for task adaptation
        """
        logger.info(f'task config!!!!: training={training}')
        task_adapt_type = cfg['task_adapt'].get('type', None)
        task_adapt_op = cfg['task_adapt'].get('op', 'REPLACE')

        # Task classes
        org_model_classes, model_classes, data_classes = \
            self.configure_task_classes(cfg, task_adapt_type, task_adapt_op)

        # Data pipeline
        if data_classes != model_classes:
            self.configure_task_data_pipeline(cfg, model_classes, data_classes)

        # Evaluation dataset
        self.configure_task_eval_dataset(cfg, model_classes)

        # Training hook for task adaptation
        self.configure_task_adapt_hook(cfg, org_model_classes, model_classes)

        # Anchor setting
        if cfg['task_adapt'].get('use_mpa_anchor', False):
            self.configure_anchor(cfg)

        # Incremental learning
        self.configure_task_cls_incr(cfg, task_adapt_type, org_model_classes, model_classes)

    def configure_task_classes(self, cfg, task_adapt_type, task_adapt_op):

        # Input classes
        org_model_classes = self.get_model_classes(cfg)
        data_classes = self.get_data_classes(cfg)

        # Model classes
        if task_adapt_op == 'REPLACE':
            if len(data_classes) == 0:
                raise ValueError('Data classes should contain at least one class!')
            model_classes = data_classes.copy()
        elif task_adapt_op == 'MERGE':
            model_classes = org_model_classes + [cls for cls in data_classes if cls not in org_model_classes]
        else:
            raise KeyError(f'{task_adapt_op} is not supported for task_adapt options!')

        if task_adapt_type == 'mpa':
            data_classes = model_classes
        cfg.task_adapt.final = model_classes
        cfg.model.task_adapt = ConfigDict(
            src_classes=org_model_classes,
            dst_classes=model_classes,
        )

        # Model architecture
        if 'roi_head' in cfg.model:
            # For Faster-RCNNs
            cfg.model.roi_head.bbox_head.num_classes = len(model_classes)
        else:
            # For other architectures (including SSD)
            cfg.model.bbox_head.num_classes = len(model_classes)

        return org_model_classes, model_classes, data_classes

    def configure_task_data_pipeline(self, cfg, model_classes, data_classes):
        # Trying to alter class indices of training data according to model class order
        tr_data_cfg = self.get_train_data_cfg(cfg)
        class_adapt_cfg = dict(type='AdaptClassLabels', src_classes=data_classes, dst_classes=model_classes)
        pipeline_cfg = tr_data_cfg.pipeline
        for i, op in enumerate(pipeline_cfg):
            if op['type'] == 'LoadAnnotations':  # insert just after this op
                op_next_ann = pipeline_cfg[i + 1] if i + 1 < len(pipeline_cfg) else {}
                if op_next_ann.get('type', '') == class_adapt_cfg['type']:
                    op_next_ann.update(class_adapt_cfg)
                else:
                    pipeline_cfg.insert(i + 1, class_adapt_cfg)
                break

    def configure_task_eval_dataset(self, cfg, model_classes):
        # - Altering model outputs according to dataset class order
        eval_types = ['val', 'test']
        for eval_type in eval_types:
            if cfg.data[eval_type]['type'] == 'TaskAdaptEvalDataset':
                cfg.data[eval_type]['model_classes'] = model_classes
            else:
                # Wrap original dataset config
                org_type = cfg.data[eval_type]['type']
                cfg.data[eval_type]['type'] = 'TaskAdaptEvalDataset'
                cfg.data[eval_type]['org_type'] = org_type
                cfg.data[eval_type]['model_classes'] = model_classes

    def configure_task_adapt_hook(self, cfg, org_model_classes, model_classes):
        task_adapt_hook = ConfigDict(
            type='TaskAdaptHook',
            src_classes=org_model_classes,
            dst_classes=model_classes,
            model_type=cfg.model.type,
        )
        update_or_add_custom_hook(cfg, task_adapt_hook)

    def configure_hyperparams(self, cfg, training, **kwargs):
        hyperparams = kwargs.get('hyperparams', None)
        if hyperparams is not None:
            bs = hyperparams.get('bs', None)
            if bs is not None:
                cfg.data.samples_per_gpu = bs

            lr = hyperparams.get('lr', None)
            if lr is not None:
                cfg.optimizer.lr = lr
