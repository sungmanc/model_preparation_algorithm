import numbers
import os
import os.path as osp
import time
import glob
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mmcv import collect_env
from mmcv.utils import get_git_hash
from mmcv import __version__
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmcv.runner import DistSamplerSeedHook
from mmcv.runner import build_optimizer, build_runner

from mmdet.parallel import MMDataCPU

from mpa.multitask.builder import build_dataset, build_dataloader, build_model


from mmdet.datasets import build_dataloader as mmdet_build_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmdet.core import EvalHook


from mpa.registry import STAGES
from .stage import MultitaskStage
from mpa.modules.hooks.eval_hook import CustomEvalHook, DistCustomEvalHook, MultitaskEvalHook
from mpa.modules.datasets.composed_dataloader import ComposedDL
from mpa.utils.logger import get_logger

logger = get_logger()

@STAGES.register_module()
class MultitaskTrainer(MultitaskStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage for detection

        - Configuration
        - Environment setup
        - Run training via MMDetection -> MMCV
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, **kwargs)
        logger.info('train!')

        # # Work directory
        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Environment
        distributed = False
        if cfg.gpu_ids is not None:
            if isinstance(cfg.get('gpu_ids'), numbers.Number):
                cfg.gpu_ids = [cfg.get('gpu_ids')]
            if len(cfg.gpu_ids) > 1:
                distributed = True

        logger.info(f'cfg.gpu_ids = {cfg.gpu_ids}, distributed = {distributed}')
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        ################
        # Data
        dataset = [build_dataset(cfg.data.train[0], 'classification'), build_dataset(cfg.data.train[1], 'detection')]

        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        # meta['config'] = cfg.pretty_text
        meta['seed'] = cfg.seed
        meta['exp_name'] = cfg.work_dir

        if distributed:
            os.environ['MASTER_ADDR'] = cfg.dist_params.get('master_addr', 'localhost')
            os.environ['MASTER_PORT'] = cfg.dist_params.get('master_port', '29500')
            mp.spawn(MultitaskTrainer.train_worker, nprocs=len(cfg.gpu_ids),
                     args=(dataset, cfg, distributed, True, timestamp, meta))
        else:
            MultitaskTrainer.train_worker(
                None,
                dataset,
                cfg,
                distributed,
                True,
                timestamp,
                meta)

        # Save outputs
        output_ckpt_path = osp.join(cfg.work_dir, 'latest.pth')
        best_ckpt_path = glob.glob(osp.join(cfg.work_dir, 'best_segm_mAP*.pth'))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        best_ckpt_path = glob.glob(osp.join(cfg.work_dir, 'best_bbox_mAP*.pth'))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        return dict(final_ckpt=output_ckpt_path)

    @staticmethod
    def train_worker(gpu, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
        logger.info(f'called train_worker() gpu={gpu}, distributed={distributed}, validate={validate}')
        if distributed:
            torch.cuda.set_device(gpu)
            dist.init_process_group(backend=cfg.dist_params.get('backend', 'nccl'),
                                    world_size=len(cfg.gpu_ids), rank=gpu)
            logger.info(f'dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}')

        # model
        cfg.model.bbox_head.num_classes = len(dataset[1].CLASSES)
        cfg.model.cls_head.num_classes = len(dataset[0].CLASSES)
        model = build_model(cfg.model)

        # prepare data loaders
        dataset = [dataset if isinstance(dataset, (list, tuple)) else [dataset]]

        data_loaders = []
        for ds in dataset:
            if isinstance(ds, list):
                sub_loaders = [
                    build_dataloader(
                        sub_ds,
                        cfg.data.samples_per_gpu,
                        cfg.data.workers_per_gpu,
                        num_gpus=len(cfg.gpu_ids),
                        dist=distributed,
                        seed=cfg.seed
                    ) for sub_ds in ds
                ]
                data_loaders.append(ComposedDL(sub_loaders))
            else:
                data_loaders.append(
                    build_dataloader(
                        ds,
                        cfg.data.samples_per_gpu,
                        cfg.data.workers_per_gpu,
                        # cfg.gpus will be ignored if distributed
                        num_gpus=len(cfg.gpu_ids),
                        dist=distributed,
                        seed=cfg.seed
                    ))

        # put model on gpus
        if torch.cuda.is_available():
            if distributed:
                find_unused_parameters = cfg.get('find_unused_parameters', False)
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
            else:
                model = MMDataParallel(
                    model.cuda(), device_ids=[torch.cuda.current_device()])
        else:
            model = MMDataCPU(model)

        # build runner
        optimizer = build_optimizer(model, cfg.optimizer)

        if cfg.get('runner') is None:
            cfg.runner = {
                'type': 'EpochBasedRunner',
                'max_epochs': cfg.total_epochs
            }
            warnings.warn(
                'config is now expected to have a `runner` section, '
                'please set `runner` in your config.', UserWarning)

        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
        
        # an ugly walkaround to make the .log and .log.json filenames the same
        runner.timestamp = timestamp

        # register hooks
        runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                       cfg.checkpoint_config, cfg.log_config,
                                       cfg.get('momentum_config', None))

        # When distributed training, it is only useful in conjunction with 'EpochBasedRunner`,
        # while `IterBasedRunner` achieves the same purpose with `IterLoader`.
        if distributed:
            runner.register_hook(DistSamplerSeedHook())

        for hook in cfg.get('custom_hooks', ()):
            runner.register_hook_from_cfg(hook)
        
        # register eval hooks
        
        if validate:
            val_dataset = [build_dataset(cfg.data.val[0], 'classification', dict(test_mode=True)), build_dataset(cfg.data.val[1], 'detection', dict(test_mode=True))]
            val_dataloaders = [
                build_dataloader(
                    val_dataset[0],
                    1,
                    cfg.data.workers_per_gpu,
                    seed=cfg.seed,
                    shuffle=False
                ),
                build_dataloader(
                    val_dataset[1],
                    1,
                    cfg.data.workers_per_gpu,
                    seed=cfg.seed,
                    shuffle=False
                ),
            ]
            
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = MultitaskEvalHook #TODO: consider distributed environment
            runner.register_hook(eval_hook(val_dataloaders, **eval_cfg), priority='HIGHEST')
    
         
        if cfg.get('resume_from', False):
            runner.resume(cfg.resume_from)
        elif cfg.get('load_from', False):
            if gpu is None:
                runner.load_checkpoint(cfg.load_from)
            else:
                runner.load_checkpoint(cfg.load_from, map_location=f'cuda:{gpu}')

        runner.run(data_loaders, cfg.workflow)
