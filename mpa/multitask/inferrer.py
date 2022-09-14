import os.path as osp
import numpy as np
import torch

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mpa.multitask.builder import build_dataset, build_dataloader, build_model

from mpa.registry import STAGES
from mpa.multitask.stage import MultitaskStage
from mpa.modules.utils.task_adapt import prob_extractor
from mpa.modules.datasets.composed_dataloader import ComposedDL
from mpa.utils.logger import get_logger

from mmdet.core import encode_mask_results

logger = get_logger()


@STAGES.register_module()
class MultitaskInferrer(MultitaskStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run inference stage

        - Configuration
        - Environment setup
        - Run inference via mmcls -> mmcv
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        outputs = self._infer(cfg)
        return outputs

    def _infer(self, cfg):
        self.dataset = [build_dataset(cfg.data.test[0], 'classification'), build_dataset(cfg.data.test[1], 'detection')]

        data_loaders = [
            build_dataloader(
                self.dataset[0],
                cfg.data.samples_per_gpu[0],
                cfg.data.workers_per_gpu,
                seed=cfg.seed,
                shuffle=False
            ),
            build_dataloader(
                self.dataset[1],
                1,
                cfg.data.workers_per_gpu,
                seed=cfg.seed,
                shuffle=False
            ),
        ]
        
        # build the model and load checkpoint
        cfg.model.bbox_head.num_classes = len(self.dataset[1].CLASSES)
        cfg.model.cls_head.num_classes = len(self.dataset[0].CLASSES)
        
        model = build_model(cfg.model)
        
        if cfg.load_from is not None:
            logger.info('load checkpoint from ' + cfg.load_from)
            _ = load_checkpoint(model, cfg.load_from, map_location='cpu')
        
        model.eval()
        model = MMDataParallel(model, device_ids=[0])

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        MultitaskStage.set_inference_progress_callback(model, cfg)

        results = self.single_gpu_test(model, data_loaders)
        
        return results
    
    def single_gpu_test(self, model, data_loader):
        model.eval()
        task_names = ['cls_results', 'det_results']
        
        results = {}
        for i, task in enumerate(task_names):
            results[task] = []
            logger.info('Evaluating {} task '.format(task))
            dataset = self.dataset[i]
            prog_bar = mmcv.ProgressBar(len(dataset))
            for data in data_loader[i]:
                data['task'] = task
                with torch.no_grad():
                    result = model(return_loss=False, **data)
                batch_size = len(result)
                
                if task == 'cls_results':
                    if not isinstance(result, dict):
                        result = np.array(result, dtype=np.float32)
                        for r in result:
                            results[task].append(r)
                elif task == 'det_results':
                    results[task].extend(result)                   
                for _ in range(batch_size):
                    prog_bar.update()

        return results
    
