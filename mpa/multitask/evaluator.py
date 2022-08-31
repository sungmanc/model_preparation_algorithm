from os import path as osp

from mpa.registry import STAGES
from .inferrer import MultitaskInferrer

from mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class MultitaskEvaluator(MultitaskInferrer):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run evaluation stage

        - Run inference
        - Run evaluation via MMDetection -> MMCV
        """
        self.eval = True
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            logger.warning(f'mode for this stage {mode}')
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg.pop('by_epoch', False)
        eval_cfg = eval_cfg['metric']
        
        cls_eval_cfg = {}
        cls_eval_cfg['metric'] = eval_cfg.get('cls_metric')
        
        det_eval_cfg = {}
        det_eval_cfg['metric'] = eval_cfg.get('det_metric')

        # Save config
        cfg.dump(osp.join(cfg.work_dir, 'config.yaml'))
        logger.info(f'Config:\n{cfg.pretty_text}')

        # Inference
        infer_results = super()._infer(cfg)

        cls_results = self.dataset[0].evaluate(infer_results['cls_results'], **cls_eval_cfg)
        logger.info(f'\n{cls_results}')
        det_results = self.dataset[1].evaluate(infer_results['det_results'], **det_eval_cfg)
        logger.info(f'\n{det_results}')
        
        return infer_results
