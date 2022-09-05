from asyncio.log import logger
from collections import OrderedDict
from unittest import loader
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from mmcv import Config

from mmdet.models import build_detector

from mpa.multitask.builder import MODELS
from mpa.multitask.builder import build_head, build_neck

from mpa.utils.visualize import _visualize_det_img

@MODELS.register_module()
class MultitaskModel(nn.Module):
    """
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 cls_neck=None,
                 cls_head=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(MultitaskModel, self).__init__()

        # build ATSS detector
        detector_cfg = dict(
            type='ATSS',
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head
        )

        self.detector = build_detector(detector_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        self.cls_neck = build_neck(cls_neck)
        self.cls_head = build_head(cls_head)

    def forward_train(self, **data):
        
        cls_imgs, cls_img_metas, cls_labels = data['img'], data['img_metas'], data['gt_label']
        det_imgs, det_img_metas, det_labels, det_bboxes = data['extra_0']['img'], data['extra_0']['img_metas'], data['extra_0']['gt_labels'], data['extra_0']['gt_bboxes']
        
        #### forward for classification
        cls_x = self.detector.backbone(cls_imgs)
        cls_x = self.cls_neck(cls_x[-1])
        cls_loss = self.cls_head.forward_train(cls_x, cls_labels)['loss']
        
        #### forward for detection
        det_loss = self.detector.forward_train(det_imgs, det_img_metas, det_bboxes, det_labels)
        
        total_loss = det_loss
        total_loss['cls_loss'] = cls_loss

        return total_loss

    def forward_test(self, **data):
        cls_imgs, cls_img_metas = data['img'], data['img_metas']
        det_imgs, det_img_metas = data['extra_0']['img'], data['extra_0']['img_metas']
        
        #### forward for classification
        cls_x = self.detector.backbone(cls_imgs)
        cls_x = self.cls_neck(cls_x[-1])
        cls_results = self.cls_head.simple_test(cls_x)
        
        #### forward for detection
        det_results = self.detector.forward_test(det_imgs, det_img_metas)

        return cls_results, det_results
    
    def forward(self, return_loss=True, **data):
        """
        """
        if 'extra_0' in data.keys():
            if return_loss:
                return self.forward_train(**data)
            else:
                return self.forward_test(**data)
        else:
            if data['task'] == 'cls_results':
                return self.forward_cls(**data)
            else:
                return self.forward_det(**data)
    
    def forward_cls(self, **data):
        cls_imgs = data['img']
        
        cls_x = self.detector.backbone(cls_imgs)
        cls_x = self.cls_neck(cls_x[-1])
        return self.cls_head.simple_test(cls_x)

    def forward_det(self, **data):
        det_imgs, det_img_metas = data['img'], data['img_metas']
        det_results = self.detector.forward_test(det_imgs, det_img_metas, rescale=True)
        return det_results
        
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def val_step(self, *args):
        pass

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    
