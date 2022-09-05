import os
import copy
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mpa.multitask.builder import DATASETS
from mpa.multitask.builder import build_pipeline

from mpa.utils.logger import get_logger

logger = get_logger()


@DATASETS.register_module()
class MultitaskDataset(Dataset):
    """
    """
    def __init__(self, cls_data_prefix, cls_pipeline, det_data_prefix, det_ann_file, det_pipeline, **kwargs):
        ### classification dataset preparation
        self.cls_data_prefix = cls_data_prefix
        self.cls_classes = self.get_classes_from_dir(self.cls_data_prefix)
        if isinstance(self.cls_classes, list):
            self.cls_classes.sort()
        self.cls_num_classes = len(self.cls_classes)

        self.data_infos = self.load_annotations()
        self.cls_pipeline = Compose([build_pipeline(p) for p in cls_pipeline])
        
        ### detection dataset preparation
        
    def load_annotations(self):
        data_infos = []
        for cls in self.cls_classes:
            cls_idx = self._class_to_idx('classification')[cls]
            cls_path = os.path.join(os.path.abspath(self.cls_data_prefix), cls)
            for file in os.listdir(cls_path):
                data_info = {'img_prefix': cls_path, 'img_info':{'filename':file}, 'gt_label': cls_idx}
                data_infos.append(data_info)
        return data_infos

    def get_classes_from_dir(self, root):
        classes = []
        path_list = os.listdir(root)
        for p in path_list:
            if os.path.isdir(os.path.join(root, p)):
                if p not in classes:
                    classes.append(p)
            else:
                raise ValueError("This folder structure is not suitable for label data")
        return classes

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.pipeline is None:
            return self.data_infos[idx]

        data_infos = [
            copy.deepcopy(self.data_infos[idx]) for _ in range(self.num_pipes)
        ]
        if isinstance(self.pipeline, dict):
            results = {}
            for i, (k, v) in enumerate(self.pipeline.items()):
                results[k] = self.pipeline[k](data_infos[i])
        else:
            results = self.pipeline(data_infos[0])

        return results

    def _class_to_idx(self, task):
        """Map mapping class name to class index.
        Returns:
            dict: mapping from class name to class index.
        """
        if task == 'classification':
            return {_class: i for i, _class in enumerate(self.cls_classes)}
        elif task == 'detection':
            return {_class: i for i, _class in enumerate(self.det_classes)}
        else:
            print('task must be included in {classification, detection}, but got {}'.format(task))
            raise
    
    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset with new metric 'class_accuracy'

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
                'accuracy', 'precision', 'recall', 'f1_score', 'support', 'class_accuracy'
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5) if self.num_classes >= 5 else (1, )}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric

        if 'class_accuracy' in metrics:
            metrics.remove('class_accuracy')
            self.class_acc = True

        eval_results = super().evaluate(results, metrics, metric_options, logger)

        # Add Evaluation Accuracy score per Class
        if self.class_acc:
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            accuracies = self.class_accuracy(results, gt_labels)
            eval_results.update({f'{c} accuracy': a for c, a in zip(self.CLASSES, accuracies)})
            eval_results.update({'mean accuracy': np.mean(accuracies)})

        return eval_results

    def class_accuracy(self, results, gt_labels):
        accracies = []
        pred_label = results.argsort(axis=1)[:, -1:][:, ::-1]
        for i in range(self.num_classes):
            cls_pred = pred_label == i
            cls_pred = cls_pred[gt_labels == i]
            cls_acc = np.sum(cls_pred) / len(cls_pred)
            accracies.append(cls_acc)
        return accracies

    @property
    def samples_per_gpu(self):
        return self._samples_per_gpu

    @property
    def workers_per_gpu(self):
        return self._workers_per_gpu

