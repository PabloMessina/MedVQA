from medvqa.metrics.bbox.utils import compute_iou
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric
from multiprocessing import Pool
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
# import time

# HACK to make multiprocessing work. Based on https://stackoverflow.com/a/37746961/2801404
_shared_pred_presence = None
_shared_pred_coords = None
_shared_gt_presence = None
_shared_gt_coords = None

def _compute_score(task, metric_fn):
    iou_thrs, c, n = task
    y_score = np.zeros((n,), dtype=np.float32)
    y_true = np.zeros((n,), dtype=np.int8)
    a = 4 * c
    b = 4 * c + 4
    for i in range(n):
        if _shared_pred_presence[i][c] > 0 and compute_iou(_shared_pred_coords[i][a:b], _shared_gt_coords[i][a:b]) >= iou_thrs:
            y_score[i] = 1
        else:
            y_score[i] = 0
        y_true[i] = _shared_gt_presence[i][c]
    return metric_fn(y_true, y_score)

def _compute_f1(task):
    return _compute_score(task, f1_score)

def _compute_precision(task):
    return _compute_score(task, precision_score)

def _compute_recall(task):
    return _compute_score(task, recall_score)

class DatasetAwareBboxMeanF1(DatasetAwareMetric):
    """Computes mean F1 score for bounding boxes.
    """

    def __init__(self, output_transform, allowed_dataset_ids, n_classes, iou_thresholds=[0.9]):
        super().__init__(output_transform, allowed_dataset_ids)
        self._iou_thresholds = iou_thresholds
        self._n_classes = n_classes
        self._gt_coords = []
        self._gt_presence = []
        self._pred_coords = []
        self._pred_presence = []
    
    def reset(self):
        self._gt_coords.clear()
        self._gt_presence.clear()
        self._pred_coords.clear()
        self._pred_presence.clear()

    def update(self, output):
        pred_coords, gt_coords, pred_presence, gt_presence = output
        n = pred_coords.shape[0]
        for i in range(n):
            self._pred_coords.append(pred_coords[i])
            self._pred_presence.append(pred_presence[i])
            self._gt_coords.append(gt_coords[i])
            self._gt_presence.append(gt_presence[i])

    # def compute(self):
    #     # start = time.time()
    #     n = len(self._pred_coords)
    #     assert n == len(self._pred_presence)
    #     assert n == len(self._gt_coords)
    #     assert n == len(self._gt_presence)
    #     y_score = np.zeros((n,), dtype=np.float32)
    #     y_true = np.zeros((n,), dtype=np.int8)
    #     f1_sum = 0        
    #     for iou_thrs in self._iou_thresholds:
    #         for c in range(self._n_classes):
    #             a = 4 * c
    #             b = 4 * c + 4
    #             for i in range(n):
    #                 if self._pred_presence[i][c] > 0 and compute_iou(self._pred_coords[i][a:b], self._gt_coords[i][a:b]) >= iou_thrs:
    #                     y_score[i] = 1
    #                 else:
    #                     y_score[i] = 0
    #                 y_true[i] = self._gt_presence[i][c]
    #             f1_sum += f1_score(y_true, y_score)
    #     mean_f1 = f1_sum / (self._n_classes * len(self._iou_thresholds))
    #     # print('compute time: {}'.format(time.time() - start))
    #     return mean_f1

    def compute(self, num_workers=4):
        # Use multiprocessing to compute the metric in parallel.
        # This is useful when the metric is computed on a large dataset.
        # print(f'Calling compute with {num_workers} workers.')
        # start = time.time()
        n = len(self._pred_coords)
        assert n == len(self._pred_presence)
        assert n == len(self._gt_coords)
        assert n == len(self._gt_presence)
        global _shared_pred_presence
        global _shared_pred_coords
        global _shared_gt_presence
        global _shared_gt_coords
        _shared_pred_presence = self._pred_presence
        _shared_pred_coords = self._pred_coords
        _shared_gt_presence = self._gt_presence
        _shared_gt_coords = self._gt_coords
        tasks = []
        for iou_thrs in self._iou_thresholds:
            for c in range(self._n_classes):
                tasks.append((iou_thrs, c, n))
        # print(f'Computing {len(tasks)} tasks.')
        with Pool(num_workers) as p:
            f1_scores = p.map(_compute_f1, tasks)
        mean_f1 = sum(f1_scores) / len(f1_scores)
        # print('compute time: {}'.format(time.time() - start))
        return mean_f1

class DatasetAwareBboxMeanPrecision(DatasetAwareMetric):
    """Computes mean precision for bounding boxes.
    """

    def __init__(self, output_transform, allowed_dataset_ids, n_classes, iou_thresholds=[0.9]):
        super().__init__(output_transform, allowed_dataset_ids)
        self._iou_thresholds = iou_thresholds
        self._n_classes = n_classes
        self._gt_coords = []
        self._gt_presence = []
        self._pred_coords = []
        self._pred_presence = []
    
    def reset(self):
        self._gt_coords.clear()
        self._gt_presence.clear()
        self._pred_coords.clear()
        self._pred_presence.clear()

    def update(self, output):
        pred_coords, gt_coords, pred_presence, gt_presence = output
        n = pred_coords.shape[0]
        for i in range(n):
            self._pred_coords.append(pred_coords[i])
            self._pred_presence.append(pred_presence[i])
            self._gt_coords.append(gt_coords[i])
            self._gt_presence.append(gt_presence[i])

    def compute(self, num_workers=4):
        n = len(self._pred_coords)
        assert n == len(self._pred_presence)
        assert n == len(self._gt_coords)
        assert n == len(self._gt_presence)
        global _shared_pred_presence
        global _shared_pred_coords
        global _shared_gt_presence
        global _shared_gt_coords
        _shared_pred_presence = self._pred_presence
        _shared_pred_coords = self._pred_coords
        _shared_gt_presence = self._gt_presence
        _shared_gt_coords = self._gt_coords
        tasks = []
        for iou_thrs in self._iou_thresholds:
            for c in range(self._n_classes):
                tasks.append((iou_thrs, c, n))
        with Pool(num_workers) as p:
            precision_scores = p.map(_compute_precision, tasks)
        mean_precision = sum(precision_scores) / len(precision_scores)
        return mean_precision

class DatasetAwareBboxMeanRecall(DatasetAwareMetric):
    """Computes mean precision for bounding boxes.
    """

    def __init__(self, output_transform, allowed_dataset_ids, n_classes, iou_thresholds=[0.9]):
        super().__init__(output_transform, allowed_dataset_ids)
        self._iou_thresholds = iou_thresholds
        self._n_classes = n_classes
        self._gt_coords = []
        self._gt_presence = []
        self._pred_coords = []
        self._pred_presence = []
    
    def reset(self):
        self._gt_coords.clear()
        self._gt_presence.clear()
        self._pred_coords.clear()
        self._pred_presence.clear()

    def update(self, output):
        pred_coords, gt_coords, pred_presence, gt_presence = output
        n = pred_coords.shape[0]
        for i in range(n):
            self._pred_coords.append(pred_coords[i])
            self._pred_presence.append(pred_presence[i])
            self._gt_coords.append(gt_coords[i])
            self._gt_presence.append(gt_presence[i])

    def compute(self, num_workers=4):
        n = len(self._pred_coords)
        assert n == len(self._pred_presence)
        assert n == len(self._gt_coords)
        assert n == len(self._gt_presence)
        global _shared_pred_presence
        global _shared_pred_coords
        global _shared_gt_presence
        global _shared_gt_coords
        _shared_pred_presence = self._pred_presence
        _shared_pred_coords = self._pred_coords
        _shared_gt_presence = self._gt_presence
        _shared_gt_coords = self._gt_coords
        tasks = []
        for iou_thrs in self._iou_thresholds:
            for c in range(self._n_classes):
                tasks.append((iou_thrs, c, n))
        with Pool(num_workers) as p:
            recall_scores = p.map(_compute_recall, tasks)
        mean_recall = sum(recall_scores) / len(recall_scores)
        return mean_recall
        
        
                           
