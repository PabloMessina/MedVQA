from medvqa.metrics.bbox.utils import (
    compute_multiple_f1_scores,
    compute_multiple_precision_scores,
    compute_multiple_recall_scores,
)
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric
import numpy as np

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

    def compute(self, num_workers=5):
        n = len(self._pred_coords)
        assert n == len(self._pred_presence)
        assert n == len(self._gt_coords)
        assert n == len(self._gt_presence)
        f1_scores = compute_multiple_f1_scores(
            pred_coords=self._pred_coords,
            pred_presences=self._pred_presence,
            gt_coords=self._gt_coords,
            gt_presences=self._gt_presence,
            iou_thresholds=self._iou_thresholds,
            num_workers=num_workers,
        )
        mean_f1 = np.mean(f1_scores)
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

    def compute(self, num_workers=5):
        n = len(self._pred_coords)
        assert n == len(self._pred_presence)
        assert n == len(self._gt_coords)
        assert n == len(self._gt_presence)
        precision_scores = compute_multiple_precision_scores(
            pred_coords=self._pred_coords,
            pred_presences=self._pred_presence,
            gt_coords=self._gt_coords,
            gt_presences=self._gt_presence,
            iou_thresholds=self._iou_thresholds,
            num_workers=num_workers,
        )
        mean_precision = np.mean(precision_scores)
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

    def compute(self, num_workers=5):
        n = len(self._pred_coords)
        assert n == len(self._pred_presence)
        assert n == len(self._gt_coords)
        assert n == len(self._gt_presence)
        recall_scores = compute_multiple_recall_scores(
            pred_coords=self._pred_coords,
            pred_presences=self._pred_presence,
            gt_coords=self._gt_coords,
            gt_presences=self._gt_presence,
            iou_thresholds=self._iou_thresholds,
            num_workers=num_workers,
        )
        mean_recall = np.mean(recall_scores)
        return mean_recall
        
                           
