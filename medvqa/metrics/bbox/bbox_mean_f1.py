from medvqa.metrics.bbox.utils import compute_iou
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric
from sklearn.metrics import f1_score
import numpy as np
# import time

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
        # start = time.time()
        pred_coords, gt_coords, pred_presence, gt_presence = output
        n = pred_coords.shape[0]
        for i in range(n):
            self._pred_coords.append(pred_coords[i])
            self._pred_presence.append(pred_presence[i])
            self._gt_coords.append(gt_coords[i])
            self._gt_presence.append(gt_presence[i])
        # print('update time: {}'.format(time.time() - start))

    def compute(self):
        # start = time.time()
        n = len(self._pred_coords)
        assert n == len(self._pred_presence)
        assert n == len(self._gt_coords)
        assert n == len(self._gt_presence)
        y_score = np.zeros((n,), dtype=np.float32)
        y_true = np.zeros((n,), dtype=np.int8)
        f1_sum = 0        
        for iou_thrs in self._iou_thresholds:
            for c in range(self._n_classes):
                a = 4 * c
                b = 4 * c + 4
                for i in range(n):
                    if self._pred_presence[i][c] > 0 and compute_iou(self._pred_coords[i][a:b], self._gt_coords[i][a:b]) >= iou_thrs:
                        y_score[i] = 1
                    else:
                        y_score[i] = 0
                    y_true[i] = self._gt_presence[i][c]
                f1_sum += f1_score(y_true, y_score)
        mean_f1 = f1_sum / (self._n_classes * len(self._iou_thresholds))
        # print('compute time: {}'.format(time.time() - start))
        return mean_f1
        
                           
