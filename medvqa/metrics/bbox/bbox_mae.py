import torch
import numpy as np
from ignite.exceptions import NotComputableError
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric

class DatasetAwareBboxMAE(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids):
        self._acc_score = 0
        self._count = 0
        super().__init__(output_transform, allowed_dataset_ids)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0

    def update(self, output):
        pred_coords, gt_coords, gt_presence = output
        n, m = gt_presence.shape
        if torch.is_tensor(pred_coords):
            for i in range(n):
                ae = torch.abs(pred_coords[i] - gt_coords[i])
                for j in range(m):
                    if gt_presence[i, j] == 1:
                        self._acc_score += ae[j*4:(j+1)*4].mean()
                        self._count += 1
        else: # numpy
            for i in range(n):
                ae = np.abs(pred_coords[i] - gt_coords[i])
                for j in range(m):
                    if gt_presence[i, j] == 1:
                        self._acc_score += ae[j*4:(j+1)*4].mean()
                        self._count += 1

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Bbox MAE needs at least one example before it can be computed.')
        return self._acc_score / self._count    