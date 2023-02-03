import torch
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
        pred_coords, gt_coords = output
        n = pred_coords.shape[0]
        for i in range(n):
            pred = pred_coords[i]
            gt = gt_coords[i]
            score = torch.mean(torch.abs(pred - gt)).item()
            self._acc_score += score
        self._count += n

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Bbox MAE needs at least one example before it can be computed.')
        return self._acc_score / self._count    