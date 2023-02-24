from ignite.exceptions import NotComputableError
from medvqa.metrics.bbox.utils import compute_iou
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric

class DatasetAwareBboxIOU(DatasetAwareMetric):

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
        assert m * 4 == gt_coords.shape[1] # each bounding box is represented by 4 coordinates
        for i in range(n):
            for j in range(0, m):
                if gt_presence[i, j] == 1:
                    pred = pred_coords[i, j*4:(j+1)*4]
                    gt = gt_coords[i, j*4:(j+1)*4]
                    self._acc_score += compute_iou(pred, gt)
                    self._count += 1

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Bbox IOU needs at least one example before it can be computed.')
        return self._acc_score / self._count