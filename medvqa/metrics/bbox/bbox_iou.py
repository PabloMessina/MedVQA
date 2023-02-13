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
        pred_coords, gt_coords = output
        n, m = pred_coords.shape
        assert m % 4 == 0 # each bounding box is represented by 4 coordinates
        for i in range(n):
            pred = pred_coords[i]
            gt = gt_coords[i]
            score = 0
            for j in range(0, m, 4):
                score += compute_iou(pred[j:j+4], gt[j:j+4])
            score /= m // 4 # average over all bounding boxes
            self._acc_score += score
        self._count += n

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Bbox IOU needs at least one example before it can be computed.')
        return self._acc_score / self._count