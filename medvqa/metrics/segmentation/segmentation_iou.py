import torch
from medvqa.metrics.condition_aware_metric import ConditionAwareMetric

class ConditionAwareSegmaskIOU(ConditionAwareMetric):

    def __init__(self, output_transform, condition_function=lambda _: True):
        super().__init__(output_transform, condition_function)
        self._acc_score = 0
        self._count = 0

    def reset(self):
        self._acc_score = 0
        self._count = 0

    def update(self, output):
        pred_mask, gt_mask = output
        gt_has_area = gt_mask.sum(-1) > 0 # (bs, n_classes)
        # intersection = (pred_mask & gt_mask).sum(-1)
        intersection = (pred_mask * gt_mask).sum(-1)
        # union = (pred_mask | gt_mask).sum(-1)
        union = torch.max(pred_mask, gt_mask).sum(-1)
        bs = pred_mask.shape[0]
        nc = pred_mask.shape[1]
        for i in range(bs):
            for j in range(nc):
                if gt_has_area[i, j]:
                    iou = (intersection[i, j] / union[i, j]).item()
                else:
                    iou = union[i, j].item() < 1e-6 # if gt has no area, then return 1 if pred has no area, else 0
                self._acc_score += iou
        self._count += bs * nc

    def compute(self):
        if self._count == 0:
            print('WARNING: Bbox IOU defaulting to 0 since self._count is 0')
            return 0
        return self._acc_score / self._count