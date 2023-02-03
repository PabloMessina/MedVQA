from ignite.exceptions import NotComputableError
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric

class DatasetAwareBboxIOU(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids):
        self._acc_score = 0
        self._count = 0
        super().__init__(output_transform, allowed_dataset_ids)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0

    def _compute_iou(self, pred, gt):
        # compute intersection over union (IoU)
        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(pred[0], gt[0])
        yA = max(pred[1], gt[1])
        xB = min(pred[2], gt[2])
        yB = min(pred[3], gt[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
        boxBArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou.item() # convert to float

    def update(self, output):
        pred_coords, gt_coords = output
        n, m = pred_coords.shape
        assert m % 4 == 0 # each bounding box is represented by 4 coordinates
        for i in range(n):
            pred = pred_coords[i]
            gt = gt_coords[i]
            score = 0
            for j in range(0, m, 4):
                score += self._compute_iou(pred[j:j+4], gt[j:j+4])
            score /= m // 4 # average over all bounding boxes
            self._acc_score += score
        self._count += n

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Bbox IOU needs at least one example before it can be computed.')
        return self._acc_score / self._count