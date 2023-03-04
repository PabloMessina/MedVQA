from medvqa.metrics.bbox.utils import compute_iou
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric

class DatasetAwareBboxIOU(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids, use_detectron2=False):
        self._acc_score = 0
        self._count = 0
        self._use_detectron2 = use_detectron2
        super().__init__(output_transform, allowed_dataset_ids)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0

    def update(self, output):
        if self._use_detectron2:
            pred_boxes, pred_classes, scores, gt_coords, gt_presence = output
            n = len(gt_presence)
            for i in range(n):
                for j in range(len(pred_boxes[i])):
                    if scores[i][j] < 0.5:
                        continue
                    cls = pred_classes[i][j].item()
                    if gt_presence[i][cls] == 1:
                        iou = compute_iou(pred_boxes[i][j], gt_coords[i][cls])
                        self._acc_score += iou
                        self._count += 1
        else:
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
            print('WARNING: Bbox IOU defaulting to 0 since self._count is 0')
            return 0
        return self._acc_score / self._count