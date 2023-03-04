import torch
import numpy as np
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric

class DatasetAwareBboxMAE(DatasetAwareMetric):

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
                        ae = torch.abs(pred_boxes[i][j] - gt_coords[i][cls])
                        self._acc_score += ae.mean()
                        self._count += 1
        else:
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
            print('WARNING: Bbox MAE defaulting to 0 since self._count is 0')
            return 0
        return self._acc_score / self._count