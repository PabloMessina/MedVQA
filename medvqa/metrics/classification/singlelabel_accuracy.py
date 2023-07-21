from medvqa.metrics.condition_aware_metric import ConditionAwareMetric
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric
from ignite.exceptions import NotComputableError

class DatasetAwareSingleLabelAccuracy(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids, record_scores=False, ignore_index=-100):
        super().__init__(output_transform, allowed_dataset_ids)
        self.ignore_index = ignore_index
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        if record_scores:
            self._scores = []
    
    def reset(self):
        self._acc_score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()

    def update(self, output):
        pred_labels, gt_labels = output
        n = pred_labels.size(0)
        for i in range(n):
            pred = pred_labels[i]
            gt = gt_labels[i]
            score = (gt == pred).item() # 0 or 1
            if self.ignore_index != gt:
                self._acc_score += score
                self._count += 1
            if self.record_scores:
                self._scores.append(score)

    def compute(self):
        if self._count == 0:
            raise NotComputableError('DatasetAwareSinglelabelAccuracy must have at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count
    
class ConditionAwareSingleLabelAccuracy(ConditionAwareMetric):

    def __init__(self, output_transform, condition_function=lambda _: True):
        super().__init__(output_transform, condition_function)

    def reset(self):
        self._acc_score = 0
        self._count = 0

    def update(self, output):
        pred_labels, gt_labels = output
        n = pred_labels.size(0)
        self._count += n
        self._acc_score += (pred_labels == gt_labels).sum().item()

    def compute(self):
        if self._count == 0:
            raise NotComputableError('ConditionAwareSingleLabelAccuracy must have at least one example before it can be computed.')
        return self._acc_score / self._count
    
class InstanceConditionedTripletAccuracy(ConditionAwareMetric):

    def __init__(self, output_transform, accepted_id, condition_function=lambda _: True):
        super().__init__(output_transform, condition_function)
        self._accepted_id = accepted_id

    def reset(self):
        self._acc_score = 0
        self._count = 0

    def update(self, output):
        triplet_scores, ids = output
        n = triplet_scores.size(0)
        for i in range(n):
            if ids[i] == self._accepted_id:
                self._count += 1
                if triplet_scores[i] > 0:
                    self._acc_score += 1

    def compute(self):
        if self._count == 0:
            raise NotComputableError('InstanceConditionedTripletAccuracy must have at least one example before it can be computed.')
        return self._acc_score / self._count