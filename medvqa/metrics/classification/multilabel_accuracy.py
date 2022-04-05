from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from sklearn.metrics import accuracy_score

class MultiLabelAccuracy(Metric):

    def __init__(self, output_transform=lambda x: x, device=None, record_scores=False):
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        if record_scores:
            self._scores = []
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()
        super().reset()

    def update(self, output):
        pred_tags, gt_tags = output
        n = pred_tags.size(0)
        for i in range(n):
            pred = pred_tags[i]
            gt = gt_tags[i]            
            score = accuracy_score(gt, pred)
            self._acc_score += score
            if self.record_scores:
                self._scores.append(score)
        self._count += n

    def compute(self):
        if self._count == 0:
            raise NotComputableError('MultiLabel accuracy needs at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count