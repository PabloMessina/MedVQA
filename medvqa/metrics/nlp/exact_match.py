from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

class ExactMatch(Metric):

    def __init__(self, output_transform=lambda x: x, device=None, record_scores=False):
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        if record_scores:
            self._scores = []
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()
        super().reset()

    def update(self, output):
        pred_sentences, gt_sentences = output
        for pred_s, gt_s in zip(pred_sentences, gt_sentences):
            score = 1 if pred_s == gt_s else 0
            self._acc_score += score
            if self.record_scores:
                self._scores.append(score)
        self._count += len(pred_sentences)

    def compute(self):
        if self._count == 0:
            raise NotComputableError('ExactMatch must have at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count