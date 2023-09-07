from ignite.metrics import Metric
from medvqa.utils.nlp import indexes_to_string
from pycocoevalcap.rouge import rouge

class RougeL(Metric):

    def __init__(self, output_transform=lambda x: x, device=None, record_scores=False, using_ids=True):
        self.scorer = rouge.Rouge()
        self.record_scores = record_scores
        self.using_ids = using_ids
        if self.record_scores:
            self._scores = []
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._n_samples = 0
        self._current_score = 0
        if self.record_scores:
            self._scores.clear()
        super().reset()
    
    def update(self, output):
        pred_sentences, gt_sentences = output
        if self.using_ids:
            for pred_s, gt_s in zip(pred_sentences, gt_sentences):
                pred_s = indexes_to_string(pred_s)
                gt_s = indexes_to_string(gt_s)
                score = self.scorer.calc_score([pred_s], [gt_s])
                self._current_score += score
                self._n_samples += 1
                if self.record_scores:
                    self._scores.append(score)
        else:
            for pred_s, gt_s in zip(pred_sentences, gt_sentences):
                score = self.scorer.calc_score([pred_s], [gt_s])
                self._current_score += score
                self._n_samples += 1
                if self.record_scores:
                    self._scores.append(score)
        
    def compute(self):
        if self.record_scores:
            return self._scores
        return self._current_score / self._n_samples if self._n_samples > 0 else 0.0