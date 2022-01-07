from ignite.metrics import Metric
from medvqa.utils.nlp import indexes_to_string
from pycocoevalcap.cider import cider_scorer

class CiderD(Metric):

    def __init__(self, n=4, output_transform=lambda x: x, device=None, record_scores=False):
        self._n = n
        self.record_scores = record_scores
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self.scorer = cider_scorer.CiderScorer(n=self._n)
        super().reset()

    def update(self, output):
        pred_sentences, gt_sentences = output
        for pred_s, gt_s in zip(pred_sentences, gt_sentences):
            pred_s = indexes_to_string(pred_s)
            gt_s = indexes_to_string(gt_s)
            self.scorer += (pred_s, [gt_s])

    def compute(self):
        mean_score, scores = self.scorer.compute_score()
        if self.record_scores:
            return scores
        return mean_score
