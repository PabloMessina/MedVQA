import warnings
from ignite.metrics import Metric
from pycocoevalcap.bleu import bleu_scorer
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric
from medvqa.utils.text_data_utils import indexes_to_string

warnings.simplefilter('ignore', UserWarning)

class Bleu(Metric):

    def __init__(self, output_transform=lambda x: x, device=None, record_scores=False, using_ids=True):
        self.record_scores = record_scores
        self.using_ids = using_ids
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self.scorer = bleu_scorer.BleuScorer(n=4)
        super().reset()

    def update(self, output):
        pred_sentences, gt_sentences = output
        if self.using_ids:
            for pred_s, gt_s in zip(pred_sentences, gt_sentences):
                pred_s = indexes_to_string(pred_s)
                gt_s = indexes_to_string(gt_s)
                self.scorer += (pred_s, [gt_s])
        else:
            for pred_s, gt_s in zip(pred_sentences, gt_sentences):
                self.scorer += (pred_s, [gt_s])

    def compute(self):
        scores, scores_by_instance = self.scorer.compute_score()
        if self.record_scores:
            return scores, scores_by_instance
        return scores

class DatasetAwareBleu(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids, record_scores=False):
        self.record_scores = record_scores
        super().__init__(output_transform, allowed_dataset_ids)
    
    def reset(self):
        self.scorer = bleu_scorer.BleuScorer(n=4)

    def update(self, output):
        pred_sentences, gt_sentences = output
        for pred_s, gt_s in zip(pred_sentences, gt_sentences):
            pred_s = indexes_to_string(pred_s)
            gt_s = indexes_to_string(gt_s)
            self.scorer += (pred_s, [gt_s])

    def compute(self):
        scores, scores_by_instance = self.scorer.compute_score()
        if self.record_scores:
            return scores, scores_by_instance
        return scores