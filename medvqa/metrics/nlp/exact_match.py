from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from medvqa.metrics.condition_aware_metric import ConditionAwareMetric
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric

class ExactMatch(Metric):

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

class DatasetAwareExactMatch(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids, record_scores=False):
        super().__init__(output_transform, allowed_dataset_ids)
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

class ConditionAwareSeq2SeqExactMatch(ConditionAwareMetric):

    def __init__(self, output_transform, tokenizer, condition_function=lambda _: True, check_is_prefix=False):
        self.tokenizer = tokenizer
        self._acc_score = 0
        self._count = 0
        self.check_is_prefix = check_is_prefix
        super().__init__(output_transform, condition_function)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0

    def update(self, output):
        pred_ids, gt_texts = output
        pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        if self.check_is_prefix:
            for pred_text, gt_text in zip(pred_texts, gt_texts):
                score = 1 if pred_text.startswith(gt_text) else 0
                self._acc_score += score
        else:
            for pred_text, gt_text in zip(pred_texts, gt_texts):
                score = 1 if pred_text == gt_text else 0
                self._acc_score += score
        self._count += len(pred_texts)

    def compute(self):
        if self._count == 0:
            raise NotComputableError('ExactMatch must have at least one example before it can be computed.')
        return self._acc_score / self._count