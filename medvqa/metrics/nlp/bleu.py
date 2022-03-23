from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import warnings
warnings.simplefilter('ignore', UserWarning)

class Bleu(Metric):

    def __init__(self, k=None, output_transform=lambda x: x, device=None, record_scores=False):
        self._acc_bleu = 0
        self._count = 0
        if k is None:
            self._weights = (0.25, 0.25, 0.25, 0.25)
            self._smoothing_function = SmoothingFunction().method1
        else:
            self._weights = [0] * k
            self._weights[k-1] = 1
            self._smoothing_function = None        
        self.record_scores = record_scores
        if record_scores:
            self._scores = []
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._acc_bleu = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()
        super().reset()

    def update(self, output):
        pred_sentences, gt_sentences = output
        for pred_s, gt_s in zip(pred_sentences, gt_sentences):
            if len(pred_s) == 0:
                bleu = 0
            else:
                bleu = sentence_bleu((gt_s,), pred_s,
                        weights=self._weights,
                        smoothing_function = self._smoothing_function,
                        auto_reweigh=True)
            self._acc_bleu += bleu
            if self.record_scores:
                self._scores.append(bleu)
        self._count += len(pred_sentences)

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Bleu must have at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_bleu / self._count