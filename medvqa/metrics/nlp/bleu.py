from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from nltk.translate.bleu_score import sentence_bleu

import warnings
warnings.simplefilter('ignore', UserWarning)

class Bleu(Metric):

    def __init__(self, output_transform, device):
        # print('Bleu:: __init__')
        self._acc_bleu = 0
        self._count = 0
        self._smoothing_function = None
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        # print('Bleu:: reset()')
        self._acc_bleu = 0
        self._count = 0
        super().reset()

    def update(self, output):
        # print('Bleu:: update(), len(output) =', len(output))
        pred_sentences, gt_sentences = output

        # print('  len(pred_sentences) =', len(pred_sentences))
        # print('  len(gt_sentences) =', len(gt_sentences))

        for pred_s, gt_s in zip(pred_sentences, gt_sentences):

            if len(pred_s) == 0:
                bleu = 0
            else:
                bleu = sentence_bleu((gt_s,), pred_s,
                        smoothing_function = self._smoothing_function,
                        auto_reweigh=True)
            self._acc_bleu += bleu
        
        self._count += len(pred_sentences)

        # print('  self._count =', self._count)

    def compute(self):
        # print('Bleu:: compute(), self._count =', self._count)
        if self._count == 0:
            raise NotComputableError('Bleu must have at least one example before it can be computed.')
        return self._acc_bleu / self._count