import random
from medvqa.metrics.condition_aware_metric import ConditionAwareMetric
from medvqa.utils.logging_utils import print_bold, print_magenta

class ConditionAwareSeq2SeqOutputLogger(ConditionAwareMetric):

    def __init__(self, output_transform, condition_function=lambda _: True, tokenizer=None):
        self.tokenizer = tokenizer
        self.random_output = None
        super().__init__(output_transform, condition_function)
    
    def reset(self):
        self.random_output = None

    def update(self, pred_output):
        if self.random_output is None:
            rand_idx = random.randint(0, len(pred_output)-1)
            if self.tokenizer is None:
                self.random_output = pred_output[rand_idx]
            else:
                self.random_output = self.tokenizer.decode(pred_output[rand_idx], skip_special_tokens=True)
            assert type(self.random_output) == str, self.random_output

    def compute(self):
        assert self.random_output is not None, 'Random output is None'
        print_bold('Random output:')
        print_magenta(self.random_output, bold=True)