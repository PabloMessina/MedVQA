import random
from medvqa.metrics.condition_aware_metric import ConditionAwareMetric
from medvqa.utils.logging import print_bold, print_magenta

class ConditionAwareT5ReportLogger(ConditionAwareMetric):

    def __init__(self, output_transform, condition_function=lambda _: True, t5_tokenizer=None):
        self.t5_tokenizer = t5_tokenizer
        self.random_report = None
        super().__init__(output_transform, condition_function)
    
    def reset(self):
        self.random_report = None

    def update(self, pred_reports):
        if self.random_report is None:
            rand_idx = random.randint(0, len(pred_reports)-1)
            if self.t5_tokenizer is None:
                self.random_report = pred_reports[rand_idx]
            else:
                self.random_report = self.t5_tokenizer.decode(pred_reports[rand_idx], skip_special_tokens=True)
            assert type(self.random_report) == str, self.random_report

    def compute(self):
        assert self.random_report is not None, 'Random report is not set'
        print_bold('Random report:')
        print_magenta(self.random_report, bold=True)