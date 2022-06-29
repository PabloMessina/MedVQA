from ignite.exceptions import NotComputableError
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric

class DatasetAwareLoss(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids):
        super().__init__(output_transform, allowed_dataset_ids)
        self._acc_loss = 0
        self._count = 0
    
    def reset(self):
        self._acc_loss = 0
        self._count = 0

    def update(self, loss):
        self._acc_loss += loss
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise NotComputableError('DatasetAwareLoss needs at least one example before it can be computed.')
        return self._acc_loss / self._count
