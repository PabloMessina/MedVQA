from ignite.exceptions import NotComputableError
from ignite.engine import Events

class DatasetAwareSinglelabelAccuracy:

    def __init__(self, output_transform, allowed_dataset_ids, record_scores=False, ignore_index=-100):
        self.allowed_dataset_ids = allowed_dataset_ids
        self.output_transform = output_transform
        self.ignore_index = ignore_index
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
        pred_labels, gt_labels = output
        n = pred_labels.size(0)
        for i in range(n):
            pred = pred_labels[i]
            gt = gt_labels[i]
            score = (gt == pred).item() # 0 or 1
            if self.ignore_index != gt:
                self._acc_score += score
                self._count += 1
            if self.record_scores:
                self._scores.append(score)
    
    def attach(self, engine, metric_alias):
        
        def epoch_started_handler(unused_engine):
            self.reset()

        def iteration_completed_handler(engine):
            output = engine.state.output
            dataset_id = output['dataset_id'] # make sure your step_fn returns this
            if dataset_id in self.allowed_dataset_ids:
                self.update(self.output_transform(output))

        def epoch_completed_handler(engine):
            engine.state.metrics[metric_alias] = self.compute()

        engine.add_event_handler(Events.EPOCH_STARTED, epoch_started_handler)
        engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_completed_handler)
        engine.add_event_handler(Events.EPOCH_COMPLETED, epoch_completed_handler)

    def compute(self):
        if self._count == 0:
            raise NotComputableError('DatasetAwareMultilabelF1score needs at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count
