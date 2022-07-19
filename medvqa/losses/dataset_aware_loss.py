from ignite.exceptions import NotComputableError
from ignite.engine import Events

class DatasetAwareLoss:

    def __init__(self, output_transform, allowed_dataset_ids):
        self.allowed_dataset_ids = allowed_dataset_ids
        self.output_transform = output_transform
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