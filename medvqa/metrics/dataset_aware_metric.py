from ignite.engine import Events

class DatasetAwareMetric:

    def __init__(self, output_transform, allowed_dataset_ids):
        self.allowed_dataset_ids = allowed_dataset_ids
        self.output_transform = output_transform
    
    def reset(self):
        raise NotImplementedError('Make sure your specialized class implements this function')

    def update(self, *args):
        raise NotImplementedError('Make sure your specialized class implements this function')
    
    def compute(self):
        raise NotImplementedError('Make sure your specialized class implements this function')
    
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
