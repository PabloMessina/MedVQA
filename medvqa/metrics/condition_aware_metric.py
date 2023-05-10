from ignite.engine import Events

class ConditionAwareMetric:

    def __init__(self, output_transform, condition_function):
        self.output_transform = output_transform
        self.condition_function = condition_function
    
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
            if self.condition_function(output):
                self.update(self.output_transform(output))

        def epoch_completed_handler(engine):
            engine.state.metrics[metric_alias] = self.compute()

        engine.add_event_handler(Events.EPOCH_STARTED, epoch_started_handler)
        engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_completed_handler)
        engine.add_event_handler(Events.EPOCH_COMPLETED, epoch_completed_handler)