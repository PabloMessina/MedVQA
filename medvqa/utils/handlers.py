from ignite.handlers import Checkpoint, DiskSaver
from ignite.engine import Events
import operator

def get_log_epoch_started_handler(model_wrapper):
    epoch_offset = model_wrapper.get_epoch()
    def handler(trainer):
        epoch = epoch_offset + trainer.state.epoch
        max_epochs = epoch_offset + trainer.state.max_epochs
        print(f'---- Epoch {epoch}/{max_epochs}')
        model_wrapper.set_epoch(epoch)
    return handler

def get_log_iteration_handler(log_every=5):

    def handler(engine):
        i = engine.state.iteration
        if (i+1) % log_every == 0:
            print(f'   iteration {i}', end='\r')

    return handler

def get_log_metrics_handlers(timer, metrics_to_print):

    def handler(engine):
        metrics = engine.state.metrics        
        scores = []
        for m in metrics_to_print:
            score = metrics.get(m)
            if hasattr(score, '__len__'):
                score = sum(score) / len(score)
            scores.append(score)

        metrics_str = ', '.join(f'{m} {s}' for m, s in zip(metrics_to_print, scores))
        duration = timer._elapsed()
        print(f'{metrics_str}, {duration}')
    
    return handler

def get_lr_sch_handler(trainer, validator, lr_scheduler, merge_metrics_fn):

    def handler():
        value = merge_metrics_fn(trainer.state.metrics, validator.state.metrics)
        lr_scheduler.step(value)

    return handler

def get_checkpoint_handler(model_wrapper, folder_path, trainer, epoch_offset, score_name, score_fn):    
    checkpoint = Checkpoint(
        to_save=model_wrapper.to_save(),
        save_handler=DiskSaver(folder_path, require_empty=False, atomic=False),
        global_step_transform=lambda *_: trainer.state.epoch + epoch_offset,
        score_function=score_fn,
        score_name=score_name,
        greater_or_equal=True,
    )
    return checkpoint

class Accumulator():

    def __init__(self, output_transform):
        self._list = []
        self._output_transform = output_transform
    
    def reset(self):
        self._list.clear()

    def update(self, output):
        self._list.extend(self._output_transform(output))

    def get_list(self):
        return self._list

def attach_accumulator(engine, output_name):
    accumulator = Accumulator(output_transform=operator.itemgetter(output_name))
    
    def epoch_started_handler(engine):
        accumulator.reset()
    def iteration_completed_handler(engine):
        # print('Debugging::accumulator')
        # print('   questions[0] = ', engine.state.output['questions'][0])
        # print('   pred_questions[0] = ', engine.state.output['pred_questions'][0])
        accumulator.update(engine.state.output)
    def epoch_completed_handler(engine):
        engine.state.metrics[output_name] = accumulator.get_list()

    engine.add_event_handler(Events.EPOCH_STARTED, epoch_started_handler)
    engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_completed_handler)
    engine.add_event_handler(Events.EPOCH_COMPLETED, epoch_completed_handler)
    
