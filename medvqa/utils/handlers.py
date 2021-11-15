from ignite.handlers import Checkpoint, DiskSaver

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
        metrics_str = ', '.join(f'{m} {metrics.get(m)}' for m in metrics_to_print)
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

