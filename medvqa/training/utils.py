import torch
import logging
from medvqa.models.checkpoint import load_model_state_dict
from medvqa.utils.common import activate_determinism, deactivate_determinism
from medvqa.utils.logging_utils import ANSI_RED_BOLD, ANSI_RESET, log_title

logger = logging.getLogger(__name__)

def append_metric_name(train_list, val_list, log_list, metric_name, train=True, val=True, log=True):
    if train: train_list.append(metric_name)
    if val: val_list.append(metric_name)
    if log: log_list.append(metric_name)

def batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

def run_validation_engine(validator_engine, val_dataloader, val_dataloader_size, use_determinism):
    logger.info('(2) Validation stage ...')
    if use_determinism:
        activate_determinism(verbose=False) # deterministic validation
    validator_engine.run(val_dataloader, max_epochs=1, epoch_length=val_dataloader_size)
    if use_determinism:
        deactivate_determinism() # back to non-deterministic training

def run_common_boilerplate_code_and_start_training(
    update_lr_batchwise,
    lr_scheduler,
    lr_scheduler_kwargs,
    score_fn,
    model,
    optimizer,
    save,
    checkpoint_folder_path,
    build_custom_checkpoint_folder_path,
    metadata_kwargs,
    device,
    trainer_engine,
    validator_engine,
    train_metrics_to_merge,
    val_metrics_to_merge,
    metrics_to_print,
    train_dataloader,
    val_dataloader,
    epochs,
    batches_per_epoch,
    val_dataloader_size,
    model_kwargs,
    override_lr,
    use_determinism_during_validation=True,
):
    from ignite.engine import Events
    from ignite.handlers.timing import Timer
    from ignite.contrib.handlers.tqdm_logger import ProgressBar
    from medvqa.utils.handlers_utils import (
        get_checkpoint_handler,
        get_log_metrics_handler,
        get_log_checkpoint_saved_handler,
        get_log_epoch_started_handler,
        get_lr_sch_handler,
    )
    from medvqa.models.checkpoint import get_checkpoint_filepath, save_metadata
    from medvqa.metrics.utils import get_hybrid_score_name
    from medvqa.models.checkpoint.model_wrapper import ModelWrapper
    import torch

    # Timer
    timer = Timer()
    timer.attach(trainer_engine, start=Events.EPOCH_STARTED)
    timer.attach(validator_engine, start=Events.EPOCH_STARTED)

    # Learning rate scheduler
    if not update_lr_batchwise:
        log_title(logger, 'Defining learning rate scheduler handler')
        lr_sch_handler = get_lr_sch_handler(lr_scheduler, lr_scheduler_kwargs['name'], score_fn=score_fn)

    # Checkpoint saving
    model_wrapper = ModelWrapper(model, optimizer, lr_scheduler)
    if checkpoint_folder_path is None: # first time
        if save: # only if we want to save checkpoints to disk
            log_title(logger, 'Defining checkpoint folder path')
            checkpoint_folder_path = build_custom_checkpoint_folder_path()
            logger.info(f'{ANSI_RED_BOLD}checkpoint_folder_path = {checkpoint_folder_path}{ANSI_RESET}')
            save_metadata(checkpoint_folder_path, **metadata_kwargs)
        # Pretrained weights
        pretrained_checkpoint_path = model_kwargs.get('pretrained_checkpoint_path', None)
        pretrained_checkpoint_folder_path = model_kwargs.get('pretrained_checkpoint_folder_path', None)
        pretrained_checkpoint_folder_paths = model_kwargs.get('pretrained_checkpoint_folder_paths', None)
        if (pretrained_checkpoint_path or pretrained_checkpoint_folder_path or pretrained_checkpoint_folder_paths):
            if pretrained_checkpoint_folder_path:
                pretrained_checkpoint_folder_paths = [pretrained_checkpoint_folder_path]
            log_title(logger, f'Loading pretrained weights')
            if pretrained_checkpoint_path:
                logger.info(f'pretrained_checkpoint_path = {pretrained_checkpoint_path}')
                checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
                load_model_state_dict(model_wrapper.model, checkpoint['model'])
                logger.info('Checkpoint successfully loaded!')
            if pretrained_checkpoint_folder_paths:
                for pretrained_checkpoint_folder_path in pretrained_checkpoint_folder_paths:
                    pretrained_checkpoint_path = get_checkpoint_filepath(pretrained_checkpoint_folder_path)
                    logger.info(f'pretrained_checkpoint_path = {pretrained_checkpoint_path}')
                    checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
                    load_model_state_dict(model_wrapper.model, checkpoint['model'])
                    logger.info('Checkpoint successfully loaded!')
    else: # resuming
        checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
        log_title(logger, 'Loading model from checkpoint')
        logger.info(f'checkpoint_path = {checkpoint_path}')
        model_wrapper.load_checkpoint(checkpoint_path, device, model_only=override_lr)
    
    if save: # only if we want to save checkpoints to disk
        checkpoint_handler = get_checkpoint_handler(model_wrapper, checkpoint_folder_path, trainer_engine,
                                                    epoch_offset=model_wrapper.get_epoch(),
                                                    score_name=get_hybrid_score_name(train_metrics_to_merge, val_metrics_to_merge),
                                                    score_fn=score_fn)

    # Logging & Progress bar
    log_title(logger, 'Defining log_metrics_handler')

    log_metrics_handler = get_log_metrics_handler(
        timer,
        metrics_to_print=metrics_to_print,
        log_to_disk=save,
        checkpoint_folder=checkpoint_folder_path,
    )

    log_checkpoint_saved_handler = get_log_checkpoint_saved_handler(checkpoint_folder_path)

    # --- Progress Bar Setup ---
    # Create progress bars
    pbar_train = ProgressBar(persist=True, desc='Training', mininterval=2, miniters=5, ncols=70)
    pbar_val = ProgressBar(persist=True, desc='Validation', mininterval=2, miniters=10, ncols=50)

    # Attach progress bars to engines
    pbar_train.attach(
        trainer_engine, output_transform=lambda x: {'loss': x.get('loss', 0.0)}
    )
    pbar_val.attach(validator_engine)
    # --- End Progress Bar Setup ---
    
    # --- Attach other handlers ---
    trainer_engine.add_event_handler(
        Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper)
    )
    trainer_engine.add_event_handler(
        Events.EPOCH_STARTED,
        lambda: logger.info(
            f'(1) Training stage (lr = {optimizer.param_groups[0]["lr"]:.6f}) ...'
        ),
    )
    trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    trainer_engine.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda : run_validation_engine(validator_engine, val_dataloader,
                                       val_dataloader_size, use_determinism_during_validation),
    )
    validator_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

    if not update_lr_batchwise:
        validator_engine.add_event_handler(Events.EPOCH_COMPLETED, lr_sch_handler)
    if save:  # only if we want to save checkpoints to disk
        validator_engine.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler
        )
        validator_engine.add_event_handler(
            Events.EPOCH_COMPLETED, log_checkpoint_saved_handler
        )
    # --- End Attach Handlers ---

    # Start training
    log_title(logger, 'Running trainer engine')
    trainer_engine.run(train_dataloader, max_epochs=epochs, epoch_length=batches_per_epoch)