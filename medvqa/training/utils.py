import torch

from medvqa.models.checkpoint import load_model_state_dict

def append_metric_name(train_list, val_list, log_list, metric_name, train=True, val=True, log=True):
    if train: train_list.append(metric_name)
    if val: val_list.append(metric_name)
    if log: log_list.append(metric_name)

def batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

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
    count_print,
    override_lr,
):
    from ignite.engine import Events
    from ignite.handlers.timing import Timer
    from medvqa.utils.handlers import (
        get_checkpoint_handler,
        get_log_metrics_handler,
        get_log_iteration_handler,
        get_log_checkpoint_saved_handler,
        get_log_epoch_started_handler,
        get_lr_sch_handler,
    )
    from medvqa.models.checkpoint import get_checkpoint_filepath, save_metadata
    from medvqa.metrics.utils import get_hybrid_score_name
    from medvqa.models.checkpoint.model_wrapper import ModelWrapper
    from medvqa.utils.logging import print_red
    import torch

    # Timer
    timer = Timer()
    timer.attach(trainer_engine, start=Events.EPOCH_STARTED)
    timer.attach(validator_engine, start=Events.EPOCH_STARTED)

    # Learning rate scheduler
    if not update_lr_batchwise:
        count_print('Defining learning rate scheduler handler ...')
        lr_sch_handler = get_lr_sch_handler(lr_scheduler, lr_scheduler_kwargs['name'], score_fn=score_fn)

    # Learning rate scheduler
    if not update_lr_batchwise:
        count_print('Defining learning rate scheduler handler ...')
        lr_sch_handler = get_lr_sch_handler(lr_scheduler, lr_scheduler_kwargs['name'], score_fn=score_fn)    

    # Checkpoint saving
    model_wrapper = ModelWrapper(model, optimizer, lr_scheduler)
    if checkpoint_folder_path is None: # first time
        if save: # only if we want to save checkpoints to disk
            count_print('Defining checkpoint folder path ...')
            checkpoint_folder_path = build_custom_checkpoint_folder_path()
            print_red('checkpoint_folder_path =', checkpoint_folder_path, bold=True)
            save_metadata(checkpoint_folder_path,
                        **metadata_kwargs)
        # Pretrained weights
        pretrained_checkpoint_path = model_kwargs.get('pretrained_checkpoint_path', None)
        pretrained_checkpoint_folder_path = model_kwargs.get('pretrained_checkpoint_folder_path', None)
        pretrained_checkpoint_folder_paths = model_kwargs.get('pretrained_checkpoint_folder_paths', None)
        if (pretrained_checkpoint_path or pretrained_checkpoint_folder_path or pretrained_checkpoint_folder_paths):
            if pretrained_checkpoint_folder_path:
                pretrained_checkpoint_folder_paths = [pretrained_checkpoint_folder_path]
            count_print(f'Loading pretrained weights ...')
            if pretrained_checkpoint_path:
                print(f'pretrained_checkpoint_path = {pretrained_checkpoint_path}')
                checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
                load_model_state_dict(model_wrapper.model, checkpoint['model'])
                print('Checkpoint successfully loaded!')
            if pretrained_checkpoint_folder_paths:
                for pretrained_checkpoint_folder_path in pretrained_checkpoint_folder_paths:
                    pretrained_checkpoint_path = get_checkpoint_filepath(pretrained_checkpoint_folder_path)
                    print(f'pretrained_checkpoint_path = {pretrained_checkpoint_path}')
                    checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
                    load_model_state_dict(model_wrapper.model, checkpoint['model'])
                    print('Checkpoint successfully loaded!')
    else: # resuming
        checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
        count_print('Loading model from checkpoint ...')
        print('checkpoint_path =', checkpoint_path)
        model_wrapper.load_checkpoint(checkpoint_path, device, model_only=override_lr)
    
    if save: # only if we want to save checkpoints to disk
        checkpoint_handler = get_checkpoint_handler(model_wrapper, checkpoint_folder_path, trainer_engine,
                                                    epoch_offset=model_wrapper.get_epoch(),
                                                    score_name=get_hybrid_score_name(train_metrics_to_merge, val_metrics_to_merge),
                                                    score_fn=score_fn)

    # Logging
    count_print('Defining log_metrics_handler ...')

    log_metrics_handler = get_log_metrics_handler(timer,
                                                   metrics_to_print=metrics_to_print,
                                                   log_to_disk=save,
                                                   checkpoint_folder=checkpoint_folder_path)
    log_iteration_handler = get_log_iteration_handler()
    log_checkpoint_saved_handler = get_log_checkpoint_saved_handler(checkpoint_folder_path)
    
    # Attach handlers
    trainer_engine.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    trainer_engine.add_event_handler(Events.EPOCH_STARTED, lambda : print(f'(1) Training stage (lr = {optimizer.param_groups[0]["lr"]:.6f}) ...'))
    trainer_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda : validator_engine.run(val_dataloader,
                                     max_epochs=1, epoch_length=val_dataloader_size))
    validator_engine.add_event_handler(Events.EPOCH_STARTED, lambda : print('(2) Validation stage ...'))
    validator_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    validator_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    if not update_lr_batchwise:
        validator_engine.add_event_handler(Events.EPOCH_COMPLETED, lr_sch_handler)
    if save: # only if we want to save checkpoints to disk
        validator_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
        validator_engine.add_event_handler(Events.EPOCH_COMPLETED, log_checkpoint_saved_handler)

    # Start training
    count_print('Running trainer engine ...')
    trainer_engine.run(train_dataloader, max_epochs=epochs, epoch_length=batches_per_epoch)