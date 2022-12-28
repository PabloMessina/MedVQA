import argparse
import os

import torch
from ignite.engine import Events
from ignite.handlers.timing import Timer

from transformers import ViTFeatureExtractor, ViTMAEForPreTraining
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    get_mae_collate_batch_fn,
    multi_cyclic_dataloaders_generator,
)
from medvqa.datasets.image_processing import get_pretrain_vit_mae_image_transform
from medvqa.datasets.padchest.padchest_dataset_management import PadChest_MAE_Trainer

from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.losses.optimizers import create_optimizer
from medvqa.metrics import attach_loss
from medvqa.metrics.utils import get_hybrid_score_name, get_merge_metrics_fn
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata, save_metadata
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.models.common import load_model_state_dict
from medvqa.training.utils import append_metric_name

from medvqa.utils.common import WORKSPACE_DIR, parsed_args_to_dict
from medvqa.utils.constants import PADCHEST_DATASET_ID, MetricNames
from medvqa.utils.files import get_checkpoint_folder_path
from medvqa.utils.handlers import get_checkpoint_handler, get_log_epoch_started_handler, get_log_iteration_handler, get_log_metrics_handlers, get_lr_sch_handler
from medvqa.utils.logging import CountPrinter, print_blue, print_red
from medvqa.training.mae import get_engine

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # --- Required arguments --- #
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batches-per-epoch', type=int, required=True)

    # --- Optional arguments --- #

    # Resuming training args
    parser.add_argument('--checkpoint-folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')
    
    # Model args
    parser.add_argument('--pretrained-model-name-or-path', type=str, default='facebook/vit-mae-base')
    
    # Optimizer args
    parser.add_argument('--optimizer-name', type=str, default='adamw')
    parser.add_argument('--num-accumulation-steps', type=int, default=1)

    # LR args
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--override-lr', dest='override_lr', action='store_true')
    parser.set_defaults(override_lr=False)
    
    # LR scheduler args
    parser.add_argument('--scheduler', type=str, default='reduce-lr-on-plateau')
    parser.add_argument('--lr-decay', type=float, default=0.76, help='Learning rate decay')
    parser.add_argument('--lr-decay-patience', type=int, default=2, help='Learning rate decay patience')
    parser.add_argument('--warmup-and-decay-args', type=str, default=None)
    parser.add_argument('--warmup-and-cosine-args', type=str, default=None)

    # Training args
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='GPU')
    parser.add_argument('--padchest-weight', type=float, default=1.0)
    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)

    # PadChest args
    parser.add_argument('--use-padchest', dest='use_padchest', action='store_true')
    parser.set_defaults(use_padchest=False)
    parser.add_argument('--padchest-training-data-mode', type=str, default='all')
    parser.add_argument('--padchest-use-validation', dest='padchest_use_validation', action='store_true')
    parser.set_defaults(padchest_use_validation=False)
    parser.add_argument('--padchest-train-study-ids-path', type=str, default=None)
    parser.add_argument('--padchest-val-study-ids-path', type=str, default=None)
    parser.add_argument('--padchest-test-study-ids-path', type=str, default=None)

    # Saving args
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = {
    MetricNames.LOSS: -1.0,
}

def _train_model(    
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    trainer_engine_kwargs,
    validator_engine_kwargs,
    training_kwargs,
    dataloading_kwargs,
    padchest_trainer_kwargs,
    epochs,
    batches_per_epoch,
    num_workers,
    device='GPU',
    checkpoint_folder_path=None,
    save=True,
    override_lr=False,
    debug=False,
):
    count_print = CountPrinter()

    # Pull out some args from kwargs
    batch_size = dataloading_kwargs['batch_size']
    use_padchest = training_kwargs['use_padchest']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print(f'Device: {device}')

    # Create model
    count_print('Creating instance of ViTMAEForPreTraining ...')
    print(f'pretrained_model_name_or_path: {model_kwargs["pretrained_model_name_or_path"]}')
    model = ViTMAEForPreTraining.from_pretrained(model_kwargs['pretrained_model_name_or_path'])
    model = model.to(device)

    # Optimizer
    count_print('Creating optimizer ...')
    optimizer = create_optimizer(params=model.parameters(), **optimizer_kwargs)

    # Learning rate scheduler
    count_print('Creating learning rate scheduler ...')
    lr_scheduler, update_lr_batchwise = create_lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)

    # Trainer and Validator engines
    count_print('Creating trainer and validator engines ...')
    trainer = get_engine(model=model, device=device, optimizer=optimizer, update_lr_batchwise=update_lr_batchwise,
                         lr_scheduler=lr_scheduler, **trainer_engine_kwargs)
    validator = get_engine(model=model, device=device, **validator_engine_kwargs)    

    # Create feature extractor
    count_print('Creating instance of ViTFeatureExtractor ...')
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_kwargs['pretrained_model_name_or_path'])

    # Create image transform
    count_print('Creating image transform ...')
    image_transform = get_pretrain_vit_mae_image_transform(feature_extractor)

    # Create PadChest MAE trainer
    if use_padchest:
        count_print('Creating PadChest MAE trainer ...')
        padchest_mae_trainer = PadChest_MAE_Trainer(
            transform=image_transform,
            batch_size=batch_size,
            collate_batch_fn=get_mae_collate_batch_fn(PADCHEST_DATASET_ID),
            num_workers=num_workers,
            **padchest_trainer_kwargs,
        )

    if debug: # if debugging
        output = {}
        if use_padchest: output['padchest_mae_trainer'] = padchest_mae_trainer
        return output

    # Create complex dataloaders
    count_print('Creating dataloaders ...')

    _train_weights = []
    _train_dataloaders = []
    _val_dataloaders = []
    _dataset_names = []

    if use_padchest:
        _dataset_names.append('padchest')
        _train_weights.append(dataloading_kwargs['padchest_weight'])
        _train_dataloaders.append(padchest_mae_trainer.train_dataloader)
        if padchest_mae_trainer.use_validation_set:
            _val_dataloaders.append(padchest_mae_trainer.val_dataloader)
    
    assert len(_train_dataloaders) > 0, 'No training datasets specified!'
    assert len(_val_dataloaders) > 0, 'No validation datasets specified!'
    assert len(_train_dataloaders) == len(_train_weights), 'Number of training datasets and training weights must be equal!'
    print(f'len(_train_dataloaders) = {len(_train_dataloaders)}')
    print(f'len(_val_dataloaders) = {len(_val_dataloaders)}')
    print(f'_train_weights = {_train_weights}')

    # final train dataloader
    if len(_train_dataloaders) > 1:
        train_dataloader = balanced_dataloaders_generator(_train_dataloaders, _train_weights)
    else:
        train_dataloader = _train_dataloaders[0]

    # final val dataloader
    val_dataloader_size = sum(len(d) for d in _val_dataloaders)
    val_dataloader = multi_cyclic_dataloaders_generator(_val_dataloaders)

    merged_dataset_name = '+'.join(_dataset_names)
    print(f'merged_dataset_name = {merged_dataset_name}')

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')
    
    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []

    attach_loss('loss', trainer, device)
    attach_loss('loss', validator, device)
    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'loss')

    # Timer
    timer = Timer()
    timer.attach(trainer, start=Events.EPOCH_STARTED)
    timer.attach(validator, start=Events.EPOCH_STARTED)

    # Score function
    merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0.1, 0.9)
    score_fn = lambda _ : 1. / (1. + merge_metrics_fn(trainer.state.metrics, validator.state.metrics)) # minimize loss

    # Learning rate scheduler
    if not update_lr_batchwise:
        count_print('Creating learning rate scheduler handler ...')
        lr_sch_handler = get_lr_sch_handler(lr_scheduler, lr_scheduler_kwargs['name'], score_fn=score_fn)

    # Checkpoint saving
    model_wrapper = ModelWrapper(model, optimizer, lr_scheduler)

    pretrained_checkpoint_folder_path = model_kwargs.get('pretrained_checkpoint_folder_path', None)
    
    if checkpoint_folder_path is None: # first time
        if save: # only if we want to save checkpoints to disk
            count_print('Defining checkpoint folder path ...')
            checkpoint_folder_path = get_checkpoint_folder_path('mae', merged_dataset_name, 'VitMAE',
                f'dws={",".join(map(str, _train_weights))}' \
                    if len(_train_weights) > 1 else None,
            )
            print_red('checkpoint_folder_path =', checkpoint_folder_path)
            save_metadata(checkpoint_folder_path,
                        model_kwargs = model_kwargs,
                        optimizer_kwargs = optimizer_kwargs,
                        lr_scheduler_kwargs = lr_scheduler_kwargs,                        
                        padchest_trainer_kwargs = padchest_trainer_kwargs,
                        dataloading_kwargs = dataloading_kwargs,
                        training_kwargs = training_kwargs,
                        trainer_engine_kwargs = trainer_engine_kwargs,
                        validator_engine_kwargs = validator_engine_kwargs)

        if pretrained_checkpoint_folder_path is not None:
            pretrained_checkpoint_path = get_checkpoint_filepath(pretrained_checkpoint_folder_path)
            count_print(f'Loading pretrained weights ...')
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
        checkpoint_handler = get_checkpoint_handler(model_wrapper, checkpoint_folder_path, trainer,
                                                    epoch_offset=model_wrapper.get_epoch(),
                                                    score_name=get_hybrid_score_name(train_metrics_to_merge, val_metrics_to_merge),
                                                    score_fn=score_fn)

    # Logging
    count_print('Creating log_metrics_handler ...')
    log_metrics_handler = get_log_metrics_handlers(timer,
                                                   metrics_to_print=metrics_to_print,
                                                   log_to_disk=save,
                                                   checkpoint_folder=checkpoint_folder_path)
    log_iteration_handler = get_log_iteration_handler()

    # Attach handlers
    trainer.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda : print(f'(1) Training stage (lr = {optimizer.param_groups[0]["lr"]:.6f}) ...'))
    trainer.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda : validator.run(val_dataloader,
                                     max_epochs=1, epoch_length=val_dataloader_size))
    validator.add_event_handler(Events.EPOCH_STARTED, lambda : print('(2) Validation stage ...'))
    validator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    validator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    if not update_lr_batchwise:
        validator.add_event_handler(Events.EPOCH_COMPLETED, lr_sch_handler)
    if save: # only if we want to save checkpoints to disk
        validator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # Start training
    count_print('Running trainer engine ...')
    trainer.run(train_dataloader,
                max_epochs = epochs,
                epoch_length = batches_per_epoch)


def train_from_scratch(
    # Model args
    pretrained_model_name_or_path,
    # Optimizer args
    optimizer_name,
    lr,
    # lr_scheduler args
    scheduler,
    lr_decay,
    lr_decay_patience,
    warmup_and_decay_args,
    warmup_and_cosine_args,
    # PadChest args
    use_padchest,
    padchest_training_data_mode,
    padchest_use_validation,
    padchest_train_study_ids_path,
    padchest_val_study_ids_path,
    padchest_test_study_ids_path,
    # Dataloading args
    batch_size,
    # Fixed training args
    use_amp,
    num_accumulation_steps,
    padchest_weight,
    # Variable training args
    epochs,
    batches_per_epoch,
    num_workers,
    # GPU
    device,
    # Other args
    save,
    debug=False,
):
    print_blue('------------------- Training ViTMAE from scratch -------------------')
    model_kwargs = dict(
        pretrained_model_name_or_path = pretrained_model_name_or_path,
    )
    optimizer_kwargs = dict(
        name = optimizer_name,
        lr = lr,
    )
    lr_scheduler_kwargs = dict(
        name = scheduler,
        factor = lr_decay,
        patience = lr_decay_patience,
        warmup_and_decay_args = warmup_and_decay_args,
        warmup_and_cosine_args = warmup_and_cosine_args,
        n_batches_per_epoch = batches_per_epoch,
    )
    trainer_engine_kwargs = dict(
        use_amp=use_amp,
        training=True,
        num_accumulation_steps=num_accumulation_steps,
    )
    validator_engine_kwargs = dict(
        training=False,
        use_amp=use_amp,
    )
    training_kwargs = dict(
        use_padchest=use_padchest,
    )
    dataloading_kwargs = dict(
        batch_size=batch_size,
        padchest_weight=padchest_weight,
    )
    padchest_trainer_kwargs = dict(
        train_study_ids_path=padchest_train_study_ids_path,
        val_study_ids_path=padchest_val_study_ids_path,
        test_study_ids_path=padchest_test_study_ids_path,
        training_data_mode=padchest_training_data_mode,
        use_validation_set=padchest_use_validation,
    )
    return _train_model(
        model_kwargs=model_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        trainer_engine_kwargs=trainer_engine_kwargs,
        validator_engine_kwargs=validator_engine_kwargs,
        training_kwargs=training_kwargs,
        dataloading_kwargs=dataloading_kwargs,
        padchest_trainer_kwargs=padchest_trainer_kwargs,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        num_workers=num_workers,
        device=device,
        save=save,
        debug=debug,
    )

def resume_training(
    checkpoint_folder,
    scheduler,
    optimizer_name,
    lr,
    lr_decay,
    lr_decay_patience,
    warmup_and_decay_args,
    warmup_and_cosine_args,
    num_workers,
    epochs = 1,
    batches_per_epoch = 1000,
    device = 'GPU',
    save = True,
    override_lr = False,
    debug = False,
    **unused_kwargs,
):
    
    print_blue('----- Resuming training ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']    
    padchest_trainer_kwargs = metadata['padchest_trainer_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']    
    training_kwargs = metadata['training_kwargs']
    trainer_engine_kwargs = metadata['trainer_engine_kwargs']
    validator_engine_kwargs = metadata['validator_engine_kwargs']

    if override_lr:
        optimizer_kwargs = dict(
            name = optimizer_name,
            lr = lr,
        )
        lr_scheduler_kwargs = dict(
            name = scheduler,
            factor = lr_decay,
            patience = lr_decay_patience,
            warmup_and_decay_args = warmup_and_decay_args,
            warmup_and_cosine_args = warmup_and_cosine_args,
            n_batches_per_epoch = batches_per_epoch,
        )
    return _train_model(
        model_kwargs=model_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        trainer_engine_kwargs=trainer_engine_kwargs,
        validator_engine_kwargs=validator_engine_kwargs,
        training_kwargs=training_kwargs,
        dataloading_kwargs=dataloading_kwargs,
        padchest_trainer_kwargs=padchest_trainer_kwargs,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        num_workers=num_workers,
        device=device,
        checkpoint_folder_path = checkpoint_folder,
        save=save,
        debug=debug,
    )

def debug_main(args):
    args = parse_args(args)
    args = parsed_args_to_dict(args)
    if args['checkpoint_folder'] is not None:
        return resume_training(**args, debug=True)
    else:
        del args['checkpoint_folder']
        del args['override_lr']
        return train_from_scratch(**args, debug=True)

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    if args['checkpoint_folder'] is not None:
        resume_training(**args)
    else:
        del args['checkpoint_folder']
        del args['override_lr']
        train_from_scratch(**args)
