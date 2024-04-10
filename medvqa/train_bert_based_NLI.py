import  os
import argparse
import torch
from medvqa.datasets.nli.nli_dataset_management import BertNLITrainer

from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.nlp.nli import BertBasedNLI
from medvqa.models.huggingface_utils import SupportedHuggingfaceMedicalBERTModels

from medvqa.training.utils import append_metric_name, run_common_boilerplate_code_and_start_training
from medvqa.utils.common import WORKSPACE_DIR, DictWithDefault
from medvqa.metrics import (
    attach_condition_aware_loss,
    attach_condition_aware_accuracy,
)
from medvqa.models.checkpoint import load_metadata

from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.files import get_checkpoint_folder_path
from medvqa.training.nli import get_bert_based_nli_engine
from medvqa.datasets.dataloading_utils import get_bert_based_nli_collate_batch_fn
from medvqa.metrics.utils import get_merge_metrics_fn
from medvqa.utils.logging import CountPrinter, print_blue

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # --- Required arguments

    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs the model will be trained')
    parser.add_argument('--batches_per_epoch', type=int, required=True, help='Number of batches per epoch')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')

    # --- Other arguments

    parser.add_argument('--checkpoint_folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')

    # Model arguments
    parser.add_argument('--huggingface_model_name', type=str, default=None, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--nli_hidden_size', type=int, default=512)
    parser.add_argument('--pretrained_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--freeze_huggingface_model', action='store_true', default=False)
    parser.add_argument('--merged_input', action='store_true', default=False)
    
    # Optimization arguments
    parser.add_argument('--optimizer_name', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='exp-warmup+decay+cyclicdecay')
    parser.add_argument('--warmup_decay_and_cyclic_decay_args', type=str, default=None)
    parser.add_argument('--iters_to_accumulate', type=int, default=1, help='For gradient accumulation')
    parser.add_argument('--override_lr', action='store_true', default=False)

    # Dataset and dataloading arguments
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for parallel dataloading')    
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--integrated_nli_jsonl_filepath', type=str, default=None)
    parser.add_argument('--integrated_sentence_facts_jsonl_filepath', type=str, default=None)

    # Checkpoint saving arguments
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no_save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = DictWithDefault(default=1.0) # Default weight is 1.0
   
def train_model(
        model_kwargs,
        optimizer_kwargs,
        lr_scheduler_kwargs,
        nli_trainer_kwargs,
        dataloading_kwargs,
        collate_batch_fn_kwargs,
        training_kwargs,
        trainer_engine_kwargs,
        validator_engine_kwargs,
        epochs,
        batches_per_epoch,
        num_workers,
        device='GPU',
        checkpoint_folder_path=None,
        save=True,
        override_lr=False,
        ):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    batch_size = dataloading_kwargs['batch_size']

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Create model
    count_print('Creating instance of BertBasedNLI ...')
    model = BertBasedNLI(**model_kwargs)
    model = model.to(device)

    # Optimizer
    count_print('Defining optimizer ...')
    optimizer = create_optimizer(params=model.parameters(), **optimizer_kwargs)

    # Learning rate scheduler
    count_print('Defining scheduler ...')
    lr_scheduler, update_lr_batchwise = create_lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)

    # Create trainer and validator engines
    count_print('Creating trainer and validator engines ...')
    trainer_engine = get_bert_based_nli_engine(model=model, optimizer=optimizer, device=device, 
                                update_lr_batchwise=update_lr_batchwise, lr_scheduler=lr_scheduler, **trainer_engine_kwargs)
    validator_engine = get_bert_based_nli_engine(model=model, device=device, **validator_engine_kwargs)
    
    # Define collate batch functions
    count_print('Defining collate batch functions ...')
    nli_collate_batch_fn = get_bert_based_nli_collate_batch_fn(**collate_batch_fn_kwargs)
    
    # Create dataloaders
    nli_trainer = BertNLITrainer(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_batch_fn=nli_collate_batch_fn,
        **nli_trainer_kwargs,
    )
    train_dataloader = nli_trainer.train_dataloader
    val_dataloader = nli_trainer.test_dataloader
    val_dataloader_size = len(val_dataloader)
    trainer_name = nli_trainer.name
    print('nli_trainer.name = ', trainer_name)
    
    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []
    # Losses
    attach_condition_aware_loss(trainer_engine, 'loss')
    metrics_to_print.append('loss')
    attach_condition_aware_loss(trainer_engine, 'nli_loss')
    metrics_to_print.append('nli_loss')
    # Metrics
    attach_condition_aware_accuracy(trainer_engine, pred_field_name='pred_labels', gt_field_name='gt_labels', metric_name='nli_acc')
    attach_condition_aware_accuracy(validator_engine, pred_field_name='pred_labels', gt_field_name='gt_labels', metric_name='nli_acc')
    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'nli_acc')

    # Score function
    assert len(val_metrics_to_merge) > 0
    if len(train_metrics_to_merge) > 0:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0.1, 0.9)
        score_fn = lambda _ : merge_metrics_fn(trainer_engine.state.metrics, validator_engine.state.metrics)
    else:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0, 1)
        score_fn = lambda _ : merge_metrics_fn(validator_engine.state.metrics)

    # Run common boilerplate code and start training
    run_common_boilerplate_code_and_start_training(
        update_lr_batchwise=update_lr_batchwise,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        score_fn=score_fn,
        model=model,
        optimizer=optimizer,
        save=save,
        checkpoint_folder_path=checkpoint_folder_path,
        build_custom_checkpoint_folder_path=lambda: get_checkpoint_folder_path('fact_embedding', trainer_name, model.get_name()),
        metadata_kwargs=dict(
            model_kwargs=model_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            nli_trainer_kwargs=nli_trainer_kwargs,
            dataloading_kwargs=dataloading_kwargs,
            collate_batch_fn_kwargs=collate_batch_fn_kwargs,
            training_kwargs=training_kwargs,
            trainer_engine_kwargs=trainer_engine_kwargs,
            validator_engine_kwargs=validator_engine_kwargs,
        ),
        device=device,
        trainer_engine=trainer_engine,
        validator_engine=validator_engine,
        train_metrics_to_merge=train_metrics_to_merge,
        val_metrics_to_merge=val_metrics_to_merge,
        metrics_to_print=metrics_to_print,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        val_dataloader_size=val_dataloader_size,
        model_kwargs=model_kwargs,
        count_print=count_print,
        override_lr=override_lr,
    )

def train_from_scratch(
    # Model args
    huggingface_model_name,
    nli_hidden_size,
    pretrained_checkpoint_folder_path,
    freeze_huggingface_model,
    merged_input,
    # Optimizer args
    optimizer_name,
    lr,
    # lr_scheduler args
    scheduler,
    warmup_decay_and_cyclic_decay_args,
    # Dataloading args
    batch_size,
    num_workers,
    # Fixed traning args
    use_amp,
    iters_to_accumulate,
    integrated_nli_jsonl_filepath,
    integrated_sentence_facts_jsonl_filepath,
    # Variable traning args
    epochs,
    batches_per_epoch,
    # GPU
    device,
    # Other args
    save,
):
    print_blue('----- Training model from scratch ------', bold=True)
    
    model_kwargs = dict(
        huggingface_model_name=huggingface_model_name,
        hidden_size=nli_hidden_size,
        freeze_huggingface_model=freeze_huggingface_model,
        pretrained_checkpoint_folder_path=pretrained_checkpoint_folder_path,
        merged_input=merged_input,
    )

    optimizer_kwargs = dict(
        name=optimizer_name,
        lr=lr,
    )

    lr_scheduler_kwargs = dict(
        name=scheduler,
        warmup_decay_and_cyclic_decay_args=warmup_decay_and_cyclic_decay_args,
    )
    
    dataloading_kwargs = dict(
        batch_size=batch_size,
    )
    
    nli_trainer_kwargs = dict(
        integrated_nli_jsonl_filepath=integrated_nli_jsonl_filepath,
        integrated_sentence_facts_jsonl_filepath=integrated_sentence_facts_jsonl_filepath,
        merged=merged_input,
    )

    collate_batch_fn_kwargs = dict(
        huggingface_model_name=huggingface_model_name,
        merged_input=merged_input,
    )

    trainer_engine_kwargs = dict(
        iters_to_accumulate=iters_to_accumulate,
        use_amp=use_amp,
        training=True,
        merged_input=merged_input,
    )

    validator_engine_kwargs = dict(
        use_amp=use_amp,
        validating=True,
        merged_input=merged_input,
    )
    
    training_kwargs = dict(
        use_amp=use_amp,
    )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                nli_trainer_kwargs=nli_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                training_kwargs=training_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                num_workers=num_workers,
                device=device,
                save=save)

def resume_training(
        checkpoint_folder,
        scheduler,
        optimizer_name,
        lr,
        warmup_decay_and_cyclic_decay_args,
        num_workers,    
        epochs=1,
        batches_per_epoch=1000,
        device='GPU',
        save=True,
        override_lr=False,
        **unused_kwargs,
        ):
    print_blue('----- Resuming training ------', bold=True)

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    nli_trainer_kwargs = metadata['nli_trainer_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
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
            warmup_decay_and_cyclic_decay_args = warmup_decay_and_cyclic_decay_args,
            n_batches_per_epoch = batches_per_epoch,
        )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                nli_trainer_kwargs=nli_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                training_kwargs=training_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                num_workers=num_workers,
                device=device,
                checkpoint_folder_path=checkpoint_folder,
                save=save,
                override_lr=override_lr)

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    if args['checkpoint_folder'] is not None:
        resume_training(**args)
    else:
        del args['checkpoint_folder']
        del args['override_lr']
        train_from_scratch(**args)