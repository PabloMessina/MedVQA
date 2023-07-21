import  os
import argparse
import torch
import ast

from medvqa.datasets.fact_embedding.fact_embedding_dataset_management import (
    _LABEL_TO_CATEGORY,
    _LABEL_TO_COMPARISON_STATUS,
    _LABEL_TO_HEALTH_STATUS,
    FactEmbeddingTrainer,
)
from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.nlp.fact_encoder import FactEncoder, HuggingfaceModels

from medvqa.training.utils import append_metric_name, run_common_boilerplate_code_and_start_training
from medvqa.utils.common import WORKSPACE_DIR, DictWithDefault
from medvqa.metrics import (
    attach_condition_aware_loss,
    attach_condition_aware_triplet_accuracy,
    attach_condition_aware_accuracy,
)
from medvqa.models.checkpoint import load_metadata

from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.files import get_checkpoint_folder_path
from medvqa.training.fact_embedding import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    get_fact_embedding_collate_batch_fn,
)
from medvqa.metrics.utils import get_merge_metrics_fn
from medvqa.utils.logging import CountPrinter, print_blue

class _TripletRuleWeightsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            parsed_dict = ast.literal_eval(values)
            try:
                # Check that parsed_dict is a dictionary mapping string keys to non-empty lists of floats
                assert isinstance(parsed_dict, dict)
                for key, value in parsed_dict.items():
                    assert isinstance(key, str)
                    assert isinstance(value, list)
                    assert len(value) > 0
                    for v in value:
                        assert isinstance(v, float)
                # If all checks pass, set the attribute
                setattr(namespace, self.dest, parsed_dict)
            except AssertionError:
                # If any of the checks fail, raise an error
                raise ValueError("Invalid input format. Must be a dictionary mapping string keys to non-empty lists of floats")
        except (ValueError, SyntaxError) as e:
            raise ValueError("Invalid input format. Error: {}".format(str(e)))

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
    parser.add_argument('--huggingface_model_name', type=str, default=None, choices=HuggingfaceModels.get_all())
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--pretrained_checkpoint_folder_path', type=str, default=None)
    
    # Optimization arguments
    parser.add_argument('--optimizer_name', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='reduce-lr-on-plateau')
    parser.add_argument('--lr_decay', type=float, default=0.76, help='Learning rate decay')
    parser.add_argument('--lr_decay_patience', type=int, default=2, help='Learning rate decay patience')
    parser.add_argument('--warmup_and_decay_args', type=str, default=None)
    parser.add_argument('--warmup_and_cosine_args', type=str, default=None)
    parser.add_argument('--warmup_decay_and_cyclic_decay_args', type=str, default=None)
    parser.add_argument('--iters_to_accumulate', type=int, default=1, help='For gradient accumulation')
    parser.add_argument('--override_lr', action='store_true', default=False)

    # Dataset and dataloading arguments
    parser.add_argument('--triplets_filepath', type=str, default=None, help='Path to triplets file')
    parser.add_argument('--triplet_rule_weights', type=str, action=_TripletRuleWeightsAction, help='Weights for triplet rules')
    parser.add_argument('--integrated_facts_metadata_jsonl_filepath', type=str, default=None, help='Path to integrated facts metadata file')
    parser.add_argument('--paraphrases_jsonl_filepaths', type=str, nargs='+', default=None, help='Path to paraphrases files')
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of dataset to use')
    parser.add_argument('--triplets_weight', type=float, default=1.0, help='Weight for triplet sampling during training')
    parser.add_argument('--classification_weight', type=float, default=1.0, help='Weight for classification sampling during training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for parallel dataloading')    
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # Checkpoint saving arguments
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no_save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = DictWithDefault(1.0) # Default weight is 1.0
   
def train_model(
        model_kwargs,
        optimizer_kwargs,
        lr_scheduler_kwargs,
        fact_embedding_trainer_kwargs,
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
    count_print('Creating instance of Seq2SeqModel ...')
    model = FactEncoder(**model_kwargs)
    model = model.to(device)

    # Optimizer
    count_print('Defining optimizer ...')
    optimizer = create_optimizer(params=model.parameters(), **optimizer_kwargs)

    # Learning rate scheduler
    count_print('Defining scheduler ...')
    lr_scheduler, update_lr_batchwise = create_lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)

    # Create trainer and validator engines
    count_print('Creating trainer and validator engines ...')
    trainer_engine = get_engine(model=model, optimizer=optimizer, device=device, 
                                update_lr_batchwise=update_lr_batchwise, lr_scheduler=lr_scheduler, **trainer_engine_kwargs)
    validator_engine = get_engine(model=model, device=device, **validator_engine_kwargs)
    
    # Define collate batch functions
    count_print('Defining collate batch functions ...')
    triplet_collate_batch_fn = get_fact_embedding_collate_batch_fn(**collate_batch_fn_kwargs['triplets'])
    classification_collate_batch_fn = get_fact_embedding_collate_batch_fn(**collate_batch_fn_kwargs['classification'])  
    
    # Create dataloaders
    fact_embedding_trainer = FactEmbeddingTrainer(
        batch_size=batch_size,
        triplet_collate_batch_fn=triplet_collate_batch_fn,
        classification_collate_batch_fn=classification_collate_batch_fn,
        num_workers=num_workers,
        **fact_embedding_trainer_kwargs,
    )
    
    _train_weights = []
    _train_dataloaders = []
    _train_dataloaders.append(fact_embedding_trainer.train_triplet_dataloader)
    _train_weights.append(dataloading_kwargs['triplets_weight'])
    _train_dataloaders.append(fact_embedding_trainer.train_classification_dataloader)
    _train_weights.append(dataloading_kwargs['classification_weight'])
    train_dataloader = balanced_dataloaders_generator(_train_dataloaders, _train_weights)
    
    val_dataloader = fact_embedding_trainer.val_dataloader
    val_dataloader_size = len(val_dataloader)

    trainer_name = fact_embedding_trainer.name
    print('fact_embedding_trainer.name = ', trainer_name)
    
    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []
    
    attach_condition_aware_loss(trainer_engine, 'loss')
    metrics_to_print.append('loss')

    attach_condition_aware_loss(trainer_engine, 'triplet_loss', condition_function=lambda x: x['flag'] == 't')
    metrics_to_print.append('triplet_loss')

    for rule_id in fact_embedding_trainer.rule_ids:
        metric_name = f'tacc({rule_id})'
        attach_condition_aware_triplet_accuracy(validator_engine, rule_id, metric_name)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, metric_name, train=False)

    _cond_func = lambda x: x['flag'] == 'c' # only consider classification samples
    
    attach_condition_aware_loss(trainer_engine, 'c_loss', condition_function=_cond_func)
    metrics_to_print.append('c_loss') # category classification loss
    attach_condition_aware_loss(trainer_engine, 'hs_loss', condition_function=_cond_func)
    metrics_to_print.append('hs_loss') # health status classification loss
    attach_condition_aware_loss(trainer_engine, 'cs_loss', condition_function=_cond_func)
    metrics_to_print.append('cs_loss') # comparison status classification loss

    attach_condition_aware_accuracy(trainer_engine, pred_field_name='pred_category', gt_field_name='gt_category',
                                    condition_function=_cond_func, metric_name='cacc')
    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cacc', val=False)

    attach_condition_aware_accuracy(trainer_engine, pred_field_name='pred_health_status', gt_field_name='gt_health_status',
                                    condition_function=_cond_func, metric_name='hsacc')
    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'hsacc', val=False)

    attach_condition_aware_accuracy(trainer_engine, pred_field_name='pred_comparison_status', gt_field_name='gt_comparison_status',
                                    condition_function=_cond_func, metric_name='csacc')
    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'csacc', val=False)

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
            fact_embedding_trainer_kwargs=fact_embedding_trainer_kwargs,
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
    embedding_size,
    pretrained_checkpoint_folder_path,
    # Optimizer args
    optimizer_name,
    lr,
    # lr_scheduler args
    scheduler,
    lr_decay,
    lr_decay_patience,
    warmup_and_decay_args,
    warmup_and_cosine_args,
    warmup_decay_and_cyclic_decay_args,
    # Dataset args
    triplets_filepath,
    triplet_rule_weights,
    integrated_facts_metadata_jsonl_filepath,
    paraphrases_jsonl_filepaths,
    dataset_name,
    triplets_weight,
    classification_weight,
    # Dataloading args
    batch_size,
    num_workers,
    # Fixed traning args
    use_amp,
    iters_to_accumulate,
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
        embedding_size=embedding_size,
        pretrained_checkpoint_folder_path=pretrained_checkpoint_folder_path,
        classify_category=True,
        n_categories=len(_LABEL_TO_CATEGORY),
        classify_health_status=True,
        n_health_statuses=len(_LABEL_TO_HEALTH_STATUS),
        classify_comparison_status=True,
        n_comparison_statuses=len(_LABEL_TO_COMPARISON_STATUS),
    )

    optimizer_kwargs = dict(
        name=optimizer_name,
        lr=lr,
    )

    lr_scheduler_kwargs = dict(
        name=scheduler,
        factor=lr_decay,
        patience=lr_decay_patience,
        warmup_and_decay_args=warmup_and_decay_args,
        warmup_and_cosine_args=warmup_and_cosine_args,
        warmup_decay_and_cyclic_decay_args=warmup_decay_and_cyclic_decay_args,
        n_batches_per_epoch=batches_per_epoch,
    )
    
    dataloading_kwargs = dict(
        batch_size=batch_size,
        triplets_weight=triplets_weight,
        classification_weight=classification_weight,
    )
    
    fact_embedding_trainer_kwargs = dict(
        triplets_filepath=triplets_filepath,
        triplet_rule_weights=triplet_rule_weights,
        integrated_facts_metadata_jsonl_filepath=integrated_facts_metadata_jsonl_filepath,
        paraphrases_jsonl_filepaths=paraphrases_jsonl_filepaths,
        dataset_name=dataset_name,
    )

    collate_batch_fn_kwargs = dict(
        triplets=dict(
            huggingface_model_name=huggingface_model_name,
            for_triplet_ranking=True,
        ),
        classification=dict(
            huggingface_model_name=huggingface_model_name,
            for_classification=True,
        ),
    )

    trainer_engine_kwargs = dict(
        iters_to_accumulate=iters_to_accumulate,
        use_amp=use_amp,
        training=True,
    )

    validator_engine_kwargs = dict(
        use_amp=use_amp,
        validating=True,
    )
    
    training_kwargs = dict(
        use_amp=use_amp,
    )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                fact_embedding_trainer_kwargs=fact_embedding_trainer_kwargs,
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
        lr_decay,
        lr_decay_patience,
        warmup_and_decay_args,
        warmup_and_cosine_args,
        warmup_decay_and_cyclic_decay_args,
        num_workers,    
        epochs = 1,
        batches_per_epoch = 1000,
        device = 'GPU',
        save = True,
        override_lr = False,
        **unused_kwargs,
        ):
    print_blue('----- Resuming training ------', bold=True)

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    fact_embedding_trainer_kwargs = metadata['fact_embedding_trainer_kwargs']
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
            factor = lr_decay,
            patience = lr_decay_patience,
            warmup_and_decay_args = warmup_and_decay_args,
            warmup_and_cosine_args = warmup_and_cosine_args,
            warmup_decay_and_cyclic_decay_args = warmup_decay_and_cyclic_decay_args,
            n_batches_per_epoch = batches_per_epoch,
        )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                fact_embedding_trainer_kwargs=fact_embedding_trainer_kwargs,
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