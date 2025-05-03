import  os
import argparse
import torch
import json

from ignite.engine import Events
from ignite.handlers.timing import Timer

from medvqa.datasets.seq2seq.seq2seq_dataset_management import Seq2SeqTaskNames, Seq2SeqTrainer
from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.checkpoint import load_model_state_dict
from medvqa.models.nlp.seq2seq import Seq2SeqModel, Seq2SeqModels

from medvqa.training.utils import append_metric_name
from medvqa.utils.common import WORKSPACE_DIR, DictWithDefault
from medvqa.metrics import (
    attach_condition_aware_loss,
    attach_condition_aware_seq2seq_exactmatch,
    # attach_condition_aware_seq2seq_output_logger,
    attach_loss,
)
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    load_metadata,
    save_metadata,
)
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.handlers_utils import (
    get_log_checkpoint_saved_handler,
    get_log_metrics_handler,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
    get_lr_sch_handler,
    get_checkpoint_handler,
)
from medvqa.utils.files_utils import (
    get_checkpoint_folder_path,
)
from medvqa.training.seq2seq import get_engine
from medvqa.datasets.dataloading_utils import (
    cyclic_dataloader_generator,
    get_seq2seq_collate_batch_fn,
)
from medvqa.metrics.utils import (
    get_merge_metrics_fn,
    get_hybrid_score_name,
)
from medvqa.utils.logging_utils import CountPrinter, print_blue, print_red

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # --- Required arguments

    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs the model will be trained')
    parser.add_argument('--batches_per_epoch', type=int, required=True, help='Number of batches per epoch')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')

    # --- Optional arguments

    parser.add_argument('--checkpoint_folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')

    # Model arguments
    parser.add_argument('--seq2seq_model_name', type=str, default=None, choices=Seq2SeqModels.get_all_models())
    parser.add_argument('--t5_model_name', type=str, default='t5-small')
    parser.add_argument('--bart_model_name', type=str, default='facebook/bart-base')
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
    parser.add_argument('--override_lr', action='store_true')

    # Data loading arguments
    parser.add_argument('--task_name', type=str, default=None, choices=Seq2SeqTaskNames.get_all())
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--multitask_name_list', type=str, nargs='+', default=None, choices=Seq2SeqTaskNames.get_all())
    parser.add_argument('--task2weight', type=json.loads, default={}, help='JSON mapping task names to weights')
    parser.add_argument('--val_size', type=int, default=200, help='Number of samples to use for validation')
    parser.add_argument('--mlm_min_token_count', type=int, default=20, help='Minimum number of tokens to mask for MLM')
    parser.add_argument('--mlm_masking_fraction', type=float, default=0.15, help='Fraction of tokens to mask for MLM')
    parser.add_argument('--integrated_facts_metadata_jsonl_filepath', type=str, default=None,
                        help='Path to json file with integrated fact metadata')
    parser.add_argument('--paraphrased_inputs_jsonl_filepaths', type=str, nargs='+', default=None,
                        help='List of paths to jsonl files with paraphrased inputs')
    parser.add_argument('--chest_imagenome_phrases2labels_filepath', type=str, default=None,
                        help='Path to pickle file with chest imagenome phrases to labels mapping')
    parser.add_argument('--sentence_to_facts_input_output_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--fact_to_metadata_input_output_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--fact_to_metadata_v2_input_output_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--fact_to_comparison_input_output_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--chest_imagenome_obs_input_output_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--chest_imagenome_anatloc_input_output_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--report_nli_input_output_train_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--report_nli_input_output_val_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--report_to_negative_facts_input_output_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--integrated_nli_jsonl_filepath', type=str, default=None)
    parser.add_argument('--integrated_report_facts_jsonl_filepath', type=str, default=None)
    parser.add_argument('--interpret_cxr__label_based_predictions_filepath', type=str, default=None)
    parser.add_argument('--interpret_cxr_challenge_data_dir', type=str, default=None)
    parser.add_argument('--mimiccxr_integrated_report_nli_data_filepath', type=str, default=None)
    parser.add_argument('--report_section_to_generate', type=str, default=None, choices=['impression', 'findings'])
    parser.add_argument('--include_public_test_in_train', action='store_true')
    parser.add_argument('--best_k_classes', type=int, default=None)
    parser.add_argument('--use_sentence2facts_for_nli', action='store_true')
    parser.add_argument('--use_anli', action='store_true')
    parser.add_argument('--use_multinli', action='store_true')
    parser.add_argument('--use_snli', action='store_true')
    parser.add_argument('--use_report_nli', action='store_true')
    parser.add_argument('--use_fact_based_reports_in_mlm', action='store_true')
    parser.add_argument('--use_report_nli_entailment_dataset', action='store_true')
    parser.add_argument('--use_report_nli_paraphrases_dataset', action='store_true')
    parser.add_argument('--use_numeric_templates', action='store_true')
    parser.add_argument('--only_validate_nli', action='store_true')
    parser.add_argument('--nli1_only_on_train', action='store_true')
    parser.add_argument('--nli1_only_on_val', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for parallel dataloading')    
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    parser.add_argument('--use_amp', action='store_true')

    # Checkpoint saving arguments
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no_save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = DictWithDefault(default=1.0) # Default weight is 1.0

def _metric_getter(metrics_dict, key):
    if '_loss' in key:
        return 1 / (1 + metrics_dict[key]) # convert loss to score
    return metrics_dict[key]
   
def train_model(
        model_kwargs,
        optimizer_kwargs,
        lr_scheduler_kwargs,
        seq2seq_trainer_kwargs,
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
    use_t5 = training_kwargs['use_t5']
    use_flan_t5 = training_kwargs['use_flan_t5']
    use_bart = training_kwargs['use_bart']
    nli1_only_on_val = seq2seq_trainer_kwargs['nli1_only_on_val']
    only_validate_nli = seq2seq_trainer_kwargs['only_validate_nli']

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Create model
    count_print('Creating instance of Seq2SeqModel ...')
    model = Seq2SeqModel(**model_kwargs)
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
    
    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    collate_batch_fn = get_seq2seq_collate_batch_fn(**collate_batch_fn_kwargs)

    # Create Seq2Seq trainer
    count_print('Creating Seq2SeqTrainer ...')
    seq2seq_trainer = Seq2SeqTrainer(
        batch_size=batch_size,
        collate_batch_fn=collate_batch_fn,
        num_workers=num_workers,
        **seq2seq_trainer_kwargs,
    )
    train_dataloader = cyclic_dataloader_generator(seq2seq_trainer.train_dataloader)
    val_dataloader = seq2seq_trainer.val_dataloader
    val_dataloader_size = len(val_dataloader)

    seq2seq_trainer_name = seq2seq_trainer.name
    print('seq2seq_trainer.name = ', seq2seq_trainer_name)
    
    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []

    attach_loss('loss', trainer_engine, device)
    # for logging
    metrics_to_print.append('loss')
    
    if use_t5 or use_flan_t5 or use_bart:
        attach_condition_aware_loss(trainer_engine, 'seq2seq_loss')
        attach_condition_aware_loss(validator_engine, 'seq2seq_loss')
        if only_validate_nli and nli1_only_on_val:
            if use_t5:
                from transformers import T5TokenizerFast
                tokenizer = T5TokenizerFast.from_pretrained(model_kwargs['model_name'])
            elif use_flan_t5:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_kwargs['model_name'])
            elif use_bart:
                from transformers import BartTokenizerFast
                tokenizer = BartTokenizerFast.from_pretrained(model_kwargs['model_name'])
            attach_condition_aware_seq2seq_exactmatch(validator_engine,
                                                      pred_field_name='pred_output_ids',
                                                      gt_field_name='gt_text', metric_name='exact_match',
                                                      tokenizer=tokenizer, check_is_prefix=True)
        # attach_condition_aware_seq2seq_output_logger(validator_engine, tokenizer, field_name='pred_output_ids')
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'seq2seq_loss')
        if only_validate_nli and nli1_only_on_val:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'exact_match', train=False)
    
    # Timer
    timer = Timer()
    timer.attach(trainer_engine, start=Events.EPOCH_STARTED)
    timer.attach(validator_engine, start=Events.EPOCH_STARTED)

    # Score function
    assert len(val_metrics_to_merge) > 0
    if len(train_metrics_to_merge) > 0:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0.1, 0.9, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(trainer_engine.state.metrics, validator_engine.state.metrics)
    else:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0, 1, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(validator_engine.state.metrics)

    # Learning rate scheduler
    if not update_lr_batchwise:
        count_print('Defining learning rate scheduler handler ...')
        lr_sch_handler = get_lr_sch_handler(lr_scheduler, lr_scheduler_kwargs['name'], score_fn=score_fn)    

    # Checkpoint saving
    model_wrapper = ModelWrapper(model, optimizer, lr_scheduler)
    pretrained_checkpoint_folder_path = model_kwargs.get('pretrained_checkpoint_folder_path', None)    
    if checkpoint_folder_path is None: # first time
        if save: # only if we want to save checkpoints to disk
            count_print('Defining checkpoint folder path ...')
            checkpoint_folder_path = get_checkpoint_folder_path('seq2seq', seq2seq_trainer_name, model.get_name())
            print_red('checkpoint_folder_path =', checkpoint_folder_path, bold=True)
            save_metadata(checkpoint_folder_path,
                        model_kwargs=model_kwargs,
                        optimizer_kwargs=optimizer_kwargs,
                        lr_scheduler_kwargs=lr_scheduler_kwargs,
                        seq2seq_trainer_kwargs=seq2seq_trainer_kwargs,
                        dataloading_kwargs=dataloading_kwargs,
                        collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                        training_kwargs=training_kwargs,
                        trainer_engine_kwargs=trainer_engine_kwargs,
                        validator_engine_kwargs=validator_engine_kwargs)
        if pretrained_checkpoint_folder_path is not None:
            count_print(f'Loading pretrained weights ...')
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


def train_from_scratch(
    # Model args
    seq2seq_model_name,
    t5_model_name,
    bart_model_name,
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
    integrated_facts_metadata_jsonl_filepath,
    paraphrased_inputs_jsonl_filepaths,
    chest_imagenome_phrases2labels_filepath,
    sentence_to_facts_input_output_jsonl_filepaths,
    fact_to_metadata_input_output_jsonl_filepaths,
    fact_to_metadata_v2_input_output_jsonl_filepaths,
    fact_to_comparison_input_output_jsonl_filepaths,
    chest_imagenome_obs_input_output_jsonl_filepaths,
    chest_imagenome_anatloc_input_output_jsonl_filepaths,
    report_nli_input_output_train_jsonl_filepaths,
    report_nli_input_output_val_jsonl_filepaths,
    report_to_negative_facts_input_output_jsonl_filepaths,
    integrated_nli_jsonl_filepath,
    integrated_report_facts_jsonl_filepath,
    use_sentence2facts_for_nli,
    use_anli,
    use_multinli,
    use_snli,
    use_report_nli,
    use_fact_based_reports_in_mlm,
    use_report_nli_entailment_dataset,
    use_report_nli_paraphrases_dataset,
    use_numeric_templates,
    task_name,
    experiment_name,
    multitask_name_list,
    task2weight,
    val_size,
    mlm_min_token_count,
    mlm_masking_fraction,
    only_validate_nli,
    nli1_only_on_train,
    nli1_only_on_val,
    interpret_cxr__label_based_predictions_filepath,
    interpret_cxr_challenge_data_dir,
    mimiccxr_integrated_report_nli_data_filepath,
    report_section_to_generate,
    include_public_test_in_train,
    best_k_classes,
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

    use_t5 = seq2seq_model_name == Seq2SeqModels.T5
    use_flan_t5 = seq2seq_model_name == Seq2SeqModels.FLAN_T5
    use_bart = seq2seq_model_name == Seq2SeqModels.BART
    
    if use_t5:
        assert t5_model_name is not None

    # Model
    if use_t5 or use_flan_t5:
        model_name = t5_model_name
    elif use_bart:
        model_name = bart_model_name
    else:
        model_name = None
    model_kwargs = dict(
        seq2seq_model_name=seq2seq_model_name,
        model_name=model_name,
        pretrained_checkpoint_folder_path=pretrained_checkpoint_folder_path,
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
    )
    
    seq2seq_trainer_kwargs = dict(
        integrated_facts_metadata_jsonl_filepath=integrated_facts_metadata_jsonl_filepath,
        paraphrased_inputs_jsonl_filepaths=paraphrased_inputs_jsonl_filepaths,
        chest_imagenome_phrases2labels_filepath=chest_imagenome_phrases2labels_filepath,
        sentence_to_facts_input_output_jsonl_filepaths=sentence_to_facts_input_output_jsonl_filepaths,
        fact_to_metadata_input_output_jsonl_filepaths=fact_to_metadata_input_output_jsonl_filepaths,
        fact_to_comparison_input_output_jsonl_filepaths=fact_to_comparison_input_output_jsonl_filepaths,
        fact_to_metadata_v2_input_output_jsonl_filepaths=fact_to_metadata_v2_input_output_jsonl_filepaths,
        report_to_negative_facts_input_output_jsonl_filepaths=report_to_negative_facts_input_output_jsonl_filepaths,
        chest_imagenome_obs_input_output_jsonl_filepaths=chest_imagenome_obs_input_output_jsonl_filepaths,
        chest_imagenome_anatloc_input_output_jsonl_filepaths=chest_imagenome_anatloc_input_output_jsonl_filepaths,
        integrated_nli_jsonl_filepath=integrated_nli_jsonl_filepath,
        use_sentence2facts_for_nli=use_sentence2facts_for_nli,
        use_anli=use_anli, use_multinli=use_multinli, use_snli=use_snli,
        use_report_nli=use_report_nli,
        report_nli_input_output_train_jsonl_filepaths=report_nli_input_output_train_jsonl_filepaths,
        report_nli_input_output_val_jsonl_filepaths=report_nli_input_output_val_jsonl_filepaths,
        task_name=task_name,
        experiment_name=experiment_name,
        multitask_name_list=multitask_name_list,
        task2weight=task2weight,
        use_fact_based_reports_in_mlm=use_fact_based_reports_in_mlm,
        use_report_nli_entailment_dataset=use_report_nli_entailment_dataset,
        use_report_nli_paraphrases_dataset=use_report_nli_paraphrases_dataset,
        integrated_report_facts_jsonl_filepath=integrated_report_facts_jsonl_filepath,
        mlm_min_token_count=mlm_min_token_count,
        mlm_masking_fraction=mlm_masking_fraction,
        only_validate_nli=only_validate_nli,
        nli1_only_on_train=nli1_only_on_train,
        nli1_only_on_val=nli1_only_on_val,
        val_size=val_size,
        interpret_cxr__label_based_predictions_filepath=interpret_cxr__label_based_predictions_filepath,
        interpret_cxr_challenge_data_dir=interpret_cxr_challenge_data_dir,
        mimiccxr_integrated_report_nli_data_filepath=mimiccxr_integrated_report_nli_data_filepath,
        report_section_to_generate=report_section_to_generate,
        include_public_test_in_train=include_public_test_in_train,
        best_k_classes=best_k_classes,
        use_numeric_templates=use_numeric_templates,
        filter_for_t5=use_t5 or use_flan_t5,
    )

    collate_batch_fn_kwargs = dict(
        use_t5=use_t5,
        use_flan_t5=use_flan_t5,
        use_bart=use_bart,
        model_name=model_name,
    )

    trainer_engine_kwargs = dict(
        iters_to_accumulate=iters_to_accumulate,
        use_amp=use_amp,
        training=True,
        use_t5=use_t5,
        use_flan_t5=use_flan_t5,
        use_bart=use_bart,
    )
    validator_engine_kwargs = dict(
        use_amp=use_amp,
        validating=True,
        use_t5=use_t5,
        use_flan_t5=use_flan_t5,
        use_bart=use_bart,
    )
    
    training_kwargs = dict(
        use_amp=use_amp,
        use_t5=use_t5,
        use_flan_t5=use_flan_t5,
        use_bart=use_bart,
    )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                seq2seq_trainer_kwargs=seq2seq_trainer_kwargs,
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
    seq2seq_trainer_kwargs = metadata['seq2seq_trainer_kwargs']
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
                seq2seq_trainer_kwargs=seq2seq_trainer_kwargs,
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