from copy import deepcopy
import  os
import argparse

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.image_processing import get_image_transform
from medvqa.datasets.mimiccxr.mimiccxr_image2report_dataset_management import MIMICCXR_Image2ReportTrainer
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.eval_labels2report import _compute_and_save_report_level_metrics__mimiccxr
from medvqa.models.report_generation.image2report import Image2ReportModel

from medvqa.utils.constants import (
    DATASET_NAMES,
    MIMICCXR_DATASET_ID,
    MetricNames,
)
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_dataset_aware_ciderd,
    attach_dataset_aware_weighted_medical_completeness,
    attach_dataset_aware_chest_imagenome_labels_auc,
    attach_dataset_aware_chest_imagenome_labels_prcauc,
    attach_dataset_aware_chexpert_labels_auc,
    attach_dataset_aware_chexpert_labels_prcauc,
)
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    load_metadata,
)
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.handlers_utils import (
    attach_accumulator,
    get_log_metrics_handler,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
)
from medvqa.utils.files_utils import get_results_folder_path
from medvqa.training.image2report import get_engine
from medvqa.datasets.dataloading_utils import get_image2report_collate_batch_fn
from medvqa.utils.logging_utils import CountPrinter, print_blue

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # --- Required arguments
    
    parser.add_argument('--checkpoint-folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')

    # --- Optional arguments

    # Data loading arguments
    parser.add_argument('--batch-size', type=int, default=45, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for parallel dataloading')
    parser.add_argument('--max-chexpert-labeler-processes', type=int, default=8, help='Maximum number of processes to use for Chexpert labeler')
    parser.add_argument('--save-reports', action='store_true', default=False, help='Save generated reports to disk')
    
    return parser.parse_args(args=args)

def evaluate_model(
        model_kwargs,
        tokenizer_kwargs,
        mimiccxr_trainer_kwargs,
        val_image_transform_kwargs,
        dataloading_kwargs,
        collate_batch_fn_kwargs,
        training_kwargs,
        validator_engine_kwargs,
        auxiliary_tasks_kwargs,
        num_workers,
        device='GPU',
        checkpoint_folder_path=None,
        max_chexpert_labeler_processes=8,
        save_reports=False,
        ):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    batch_size = dataloading_kwargs['batch_size']
    eval_mimiccxr = training_kwargs['train_mimiccxr']
    assert eval_mimiccxr, 'MIMIC-CXR must be used for evaluation'
    
    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs['classify_chest_imagenome']

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    tokenizer = Tokenizer(**tokenizer_kwargs)

    # Create model
    count_print('Creating instance of Image2ReportModel ...')
    # temporary HACK: remove
    if 'input_pos_encoding_mode' not in model_kwargs:
        model_kwargs['input_pos_encoding_mode'] = 'sinusoidal'
    model = Image2ReportModel(vocab_size=tokenizer.vocab_size,
                              start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                              device=device, **model_kwargs)
    model = model.to(device)

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    evaluator_engine = get_engine(model=model, tokenizer=tokenizer, device=device, **validator_engine_kwargs)
    
    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    if eval_mimiccxr:
        mimiccxr_collate_batch_fn = get_image2report_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR])

    # Create MIMIC-CXR trainer
    if eval_mimiccxr:
        count_print('Creating MIMIC-CXR Image2Report trainer ...')
        mimiccxr_trainer_kwargs['data_augmentation_enabled'] = False
        mimiccxr_trainer = MIMICCXR_Image2ReportTrainer(
            tokenizer=tokenizer,
            batch_size=batch_size,
            collate_batch_fn=mimiccxr_collate_batch_fn,
            num_workers=num_workers,
            use_test_set=True,
            test_image_transform = get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            **mimiccxr_trainer_kwargs,
        )
    
    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    _mim_datasets = [MIMICCXR_DATASET_ID]
    _chexpert_labels_datasets = _mim_datasets
    metrics_to_print = []
    
    attach_dataset_aware_ciderd(evaluator_engine, _mim_datasets, field='reports')
    attach_dataset_aware_weighted_medical_completeness(evaluator_engine, tokenizer, _mim_datasets, field='reports')
    # for logging
    metrics_to_print.append(MetricNames.CIDER_D)
    metrics_to_print.append(MetricNames.WMEDCOMP)
    
    if classify_chexpert:
        attach_dataset_aware_chexpert_labels_auc(evaluator_engine, _chexpert_labels_datasets, 'cpu')
        attach_dataset_aware_chexpert_labels_prcauc(evaluator_engine, _chexpert_labels_datasets, 'cpu')
        # for logging
        metrics_to_print.append(MetricNames.CHXLABEL_AUC)
        metrics_to_print.append(MetricNames.CHXLABEL_PRCAUC)

    if classify_chest_imagenome:
        attach_dataset_aware_chest_imagenome_labels_auc(evaluator_engine, _mim_datasets, 'cpu')
        attach_dataset_aware_chest_imagenome_labels_prcauc(evaluator_engine, _mim_datasets, 'cpu')
        # for logging
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELAUC)        
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELPRCAUC)
    
    # Timer
    timer = Timer()
    timer.attach(evaluator_engine, start=Events.EPOCH_STARTED)

    # Checkpoint saving
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True)

    # Logging
    count_print('Defining log_metrics_handler ...')

    log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()
    
    # Attach handlers
    evaluator_engine.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    evaluator_engine.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    # Accumulators
    attach_accumulator(evaluator_engine, 'idxs')
    attach_accumulator(evaluator_engine, 'pred_reports')

    # Start evaluation
    if eval_mimiccxr:
        count_print('Running evaluation engine ...')
        evaluator_engine.run(mimiccxr_trainer.test_dataloader)

        results_dict = {}
        results_dict['mimiccxr_metrics'] = deepcopy(evaluator_engine.state.metrics)
        results_dict['mimiccxr_dataset'] = mimiccxr_trainer.test_dataset
        assert checkpoint_folder_path is not None
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        qa_adapted_reports_filename = mimiccxr_trainer_kwargs['qa_adapted_reports_filename']
        results_dict['mimiccxr_report_metrics'] = _compute_and_save_report_level_metrics__mimiccxr(
            results_dict, 'mimiccxr_test_set', tokenizer, qa_adapted_reports_filename,
            results_folder_path, max_processes=max_chexpert_labeler_processes,
            save_reports=save_reports,
        )
        return results_dict

def evaluate(
        checkpoint_folder,
        num_workers,
        batch_size,
        device='GPU',
        max_chexpert_labeler_processes=8,
        save_reports=False,
        ):
    print_blue('----- Evaluating model ------', bold=True)

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
    val_image_transform_kwargs = metadata['val_image_transform_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    training_kwargs = metadata['training_kwargs']
    validator_engine_kwargs = metadata['validator_engine_kwargs']                
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']
    
    dataloading_kwargs['batch_size'] = batch_size

    return evaluate_model(
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                training_kwargs=training_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                auxiliary_tasks_kwargs=auxiliary_tasks_kwargs,
                num_workers=num_workers,
                device=device,
                checkpoint_folder_path=checkpoint_folder,
                max_chexpert_labeler_processes=max_chexpert_labeler_processes,
                save_reports=save_reports,
                )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)