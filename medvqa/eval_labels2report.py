from copy import deepcopy
import  os
import argparse

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.mimiccxr import load_mimiccxr_reports_detailed_metadata
from medvqa.datasets.mimiccxr.mimiccxr_labels2report_dataset_management import MIMICCXR_Labels2ReportTrainer
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.evaluation.report_generation import compute_report_level_metrics
from medvqa.models.report_generation.labels2report import Labels2ReportModel

from medvqa.utils.constants import (
    DATASET_NAMES,
    MIMICCXR_DATASET_ID,
    MetricNames,
)
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_condition_aware_t5_report_logger,
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
from medvqa.utils.handlers import (
    attach_accumulator,
    get_log_metrics_handler,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
)
from medvqa.utils.files import (
    get_results_folder_path,
    save_pickle,
)
from medvqa.training.labels2report import get_engine
from medvqa.datasets.dataloading_utils import get_labels2report_collate_batch_fn
from medvqa.utils.logging import CountPrinter, print_blue, print_bold

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
    parser.add_argument('--save-input-labels', action='store_true', default=False, help='Save input labels to disk')
    parser.add_argument('--force-gt-input-labels', action='store_true', default=False, help='Force ground truth as input labels')
    parser.add_argument('--num-beams', type=int, default=1, help='Number of beams for beam search')
    
    return parser.parse_args(args=args)

def _compute_and_save_report_level_metrics__mimiccxr(results_dict, eval_dataset_name, tokenizer, qa_adapted_reports_filename,
                                                     results_folder_path, max_processes, save_reports=False,
                                                     save_input_labels=False, get_input_label_breakdown=None,
                                                     force_gt_input_labels=False, use_t5=False, num_beams=None):
    idxs = results_dict['mimiccxr_metrics']['idxs']
    dataset = results_dict['mimiccxr_dataset']
    if use_t5:
        assert num_beams is not None
        gt_reports = [dataset.reports[dataset.report_ids[idx]] for idx in idxs]
        assert type(gt_reports[0]) == str
        gen_reports = results_dict['mimiccxr_metrics']['pred_reports']
        assert type(gen_reports[0]) == str
    else:
        gt_reports = [tokenizer.ids2string(tokenizer.clean_sentence(dataset.reports[dataset.report_ids[idx]])) for idx in idxs]
        gen_reports = [tokenizer.ids2string(x) for x in results_dict['mimiccxr_metrics']['pred_reports']]
    report_metrics = compute_report_level_metrics(gt_reports, gen_reports, tokenizer, max_processes=max_processes)
    if force_gt_input_labels:
        eval_dataset_name = f'{eval_dataset_name}(gt)'
    if use_t5:
        save_path = os.path.join(results_folder_path, f'{eval_dataset_name}_report_level_metrics(num_beams={num_beams}).pkl')
    else:
        save_path = os.path.join(results_folder_path, f'{eval_dataset_name}_report_level_metrics.pkl')
    save_pickle(report_metrics, save_path)
    print(f'Report-level metrics successfully saved to ', end='')
    print_bold(save_path)
    if save_reports:
        if use_t5:
            save_path = os.path.join(results_folder_path, f'{eval_dataset_name}_reports(num_beams={num_beams}).pkl')
        else:
            save_path = os.path.join(results_folder_path, f'{eval_dataset_name}_reports.pkl')
        metadata = load_mimiccxr_reports_detailed_metadata(qa_adapted_reports_filename)
        gt_report_paths = [metadata['filepaths'][dataset.report_ids[idx]] for idx in idxs]
        to_save = { 'gen_reports': gen_reports, 'gt_reports': gt_reports, 'gt_report_paths': gt_report_paths }
        save_pickle(to_save, save_path)
        print(f'Generated reports successfully saved to ', end='')
        print_bold(save_path)
    if save_input_labels:
        assert get_input_label_breakdown is not None
        save_path = os.path.join(results_folder_path, f'{eval_dataset_name}_input_labels.pkl')
        if os.path.exists(save_path):
            print(f'Input labels already exist at ', end='')
            print_bold(save_path)
        else:
            input_label_breakdown = get_input_label_breakdown(idxs)
            save_pickle(input_label_breakdown, save_path)
            print(f'Input labels successfully saved to ', end='')
            print_bold(save_path)
    return report_metrics

def evaluate_model(
        model_kwargs,
        tokenizer_kwargs,
        mimiccxr_trainer_kwargs,
        dataloading_kwargs,
        collate_batch_fn_kwargs,
        training_kwargs,
        validator_engine_kwargs,
        auxiliary_tasks_kwargs,
        num_workers,
        device='GPU',
        checkpoint_folder_path=None,
        max_chexpert_labeler_processes=8,
        num_beams=1,
        save_reports=False,
        save_input_labels=False,
        force_gt_input_labels=False,
        ):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    batch_size = dataloading_kwargs['batch_size']
    eval_mimiccxr = training_kwargs['train_mimiccxr']
    assert eval_mimiccxr, 'MIMIC-CXR must be used for evaluation'
    use_t5 = training_kwargs['use_t5']
    
    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs['classify_chest_imagenome']

    # Sanity check
    if force_gt_input_labels:
        assert training_kwargs['train_on_gt_and_eval_on_predictions']
        assert mimiccxr_trainer_kwargs['use_ensemble_predictions']
        assert mimiccxr_trainer_kwargs['precomputed_sigmoid_paths'] is not None
        assert mimiccxr_trainer_kwargs['use_hard_predictions']
        assert mimiccxr_trainer_kwargs['precomputed_thresholds_paths'] is not None
        assert not model_kwargs['support_two_label_sources']

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    tokenizer = Tokenizer(**tokenizer_kwargs)
    
    # Create model
    count_print('Creating instance of Labels2ReportModel ...')
    model = Labels2ReportModel(vocab_size=tokenizer.vocab_size,
                               start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                               device=device, **model_kwargs)
    model = model.to(device)

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    engine_kwargs = validator_engine_kwargs.copy()
    engine_kwargs['validating'] = False
    engine_kwargs['testing'] = True
    engine_kwargs['model'] = model
    engine_kwargs['device'] = device
    engine_kwargs['num_beams'] = num_beams
    if use_t5:
        from transformers import T5Tokenizer
        t5_tokenizer = T5Tokenizer.from_pretrained(model_kwargs['t5_model_name'])
        _tokenizer = t5_tokenizer
    else:
        _tokenizer = tokenizer
    engine_kwargs['tokenizer'] = _tokenizer
    evaluator_engine = get_engine(**engine_kwargs)
    
    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    if eval_mimiccxr:
        kwargs = collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR]
        if force_gt_input_labels:
            assert not kwargs.get('randomly_drop_labels', False)
            assert not kwargs['use_ground_truth_as_prediction']
            kwargs['use_ground_truth_as_prediction'] = True
        mimiccxr_collate_batch_fn = get_labels2report_collate_batch_fn(**kwargs)

    # Create MIMIC-CXR trainer
    if eval_mimiccxr:
        count_print('Creating MIMIC-CXR Labels2Report trainer ...')
        mimiccxr_trainer_kwargs['use_test_set'] = True
        if force_gt_input_labels:
            mimiccxr_trainer_kwargs['use_ensemble_predictions'] = False
            mimiccxr_trainer_kwargs['precomputed_sigmoid_paths'] = None
            mimiccxr_trainer_kwargs['use_hard_predictions'] = False
            mimiccxr_trainer_kwargs['precomputed_thresholds_paths'] = None
            mimiccxr_trainer_kwargs['filter_labels'] = False
            if mimiccxr_trainer_kwargs['use_chest_imagenome']:
                mimiccxr_trainer_kwargs['reorder_chest_imagenome_labels'] = True
        mimiccxr_trainer = MIMICCXR_Labels2ReportTrainer(
            tokenizer=tokenizer,
            batch_size=batch_size,
            collate_batch_fn=mimiccxr_collate_batch_fn,            
            num_workers=num_workers,
            **mimiccxr_trainer_kwargs,
        )
    
    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    _mim_datasets = [MIMICCXR_DATASET_ID]
    _chexpert_labels_datasets = _mim_datasets
    metrics_to_print = []
    
    if use_t5:
        attach_condition_aware_t5_report_logger(evaluator_engine)
    else:
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
            save_reports=save_reports, save_input_labels=save_input_labels,
            get_input_label_breakdown=mimiccxr_trainer.get_input_labels_breakdown,
            force_gt_input_labels=force_gt_input_labels,
            use_t5=use_t5, num_beams=num_beams,
        )
        return results_dict

def evaluate(
        checkpoint_folder,
        num_workers,
        batch_size,
        device='GPU',
        max_chexpert_labeler_processes=8,
        save_reports=False,
        save_input_labels=False,
        force_gt_input_labels=False,
        num_beams=1,
        ):
    print_blue('----- Evaluating model ------', bold=True)

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
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
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                training_kwargs=training_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                auxiliary_tasks_kwargs=auxiliary_tasks_kwargs,
                num_workers=num_workers,
                device=device,
                checkpoint_folder_path=checkpoint_folder,
                max_chexpert_labeler_processes=max_chexpert_labeler_processes,
                num_beams=num_beams,
                save_reports=save_reports,
                save_input_labels=save_input_labels,
                force_gt_input_labels=force_gt_input_labels,
                )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)