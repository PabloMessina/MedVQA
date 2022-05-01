import  os
import argparse
from copy import deepcopy

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer

from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.datasets.dataloading_utils import qa_collate_batch_fn
from medvqa.metrics import (
    attach_bleu_question,
    attach_bleu,
    attach_rougel,
    attach_ciderd,
    attach_medical_completeness,
    attach_weighted_medical_completeness,    
    attach_loss
)
from medvqa.metrics.medical.chexpert import ChexpertLabeler
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    load_metadata,
)
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.utils.common import (
    WORKSPACE_DIR,
    parsed_args_to_dict,
)    
from medvqa.utils.handlers import (
    get_log_metrics_handlers,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
    attach_accumulator,
)
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.models.qa import OpenEndedQA
from medvqa.utils.files import (
    get_cached_json_file,
    get_results_folder_path,
    save_to_pickle,
)
from medvqa.training.qa import get_engine
from medvqa.datasets.mimiccxr.mimiccxr_qa_dataset_management import MIMICCXR_QA_Evaluator
from medvqa.datasets.iuxray.iuxray_qa_dataset_management import IUXRAY_QA_Trainer
from medvqa.utils.logging import CountPrinter
from medvqa.evaluation.vqa import compute_aggregated_metrics
from medvqa.evaluation.report_generation import (
    recover_reports,
    compute_report_level_metrics,
)
from medvqa.datasets.mimiccxr import mimiccxr_vqa_dataset_management

_METRIC_NAMES = [
    'bleu_question',
    'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4',
    'rougeL',
    'ciderD',
    'chexpert_accuracy',
    'chexpert_prf1s',
    'medcomp',
    'wmedcomp',
]

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--checkpoint-folder', type=str, required=True,
                        help='Relative path to folder with checkpoint to evaluate')
    
    # optional arguments    
    parser.add_argument('--batch-size', type=int, default=140,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for parallel dataloading')

    parser.add_argument('--iuxray', dest='eval_iuxray', action='store_true')
    parser.add_argument('--no-iuxray', dest='eval_iuxray', action='store_false')
    parser.set_defaults(eval_iuxray=True)

    parser.add_argument('--mimiccxr', dest='eval_mimiccxr', action='store_true')
    parser.add_argument('--no-mimiccxr', dest='eval_mimiccxr', action='store_false')
    parser.set_defaults(eval_mimiccxr=True)
    
    return parser.parse_args()

def _append_chexpert_labels(metrics_dict, pred_answers, gt_dataset, idxs, tokenizer):
    gt_answers = [tokenizer.ids2string(tokenizer.clean_sentence(gt_dataset.answers[i])) for i in idxs]
    pred_answers = [tokenizer.ids2string(x) for x in pred_answers]
    labeler = ChexpertLabeler()
    metrics_dict['chexpert_labels_gt'] = labeler.get_labels(gt_answers,
                                                            update_cache_on_disk=True)
    metrics_dict['chexpert_labels_gen'] = labeler.get_labels(pred_answers,
                                                            update_cache_on_disk=True)

def _compute_and_save_aggregated_metrics(results_dict, dataset_name, tokenizer, metric_names,
                                         results_folder_path):    
    agg_metrics = compute_aggregated_metrics(metrics_dict=results_dict[f'{dataset_name}_metrics'],
                                             dataset=results_dict[f'{dataset_name}_dataset'],
                                             tokenizer=tokenizer,
                                             metric_names=metric_names)
    save_path = os.path.join(results_folder_path, f'{dataset_name}_metrics.pkl')
    save_to_pickle(agg_metrics, save_path)
    print (f'Aggregated metrics successfully saved to {save_path}')
    return agg_metrics

def _compute_and_save_report_level_metrics(results_dict, dataset_name, tokenizer, qa_adapted_dataset,
                                           results_folder_path):

    metrics = compute_report_level_metrics(results_dict[f'{dataset_name}_reports']['gt_reports'],
                                           results_dict[f'{dataset_name}_reports']['gen_reports'],
                                           tokenizer, qa_adapted_dataset)
    save_path = os.path.join(results_folder_path, f'{dataset_name}_report_level_metrics.pkl')
    save_to_pickle(metrics, save_path)
    print (f'Report-level metrics successfully saved to {save_path}')
    return metrics


def _evaluate_model(
    tokenizer_kwargs,
    model_kwargs,
    mimiccxr_qa_evaluator_kwargs,
    iuxray_qa_trainer_kwargs,
    batch_size,
    num_workers,
    device = 'GPU',
    checkpoint_folder_path = None,
    return_results = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
):

    assert eval_iuxray or eval_mimiccxr

    # Pull out some args from kwargs
    
    # Filenames
    iuxray_preprocessed_data_filename = iuxray_qa_trainer_kwargs['preprocessed_data_filename']
    iuxray_qa_adapted_reports_filename = iuxray_qa_trainer_kwargs['qa_adapted_reports_filename']    
    mimiccxr_qa_adapted_reports_filename = mimiccxr_qa_evaluator_kwargs['qa_adapted_reports_filename']
    assert iuxray_preprocessed_data_filename is not None
    assert iuxray_qa_adapted_reports_filename is not None
    assert mimiccxr_qa_adapted_reports_filename is not None
    
    count_print = CountPrinter()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Load qa adapted reports
    count_print('Loading iuxray and mimiccxr QA adapted reports ...')
    iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, iuxray_qa_adapted_reports_filename)
    mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, mimiccxr_qa_adapted_reports_filename)
    iuxray_qa_reports = get_cached_json_file(iuxray_qa_adapted_reports_path)
    mimiccxr_qa_reports = get_cached_json_file(mimiccxr_qa_adapted_reports_path)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    vocab_min_freq = tokenizer_kwargs['vocab_min_freq']
    tokenizer = Tokenizer(qa_adapted_filenames=[iuxray_qa_adapted_reports_filename,
                                                mimiccxr_qa_adapted_reports_filename],
                          qa_adapted_datasets=[iuxray_qa_reports, mimiccxr_qa_reports],
                          min_freq=vocab_min_freq)
    
    # Create model
    count_print('Creating instance of OpenEndedQA model ...')
    model = OpenEndedQA(vocab_size=tokenizer.vocab_size,
                        start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                        device=device,
                        **model_kwargs)
    model = model.to(device)

    # Criterion    
    count_print('Creating evaluator engine ...')
    evaluator = get_engine(model, tokenizer, device, training=False)

    # Create MIMIC-CXR qa evaluator
    if eval_mimiccxr:
        count_print('Creating MIMIC-CXR qa evaluator ...')
        mimiccxr_preprocessed_data_filename = \
            mimiccxr_vqa_dataset_management.get_test_preprocessing_save_path(
                mimiccxr_qa_adapted_reports_filename, tokenizer)
        mimiccxr_qa_evaluator = MIMICCXR_QA_Evaluator(
            batch_size = batch_size,
            collate_batch_fn = qa_collate_batch_fn,
            num_workers = num_workers,
            preprocessed_data_filename = mimiccxr_preprocessed_data_filename
        )
    
    # Create IU X-Ray qa trainer
    if eval_iuxray:
        count_print('Creating IU X-Ray qa trainer ...')
        iuxray_qa_trainer = IUXRAY_QA_Trainer(
            batch_size = batch_size,
            collate_batch_fn = qa_collate_batch_fn,
            num_workers = num_workers,
            preprocessed_data_filename = iuxray_preprocessed_data_filename,
            validation_only = True,
        )

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    # Metrics
    attach_bleu_question(evaluator, device, record_scores=return_results)
    attach_bleu(evaluator, device, record_scores=return_results, ks=[1,2,3,4])
    attach_rougel(evaluator, device, record_scores=return_results)
    attach_ciderd(evaluator, device, record_scores=return_results)
    attach_medical_completeness(evaluator, device, tokenizer, record_scores=return_results)
    attach_weighted_medical_completeness(evaluator, device, tokenizer, record_scores=return_results)    

    if return_results:
        attach_accumulator(evaluator, 'idxs')
        attach_accumulator(evaluator, 'pred_answers')
        attach_accumulator(evaluator, 'pred_questions')

    # Losses
    attach_loss('loss', evaluator, device)
    
    # Timer
    timer = Timer()
    timer.attach(evaluator, start=Events.EPOCH_STARTED)
    
    # Logging
    metrics_to_print=['loss', 'bleu_question', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4',
                      'rougeL', 'ciderD', 'medcomp', 'wmedcomp']

    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()

    # Load saved checkpoint
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True)

    # Attach handlers
    evaluator.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation

    metrics_to_aggregate = _METRIC_NAMES[:]

    if return_results:
        results_dict = dict(tokenizer = tokenizer)
        results_folder_path = get_results_folder_path(checkpoint_folder_path)

    if eval_iuxray:
        print('\n========================')
        count_print('Running evaluator engine on IU X-Ray validation split ...')
        print('len(dataset) =', len(iuxray_qa_trainer.val_dataset))
        print('len(dataloader) =', len(iuxray_qa_trainer.val_dataloader))
        evaluator.run(iuxray_qa_trainer.val_dataloader)
        if return_results:
            results_dict['iuxray_metrics'] = deepcopy(evaluator.state.metrics)            
            results_dict['iuxray_dataset'] = iuxray_qa_trainer.val_dataset
            _append_chexpert_labels(
                results_dict['iuxray_metrics'],
                results_dict['iuxray_metrics']['pred_answers'],
                results_dict['iuxray_dataset'],
                results_dict['iuxray_metrics']['idxs'],
                tokenizer,
            )
            results_dict['iuxray_agg_metrics'] = _compute_and_save_aggregated_metrics(
                                                    results_dict, 'iuxray', tokenizer,
                                                    metrics_to_aggregate, results_folder_path)
            results_dict['iuxray_reports'] = recover_reports(
                results_dict['iuxray_metrics'],
                results_dict['iuxray_dataset'],
                tokenizer, iuxray_qa_reports,
            )
            results_dict['iuxray_report_metrics'] = _compute_and_save_report_level_metrics(
                results_dict, 'iuxray', tokenizer, iuxray_qa_reports, results_folder_path)

    if eval_mimiccxr:
        print('\n========================')
        count_print('Running evaluator engine on MIMIC-CXR test split ...')
        print('len(dataset) =', len(mimiccxr_qa_evaluator.test_dataset))
        print('len(dataloader) =', len(mimiccxr_qa_evaluator.test_dataloader))
        evaluator.run(mimiccxr_qa_evaluator.test_dataloader)
        if return_results:
            results_dict['mimiccxr_metrics'] = deepcopy(evaluator.state.metrics)
            results_dict['mimiccxr_dataset'] = mimiccxr_qa_evaluator.test_dataset
            _append_chexpert_labels(
                results_dict['mimiccxr_metrics'],
                results_dict['mimiccxr_metrics']['pred_answers'],
                results_dict['mimiccxr_dataset'],
                results_dict['mimiccxr_metrics']['idxs'],
                tokenizer,
            )
            results_dict['mimiccxr_agg_metrics'] = _compute_and_save_aggregated_metrics(
                                                    results_dict, 'mimiccxr', tokenizer,
                                                    metrics_to_aggregate, results_folder_path)
            results_dict['mimiccxr_reports'] = recover_reports(
                results_dict['mimiccxr_metrics'],
                results_dict['mimiccxr_dataset'],
                tokenizer, mimiccxr_qa_reports,
            )
            results_dict['mimiccxr_report_metrics'] = _compute_and_save_report_level_metrics(
                results_dict, 'mimiccxr', tokenizer, mimiccxr_qa_reports, results_folder_path)

    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder,
    batch_size = 100,
    num_workers = 0,
    device = 'GPU',
    return_results = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
):
    print('----- Evaluating model ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)    
    metadata = load_metadata(checkpoint_folder)
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    model_kwargs = metadata['model_kwargs']
    mimiccxr_qa_evaluator_kwargs = metadata['mimiccxr_qa_trainer_kwargs']    
    iuxray_qa_trainer_kwargs = metadata['iuxray_qa_trainer_kwargs']

    return _evaluate_model(
                tokenizer_kwargs,
                model_kwargs,
                mimiccxr_qa_evaluator_kwargs,
                iuxray_qa_trainer_kwargs,
                device = device,
                batch_size = batch_size,
                num_workers = num_workers,
                checkpoint_folder_path = checkpoint_folder,
                return_results = return_results,
                eval_iuxray = eval_iuxray,
                eval_mimiccxr = eval_mimiccxr,
            )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)