import  os
import argparse
from copy import deepcopy

import torch
import torch.nn as nn

from ignite.engine import Events, Engine
from ignite.handlers.timing import Timer

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
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.utils.handlers import (
    get_log_metrics_handlers,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
    attach_accumulator,
)
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.models.vqa.open_ended_vqa import OpenEndedVQA
from medvqa.utils.files import (
    load_json_file,
    get_results_folder_path,
    save_to_pickle,
)
from medvqa.training.vqa import get_step_fn
from medvqa.datasets.dataloading_utils import (
    collate_batch_fn
)
from medvqa.datasets.mimiccxr.vqa import MIMICCXR_VQA_Evaluator
from medvqa.datasets.mimiccxr import (
    MIMICCXR_QA_ADAPTED_REPORTS_JSON_PATH,
)
from medvqa.datasets.iuxray.vqa import IUXRAY_VQA_Trainer
from medvqa.datasets.iuxray import (
    IUXRAY_QA_ADAPTED_REPORTS_JSON_PATH,
)
from medvqa.utils.images import get_image_transform
from medvqa.utils.logging import CountPrinter
from medvqa.evaluation.vqa import compute_aggregated_metrics

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
    parser.add_argument('--checkpoint-folder', type=str,
                        help='Relative path to folder with checkpoint to evaluate')

    # optional arguments    
    parser.add_argument('--batch-size', type=int, default=140,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')
    
    return parser.parse_args()

def _append_chexpert_labels(metrics_dict, pred_answers, gt_dataset, idxs, tokenizer):
    gt_answers = [tokenizer.ids2string(tokenizer.clean_sentence(gt_dataset.answers[i])) for i in idxs]
    pred_answers = [tokenizer.ids2string(x) for x in pred_answers]
    labeler = ChexpertLabeler()
    metrics_dict['chexpert_labels_gt'] = labeler.get_labels(gt_answers,
                                                            update_cache_on_disk=True)
    metrics_dict['chexpert_labels_gen'] = labeler.get_labels(pred_answers)

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


def _evaluate_model(
    tokenizer_kwargs,
    model_kwargs,
    mimiccxr_vqa_evaluator_kwargs,
    iuxray_vqa_trainer_kwargs,
    device = 'GPU',
    checkpoint_folder_path = None,
    return_results = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
):

    assert eval_iuxray or eval_mimiccxr
    
    count_print = CountPrinter()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Load qa adapted reports
    count_print('Loading iuxray and mimiccxr QA adapted reports ...')
    iuxray_qa_reports = load_json_file(IUXRAY_QA_ADAPTED_REPORTS_JSON_PATH)
    mimiccxr_qa_reports = load_json_file(MIMICCXR_QA_ADAPTED_REPORTS_JSON_PATH)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    vocab_min_freq = tokenizer_kwargs['vocab_min_freq']
    tokenizer = Tokenizer(vocab_filepath=f'vocab/id2tokens_vqa_mimiccxr+iuxray_minf={vocab_min_freq}.pickle',
                          qa_adapted_datasets=[iuxray_qa_reports, mimiccxr_qa_reports],
                          min_freq=vocab_min_freq)
    
    # Create model
    count_print('Creating instance of OpenEndedVQA model ...')    
    model = OpenEndedVQA(vocab_size=tokenizer.vocab_size,
                              start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                         device=device,
                         **model_kwargs)
    model = model.to(device)

    # Criterion
    count_print('Defining loss criterion ...')
    nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    eval_step = get_step_fn(model, None, nlg_criterion, tokenizer,
                            training=False, device=device)    
    evaluator = Engine(eval_step)    
    
    # Default image transform
    count_print('Defining image transform ...')
    img_transform = get_image_transform()

    # Create MIMIC-CXR vqa evaluator
    count_print('Creating MIMIC-CXR vqa evaluator ...')
    mimiccxr_vqa_evaluator = MIMICCXR_VQA_Evaluator(
        transform = img_transform,
        collate_batch_fn = collate_batch_fn,
        tokenizer = tokenizer,
        mimiccxr_qa_reports = mimiccxr_qa_reports,
        **mimiccxr_vqa_evaluator_kwargs,
    )
    
    # Create IU X-Ray vqa trainer
    count_print('Creating IU X-Ray vqa trainer ...')
    iuxray_vqa_trainer = IUXRAY_VQA_Trainer(
        transform = img_transform,
        collate_batch_fn = collate_batch_fn,
        tokenizer = tokenizer,
        iuxray_qa_reports = iuxray_qa_reports,
        **iuxray_vqa_trainer_kwargs,
    )

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    # metrics
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

    # losses
    attach_loss('loss', evaluator, device)
    
    # timer
    timer = Timer()
    timer.attach(evaluator, start=Events.EPOCH_STARTED)
    
    # logging
    log_metrics_handler = get_log_metrics_handlers(
        timer, metrics_to_print=['loss', 'bleu_question', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4',
                                 'rougeL', 'ciderD', 'medcomp', 'wmedcomp'])
    log_iteration_handler = get_log_iteration_handler()

    # load saved checkpoint
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True)

    # attaching handlers
    evaluator.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation

    if return_results:
        results_dict = dict(tokenizer = tokenizer)
        results_folder_path = get_results_folder_path(checkpoint_folder_path)

    if eval_iuxray:
        print('\n========================')
        count_print('Running evaluator engine on IU X-Ray validation split ...')
        print('len(dataset) =', len(iuxray_vqa_trainer.val_dataset))
        print('len(dataloader) =', len(iuxray_vqa_trainer.val_dataloader))
        evaluator.run(iuxray_vqa_trainer.val_dataloader)
        if return_results:
            results_dict['iuxray_metrics'] = deepcopy(evaluator.state.metrics)            
            results_dict['iuxray_dataset'] = iuxray_vqa_trainer.val_dataset
            _append_chexpert_labels(
                results_dict['iuxray_metrics'],
                results_dict['iuxray_metrics']['pred_answers'],
                results_dict['iuxray_dataset'],
                results_dict['iuxray_metrics']['idxs'],
                tokenizer,
            )
            results_dict['iuxray_agg_metrics'] = _compute_and_save_aggregated_metrics(
                                                    results_dict, 'iuxray', tokenizer,
                                                    _METRIC_NAMES, results_folder_path)

    if eval_mimiccxr:
        print('\n========================')
        count_print('Running evaluator engine on MIMIC-CXR test split ...')
        print('len(dataset) =', len(mimiccxr_vqa_evaluator.test_dataset))
        print('len(dataloader) =', len(mimiccxr_vqa_evaluator.test_dataloader))
        evaluator.run(mimiccxr_vqa_evaluator.test_dataloader)
        if return_results:
            results_dict['mimiccxr_metrics'] = deepcopy(evaluator.state.metrics)
            results_dict['mimiccxr_dataset'] = mimiccxr_vqa_evaluator.test_dataset
            _append_chexpert_labels(
                results_dict['mimiccxr_metrics'],
                results_dict['mimiccxr_metrics']['pred_answers'],
                results_dict['mimiccxr_dataset'],
                results_dict['mimiccxr_metrics']['idxs'],
                tokenizer,
            )
            results_dict['mimiccxr_agg_metrics'] = _compute_and_save_aggregated_metrics(
                                                    results_dict, 'mimiccxr', tokenizer,
                                                    _METRIC_NAMES, results_folder_path)

    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder,
    batch_size = 100,
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
    mimiccxr_vqa_evaluator_kwargs = metadata['mimiccxr_vqa_trainer_kwargs']
    mimiccxr_vqa_evaluator_kwargs['batch_size'] = batch_size
    iuxray_vqa_trainer_kwargs = metadata['iuxray_vqa_trainer_kwargs']
    iuxray_vqa_trainer_kwargs['batch_size'] = batch_size

    return _evaluate_model(
                tokenizer_kwargs,
                model_kwargs,
                mimiccxr_vqa_evaluator_kwargs,
                iuxray_vqa_trainer_kwargs,
                device = device,
                checkpoint_folder_path = checkpoint_folder,
                return_results = return_results,
                eval_iuxray = eval_iuxray,
                eval_mimiccxr = eval_mimiccxr,
            )

if __name__ == '__main__':

    args = parse_args()
    args = {k : v for k, v in vars(args).items() if v is not None}
    print('script\'s arguments:')
    for k, v in args.items():
        print(f'   {k}: {v}')
    evaluate_model(**args)