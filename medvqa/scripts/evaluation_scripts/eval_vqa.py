import  os
import argparse
from copy import deepcopy

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.models.common import AnswerDecoding

from medvqa.models.vqa.open_ended_vqa import QuestionEncoding
from medvqa.utils.constants import (
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID,
    MetricNames,
)
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.metrics import (
    attach_exactmatch_question,
    attach_bleu,
    attach_chexpert_labels_accuracy,
    attach_chexpert_labels_macroavgf1,
    attach_chexpert_labels_microavgf1,
    attach_chexpert_labels_roc_auc,
    attach_question_labels_f1score,
    attach_rougel,
    attach_meteor,
    attach_ciderd,
    attach_medical_completeness,
    attach_weighted_medical_completeness,
    attach_medical_tags_f1score,
    attach_chexpert_labels_accuracy,
    attach_dataset_aware_orientation_accuracy,
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
from medvqa.utils.handlers_utils import (
    get_log_metrics_handlers,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
    attach_accumulator,
)
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.models.vqa.open_ended_vqa import OpenEndedVQA
from medvqa.utils.files_utils import (
    get_cached_json_file,
    get_results_folder_path,
    save_to_pickle,
)
from medvqa.training.vqa import get_engine
from medvqa.datasets.dataloading_utils import get_vqa_collate_batch_fn
from medvqa.datasets.mimiccxr.mimiccxr_vqa_dataset_management import MIMICCXR_VQA_Evaluator
from medvqa.datasets.iuxray.iuxray_vqa_dataset_management import IUXRAY_VQA_Trainer
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging_utils import CountPrinter
from medvqa.evaluation.vqa import compute_aggregated_metrics

_METRIC_NAMES = [
    MetricNames.BLEU,
    MetricNames.ROUGE_L,
    MetricNames.METEOR,
    MetricNames.CIDER_D,
    MetricNames.CHEXPERT_ACCURACY,
    MetricNames.CHEXPERT_PRF1S,
    MetricNames.MEDCOMP,
    MetricNames.WMEDCOMP,
]

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--checkpoint-folder', type=str,
                        help='Relative path to folder with checkpoint to evaluate')

    # optional arguments
    parser.add_argument('--answer-decoding', type=str, default='greedy-search')
    parser.add_argument('--beam-search-k', type=int, default=None)
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

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)
    
    return parser.parse_args()

def _append_chexpert_labels(metrics_dict, pred_answers, gt_dataset, idxs, tokenizer):
    gt_answers = [tokenizer.ids2string(tokenizer.clean_sentence(gt_dataset.answers[i])) for i in idxs]
    pred_answers = [tokenizer.ids2string(x) for x in pred_answers]
    labeler = ChexpertLabeler()
    metrics_dict['chexpert_labels_gt'] = labeler.get_labels(gt_answers,
                                                            update_cache_on_disk=True)
    metrics_dict['chexpert_labels_gen'] = labeler.get_labels(pred_answers)

def _compute_and_save_aggregated_metrics(results_dict, dataset_name, tokenizer, metric_names,
                                         results_folder_path, one_hot_questions=False, qa_reports=None):
    agg_metrics = compute_aggregated_metrics(metrics_dict=results_dict[f'{dataset_name}_metrics'],
                                             dataset=results_dict[f'{dataset_name}_dataset'],
                                             tokenizer=tokenizer,
                                             metric_names=metric_names,
                                             one_hot_questions=one_hot_questions,
                                             qa_reports=qa_reports)
    save_path = os.path.join(results_folder_path, f'{dataset_name}_metrics.pkl')
    save_to_pickle(agg_metrics, save_path)
    print (f'Aggregated metrics successfully saved to {save_path}')
    return agg_metrics

def _evaluate_model(
    tokenizer_kwargs,
    model_kwargs,
    mimiccxr_vqa_evaluator_kwargs,
    iuxray_vqa_trainer_kwargs,
    dataloading_kwargs,
    auxiliary_tasks_kwargs,
    answer_decoding,
    beam_search_k = None,
    num_workers = 0,
    device = 'GPU',
    checkpoint_folder_path = None,
    return_results = False,
    use_amp = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
):

    assert eval_iuxray or eval_mimiccxr

    # Pull out some args from kwargs
    question_encoding = model_kwargs.get('question_encoding', QuestionEncoding.BILSTM)
    verbose_question = question_encoding != QuestionEncoding.ONE_HOT

    # auxiliary task: medical tags prediction
    classify_tags = auxiliary_tasks_kwargs['classify_tags']
    n_medical_tags = auxiliary_tasks_kwargs['n_medical_tags']
    iuxray_medical_tags_per_report_filename = auxiliary_tasks_kwargs['iuxray_medical_tags_per_report_filename']
    mimiccxr_medical_tags_per_report_filename = auxiliary_tasks_kwargs['mimiccxr_medical_tags_per_report_filename']
    if classify_tags:
        assert n_medical_tags is not None
        assert iuxray_medical_tags_per_report_filename is not None
        assert mimiccxr_medical_tags_per_report_filename is not None
    
    # auxiliary task: orientation classification
    classify_orientation = auxiliary_tasks_kwargs['classify_orientation']

    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    iuxray_chexpert_labels_filename = auxiliary_tasks_kwargs['iuxray_chexpert_labels_filename']
    mimiccxr_chexpert_labels_filename = auxiliary_tasks_kwargs['mimiccxr_chexpert_labels_filename']

    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)
    n_questions_aux_task = auxiliary_tasks_kwargs.get('n_questions_aux_task', None)
    iuxray_question_labels_filename = auxiliary_tasks_kwargs.get('iuxray_question_labels_filename', None)
    mimiccxr_question_labels_filename = auxiliary_tasks_kwargs.get('mimiccxr_question_labels_filename', None)
    if classify_questions:
        assert n_questions_aux_task is not None
        if eval_iuxray: assert iuxray_question_labels_filename is not None
        if eval_mimiccxr: assert mimiccxr_question_labels_filename is not None
    
    # QA dataset filenames
    iuxray_qa_adapted_reports_filename = iuxray_vqa_trainer_kwargs['qa_adapted_reports_filename']
    mimiccxr_qa_adapted_reports_filename = mimiccxr_vqa_evaluator_kwargs['qa_adapted_reports_filename']
    assert iuxray_qa_adapted_reports_filename is not None
    assert mimiccxr_qa_adapted_reports_filename is not None
    
    if answer_decoding == AnswerDecoding.BEAM_SEARCH:
        assert beam_search_k is not None

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
    medical_tokenization = tokenizer_kwargs['medical_tokenization']
    medical_terms_frequency_filename = tokenizer_kwargs['medical_terms_frequency_filename']
    assert medical_tokenization == (medical_terms_frequency_filename is not None)
    tokenizer = Tokenizer(qa_adapted_dataset_paths=[iuxray_qa_adapted_reports_path,
                                                    mimiccxr_qa_adapted_reports_path],
                          min_freq=vocab_min_freq,
                          medical_terms_frequency_filename=medical_terms_frequency_filename)
    
    # Create model
    count_print('Creating instance of OpenEndedVQA model ...')    
    model = OpenEndedVQA(vocab_size=tokenizer.vocab_size,
                         start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                         padding_idx=tokenizer.token2id[tokenizer.PAD_TOKEN],
                         eos_idx=tokenizer.token2id[tokenizer.END_TOKEN],
                         device=device, **model_kwargs)
    model = model.to(device)

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    evaluator = get_engine(model, tokenizer, classify_tags, classify_orientation, classify_chexpert,
                           classify_questions, question_encoding, answer_decoding,
                           device, beam_search_k=beam_search_k, use_amp=use_amp, training=False)
    
    # Default image transform
    count_print('Defining image transform ...')
    img_transform = get_image_transform()

    # Define collate_batch_fn    
    one_hot_question_offsets = dataloading_kwargs.get('one_hot_question_offsets', None)
    if not verbose_question: assert one_hot_question_offsets is not None

    mimiccxr_collate_batch_fn = get_vqa_collate_batch_fn(MIMICCXR_DATASET_ID,
                                                    verbose_question = verbose_question,
                                                    one_hot_question_offset = one_hot_question_offsets[str(MIMICCXR_DATASET_ID)],
                                                    classify_tags = classify_tags,
                                                    n_tags = n_medical_tags,
                                                    classify_orientation = classify_orientation,
                                                    classify_chexpert = classify_chexpert,
                                                    classify_questions = classify_questions)
    iuxray_collate_batch_fn = get_vqa_collate_batch_fn(IUXRAY_DATASET_ID,
                                                   verbose_question = verbose_question,
                                                   one_hot_question_offset = one_hot_question_offsets[str(IUXRAY_DATASET_ID)],
                                                   classify_tags = classify_tags,
                                                   n_tags = n_medical_tags,
                                                   classify_orientation = classify_orientation,
                                                   classify_chexpert = classify_chexpert,
                                                   classify_questions = classify_questions)

    # Create MIMIC-CXR vqa evaluator
    if eval_mimiccxr:
        count_print('Creating MIMIC-CXR vqa evaluator ...')
        mimiccxr_vqa_evaluator = MIMICCXR_VQA_Evaluator(
            transform = img_transform,
            collate_batch_fn = mimiccxr_collate_batch_fn,
            num_workers = num_workers,
            tokenizer = tokenizer,
            mimiccxr_qa_reports = mimiccxr_qa_reports,
            classify_tags = classify_tags,
            medical_tags_per_report_filename = mimiccxr_medical_tags_per_report_filename,
            classify_orientation = classify_orientation,
            classify_chexpert = classify_chexpert,
            chexpert_labels_filename = mimiccxr_chexpert_labels_filename,
            classify_questions = classify_questions,
            question_labels_filename = mimiccxr_question_labels_filename,
            verbose_question = verbose_question,
            **mimiccxr_vqa_evaluator_kwargs,
        )
    
    # Create IU X-Ray vqa trainer
    if eval_iuxray:
        count_print('Creating IU X-Ray vqa trainer ...')
        iuxray_vqa_trainer = IUXRAY_VQA_Trainer(
            transform = img_transform,
            collate_batch_fn = iuxray_collate_batch_fn,
            num_workers = num_workers,
            tokenizer = tokenizer,        
            iuxray_qa_reports = iuxray_qa_reports,
            classify_tags = classify_tags,
            medical_tags_per_report_filename = iuxray_medical_tags_per_report_filename,
            classify_orientation = classify_orientation,
            classify_chexpert = classify_chexpert,
            chexpert_labels_filename = iuxray_chexpert_labels_filename,
            classify_questions = classify_questions,
            question_labels_filename = iuxray_question_labels_filename,
            validation_only = True,
            ignore_medical_tokenization = tokenizer.medical_tokenization,
            verbose_question = verbose_question,
            **iuxray_vqa_trainer_kwargs,
        )

    # Attach metrics, timer and events to engines    
    count_print('Attaching metrics, timer and events to engines ...')

    # Metrics
    attach_bleu(evaluator, device, record_scores=True)
    attach_rougel(evaluator, device, record_scores=True)
    attach_meteor(evaluator, device, record_scores=True)
    attach_ciderd(evaluator, device, record_scores=True)
    attach_medical_completeness(evaluator, device, tokenizer, record_scores=True)
    attach_weighted_medical_completeness(evaluator, device, tokenizer, record_scores=True)
    if verbose_question:
        attach_exactmatch_question(evaluator, device, record_scores=True)
    if classify_tags:
        attach_medical_tags_f1score(evaluator, device, record_scores=True)
    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(evaluator, record_scores=True)
    if classify_chexpert:
        attach_chexpert_labels_accuracy(evaluator, device, record_scores=True)
        attach_chexpert_labels_macroavgf1(evaluator, device)
        attach_chexpert_labels_microavgf1(evaluator, device)
        attach_chexpert_labels_roc_auc(evaluator, 'cpu')
    if classify_questions:
        attach_question_labels_f1score(evaluator, device, record_scores=True)

    # Accumulators
    attach_accumulator(evaluator, 'idxs')
    attach_accumulator(evaluator, 'pred_answers')
    if verbose_question:
        attach_accumulator(evaluator, 'pred_questions')
    if classify_tags:
        attach_accumulator(evaluator, 'pred_tags')
    if classify_orientation:
        attach_accumulator(evaluator, 'pred_orientation')
    if classify_chexpert:
        attach_accumulator(evaluator, 'pred_chexpert')
    if classify_questions:
        attach_accumulator(evaluator, 'pred_qlabels')
    
    # Timer
    timer = Timer()
    timer.attach(evaluator, start=Events.EPOCH_STARTED)
    
    # Logging
    metrics_to_print=[MetricNames.BLEU, MetricNames.ROUGE_L, MetricNames.METEOR,
                    MetricNames.CIDER_D, MetricNames.MEDCOMP, MetricNames.WMEDCOMP]
    if verbose_question:
        metrics_to_print.append(MetricNames.EXACTMATCH_QUESTION)
    if classify_tags:
        metrics_to_print.append(MetricNames.MEDTAGF1)
    if classify_orientation:
        metrics_to_print.append(MetricNames.ORIENACC)
    if classify_chexpert:
        metrics_to_print.append(MetricNames.CHXLABELMICROAVGF1)
        metrics_to_print.append(MetricNames.CHXLABELMACROAVGF1)
        metrics_to_print.append(MetricNames.CHXLABELACC)
        metrics_to_print.append(MetricNames.CHXLABEL_ROCAUC)
    if classify_questions:
        metrics_to_print.append(MetricNames.QLABELSF1)

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
    if verbose_question:
        metrics_to_aggregate.append(MetricNames.EXACTMATCH_QUESTION)
    if classify_tags:
        metrics_to_aggregate.append(MetricNames.MEDTAGF1)
    if classify_orientation:
        metrics_to_aggregate.append(MetricNames.ORIENACC)
    if classify_chexpert: 
        metrics_to_aggregate.append(MetricNames.CHXLABELMICROAVGF1)
        metrics_to_aggregate.append(MetricNames.CHXLABELMACROAVGF1)
        metrics_to_aggregate.append(MetricNames.CHXLABELACC)
        metrics_to_aggregate.append(MetricNames.CHXLABEL_ROCAUC)
    if classify_questions:
        metrics_to_aggregate.append(MetricNames.QLABELSF1)

    # Run evaluation
    results_dict = dict(tokenizer = tokenizer)
    results_folder_path = get_results_folder_path(checkpoint_folder_path)

    if eval_iuxray:
        print('\n========================')
        count_print('Running evaluator engine on IU X-Ray validation split ...')
        print('len(dataset) =', len(iuxray_vqa_trainer.val_dataset))
        print('len(dataloader) =', len(iuxray_vqa_trainer.val_dataloader))
        evaluator.run(iuxray_vqa_trainer.val_dataloader)
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
                                                metrics_to_aggregate, results_folder_path,
                                                one_hot_questions = not verbose_question,
                                                qa_reports = iuxray_qa_reports)

    if eval_mimiccxr:
        print('\n========================')
        count_print('Running evaluator engine on MIMIC-CXR test split ...')
        print('len(dataset) =', len(mimiccxr_vqa_evaluator.test_dataset))
        print('len(dataloader) =', len(mimiccxr_vqa_evaluator.test_dataloader))
        evaluator.run(mimiccxr_vqa_evaluator.test_dataloader)        
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
                                                metrics_to_aggregate, results_folder_path,
                                                one_hot_questions = not verbose_question,
                                                qa_reports = mimiccxr_qa_reports)

    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder,
    answer_decoding,
    beam_search_k=None,
    batch_size = 100,
    num_workers = 0,
    device = 'GPU',
    return_results = False,
    use_amp = False,
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
    dataloading_kwargs = metadata['dataloading_kwargs']
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']

    return _evaluate_model(
                tokenizer_kwargs,
                model_kwargs,
                mimiccxr_vqa_evaluator_kwargs,
                iuxray_vqa_trainer_kwargs,
                dataloading_kwargs,
                auxiliary_tasks_kwargs = auxiliary_tasks_kwargs,
                device = device,
                answer_decoding = answer_decoding,
                beam_search_k = beam_search_k,
                num_workers = num_workers,
                checkpoint_folder_path = checkpoint_folder,
                return_results = return_results,
                use_amp = use_amp,
                eval_iuxray = eval_iuxray,
                eval_mimiccxr = eval_mimiccxr,
            )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)