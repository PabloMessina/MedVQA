import  os
import numpy as np
import argparse
from copy import deepcopy

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer

from medvqa.models.vqa.open_ended_vqa import QuestionEncoding
from medvqa.utils.constants import (
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID,
    ReportEvalMode,
)
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.metrics import (
    attach_exactmatch_question,
    attach_chexpert_labels_f1score,
    attach_question_labels_f1score,
    attach_medical_tags_f1score,
    attach_chexpert_labels_accuracy,
    attach_dataset_aware_orientation_accuracy,
    attach_loss
)
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    load_metadata,
)
from medvqa.utils.common import (
    WORKSPACE_DIR,
    parsed_args_to_dict,
)    
from medvqa.utils.handlers import (
    get_log_metrics_handlers,
    get_log_iteration_handler,
    attach_accumulator,
)
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.models.vqa.open_ended_vqa import OpenEndedVQA
from medvqa.utils.files import (
    get_cached_json_file,
    get_results_folder_path,
    save_to_pickle,
)
from medvqa.training.vqa import get_engine
from medvqa.datasets.dataloading_utils import get_vqa_collate_batch_fn
from medvqa.datasets.mimiccxr.mimiccxr_vqa_dataset_management import MIMICCXR_VQA_Evaluator
from medvqa.datasets.iuxray.iuxray_vqa_dataset_management import IUXRAY_VQA_Trainer
from medvqa.datasets.image_processing import get_image_transform
from medvqa.datasets.preprocessing import get_sentences
from medvqa.utils.logging import CountPrinter
from medvqa.evaluation.report_generation import (
    recover_reports,
    compute_report_level_metrics,
)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--checkpoint-folder', type=str, required=True,
                        help='Relative path to folder with checkpoint to evaluate')
    parser.add_argument('--eval-mode', type=str, required=True)

    # optional arguments    
    parser.add_argument('--n-questions-per-report', type=int, default=None)
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

def _compute_and_save_report_level_metrics(results_dict, dataset_name, tokenizer, results_folder_path, parenthesis_text=None):
    metrics = compute_report_level_metrics(results_dict[f'{dataset_name}_reports']['gt_reports'],
                                           results_dict[f'{dataset_name}_reports']['gen_reports'],
                                           tokenizer)
    if parenthesis_text is not None:
        save_path = os.path.join(results_folder_path, f'{dataset_name}_report_level_metrics({parenthesis_text}).pkl')
    else:
        save_path = os.path.join(results_folder_path, f'{dataset_name}_report_level_metrics.pkl')
    save_to_pickle(metrics, save_path)
    print (f'Report-level metrics successfully saved to {save_path}')
    return metrics

def _get_eval_mode_text(eval_mode, n_questions_per_report):
    strings = [f'eval_mode={eval_mode}']
    if eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION.value or\
       eval_mode == ReportEvalMode.MOST_POPULAR.value:
       assert n_questions_per_report is not None
       strings.append(f'n_questions_per_report={n_questions_per_report}')
    return ';'.join(strings)

def _estimate_maximum_answer_length(qa_adapted_datasets, tokenizer):
    lengths = [None] * (10 * sum(len(x['reports']) for x in qa_adapted_datasets))
    for i, s in enumerate(get_sentences(qa_adapted_datasets, include_unmatched=False)):
        lengths[i] = len(tokenizer.tokenize(s))
    lengths = lengths[:i]
    mean = np.mean(lengths)
    sqrt = np.std(lengths)
    return int(mean + 3 * sqrt)

def _evaluate_model(
    tokenizer_kwargs,
    model_kwargs,
    mimiccxr_vqa_evaluator_kwargs,
    iuxray_vqa_trainer_kwargs,
    auxiliary_tasks_kwargs,
    eval_mode,
    n_questions_per_report = None,
    num_workers = 0,
    device = 'GPU',
    checkpoint_folder_path = None,
    return_results = False,
    use_amp = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
):
    assert eval_iuxray or eval_mimiccxr
    assert eval_mode is not None
    if eval_mode == ReportEvalMode.MOST_POPULAR.value or\
       eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION.value:
       assert n_questions_per_report is not None

    # Pull out some args from kwargs
    question_encoding = model_kwargs.get('question_encoding', QuestionEncoding.BILSTM)
    verbose_question = question_encoding != QuestionEncoding.ONE_HOT

    # auxiliary task: medical tags prediction
    use_tags = auxiliary_tasks_kwargs['use_tags']
    n_medical_tags = auxiliary_tasks_kwargs['n_medical_tags']
    iuxray_medical_tags_per_report_filename = auxiliary_tasks_kwargs['iuxray_medical_tags_per_report_filename']
    mimiccxr_medical_tags_per_report_filename = auxiliary_tasks_kwargs['mimiccxr_medical_tags_per_report_filename']
    if use_tags:
        assert n_medical_tags is not None
        assert iuxray_medical_tags_per_report_filename is not None
        assert mimiccxr_medical_tags_per_report_filename is not None
    
    # auxiliary task: orientation classification
    use_orientation = auxiliary_tasks_kwargs['use_orientation']

    # auxiliary task: chexpert labels
    use_chexpert = auxiliary_tasks_kwargs['use_chexpert']
    iuxray_chexpert_labels_filename = auxiliary_tasks_kwargs['iuxray_chexpert_labels_filename']
    mimiccxr_chexpert_labels_filename = auxiliary_tasks_kwargs['mimiccxr_chexpert_labels_filename']

    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)
    n_questions = auxiliary_tasks_kwargs.get('n_questions', None)
    iuxray_question_labels_filename = auxiliary_tasks_kwargs.get('iuxray_question_labels_filename', None)
    mimiccxr_question_labels_filename = auxiliary_tasks_kwargs.get('mimiccxr_question_labels_filename', None)
    if classify_questions:
        assert n_questions is not None
        if eval_iuxray: assert iuxray_question_labels_filename is not None
        if eval_mimiccxr: assert mimiccxr_question_labels_filename is not None
    # QA dataset filenames
    iuxray_qa_adapted_reports_filename = iuxray_vqa_trainer_kwargs['qa_adapted_reports_filename']
    mimiccxr_qa_adapted_reports_filename = mimiccxr_vqa_evaluator_kwargs['qa_adapted_reports_filename']
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
    medical_tokenization = tokenizer_kwargs['medical_tokenization']
    medical_terms_frequency_filename = tokenizer_kwargs['medical_terms_frequency_filename']
    assert medical_tokenization == (medical_terms_frequency_filename is not None)
    tokenizer = Tokenizer(qa_adapted_dataset_paths=[iuxray_qa_adapted_reports_path,
                                                    mimiccxr_qa_adapted_reports_path],
                          min_freq=vocab_min_freq,
                          medical_terms_frequency_filename=medical_terms_frequency_filename)
    
    count_print('Estimating maximum answer length ...')
    max_answer_length = _estimate_maximum_answer_length([iuxray_qa_reports, mimiccxr_qa_reports], tokenizer)
    print('max_answer_length =', max_answer_length)    
    
    # Default image transform
    count_print('Defining image transform ...')
    img_transform = get_image_transform()

    # Define collate_batch_fn    
    mimiccxr_collate_batch_fn = get_vqa_collate_batch_fn(MIMICCXR_DATASET_ID,
                                                        verbose_question = verbose_question,
                                                        include_answer=False,
                                                        use_tags = use_tags,
                                                        n_tags = n_medical_tags,
                                                        use_orientation = use_orientation,
                                                        use_chexpert = use_chexpert,
                                                        classify_questions = classify_questions)    
    iuxray_collate_batch_fn = get_vqa_collate_batch_fn(IUXRAY_DATASET_ID,
                                                   verbose_question = verbose_question,
                                                   include_answer=False,
                                                   use_tags = use_tags,
                                                   n_tags = n_medical_tags,
                                                   use_orientation = use_orientation,
                                                   use_chexpert = use_chexpert,
                                                   classify_questions = classify_questions)

    # Load saved checkpoint    
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)    
    
    # Create MIMIC-CXR vqa evaluator
    if eval_mimiccxr:
        count_print('Creating MIMIC-CXR vqa evaluator ...')
        mimiccxr_vqa_evaluator = MIMICCXR_VQA_Evaluator(
            transform = img_transform,
            collate_batch_fn = mimiccxr_collate_batch_fn,
            num_workers = num_workers,
            tokenizer = tokenizer,
            use_tags = use_tags,
            medical_tags_per_report_filename = mimiccxr_medical_tags_per_report_filename,
            use_orientation = use_orientation,
            use_chexpert = use_chexpert,
            chexpert_labels_filename = mimiccxr_chexpert_labels_filename,
            classify_questions = classify_questions,
            question_labels_filename = mimiccxr_question_labels_filename,
            report_eval_mode = eval_mode,
            image_local_feat_size = model_kwargs['image_local_feat_size'],
            n_questions = n_questions,
            pretrained_weights = checkpoint['model'],
            pretrained_checkpoint_path = checkpoint_path,
            n_questions_per_report = n_questions_per_report,
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
            use_tags = use_tags,
            medical_tags_per_report_filename = iuxray_medical_tags_per_report_filename,
            use_orientation = use_orientation,
            use_chexpert = use_chexpert,
            chexpert_labels_filename = iuxray_chexpert_labels_filename,
            classify_questions = classify_questions,
            question_labels_filename = mimiccxr_question_labels_filename,
            validation_only = True,
            report_eval_mode = eval_mode,
            ignore_medical_tokenization = tokenizer.medical_tokenization,
            verbose_question = verbose_question,
            **iuxray_vqa_trainer_kwargs,
        )

    # Create model
    count_print('Creating instance of OpenEndedVQA model ...')
    model = OpenEndedVQA(vocab_size=tokenizer.vocab_size,
                         start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                         device=device, 
                         n_medical_tags=n_medical_tags,
                         n_questions=n_questions,
                         classify_orientation=use_orientation,
                         classify_chexpert=use_chexpert,
                         classify_questions=classify_questions,
                         **model_kwargs)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    evaluator = get_engine(model, tokenizer, use_tags, use_orientation, use_chexpert,
                           classify_questions, question_encoding,
                           device, use_amp=use_amp, training=False, include_answer=False,
                           max_answer_length=max_answer_length)

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    # Metrics
    if verbose_question:
        attach_exactmatch_question(evaluator, device, record_scores=True)
    if use_tags:
        attach_medical_tags_f1score(evaluator, device, record_scores=True)
    if use_orientation:
        attach_dataset_aware_orientation_accuracy(evaluator, record_scores=True)
    if use_chexpert:
        attach_chexpert_labels_f1score(evaluator, device, record_scores=True)
        attach_chexpert_labels_accuracy(evaluator, device, record_scores=True)
    if classify_questions:
        attach_question_labels_f1score(evaluator, device, record_scores=True)

    # Accumulators
    attach_accumulator(evaluator, 'idxs')
    attach_accumulator(evaluator, 'pred_answers')
    if verbose_question:
        attach_accumulator(evaluator, 'pred_questions')
    if use_tags:
        attach_accumulator(evaluator, 'pred_tags')
    if use_orientation:
        attach_accumulator(evaluator, 'pred_orientation')
    if use_chexpert:
        attach_accumulator(evaluator, 'pred_chexpert')
    if classify_questions:
        attach_accumulator(evaluator, 'pred_qlabels')

    # Losses
    attach_loss('loss', evaluator, device)
    
    # Timer
    timer = Timer()
    timer.attach(evaluator, start=Events.EPOCH_STARTED)
    
    # Logging
    metrics_to_print=['loss']
    if verbose_question:
        metrics_to_print.append('exactmatch_question')
    if use_tags:
        metrics_to_print.append('medtagf1')
    if use_orientation:
        metrics_to_print.append('orienacc')
    if use_chexpert:
        metrics_to_print.append('chxlabelf1')
        metrics_to_print.append('chxlabelacc')
    if classify_questions:
        metrics_to_print.append('qlabelsf1')

    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()    

    # Attach handlers    
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation
    results_dict = dict(tokenizer = tokenizer)
    results_folder_path = get_results_folder_path(checkpoint_folder_path)    
    eval_mode_text = _get_eval_mode_text(eval_mode, n_questions_per_report)

    if eval_iuxray:
        print('\n========================')
        count_print('Running evaluator engine on IU X-Ray validation split ...')
        print('len(dataset) =', len(iuxray_vqa_trainer.val_dataset))
        print('len(dataloader) =', len(iuxray_vqa_trainer.val_dataloader))
        evaluator.run(iuxray_vqa_trainer.val_dataloader)
        
        results_dict['iuxray_metrics'] = deepcopy(evaluator.state.metrics)            
        results_dict['iuxray_dataset'] = iuxray_vqa_trainer.val_dataset
        results_dict['iuxray_reports'] = recover_reports(
            results_dict['iuxray_metrics'],
            results_dict['iuxray_dataset'],
            tokenizer, iuxray_qa_reports,
            verbose_question=verbose_question,
        )
        results_dict['iuxray_report_metrics'] = _compute_and_save_report_level_metrics(
            results_dict, 'iuxray', tokenizer, results_folder_path, parenthesis_text=eval_mode_text)

    if eval_mimiccxr:
        print('\n========================')
        count_print('Running evaluator engine on MIMIC-CXR test split ...')
        print('len(dataset) =', len(mimiccxr_vqa_evaluator.test_dataset))
        print('len(dataloader) =', len(mimiccxr_vqa_evaluator.test_dataloader))
        evaluator.run(mimiccxr_vqa_evaluator.test_dataloader)
        
        results_dict['mimiccxr_metrics'] = deepcopy(evaluator.state.metrics)
        results_dict['mimiccxr_dataset'] = mimiccxr_vqa_evaluator.test_dataset            
        results_dict['mimiccxr_reports'] = recover_reports(
            results_dict['mimiccxr_metrics'],
            results_dict['mimiccxr_dataset'],
            tokenizer, mimiccxr_qa_reports,
            verbose_question=verbose_question,
        )
        results_dict['mimiccxr_report_metrics'] = _compute_and_save_report_level_metrics(
            results_dict, 'mimiccxr', tokenizer, results_folder_path, parenthesis_text=eval_mode_text)

    torch.cuda.empty_cache()
    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder,
    eval_mode,
    n_questions_per_report = None,
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
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']

    return _evaluate_model(
                tokenizer_kwargs,
                model_kwargs,
                mimiccxr_vqa_evaluator_kwargs,
                iuxray_vqa_trainer_kwargs,
                auxiliary_tasks_kwargs,
                eval_mode,
                n_questions_per_report = n_questions_per_report,
                device = device,
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