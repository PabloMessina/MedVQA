import  os
import numpy as np
import argparse
import re
from copy import deepcopy

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer

from medvqa.models.vqa.open_ended_vqa import (
    QuestionEncoding,
    does_include_image,
    does_include_visual_features,
)
from medvqa.utils.constants import (
    CHEXPERT_DATASET_ID,
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID,
    VINBIG_DATASET_ID,
    MetricNames,
    ReportEvalMode,
)
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR, MIMICCXR_EvalViewModes
from medvqa.metrics import (
    attach_dataset_aware_chest_imagenome_labels_accuracy,
    attach_dataset_aware_chest_imagenome_labels_macroavgf1,
    attach_dataset_aware_chest_imagenome_labels_microavgf1,
    attach_dataset_aware_chest_imagenome_labels_roc_auc,
    attach_dataset_aware_chexpert_labels_accuracy,
    attach_dataset_aware_chexpert_labels_macroavgf1,
    attach_dataset_aware_chexpert_labels_microavgf1,
    attach_dataset_aware_chexpert_labels_roc_auc,
    attach_dataset_aware_question_labels_macroavgf1,
    attach_exactmatch_question,
    attach_medical_tags_f1score,
    attach_dataset_aware_orientation_accuracy,
)
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    get_matching_checkpoint_epoch,
    load_metadata,
    split_checkpoint_name,
)
from medvqa.utils.common import (
    WORKSPACE_DIR,
    parsed_args_to_dict,
)    
from medvqa.utils.handlers_utils import (
    get_log_metrics_handlers,
    get_log_iteration_handler,
    attach_accumulator,
)
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.models.vqa.open_ended_vqa import OpenEndedVQA
from medvqa.utils.files_utils import (
    get_cached_json_file,
    get_results_folder_path,
    save_pickle,
)
from medvqa.training.vqa import get_engine
from medvqa.datasets.dataloading_utils import get_vqa_collate_batch_fn
from medvqa.datasets.mimiccxr.mimiccxr_vqa_dataset_management import MIMICCXR_VQA_Evaluator
# from medvqa.datasets.iuxray.iuxray_vqa_dataset_management import IUXRAY_VQA_Trainer
from medvqa.datasets.image_processing import get_image_transform
from medvqa.datasets.preprocessing import get_sentences
from medvqa.utils.logging_utils import CountPrinter, print_blue
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
    parser.add_argument('--qclass-threshold', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=140,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for parallel dataloading')
    parser.add_argument('--answer-decoding', type=str, default='greedy-search')

    parser.add_argument('--eval-checkpoint-folder', type=str, default=None,
                        help='Optional checkpoint folder to load weights from for evaluation')
    parser.add_argument('--precomputed-question-probs-path', type=str, default=None)
    parser.add_argument('--precomputed-question-thresholds-path', type=str, default=None)

    parser.add_argument('--use-random-image', dest='use_random_image', action='store_true')
    parser.set_defaults(use_random_image=False)

    parser.add_argument('--eval-iuxray', dest='eval_iuxray', action='store_true')
    parser.set_defaults(eval_iuxray=False)

    parser.add_argument('--eval-mimiccxr', dest='eval_mimiccxr', action='store_true')
    parser.set_defaults(eval_mimiccxr=False)

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)

    parser.add_argument('--max-processes-for-chexpert-labeler', type=int, default=10)

    parser.add_argument('--save-for-error-analysis', dest='save_for_error_analysis', action='store_true')
    parser.set_defaults(save_for_error_analysis=False)

    parser.add_argument('--mimiccxr-eval-view-mode', type=str, default=MIMICCXR_EvalViewModes.FRONT_ALL)

    return parser.parse_args()

def _compute_and_save_report_level_metrics(results_dict, dataset_name, tokenizer, results_folder_path,
                                           parenthesis_text=None, max_processes=10):
    metrics = compute_report_level_metrics(results_dict[f'{dataset_name}_reports']['gt_reports'],
                                           results_dict[f'{dataset_name}_reports']['gen_reports'],
                                           tokenizer, max_processes=max_processes)
    if parenthesis_text is not None:
        save_path = os.path.join(results_folder_path, f'{dataset_name}_report_level_metrics({parenthesis_text}).pkl')
    else:
        save_path = os.path.join(results_folder_path, f'{dataset_name}_report_level_metrics.pkl')
    save_pickle(metrics, save_path)
    print (f'Report-level metrics successfully saved to {save_path}')
    return metrics

def _save_results_for_error_analysis(results_dict, dataset_name, results_folder_path,
        classify_orientation, classify_chexpert, classify_questions, classify_chest_imagenome,
        chest_imagenome_labels, parenthesis_text=None):

    if classify_chest_imagenome:
        assert chest_imagenome_labels is not None

    metrics = results_dict[f'{dataset_name}_metrics']
    dataset = results_dict[f'{dataset_name}_dataset']
    reports = results_dict[f'{dataset_name}_reports']
    report_metrics = results_dict[f'{dataset_name}_report_metrics']
    
    idxs = metrics['idxs']
    rids = [x['rid'] for x in reports['gt_reports']]
    rid2i = dict()
    for i, idx in enumerate(metrics['idxs']):
        rid2i[dataset.report_ids[idx]] = i

    image_encoder_pred = dict()
    if classify_orientation:
        met = metrics['pred_orientation']
        image_encoder_pred['orientation'] = [met[rid2i[rid]].item() for rid in rids]
    if classify_chexpert:
        met = metrics['pred_chexpert']
        image_encoder_pred['chexpert'] = [met[rid2i[rid]].numpy() for rid in rids]
    if classify_questions:
        met = metrics['pred_qlabels']
        image_encoder_pred['qlabels'] = [met[rid2i[rid]].numpy() for rid in rids]
    if classify_chest_imagenome:
        met = metrics['pred_chest_imagenome']
        image_encoder_pred['chest_imagenome'] = [met[rid2i[rid]].numpy() for rid in rids]

    image_encoder_gt = dict()
    if classify_orientation:
        image_encoder_gt['orientation'] = [dataset.orientations[idxs[rid2i[rid]]] for rid in rids]
    if classify_chexpert:
        image_encoder_gt['chexpert'] = [dataset.chexpert_labels[rid] for rid in rids]
    if classify_questions:
        image_encoder_gt['qlabels'] = [dataset.question_labels[rid] for rid in rids]
    if classify_chest_imagenome:
        image_encoder_gt['chest_imagenome'] = [chest_imagenome_labels[rid] for rid in rids]

    output = {
        'images': [dataset.images[i] for i in idxs],
        'reports':  results_dict[f'{dataset_name}_reports'],
        'report_metrics': report_metrics,
        'image_encoder_gt': image_encoder_gt,
        'image_encoder_pred': image_encoder_pred,        
    }
    parenthesis_text = f'({parenthesis_text})' if parenthesis_text else ''
    save_path = os.path.join(results_folder_path, f'{dataset_name}_report_results_for_error_analysis{parenthesis_text}.pkl')
    save_pickle(output, save_path)
    print (f'Report-level results for error analysis successfully saved to {save_path}')

def _get_eval_mode_text(eval_mode, n_questions_per_report, qclass_threshold, checkpoint_path,
                        precomputed_question_probs_path, use_random_image, eval_view_mode):
    strings = [f'eval_mode={eval_mode}']
    strings.append(f'eval_view_mode={eval_view_mode}')
    if eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION or\
       eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION or\
       eval_mode == ReportEvalMode.MOST_POPULAR:
        assert n_questions_per_report is not None
        strings.append(f'n_q_per_rep={n_questions_per_report}')
    if eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION or \
       eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
        if qclass_threshold is not None:
            strings.append(f'qclass_thr={qclass_threshold}')
    if eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION or \
       eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
        assert checkpoint_path is not None or precomputed_question_probs_path is not None
        if precomputed_question_probs_path is not None:
            if 'ensemble' in precomputed_question_probs_path:
                tmp = re.findall(r'_ensemble\((.*)\)_', precomputed_question_probs_path)[0]
                strings.append(f'ensemble=({tmp})')
            else:
                timestamp = os.path.basename(os.path.abspath(os.path.join(precomputed_question_probs_path, os.pardir)))[:15]
                strings.append(f'probs={timestamp}')
                strings.append(f'epoch={get_matching_checkpoint_epoch(precomputed_question_probs_path)}')
        else:
            timestamp = os.path.basename(os.path.abspath(os.path.join(checkpoint_path, os.pardir)))[:15]
            strings.append(f'chkpt={timestamp}')
            strings.append(f'epoch={split_checkpoint_name(os.path.basename(checkpoint_path)).epoch}')
    if eval_mode == ReportEvalMode.NEAREST_NEIGHBOR:
       assert checkpoint_path is not None       
       timestamp = os.path.basename(os.path.abspath(os.path.join(checkpoint_path, os.pardir)))[:15]
       strings.append(f'chkpt={timestamp}')
       strings.append(f'epoch={split_checkpoint_name(os.path.basename(checkpoint_path)).epoch}')
    if use_random_image:
        strings.append('rand-img')
    
    return ';'.join(strings)

def _estimate_maximum_answer_length(qa_adapted_datasets, tokenizer):
    lengths = [None] * (10 * sum(len(x['reports']) for x in qa_adapted_datasets))
    for i, s in enumerate(get_sentences(qa_adapted_datasets, include_unmatched=False)):
        lengths[i] = len(tokenizer.tokenize(s))
    lengths = lengths[:i]
    mean = np.mean(lengths)
    sqrt = np.std(lengths)
    return int(mean + 3 * sqrt)

def _get_one_hot_question_offset(offsets_dict, dataset_id, eval_mode):
    if offsets_dict is None:
        return None
    if eval_mode == ReportEvalMode.CHEXPERT_LABELS:
        return offsets_dict[str(CHEXPERT_DATASET_ID)]
    if eval_mode == ReportEvalMode.VINBIG_DISEASES:
        return offsets_dict[str(VINBIG_DATASET_ID)]
    return offsets_dict[str(dataset_id)]

def _evaluate_model(
    tokenizer_kwargs,
    model_kwargs,
    dataloading_kwargs,
    image_transform_kwargs,
    mimiccxr_vqa_evaluator_kwargs,
    iuxray_vqa_trainer_kwargs,
    auxiliary_tasks_kwargs,
    trainer_engine_kwargs,
    answer_decoding,
    eval_mode,
    n_questions_per_report = None,
    qclass_threshold = None,
    num_workers = 0,
    device = 'GPU',
    checkpoint_folder_path = None,
    eval_checkpoint_folder_path = None,
    precomputed_question_probs_path = None,
    precomputed_question_thresholds_path = None,
    return_results = False,
    use_random_image = False,
    use_amp = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
    max_processes_for_chexpert_labeler = 10,
    save_for_error_analysis = False,
    mimiccxr_eval_view_mode = None,
):
    assert eval_iuxray or eval_mimiccxr
    assert eval_mode is not None
    if eval_mode == ReportEvalMode.MOST_POPULAR or\
       eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION:
       assert n_questions_per_report is not None

    # Pull out some args from kwargs
    question_encoding = model_kwargs.get('question_encoding', QuestionEncoding.BILSTM)
    verbose_question = question_encoding != QuestionEncoding.ONE_HOT
    visual_input_mode = model_kwargs['visual_input_mode']
    include_image = does_include_image(visual_input_mode)
    include_visual_features = does_include_visual_features(visual_input_mode)
    use_merged_findings = trainer_engine_kwargs.get('use_merged_findings', False)

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
    # iuxray_chexpert_labels_filename = auxiliary_tasks_kwargs['iuxray_chexpert_labels_filename']
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs['classify_chest_imagenome']
    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)
    n_questions_aux_task = auxiliary_tasks_kwargs.get('n_questions_aux_task', None)
    iuxray_question_labels_filename = auxiliary_tasks_kwargs.get('iuxray_question_labels_filename', None)
    mimiccxr_question_labels_filename = auxiliary_tasks_kwargs.get('mimiccxr_question_labels_filename', None)
    if classify_questions:
        assert n_questions_aux_task is not None
        if eval_iuxray: assert iuxray_question_labels_filename is not None
        if eval_mimiccxr: assert mimiccxr_question_labels_filename is not None
    
    if question_encoding == QuestionEncoding.ONE_HOT:
        assert model_kwargs['n_questions'] is not None    
    
    count_print = CountPrinter()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Load qa adapted reports
    count_print('Loading QA adapted reports ...')
    qa_adapted_datasets = []
    if eval_mimiccxr:
        mimiccxr_qa_adapted_reports_filename = mimiccxr_vqa_evaluator_kwargs['qa_adapted_reports_filename']
        assert mimiccxr_qa_adapted_reports_filename is not None
        mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, mimiccxr_qa_adapted_reports_filename)
        mimiccxr_qa_reports = get_cached_json_file(mimiccxr_qa_adapted_reports_path)
        qa_adapted_datasets.append(mimiccxr_qa_reports)
    if eval_iuxray:
        iuxray_qa_adapted_reports_filename = iuxray_vqa_trainer_kwargs['qa_adapted_reports_filename']    
        assert iuxray_qa_adapted_reports_filename is not None
        iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, iuxray_qa_adapted_reports_filename)
        iuxray_qa_reports = get_cached_json_file(iuxray_qa_adapted_reports_path)
        qa_adapted_datasets.append(iuxray_qa_reports)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    del tokenizer_kwargs['use_medical_tokenization']
    tokenizer = Tokenizer(**tokenizer_kwargs)
    
    count_print('Estimating maximum answer length ...')
    max_answer_length = _estimate_maximum_answer_length(qa_adapted_datasets, tokenizer)
    print('max_answer_length =', max_answer_length)    
    
    # Default image transform (no augmentations)
    count_print('Defining image transform ...')
    img_transform = get_image_transform(**image_transform_kwargs)

    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    
    one_hot_question_offsets = dataloading_kwargs.get('one_hot_question_offsets', None)
    if not verbose_question: assert one_hot_question_offsets is not None
    
    if eval_mimiccxr:
        mimiccxr_collate_batch_fn = get_vqa_collate_batch_fn(MIMICCXR_DATASET_ID,
                                                            verbose_question = verbose_question,
                                                            one_hot_question_offset = _get_one_hot_question_offset(
                                                                one_hot_question_offsets, MIMICCXR_DATASET_ID, eval_mode),
                                                            include_image = include_image,
                                                            include_visual_features = include_visual_features,
                                                            include_answer=False,
                                                            classify_tags = classify_tags,
                                                            n_tags = n_medical_tags,
                                                            classify_orientation = classify_orientation,
                                                            classify_chexpert = classify_chexpert,
                                                            classify_chest_imagenome = classify_chest_imagenome,
                                                            classify_questions = classify_questions)
    
    # if eval_iuxray:
    #     iuxray_collate_batch_fn = get_vqa_collate_batch_fn(IUXRAY_DATASET_ID,
    #                                                 verbose_question = verbose_question,
    #                                                 one_hot_question_offset = _get_one_hot_question_offset(
    #                                                             one_hot_question_offsets, IUXRAY_DATASET_ID, eval_mode),
    #                                                 include_answer=False,
    #                                                 classify_tags = classify_tags,
    #                                                 n_tags = n_medical_tags,
    #                                                 classify_orientation = classify_orientation,
    #                                                 classify_chexpert = classify_chexpert,
    #                                                 classify_questions = classify_questions)

    # Load saved checkpoint    
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    # Load optional checkpoint
    if eval_checkpoint_folder_path is not None:
        eval_checkpoint_path = get_checkpoint_filepath(eval_checkpoint_folder_path)        
        eval_checkpoint = torch.load(eval_checkpoint_path)
        pretrained_weights = eval_checkpoint['model']
        pretrained_checkpoint_path = eval_checkpoint_path
    else:
        pretrained_weights = checkpoint['model'] # default -> same model
        pretrained_checkpoint_path = checkpoint_path
    print('pretrained_checkpoint_path =', pretrained_checkpoint_path)

    # Create MIMIC-CXR vqa evaluator
    if eval_mimiccxr:
        count_print('Creating MIMIC-CXR vqa evaluator ...')
        mimiccxr_vqa_evaluator = MIMICCXR_VQA_Evaluator(
            transform = img_transform,
            collate_batch_fn = mimiccxr_collate_batch_fn,
            num_workers = num_workers,
            tokenizer = tokenizer,            
            report_eval_mode = eval_mode,
            image_local_feat_size = model_kwargs['image_local_feat_size'],
            n_questions_aux_task = model_kwargs['n_questions_aux_task'],
            pretrained_weights = pretrained_weights,
            pretrained_checkpoint_path = pretrained_checkpoint_path,
            precomputed_question_probs_path = precomputed_question_probs_path,
            precomputed_question_thresholds_path = precomputed_question_thresholds_path,
            n_questions_per_report = n_questions_per_report,
            qclass_threshold = qclass_threshold,
            eval_view_mode = mimiccxr_eval_view_mode,
            **mimiccxr_vqa_evaluator_kwargs,
        )
    
    # Create IU X-Ray vqa trainer
    # if eval_iuxray:
    #     count_print('Creating IU X-Ray vqa trainer ...')
    #     iuxray_vqa_trainer = IUXRAY_VQA_Trainer(
    #         transform = img_transform,
    #         collate_batch_fn = iuxray_collate_batch_fn,
    #         num_workers = num_workers,
    #         tokenizer = tokenizer,        
    #         iuxray_qa_reports = iuxray_qa_reports,
    #         classify_tags = classify_tags,
    #         medical_tags_per_report_filename = iuxray_medical_tags_per_report_filename,
    #         classify_orientation = classify_orientation,
    #         classify_chexpert = classify_chexpert,
    #         chexpert_labels_filename = iuxray_chexpert_labels_filename,
    #         classify_questions = classify_questions,
    #         question_labels_filename = iuxray_question_labels_filename,
    #         validation_only = True,
    #         report_eval_mode = eval_mode,
    #         ignore_medical_tokenization = tokenizer.medical_tokenization,
    #         verbose_question = verbose_question,
    #         **iuxray_vqa_trainer_kwargs,
    #     )

    # Create model
    count_print('Creating instance of OpenEndedVQA model ...')
    model = OpenEndedVQA(vocab_size=tokenizer.vocab_size,
                         start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                         device=device, **model_kwargs)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    evaluator = get_engine(model, tokenizer, classify_tags, classify_orientation, classify_chexpert,
                           classify_questions, classify_chest_imagenome, question_encoding, answer_decoding,
                           device, use_amp=use_amp, training=False, include_answer=False,
                           include_image=include_image, include_visual_features=include_visual_features,
                           max_answer_length=max_answer_length,
                           use_merged_findings=use_merged_findings)

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    # Metrics

    _iu_mim_datasets = [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID]
    _chexpert_labels_datasets = _iu_mim_datasets[:]
    _orientation_datasets = _iu_mim_datasets[:]

    if use_merged_findings:
        _findings_remapper = trainer_engine_kwargs['findings_remapper']
        _chexpert_class_indices = _findings_remapper[str(CHEXPERT_DATASET_ID)]
    else:
        _chexpert_class_indices = None

    if verbose_question:
        attach_exactmatch_question(evaluator, device, record_scores=True)
    if classify_tags:
        attach_medical_tags_f1score(evaluator, device, record_scores=True)
    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(evaluator, _orientation_datasets, record_scores=True)
    if classify_chexpert:
        attach_dataset_aware_chexpert_labels_accuracy(evaluator, _chexpert_labels_datasets, _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_macroavgf1(evaluator, _chexpert_labels_datasets, _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_microavgf1(evaluator, _chexpert_labels_datasets, _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_roc_auc(evaluator, _chexpert_labels_datasets, 'cpu', _chexpert_class_indices)
    if classify_chest_imagenome:
        attach_dataset_aware_chest_imagenome_labels_accuracy(evaluator, [MIMICCXR_DATASET_ID])
        attach_dataset_aware_chest_imagenome_labels_macroavgf1(evaluator, [MIMICCXR_DATASET_ID])
        attach_dataset_aware_chest_imagenome_labels_microavgf1(evaluator, [MIMICCXR_DATASET_ID])
        attach_dataset_aware_chest_imagenome_labels_roc_auc(evaluator, [MIMICCXR_DATASET_ID], 'cpu')
    if classify_questions:
        attach_dataset_aware_question_labels_macroavgf1(evaluator, _iu_mim_datasets)

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
    if classify_chest_imagenome:
        attach_accumulator(evaluator, 'pred_chest_imagenome')
    if classify_questions:
        attach_accumulator(evaluator, 'pred_qlabels')
    
    # Timer
    timer = Timer()
    timer.attach(evaluator, start=Events.EPOCH_STARTED)
    
    # Logging
    metrics_to_print = []
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
    if classify_chest_imagenome:
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELMICROAVGF1)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELMACROAVGF1)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELACC)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELROCAUC)
    if classify_questions:
        metrics_to_print.append(MetricNames.QLABELS_MACROAVGF1)

    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()

    # Attach handlers    
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation
    results_dict = dict(tokenizer = tokenizer)
    results_folder_path = get_results_folder_path(checkpoint_folder_path)        

    if eval_mimiccxr:
        print('\n========================')
        count_print('Running evaluator engine on MIMIC-CXR test split ...')
        print('len(dataset) =', len(mimiccxr_vqa_evaluator.test_dataset))
        print('len(dataloader) =', len(mimiccxr_vqa_evaluator.test_dataloader))
        evaluator.run(mimiccxr_vqa_evaluator.test_dataloader)

        assert mimiccxr_eval_view_mode is not None

        eval_mode_text = _get_eval_mode_text(eval_mode, n_questions_per_report, qclass_threshold,
                                         pretrained_checkpoint_path, precomputed_question_probs_path,
                                         use_random_image, mimiccxr_eval_view_mode)
        
        print_blue('Computing metrics ...')
        results_dict['mimiccxr_metrics'] = deepcopy(evaluator.state.metrics)
        results_dict['mimiccxr_dataset'] = mimiccxr_vqa_evaluator.test_dataset            
        results_dict['mimiccxr_reports'] = recover_reports(
            results_dict['mimiccxr_metrics'],
            results_dict['mimiccxr_dataset'],
            tokenizer, eval_mode, mimiccxr_qa_reports,
            verbose_question=verbose_question,
        )
        print(f'recovered reports: len(gen_reports)={len(results_dict["mimiccxr_reports"]["gen_reports"])}, '
                f'len(gt_reports)={len(results_dict["mimiccxr_reports"]["gt_reports"])}')
        results_dict['mimiccxr_report_metrics'] = _compute_and_save_report_level_metrics(
            results_dict, 'mimiccxr', tokenizer, results_folder_path, parenthesis_text=eval_mode_text,
            max_processes=max_processes_for_chexpert_labeler)

        if save_for_error_analysis:
            _save_results_for_error_analysis(
                results_dict=results_dict,
                dataset_name='mimiccxr',
                results_folder_path=results_folder_path,
                classify_orientation=classify_orientation,
                classify_chexpert=classify_chexpert,
                classify_questions=classify_questions,
                classify_chest_imagenome=classify_chest_imagenome,
                chest_imagenome_labels=mimiccxr_vqa_evaluator.chest_imagenome_labels,
                parenthesis_text=eval_mode_text)

    # if eval_iuxray:
    #     print('\n========================')
    #     count_print('Running evaluator engine on IU X-Ray validation split ...')
    #     print('len(dataset) =', len(iuxray_vqa_trainer.val_dataset))
    #     print('len(dataloader) =', len(iuxray_vqa_trainer.val_dataloader))
    #     evaluator.run(iuxray_vqa_trainer.val_dataloader)
        
    #     results_dict['iuxray_metrics'] = deepcopy(evaluator.state.metrics)            
    #     results_dict['iuxray_dataset'] = iuxray_vqa_trainer.val_dataset
    #     results_dict['iuxray_reports'] = recover_reports(
    #         results_dict['iuxray_metrics'],
    #         results_dict['iuxray_dataset'],
    #         tokenizer, eval_mode, iuxray_qa_reports,
    #         verbose_question=verbose_question,
    #     )
    #     results_dict['iuxray_report_metrics'] = _compute_and_save_report_level_metrics(
    #         results_dict, 'iuxray', tokenizer, results_folder_path, parenthesis_text=eval_mode_text,
    #         max_processes=max_processes_for_chexpert_labeler)

    torch.cuda.empty_cache()
    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder,
    eval_mode,
    answer_decoding,
    n_questions_per_report = None,
    qclass_threshold = None,
    eval_checkpoint_folder = None,
    precomputed_question_probs_path = None,
    precomputed_question_thresholds_path = None,
    use_random_image = False,
    batch_size = 100,
    num_workers = 0,    
    device = 'GPU',
    return_results = False,
    use_amp = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
    max_processes_for_chexpert_labeler = 10,
    save_for_error_analysis = False,
    mimiccxr_eval_view_mode = None,
):
    print('----- Evaluating model ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    if eval_checkpoint_folder is not None:
        eval_checkpoint_folder =os.path.join(WORKSPACE_DIR, eval_checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    model_kwargs = metadata['model_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    image_transform_kwargs = metadata['train_image_transform_kwargs']
    image_transform_kwargs['augmentation_mode'] = None # no data augmentation
    if eval_mimiccxr:
        mimiccxr_vqa_evaluator_kwargs = metadata['mimiccxr_vqa_trainer_kwargs']
        mimiccxr_vqa_evaluator_kwargs['batch_size'] = batch_size
        mimiccxr_vqa_evaluator_kwargs['use_random_image'] = use_random_image
    else:
        mimiccxr_vqa_evaluator_kwargs = None
    if eval_iuxray:
        iuxray_vqa_trainer_kwargs = metadata['iuxray_vqa_trainer_kwargs']
        iuxray_vqa_trainer_kwargs['batch_size'] = batch_size
    else:
        iuxray_vqa_trainer_kwargs = None
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']
    trainer_engine_kwargs = metadata['trainer_engine_kwargs']

    return _evaluate_model(
                tokenizer_kwargs,
                model_kwargs,
                dataloading_kwargs,
                image_transform_kwargs,
                mimiccxr_vqa_evaluator_kwargs,
                iuxray_vqa_trainer_kwargs,
                auxiliary_tasks_kwargs,
                trainer_engine_kwargs,
                answer_decoding,
                eval_mode,
                n_questions_per_report = n_questions_per_report,
                qclass_threshold = qclass_threshold,
                device = device,
                num_workers = num_workers,
                checkpoint_folder_path = checkpoint_folder,
                eval_checkpoint_folder_path = eval_checkpoint_folder,
                precomputed_question_probs_path = precomputed_question_probs_path,
                precomputed_question_thresholds_path = precomputed_question_thresholds_path,
                return_results = return_results,
                use_random_image = use_random_image,
                use_amp = use_amp,
                eval_iuxray = eval_iuxray,
                eval_mimiccxr = eval_mimiccxr,
                max_processes_for_chexpert_labeler = max_processes_for_chexpert_labeler,
                save_for_error_analysis = save_for_error_analysis,
                mimiccxr_eval_view_mode = mimiccxr_eval_view_mode,
            )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)