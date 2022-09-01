import  os
import argparse
import numpy as np

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.utils import deduplicate_indices
from medvqa.metrics.classification.multilabel_prf1 import MultiLabelPRF1
from medvqa.models.report_generation.templates.chex_v1 import TEMPLATES_CHEXPERT_v1

from medvqa.models.vqa.open_ended_vqa import (
    QuestionEncoding,
    does_include_image,
    does_include_visual_features,
)
from medvqa.utils.constants import (
    CHEXPERT_DATASET_ID,
    CHEXPERT_LABELS,
    MIMICCXR_DATASET_ID,
    MetricNames,
    ReportEvalMode,
)
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.metrics import (
    attach_dataset_aware_chexpert_labels_accuracy,
    attach_dataset_aware_chexpert_labels_macroavgf1,
    attach_dataset_aware_chexpert_labels_microavgf1,
    attach_dataset_aware_chexpert_labels_roc_auc,
    attach_dataset_aware_question_labels_macroavgf1,
    attach_dataset_aware_question_labels_microavgf1,
    attach_exactmatch_question,
    attach_medical_tags_f1score,
    attach_dataset_aware_orientation_accuracy,
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
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter
from medvqa.evaluation.report_generation import recover_reports

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
    parser.add_argument('--answer-decoding', type=str, default='greedy-search')

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)
    
    return parser.parse_args()

def _estimate_maximum_answer_length(tokenizer):
    def answer_generator():
        for label in CHEXPERT_LABELS:
            for i in range(2):
                yield TEMPLATES_CHEXPERT_v1[label][i]
    return max(len(tokenizer.tokenize(s)) for s in answer_generator()) + 3

def _recover_chexpert(chexpert, pred_chexpert, idxs, report_ids, gen_reports):
        rids = [report_ids[i] for i in idxs]
        indices = deduplicate_indices(list(range(len(rids))), rids)
        _pred_chexpert_vqa = []
        for r in gen_reports:
            labels = []
            for l in CHEXPERT_LABELS:
                i = r['q'].index(l)
                a = r['a'][i]
                if TEMPLATES_CHEXPERT_v1[l][0] == a:
                    labels.append(0)
                elif TEMPLATES_CHEXPERT_v1[l][1] == a:
                    labels.append(1)
                else: assert False
            _pred_chexpert_vqa.append(labels)
        _pred_chexpert_vqa = np.array(_pred_chexpert_vqa)
        return dict(
            chexpert = torch.stack([chexpert[i] for i in indices]),
            pred_chexpert = torch.stack([pred_chexpert[i] for i in indices]),
            pred_chexpert_vqa = _pred_chexpert_vqa,
        )

def _compute_metrics(chexpert, pred_chexpert, pred_chexpert_vqa):
    output = {}
    met = MultiLabelPRF1(device='cpu')
    names = ['pred_chexpert_vqa', 'pred_chexpert', 'chexpert']
    labels = [pred_chexpert_vqa, pred_chexpert, chexpert]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            met.update((labels[i], labels[j]))
            output[(names[i], names[j])] = met.compute()
            met.reset()
    return output

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
    num_workers = 0,
    device = 'GPU',
    checkpoint_folder_path = None,
    use_amp = False,
):
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
    assert classify_chexpert

    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)
    n_questions_aux_task = auxiliary_tasks_kwargs.get('n_questions_aux_task', None)        
    if classify_questions:
        assert n_questions_aux_task is not None        
    
    if question_encoding == QuestionEncoding.ONE_HOT:
        assert model_kwargs['n_questions'] is not None
    
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
    max_answer_length = _estimate_maximum_answer_length(tokenizer)
    print('max_answer_length =', max_answer_length)
    
    # Default image transform (no augmentations)
    count_print('Defining image transform ...')
    img_transform = get_image_transform(**image_transform_kwargs)

    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    
    one_hot_question_offsets = dataloading_kwargs.get('one_hot_question_offsets', None)
    if not verbose_question: assert one_hot_question_offsets is not None
    
    mimiccxr_collate_batch_fn = get_vqa_collate_batch_fn(
        MIMICCXR_DATASET_ID,
        verbose_question = verbose_question,
        one_hot_question_offset = one_hot_question_offsets[str(CHEXPERT_DATASET_ID)],
        include_image = include_image,
        include_visual_features = include_visual_features,
        include_answer=False,
        classify_tags = classify_tags,
        n_tags = n_medical_tags,
        classify_orientation = classify_orientation,
        classify_chexpert = classify_chexpert,
        classify_questions = classify_questions)

    # Load saved checkpoint    
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    
    # Create MIMIC-CXR vqa evaluator
    count_print('Creating MIMIC-CXR vqa evaluator ...')
    mimiccxr_vqa_evaluator = MIMICCXR_VQA_Evaluator(
        transform = img_transform,
        collate_batch_fn = mimiccxr_collate_batch_fn,
        num_workers = num_workers,
        tokenizer = tokenizer,
        report_eval_mode = ReportEvalMode.CHEXPERT_LABELS,
        image_local_feat_size = model_kwargs['image_local_feat_size'],
        n_questions_aux_task = model_kwargs['n_questions_aux_task'],
        pretrained_weights = checkpoint['model'],
        pretrained_checkpoint_path = checkpoint_path,
        **mimiccxr_vqa_evaluator_kwargs,
    )

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
                           classify_questions, question_encoding, answer_decoding,
                           device, use_amp=use_amp, training=False, include_answer=False,
                           include_image=include_image, include_visual_features=include_visual_features,
                           max_answer_length=max_answer_length,
                           use_merged_findings=use_merged_findings)

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    # Metrics
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
        attach_dataset_aware_orientation_accuracy(evaluator, [MIMICCXR_DATASET_ID], record_scores=True)
    if classify_chexpert:
        attach_dataset_aware_chexpert_labels_accuracy(evaluator, [MIMICCXR_DATASET_ID], _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_macroavgf1(evaluator, [MIMICCXR_DATASET_ID], _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_microavgf1(evaluator, [MIMICCXR_DATASET_ID], _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_roc_auc(evaluator, [MIMICCXR_DATASET_ID], 'cpu', _chexpert_class_indices)        
    if classify_questions:
        attach_dataset_aware_question_labels_macroavgf1(evaluator, [MIMICCXR_DATASET_ID])
        attach_dataset_aware_question_labels_microavgf1(evaluator, [MIMICCXR_DATASET_ID])

    # Accumulators
    attach_accumulator(evaluator, 'idxs')
    attach_accumulator(evaluator, 'pred_answers')
    attach_accumulator(evaluator, 'chexpert')
    attach_accumulator(evaluator, 'pred_chexpert')
    
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
    if classify_questions:
        metrics_to_print.append(MetricNames.QLABELS_MACROAVGF1)
        metrics_to_print.append(MetricNames.QLABELS_MICROAVGF1)

    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()

    # Attach handlers
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation
    count_print('Running evaluator engine on MIMIC-CXR test split ...')
    print('len(dataset) =', len(mimiccxr_vqa_evaluator.test_dataset))
    print('len(dataloader) =', len(mimiccxr_vqa_evaluator.test_dataloader))
    evaluator.run(mimiccxr_vqa_evaluator.test_dataloader)

    if use_merged_findings:
        _findings_remapper = trainer_engine_kwargs['findings_remapper']
        _chexpert_class_indices = _findings_remapper[str(CHEXPERT_DATASET_ID)]

    results_folder_path = get_results_folder_path(checkpoint_folder_path)
    reports = recover_reports(
        metrics_dict=evaluator.state.metrics,
        dataset=mimiccxr_vqa_evaluator.test_dataset,
        tokenizer=tokenizer,
        qa_adapted_dataset=mimiccxr_qa_reports,
        verbose_question=verbose_question,
        report_eval_mode=ReportEvalMode.CHEXPERT_LABELS,
    )
    _gt_chexpert = evaluator.state.metrics['chexpert']
    _pred_chexpert = evaluator.state.metrics['pred_chexpert']
    if use_merged_findings:
        _gt_chexpert = torch.stack(_gt_chexpert)[:, _chexpert_class_indices]
        _pred_chexpert = torch.stack(_pred_chexpert)[:, _chexpert_class_indices]
    chexperts = _recover_chexpert(
        chexpert=_gt_chexpert,
        pred_chexpert=_pred_chexpert,
        idxs=evaluator.state.metrics['idxs'],
        report_ids=mimiccxr_vqa_evaluator.test_dataset.report_ids,
        gen_reports=reports['gen_reports'],
    ) 
    output = dict(
        gt_reports=reports['gt_reports'],
        gen_reports=reports['gen_reports'],
        gt_chexpert=_gt_chexpert,
        pred_chexpert=_pred_chexpert,
        metrics=_compute_metrics(**chexperts),
    )
    save_path = os.path.join(results_folder_path, f'mimiccxr_chexpert_based_output.pkl')
    save_to_pickle(output, save_path)
    print (f'Chexpert-based output saved to {save_path}')

    torch.cuda.empty_cache()

def evaluate_model(
    checkpoint_folder,
    answer_decoding,
    batch_size = 100,
    num_workers = 0,    
    device = 'GPU',
    use_amp = False,
):
    print('----- Evaluating model ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)    
    metadata = load_metadata(checkpoint_folder)
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    model_kwargs = metadata['model_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    image_transform_kwargs = metadata['image_transform_kwargs']
    image_transform_kwargs['augmentation_mode'] = None # no data augmentation
    mimiccxr_vqa_evaluator_kwargs = metadata['mimiccxr_vqa_trainer_kwargs']
    mimiccxr_vqa_evaluator_kwargs['batch_size'] = batch_size
    iuxray_vqa_trainer_kwargs = metadata['iuxray_vqa_trainer_kwargs']
    iuxray_vqa_trainer_kwargs['batch_size'] = batch_size
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']
    trainer_engine_kwargs = metadata.get('trainer_engine_kwargs', {})

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
                device = device,
                num_workers = num_workers,
                checkpoint_folder_path = checkpoint_folder,
                use_amp = use_amp,
            )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)