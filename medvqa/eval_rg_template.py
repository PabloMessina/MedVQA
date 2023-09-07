import gc
import  os
import argparse
import numpy as np

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_label_names_and_templates,
    load_chest_imagenome_labels,
)
from medvqa.datasets.dataloading_utils import get_vision_collate_batch_fn
from medvqa.datasets.image_processing import get_image_transform
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.evaluation.visual_module import calibrate_thresholds_on_mimiccxr_validation_set
from medvqa.metrics.medical.chexbert import CheXbertLabeler
from medvqa.metrics.medical.chexpert import ChexpertLabeler
from medvqa.models.report_generation.templates.chexpert import TEMPLATES_CHEXPERT_v1, TEMPLATES_CHEXPERT_v2
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule

from medvqa.utils.constants import (
    CHEXBERT_LABELS,
    CHEXPERT_LABELS,
    DATASET_NAMES,
    MIMICCXR_DATASET_ID,
    MetricNames,
)
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR, get_imageId2reportId, load_mimiccxr_reports_detailed_metadata
from medvqa.metrics import (
    attach_chexpert_labels_prf1,
    attach_chexpert_labels_roc_auc,
    attach_dataset_aware_chest_imagenome_bbox_iou,
    attach_medical_tags_f1score,
    attach_chexpert_labels_accuracy,
    attach_dataset_aware_orientation_accuracy,
    attach_question_labels_prf1,
    attach_chest_imagenome_labels_accuracy,
    attach_chest_imagenome_labels_prf1,
    attach_chest_imagenome_labels_roc_auc,
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
    get_log_metrics_handler,
    get_log_iteration_handler,
    attach_accumulator,
)
from medvqa.utils.files import (
    get_cached_json_file,
    get_cached_pickle_file,
    get_checkpoint_folder_path,
    get_results_folder_path,
    save_pickle,
)
from medvqa.training.vision import get_engine
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleTrainer
from medvqa.utils.logging import CountPrinter, print_blue, print_bold
from medvqa.evaluation.report_generation import (
    TemplateBasedModes,
    compute_report_level_metrics,
    recover_reports__template_based,
)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--template_based_mode', type=str, required=True, choices=TemplateBasedModes.get_choices())

    # optional arguments
    parser.add_argument('--checkpoint_folder', type=str)
    parser.add_argument('--calibrate_thresholds', action='store_true', default=False)
    parser.add_argument('--calibration_score_name', type=str, default='f1', choices=['f1', 'precision', 'accuracy'])
    parser.add_argument('--batch_size', type=int, default=140)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_processes_for_chexpert_labeler', type=int, default=4)
    parser.add_argument('--mimiccxr_qa_adapted_reports_filename', type=str, default=None)
    parser.add_argument('--chest_imagenome_label_names_filename', type=str)
    parser.add_argument('--chest_imagenome_labels_filename', type=str)
    parser.add_argument('--top_k_chexpert_labels', type=int, default=None, help='if None, use all labels')
    parser.add_argument('--top_k_chest_imagenome_labels', type=int, default=None, help='if None, use all labels')
    parser.add_argument('--label_score_threshold', type=float, default=None, help='if not None, keep only labels with score >= threshold')
    parser.add_argument('--background_findings_and_impression_per_report_filepath', type=str, default=None)
    parser.add_argument('--fact_embedding_cluster_labels_per_report_filepath', type=str, default=None)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--cache_computations', action='store_true', default=False)
    
    return parser.parse_args()

_BEST_CHEXPERT_ORDER = [
    'Cardiomegaly',
    'Enlarged Cardiomediastinum',
    'Consolidation',
    'Lung Opacity',
    'Atelectasis',
    'Support Devices',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Edema',
    'Lung Lesion',
    'Fracture',
]

def _compute_and_save_report_level_metrics(
        gt_reports, gen_reports, dataset_name, results_folder_path, max_processes, template_based_mode,
        calibrate_thresholds=False, calibration_score_name=None, top_k_chexpert_labels=None, tot_chexpert_labels=None,
        top_k_chest_imagenome_labels=None, tot_chest_imagenome_labels=None, label_score_threshold=None
    ):
    metrics = compute_report_level_metrics(gt_reports, gen_reports, max_processes=max_processes)
    template_based_mode = template_based_mode.replace('_','-')
    strings = ['template-based', template_based_mode]
    if calibrate_thresholds:
        strings.append('thrs-calib')
        assert calibration_score_name is not None
        strings.append(f'calib-score={calibration_score_name}')
    if label_score_threshold is not None: strings.append(f'lbl-thr={label_score_threshold:.2f}')
    if top_k_chexpert_labels is not None:
        assert tot_chexpert_labels is not None
        strings.append(f'chxp-top{top_k_chexpert_labels}(tot={tot_chexpert_labels})')
    if top_k_chest_imagenome_labels is not None:
        assert tot_chest_imagenome_labels is not None
        strings.append(f'chst-img-top{top_k_chest_imagenome_labels}(tot={tot_chest_imagenome_labels})')
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_report_level_metrics(eval_mode={",".join(strings)}).pkl')
    save_pickle(metrics, save_path)
    print('Report-level metrics successfully saved to ', end=''); print_bold(save_path)
    return metrics

def _save_gen_reports(gen_reports, report_idxs, dataset_name, results_folder_path, template_based_mode):
    template_based_mode = template_based_mode.replace('_','-')
    strings = ['template-based', template_based_mode]
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_gen_reports(eval_mode={",".join(strings)}).pkl')
    save_pickle({
        'gen_reports': gen_reports,
        'report_idxs': report_idxs,
    }, save_path)
    print('Generated reports successfully saved to ', end=''); print_bold(save_path)

def _recover_tokenizer_kwargs(metadata):
    return metadata['tokenizer_kwargs']

def _recover_model_kwargs(metadata):
    kwargs = metadata['model_kwargs']
    kwargs.update(metadata['auxiliary_tasks_kwargs'])
    return kwargs

def _recover_vision_dataset_manager_kwargs(
        dataset_name, metadata, batch_size, num_workers, qa_adapted_reports_filename):
    keys = [
        f'{dataset_name}_vision_trainer_kwargs',
        f'{dataset_name}_vqa_trainer_kwargs',
        f'{dataset_name}_trainer_kwargs',
    ]
    kwargs = None
    for key in keys:
        if key in metadata:
            kwargs = metadata[key]
            break    
    assert kwargs is not None
    kwargs['use_test_set'] = True
    kwargs['data_augmentation_enabled'] = False
    kwargs['batch_size'] = batch_size
    kwargs['num_workers'] = num_workers
    if qa_adapted_reports_filename is not None:
        kwargs['qa_adapted_reports_filename'] = qa_adapted_reports_filename

    # Define collate_batch_fn
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    mimiccxr_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR])
    kwargs['collate_batch_fn'] = mimiccxr_collate_batch_fn

    # Define test image transform
    image_transform_kwargs = metadata['val_image_transform_kwargs']
    image_transform = get_image_transform(**image_transform_kwargs[DATASET_NAMES.MIMICCXR])
    kwargs['test_image_transform'] = image_transform
    
    return kwargs

def _recover_mimiccxr_vision_evaluator_kwargs(
        metadata, batch_size, num_workers, qa_adapted_reports_filename=None):
    return _recover_vision_dataset_manager_kwargs(
        'mimiccxr', metadata, batch_size, num_workers, qa_adapted_reports_filename)

def _evaluate_chest_imagenome_template_based_oracle(
        chest_imagenome_label_names_filename,
        chest_imagenome_labels_filename,
        background_findings_and_impression_per_report_filepath,
        max_processes_for_chexpert_labeler,
        template_based_mode,
):
    print('Evaluating chest imagenome template-based oracle ...')
    assert chest_imagenome_label_names_filename is not None
    assert chest_imagenome_labels_filename is not None
    label_names, label_templates = load_chest_imagenome_label_names_and_templates(chest_imagenome_label_names_filename)
    labels_dict = load_chest_imagenome_labels(chest_imagenome_labels_filename)
    mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata(background_findings_and_impression_per_report_filepath=
                                                                         background_findings_and_impression_per_report_filepath)
    test_idxs = [i for i, split in enumerate(mimiccxr_detailed_metadata['splits']) if split == 'test']

    gt_reports = []
    gen_reports = []
    actual_report_idxs = []
    for idx in test_idxs:
        # gen report
        label_found = False
        dicom_id_view_pairs = mimiccxr_detailed_metadata['dicom_id_view_pos_pairs'][idx]
        for dicom_id, _ in dicom_id_view_pairs:
            if dicom_id in labels_dict:
                label = labels_dict[dicom_id]
                label_found = True                
                break
        if not label_found:
            continue # skip this report
        gen_report = ""
        for i, label_name in enumerate(label_names):
            text = label_templates[label_name][label[i]]
            if text:
                if gen_report:
                    gen_report += ' ' if gen_report[-1] == '.' else '. '
                gen_report += text
        gen_reports.append(gen_report)
        actual_report_idxs.append(idx)
        # gt report
        findings = mimiccxr_detailed_metadata['findings'][idx]
        impression = mimiccxr_detailed_metadata['impressions'][idx]
        gt_report = ""
        for text in (findings, impression):
            if gt_report:
                gt_report += ' ' if gt_report[-1] == '.' else '. '
            gt_report += text
        gt_reports.append(gt_report)    

    results_folder_path = get_results_folder_path(get_checkpoint_folder_path('report_gen', 'mimiccxr', 'oracle'))
    _compute_and_save_report_level_metrics(
        gt_reports, gen_reports, 'mimiccxr', results_folder_path,
        max_processes_for_chexpert_labeler, template_based_mode)
    _save_gen_reports(gen_reports, actual_report_idxs, 'mimiccxr', results_folder_path, template_based_mode)

def _evaluate_chexpert_labels_template_based_oracle(
        background_findings_and_impression_per_report_filepath,
        max_processes_for_chexpert_labeler,
        template_based_mode,
):
    print('Evaluating chexpert labels template-based oracle ...')
    mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata(background_findings_and_impression_per_report_filepath=
                                                                         background_findings_and_impression_per_report_filepath)
    test_idxs = [i for i, split in enumerate(mimiccxr_detailed_metadata['splits']) if split == 'test']

    # gt reports
    gt_reports = []
    for idx in test_idxs:
        findings = mimiccxr_detailed_metadata['findings'][idx]
        impression = mimiccxr_detailed_metadata['impressions'][idx]
        gt_report = ""
        for text in (findings, impression):
            if gt_report:
                gt_report += ' ' if gt_report[-1] == '.' else '. '
            gt_report += text
        gt_reports.append(gt_report)
    
    # extract chexpert labels from gt reports
    chexpert_labeler = ChexpertLabeler(verbose=True)
    gt_labels = chexpert_labeler.get_labels(gt_reports, update_cache_on_disk=True,
                                            max_processes=max_processes_for_chexpert_labeler)
    assert len(gt_labels) == len(gt_reports)

    # gen reports
    gen_reports = []
    for labels in gt_labels:
        assert len(labels) == len(_BEST_CHEXPERT_ORDER) + 1
        sentences = []
        for label_name in _BEST_CHEXPERT_ORDER:
            label_idx = CHEXPERT_LABELS.index(label_name)
            sentences.append(TEMPLATES_CHEXPERT_v1[label_name][labels[label_idx]])
        gen_report = '. '.join(s for s in sentences if s)
        gen_reports.append(gen_report)

    results_folder_path = get_results_folder_path(get_checkpoint_folder_path('report_gen', 'mimiccxr', 'oracle'))
    _compute_and_save_report_level_metrics(
        gt_reports, gen_reports, 'mimiccxr', results_folder_path,
        max_processes_for_chexpert_labeler, template_based_mode)
    _save_gen_reports(gen_reports, test_idxs, 'mimiccxr', results_folder_path, template_based_mode)

def _evaluate_chexbert_labels_template_based_oracle(
        background_findings_and_impression_per_report_filepath,
        max_processes_for_chexpert_labeler,
        template_based_mode,
):
    print('Evaluating chexbert labels template-based oracle ...')
    mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata(background_findings_and_impression_per_report_filepath=
                                                                         background_findings_and_impression_per_report_filepath)
    test_idxs = [i for i, split in enumerate(mimiccxr_detailed_metadata['splits']) if split == 'test']

    # gt reports
    gt_reports = []
    for idx in test_idxs:
        findings = mimiccxr_detailed_metadata['findings'][idx]
        impression = mimiccxr_detailed_metadata['impressions'][idx]
        gt_report = ""
        for text in (findings, impression):
            if gt_report:
                gt_report += ' ' if gt_report[-1] == '.' else '. '
            gt_report += text
        gt_reports.append(gt_report)
    
    # extract chexbert labels from gt reports
    chexbert_labeler = CheXbertLabeler(verbose=True)
    gt_labels = chexbert_labeler.get_labels(gt_reports, update_cache_on_disk=True)
    assert len(gt_labels) == len(gt_reports)

    # gen reports
    gen_reports = []
    for labels in gt_labels:
        assert len(labels) == len(_BEST_CHEXPERT_ORDER) + 1
        sentences = []
        for label_name in _BEST_CHEXPERT_ORDER:
            label_idx = CHEXBERT_LABELS.index(label_name)
            sentences.append(TEMPLATES_CHEXPERT_v1[label_name][labels[label_idx]])
        gen_report = '. '.join(s for s in sentences if s)
        gen_reports.append(gen_report)

    results_folder_path = get_results_folder_path(get_checkpoint_folder_path('report_gen', 'mimiccxr', 'oracle'))
    _compute_and_save_report_level_metrics(
        gt_reports, gen_reports, 'mimiccxr', results_folder_path,
        max_processes_for_chexpert_labeler, template_based_mode)
    _save_gen_reports(gen_reports, test_idxs, 'mimiccxr', results_folder_path, template_based_mode)

    
def _evaluate_fact_embedding_template_based_oracle(
        background_findings_and_impression_per_report_filepath,
        fact_embedding_cluster_labels_per_report_filepath,
        max_processes_for_chexpert_labeler,
        template_based_mode,
):
    print('Evaluating fact embedding template-based oracle ...')
    assert background_findings_and_impression_per_report_filepath is not None
    assert fact_embedding_cluster_labels_per_report_filepath is not None
    
    mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata(background_findings_and_impression_per_report_filepath=
                                                                         background_findings_and_impression_per_report_filepath)
    test_idxs = [i for i, split in enumerate(mimiccxr_detailed_metadata['splits']) if split == 'test']
    feclpr_data = get_cached_pickle_file(fact_embedding_cluster_labels_per_report_filepath)
    top_label_names = feclpr_data['top_label_names']
    top_label_ids = feclpr_data['top_label_ids']
    label_id_2_name = {x:y for x,y in zip(top_label_ids, top_label_names)}
    labeled_reports = feclpr_data['labeled_reports']
    
    gt_reports = []
    gen_reports = []
    for i in test_idxs:
        assert mimiccxr_detailed_metadata['filepaths'][i] == labeled_reports[i]['report_path']
        # gt report
        findings = mimiccxr_detailed_metadata['findings'][i]
        impression = mimiccxr_detailed_metadata['impressions'][i]
        gt_report = ""
        for text in (findings, impression):
            if gt_report:
                gt_report += ' ' if gt_report[-1] == '.' else '. '
            gt_report += text
        gt_reports.append(gt_report)
        # gen report
        gen_report = ""
        for label in labeled_reports[i]['labels']:
            label_name = label_id_2_name[label]
            if gen_report:
                gen_report += ' ' if gen_report[-1] == '.' else '. '
            gen_report += label_name
        gen_reports.append(gen_report)

    results_folder_path = get_results_folder_path(get_checkpoint_folder_path('report_gen', 'mimiccxr', 'oracle'))
    _compute_and_save_report_level_metrics(
        gt_reports, gen_reports, 'mimiccxr', results_folder_path,
        max_processes_for_chexpert_labeler, template_based_mode)
    _save_gen_reports(gen_reports, test_idxs, 'mimiccxr', results_folder_path, template_based_mode)

def _compute_probs_and_gt_labels_for_mimiccxr_test_set(
        auxiliary_tasks_kwargs, checkpoint_folder_path, model_kwargs, mimiccxr_vision_evaluator_kwargs,
        use_amp, save_probs_and_gt_labels=True):

    # Pull out some args from kwargs
    # auxiliary task: medical tags prediction
    classify_tags = auxiliary_tasks_kwargs.get('classify_tags', False)
    # auxiliary task: gender classification
    classify_gender = auxiliary_tasks_kwargs.get('classify_gender', False)
    # auxiliary task: orientation classification
    classify_orientation = auxiliary_tasks_kwargs.get('classify_orientation', False)
    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs['classify_chest_imagenome']
    # auxiliary task: chest imagenome bounding boxes
    predict_bboxes_chest_imagenome = auxiliary_tasks_kwargs['predict_bboxes_chest_imagenome']
    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)

    assert classify_chexpert or classify_chest_imagenome

    results_folder_path = get_results_folder_path(checkpoint_folder_path)

    skip_chexpert = not classify_chexpert
    skip_chest_imagenome = not classify_chest_imagenome
    if classify_chexpert:
        chexpert_probs_path = os.path.join(results_folder_path, 'dicom_id_to_pred_chexpert_probs__mimiccxr_test_set.pkl')
        chexpert_gt_path = os.path.join(results_folder_path, 'dicom_id_to_gt_chexpert_labels__mimiccxr_test_set.pkl')
        if os.path.exists(chexpert_probs_path):
            assert os.path.exists(chexpert_gt_path)
            print_bold('Loading cached chexpert predictions and ground truth...')
            dicom_id_to_pred_chexpert_probs = get_cached_pickle_file(chexpert_probs_path)
            dicom_id_to_gt_chexpert_labels = get_cached_pickle_file(chexpert_gt_path)
            print(f'len(dicom_id_to_pred_chexpert_probs): {len(dicom_id_to_pred_chexpert_probs)}')
            print(f'len(dicom_id_to_gt_chexpert_labels): {len(dicom_id_to_gt_chexpert_labels)}')
            assert len(dicom_id_to_pred_chexpert_probs) == len(dicom_id_to_gt_chexpert_labels)
            skip_chexpert = True
    if classify_chest_imagenome:
        assert auxiliary_tasks_kwargs['classify_chest_imagenome']
        chest_imagenome_probs_path = os.path.join(results_folder_path, 'dicom_id_to_pred_chest_imagenome_probs__mimiccxr_test_set.pkl')
        chest_imagenome_gt_path = os.path.join(results_folder_path, 'dicom_id_to_gt_chest_imagenome_labels__mimiccxr_test_set.pkl')
        if os.path.exists(chest_imagenome_probs_path):
            assert os.path.exists(chest_imagenome_gt_path)
            print_bold('Loading cached chest imagenome predictions and ground truth...')
            dicom_id_to_pred_chest_imagenome_probs = get_cached_pickle_file(chest_imagenome_probs_path)
            dicom_id_to_gt_chest_imagenome_labels = get_cached_pickle_file(chest_imagenome_gt_path)
            print(f'len(dicom_id_to_pred_chest_imagenome_probs): {len(dicom_id_to_pred_chest_imagenome_probs)}')
            print(f'len(dicom_id_to_gt_chest_imagenome_labels): {len(dicom_id_to_gt_chest_imagenome_labels)}')
            assert len(dicom_id_to_pred_chest_imagenome_probs) == len(dicom_id_to_gt_chest_imagenome_labels)
            skip_chest_imagenome = True

    if skip_chexpert and skip_chest_imagenome:
        output = {}
        if classify_chexpert:
            output['dicom_id_to_pred_chexpert_probs'] = dicom_id_to_pred_chexpert_probs
            output['dicom_id_to_gt_chexpert_labels'] = dicom_id_to_gt_chexpert_labels
        if classify_chest_imagenome:
            output['dicom_id_to_pred_chest_imagenome_probs'] = dicom_id_to_pred_chest_imagenome_probs
            output['dicom_id_to_gt_chest_imagenome_labels'] = dicom_id_to_gt_chest_imagenome_labels
        return output

    print_blue('Computing probs and gt labels for mimiccxr test set from scratch...', bold=True)
    
    count_print = CountPrinter()
        
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    count_print('device =', device)

    # Load saved checkpoint    
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    # Create model
    count_print('Creating model ...')
    model = MultiPurposeVisualModule(**model_kwargs)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Create MIMIC-CXR visual module evaluator
    count_print('Creating MIMIC-CXR visual module evaluator ...')
    mimiccxr_vision_evaluator = MIMICCXR_VisualModuleTrainer(**mimiccxr_vision_evaluator_kwargs)

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    _engine_kwargs = dict(
        model=model, classify_tags=classify_tags, classify_orientation=classify_orientation,
        classify_gender=classify_gender, classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        pass_pred_bbox_coords_as_input=mimiccxr_vision_evaluator_kwargs.get('pass_pred_bbox_coords_to_model', False),
        device=device, use_amp=use_amp, training=False,
        using_yolov8=mimiccxr_vision_evaluator_kwargs.get('use_yolov8', False),
    )
    evaluator_engine = get_engine(**_engine_kwargs)

    # Attach metrics, timer and events to engines
    count_print('Attaching metrics, timer and events to engines ...')

    # Metrics & Logging
    metrics_to_print = []

    if classify_tags:
        attach_medical_tags_f1score(evaluator_engine, device)
        metrics_to_print.append(MetricNames.MEDTAGF1)
    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(evaluator_engine)
        metrics_to_print.append(MetricNames.ORIENACC)
    if classify_chexpert:
        attach_chexpert_labels_accuracy(evaluator_engine, device)        
        attach_chexpert_labels_prf1(evaluator_engine, device)
        attach_chexpert_labels_roc_auc(evaluator_engine, 'cpu')
        metrics_to_print.append(MetricNames.CHXLABEL_PRF1)
        metrics_to_print.append(MetricNames.CHXLABELACC)
        metrics_to_print.append(MetricNames.CHXLABEL_ROCAUC)
    if classify_questions:
        attach_question_labels_prf1(evaluator_engine, device)
        metrics_to_print.append(MetricNames.QLABELS_PRF1)
    if classify_chest_imagenome:
        attach_chest_imagenome_labels_accuracy(evaluator_engine, device)
        attach_chest_imagenome_labels_prf1(evaluator_engine, device)
        attach_chest_imagenome_labels_roc_auc(evaluator_engine, 'cpu')
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABEL_PRF1)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELACC)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELROCAUC)
    if predict_bboxes_chest_imagenome:
        if mimiccxr_vision_evaluator_kwargs.get('use_yolov8', False):
            attach_dataset_aware_chest_imagenome_bbox_iou(evaluator_engine, [MIMICCXR_DATASET_ID], use_yolov8=True)
        else:
            attach_dataset_aware_chest_imagenome_bbox_iou(evaluator_engine, [MIMICCXR_DATASET_ID])
        metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXIOU)

    # Accumulators
    attach_accumulator(evaluator_engine, 'idxs')
    if classify_chexpert:
        attach_accumulator(evaluator_engine, 'pred_chexpert_probs')
        attach_accumulator(evaluator_engine, 'chexpert')
    if classify_chest_imagenome:
        attach_accumulator(evaluator_engine, 'pred_chest_imagenome_probs')
        attach_accumulator(evaluator_engine, 'chest_imagenome')

    # Timer
    timer = Timer()
    timer.attach(evaluator_engine, start=Events.EPOCH_STARTED)

    # Attach handlers
    log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()
    evaluator_engine.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

    # Run evaluation
    count_print(f'Running evaluator engine on MIMIC-CXR test set ...')
    print('len(mimiccxr_vision_evaluator.test_dataset) =', len(mimiccxr_vision_evaluator.test_dataset))
    print('len(mimiccxr_vision_evaluator.test_dataloader) =', len(mimiccxr_vision_evaluator.test_dataloader))
    evaluator_engine.run(mimiccxr_vision_evaluator.test_dataloader)
    
    # Prepare probabilities and ground truth labels
    count_print(f'Preparing probs and gt labels ...')
    output = {}
    label_names = []
    if classify_chest_imagenome:
        label_names.append('chest_imagenome')
    if classify_chexpert:
        label_names.append('chexpert')
    
    for label_name in label_names:
        pred_probs_key = f'pred_{label_name}_probs'
        gt_labels_key = f'{label_name}'
        dicom_id_to_pred_probs = {}
        dicom_id_to_gt_labels = {}
        pred_probs = evaluator_engine.state.metrics[pred_probs_key]
        gt_labels = evaluator_engine.state.metrics[gt_labels_key]
        assert len(mimiccxr_vision_evaluator.test_indices) == len(pred_probs)
        for i, idx in enumerate(mimiccxr_vision_evaluator.test_indices):
            dicom_id = mimiccxr_vision_evaluator.dicom_ids[idx]
            dicom_id_to_pred_probs[dicom_id] = pred_probs[i].detach().cpu().numpy()
            dicom_id_to_gt_labels[dicom_id] = gt_labels[i].detach().cpu().numpy()
        
        output[f'dicom_id_to_pred_{label_name}_probs'] = dicom_id_to_pred_probs
        output[f'dicom_id_to_gt_{label_name}_labels'] = dicom_id_to_gt_labels
        
        # Save probabilities and ground truth labels        
        if save_probs_and_gt_labels:
            pred_probs_save_name = f'dicom_id_to_pred_{label_name}_probs__mimiccxr_test_set.pkl'
            pred_probs_save_path = os.path.join(results_folder_path, pred_probs_save_name)
            gt_labels_save_name = f'dicom_id_to_gt_{label_name}_labels__mimiccxr_test_set.pkl'
            gt_labels_save_path = os.path.join(results_folder_path, gt_labels_save_name)
            save_pickle(dicom_id_to_pred_probs, pred_probs_save_path)
            save_pickle(dicom_id_to_gt_labels, gt_labels_save_path)
            print('Probabilities saved to ', end=''); print_bold(pred_probs_save_path)
            print('Ground truth labels saved to ', end=''); print_bold(gt_labels_save_path)

    # Release GPU memory
    del evaluator_engine
    del mimiccxr_vision_evaluator
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Return output
    return output

def _find_top_k_label_indices(dicom_id_to_gt_labels, dicom_id_to_pred_probs, thresholds, k, label_names,
                              score_name, score_threshold):
    assert (k is None) != (score_threshold is None), 'Exactly one of k and score_threshold must be specified'
    # Find the top k label classes with the highest scores
    assert score_name in ['f1', 'precision', 'accuracy']
    if score_name == 'f1':
        from sklearn.metrics import f1_score
        score_func = f1_score
    elif score_name == 'precision':
        from sklearn.metrics import precision_score
        score_func = precision_score
    elif score_name == 'accuracy':
        from sklearn.metrics import accuracy_score
        score_func = accuracy_score
    dicom_ids = list(dicom_id_to_gt_labels.keys())
    gt_labels = np.array([dicom_id_to_gt_labels[dicom_id] for dicom_id in dicom_ids])
    pred_probs = np.array([dicom_id_to_pred_probs[dicom_id] for dicom_id in dicom_ids])
    pred_labels = (pred_probs > thresholds).astype(int)

    # # DEBUGGING
    # print_magenta('DEBUGGING--BEGIN', bold=True)
    # import random
    # image2reportId = get_imageId2reportId()
    # rand_i = random.randint(0, len(dicom_ids) - 1)
    # random_dicom_id = dicom_ids[rand_i]
    # print('random_dicom_id =', random_dicom_id)
    # rid = image2reportId[random_dicom_id]
    # metadata = load_mimiccxr_reports_detailed_metadata()
    # filepath = metadata['filepaths'][rid]
    # print('filepath =', filepath)
    # print('Original report:')
    # with open(filepath, 'r') as f:
    #     print(f.read())
    # print('Ground truth labels:')
    # print(gt_labels[rand_i])
    # aux = [str(label_names[i]) for i in range(len(label_names)) if gt_labels[rand_i, i] == 1]
    # print('\n'.join(aux))
    # from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    #     load_scene_graph, visualize_scene_graph
    # )
    # visualize_scene_graph(load_scene_graph(random_dicom_id))
    # print_magenta('DEBUGGING--END', bold=True)
    # print()

    score_list = []
    for i in range(gt_labels.shape[1]):
        score_list.append(score_func(gt_labels[:, i], pred_labels[:, i]))
    score_list = np.array(score_list)
    if score_threshold is not None:
        top_k_label_indices = np.where(score_list >= score_threshold)[0]
        top_k_label_indices = top_k_label_indices[np.argsort(score_list[top_k_label_indices])[::-1]]
    elif k is not None:
        top_k_label_indices = np.argsort(score_list)[::-1][:k]
    else: assert False
    # Print top k label classes with the highest scores
    k = len(top_k_label_indices)
    print_bold(f'Top {k} label classes with the highest {score_name} scores:')
    if k > 20:
        idxs_to_print = set(np.linspace(0, k-1, 20, dtype=int))
    else:
        idxs_to_print = set(range(k))
    for idx, i in enumerate(top_k_label_indices):
        if idx not in idxs_to_print:
            continue
        print_bold(f'{idx+1}. {label_names[i]}: {score_list[i]}')
    return top_k_label_indices

def _evaluate_model(
    tokenizer_kwargs,
    model_kwargs,
    mimiccxr_vision_evaluator_kwargs,
    auxiliary_tasks_kwargs,
    template_based_mode,
    max_processes_for_chexpert_labeler=4,
    checkpoint_folder_path=None,
    use_amp=False,
    eval_mimiccxr=False,
    calibrate_thresholds=False,
    calibration_score_name='f1',
    top_k_chexpert_labels=None,
    top_k_chest_imagenome_labels=None,
    label_score_threshold=None,
    mimiccxr_qa_adapted_reports_filename=None,
    chest_imagenome_label_names_filename=None,
    chest_imagenome_labels_filename=None,
    background_findings_and_impression_per_report_filepath=None,
    fact_embedding_cluster_labels_per_report_filepath=None,
    cache_computations=False,
):
    if eval_mimiccxr:
        if template_based_mode == TemplateBasedModes.CHEST_IMAGENOME_LABELS__ORACLE:
            assert chest_imagenome_label_names_filename is not None
            assert chest_imagenome_labels_filename is not None
            assert background_findings_and_impression_per_report_filepath is not None
            _evaluate_chest_imagenome_template_based_oracle(
                chest_imagenome_label_names_filename=chest_imagenome_label_names_filename,
                chest_imagenome_labels_filename=chest_imagenome_labels_filename,
                background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                template_based_mode=template_based_mode,
            )
        elif template_based_mode == TemplateBasedModes.CHEXPERT_LABELS__ORACLE:
            _evaluate_chexpert_labels_template_based_oracle(
                background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                template_based_mode=template_based_mode,
            )
        elif template_based_mode == TemplateBasedModes.CHEXBERT_LABELS__ORACLE:
            _evaluate_chexbert_labels_template_based_oracle(
                background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                template_based_mode=template_based_mode,
            )
        elif template_based_mode == TemplateBasedModes.FACT_EMBEDDING_LABELS__ORACLE:
            assert background_findings_and_impression_per_report_filepath is not None
            assert fact_embedding_cluster_labels_per_report_filepath is not None
            _evaluate_fact_embedding_template_based_oracle(
                background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
                fact_embedding_cluster_labels_per_report_filepath=fact_embedding_cluster_labels_per_report_filepath,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                template_based_mode=template_based_mode,
            )
        else: # No oracles -> we need generate reports from the predictions of a model

            use_chexpert = template_based_mode in [TemplateBasedModes.CHEXPERT_LABELS,
                                                   TemplateBasedModes.CHEXPERT_AND_CHEST_IMAGENOME_LABELS]
            use_chest_imagenome = template_based_mode in [TemplateBasedModes.CHEST_IMAGENOME_LABELS,
                                                          TemplateBasedModes.CHEXPERT_AND_CHEST_IMAGENOME_LABELS]
            assert use_chexpert or use_chest_imagenome

            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            
            # Obtain model's predictions on MIMIC-CXR test set
            tmp = _compute_probs_and_gt_labels_for_mimiccxr_test_set(
                auxiliary_tasks_kwargs=auxiliary_tasks_kwargs,
                checkpoint_folder_path=checkpoint_folder_path,
                model_kwargs=model_kwargs,
                mimiccxr_vision_evaluator_kwargs=mimiccxr_vision_evaluator_kwargs,
                use_amp=use_amp,
                save_probs_and_gt_labels=cache_computations,
            )
            if use_chexpert:
                dicom_id_to_pred_chexpert_probs = tmp['dicom_id_to_pred_chexpert_probs']
                dicom_id_to_gt_chexpert_labels = tmp['dicom_id_to_gt_chexpert_labels']
            if use_chest_imagenome:
                dicom_id_to_pred_chest_imagenome_probs = tmp['dicom_id_to_pred_chest_imagenome_probs']
                dicom_id_to_gt_chest_imagenome_labels = tmp['dicom_id_to_gt_chest_imagenome_labels']

            # Calibrate thresholds on MIMIC-CXR validation set
            if calibrate_thresholds:
                kwargs = mimiccxr_vision_evaluator_kwargs.copy()
                kwargs['use_test_set'] = False
                kwargs['use_chest_imagenome_label_gold_set'] = False
                kwargs['use_val_set_only'] = True
                kwargs['val_image_transform'] = kwargs['test_image_transform']
                assert kwargs['val_image_transform'] is not None

                def _get_model_and_device():
                    # Define device
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # Load saved checkpoint    
                    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
                    print('Loading model from checkpoint ...')
                    print('checkpoint_path = ', checkpoint_path)
                    checkpoint = torch.load(checkpoint_path)
                    # Load model
                    print('Loading model...')
                    model = MultiPurposeVisualModule(**model_kwargs)
                    model = model.to(device)
                    model.load_state_dict(checkpoint['model'], strict=False)
                    return model, device

                thresholds_dict = calibrate_thresholds_on_mimiccxr_validation_set(
                    model_and_device_getter=_get_model_and_device,
                    use_amp=use_amp,
                    mimiccxr_vision_evaluator_kwargs=kwargs,
                    classify_chexpert=use_chexpert,
                    classify_chest_imagenome=use_chest_imagenome,
                    cache_thresholds=cache_computations,
                    cache_probs=cache_computations,
                    results_folder_path=results_folder_path,
                    score_name=calibration_score_name,
                )
                # Release GPU memory (just in case)
                torch.cuda.empty_cache()

            if calibrate_thresholds and use_chexpert:
                chexpert_thresholds = thresholds_dict['chexpert']
            else:
                chexpert_thresholds = None
            if calibrate_thresholds and use_chest_imagenome:
                chest_imagenome_thresholds = thresholds_dict['chest_imagenome']
            else:
                chest_imagenome_thresholds = None

            # Determine template-based algorithm to generate reports from predictions

            def _get_report_ids_and_pred_probs(dicom_id_to_probs):
                imageId2reportId = get_imageId2reportId()
                seen_report_ids = set()
                report_ids = []
                pred_probs = []
                for dicom_id, probs in dicom_id_to_probs.items():
                    report_id = imageId2reportId[dicom_id]
                    if report_id in seen_report_ids:
                        continue
                    seen_report_ids.add(report_id)
                    report_ids.append(report_id)
                    pred_probs.append(probs)
                return report_ids, np.array(pred_probs)
            
            if use_chexpert:
                print_blue('Getting label names and templates from CheXpert ...', bold=True)
                label_names = CHEXPERT_LABELS
                label_order = _BEST_CHEXPERT_ORDER
                label_templates = TEMPLATES_CHEXPERT_v2
                label_thresholds = chexpert_thresholds
                if label_thresholds is None:
                    label_thresholds = np.array([0.5] * len(label_names)) # default thresholds
                chexpert_report_ids, pred_probs = _get_report_ids_and_pred_probs(dicom_id_to_pred_chexpert_probs)
                if top_k_chexpert_labels is not None or label_score_threshold is not None:
                    top_k_label_indices = _find_top_k_label_indices(dicom_id_to_gt_chexpert_labels,
                                                                    dicom_id_to_pred_chexpert_probs,
                                                                    chexpert_thresholds, top_k_chexpert_labels,
                                                                    CHEXPERT_LABELS, calibration_score_name,
                                                                    label_score_threshold)
                    tot_chexpert_labels = len(CHEXPERT_LABELS)
                    top_k_chexpert_labels = len(top_k_label_indices)
                else:
                    top_k_label_indices = None
                    tot_chexpert_labels = None
                chexpert_reports = recover_reports__template_based(
                    report_ids=chexpert_report_ids,
                    pred_probs=pred_probs,
                    qa_adapted_dataset=get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR,
                                                                        mimiccxr_qa_adapted_reports_filename)),
                    label_names=label_names,
                    label_templates=label_templates,
                    label_thresholds=label_thresholds,
                    label_order=label_order,
                    top_k_label_indices=top_k_label_indices,
                )
            else:
                tot_chexpert_labels = None

            if use_chest_imagenome:
                print_blue('Getting label names and templates from Chest-Imagenome ...', bold=True)
                label_names_filename = mimiccxr_vision_evaluator_kwargs['chest_imagenome_label_names_filename']
                label_names, label_templates = load_chest_imagenome_label_names_and_templates(
                    label_names_filename, apply_anatomy_reordering=True)
                label_order = label_names # TODO: come up with a better order
                label_thresholds = chest_imagenome_thresholds
                if label_thresholds is None:
                    label_thresholds = np.array([0.5] * len(label_names)) # default thresholds
                chest_imagenome_report_ids, pred_probs = _get_report_ids_and_pred_probs(
                    dicom_id_to_pred_chest_imagenome_probs)
                if top_k_chest_imagenome_labels is not None or label_score_threshold is not None:
                    top_k_label_indices = _find_top_k_label_indices(dicom_id_to_gt_chest_imagenome_labels,
                                                                    dicom_id_to_pred_chest_imagenome_probs,
                                                                    chest_imagenome_thresholds, top_k_chest_imagenome_labels,
                                                                    label_names, calibration_score_name,
                                                                    label_score_threshold)
                    tot_chest_imagenome_labels = len(label_names)
                    top_k_chest_imagenome_labels = len(top_k_label_indices)
                else:
                    top_k_label_indices = None
                    tot_chest_imagenome_labels = None
                chest_imagenome_reports = recover_reports__template_based(
                    report_ids=chest_imagenome_report_ids,
                    pred_probs=pred_probs,
                    qa_adapted_dataset=get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR,
                                                                        mimiccxr_qa_adapted_reports_filename)),
                    label_names=label_names,
                    label_templates=label_templates,
                    label_thresholds=label_thresholds,
                    label_order=label_order,
                    top_k_label_indices=top_k_label_indices,
                )
            else:
                tot_chest_imagenome_labels = None

            if template_based_mode == TemplateBasedModes.CHEXPERT_LABELS:
                reports = chexpert_reports
            elif template_based_mode == TemplateBasedModes.CHEST_IMAGENOME_LABELS:
                reports = chest_imagenome_reports
            elif template_based_mode == TemplateBasedModes.CHEXPERT_AND_CHEST_IMAGENOME_LABELS:
                # Sanity check
                assert len(chexpert_report_ids) == len(chest_imagenome_report_ids)
                assert set(chexpert_report_ids) == set(chest_imagenome_report_ids)
                # Merge reports
                print_blue('Merging reports from CheXpert and Chest-Imagenome ...', bold=True)
                chexpert_rid2idx = {rid: idx for idx, rid in enumerate(chexpert_report_ids)}
                chest_imagenome_rid2idx = {rid: idx for idx, rid in enumerate(chest_imagenome_report_ids)}
                reports = {'gt_reports': [], 'gen_reports': []}
                for rid in chexpert_report_ids:
                    chx_i = chexpert_rid2idx[rid]
                    chimg_i = chest_imagenome_rid2idx[rid]
                    assert chexpert_reports['gt_reports'][chx_i]['rid'] == chest_imagenome_reports['gt_reports'][chimg_i]['rid']
                    assert chexpert_reports['gt_reports'][chx_i]['text'] == chest_imagenome_reports['gt_reports'][chimg_i]['text']
                    gt_report = chexpert_reports['gt_reports'][chx_i]
                    chx_rep = chexpert_reports['gen_reports'][chx_i]
                    chimg_rep = chest_imagenome_reports['gen_reports'][chimg_i]
                    merged_report = {
                        'q': chx_rep['q'] + chimg_rep['q'],
                        'a': chx_rep['a'] + chimg_rep['a'],
                    }
                    reports['gt_reports'].append(gt_report)
                    reports['gen_reports'].append(merged_report)
            else:
                raise ValueError(f'Unknown template_based_mode: {template_based_mode}')
            
            tokenizer_kwargs = {'qa_adapted_dataset_paths': [os.path.join(MIMICCXR_CACHE_DIR, mimiccxr_qa_adapted_reports_filename)] }
            tokenizer = Tokenizer(**tokenizer_kwargs)
            _compute_and_save_report_level_metrics(
                reports['gt_reports'], reports['gen_reports'], 'mimiccxr', tokenizer, results_folder_path,
                max_processes_for_chexpert_labeler, template_based_mode, calibrate_thresholds, calibration_score_name,
                top_k_chexpert_labels, tot_chexpert_labels,
                top_k_chest_imagenome_labels, tot_chest_imagenome_labels,
                label_score_threshold)

def evaluate_model(
    checkpoint_folder,
    template_based_mode,
    batch_size=100,
    num_workers=0,
    max_processes_for_chexpert_labeler=4,
    use_amp=False,
    calibrate_thresholds=False,
    calibration_score_name='f1',
    mimiccxr_qa_adapted_reports_filename=None,
    chest_imagenome_label_names_filename=None,
    chest_imagenome_labels_filename=None,
    top_k_chexpert_labels=None,
    top_k_chest_imagenome_labels=None,
    label_score_threshold=None,
    background_findings_and_impression_per_report_filepath=None,
    fact_embedding_cluster_labels_per_report_filepath=None,
    cache_computations=False,
):
    print()
    print_blue('----- Evaluating model ------', bold=True)

    if checkpoint_folder is not None:
        checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
        metadata = load_metadata(checkpoint_folder)
        tokenizer_kwargs = _recover_tokenizer_kwargs(metadata)
        model_kwargs = _recover_model_kwargs(metadata)
        mimiccxr_vision_evaluator_kwargs = _recover_mimiccxr_vision_evaluator_kwargs(
            metadata, batch_size, num_workers, mimiccxr_qa_adapted_reports_filename)
        auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']
    else:
        tokenizer_kwargs = None
        model_kwargs = None
        mimiccxr_vision_evaluator_kwargs = None
        auxiliary_tasks_kwargs = None

    _evaluate_model(
        tokenizer_kwargs,
        model_kwargs,
        mimiccxr_vision_evaluator_kwargs,
        auxiliary_tasks_kwargs,
        template_based_mode,
        max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
        checkpoint_folder_path=checkpoint_folder,
        use_amp=use_amp,
        eval_mimiccxr=True, # TODO: eventually support other datasets
        calibrate_thresholds=calibrate_thresholds,
        calibration_score_name=calibration_score_name,
        top_k_chexpert_labels=top_k_chexpert_labels,
        top_k_chest_imagenome_labels=top_k_chest_imagenome_labels,
        label_score_threshold=label_score_threshold,
        mimiccxr_qa_adapted_reports_filename=mimiccxr_qa_adapted_reports_filename,
        chest_imagenome_label_names_filename=chest_imagenome_label_names_filename,
        chest_imagenome_labels_filename=chest_imagenome_labels_filename,
        background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
        fact_embedding_cluster_labels_per_report_filepath=fact_embedding_cluster_labels_per_report_filepath,
        cache_computations=cache_computations,
    )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)