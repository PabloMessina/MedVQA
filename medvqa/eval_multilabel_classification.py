import  os
import glob
import argparse
from copy import deepcopy
import numpy as np
from tqdm import tqdm

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_label_names,
    load_chest_imagenome_labels,
)
from medvqa.datasets.dataloading_utils import get_vision_collate_batch_fn
from medvqa.datasets.image_processing import get_image_transform
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBig_VisualModuleTrainer
from medvqa.evaluation.bootstrapping import stratified_multilabel_bootstrap_metrics
from medvqa.evaluation.visual_module import (
    calibrate_thresholds_on_mimiccxr_validation_set,
    calibrate_weights_and_thresholds_for_ensemble,
    merge_probabilities,
)
from medvqa.metrics.classification.auc import auc_fn
from medvqa.metrics.classification.multilabel_accuracy import MultiLabelAccuracy
from medvqa.metrics.classification.multilabel_prf1 import MultiLabelPRF1
from medvqa.metrics.classification.roc_auc import roc_auc_fn
from medvqa.metrics.classification.prc_auc import prc_auc_fn, prc_auc_score
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule
from medvqa.models.vqa.open_ended_vqa import OpenEndedVQA

from medvqa.utils.constants import (
    DATASET_NAMES,
    MIMICCXR_DATASET_ID,
    MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE,
    VINBIG_LABELS,
    MetricNames,
)
from medvqa.metrics import (
    attach_chexpert_labels_prf1,
    attach_chexpert_labels_roc_auc,
    attach_condition_aware_class_averaged_prc_auc,
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
    get_model_name_from_checkpoint_path,
    load_metadata,
    load_model_state_dict,
)
from medvqa.utils.common import (
    RESULTS_DIR,
    parsed_args_to_dict,
)    
from medvqa.utils.handlers_utils import (
    get_log_metrics_handler,
    get_log_iteration_handler,
    attach_accumulator,
)
from medvqa.utils.files_utils import (
    get_checkpoint_folder_path,
    get_results_folder_path,
    load_pickle,
    save_pickle,
)
from medvqa.training.vision import get_engine
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleTrainer
from medvqa.utils.logging_utils import CountPrinter, print_blue, print_magenta, print_orange

class EvalDatasets:
    MIMICCXR_TEST_SET = 'mimiccxr_test_set'
    CHEST_IMAGENOME_GOLD = 'chest_imagenome_gold'
    VINBIG_TEST_SET = 'vinbig_test_set'
    @staticmethod
    def get_choices():
        return [
            EvalDatasets.MIMICCXR_TEST_SET,
            EvalDatasets.CHEST_IMAGENOME_GOLD,
            EvalDatasets.VINBIG_TEST_SET,
        ]

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--eval_dataset_name', type=str, required=True, choices=EvalDatasets.get_choices())

    # optional arguments
    parser.add_argument('--checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=140)
    parser.add_argument('--device', type=str, default='GPU')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--calibrate_thresholds', action='store_true')
    parser.add_argument('--save_probs', action='store_true')
    parser.add_argument('--use_ensemble', action='store_true')
    parser.add_argument('--model_names_for_ensemble', type=str, nargs='+', default=None)
    parser.add_argument('--label_name_for_ensemble', type=str, default=None)
    parser.add_argument('--chest_imagenome_label_names_filename', type=str, default=None)
    parser.add_argument('--chest_imagenome_labels_filename', type=str, default=None)
    parser.add_argument('--cheat', action='store_true')
    
    return parser.parse_args()

def _compute_and_save_metrics(
    metrics_dict, metrics_to_save, dataset_name, results_folder_path,
    chexpert_thresholds=None, chest_imagenome_thresholds=None,
    chest_imagenome_label_names=None,
):
    metrics = {k:metrics_dict[k] for k in metrics_to_save}
    strings = []    

    if chexpert_thresholds is not None:
        strings.append('chexp-calib-thresh')
        pred_chexpert_probs = metrics_dict['pred_chexpert_probs']
        pred_chexpert_probs = torch.stack(pred_chexpert_probs).numpy()
        pred_chexpert_labels = (pred_chexpert_probs > chexpert_thresholds).astype(int)
        gt_chexpert_labels = metrics_dict['chexpert']
        gt_chexpert_labels = torch.stack(gt_chexpert_labels).numpy()
        # compute chexpert accuracy
        met = MultiLabelAccuracy(device='cpu')
        met.update((pred_chexpert_labels, gt_chexpert_labels))
        metrics[MetricNames.CHXLABELACC] = met.compute()
        # compute chexpert prf1
        met = MultiLabelPRF1(device='cpu')
        met.update((pred_chexpert_labels, gt_chexpert_labels))
        metrics[MetricNames.CHXLABEL_PRF1] = met.compute()
        # compute chexpert roc auc
        metrics[MetricNames.CHXLABEL_ROCAUC] = roc_auc_fn(
            pred_chexpert_probs, gt_chexpert_labels)
        # compute chexpert auc
        metrics[MetricNames.CHXLABEL_AUC] = auc_fn(
            pred_chexpert_probs, gt_chexpert_labels)
        # compute chexpert prc auc
        metrics[MetricNames.CHXLABEL_PRCAUC] = prc_auc_fn(
            gt_chexpert_labels, pred_chexpert_probs)

    if chest_imagenome_thresholds is not None:
        assert chest_imagenome_label_names is not None
        strings.append('chest-imagenome-calib-thresh')
        pred_chest_imagenome_probs = metrics_dict['pred_chest_imagenome_probs']
        pred_chest_imagenome_probs = torch.stack(pred_chest_imagenome_probs).numpy()
        pred_chest_imagenome_labels = (pred_chest_imagenome_probs > chest_imagenome_thresholds).astype(int)
        gt_chest_imagenome_labels = metrics_dict['chest_imagenome']
        gt_chest_imagenome_labels = torch.stack(gt_chest_imagenome_labels).numpy()
        # compute chest imagenome accuracy
        met = MultiLabelAccuracy(device='cpu')
        met.update((pred_chest_imagenome_labels, gt_chest_imagenome_labels))
        metrics[MetricNames.CHESTIMAGENOMELABELACC] = met.compute()
        # compute chest imagenome prf1
        met = MultiLabelPRF1(device='cpu')
        met.update((pred_chest_imagenome_labels, gt_chest_imagenome_labels))
        metrics[MetricNames.CHESTIMAGENOMELABEL_PRF1] = met.compute()
        # compute chest imagenome roc auc
        metrics[MetricNames.CHESTIMAGENOMELABELROCAUC] = roc_auc_fn(
            pred_chest_imagenome_probs, gt_chest_imagenome_labels)
        # compute chest imagenome auc
        metrics[MetricNames.CHESTIMAGENOMELABELAUC] = auc_fn(
            pred_chest_imagenome_probs, gt_chest_imagenome_labels)
        # compute chest imagenome prc auc
        metrics[MetricNames.CHESTIMAGENOMELABELPRCAUC] = prc_auc_fn(
            gt_chest_imagenome_labels, pred_chest_imagenome_probs)
        # save label names
        metrics['chest_imagenome_label_names'] = chest_imagenome_label_names
    
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_multilabel_classification_metrics{"(" + "-".join(strings) + ")" if strings else ""}.pkl')
    save_pickle(metrics, save_path)
    print (f'Multilabel classification metrics saved to {save_path}')
    return metrics

def _compute_and_save_metrics__ensemble__chest_imagenome(
    dataset_name, merged_probs, pred_labels, gt_labels, label_names, ensemble_model_names, results_folder_path,
    ensemble_weights, ensemble_thresholds, cheating=False,
):
    metrics = {}
    # compute chest imagenome accuracy
    met = MultiLabelAccuracy(device='cpu')
    assert len(pred_labels) == len(gt_labels)
    assert len(pred_labels) > 0
    met.update((pred_labels, gt_labels))
    metrics[MetricNames.CHESTIMAGENOMELABELACC] = met.compute()
    # compute chest imagenome prf1
    met = MultiLabelPRF1(device='cpu')
    met.update((pred_labels, gt_labels))
    metrics[MetricNames.CHESTIMAGENOMELABEL_PRF1] = met.compute()
    # compute chest imagenome roc auc
    metrics[MetricNames.CHESTIMAGENOMELABELROCAUC] = roc_auc_fn(merged_probs, gt_labels)
    # compute chest imagenome auc
    metrics[MetricNames.CHESTIMAGENOMELABELAUC] = auc_fn(merged_probs, gt_labels)
    # compute precision-recall auc
    metrics[MetricNames.CHESTIMAGENOMELABELPRCAUC] = prc_auc_fn(gt_labels, merged_probs)
    # save label names
    metrics['chest_imagenome_label_names'] = label_names
    # save ensemble related arguments
    metrics['ensemble_model_names'] = ensemble_model_names
    metrics['ensemble_weights'] = ensemble_weights
    metrics['ensemble_thresholds'] = ensemble_thresholds
    
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_multilabel_classification_metrics(ensemble-chest-imagenome{",cheating" if cheating else ""}).pkl')
    save_pickle(metrics, save_path)
    print (f'Multilabel classification metrics saved to {save_path}')
    return metrics


def _recover_model_kwargs(metadata):
    kwargs = metadata['model_kwargs']
    kwargs.update(metadata['auxiliary_tasks_kwargs'])
    return kwargs

def _recover_vision_dataset_manager_kwargs(dataset_name, metadata, batch_size, num_workers, eval_dataset_name):
    keys = [
        f'{dataset_name}_trainer_kwargs',
        f'{dataset_name}_vision_trainer_kwargs',
        f'{dataset_name}_vqa_trainer_kwargs',
    ]
    kwargs = None
    for key in keys:
        if key in metadata:
            kwargs = metadata[key]
            break    
    assert kwargs is not None

    kwargs['val_batch_size'] = batch_size
    kwargs['num_workers'] = num_workers
    if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
        kwargs['use_test_set'] = True
    elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
        kwargs['use_chest_imagenome_label_gold_set'] = True
    elif eval_dataset_name == EvalDatasets.VINBIG_TEST_SET:
        kwargs['use_training_set'] = False
        kwargs['use_validation_set'] = True
    else: assert False
    kwargs['data_augmentation_enabled'] = False
    
    # Define test image transform
    image_transform_kwargs = metadata['val_image_transform_kwargs']
    if eval_dataset_name in [EvalDatasets.MIMICCXR_TEST_SET, EvalDatasets.CHEST_IMAGENOME_GOLD]:
        if DATASET_NAMES.MIMICCXR in image_transform_kwargs:
            image_transform = get_image_transform(**image_transform_kwargs[DATASET_NAMES.MIMICCXR])
        else: # for backward compatibility
            image_transform = get_image_transform(**image_transform_kwargs)
    elif eval_dataset_name == EvalDatasets.VINBIG_TEST_SET:
        image_transform = get_image_transform(**image_transform_kwargs[DATASET_NAMES.VINBIG])
    else: assert False
    if dataset_name == 'vinbig':
        kwargs['val_image_transform'] = image_transform
    else:
        kwargs['test_image_transform'] = image_transform

    # Define collate_batch_fn
    if eval_dataset_name in [EvalDatasets.MIMICCXR_TEST_SET, EvalDatasets.CHEST_IMAGENOME_GOLD]:
        use_detectron2 = kwargs.get('use_detectron2', False)
        try:
            collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
        except KeyError:
            _kwargs = dict(
                include_image=True,
                classify_tags=metadata['auxiliary_tasks_kwargs']['classify_tags'],
                n_tags=metadata['auxiliary_tasks_kwargs']['n_medical_tags'],
                classify_orientation=metadata['auxiliary_tasks_kwargs']['classify_orientation'],
                classify_gender=metadata['auxiliary_tasks_kwargs'].get('classify_gender', False),
                classify_chexpert=metadata['auxiliary_tasks_kwargs']['classify_chexpert'],
                classify_questions=metadata['auxiliary_tasks_kwargs']['classify_questions'],
                classify_chest_imagenome=metadata['auxiliary_tasks_kwargs']['classify_chest_imagenome'],
                predict_bboxes_chest_imagenome=metadata['auxiliary_tasks_kwargs']['predict_bboxes_chest_imagenome'],
            )
            if use_detectron2:
                collate_batch_fn_kwargs = {
                    DATASET_NAMES.MIMICCXR_CHEST_IMAGENOME__DETECTRON2_MODE: {
                        'dataset_id': MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE, **_kwargs },
                }
            else:
                collate_batch_fn_kwargs = {
                    DATASET_NAMES.MIMICCXR: { 'dataset_id': MIMICCXR_DATASET_ID, **_kwargs },
                }
        if use_detectron2: # special case for detectron2
            collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR_CHEST_IMAGENOME__DETECTRON2_MODE])
        else:
            collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR])
    elif eval_dataset_name == EvalDatasets.VINBIG_TEST_SET:
        collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
        collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs['vinbig'])
    else: assert False
    kwargs['collate_batch_fn'] = collate_batch_fn

    return kwargs

def _recover_mimiccxr_vision_evaluator_kwargs(metadata, batch_size, num_workers, eval_dataset_name):
    return _recover_vision_dataset_manager_kwargs('mimiccxr', metadata, batch_size, num_workers, eval_dataset_name)

def _recover_vinbig_vision_evaluator_kwargs(metadata, batch_size, num_workers, eval_dataset_name):
    return _recover_vision_dataset_manager_kwargs('vinbig', metadata, batch_size, num_workers, eval_dataset_name)

def _calibrate_thresholds_using_chexpert_for_mimiccxr(model, device, use_amp, mimiccxr_vision_evaluator_kwargs,
                                                        trainer_engine_kwargs, save_probs=False, results_folder_path=None):
    print_blue('Calibrating thresholds using MIMICCXR validation dataset and CheXpert labels')
    return calibrate_thresholds_on_mimiccxr_validation_set(
        model_and_device_getter=lambda : (model, device),
        use_amp=use_amp,
        mimiccxr_vision_evaluator_kwargs=mimiccxr_vision_evaluator_kwargs,
        trainer_engine_kwargs=trainer_engine_kwargs,
        classify_chexpert=True, classify_chest_imagenome=False,
        cache_probs=save_probs, results_folder_path=results_folder_path)['chexpert']

def _calibrate_thresholds_using_chest_imagenome_for_mimiccxr(model, device, use_amp, mimiccxr_vision_evaluator_kwargs,
                                                            trainer_engine_kwargs, save_probs=False, results_folder_path=None):
    print_blue('Calibrating thresholds using MIMICCXR validation dataset and Chest-ImaGenome labels')
    return calibrate_thresholds_on_mimiccxr_validation_set(
        model_and_device_getter=lambda : (model, device),
        use_amp=use_amp,
        mimiccxr_vision_evaluator_kwargs=mimiccxr_vision_evaluator_kwargs,
        trainer_engine_kwargs=trainer_engine_kwargs,
        classify_chexpert=False, classify_chest_imagenome=True,
        cache_probs=save_probs, results_folder_path=results_folder_path)['chest_imagenome']
        
def _evaluate_model(
    eval_dataset_name,
    model_kwargs,
    trainer_engine_kwargs,
    mimiccxr_vision_evaluator_kwargs,
    vinbig_vision_evaluator_kwargs,
    auxiliary_tasks_kwargs,
    device='GPU',
    checkpoint_folder_path=None,
    use_amp=False,
    calibrate_thresholds=False,
    save_probs=False,
    use_ensemble=False,
    model_names_for_ensemble=None,
    label_name_for_ensemble=None,
    chest_imagenome_label_names_filename=None,
    chest_imagenome_labels_filename=None,
    return_results=False,
    cheat=False,
):    
    eval_mimiccxr_test_set = eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET
    eval_chest_imagenome_gold = eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD
    eval_vinbig_test_set = eval_dataset_name == EvalDatasets.VINBIG_TEST_SET
    assert sum([eval_mimiccxr_test_set, eval_chest_imagenome_gold, eval_vinbig_test_set]) == 1

    if use_ensemble: # do things differently if using ensemble
        assert model_names_for_ensemble is not None
        assert label_name_for_ensemble is not None
        assert len(model_names_for_ensemble) > 0
        assert calibrate_thresholds # always calibrate thresholds when using ensemble
        assert eval_mimiccxr_test_set # not implemented for other datasets

        if label_name_for_ensemble == 'chest_imagenome':
            assert chest_imagenome_label_names_filename is not None
            assert chest_imagenome_labels_filename is not None
            if not cheat:
                # Collect validation set probabilities
                print('Collecting validation set probabilities')
                val_probs_list = []
                for model_name in tqdm(model_names_for_ensemble):
                    val_probs_path = os.path.join(RESULTS_DIR, model_name, f'dicom_id_to_pred_probs__mimiccxr_val__{label_name_for_ensemble}.pkl')
                    assert os.path.exists(val_probs_path), f'Could not find {val_probs_path}'
                    val_probs_list.append(load_pickle(val_probs_path))
            # Collect test set probabilities
            print('Collecting test set probabilities')
            test_probs_list = []
            for model_name in tqdm(model_names_for_ensemble):
                test_probs_path = os.path.join(RESULTS_DIR, model_name, f'dicom_id_to_pred_{label_name_for_ensemble}_probs__mimiccxr_test_set.pkl')
                assert os.path.exists(test_probs_path), f'Could not find {test_probs_path}'
                test_probs_list.append(load_pickle(test_probs_path))
            # Collect label names used by each model (we will ensemble only the labels that are common to all models)
            print('Collecting label names used by each model')
            label_names_list = []
            for model_name in tqdm(model_names_for_ensemble):
                results_folder_path = os.path.join(RESULTS_DIR, model_name)
                assert os.path.exists(results_folder_path), f'Could not find {results_folder_path}'
                # find all files containing "_multilabel_classification_metrics"
                metrics_paths = glob.glob(os.path.join(results_folder_path, '*_multilabel_classification_metrics*.pkl'))
                assert len(metrics_paths) == 1, f'Found {len(metrics_paths)} metrics files in {results_folder_path}'
                metrics_path = metrics_paths[0]
                metrics = load_pickle(metrics_path)
                label_names_list.append(metrics['chest_imagenome_label_names'])
            # Collect ground truth labels
            print('Collecting ground truth labels')
            dicom_id_2_gt_labels = load_chest_imagenome_labels(chest_imagenome_labels_filename)
            gt_label_names = load_chest_imagenome_label_names(chest_imagenome_label_names_filename)
            if not cheat:
                # Calibrate thresholds on validation set
                print('Calibrating thresholds on validation set')
                weights, thresholds = calibrate_weights_and_thresholds_for_ensemble(
                    dicom_id_2_probs_list=val_probs_list,
                    label_names_list=label_names_list,
                    dicom_id_2_gt_labels=dicom_id_2_gt_labels,
                    gt_label_names=gt_label_names,
                )
            else:
                # Calibrate thresholds on test set (cheating)
                print('Calibrating thresholds on test set (we are cheating)')
                weights, thresholds = calibrate_weights_and_thresholds_for_ensemble(
                    dicom_id_2_probs_list=test_probs_list,
                    label_names_list=label_names_list,
                    dicom_id_2_gt_labels=dicom_id_2_gt_labels,
                    gt_label_names=gt_label_names,
                )

            # Ensemble test set probabilities
            print('Ensembling test set probabilities')
            test_merged_probs, test_pred_labels, test_gt_labels, test_label_names = merge_probabilities(
                dicom_id_2_probs_list=test_probs_list,
                label_names_list=label_names_list,
                dicom_id_2_gt_labels=dicom_id_2_gt_labels,
                gt_label_names=gt_label_names,
                weights=weights,
                thresholds=thresholds,
            )
            # Compute and save metrics
            ensemble_results_folder_path = get_results_folder_path(get_checkpoint_folder_path('visual_module', 'chest-imagenome', 'ensemble'))
            metrics = _compute_and_save_metrics__ensemble__chest_imagenome(
                dataset_name='mimiccxr_test_set',
                merged_probs=test_merged_probs,
                pred_labels=test_pred_labels,
                gt_labels=test_gt_labels,
                label_names=test_label_names,
                ensemble_model_names=model_names_for_ensemble,
                results_folder_path=ensemble_results_folder_path,
                ensemble_weights=weights,
                ensemble_thresholds=thresholds,
                cheating=cheat,
            )
            # Save ensemble probabilities
            if save_probs:
                save_pickle(test_merged_probs, os.path.join(ensemble_results_folder_path,
                                f'dicom_id_to_pred_{label_name_for_ensemble}_probs__mimiccxr_test_set.pkl'))
            # Return results
            if return_results:
                return metrics
            else:
                return
        elif label_name_for_ensemble == 'chexpert':
            raise NotImplementedError
        else:
            raise ValueError(f'Unknown label name for ensemble: {label_name_for_ensemble}')

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
    # auxiliary task: vinbig bounding boxes
    predict_bboxes_vinbig = auxiliary_tasks_kwargs['predict_bboxes_vinbig']
    # auxiliary task: vinbig labels
    classify_labels_vinbig = auxiliary_tasks_kwargs['classify_labels_vinbig']
    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)
    
    count_print = CountPrinter()
        
    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Load saved checkpoint    
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    # Create model
    count_print('Creating model ...')
    model_name = get_model_name_from_checkpoint_path(checkpoint_path)
    print('model_name =', model_name)
    if model_name == 'vqa':
        model = OpenEndedVQA(**model_kwargs, device=device, use_visual_module_only=True,
                            vocab_size=None, start_idx=None)
    elif model_name == 'visual_module':
        model = MultiPurposeVisualModule(**model_kwargs)
    else:
        raise ValueError(f'Invalid model_name: {model_name}')
    model = model.to(device)
    load_model_state_dict(model, checkpoint['model'])
    
    # Create MIMIC-CXR visual module evaluator
    if eval_mimiccxr_test_set or eval_chest_imagenome_gold:
        count_print('Creating MIMIC-CXR visual module evaluator ...')
        if eval_mimiccxr_test_set:
            assert mimiccxr_vision_evaluator_kwargs['use_test_set']
        if eval_chest_imagenome_gold:
            assert mimiccxr_vision_evaluator_kwargs['use_chest_imagenome_label_gold_set']
        mimiccxr_vision_evaluator = MIMICCXR_VisualModuleTrainer(**mimiccxr_vision_evaluator_kwargs)

    # Create VINBIG visual module evaluator
    elif eval_vinbig_test_set:
        count_print('Creating VINBIG visual module evaluator ...')
        vinbig_vision_evaluator = VinBig_VisualModuleTrainer(**vinbig_vision_evaluator_kwargs)

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    _engine_kwargs = dict(
        model=model, classify_tags=classify_tags, classify_orientation=classify_orientation,
        classify_gender=classify_gender, classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        classify_chest_imagenome=classify_chest_imagenome,
        classify_labels_vinbig=classify_labels_vinbig,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
        device=device, use_amp=use_amp, training=False,
        yolov8_use_multiple_detection_layers=trainer_engine_kwargs.get('yolov8_use_multiple_detection_layers', False),
    )
    if mimiccxr_vision_evaluator_kwargs is not None:
        _engine_kwargs['pass_pred_bbox_coords_as_input'] = mimiccxr_vision_evaluator_kwargs.get('pass_pred_bbox_coords_to_model', False)
        _engine_kwargs['use_yolov8'] = mimiccxr_vision_evaluator_kwargs.get('use_yolov8', False)
    else:
        _engine_kwargs['pass_pred_bbox_coords_as_input'] = False
        _engine_kwargs['use_yolov8'] = False
    if eval_chest_imagenome_gold:
        _engine_kwargs['valid_chest_imagenome_label_indices'] = mimiccxr_vision_evaluator.valid_chest_imagenome_label_indices
    evaluator_engine = get_engine(**_engine_kwargs)

    # Attach metrics, timer and events to engines
    count_print('Attaching metrics, timer and events to engines ...')

    # Metrics
    if classify_tags:
        attach_medical_tags_f1score(evaluator_engine, device)

    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(evaluator_engine)

    if classify_chexpert:
        attach_chexpert_labels_accuracy(evaluator_engine, device)        
        attach_chexpert_labels_prf1(evaluator_engine, device)
        attach_chexpert_labels_roc_auc(evaluator_engine, 'cpu')

    if classify_questions:
        attach_question_labels_prf1(evaluator_engine, device)

    if classify_chest_imagenome:
        attach_chest_imagenome_labels_accuracy(evaluator_engine, device)
        attach_chest_imagenome_labels_prf1(evaluator_engine, device)
        attach_chest_imagenome_labels_roc_auc(evaluator_engine, 'cpu')

    if classify_labels_vinbig:
        _cond_func = lambda x: x['flag'] == 'vinbig'
        attach_condition_aware_class_averaged_prc_auc(evaluator_engine, 'pred_vinbig_probs', 'vinbig_labels', None, 'vnb_macro_prcauc', _cond_func)

    # Accumulators
    if classify_chexpert:
        attach_accumulator(evaluator_engine, 'chexpert')
        attach_accumulator(evaluator_engine, 'pred_chexpert_probs')
    if classify_chest_imagenome:
        attach_accumulator(evaluator_engine, 'chest_imagenome')
        attach_accumulator(evaluator_engine, 'pred_chest_imagenome_probs')
    if classify_labels_vinbig:
        attach_accumulator(evaluator_engine, 'vinbig_labels')
        attach_accumulator(evaluator_engine, 'pred_vinbig_probs')
    
    # Timer
    timer = Timer()
    timer.attach(evaluator_engine, start=Events.EPOCH_STARTED)
    
    # Logging
    metrics_to_print=[]
    if classify_tags:
        metrics_to_print.append(MetricNames.MEDTAGF1)
    if classify_orientation:
        metrics_to_print.append(MetricNames.ORIENACC)
    if classify_chexpert:
        metrics_to_print.append(MetricNames.CHXLABEL_PRF1)
        metrics_to_print.append(MetricNames.CHXLABELACC)
        metrics_to_print.append(MetricNames.CHXLABEL_ROCAUC)
    if classify_questions:
        metrics_to_print.append(MetricNames.QLABELS_PRF1)
    if classify_chest_imagenome:
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABEL_PRF1)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELACC)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELROCAUC)
    if classify_labels_vinbig:
        metrics_to_print.append('vnb_macro_prcauc')

    log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()    
    
    # Attach handlers    
    evaluator_engine.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation
    results_dict = {}

    if eval_mimiccxr_test_set or eval_chest_imagenome_gold:
        count_print(f'Running evaluator engine on {eval_dataset_name} ...')
        print('len(mimiccxr_vision_evaluator.test_dataset) =', len(mimiccxr_vision_evaluator.test_dataset))
        print('len(mimiccxr_vision_evaluator.test_dataloader) =', len(mimiccxr_vision_evaluator.test_dataloader))
        evaluator_engine.run(mimiccxr_vision_evaluator.test_dataloader)
        results_dict['mimiccxr_metrics'] = deepcopy(evaluator_engine.state.metrics)
        results_dict['mimiccxr_dataset'] = mimiccxr_vision_evaluator.test_dataset        
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        
        # Save probabilities
        if save_probs:
            print(f'Saving probabilities on {eval_dataset_name} ...')
            _tuples = []
            if classify_chest_imagenome:
                _tuples.append(('pred_chest_imagenome_probs', f'dicom_id_to_pred_chest_imagenome_probs__{eval_dataset_name}.pkl'))
            if classify_chexpert:
                _tuples.append(('pred_chexpert_probs', f'dicom_id_to_pred_chexpert_probs__{eval_dataset_name}.pkl'))
            for pred_probs_key, save_name in _tuples:
                dicom_id_to_pred_probs = {}            
                pred_probs = evaluator_engine.state.metrics[pred_probs_key]
                assert len(mimiccxr_vision_evaluator.test_indices) == len(pred_probs)
                for i, idx in enumerate(mimiccxr_vision_evaluator.test_indices):
                    dicom_id = mimiccxr_vision_evaluator.dicom_ids[idx]
                    dicom_id_to_pred_probs[dicom_id] = pred_probs[i].detach().cpu().numpy()
                save_path = os.path.join(results_folder_path, save_name)
                save_pickle(dicom_id_to_pred_probs, save_path)
                print('Probabilities saved to:', save_path)

        if calibrate_thresholds:
            kwargs = mimiccxr_vision_evaluator_kwargs.copy()
            kwargs['use_test_set'] = False
            kwargs['use_chest_imagenome_label_gold_set'] = False
            kwargs['use_val_set_only'] = True
            kwargs['val_image_transform'] = kwargs['test_image_transform']
            assert kwargs['val_image_transform'] is not None
        
        # Calibrate thresholds
        if classify_chexpert and calibrate_thresholds:
            chexpert_thresholds = _calibrate_thresholds_using_chexpert_for_mimiccxr(
                model, device, use_amp, kwargs, trainer_engine_kwargs, save_probs, results_folder_path)
        else:
            chexpert_thresholds = None
        if classify_chest_imagenome and calibrate_thresholds:
            chest_imagenome_thresholds = _calibrate_thresholds_using_chest_imagenome_for_mimiccxr(
                model, device, use_amp, kwargs, trainer_engine_kwargs, save_probs, results_folder_path)
            if eval_chest_imagenome_gold:
                print(f'chest_imagenome_thresholds.shape (before) = {chest_imagenome_thresholds.shape}')
                chest_imagenome_thresholds = chest_imagenome_thresholds[mimiccxr_vision_evaluator.valid_chest_imagenome_label_indices]
                print(f'chest_imagenome_thresholds.shape (after) = {chest_imagenome_thresholds.shape}')
        else:
            chest_imagenome_thresholds = None

        # Compute and save metrics
        results_dict['final_mimiccxr_metrics'] = _compute_and_save_metrics(
            metrics_dict=results_dict['mimiccxr_metrics'],
            metrics_to_save=metrics_to_print,
            dataset_name=eval_dataset_name,
            results_folder_path=results_folder_path,
            chexpert_thresholds=chexpert_thresholds,
            chest_imagenome_thresholds=chest_imagenome_thresholds,
            chest_imagenome_label_names=mimiccxr_vision_evaluator.chest_imagenome_label_names,
        )

    elif eval_vinbig_test_set:

        count_print(f'Running evaluator engine on {eval_dataset_name} ...')
        evaluator_engine.run(vinbig_vision_evaluator.val_dataloader)
        results_dict['vinbig_metrics'] = deepcopy(evaluator_engine.state.metrics)
        results_dict['vinbig_dataset'] = vinbig_vision_evaluator.val_dataset
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        
        metrics = results_dict['vinbig_metrics']
        pred_probs = metrics['pred_vinbig_probs']
        pred_probs = torch.stack(pred_probs).cpu().numpy()
        assert pred_probs.ndim == 2
        gt_labels = metrics['vinbig_labels']
        gt_labels = torch.stack(gt_labels).cpu().numpy()
        assert gt_labels.ndim == 2
        assert pred_probs.shape == gt_labels.shape

        label_names = vinbig_vision_evaluator.label_names
        
        # Remove classes without any ground truth
        gt_labels_sum = gt_labels.sum(axis=0)
        no_gt_classes = np.where(gt_labels_sum == 0)[0]
        if len(no_gt_classes) > 0:
            print_orange('NOTE: Removing the following classes without any positive classification labels:', bold=True)
            for i in no_gt_classes:
                print_orange(f'  {label_names[i]}')
            pred_probs = np.delete(pred_probs, no_gt_classes, axis=1)
            gt_labels = np.delete(gt_labels, no_gt_classes, axis=1)
            label_names = [x for i, x in enumerate(label_names) if i not in no_gt_classes]
            print(f'pred_probs.shape = {pred_probs.shape}')
            print(f'gt_labels.shape = {gt_labels.shape}')
            print(f'len(label_names) = {len(label_names)}')

        # PRC-AUC
        prc_auc_metrics = prc_auc_fn(gt_labels, pred_probs)
        print_magenta(f'PRC-AUC(macro_avg): {prc_auc_metrics["macro_avg"]}', bold=True)
        print_magenta(f'PRC-AUC(micro_avg): {prc_auc_metrics["micro_avg"]}', bold=True)

        prc_auc_metrics_with_boot = stratified_multilabel_bootstrap_metrics(
            gt_labels=gt_labels, pred_probs=pred_probs, metric_fn=prc_auc_score, num_bootstraps=500)

        for class_name, mean, std in zip(VINBIG_LABELS,
                                        prc_auc_metrics_with_boot['mean_per_class'],
                                        prc_auc_metrics_with_boot['std_per_class']):
            print(f'PRC-AUC({class_name}): {mean} ± {std}')
        print_magenta(f'PRC-AUC(macro_avg) with bootstrapping: {prc_auc_metrics_with_boot["mean_macro_avg"]} ± {prc_auc_metrics_with_boot["std_macro_avg"]}', bold=True)
        print_magenta(f'PRC-AUC(micro_avg) with bootstrapping: {prc_auc_metrics_with_boot["mean_micro_avg"]} ± {prc_auc_metrics_with_boot["std_micro_avg"]}', bold=True)

        metrics_dict = {
            'pred_probs': pred_probs,
            'gt_labels': gt_labels,
            'label_names': label_names,
            'prc_auc': prc_auc_metrics,
            'prc_auc_with_bootstrapping': prc_auc_metrics_with_boot,
        }
        metrics_to_save = list(metrics_dict.keys())

        # Compute and save metrics
        results_dict['final_vinbig_metrics'] = _compute_and_save_metrics(
            metrics_dict=metrics_dict,
            metrics_to_save=metrics_to_save,
            dataset_name=eval_dataset_name,
            results_folder_path=results_folder_path,
            chexpert_thresholds=None,
            chest_imagenome_thresholds=None,
            chest_imagenome_label_names=None,
        )
    

    torch.cuda.empty_cache()
    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder_path,
    eval_dataset_name,
    batch_size=100,
    num_workers=0,
    device='GPU',
    return_results=False,
    use_amp=False,
    calibrate_thresholds=False,
    save_probs=False,
    use_ensemble=False,
    model_names_for_ensemble=None,
    label_name_for_ensemble=None,
    chest_imagenome_label_names_filename=None,
    chest_imagenome_labels_filename=None,
    cheat=False,
):
    print()
    print_blue('----- Evaluating model ------', bold=True)

    # Sanity checks
    if use_ensemble:
        assert model_names_for_ensemble is not None
        assert label_name_for_ensemble is not None
    else:
        assert checkpoint_folder_path is not None

    if not use_ensemble:
        metadata = load_metadata(checkpoint_folder_path)
        model_kwargs = _recover_model_kwargs(metadata)
        trainer_engine_kwargs = metadata['trainer_engine_kwargs']
        auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']
        mimiccxr_vision_evaluator_kwargs = None
        vinbig_vision_evaluator_kwargs = None
        if eval_dataset_name in [EvalDatasets.MIMICCXR_TEST_SET, EvalDatasets.CHEST_IMAGENOME_GOLD]:
            mimiccxr_vision_evaluator_kwargs = _recover_mimiccxr_vision_evaluator_kwargs(
                metadata, batch_size, num_workers, eval_dataset_name)
        elif eval_dataset_name == EvalDatasets.VINBIG_TEST_SET:
            vinbig_vision_evaluator_kwargs = _recover_vinbig_vision_evaluator_kwargs(
                metadata, batch_size, num_workers, eval_dataset_name)
        else:
            assert False, f'Unknown eval_dataset_name: {eval_dataset_name}'
    else:
        model_kwargs = None
        trainer_engine_kwargs = None
        auxiliary_tasks_kwargs = None
        mimiccxr_vision_evaluator_kwargs = None
        vinbig_vision_evaluator_kwargs = None

    return _evaluate_model(
                eval_dataset_name=eval_dataset_name,
                model_kwargs=model_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                mimiccxr_vision_evaluator_kwargs=mimiccxr_vision_evaluator_kwargs,
                vinbig_vision_evaluator_kwargs=vinbig_vision_evaluator_kwargs,
                auxiliary_tasks_kwargs=auxiliary_tasks_kwargs,
                device=device,
                checkpoint_folder_path=checkpoint_folder_path,
                use_amp=use_amp,
                calibrate_thresholds=calibrate_thresholds,
                use_ensemble=use_ensemble,
                model_names_for_ensemble=model_names_for_ensemble,
                label_name_for_ensemble=label_name_for_ensemble,
                chest_imagenome_label_names_filename=chest_imagenome_label_names_filename,
                chest_imagenome_labels_filename=chest_imagenome_labels_filename,
                save_probs=save_probs,
                return_results=return_results,
                cheat=cheat,
            )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)