import  os
import argparse
from copy import deepcopy

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.dataloading_utils import get_vision_collate_batch_fn
from medvqa.datasets.image_processing import get_image_transform
from medvqa.evaluation.visual_module import calibrate_thresholds_for_mimiccxr_test_set
from medvqa.metrics.classification.multilabel_accuracy import MultiLabelAccuracy
from medvqa.metrics.classification.multilabel_prf1 import MultiLabelPRF1
from medvqa.metrics.classification.roc_auc import roc_auc_fn
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule
from medvqa.models.vqa.open_ended_vqa import OpenEndedVQA

from medvqa.utils.constants import (
    DATASET_NAMES,
    MIMICCXR_DATASET_ID,
    MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE,
    MetricNames,
)
from medvqa.metrics import (
    attach_chexpert_labels_prf1,
    attach_chexpert_labels_roc_auc,
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
from medvqa.utils.files import (
    get_results_folder_path,
    save_to_pickle,
)
from medvqa.training.vision import get_engine
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleTrainer
from medvqa.utils.logging import CountPrinter, print_blue

class EvalDatasets:
    MIMICCXR_TEST_SET = 'mimiccxr_test_set'

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--eval-dataset-name', type=str, required=True)
    parser.add_argument('--checkpoint-folder', type=str, required=True)

    # optional arguments
    parser.add_argument('--batch-size', type=int, default=140)
    parser.add_argument('--device', type=str, default='GPU')
    parser.add_argument('--num-workers', type=int, default=0)

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)

    parser.add_argument('--calibrate-thresholds', dest='calibrate_thresholds', action='store_true')
    parser.set_defaults(calibrate_thresholds=False)
    
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
        pred_chexpert_labels = (pred_chexpert_probs >= chexpert_thresholds).astype(int)
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

    if chest_imagenome_thresholds is not None:
        assert chest_imagenome_label_names is not None
        strings.append('chest-imagenome-calib-thresh')
        pred_chest_imagenome_probs = metrics_dict['pred_chest_imagenome_probs']
        pred_chest_imagenome_probs = torch.stack(pred_chest_imagenome_probs).numpy()
        pred_chest_imagenome_labels = (pred_chest_imagenome_probs >= chest_imagenome_thresholds).astype(int)
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
        # save label names
        metrics['chest_imagenome_label_names'] = chest_imagenome_label_names
    
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_multilabel_classification_metrics{"(" + "-".join(strings) + ")" if strings else ""}.pkl')
    save_to_pickle(metrics, save_path)
    print (f'Multilabel classification metrics saved to {save_path}')
    return metrics

def _recover_model_kwargs(metadata):
    kwargs = metadata['model_kwargs']
    kwargs.update(metadata['auxiliary_tasks_kwargs'])
    return kwargs

def _recover_vision_dataset_manager_kwargs(dataset_name, metadata, batch_size, num_workers):
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

    kwargs['batch_size'] = batch_size
    kwargs['num_workers'] = num_workers
    kwargs['use_test_set'] = True
    kwargs['data_augmentation_enabled'] = False
    
    # Define test image transform
    image_transform_kwargs = metadata['val_image_transform_kwargs']
    if DATASET_NAMES.MIMICCXR in image_transform_kwargs:
        image_transform = get_image_transform(**image_transform_kwargs[DATASET_NAMES.MIMICCXR])
    else: # for backward compatibility
        image_transform = get_image_transform(**image_transform_kwargs)
    kwargs['test_image_transform'] = image_transform

    # Define collate_batch_fn
    use_detectron2 = kwargs.get('use_detectron2', False)
    try:
        collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    except KeyError:
        _kwargs = dict(
            include_image=True,
            classify_tags=metadata['auxiliary_tasks_kwargs']['classify_tags'],
            n_tags=metadata['auxiliary_tasks_kwargs']['n_medical_tags'],
            classify_orientation=metadata['auxiliary_tasks_kwargs']['classify_orientation'],
            classify_gender=metadata['auxiliary_tasks_kwargs']['classify_orientation'],
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
        mimiccxr_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR_CHEST_IMAGENOME__DETECTRON2_MODE])
    else:
        mimiccxr_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR])
    kwargs['collate_batch_fn'] = mimiccxr_collate_batch_fn

    return kwargs

def _recover_mimiccxr_vision_evaluator_kwargs(metadata, batch_size, num_workers):
    return _recover_vision_dataset_manager_kwargs('mimiccxr', metadata, batch_size, num_workers)

def _calibrate_thresholds_using_chexpert_for_mimiccxr(model, device, use_amp, mimiccxr_vision_evaluator_kwargs):
    print_blue('Calibrating thresholds using MIMICCXR validation dataset and CheXpert labels')
    return calibrate_thresholds_for_mimiccxr_test_set(
        model, device, use_amp, mimiccxr_vision_evaluator_kwargs,
        classify_chexpert=True, classify_chest_imagenome=False)

def _calibrate_thresholds_using_chest_imagenome_for_mimiccxr(model, device, use_amp, mimiccxr_vision_evaluator_kwargs):
    print_blue('Calibrating thresholds using MIMICCXR validation dataset and Chest-ImaGenome labels')
    return calibrate_thresholds_for_mimiccxr_test_set(
        model, device, use_amp, mimiccxr_vision_evaluator_kwargs,
        classify_chexpert=False, classify_chest_imagenome=True, max_iterations=5)
        
def _evaluate_model(
    eval_dataset_name,
    model_kwargs,
    mimiccxr_vision_evaluator_kwargs,
    auxiliary_tasks_kwargs,
    device='GPU',
    checkpoint_folder_path=None,
    use_amp=False,
    calibrate_thresholds=False,
    return_results=False,
):
    # Sanity checks
    assert eval_dataset_name in [EvalDatasets.MIMICCXR_TEST_SET]
    
    eval_mimiccxr = eval_dataset_name in [EvalDatasets.MIMICCXR_TEST_SET]

    # Pull out some args from kwargs
    # auxiliary task: medical tags prediction
    classify_tags = auxiliary_tasks_kwargs['classify_tags']
    # auxiliary task: gender classification
    classify_gender = auxiliary_tasks_kwargs['classify_gender']
    # auxiliary task: orientation classification
    classify_orientation = auxiliary_tasks_kwargs['classify_orientation']
    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs['classify_chest_imagenome']
    # auxiliary task: chest imagenome bounding boxes
    predict_bboxes_chest_imagenome = auxiliary_tasks_kwargs['predict_bboxes_chest_imagenome']
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
    if model_name == 'vqa':
        model = OpenEndedVQA(**model_kwargs, device=device, use_visual_module_only=True,
                            vocab_size=None, start_idx=None)
    elif model_name == 'visual_module':
        model = MultiPurposeVisualModule(**model_kwargs)
    else:
        raise ValueError(f'Invalid model_name: {model_name}')
    model = model.to(device)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    evaluator = get_engine(model=model, classify_tags=classify_tags, classify_orientation=classify_orientation,
                            classify_gender=classify_gender, classify_chexpert=classify_chexpert,
                            classify_questions=classify_questions,
                            classify_chest_imagenome=classify_chest_imagenome,
                            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
                            device=device, use_amp=use_amp, training=False)
    
    # Create MIMIC-CXR visual module evaluator
    if eval_mimiccxr:
        count_print('Creating MIMIC-CXR visual module evaluator ...')
        assert mimiccxr_vision_evaluator_kwargs['use_test_set']
        mimiccxr_vision_evaluator = MIMICCXR_VisualModuleTrainer(**mimiccxr_vision_evaluator_kwargs)

    # Attach metrics, timer and events to engines
    count_print('Attaching metrics, timer and events to engines ...')

    # Metrics
    if classify_tags:
        attach_medical_tags_f1score(evaluator, device)

    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(evaluator)

    if classify_chexpert:
        attach_chexpert_labels_accuracy(evaluator, device)        
        attach_chexpert_labels_prf1(evaluator, device)
        attach_chexpert_labels_roc_auc(evaluator, 'cpu')

    if classify_questions:
        attach_question_labels_prf1(evaluator, device)

    if classify_chest_imagenome:
        attach_chest_imagenome_labels_accuracy(evaluator, device)
        attach_chest_imagenome_labels_prf1(evaluator, device)
        attach_chest_imagenome_labels_roc_auc(evaluator, 'cpu')

    # Accumulators
    attach_accumulator(evaluator, 'idxs')
    if classify_chexpert:
        attach_accumulator(evaluator, 'chexpert')
        attach_accumulator(evaluator, 'pred_chexpert_probs')
    if classify_chest_imagenome:
        attach_accumulator(evaluator, 'chest_imagenome')
        attach_accumulator(evaluator, 'pred_chest_imagenome_probs')
    
    # Timer
    timer = Timer()
    timer.attach(evaluator, start=Events.EPOCH_STARTED)
    
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

    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()    
    
    # Attach handlers    
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation
    results_dict = {}

    if eval_mimiccxr:
        count_print('Running evaluator engine on MIMIC-CXR test set ...')
        print('len(mimiccxr_vision_evaluator.test_dataset) =', len(mimiccxr_vision_evaluator.test_dataset))
        print('len(mimiccxr_vision_evaluator.test_dataloader) =', len(mimiccxr_vision_evaluator.test_dataloader))
        evaluator.run(mimiccxr_vision_evaluator.test_dataloader)        
        results_dict['mimiccxr_metrics'] = deepcopy(evaluator.state.metrics)
        results_dict['mimiccxr_dataset'] = mimiccxr_vision_evaluator.test_dataset

        if calibrate_thresholds:
            kwargs = mimiccxr_vision_evaluator_kwargs.copy()
            kwargs['use_test_set'] = False
            kwargs['use_val_set_only'] = True
            kwargs['val_image_transform'] = kwargs['test_image_transform']
            assert kwargs['val_image_transform'] is not None
        
        # Calibrate thresholds
        if classify_chexpert and calibrate_thresholds:
            chexpert_thresholds = _calibrate_thresholds_using_chexpert_for_mimiccxr(model, device, use_amp, kwargs)
        else:
            chexpert_thresholds = None
        if classify_chest_imagenome and calibrate_thresholds:
            chest_imagenome_thresholds = _calibrate_thresholds_using_chest_imagenome_for_mimiccxr(model, device, use_amp, kwargs)
        else:
            chest_imagenome_thresholds = None

        # Compute and save metrics
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        results_dict['final_mimiccxr_metrics'] = _compute_and_save_metrics(
            metrics_dict=results_dict['mimiccxr_metrics'],
            metrics_to_save=metrics_to_print,
            dataset_name=eval_dataset_name,
            results_folder_path=results_folder_path,
            chexpert_thresholds=chexpert_thresholds,
            chest_imagenome_thresholds=chest_imagenome_thresholds,
            chest_imagenome_label_names=mimiccxr_vision_evaluator.chest_imagenome_label_names,
        )
    

    torch.cuda.empty_cache()
    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder,
    eval_dataset_name,
    batch_size=100,
    num_workers=0,
    device='GPU',
    return_results=False,
    use_amp=False,
    calibrate_thresholds=False,
):
    print()
    print_blue('----- Evaluating model ------')

    # Sanity checks
    assert checkpoint_folder is not None
    assert eval_dataset_name in [EvalDatasets.MIMICCXR_TEST_SET]

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)        
    model_kwargs = _recover_model_kwargs(metadata)
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']    

    if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
        mimiccxr_vision_evaluator_kwargs = _recover_mimiccxr_vision_evaluator_kwargs(metadata, batch_size, num_workers)
    else:
        mimiccxr_vision_evaluator_kwargs = None

    return _evaluate_model(
                eval_dataset_name=eval_dataset_name,
                model_kwargs=model_kwargs,
                mimiccxr_vision_evaluator_kwargs=mimiccxr_vision_evaluator_kwargs,
                auxiliary_tasks_kwargs=auxiliary_tasks_kwargs,
                device=device,
                checkpoint_folder_path=checkpoint_folder,
                use_amp=use_amp,
                calibrate_thresholds=calibrate_thresholds,
                return_results=return_results,
            )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)