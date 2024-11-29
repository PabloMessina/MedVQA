import argparse
import os
import torch
import numpy as np

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.chexlocalize import CHEXLOCALIZE_CLASS_NAMES
from medvqa.datasets.chexlocalize.chexlocalize_dataset_management import CheXlocalizePhraseGroundingTrainer
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBigPhraseGroundingTrainer
from medvqa.metrics.bbox.utils import calculate_exact_iou_union, compute_mAP__yolov11, compute_mean_iou_per_class__yolov11, find_optimal_conf_iou_thresholds
from medvqa.metrics.classification.prc_auc import prc_auc_fn
from medvqa.utils.files import get_results_folder_path, save_pickle
from medvqa.utils.handlers import (
    attach_accumulator,
    get_log_metrics_handler,
    get_log_iteration_handler,
)

from medvqa.models.checkpoint import get_checkpoint_filepath
from medvqa.models.checkpoint.model_wrapper import ModelWrapper

from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED,
    CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
    get_chest_imagenome_gold_class_mask,
)
from medvqa.datasets.mimiccxr.mimiccxr_phrase_grounding_dataset_management import MIMICCXR_PhraseGroundingTrainer

from medvqa.models.phrase_grounding.phrase_grounder import PhraseGrounder

from medvqa.utils.constants import (
    DATASET_NAMES,
    VINBIG_BBOX_NAMES,
    VINBIG_NUM_BBOX_CLASSES,
    MetricNames,
)
from medvqa.metrics import (
    attach_condition_aware_accuracy,
    attach_condition_aware_bbox_iou_per_class,
    attach_condition_aware_chest_imagenome_bbox_iou,
    attach_condition_aware_segmask_iou,
    attach_condition_aware_segmask_iou_per_class,
)
from medvqa.models.checkpoint import (
    load_metadata,
)
from medvqa.utils.common import parsed_args_to_dict
from medvqa.training.phrase_grounding import get_engine
from medvqa.datasets.dataloading_utils import (
    get_phrase_grounding_collate_batch_fn,
)
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter, print_blue, print_bold, print_magenta

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # --- Required arguments

    parser.add_argument('--checkpoint_folder_path', type=str, default=None, help='Path to the checkpoint folder of the model to be evaluated')
    parser.add_argument('--max_images_per_batch', type=int, required=True, help='Max number of images per batch')
    parser.add_argument('--max_phrases_per_batch', type=int, required=True, help='Max number of phrases per batch')
    parser.add_argument('--max_phrases_per_image', type=int, required=True, help='Max number of phrases per image')

    # --- Other arguments

    # Dataset and dataloading arguments
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    parser.add_argument('--mscxr_phrase2embedding_filepath', type=str, default=None, help='Path to the MS-CXR phrase2embedding file')

    # Evaluation arguments
    parser.add_argument('--eval_chest_imagenome_gold', action='store_true')
    parser.add_argument('--eval_mscxr', action='store_true')
    parser.add_argument('--eval_chexlocalize', action='store_true')
    parser.add_argument('--eval_vinbig', action='store_true')
    parser.add_argument('--vinbig_use_training_indices_for_validation', action='store_true')
    parser.add_argument('--optimize_thresholds', action='store_true')
    parser.add_argument('--candidate_iou_thresholds', type=float, nargs='+', default=None)
    parser.add_argument('--candidate_conf_thresholds', type=float, nargs='+', default=None)
    parser.add_argument('--map_iou_thresholds', type=float, nargs='+', default=[0., 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--max_det', type=int, default=50)
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--use_classifier_confs_for_map', action='store_true', help='Use classifier confidences for mAP computation')

    
    return parser.parse_args(args=args)

def _evaluate_model(
    checkpoint_folder_path,
    model_kwargs,
    mimiccxr_trainer_kwargs,
    chexlocalize_trainer_kwargs,
    vinbig_trainer_kwargs,
    collate_batch_fn_kwargs,
    val_image_transform_kwargs,
    evaluation_engine_kwargs,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    num_workers,
    eval_chest_imagenome_gold,
    eval_mscxr,
    eval_chexlocalize,
    eval_vinbig,
    mscxr_phrase2embedding_filepath,
    device,
    vinbig_use_training_indices_for_validation,
    optimize_thresholds,
    candidate_iou_thresholds,
    candidate_conf_thresholds,
    map_iou_thresholds,
    max_det,
    use_amp,
    use_classifier_confs_for_map,
):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    use_yolov8 = (mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_yolov8', False))
    do_visual_grounding_with_bbox_regression = evaluation_engine_kwargs.get('do_visual_grounding_with_bbox_regression', False)
    do_visual_grounding_with_segmentation = evaluation_engine_kwargs.get('do_visual_grounding_with_segmentation', False)
    print(f'do_visual_grounding_with_bbox_regression = {do_visual_grounding_with_bbox_regression}')
    print(f'do_visual_grounding_with_segmentation = {do_visual_grounding_with_segmentation}')

    # Sanity checks
    assert sum([eval_chest_imagenome_gold, eval_mscxr, eval_chexlocalize, eval_vinbig]) > 0 # at least one dataset must be evaluated

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Create model
    count_print('Creating instance of PhraseGrounder ...')
    model = PhraseGrounder(**model_kwargs)
    model = model.to(device)

    # Load model from checkpoint
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path =', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True, strict=False)

    # Create phrase grounding trainers

    if eval_chest_imagenome_gold or eval_mscxr:

        count_print('Creating MIMIC-CXR Phrase Grounding Trainer ...')
        if eval_chest_imagenome_gold:
            bbox_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cibg'])
        else:
            bbox_grounding_collate_batch_fn = None
        if eval_mscxr:
            phrase_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['pg'])
        else:
            phrase_grounding_collate_batch_fn = None
        mimiccxr_trainer_kwargs['use_facts_for_train'] = False
        mimiccxr_trainer_kwargs['use_facts_for_test'] = False
        mimiccxr_trainer_kwargs['use_mscxr_for_train'] = False
        mimiccxr_trainer_kwargs['use_mscxr_for_test'] = eval_mscxr
        mimiccxr_trainer_kwargs['use_cxrlt2024_challenge_split'] = False 
        mimiccxr_trainer_kwargs['use_cxrlt2024_official_labels'] = False
        mimiccxr_trainer_kwargs['use_cxrlt2024_custom_labels'] = False
        mimiccxr_trainer_kwargs['use_chest_imagenome_for_train'] = False
        mimiccxr_trainer_kwargs['use_chest_imagenome_gold_for_test'] = eval_chest_imagenome_gold
        if mscxr_phrase2embedding_filepath is not None:
            mimiccxr_trainer_kwargs['mscxr_phrase2embedding_filepath'] = mscxr_phrase2embedding_filepath

        mimiccxr_trainer = MIMICCXR_PhraseGroundingTrainer(
            test_image_transform = get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            max_phrases_per_image=max_phrases_per_image,
            bbox_grounding_collate_batch_fn=bbox_grounding_collate_batch_fn,
            phrase_grounding_collate_batch_fn=phrase_grounding_collate_batch_fn,
            num_test_workers=num_workers,
            **mimiccxr_trainer_kwargs,
        )

    if eval_chexlocalize:

        count_print('Creating CheXlocalize Phrase Grounding Trainer ...')
        chexlocalize_trainer_kwargs['use_training_set'] = False
        chexlocalize_trainer_kwargs['use_validation_set'] = True
        chexlocalize_trainer = CheXlocalizePhraseGroundingTrainer(
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE]),
            collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cl']),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            num_val_workers=num_workers,
            **chexlocalize_trainer_kwargs,
        )
    
    if eval_vinbig:

        count_print('Creating VinBig Phrase Grounding Trainer ...')
        vinbig_trainer_kwargs['use_training_set'] = False
        vinbig_trainer_kwargs['use_validation_set'] = True
        vinbig_trainer = VinBigPhraseGroundingTrainer(
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.VINBIG]),
            collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['vbg']),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            num_val_workers=num_workers,
            use_training_indices_for_validation=vinbig_use_training_indices_for_validation,
            **vinbig_trainer_kwargs,
        )

    # Evaluate on datasets
    
    if eval_chest_imagenome_gold:

        count_print('----- Evaluating on Chest ImaGenome Gold Bbox Phrase Grounding -----')

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

        # Attach metrics
        _cond_func = lambda x: x['flag'] == 'cibg'
        if use_yolov8:
            _gold_class_mask = get_chest_imagenome_gold_class_mask()
            attach_condition_aware_chest_imagenome_bbox_iou(evaluation_engine, _cond_func, use_yolov8=True, class_mask=_gold_class_mask)
        if do_visual_grounding_with_bbox_regression:
            attach_condition_aware_bbox_iou_per_class(evaluation_engine,
                                                      field_names=['predicted_bboxes', 'chest_imagenome_bbox_coords', 'chest_imagenome_bbox_presence'],
                                                      metric_name='bbox_iou', nc=CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
                                                      condition_function=_cond_func)
        else:
            attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
                                                        nc=CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
                                                        condition_function=_cond_func)
        # Attach accumulators
        if do_visual_grounding_with_bbox_regression:
            attach_accumulator(evaluation_engine, 'predicted_bboxes')
            attach_accumulator(evaluation_engine, 'chest_imagenome_bbox_coords')
            attach_accumulator(evaluation_engine, 'chest_imagenome_bbox_presence')
        else:
            attach_accumulator(evaluation_engine, 'pred_mask')
            attach_accumulator(evaluation_engine, 'gt_mask')

        # for logging
        metrics_to_print = []
        if use_yolov8:
            metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXIOU)
        if do_visual_grounding_with_bbox_regression:
            metrics_to_print.append('bbox_iou')
        else:
            metrics_to_print.append('segmask_iou')

        # Timer
        timer = Timer()
        timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

        # Logging
        print_blue('Defining log_metrics_handler ...', bold=True)
        log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
        log_iteration_handler = get_log_iteration_handler()
        
        # Attach handlers
        evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
        evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

        # Start evaluation
        print_blue('Running engine ...', bold=True)
        evaluation_engine.run(mimiccxr_trainer.test_chest_imagenome_gold_dataloader)

        # Print final metrics
        print_blue('Final metrics:', bold=True)
        metrics = evaluation_engine.state.metrics
        if use_yolov8:
            # 1) chest imagenome bbox iou
            metric_name = MetricNames.CHESTIMAGENOMEBBOXIOU
            print(f'{metric_name}: {metrics[metric_name]}')
            print()
        # 2) bbox iou / segmask iou
        if do_visual_grounding_with_bbox_regression:
            metric_name = 'bbox_iou'
        else:
            metric_name = 'segmask_iou'
        print(f'{metric_name}:')
        from tabulate import tabulate
        table = []
        for bbox_name, iou in zip(CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED, metrics[metric_name]):
            table.append([bbox_name, iou])
        print(tabulate(table, headers=['bbox_name', 'iou'], tablefmt='latex_raw'))

        # Save metrics to file
        dataset = mimiccxr_trainer.test_chest_imagenome_gold_dataset
        image_paths = dataset.image_paths
        phrases = mimiccxr_trainer.test_chest_imagenome_gold_bbox_phrases

        if do_visual_grounding_with_bbox_regression:
            pred_bboxes = metrics['predicted_bboxes']
            gt_bboxes = metrics['chest_imagenome_bbox_coords']
            gt_presence = metrics['chest_imagenome_bbox_presence']
            assert len(image_paths) == len(pred_bboxes) == len(gt_bboxes) == len(gt_presence)
            print_blue('Saving metrics to file ...', bold=True)
            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            save_path = os.path.join(results_folder_path, f'chest_imagenome_gold_metrics_bbox_regression.pkl')
            output = dict(
                image_paths=[],
                phrases=[],
                pred_bboxes=[],
                gt_bboxes=[],
                ious=[],
                bbox_iou=metrics['bbox_iou'],
            )
            for i in range(len(image_paths)):
                for j in range(len(pred_bboxes[i])):
                    if gt_presence[i][j] == 1:
                        if pred_bboxes[i][j] is None:
                            iou = 0
                        else:
                            iou = calculate_exact_iou_union(pred_bboxes[i][j][0], gt_bboxes[i][j])
                        output['image_paths'].append(image_paths[i])
                        output['phrases'].append(phrases[j])
                        output['pred_bboxes'].append(pred_bboxes[i][j][0].cpu().numpy() if pred_bboxes[i][j] is not None else None)
                        output['gt_bboxes'].append(gt_bboxes[i][j].cpu().numpy())
                        output['ious'].append(iou)
        else:
            gt_masks = metrics['gt_mask']
            pred_masks = metrics['pred_mask']
            assert len(image_paths) == len(pred_masks) == len(gt_masks)
            print_blue('Saving metrics to file ...', bold=True)
            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            save_path = os.path.join(results_folder_path, f'chest_imagenome_gold_metrics_segmask.pkl')
            output = dict(
                image_paths=[],
                phrases=[],
                pred_masks=[],
                gt_masks=[],
                ious=[],
                segmask_iou=metrics['segmask_iou'],
            )
            for i in range(len(image_paths)):
                for j in range(len(pred_masks[i])):
                    intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
                    union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
                    iou = intersection / union
                    iou = iou.item()
                    output['image_paths'].append(image_paths[i])
                    output['phrases'].append(phrases[j])
                    output['pred_masks'].append(pred_masks[i][j].cpu().numpy())
                    output['gt_masks'].append(gt_masks[i][j].cpu().numpy())
                    output['ious'].append(iou)

        print_magenta('mean_iou =', sum(output['ious']) / len(output['ious']), bold=True)
        save_pickle(output, save_path)
        print(f'Saved metrics to {save_path}')

    if eval_mscxr:

        count_print('----- Evaluating on MSCXR Phrase Grounding -----')

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

        # Attach metrics
        metrics_to_print = []
        _cond_func = lambda x: x['flag'] == 'pg'
        attach_condition_aware_segmask_iou(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou', _cond_func)
        metrics_to_print.append('segmask_iou')

        # Attach accumulators
        attach_accumulator(evaluation_engine, 'pred_mask')
        attach_accumulator(evaluation_engine, 'gt_mask')

        # Timer
        timer = Timer()
        timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

        # Logging
        print_blue('Defining log_metrics_handler ...', bold=True)
        log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
        log_iteration_handler = get_log_iteration_handler()
        
        # Attach handlers
        evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
        evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

        # Start evaluation
        print_blue('Running engine ...', bold=True)
        evaluation_engine.run(mimiccxr_trainer.test_mscxr_dataloader)

        # Print final metrics
        print_blue('Final metrics:', bold=True)
        metrics = evaluation_engine.state.metrics
        # 1) segmask iou
        metric_name = 'segmask_iou'
        print(f'{metric_name}: {metrics[metric_name]}')

        # Save metrics to file
        image_paths = []
        phrases = []
        for dataloader in  mimiccxr_trainer.test_mscxr_dataloader.dataloaders:
            dataset = dataloader.dataset
            for i in dataset.indices:
                image_paths.append(dataset.image_paths[i])
                phrases.append(dataset.phrases[i])
            print(f'len(image_paths) = {len(image_paths)}')
        assert len(image_paths) == len(metrics['pred_mask']) == len(metrics['gt_mask'])
        print_blue('Saving metrics to file ...', bold=True)
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        save_path = os.path.join(results_folder_path, f'mscxr_metrics.pkl')
        pred_masks = metrics['pred_mask']
        gt_masks = metrics['gt_mask']
        output = dict(
            image_paths=[],
            phrases=[],
            pred_masks=[],
            gt_masks=[],
            ious=[],
            segmask_iou=metrics['segmask_iou'],
        )
        for i in range(len(image_paths)):
            for j in range(len(pred_masks[i])):
                intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
                union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
                iou = intersection / union
                iou = iou.item()
                output['image_paths'].append(image_paths[i])
                output['phrases'].append(phrases[i][j])
                output['pred_masks'].append(pred_masks[i][j].cpu().numpy())
                output['gt_masks'].append(gt_masks[i][j].cpu().numpy())
                output['ious'].append(iou)
        print_magenta('mean_iou =', sum(output['ious']) / len(output['ious']), bold=True)
        save_pickle(output, save_path)
        print(f'Saved metrics to {save_path}')

    if eval_chexlocalize:

        count_print('----- Evaluating on CheXlocalize Phrase Grounding -----')

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

        # Attach metrics
        metrics_to_print = []
        _cond_func = lambda x: x['flag'] == 'cl'
        attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
                                                      nc=len(CHEXLOCALIZE_CLASS_NAMES),
                                                      condition_function=_cond_func)
        attach_condition_aware_accuracy(evaluation_engine, 'pred_labels', 'gt_labels', 'classif_acc', _cond_func)
        metrics_to_print.append('segmask_iou')
        metrics_to_print.append('classif_acc')

        # Attach accumulators
        attach_accumulator(evaluation_engine, 'pred_mask')
        attach_accumulator(evaluation_engine, 'gt_mask')
        attach_accumulator(evaluation_engine, 'pred_probs')
        attach_accumulator(evaluation_engine, 'gt_labels')

        # Timer
        timer = Timer()
        timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

        # Logging
        print_blue('Defining log_metrics_handler ...', bold=True)
        log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
        log_iteration_handler = get_log_iteration_handler()
        
        # Attach handlers
        evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
        evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

        # Start evaluation
        print_blue('Running engine ...', bold=True)
        evaluation_engine.run(chexlocalize_trainer.val_dataloader)

        # Print final metrics
        print_blue('Final metrics:', bold=True)
        metrics = evaluation_engine.state.metrics
        # 1) segmask iou
        metric_name = 'segmask_iou'
        print(f'{metric_name}: {metrics[metric_name]}')
        # 2) classif acc
        metric_name = 'classif_acc'
        print(f'{metric_name}: {metrics[metric_name]}')
        # 3) PRC-AUC
        pred_probs = metrics['pred_probs']
        pred_probs = torch.tensor(pred_probs).cpu().numpy()
        assert pred_probs.ndim == 1
        gt_labels = metrics['gt_labels']
        gt_labels = torch.tensor(gt_labels).cpu().numpy()
        assert gt_labels.ndim == 1
        pred_probs =  pred_probs.reshape(-1, len(CHEXLOCALIZE_CLASS_NAMES))
        gt_labels = gt_labels.reshape(-1, len(CHEXLOCALIZE_CLASS_NAMES))
        assert pred_probs.shape == gt_labels.shape
        assert pred_probs.shape[0] == len(chexlocalize_trainer.val_dataset)
        prc_auc_metrics = prc_auc_fn(pred_probs, gt_labels)
        for class_name, prc_auc in zip(CHEXLOCALIZE_CLASS_NAMES, prc_auc_metrics['per_class']):
            print(f'  PRC-AUC({class_name}): {prc_auc}')
        print(f'PRC-AUC(macro_avg): {prc_auc_metrics["macro_avg"]}')
        print(f'PRC-AUC(micro_avg): {prc_auc_metrics["micro_avg"]}')

        # Save metrics to file
        dataset = chexlocalize_trainer.val_dataset
        image_paths = dataset.image_paths
        phrases = chexlocalize_trainer.class_phrases
        gt_masks = metrics['gt_mask']
        pred_masks = metrics['pred_mask']
        assert len(image_paths) == len(pred_masks) == len(gt_masks)
        print_blue('Saving metrics to file ...', bold=True)
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        save_path = os.path.join(results_folder_path, f'chexlocalize_metrics.pkl')
        output = dict(
            image_paths=[],
            phrases=[],
            pred_masks=[],
            gt_masks=[],
            ious=[],
            segmask_iou=metrics['segmask_iou'],
            classif_acc=metrics['classif_acc'],
            prc_auc=prc_auc_metrics,
        )
        for i in range(len(image_paths)):
            for j in range(len(pred_masks[i])):
                if gt_labels[i,j] == 1:
                    intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
                    union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
                    iou = intersection / union
                    iou = iou.item()
                    output['image_paths'].append(image_paths[i])
                    output['phrases'].append(phrases[j])
                    output['pred_masks'].append(pred_masks[i][j].cpu().numpy())
                    output['gt_masks'].append(gt_masks[i][j].cpu().numpy())
                    output['ious'].append(iou)
                else:
                    assert torch.all(gt_masks[i][j] == 0)

        print_magenta('mean_iou =', sum(output['ious']) / len(output['ious']), bold=True)
        save_pickle(output, save_path)
        print(f'Saved metrics to {save_path}')

    if eval_vinbig:

        count_print('----- Evaluating on VinDr-CXR Phrase Grounding -----')

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        if optimize_thresholds:
            evaluation_engine_kwargs['skip_nms'] = True # We need to skip NMS to optimize thresholds
        evaluation_engine_kwargs['use_amp'] = use_amp
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

        # Attach metrics
        metrics_to_print = []
        _cond_func = lambda x: x['flag'] == 'vbg'
        # if do_visual_grounding_with_bbox_regression:
        #     if not optimize_thresholds:
        #         attach_condition_aware_bbox_iou_per_class(evaluation_engine,
        #                                                 field_names=['predicted_bboxes', 'vinbig_bbox_coords', 'vinbig_bbox_classes'],
        #                                                 metric_name='bbox_iou', nc=len(VINBIG_BBOX_NAMES), condition_function=_cond_func,
        #                                                 for_vinbig=True)
        #         metrics_to_print.append('bbox_iou')
        if do_visual_grounding_with_segmentation:
            attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
                                                        nc=len(VINBIG_BBOX_NAMES), condition_function=_cond_func)
            metrics_to_print.append('segmask_iou')

        # Attach accumulators
        attach_accumulator(evaluation_engine, 'pred_probs')
        attach_accumulator(evaluation_engine, 'gt_labels')
        if do_visual_grounding_with_bbox_regression:
            if optimize_thresholds:
                attach_accumulator(evaluation_engine, 'pred_bbox_probs')
                attach_accumulator(evaluation_engine, 'pred_bbox_coords')
            else:
                attach_accumulator(evaluation_engine, 'predicted_bboxes')
            attach_accumulator(evaluation_engine, 'vinbig_bbox_coords')
            attach_accumulator(evaluation_engine, 'vinbig_bbox_classes')
        elif do_visual_grounding_with_segmentation:
            attach_accumulator(evaluation_engine, 'pred_mask')
            attach_accumulator(evaluation_engine, 'gt_mask')

        # Timer
        timer = Timer()
        timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

        # Logging
        print_blue('Defining log_metrics_handler ...', bold=True)
        log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
        log_iteration_handler = get_log_iteration_handler()
        
        # Attach handlers
        evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
        evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

        # Start evaluation
        print_blue('Running engine ...', bold=True)
        evaluation_engine.run(vinbig_trainer.val_dataloader)

        # Print final metrics
        print_blue('Final metrics:', bold=True)
        metrics = evaluation_engine.state.metrics
        # # 1) bbox iou
        # if do_visual_grounding_with_bbox_regression:
        #     if not optimize_thresholds:
        #         metric_name = 'bbox_iou'
        #         print(f'{metric_name}: {metrics[metric_name]}')
        # 1) segmask iou
        if do_visual_grounding_with_segmentation:
            metric_name = 'segmask_iou'
            print(f'{metric_name}: {metrics[metric_name]}')        
        # 2) PRC-AUC
        dataset = vinbig_trainer.val_dataset
        phrases = vinbig_trainer.phrases
        pred_probs = metrics['pred_probs']
        pred_probs = torch.stack(pred_probs).cpu().numpy()
        assert pred_probs.ndim == 2
        gt_labels = metrics['gt_labels']
        gt_labels = torch.stack(gt_labels).cpu().numpy()
        assert gt_labels.ndim == 2
        assert pred_probs.shape == gt_labels.shape
        assert pred_probs.shape[0] == len(dataset)
        prc_auc_metrics = prc_auc_fn(pred_probs, gt_labels)
        for class_name, prc_auc in zip(phrases, prc_auc_metrics['per_class']):
            print(f'PRC-AUC({class_name}): {prc_auc}')
        print_magenta(f'PRC-AUC(macro_avg): {prc_auc_metrics["macro_avg"]}', bold=True)
        print_magenta(f'PRC-AUC(micro_avg): {prc_auc_metrics["micro_avg"]}', bold=True)
        # 3) mAP
        from sklearn.metrics import average_precision_score
        mAP_micro = average_precision_score(gt_labels, pred_probs, average='micro')
        print_magenta(f'mAP(micro_avg): {mAP_micro}', bold=True)
        mAP_macro = average_precision_score(gt_labels, pred_probs, average='macro')
        print_magenta(f'mAP(macro_avg): {mAP_macro}', bold=True)

        # Save metrics to file
        image_paths = [dataset.image_paths[i] for i in dataset.indices]

        if do_visual_grounding_with_bbox_regression:
            gt_bboxes = metrics['vinbig_bbox_coords']
            gt_classes = metrics['vinbig_bbox_classes']
            assert len(image_paths) == len(gt_bboxes) == len(gt_classes)
            gt_coords_list = [[[] for _ in range(VINBIG_NUM_BBOX_CLASSES)] for _ in range(len(gt_bboxes))]
            for i in range(len(gt_bboxes)):
                for bbox, cls in zip(gt_bboxes[i], gt_classes[i]):
                    gt_coords_list[i][cls].append(bbox)
            # convert to numpy
            for i in range(len(gt_coords_list)):
                for j in range(len(gt_coords_list[i])):
                    gt_coords_list[i][j] = np.stack(gt_coords_list[i][j]) if len(gt_coords_list[i][j]) > 0 else np.empty((0, 4))

            if use_classifier_confs_for_map:
                classifier_confs = pred_probs[:, :VINBIG_NUM_BBOX_CLASSES] # (num_samples, num_classes)
                print_bold('Using classifier confidences for mAP computation')
                print(f'classifier_confs.shape = {classifier_confs.shape}')
            else:
                classifier_confs = None

            if optimize_thresholds:
                assert candidate_iou_thresholds is not None
                assert candidate_conf_thresholds is not None
                assert max_det is not None
                pred_bbox_probs = metrics['pred_bbox_probs']
                pred_bbox_coords = metrics['pred_bbox_coords']
                num_regions = pred_bbox_probs[0].shape[1]
                num_classes = VINBIG_NUM_BBOX_CLASSES
                assert pred_bbox_probs[0].ndim == 3 # (num_classes, num_regions, 1)
                assert pred_bbox_coords[0].ndim == 3 # (num_classes, num_regions, 4)
                assert pred_bbox_probs[0].shape == (num_classes, num_regions, 1)
                assert pred_bbox_coords[0].shape == (num_classes, num_regions, 4)
                out = find_optimal_conf_iou_thresholds(
                    gt_coords_list=gt_coords_list,
                    pred_boxes_list=pred_bbox_coords,
                    pred_confs_list=pred_bbox_probs,
                    iou_thresholds=candidate_iou_thresholds,
                    conf_thresholds=candidate_conf_thresholds,
                    classifier_confs=classifier_confs,
                    max_det=max_det,
                    verbose=True,
                )
                best_iou_threshold = out['best_iou_threshold']
                best_conf_threshold = out['best_conf_threshold']
                pred_boxes_list = out['pred_boxes_list']
                pred_classes_list = out['pred_classes_list']
                pred_confs_list = out['pred_confs_list']

            else:
                pred_bboxes = metrics['predicted_bboxes']
                assert len(image_paths) == len(pred_bboxes)
                pred_boxes_list = []
                pred_classes_list = []
                pred_confs_list = []
                _empty_boxes = np.empty((0, 4))
                _empty_confs = np.empty((0,))
                for preds in pred_bboxes:
                    boxes = []
                    confs = []
                    classes = []
                    for i, pred in enumerate(preds):
                        if pred is not None:
                            boxes.append(pred[0].cpu().numpy())
                            classes.extend([i] * len(pred[0]))
                            confs.append(pred[1].cpu().numpy())
                    pred_boxes_list.append(np.concatenate(boxes) if len(boxes) > 0 else _empty_boxes)
                    pred_confs_list.append(np.concatenate(confs) if len(confs) > 0 else _empty_confs)
                    pred_classes_list.append(np.array(classes, dtype=np.int64))

            # Compute metrics
            tmp = compute_mean_iou_per_class__yolov11(
                pred_boxes=pred_boxes_list,
                pred_classes=pred_classes_list,
                gt_coords=gt_coords_list,
                compute_iou_per_sample=True,
                compute_micro_average_iou=True,
                return_counts=True,
            )
            class_ious = tmp['class_ious']
            sample_ious = tmp['sample_ious']
            class_counts = tmp['class_counts']
            sample_counts = tmp['sample_counts']
            micro_iou = tmp['micro_iou']
            class_idxs = np.where(class_counts > 0)[0]
            mean_iou = class_ious[class_idxs].mean()
            sample_idxs = np.where(sample_counts > 0)[0]
            sample_iou = sample_ious[sample_idxs].mean()
            if optimize_thresholds:
                print_magenta(f'best_iou_threshold: {best_iou_threshold}', bold=True)
                print_magenta(f'best_conf_threshold: {best_conf_threshold}', bold=True)
            for class_name, iou, count in zip(VINBIG_BBOX_NAMES, class_ious, class_counts):
                print(f'mean_iou({class_name}): {iou} ({count} samples)')
            print_magenta(f'mean_iou: {mean_iou}', bold=True)
            print(f'\t{class_idxs.shape[0]} / {len(VINBIG_BBOX_NAMES)} classes have at least one ground truth bbox')
            print_magenta(f'mean_sample_iou: {sample_iou}', bold=True)
            print(f'\t{sample_idxs.shape[0]} / {len(gt_coords_list)} samples have at least one ground truth bbox')
            print_magenta(f'micro_iou: {micro_iou} (count={sample_counts.sum()})', bold=True)

            map_per_class = compute_mAP__yolov11(
                pred_boxes=pred_boxes_list,
                pred_classes=pred_classes_list,
                pred_confs=pred_confs_list,
                classifier_confs=classifier_confs,
                gt_coords=gt_coords_list,
                iou_thresholds=map_iou_thresholds,
            )
            
            for iou_thresh, map_ in zip(map_iou_thresholds, map_per_class.mean(axis=1)):
                print_magenta(f'mAP@{iou_thresh}: {map_}', bold=True)
            
            print_blue('Saving metrics to file ...', bold=True)
            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            strings = [
                'bbox_regression',
                f'{len(vinbig_trainer.val_dataset)}',
            ]
            if optimize_thresholds:
                strings.append(f'opt_thr({best_iou_threshold:.2f},{best_conf_threshold:.2f},{max_det})')
            if use_classifier_confs_for_map:
                strings.append('use_classifier_confs')
            save_path = os.path.join(results_folder_path, f'vindrcxr_metrics({",".join(strings)}).pkl')
            output = dict(
                image_paths=image_paths,
                pred_boxes_list=pred_boxes_list,
                pred_classes_list=pred_classes_list,
                pred_confs_list=pred_confs_list,
                classifier_confs=classifier_confs,
                gt_bboxes=gt_coords_list,
                class_ious=class_ious,
                sample_ious=sample_ious,
                micro_iou=micro_iou,
                macro_iou=mean_iou,
                map_iou_thresholds=map_iou_thresholds,
                map_per_class=map_per_class,
                prc_auc=prc_auc_metrics,
            )
            if optimize_thresholds:
                output['best_iou_threshold'] = best_iou_threshold
                output['best_conf_threshold'] = best_conf_threshold
                output['max_det'] = max_det

        elif do_visual_grounding_with_segmentation:
            gt_masks = metrics['gt_mask']
            pred_masks = metrics['pred_mask']
            assert len(image_paths) == len(pred_masks) == len(gt_masks),\
                (f'len(image_paths) = {len(image_paths)}, len(pred_masks) = {len(pred_masks)}, len(gt_masks) = {len(gt_masks)}')
            print_blue('Saving metrics to file ...', bold=True)
            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            save_path = os.path.join(results_folder_path, f'vindrcxr_metrics(segmask,{len(vinbig_trainer.val_dataset)}).pkl')
            output = dict(
                image_paths=[],
                phrases=[],
                pred_masks=[],
                gt_masks=[],
                ious=[],
                segmask_iou=metrics['segmask_iou'],
                prc_auc=prc_auc_metrics,
            )
            for i in range(len(image_paths)):
                for j in range(len(pred_masks[i])):
                    if gt_labels[i, j] == 1:
                        intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
                        union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
                        iou = intersection / union
                        iou = iou.item()
                        output['image_paths'].append(image_paths[i])
                        output['phrases'].append(phrases[j])
                        output['pred_masks'].append(pred_masks[i][j].cpu().numpy())
                        output['gt_masks'].append(gt_masks[i][j].cpu().numpy())
                        output['ious'].append(iou)
                    else:
                        assert torch.all(gt_masks[i][j] == 0)

            print_magenta('mean_iou =', sum(output['ious']) / len(output['ious']), bold=True)

        if do_visual_grounding_with_bbox_regression or do_visual_grounding_with_segmentation:
            save_pickle(output, save_path)
            print(f'Saved metrics to {save_path}')

def evaluate(
    checkpoint_folder_path,
    num_workers,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    eval_chest_imagenome_gold,
    eval_mscxr,
    eval_chexlocalize,
    eval_vinbig,
    mscxr_phrase2embedding_filepath,
    device,
    vinbig_use_training_indices_for_validation,
    optimize_thresholds,
    candidate_iou_thresholds,
    candidate_conf_thresholds,
    map_iou_thresholds,
    max_det,
    use_amp,
    use_classifier_confs_for_map,
):
    print_blue('----- Evaluating model -----', bold=True)

    metadata = load_metadata(checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
    chexlocalize_trainer_kwargs = metadata['chexlocalize_trainer_kwargs']
    vinbig_trainer_kwargs = metadata['vinbig_trainer_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    try:
        val_image_transform_kwargs = metadata['val_image_transform_kwargs']
    except KeyError: # HACK: when val_image_transform_kwargs is missing due to a bug
        val_image_transform_kwargs = {
            DATASET_NAMES.MIMICCXR: dict(
                image_size=(416, 416),
                augmentation_mode=None,
                use_bbox_aware_transform=True,
                for_yolov8=True,
            )
        }
    validator_engine_kwargs = metadata['validator_engine_kwargs']

    return _evaluate_model(
                checkpoint_folder_path=checkpoint_folder_path,
                model_kwargs=model_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                chexlocalize_trainer_kwargs=chexlocalize_trainer_kwargs,
                vinbig_trainer_kwargs=vinbig_trainer_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                evaluation_engine_kwargs=validator_engine_kwargs,
                max_images_per_batch=max_images_per_batch,
                max_phrases_per_batch=max_phrases_per_batch,
                max_phrases_per_image=max_phrases_per_image,
                num_workers=num_workers,
                eval_chest_imagenome_gold=eval_chest_imagenome_gold,
                eval_mscxr=eval_mscxr,
                eval_chexlocalize=eval_chexlocalize,
                eval_vinbig=eval_vinbig,
                mscxr_phrase2embedding_filepath=mscxr_phrase2embedding_filepath,
                device=device,
                vinbig_use_training_indices_for_validation=vinbig_use_training_indices_for_validation,
                optimize_thresholds=optimize_thresholds,
                candidate_iou_thresholds=candidate_iou_thresholds,
                candidate_conf_thresholds=candidate_conf_thresholds,
                map_iou_thresholds=map_iou_thresholds,
                max_det=max_det,
                use_amp=use_amp,
                use_classifier_confs_for_map=use_classifier_confs_for_map,
            )


if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)