import argparse
import os
import torch
import numpy as np
from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBig_VisualModuleTrainer
from medvqa.metrics import attach_condition_aware_bbox_iou_per_class
from medvqa.metrics.bbox.utils import (
    compute_mAP__yolov11,
    compute_mean_iou_per_class__yolov11,
    find_optimal_conf_iou_thresholds,
)
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule
from medvqa.utils.files import get_results_folder_path, save_pickle
from medvqa.utils.handlers import (
    attach_accumulator,
    get_log_metrics_handler,
    get_log_iteration_handler,
)
from medvqa.models.checkpoint import get_checkpoint_filepath
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.utils.constants import (
    DATASET_NAMES,
    VINBIG_BBOX_NAMES,
    VINBIG_NUM_BBOX_CLASSES,
)
from medvqa.models.checkpoint import (
    load_metadata,
)
from medvqa.utils.common import parsed_args_to_dict
from medvqa.training.vision import get_engine
from medvqa.datasets.dataloading_utils import get_vision_collate_batch_fn
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter, print_blue, print_magenta

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # --- Required arguments

    parser.add_argument('--checkpoint_folder_path', type=str, required=True, help='Path to the folder containing the model checkpoint')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    
    # --- Other arguments

    # Dataset and dataloading arguments
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    parser.add_argument('--vinbig_use_training_indices_for_validation', action='store_true')

    # Evaluation arguments
    parser.add_argument('--eval_vinbig', action='store_true')
    parser.add_argument('--optimize_thresholds', action='store_true')
    parser.add_argument('--candidate_iou_thresholds', type=float, nargs='+', default=None)
    parser.add_argument('--candidate_conf_thresholds', type=float, nargs='+', default=None)
    parser.add_argument('--map_iou_thresholds', type=float, nargs='+', default=[0., 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--max_det', type=int, default=100)
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')

    return parser.parse_args(args=args)

def _evaluate_model(
    checkpoint_folder_path,
    model_kwargs,
    vinbig_trainer_kwargs,
    collate_batch_fn_kwargs,
    val_image_transform_kwargs,
    evaluation_engine_kwargs,
    batch_size,
    num_workers,
    eval_vinbig,
    vinbig_use_training_indices_for_validation,
    device,
    optimize_thresholds,
    candidate_iou_thresholds,
    candidate_conf_thresholds,
    map_iou_thresholds,
    max_det,
    use_amp,
):
    count_print = CountPrinter()

    # Sanity checks
    assert sum([eval_vinbig]) > 0 # at least one dataset must be evaluated

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Create model
    count_print('Creating instance of PhraseGrounder ...')
    model = MultiPurposeVisualModule(**model_kwargs, device=device)
    model = model.to(device)

    # Load model from checkpoint
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path =', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True, strict=False)
    
    # Evaluate on VinDr-CXR
    if eval_vinbig:

        count_print('----- Evaluating on VinDr-CXR -----')

        count_print('Creating visual module trainer ...')
        vinbig_trainer_kwargs['use_training_set'] = False
        vinbig_trainer_kwargs['use_validation_set'] = True
        vinbig_trainer = VinBig_VisualModuleTrainer(
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.VINBIG]),
            collate_batch_fn=get_vision_collate_batch_fn(**collate_batch_fn_kwargs['vinbig']),
            batch_size=batch_size,
            num_workers=num_workers,
            use_training_indices_for_validation=vinbig_use_training_indices_for_validation,
            **vinbig_trainer_kwargs,
        )

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        if optimize_thresholds:
            evaluation_engine_kwargs['apply_nms'] = False # We need to skip NMS to optimize thresholds
        evaluation_engine_kwargs['use_amp'] = use_amp
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

        # Attach metrics and accumulators
        _cond_func = lambda x: x['flag'] == 'vinbig'
        metrics_to_print = []
        if optimize_thresholds:
            attach_accumulator(evaluation_engine, 'resized_shape', append_instead_of_extend=True)
            attach_accumulator(evaluation_engine, 'yolov11_predictions', append_instead_of_extend=True)
        else:
            attach_accumulator(evaluation_engine, 'yolov11_predictions')
            attach_condition_aware_bbox_iou_per_class(evaluation_engine, ('yolov11_predictions', 'vinbig_bbox_coords', 'vinbig_bbox_classes'),
                                                      'vnb_y11_bbox_iou', VINBIG_NUM_BBOX_CLASSES, _cond_func, for_vinbig=True, use_yolov8=True)
            metrics_to_print.append('vnb_y11_bbox_iou')
        attach_accumulator(evaluation_engine, 'vinbig_bbox_coords')
        attach_accumulator(evaluation_engine, 'vinbig_bbox_classes')

        # Timer
        timer = Timer()
        timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

        # Logging
        count_print('Defining log_metrics_handler ...')
        log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
        log_iteration_handler = get_log_iteration_handler()
        
        # Attach handlers
        evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
        evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

        # Start evaluation
        count_print('Running engine ...')
        evaluation_engine.run(vinbig_trainer.val_dataloader)

        # Compute metrics
        count_print('Computing metrics ...')
        
        metrics = evaluation_engine.state.metrics
        dataset = vinbig_trainer.val_dataset
        image_paths = [dataset.image_paths[i] for i in dataset.indices]
        
        gt_bboxes = metrics['vinbig_bbox_coords']
        gt_classes = metrics['vinbig_bbox_classes']
        assert len(image_paths) == len(gt_bboxes) == len(gt_classes)

        if not optimize_thresholds:
            assert 'vnb_y11_bbox_iou' in metrics
            print('metrics["vnb_y11_bbox_iou"] =', metrics['vnb_y11_bbox_iou'])

        # Prepare ground truth bboxes
        gt_coords_list = [[[] for _ in range(VINBIG_NUM_BBOX_CLASSES)] for _ in range(len(gt_bboxes))]
        for i in range(len(gt_bboxes)):
            for bbox, cls in zip(gt_bboxes[i], gt_classes[i]):
                gt_coords_list[i][cls].append(bbox)
        for i in range(len(gt_coords_list)):
            for j in range(len(gt_coords_list[i])):
                gt_coords_list[i][j] = np.stack(gt_coords_list[i][j]) if len(gt_coords_list[i][j]) > 0 else np.empty((0, 4))

        if optimize_thresholds:
            # Compute optimal thresholds
            assert candidate_iou_thresholds is not None
            assert candidate_conf_thresholds is not None
            assert max_det is not None
            yolov11_predictions = metrics['yolov11_predictions']
            resized_shapes = metrics['resized_shape']
            assert len(yolov11_predictions) == len(resized_shapes)
            assert sum(len(preds) for preds in yolov11_predictions) == len(image_paths)
            assert all(len(preds) == len(rs) for preds, rs in zip(yolov11_predictions, resized_shapes))
            print('yolov11_predictions[0].shape =', yolov11_predictions[0].shape)
            print('resized_shapes[0][0] =', resized_shapes[0][0])
            
            out = find_optimal_conf_iou_thresholds(
                gt_coords_list=gt_coords_list,
                yolo_predictions_list=yolov11_predictions,
                resized_shape_list=resized_shapes,
                iou_thresholds=candidate_iou_thresholds,
                conf_thresholds=candidate_conf_thresholds,
                max_det=max_det,
                verbose=True,
            )
            best_iou_threshold = out['best_iou_threshold']
            best_conf_threshold = out['best_conf_threshold']
            pred_boxes_list = out['pred_boxes_list']
            pred_classes_list = out['pred_classes_list']
            pred_confs_list = out['pred_confs_list']
        else:
            yolov11_predictions = metrics['yolov11_predictions']
            assert len(image_paths) == len(yolov11_predictions)
            assert yolov11_predictions[0].ndim == 2 # (num_boxes, 6) (6 means [x1, y1, x2, y2, conf, cls])
            assert yolov11_predictions[0].shape[-1] == 6
            pred_boxes_list = []
            pred_classes_list = []
            pred_confs_list = []
            for pred in yolov11_predictions:
                pred = pred.cpu().numpy()
                pred_boxes_list.append(pred[:, :4])
                pred_confs_list.append(pred[:, 4])
                pred_classes_list.append(pred[:, 5].astype(int))

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

        res = compute_mAP__yolov11(
            pred_boxes=pred_boxes_list,
            pred_classes=pred_classes_list,
            pred_confs=pred_confs_list,
            gt_coords=gt_coords_list,
            iou_thresholds=map_iou_thresholds,
            compute_micro_average=True,
        )
        class_aps = res['class_aps']
        micro_aps = res['micro_aps']
        
        for iou_thresh, map_ in zip(map_iou_thresholds, class_aps.mean(axis=1)):
            print_magenta(f'mAP@{iou_thresh}: {map_}', bold=True)

        for iou_thresh, ap in zip(map_iou_thresholds, micro_aps):
            print_magenta(f'micro_AP@{iou_thresh}: {ap}', bold=True)
            
        print_blue('Saving metrics to file ...', bold=True)
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        strings = [
            'bbox_regression',
            f'{len(vinbig_trainer.val_dataset)}',
        ]
        if optimize_thresholds:
            strings.append(f'opt_thr({best_iou_threshold:.2f},{best_conf_threshold:.2f},{max_det})')
        save_path = os.path.join(results_folder_path, f'vindrcxr_metrics({",".join(strings)}).pkl')
        output = dict(
            image_paths=image_paths,
            pred_boxes_list=pred_boxes_list,
            pred_classes_list=pred_classes_list,
            pred_confs_list=pred_confs_list,
            gt_bboxes=gt_coords_list,
            class_ious=class_ious,
            sample_ious=sample_ious,
            micro_iou=micro_iou,
            macro_iou=mean_iou,
            map_iou_thresholds=map_iou_thresholds,
            class_aps=class_aps,
            micro_aps=micro_aps,
        )
        if optimize_thresholds:
            output['best_iou_threshold'] = best_iou_threshold
            output['best_conf_threshold'] = best_conf_threshold
            output['max_det'] = max_det
        
        save_pickle(output, save_path)
        print(f'Saved metrics to {save_path}')

def evaluate(
    checkpoint_folder_path,
    num_workers,
    batch_size,
    eval_vinbig,
    vinbig_use_training_indices_for_validation,
    device,
    optimize_thresholds,
    candidate_iou_thresholds,
    candidate_conf_thresholds,
    map_iou_thresholds,
    max_det,
    use_amp,
):
    print_blue('----- Evaluating model -----', bold=True)

    metadata = load_metadata(checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    vinbig_trainer_kwargs = metadata['vinbig_trainer_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    val_image_transform_kwargs = metadata['val_image_transform_kwargs']
    validator_engine_kwargs = metadata['validator_engine_kwargs']

    return _evaluate_model(
                checkpoint_folder_path=checkpoint_folder_path,
                model_kwargs=model_kwargs,
                vinbig_trainer_kwargs=vinbig_trainer_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                evaluation_engine_kwargs=validator_engine_kwargs,
                batch_size=batch_size,
                num_workers=num_workers,
                eval_vinbig=eval_vinbig,
                vinbig_use_training_indices_for_validation=vinbig_use_training_indices_for_validation,
                device=device,
                optimize_thresholds=optimize_thresholds,
                candidate_iou_thresholds=candidate_iou_thresholds,
                candidate_conf_thresholds=candidate_conf_thresholds,
                map_iou_thresholds=map_iou_thresholds,
                max_det=max_det,
                use_amp=use_amp,
            )


if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)