import  os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    get_anaxnet_bbox_coords_and_presence_sorted_indices,
    get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_dicom_ids,
    load_chest_imagenome_gold_bboxes,
    load_chest_imagenome_silver_bboxes,
    get_chest_imagenome_train_average_bbox_coords,
)
from medvqa.datasets.dataloading_utils import get_vision_collate_batch_fn, simple_yolov8_collate_batch_fn
from medvqa.datasets.image_processing import get_image_transform, ImageDataset
from medvqa.datasets.mimiccxr import (
    # get_mimiccxr_small_image_path,
    get_mimiccxr_medium_image_path,
    load_mimiccxr_reports_detailed_metadata,
    get_imageId2PartPatientStudy,
)
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleTrainer
from medvqa.metrics.bbox.utils import (
    compute_mae_per_class,
    compute_mae_per_class__detectron2,
    compute_mae_per_class__yolov8,
    compute_mean_iou_per_class,
    compute_mean_iou_per_class__detectron2,
    compute_mean_iou_per_class__yolov8,
    compute_multiple_prf1_scores,
    compute_multiple_prf1_scores__detectron2,
    compute_multiple_prf1_scores__yolov8,
)
from medvqa.models.checkpoint import get_checkpoint_filepath, get_model_name_from_checkpoint_path, load_metadata
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule
from medvqa.models.vqa.open_ended_vqa import OpenEndedVQA
from medvqa.training.vision import get_engine
from medvqa.utils.common import (
    WORKSPACE_DIR,
    parsed_args_to_dict,
)
from medvqa.utils.constants import DATASET_NAMES, MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE
from medvqa.utils.files_utils import (
    get_checkpoint_folder_path,
    get_results_folder_path,
    save_to_pickle,
)
from medvqa.utils.handlers_utils import attach_accumulator

from medvqa.utils.logging_utils import print_blue

class EvalMode:
    CHEST_IMAGENOME__AVERAGE_BBOX = 'chest_imagenome_average_bbox'
    CHEST_IMAGENOME__TRAINED_MODEL = 'chest_imagenome_trained_model'

class EvalDatasets:
    MIMICCXR_TEST_SET = 'mimiccxr_test_set'
    CHEST_IMAGENOME_GOLD = 'chest_imagenome_gold'

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--eval-mode', type=str, required=True)
    parser.add_argument('--eval-dataset-name', type=str, required=True)

    # optional arguments
    parser.add_argument('--checkpoint-folder', type=str)
    parser.add_argument('--batch-size', type=int, default=140)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--clamp-bbox-coords', action='store_true')
    parser.set_defaults(clamp_bbox_coords=False)
    parser.add_argument('--decent-images-only', action='store_true')
    parser.set_defaults(decent_images_only=False)
    parser.add_argument('--save-predictions', action='store_true')
    parser.set_defaults(save_predictions=False)

    
    return parser.parse_args()

def _compute_and_save_bbox_metrics(test_bbox_coords, test_bbox_presences, bbox_names, eval_mode,
                                    eval_dataset_name, results_folder_path, clamp_bbox_coords, decent_images_only,
                                    use_detectron2=False, use_yolov8=False, pred_boxes=None, pred_classes=None,
                                    scores=None, valid_classes=None, pred_bbox_coords=None, pred_bbox_presences=None,
                                    save_predictions=False, dicom_ids=None):

    assert sum([use_detectron2, use_yolov8]) <= 1

    if use_detectron2:
        assert pred_boxes is not None
        assert pred_classes is not None
        assert scores is not None
        assert len(pred_boxes) == len(pred_classes) == len(scores)
        assert len(pred_boxes) == len(test_bbox_coords)
        assert len(pred_boxes) == len(test_bbox_presences)
    elif use_yolov8:
        assert pred_boxes is not None
        assert pred_classes is not None
        assert len(pred_boxes) == len(pred_classes)
        assert len(pred_boxes) == len(test_bbox_coords)
        assert len(pred_boxes) == len(test_bbox_presences)
    else:
        assert pred_bbox_coords is not None
        assert pred_bbox_presences is not None
        assert len(test_bbox_coords) == len(test_bbox_presences)
        assert len(pred_bbox_coords) == len(pred_bbox_presences)
        assert len(test_bbox_coords) == len(pred_bbox_coords)    

    metrics = {}
    
    # Mean Absolute Error
    print('Computing Mean Absolute Error (MAE) ...')
    if use_detectron2:
        metrics['mae'] = compute_mae_per_class__detectron2(pred_boxes, pred_classes, scores, test_bbox_coords, test_bbox_presences, valid_classes)
    elif use_yolov8:
        metrics['mae'] = compute_mae_per_class__yolov8(pred_boxes, pred_classes, test_bbox_coords, test_bbox_presences, valid_classes)
    else:
        metrics['mae'] = compute_mae_per_class(pred_bbox_coords, test_bbox_coords, test_bbox_presences)
    metrics['mean_mae'] = np.mean(metrics['mae'])
    
    # Mean Intersection Over Union (IOU)
    print('Computing Mean Intersection Over Union (IOU) ...')
    if use_detectron2:
        metrics['iou'] = compute_mean_iou_per_class__detectron2(pred_boxes, pred_classes, scores, test_bbox_coords, test_bbox_presences, valid_classes)
    elif use_yolov8:
        metrics['iou'] = compute_mean_iou_per_class__yolov8(pred_boxes, pred_classes, test_bbox_coords, test_bbox_presences, valid_classes)
    else:
        metrics['iou'] = compute_mean_iou_per_class(pred_bbox_coords, test_bbox_coords, test_bbox_presences)
    metrics['mean_iou'] = np.mean(metrics['iou'])
    
    # Precision, Recall, and F1 Score
    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f'Computing Precision, Recall, and F1 Score at IOU thresholds {iou_thresholds} ...')
    if use_detectron2:
        scores = compute_multiple_prf1_scores__detectron2(
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            scores=scores,
            gt_coords=test_bbox_coords,
            gt_presences=test_bbox_presences,
            iou_thresholds=iou_thresholds,
            valid_classes=valid_classes,
            num_workers=7,
        )
    elif use_yolov8:
        scores = compute_multiple_prf1_scores__yolov8(
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            gt_coords=test_bbox_coords,
            gt_presences=test_bbox_presences,
            iou_thresholds=iou_thresholds,
            valid_classes=valid_classes,
            num_workers=7,
        )
    else:
        scores = compute_multiple_prf1_scores(
            pred_coords=pred_bbox_coords,
            pred_presences=pred_bbox_presences,
            gt_coords=test_bbox_coords,
            gt_presences=test_bbox_presences,
            iou_thresholds=iou_thresholds,
            num_workers=5,
        )
    assert scores.shape == (len(iou_thresholds), len(bbox_names), 3)
    mean_p = 0
    mean_r = 0
    mean_f1 = 0
    for i, iou_thrs in enumerate(iou_thresholds):
        metrics[f'p@{iou_thrs}'] = scores[i, :, 0]
        metrics[f'r@{iou_thrs}'] = scores[i, :, 1]
        metrics[f'f1@{iou_thrs}'] = scores[i, :, 2]
        metrics[f'mean_p@{iou_thrs}'] = metrics[f'p@{iou_thrs}'].mean()
        metrics[f'mean_r@{iou_thrs}'] = metrics[f'r@{iou_thrs}'].mean()
        metrics[f'mean_f1@{iou_thrs}'] = metrics[f'f1@{iou_thrs}'].mean()
        mean_p += metrics[f'mean_p@{iou_thrs}']
        mean_r += metrics[f'mean_r@{iou_thrs}']
        mean_f1 += metrics[f'mean_f1@{iou_thrs}']
    mean_p /= len(iou_thresholds)
    mean_r /= len(iou_thresholds)
    mean_f1 /= len(iou_thresholds)
    metrics['mean_p'] = mean_p
    metrics['mean_r'] = mean_r
    metrics['mean_f1'] = mean_f1
    metrics['bbox_names'] = bbox_names
    
    # Save metrics    
    save_path = os.path.join(results_folder_path,
        (f'{eval_dataset_name}__bbox_metrics(eval_mode={eval_mode}'
         f'{",clamped" if clamp_bbox_coords else ""}{",decent" if decent_images_only else ""}).pkl'))
    save_to_pickle(metrics, save_path)
    print(f'Saved bbox metrics to {save_path}')
    
    # Save predictions
    if save_predictions:
        assert dicom_ids is not None
        save_path = os.path.join(results_folder_path,
            (f'{eval_dataset_name}__bbox_predictions(eval_mode={eval_mode}'
             f'{",clamped" if clamp_bbox_coords else ""}{",decent" if decent_images_only else ""}).pkl'))
        save_to_pickle({
            'dicom_ids': dicom_ids,
            'pred_bbox_coords': pred_bbox_coords,
            'pred_bbox_presences': pred_bbox_presences,
            'test_bbox_coords': test_bbox_coords,
            'test_bbox_presences': test_bbox_presences,
            'bbox_names': bbox_names,
        }, save_path)
        print(f'Saved bbox predictions to {save_path}')

def _evaluate_model(
    eval_mode,
    eval_dataset_name,
    batch_size=None,
    num_workers=None,
    image_transform_kwargs=None,
    collate_batch_fn_kwargs=None,
    mimiccxr_trainer_kwargs=None,
    validator_engine_kwargs=None,
    model_kwargs=None,
    checkpoint_folder_path=None,
    clamp_bbox_coords=False,
    decent_images_only=False,
    save_predictions=False,
):
    # Check if we need to filter out labels
    use_anaxnet_bbox_subset = (model_kwargs is not None and model_kwargs.get('use_anaxnet_bbox_subset', False))
    use_gold_bbox_subset = eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD
    filter_bbox_classes = False
    if use_anaxnet_bbox_subset:
        filter_bbox_classes = True
        gt_coord_indices, gt_presence_indices = get_anaxnet_bbox_coords_and_presence_sorted_indices(use_gold_bbox_subset)
        model_coord_indices, model_presence_indices = get_anaxnet_bbox_coords_and_presence_sorted_indices(
                use_gold_bbox_subset, for_model_output=True)
    elif use_gold_bbox_subset:
        filter_bbox_classes = True
        gt_coord_indices, gt_presence_indices = get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices()
        model_coord_indices = gt_coord_indices
        model_presence_indices = gt_presence_indices
    if filter_bbox_classes:
        bbox_names = [CHEST_IMAGENOME_BBOX_NAMES[i] for i in gt_presence_indices]
    else:
        bbox_names = CHEST_IMAGENOME_BBOX_NAMES
    print(f'len(bbox_names) = {len(bbox_names)}')

    if decent_images_only:
        decent_dicom_ids = set(load_chest_imagenome_dicom_ids(decent_images_only=True))

    if eval_mode == EvalMode.CHEST_IMAGENOME__AVERAGE_BBOX:
                
        # Compute the average bbox for each class from the training set
        avg_bbox_coords = get_chest_imagenome_train_average_bbox_coords(
            clamp_bbox_coords=clamp_bbox_coords, use_decent_images_only=decent_images_only)

        # Collect the test set bbox coords and presences
        if save_predictions:
            dicom_ids = []
        if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
            bboxes_dict = load_chest_imagenome_silver_bboxes()
            mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata()
            test_idxs = [i for i, split in enumerate(mimiccxr_detailed_metadata['splits']) if split == 'test']
            test_bbox_coords = []
            test_bbox_presences = []
            for idx in test_idxs:
                dicom_id_view_pairs = mimiccxr_detailed_metadata['dicom_id_view_pos_pairs'][idx]
                for dicom_id, _ in dicom_id_view_pairs:
                    if decent_images_only and dicom_id not in decent_dicom_ids:
                        continue
                    if dicom_id in bboxes_dict:
                        bbox = bboxes_dict[dicom_id]
                        test_bbox_coords.append(bbox['coords'])
                        test_bbox_presences.append(bbox['presence'])
                        if save_predictions:
                            dicom_ids.append(dicom_id)
        elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
            test_bbox_coords = []
            test_bbox_presences = []
            gold_bboxes = load_chest_imagenome_gold_bboxes()
            for dicom_id, bbox in gold_bboxes.items():
                if decent_images_only and dicom_id not in decent_dicom_ids:
                    continue
                test_bbox_coords.append(bbox['coords'])
                test_bbox_presences.append(bbox['presence'])
                if save_predictions:
                    dicom_ids.append(dicom_id)
        else:
            raise ValueError(f'Invalid eval_dataset_name: {eval_dataset_name}')
        
        # Convert to numpy arrays
        test_bbox_coords = np.array(test_bbox_coords)
        test_bbox_presences = np.array(test_bbox_presences)

        # Clamp the bbox coords to [0, 1]
        if clamp_bbox_coords:
            test_bbox_coords.clip(0, 1, out=test_bbox_coords)

        # Filter out bboxes if required
        if filter_bbox_classes:
            test_bbox_coords = test_bbox_coords[:, gt_coord_indices]
            test_bbox_presences = test_bbox_presences[:, gt_presence_indices]

        # Prepare predictions
        pred_bbox_coords = np.tile(avg_bbox_coords, (len(test_bbox_coords), 1))
        pred_bbox_presences = np.ones((len(test_bbox_coords), CHEST_IMAGENOME_NUM_BBOX_CLASSES))
        if filter_bbox_classes:            
            pred_bbox_coords = pred_bbox_coords[:, model_coord_indices]
            pred_bbox_presences = pred_bbox_presences[:, model_presence_indices]

        assert pred_bbox_coords.shape == test_bbox_coords.shape
        assert pred_bbox_presences.shape == test_bbox_presences.shape
        
        print(f'pred_bbox_coords.shape: {pred_bbox_coords.shape}')
        print(f'pred_bbox_presences.shape: {pred_bbox_presences.shape}')
        print(f'test_bbox_coords.shape: {test_bbox_coords.shape}')
        print(f'test_bbox_presences.shape: {test_bbox_presences.shape}')
        
        results_folder_path = get_results_folder_path(get_checkpoint_folder_path('bbox', 'chest-imagenome', 'average'))
        _compute_and_save_bbox_metrics(
            test_bbox_coords=test_bbox_coords,
            test_bbox_presences=test_bbox_presences,
            pred_bbox_coords=pred_bbox_coords,
            pred_bbox_presences=pred_bbox_presences,
            bbox_names=bbox_names,
            eval_mode=eval_mode,
            eval_dataset_name=eval_dataset_name,
            results_folder_path=results_folder_path,
            clamp_bbox_coords=clamp_bbox_coords,
            save_predictions=save_predictions,
            dicom_ids=dicom_ids if save_predictions else None,
        )
    elif eval_mode == EvalMode.CHEST_IMAGENOME__TRAINED_MODEL:
        assert batch_size is not None
        assert num_workers is not None
        assert image_transform_kwargs is not None
        assert collate_batch_fn_kwargs is not None
        assert model_kwargs is not None
        assert checkpoint_folder_path is not None
        assert mimiccxr_trainer_kwargs is not None
        assert validator_engine_kwargs is not None
        assert mimiccxr_trainer_kwargs['use_decent_images_only'] == decent_images_only

        use_detectron2 = mimiccxr_trainer_kwargs.get('use_detectron2', False)
        use_yolov8 = mimiccxr_trainer_kwargs.get('use_yolov8', False)

        # Define image transform
        if DATASET_NAMES.MIMICCXR in image_transform_kwargs:
            image_transform = get_image_transform(**image_transform_kwargs[DATASET_NAMES.MIMICCXR])
        else: # for backward compatibility
            image_transform = get_image_transform(**image_transform_kwargs)

        # Define collate_batch_fn
        if use_detectron2:
            collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR_CHEST_IMAGENOME__DETECTRON2_MODE])
        elif use_yolov8:
            collate_batch_fn = simple_yolov8_collate_batch_fn
        else:
            collate_batch_fn = None # use default collate_fn

        # Define test bbox dataset and dataloader
        if use_detectron2:
            if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
                assert mimiccxr_trainer_kwargs['use_test_set'] == True
                mimiccxr_trainer = MIMICCXR_VisualModuleTrainer(
                    test_image_transform = image_transform,
                    batch_size = batch_size,
                    collate_batch_fn = collate_batch_fn,
                    num_workers = num_workers,
                    **mimiccxr_trainer_kwargs,
                )
                test_dataloader = mimiccxr_trainer.test_dataloader
                dicom_ids = []
                test_bbox_coords = []
                test_bbox_presences = []
                for d in mimiccxr_trainer.test_dataset.dataset_dicts:
                    dicom_ids.append(d['image_id'])
                    test_bbox_coords.append(d['coords'])
                    test_bbox_presences.append(d['presence'])
            elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
                assert mimiccxr_trainer_kwargs['use_chest_imagenome_gold_set'] == True
                mimiccxr_trainer = MIMICCXR_VisualModuleTrainer(
                    test_image_transform = image_transform,
                    batch_size = batch_size,
                    collate_batch_fn = collate_batch_fn,
                    num_workers = num_workers,
                    **mimiccxr_trainer_kwargs,
                )
                test_dataloader = mimiccxr_trainer.test_dataloader
                dicom_ids = []
                test_bbox_coords = []
                test_bbox_presences = []
                for d in mimiccxr_trainer.test_dataset.dataset_dicts:
                    dicom_ids.append(d['image_id'])
                    test_bbox_coords.append(d['coords'])
                    test_bbox_presences.append(d['presence'])
            else:
                raise ValueError(f'Invalid eval_dataset_name: {eval_dataset_name}')
        else:
            # Collect the test set image paths, bbox coords and presences
            test_image_paths = []
            test_bbox_coords = []
            test_bbox_presences = []
            if save_predictions:
                dicom_ids = []
            if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
                bboxes_dict = load_chest_imagenome_silver_bboxes()
                mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata()
                test_idxs = [i for i, split in enumerate(mimiccxr_detailed_metadata['splits']) if split == 'test']            
                for idx in test_idxs:
                    dicom_id_view_pairs = mimiccxr_detailed_metadata['dicom_id_view_pos_pairs'][idx]
                    part_id = mimiccxr_detailed_metadata['part_ids'][idx]
                    subject_id = mimiccxr_detailed_metadata['subject_ids'][idx]
                    study_id = mimiccxr_detailed_metadata['study_ids'][idx]
                    for dicom_id, _ in dicom_id_view_pairs:
                        if decent_images_only and dicom_id not in decent_dicom_ids:
                            continue
                        if dicom_id in bboxes_dict:
                            image_path = get_mimiccxr_medium_image_path(part_id, subject_id, study_id, dicom_id)
                            test_image_paths.append(image_path)
                            bbox = bboxes_dict[dicom_id]
                            test_bbox_coords.append(bbox['coords'].reshape(-1, 4))
                            test_bbox_presences.append(bbox['presence'])
                            if save_predictions:
                                dicom_ids.append(dicom_id)
            elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
                gold_bboxes = load_chest_imagenome_gold_bboxes()
                imageId2PartPatientStudy = get_imageId2PartPatientStudy()
                for dicom_id, bbox in gold_bboxes.items():
                    if decent_images_only and dicom_id not in decent_dicom_ids:
                        continue
                    part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
                    image_path = get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id)
                    test_image_paths.append(image_path)
                    test_bbox_coords.append(bbox['coords'].reshape(-1, 4))
                    test_bbox_presences.append(bbox['presence'])
                    if save_predictions:
                        dicom_ids.append(dicom_id)
            else:
                raise ValueError(f'Invalid eval_dataset_name: {eval_dataset_name}')
            test_dataset = ImageDataset(
                image_paths=test_image_paths,
                image_transform=image_transform,
                use_yolov8=use_yolov8,
            )
            test_dataloader = DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        collate_fn=collate_batch_fn,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True)

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')

        # Load saved checkpoint
        checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
        checkpoint = torch.load(checkpoint_path)

        # Create model
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

        # Obtain predictions
        if use_detectron2:
            test_engine = get_engine(model=model, device=device, **validator_engine_kwargs)
            attach_accumulator(test_engine, 'pred_boxes')
            attach_accumulator(test_engine, 'pred_classes')
            attach_accumulator(test_engine, 'scores')
            test_engine.run(test_dataloader)
            pred_boxes = test_engine.state.metrics['pred_boxes']
            pred_classes = test_engine.state.metrics['pred_classes']
            scores = test_engine.state.metrics['scores']
        elif use_yolov8:
            pred_boxes = []
            pred_classes = []
            with torch.no_grad():
                model.eval()
                for batch in tqdm(test_dataloader):
                    images = batch['i'].to(device)
                    resized_shapes = batch['resized_shape']
                    output = model(raw_images=images, mimiccxr_forward=True, skip_mlc=True)
                    batch_predictions = output['yolov8_predictions']
                    assert len(resized_shapes) == len(batch_predictions)
                    for i in range(len(resized_shapes)):
                        resized_shape = resized_shapes[i]
                        pred = batch_predictions[i].detach().cpu()
                        pred[:, :4] /= torch.tensor([resized_shape[1], resized_shape[0], resized_shape[1], resized_shape[0]], dtype=torch.float32)
                        pred_boxes.append(pred[:, :4])
                        pred_classes.append(pred[:, 5].int())
        else:
            pred_bbox_coords = []
            pred_bbox_presences = []
            with torch.no_grad():
                model.eval()
                for batch in tqdm(test_dataloader):
                    images = batch['i'].to(device)
                    output = model(raw_images=images, mimiccxr_forward=True)
                    bbox_coords = output['pred_chest_imagenome_bbox_coords'].detach().cpu().numpy()
                    bbox_presence = output['pred_chest_imagenome_bbox_presence'].detach().cpu().numpy()
                    for i in range(len(bbox_coords)):
                        pred_bbox_coords.append(bbox_coords[i])
                        pred_bbox_presences.append(bbox_presence[i])

        # Convert to numpy arrays
        test_bbox_coords = np.array(test_bbox_coords)
        test_bbox_presences = np.array(test_bbox_presences)
        if not use_detectron2 and not use_yolov8:
            pred_bbox_coords = np.array(pred_bbox_coords)
            pred_bbox_presences = np.array(pred_bbox_presences)

        # Clamp test bbox coords to [0, 1]
        if not use_detectron2 and clamp_bbox_coords:
            test_bbox_coords.clip(0, 1, out=test_bbox_coords)
        
        # Filter out bboxes if required
        if filter_bbox_classes:
            if not use_detectron2 and not use_yolov8:
                test_bbox_coords = test_bbox_coords[:, gt_coord_indices]
                test_bbox_presences = test_bbox_presences[:, gt_presence_indices]
                pred_bbox_coords = pred_bbox_coords[:, model_coord_indices]
                pred_bbox_presences = pred_bbox_presences[:, model_presence_indices]
            valid_classes = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES, dtype=bool)
            valid_classes[gt_presence_indices] = True
            print('valid_classes: ', valid_classes)
            assert len(valid_classes) == CHEST_IMAGENOME_NUM_BBOX_CLASSES
        else:
            valid_classes = None

        if not use_detectron2 and not use_yolov8:
            assert test_bbox_coords.shape == pred_bbox_coords.shape
            assert test_bbox_presences.shape == pred_bbox_presences.shape
            print(f'pred_bbox_coords.shape: {pred_bbox_coords.shape}')
            print(f'pred_bbox_presences.shape: {pred_bbox_presences.shape}')
            print(f'test_bbox_coords.shape: {test_bbox_coords.shape}')
            print(f'test_bbox_presences.shape: {test_bbox_presences.shape}')
        
        # Compute and save metrics
        if use_detectron2:
            _compute_and_save_bbox_metrics(
                use_detectron2=use_detectron2,
                test_bbox_coords=test_bbox_coords,
                test_bbox_presences=test_bbox_presences,
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                scores=scores,
                valid_classes=valid_classes,
                bbox_names=bbox_names,
                eval_mode=eval_mode,
                eval_dataset_name=eval_dataset_name,
                results_folder_path=get_results_folder_path(checkpoint_folder_path),
                clamp_bbox_coords=clamp_bbox_coords,
                decent_images_only=decent_images_only,
                save_predictions=save_predictions,
                dicom_ids=dicom_ids if save_predictions else None,
            )
        elif use_yolov8:
            _compute_and_save_bbox_metrics(
                use_yolov8=use_yolov8,
                test_bbox_coords=test_bbox_coords,
                test_bbox_presences=test_bbox_presences,
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                valid_classes=valid_classes,
                bbox_names=bbox_names,
                eval_mode=eval_mode,
                eval_dataset_name=eval_dataset_name,
                results_folder_path=get_results_folder_path(checkpoint_folder_path),
                clamp_bbox_coords=clamp_bbox_coords,
                decent_images_only=decent_images_only,
                save_predictions=save_predictions,
                dicom_ids=dicom_ids if save_predictions else None,
            )
        else:
            _compute_and_save_bbox_metrics(
                use_detectron2=use_detectron2,
                test_bbox_coords=test_bbox_coords,
                test_bbox_presences=test_bbox_presences,
                pred_bbox_coords=pred_bbox_coords,
                pred_bbox_presences=pred_bbox_presences,
                bbox_names=bbox_names,
                eval_mode=eval_mode,
                eval_dataset_name=eval_dataset_name,
                results_folder_path=get_results_folder_path(checkpoint_folder_path),
                clamp_bbox_coords=clamp_bbox_coords,
                decent_images_only=decent_images_only,
                save_predictions=save_predictions,
                dicom_ids=dicom_ids if save_predictions else None,
            )
    else:
        raise ValueError(f'Invalid eval_mode: {eval_mode}')

def evaluate_model(
    eval_mode,
    eval_dataset_name,
    batch_size=None,
    num_workers=None,
    checkpoint_folder=None,
    clamp_bbox_coords=False,
    decent_images_only=False,
    save_predictions=False,
):
    print()
    print_blue('----- Evaluating model ------')

    if checkpoint_folder is not None:
        checkpoint_folder_path = os.path.join(WORKSPACE_DIR, checkpoint_folder)
        metadata = load_metadata(checkpoint_folder_path)
        image_transform_kwargs = metadata['val_image_transform_kwargs']
        image_transform_kwargs['augmentation_mode'] = None # no data augmentation during evaluation
        model_kwargs = metadata['model_kwargs']
        mimiccxr_trainer_kwargs = None
        for key in [ # to handle different naming conventions and backward compatibility
            'mimiccxr_vqa_trainer_kwargs',
            'mimiccxr_visual_trainer_kwargs',
            'mimiccxr_vision_trainer_kwargs',
            'mimiccxr_trainer_kwargs',
        ]:
            if key in metadata and eval_mode == EvalMode.CHEST_IMAGENOME__TRAINED_MODEL:
                mimiccxr_trainer_kwargs = metadata[key]
                mimiccxr_trainer_kwargs['use_decent_images_only'] = decent_images_only
                # set clamp_bbox_coords to True if the model was trained with clamp_bbox_coords=True
                clamp_bbox_coords = metadata[key]['clamp_bboxes_chest_imagenome']
                print(f'clamp_bbox_coords: {clamp_bbox_coords}')
                if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
                    mimiccxr_trainer_kwargs['use_test_set'] = True
                    mimiccxr_trainer_kwargs['use_chest_imagenome_gold_set'] = False
                elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
                    mimiccxr_trainer_kwargs['use_test_set'] = False
                    mimiccxr_trainer_kwargs['use_chest_imagenome_gold_set'] = True
                break
        try:
            collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
        except KeyError:
            if mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_detectron2', False):
                collate_batch_fn_kwargs = { DATASET_NAMES.MIMICCXR_CHEST_IMAGENOME__DETECTRON2_MODE: 
                    {
                        'dataset_id': MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE,
                        'include_image': True,
                        'predict_bboxes_chest_imagenome': True,
                     }
                }
            else:
                collate_batch_fn_kwargs = None
        validator_engine_kwargs = metadata['validator_engine_kwargs']
    else:
        checkpoint_folder_path = None
        image_transform_kwargs = None
        model_kwargs = None
        collate_batch_fn_kwargs = None
        mimiccxr_trainer_kwargs = None
        validator_engine_kwargs = None

    return _evaluate_model(
        eval_mode=eval_mode,
        eval_dataset_name=eval_dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        image_transform_kwargs=image_transform_kwargs,
        collate_batch_fn_kwargs=collate_batch_fn_kwargs,
        mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
        validator_engine_kwargs=validator_engine_kwargs,
        model_kwargs=model_kwargs,
        checkpoint_folder_path=checkpoint_folder_path,
        clamp_bbox_coords=clamp_bbox_coords,
        decent_images_only=decent_images_only,
        save_predictions=save_predictions,
    )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)