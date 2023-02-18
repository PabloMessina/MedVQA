import  os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_GOLD_BBOX_NAMES,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_gold_bboxes,
    load_chest_imagenome_silver_bboxes,
    load_chest_imagenome_train_average_bbox_coords,
)
from medvqa.datasets.image_processing import get_image_transform, ImageDataset
from medvqa.datasets.mimiccxr import get_mimiccxr_small_image_path, load_mimiccxr_reports_detailed_metadata
from medvqa.datasets.mimiccxr.preprocessing import get_imageId2PartPatientStudy
from medvqa.metrics.bbox.bbox_iou import DatasetAwareBboxIOU
from medvqa.metrics.bbox.bbox_mae import DatasetAwareBboxMAE
from medvqa.metrics.bbox.bbox_mean_prf1 import (
    DatasetAwareBboxMeanF1,
    DatasetAwareBboxMeanPrecision,
    DatasetAwareBboxMeanRecall,
)
from medvqa.models.checkpoint import get_checkpoint_filepath, get_model_name_from_checkpoint_path, load_metadata
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule
from medvqa.models.vqa.open_ended_vqa import OpenEndedVQA
from medvqa.utils.common import (
    WORKSPACE_DIR,
    parsed_args_to_dict,
)
from medvqa.utils.constants import DATASET_NAMES
from medvqa.utils.files import (
    get_checkpoint_folder_path,
    get_results_folder_path,
    save_to_pickle,
)

from medvqa.utils.logging import print_blue

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
    parser.add_argument('--mimiccxr-qa-adapted-reports-filename', type=str)
    parser.add_argument('--checkpoint-folder', type=str)
    parser.add_argument('--batch-size', type=int, default=140)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--clamp-bbox-coords', action='store_true')
    parser.set_defaults(clamp_bbox_coords=False)
    parser.add_argument('--save-predictions', action='store_true')
    parser.set_defaults(save_predictions=False)

    
    return parser.parse_args()

def _compute_and_save_bbox_metrics(test_bbox_coords, test_bbox_presences, pred_bbox_coords, pred_bbox_presences,
                                    n_classes, eval_mode, eval_dataset_name, results_folder_path, clamp_bbox_coords,
                                    save_predictions=False, dicom_ids=None):

    assert len(test_bbox_coords) == len(test_bbox_presences)
    assert len(pred_bbox_coords) == len(pred_bbox_presences)
    assert len(test_bbox_coords) == len(pred_bbox_coords)    

    metrics = {}
    # Mean Absolute Error
    print('Computing Mean Absolute Error (MAE) ...')
    mae = DatasetAwareBboxMAE(lambda x:x, None)
    mae.update((pred_bbox_coords, test_bbox_coords))
    metrics['mae'] = mae.compute()
    # Mean Intersection Over Union (IOU)
    print('Computing Mean Intersection Over Union (IOU) ...')
    iou = DatasetAwareBboxIOU(lambda x:x, None)
    iou.update((pred_bbox_coords, test_bbox_coords))
    metrics['iou'] = iou.compute()
    # Mean F1 Score
    print('Computing Mean F1 Score ...')
    count = 0
    for iou_thrs in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f'   Computing F1 Score at IOU threshold {iou_thrs} ...')
        meanf1 = DatasetAwareBboxMeanF1(lambda x:x, None, n_classes, [iou_thrs])
        meanf1.update((pred_bbox_coords, test_bbox_coords, pred_bbox_presences, test_bbox_presences))
        metrics[f'f1@{iou_thrs}'] = meanf1.compute()
        metrics['meanf1'] = metrics.get('meanf1', 0) + metrics[f'f1@{iou_thrs}']
        count += 1
    metrics['meanf1'] /= count
    # Mean Precision
    print('Computing Mean Precision ...')
    count = 0
    for iou_thrs in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f'   Computing Precision at IOU threshold {iou_thrs} ...')
        meanp = DatasetAwareBboxMeanPrecision(lambda x:x, None, n_classes, [iou_thrs])
        meanp.update((pred_bbox_coords, test_bbox_coords, pred_bbox_presences, test_bbox_presences))
        metrics[f'p@{iou_thrs}'] = meanp.compute()
        metrics['meanp'] = metrics.get('meanp', 0) + metrics[f'p@{iou_thrs}']
        count += 1
    metrics['meanp'] /= count
    # Mean Recall
    print('Computing Mean Recall ...')
    count = 0
    for iou_thrs in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f'   Computing Recall at IOU threshold {iou_thrs} ...')
        meanr = DatasetAwareBboxMeanRecall(lambda x:x, None, n_classes, [iou_thrs])
        meanr.update((pred_bbox_coords, test_bbox_coords, pred_bbox_presences, test_bbox_presences))
        metrics[f'r@{iou_thrs}'] = meanr.compute()
        metrics['meanr'] = metrics.get('meanr', 0) + metrics[f'r@{iou_thrs}']
        count += 1
    metrics['meanr'] /= count
    # Save metrics    
    save_path = os.path.join(results_folder_path,
        f'{eval_dataset_name}__bbox_metrics(eval_mode={eval_mode}{",clamped" if clamp_bbox_coords else ""}).pkl')
    save_to_pickle(metrics, save_path)
    print(f'Saved bbox metrics to {save_path}')
    # Save predictions
    if save_predictions:
        assert dicom_ids is not None
        save_path = os.path.join(results_folder_path,
            f'{eval_dataset_name}__bbox_predictions(eval_mode={eval_mode}{",clamped" if clamp_bbox_coords else ""}).pkl')
        save_to_pickle({
            'dicom_ids': dicom_ids,
            'pred_bbox_coords': pred_bbox_coords,
            'pred_bbox_presences': pred_bbox_presences,
            'test_bbox_coords': test_bbox_coords,
            'test_bbox_presences': test_bbox_presences,            
        }, save_path)
        print(f'Saved bbox predictions to {save_path}')

def _evaluate_model(
    eval_mode,
    eval_dataset_name,
    batch_size=None,
    num_workers=None,
    image_transform_kwargs=None,
    model_kwargs=None,
    checkpoint_folder_path=None,
    mimiccxr_qa_adapted_reports_filename=None,
    clamp_bbox_coords=False,
    save_predictions=False,
):

    if eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
        coord_indices = []
        presence_indices = []
        for i, bbox_name in enumerate(CHEST_IMAGENOME_BBOX_NAMES):
            if bbox_name in CHEST_IMAGENOME_GOLD_BBOX_NAMES:
                for j in range(4):
                    coord_indices.append(i*4 + j)
                presence_indices.append(i)

    if eval_mode == EvalMode.CHEST_IMAGENOME__AVERAGE_BBOX:
        assert mimiccxr_qa_adapted_reports_filename is not None
                
        # Compute the average bbox for each class from the training set
        avg_bbox_coords = load_chest_imagenome_train_average_bbox_coords(mimiccxr_qa_adapted_reports_filename, clamp_bbox_coords)

        # Collect the test set bbox coords and presences
        if save_predictions:
            dicom_ids = []
        if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
            bboxes_dict = load_chest_imagenome_silver_bboxes()
            mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata(mimiccxr_qa_adapted_reports_filename)
            test_idxs = [i for i, split in enumerate(mimiccxr_detailed_metadata['splits']) if split == 'test']
            test_bbox_coords = []
            test_bbox_presences = []
            for idx in test_idxs:
                dicom_id_view_pairs = mimiccxr_detailed_metadata['dicom_id_view_pos_pairs'][idx]
                for dicom_id, _ in dicom_id_view_pairs:
                    if dicom_id in bboxes_dict:
                        bbox = bboxes_dict[dicom_id]
                        test_bbox_coords.append(bbox['coords'])
                        test_bbox_presences.append(bbox['presence'])
                        if save_predictions:
                            dicom_ids.append(dicom_id)
            n_bbox_classes = CHEST_IMAGENOME_NUM_BBOX_CLASSES
        elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
            test_bbox_coords = []
            test_bbox_presences = []
            gold_bboxes = load_chest_imagenome_gold_bboxes()
            for dicom_id, bbox in gold_bboxes.items():
                test_bbox_coords.append(bbox['coords'][coord_indices])
                test_bbox_presences.append(bbox['presence'][presence_indices])
                if save_predictions:
                    dicom_ids.append(dicom_id)
            n_bbox_classes = CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES
        else:
            raise ValueError(f'Invalid eval_dataset_name: {eval_dataset_name}')

        # Clamp the bbox coords to [0, 1]
        if clamp_bbox_coords:
            for coords in test_bbox_coords:
                coords.clip(0, 1, out=coords)

        # Prepare predictions
        pred_bbox_coords = np.tile(avg_bbox_coords, (len(test_bbox_coords), 1))
        pred_bbox_presences = np.ones((len(test_bbox_coords), CHEST_IMAGENOME_NUM_BBOX_CLASSES))

        if eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
            # Filter out the bboxes that are not in the gold set
            pred_bbox_coords = pred_bbox_coords[:, coord_indices]
            pred_bbox_presences = pred_bbox_presences[:, presence_indices]

        print(f'pred_bbox_coords.shape: {pred_bbox_coords.shape}')
        print(f'pred_bbox_presences.shape: {pred_bbox_presences.shape}')
        print(f'test_bbox_coords[0].shape: {test_bbox_coords[0].shape}')
        print(f'test_bbox_presences[0].shape: {test_bbox_presences[0].shape}')
        
        results_folder_path = get_results_folder_path(get_checkpoint_folder_path('bbox', 'chest-imagenome', 'average'))
        _compute_and_save_bbox_metrics(
            test_bbox_coords=test_bbox_coords,
            test_bbox_presences=test_bbox_presences,
            pred_bbox_coords=pred_bbox_coords,
            pred_bbox_presences=pred_bbox_presences,
            n_classes=n_bbox_classes,
            eval_mode=eval_mode,
            eval_dataset_name=eval_dataset_name,
            results_folder_path=results_folder_path,
            clamp_bbox_coords=clamp_bbox_coords,
            save_predictions=save_predictions,
            dicom_ids=dicom_ids if save_predictions else None,
        )
    elif eval_mode == EvalMode.CHEST_IMAGENOME__TRAINED_MODEL:
        assert mimiccxr_qa_adapted_reports_filename is not None
        assert batch_size is not None
        assert num_workers is not None
        assert image_transform_kwargs is not None
        assert model_kwargs is not None
        assert checkpoint_folder_path is not None
        
        # Collect the test set image paths, bbox coords and presences
        test_image_paths = []
        test_bbox_coords = []
        test_bbox_presences = []
        if save_predictions:
                dicom_ids = []
        if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
            bboxes_dict = load_chest_imagenome_silver_bboxes()
            mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata(mimiccxr_qa_adapted_reports_filename)
            test_idxs = [i for i, split in enumerate(mimiccxr_detailed_metadata['splits']) if split == 'test']            
            for idx in test_idxs:
                dicom_id_view_pairs = mimiccxr_detailed_metadata['dicom_id_view_pos_pairs'][idx]
                part_id = mimiccxr_detailed_metadata['part_ids'][idx]
                subject_id = mimiccxr_detailed_metadata['subject_ids'][idx]
                study_id = mimiccxr_detailed_metadata['study_ids'][idx]
                for dicom_id, _ in dicom_id_view_pairs:
                    if dicom_id in bboxes_dict:
                        image_path = get_mimiccxr_small_image_path(part_id, subject_id, study_id, dicom_id)
                        test_image_paths.append(image_path)
                        bbox = bboxes_dict[dicom_id]
                        bbox_coords = bbox['coords']                        
                        test_bbox_coords.append(bbox_coords)
                        test_bbox_presences.append(bbox['presence'])
                        if save_predictions:
                            dicom_ids.append(dicom_id)
            n_bbox_classes = CHEST_IMAGENOME_NUM_BBOX_CLASSES
        elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
            gold_bboxes = load_chest_imagenome_gold_bboxes()
            imageId2PartPatientStudy = get_imageId2PartPatientStudy()
            for dicom_id, bbox in gold_bboxes.items():
                part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
                image_path = get_mimiccxr_small_image_path(part_id, patient_id, study_id, dicom_id)
                test_image_paths.append(image_path)
                test_bbox_coords.append(bbox['coords'][coord_indices])
                test_bbox_presences.append(bbox['presence'][presence_indices])
                if save_predictions:
                    dicom_ids.append(dicom_id)
            n_bbox_classes = CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES
        else:
            raise ValueError(f'Invalid eval_dataset_name: {eval_dataset_name}')

        # Clamp bbox coords to [0, 1]
        if clamp_bbox_coords:
            for coords in test_bbox_coords:
                coords.clip(0, 1, out=coords)

        # Define image transform
        if DATASET_NAMES.MIMICCXR in image_transform_kwargs:
            image_transform = get_image_transform(**image_transform_kwargs[DATASET_NAMES.MIMICCXR])
        else: # for backward compatibility
            image_transform = get_image_transform(**image_transform_kwargs)

        # Define test bbox dataset and dataloader
        test_dataset = ImageDataset(
            image_paths=test_image_paths,
            image_transform=image_transform,
        )
        test_dataloader = DataLoader(test_dataset,
                                    batch_size=batch_size,
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
            model = OpenEndedVQA(**model_kwargs, device=device, use_image_encoder_only=True,
                                vocab_size=None, start_idx=None)
        elif model_name == 'visual_module':
            model = MultiPurposeVisualModule(**model_kwargs)
        else:
            raise ValueError(f'Invalid model_name: {model_name}')
        model = model.to(device)
        model.load_state_dict(checkpoint['model'], strict=False)

        # Obtain predictions
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

        # Conver to numpy arrays
        test_bbox_coords = np.array(test_bbox_coords)
        test_bbox_presences = np.array(test_bbox_presences)
        pred_bbox_coords = np.array(pred_bbox_coords)
        pred_bbox_presences = np.array(pred_bbox_presences)

        if eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
            # Filter out the bboxes that are not in the gold set
            pred_bbox_coords = pred_bbox_coords[:, coord_indices]
            pred_bbox_presences = pred_bbox_presences[:, presence_indices]
        
        # Compute and save metrics
        _compute_and_save_bbox_metrics(
            test_bbox_coords=test_bbox_coords,
            test_bbox_presences=test_bbox_presences,
            pred_bbox_coords=pred_bbox_coords,
            pred_bbox_presences=pred_bbox_presences,
            n_classes=n_bbox_classes,
            eval_mode=eval_mode,
            eval_dataset_name=eval_dataset_name,
            results_folder_path=get_results_folder_path(checkpoint_folder_path),
            clamp_bbox_coords=clamp_bbox_coords,
            save_predictions=save_predictions,
            dicom_ids=dicom_ids if save_predictions else None,
        )
    else:
        raise ValueError(f'Invalid eval_mode: {eval_mode}')

def evaluate_model(
    eval_mode,
    eval_dataset_name,
    mimiccxr_qa_adapted_reports_filename=None,
    batch_size=None,
    num_workers=None,
    checkpoint_folder=None,
    clamp_bbox_coords=False,
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
        for key in [ # to handle different naming conventions and backward compatibility
            'mimiccxr_vqa_trainer_kwargs',
            'mimiccxr_visual_trainer_kwargs',
            'mimiccxr_vision_trainer_kwargs',
            'mimiccxr_trainer_kwargs',
        ]:
            if key in metadata and eval_mode == EvalMode.CHEST_IMAGENOME__TRAINED_MODEL:
                # set clamp_bbox_coords to True if the model was trained with clamp_bbox_coords=True
                clamp_bbox_coords = metadata[key]['clamp_bboxes_chest_imagenome']
                print(f'clamp_bbox_coords: {clamp_bbox_coords}')
                break
    else:
        checkpoint_folder_path = None
        image_transform_kwargs = None
        model_kwargs = None

    return _evaluate_model(
        eval_mode=eval_mode,
        eval_dataset_name=eval_dataset_name,
        mimiccxr_qa_adapted_reports_filename=mimiccxr_qa_adapted_reports_filename,
        batch_size=batch_size,
        num_workers=num_workers,
        image_transform_kwargs=image_transform_kwargs,
        model_kwargs=model_kwargs,
        checkpoint_folder_path=checkpoint_folder_path,
        clamp_bbox_coords=clamp_bbox_coords,
        save_predictions=save_predictions,
    )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)