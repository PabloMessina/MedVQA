import argparse
import io
import math
import os
import random
from collections import defaultdict
from pprint import pprint
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from medvqa.datasets.chestxdet.chestxdet_phrase_grounding_dataset_management import (
    ChestXDetInferenceDataset,
)
from medvqa.datasets.image_transforms_factory import create_image_transforms
from medvqa.datasets.mimiccxr import MIMICCXR_ImageSizeModes
from medvqa.datasets.mimiccxr.mimiccxr_phrase_grounding_dataset_management import (
    MIMICCXR_PhraseGroundingTrainer,
)
from medvqa.datasets.ms_cxr import (
    get_ms_cxr_category_names,
    get_ms_cxr_phrase_to_category_name,
)
from medvqa.datasets.padchest.padchest_dataset_management import PadChestGRPhraseTrainer
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBigPhraseTrainer
from medvqa.evaluation.bootstrapping import apply_bootstrapping
from medvqa.metrics.bbox.utils import (
    compute_iou_with_nms,
    compute_probability_map_iou,
    find_optimal_conf_iou_max_det_thresholds__single_class,
    find_optimal_probability_map_conf_threshold,
)
from medvqa.metrics.classification.prc_auc import prc_auc_score
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.models.phrase_grounding.phrase_grounder import PhraseGrounder
from medvqa.utils.bbox_utils import (
    convert_bboxes_into_presence_map,
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
)
# from medvqa.utils.common import activate_determinism
from medvqa.utils.constants import DATASET_NAMES
from medvqa.utils.files_utils import load_config_yaml, load_json, load_pickle, save_pickle
from medvqa.utils.logging_utils import (
    CountPrinter,
    print_blue,
    print_bold,
    print_orange,
    setup_logging,
)
from medvqa.utils.metrics_utils import (
    calculate_cnr,
    calculate_dice,
    calculate_rand_index,
    calculate_segmentation_iou,
    calculate_segmentation_precision,
    calculate_segmentation_recall,
    calculate_soft_dice,
    find_optimal_probability_map_conf_threshold_for_masks,
)

setup_logging()


def run_inference_and_save_predictions_on_mscxr(
    model: PhraseGrounder,
    checkpoint_folder_path: str,
    val_image_transform_kwargs: dict,
    max_images_per_batch: int,
    max_phrases_per_batch: int,
    max_phrases_per_image: int,
    num_workers: int,
    mscxr_phrase2embedding_filepath: str,
    mimicxr_dicom_id_to_pos_neg_facts_filepath: str,
    device: torch.device,
    override_bbox_format: str = None,
):
    # Get image transform kwargs
    try: 
        image_transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]
    except KeyError:
        image_transform_kwargs = next(iter(val_image_transform_kwargs.values())) # get the first value

    # Initialize trainer
    mimiccxr_trainer = MIMICCXR_PhraseGroundingTrainer(
        use_mscxr_for_test=True,
        test_image_transform = create_image_transforms(**image_transform_kwargs),
        max_images_per_batch=max_images_per_batch,
        max_phrases_per_batch=max_phrases_per_batch,
        max_phrases_per_image=max_phrases_per_image,
        num_test_workers=num_workers,
        mscxr_do_grounding_only=False,
        mscxr_test_on_all_images=True,
        bbox_format=override_bbox_format,
        source_image_size_mode=MIMICCXR_ImageSizeModes.MEDIUM_512,
        mscxr_phrase2embedding_filepath=mscxr_phrase2embedding_filepath,
        dicom_id_to_pos_neg_facts_filepath=mimicxr_dicom_id_to_pos_neg_facts_filepath,
    )

    # Get dataset and dataloader
    dataset = mimiccxr_trainer.mscxr_test_dataset
    dataloader = mimiccxr_trainer.mscxr_test_dataloader

    # Aux variables
    train_preds_and_gt = {
        'pred_bbox_prob_maps': [],
        'pred_bbox_coord_maps': [],
        'pred_classification_probs': [],
        'gt_bbox_coords': [],
        'phrases': [],
        'categories': [],
        'image_paths': [],
    }
    val_preds_and_gt = {
        'pred_bbox_prob_maps': [],
        'pred_bbox_coord_maps': [],
        'pred_classification_probs': [],
        'gt_bbox_coords': [],
        'phrases': [],
        'categories': [],
        'image_paths': [],
    }
    test_preds_and_gt = {
        'pred_bbox_prob_maps': [],
        'pred_bbox_coord_maps': [],
        'pred_classification_probs': [],
        'gt_bbox_coords': [],
        'phrases': [],
        'categories': [],
        'image_paths': [],
    }
    n_train = len(mimiccxr_trainer.mscxr_train_indices)
    n_val = len(mimiccxr_trainer.mscxr_val_indices)
    n_test = len(mimiccxr_trainer.mscxr_test_indices)

    phrase_to_category_name = get_ms_cxr_phrase_to_category_name()

    # Run inference and save predictions
    print_blue('Running inference and saving predictions on MS-CXR ...', bold=True)
    model.eval()
    H, W = None, None
    idx = 0

    train_prc_aucs = []
    val_prc_aucs = []
    test_prc_aucs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', unit='batch', mininterval=2):
            images = batch['i'].to(device)
            phrase_embeddings = batch['pe'].to(device)
            phrase_classification_labels = batch['pcl']
            bboxes = batch['bboxes']
            classes = batch['classes']
            output = model(
                raw_images=images,
                phrase_embeddings=phrase_embeddings,
                predict_bboxes=True,
                only_compute_features=True,
                apply_nms=False, # Skip NMS in order to get all predictions
            )
            phrase_classifier_logits = output['phrase_classifier_logits']
            visual_grounding_bbox_prob_logits = output['visual_grounding_confidence_logits'] # (B, N, H * W, 1)
            visual_grounding_bbox_prob_logits = visual_grounding_bbox_prob_logits.squeeze(-1) # (B, N, H * W)
            visual_grounding_bbox_coord_logits = output['visual_grounding_bbox_logits'] # (B, N, H * W, 4)
            visual_grounding_bbox_coord_logits = visual_grounding_bbox_coord_logits.cpu().numpy()
            phrase_classifier_probs = torch.sigmoid(phrase_classifier_logits).cpu().numpy() # (B, N)
            visual_grounding_bbox_probs = torch.sigmoid(visual_grounding_bbox_prob_logits).cpu().numpy()
            assert visual_grounding_bbox_probs.ndim == 3
            assert visual_grounding_bbox_coord_logits.ndim == 4

            if H is None:
                # Get the integer square root of H * W, assuming H = W
                H = W = math.isqrt(visual_grounding_bbox_prob_logits.shape[-1])
                assert H * W == visual_grounding_bbox_prob_logits.shape[-1]

            batch_size = images.size(0)

            for b in range(batch_size):
            
                i = dataset.indices[idx]
                phrase_idxs = dataset.phrase_idxs[i]
                image_path = dataset.image_paths[i]
                prc_auc = prc_auc_score(phrase_classification_labels[b], phrase_classifier_probs[b])

                if idx < n_train:
                    preds_and_gt = train_preds_and_gt
                    train_prc_aucs.append(prc_auc)
                elif idx < n_train + n_val:
                    preds_and_gt = val_preds_and_gt
                    val_prc_aucs.append(prc_auc)
                else:
                    preds_and_gt = test_preds_and_gt
                    test_prc_aucs.append(prc_auc)
                
                for j, phrase_idx in enumerate(phrase_idxs):
                    phrase = mimiccxr_trainer.mscxr_phrases[phrase_idx]
                    phrase_bboxes = [bbox for bbox, cls in zip(bboxes[b], classes[b]) if cls == j]
                    assert len(phrase_bboxes) > 0
                    pred_bbox_prob_map = visual_grounding_bbox_probs[b, j] # (H * W,)
                    pred_bbox_coord_map = visual_grounding_bbox_coord_logits[b, j] # (H * W, 4)
                    pred_bbox_prob_map = pred_bbox_prob_map.reshape(H, W)
                    pred_bbox_coord_map = pred_bbox_coord_map.reshape(H, W, 4)
                    preds_and_gt['pred_bbox_prob_maps'].append(pred_bbox_prob_map)
                    preds_and_gt['pred_bbox_coord_maps'].append(pred_bbox_coord_map)
                    preds_and_gt['pred_classification_probs'].append(phrase_classifier_probs[b, j])
                    preds_and_gt['gt_bbox_coords'].append(phrase_bboxes)
                    preds_and_gt['phrases'].append(phrase)
                    preds_and_gt['categories'].append(phrase_to_category_name[phrase])
                    preds_and_gt['image_paths'].append(image_path)

                idx += 1

    # Print a few stats
    print_blue('Stats:', bold=True)
    print(f'H = {H}, W = {W}')
    print(f'n_train = {n_train}')
    print(f'n_val = {n_val}')
    print(f'n_test = {n_test}')
    print(f'len(train_preds_and_gt["pred_bbox_prob_maps"]) = {len(train_preds_and_gt["pred_bbox_prob_maps"])}')
    print(f'len(val_preds_and_gt["pred_bbox_prob_maps"]) = {len(val_preds_and_gt["pred_bbox_prob_maps"])}')
    print(f'len(test_preds_and_gt["pred_bbox_prob_maps"]) = {len(test_preds_and_gt["pred_bbox_prob_maps"])}')

    # Print PRC-AUCs
    print_blue('PRC-AUCs:', bold=True)
    print(f'Train PRC-AUC: {sum(train_prc_aucs) / len(train_prc_aucs)}')
    print(f'Val PRC-AUC: {sum(val_prc_aucs) / len(val_prc_aucs)}')
    print(f'Test PRC-AUC: {sum(test_prc_aucs) / len(test_prc_aucs)}')

    # Save predictions and ground truth to file
    print_blue('Saving predictions and ground truth to file ...', bold=True)
    save_path = os.path.join(checkpoint_folder_path, 'results','mscxr_predictions_and_gt.pkl')
    output = dict(
        train_preds_and_gt=train_preds_and_gt,
        val_preds_and_gt=val_preds_and_gt,
        test_preds_and_gt=test_preds_and_gt,
        bbox_format=override_bbox_format or dataset.bbox_format,
    )
    save_pickle(output, save_path)
    print_bold(f'Saved predictions and ground truth to {save_path}')


def run_inference_and_save_predictions_on_padchest_gr(
    model: PhraseGrounder,
    checkpoint_folder_path: str,
    val_image_transform_kwargs: dict,
    max_images_per_batch: int,
    num_workers: int,
    padchest_gr_phrase_embeddings_filepath: str,
    device: torch.device,
    override_bbox_format: str = None,
):
    # Get image transform kwargs
    try: 
        image_transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.PADCHEST_GR]
    except KeyError:
        image_transform_kwargs = next(iter(val_image_transform_kwargs.values())) # get the first value

    # Initialize trainer
    padchestgr_trainer = PadChestGRPhraseTrainer(
        phrase_embeddings_filepath=padchest_gr_phrase_embeddings_filepath,
        bbox_format=override_bbox_format,
        use_test_set=True,
        max_images_per_batch=max_images_per_batch,
        test_batch_size_factor=1.0,
        test_image_transforms_kwargs=image_transform_kwargs,
        num_test_workers=num_workers,
        data_augmentation_enabled=False,
        include_labels_as_phrases=False,
    )

    # Get dataset and dataloader
    dataset = padchestgr_trainer.test_dataset
    dataloader = padchestgr_trainer.test_dataloader

    # Aux variables
    test_preds_and_gt = {
        'pred_bbox_prob_maps': [],
        'pred_bbox_coord_maps': [],
        'pred_classification_probs': [],
        'gt_bbox_coords': [],
        'phrases': [],
        'image_paths': [],
    }

    # Run inference and save predictions
    print_blue('Running inference and saving predictions on PadChest-GR ...', bold=True)
    model.eval()
    H, W = None, None
    idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', unit='batch', mininterval=2):
            images = batch['i'].to(device) # (B, 3, H, W)
            phrase_embeddings = batch['pe'].to(device) # (B, D)
            bboxes = batch['bboxes']
            output = model(
                raw_images=images,
                phrase_embeddings=phrase_embeddings.unsqueeze(1), # (B, 1, D), add a singleton dimension for num_facts
                predict_bboxes=True,
                only_compute_features=True,
                apply_nms=False, # Skip NMS in order to get all predictions
            )
            phrase_classifier_logits = output['phrase_classifier_logits'] # (B, 1)
            visual_grounding_bbox_prob_logits = output['visual_grounding_confidence_logits'] # (B, 1, H * W, 1)
            visual_grounding_bbox_prob_logits = visual_grounding_bbox_prob_logits.squeeze(-1) # (B, 1, H * W)
            visual_grounding_bbox_coord_logits = output['visual_grounding_bbox_logits'] # (B, 1, H * W, 4)
            visual_grounding_bbox_coord_logits = visual_grounding_bbox_coord_logits.cpu().numpy()
            phrase_classifier_probs = torch.sigmoid(phrase_classifier_logits).cpu().numpy() # (B, 1)
            visual_grounding_bbox_probs = torch.sigmoid(visual_grounding_bbox_prob_logits).cpu().numpy() # (B, 1, H * W)
            assert visual_grounding_bbox_probs.ndim == 3
            assert visual_grounding_bbox_coord_logits.ndim == 4

            if H is None:
                # Get the integer square root of H * W, assuming H = W
                H = W = math.isqrt(visual_grounding_bbox_prob_logits.shape[-1])
                assert H * W == visual_grounding_bbox_prob_logits.shape[-1]

            batch_size = images.size(0)

            for b in range(batch_size):
                
                image_path = dataset.image_paths[idx]
                phrase = dataset.phrase_texts[idx]
                phrase_bboxes = bboxes[b]
                assert len(phrase_bboxes) > 0
                pred_bbox_prob_map = visual_grounding_bbox_probs[b, 0] # (H * W,)
                pred_bbox_coord_map = visual_grounding_bbox_coord_logits[b, 0] # (H * W, 4)
                pred_bbox_prob_map = pred_bbox_prob_map.reshape(H, W)
                pred_bbox_coord_map = pred_bbox_coord_map.reshape(H, W, 4)
                test_preds_and_gt['pred_bbox_prob_maps'].append(pred_bbox_prob_map)
                test_preds_and_gt['pred_bbox_coord_maps'].append(pred_bbox_coord_map)
                test_preds_and_gt['pred_classification_probs'].append(phrase_classifier_probs[b, 0])
                test_preds_and_gt['gt_bbox_coords'].append(phrase_bboxes)
                test_preds_and_gt['phrases'].append(phrase)
                test_preds_and_gt['image_paths'].append(image_path)

                idx += 1

    # Print a few stats
    print_blue('Stats:', bold=True)
    print(f'H = {H}, W = {W}')
    print(f'len(test_preds_and_gt["pred_bbox_prob_maps"]) = {len(test_preds_and_gt["pred_bbox_prob_maps"])}')
    
    # Save predictions and ground truth to file
    print_blue('Saving predictions and ground truth to file ...', bold=True)
    save_path = os.path.join(checkpoint_folder_path, 'results', 'padchestgr_predictions_and_gt.pkl')
    output = dict(
        test_preds_and_gt=test_preds_and_gt,
        bbox_format=override_bbox_format or dataset.bbox_format,
    )
    save_pickle(output, save_path)
    print_bold(f'Saved predictions and ground truth to {save_path}')


def run_inference_and_save_predictions_on_vindrcxr(
    model: PhraseGrounder,
    checkpoint_folder_path: str,
    val_image_transform_kwargs: dict,
    max_images_per_batch: int,
    num_workers: int,
    vindrcxr_phrase_embeddings_filepath: str,
    device: torch.device,
    override_bbox_format: str = None,
):
    # Get image transform kwargs
    try: 
        image_transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.PADCHEST_GR]
    except KeyError:
        image_transform_kwargs = next(iter(val_image_transform_kwargs.values())) # get the first value

    # Initialize trainer
    vinbig_trainer = VinBigPhraseTrainer(
        task_mode='grounding',
        mask_height=100, # Not really used, but needed for the trainer
        mask_width=100, # Not really used, but needed for the trainer
        phrase_embeddings_filepath=vindrcxr_phrase_embeddings_filepath,
        max_images_per_batch=max_images_per_batch,
        use_training_set=False,
        use_validation_set=True,
        num_val_workers=num_workers,
        val_image_transform=create_image_transforms(**image_transform_kwargs),
        bbox_format=override_bbox_format,
    )

    # Get dataset and dataloader
    dataset = vinbig_trainer.val_grounding_dataset
    dataloader = vinbig_trainer.val_grounding_dataloader

    # Aux variables
    test_preds_and_gt = {
        'pred_bbox_prob_maps': [],
        'pred_bbox_coord_maps': [],
        'pred_classification_probs': [],
        'gt_bbox_coords': [],
        'phrases': [],
        'classes': [],
        'image_paths': [],
    }

    # Run inference and save predictions
    print_blue('Running inference and saving predictions on VinDr-CXR ...', bold=True)
    model.eval()
    H, W = None, None
    idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', unit='batch', mininterval=2):
            images = batch['i'].to(device) # (B, 3, H, W)
            phrase_embeddings = batch['pe'].to(device) # (B, D)
            bboxes = batch['bboxes']
            output = model(
                raw_images=images,
                phrase_embeddings=phrase_embeddings.unsqueeze(1), # (B, 1, D), add a singleton dimension for num_facts
                predict_bboxes=True,
                only_compute_features=True,
                apply_nms=False, # Skip NMS in order to get all predictions
            )
            phrase_classifier_logits = output['phrase_classifier_logits'] # (B, 1)
            visual_grounding_bbox_prob_logits = output['visual_grounding_confidence_logits'] # (B, 1, H * W, 1)
            visual_grounding_bbox_prob_logits = visual_grounding_bbox_prob_logits.squeeze(-1) # (B, 1, H * W)
            visual_grounding_bbox_coord_logits = output['visual_grounding_bbox_logits'] # (B, 1, H * W, 4)
            visual_grounding_bbox_coord_logits = visual_grounding_bbox_coord_logits.cpu().numpy()
            phrase_classifier_probs = torch.sigmoid(phrase_classifier_logits).cpu().numpy() # (B, 1)
            visual_grounding_bbox_probs = torch.sigmoid(visual_grounding_bbox_prob_logits).cpu().numpy() # (B, 1, H * W)
            assert visual_grounding_bbox_probs.ndim == 3
            assert visual_grounding_bbox_coord_logits.ndim == 4

            if H is None:
                # Get the integer square root of H * W, assuming H = W
                H = W = math.isqrt(visual_grounding_bbox_prob_logits.shape[-1])
                assert H * W == visual_grounding_bbox_prob_logits.shape[-1]

            batch_size = images.size(0)

            for b in range(batch_size):
                i = dataset.indices[idx]
                image_path = dataset.image_paths[i]
                phrase_idx = dataset.phrase_idxs[i]
                phrase = vinbig_trainer.phrases[phrase_idx]
                label_name = vinbig_trainer.label_names[phrase_idx]
                phrase_bboxes = bboxes[b]
                assert len(phrase_bboxes) > 0
                pred_bbox_prob_map = visual_grounding_bbox_probs[b, 0] # (H * W,)
                pred_bbox_coord_map = visual_grounding_bbox_coord_logits[b, 0] # (H * W, 4)
                pred_bbox_prob_map = pred_bbox_prob_map.reshape(H, W)
                pred_bbox_coord_map = pred_bbox_coord_map.reshape(H, W, 4)
                test_preds_and_gt['pred_bbox_prob_maps'].append(pred_bbox_prob_map)
                test_preds_and_gt['pred_bbox_coord_maps'].append(pred_bbox_coord_map)
                test_preds_and_gt['pred_classification_probs'].append(phrase_classifier_probs[b, 0])
                test_preds_and_gt['gt_bbox_coords'].append(phrase_bboxes)
                test_preds_and_gt['phrases'].append(phrase)
                test_preds_and_gt['classes'].append(label_name)
                test_preds_and_gt['image_paths'].append(image_path)

                idx += 1

    # Print a few stats
    print_blue('Stats:', bold=True)
    print(f'H = {H}, W = {W}')
    print(f'len(test_preds_and_gt["pred_bbox_prob_maps"]) = {len(test_preds_and_gt["pred_bbox_prob_maps"])}')
    
    # Save predictions and ground truth to file
    print_blue('Saving predictions and ground truth to file ...', bold=True)
    save_path = os.path.join(checkpoint_folder_path, 'results', 'vindrcxr_predictions_and_gt.pkl')
    output = dict(
        test_preds_and_gt=test_preds_and_gt,
        bbox_format=override_bbox_format or dataset.bbox_format,
    )
    save_pickle(output, save_path)
    print_bold(f'Saved predictions and ground truth to {save_path}')



def run_inference_and_save_predictions_on_chest_imagenome(
    model: PhraseGrounder,
    checkpoint_folder_path: str,
    val_image_transform_kwargs: dict,
    max_images_per_batch: int,
    max_phrases_per_batch: int,
    max_phrases_per_image: int,
    num_workers: int,
    chest_imagenome_augmented_phrase_groundings_filepath: str,
    chest_imagenome_phrase_embeddings_filepath: str,
    chest_imagenome_bbox_phrase_embeddings_filepath: str,
    device: torch.device,
    override_bbox_format: str = None,
):
    # Get image transform kwargs
    try: 
        image_transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]
    except KeyError:
        image_transform_kwargs = next(iter(val_image_transform_kwargs.values())) # get the first value

    # Initialize trainer
    mimiccxr_trainer = MIMICCXR_PhraseGroundingTrainer(
        use_chest_imagenome_for_test=True,
        test_image_transform = create_image_transforms(**image_transform_kwargs),
        max_images_per_batch=max_images_per_batch,
        max_phrases_per_batch=max_phrases_per_batch,
        max_phrases_per_image=max_phrases_per_image,
        num_test_workers=num_workers,
        bbox_format=override_bbox_format,
        source_image_size_mode=MIMICCXR_ImageSizeModes.MEDIUM_512,
        chest_imagenome_augmented_phrase_groundings_filepath=chest_imagenome_augmented_phrase_groundings_filepath,
        chest_imagenome_phrase_embeddings_filepath=chest_imagenome_phrase_embeddings_filepath,
        chest_imagenome_bbox_phrase_embeddings_filepath=chest_imagenome_bbox_phrase_embeddings_filepath,
        mask_width=100, # Not really used, but needed for the trainer
        mask_height=100, # Not really used, but needed for the trainer
    )

    # Get dataset and dataloader
    dataset = mimiccxr_trainer.chest_imagenome_alg_test_dataset
    dataloader = mimiccxr_trainer.chest_imagenome_alg_test_dataloader

    # Aux variables
    test_preds_and_gt = {
        'pred_bbox_prob_maps': [],
        'pred_bbox_coord_maps': [],
        'pred_classification_probs': [],
        'gt_bbox_coords': [],
        'phrases': [],
        'classes': [],
        'image_paths': [],
    }

    # Run inference and save predictions
    print_blue('Running inference and saving predictions on Chest-ImaGenome ...', bold=True)
    model.eval()
    H, W = None, None
    idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', unit='batch', mininterval=2):
            images = batch['i'].to(device) # (B, 3, H, W)
            phrase_embeddings = batch['pe'].to(device) # (B, N, D)
            bboxes_list = batch['bboxes'] # List of lists of bounding boxes
            classes_list = batch['classes'] # List of lists of classes
            output = model(
                raw_images=images,
                phrase_embeddings=phrase_embeddings,
                predict_bboxes=True,
                only_compute_features=True,
                apply_nms=False, # Skip NMS in order to get all predictions
            )
            phrase_classifier_logits = output['phrase_classifier_logits']
            visual_grounding_bbox_prob_logits = output['visual_grounding_confidence_logits'] # (B, N, H * W, 1)
            visual_grounding_bbox_prob_logits = visual_grounding_bbox_prob_logits.squeeze(-1) # (B, N, H * W)
            visual_grounding_bbox_coord_logits = output['visual_grounding_bbox_logits'] # (B, N, H * W, 4)
            visual_grounding_bbox_coord_logits = visual_grounding_bbox_coord_logits.cpu().numpy()
            phrase_classifier_probs = torch.sigmoid(phrase_classifier_logits).cpu().numpy() # (B, N)
            visual_grounding_bbox_probs = torch.sigmoid(visual_grounding_bbox_prob_logits).cpu().numpy()
            assert visual_grounding_bbox_probs.ndim == 3
            assert visual_grounding_bbox_coord_logits.ndim == 4

            if H is None:
                # Get the integer square root of H * W, assuming H = W
                H = W = math.isqrt(visual_grounding_bbox_prob_logits.shape[-1])
                assert H * W == visual_grounding_bbox_prob_logits.shape[-1]

            batch_size = images.size(0)

            for b in range(batch_size):

                image_path = dataset.image_paths[idx]
                
                for cls, bboxes in zip(classes_list[b], bboxes_list[b]):
                    phrase = dataset.anatomical_locations[cls]
                    phrase_bboxes = bboxes
                    assert len(phrase_bboxes) > 0
                    pred_bbox_prob_map = visual_grounding_bbox_probs[b, cls] # (H * W,)
                    pred_bbox_coord_map = visual_grounding_bbox_coord_logits[b, cls] # (H * W, 4)
                    pred_bbox_prob_map = pred_bbox_prob_map.reshape(H, W)
                    pred_bbox_coord_map = pred_bbox_coord_map.reshape(H, W, 4)
                    test_preds_and_gt['pred_bbox_prob_maps'].append(pred_bbox_prob_map)
                    test_preds_and_gt['pred_bbox_coord_maps'].append(pred_bbox_coord_map)
                    test_preds_and_gt['pred_classification_probs'].append(phrase_classifier_probs[b, cls])
                    test_preds_and_gt['gt_bbox_coords'].append(phrase_bboxes)
                    test_preds_and_gt['phrases'].append(phrase)
                    test_preds_and_gt['classes'].append(phrase) # The class is the same as the phrase in this case
                    test_preds_and_gt['image_paths'].append(image_path)

                idx += 1

    # Print a few stats
    print_blue('Stats:', bold=True)
    print(f'H = {H}, W = {W}')
    print(f'len(test_preds_and_gt["pred_bbox_prob_maps"]) = {len(test_preds_and_gt["pred_bbox_prob_maps"])}')

    # Save predictions and ground truth to file
    print_blue('Saving predictions and ground truth to file ...', bold=True)
    save_path = os.path.join(checkpoint_folder_path, 'results', 'chestimagenome_predictions_and_gt.pkl')
    output = dict(
        test_preds_and_gt=test_preds_and_gt,
        bbox_format=override_bbox_format or dataset.bbox_format,
    )
    save_pickle(output, save_path)
    print_bold(f'Saved predictions and ground truth to {save_path}')


def run_inference_and_save_predictions_on_chest_x_det(
    model: PhraseGrounder,
    checkpoint_folder_path: str,
    val_image_transform_kwargs: dict,
    max_images_per_batch: int,
    num_workers: int,
    device: torch.device,
    train_json_path: str,
    train_image_dir: str,
    test_json_path: str,
    test_image_dir: str,
    label2embedding_path: str,
    override_bbox_format: str = None,
):
    # Get image transform kwargs
    image_transform_kwargs = next(iter(val_image_transform_kwargs.values())) # get the first value

    # Initialize trainer
    image_transforms = create_image_transforms(**image_transform_kwargs)
    train_dataset = ChestXDetInferenceDataset(
        json_path=train_json_path,
        image_dir=train_image_dir,
        image_transform=image_transforms,
        mask_res=(100, 100),
        bbox_format=override_bbox_format,
        label2embedding_path=label2embedding_path,
    )
    test_dataset = ChestXDetInferenceDataset(
        json_path=test_json_path,
        image_dir=test_image_dir,
        image_transform=image_transforms,
        mask_res=(100, 100),
        bbox_format=override_bbox_format,
        label2embedding_path=label2embedding_path,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=max_images_per_batch,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=max_images_per_batch,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_dataset.collate_fn,
    )

    # Aux variables
    train_preds_and_gt = {
        'pred_bbox_prob_maps': [],
        'pred_bbox_coord_maps': [],
        'pred_classification_probs': [],
        'gt_bbox_coords': [],
        'gt_polygons': [],
        'gt_mask': [],
        'phrases': [],
        'classes': [],
        'image_paths': [],
    }
    test_preds_and_gt = {
        'pred_bbox_prob_maps': [],
        'pred_bbox_coord_maps': [],
        'pred_classification_probs': [],
        'gt_bbox_coords': [],
        'gt_polygons': [],
        'gt_mask': [],
        'phrases': [],
        'classes': [],
        'image_paths': [],
    }

    # Run inference and save predictions
    print_blue('Running inference and saving predictions on Chest-X-Det ...', bold=True)
    model.eval()
    H, W = None, None
    
    with torch.no_grad():
        
        for dataloader, preds_and_gt in zip(
            [train_dataloader, test_dataloader],
            [train_preds_and_gt, test_preds_and_gt]
        ):
            for batch in tqdm(dataloader, desc='Evaluating', unit='batch', mininterval=2):
                image_paths = batch['image_paths']
                images = batch['pixel_values'].to(device) # (B, 3, H, W)
                phrase_embeddings = batch['embeddings'].to(device) # (B, D)
                bboxes_list = batch['bboxes'] # List of lists of bounding boxes
                polygons_list = batch['polygons'] # List of lists of polygons
                mask_list = batch['masks'] # List of masks
                labels = batch['labels'] # List of labels (str)
                output = model(
                    raw_images=images,
                    phrase_embeddings=phrase_embeddings.unsqueeze(1), # (B, 1, D), add a singleton dimension for num_facts
                    predict_bboxes=True,
                    only_compute_features=True,
                    apply_nms=False, # Skip NMS in order to get all predictions
                )
                phrase_classifier_logits = output['phrase_classifier_logits']
                visual_grounding_bbox_prob_logits = output['visual_grounding_confidence_logits'] # (B, 1, H * W, 1)
                visual_grounding_bbox_prob_logits = visual_grounding_bbox_prob_logits.squeeze(-1) # (B, 1, H * W)
                visual_grounding_bbox_coord_logits = output['visual_grounding_bbox_logits'] # (B, 1, H * W, 4)
                visual_grounding_bbox_coord_logits = visual_grounding_bbox_coord_logits.cpu().numpy()
                phrase_classifier_probs = torch.sigmoid(phrase_classifier_logits).cpu().numpy() # (B, 1)
                visual_grounding_bbox_probs = torch.sigmoid(visual_grounding_bbox_prob_logits).cpu().numpy()
                assert visual_grounding_bbox_probs.ndim == 3
                assert visual_grounding_bbox_coord_logits.ndim == 4

                if H is None:
                    # Get the integer square root of H * W, assuming H = W
                    H = W = math.isqrt(visual_grounding_bbox_prob_logits.shape[-1])
                    assert H * W == visual_grounding_bbox_prob_logits.shape[-1]

                batch_size = images.size(0)

                for b in range(batch_size):

                    image_path = image_paths[b]
                    pred_bbox_prob_map = visual_grounding_bbox_probs[b, 0] # (H * W,)
                    pred_bbox_coord_map = visual_grounding_bbox_coord_logits[b, 0] # (H * W, 4)
                    pred_bbox_prob_map = pred_bbox_prob_map.reshape(H, W)
                    pred_bbox_coord_map = pred_bbox_coord_map.reshape(H, W, 4)
                    preds_and_gt['pred_bbox_prob_maps'].append(pred_bbox_prob_map)
                    preds_and_gt['pred_bbox_coord_maps'].append(pred_bbox_coord_map)
                    preds_and_gt['pred_classification_probs'].append(phrase_classifier_probs[b, 0])
                    preds_and_gt['gt_bbox_coords'].append(bboxes_list[b])
                    preds_and_gt['gt_polygons'].append(polygons_list[b])
                    preds_and_gt['gt_mask'].append(mask_list[b])
                    preds_and_gt['phrases'].append(labels[b]) # The phrase is the same as the label in this case
                    preds_and_gt['classes'].append(labels[b]) # The class is the same as the label in this case
                    preds_and_gt['image_paths'].append(image_path)

    # Print a few stats
    print_blue('Stats:', bold=True)
    print(f'H = {H}, W = {W}')
    print(f'len(train_preds_and_gt["pred_bbox_prob_maps"]) = {len(train_preds_and_gt["pred_bbox_prob_maps"])}')
    print(f'len(test_preds_and_gt["pred_bbox_prob_maps"]) = {len(test_preds_and_gt["pred_bbox_prob_maps"])}')

    # Save predictions and ground truth to file
    print_blue('Saving predictions and ground truth to file ...', bold=True)
    save_path = os.path.join(checkpoint_folder_path, 'results', 'chestxdet_predictions_and_gt.pkl')
    output = dict(
        train_preds_and_gt=train_preds_and_gt,
        test_preds_and_gt=test_preds_and_gt,
        bbox_format=override_bbox_format or train_dataset.bbox_format,
    )
    save_pickle(output, save_path)
    print_bold(f'Saved predictions and ground truth to {save_path}')


def _compute_probability_map_iou(
    bbox_format: str,
    preds_and_gt: dict,
    threshold: float = 0.5,
):
    ious = [compute_probability_map_iou(prob_map, gt_bboxes, threshold, bbox_format=bbox_format) for\
                prob_map, gt_bboxes in zip(preds_and_gt['pred_bbox_prob_maps'],
                                            preds_and_gt['gt_bbox_coords'])]
    iou_with_bootstrapping = apply_bootstrapping(metric_values=ious, metric_name='iou',
                                                num_bootstraps=500, num_processes=6)
    print_bold('IoU with bootstrapping:')
    pprint(iou_with_bootstrapping)
    return ious, iou_with_bootstrapping

def _compute_probability_map_iou_on_mscxr(
    split: str,
    bbox_format: str,
    preds_and_gt: dict,
    category_to_indices: list[list[int]],
    category_names: list[str],
    threshold: float = 0.5,
):
    ious = [compute_probability_map_iou(prob_map, gt_bboxes, threshold, bbox_format=bbox_format) for\
                prob_map, gt_bboxes in zip(preds_and_gt['pred_bbox_prob_maps'],
                                            preds_and_gt['gt_bbox_coords'])]
    iou_with_bootstrapping = apply_bootstrapping(
        metric_values=ious, class_to_indices=category_to_indices,
        class_names=category_names, metric_name='iou', num_bootstraps=500, num_processes=6)
    print_bold(f'{split.capitalize()} IoU with bootstrapping:')
    pprint(iou_with_bootstrapping)
    return ious, iou_with_bootstrapping


def _compute_bbox_iou(
    preds_and_gt: dict,
    candidate_iou_thresholds: list[float],
    candidate_conf_thresholds: list[float],
    bbox_format: str,
):
    print_blue('Finding optimal IoU and conf thresholds for IoU based on bounding box coordinates ...', bold=True)
    tmp = find_optimal_conf_iou_max_det_thresholds__single_class(
        gt_coords_list=preds_and_gt['gt_bbox_coords'],
        pred_boxes_list=preds_and_gt['pred_bbox_coord_maps'],
        pred_confs_list=preds_and_gt['pred_bbox_prob_maps'],
        iou_thresholds=candidate_iou_thresholds,
        conf_thresholds=candidate_conf_thresholds,
        verbose=False,
        bbox_format=bbox_format,
    )
    best_iou_th = tmp['best_iou_threshold']
    best_conf_th = tmp['best_conf_threshold']
    best_pre_nms_max_det = tmp['best_pre_nms_max_det']
    best_post_nms_max_det = tmp['best_post_nms_max_det']
    print(f'best_iou_threshold = {best_iou_th}')
    print(f'best_conf_threshold = {best_conf_th}')
    print(f'best_pre_nms_max_det = {best_pre_nms_max_det}')
    print(f'best_post_nms_max_det = {best_post_nms_max_det}')            
    ious = []
    for pred_bbox_coord_map, pred_bbox_prob_map, gt_bbox_coords in zip(preds_and_gt['pred_bbox_coord_maps'],
                                                                    preds_and_gt['pred_bbox_prob_maps'],
                                                                    preds_and_gt['gt_bbox_coords']):
        iou = compute_iou_with_nms(
            gt_bboxes=gt_bbox_coords,
            pred_bbox_coords=pred_bbox_coord_map.reshape(-1, 4),
            pred_bbox_probs=pred_bbox_prob_map.reshape(-1),
            iou_th=best_iou_th,
            conf_th=best_conf_th,
            pre_nms_max_det=best_pre_nms_max_det,
            post_nms_max_det=best_post_nms_max_det,
            bbox_format=bbox_format,
        )
        ious.append(iou)
    iou_with_bootstrapping = apply_bootstrapping(metric_values=ious, metric_name='iou',
                                                num_bootstraps=500, num_processes=6)
    print_bold('IoU with bootstrapping:')
    pprint(iou_with_bootstrapping)
    return (
        best_iou_th, best_conf_th, best_pre_nms_max_det,
        best_post_nms_max_det, ious, iou_with_bootstrapping
    )

def _compute_bbox_iou_on_mscxr(
    split: str,
    preds_and_gt: dict,
    class_to_indices: list[list[int]],
    category_names: list[str],
    candidate_iou_thresholds: list[float],
    candidate_conf_thresholds: list[float],
    bbox_format: str,
):
    print_blue(f'Finding optimal IoU and conf thresholds for IoU based on bounding box coordinates on the {split} set ...', bold=True)
    tmp = find_optimal_conf_iou_max_det_thresholds__single_class(
        gt_coords_list=preds_and_gt['gt_bbox_coords'],
        pred_boxes_list=preds_and_gt['pred_bbox_coord_maps'],
        pred_confs_list=preds_and_gt['pred_bbox_prob_maps'],
        iou_thresholds=candidate_iou_thresholds,
        conf_thresholds=candidate_conf_thresholds,
        verbose=False,
        bbox_format=bbox_format,
    )
    best_iou_th = tmp['best_iou_threshold']
    best_conf_th = tmp['best_conf_threshold']
    best_pre_nms_max_det = tmp['best_pre_nms_max_det']
    best_post_nms_max_det = tmp['best_post_nms_max_det']
    print(f'{split}_best_iou_threshold = {best_iou_th}')
    print(f'{split}_best_conf_threshold = {best_conf_th}')
    print(f'{split}_best_pre_nms_max_det = {best_pre_nms_max_det}')
    print(f'{split}_best_post_nms_max_det = {best_post_nms_max_det}')            
    ious = []
    for pred_bbox_coord_map, pred_bbox_prob_map, gt_bbox_coords in zip(preds_and_gt['pred_bbox_coord_maps'],
                                                                    preds_and_gt['pred_bbox_prob_maps'],
                                                                    preds_and_gt['gt_bbox_coords']):
        iou = compute_iou_with_nms(
            gt_bboxes=gt_bbox_coords,
            pred_bbox_coords=pred_bbox_coord_map.reshape(-1, 4),
            pred_bbox_probs=pred_bbox_prob_map.reshape(-1),
            iou_th=best_iou_th,
            conf_th=best_conf_th,
            pre_nms_max_det=best_pre_nms_max_det,
            post_nms_max_det=best_post_nms_max_det,
            bbox_format=bbox_format,
        )
        ious.append(iou)
    iou_with_bootstrapping = apply_bootstrapping(
        metric_values=ious, class_to_indices=class_to_indices,
        class_names=category_names, metric_name='iou', num_bootstraps=500, num_processes=6)
    print_bold(f'{split.capitalize()} IoU with bootstrapping:')
    pprint(iou_with_bootstrapping)
    return (
        best_iou_th, best_conf_th, best_pre_nms_max_det,
        best_post_nms_max_det, ious, iou_with_bootstrapping
    )


def _compute_avg_classification_prob(preds_and_gt: dict):
    print_blue('Computing average classification probability ...', bold=True)
    avg_classification_prob_with_bootstrapping = apply_bootstrapping(
        metric_values=preds_and_gt['pred_classification_probs'],
        metric_name='avg_classification_prob', num_bootstraps=500, num_processes=6)
    print_bold('Avg Classification Prob with bootstrapping:')
    pprint(avg_classification_prob_with_bootstrapping)
    return avg_classification_prob_with_bootstrapping

def _compute_avg_classification_prob_on_mscxr(
    split: str,
    category_names: list[str],
    preds_and_gt: dict,
    category_to_indices: list[list[int]],
):
    print_blue(f'Computing average classification probability on the {split} set ...', bold=True)
    avg_classification_prob_with_bootstrapping = apply_bootstrapping(
        metric_values=preds_and_gt['pred_classification_probs'], class_to_indices=category_to_indices,
        class_names=category_names, metric_name='avg_classification_prob', num_bootstraps=500, num_processes=6)
    print_bold(f'{split.capitalize()} Avg Classification Prob with bootstrapping:')
    pprint(avg_classification_prob_with_bootstrapping)
    return avg_classification_prob_with_bootstrapping


def _compute_segmentation_metric(
    preds_and_gt: dict,
    bbox_format: str,
    mask_resolution: tuple = (100, 100),
    metric_fn=None,
    metric_name: str = "",
    metric_kwargs: dict = None,
    num_bootstraps: int = 500,
    num_processes: int = 6,
    use_tqdm: bool = False,
    category_to_indices: list[list[int]] = None,
    category_names: list[str] = None,
    split: str = None,
    print_fn=None,
    gt_masks_list: list = None,  # <-- NEW ARG
):
    if print_fn is None:
        print_fn = print
    if metric_kwargs is None:
        metric_kwargs = {}

    if split is not None:
        print_fn(f'Computing {metric_name} for the {split} split ...')
    else:
        print_fn(f'Computing {metric_name} ...')
    gt_bboxes_list = preds_and_gt["gt_bbox_coords"]
    prob_maps_list = [
        np.array(p) for p in preds_and_gt["pred_bbox_prob_maps"]
    ]
    num_samples = len(gt_bboxes_list)

    # 1. Compute metric for each sample
    metric_values = []
    for i in range(num_samples):
        if gt_masks_list is not None:
            mask = gt_masks_list[i]
        else:
            if bbox_format == 'cxcywh':
                xyxy_bboxes = [
                    cxcywh_to_xyxy(bbox) for bbox in gt_bboxes_list[i]
                ]
            elif bbox_format == 'xyxy':
                xyxy_bboxes = gt_bboxes_list[i]
            else:
                raise ValueError(f"Unsupported bbox format: {bbox_format}")
            mask = convert_bboxes_into_presence_map(
                xyxy_bboxes, mask_resolution
            )
        metric_value = metric_fn(mask, prob_maps_list[i], **metric_kwargs)
        metric_values.append(metric_value)

    # 2. Apply stratified bootstrapping
    metric_with_bootstrapping = apply_bootstrapping(
        metric_values=np.array(metric_values),
        class_to_indices=category_to_indices,
        class_names=category_names,
        metric_name=metric_name,
        num_bootstraps=num_bootstraps,
        num_processes=num_processes,
        use_tqdm=use_tqdm,
    )

    # 3. Print bootstrapped statistics
    print_fn(f"Bootstrapped {metric_name} statistics:")
    print_fn(f"  Mean {metric_name}: {metric_with_bootstrapping[metric_name]['mean']}")
    print_fn(f"  Std {metric_name}: {metric_with_bootstrapping[metric_name]['std']}")

    return metric_values, metric_with_bootstrapping


def _compute_segmentation_iou(
    preds_and_gt, bbox_format=None, category_to_indices=None, category_names=None,
    split=None, mask_resolution=(100, 100), threshold=0.5, gt_masks_list=None,
):
    return _compute_segmentation_metric(
        split=split,
        preds_and_gt=preds_and_gt,
        category_to_indices=category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=mask_resolution,
        metric_fn=calculate_segmentation_iou,
        metric_name="iou",
        metric_kwargs={'threshold': threshold},
        gt_masks_list=gt_masks_list,
    )


def _compute_cnr(
    preds_and_gt, bbox_format=None, category_to_indices=None, category_names=None,
    split=None, mask_resolution=(100, 100), gt_masks_list=None
):
    return _compute_segmentation_metric(
        split=split,
        preds_and_gt=preds_and_gt,
        category_to_indices=category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=mask_resolution,
        metric_fn=calculate_cnr,
        metric_name="cnr",
        metric_kwargs={},
        gt_masks_list=gt_masks_list,
    )


def _compute_soft_dice(
    preds_and_gt, bbox_format=None, category_to_indices=None, category_names=None,
    split=None, mask_resolution=(100, 100), gt_masks_list=None
):
    return _compute_segmentation_metric(
        split=split,
        preds_and_gt=preds_and_gt,
        category_to_indices=category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=mask_resolution,
        metric_fn=calculate_soft_dice,
        metric_name="soft_dice",
        metric_kwargs={},
        gt_masks_list=gt_masks_list,
    )


def _compute_dice(
    preds_and_gt, bbox_format=None, category_to_indices=None, category_names=None,
    split=None, mask_resolution=(100, 100), threshold=0.5, gt_masks_list=None
):
    return _compute_segmentation_metric(
        split=split,
        preds_and_gt=preds_and_gt,
        category_to_indices=category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=mask_resolution,
        metric_fn=calculate_dice,
        metric_name="dice",
        metric_kwargs={"threshold": threshold},
        gt_masks_list=gt_masks_list,
    )


def _compute_rand_index(
    preds_and_gt, bbox_format=None, category_to_indices=None, category_names=None,
    split=None, mask_resolution=(100, 100), threshold=0.5, gt_masks_list=None
):
    return _compute_segmentation_metric(
        split=split,
        preds_and_gt=preds_and_gt,
        category_to_indices=category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=mask_resolution,
        metric_fn=calculate_rand_index,
        metric_name="rand_index",
        metric_kwargs={"threshold": threshold},
        gt_masks_list=gt_masks_list,
    )


def _compute_precision(
    preds_and_gt, bbox_format=None, category_to_indices=None, category_names=None,
    split=None, mask_resolution=(100, 100), threshold=0.5, gt_masks_list=None
):
    return _compute_segmentation_metric(
        split=split,
        preds_and_gt=preds_and_gt,
        category_to_indices=category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=mask_resolution,
        metric_fn=calculate_segmentation_precision,
        metric_name="precision",
        metric_kwargs={"threshold": threshold},
        gt_masks_list=gt_masks_list,
    )


def _compute_recall(
    preds_and_gt, bbox_format=None, category_to_indices=None, category_names=None,
    split=None, mask_resolution=(100, 100), threshold=0.5, gt_masks_list=None
):
    return _compute_segmentation_metric(
        split=split,
        preds_and_gt=preds_and_gt,
        category_to_indices=category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=mask_resolution,
        metric_fn=calculate_segmentation_recall,
        metric_name="recall",
        metric_kwargs={"threshold": threshold},
        gt_masks_list=gt_masks_list,
    )


def compute_and_save_metrics_on_mscxr(
    predictions_and_gt_filepath: str,
    candidate_conf_thresholds: list[float],
    candidate_iou_thresholds: list[float],
):
    assert candidate_conf_thresholds is not None
    assert candidate_iou_thresholds is not None

    # Load predictions and ground truth
    print(f'Loading predictions and ground truth from {predictions_and_gt_filepath} ...')
    preds_and_gt = load_pickle(predictions_and_gt_filepath)
    train_preds_and_gt = preds_and_gt['train_preds_and_gt']
    val_preds_and_gt = preds_and_gt['val_preds_and_gt']
    test_preds_and_gt = preds_and_gt['test_preds_and_gt']
    bbox_format = preds_and_gt['bbox_format']

    category_names = get_ms_cxr_category_names()
    category_name_to_idx = {name: idx for idx, name in enumerate(category_names)}
    phrase_to_category_name = get_ms_cxr_phrase_to_category_name()

    def _get_category_to_indices(category_names, phrases):
        category_to_indices = [[] for _ in range(len(category_names))]
        for i, phrase in enumerate(phrases):
            category_name = phrase_to_category_name[phrase]
            category_idx = category_name_to_idx[category_name]
            category_to_indices[category_idx].append(i)
        return category_to_indices
    
    train_category_to_indices = _get_category_to_indices(category_names, train_preds_and_gt['phrases'])
    val_category_to_indices = _get_category_to_indices(category_names, val_preds_and_gt['phrases'])
    test_category_to_indices = _get_category_to_indices(category_names, test_preds_and_gt['phrases'])

    # --- Compute IoU based on bbox probability maps on train, val, and test sets with bootstrapping ---
    print_blue('Computing IoU based on bbox probability maps on train, val, and test sets ...', bold=True)
    # Train
    tmp = find_optimal_probability_map_conf_threshold(
        prob_maps=np.array(train_preds_and_gt['pred_bbox_prob_maps']), # (N, H, W)
        gt_bboxes_list=train_preds_and_gt['gt_bbox_coords'],
        bbox_format=bbox_format,
    )
    train_best_prob_conf_th = tmp['best_conf_th']
    print(f'train_best_prob_conf_th = {train_best_prob_conf_th}')
    train_prob_ious, train_prob_iou_with_bootstrapping = _compute_probability_map_iou_on_mscxr(
        split='train',
        bbox_format=bbox_format,
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_category_to_indices,
        category_names=category_names,
        threshold=train_best_prob_conf_th,
    )
    # Val
    tmp = find_optimal_probability_map_conf_threshold(
        prob_maps=np.array(val_preds_and_gt['pred_bbox_prob_maps']), # (N, H, W)
        gt_bboxes_list=val_preds_and_gt['gt_bbox_coords'],
        bbox_format=bbox_format,
    )
    val_best_prob_conf_th = tmp['best_conf_th']
    print(f'val_best_prob_conf_th = {val_best_prob_conf_th}')
    val_prob_ious, val_prob_iou_with_bootstrapping = _compute_probability_map_iou_on_mscxr(
        split='val',
        bbox_format=bbox_format,
        preds_and_gt=val_preds_and_gt,
        category_to_indices=val_category_to_indices,
        category_names=category_names,
        threshold=val_best_prob_conf_th,
    )
    # Test
    tmp = find_optimal_probability_map_conf_threshold(
        prob_maps=np.array(test_preds_and_gt['pred_bbox_prob_maps']), # (N, H, W)
        gt_bboxes_list=test_preds_and_gt['gt_bbox_coords'],
        bbox_format=bbox_format,
    )
    test_best_prob_conf_th = tmp['best_conf_th']
    print(f'test_best_prob_conf_th = {test_best_prob_conf_th}')
    test_prob_ious, test_prob_iou_with_bootstrapping = _compute_probability_map_iou_on_mscxr(
        split='test',
        bbox_format=bbox_format,
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_category_to_indices,
        category_names=category_names,
        threshold=test_best_prob_conf_th,
    )

    # --- Compute IoU based on bbox coordinates on train, val, and test sets with bootstrapping ---
    print_blue('Computing IoU based on bbox coordinates on train, val, and test sets ...', bold=True)
    # Train
    (
        train_best_bbox_iou_th, train_best_bbox_conf_th, train_best_bbox_pre_nms_max_det,
        train_best_bbox_post_nms_max_det, train_bbox_ious, train_bbox_iou_with_bootstrapping
    ) = _compute_bbox_iou_on_mscxr(
        split='train',
        preds_and_gt=train_preds_and_gt,
        class_to_indices=train_category_to_indices,
        category_names=category_names,
        candidate_iou_thresholds=candidate_iou_thresholds,
        candidate_conf_thresholds=candidate_conf_thresholds,
        bbox_format=bbox_format,
    )
    # Val
    (
        val_best_bbox_iou_th, val_best_bbox_conf_th, val_best_bbox_pre_nms_max_det,
        val_best_bbox_post_nms_max_det, val_bbox_ious, val_bbox_iou_with_bootstrapping
    ) = _compute_bbox_iou_on_mscxr(
        split='val',
        preds_and_gt=val_preds_and_gt,
        class_to_indices=val_category_to_indices,
        category_names=category_names,
        candidate_iou_thresholds=candidate_iou_thresholds,
        candidate_conf_thresholds=candidate_conf_thresholds,
        bbox_format=bbox_format,
    )
    # Test
    (
        test_best_bbox_iou_th, test_best_bbox_conf_th, test_best_bbox_pre_nms_max_det,
        test_best_bbox_post_nms_max_det, test_bbox_ious, test_bbox_iou_with_bootstrapping
    ) = _compute_bbox_iou_on_mscxr(
        split='test',
        preds_and_gt=test_preds_and_gt,
        class_to_indices=test_category_to_indices,
        category_names=category_names,
        candidate_iou_thresholds=candidate_iou_thresholds,
        candidate_conf_thresholds=candidate_conf_thresholds,
        bbox_format=bbox_format,
    )    

    # --- Compute average classification probability on train, val, and test sets with bootstrapping ---
    print_blue('Computing average classification probability on train, val, and test sets ...', bold=True)
    # Train
    train_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob_on_mscxr(
        'train', category_names, train_preds_and_gt, train_category_to_indices)
    # Val
    val_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob_on_mscxr(
        'val', category_names, val_preds_and_gt, val_category_to_indices)
    # Test
    test_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob_on_mscxr(
        'test', category_names, test_preds_and_gt, test_category_to_indices)
    
    # --- Compute CNR on train, val, and test sets with bootstrapping ---
    print_blue('Computing CNR on train, val, and test sets ...', bold=True)
    train_cnrs, train_cnr_with_bootstrapping = _compute_cnr(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )
    val_cnrs, val_cnr_with_bootstrapping = _compute_cnr(
        split='val',
        preds_and_gt=val_preds_and_gt,
        category_to_indices=val_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )
    test_cnrs, test_cnr_with_bootstrapping = _compute_cnr(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )

    # --- Compute Soft Dice on train, val, and test sets with bootstrapping ---
    print_blue('Computing Soft Dice on train, val, and test sets ...', bold=True)
    train_soft_dices, train_soft_dice_with_bootstrapping = _compute_soft_dice(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )
    val_soft_dices, val_soft_dice_with_bootstrapping = _compute_soft_dice(
        split='val',
        preds_and_gt=val_preds_and_gt,
        category_to_indices=val_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )
    test_soft_dices, test_soft_dice_with_bootstrapping = _compute_soft_dice(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )

    # --- Compute Dice on train, val, and test sets with bootstrapping ---
    print_blue('Computing Dice on train, val, and test sets ...', bold=True)
    train_dices, train_dice_with_bootstrapping = _compute_dice(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=train_best_prob_conf_th,
    )
    val_dices, val_dice_with_bootstrapping = _compute_dice(
        split='val',
        preds_and_gt=val_preds_and_gt,
        category_to_indices=val_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=val_best_prob_conf_th,
    )
    test_dices, test_dice_with_bootstrapping = _compute_dice(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Compute Rand Index on train, val, and test sets with bootstrapping ---
    print_blue('Computing Rand Index on train, val, and test sets ...', bold=True)
    train_rand_indices, train_rand_index_with_bootstrapping = _compute_rand_index(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=train_best_prob_conf_th,
    )
    val_rand_indices, val_rand_index_with_bootstrapping = _compute_rand_index(
        split='val',
        preds_and_gt=val_preds_and_gt,
        category_to_indices=val_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=val_best_prob_conf_th,
    )
    test_rand_indices, test_rand_index_with_bootstrapping = _compute_rand_index(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Compute Precision on train, val, and test sets with bootstrapping ---
    print_blue('Computing Precision on train, val, and test sets ...', bold=True)
    train_precisions, train_precision_with_bootstrapping = _compute_precision(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=train_best_prob_conf_th,
    )
    val_precisions, val_precision_with_bootstrapping = _compute_precision(
        split='val',
        preds_and_gt=val_preds_and_gt,
        category_to_indices=val_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=val_best_prob_conf_th,
    )
    test_precisions, test_precision_with_bootstrapping = _compute_precision(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )    

    # --- Compute Recall on train, val, and test sets with bootstrapping ---
    print_blue('Computing Recall on train, val, and test sets ...', bold=True)
    train_recalls, train_recall_with_bootstrapping = _compute_recall(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=train_best_prob_conf_th,
    )
    val_recalls, val_recall_with_bootstrapping = _compute_recall(
        split='val',
        preds_and_gt=val_preds_and_gt,
        category_to_indices=val_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=val_best_prob_conf_th,
    )
    test_recalls, test_recall_with_bootstrapping = _compute_recall(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_category_to_indices,
        category_names=category_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Save metrics to file ---
    print_blue('Saving metrics to file ...', bold=True)
    save_path = predictions_and_gt_filepath + '.metrics.pkl' # append .metrics.pkl to the predictions_and_gt_filepath
    output = dict(
        train_metrics=dict(
            best_prob_conf_th=train_best_prob_conf_th,
            best_bbox_iou_th=train_best_bbox_iou_th,
            best_bbox_conf_th=train_best_bbox_conf_th,
            best_bbox_pre_nms_max_det=train_best_bbox_pre_nms_max_det,
            best_bbox_post_nms_max_det=train_best_bbox_post_nms_max_det,
            prob_ious=train_prob_ious,
            prob_iou_with_bootstrapping=train_prob_iou_with_bootstrapping,
            bbox_ious=train_bbox_ious,
            bbox_iou_with_bootstrapping=train_bbox_iou_with_bootstrapping,
            avg_classification_prob_with_bootstrappig=train_avg_classification_prob_with_bootstrappig,
            cnrs=train_cnrs,
            cnr_with_bootstrapping=train_cnr_with_bootstrapping,
            soft_dices=train_soft_dices,
            soft_dice_with_bootstrapping=train_soft_dice_with_bootstrapping,
            dices=train_dices,
            dice_with_bootstrapping=train_dice_with_bootstrapping,
            rand_indices=train_rand_indices,
            rand_index_with_bootstrapping=train_rand_index_with_bootstrapping,
            precisions=train_precisions,
            precision_with_bootstrapping=train_precision_with_bootstrapping,
            recalls=train_recalls,
            recall_with_bootstrapping=train_recall_with_bootstrapping,
        ),
        val_metrics=dict(
            best_prob_conf_th=val_best_prob_conf_th,
            best_bbox_iou_th=val_best_bbox_iou_th,
            best_bbox_conf_th=val_best_bbox_conf_th,
            best_bbox_pre_nms_max_det=val_best_bbox_pre_nms_max_det,
            best_bbox_post_nms_max_det=val_best_bbox_post_nms_max_det,
            prob_ious=val_prob_ious,
            prob_iou_with_bootstrapping=val_prob_iou_with_bootstrapping,
            bbox_ious=val_bbox_ious,
            bbox_iou_with_bootstrapping=val_bbox_iou_with_bootstrapping,
            avg_classification_prob_with_bootstrappig=val_avg_classification_prob_with_bootstrappig,
            cnrs=val_cnrs,
            cnr_with_bootstrapping=val_cnr_with_bootstrapping,
            soft_dices=val_soft_dices,
            soft_dice_with_bootstrapping=val_soft_dice_with_bootstrapping,
            dices=val_dices,
            dice_with_bootstrapping=val_dice_with_bootstrapping,
            rand_indices=val_rand_indices,
            rand_index_with_bootstrapping=val_rand_index_with_bootstrapping,
            precisions=val_precisions,
            precision_with_bootstrapping=val_precision_with_bootstrapping,
            recalls=val_recalls,
            recall_with_bootstrapping=val_recall_with_bootstrapping,
        ),
        test_metrics=dict(
            best_prob_conf_th=test_best_prob_conf_th,
            best_bbox_iou_th=test_best_bbox_iou_th,
            best_bbox_conf_th=test_best_bbox_conf_th,
            best_bbox_pre_nms_max_det=test_best_bbox_pre_nms_max_det,
            best_bbox_post_nms_max_det=test_best_bbox_post_nms_max_det,
            prob_ious=test_prob_ious,
            prob_iou_with_bootstrapping=test_prob_iou_with_bootstrapping,
            bbox_ious=test_bbox_ious,
            bbox_iou_with_bootstrapping=test_bbox_iou_with_bootstrapping,
            avg_classification_prob_with_bootstrappig=test_avg_classification_prob_with_bootstrappig,
            cnrs=test_cnrs,
            cnr_with_bootstrapping=test_cnr_with_bootstrapping,
            soft_dices=test_soft_dices,
            soft_dice_with_bootstrapping=test_soft_dice_with_bootstrapping,
            dices=test_dices,
            dice_with_bootstrapping=test_dice_with_bootstrapping,
            rand_indices=test_rand_indices,
            rand_index_with_bootstrapping=test_rand_index_with_bootstrapping,
            precisions=test_precisions,
            precision_with_bootstrapping=test_precision_with_bootstrapping,
            recalls=test_recalls,
            recall_with_bootstrapping=test_recall_with_bootstrapping,
        ),
    )
    save_pickle(output, save_path)
    print_bold(f'Saved metrics to {save_path}')


def compute_and_save_metrics_on_padchest_gr(
    predictions_and_gt_filepath: str,
    candidate_conf_thresholds: list[float],
    candidate_iou_thresholds: list[float],
):
    assert candidate_iou_thresholds is not None

    # Load predictions and ground truth
    print(f'Loading predictions and ground truth from {predictions_and_gt_filepath} ...')
    preds_and_gt = load_pickle(predictions_and_gt_filepath)
    test_preds_and_gt = preds_and_gt['test_preds_and_gt']
    bbox_format = preds_and_gt['bbox_format']

    # --- Compute IoU based on bbox probability maps on the test set with bootstrapping ---
    print_blue('Computing IoU based on bbox probability maps on the test set ...', bold=True)
    tmp = find_optimal_probability_map_conf_threshold(
        prob_maps=np.array(test_preds_and_gt['pred_bbox_prob_maps']), # (N, H, W)
        gt_bboxes_list=test_preds_and_gt['gt_bbox_coords'],
        bbox_format=bbox_format,
    )
    test_best_prob_conf_th = tmp['best_conf_th']
    print(f'test_best_prob_conf_th = {test_best_prob_conf_th}')
    test_prob_ious, test_prob_iou_with_bootstrapping = _compute_probability_map_iou(
        bbox_format=bbox_format,
        preds_and_gt=test_preds_and_gt,
        threshold=test_best_prob_conf_th,
    )

    # --- Compute IoU based on bbox coordinates on the test set with bootstrapping ---
    if candidate_conf_thresholds is None:
        candidate_conf_thresholds = [test_best_prob_conf_th]  # Use the best prob conf threshold as the only candidate conf threshold
    print_blue('Computing IoU based on bbox coordinates on the test set ...', bold=True)
    (
        test_best_bbox_iou_th, test_best_bbox_conf_th, test_best_bbox_pre_nms_max_det,
        test_best_bbox_post_nms_max_det, test_bbox_ious, test_bbox_iou_with_bootstrapping
    ) = _compute_bbox_iou(
        preds_and_gt=test_preds_and_gt,
        candidate_iou_thresholds=candidate_iou_thresholds,
        candidate_conf_thresholds=candidate_conf_thresholds,
        bbox_format=bbox_format,
    )

    # --- Compute average classification probability on the test set with bootstrapping ---
    print_blue('Computing average classification probability on the test set ...', bold=True)
    test_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob(test_preds_and_gt)
    
    # --- Compute CNR on the test set with bootstrapping ---
    print_blue('Computing CNR on the test set ...', bold=True)
    test_cnrs, test_cnr_with_bootstrapping = _compute_cnr(
        preds_and_gt=test_preds_and_gt,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )

    # --- Compute Soft Dice on the test set with bootstrapping ---
    print_blue('Computing Soft Dice on the test set ...', bold=True)
    test_soft_dices, test_soft_dice_with_bootstrapping = _compute_soft_dice(
        preds_and_gt=test_preds_and_gt,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )

    # --- Compute Dice on the test set with bootstrapping ---
    print_blue('Computing Dice on the test set ...', bold=True)
    test_dices, test_dice_with_bootstrapping = _compute_dice(
        preds_and_gt=test_preds_and_gt,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Compute Rand Index on the test set with bootstrapping ---
    print_blue('Computing Rand Index on the test set ...', bold=True)
    test_rand_indices, test_rand_index_with_bootstrapping = _compute_rand_index(
        preds_and_gt=test_preds_and_gt,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Compute Precision on the test set with bootstrapping ---
    print_blue('Computing Precision on the test set ...', bold=True)
    test_precisions, test_precision_with_bootstrapping = _compute_precision(
        preds_and_gt=test_preds_and_gt,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )    

    # --- Compute Recall on the test set with bootstrapping ---
    print_blue('Computing Recall on the test set ...', bold=True)
    test_recalls, test_recall_with_bootstrapping = _compute_recall(
        preds_and_gt=test_preds_and_gt,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Save metrics to file ---
    print_blue('Saving metrics to file ...', bold=True)
    save_path = predictions_and_gt_filepath + '.metrics.pkl' # append .metrics.pkl to the predictions_and_gt_filepath
    output = dict(
        test_metrics=dict(
            best_prob_conf_th=test_best_prob_conf_th,
            best_bbox_iou_th=test_best_bbox_iou_th,
            best_bbox_conf_th=test_best_bbox_conf_th,
            best_bbox_pre_nms_max_det=test_best_bbox_pre_nms_max_det,
            best_bbox_post_nms_max_det=test_best_bbox_post_nms_max_det,
            prob_ious=test_prob_ious,
            prob_iou_with_bootstrapping=test_prob_iou_with_bootstrapping,
            bbox_ious=test_bbox_ious,
            bbox_iou_with_bootstrapping=test_bbox_iou_with_bootstrapping,
            avg_classification_prob_with_bootstrappig=test_avg_classification_prob_with_bootstrappig,
            cnrs=test_cnrs,
            cnr_with_bootstrapping=test_cnr_with_bootstrapping,
            soft_dices=test_soft_dices,
            soft_dice_with_bootstrapping=test_soft_dice_with_bootstrapping,
            dices=test_dices,
            dice_with_bootstrapping=test_dice_with_bootstrapping,
            rand_indices=test_rand_indices,
            rand_index_with_bootstrapping=test_rand_index_with_bootstrapping,
            precisions=test_precisions,
            precision_with_bootstrapping=test_precision_with_bootstrapping,
            recalls=test_recalls,
            recall_with_bootstrapping=test_recall_with_bootstrapping,
        ),
    )
    save_pickle(output, save_path)
    print_bold(f'Saved metrics to {save_path}')


def compute_and_save_metrics_on_vindrcxr(
    predictions_and_gt_filepath: str,
    candidate_iou_thresholds: list[float],
):
    assert candidate_iou_thresholds is not None

    # Load predictions and ground truth
    print(f'Loading predictions and ground truth from {predictions_and_gt_filepath} ...')
    preds_and_gt = load_pickle(predictions_and_gt_filepath)
    test_preds_and_gt = preds_and_gt['test_preds_and_gt']
    bbox_format = preds_and_gt['bbox_format']

    class_names = sorted(list(set(preds_and_gt['test_preds_and_gt']['classes'])))
    print(f'Found {len(class_names)} unique classes: {class_names}')
    class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    test_class_name_to_indices = [[] for _ in range(len(class_names))]
    for i, class_name in enumerate(test_preds_and_gt['classes']):
        class_idx = class_name_to_idx[class_name]
        test_class_name_to_indices[class_idx].append(i)

    # --- Compute IoU based on bbox probability maps on test set with bootstrapping ---
    print_blue('Computing IoU based on bbox probability maps on test set ...', bold=True)
    tmp = find_optimal_probability_map_conf_threshold(
        prob_maps=np.array(test_preds_and_gt['pred_bbox_prob_maps']), # (N, H, W)
        gt_bboxes_list=test_preds_and_gt['gt_bbox_coords'],
        bbox_format=bbox_format,
    )
    test_best_prob_conf_th = tmp['best_conf_th']
    print(f'test_best_prob_conf_th = {test_best_prob_conf_th}')
    test_prob_ious, test_prob_iou_with_bootstrapping = _compute_probability_map_iou_on_mscxr( # reuse the same function as MS-CXR
        split='test',
        bbox_format=bbox_format,
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        threshold=test_best_prob_conf_th,
    )

    # --- Compute IoU based on bbox coordinates on test set with bootstrapping ---
    print_blue('Computing IoU based on bbox coordinates on test set ...', bold=True)
    (
        test_best_bbox_iou_th, test_best_bbox_conf_th, test_best_bbox_pre_nms_max_det,
        test_best_bbox_post_nms_max_det, test_bbox_ious, test_bbox_iou_with_bootstrapping
    ) = _compute_bbox_iou_on_mscxr( # reuse the same function as MS-CXR
        split='test',
        preds_and_gt=test_preds_and_gt,
        class_to_indices=test_class_name_to_indices,
        category_names=class_names,
        candidate_iou_thresholds=candidate_iou_thresholds,
        candidate_conf_thresholds=[test_best_prob_conf_th], # use the best prob conf th as the only candidate conf threshold
        bbox_format=bbox_format,
    )    

    # --- Compute average classification probability on test set with bootstrapping ---
    print_blue('Computing average classification probability on test set ...', bold=True)
    test_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob_on_mscxr(
        'test', class_names, test_preds_and_gt, test_class_name_to_indices)
    
    # --- Compute CNR on test set with bootstrapping ---
    print_blue('Computing CNR on test set ...', bold=True)
    test_cnrs, test_cnr_with_bootstrapping = _compute_cnr(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )

    # --- Compute Soft Dice on test set with bootstrapping ---
    print_blue('Computing Soft Dice on test set ...', bold=True)
    test_soft_dices, test_soft_dice_with_bootstrapping = _compute_soft_dice(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )

    # --- Compute Dice on test set with bootstrapping ---
    print_blue('Computing Dice on test set ...', bold=True)
    test_dices, test_dice_with_bootstrapping = _compute_dice(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Compute Rand Index on test set with bootstrapping ---
    print_blue('Computing Rand Index test set ...', bold=True)
    test_rand_indices, test_rand_index_with_bootstrapping = _compute_rand_index(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Compute Precision on test set with bootstrapping ---
    print_blue('Computing Precision on test set ...', bold=True)
    test_precisions, test_precision_with_bootstrapping = _compute_precision(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )    

    # --- Compute Recall on test set with bootstrapping ---
    print_blue('Computing Recall on test set ...', bold=True)
    test_recalls, test_recall_with_bootstrapping = _compute_recall(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Save metrics to file ---
    print_blue('Saving metrics to file ...', bold=True)
    save_path = predictions_and_gt_filepath + '.metrics.pkl' # append .metrics.pkl to the predictions_and_gt_filepath
    output = dict(
        test_metrics=dict(
            best_prob_conf_th=test_best_prob_conf_th,
            best_bbox_iou_th=test_best_bbox_iou_th,
            best_bbox_conf_th=test_best_bbox_conf_th,
            best_bbox_pre_nms_max_det=test_best_bbox_pre_nms_max_det,
            best_bbox_post_nms_max_det=test_best_bbox_post_nms_max_det,
            prob_ious=test_prob_ious,
            prob_iou_with_bootstrapping=test_prob_iou_with_bootstrapping,
            bbox_ious=test_bbox_ious,
            bbox_iou_with_bootstrapping=test_bbox_iou_with_bootstrapping,
            avg_classification_prob_with_bootstrappig=test_avg_classification_prob_with_bootstrappig,
            cnrs=test_cnrs,
            cnr_with_bootstrapping=test_cnr_with_bootstrapping,
            soft_dices=test_soft_dices,
            soft_dice_with_bootstrapping=test_soft_dice_with_bootstrapping,
            dices=test_dices,
            dice_with_bootstrapping=test_dice_with_bootstrapping,
            rand_indices=test_rand_indices,
            rand_index_with_bootstrapping=test_rand_index_with_bootstrapping,
            precisions=test_precisions,
            precision_with_bootstrapping=test_precision_with_bootstrapping,
            recalls=test_recalls,
            recall_with_bootstrapping=test_recall_with_bootstrapping,
        ),
    )
    save_pickle(output, save_path)
    print_bold(f'Saved metrics to {save_path}')


def compute_and_save_metrics_on_chest_imagenome(
    predictions_and_gt_filepath: str,
    candidate_conf_thresholds: list[float],
    candidate_iou_thresholds: list[float],
    num_samples_per_class: int = 300, # To make the computation faster, we only sample a subset of the data
):
    assert candidate_iou_thresholds is not None

    # Load predictions and ground truth
    print(f'Loading predictions and ground truth from {predictions_and_gt_filepath} ...')
    preds_and_gt = load_pickle(predictions_and_gt_filepath)
    test_preds_and_gt = preds_and_gt['test_preds_and_gt']
    bbox_format = preds_and_gt['bbox_format']

    class_names = sorted(list(set(preds_and_gt['test_preds_and_gt']['classes'])))
    print(f'Found {len(class_names)} unique classes: {class_names}')
    class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    test_class_name_to_indices = [[] for _ in range(len(class_names))]
    for i, class_name in enumerate(test_preds_and_gt['classes']):
        class_idx = class_name_to_idx[class_name]
        test_class_name_to_indices[class_idx].append(i)

    test_preds_and_gt['gt_bbox_coords'] = [[x] for x in test_preds_and_gt['gt_bbox_coords']] # convert to list of lists for compatibility

    # Sample a subset of the data for faster computation
    sampled_indices = []
    test_class_name_to_indices_ = [[] for _ in range(len(class_names))]
    for class_idx, indices in enumerate(test_class_name_to_indices):
        offset = len(sampled_indices)
        if len(indices) > num_samples_per_class:
            sampled_indices.extend(random.sample(indices, num_samples_per_class))
        else:
            sampled_indices.extend(indices)
        test_class_name_to_indices_[class_idx] = [i + offset for i in range(len(sampled_indices) - offset)]
    test_class_name_to_indices = test_class_name_to_indices_ # update the indices to the sampled ones
    print(f'Sampled {len(sampled_indices)} indices from the test set.')
    for key in [
        'pred_bbox_prob_maps', 'pred_bbox_coord_maps', 'pred_classification_probs',
        'gt_bbox_coords', 'phrases', 'classes', 'image_paths'
    ]:
        test_preds_and_gt[key] = [test_preds_and_gt[key][i] for i in sampled_indices]

    # --- Compute IoU based on bbox probability maps on test set with bootstrapping ---
    print_blue('Computing IoU based on bbox probability maps on test set ...', bold=True)
    tmp = find_optimal_probability_map_conf_threshold(
        prob_maps=np.array(test_preds_and_gt['pred_bbox_prob_maps']), # (N, H, W)
        gt_bboxes_list=test_preds_and_gt['gt_bbox_coords'],
        bbox_format=bbox_format,
    )
    test_best_prob_conf_th = tmp['best_conf_th']
    print(f'test_best_prob_conf_th = {test_best_prob_conf_th}')
    test_prob_ious, test_prob_iou_with_bootstrapping = _compute_probability_map_iou_on_mscxr( # reuse the same function as MS-CXR
        split='test',
        bbox_format=bbox_format,
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        threshold=test_best_prob_conf_th,
    )
    if candidate_conf_thresholds is None:
        candidate_conf_thresholds = [
            test_best_prob_conf_th, # Use the best confidence threshold as a candidate
        ]

    print(f'Using candidate confidence thresholds: {candidate_conf_thresholds}')

    # --- Compute IoU based on bbox coordinates on test set with bootstrapping ---
    print_blue('Computing IoU based on bbox coordinates on test set ...', bold=True)
    (
        test_best_bbox_iou_th, test_best_bbox_conf_th, test_best_bbox_pre_nms_max_det,
        test_best_bbox_post_nms_max_det, test_bbox_ious, test_bbox_iou_with_bootstrapping
    ) = _compute_bbox_iou_on_mscxr( # reuse the same function as MS-CXR
        split='test',
        preds_and_gt=test_preds_and_gt,
        class_to_indices=test_class_name_to_indices,
        category_names=class_names,
        candidate_iou_thresholds=candidate_iou_thresholds,
        candidate_conf_thresholds=candidate_conf_thresholds,
        bbox_format=bbox_format,
    )    

    # --- Compute average classification probability on test set with bootstrapping ---
    print_blue('Computing average classification probability on test set ...', bold=True)
    test_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob_on_mscxr(
        'test', class_names, test_preds_and_gt, test_class_name_to_indices)
    
    # --- Compute CNR on test set with bootstrapping ---
    print_blue('Computing CNR on test set ...', bold=True)
    test_cnrs, test_cnr_with_bootstrapping = _compute_cnr(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )

    # --- Compute Soft Dice on test set with bootstrapping ---
    print_blue('Computing Soft Dice on test set ...', bold=True)
    test_soft_dices, test_soft_dice_with_bootstrapping = _compute_soft_dice(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100)
    )

    # --- Compute Dice on test set with bootstrapping ---
    print_blue('Computing Dice on test set ...', bold=True)
    test_dices, test_dice_with_bootstrapping = _compute_dice(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Compute Rand Index on test set with bootstrapping ---
    print_blue('Computing Rand Index test set ...', bold=True)
    test_rand_indices, test_rand_index_with_bootstrapping = _compute_rand_index(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Compute Precision on test set with bootstrapping ---
    print_blue('Computing Precision on test set ...', bold=True)
    test_precisions, test_precision_with_bootstrapping = _compute_precision(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )    

    # --- Compute Recall on test set with bootstrapping ---
    print_blue('Computing Recall on test set ...', bold=True)
    test_recalls, test_recall_with_bootstrapping = _compute_recall(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        bbox_format=bbox_format,
        mask_resolution=(100, 100),
        threshold=test_best_prob_conf_th,
    )

    # --- Save metrics to file ---
    print_blue('Saving metrics to file ...', bold=True)
    save_path = predictions_and_gt_filepath + '.metrics.pkl' # append .metrics.pkl to the predictions_and_gt_filepath
    output = dict(
        test_metrics=dict(
            sampled_indices=sampled_indices, # Keep track of the sampled indices for reproducibility
            best_prob_conf_th=test_best_prob_conf_th,
            best_bbox_iou_th=test_best_bbox_iou_th,
            best_bbox_conf_th=test_best_bbox_conf_th,
            best_bbox_pre_nms_max_det=test_best_bbox_pre_nms_max_det,
            best_bbox_post_nms_max_det=test_best_bbox_post_nms_max_det,
            prob_ious=test_prob_ious,
            prob_iou_with_bootstrapping=test_prob_iou_with_bootstrapping,
            bbox_ious=test_bbox_ious,
            bbox_iou_with_bootstrapping=test_bbox_iou_with_bootstrapping,
            avg_classification_prob_with_bootstrappig=test_avg_classification_prob_with_bootstrappig,
            cnrs=test_cnrs,
            cnr_with_bootstrapping=test_cnr_with_bootstrapping,
            soft_dices=test_soft_dices,
            soft_dice_with_bootstrapping=test_soft_dice_with_bootstrapping,
            dices=test_dices,
            dice_with_bootstrapping=test_dice_with_bootstrapping,
            rand_indices=test_rand_indices,
            rand_index_with_bootstrapping=test_rand_index_with_bootstrapping,
            precisions=test_precisions,
            precision_with_bootstrapping=test_precision_with_bootstrapping,
            recalls=test_recalls,
            recall_with_bootstrapping=test_recall_with_bootstrapping,
        ),
    )
    save_pickle(output, save_path)
    print_bold(f'Saved metrics to {save_path}')


def compute_and_save_metrics_on_chest_x_det(
    predictions_and_gt_filepath: str,
):
    # Load predictions and ground truth
    print(f'Loading predictions and ground truth from {predictions_and_gt_filepath} ...')
    preds_and_gt = load_pickle(predictions_and_gt_filepath)
    train_preds_and_gt = preds_and_gt['train_preds_and_gt']
    test_preds_and_gt = preds_and_gt['test_preds_and_gt']
    train_gt_masks = train_preds_and_gt['gt_mask']
    test_gt_masks = test_preds_and_gt['gt_mask']

    class_names = sorted(list(set(preds_and_gt['train_preds_and_gt']['classes'])))
    class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def _get_class_name_to_indices(class_names, classes):
        class_name_to_indices = [[] for _ in range(len(class_names))]
        for i, class_name in enumerate(classes):
            class_idx = class_name_to_idx[class_name]
            class_name_to_indices[class_idx].append(i)
        return class_name_to_indices
    
    train_class_name_to_indices = _get_class_name_to_indices(class_names, train_preds_and_gt['classes'])
    test_class_name_to_indices = _get_class_name_to_indices(class_names, test_preds_and_gt['classes'])
    

    # --- Compute average classification probability on train and test sets with bootstrapping ---
    print_blue('Computing average classification probability on train and test sets ...', bold=True)
    # Train
    train_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob_on_mscxr(
        'train', class_names, train_preds_and_gt, train_class_name_to_indices)
    # Test
    test_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob_on_mscxr(
        'test', class_names, test_preds_and_gt, test_class_name_to_indices)
    

    # --- Compute IoU based on bbox probability maps on train and test sets with bootstrapping ---
    print_blue('Computing IoU based on bbox probability maps on train and test sets ...', bold=True)
    # Train
    tmp = find_optimal_probability_map_conf_threshold_for_masks(
        prob_maps=np.array(train_preds_and_gt['pred_bbox_prob_maps']), # (N, H, W)
        gt_masks=train_gt_masks,
    )
    train_best_prob_conf_th = tmp['best_conf_th']
    print(f'train_best_prob_conf_th = {train_best_prob_conf_th}')
    train_prob_ious, train_prob_iou_with_bootstrapping = _compute_segmentation_iou(
        split='train',        
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_class_name_to_indices,
        category_names=class_names,
        gt_masks_list=train_gt_masks,
        threshold=train_best_prob_conf_th,
    )
    # Test
    tmp = find_optimal_probability_map_conf_threshold_for_masks(
        prob_maps=np.array(test_preds_and_gt['pred_bbox_prob_maps']), # (N, H, W)
        gt_masks=test_gt_masks,
    )
    test_best_prob_conf_th = tmp['best_conf_th']
    print(f'test_best_prob_conf_th = {test_best_prob_conf_th}')
    test_prob_ious, test_prob_iou_with_bootstrapping = _compute_segmentation_iou(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        gt_masks_list=test_gt_masks,
        threshold=test_best_prob_conf_th,
    )
    
    # --- Compute CNR on train and test sets with bootstrapping ---
    print_blue('Computing CNR on train and test sets ...', bold=True)
    train_cnrs, train_cnr_with_bootstrapping = _compute_cnr(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_class_name_to_indices,
        category_names=class_names,
        gt_masks_list=train_gt_masks,
    )
    test_cnrs, test_cnr_with_bootstrapping = _compute_cnr(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        gt_masks_list=test_gt_masks,
    )

    # --- Compute Soft Dice on train and test sets with bootstrapping ---
    print_blue('Computing Soft Dice on train and test sets ...', bold=True)
    train_soft_dices, train_soft_dice_with_bootstrapping = _compute_soft_dice(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_class_name_to_indices,
        category_names=class_names,
        gt_masks_list=train_gt_masks,
    )
    test_soft_dices, test_soft_dice_with_bootstrapping = _compute_soft_dice(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        gt_masks_list=test_gt_masks,
    )

    # --- Compute Dice on train and test sets with bootstrapping ---
    print_blue('Computing Dice on train and test sets ...', bold=True)
    train_dices, train_dice_with_bootstrapping = _compute_dice(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_class_name_to_indices,
        category_names=class_names,
        threshold=train_best_prob_conf_th,
        gt_masks_list=train_gt_masks,
    )
    test_dices, test_dice_with_bootstrapping = _compute_dice(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        threshold=test_best_prob_conf_th,
        gt_masks_list=test_gt_masks,
    )

    # --- Compute Rand Index on train and test sets with bootstrapping ---
    print_blue('Computing Rand Index on train and test sets ...', bold=True)
    train_rand_indices, train_rand_index_with_bootstrapping = _compute_rand_index(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_class_name_to_indices,
        category_names=class_names,
        threshold=train_best_prob_conf_th,
        gt_masks_list=train_gt_masks,
    )
    test_rand_indices, test_rand_index_with_bootstrapping = _compute_rand_index(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        threshold=test_best_prob_conf_th,
        gt_masks_list=test_gt_masks,
    )

    # --- Compute Precision on train and test sets with bootstrapping ---
    print_blue('Computing Precision on train and test sets ...', bold=True)
    train_precisions, train_precision_with_bootstrapping = _compute_precision(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_class_name_to_indices,
        category_names=class_names,
        threshold=train_best_prob_conf_th,
        gt_masks_list=train_gt_masks,
    )
    test_precisions, test_precision_with_bootstrapping = _compute_precision(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names=class_names,
        threshold=test_best_prob_conf_th,
        gt_masks_list=test_gt_masks,
    )    

    # --- Compute Recall on train and test sets with bootstrapping ---
    print_blue('Computing Recall on train and test sets ...', bold=True)
    train_recalls, train_recall_with_bootstrapping = _compute_recall(
        split='train',
        preds_and_gt=train_preds_and_gt,
        category_to_indices=train_class_name_to_indices,
        category_names=class_names,
        threshold=train_best_prob_conf_th,
        gt_masks_list=train_gt_masks,
    )
    test_recalls, test_recall_with_bootstrapping = _compute_recall(
        split='test',
        preds_and_gt=test_preds_and_gt,
        category_to_indices=test_class_name_to_indices,
        category_names= class_names,
        threshold=test_best_prob_conf_th,
        gt_masks_list=test_gt_masks,
    )

    # --- Save metrics to file ---
    print_blue('Saving metrics to file ...', bold=True)
    save_path = predictions_and_gt_filepath + '.metrics.pkl' # append .metrics.pkl to the predictions_and_gt_filepath
    output = dict(
        train_metrics=dict(
            best_prob_conf_th=train_best_prob_conf_th,
            prob_ious=train_prob_ious,
            prob_iou_with_bootstrapping=train_prob_iou_with_bootstrapping,
            avg_classification_prob_with_bootstrappig=train_avg_classification_prob_with_bootstrappig,
            cnrs=train_cnrs,
            cnr_with_bootstrapping=train_cnr_with_bootstrapping,
            soft_dices=train_soft_dices,
            soft_dice_with_bootstrapping=train_soft_dice_with_bootstrapping,
            dices=train_dices,
            dice_with_bootstrapping=train_dice_with_bootstrapping,
            rand_indices=train_rand_indices,
            rand_index_with_bootstrapping=train_rand_index_with_bootstrapping,
            precisions=train_precisions,
            precision_with_bootstrapping=train_precision_with_bootstrapping,
            recalls=train_recalls,
            recall_with_bootstrapping=train_recall_with_bootstrapping,
        ),
        test_metrics=dict(
            best_prob_conf_th=test_best_prob_conf_th,
            prob_ious=test_prob_ious,
            prob_iou_with_bootstrapping=test_prob_iou_with_bootstrapping,
            avg_classification_prob_with_bootstrappig=test_avg_classification_prob_with_bootstrappig,
            cnrs=test_cnrs,
            cnr_with_bootstrapping=test_cnr_with_bootstrapping,
            soft_dices=test_soft_dices,
            soft_dice_with_bootstrapping=test_soft_dice_with_bootstrapping,
            dices=test_dices,
            dice_with_bootstrapping=test_dice_with_bootstrapping,
            rand_indices=test_rand_indices,
            rand_index_with_bootstrapping=test_rand_index_with_bootstrapping,
            precisions=test_precisions,
            precision_with_bootstrapping=test_precision_with_bootstrapping,
            recalls=test_recalls,
            recall_with_bootstrapping=test_recall_with_bootstrapping,
        ),
    )
    save_pickle(output, save_path)
    print_bold(f'Saved metrics to {save_path}')



def image_to_binary(image_pil, format="PNG"):
    byte_arr = io.BytesIO()
    image_pil.save(byte_arr, format=format)
    return byte_arr.getvalue()

def export_data_for_mscxr(bbox_format: str, output_filepath: str):
    from PIL import Image

    from medvqa.datasets.mimiccxr import (
        MIMICCXR_DicomIdToImagePath,
        MIMICCXR_ImageSizeModes,
    )
    from medvqa.datasets.ms_cxr import (
        get_ms_cxr_dicom_id_2_phrases_and_bboxes,
        get_ms_cxr_dicom_id_2_split,
    )
    dicom_id_2_phrases_and_bboxes = get_ms_cxr_dicom_id_2_phrases_and_bboxes(bbox_format=bbox_format)
    dicom_id_2_split = get_ms_cxr_dicom_id_2_split()
    dicom_id_2_image_path = MIMICCXR_DicomIdToImagePath(image_size_mode=MIMICCXR_ImageSizeModes.MEDIUM_512)
    data_to_export = []
    for dicom_id, (phrases, bboxes) in dicom_id_2_phrases_and_bboxes.items():
        split = dicom_id_2_split[dicom_id]
        if split == 'train':
            continue # Skip training data
        image_path = dicom_id_2_image_path(dicom_id)
        image = Image.open(image_path)
        image_binary = image_to_binary(image, format="JPEG")
        data_to_export.append({
            'dicom_id': dicom_id,
            'image_path': image_path,
            'image': image_binary,
            'phrases': phrases,
            'bboxes': bboxes,
            'split': split,
        })
    save_pickle(data_to_export, output_filepath)
    print(f'Exported {len(data_to_export)} images to {output_filepath}')


def export_data_for_padchest_gr(bbox_format: str, output_filepath: str):
    """
    Exports PadChest-GR image-phrase grounding data to a pickle file.
    Only includes test split, and stores images as binary data.

    Args:
        bbox_format (str): The desired bounding box format ('cxcywh' or 'xyxy').
        output_filepath (str): The path to save the pickle file.
    """
    from PIL import Image

    from medvqa.datasets.padchest import (
        PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH,
        PADCHEST_GR_JPG_DIR,
        PADCHEST_GR_MASTER_TABLE_CSV_PATH,
    )

    # --- Load and Filter Metadata (from Dataset __init__) ---
    print(f"Loading master CSV from: {PADCHEST_GR_MASTER_TABLE_CSV_PATH}")
    df = pd.read_csv(PADCHEST_GR_MASTER_TABLE_CSV_PATH)

    # Load only test data
    allowed_splits = ["test"]
    print(f"Filtering dataset for splits: {allowed_splits}")
    df = df[df['split'].isin(allowed_splits)]

    if df.empty:
        raise ValueError(f"No data found for splits {allowed_splits} in {PADCHEST_GR_MASTER_TABLE_CSV_PATH}")
    study_ids_in_splits = df['StudyID'].unique().tolist()
    print(
        f"Found {len(study_ids_in_splits)} unique studies for splits '{allowed_splits}'"
    )

    # --- Load Reports Data ---
    print(f"Loading reports JSON from: {PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH}")
    reports_json_list = load_json(PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH)
    reports_data = {item['StudyID']: item for item in reports_json_list}
    print(f"Loaded {len(reports_data)} reports from JSON.")

    # --- Pre-process Data into Image-Phrase Pairs ---
    print("Processing data into image-phrase pairs for export...")

    data_to_export: List[Dict[str, Any]] = []
    language = "en"
    image_format = "jpg"
    lang_key = f'sentence_{language}'

    for study_id in tqdm(study_ids_in_splits, desc="Processing Studies for Export"):
        report_info = reports_data.get(study_id)
        if report_info is None:
            print_orange(f"WARNING: StudyID {study_id} not found in reports_data. Skipping.")
            continue

        image_id = report_info['ImageID']
        base_name, _ = os.path.splitext(image_id)
        image_id_with_ext = f"{base_name}.{image_format}"
        image_path = os.path.join(PADCHEST_GR_JPG_DIR, image_id_with_ext)

        if not os.path.exists(image_path):
            print_orange(f"WARNING: Image not found: {image_path}. Skipping study {study_id}.")
            continue

        # Load image and convert to binary *once per image*
        pil_image = Image.open(image_path)
        image_binary = image_to_binary(pil_image, format="JPEG") # Use JPEG for export

        # Process findings for this report
        phrases = []
        bboxes = []

        findings = report_info.get('findings', [])
        for finding in findings:
            # Clean sentence
            sentence_text = finding.get(lang_key, "")
            sentence_text = sentence_text.strip()
            if sentence_text.endswith('.'):
                sentence_text = sentence_text[:-1]
            if not sentence_text: # Ensure not empty
                continue                

            original_boxes_xyxy = finding.get('boxes')
            if not original_boxes_xyxy: # Must have boxes
                continue
            # Convert boxes to the required format for export
            if bbox_format == "cxcywh":
                processed_boxes = [xyxy_to_cxcywh(box) for box in original_boxes_xyxy]
            else: # xyxy
                processed_boxes = original_boxes_xyxy

            phrases.append(sentence_text)
            bboxes.append(processed_boxes)

        # Check if we have any phrases and bboxes to export
        if not phrases or not bboxes:
            continue

        data_to_export.append({
            'study_id': study_id,
            'image_id': image_id,
            'image_path': image_path,
            'image': image_binary, # Binary image data
            'phrases': phrases, # List of phrases for this study
            'bboxes': bboxes, # List of lists of bounding boxes
            'split': df[df['StudyID'] == study_id]['split'].iloc[0], # Get the actual split
        })

    if not data_to_export:
        raise ValueError(
            f"No grounded phrases found for splits {allowed_splits} "
            f"with language '{language}'. Check data and paths."
        )

    save_pickle(data_to_export, output_filepath)
    print(f'Exported {len(data_to_export)} data entries to {output_filepath}')


def export_data_for_vindrcxr(bbox_format: str, output_filepath: str):
    """
    Exports VinDr-CXR image-phrase grounding data to a pickle file.
    Only includes test split, and stores images as binary data.
    Phrases are derived from bounding box class names.

    Args:
        bbox_format (str): The desired bounding box format ('cxcywh' or 'xyxy').
        output_filepath (str): The path to save the pickle file.
    """
    from PIL import Image

    from medvqa.datasets.vinbig import (
        get_medium_size_image_path,
        load_labels,
        load_test_image_id_2_bboxes,
    )
    from medvqa.utils.constants import VINBIG_BBOX_NAMES

    assert bbox_format in ["cxcywh", "xyxy"], \
        f"Invalid bbox_format '{bbox_format}'. Must be one of ['cxcywh', 'xyxy']."

    # --- Load Test Image IDs and their Paths ---
    print("Loading image-level labels to get test image IDs...")
    _, test_image_id_to_labels = load_labels()
    test_image_ids = list(test_image_id_to_labels.keys())
    
    print(f"Found {len(test_image_ids)} unique test image IDs.")

    print("Determining image paths for test images...")
    image_id_to_path = {img_id: get_medium_size_image_path(img_id) for img_id in test_image_ids}
    
    # --- Load Bounding Boxes for Test Set ---
    print("Loading bounding boxes for the test set...")
    # This function returns (bbox_list, class_id_list) for each image_id
    image_id_2_bboxes_data = load_test_image_id_2_bboxes(
        for_training=True, normalize=True, bbox_format=bbox_format,
    )
    
    print("Processing data into image-phrase pairs for export...")

    data_to_export: List[Dict[str, Any]] = []

    for image_id in tqdm(test_image_ids, desc="Processing Images for Export"):
        image_path = image_id_to_path.get(image_id)
        if not image_path or not os.path.exists(image_path):
            print_orange(f"WARNING: Image not found: {image_path or image_id}. Skipping.")
            continue

        # Load image and convert to binary *once per image*
        pil_image = Image.open(image_path)
        image_binary = image_to_binary(pil_image, format="JPEG") # Use JPEG for export

        # Get bounding box data for this image
        # image_id_2_bboxes_data returns (bbox_coords_list, class_ids_list)
        bbox_coords_list, class_ids_list = image_id_2_bboxes_data.get(image_id, ([], []))

        # Group bounding boxes by their class name to form phrases and associated boxes
        # This creates a structure where one phrase (class name) might be associated
        # with multiple bounding boxes if that class appears multiple times in the image.
        phrases_to_bboxes_map = defaultdict(list)
        for bbox_coords, class_id in zip(bbox_coords_list, class_ids_list):
            phrase = VINBIG_BBOX_NAMES[class_id]
            phrases_to_bboxes_map[phrase].append(bbox_coords)
        
        # Now, flatten this map into the desired export format
        # Each unique phrase with its associated list of bounding boxes becomes an entry
        # If an image has 3 "Cardiomegaly" bboxes and 1 "Aortic Enlargement" bbox,
        # it will create two entries: one for "Cardiomegaly" (with 3 bboxes) and one
        # for "Aortic Enlargement" (with 1 bbox), each pointing to the same binary image.
        
        # If there are no grounded phrases (i.e., no bounding boxes), skip this image
        if not phrases_to_bboxes_map:
            continue

        phrases = []
        bboxes = []
        for phrase, bboxes_for_phrase in phrases_to_bboxes_map.items():
            assert len(bboxes_for_phrase) > 0
            phrases.append(phrase)
            bboxes.append(bboxes_for_phrase)

        data_to_export.append({
            'image_id': image_id,
            'image_path': image_path,
            'image': image_binary, # Binary image data
            'phrases': phrases, # List of phrases for this image
            'bboxes': bboxes, # List of lists of bounding boxes
            'split': 'test', # Hardcoded as we only process test split
        })

    if not data_to_export:
        raise ValueError(
            "No grounded phrases (bounding boxes) found for the test split. "
            "Check data and paths."
        )

    save_pickle(data_to_export, output_filepath)
    print(f'Exported {len(data_to_export)} data entries to {output_filepath}')


def export_data_for_chest_imagenome(
        bbox_format: str,
        chest_imagenome_augmented_phrase_groundings_filepath: str,
        output_filepath: str
    ):
    """
    Exports Chest ImaGenome anatomical location grounding data to a pickle file.
    Only includes the test split, and stores images as binary data.
    Bounding boxes are cleaned and validated before export.
    Phrases are the anatomical locations (e.g., "right lung").

    Args:
        bbox_format (str): The desired bounding box format ('cxcywh' or 'xyxy').
        chest_imagenome_augmented_phrase_groundings_filepath (str): The path to the Chest ImaGenome augmented phrase groundings file.
        output_filepath (str): The path to save the pickle file.
    """
    from PIL import Image

    from medvqa.datasets.mimiccxr import (
        MIMICCXR_DicomIdToImagePath,
        MIMICCXR_ImageSizeModes,
        get_split2imageIds,
    )

    assert bbox_format in ["cxcywh", "xyxy"], \
        f"Invalid bbox_format '{bbox_format}'. Must be one of ['cxcywh', 'xyxy']."

    # --- Load Splits and Filter for Test Set ---
    print("Loading Chest ImaGenome splits...")
    split_to_image_ids = get_split2imageIds()
    test_dicom_ids = set(split_to_image_ids['test'])
    print(f"Found {len(test_dicom_ids)} unique DICOM IDs in the test set.")

    # --- Load Augmented Data ---
    print(f"Loading Chest ImaGenome augmented data from: {chest_imagenome_augmented_phrase_groundings_filepath}")
    augmented_data = load_pickle(chest_imagenome_augmented_phrase_groundings_filepath)
    
    # Filter for test set
    test_augmented_data = [item for item in augmented_data if item['dicom_id'] in test_dicom_ids]
    print(f"Filtered to {len(test_augmented_data)} data entries for the test split.")

    # --- Initialize Image Path Mapper ---
    dicom_id_to_image_path = MIMICCXR_DicomIdToImagePath(
        image_size_mode=MIMICCXR_ImageSizeModes.MEDIUM_512
    )

    # --- Process Data for Export ---
    print("Processing data into image-phrase pairs for export...")
    data_to_export: List[Dict[str, Any]] = []
    total_skipped_bboxes = 0

    for item in tqdm(test_augmented_data, desc="Processing Images for Export"):
        dicom_id = item['dicom_id']
        
        # --- Bounding Box Cleaning Step ---
        cleaned_location2bbox = {}
        original_location2bbox = item.get('location2bbox', {})

        for loc_name, bbox in original_location2bbox.items():
            bbox = original_location2bbox[loc_name]
            x_min, y_min, x_max, y_max = bbox
            
            # Clamp coordinates to [0, 1] range
            x_min_c = max(0, min(1, x_min))
            y_min_c = max(0, min(1, y_min))
            x_max_c = max(0, min(1, x_max))
            y_max_c = max(0, min(1, y_max))
            
            # Check for invalid boxes (zero or negative area)
            if x_min_c >= x_max_c or y_min_c >= y_max_c:
                total_skipped_bboxes += 1
                continue
            
            cleaned_location2bbox[loc_name] = [x_min_c, y_min_c, x_max_c, y_max_c]
        
        # If after cleaning there are no bboxes left, skip this item
        if not cleaned_location2bbox:
            continue

        # --- Continue with Export Logic using Cleaned Data ---
        image_path = dicom_id_to_image_path(dicom_id)
        if not os.path.exists(image_path):
            print_orange(f"WARNING: Image not found: {image_path}. Skipping DICOM ID {dicom_id}.")
            continue

        pil_image = Image.open(image_path)
        image_binary = image_to_binary(pil_image, format="JPEG")

        # Group bounding boxes by their anatomical location name (phrase)
        phrases_to_bboxes_map = defaultdict(list)
        for phrase, bbox_coords_xyxy in cleaned_location2bbox.items():
            if bbox_format == "cxcywh":
                processed_bbox = xyxy_to_cxcywh(bbox_coords_xyxy)
            else: # xyxy
                processed_bbox = bbox_coords_xyxy
            phrases_to_bboxes_map[phrase].append(processed_bbox)

        phrases = list(phrases_to_bboxes_map.keys())
        bboxes = list(phrases_to_bboxes_map.values())

        data_to_export.append({
            'dicom_id': dicom_id,
            'image_path': image_path,
            'image': image_binary,
            'phrases': phrases,
            'bboxes': bboxes,
            'split': 'test',
        })

    if total_skipped_bboxes > 0:
        print_orange(f"WARNING: Skipped a total of {total_skipped_bboxes} bboxes due to invalid coordinates.")

    if not data_to_export:
        raise ValueError("No valid grounded anatomical locations found for the test split after cleaning.")

    save_pickle(data_to_export, output_filepath)
    print(f'Exported {len(data_to_export)} data entries to {output_filepath}')


def export_data_for_chest_x_det(
    bbox_format: str,
    test_json_path: str,
    test_image_dir: str,
    output_filepath: str,
):
    """
    Exports ChestX-Det data to a pickle file for phrase grounding.
    Only includes the test split, and stores images as binary data.
    Includes both bounding boxes and polygons.

    Args:
        bbox_format (str): The desired bounding box format ('cxcywh' or 'xyxy').
        test_json_path (str): Path to the ChestX-Det test set JSON file.
        test_image_dir (str): Path to the directory containing test images.
        output_filepath (str): The path to save the pickle file.
    """
    from PIL import Image

    from medvqa.datasets.chestxdet.chestxdet_phrase_grounding_dataset_management import (
        normalize_bbox,
        normalize_polygon,
    )

    assert bbox_format in ["cxcywh", "xyxy"], \
        f"Invalid bbox_format '{bbox_format}'. Must be one of ['cxcywh', 'xyxy']."

    # --- Load Test Data ---
    print(f"Loading ChestX-Det test data from: {test_json_path}")
    test_data = load_json(test_json_path)
    print(f"Found {len(test_data)} image entries in the JSON file.")

    # --- Process Data for Export ---
    print("Processing data into image-phrase-bbox-polygon entries for export...")
    data_to_export: List[Dict[str, Any]] = []

    for item in tqdm(test_data, desc="Processing Images for Export"):
        file_name = item['file_name']
        image_path = os.path.join(test_image_dir, file_name)

        if not os.path.exists(image_path):
            print_orange(f"WARNING: Image not found: {image_path}. Skipping.")
            continue

        # Load image, get its size, and convert to binary
        pil_image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        img_w, img_h = pil_image.size
        image_binary = image_to_binary(pil_image, format="JPEG")

        # Group bboxes and polygons by their label (phrase)
        phrases_to_bboxes_map = defaultdict(list)
        phrases_to_polygons_map = defaultdict(list)

        for sym, bbox, polygon in zip(item['syms'], item['boxes'], item['polygons']):
            # Normalize coordinates
            norm_bbox = normalize_bbox(bbox, img_w, img_h, bbox_format)
            norm_polygon = normalize_polygon(polygon, img_w, img_h)
            
            phrases_to_bboxes_map[sym].append(norm_bbox)
            phrases_to_polygons_map[sym].append(norm_polygon)

        # If there are no valid phrases/bboxes, skip
        if not phrases_to_bboxes_map:
            continue

        # Structure the data for this image: one entry per image
        phrases = list(phrases_to_bboxes_map.keys())
        # Ensure bboxes and polygons lists are aligned with the phrases list
        bboxes = [phrases_to_bboxes_map[p] for p in phrases]
        polygons = [phrases_to_polygons_map[p] for p in phrases]

        data_to_export.append({
            'file_name': file_name,
            'image_path': image_path,
            'image': image_binary,
            'phrases': phrases,
            'bboxes': bboxes,
            'polygons': polygons,
            'split': 'test',
        })

    if not data_to_export:
        raise ValueError("No valid data found to export from the test set.")

    save_pickle(data_to_export, output_filepath)
    print(f'Exported {len(data_to_export)} data entries to {output_filepath}')


# def _evaluate_model(
#     checkpoint_folder_path,
#     model_kwargs,
#     mimiccxr_trainer_kwargs,
#     chexlocalize_trainer_kwargs,
#     vinbig_trainer_kwargs,
#     val_image_transform_kwargs,
#     evaluation_engine_kwargs,
#     max_images_per_batch,
#     max_phrases_per_batch,
#     max_phrases_per_image,
#     num_workers,
#     eval_chest_imagenome_gold,
#     eval_mscxr,
#     eval_chexlocalize,
#     eval_vinbig,
#     mscxr_phrase2embedding_filepath,
#     mimicxr_dicom_id_to_pos_neg_facts_filepath,
#     device,
#     vinbig_use_training_indices_for_validation,
#     optimize_thresholds,
#     candidate_iou_thresholds,
#     candidate_conf_thresholds,
#     map_iou_thresholds,
#     use_amp,
#     use_classifier_confs_for_map,
#     checkpoint_folder_path_to_borrow_metadata_from,
#     override_bbox_format,
# ):
#     count_print = CountPrinter()
    
#     # Pull out some args from kwargs
#     use_yolov8 = (mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_yolov8', False))
#     use_fact_conditioned_yolo = model_kwargs['raw_image_encoding'] == RawImageEncoding.YOLOV11_FACT_CONDITIONED
#     do_visual_grounding_with_bbox_regression = evaluation_engine_kwargs.get('do_visual_grounding_with_bbox_regression', False)
#     do_visual_grounding_with_segmentation = evaluation_engine_kwargs.get('do_visual_grounding_with_segmentation', False)
#     print(f'do_visual_grounding_with_bbox_regression = {do_visual_grounding_with_bbox_regression}')
#     print(f'do_visual_grounding_with_segmentation = {do_visual_grounding_with_segmentation}')

#     # Sanity checks
#     assert sum([eval_chest_imagenome_gold, eval_mscxr, eval_chexlocalize, eval_vinbig]) > 0 # at least one dataset must be evaluated

#     # device
#     device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
#     count_print('device =', device)

#     # Create model
#     count_print('Creating instance of PhraseGrounder ...')
#     model = PhraseGrounder(**model_kwargs)
#     model = model.to(device)

#     # Load model from checkpoint
#     model_wrapper = ModelWrapper(model)
#     checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
#     count_print('Loading model from checkpoint ...')
#     print('checkpoint_path =', checkpoint_path)
#     model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True, strict=False)

#     # Create phrase grounding trainers

#     # if eval_chest_imagenome_gold or eval_mscxr:

#     #     if checkpoint_folder_path_to_borrow_metadata_from is not None:
#     #         metadata = load_metadata(checkpoint_folder_path_to_borrow_metadata_from)
#     #         # collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
#     #         mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
        
#     #     try: 
#     #         image_transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]
#     #     except KeyError:
#     #         image_transform_kwargs = next(iter(val_image_transform_kwargs.values())) # get the first value

#     #     count_print('Creating MIMIC-CXR Phrase Grounding Trainer ...')
#     #     # if eval_chest_imagenome_gold:
#     #     #     bbox_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cibg'])
#     #     # else:
#     #     #     bbox_grounding_collate_batch_fn = None
#     #     # if eval_mscxr:
#     #     #     mscxr_phrase_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['mscxr'])
#     #     # else:
#     #     #     mscxr_phrase_grounding_collate_batch_fn = None
#     #     mimiccxr_trainer_kwargs['use_facts_for_train'] = False
#     #     mimiccxr_trainer_kwargs['use_facts_for_test'] = False
#     #     mimiccxr_trainer_kwargs['use_mscxr_for_train'] = False
#     #     mimiccxr_trainer_kwargs['use_mscxr_for_val'] = False
#     #     mimiccxr_trainer_kwargs['use_mscxr_for_test'] = eval_mscxr
#     #     mimiccxr_trainer_kwargs['mscxr_test_on_all_images'] = eval_mscxr # if True, test on all MSCXR
#     #     mimiccxr_trainer_kwargs['use_cxrlt2024_challenge_split'] = False 
#     #     mimiccxr_trainer_kwargs['use_cxrlt2024_official_labels'] = False
#     #     mimiccxr_trainer_kwargs['use_cxrlt2024_custom_labels'] = False
#     #     mimiccxr_trainer_kwargs['use_chest_imagenome_for_train'] = False
#     #     mimiccxr_trainer_kwargs['use_chest_imagenome_gold_for_test'] = eval_chest_imagenome_gold
#     #     if mscxr_phrase2embedding_filepath is not None:
#     #         mimiccxr_trainer_kwargs['mscxr_phrase2embedding_filepath'] = mscxr_phrase2embedding_filepath
#     #     if override_bbox_format:
#     #         print_orange('Overriding bbox format to', override_bbox_format)
#     #         mimiccxr_trainer_kwargs['bbox_format'] = override_bbox_format

#     #     mimiccxr_trainer = MIMICCXR_PhraseGroundingTrainer(
#     #         test_image_transform = create_image_transforms(**image_transform_kwargs),
#     #         max_images_per_batch=max_images_per_batch,
#     #         max_phrases_per_batch=max_phrases_per_batch,
#     #         max_phrases_per_image=max_phrases_per_image,
#     #         # bbox_grounding_collate_batch_fn=bbox_grounding_collate_batch_fn,
#     #         # mscxr_phrase_grounding_collate_batch_fn=mscxr_phrase_grounding_collate_batch_fn,
#     #         num_test_workers=num_workers,
#     #         **mimiccxr_trainer_kwargs,
#     #     )

#     if eval_chexlocalize:

#         count_print('Creating CheXlocalize Phrase Grounding Trainer ...')
#         chexlocalize_trainer_kwargs['use_training_set'] = False
#         chexlocalize_trainer_kwargs['use_validation_set'] = True
#         chexlocalize_trainer = CheXlocalizePhraseGroundingTrainer(
#             val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE]),
#             collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cl']),
#             max_images_per_batch=max_images_per_batch,
#             max_phrases_per_batch=max_phrases_per_batch,
#             num_val_workers=num_workers,
#             **chexlocalize_trainer_kwargs,
#         )
    
#     if eval_vinbig:

#         count_print('Creating VinBig Phrase Grounding Trainer ...')
#         if vinbig_trainer_kwargs is None:
#             assert checkpoint_folder_path_to_borrow_metadata_from is not None
#             metadata = load_metadata(checkpoint_folder_path_to_borrow_metadata_from)
#             vinbig_trainer_kwargs = metadata['vinbig_trainer_kwargs']
#             try: 
#                 transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.VINBIG]
#             except KeyError:
#                 first_key = list(val_image_transform_kwargs.keys())[0]
#                 transform_kwargs = val_image_transform_kwargs[first_key]
#                 print_orange(f'Using transform_kwargs from {first_key} for VinBig')
#             try:
#                 collate_batch_fn_kwargs = collate_batch_fn_kwargs['vbg']
#             except KeyError:
#                 collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']['vbg']
#                 print_orange(f'Borrowing collate_batch_fn_kwargs from {checkpoint_folder_path_to_borrow_metadata_from} for VinBig')
#         else:
#             transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.VINBIG]
#             collate_batch_fn_kwargs = collate_batch_fn_kwargs['vbg']
#         vinbig_trainer_kwargs['use_training_set'] = False
#         vinbig_trainer_kwargs['use_validation_set'] = True
#         vinbig_trainer = VinBigPhraseGroundingTrainer(
#             val_image_transform=get_image_transform(**transform_kwargs),
#             collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs),
#             max_images_per_batch=max_images_per_batch,
#             max_phrases_per_batch=max_phrases_per_batch,
#             num_val_workers=num_workers,
#             use_training_indices_for_validation=vinbig_use_training_indices_for_validation,
#             **vinbig_trainer_kwargs,
#         )

#     # Evaluate on datasets
    
#     if eval_chest_imagenome_gold:

#         count_print('----- Evaluating on Chest ImaGenome Gold Bbox Phrase Grounding -----')

#         # Create evaluation engine
#         print_blue('Creating evaluation engine ...', bold=True)
#         evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

#         # Attach metrics
#         _cond_func = lambda x: x['flag'] == 'cibg'
#         if use_yolov8:
#             _gold_class_mask = get_chest_imagenome_gold_class_mask()
#             attach_condition_aware_chest_imagenome_bbox_iou(evaluation_engine, _cond_func, use_yolov8=True, class_mask=_gold_class_mask)
#         if do_visual_grounding_with_bbox_regression:
#             attach_condition_aware_bbox_iou_per_class(evaluation_engine,
#                                                       field_names=['predicted_bboxes', 'chest_imagenome_bbox_coords', 'chest_imagenome_bbox_presence'],
#                                                       metric_name='bbox_iou', nc=CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
#                                                       condition_function=_cond_func)
#         else:
#             attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
#                                                         nc=CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
#                                                         condition_function=_cond_func)
#         # Attach accumulators
#         if do_visual_grounding_with_bbox_regression:
#             attach_accumulator(evaluation_engine, 'predicted_bboxes')
#             attach_accumulator(evaluation_engine, 'chest_imagenome_bbox_coords')
#             attach_accumulator(evaluation_engine, 'chest_imagenome_bbox_presence')
#         else:
#             attach_accumulator(evaluation_engine, 'pred_mask')
#             attach_accumulator(evaluation_engine, 'gt_mask')

#         # for logging
#         metrics_to_print = []
#         if use_yolov8:
#             metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXIOU)
#         if do_visual_grounding_with_bbox_regression:
#             metrics_to_print.append('bbox_iou')
#         else:
#             metrics_to_print.append('segmask_iou')

#         # Timer
#         timer = Timer()
#         timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

#         # Logging
#         print_blue('Defining log_metrics_handler ...', bold=True)
#         log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
#         log_iteration_handler = get_log_iteration_handler()
        
#         # Attach handlers
#         evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
#         evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

#         # Start evaluation
#         print_blue('Running engine ...', bold=True)
#         evaluation_engine.run(mimiccxr_trainer.test_chest_imagenome_gold_dataloader)

#         # Print final metrics
#         print_blue('Final metrics:', bold=True)
#         metrics = evaluation_engine.state.metrics
#         if use_yolov8:
#             # 1) chest imagenome bbox iou
#             metric_name = MetricNames.CHESTIMAGENOMEBBOXIOU
#             print(f'{metric_name}: {metrics[metric_name]}')
#             print()
#         # 2) bbox iou / segmask iou
#         if do_visual_grounding_with_bbox_regression:
#             metric_name = 'bbox_iou'
#         else:
#             metric_name = 'segmask_iou'
#         print(f'{metric_name}:')
#         from tabulate import tabulate
#         table = []
#         for bbox_name, iou in zip(CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED, metrics[metric_name]):
#             table.append([bbox_name, iou])
#         print(tabulate(table, headers=['bbox_name', 'iou'], tablefmt='latex_raw'))

#         # Save metrics to file
#         dataset = mimiccxr_trainer.test_chest_imagenome_gold_dataset
#         image_paths = dataset.image_paths
#         phrases = mimiccxr_trainer.test_chest_imagenome_gold_bbox_phrases

#         if do_visual_grounding_with_bbox_regression:
#             pred_bboxes = metrics['predicted_bboxes']
#             gt_bboxes = metrics['chest_imagenome_bbox_coords']
#             gt_presence = metrics['chest_imagenome_bbox_presence']
#             assert len(image_paths) == len(pred_bboxes) == len(gt_bboxes) == len(gt_presence)
#             print_blue('Saving metrics to file ...', bold=True)
#             results_folder_path = get_results_folder_path(checkpoint_folder_path)
#             save_path = os.path.join(results_folder_path, f'chest_imagenome_gold_metrics_bbox_regression.pkl')
#             output = dict(
#                 image_paths=[],
#                 phrases=[],
#                 pred_bboxes=[],
#                 gt_bboxes=[],
#                 ious=[],
#                 bbox_iou=metrics['bbox_iou'],
#             )
#             for i in range(len(image_paths)):
#                 for j in range(len(pred_bboxes[i])):
#                     if gt_presence[i][j] == 1:
#                         if pred_bboxes[i][j] is None:
#                             iou = 0
#                         else:
#                             iou = compute_mean_bbox_union_iou(pred_bboxes[i][j][0], gt_bboxes[i][j])
#                         output['image_paths'].append(image_paths[i])
#                         output['phrases'].append(phrases[j])
#                         output['pred_bboxes'].append(pred_bboxes[i][j][0].cpu().numpy() if pred_bboxes[i][j] is not None else None)
#                         output['gt_bboxes'].append(gt_bboxes[i][j].cpu().numpy())
#                         output['ious'].append(iou)
#         else:
#             gt_masks = metrics['gt_mask']
#             pred_masks = metrics['pred_mask']
#             assert len(image_paths) == len(pred_masks) == len(gt_masks)
#             print_blue('Saving metrics to file ...', bold=True)
#             results_folder_path = get_results_folder_path(checkpoint_folder_path)
#             save_path = os.path.join(results_folder_path, f'chest_imagenome_gold_metrics_segmask.pkl')
#             output = dict(
#                 image_paths=[],
#                 phrases=[],
#                 pred_masks=[],
#                 gt_masks=[],
#                 ious=[],
#                 segmask_iou=metrics['segmask_iou'],
#             )
#             for i in range(len(image_paths)):
#                 for j in range(len(pred_masks[i])):
#                     intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
#                     union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
#                     iou = intersection / union
#                     iou = iou.item()
#                     output['image_paths'].append(image_paths[i])
#                     output['phrases'].append(phrases[j])
#                     output['pred_masks'].append(pred_masks[i][j].cpu().numpy())
#                     output['gt_masks'].append(gt_masks[i][j].cpu().numpy())
#                     output['ious'].append(iou)

#         print_magenta('mean_iou =', sum(output['ious']) / len(output['ious']), bold=True)
#         save_pickle(output, save_path)
#         print(f'Saved metrics to {save_path}')

#     if eval_mscxr:

#         count_print('----- Evaluating on MSCXR Phrase Grounding -----')
        
        

#     if eval_chexlocalize:

#         count_print('----- Evaluating on CheXlocalize Phrase Grounding -----')

#         # Create evaluation engine
#         print_blue('Creating evaluation engine ...', bold=True)
#         evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

#         # Attach metrics
#         metrics_to_print = []
#         _cond_func = lambda x: x['flag'] == 'cl'
#         attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
#                                                       nc=len(CHEXLOCALIZE_CLASS_NAMES),
#                                                       condition_function=_cond_func)
#         attach_condition_aware_accuracy(evaluation_engine, 'pred_labels', 'gt_labels', 'classif_acc', _cond_func)
#         metrics_to_print.append('segmask_iou')
#         metrics_to_print.append('classif_acc')

#         # Attach accumulators
#         attach_accumulator(evaluation_engine, 'pred_mask')
#         attach_accumulator(evaluation_engine, 'gt_mask')
#         attach_accumulator(evaluation_engine, 'pred_probs')
#         attach_accumulator(evaluation_engine, 'gt_labels')

#         # Timer
#         timer = Timer()
#         timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

#         # Logging
#         print_blue('Defining log_metrics_handler ...', bold=True)
#         log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
#         log_iteration_handler = get_log_iteration_handler()
        
#         # Attach handlers
#         evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
#         evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

#         # Start evaluation
#         print_blue('Running engine ...', bold=True)
#         evaluation_engine.run(chexlocalize_trainer.val_dataloader)

#         # Print final metrics
#         print_blue('Final metrics:', bold=True)
#         metrics = evaluation_engine.state.metrics
#         # 1) segmask iou
#         metric_name = 'segmask_iou'
#         print(f'{metric_name}: {metrics[metric_name]}')
#         # 2) classif acc
#         metric_name = 'classif_acc'
#         print(f'{metric_name}: {metrics[metric_name]}')
#         # 3) PRC-AUC
#         pred_probs = metrics['pred_probs']
#         pred_probs = torch.tensor(pred_probs).cpu().numpy()
#         assert pred_probs.ndim == 1
#         gt_labels = metrics['gt_labels']
#         gt_labels = torch.tensor(gt_labels).cpu().numpy()
#         assert gt_labels.ndim == 1
#         pred_probs =  pred_probs.reshape(-1, len(CHEXLOCALIZE_CLASS_NAMES))
#         gt_labels = gt_labels.reshape(-1, len(CHEXLOCALIZE_CLASS_NAMES))
#         assert pred_probs.shape == gt_labels.shape
#         assert pred_probs.shape[0] == len(chexlocalize_trainer.val_dataset)
#         prc_auc_metrics = prc_auc_fn(pred_probs, gt_labels)
#         for class_name, prc_auc in zip(CHEXLOCALIZE_CLASS_NAMES, prc_auc_metrics['per_class']):
#             print(f'  PRC-AUC({class_name}): {prc_auc}')
#         print(f'PRC-AUC(macro_avg): {prc_auc_metrics["macro_avg"]}')
#         print(f'PRC-AUC(micro_avg): {prc_auc_metrics["micro_avg"]}')

#         # Save metrics to file
#         dataset = chexlocalize_trainer.val_dataset
#         image_paths = dataset.image_paths
#         phrases = chexlocalize_trainer.class_phrases
#         gt_masks = metrics['gt_mask']
#         pred_masks = metrics['pred_mask']
#         assert len(image_paths) == len(pred_masks) == len(gt_masks)
#         print_blue('Saving metrics to file ...', bold=True)
#         results_folder_path = get_results_folder_path(checkpoint_folder_path)
#         save_path = os.path.join(results_folder_path, f'chexlocalize_metrics.pkl')
#         output = dict(
#             image_paths=[],
#             phrases=[],
#             pred_masks=[],
#             gt_masks=[],
#             ious=[],
#             segmask_iou=metrics['segmask_iou'],
#             classif_acc=metrics['classif_acc'],
#             prc_auc=prc_auc_metrics,
#         )
#         for i in range(len(image_paths)):
#             for j in range(len(pred_masks[i])):
#                 if gt_labels[i,j] == 1:
#                     intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
#                     union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
#                     iou = intersection / union
#                     iou = iou.item()
#                     output['image_paths'].append(image_paths[i])
#                     output['phrases'].append(phrases[j])
#                     output['pred_masks'].append(pred_masks[i][j].cpu().numpy())
#                     output['gt_masks'].append(gt_masks[i][j].cpu().numpy())
#                     output['ious'].append(iou)
#                 else:
#                     assert torch.all(gt_masks[i][j] == 0)

#         print_magenta('mean_iou =', sum(output['ious']) / len(output['ious']), bold=True)
#         save_pickle(output, save_path)
#         print(f'Saved metrics to {save_path}')

#     if eval_vinbig:

#         count_print('----- Evaluating on VinDr-CXR Phrase Grounding -----')

#         # Create evaluation engine
#         print_blue('Creating evaluation engine ...', bold=True)
#         if optimize_thresholds:
#             evaluation_engine_kwargs['skip_nms'] = True # We need to skip NMS to optimize thresholds
#         evaluation_engine_kwargs['use_amp'] = use_amp
#         evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)
#         use_vinbig_with_modified_labels = vinbig_trainer_kwargs.get('use_vinbig_with_modified_labels', False)
        
#         if use_vinbig_with_modified_labels:
#             print_orange('NOTE: Using VinDr-CXR with modified labels', bold=True)
#             from medvqa.datasets.vinbig import VINBIG_BBOX_NAMES__MODIFIED
#             vinbig_bbox_names = VINBIG_BBOX_NAMES__MODIFIED
#             vinbig_num_bbox_classes = len(VINBIG_BBOX_NAMES__MODIFIED)
#         else:
#             vinbig_bbox_names = VINBIG_BBOX_NAMES
#             vinbig_num_bbox_classes = VINBIG_NUM_BBOX_CLASSES            

#         # Attach metrics
#         metrics_to_print = []
#         _cond_func = lambda x: x['flag'] == 'vbg'
#         if do_visual_grounding_with_bbox_regression:
#             if not optimize_thresholds:
#                 if use_fact_conditioned_yolo:
#                     attach_condition_aware_bbox_iou_per_class(evaluation_engine,
#                                                         field_names=['yolo_predictions', 'vinbig_bbox_coords', 'vinbig_bbox_classes'],
#                                                         metric_name='bbox_iou', nc=vinbig_num_bbox_classes, condition_function=_cond_func,
#                                                         for_vinbig=True, use_fact_conditioned_yolo=True)
#                 else:
#                     attach_condition_aware_bbox_iou_per_class(evaluation_engine,
#                                                             field_names=['predicted_bboxes', 'vinbig_bbox_coords', 'vinbig_bbox_classes'],
#                                                             metric_name='bbox_iou', nc=vinbig_num_bbox_classes, condition_function=_cond_func,
#                                                             for_vinbig=True)
#                 metrics_to_print.append('bbox_iou')
#         if do_visual_grounding_with_segmentation:
#             attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
#                                                         nc=vinbig_num_bbox_classes, condition_function=_cond_func)
#             metrics_to_print.append('segmask_iou')

#         # Attach accumulators
#         attach_accumulator(evaluation_engine, 'pred_probs')
#         attach_accumulator(evaluation_engine, 'gt_labels')
#         if do_visual_grounding_with_bbox_regression:
#             if use_fact_conditioned_yolo:
#                 if optimize_thresholds:
#                     attach_accumulator(evaluation_engine, 'yolo_predictions', append_instead_of_extend=True)
#                     attach_accumulator(evaluation_engine, 'resized_shape', append_instead_of_extend=True)
#                 else:
#                     attach_accumulator(evaluation_engine, 'yolo_predictions')
#             else:
#                 if optimize_thresholds:
#                     attach_accumulator(evaluation_engine, 'pred_bbox_probs')
#                     attach_accumulator(evaluation_engine, 'pred_bbox_coords')
#                 else:
#                     attach_accumulator(evaluation_engine, 'predicted_bboxes')
#             attach_accumulator(evaluation_engine, 'vinbig_bbox_coords')
#             attach_accumulator(evaluation_engine, 'vinbig_bbox_classes')
#         elif do_visual_grounding_with_segmentation:
#             attach_accumulator(evaluation_engine, 'pred_mask')
#             attach_accumulator(evaluation_engine, 'gt_mask')

#         # Timer
#         timer = Timer()
#         timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

#         # Logging
#         print_blue('Defining log_metrics_handler ...', bold=True)
#         log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
#         log_iteration_handler = get_log_iteration_handler()
        
#         # Attach handlers
#         evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
#         evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

#         # Run evaluation
#         print_blue('Running engine ...', bold=True)
#         evaluation_engine.run(vinbig_trainer.val_dataloader)
#         metrics = evaluation_engine.state.metrics

#         # Print some running metrics
        
#         # 1) bbox iou
#         if do_visual_grounding_with_bbox_regression:
#             if not optimize_thresholds:
#                 metric_name = 'bbox_iou'
#                 print(f'{metric_name}: {metrics[metric_name]}')
#         # 1) segmask iou
#         if do_visual_grounding_with_segmentation:
#             metric_name = 'segmask_iou'
#             print(f'{metric_name}: {metrics[metric_name]}')

#         # Compute metrics and prepare output to save to file

#         output_to_save = dict()
        
#         # --- Classification metrics

#         dataset = vinbig_trainer.val_dataset
#         phrases = vinbig_trainer.phrases
#         classification_label_names = vinbig_trainer.actual_label_names[:] # copy
#         pred_probs = metrics['pred_probs']
#         pred_probs = torch.stack(pred_probs).cpu().numpy()
#         assert pred_probs.ndim == 2
#         gt_labels = metrics['gt_labels']
#         gt_labels = torch.stack(gt_labels).cpu().numpy()
#         assert gt_labels.ndim == 2
#         assert pred_probs.shape == gt_labels.shape
#         assert pred_probs.shape[0] == len(dataset)

#         classif_pred_probs = pred_probs.copy() # (num_samples, num_classes)
#         classif_gt_labels = gt_labels.copy() # (num_samples, num_classes)

#         # Convert "Abnormal finding" to "No finding" if applicable
#         if use_vinbig_with_modified_labels:
#             print_orange('NOTE: Converting "Abnormal finding" to "No finding" for VinDr-CXR with modified labels', bold=True)
#             assert "Abnormal finding" in classification_label_names
#             assert "No finding" not in classification_label_names
#             abnormal_finding_idx = classification_label_names.index("Abnormal finding")
#             classification_label_names[abnormal_finding_idx] = "No finding"
#             classif_gt_labels[:, abnormal_finding_idx] = np.logical_not(classif_gt_labels[:, abnormal_finding_idx])
#             classif_pred_probs[:, abnormal_finding_idx] = 1 - classif_pred_probs[:, abnormal_finding_idx]

#         # Remove classes without any ground truth
#         classif_gt_labels_sum = classif_gt_labels.sum(axis=0)
#         no_gt_classes = np.where(classif_gt_labels_sum == 0)[0]
#         if len(no_gt_classes) > 0:
#             print_orange('NOTE: Removing the following classes without any positive classification labels:', bold=True)
#             for i in no_gt_classes:
#                 print_orange(f'  {classification_label_names[i]}')
#             classif_pred_probs = np.delete(classif_pred_probs, no_gt_classes, axis=1)
#             classif_gt_labels = np.delete(classif_gt_labels, no_gt_classes, axis=1)
#             classification_label_names = [x for i, x in enumerate(classification_label_names) if i not in no_gt_classes]
#             print(f'classif_pred_probs.shape = {classif_pred_probs.shape}')
#             print(f'classif_gt_labels.shape = {classif_gt_labels.shape}')
#             print(f'len(classification_label_names) = {len(classification_label_names)}')

#         # Compute PRC-AUC without bootstrapping
#         prc_auc_metrics = prc_auc_fn(classif_gt_labels, classif_pred_probs)

#         # Compute PRC-AUC with bootstrapping
#         prc_auc_metrics_with_boot = stratified_multilabel_bootstrap_metrics(
#             gt_labels=classif_gt_labels, pred_probs=classif_pred_probs, metric_fn=prc_auc_score, num_bootstraps=500)
        
#         # Save classification metrics to output
#         output_to_save['classification'] = dict(
#             classification_label_names=classification_label_names,
#             pred_probs=classif_pred_probs,
#             gt_labels=classif_gt_labels,
#             prc_auc=prc_auc_metrics,
#             prc_auc_with_bootstrapping=prc_auc_metrics_with_boot,
#         )

#         # Print some classification metrics
#         for class_name, mean, std in zip(classification_label_names,
#                                          prc_auc_metrics_with_boot['mean_per_class'],
#                                          prc_auc_metrics_with_boot['std_per_class']):
#             if class_name in VINBIG_RAD_DINO_CLASSES:
#                 print_bold(f'PRC-AUC({class_name}): {mean} ± {std}')
#             else:
#                 print(f'PRC-AUC({class_name}): {mean} ± {std}')
#         print_magenta(f'PRC-AUC(macro_avg) with bootstrapping: {prc_auc_metrics_with_boot["mean_macro_avg"]} ± {prc_auc_metrics_with_boot["std_macro_avg"]}', bold=True)
#         print_magenta(f'PRC-AUC(macro_avg): {prc_auc_metrics["macro_avg"]}', bold=True)

#         #  --- Visual grounding (object detection / segmentation) metrics

#         image_paths = [dataset.image_paths[i] for i in dataset.indices]
#         output_to_save['image_paths'] = image_paths # for saving to file

#         if do_visual_grounding_with_bbox_regression:

#             for iou_thr in VINBIG_CHEX_IOU_THRESHOLDS: assert iou_thr in map_iou_thresholds
#             assert VINBIGDATA_CHALLENGE_IOU_THRESHOLD in map_iou_thresholds

#             gt_bboxes = metrics['vinbig_bbox_coords']
#             gt_classes = metrics['vinbig_bbox_classes']
#             assert len(image_paths) == len(gt_bboxes) == len(gt_classes)
#             gt_coords_list = [[[] for _ in range(vinbig_num_bbox_classes)] for _ in range(len(gt_bboxes))]
#             for i in range(len(gt_bboxes)):
#                 for bbox, cls in zip(gt_bboxes[i], gt_classes[i]):
#                     gt_coords_list[i][cls].append(bbox)
            
#             # Convert to numpy
#             for i in range(len(gt_coords_list)):
#                 for j in range(len(gt_coords_list[i])):
#                     gt_coords_list[i][j] = np.stack(gt_coords_list[i][j]) if len(gt_coords_list[i][j]) > 0 else np.empty((0, 4))

#             if use_classifier_confs_for_map:
#                 classifier_confs = pred_probs[:, :vinbig_num_bbox_classes] # (num_samples, num_classes)
#                 print_bold('Using classifier confidences for mAP computation')
#                 print(f'classifier_confs.shape = {classifier_confs.shape}')
#             else:
#                 classifier_confs = None
            
#             if optimize_thresholds: # Optimize thresholds
#                 assert candidate_iou_thresholds is not None
#                 assert candidate_conf_thresholds is not None
#                 num_classes = vinbig_num_bbox_classes
#                 if use_fact_conditioned_yolo:
#                     out = find_optimal_conf_iou_thresholds(
#                         gt_coords_list=gt_coords_list,
#                         yolo_predictions_list=metrics['yolo_predictions'],
#                         resized_shape_list=metrics['resized_shape'],
#                         is_fact_conditioned_yolo=True,
#                         iou_thresholds=candidate_iou_thresholds,
#                         conf_thresholds=candidate_conf_thresholds,
#                         classifier_confs=classifier_confs,
#                         num_classes=num_classes,
#                         verbose=True,
#                     )
#                 else:
#                     pred_bbox_probs = metrics['pred_bbox_probs']
#                     pred_bbox_coords = metrics['pred_bbox_coords']
#                     num_regions = pred_bbox_probs[0].shape[1]
#                     assert pred_bbox_probs[0].ndim == 3 # (num_classes, num_regions, 1)
#                     assert pred_bbox_coords[0].ndim == 3 # (num_classes, num_regions, 4)
#                     assert pred_bbox_probs[0].shape == (num_classes, num_regions, 1)
#                     assert pred_bbox_coords[0].shape == (num_classes, num_regions, 4)
#                     out = find_optimal_conf_iou_thresholds(
#                         gt_coords_list=gt_coords_list,
#                         pred_boxes_list=pred_bbox_coords,
#                         pred_confs_list=pred_bbox_probs,
#                         iou_thresholds=candidate_iou_thresholds,
#                         conf_thresholds=candidate_conf_thresholds,
#                         classifier_confs=classifier_confs,
#                         verbose=True,
#                     )
#                 best_iou_threshold = out['best_iou_threshold']
#                 best_conf_threshold = out['best_conf_threshold']
#                 pred_boxes_list = out['pred_boxes_list']
#                 pred_classes_list = out['pred_classes_list']
#                 pred_confs_list = out['pred_confs_list']

#             else:
#                 if use_fact_conditioned_yolo:
#                     pred_bboxes = metrics['yolo_predictions']
#                     assert len(image_paths) == len(pred_bboxes)
#                     pred_boxes_list = []
#                     pred_confs_list = []
#                     pred_classes_list = []
#                     for preds in pred_bboxes:
#                         assert len(preds) == vinbig_num_bbox_classes
#                         boxes = []
#                         confs = []
#                         classes = []
#                         for i, pred in enumerate(preds):
#                             pred = pred.cpu().numpy()
#                             boxes.append(pred[:, :4])
#                             confs.append(pred[:, 4])
#                             classes.append(np.full((len(pred),), i))
#                         pred_boxes_list.append(np.concatenate(boxes, axis=0))
#                         pred_confs_list.append(np.concatenate(confs, axis=0))
#                         pred_classes_list.append(np.concatenate(classes, axis=0))
#                 else:
#                     pred_bboxes = metrics['predicted_bboxes']
#                     assert len(image_paths) == len(pred_bboxes)
#                     pred_boxes_list = []
#                     pred_confs_list = []
#                     pred_classes_list = []
#                     for preds in pred_bboxes:
#                         assert len(preds) == 3 # (boxes, confs, classes)
#                         pred_boxes_list.append(preds[0].cpu().numpy())
#                         pred_confs_list.append(preds[1].cpu().numpy())
#                         pred_classes_list.append(preds[2].cpu().numpy())

#             # Remove classes without any ground truth
#             gt_counts_per_class = np.zeros(vinbig_num_bbox_classes, dtype=int)
#             for i in range(len(gt_coords_list)):
#                 for j in range(len(gt_coords_list[i])):
#                     gt_counts_per_class[j] += len(gt_coords_list[i][j])
#             no_gt_classes = np.where(gt_counts_per_class == 0)[0]
#             with_gt_classes = np.where(gt_counts_per_class > 0)[0]
#             if len(no_gt_classes) > 0:
#                 print_orange('NOTE: Removing the following classes without any bounding box annotations:', bold=True)
#                 for i in no_gt_classes:
#                     print_orange(f'  {vinbig_bbox_names[i]}')
#                 print(f'gt_counts_per_class = {gt_counts_per_class}')
#                 # Clean gt_coords_list
#                 for i in range(len(gt_coords_list)):
#                     gt_coords_list[i] = [gt_coords_list[i][j] for j in with_gt_classes]
#                 print(f'len(gt_coords_list) = {len(gt_coords_list)}')
#                 print(f'len(gt_coords_list[0]) = {len(gt_coords_list[0])}')
#                 # Clean classifier_confs
#                 if use_classifier_confs_for_map:
#                     classifier_confs = classifier_confs[:, with_gt_classes]
#                     print(f'classifier_confs.shape = {classifier_confs.shape}')
#                 # Clean vinbig_bbox_names
#                 vinbig_bbox_names = [x for i, x in enumerate(vinbig_bbox_names) if i in with_gt_classes]
#                 vinbig_num_bbox_classes = len(vinbig_bbox_names)
#                 print(f'len(vinbig_bbox_names) = {len(vinbig_bbox_names)}')
#                 # Clean pred_boxes_list, pred_classes_list, pred_confs_list
#                 old_class_idx_to_new_class_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(with_gt_classes)}
#                 for i in range(len(pred_boxes_list)):
#                     if len(pred_classes_list[i]) > 0:
#                         valid_idxs = np.where(np.isin(pred_classes_list[i], with_gt_classes))[0]
#                         pred_boxes_list[i] = pred_boxes_list[i][valid_idxs]
#                         pred_confs_list[i] = pred_confs_list[i][valid_idxs]
#                         pred_classes_list[i] = np.array([old_class_idx_to_new_class_idx[x] for x in pred_classes_list[i][valid_idxs]])
#                 print(f'len(pred_boxes_list) = {len(pred_boxes_list)}')
#                 print(f'len(pred_classes_list) = {len(pred_classes_list)}')
#                 print(f'len(pred_confs_list) = {len(pred_confs_list)}')


#             if optimize_thresholds: # Print optimal thresholds
#                 print_magenta(f'best_iou_threshold: {best_iou_threshold}', bold=True)
#                 print_magenta(f'best_conf_threshold: {best_conf_threshold}', bold=True)

#             # Compute metrics without bootstrapping
            
#             # 1. IoU
#             tmp = compute_mean_iou_per_class__yolov11(
#                 pred_boxes=pred_boxes_list,
#                 pred_classes=pred_classes_list,
#                 gt_coords=gt_coords_list,
#                 compute_iou_per_sample=True,
#                 compute_micro_average_iou=True,
#                 return_counts=True,
#             )
#             class_ious = tmp['class_ious']
#             sample_ious = tmp['sample_ious']
#             class_counts = tmp['class_counts']
#             sample_counts = tmp['sample_counts']
#             micro_iou = tmp['micro_iou']
#             class_idxs = np.where(class_counts > 0)[0]
#             macro_iou = class_ious[class_idxs].mean()
#             sample_idxs = np.where(sample_counts > 0)[0]
#             sample_iou = sample_ious[sample_idxs].mean()

#             # 2. mAP
#             tmp = compute_mAP__yolov11(
#                 pred_boxes=pred_boxes_list,
#                 pred_classes=pred_classes_list,
#                 pred_confs=pred_confs_list,
#                 classifier_confs=classifier_confs,
#                 gt_coords=gt_coords_list,
#                 iou_thresholds=map_iou_thresholds,
#                 compute_micro_average=True,
#             )
#             class_aps = tmp['class_aps']
#             micro_aps = tmp['micro_aps']

#             # 2.1 vinbigdata mAP
#             class_idxs = [vinbig_bbox_names.index(x) for x in VINBIGDATA_CHALLENGE_CLASSES]
#             iou_idx = map_iou_thresholds.index(VINBIGDATA_CHALLENGE_IOU_THRESHOLD)
#             vbdc_mAP = class_aps[iou_idx, class_idxs].mean() # vbdc = vinbigdata challenge

#             # 2.2 ChEX mAP
#             class_idxs = [vinbig_bbox_names.index(x) for x in VINBIG_CHEX_CLASSES]
#             iou_idxs = [map_iou_thresholds.index(x) for x in VINBIG_CHEX_IOU_THRESHOLDS]
#             chex_mAP = class_aps[iou_idxs][:, class_idxs].mean()

#             # Update output
#             output_to_save['detection'] = dict(
#                 pred_boxes_list=pred_boxes_list,
#                 pred_classes_list=pred_classes_list,
#                 pred_confs_list=pred_confs_list,
#                 classifier_confs=classifier_confs,
#                 gt_bboxes=gt_coords_list,
#                 bbox_class_names=vinbig_bbox_names,
#                 map_iou_thresholds=map_iou_thresholds, # (num_iou_thresholds,)
#                 sample_ious=sample_ious, # (num_samples,)
#                 metrics_without_bootstrapping=dict(
#                     class_ious=class_ious, # (num_classes,)
#                     micro_iou=micro_iou, # scalar
#                     macro_iou=macro_iou, # scalar
#                     class_aps=class_aps, # (num_iou_thresholds, num_classes)
#                     micro_aps=micro_aps, # (num_iou_thresholds,)
#                     vbdc_mAP=vbdc_mAP, # scalar
#                     chex_mAP=chex_mAP, # scalar
#                 ),
#             )

#             # Compute metrics with bootstrapping
#             iou_map_metrics_with_boot = stratified_vinbig_bootstrap_iou_map(
#                 pred_boxes_list=pred_boxes_list,
#                 pred_classes_list=pred_classes_list,
#                 pred_confs_list=pred_confs_list,
#                 classifier_confs=classifier_confs,
#                 gt_coords_list=gt_coords_list,
#                 vinbig_bbox_names=vinbig_bbox_names,
#                 map_iou_thresholds=map_iou_thresholds,
#                 compute_mean_iou_per_class_fn=compute_mean_iou_per_class__yolov11,
#                 compute_mAP_fn=compute_mAP__yolov11,
#                 num_bootstraps=60,
#                 num_processes=12,
#             )

#             # Update output
#             output_to_save['detection']['metrics_with_bootstrapping'] = iou_map_metrics_with_boot

#             if optimize_thresholds:
#                 output_to_save['best_iou_threshold'] = best_iou_threshold
#                 output_to_save['best_conf_threshold'] = best_conf_threshold

#             # Print some metrics
#             for class_name, iou, count in zip(vinbig_bbox_names, class_ious, class_counts):
#                 print(f'mean_iou({class_name}): {iou} ({count} samples)')
#             print_magenta(f'macro_iou: {macro_iou}', bold=True)
#             print_magenta(f'mean_sample_iou: {sample_iou}', bold=True)
#             print(f'\t{sample_idxs.shape[0]} / {len(gt_coords_list)} samples have at least one ground truth bbox')
#             print_magenta(f'micro_iou: {micro_iou} (count={sample_counts.sum()})', bold=True)
            
#             for iou_thresh, map_ in zip(map_iou_thresholds, class_aps.mean(axis=1)):
#                 print_magenta(f'mAP@{iou_thresh}: {map_}', bold=True)

#             for iou_thresh, ap in zip(map_iou_thresholds, micro_aps):
#                 print_magenta(f'micro_AP@{iou_thresh}: {ap}', bold=True)

#             print_magenta(f'vbdc_mAP: {vbdc_mAP}', bold=True)
#             print_magenta(f'chex_mAP: {chex_mAP}', bold=True)

#             print_magenta(f'micro_iou (with bootstrap): {iou_map_metrics_with_boot["micro_iou"]["mean"]} ± {iou_map_metrics_with_boot["micro_iou"]["std"]}', bold=True)
#             print_magenta(f'macro_iou (with bootstrap): {iou_map_metrics_with_boot["macro_iou"]["mean"]} ± {iou_map_metrics_with_boot["macro_iou"]["std"]}', bold=True)
#             print_magenta(f'vbdc_mAP (with bootstrap): {iou_map_metrics_with_boot["vbdc_mAP"]["mean"]} ± {iou_map_metrics_with_boot["vbdc_mAP"]["std"]}', bold=True)
#             print_magenta(f'chex_mAP (with bootstrap): {iou_map_metrics_with_boot["chex_mAP"]["mean"]} ± {iou_map_metrics_with_boot["chex_mAP"]["std"]}', bold=True)
            
#             # Save metrics to file
#             print_blue('Saving metrics to file ...', bold=True)
#             results_folder_path = get_results_folder_path(checkpoint_folder_path)
#             strings = [
#                 'detection',
#                 f'{len(vinbig_trainer.val_dataset)}',
#             ]
#             if optimize_thresholds:
#                 strings.append(f'opt_thr({best_iou_threshold:.2f},{best_conf_threshold:.2f})')
#             if use_classifier_confs_for_map:
#                 strings.append('use_classifier_confs')
#             save_path = os.path.join(results_folder_path, f'vindrcxr_metrics({",".join(strings)}).pkl')

#         elif do_visual_grounding_with_segmentation:

#             gt_masks = metrics['gt_mask']
#             pred_masks = metrics['pred_mask']
#             assert len(image_paths) == len(pred_masks) == len(gt_masks),\
#                 (f'len(image_paths) = {len(image_paths)}, len(pred_masks) = {len(pred_masks)}, len(gt_masks) = {len(gt_masks)}')
#             print_blue('Saving metrics to file ...', bold=True)
#             results_folder_path = get_results_folder_path(checkpoint_folder_path)
#             save_path = os.path.join(results_folder_path, f'vindrcxr_metrics(segmask,{len(vinbig_trainer.val_dataset)}).pkl')
#             output_to_save = dict(
#                 image_paths=[],
#                 phrases=[],
#                 pred_masks=[],
#                 gt_masks=[],
#                 ious=[],
#                 segmask_iou=metrics['segmask_iou'],
#                 prc_auc=prc_auc_metrics,
#             )
#             for i in range(len(image_paths)):
#                 for j in range(len(pred_masks[i])):
#                     if gt_labels[i, j] == 1:
#                         intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
#                         union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
#                         iou = intersection / union
#                         iou = iou.item()
#                         output_to_save['image_paths'].append(image_paths[i])
#                         output_to_save['phrases'].append(phrases[j])
#                         output_to_save['pred_masks'].append(pred_masks[i][j].cpu().numpy())
#                         output_to_save['gt_masks'].append(gt_masks[i][j].cpu().numpy())
#                         output_to_save['ious'].append(iou)
#                     else:
#                         assert torch.all(gt_masks[i][j] == 0)

#             print_magenta('mean_iou =', sum(output_to_save['ious']) / len(output_to_save['ious']), bold=True)

#         else:

#             results_folder_path = get_results_folder_path(checkpoint_folder_path)
#             save_path = os.path.join(results_folder_path, f'vindrcxr_metrics(classification,{len(vinbig_trainer.val_dataset)}).pkl')
        
#         save_pickle(output_to_save, save_path)
#         print(f'Saved metrics to {save_path}')


def _run_inference(args):

    count_print = CountPrinter()

    print_blue('----- Running Inference -----', bold=True)

    # Force deterministic behavior
    # activate_determinism()    

    metadata = load_metadata(args.checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    val_image_transform_kwargs = metadata['val_image_transform_kwargs']

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    count_print('device =', device)

    # Create model
    count_print('Creating instance of PhraseGrounder ...')
    model = PhraseGrounder(**model_kwargs)
    model = model.to(device)

    # Load model from checkpoint
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(args.checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path =', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True, strict=False)

    if args.dataset_name == 'mscxr':
        
        count_print('----- Evaluating on MSCXR Phrase Grounding -----')
        run_inference_and_save_predictions_on_mscxr(
            model=model,
            checkpoint_folder_path=args.checkpoint_folder_path,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=args.max_images_per_batch,
            max_phrases_per_batch=args.max_phrases_per_batch,
            max_phrases_per_image=args.max_phrases_per_image,
            num_workers=args.num_workers,
            mscxr_phrase2embedding_filepath=args.mscxr_phrase2embedding_filepath,
            mimicxr_dicom_id_to_pos_neg_facts_filepath=args.mimicxr_dicom_id_to_pos_neg_facts_filepath,
            device=device,
            override_bbox_format=args.override_bbox_format,
        )

    elif args.dataset_name == 'padchest-gr':

        count_print('----- Evaluating on PadChest-GR Phrase Grounding -----')
        run_inference_and_save_predictions_on_padchest_gr(
            model=model,
            checkpoint_folder_path=args.checkpoint_folder_path,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=args.max_images_per_batch,
            num_workers=args.num_workers,
            padchest_gr_phrase_embeddings_filepath=args.padchest_gr_phrase_embeddings_filepath,
            device=device,
            override_bbox_format=args.override_bbox_format,
        )

    elif args.dataset_name == 'vindrcxr':

        count_print('----- Evaluating on VinDr-CXR Phrase Grounding -----')
        run_inference_and_save_predictions_on_vindrcxr(
            model=model,
            checkpoint_folder_path=args.checkpoint_folder_path,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=args.max_images_per_batch,
            num_workers=args.num_workers,
            vindrcxr_phrase_embeddings_filepath=args.vindrcxr_phrase_embeddings_filepath,
            device=device,
            override_bbox_format=args.override_bbox_format,
        )

    elif args.dataset_name == 'chest-imagenome':

        count_print('----- Evaluating on Chest-ImaGenome Phrase Grounding -----')
        run_inference_and_save_predictions_on_chest_imagenome(
            model=model,
            checkpoint_folder_path=args.checkpoint_folder_path,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=args.max_images_per_batch,
            max_phrases_per_batch=args.max_phrases_per_batch,
            max_phrases_per_image=args.max_phrases_per_image,
            num_workers=args.num_workers,
            chest_imagenome_augmented_phrase_groundings_filepath=args.chest_imagenome_augmented_phrase_groundings_filepath,
            chest_imagenome_bbox_phrase_embeddings_filepath=args.chest_imagenome_bbox_phrase_embeddings_filepath,
            chest_imagenome_phrase_embeddings_filepath=args.chest_imagenome_phrase_embeddings_filepath,
            device=device,
            override_bbox_format=args.override_bbox_format,
        )

    elif args.dataset_name == 'chest-x-det':

        count_print('----- Evaluating on Chest-X-Det Phrase Grounding -----')
        run_inference_and_save_predictions_on_chest_x_det(
            model=model,
            checkpoint_folder_path=args.checkpoint_folder_path,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=args.max_images_per_batch,
            num_workers=args.num_workers,
            device=device,
            train_json_path= args.chestxdet_train_json_path,
            train_image_dir=args.chestxdet_train_image_dir,
            test_json_path=args.chestxdet_test_json_path,
            test_image_dir=args.chestxdet_test_image_dir,
            label2embedding_path=args.chestxdet_label2embedding_path,
            override_bbox_format=args.override_bbox_format,
        )

    else:
        raise ValueError(f'Unsupported dataset_name: {args.dataset_name}.')


def _export_data(args):

    if args.dataset_name == 'mscxr':
        
        print_blue('----- Exporting data for MSCXR Phrase Grounding -----', bold=True)
        export_data_for_mscxr(
            bbox_format=args.bbox_format,
            output_filepath=args.output_filepath,
        )

    elif args.dataset_name == 'padchest-gr':

        print_blue('----- Exporting data for PadChest-GR Phrase Grounding -----', bold=True)
        export_data_for_padchest_gr(
            bbox_format=args.bbox_format,
            output_filepath=args.output_filepath,
        )

    elif args.dataset_name == 'vindrcxr':

        print_blue('----- Exporting data for VinDr-CXR Phrase Grounding -----', bold=True)
        export_data_for_vindrcxr(
            bbox_format=args.bbox_format,
            output_filepath=args.output_filepath,
        )

    elif args.dataset_name == 'chest-imagenome':

        print_blue('----- Exporting data for Chest-ImaGenome Phrase Grounding -----', bold=True)
        export_data_for_chest_imagenome(
            bbox_format=args.bbox_format,
            chest_imagenome_augmented_phrase_groundings_filepath=args.chest_imagenome_augmented_phrase_groundings_filepath,
            output_filepath=args.output_filepath,
        )

    elif args.dataset_name == 'chest-x-det':

        print_blue('----- Exporting data for Chest-X-Det Phrase Grounding -----', bold=True)
        export_data_for_chest_x_det(
            bbox_format=args.bbox_format,
            test_json_path=args.chestxdet_test_json_path,
            test_image_dir=args.chestxdet_test_image_dir,
            output_filepath=args.output_filepath,
        )


def _run_metrics_computation(args):

    if args.dataset_name == 'mscxr':
        
        print_blue('----- Computing metrics on MSCXR Phrase Grounding -----', bold=True)
        compute_and_save_metrics_on_mscxr(
            predictions_and_gt_filepath=args.predictions_and_gt_filepath,
            candidate_conf_thresholds=args.candidate_conf_thresholds,
            candidate_iou_thresholds=args.candidate_iou_thresholds,
        )

    elif args.dataset_name == 'padchest-gr':

        print_blue('----- Computing metrics on PadChest-GR Phrase Grounding -----', bold=True)
        compute_and_save_metrics_on_padchest_gr(
            predictions_and_gt_filepath=args.predictions_and_gt_filepath,
            candidate_conf_thresholds=args.candidate_conf_thresholds,
            candidate_iou_thresholds=args.candidate_iou_thresholds,
        )

    elif args.dataset_name == 'vindrcxr':

        print_blue('----- Computing metrics on VinDr-CXR Phrase Grounding -----', bold=True)
        compute_and_save_metrics_on_vindrcxr(
            predictions_and_gt_filepath=args.predictions_and_gt_filepath,
            candidate_iou_thresholds=args.candidate_iou_thresholds,
        )

    elif args.dataset_name == 'chest-imagenome':

        print_blue('----- Computing metrics on Chest-ImaGenome Phrase Grounding -----', bold=True)
        compute_and_save_metrics_on_chest_imagenome(
            predictions_and_gt_filepath=args.predictions_and_gt_filepath,
            candidate_conf_thresholds=args.candidate_conf_thresholds,
            candidate_iou_thresholds=args.candidate_iou_thresholds,
            num_samples_per_class=args.num_samples_per_class,
        )

    elif args.dataset_name == 'chest-x-det':

        print_blue('----- Computing metrics on Chest-X-Det Phrase Grounding -----', bold=True)
        compute_and_save_metrics_on_chest_x_det(
            predictions_and_gt_filepath=args.predictions_and_gt_filepath,
        )

    else:
        raise ValueError(f'Unsupported dataset_name: {args.dataset_name}.')


# def evaluate(
#     checkpoint_folder_path,
#     num_workers,
#     max_images_per_batch,
#     max_phrases_per_batch,
#     max_phrases_per_image,
#     eval_chest_imagenome_gold,
#     eval_mscxr,
#     eval_chexlocalize,
#     eval_vinbig,
#     mscxr_phrase2embedding_filepath,
#     mimicxr_dicom_id_to_pos_neg_facts_filepath,
#     device,
#     vinbig_use_training_indices_for_validation,
#     optimize_thresholds,
#     candidate_iou_thresholds,
#     candidate_conf_thresholds,
#     map_iou_thresholds,
#     use_amp,
#     use_classifier_confs_for_map,
#     checkpoint_folder_path_to_borrow_metadata_from,
#     override_bbox_format,
# ):  
#     # Force deterministic behavior
#     activate_determinism()
    
#     print_blue('----- Evaluating model -----', bold=True)

#     metadata = load_metadata(checkpoint_folder_path)
#     model_kwargs = metadata['model_kwargs']
#     mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
#     chexlocalize_trainer_kwargs = metadata['chexlocalize_trainer_kwargs']
#     vinbig_trainer_kwargs = metadata['vinbig_trainer_kwargs']
#     # collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
#     try:
#         val_image_transform_kwargs = metadata['val_image_transform_kwargs']
#     except KeyError: # HACK: when val_image_transform_kwargs is missing due to a bug
#         val_image_transform_kwargs = {
#             DATASET_NAMES.MIMICCXR: dict(
#                 image_size=(416, 416),
#                 augmentation_mode=None,
#                 use_bbox_aware_transform=True,
#                 for_yolov8=True,
#             )
#         }
#     validator_engine_kwargs = metadata['validator_engine_kwargs']

#     return _evaluate_model(
#                 checkpoint_folder_path=checkpoint_folder_path,
#                 model_kwargs=model_kwargs,
#                 mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
#                 chexlocalize_trainer_kwargs=chexlocalize_trainer_kwargs,
#                 vinbig_trainer_kwargs=vinbig_trainer_kwargs,
#                 # collate_batch_fn_kwargs=collate_batch_fn_kwargs,
#                 val_image_transform_kwargs=val_image_transform_kwargs,
#                 evaluation_engine_kwargs=validator_engine_kwargs,
#                 max_images_per_batch=max_images_per_batch,
#                 max_phrases_per_batch=max_phrases_per_batch,
#                 max_phrases_per_image=max_phrases_per_image,
#                 num_workers=num_workers,
#                 eval_chest_imagenome_gold=eval_chest_imagenome_gold,
#                 eval_mscxr=eval_mscxr,
#                 eval_chexlocalize=eval_chexlocalize,
#                 eval_vinbig=eval_vinbig,
#                 mscxr_phrase2embedding_filepath=mscxr_phrase2embedding_filepath,
#                 mimicxr_dicom_id_to_pos_neg_facts_filepath=mimicxr_dicom_id_to_pos_neg_facts_filepath,
#                 device=device,
#                 vinbig_use_training_indices_for_validation=vinbig_use_training_indices_for_validation,
#                 optimize_thresholds=optimize_thresholds,
#                 candidate_iou_thresholds=candidate_iou_thresholds,
#                 candidate_conf_thresholds=candidate_conf_thresholds,
#                 map_iou_thresholds=map_iou_thresholds,
#                 use_amp=use_amp,
#                 use_classifier_confs_for_map=use_classifier_confs_for_map,
#                 checkpoint_folder_path_to_borrow_metadata_from=checkpoint_folder_path_to_borrow_metadata_from,
#                 override_bbox_format=override_bbox_format,
#             )

def main():
    # --- Step 1: Pre-parse to find the config file ---
    # We look for --config_filepath manually in sys.argv to load defaults early
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--config_filepath', help='Path to YAML config file')
    known_args, _ = conf_parser.parse_known_args()

    defaults = {}
    if known_args.config_filepath:
        defaults = load_config_yaml(known_args.config_filepath)
        print(f"Loaded configuration from: {known_args.config_filepath}")

    # --- Step 2: Main Parser Setup ---
    parser = argparse.ArgumentParser()

    # Add subcommands
    subparsers = parser.add_subparsers(dest='subcommand', required=True)

    # 1. Inference Parser
    inference_parser = subparsers.add_parser('inference', help='Run inference and save predictions', parents=[conf_parser])
    
    dataset_choices = ['mscxr', 'padchest-gr', 'vindrcxr', 'chest-imagenome', 'chest-x-det']

    # --- Arguments (Note: required=True REMOVED for args that can be in YAML) ---
    inference_parser.add_argument('--dataset_name', type=str, choices=dataset_choices, help='Name of the dataset to run inference on')
    inference_parser.add_argument('--checkpoint_folder_path', type=str, help='Path to the checkpoint folder of the model to be evaluated')
    inference_parser.add_argument('--max_images_per_batch', type=int, help='Max number of images per batch')
    inference_parser.add_argument('--max_phrases_per_batch', type=int, help='Max number of phrases per batch')
    inference_parser.add_argument('--max_phrases_per_image', type=int, help='Max number of phrases per image')

    # Optional args (defaults provided here act as fallbacks if not in YAML)
    inference_parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    inference_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    inference_parser.add_argument('--mscxr_phrase2embedding_filepath', type=str, default=None, help='Path to the MS-CXR phrase2embedding file')
    inference_parser.add_argument('--padchest_gr_phrase_embeddings_filepath', type=str, default=None, help='Path to the PadChest-GR phrase embeddings file')
    inference_parser.add_argument('--vindrcxr_phrase_embeddings_filepath', type=str, default=None, help='Path to the VinDr-CXR phrase embeddings file')
    inference_parser.add_argument('--mimicxr_dicom_id_to_pos_neg_facts_filepath', type=str, default=None, help='Path to the MIMIC-CXR DICOM ID to pos/neg facts file')
    inference_parser.add_argument('--chest_imagenome_augmented_phrase_groundings_filepath', type=str, default=None,
                                  help='Path to the Chest-ImaGenome augmented phrase groundings file')
    inference_parser.add_argument('--chest_imagenome_bbox_phrase_embeddings_filepath', type=str, default=None,
                                    help='Path to the Chest-ImaGenome bbox phrase embeddings file')
    inference_parser.add_argument('--chest_imagenome_phrase_embeddings_filepath', type=str, default=None,
                                    help='Path to the Chest-ImaGenome phrase embeddings file')
    inference_parser.add_argument('--chestxdet_train_json_path', type=str, default=None, help='Path to the Chest-X-Det train JSON file')
    inference_parser.add_argument('--chestxdet_train_image_dir', type=str, default=None, help='Path to the Chest-X-Det train image directory')
    inference_parser.add_argument('--chestxdet_test_json_path', type=str, default=None, help='Path to the Chest-X-Det test JSON file')
    inference_parser.add_argument('--chestxdet_test_image_dir', type=str, default=None, help='Path to the Chest-X-Det test image directory')
    inference_parser.add_argument('--chestxdet_label2embedding_path', type=str, default=None, help='Path to the Chest-X-Det label to embedding file')
    inference_parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision for inference')
    inference_parser.add_argument('--override_bbox_format', type=str, default=None, choices=['xyxy', 'cxcywh'], help='Override the bbox format used in the dataset')

    # Apply YAML defaults
    if defaults:
        inference_parser.set_defaults(**defaults)

    # 2. Metrics Parser
    metrics_parser = subparsers.add_parser('metrics', help='Compute metrics')
    metrics_parser.add_argument('--dataset_name', type=str, required=True, choices=dataset_choices, help='Name of the dataset to compute metrics on')
    metrics_parser.add_argument('--predictions_and_gt_filepath', type=str, required=True, help='Path to the predictions and ground truth file')
    metrics_parser.add_argument('--candidate_conf_thresholds', type=float, nargs='+', help='Candidate confidence thresholds for IoU computation')
    metrics_parser.add_argument('--candidate_iou_thresholds', type=float, nargs='+', help='Candidate IoU thresholds for IoU computation')
    metrics_parser.add_argument('--num_samples_per_class', type=int, default=300, help='Number of samples per class for metrics computation')

    # 3. Export Parser
    export_data_parser = subparsers.add_parser('export', help='Export data')
    export_data_parser.add_argument('--dataset_name', type=str, required=True, choices=dataset_choices, help='Name of the dataset to export data for')
    export_data_parser.add_argument('--bbox_format', type=str, default='cxcywh', choices=['xyxy', 'cxcywh'], help='Bounding box format to use for exporting data')
    export_data_parser.add_argument('--output_filepath', type=str, required=True, help='Output file path for the exported data')
    export_data_parser.add_argument('--chest_imagenome_augmented_phrase_groundings_filepath', type=str, default=None,
                                    help='Path to the Chest-ImaGenome augmented phrase groundings file for exporting data')
    export_data_parser.add_argument('--chestxdet_test_json_path', type=str, default=None, help='Path to the Chest-X-Det test JSON file for exporting data')
    export_data_parser.add_argument('--chestxdet_test_image_dir', type=str, default=None, help='Path to the Chest-X-Det test image directory for exporting data')


    # # --- Required arguments

    # parser.add_argument('--checkpoint_folder_path', type=str, default=None, help='Path to the checkpoint folder of the model to be evaluated')
    # parser.add_argument('--max_images_per_batch', type=int, required=True, help='Max number of images per batch')
    # parser.add_argument('--max_phrases_per_batch', type=int, required=True, help='Max number of phrases per batch')
    # parser.add_argument('--max_phrases_per_image', type=int, required=True, help='Max number of phrases per image')

    # # --- Other arguments

    # # Dataset and dataloading arguments
    # parser.add_argument('--num_workers', type=int, default=0)
    # parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    # parser.add_argument('--mscxr_phrase2embedding_filepath', type=str, default=None, help='Path to the MS-CXR phrase2embedding file')
    # parser.add_argument('--mimicxr_dicom_id_to_pos_neg_facts_filepath', type=str, default=None, help='Path to the MIMIC-CXR DICOM ID to pos/neg facts file')
    # parser.add_argument('--vinbig_use_training_indices_for_validation', action='store_true')
    # parser.add_argument('--checkpoint_folder_path_to_borrow_metadata_from', type=str, default=None, help='Path to metadata file to borrow trainer kwargs from')
    # parser.add_argument('--override_bbox_format', type=str, default=None, choices=['xyxy', 'cxcywh'], help='Override the bbox format used in the dataset')

    # # Evaluation arguments
    # parser.add_argument('--eval_chest_imagenome_gold', action='store_true')
    # parser.add_argument('--eval_mscxr', action='store_true')
    # parser.add_argument('--eval_chexlocalize', action='store_true')
    # parser.add_argument('--eval_vinbig', action='store_true')
    # parser.add_argument('--optimize_thresholds', action='store_true')
    # parser.add_argument('--candidate_iou_thresholds', type=float, nargs='+', default=None)
    # parser.add_argument('--candidate_conf_thresholds', type=float, nargs='+', default=None)
    # parser.add_argument('--map_iou_thresholds', type=float, nargs='+', default=[0., 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    # parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    # parser.add_argument('--use_classifier_confs_for_map', action='store_true', help='Use classifier confidences for mAP computation')
    
    # --- Step 3: Parse and Validate ---
    args = parser.parse_args()

    if args.subcommand == 'inference':
        # Manual validation for inference args since we removed required=True
        required_args = [
            'dataset_name', 'checkpoint_folder_path', 
            'max_images_per_batch', 'max_phrases_per_batch', 
            'max_phrases_per_image'
        ]
        missing = [arg for arg in required_args if getattr(args, arg) is None]
        if missing:
            parser.error(f"Inference subcommand requires the following arguments (via CLI or YAML): {', '.join(missing)}")
        
        _run_inference(args=args)

    elif args.subcommand == 'metrics':
        _run_metrics_computation(args=args)
    elif args.subcommand == 'export':
        _export_data(args=args)

if __name__ == '__main__':
    main()