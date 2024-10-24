import math
import os
import numpy as np
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.augmentation import ChestImagenomeAlbumentationAdapter
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
    get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_gold_bboxes,
    load_chest_imagenome_silver_bboxes,
    load_gold_bbox_dicom_ids,
    load_nondecent_chest_imagenome_dicom_ids,
)
from medvqa.datasets.dataloading_utils import CompositeInfiniteDataset, SequentialDataLoader, group_indices_for_balanced_sampling
from medvqa.datasets.image_processing import ImageFactBasedMultilabelClassificationDataset, ImageFactClassificationDataset
from medvqa.datasets.ms_cxr import get_ms_cxr_dicom_id_2_phrases_and_masks, get_ms_cxr_dicom_ids
from medvqa.datasets.segmentation_utils import compute_mask_from_bounding_box
from medvqa.datasets.mimiccxr import (
    MIMICCXR_LARGE_FAST_CACHE_DIR,
    MIMICCXR_ImageSizeModes,
    MIMICCXR_ViewModes,
    get_cxrlt2024_train_dev_dicom_ids,
    get_dicom_id_and_orientation_list,
    get_image_path_getter,
    get_imageId2PartPatientStudy,
    get_mimiccxr_train_dicom_ids,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.constants import LABEL_BASED_FACTS
from medvqa.utils.files import get_cached_pickle_file, load_pickle, print_file_size, save_pickle
from medvqa.utils.logging import print_bold, print_magenta, print_orange, print_red

class MIMICCXR_FactGroundingDataset(ImageFactClassificationDataset):

    def __init__(self, image_paths, image_transform, fact_embeddings, positive_facts, negative_facts, indices, num_facts,
                 infinite=False, shuffle=False, use_weights=False, weights_filepath=None):
        super().__init__(
            image_paths=image_paths,
            image_transform=image_transform,
            fact_embeddings=fact_embeddings,
            positive_facts=positive_facts,
            negative_facts=negative_facts,
            indices=indices,
            num_facts=num_facts,
            infinite=infinite,
            shuffle=shuffle,
        )
        if use_weights:
            assert weights_filepath is not None
            self.use_weights = True
            data = get_cached_pickle_file(weights_filepath)
            self.cluster_assignments = data['cluster_assignments']
            self.cluster_weights = data['cluster_weights']
            self.label_weights = data['label_weights']
            self.num_labels = len(self.label_weights)
            assert self.num_labels == len(LABEL_BASED_FACTS) # sanity check
        else:
            self.use_weights = False
    
    def __getitem__(self, i):
        output = super().__getitem__(i)
        if self.use_weights: # add phrase weights
            fact_indices = output['fidxs']
            labels = output['l']
            weights = np.empty(len(labels), dtype=np.float32)
            for i, label in enumerate(labels):
                fidx = fact_indices[i]
                if fidx < self.num_labels: # use label-specific weight
                    weights[i] = self.label_weights[fidx, 1 - label] # 0 -> positive fact, 1 -> negative fact
                else: # use cluster-specific weight
                    weights[i] = self.cluster_weights[self.cluster_assignments[fidx], 1 - label] # 0 -> positive fact, 1 -> negative fact
            output['pw'] = weights # phrase weights
        return output
    
class MIMICCXR_PhraseGroundingDataset(Dataset):

    def __init__(self, image_paths, image_transform, phrases, phrase_embeddings, phrase_grounding_masks, indices):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrases = phrases
        self.phrase_embeddings = phrase_embeddings
        self.phrase_grounding_masks = phrase_grounding_masks
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        # print(f'MIMICCXR_PhraseGroundingDataset.__getitem__({i})')
        idx = self.indices[i]
        image_path = self.image_paths[idx]
        phrase_embeddings = self.phrase_embeddings[idx]
        phrase_grounding_masks = self.phrase_grounding_masks[idx]
        image, phrase_grounding_masks = self.image_transform(image_path, phrase_grounding_masks)
        return {
            'i': image,
            'pe': phrase_embeddings,
            'pgm': phrase_grounding_masks,
        }
    
class MIMICCXR_CXRLT2024_ClassificationDataset(Dataset):

    def __init__(self, image_paths, image_transform, phrase_embeddings, phrase_idxs, phrase_classification_labels,
                 indices, shuffle=False):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_embeddings = phrase_embeddings
        self.phrase_idxs = phrase_idxs
        self.phrase_classification_labels = phrase_classification_labels
        self.indices = indices
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        image_path = self.image_paths[idx]
        image = self.image_transform(image_path)
        phrase_idxs = self.phrase_idxs[idx]
        phrase_embeddings = self.phrase_embeddings[phrase_idxs]
        phrase_classification_labels = self.phrase_classification_labels[idx]
        return {
            'i': image,
            'pe': phrase_embeddings,
            'pi': phrase_idxs,
            'pcl': phrase_classification_labels,
        }
    
def _create_target_tensors(bbox_coords, bbox_presence, feature_map_size):
    """
    Creates a target tensor for bounding boxes on a feature map.
    
    Args:
        bbox_coords: A tensor of shape (N, 4) representing bounding box coordinates.
        bbox_presence: A tensor of shape (N,) representing the presence of bounding boxes.
        feature_map_size: Tuple (H, W) representing the size of the feature map.
    
    Returns:
        A tensor of shape (N, H*W, 4) representing the target tensor with bounding box coordinates (x_min, y_min, x_max, y_max).
        A tensor of shape (N, H*W) representing the target tensor with bounding box presence.
    """
    H, W = feature_map_size
    N = len(bbox_coords)
    target_coords = torch.zeros(N, H, W, 4)
    target_presence = torch.zeros(N, H, W)
    
    for i in range(N):
        if bbox_presence[i] == 0:
            continue
        x_min, y_min, x_max, y_max = bbox_coords[i]
        x_min_scaled = math.floor(x_min * W)
        y_min_scaled = math.floor(y_min * H)
        x_max_scaled = math.ceil(x_max * W)
        y_max_scaled = math.ceil(y_max * H)
        target_coords[i, y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled, :] = torch.tensor([x_min, y_min, x_max, y_max])
        target_presence[i, y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled] = 1

    # Reshape target tensors
    target_coords = target_coords.view(N, H*W, 4)
    target_presence = target_presence.view(N, H*W)
    
    return target_coords, target_presence

class MIMICCXR_BBoxGroundingDataset(Dataset):

    def __init__(self, image_paths, image_transform, phrase_embeddings, phrase_classification_labels,
                 predict_bboxes=False, num_bbox_classes=None, feature_map_size=None, bbox_coords=None, bbox_presence=None,
                 phrase_grounding_masks=None, data_augmentation_enabled=False, for_training=True):
        
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_embeddings = phrase_embeddings
        self.phrase_classification_labels = phrase_classification_labels
        self.predict_bboxes = predict_bboxes
        self.data_augmentation_enabled = data_augmentation_enabled
        self.for_training = for_training
        if predict_bboxes:
            assert num_bbox_classes is not None
            assert feature_map_size is not None
            assert bbox_coords is not None
            assert bbox_presence is not None
            self.num_bbox_classes = num_bbox_classes
            self.feature_map_size = feature_map_size
            self.bbox_coords = bbox_coords
            self.bbox_presence = bbox_presence
            if data_augmentation_enabled:
                self.albumentation_adapter = ChestImagenomeAlbumentationAdapter(num_bbox_classes)
        else:
            assert phrase_grounding_masks is not None
            self.phrase_grounding_masks = phrase_grounding_masks

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        image_path = self.image_paths[i]
        phrase_embeddings = self.phrase_embeddings
        phrase_classification_labels = self.phrase_classification_labels[i]
        if self.predict_bboxes:
            bbox_coords = self.bbox_coords[i]
            bbox_presence = self.bbox_presence[i]
            if self.data_augmentation_enabled:
                image, bbox_coords, bbox_presence = self.image_transform(
                    image_path=image_path,
                    bboxes=bbox_coords,
                    presence=bbox_presence,
                    albumentation_adapter=self.albumentation_adapter,
                )
            else:
                image = self.image_transform(image_path)
            if self.for_training:
                bbox_target_coords, bbox_target_presence = _create_target_tensors(bbox_coords, bbox_presence, self.feature_map_size)
                return {
                    'i': image,
                    'pe': phrase_embeddings,
                    'pcl': phrase_classification_labels,
                    'btc': bbox_target_coords,
                    'btp': bbox_target_presence,
                }
            else:
                return {
                    'i': image,
                    'pe': phrase_embeddings,
                    'pcl': phrase_classification_labels,
                    'bc': bbox_coords,
                    'bp': bbox_presence,
                }
        else:
            phrase_grounding_masks = self.phrase_grounding_masks[i]
            image, phrase_grounding_masks, phrase_classification_labels = self.image_transform(
                image_path, phrase_grounding_masks, phrase_classification_labels)
            return {
                'i': image,
                'pe': phrase_embeddings,
                'pgm': phrase_grounding_masks,
                'pcl': phrase_classification_labels,
            }
    
def _compute_mask_from_bounding_boxes(mask_height, mask_width, bbox_coords, bbox_presence):
    assert len(bbox_coords.shape) == 2
    assert bbox_coords.shape[1] == 4
    assert len(bbox_presence.shape) == 1
    assert bbox_presence.shape[0] == bbox_coords.shape[0]
    mask = np.zeros((len(bbox_coords), mask_height * mask_width), dtype=np.float32)
    for i in range(len(bbox_coords)):
        if bbox_presence[i] == 1:
            x1, y1, x2, y2 = bbox_coords[i]
            mask[i] = compute_mask_from_bounding_box(mask_height, mask_width, x1, y1, x2, y2, flatten=True, binary=True)
    return mask

def _clean_bbox_coords_and_presence(bbox_coords, bbox_presence):
    bbox_coords = bbox_coords.reshape(-1, 4)
    assert len(bbox_coords) == len(bbox_presence)
    bbox_coords.clip(0, 1, out=bbox_coords)
    for i in range(len(bbox_coords)):
        w = bbox_coords[i, 2] - bbox_coords[i, 0]
        h = bbox_coords[i, 3] - bbox_coords[i, 1]
        if w <= 0 or h <= 0:
            bbox_presence[i] = 0
        else:
            assert bbox_presence[i] == 1
    return bbox_coords, bbox_presence

_shared_mask_height = None
_shared_mask_width = None
_shared_did2bboxes = None
def _clean_bbox_coords_and_presence_and_compute_mask(dicom_id):
    bboxes = _shared_did2bboxes[dicom_id]
    bbox_coords = bboxes['coords']
    bbox_presence = bboxes['presence']
    bbox_coords, bbox_presence = _clean_bbox_coords_and_presence(bbox_coords, bbox_presence)
    mask = _compute_mask_from_bounding_boxes(_shared_mask_height, _shared_mask_width, bbox_coords, bbox_presence)
    return bbox_coords, bbox_presence, mask

def _precompute_bbox_coords_and_presence_and_mask(mask_height, mask_width, did2bboxes, num_workers=None):
    save_path = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, f'bbox_coords_and_presence_and_mask({mask_height},{mask_width},{len(did2bboxes)}).pkl')
    if os.path.exists(save_path):
        print_bold(f'Loading precomputed bbox_coords_and_presence_and_mask from {save_path}...')
        print_file_size(save_path)
        return get_cached_pickle_file(save_path)

    print_bold(f'Precomputing bbox_coords_and_presence_and_mask({mask_height},{mask_width},{len(did2bboxes)})...')
    global _shared_mask_height, _shared_mask_width, _shared_did2bboxes
    _shared_mask_height = mask_height
    _shared_mask_width = mask_width
    _shared_did2bboxes = did2bboxes
    dicom_ids = list(did2bboxes.keys())
    dicom_ids.sort() # sort for reproducibility
    import multiprocessing as mp
    if num_workers is None:
        num_workers = mp.cpu_count()
    with mp.Pool(num_workers) as pool:
        results = pool.map(_clean_bbox_coords_and_presence_and_compute_mask, dicom_ids)
    bbox_coords = np.array([r[0] for r in results])
    bbox_presence = np.array([r[1] for r in results])
    phrase_grounding_masks = np.array([r[2] for r in results], dtype=np.uint8) # use uint8 in order to save memory
    print(f'bbox_coords.shape = {bbox_coords.shape}')
    print(f'bbox_presence.shape = {bbox_presence.shape}')
    print(f'phrase_grounding_masks.shape = {phrase_grounding_masks.shape}')
    output = {
        'dicom_ids': dicom_ids, # sorted
        'bbox_coords': bbox_coords,
        'bbox_presence': bbox_presence,
        'phrase_grounding_masks': phrase_grounding_masks,
    }
    save_pickle(output, save_path)
    return output

def visualize_mask(mask, mask_height, mask_width): # for debugging
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    # use gray scale, where white = 1 and black = 0
    plt.imshow(mask.reshape(mask_height, mask_width), cmap='gray', vmin=0, vmax=1)
    plt.show()

_LONG_TAIL = 0
_MIDDLE_TAIL = 1
_SHORT_TAIL = 2

def _assign_distribution_classes_to_reports(report_fact_nli_integrated_data_filepath,
                                            distribution_thresholds=(0.02, 0.05)):
    assert len(distribution_thresholds) == 2
    assert 0 < distribution_thresholds[0] < distribution_thresholds[1] < 1
    print_bold(f'Assigning distribution classes to reports...')
    # Load the data
    print(f'Loading mimiccxr_report_fact_nli_integrated_data from {report_fact_nli_integrated_data_filepath}...')
    data = load_pickle(report_fact_nli_integrated_data_filepath)
    labels = data['label_based_nli_predictions']['nli_predictions']
    print(f'labels.shape = {labels.shape}')
    n_reports = labels.shape[0]
    binary_labels = labels == 0
    count_per_class = binary_labels.sum(0)
    count_no_positives = (binary_labels.sum(1) == 0).sum() # number of reports with no positive class
    distribution_classes = np.zeros(n_reports, dtype=np.int8)    
    t0, t1 = distribution_thresholds
    if count_no_positives < t0 * n_reports:
        no_positive_class = _LONG_TAIL
    elif count_no_positives < t1 * n_reports:
        no_positive_class = _MIDDLE_TAIL
    else:
        no_positive_class = _SHORT_TAIL
    is_long_tail = count_per_class < t0 * n_reports
    is_middle_tail = (t0 * n_reports <= count_per_class) & (count_per_class < t1 * n_reports)
    is_short_tail = count_per_class >= t1 * n_reports
    
    for i in range(n_reports):
        if binary_labels[i].sum() == 0: # no positive class
            distribution_classes[i] = no_positive_class
        elif is_long_tail[binary_labels[i]].any(): # at least one long tail class
            distribution_classes[i] = _LONG_TAIL
        elif is_middle_tail[binary_labels[i]].any(): # at least one middle tail class
            distribution_classes[i] = _MIDDLE_TAIL
        else: # all short tail classes
            assert is_short_tail[binary_labels[i]].all()
            distribution_classes[i] = _SHORT_TAIL

    print(f'count_no_positives = {count_no_positives}')
    print(f'number of long tail classes = {(is_long_tail).sum()}')
    print(f'number of middle tail classes = {(is_middle_tail).sum()}')
    print(f'number of short tail classes = {(is_short_tail).sum()}')
    print(f'number of long tail reports = {(distribution_classes == _LONG_TAIL).sum()}')
    print(f'number of middle tail reports = {(distribution_classes == _MIDDLE_TAIL).sum()}')
    print(f'number of short tail reports = {(distribution_classes == _SHORT_TAIL).sum()}')

    return distribution_classes

class MIMICCXR_PhraseGroundingTrainer:

    def __init__(self, 
                max_images_per_batch, max_phrases_per_batch, max_phrases_per_image,
                num_train_workers=None, num_test_workers=None,
                train_image_transform=None, test_image_transform=None,
                fact_grounding_collate_batch_fn=None,
                phrase_grounding_collate_batch_fn=None,
                bbox_grounding_collate_batch_fn=None,
                cxrlt2024_image_phrase_classifier_collate_batch_fn=None,
                cxrlt2024_multilabel_classifier_collate_batch_fn=None,
                test_batch_size_factor=1,
                mask_width=None, mask_height=None,
                use_facts_for_train=False,
                use_facts_for_test=False,
                dicom_id_to_pos_neg_facts_filepath=None,
                use_mscxr_for_train=False,
                use_mscxr_for_test=False,
                mscxr_phrase2embedding_filepath=None,
                use_chest_imagenome_for_train=False,
                use_chest_imagenome_gold_for_test=False,
                chest_imagenome_bbox_phrase_embeddings_filepath=None,
                source_image_size_mode=MIMICCXR_ImageSizeModes.SMALL_256x256,
                exclude_noisy_images=False,
                balance_long_middle_short_tail=False,
                long_middle_short_tail_thresholds=(0.02, 0.05),
                report_fact_nli_integrated_data_filepath=None,
                use_weighted_phrase_classifier_loss=False,
                cluster_and_label_weights_for_facts_filepath=None,
                use_interpret_cxr_challenge_split=False,
                interpret_cxr_challenge_split_filepath=None,
                use_cxrlt2024_challenge_split=False,
                use_cxrlt2024_custom_labels=False,
                use_cxrlt2024_official_labels=False,
                use_all_cxrlt2024_official_labels_for_training=False,
                cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath=None,
                cxrlt2024_official_training_labels_for_fact_classification_filepath=None,
                cxrlt2024_do_balanced_sampling=False,
                do_visual_grounding_with_bbox_regression=False,
                data_augmentation_enabled=False,
                **unused_kwargs,
            ):

        if len(unused_kwargs) > 0:
            # Print warning in orange and bold
            print('\033[93m\033[1mWarning: unused kwargs in MIMICCXR_VisualModuleTrainer: {}\033[0m'.format(unused_kwargs))
        # Sanity checks
        assert sum([use_facts_for_train, use_chest_imagenome_for_train, use_mscxr_for_train,
                    use_mscxr_for_test, use_chest_imagenome_gold_for_test, use_cxrlt2024_challenge_split]) > 0 # at least one of them must be True
        
        self.use_facts_for_train = use_facts_for_train
        self.use_chest_imagenome_for_train = use_chest_imagenome_for_train
        self.use_mscxr_for_train = use_mscxr_for_train
        self.use_mscxr_for_test = use_mscxr_for_test
        self.use_chest_imagenome_gold_for_test = use_chest_imagenome_gold_for_test
        self.use_cxrlt2024_challenge_split = use_cxrlt2024_challenge_split
        self.use_cxrlt2024_custom_labels = use_cxrlt2024_custom_labels
        self.use_cxrlt2024_official_labels = use_cxrlt2024_official_labels
        self.use_all_cxrlt2024_official_labels_for_training = use_all_cxrlt2024_official_labels_for_training

        forbidden_train_dicom_ids = set()

        if exclude_noisy_images:
            noisy_dicom_ids = set(load_nondecent_chest_imagenome_dicom_ids())
            forbidden_train_dicom_ids |= noisy_dicom_ids

        if use_cxrlt2024_challenge_split:
            assert use_cxrlt2024_custom_labels or use_cxrlt2024_official_labels
            assert not use_mscxr_for_test # only for training
            assert not use_chest_imagenome_gold_for_test # only for training
            assert not use_facts_for_test # only for training
            assert not use_interpret_cxr_challenge_split
            cxrlt2024_train_dicom_ids, cxrlt2024_dev_dicom_ids = get_cxrlt2024_train_dev_dicom_ids()
            cxrlt2024_train_dicom_ids = set(cxrlt2024_train_dicom_ids)
            cxrlt2024_dev_dicom_ids = set(cxrlt2024_dev_dicom_ids)
            print(f'len(cxrlt2024_train_dicom_ids) = {len(cxrlt2024_train_dicom_ids)}')
            print(f'len(cxrlt2024_dev_dicom_ids) = {len(cxrlt2024_dev_dicom_ids)}')
            forbidden_train_dicom_ids |= cxrlt2024_dev_dicom_ids # exclude dev set

        if use_cxrlt2024_official_labels:
            assert use_cxrlt2024_challenge_split
            assert cxrlt2024_official_training_labels_for_fact_classification_filepath is not None
            assert cxrlt2024_multilabel_classifier_collate_batch_fn is not None

        if use_cxrlt2024_custom_labels:
            # assert use_cxrlt2024_challenge_split
            assert cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath is not None
            assert cxrlt2024_image_phrase_classifier_collate_batch_fn is not None

        if use_mscxr_for_test:
            ms_cxr_dicom_ids = set(get_ms_cxr_dicom_ids())
            if not use_mscxr_for_train:
                forbidden_train_dicom_ids |= ms_cxr_dicom_ids

        if use_chest_imagenome_gold_for_test:
            gold_dicom_ids = set(load_gold_bbox_dicom_ids())
            forbidden_train_dicom_ids |= gold_dicom_ids

        print(f'len(forbidden_train_dicom_ids) = {len(forbidden_train_dicom_ids)}')

        # Create train and dev datasets for CXR-LT-2024 challenge
        if use_cxrlt2024_challenge_split or self.use_cxrlt2024_custom_labels:
            print_magenta('Preparing CXR-LT-2024 challenge datasets and dataloaders for training/testing...', bold=True)

            imageId2PartPatientStudy = get_imageId2PartPatientStudy()
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            if self.use_cxrlt2024_custom_labels:

                assert cxrlt2024_image_phrase_classifier_collate_batch_fn is not None
                assert cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath is not None

                cxrlt2024_dicom_id_to_pos_neg_facts = load_pickle(cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath)
                print(f'len(cxrlt2024_dicom_id_to_pos_neg_facts["train"]) = {len(cxrlt2024_dicom_id_to_pos_neg_facts["train"])}')
                print(f'len(cxrlt2024_dicom_id_to_pos_neg_facts["dev"]) = {len(cxrlt2024_dicom_id_to_pos_neg_facts["dev"])}')

                BIG_ENOGUGH = 1000000
                phrases = cxrlt2024_dicom_id_to_pos_neg_facts['class_sentences']
                phrase_embeddings = cxrlt2024_dicom_id_to_pos_neg_facts['class_sentence_embeddings']
                image_paths = [None] * BIG_ENOGUGH
                phrase_idxs = [None] * BIG_ENOGUGH
                phrase_classification_labels = [None] * BIG_ENOGUGH
                idx = 0

                train_indices = []
                dev_indices = []

                if use_cxrlt2024_challenge_split:
                    
                    for dicom_id, (neg_idxs, pos_idxs) in cxrlt2024_dicom_id_to_pos_neg_facts['train'].items():
                        if dicom_id in forbidden_train_dicom_ids:
                            continue
                        assert dicom_id in cxrlt2024_train_dicom_ids
                        part_id, subject_id, study_id = imageId2PartPatientStudy[dicom_id]
                        image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                        phrase_idxs[idx] = neg_idxs + pos_idxs
                        phrase_classification_labels[idx] = np.concatenate([np.zeros(len(neg_idxs), dtype=np.int64),
                                                                            np.ones(len(pos_idxs), dtype=np.int64)])
                        train_indices.append(idx)
                        idx += 1

                    for dicom_id, (neg_idxs, pos_idxs) in cxrlt2024_dicom_id_to_pos_neg_facts['dev'].items():
                        part_id, subject_id, study_id = imageId2PartPatientStudy[dicom_id]
                        image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                        phrase_idxs[idx] = neg_idxs + pos_idxs
                        phrase_classification_labels[idx] = np.concatenate([np.zeros(len(neg_idxs), dtype=np.int64),
                                                                            np.ones(len(pos_idxs), dtype=np.int64)])
                        dev_indices.append(idx)
                        idx += 1

                else:

                    # Use MIMIC-CXR's original train/dev split by default
                    print_orange('Using MIMIC-CXR\'s original train/dev split for CXR-LT-2024 challenge', bold=True)
                    train_dicom_ids = set(get_mimiccxr_train_dicom_ids())
                    for key in ['train', 'dev']:
                        for dicom_id, (neg_idxs, pos_idxs) in cxrlt2024_dicom_id_to_pos_neg_facts[key].items():
                            if dicom_id in train_dicom_ids:
                                if dicom_id in forbidden_train_dicom_ids:
                                    continue
                                part_id, subject_id, study_id = imageId2PartPatientStudy[dicom_id]
                                image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                                phrase_idxs[idx] = neg_idxs + pos_idxs
                                phrase_classification_labels[idx] = np.concatenate([np.zeros(len(neg_idxs), dtype=np.int64),
                                                                                    np.ones(len(pos_idxs), dtype=np.int64)])
                                train_indices.append(idx)
                                idx += 1
                            else:
                                part_id, subject_id, study_id = imageId2PartPatientStudy[dicom_id]
                                image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                                phrase_idxs[idx] = neg_idxs + pos_idxs
                                phrase_classification_labels[idx] = np.concatenate([np.zeros(len(neg_idxs), dtype=np.int64),
                                                                                    np.ones(len(pos_idxs), dtype=np.int64)])
                                dev_indices.append(idx)
                                idx += 1

                print(f'Total number of images: {idx}')
                image_paths = image_paths[:idx]
                phrase_idxs = phrase_idxs[:idx]
                phrase_classification_labels = phrase_classification_labels[:idx]

                # Create dataset and dataloader for training
                print_bold('Building cxrlt2024 train phrase classifier dataloader...')
                num_phrases_2_idxs = {}
                for idx in train_indices:
                    num_phrases = len(phrase_idxs[idx])
                    try:
                        num_phrases_2_idxs[num_phrases].append(idx)
                    except KeyError:
                        num_phrases_2_idxs[num_phrases] = [idx]
                train_dataloaders = []
                for num_phrases, indices in num_phrases_2_idxs.items():
                    print(f'num_phrases = {num_phrases}, len(indices) = {len(indices)}')
                    dataset = MIMICCXR_CXRLT2024_ClassificationDataset(
                        image_paths=image_paths, image_transform=train_image_transform,
                        phrase_embeddings=phrase_embeddings, phrase_idxs=phrase_idxs,
                        phrase_classification_labels=phrase_classification_labels,
                        indices=indices, shuffle=True)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=max(min(max_images_per_batch, max_phrases_per_batch // num_phrases), 1), # at least 1
                        shuffle=True,
                        num_workers=num_train_workers,
                        collate_fn=cxrlt2024_image_phrase_classifier_collate_batch_fn,
                        pin_memory=True,
                    )
                    train_dataloaders.append(dataloader)
                self.cxrlt2024_custom_train_dataloader = SequentialDataLoader(train_dataloaders)

                # Create dataset and dataloader for dev
                print_bold('Building cxrlt2024 dev phrase classifier dataloader...')
                num_phrases_2_idxs = {}
                for idx in dev_indices:
                    num_phrases = len(phrase_idxs[idx])
                    try:
                        num_phrases_2_idxs[num_phrases].append(idx)
                    except KeyError:
                        num_phrases_2_idxs[num_phrases] = [idx]
                dev_dataloaders = []
                for num_phrases, indices in num_phrases_2_idxs.items():
                    print(f'num_phrases = {num_phrases}, len(indices) = {len(indices)}')
                    dataset = MIMICCXR_CXRLT2024_ClassificationDataset(
                        image_paths=image_paths, image_transform=test_image_transform,
                        phrase_embeddings=phrase_embeddings, phrase_idxs=phrase_idxs,
                        phrase_classification_labels=phrase_classification_labels,
                        indices=indices, shuffle=False)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=int(max(min(max_images_per_batch, max_phrases_per_batch // num_phrases), 1) * test_batch_size_factor), # at least 1
                        shuffle=False,
                        num_workers=num_test_workers,
                        collate_fn=cxrlt2024_image_phrase_classifier_collate_batch_fn,
                        pin_memory=True,
                    )
                    dev_dataloaders.append(dataloader)
                self.cxrlt2024_custom_dev_dataloader = SequentialDataLoader(dev_dataloaders)

            if self.use_cxrlt2024_official_labels:
                # We will create an additional dataloader for training the fact classifier
                # using the official training labels provided by the challenge organizers
                print_bold('Building cxrlt2024 train/val dataloader for fact classifier using official labels...')
                data = load_pickle(cxrlt2024_official_training_labels_for_fact_classification_filepath)
                dicom_ids = data['dicom_ids']
                labels = data['labels']
                train_indices = data['train_indices']
                val_indices = data['val_indices']
                class_names = data['class_names']
                class_embeddings = data['class_embeddings']
                assert labels.shape == (len(dicom_ids), len(class_embeddings))
                assert len(train_indices) + len(val_indices) == len(dicom_ids)
                print(f'len(dicom_ids) = {len(dicom_ids)}')
                print(f'len(train_indices) = {len(train_indices)}')
                print(f'len(val_indices) = {len(val_indices)}')
                print(f'labels.shape = {labels.shape}')
                print(f'class_embeddings.shape = {class_embeddings.shape}')
                if use_all_cxrlt2024_official_labels_for_training:
                    print_orange('Using all CXR-LT-2024 official labels for training the fact classifier', bold=True)
                    train_indices += val_indices
                    val_indices = []
                    print(f'len(train_indices) = {len(train_indices)}')
                    print(f'len(val_indices) = {len(val_indices)}')
                image_paths = []
                for dicom_id in dicom_ids:
                    part_id, subject_id, study_id = imageId2PartPatientStudy[dicom_id]
                    image_paths.append(image_path_getter(part_id, subject_id, study_id, dicom_id))

                # Clean train_indices
                len_before = len(train_indices)
                train_indices = [i for i in train_indices if dicom_ids[i] not in forbidden_train_dicom_ids]
                len_after = len(train_indices)
                if len_before != len_after:
                    print_orange(f'{len_before - len_after} train indices removed due to forbidden_train_dicom_ids')

                if cxrlt2024_do_balanced_sampling:
                    print_orange('Balanced sampling is enabled for CXR-LT-2024 fact classifier training')
                    # Group train indices for balanced sampling
                    grouped_indices = group_indices_for_balanced_sampling(
                        label_matrix=labels, indices=train_indices, label_names=class_names, min_group_size=100)
                    
                    # Create train dataset
                    train_datasets = []
                    train_weights = []
                    for indices in grouped_indices:
                        dataset = ImageFactBasedMultilabelClassificationDataset(
                            image_paths=image_paths,
                            image_transform=train_image_transform,
                            phrase_embeddings=class_embeddings,
                            phrase_classification_labels=labels,
                            indices=indices,
                            infinite=True,
                            shuffle_indices=True,
                        )
                        weight = math.log2(len(indices)) ** 3
                        train_datasets.append(dataset)
                        train_weights.append(weight)
                        print(f'  len(indices) = {len(indices)}, weight = {weight}')
                    self.cxrlt2024_official_train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
                    batch_size = max(min(max_images_per_batch, max_phrases_per_batch // len(class_names)), 1) # at least 1 image per batch
                    self.cxrlt2024_official_train_dataloader = DataLoader(self.cxrlt2024_official_train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_train_workers,
                                                    collate_fn=cxrlt2024_multilabel_classifier_collate_batch_fn,
                                                    pin_memory=True)
                else:
                    print_orange('Normal sampling is enabled for CXR-LT-2024 fact classifier training')
                    self.cxrlt2024_official_train_dataset = ImageFactBasedMultilabelClassificationDataset(
                        image_paths=image_paths,
                        image_transform=train_image_transform,
                        phrase_embeddings=class_embeddings,
                        phrase_classification_labels=labels,
                        indices=train_indices,
                        infinite=False,
                        shuffle_indices=True,
                    )
                    batch_size = max(min(max_images_per_batch, max_phrases_per_batch // len(class_names)), 1) # at least 1 image per batch
                    self.cxrlt2024_official_train_dataloader = DataLoader(self.cxrlt2024_official_train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_train_workers,
                                                    collate_fn=cxrlt2024_multilabel_classifier_collate_batch_fn,
                                                    pin_memory=True)
                
                # Create val dataset
                if len(val_indices) > 0:
                    print_orange('Creating CXR-LT-2024 official val dataset and dataloader...')
                    val_dataset = ImageFactBasedMultilabelClassificationDataset(
                        image_paths=image_paths,
                        image_transform=test_image_transform,
                        phrase_embeddings=class_embeddings,
                        phrase_classification_labels=labels,
                        indices=val_indices,
                        infinite=False,
                        shuffle_indices=False,
                    )
                    self.cxrlt2024_official_val_dataloader = DataLoader(val_dataset,
                                                    batch_size=int(batch_size * test_batch_size_factor),
                                                    shuffle=False,
                                                    num_workers=num_test_workers,
                                                    collate_fn=cxrlt2024_multilabel_classifier_collate_batch_fn,
                                                    pin_memory=True)

        # Create train mimiccxr facts dataset
        if use_facts_for_train or use_facts_for_test:
            print_magenta('Preparing MIMIC-CXR-Facts datasets and dataloaders for training/testing...', bold=True)
            assert dicom_id_to_pos_neg_facts_filepath is not None
            assert fact_grounding_collate_batch_fn is not None
            assert num_train_workers is not None
            assert train_image_transform is not None

            tmp = load_pickle(dicom_id_to_pos_neg_facts_filepath)
            fact_embeddings = tmp['embeddings']
            dicom_id_to_pos_neg_facts = tmp['dicom_id_to_pos_neg_facts']
            print(f'fact_embeddings.shape = {fact_embeddings.shape}')

            if balance_long_middle_short_tail and use_facts_for_train: # only for training
                assert report_fact_nli_integrated_data_filepath is not None
                distribution_classes = _assign_distribution_classes_to_reports(
                    report_fact_nli_integrated_data_filepath=report_fact_nli_integrated_data_filepath,
                    distribution_thresholds=long_middle_short_tail_thresholds)

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            positive_facts = [None] * BIG_ENOGUGH
            negative_facts = [None] * BIG_ENOGUGH
            report_idxs = [None] * BIG_ENOGUGH
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()

            if use_facts_for_train:
                train_indices = []
            if use_facts_for_test:
                test_indices = []

            if use_interpret_cxr_challenge_split:
                assert interpret_cxr_challenge_split_filepath is not None
                print_bold(f'Using split from {interpret_cxr_challenge_split_filepath}')
                challenge_split = load_pickle(interpret_cxr_challenge_split_filepath)
                challenge_train_dicom_ids = set(challenge_split['train'])
                challenge_val_dicom_ids = set(challenge_split['val'])
                # ignore test set because it is kept hidden in the challenge

            idx = 0
            for ridx, (part_id, subject_id, study_id, dicom_id_view_pairs, split) in \
                tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs'],
                    mimiccxr_metadata['splits'])), mininterval=2):
                for dicom_id, view in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id not in dicom_id_to_pos_neg_facts:
                        continue
                    
                    if use_interpret_cxr_challenge_split:
                        if dicom_id in challenge_train_dicom_ids and use_facts_for_train:
                            split = 'train'
                        elif dicom_id in challenge_val_dicom_ids and use_facts_for_test:
                            split = 'validate'
                        else:
                            continue # ignore test set

                    if use_cxrlt2024_challenge_split:
                        if dicom_id in cxrlt2024_train_dicom_ids:
                            split = 'train'
                        else:
                            continue # ignore dev set

                    if split == 'validate' or split == 'test':
                        if use_facts_for_test:
                            test_indices.append(idx)
                        else:
                            if dicom_id in forbidden_train_dicom_ids:
                                continue
                            train_indices.append(idx)
                    elif split == 'train':
                        if use_facts_for_train:
                            if dicom_id in forbidden_train_dicom_ids:
                                continue
                            train_indices.append(idx)
                        else:
                            continue
                    else:
                        raise ValueError(f'Invalid split: {split}')
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    pos_neg_facts = dicom_id_to_pos_neg_facts[dicom_id]
                    assert len(pos_neg_facts) == 2
                    positive_facts[idx] = pos_neg_facts[0]
                    negative_facts[idx] = pos_neg_facts[1]
                    report_idxs[idx] = ridx
                    idx += 1

            print(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
            positive_facts = positive_facts[:idx]
            negative_facts = negative_facts[:idx]
            report_idxs = report_idxs[:idx]
            aux = 0
            if use_facts_for_train:
                print(f'len(train_indices) = {len(train_indices)}')
                aux += len(train_indices)
            if use_facts_for_test:
                print(f'len(test_indices) = {len(test_indices)}')
                aux += len(test_indices)
            assert aux == idx # sanity check

            # Calculate the average number of facts per image
            if use_facts_for_train:
                aux = 0
                for i in train_indices:
                    pos_facts = positive_facts[i]
                    neg_facts = negative_facts[i]
                    assert len(pos_facts) + len(neg_facts) > 0 # at least one fact
                    aux += max(len(pos_facts), len(neg_facts))
                avg_facts_per_image = aux / len(train_indices)
                train_num_facts_per_image = min(max_phrases_per_image, int(avg_facts_per_image))
                print(f'avg_facts_per_image = {avg_facts_per_image}')
                print(f'train_num_facts_per_image = {train_num_facts_per_image}')
            if use_facts_for_test:
                aux = 0
                for i in test_indices:
                    pos_facts = positive_facts[i]
                    neg_facts = negative_facts[i]
                    assert len(pos_facts) + len(neg_facts) > 0 # at least one fact
                    aux += max(len(pos_facts), len(neg_facts))
                avg_facts_per_image = aux / len(test_indices)
                test_num_facts_per_image = min(max_phrases_per_image, int(avg_facts_per_image))
                print(f'avg_facts_per_image = {avg_facts_per_image}')
                print(f'test_num_facts_per_image = {test_num_facts_per_image}')

            # Create dataset and dataloader for training
            if use_facts_for_train:
                print_bold('Building train fact dataloader...')
                batch_size = max(min(max_images_per_batch, max_phrases_per_batch // train_num_facts_per_image), 1) # at least 1
                print(f'batch_size = {batch_size}')

                if balance_long_middle_short_tail:
                    print('Balancing long, middle, and short tail classes...')
                    long_tail_indices = [i for i in train_indices if distribution_classes[report_idxs[i]] == _LONG_TAIL]
                    middle_tail_indices = [i for i in train_indices if distribution_classes[report_idxs[i]] == _MIDDLE_TAIL]
                    short_tail_indices = [i for i in train_indices if distribution_classes[report_idxs[i]] == _SHORT_TAIL]
                    assert len(long_tail_indices) > 0
                    assert len(middle_tail_indices) > 0
                    assert len(short_tail_indices) > 0
                    assert len(long_tail_indices) + len(middle_tail_indices) + len(short_tail_indices) == len(train_indices)
                    indices_list = [long_tail_indices, middle_tail_indices, short_tail_indices]
                    indices_list.sort(key=lambda x: len(x), reverse=True)
                    while len(indices_list) >= 2 and len(indices_list[-1]) < 100:
                        indices_list[-2].extend(indices_list[-1])
                        indices_list.pop()
                    assert len(indices_list) >= 1
                    assert len(indices_list[0]) > 0
                    datasets = []
                    for indices_ in indices_list:
                        dataset = MIMICCXR_FactGroundingDataset(
                            image_paths=image_paths, image_transform=train_image_transform,
                            fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                            indices=indices_, num_facts=train_num_facts_per_image, shuffle=True, infinite=True,
                            use_weights=use_weighted_phrase_classifier_loss,
                            weights_filepath=cluster_and_label_weights_for_facts_filepath)
                        datasets.append(dataset)
                    if len(datasets) == 1:
                        train_fact_dataset = datasets[0]
                    else:
                        weights = [1] * len(datasets) # equal weights
                        train_fact_dataset = CompositeInfiniteDataset(datasets, weights)
                    train_fact_dataloader = DataLoader(
                        train_fact_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_train_workers,
                        collate_fn=fact_grounding_collate_batch_fn,
                        pin_memory=True,
                    )
                else:
                    print('Normal (unbalanced) training...')
                    train_fact_dataset = MIMICCXR_FactGroundingDataset(
                        image_paths=image_paths, image_transform=train_image_transform,
                        fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                        indices=train_indices, num_facts=train_num_facts_per_image,
                        use_weights=use_weighted_phrase_classifier_loss,
                        weights_filepath=cluster_and_label_weights_for_facts_filepath)
                    train_fact_dataloader = DataLoader(
                        train_fact_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_train_workers,
                        collate_fn=fact_grounding_collate_batch_fn,
                        pin_memory=True,
                    )
                self.train_fact_dataset = train_fact_dataset
                self.train_fact_dataloader = train_fact_dataloader
                print(f'len(self.train_fact_dataloader) = {len(self.train_fact_dataloader)}')

            # Create dataset and dataloader for testing
            if use_facts_for_test:
                print_bold('Building test fact dataloaders...')
                test_fact_dataset = MIMICCXR_FactGroundingDataset(
                    image_paths=image_paths, image_transform=test_image_transform,
                    fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                    indices=test_indices, num_facts=test_num_facts_per_image,
                    use_weights=use_weighted_phrase_classifier_loss,
                    weights_filepath=cluster_and_label_weights_for_facts_filepath)
                batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // test_num_facts_per_image), 1) * test_batch_size_factor)
                test_fact_dataloader = DataLoader(
                    test_fact_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_test_workers,
                    collate_fn=fact_grounding_collate_batch_fn,
                    pin_memory=True,
                )
                self.test_fact_dataset = test_fact_dataset
                self.test_fact_dataloader = test_fact_dataloader
                print(f'len(self.test_fact_dataloader) = {len(self.test_fact_dataloader)}')

        # Create mscxr train/test dataset and dataloader
        if use_mscxr_for_train or use_mscxr_for_test:
            print_magenta('Preparing MS-CXR dataset and dataloader for training/testing...', bold=True)
            assert mask_width is not None
            assert mask_height is not None
            assert mscxr_phrase2embedding_filepath is not None
            assert phrase_grounding_collate_batch_fn is not None
            if use_mscxr_for_train:
                assert num_train_workers is not None
                assert train_image_transform is not None
            if use_mscxr_for_test:
                assert num_test_workers is not None
                assert test_image_transform is not None                
            
            dicom_id_2_phrases_and_masks = get_ms_cxr_dicom_id_2_phrases_and_masks(mask_height, mask_width)
            print(f'len(dicom_id_2_phrases_and_masks) = {len(dicom_id_2_phrases_and_masks)}')

            phrase2embedding = load_pickle(mscxr_phrase2embedding_filepath)
            print(f'len(phrase2embedding) = {len(phrase2embedding)}')

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            dicom_ids = [None] * BIG_ENOGUGH
            phrases = [None] * BIG_ENOGUGH
            phrase_embeddings = [None] * BIG_ENOGUGH
            phrase_grounding_masks = [None] * BIG_ENOGUGH
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()

            idx = 0
            for _, (part_id, subject_id, study_id, dicom_id_view_pairs) in \
                tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs'])), mininterval=2):
                for dicom_id, _ in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id not in dicom_id_2_phrases_and_masks:
                        continue
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    phrases_, masks_ = dicom_id_2_phrases_and_masks[dicom_id]
                    phrases[idx] = phrases_
                    phrase_embeddings[idx] = np.array([phrase2embedding[p] for p in phrases_])
                    phrase_grounding_masks[idx] = np.array(masks_)
                    idx += 1

            print(f'Total number of images: {idx}')
            assert idx > 0
            image_paths = image_paths[:idx]
            dicom_ids = dicom_ids[:idx]
            phrases = phrases[:idx]
            phrase_embeddings = phrase_embeddings[:idx]
            phrase_grounding_masks = phrase_grounding_masks[:idx]

            # Create a mapping from number of phrases to indices
            num_phrases_2_idxs = {}
            for i, pe in enumerate(phrase_embeddings):
                num_phrases = len(pe)
                try:
                    num_phrases_2_idxs[num_phrases].append(i)
                except KeyError:
                    num_phrases_2_idxs[num_phrases] = [i]

            # Create datasets and dataloaders for training and testing
            if use_mscxr_for_train:
                train_mscxr_dataloaders = []
            if use_mscxr_for_test:
                test_mscxr_dataloaders = []
            for num_phrases, indices in num_phrases_2_idxs.items():
                print(f'Number of phrases: {num_phrases}, # images: {len(indices)}')
                # Sanity checks
                for i in indices:
                    assert phrase_embeddings[i].shape == phrase_embeddings[indices[0]].shape
                    assert phrase_grounding_masks[i].shape == phrase_grounding_masks[indices[0]].shape
                    assert phrase_embeddings[i].shape[0] == phrase_grounding_masks[i].shape[0] == num_phrases
                
                if use_mscxr_for_train:
                    clean_indices = [i for i in indices if dicom_ids[i] not in forbidden_train_dicom_ids] # exclude forbidden images
                    if use_cxrlt2024_challenge_split:
                        clean_indices = [i for i in clean_indices if dicom_ids[i] in cxrlt2024_train_dicom_ids] # it has to be in the train set
                    if len(clean_indices) < len(indices):
                        print_orange(f'Excluding {len(indices) - len(clean_indices)} images from training...', bold=True)
                        if len(clean_indices) == 0:
                            print_red('WARNING: No images left for training!', bold=True)
                            continue
                    batch_size = max(min(max_images_per_batch, max_phrases_per_batch // num_phrases), 1)
                    dataset = MIMICCXR_PhraseGroundingDataset(
                        image_paths=image_paths, image_transform=train_image_transform, phrases=phrases,
                        phrase_embeddings=phrase_embeddings, phrase_grounding_masks=phrase_grounding_masks,
                        indices=clean_indices)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_train_workers,
                        collate_fn=phrase_grounding_collate_batch_fn,
                        pin_memory=True,
                    )
                    train_mscxr_dataloaders.append(dataloader)
                if use_mscxr_for_test:
                    batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // num_phrases), 1) * test_batch_size_factor)
                    dataset = MIMICCXR_PhraseGroundingDataset(
                        image_paths=image_paths, image_transform=test_image_transform, phrases=phrases,
                        phrase_embeddings=phrase_embeddings, phrase_grounding_masks=phrase_grounding_masks,
                        indices=indices)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_test_workers,
                        collate_fn=phrase_grounding_collate_batch_fn,
                        pin_memory=True,
                    )
                    test_mscxr_dataloaders.append(dataloader)
            if use_mscxr_for_train:
                self.train_mscxr_dataloader = SequentialDataLoader(train_mscxr_dataloaders)
            if use_mscxr_for_test:
                self.test_mscxr_dataloader = SequentialDataLoader(test_mscxr_dataloaders)

        # Create train chest imagenome dataset
        if use_chest_imagenome_for_train:
            print_magenta('Preparing Chest Imagenome dataset and dataloader for training...', bold=True)

            assert chest_imagenome_bbox_phrase_embeddings_filepath is not None
            assert mask_width is not None
            assert mask_height is not None
            assert bbox_grounding_collate_batch_fn is not None
            assert num_train_workers is not None
            assert train_image_transform is not None

            print(f'Loading bbox_phrase_embeddings and bbox_phrases from {chest_imagenome_bbox_phrase_embeddings_filepath}...')
            tmp = get_cached_pickle_file(chest_imagenome_bbox_phrase_embeddings_filepath)
            bbox_phrase_embeddings = tmp['bbox_phrase_embeddings']
            bbox_phrases = tmp['bbox_phrases']
            assert bbox_phrase_embeddings.shape[0] == len(bbox_phrases)
            print(f'bbox_phrase_embeddings.shape = {bbox_phrase_embeddings.shape}')
            print(f'len(bbox_phrases) = {len(bbox_phrases)}')
            for phrase in bbox_phrases:
                print('\t', phrase)

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            chest_imagenome_bbox_coords = [None] * BIG_ENOGUGH
            chest_imagenome_bbox_presence = [None] * BIG_ENOGUGH
            phrase_grounding_masks = [None] * BIG_ENOGUGH
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()
            did2bboxes = load_chest_imagenome_silver_bboxes()
            tmp = _precompute_bbox_coords_and_presence_and_mask(mask_height, mask_width, did2bboxes)
            dicom_ids = tmp['dicom_ids']
            did2idx = {dicom_id: idx for idx, dicom_id in enumerate(dicom_ids)}
            bbox_coords_array = tmp['bbox_coords']
            bbox_presence_array = tmp['bbox_presence']
            phrase_grounding_masks_array = tmp['phrase_grounding_masks']
            assert phrase_grounding_masks_array.dtype == np.uint8
            assert phrase_grounding_masks_array.min() == 0
            assert phrase_grounding_masks_array.max() == 1

            idx = 0
            for part_id, subject_id, study_id, dicom_id_view_pairs in \
                tqdm(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs']), mininterval=2):
                for dicom_id, _ in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id not in did2idx:
                        continue
                    if dicom_id in forbidden_train_dicom_ids:
                        continue
                    if use_cxrlt2024_challenge_split:
                        if dicom_id not in cxrlt2024_train_dicom_ids: # it has to be in the challenge's official training set
                            continue
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    i = did2idx[dicom_id]
                    chest_imagenome_bbox_coords[idx] = bbox_coords_array[i]
                    chest_imagenome_bbox_presence[idx] = bbox_presence_array[i]
                    phrase_grounding_masks[idx] = phrase_grounding_masks_array[i]
                    idx += 1

            print(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
            chest_imagenome_bbox_coords = chest_imagenome_bbox_coords[:idx]
            chest_imagenome_bbox_presence = chest_imagenome_bbox_presence[:idx]
            phrase_grounding_masks = phrase_grounding_masks[:idx]

            # Create dataset and dataloader for training
            if do_visual_grounding_with_bbox_regression:
                self.train_chest_imagenome_dataset = MIMICCXR_BBoxGroundingDataset(
                    image_paths=image_paths,
                    image_transform=train_image_transform,
                    phrase_embeddings=bbox_phrase_embeddings,
                    phrase_classification_labels=chest_imagenome_bbox_presence, # use bbox_presence as classification labels
                    predict_bboxes=do_visual_grounding_with_bbox_regression,
                    num_bbox_classes=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
                    feature_map_size=(mask_height, mask_width),
                    bbox_coords=chest_imagenome_bbox_coords,
                    bbox_presence=chest_imagenome_bbox_presence,
                    data_augmentation_enabled=data_augmentation_enabled,
                    for_training=True)
            else:
                self.train_chest_imagenome_dataset = MIMICCXR_BBoxGroundingDataset(
                    image_paths=image_paths,
                    image_transform=train_image_transform,
                    phrase_embeddings=bbox_phrase_embeddings,
                    phrase_classification_labels=chest_imagenome_bbox_presence, # use bbox_presence as classification labels
                    predict_bboxes=do_visual_grounding_with_bbox_regression,
                    phrase_grounding_masks=phrase_grounding_masks,
                    data_augmentation_enabled=data_augmentation_enabled,
                    for_training=True)
            batch_size = max(min(max_images_per_batch, max_phrases_per_batch // len(bbox_phrases)), 1) # at least 1 image per batch
            self.train_chest_imagenome_dataloader = DataLoader(
                self.train_chest_imagenome_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_train_workers,
                collate_fn=lambda batch: bbox_grounding_collate_batch_fn(
                    batch, training_mode=True, do_visual_grounding_with_bbox_regression=do_visual_grounding_with_bbox_regression),
                pin_memory=True,
            )
        
        # Create chest imagenome test dataset and dataloader
        if use_chest_imagenome_gold_for_test:
            print_magenta('Preparing Chest Imagenome dataset and dataloader for testing...', bold=True)
            assert mask_width is not None
            assert mask_height is not None
            assert chest_imagenome_bbox_phrase_embeddings_filepath is not None
            assert bbox_grounding_collate_batch_fn is not None
            assert num_test_workers is not None
            assert test_image_transform is not None

            _, gold_pres_indices = get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices()

            print(f'Loading bbox_phrase_embeddings and bbox_phrases from {chest_imagenome_bbox_phrase_embeddings_filepath}...')
            tmp = get_cached_pickle_file(chest_imagenome_bbox_phrase_embeddings_filepath)
            bbox_phrase_embeddings = tmp['bbox_phrase_embeddings']
            bbox_phrases = tmp['bbox_phrases']
            bbox_phrase_embeddings = bbox_phrase_embeddings[gold_pres_indices] # use only gold subset
            bbox_phrases = [bbox_phrases[i] for i in gold_pres_indices] # use only gold subset
            assert bbox_phrase_embeddings.shape[0] == len(bbox_phrases)
            print(f'bbox_phrase_embeddings.shape = {bbox_phrase_embeddings.shape}')
            print(f'len(bbox_phrases) = {len(bbox_phrases)}')
            for phrase in bbox_phrases:
                print('\t', phrase)
            self.test_chest_imagenome_gold_bbox_phrases = bbox_phrases

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            chest_imagenome_bbox_coords = [None] * BIG_ENOGUGH
            chest_imagenome_bbox_presence = [None] * BIG_ENOGUGH
            phrase_grounding_masks = [None] * BIG_ENOGUGH
            phrase_classification_labels = [None] * BIG_ENOGUGH
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()
            did2bboxes = load_chest_imagenome_gold_bboxes()

            idx = 0
            for rid, (part_id, subject_id, study_id, dicom_id_view_pairs) in \
                tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs'])), mininterval=2):
                for dicom_id, view in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id not in did2bboxes:
                        continue
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    bboxes = did2bboxes[dicom_id]
                    bbox_coords = bboxes['coords']
                    bbox_presence = bboxes['presence']
                    bbox_coords, bbox_presence = _clean_bbox_coords_and_presence(bbox_coords, bbox_presence)
                    bbox_coords = bbox_coords[gold_pres_indices] # use only gold subset
                    bbox_presence = bbox_presence[gold_pres_indices] # use only gold subset
                    chest_imagenome_bbox_coords[idx] = bbox_coords
                    chest_imagenome_bbox_presence[idx] = bbox_presence
                    phrase_grounding_masks[idx] = _compute_mask_from_bounding_boxes(mask_height, mask_width, bbox_coords, bbox_presence)
                    phrase_classification_labels[idx] = bbox_presence # use bbox_presence as classification labels, use only gold subset
                    idx += 1

            print(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
            chest_imagenome_bbox_coords = chest_imagenome_bbox_coords[:idx]
            chest_imagenome_bbox_presence = chest_imagenome_bbox_presence[:idx]
            phrase_grounding_masks = phrase_grounding_masks[:idx]

            # Sanity check
            for i in range(idx):
                assert chest_imagenome_bbox_presence[i].shape[0] == chest_imagenome_bbox_coords[i].shape[0] == CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,\
                    f'chest_imagenome_bbox_presence[{i}].shape[0] = {chest_imagenome_bbox_presence[i].shape[0]}, chest_imagenome_bbox_coords[{i}].shape[0] = {chest_imagenome_bbox_coords[i].shape[0]}'

            # Create dataset and dataloader for testing
            if do_visual_grounding_with_bbox_regression:
                self.test_chest_imagenome_gold_dataset = MIMICCXR_BBoxGroundingDataset(
                    image_paths=image_paths,
                    image_transform=test_image_transform,
                    phrase_embeddings=bbox_phrase_embeddings,
                    phrase_classification_labels=phrase_classification_labels,
                    predict_bboxes=do_visual_grounding_with_bbox_regression,
                    num_bbox_classes=CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
                    feature_map_size=(mask_height, mask_width),
                    bbox_coords=chest_imagenome_bbox_coords,
                    bbox_presence=chest_imagenome_bbox_presence,
                    data_augmentation_enabled=False,
                    for_training=False)
            else:
                self.test_chest_imagenome_gold_dataset = MIMICCXR_BBoxGroundingDataset(
                    image_paths=image_paths,
                    image_transform=test_image_transform,
                    phrase_embeddings=bbox_phrase_embeddings,
                    phrase_classification_labels=phrase_classification_labels,
                    predict_bboxes=do_visual_grounding_with_bbox_regression,
                    phrase_grounding_masks=phrase_grounding_masks,
                    data_augmentation_enabled=False,
                    for_training=False)
            
            batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // len(bbox_phrases)), 1) * test_batch_size_factor)
            self.test_chest_imagenome_gold_dataloader = DataLoader(
                self.test_chest_imagenome_gold_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_test_workers,
                collate_fn=lambda batch: bbox_grounding_collate_batch_fn(
                    batch, training_mode=False, do_visual_grounding_with_bbox_regression=do_visual_grounding_with_bbox_regression),
                pin_memory=True,
            )