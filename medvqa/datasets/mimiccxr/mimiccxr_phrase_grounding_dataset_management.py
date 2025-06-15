import math
import multiprocessing
import os
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import random
import torch
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.augmentation import VinBigAlbumentationAdapter
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAMES
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_nondecent_chest_imagenome_dicom_ids,
)
from medvqa.datasets.dataloading_utils import (
    CompositeInfiniteDataset,
    SequentialDataLoader,
    group_indices_for_balanced_sampling,
)
from medvqa.datasets.image_processing import (
    ImageFactBasedMultilabelClassificationDataset,
    ImageFactClassificationDataset,
)
from medvqa.datasets.ms_cxr import (
    MS_CXR_TrainingMode,
    get_ms_cxr_dicom_id_2_phrases_and_bboxes,
    get_ms_cxr_dicom_id_2_split,
    get_ms_cxr_test_dicom_ids,
    get_ms_cxr_val_dicom_ids,
)
from medvqa.datasets.mimiccxr import (
    MIMICCXR_DicomIdToImagePath,
    MIMICCXR_ImageSizeModes,
    MIMICCXR_ViewModes,
    get_cxrlt2024_train_dev_dicom_ids,
    get_dicom_id_and_orientation_list,
    get_mimiccxr_train_dicom_ids,
    get_split2imageIds,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.bbox_utils import (
    calculate_probabilistic_mask_from_bboxes,
    convert_bboxes_into_presence_map,
    convert_bboxes_into_target_tensors,
    xyxy_to_cxcywh,
)
from medvqa.utils.constants import LABEL_BASED_FACTS
from medvqa.utils.files_utils import get_cached_pickle_file, load_pickle, save_pickle
from medvqa.utils.logging_utils import ANSI_BOLD, ANSI_MAGENTA_BOLD, ANSI_RESET

logger = logging.getLogger(__name__)

class MIMICCXR_FactClassificationDataset(ImageFactClassificationDataset):

    def __init__(self, image_paths, image_transform, fact_embeddings, positive_facts, indices, num_facts_per_image,
                 use_strong_and_weak_negatives=False,
                 negative_facts=None, weak_negative_facts=None, strong_negative_facts=None,
                 infinite=False, shuffle=False, use_weights=False, weights_filepath=None):
        super().__init__(
            image_paths=image_paths,
            image_transform=image_transform,
            fact_embeddings=fact_embeddings,
            positive_facts=positive_facts,
            use_strong_and_weak_negatives=use_strong_and_weak_negatives,
            negative_facts=negative_facts,
            weak_negative_facts=weak_negative_facts,
            strong_negative_facts=strong_negative_facts,
            indices=indices,
            num_facts_per_image=num_facts_per_image,
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
    
class MSCXR_PhraseGroundingAndClassificationDataset(Dataset):

    def __init__(self, image_paths, image_transform, phrase_idxs, phrase_embeddings, phrase_bboxes_and_classes,
                 positive_fact_idxs, strong_neg_fact_idxs, weak_neg_fact_idxs, fact_embeddings,
                 indices, num_facts_per_image, feature_map_size, shuffle_indices=False,
                data_augmentation_enabled=False, for_training=True, bbox_format='xyxy'):
        assert num_facts_per_image >= 2
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_idxs = phrase_idxs
        self.phrase_bboxes_and_classes = phrase_bboxes_and_classes
        self.phrase_embeddings = phrase_embeddings
        self.positive_fact_idxs = positive_fact_idxs
        self.strong_neg_fact_idxs = strong_neg_fact_idxs
        self.weak_neg_fact_idxs = weak_neg_fact_idxs
        self.fact_embeddings = fact_embeddings
        self.indices = indices
        self.num_facts_per_image = num_facts_per_image
        self.num_neg_facts_per_image = num_facts_per_image // 2
        self.num_pos_facts_per_image = num_facts_per_image - self.num_neg_facts_per_image
        self.embedding_size = self.phrase_embeddings.shape[1]
        self.feature_map_size = feature_map_size
        self.data_augmentation_enabled = data_augmentation_enabled
        self.for_training = for_training
        self.bbox_format = bbox_format
        assert self.num_neg_facts_per_image > 0
        assert self.num_pos_facts_per_image > 0
        if shuffle_indices:
            random.shuffle(self.indices)
        if data_augmentation_enabled:
            # Reuse VinBigAlbumentationAdapter for MSCXR
            self.albumentation_adapter = VinBigAlbumentationAdapter(bbox_format=bbox_format)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):

        # Get the data for the i-th image
        i = self.indices[i]
        image_path = self.image_paths[i]
        phrase_idxs = self.phrase_idxs[i]
        phrase_bboxes, phrase_classes = self.phrase_bboxes_and_classes[i]
        pos_fact_idxs = self.positive_fact_idxs[i]
        strong_neg_fact_idxs = self.strong_neg_fact_idxs[i]
        weak_neg_fact_idxs = self.weak_neg_fact_idxs[i]

        # Select the facts for the i-th image
        assert len(phrase_idxs) < self.num_pos_facts_per_image
        remaining_pos_facts = self.num_pos_facts_per_image - len(phrase_idxs)
        if remaining_pos_facts < len(pos_fact_idxs):
            pos_fact_idxs = random.sample(pos_fact_idxs, remaining_pos_facts)
        remaining_pos_facts -= len(pos_fact_idxs)
        assert remaining_pos_facts >= 0
        num_neg_facts = self.num_neg_facts_per_image + remaining_pos_facts # fill the remaining with negative facts
        if random.random() < 0.5: # use strong negatives
            if num_neg_facts < len(strong_neg_fact_idxs):
                strong_neg_fact_idxs = random.sample(strong_neg_fact_idxs, num_neg_facts)
        else:
            strong_neg_fact_idxs = [] # no strong negatives
        remaining_neg_facts = num_neg_facts - len(strong_neg_fact_idxs)
        assert remaining_neg_facts >= 0
        assert remaining_neg_facts < len(weak_neg_fact_idxs)
        weak_neg_fact_idxs = random.sample(weak_neg_fact_idxs, remaining_neg_facts)
        fact_idxs = strong_neg_fact_idxs + weak_neg_fact_idxs + pos_fact_idxs

        assert len(phrase_idxs) + len(fact_idxs) == self.num_facts_per_image
        
        # Create phrase embeddings and phrase classification labels
        phrase_embeddings = np.empty((self.num_facts_per_image, self.embedding_size), dtype=self.phrase_embeddings.dtype)
        phrase_embeddings[:len(phrase_idxs)] = self.phrase_embeddings[phrase_idxs]
        phrase_embeddings[len(phrase_idxs):] = self.fact_embeddings[fact_idxs]

        phrase_classification_labels = np.zeros(self.num_facts_per_image, dtype=np.int64) # initialize with zeros
        phrase_classification_labels[:len(phrase_idxs)] = 1 # positive phrases
        phrase_classification_labels[-len(pos_fact_idxs):] = 1 # positive facts

        # Load the image and apply data augmentation if enabled
        if self.data_augmentation_enabled:
            image, phrase_bboxes, phrase_classes = self.image_transform(
                image_path=image_path,
                bboxes=phrase_bboxes,
                classes=phrase_classes,
                albumentation_adapter=self.albumentation_adapter,
            )
        else:
            image = self.image_transform(image_path)['pixel_values']

        if self.for_training:
            # Create target tensors for bounding boxes
            idxs_with_grounding = np.arange(len(phrase_idxs) + len(strong_neg_fact_idxs) + len(weak_neg_fact_idxs))
            bbox_target_coords, bbox_target_presence = convert_bboxes_into_target_tensors(
                bboxes=phrase_bboxes, classes=phrase_classes, num_classes=len(idxs_with_grounding),
                feature_map_size=self.feature_map_size, bbox_format=self.bbox_format,
            )
            return {
                'i': image,
                'pe': phrase_embeddings,
                'pcl': phrase_classification_labels,
                'btc': bbox_target_coords,
                'btp': bbox_target_presence,
                'gidxs': idxs_with_grounding, # indices with grounding, used for computing the loss only for the phrases with grounding
            }
        else:
            return {
                'i': image,
                'pe': phrase_embeddings,
                'pcl': phrase_classification_labels,
                'bboxes': phrase_bboxes,
                'classes': phrase_classes,
            }
    
    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle the variable-length
        sequences and different data types in the dataset.

        Args:
            batch: A list of dictionaries, each representing a sample from the dataset.

        Returns:
            A dictionary containing the batched data.
        """
        batch_dict = {'dataset_name': 'mscxr'}
        if self.for_training:
            batch_dict['i'] = torch.stack([x['i'] for x in batch])
            batch_dict['pe'] = torch.tensor(np.array([x['pe'] for x in batch]))
            batch_dict['pcl'] = torch.tensor(np.array([x['pcl'] for x in batch]))
            batch_dict['btc'] = torch.cat([x['btc'] for x in batch], dim=0) # shape: (num_phrases_with_grounding, H*W, 4)
            batch_dict['btp'] = torch.cat([x['btp'] for x in batch], dim=0) # shape: (num_phrases_with_grounding, H*W)
            gidxs = []
            offset = 0
            for x in batch:
                gidxs.append(x['gidxs'] + offset)
                offset += x['gidxs'].shape[0]
            batch_dict['gidxs'] = torch.tensor(np.concatenate(gidxs))
            assert batch_dict['gidxs'].shape[0] == batch_dict['btc'].shape[0]
            assert batch_dict['gidxs'].shape[0] == batch_dict['btp'].shape[0]
        else:
            batch_dict['i'] = torch.stack([x['i'] for x in batch])
            batch_dict['pe'] = torch.tensor(np.array([x['pe'] for x in batch]))
            batch_dict['pcl'] = torch.tensor(np.array([x['pcl'] for x in batch]))
            batch_dict['bboxes'] = [x['bboxes'] for x in batch]
            batch_dict['classes'] = [x['classes'] for x in batch]
        return batch_dict
        
class MSCXR_PhraseGroundingDataset(Dataset):
    """
    Dataset class for phrase grounding on the MSCXR dataset.

    This dataset handles loading images, applying transformations and
    augmentations, and preparing data for phrase grounding tasks, including
    generating target tensors for training.

    Args:
        image_paths: List of paths to the image files.
        image_transform: A callable function to transform the images.
                         This could be a simple normalization or a composition of transforms.
        phrase_idxs: List of indices linking each image to its corresponding phrase embedding.
        phrase_embeddings: A numpy array containing pre-computed phrase embeddings.
                           Shape should be (num_phrases, embedding_size).
        phrase_bboxes: A list where each element is a list of bounding boxes
                       (coordinates) for the phrases within the corresponding image.
        phrase_prob_masks: A list of numpy arrays, where each array is a probabilistic mask
                           for the phrase in the corresponding image. Used for generating
                           probabilistic target masks during training.
        indices: A list of integers representing the mapping from dataset index
                 to the index in the input lists (image_paths, etc.). This allows
                 shuffling and subsetting the dataset easily.
        feature_map_size: A tuple (height, width) representing the target
                          feature map size for generating ground truth tensors.
        shuffle_indices: If True, the dataset indices will be shuffled upon initialization.
        data_augmentation_enabled: If True, data augmentation (using Albumentations)
                                   will be applied to images and bounding boxes.
        for_training: If True, the __getitem__ method will return target tensors
                      suitable for training (e.g., target coordinates, presence,
                      and probabilistic mask). If False, it will return the image,
                      phrase embedding, and original bounding boxes for inference.
        bbox_format: A string specifying the format of the bounding box coordinates
                     (e.g., 'xyxy', 'xywh'). This is used for data augmentation
                     and target tensor generation.
    """
    def __init__(self,
                 image_paths: List[str],
                 image_transform: Callable,
                 phrase_idxs: List[int],
                 phrase_embeddings: np.ndarray,
                 phrase_bboxes: List[List[float]],
                 indices: List[int],
                 phrase_prob_masks: Optional[List[np.ndarray]] = None,
                 feature_map_size: Optional[Tuple[int, int]] = None,
                 shuffle_indices: bool = False,
                 data_augmentation_enabled: bool = False,
                 for_training: bool = True,
                 bbox_format: str = 'xyxy'):

        # Store the dataset input parameters as instance attributes.
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_idxs = phrase_idxs
        self.phrase_embeddings = phrase_embeddings
        self.phrase_bboxes = phrase_bboxes
        self.phrase_prob_masks = phrase_prob_masks
        self.indices = indices
        # Determine the dimensionality of the phrase embeddings.
        self.embedding_size = self.phrase_embeddings.shape[1]
        self.feature_map_size = feature_map_size
        self.data_augmentation_enabled = data_augmentation_enabled
        self.for_training = for_training
        self.bbox_format = bbox_format

        # Optionally shuffle the internal indices to randomize sample order.
        if shuffle_indices:
            random.shuffle(self.indices)

        # Sanity checks
        if for_training:
            assert self.feature_map_size is not None, "Feature map size must be provided for training."
            assert self.phrase_prob_masks is not None, "Probabilistic masks must be provided for training."

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        This is determined by the number of entries in the indices list.
        """
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        """
        Retrieves a single data sample from the dataset.

        Args:
            i: The index of the sample to retrieve (corresponds to an index
               in the internal self.indices list).

        Returns:
            A dictionary containing the data for the requested sample.
            The content of the dictionary depends on the 'for_training' flag.
            During training, it includes transformed image, phrase embedding,
            and target tensors (coords, presence, probabilistic mask).
            During inference, it includes transformed image, phrase embedding,
            and the original bounding boxes.
        """
        data_index = self.indices[i]
        image_path = self.image_paths[data_index]
        phrase_idx = self.phrase_idxs[data_index]
        phrase_bboxes = self.phrase_bboxes[data_index]
        phrase_embedding = self.phrase_embeddings[phrase_idx]
        phrase_embedding = torch.tensor(phrase_embedding, dtype=torch.float32)
        
        if self.for_training:
            prob_mask = self.phrase_prob_masks[data_index]
            if self.data_augmentation_enabled:
                out = self.image_transform(
                    image_path=image_path,
                    bboxes=phrase_bboxes,
                    bbox_labels=[0] * len(phrase_bboxes), # dummy labels for bounding boxes
                    masks=[prob_mask], # Albumentations expects a list of masks
                )
                image = out['pixel_values']
                prob_mask = out['masks'][0]
                phrase_bboxes = out['bboxes']
            else:
                image = self.image_transform(image_path)['pixel_values']
            # Convert the processed bounding boxes and probabilistic mask into
            # target tensors for training a model that outputs predictions
            # over a feature map.
            (
                target_coords,
                target_presence,
                target_prob_mask,
            ) = convert_bboxes_into_target_tensors(
                bboxes=phrase_bboxes,
                probabilistic_mask=prob_mask,
                feature_map_size=self.feature_map_size,
                bbox_format=self.bbox_format,
            )
            return {
                # 'i': Transformed image tensor, e.g., (C, H, W)
                'i': image,
                # 'pe': Phrase embedding, e.g., (Embedding Size,)
                'pe': phrase_embedding,
                # 'tbc': Target bounding box coordinates, e.g., (H_fm * W_fm, 4)
                'tbc': target_coords,
                # 'tbp': Target bounding box presence (binary), e.g., (H_fm * W_fm,)
                'tbp': target_presence,
                # 'tpm': Target probabilistic mask for loss, e.g., (H_fm * W_fm,)
                'tpm': target_prob_mask,
            }
        else:
            image = self.image_transform(image_path)['pixel_values']
            return {
                # 'i': Transformed image tensor, e.g., (C, H, W)
                'i': image,
                # 'pe': Phrase embedding, e.g., (Embedding Size,)
                'pe': phrase_embedding,
                # 'bboxes': Processed bounding boxes for the phrase(s) in the image, e.g., (Number of Bboxes, 4)
                'bboxes': phrase_bboxes,
            }
        
    def collate_fn(self, batch: List[dict]) -> dict:
        """
        Custom collate function for DataLoader to handle the variable-length
        sequences and different data types in the dataset.

        Args:
            batch: A list of dictionaries, each representing a sample from the dataset.

        Returns:
            A dictionary containing the batched data.
        """
        if self.for_training:
            # Collate the batch for training
            images = torch.stack([item['i'] for item in batch])
            phrase_embeddings = torch.stack([item['pe'] for item in batch])
            target_bboxes = torch.stack([item['tbc'] for item in batch])
            target_presence = torch.stack([item['tbp'] for item in batch])
            target_prob_masks = torch.stack([item['tpm'] for item in batch])
            batch = {
                'i': images,
                'pe': phrase_embeddings,
                'tbc': target_bboxes,
                'tbp': target_presence,
                'tpm': target_prob_masks,
            }
        else:
            # Collate the batch for inference
            images = torch.stack([item['i'] for item in batch])
            phrase_embeddings = torch.stack([item['pe'] for item in batch])
            bboxes = [item['bboxes'] for item in batch]
            batch = {
                'i': images,
                'pe': phrase_embeddings,
                'bboxes': bboxes,
            }
        batch['dataset_name'] = 'mscxr'
        return batch
    

class ChestImaGenome_PhraseGroundingDataset(Dataset):
    """
    Dataset for phrase grounding on the ChestImaGenome dataset.

    This class loads and processes data from a pre-compiled 'augmented_data'
    pickle file. It extracts image-phrase pairs, collects all associated
    bounding boxes (from the image and similar groundings), and generates
    target tensors for training a presence prediction model.

    Args:
        augmented_data: A list of dictionaries, where each dictionary contains
                        grounding information for a single DICOM image.
        image_transform: A callable function to transform the images.
        phrase_to_embedding_idx: A dictionary mapping each unique phrase string
                                 to its corresponding index in phrase_embeddings.
        phrase_embeddings: A numpy array of pre-computed phrase embeddings.
        feature_map_size: A tuple (H, W) for the target feature map size.
        for_training: If True, returns tensors for training (image, embedding,
                      target presence, target area). If False, returns data
                      for inference.
        data_augmentation_enabled: If True, applies data augmentation.
    """

    def __init__(
        self,
        augmented_data: List[Dict],
        image_transform: Callable,
        dicom_id_to_image_path: Callable,
        phrase_to_embedding_idx: Dict[str, int],
        phrase_embeddings: np.ndarray,
        feature_map_size: Tuple[int, int],
        for_training: bool = True,
        data_augmentation_enabled: bool = False,
    ):
        self.image_transform = image_transform
        self.dicom_id_to_image_path = dicom_id_to_image_path
        self.phrase_to_embedding_idx = phrase_to_embedding_idx
        self.phrase_embeddings = phrase_embeddings
        self.feature_map_size = feature_map_size
        self.for_training = for_training
        self.data_augmentation_enabled = data_augmentation_enabled

        # Lists to store the processed data points
        self.image_paths = []
        self.phrases = []
        self.phrase_idxs = []
        self.phrase_original_bboxes = []
        self.phrase_similar_bboxes = []
        self.phrase_presence_maps = []
        self.target_area_ratios = []

        self._parse_augmented_data(augmented_data)

    def _parse_augmented_data(self, augmented_data: List[Dict]):
        """
        Parses the raw augmented data to create clean data points for the dataset.
        """
        logger.info("Parsing ChestImaGenome augmented data...")
        
        skipped_due_to_invalid_bboxes = 0
        skipped_due_to_no_similar_groundings = 0
        
        for item in tqdm(augmented_data, desc="Parsing data", mininterval=1.0):
            dicom_id = item["dicom_id"]
            image_path = self.dicom_id_to_image_path(dicom_id)

            # Iterate through each phrase in the current image's data
            for phrase, locations in item["phrase2locations"].items():
                # Condition: Only include phrases with similar groundings
                similar_groundings = item["phrase2similar_grounding"][phrase]
                if not similar_groundings:
                    skipped_due_to_no_similar_groundings += 1
                    continue

                # Collect all bounding boxes for this phrase
                original_bboxes = [] # to store bounding boxes from the primary image locations
                similar_bboxes = [] # to store bounding boxes from similar groundings

                # 1. Get bboxes from the primary image locations
                for loc in locations:
                    original_bboxes.append(item["location2bbox"][loc])

                # 2. Validate the bounding boxes
                valid = True
                for i, box in enumerate(original_bboxes):
                    x_min, y_min, x_max, y_max = box
                    # Ensure coordinates are between 0 and 1
                    x_min = max(0, min(1, x_min))
                    y_min = max(0, min(1, y_min))
                    x_max = max(0, min(1, x_max))
                    y_max = max(0, min(1, y_max))
                    # Ensure the bounding box is valid
                    if x_min >= x_max or y_min >= y_max:
                        valid = False
                        break
                    # Update the bounding box with validated coordinates
                    original_bboxes[i] = [x_min, y_min, x_max, y_max]
                if not valid:
                    skipped_due_to_invalid_bboxes += 1
                    continue

                # 3. Get bboxes from similar groundings
                for grounding in similar_groundings:
                    for box in grounding["boxes"]:
                        assert len(box) == 4, "Bounding box must have 4 coordinates."
                        similar_bboxes.append(box)
                
                assert len(original_bboxes) > 0, (
                    f"No bounding boxes found for the phrase '{phrase}' in DICOM ID {dicom_id}."
                )
                assert len(similar_bboxes) > 0, (
                    f"No similar groundings found for the phrase '{phrase}' in DICOM ID {dicom_id}."
                )
 
                # 4. Convert bounding boxes to presence maps
                all_bboxes = original_bboxes + similar_bboxes
                presence_map = convert_bboxes_into_presence_map(
                    bboxes=all_bboxes,
                    feature_map_size=self.feature_map_size,
                )
                self.phrase_presence_maps.append(presence_map)

                # 5. Precompute the ratio of the target area and the presence map area
                target_area = item["phrase2target_area"][phrase]
                presence_map_area = presence_map.sum().item() / (
                    self.feature_map_size[0] * self.feature_map_size[1]
                )
                assert 0 < presence_map_area <= 1, (
                    f"Presence map area for phrase '{phrase}' in DICOM ID {dicom_id} "
                    f"is out of bounds: {presence_map_area:.4f}"
                )
                target_area_ratio = target_area / presence_map_area
                assert 0 < target_area_ratio <= 1, (
                    f"Target area ratio for phrase '{phrase}' in DICOM ID {dicom_id} "
                    f"is out of bounds: {target_area_ratio:.4f}"
                )

                # 6. Append the processed data to the lists
                self.image_paths.append(image_path)
                self.phrases.append(phrase)
                self.phrase_original_bboxes.append(original_bboxes)
                self.phrase_similar_bboxes.append(similar_bboxes)
                self.target_area_ratios.append(target_area_ratio)
                self.phrase_idxs.append(
                    self.phrase_to_embedding_idx[phrase]
                )
        if skipped_due_to_invalid_bboxes > 0:
            logger.warning(
                f"Skipped {skipped_due_to_invalid_bboxes} samples due to invalid bounding boxes."
            )
        if skipped_due_to_no_similar_groundings > 0:
            logger.warning(
                f"Skipped {skipped_due_to_no_similar_groundings} samples due to no similar groundings."
            )
        logger.info(f"Finished parsing. Found {len(self.image_paths)} valid samples.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> dict:
        image_path = self.image_paths[i]
        phrase_idx = self.phrase_idxs[i]
        presence_map = self.phrase_presence_maps[i]
        target_area_ratio = self.target_area_ratios[i]

        phrase_embedding = self.phrase_embeddings[phrase_idx]
        phrase_embedding = torch.tensor(phrase_embedding, dtype=torch.float32)

        if self.for_training:
            if self.data_augmentation_enabled:
                # Apply augmentations to image and presence map
                out = self.image_transform(
                    image_path=image_path,
                    masks=[presence_map],  # Albumentations expects a list of masks
                )
                image = out["pixel_values"]
                augmented_presence_map = out["masks"][0]
            else:
                image = self.image_transform(image_path)["pixel_values"]
                augmented_presence_map = presence_map

            # Convert the presence map into target tensors
            target_presence = torch.from_numpy(augmented_presence_map)
        else:
            # For inference, we just transform the image and keep the original presence map
            image = self.image_transform(image_path)["pixel_values"]
            # Convert the presence map into a tensor
            target_presence = torch.from_numpy(presence_map)

        return {
            "i": image,  # Transformed image tensor
            "pe": phrase_embedding,  # Phrase embedding tensor
            "tbp": target_presence.view(-1),  # Target presence map (flattened)
            "tar": target_area_ratio,  # Target area ratio
        }
    
    def collate_fn(self, batch: List[dict]) -> dict:
        """Custom collate function for the DataLoader."""
        images = torch.stack([item["i"] for item in batch])
        phrase_embeddings = torch.stack([item["pe"] for item in batch])
        target_presence = torch.stack([item["tbp"] for item in batch])
        target_area_ratios = torch.tensor(
            [item["tar"] for item in batch], dtype=torch.float32
        )
        return {
            "i": images,
            "pe": phrase_embeddings,
            "tbp": target_presence,
            "tar": target_area_ratios,
            "dataset_name": "chest-imagenome-pg",
        }


# Global variable for worker processes to avoid passing large data repeatedly
_ALG_WORKER_DATA = {}

def _init_alg_worker(shared_data: Dict):
    """
    Initializer for the multiprocessing pool.
    Sets a global variable in each worker process with shared read-only data.
    """
    global _ALG_WORKER_DATA
    _ALG_WORKER_DATA = shared_data

def _process_chunk_worker(
    data_chunk: List[Dict],
) -> Tuple[List, List, List, int]:
    """
    Worker function that processes a chunk of the augmented_data.
    It validates bboxes, creates probabilistic masks, and returns the results.
    """
    # Unpack shared data from the global variable
    dicom_id_to_image_path = _ALG_WORKER_DATA["dicom_id_to_image_path"]
    location_to_idx = _ALG_WORKER_DATA["location_to_idx"]
    bbox_format = _ALG_WORKER_DATA["bbox_format"]
    feature_map_size = _ALG_WORKER_DATA["feature_map_size"]

    # Local lists to store results for this chunk
    chunk_image_paths = []
    chunk_grounded_locs = []
    chunk_prob_masks = []
    skipped_bboxes_count = 0

    for item in data_chunk:
        image_path = dicom_id_to_image_path(item["dicom_id"])
        grounded_locs = {}

        for loc_name, loc_idx in location_to_idx.items():
            if loc_name in item["location2bbox"]:
                bbox = item["location2bbox"][loc_name]  # Original format: xyxy
                x_min, y_min, x_max, y_max = bbox
                x_min, y_min = max(0, min(1, x_min)), max(0, min(1, y_min))
                x_max, y_max = max(0, min(1, x_max)), max(0, min(1, y_max))
                if x_min >= x_max or y_min >= y_max:
                    skipped_bboxes_count += 1
                    continue
                bbox = [x_min, y_min, x_max, y_max]
                if bbox_format == "cxcywh":
                    bbox = xyxy_to_cxcywh(bbox)
                grounded_locs[loc_idx] = bbox

        if grounded_locs:
            chunk_image_paths.append(image_path)
            chunk_grounded_locs.append(grounded_locs)
            prob_masks_dict = {}
            for loc_idx, bbox in grounded_locs.items():
                mask = calculate_probabilistic_mask_from_bboxes(
                    bboxes=[bbox],
                    mask_resolution=feature_map_size,
                    bbox_format=bbox_format,
                )
                prob_masks_dict[loc_idx] = mask
            chunk_prob_masks.append(prob_masks_dict)

    return (
        chunk_image_paths,
        chunk_grounded_locs,
        chunk_prob_masks,
        skipped_bboxes_count,
    )

class ChestImaGenome_AnatomicalLocationGroundingDataset(Dataset):
    """
    Dataset for grounding a fixed set of anatomical locations on ChestImaGenome.

    For each image and for each of its grounded anatomical locations, this dataset
    pre-computes a specific probabilistic mask from the location's bounding box.
    During training, the image, all its ground-truth bounding boxes, and all their
    corresponding masks are augmented together.

    Args:
        augmented_data: List of dictionaries containing grounding info.
        image_transform: A callable function to transform images, bboxes, and masks.
        dicom_id_to_image_path: A function mapping a DICOM ID to its file path.
        bbox_phrase_embeddings_data: A dictionary loaded from the output of the
                                     precompute_bbox_phrase_embeddings.py script.
        feature_map_size: A tuple (H, W) for the target feature map size.
        for_training: If True, returns tensors for training.
        data_augmentation_enabled: If True, applies data augmentation.
        bbox_format: The format of bounding boxes (e.g., 'xyxy').
        num_processes: Number of parallel processes for data parsing.
    """

    def __init__(
        self,
        augmented_data: List[Dict],
        image_transform: Callable,
        dicom_id_to_image_path: Callable,
        bbox_phrase_embeddings_data: Dict,
        feature_map_size: Tuple[int, int],
        for_training: bool = True,
        data_augmentation_enabled: bool = False,
        bbox_format: str = "xyxy",
        num_processes: int = 1,
    ):
        self.image_transform = image_transform
        self.dicom_id_to_image_path = dicom_id_to_image_path
        self.bbox_phrase_embeddings_data = bbox_phrase_embeddings_data
        self.feature_map_size = feature_map_size
        self.for_training = for_training
        self.data_augmentation_enabled = data_augmentation_enabled
        self.bbox_format = bbox_format
        self.num_processes = num_processes

        assert bbox_format in ["xyxy", "cxcywh"], (
            f"Unsupported bbox format: {bbox_format}. "
            "Supported formats are 'xyxy' and 'cxcywh'."
        )

        self.anatomical_locations = self.bbox_phrase_embeddings_data[
            "bbox_phrases"
        ]
        assert self.anatomical_locations == CHEST_IMAGENOME_BBOX_NAMES
        self.location_to_idx = {
            name: i for i, name in enumerate(self.anatomical_locations)
        }
        self.num_locations = len(self.anatomical_locations)

        self.image_paths = []
        self.grounded_locations_per_image = []
        self.prob_masks_per_image = []

        self._parse_augmented_data(augmented_data)

    def _parse_augmented_data(self, augmented_data: List[Dict]):
        """
        Parses raw data in parallel, pre-computing a probabilistic mask for each
        grounded location's bounding box.
        """
        logger.info(
            f"Parsing ChestImaGenome data with {self.num_processes} processes (augmented_data size: {len(augmented_data)})"
        )

        # if len(augmented_data) > 10000:
        #     augmented_data = augmented_data[:10000]  # Limit to first 10k samples for faster debugging
        #     logger.warning(
        #         "Limiting augmented_data to first 10,000 samples for faster debugging. TODO: Remove this limit in production."
        #     )

        shared_data_payload = {
            "dicom_id_to_image_path": self.dicom_id_to_image_path,
            "location_to_idx": self.location_to_idx,
            "bbox_format": self.bbox_format,
            "feature_map_size": self.feature_map_size,
        }

        total_skipped_bboxes = 0

        if self.num_processes == 1:
            # Run in a single process (useful for debugging)
            _init_alg_worker(shared_data_payload)
            (
                self.image_paths,
                self.grounded_locations_per_image,
                self.prob_masks_per_image,
                total_skipped_bboxes,
            ) = _process_chunk_worker(augmented_data)
        else:
            # Split data into chunks for parallel processing
            chunk_size = math.ceil(len(augmented_data) / self.num_processes)
            chunks = [
                augmented_data[i : i + chunk_size]
                for i in range(0, len(augmented_data), chunk_size)
            ]

            with multiprocessing.Pool(
                processes=self.num_processes,
                initializer=_init_alg_worker,
                initargs=(shared_data_payload,),
            ) as pool:
                with tqdm(
                    total=len(chunks), desc="Parsing data chunks"
                ) as pbar:
                    for (
                        img_paths,
                        grounded_locs,
                        prob_masks,
                        skipped_count,
                    ) in pool.imap_unordered(_process_chunk_worker, chunks):
                        self.image_paths.extend(img_paths)
                        self.grounded_locations_per_image.extend(grounded_locs)
                        self.prob_masks_per_image.extend(prob_masks)
                        total_skipped_bboxes += skipped_count
                        pbar.update(1)

        if total_skipped_bboxes > 0:
            logger.warning(
                f"Skipped {total_skipped_bboxes} bboxes due to invalid coordinates."
            )
        logger.info(
            f"Finished parsing. Found {len(self.image_paths)} valid samples."
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> dict:
        image_path = self.image_paths[i]
        grounded_locs = self.grounded_locations_per_image[i]
        prob_masks_dict = self.prob_masks_per_image[i]

        # 1. Prepare phrase embeddings for all 36 locations
        embeddings = []
        for loc_idx in range(self.num_locations):
            if random.random() < 0.5:
                similar_phrases = self.bbox_phrase_embeddings_data[
                    "most_similar_anatomical_locations"
                ][loc_idx]
                similar_embeddings = self.bbox_phrase_embeddings_data[
                    "most_similar_anatomical_location_embeddings"
                ][loc_idx]
                if similar_phrases:
                    choice_idx = random.randint(0, len(similar_phrases) - 1)
                    chosen_embedding = similar_embeddings[choice_idx]
                else:
                    chosen_embedding = self.bbox_phrase_embeddings_data[
                        "bbox_phrase_embeddings"
                    ][loc_idx]
            else:
                chosen_embedding = self.bbox_phrase_embeddings_data[
                    "bbox_phrase_embeddings"
                ][loc_idx]
            embeddings.append(chosen_embedding)
        phrase_embeddings = torch.tensor(
            np.array(embeddings), dtype=torch.float32
        )

        # 2. Prepare ground truth bboxes, masks, and their class indices
        gt_indices = sorted(grounded_locs.keys())
        gt_bboxes = [grounded_locs[idx] for idx in gt_indices]
        masks_to_augment = [prob_masks_dict[idx] for idx in gt_indices]

        if self.for_training:
            # 3. Apply data augmentation
            if self.data_augmentation_enabled:
                out = self.image_transform(
                    image_path=image_path,
                    bboxes=gt_bboxes,
                    bbox_labels=gt_indices,
                    masks=masks_to_augment,
                )
                image = out["pixel_values"]
                augmented_bboxes = out["bboxes"]
                augmented_bbox_labels = out["bbox_labels"]
                augmented_masks = out["masks"]
                assert len(augmented_masks) == len(masks_to_augment) # Ensure no masks were dropped
                if len(augmented_bbox_labels) < len(gt_indices): # Some bboxes were dropped, should happen very rarely
                    # Filter masks whose bounding boxes were dropped during augmentation
                    valid_indices = [
                        i for i, idx in enumerate(gt_indices)
                        if idx in augmented_bbox_labels
                    ]
                    augmented_masks = [augmented_masks[i] for i in valid_indices]
                    # Update gt_indices
                    gt_indices = augmented_bbox_labels
            else:
                image = self.image_transform(image_path)["pixel_values"]
                augmented_bboxes = gt_bboxes
                augmented_masks = masks_to_augment

            # 4. Generate target tensors for each grounded location
            t_coords, t_presence, t_prob_masks = [], [], []
            for j, bbox in enumerate(augmented_bboxes):
                # Pair each augmented bbox with its corresponding augmented mask
                augmented_mask = augmented_masks[j]
                coords, presence, p_mask = convert_bboxes_into_target_tensors(
                    bboxes=[bbox],
                    probabilistic_mask=augmented_mask,
                    feature_map_size=self.feature_map_size,
                    bbox_format=self.bbox_format,
                )
                t_coords.append(coords)
                t_presence.append(presence)
                t_prob_masks.append(p_mask)

            return {
                "i": image,
                "pe": phrase_embeddings,
                "tbc": torch.stack(t_coords), # Target bounding box coordinates
                "tbp": torch.stack(t_presence), # Target bounding box presence (binary)
                "tpm": torch.stack(t_prob_masks), # Target probabilistic masks for loss
                "gidxs": torch.tensor(gt_indices, dtype=torch.long), # Ground truth indices
            }
        else:  # For inference
            image = self.image_transform(image_path)["pixel_values"]
            return {
                "i": image,
                "pe": phrase_embeddings,
                "bboxes": gt_bboxes,
                "classes": gt_indices,
            }

    def collate_fn(self, batch: List[dict]) -> dict:
        """Custom collate function to handle batching."""
        batch_dict = {"dataset_name": "chest-imagenome-alg"} # alg stands for anatomical location grounding

        if self.for_training:
            batch_dict["i"] = torch.stack([x["i"] for x in batch]) # (B, C, H, W)
            batch_dict["pe"] = torch.stack([x["pe"] for x in batch]) # (B, num_locations, embedding_size)
            batch_dict["tbc"] = torch.cat([x["tbc"] for x in batch], dim=0) # (num_grounded_locations, H*W, 4)
            batch_dict["tbp"] = torch.cat([x["tbp"] for x in batch], dim=0) # (num_grounded_locations, H*W)
            batch_dict["tpm"] = torch.cat([x["tpm"] for x in batch], dim=0) # (num_grounded_locations, H*W)

            gidxs_list = []
            for i, x in enumerate(batch):
                offset = i * self.num_locations
                gidxs_list.append(x["gidxs"] + offset)
            batch_dict["gidxs"] = torch.cat(gidxs_list) # (num_grounded_locations,)

            assert batch_dict["gidxs"].shape[0] == batch_dict["tbc"].shape[0]
            assert batch_dict["gidxs"].shape[0] == batch_dict["tbp"].shape[0]
        else:
            batch_dict["i"] = torch.stack([x["i"] for x in batch]) # (B, C, H, W)
            batch_dict["pe"] = torch.stack([x["pe"] for x in batch]) # (B, num_locations, embedding_size)
            batch_dict["bboxes"] = [x["bboxes"] for x in batch] # List of lists of bboxes
            batch_dict["classes"] = [x["classes"] for x in batch] # List of lists of class indices

        return batch_dict

    
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
    

_LONG_TAIL = 0
_MIDDLE_TAIL = 1
_SHORT_TAIL = 2

def _assign_distribution_classes_to_reports(report_fact_nli_integrated_data_filepath,
                                            distribution_thresholds=(0.02, 0.05)):
    assert len(distribution_thresholds) == 2
    assert 0 < distribution_thresholds[0] < distribution_thresholds[1] < 1
    logger.info('Assigning distribution classes to reports...')
    # Load the data
    logger.info(f'Loading mimiccxr_report_fact_nli_integrated_data from {report_fact_nli_integrated_data_filepath}...')
    data = load_pickle(report_fact_nli_integrated_data_filepath)
    labels = data['label_based_nli_predictions']['nli_predictions']
    logger.info(f'labels.shape = {labels.shape}')
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

    logger.info(f'count_no_positives = {count_no_positives}')
    logger.info(f'number of long tail classes = {(is_long_tail).sum()}')
    logger.info(f'number of middle tail classes = {(is_middle_tail).sum()}')
    logger.info(f'number of short tail classes = {(is_short_tail).sum()}')
    logger.info(f'number of long tail reports = {(distribution_classes == _LONG_TAIL).sum()}')
    logger.info(f'number of middle tail reports = {(distribution_classes == _MIDDLE_TAIL).sum()}')
    logger.info(f'number of short tail reports = {(distribution_classes == _SHORT_TAIL).sum()}')

    return distribution_classes

class MIMICCXR_PhraseGroundingTrainer:

    def __init__(self, 
                max_images_per_batch, max_phrases_per_batch, max_phrases_per_image,
                num_train_workers=None, num_test_workers=None,
                train_image_transform=None, test_image_transform=None,
                fact_grounding_collate_batch_fn=None,
                cxrlt2024_image_phrase_classifier_collate_batch_fn=None,
                cxrlt2024_multilabel_classifier_collate_batch_fn=None,
                test_batch_size_factor=1,
                mask_width=None, mask_height=None,
                use_facts_for_train=False,
                use_facts_for_test=False,
                dicom_id_to_pos_neg_facts_filepath=None,
                use_mscxr_for_train=False,
                use_mscxr_for_val=False,
                use_mscxr_for_test=False,
                mscxr_do_grounding_only=False,
                mscxr_test_on_all_images=False, # test on all images
                mscxr_training_data_mode=MS_CXR_TrainingMode.TRAIN.value,
                mscxr_phrase2embedding_filepath=None,
                use_chest_imagenome_for_train=False,
                use_chest_imagenome_for_val=False,
                use_chest_imagenome_for_test=False,
                # use_chest_imagenome_for_test=True, # TODO: set to False after done with debugging
                # use_chest_imagenome_gold_for_test=False,
                chest_imagenome_augmented_phrase_groundings_filepath=None,
                chest_imagenome_phrase_embeddings_filepath=None,
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
                data_augmentation_enabled=False,
                replace_phrase_embeddings_with_random_vectors=False,
                bbox_format='xyxy',
                num_processes=10,
                **unused_kwargs,
            ):

        if len(unused_kwargs) > 0:
            logger.warning(f'unused kwargs in MIMICCXR_PhraseGroundingTrainer: {unused_kwargs}')
        # Sanity checks
        assert sum([use_facts_for_train,
                    use_chest_imagenome_for_train,
                    use_chest_imagenome_for_val,
                    use_chest_imagenome_for_test,
                    use_mscxr_for_train,
                    use_mscxr_for_test,
                    # use_chest_imagenome_gold_for_test,
                    use_cxrlt2024_challenge_split]) > 0 # at least one of them must be True
        
        self.use_facts_for_train = use_facts_for_train
        self.use_chest_imagenome_for_train = use_chest_imagenome_for_train
        self.use_mscxr_for_train = use_mscxr_for_train
        self.use_mscxr_for_test = use_mscxr_for_test
        # self.use_chest_imagenome_gold_for_test = use_chest_imagenome_gold_for_test
        self.use_cxrlt2024_challenge_split = use_cxrlt2024_challenge_split
        self.use_cxrlt2024_custom_labels = use_cxrlt2024_custom_labels
        self.use_cxrlt2024_official_labels = use_cxrlt2024_official_labels
        self.use_all_cxrlt2024_official_labels_for_training = use_all_cxrlt2024_official_labels_for_training
        self.replace_phrase_embeddings_with_random_vectors = replace_phrase_embeddings_with_random_vectors

        forbidden_train_dicom_ids = set()

        if exclude_noisy_images:
            noisy_dicom_ids = set(load_nondecent_chest_imagenome_dicom_ids())
            forbidden_train_dicom_ids |= noisy_dicom_ids

        if use_cxrlt2024_challenge_split:
            assert use_cxrlt2024_custom_labels or use_cxrlt2024_official_labels
            assert not use_mscxr_for_test # only for training
            # assert not use_chest_imagenome_gold_for_test # only for training
            assert not use_facts_for_test # only for training
            assert not use_interpret_cxr_challenge_split
            cxrlt2024_train_dicom_ids, cxrlt2024_dev_dicom_ids = get_cxrlt2024_train_dev_dicom_ids()
            cxrlt2024_train_dicom_ids = set(cxrlt2024_train_dicom_ids)
            cxrlt2024_dev_dicom_ids = set(cxrlt2024_dev_dicom_ids)
            logger.info(f'len(cxrlt2024_train_dicom_ids) = {len(cxrlt2024_train_dicom_ids)}')
            logger.info(f'len(cxrlt2024_dev_dicom_ids) = {len(cxrlt2024_dev_dicom_ids)}')
            forbidden_train_dicom_ids |= cxrlt2024_dev_dicom_ids # exclude dev set

        if use_cxrlt2024_official_labels:
            assert use_cxrlt2024_challenge_split
            assert cxrlt2024_official_training_labels_for_fact_classification_filepath is not None
            assert cxrlt2024_multilabel_classifier_collate_batch_fn is not None

        if use_cxrlt2024_custom_labels:
            # assert use_cxrlt2024_challenge_split
            assert cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath is not None
            assert cxrlt2024_image_phrase_classifier_collate_batch_fn is not None

        if use_mscxr_for_val or use_mscxr_for_test:
            forbidden_train_dicom_ids |= set(get_ms_cxr_val_dicom_ids())
            forbidden_train_dicom_ids |= set(get_ms_cxr_test_dicom_ids())

        # if use_chest_imagenome_gold_for_test:
        #     gold_dicom_ids = set(load_gold_bbox_dicom_ids())
        #     forbidden_train_dicom_ids |= gold_dicom_ids

        logger.info(f'len(forbidden_train_dicom_ids) = {len(forbidden_train_dicom_ids)}')

        dicom_id_to_image_path = MIMICCXR_DicomIdToImagePath(image_size_mode=source_image_size_mode, verbose=True)

        # Create train and dev datasets for CXR-LT-2024 challenge
        if use_cxrlt2024_challenge_split or self.use_cxrlt2024_custom_labels:
            logger.info(f'{ANSI_MAGENTA_BOLD}Preparing CXR-LT-2024 challenge datasets and dataloaders for training/testing...', bold=True)

            if self.use_cxrlt2024_custom_labels:

                assert cxrlt2024_image_phrase_classifier_collate_batch_fn is not None
                assert cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath is not None

                cxrlt2024_dicom_id_to_pos_neg_facts = load_pickle(cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath)
                logger.info(f'len(cxrlt2024_dicom_id_to_pos_neg_facts["train"]) = {len(cxrlt2024_dicom_id_to_pos_neg_facts["train"])}')
                logger.info(f'len(cxrlt2024_dicom_id_to_pos_neg_facts["dev"]) = {len(cxrlt2024_dicom_id_to_pos_neg_facts["dev"])}')

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
                        image_paths[idx] = dicom_id_to_image_path(dicom_id)
                        phrase_idxs[idx] = neg_idxs + pos_idxs
                        phrase_classification_labels[idx] = np.concatenate([np.zeros(len(neg_idxs), dtype=np.int64),
                                                                            np.ones(len(pos_idxs), dtype=np.int64)])
                        train_indices.append(idx)
                        idx += 1

                    for dicom_id, (neg_idxs, pos_idxs) in cxrlt2024_dicom_id_to_pos_neg_facts['dev'].items():
                        image_paths[idx] = dicom_id_to_image_path(dicom_id)
                        phrase_idxs[idx] = neg_idxs + pos_idxs
                        phrase_classification_labels[idx] = np.concatenate([np.zeros(len(neg_idxs), dtype=np.int64),
                                                                            np.ones(len(pos_idxs), dtype=np.int64)])
                        dev_indices.append(idx)
                        idx += 1

                else:

                    # Use MIMIC-CXR's original train/dev split by default
                    logger.info('Using MIMIC-CXR\'s original train/dev split for CXR-LT-2024 challenge', bold=True)
                    train_dicom_ids = set(get_mimiccxr_train_dicom_ids())
                    for key in ['train', 'dev']:
                        for dicom_id, (neg_idxs, pos_idxs) in cxrlt2024_dicom_id_to_pos_neg_facts[key].items():
                            if dicom_id in train_dicom_ids:
                                if dicom_id in forbidden_train_dicom_ids:
                                    continue
                                image_paths[idx] = dicom_id_to_image_path(dicom_id)
                                phrase_idxs[idx] = neg_idxs + pos_idxs
                                phrase_classification_labels[idx] = np.concatenate([np.zeros(len(neg_idxs), dtype=np.int64),
                                                                                    np.ones(len(pos_idxs), dtype=np.int64)])
                                train_indices.append(idx)
                                idx += 1
                            else:
                                image_paths[idx] = dicom_id_to_image_path(dicom_id)
                                phrase_idxs[idx] = neg_idxs + pos_idxs
                                phrase_classification_labels[idx] = np.concatenate([np.zeros(len(neg_idxs), dtype=np.int64),
                                                                                    np.ones(len(pos_idxs), dtype=np.int64)])
                                dev_indices.append(idx)
                                idx += 1

                logger.info(f'Total number of images: {idx}')
                image_paths = image_paths[:idx]
                phrase_idxs = phrase_idxs[:idx]
                phrase_classification_labels = phrase_classification_labels[:idx]

                # Create dataset and dataloader for training
                logger.info(f'{ANSI_BOLD}Building cxrlt2024 train phrase classifier dataloader...{ANSI_RESET}')
                num_phrases_2_idxs = {}
                for idx in train_indices:
                    num_phrases = len(phrase_idxs[idx])
                    try:
                        num_phrases_2_idxs[num_phrases].append(idx)
                    except KeyError:
                        num_phrases_2_idxs[num_phrases] = [idx]
                train_dataloaders = []
                for num_phrases, indices in num_phrases_2_idxs.items():
                    logger.info(f'num_phrases = {num_phrases}, len(indices) = {len(indices)}')
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
                        persistent_workers=True, # for faster training
                    )
                    train_dataloaders.append(dataloader)
                self.cxrlt2024_custom_train_dataloader = SequentialDataLoader(train_dataloaders)

                # Create dataset and dataloader for dev
                logger.info(f'{ANSI_BOLD}Building cxrlt2024 dev phrase classifier dataloader...{ANSI_RESET}')
                num_phrases_2_idxs = {}
                for idx in dev_indices:
                    num_phrases = len(phrase_idxs[idx])
                    try:
                        num_phrases_2_idxs[num_phrases].append(idx)
                    except KeyError:
                        num_phrases_2_idxs[num_phrases] = [idx]
                dev_dataloaders = []
                for num_phrases, indices in num_phrases_2_idxs.items():
                    logger.info(f'num_phrases = {num_phrases}, len(indices) = {len(indices)}')
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
                        persistent_workers=True, # for faster evaluation
                    )
                    dev_dataloaders.append(dataloader)
                self.cxrlt2024_custom_dev_dataloader = SequentialDataLoader(dev_dataloaders)

            if self.use_cxrlt2024_official_labels:
                # We will create an additional dataloader for training the fact classifier
                # using the official training labels provided by the challenge organizers
                logger.info(f'{ANSI_BOLD}Building cxrlt2024 train/val dataloader for fact classifier using official labels...{ANSI_RESET}')
                data = load_pickle(cxrlt2024_official_training_labels_for_fact_classification_filepath)
                dicom_ids = data['dicom_ids']
                labels = data['labels']
                train_indices = data['train_indices']
                val_indices = data['val_indices']
                class_names = data['class_names']
                class_embeddings = data['class_embeddings']
                assert labels.shape == (len(dicom_ids), len(class_embeddings))
                assert len(train_indices) + len(val_indices) == len(dicom_ids)
                logger.info(f'len(dicom_ids) = {len(dicom_ids)}')
                logger.info(f'len(train_indices) = {len(train_indices)}')
                logger.info(f'len(val_indices) = {len(val_indices)}')
                logger.info(f'labels.shape = {labels.shape}')
                logger.info(f'class_embeddings.shape = {class_embeddings.shape}')
                if use_all_cxrlt2024_official_labels_for_training:
                    logger.info('Using all CXR-LT-2024 official labels for training the fact classifier', bold=True)
                    train_indices += val_indices
                    val_indices = []
                    logger.info(f'len(train_indices) = {len(train_indices)}')
                    logger.info(f'len(val_indices) = {len(val_indices)}')
                image_paths = []
                for dicom_id in dicom_ids:
                    image_paths.append(dicom_id_to_image_path(dicom_id))

                # Clean train_indices
                len_before = len(train_indices)
                train_indices = [i for i in train_indices if dicom_ids[i] not in forbidden_train_dicom_ids]
                len_after = len(train_indices)
                if len_before != len_after:
                    logger.info(f'{len_before - len_after} train indices removed due to forbidden_train_dicom_ids')

                if cxrlt2024_do_balanced_sampling:
                    logger.info('Balanced sampling is enabled for CXR-LT-2024 fact classifier training')
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
                        logger.info(f'  len(indices) = {len(indices)}, weight = {weight}')
                    self.cxrlt2024_official_train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
                    batch_size = max(min(max_images_per_batch, max_phrases_per_batch // len(class_names)), 1) # at least 1 image per batch
                    self.cxrlt2024_official_train_dataloader = DataLoader(self.cxrlt2024_official_train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_train_workers,
                                                    collate_fn=cxrlt2024_multilabel_classifier_collate_batch_fn,
                                                    pin_memory=True,
                                                    persistent_workers=True)
                else:
                    logger.info('Normal sampling is enabled for CXR-LT-2024 fact classifier training')
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
                                                    pin_memory=True,
                                                    persistent_workers=True)
                
                # Create val dataset
                if len(val_indices) > 0:
                    logger.info('Creating CXR-LT-2024 official val dataset and dataloader...')
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
                                                    pin_memory=True,
                                                    persistent_workers=True)

        # Create train mimiccxr facts dataset
        if use_facts_for_train or use_facts_for_test:
            logger.info(f'{ANSI_MAGENTA_BOLD}Preparing MIMIC-CXR-Facts datasets and dataloaders for training/testing...', bold=True)
            assert dicom_id_to_pos_neg_facts_filepath is not None
            assert fact_grounding_collate_batch_fn is not None
            assert num_train_workers is not None
            assert train_image_transform is not None

            tmp = get_cached_pickle_file(dicom_id_to_pos_neg_facts_filepath)
            fact_embeddings = tmp['embeddings']

            if replace_phrase_embeddings_with_random_vectors:
                logger.info('NOTE: Replacing fact_embeddings with random vectors', bold=True)
                save_path = f'{dicom_id_to_pos_neg_facts_filepath}.random_vectors.pkl'
                if os.path.exists(save_path):
                    logger.info(f'Random vectors already saved at {save_path}')
                    fact_embeddings = load_pickle(save_path)['fact_embeddings']
                else:
                    fact_embeddings = np.random.randn(*fact_embeddings.shape).astype(fact_embeddings.dtype) # replace with random vectors
                    fact_embeddings /= np.linalg.norm(fact_embeddings, axis=1, keepdims=True) # normalize
                    save_dict = {'fact_embeddings': fact_embeddings}
                    save_pickle(save_dict, save_path)
                    logger.info(f'Saved random vectors at {save_path}')

            try:
                # Backward compatibility
                dicom_id_to_pos_neg_facts = tmp['dicom_id_to_pos_neg_facts']
                dicom_ids_with_facts = set(dicom_id_to_pos_neg_facts.keys())
                use_strong_and_weak_negatives = False
            except KeyError:
                dicom_id_to_pos_facts = tmp['dicom_id_to_pos_facts']
                dicom_id_to_strong_neg_facts = tmp['dicom_id_to_strong_neg_facts']
                dicom_id_to_weak_neg_facts = tmp['dicom_id_to_weak_neg_facts']
                dicom_ids_with_facts = set(dicom_id_to_pos_facts.keys())
                use_strong_and_weak_negatives = True
                logger.info('NOTE: Using strong and weak negatives for training...', bold=True)

            logger.info(f'fact_embeddings.shape = {fact_embeddings.shape}')

            if balance_long_middle_short_tail and use_facts_for_train: # only for training
                assert report_fact_nli_integrated_data_filepath is not None
                distribution_classes = _assign_distribution_classes_to_reports(
                    report_fact_nli_integrated_data_filepath=report_fact_nli_integrated_data_filepath,
                    distribution_thresholds=long_middle_short_tail_thresholds)

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            positive_facts = [None] * BIG_ENOGUGH
            if use_strong_and_weak_negatives:
                weak_negative_facts = [None] * BIG_ENOGUGH
                strong_negative_facts = [None] * BIG_ENOGUGH
            else:
                negative_facts = [None] * BIG_ENOGUGH
            report_idxs = [None] * BIG_ENOGUGH

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()

            if use_facts_for_train:
                train_indices = []
            if use_facts_for_test:
                test_indices = []

            if use_interpret_cxr_challenge_split:
                assert interpret_cxr_challenge_split_filepath is not None
                logger.info(f'Using split from {interpret_cxr_challenge_split_filepath}')
                challenge_split = load_pickle(interpret_cxr_challenge_split_filepath)
                challenge_train_dicom_ids = set(challenge_split['train'])
                challenge_val_dicom_ids = set(challenge_split['val'])
                # ignore test set because it is kept hidden in the challenge

            idx = 0
            for ridx, (dicom_id_view_pairs, split) in tqdm(enumerate(zip(
                    mimiccxr_metadata['dicom_id_view_pos_pairs'],
                    mimiccxr_metadata['splits'])), mininterval=2):
                for dicom_id, _ in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id not in dicom_ids_with_facts:
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
                            continue
                    elif split == 'train':
                        if use_facts_for_train:
                            if dicom_id in forbidden_train_dicom_ids:
                                continue
                            train_indices.append(idx)
                        else:
                            continue
                    else:
                        raise ValueError(f'Invalid split: {split}')
                    image_paths[idx] = dicom_id_to_image_path(dicom_id)
                    if use_strong_and_weak_negatives:
                        pos_facts = dicom_id_to_pos_facts[dicom_id]
                        strong_neg_facts = dicom_id_to_strong_neg_facts[dicom_id]
                        weak_neg_facts = dicom_id_to_weak_neg_facts[dicom_id]
                        positive_facts[idx] = pos_facts
                        strong_negative_facts[idx] = strong_neg_facts
                        weak_negative_facts[idx] = weak_neg_facts
                    else:
                        pos_neg_facts = dicom_id_to_pos_neg_facts[dicom_id]
                        assert len(pos_neg_facts) == 2
                        positive_facts[idx] = pos_neg_facts[0]
                        negative_facts[idx] = pos_neg_facts[1]
                        report_idxs[idx] = ridx
                    idx += 1

            logger.info(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
            positive_facts = positive_facts[:idx]
            if use_strong_and_weak_negatives:
                strong_negative_facts = strong_negative_facts[:idx]
                weak_negative_facts = weak_negative_facts[:idx]
            else:
                negative_facts = negative_facts[:idx]
            report_idxs = report_idxs[:idx]
            aux = 0
            if use_facts_for_train:
                logger.info(f'len(train_indices) = {len(train_indices)}')
                aux += len(train_indices)
            if use_facts_for_test:
                logger.info(f'len(test_indices) = {len(test_indices)}')
                aux += len(test_indices)
            assert aux == idx # sanity check
            if use_facts_for_train and use_facts_for_test:
                intersection = len(set(train_indices) & set(test_indices))
                logger.info(f'len(set(train_indices) & set(test_indices)) = {intersection}')
                assert intersection == 0 # no intersection

            # Calculate the average number of facts per image
            if use_facts_for_train:
                aux = 0
                for i in train_indices:
                    pos_facts = positive_facts[i]
                    if use_strong_and_weak_negatives:
                        strong_neg_facts = strong_negative_facts[i]
                        weak_neg_facts = weak_negative_facts[i]
                        num_facts = len(pos_facts) + len(strong_neg_facts) + len(weak_neg_facts)
                        assert num_facts > 0 # at least one fact
                        aux += num_facts
                    else:
                        neg_facts = negative_facts[i]
                        num_facts = len(pos_facts) + len(neg_facts)
                        assert num_facts > 0
                        aux += num_facts
                avg_facts_per_image = aux / len(train_indices)
                train_num_facts_per_image = min(max_phrases_per_image, int(avg_facts_per_image))
                logger.info(f'avg_facts_per_image = {avg_facts_per_image}')
                logger.info(f'train_num_facts_per_image = {train_num_facts_per_image}')
            if use_facts_for_test:
                aux = 0
                for i in test_indices:                    
                    pos_facts = positive_facts[i]
                    if use_strong_and_weak_negatives:
                        strong_neg_facts = strong_negative_facts[i]
                        weak_neg_facts = weak_negative_facts[i]
                        num_facts = len(pos_facts) + len(strong_neg_facts) + len(weak_neg_facts)
                        assert num_facts > 0
                        aux += num_facts
                    else:
                        neg_facts = negative_facts[i]
                        num_facts = len(pos_facts) + len(neg_facts)
                        assert num_facts > 0
                        aux += num_facts
                avg_facts_per_image = aux / len(test_indices)
                test_num_facts_per_image = min(max_phrases_per_image, int(avg_facts_per_image))
                logger.info(f'avg_facts_per_image = {avg_facts_per_image}')
                logger.info(f'test_num_facts_per_image = {test_num_facts_per_image}')

            # Create dataset and dataloader for training
            if use_facts_for_train:
                logger.info(f'{ANSI_BOLD}Building train fact dataloader...{ANSI_RESET}')
                batch_size = max(min(max_images_per_batch, max_phrases_per_batch // train_num_facts_per_image), 1) # at least 1
                logger.info(f'batch_size = {batch_size}')

                if balance_long_middle_short_tail:
                    logger.info('Balancing long, middle, and short tail classes...')
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
                        dataset = MIMICCXR_FactClassificationDataset(
                            image_paths=image_paths, image_transform=train_image_transform,
                            fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                            indices=indices_, num_facts_per_image=train_num_facts_per_image, shuffle=True, infinite=True,
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
                        persistent_workers=True, # for faster training
                    )
                else:
                    logger.info('Normal (unbalanced) training...')
                    train_fact_dataset = MIMICCXR_FactClassificationDataset(
                        image_paths=image_paths, image_transform=train_image_transform,
                        fact_embeddings=fact_embeddings,
                        positive_facts=positive_facts,
                        use_strong_and_weak_negatives=use_strong_and_weak_negatives,
                        negative_facts=negative_facts if not use_strong_and_weak_negatives else None,
                        weak_negative_facts=weak_negative_facts if use_strong_and_weak_negatives else None,
                        strong_negative_facts=strong_negative_facts if use_strong_and_weak_negatives else None,
                        indices=train_indices, num_facts_per_image=train_num_facts_per_image,
                        use_weights=use_weighted_phrase_classifier_loss,
                        weights_filepath=cluster_and_label_weights_for_facts_filepath)
                    train_fact_dataloader = DataLoader(
                        train_fact_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_train_workers,
                        collate_fn=fact_grounding_collate_batch_fn,
                        pin_memory=True,
                        persistent_workers=True, # for faster training
                    )
                self.train_fact_dataset = train_fact_dataset
                self.train_fact_dataloader = train_fact_dataloader
                logger.info(f'len(self.train_fact_dataloader) = {len(self.train_fact_dataloader)}')

            # Create dataset and dataloader for testing
            if use_facts_for_test:
                logger.info(f'{ANSI_BOLD}Building test fact dataloaders...{ANSI_RESET}')
                test_fact_dataset = MIMICCXR_FactClassificationDataset(
                    image_paths=image_paths, image_transform=test_image_transform,
                    fact_embeddings=fact_embeddings,
                    positive_facts=positive_facts,
                    negative_facts=negative_facts if not use_strong_and_weak_negatives else None,
                    use_strong_and_weak_negatives=use_strong_and_weak_negatives,
                    weak_negative_facts=weak_negative_facts if use_strong_and_weak_negatives else None,
                    strong_negative_facts=strong_negative_facts if use_strong_and_weak_negatives else None,
                    indices=test_indices, num_facts_per_image=test_num_facts_per_image,
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
                    persistent_workers=True, # for faster evaluation
                )
                self.test_fact_dataset = test_fact_dataset
                self.test_fact_dataloader = test_fact_dataloader
                logger.info(f'len(self.test_fact_dataloader) = {len(self.test_fact_dataloader)}')

        # Create mscxr train/val/test dataset and dataloader
        if use_mscxr_for_train or use_mscxr_for_val or use_mscxr_for_test:
            if mscxr_do_grounding_only:
                assert mscxr_phrase2embedding_filepath is not None
                options = []
                if use_mscxr_for_train: options.append('train')
                if use_mscxr_for_val: options.append('val')
                if use_mscxr_for_test: options.append('test')
                logger.info(f'{ANSI_MAGENTA_BOLD}Preparing MS-CXR dataset and dataloaders for {", ".join(options)} (grounding only) ...{ANSI_RESET}')
                if use_mscxr_for_train:
                    assert num_train_workers is not None
                    assert train_image_transform is not None
                if use_mscxr_for_val or use_mscxr_for_test:
                    assert num_test_workers is not None
                    assert test_image_transform is not None
                if mscxr_test_on_all_images:
                    assert use_mscxr_for_test
                
                dicom_id_2_phrases_and_bboxes = get_ms_cxr_dicom_id_2_phrases_and_bboxes(bbox_format=bbox_format)
                logger.info(f'len(dicom_id_2_phrases_and_bboxes) = {len(dicom_id_2_phrases_and_bboxes)}')

                phrase2embedding = load_pickle(mscxr_phrase2embedding_filepath)
                logger.info(f'len(phrase2embedding) = {len(phrase2embedding)}')
                phrases = list(phrase2embedding.keys())
                phrases.sort()
                phrase2idx = {p: i for i, p in enumerate(phrases)}
                phrase_embeddings = np.array([phrase2embedding[p] for p in phrases])
                logger.info(f'phrase_embeddings.shape = {phrase_embeddings.shape}')
                
                self.mscxr_phrases = phrases
                self.mscxr_phrase2idx = phrase2idx
                self.mscxr_phrase_embeddings = phrase_embeddings

                BIG_ENOGUGH = 1000000
                image_paths = [None] * BIG_ENOGUGH
                dicom_ids = [None] * BIG_ENOGUGH
                phrase_idxs = [None] * BIG_ENOGUGH
                phrase_bboxes = [None] * BIG_ENOGUGH
                train_indices = []
                val_indices = []
                test_indices = []
                if use_mscxr_for_train:
                    phrase_prob_masks = [None] * BIG_ENOGUGH

                mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()
                dicom_id_2_split = get_ms_cxr_dicom_id_2_split()

                idx = 0
                for dicom_id_view_pairs in tqdm(mimiccxr_metadata['dicom_id_view_pos_pairs'], mininterval=2):
                    for dicom_id, _ in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                        if dicom_id not in dicom_id_2_split:
                            continue
                        image_path = dicom_id_to_image_path(dicom_id)
                        phrases_, bboxes_list_ = dicom_id_2_phrases_and_bboxes[dicom_id]
                        split = dicom_id_2_split[dicom_id]
                        for phrase_, bboxes_ in zip(phrases_, bboxes_list_):
                            phrase_idx = phrase2idx[phrase_]
                            image_paths[idx] = image_path
                            dicom_ids[idx] = dicom_id
                            phrase_idxs[idx] = phrase_idx
                            phrase_bboxes[idx] = bboxes_
                            if use_mscxr_for_train:
                                phrase_prob_masks[idx] = calculate_probabilistic_mask_from_bboxes(
                                    bboxes=bboxes_,
                                    mask_resolution=(mask_height, mask_width),
                                    bbox_format=bbox_format,
                                )
                            if split == 'train':
                                train_indices.append(idx)
                            elif split == 'val':
                                val_indices.append(idx)
                            elif split == 'test':
                                test_indices.append(idx)
                            else:
                                raise ValueError(f'Invalid split: {split}')
                            idx += 1

                assert idx > 0                
                image_paths = image_paths[:idx]
                dicom_ids = dicom_ids[:idx]
                phrase_idxs = phrase_idxs[:idx]
                phrase_bboxes = phrase_bboxes[:idx]
                if use_mscxr_for_train:
                    phrase_prob_masks = phrase_prob_masks[:idx]
                logger.info(f'Total number of instances: {idx}')
                
                self.mscxr_phrase_idxs = phrase_idxs
                self.mscxr_phrase_bboxes = phrase_bboxes
                self.mscxr_train_indices = train_indices
                self.mscxr_val_indices = val_indices
                self.mscxr_test_indices = test_indices

                if use_mscxr_for_train:
                    logger.info(f'len(train_indices) = {len(train_indices)}')
                    assert len(train_indices) > 0
                    # Create dataset and dataloader for training
                    logger.info(f'{ANSI_BOLD}Building train fact dataloader...{ANSI_RESET}')
                    batch_size = max_images_per_batch
                    logger.info(f'batch_size = {batch_size}') 
                    if mscxr_training_data_mode == MS_CXR_TrainingMode.TRAIN.value:
                        actual_train_indices = train_indices
                    elif mscxr_training_data_mode == MS_CXR_TrainingMode.VAL.value:
                        actual_train_indices = val_indices
                    elif mscxr_training_data_mode == MS_CXR_TrainingMode.TEST.value:
                        actual_train_indices = test_indices
                    elif mscxr_training_data_mode == MS_CXR_TrainingMode.ALL.value:
                        actual_train_indices = train_indices + val_indices + test_indices
                    else: raise ValueError(f'Invalid training data mode: {mscxr_training_data_mode}')
                    mscxr_train_dataset = MSCXR_PhraseGroundingDataset(
                        image_paths=image_paths,
                        image_transform=train_image_transform,
                        phrase_idxs=phrase_idxs,
                        phrase_embeddings=phrase_embeddings,
                        phrase_bboxes=phrase_bboxes,
                        phrase_prob_masks=phrase_prob_masks,
                        indices=actual_train_indices,
                        feature_map_size=(mask_height, mask_width),
                        data_augmentation_enabled=data_augmentation_enabled,
                        for_training=True,
                        bbox_format=bbox_format)
                    mscxr_train_dataloader = DataLoader(
                        mscxr_train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_train_workers,
                        collate_fn=mscxr_train_dataset.collate_fn,
                        pin_memory=True,
                        persistent_workers=True, # for faster training
                    )
                    self.mscxr_train_dataset = mscxr_train_dataset
                    self.mscxr_train_dataloader = mscxr_train_dataloader
                    logger.info(f'len(self.mscxr_train_dataset) = {len(self.mscxr_train_dataset)}')
                    logger.info(f'len(self.mscxr_train_dataloader) = {len(self.mscxr_train_dataloader)}')

                if use_mscxr_for_val:
                    logger.info(f'len(val_indices) = {len(val_indices)}')
                    assert len(val_indices) > 0
                    # Create dataset and dataloader for validation
                    logger.info(f'{ANSI_BOLD}Building val fact dataloader...{ANSI_RESET}')
                    batch_size = int(max_images_per_batch * test_batch_size_factor)
                    logger.info(f'batch_size = {batch_size}')
                    mscxr_val_dataset = MSCXR_PhraseGroundingDataset(
                        image_paths=image_paths,
                        image_transform=test_image_transform,
                        phrase_idxs=phrase_idxs,
                        phrase_embeddings=phrase_embeddings,
                        phrase_bboxes=phrase_bboxes,
                        indices=val_indices,
                        bbox_format=bbox_format,
                        for_training=False)
                    mscxr_val_dataloader = DataLoader(
                        mscxr_val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_test_workers,
                        collate_fn=mscxr_val_dataset.collate_fn,
                        pin_memory=True,
                        persistent_workers=True, # for faster evaluation
                    )
                    self.mscxr_val_dataset = mscxr_val_dataset
                    self.mscxr_val_dataloader = mscxr_val_dataloader
                    logger.info(f'len(self.mscxr_val_dataset) = {len(self.mscxr_val_dataset)}')
                    logger.info(f'len(self.mscxr_val_dataloader) = {len(self.mscxr_val_dataloader)}')

                if use_mscxr_for_test:
                    logger.info(f'len(test_indices) = {len(test_indices)}')
                    assert len(test_indices) > 0
                    # Create dataset and dataloader for testing
                    logger.info(f'{ANSI_BOLD}Building test fact dataloader...{ANSI_RESET}')
                    batch_size = int(max_images_per_batch * test_batch_size_factor)
                    logger.info(f'batch_size = {batch_size}')
                    mscxr_test_dataset = MSCXR_PhraseGroundingDataset(
                        image_paths=image_paths,
                        image_transform=test_image_transform,
                        phrase_idxs=phrase_idxs,
                        phrase_embeddings=phrase_embeddings,
                        phrase_bboxes=phrase_bboxes,
                        indices=test_indices,
                        bbox_format=bbox_format,
                        for_training=False)
                    mscxr_test_dataloader = DataLoader(
                        mscxr_test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_test_workers,
                        collate_fn=mscxr_test_dataset.collate_fn,
                        pin_memory=True,
                        persistent_workers=True, # for faster evaluation
                    )
                    self.mscxr_test_dataset = mscxr_test_dataset
                    self.mscxr_test_dataloader = mscxr_test_dataloader
                    logger.info(f'len(self.mscxr_test_dataset) = {len(self.mscxr_test_dataset)}')
                    logger.info(f'len(self.mscxr_test_dataloader) = {len(self.mscxr_test_dataloader)}')
            else:
                assert mscxr_phrase2embedding_filepath is not None
                assert dicom_id_to_pos_neg_facts_filepath is not None
                options = []
                if use_mscxr_for_train: options.append('train')
                if use_mscxr_for_val: options.append('val')
                if use_mscxr_for_test: options.append('test')
                logger.info(f'{ANSI_MAGENTA_BOLD}Preparing MS-CXR dataset and dataloaders for {", ".join(options)} ...{ANSI_RESET}')
                if use_mscxr_for_train:
                    assert num_train_workers is not None
                    assert train_image_transform is not None
                if use_mscxr_for_val or use_mscxr_for_test:
                    assert num_test_workers is not None
                    assert test_image_transform is not None
                if mscxr_test_on_all_images:
                    assert use_mscxr_for_test
                
                dicom_id_2_phrases_and_bboxes = get_ms_cxr_dicom_id_2_phrases_and_bboxes(bbox_format=bbox_format)
                logger.info(f'len(dicom_id_2_phrases_and_bboxes) = {len(dicom_id_2_phrases_and_bboxes)}')

                phrase2embedding = load_pickle(mscxr_phrase2embedding_filepath)
                logger.info(f'len(phrase2embedding) = {len(phrase2embedding)}')
                phrases = list(phrase2embedding.keys())
                phrases.sort()
                phrase2idx = {p: i for i, p in enumerate(phrases)}
                phrase_embeddings = np.array([phrase2embedding[p] for p in phrases])
                logger.info(f'phrase_embeddings.shape = {phrase_embeddings.shape}')
                
                self.mscxr_phrases = phrases
                self.mscxr_phrase2idx = phrase2idx
                self.mscxr_phrase_embeddings = phrase_embeddings

                tmp = get_cached_pickle_file(dicom_id_to_pos_neg_facts_filepath)
                fact_embeddings = tmp['embeddings']
                dicom_id_to_pos_facts = tmp['dicom_id_to_pos_facts']
                dicom_id_to_strong_neg_facts = tmp['dicom_id_to_strong_neg_facts']
                dicom_id_to_weak_neg_facts = tmp['dicom_id_to_weak_neg_facts']
                logger.info(f'fact_embeddings.shape = {fact_embeddings.shape}')

                self.mscxr_facts = tmp['facts']
                self.mscxr_fact_embeddings = fact_embeddings

                BIG_ENOGUGH = 1000000
                image_paths = [None] * BIG_ENOGUGH
                dicom_ids = [None] * BIG_ENOGUGH
                phrase_idxs = [None] * BIG_ENOGUGH
                phrase_bboxes_and_classes = [None] * BIG_ENOGUGH
                pos_fact_idxs = [None] * BIG_ENOGUGH
                strong_neg_fact_idxs = [None] * BIG_ENOGUGH
                weak_neg_fact_idxs = [None] * BIG_ENOGUGH
                if use_mscxr_for_train or mscxr_test_on_all_images:
                    train_indices = []
                if use_mscxr_for_val or mscxr_test_on_all_images:
                    val_indices = []
                if use_mscxr_for_test:
                    test_indices = []

                mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()
                dicom_id_2_split = get_ms_cxr_dicom_id_2_split()

                idx = 0
                for dicom_id_view_pairs in tqdm(mimiccxr_metadata['dicom_id_view_pos_pairs'], mininterval=2):
                    for dicom_id, _ in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                        if dicom_id not in dicom_id_2_split:
                            continue
                        image_paths[idx] = dicom_id_to_image_path(dicom_id)
                        phrases_, bboxes_list_ = dicom_id_2_phrases_and_bboxes[dicom_id]
                        phrase_idxs_ = [phrase2idx[p] for p in phrases_]
                        pos_fact_idxs_ = dicom_id_to_pos_facts[dicom_id]
                        strong_neg_fact_idxs_ = dicom_id_to_strong_neg_facts[dicom_id]
                        weak_neg_fact_idxs_ = dicom_id_to_weak_neg_facts[dicom_id]
                        dicom_ids[idx] = dicom_id
                        phrase_idxs[idx] = phrase_idxs_
                        flattened_bboxes_ = []
                        classes_ = []
                        for i, bboxes_ in enumerate(bboxes_list_): # bboxes_: List[Tuple[float, float, float, float]]
                            flattened_bboxes_.extend(bboxes_)
                            classes_.extend([i] * len(bboxes_))
                        phrase_bboxes_and_classes[idx] = (flattened_bboxes_, classes_)
                        pos_fact_idxs[idx] = pos_fact_idxs_
                        strong_neg_fact_idxs[idx] = strong_neg_fact_idxs_
                        weak_neg_fact_idxs[idx] = weak_neg_fact_idxs_
                        split = dicom_id_2_split[dicom_id]
                        if split == 'train':
                            if use_mscxr_for_train or mscxr_test_on_all_images:
                                train_indices.append(idx)
                            else:
                                continue
                        elif split == 'val':
                            if use_mscxr_for_val or mscxr_test_on_all_images:
                                val_indices.append(idx)
                            else:
                                continue
                        elif split == 'test':
                            if use_mscxr_for_test:
                                test_indices.append(idx)
                            else:
                                continue
                        else:
                            raise ValueError(f'Invalid split: {split}')
                        idx += 1

                logger.info(f'Total number of images: {idx}')
                logger.info(f'phrase_bboxes_and_classes[0] = {phrase_bboxes_and_classes[0]}')
                assert idx > 0
                image_paths = image_paths[:idx]
                dicom_ids = dicom_ids[:idx]
                phrase_idxs = phrase_idxs[:idx]
                phrase_bboxes_and_classes = phrase_bboxes_and_classes[:idx]
                pos_fact_idxs = pos_fact_idxs[:idx]
                strong_neg_fact_idxs = strong_neg_fact_idxs[:idx]
                
                self.mscxr_phrase_idxs = phrase_idxs
                self.mscxr_phrase_bboxes_and_classes = phrase_bboxes_and_classes
                self.mscxr_pos_fact_idxs = pos_fact_idxs
                self.mscxr_strong_neg_fact_idxs = strong_neg_fact_idxs
                self.mscxr_weak_neg_fact_idxs = weak_neg_fact_idxs
                if use_mscxr_for_train or mscxr_test_on_all_images:
                    self.mscxr_train_indices = train_indices
                if use_mscxr_for_val or mscxr_test_on_all_images:
                    self.mscxr_val_indices = val_indices
                if use_mscxr_for_test:
                    self.mscxr_test_indices = test_indices

                # Calculate the average number of facts per image
                def _calc_avg_facts_per_image(indices):
                    aux = 0
                    for i in indices:
                        num_facts = len(phrase_idxs[i]) + len(pos_fact_idxs[i]) + len(strong_neg_fact_idxs[i]) + len(weak_neg_fact_idxs[i])
                        assert num_facts > 0
                        aux += num_facts
                    return aux / len(indices)
                if use_mscxr_for_train:
                    avg_facts_per_image = _calc_avg_facts_per_image(train_indices)
                    train_num_facts_per_image = min(max_phrases_per_image, int(avg_facts_per_image))
                    logger.info(f'avg_facts_per_image = {avg_facts_per_image}')
                    logger.info(f'train_num_facts_per_image = {train_num_facts_per_image}')
                if use_mscxr_for_val:
                    avg_facts_per_image = _calc_avg_facts_per_image(val_indices)
                    val_num_facts_per_image = min(max_phrases_per_image, int(avg_facts_per_image))
                    logger.info(f'avg_facts_per_image = {avg_facts_per_image}')
                    logger.info(f'val_num_facts_per_image = {val_num_facts_per_image}')
                if use_mscxr_for_test:
                    avg_facts_per_image = _calc_avg_facts_per_image(test_indices)
                    test_num_facts_per_image = min(max_phrases_per_image, int(avg_facts_per_image))
                    logger.info(f'avg_facts_per_image = {avg_facts_per_image}')
                    logger.info(f'test_num_facts_per_image = {test_num_facts_per_image}')

                if use_mscxr_for_train:
                    logger.info(f'len(train_indices) = {len(train_indices)}')
                    assert len(train_indices) > 0
                    # Create dataset and dataloader for training
                    logger.info(f'{ANSI_BOLD}Building train fact dataloader...{ANSI_RESET}')
                    batch_size = max(min(max_images_per_batch, max_phrases_per_batch // train_num_facts_per_image), 1) # at least 1
                    logger.info(f'batch_size = {batch_size}')
                    mscxr_train_dataset = MSCXR_PhraseGroundingAndClassificationDataset(
                        image_paths=image_paths, image_transform=train_image_transform,
                        phrase_idxs=phrase_idxs, phrase_embeddings=phrase_embeddings,
                        phrase_bboxes_and_classes=phrase_bboxes_and_classes,
                        positive_fact_idxs=pos_fact_idxs,
                        strong_neg_fact_idxs=strong_neg_fact_idxs,
                        weak_neg_fact_idxs=weak_neg_fact_idxs,
                        fact_embeddings=fact_embeddings,
                        indices=train_indices,
                        num_facts_per_image=train_num_facts_per_image,
                        feature_map_size=(mask_height, mask_width),
                        data_augmentation_enabled=data_augmentation_enabled,
                        for_training=True, bbox_format=bbox_format)
                    mscxr_train_dataloader = DataLoader(
                        mscxr_train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_train_workers,
                        collate_fn=mscxr_train_dataset.collate_fn,
                        pin_memory=True,
                        persistent_workers=True, # for faster training
                    )
                    self.mscxr_train_dataset = mscxr_train_dataset
                    self.mscxr_train_dataloader = mscxr_train_dataloader
                    logger.info(f'len(self.mscxr_train_dataset) = {len(self.mscxr_train_dataset)}')
                    logger.info(f'len(self.mscxr_train_dataloader) = {len(self.mscxr_train_dataloader)}')

                if use_mscxr_for_val:
                    logger.info(f'len(val_indices) = {len(val_indices)}')
                    assert len(val_indices) > 0
                    # Create dataset and dataloader for validation
                    logger.info(f'{ANSI_BOLD}Building val fact dataloader...{ANSI_RESET}')
                    batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // val_num_facts_per_image), 1) * test_batch_size_factor)
                    logger.info(f'batch_size = {batch_size}')
                    mscxr_val_dataset = MSCXR_PhraseGroundingAndClassificationDataset(
                        image_paths=image_paths, image_transform=test_image_transform,
                        phrase_idxs=phrase_idxs, phrase_embeddings=phrase_embeddings,
                        phrase_bboxes_and_classes=phrase_bboxes_and_classes,
                        positive_fact_idxs=pos_fact_idxs,
                        strong_neg_fact_idxs=strong_neg_fact_idxs,
                        weak_neg_fact_idxs=weak_neg_fact_idxs,
                        fact_embeddings=fact_embeddings,
                        indices=val_indices,
                        num_facts_per_image=val_num_facts_per_image,
                        feature_map_size=(mask_height, mask_width),
                        for_training=False, bbox_format=bbox_format)
                    mscxr_val_dataloader = DataLoader(
                        mscxr_val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_test_workers,
                        collate_fn=mscxr_val_dataset.collate_fn,
                        pin_memory=True,
                        persistent_workers=True, # for faster evaluation
                    )
                    self.mscxr_val_dataset = mscxr_val_dataset
                    self.mscxr_val_dataloader = mscxr_val_dataloader
                    logger.info(f'len(self.mscxr_val_dataset) = {len(self.mscxr_val_dataset)}')
                    logger.info(f'len(self.mscxr_val_dataloader) = {len(self.mscxr_val_dataloader)}')

                if use_mscxr_for_test:
                    logger.info(f'len(test_indices) = {len(test_indices)}')
                    assert len(test_indices) > 0
                    # Create dataset and dataloader for testing
                    logger.info(f'{ANSI_BOLD}Building test fact dataloader...{ANSI_RESET}')
                    batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // test_num_facts_per_image), 1) * test_batch_size_factor)
                    logger.info(f'batch_size = {batch_size}')
                    mscxr_test_dataset = MSCXR_PhraseGroundingAndClassificationDataset(
                        image_paths=image_paths, image_transform=test_image_transform,
                        phrase_idxs=phrase_idxs, phrase_embeddings=phrase_embeddings,
                        phrase_bboxes_and_classes=phrase_bboxes_and_classes,
                        positive_fact_idxs=pos_fact_idxs,
                        strong_neg_fact_idxs=strong_neg_fact_idxs,
                        weak_neg_fact_idxs=weak_neg_fact_idxs,
                        fact_embeddings=fact_embeddings,
                        indices=(train_indices + val_indices + test_indices) if mscxr_test_on_all_images else test_indices,
                        num_facts_per_image=test_num_facts_per_image,
                        feature_map_size=(mask_height, mask_width),
                        for_training=False, bbox_format=bbox_format)
                    mscxr_test_dataloader = DataLoader(
                        mscxr_test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_test_workers,
                        collate_fn=mscxr_test_dataset.collate_fn,
                        pin_memory=True,
                        persistent_workers=True, # for faster evaluation
                    )
                    self.mscxr_test_dataset = mscxr_test_dataset
                    self.mscxr_test_dataloader = mscxr_test_dataloader
                    logger.info(f'len(self.mscxr_test_dataset) = {len(self.mscxr_test_dataset)}')
                    logger.info(f'len(self.mscxr_test_dataloader) = {len(self.mscxr_test_dataloader)}')

        
        # Create Chest ImaGenome train/val/test dataset and dataloader
        if use_chest_imagenome_for_train or use_chest_imagenome_for_val or use_chest_imagenome_for_test:
            
            # ===================================
            # 1) Chest Imagenome Phrase Grounding
            # ===================================
            logger.info(f'{ANSI_MAGENTA_BOLD}Preparing Chest Imagenome Phrase Grounding datasets and dataloaders...{ANSI_RESET}')

            assert chest_imagenome_augmented_phrase_groundings_filepath is not None
            assert chest_imagenome_phrase_embeddings_filepath is not None
            assert chest_imagenome_bbox_phrase_embeddings_filepath is not None
            assert mask_width is not None
            assert mask_height is not None
            assert num_train_workers is not None
            
            logger.info(f'Loading Chest Imagenome augmented phrase groundings from {chest_imagenome_augmented_phrase_groundings_filepath}...')
            augmented_data = load_pickle(chest_imagenome_augmented_phrase_groundings_filepath)
            logger.info(f'len(augmented_data) = {len(augmented_data)}')

            logger.info(f'Loading Chest Imagenome phrase embeddings from {chest_imagenome_phrase_embeddings_filepath}...')
            chest_imagenome_phrase_embeddings = get_cached_pickle_file(chest_imagenome_phrase_embeddings_filepath)            
            phrases = chest_imagenome_phrase_embeddings['phrases']
            chest_imagenome_phrase_embeddings = chest_imagenome_phrase_embeddings['embeddings']
            phrase_to_embedding_idx = {phrase: idx for idx, phrase in enumerate(phrases)}
            logger.info(f'len(phrases) = {len(phrases)}')
            logger.info(f'chest_imagenome_phrase_embeddings.shape = {chest_imagenome_phrase_embeddings.shape}')
            assert chest_imagenome_phrase_embeddings.shape[0] == len(phrases)

            split_to_image_ids = get_split2imageIds()

            # Create dataset and dataloader for training
            if use_chest_imagenome_for_train:
                train_dicom_ids = set(split_to_image_ids['train'])
                logger.info(f'len(train_dicom_ids) = {len(train_dicom_ids)}')
                train_dicom_ids -= forbidden_train_dicom_ids # remove forbidden dicom ids
                logger.info(f'len(train_dicom_ids) after removing forbidden ids = {len(train_dicom_ids)}')
                train_augmented_data = [item for item in augmented_data if item['dicom_id'] in train_dicom_ids]
                logger.info(f'len(train_augmented_data) = {len(train_augmented_data)}')

                self.chest_imagenome_pg_train_dataset = ChestImaGenome_PhraseGroundingDataset(
                    augmented_data=train_augmented_data,
                    image_transform=train_image_transform,
                    dicom_id_to_image_path=dicom_id_to_image_path,
                    phrase_to_embedding_idx=phrase_to_embedding_idx,
                    phrase_embeddings=chest_imagenome_phrase_embeddings,
                    feature_map_size=(mask_height, mask_width),
                    for_training=True,
                    data_augmentation_enabled=data_augmentation_enabled,
                )
                self.chest_imagenome_pg_train_dataloader = DataLoader(
                    self.chest_imagenome_pg_train_dataset,
                    batch_size=max_images_per_batch,
                    shuffle=True,
                    num_workers=num_train_workers,
                    collate_fn=self.chest_imagenome_pg_train_dataset.collate_fn,
                    pin_memory=True,
                    persistent_workers=True, # for faster training
                )
                logger.info(f'len(self.chest_imagenome_pg_train_dataset) = {len(self.chest_imagenome_pg_train_dataset)}')

            # Create dataset and dataloader for validation
            if use_chest_imagenome_for_val:
                val_dicom_ids = set(split_to_image_ids['validate'])
                val_augmented_data = [item for item in augmented_data if item['dicom_id'] in val_dicom_ids]
                logger.info(f'len(val_augmented_data) = {len(val_augmented_data)}')
                
                self.chest_imagenome_pg_val_dataset = ChestImaGenome_PhraseGroundingDataset(
                    augmented_data=val_augmented_data,
                    image_transform=test_image_transform,
                    dicom_id_to_image_path=dicom_id_to_image_path,
                    phrase_to_embedding_idx=phrase_to_embedding_idx,
                    phrase_embeddings=chest_imagenome_phrase_embeddings,
                    feature_map_size=(mask_height, mask_width),
                    for_training=False,
                    data_augmentation_enabled=False,
                )
                self.chest_imagenome_pg_val_dataloader = DataLoader(
                    self.chest_imagenome_pg_val_dataset,
                    batch_size=int(max_images_per_batch * test_batch_size_factor),
                    shuffle=False,
                    num_workers=num_test_workers,
                    collate_fn=self.chest_imagenome_pg_val_dataset.collate_fn,
                    pin_memory=True,
                    persistent_workers=True, # for faster evaluation
                )
                logger.info(f'len(self.chest_imagenome_pg_val_dataset) = {len(self.chest_imagenome_pg_val_dataset)}')
            
            # Create dataset and dataloader for testing
            if use_chest_imagenome_for_test:
                test_dicom_ids = set(split_to_image_ids['test'])
                test_augmented_data = [item for item in augmented_data if item['dicom_id'] in test_dicom_ids]
                logger.info(f'len(test_augmented_data) = {len(test_augmented_data)}')

                self.chest_imagenome_pg_test_dataset = ChestImaGenome_PhraseGroundingDataset(
                    augmented_data=test_augmented_data,
                    image_transform=test_image_transform,
                    dicom_id_to_image_path=dicom_id_to_image_path,
                    phrase_to_embedding_idx=phrase_to_embedding_idx,
                    phrase_embeddings=chest_imagenome_phrase_embeddings,
                    feature_map_size=(mask_height, mask_width),
                    for_training=False,
                    data_augmentation_enabled=False,
                )
                self.chest_imagenome_pg_test_dataloader = DataLoader(
                    self.chest_imagenome_pg_test_dataset,
                    batch_size=int(max_images_per_batch * test_batch_size_factor),
                    shuffle=False,
                    num_workers=num_test_workers,
                    collate_fn=self.chest_imagenome_pg_test_dataset.collate_fn,
                    pin_memory=True,
                    persistent_workers=True, # for faster evaluation
                )
                logger.info(f'len(self.chest_imagenome_pg_test_dataset) = {len(self.chest_imagenome_pg_test_dataset)}')

            # ================================================
            # 2) Chest Imagenome Anatomical Location Grounding
            # ================================================
            logger.info(f'{ANSI_MAGENTA_BOLD}Preparing Chest Imagenome Anatomical Location Grounding datasets and dataloaders...{ANSI_RESET}')

            bbox_phrase_embeddings_data = get_cached_pickle_file(chest_imagenome_bbox_phrase_embeddings_filepath)
            train_batch_size = max(min(max_images_per_batch,
                                       max_phrases_per_batch // len(CHEST_IMAGENOME_BBOX_NAMES)), 1) # at least 1 image per batch

            # Create dataset and dataloader for training
            if use_chest_imagenome_for_train:
                train_dicom_ids = set(split_to_image_ids['train'])
                logger.info(f'len(train_dicom_ids) = {len(train_dicom_ids)}')
                train_dicom_ids -= forbidden_train_dicom_ids # remove forbidden dicom ids
                logger.info(f'len(train_dicom_ids) after removing forbidden ids = {len(train_dicom_ids)}')
                train_augmented_data = [item for item in augmented_data if item['dicom_id'] in train_dicom_ids]
                logger.info(f'len(train_augmented_data) = {len(train_augmented_data)}')

                self.chest_imagenome_alg_train_dataset = ChestImaGenome_AnatomicalLocationGroundingDataset(
                    augmented_data=train_augmented_data,
                    image_transform=train_image_transform,
                    dicom_id_to_image_path=dicom_id_to_image_path,
                    bbox_phrase_embeddings_data=bbox_phrase_embeddings_data,
                    feature_map_size=(mask_height, mask_width),
                    for_training=True,
                    data_augmentation_enabled=data_augmentation_enabled,
                    bbox_format=bbox_format,
                    num_processes=num_processes,
                )
                self.chest_imagenome_alg_train_dataloader = DataLoader(
                    self.chest_imagenome_alg_train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    num_workers=num_train_workers,
                    collate_fn=self.chest_imagenome_alg_train_dataset.collate_fn,
                    pin_memory=True,
                    persistent_workers=True, # for faster training
                )
                logger.info(f'len(self.chest_imagenome_alg_train_dataset) = {len(self.chest_imagenome_alg_train_dataset)}')

            # Create dataset and dataloader for validation
            if use_chest_imagenome_for_val:
                val_dicom_ids = set(split_to_image_ids['validate'])
                val_augmented_data = [item for item in augmented_data if item['dicom_id'] in val_dicom_ids]
                logger.info(f'len(val_augmented_data) = {len(val_augmented_data)}')

                self.chest_imagenome_alg_val_dataset = ChestImaGenome_AnatomicalLocationGroundingDataset(
                    augmented_data=val_augmented_data,
                    image_transform=test_image_transform,
                    dicom_id_to_image_path=dicom_id_to_image_path,
                    bbox_phrase_embeddings_data=bbox_phrase_embeddings_data,
                    feature_map_size=(mask_height, mask_width),
                    for_training=False,
                    data_augmentation_enabled=False,
                    bbox_format=bbox_format,
                )
                self.chest_imagenome_alg_val_dataloader = DataLoader(
                    self.chest_imagenome_alg_val_dataset,
                    batch_size=int(train_batch_size * test_batch_size_factor),
                    shuffle=False,
                    num_workers=num_test_workers,
                    collate_fn=self.chest_imagenome_alg_val_dataset.collate_fn,
                    pin_memory=True,
                    persistent_workers=True, # for faster evaluation
                )
                logger.info(f'len(self.chest_imagenome_alg_val_dataset) = {len(self.chest_imagenome_alg_val_dataset)}')

            # Create dataset and dataloader for testing
            if use_chest_imagenome_for_test:
                test_dicom_ids = set(split_to_image_ids['test'])
                test_augmented_data = [item for item in augmented_data if item['dicom_id'] in test_dicom_ids]
                logger.info(f'len(test_augmented_data) = {len(test_augmented_data)}')

                self.chest_imagenome_alg_test_dataset = ChestImaGenome_AnatomicalLocationGroundingDataset(
                    augmented_data=test_augmented_data,
                    image_transform=test_image_transform,
                    dicom_id_to_image_path=dicom_id_to_image_path,
                    bbox_phrase_embeddings_data=bbox_phrase_embeddings_data,
                    feature_map_size=(mask_height, mask_width),
                    for_training=False,
                    data_augmentation_enabled=False,
                    bbox_format=bbox_format,
                )
                self.chest_imagenome_alg_test_dataloader = DataLoader(
                    self.chest_imagenome_alg_test_dataset,
                    batch_size=int(train_batch_size * test_batch_size_factor),
                    shuffle=False,
                    num_workers=num_test_workers,
                    collate_fn=self.chest_imagenome_alg_test_dataset.collate_fn,
                    pin_memory=True,
                    persistent_workers=True, # for faster evaluation
                )
                logger.info(f'len(self.chest_imagenome_alg_test_dataset) = {len(self.chest_imagenome_alg_test_dataset)}')