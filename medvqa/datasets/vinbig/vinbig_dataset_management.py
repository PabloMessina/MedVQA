import os
import logging
import numpy as np
import math
import pandas as pd
import random
import torch
from typing import List, Dict, Tuple, Optional, Literal, Callable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from medvqa.datasets.augmentation import VinBigAlbumentationAdapter
from medvqa.datasets.vinbig import (
    VINBIG_BBOX_NAMES__MODIFIED,
    VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH,
    VINBIG_IMAGE_LABELS_TEST_CSV_PATH,
    N_IMAGES_TRAIN,
    N_IMAGES_TEST,
    VINBIG_LABELS__MODIFIED,
    _merge_labels,
    get_medium_size_image_path,
    get_original_image_path,
    load_labels,
    load_test_image_id_2_bboxes,
    load_train_image_id_2_bboxes,
)
from medvqa.datasets.visual_module import BasicImageDataset, MAETrainerBase
from medvqa.utils.bbox_utils import calculate_probabilistic_mask_from_bboxes, convert_bboxes_into_target_tensors
from medvqa.utils.common import ChoiceEnum
from medvqa.utils.constants import VINBIG_BBOX_NAMES, VINBIG_LABEL2PHRASE, VINBIG_LABELS
from medvqa.datasets.dataloading_utils import (
    INFINITE_DATASET_LENGTH,
    CompositeInfiniteDataset,
    group_indices_for_balanced_sampling,
)
from medvqa.datasets.vqa import load_precomputed_visual_features
from medvqa.utils.files_utils import get_cached_pickle_file, load_pickle, save_pickle
from medvqa.utils.logging_utils import ANSI_BOLD, ANSI_RESET

logger = logging.getLogger(__name__)

# Define a type alias for the bbox structure if loaded
BboxDataType = Dict[str, Tuple[List[List[float]], List[int]]]

class VinBigTrainingMode(ChoiceEnum):
    TRAIN = 'train'
    TEST = 'test'
    ALL = 'all'

class VinBigPhraseTaskMode(ChoiceEnum):
    CLASSIFICATION = 'classification'
    GROUNDING = 'grounding'
    CLASSIFICATION_AND_GROUNDING = 'classification_and_grounding'


class VinBigTrainerBase:
    """
    Base class for managing the VinBigData dataset for training or evaluation.

    Handles loading image-level labels, determining image paths, defining
    train/test splits based on indices, and optionally loading and processing
    bounding box annotations.

    Attributes:
        image_ids (List[str]): Combined list of training and testing image IDs.
        labels (np.ndarray): Numpy array of shape (N_IMAGES, N_CLASSES)
            containing the multi-label classification labels for each image.
        label_names (List[str]): List of class names corresponding to the
            columns in `labels`. Depends on `use_improved_labels`.
        image_paths (List[str]): List of file paths corresponding to each
            image in `image_ids`. Path depends on `use_original_image_size`.
        train_indices (List[int]): List of indices corresponding to the
            training images within the combined `image_ids` list.
        test_indices (List[int]): List of indices corresponding to the
            testing images within the combined `image_ids` list.
        image_id_2_bboxes (Optional[BboxDataType]): Dictionary mapping image IDs
            to their bounding box data (tuple of bbox list and class_id list)
            if `load_bounding_boxes` is True, otherwise None. Bboxes are
            normalized and formatted according to `bbox_format`.
        bboxes (Optional[List[Tuple[List[List[float]], List[int]]]]): List
            containing the bounding box data for each image, aligned with
            `image_ids`. Each element is a tuple: (list_of_bboxes, list_of_class_ids).
            None if `load_bounding_boxes` is False.
    """

    def __init__(
        self,
        load_bounding_boxes: bool = False,
        use_original_image_size: bool = False,
        use_improved_labels: bool = False,
        bbox_format: Literal["xyxy", "cxcywh"] = "xyxy",
        verbose: bool = False,
    ):
        """
        Initializes the VinBigTrainerBase instance.

        Args:
            load_bounding_boxes: If True, load and process bounding box
                annotations. Defaults to False.
            use_original_image_size: If True, use paths to original resolution
                images. If False (default), use paths to medium-sized images.
            use_improved_labels: If True, use modified/improved image-level
                labels and bounding box class definitions. Defaults to False.
            bbox_format: Specifies the format for bounding boxes if loaded.
                Can be 'xyxy' (default) or 'cxcywh'. This format is passed
                to the loading functions.
            verbose: If True, print additional information during loading
                (e.g., sample bounding boxes). Defaults to False.
        """

        # --- Load Image-Level Labels ---
        # Loads labels for both train and test sets based on the improve_labels flag.
        # Assumes load_labels returns two dictionaries: image_id -> label_vector
        logger.info("Loading image-level labels...")
        train_image_id_to_labels, test_image_id_to_labels = load_labels(
            improve_labels=use_improved_labels
        )
        train_image_ids = list(train_image_id_to_labels.keys())
        test_image_ids = list(test_image_id_to_labels.keys())

        # --- Basic Sanity Checks for Label Loading ---
        # Ensure the number of loaded IDs matches expected dataset sizes.
        assert len(train_image_ids) == N_IMAGES_TRAIN, (
            f"Expected {N_IMAGES_TRAIN} train image IDs, "
            f"but found {len(train_image_ids)}"
        )
        assert len(test_image_ids) == N_IMAGES_TEST, (
            f"Expected {N_IMAGES_TEST} test image IDs, "
            f"but found {len(test_image_ids)}"
        )

        # --- Combine Train/Test Data ---
        # Create unified lists/dictionaries for easier access.
        self.image_ids: List[str] = train_image_ids + test_image_ids
        image_id_to_labels: Dict[str, np.ndarray] = {
            **train_image_id_to_labels,
            **test_image_id_to_labels,
        }

        # Determine the correct set of label names based on the flag.
        self.label_names: List[str] = (
            VINBIG_LABELS__MODIFIED if use_improved_labels else VINBIG_LABELS
        )

        # Create the final labels array, ensuring order matches image_ids.
        self.labels: np.ndarray = np.array(
            [image_id_to_labels[img_id] for img_id in self.image_ids],
            dtype=np.int8,
        )
        # Verify the shape of the final labels array.
        expected_shape = (N_IMAGES_TRAIN + N_IMAGES_TEST, len(self.label_names))
        assert self.labels.shape == expected_shape, (
            f"Expected labels shape {expected_shape}, "
            f"but got {self.labels.shape}"
        )
        logger.info(f"Loaded labels for {len(self.image_ids)} images.")

        # --- Determine Image Paths ---
        # Select image paths based on whether original or processed size is requested.
        logger.info("Determining image paths...")
        if use_original_image_size:
            self.image_paths: List[str] = [
                get_original_image_path(img_id) for img_id in self.image_ids
            ]
            logger.info("Using original image paths.")
        else:
            self.image_paths: List[str] = [
                get_medium_size_image_path(img_id) for img_id in self.image_ids
            ]
            logger.info("Using medium-size image paths.")

        # --- Define Train/Test Indices ---
        # Simple indices based on the known sizes of train/test sets.
        self.train_indices: List[int] = list(range(N_IMAGES_TRAIN))
        self.test_indices: List[int] = list(
            range(N_IMAGES_TRAIN, N_IMAGES_TRAIN + N_IMAGES_TEST)
        )

        # --- Load Bounding Boxes (Optional) ---
        self.image_id_2_bboxes: Optional[BboxDataType] = None
        self.bboxes: Optional[
            List[Tuple[List[List[float]], List[int]]]
        ] = None

        if load_bounding_boxes:
            logger.info("Loading bounding boxes...")
            # Load normalized bounding boxes, formatted for training (list format),
            # using the specified label improvement and bbox format settings.
            train_image_id_2_bboxes = load_train_image_id_2_bboxes(
                for_training=True,  # Get (bbox_list, class_list) format
                normalize=True,  # Ensure coordinates are in [0, 1]
                improve_labels=use_improved_labels,
                bbox_format=bbox_format, # Pass the desired format
            )
            test_image_id_2_bboxes = load_test_image_id_2_bboxes(
                for_training=True,  # Get (bbox_list, class_list) format
                normalize=True,  # Ensure coordinates are in [0, 1]
                improve_labels=use_improved_labels,
                bbox_format=bbox_format, # Pass the desired format
            )

            # Determine the number of bbox classes based on label settings.
            if use_improved_labels:
                num_bbox_classes = len(VINBIG_BBOX_NAMES__MODIFIED)
                bbox_class_names = VINBIG_BBOX_NAMES__MODIFIED
            else:
                num_bbox_classes = len(VINBIG_BBOX_NAMES)
                bbox_class_names = VINBIG_BBOX_NAMES
            self.bbox_class_names = bbox_class_names
            logger.info(f"Expecting {num_bbox_classes} bbox classes.")

            # Combine train/test bbox dictionaries, handling images with no bboxes.
            image_id_2_bboxes_combined: BboxDataType = {}
            for img_id in self.image_ids:
                if img_id in train_image_id_2_bboxes:
                    image_id_2_bboxes_combined[img_id] = train_image_id_2_bboxes[img_id]
                elif img_id in test_image_id_2_bboxes:
                    image_id_2_bboxes_combined[img_id] = test_image_id_2_bboxes[img_id]
                else:
                    # Assign empty lists for images without any bounding boxes
                    image_id_2_bboxes_combined[img_id] = ([], [])

            self.image_id_2_bboxes = image_id_2_bboxes_combined
            # Create a list of bbox data aligned with self.image_ids
            self.bboxes = [
                self.image_id_2_bboxes[img_id] for img_id in self.image_ids
            ]

            # --- Bounding Box Sanity Checks ---
            logger.info("Performing bounding box sanity checks...")
            for i, (bbox_list, class_id_list) in enumerate(self.bboxes):
                img_id = self.image_ids[i]
                # Check structure: tuple of two lists
                assert isinstance(bbox_list, list) and isinstance(class_id_list, list), \
                    f"Image {img_id}: Expected tuple of two lists, got {type(bbox_list)}, {type(class_id_list)}"
                # Check lengths match
                assert len(bbox_list) == len(class_id_list), \
                    f"Image {img_id}: Mismatch between bbox count ({len(bbox_list)}) and class ID count ({len(class_id_list)})"

                # Check individual bounding boxes
                for j, bbox_coords in enumerate(bbox_list):
                    assert len(bbox_coords) == 4, \
                        f"Image {img_id}, Bbox {j}: Expected 4 coordinates, got {len(bbox_coords)}"
                    # Check coordinate range (assuming normalization to [0, 1])
                    for coord in bbox_coords:
                        assert 0 <= coord <= 1, \
                            f"Image {img_id}, Bbox {j}: Coordinate {coord} out of range [0, 1]"
                    # Check bbox format
                    if bbox_format == "xyxy":
                        assert bbox_coords[0] < bbox_coords[2] and bbox_coords[1] < bbox_coords[3], \
                            f"Image {img_id}, Bbox {j}: Invalid coordinates for xyxy format: {bbox_coords}"
                    elif bbox_format == "cxcywh":
                        assert bbox_coords[0] - bbox_coords[2] / 2 >= 0 and \
                               bbox_coords[0] + bbox_coords[2] / 2 <= 1 and \
                               bbox_coords[1] - bbox_coords[3] / 2 >= 0 and \
                               bbox_coords[1] + bbox_coords[3] / 2 <= 1, \
                            f"Image {img_id}, Bbox {j}: Invalid coordinates for cxcywh format: {bbox_coords}"
                    else:
                        raise ValueError(f"Unknown bbox format: {bbox_format}")

                # Check class IDs
                for j, class_id in enumerate(class_id_list):
                    assert 0 <= class_id < num_bbox_classes, \
                        f"Image {img_id}, Bbox {j}: Class ID {class_id} out of range [0, {num_bbox_classes - 1}]"
            logger.info(f"Sanity checks complete.")

            logger.info(
                f"Loaded bounding box data for {len(self.image_id_2_bboxes)} images."
            )
            if verbose:
                # Print info for 5 random images with bounding boxes
                indices_with_bboxes = [
                    i for i, (b, c) in enumerate(self.bboxes) if len(b) > 0
                ]
                if indices_with_bboxes:
                    logger.info("  Sample bounding box data (up to 5 random images):")
                    sample_indices = random.sample(
                        indices_with_bboxes, min(5, len(indices_with_bboxes))
                    )
                    for i in sample_indices:
                        num_boxes = len(self.bboxes[i][0])
                        logger.info(
                            f"    Image ID {self.image_ids[i]} (Index {i}): "
                            f"{num_boxes} boxes. "
                            f"First box (if any): {self.bboxes[i][0][0] if num_boxes > 0 else 'N/A'}, "
                            f"Class ID: {self.bboxes[i][1][0] if num_boxes > 0 else 'N/A'}"
                        )
                else:
                    logger.warning("No images with bounding boxes found to sample.")
        else:
            logger.info("Bounding box loading skipped.")

        logger.info("VinBigTrainerBase initialization complete.")


class VinBig_VQA_Trainer(VinBigTrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers, tokenizer,
                include_image=True,
                use_precomputed_visual_features=False,
                precomputed_visual_features_path=None,
                training_data_mode=VinBigTrainingMode.ALL,
                use_validation=True,
                use_merged_findings=False,
                findings_remapper=None,
                n_findings=None,
        ):
        super().__init__(
            use_merged_findings=use_merged_findings,
            findings_remapper=findings_remapper,
            n_findings=n_findings,
        )
        
        self.transform = transform
        self.include_image = include_image
        self.use_precomputed_visual_features = use_precomputed_visual_features
        self.tokenizer = tokenizer
        self.training_data_mode = training_data_mode
        self.use_validation = use_validation

        if use_precomputed_visual_features:
            assert precomputed_visual_features_path is not None
            features, idx2visfeatidx = load_precomputed_visual_features(
                precomputed_visual_features_path,
                self.image_paths,
            )
            self.precomputed_visual_features = features
            self.idx2visfeatidx = idx2visfeatidx
        else:
            self.precomputed_visual_features = None
            self.idx2visfeatidx = None
        
        print('Generating train dataset and dataloader')
        if training_data_mode == VinBigTrainingMode.TRAIN:
            train_indices = self.train_indices
        elif training_data_mode == VinBigTrainingMode.TEST:
            train_indices = self.test_indices
        elif training_data_mode == VinBigTrainingMode.ALL:
            train_indices = self.train_indices + self.test_indices
        else: assert False, f'Unknown training_data_mode = {training_data_mode}'         
        
        self.train_dataset, self.train_dataloader = self._generate_dataset_and_dataloader(
            train_indices, batch_size, collate_batch_fn, num_workers, infinite=True, min_pos_to_include=10,
        )

        if use_validation:
            print('Generating val dataset and dataloader')
            self.val_dataset, self.val_dataloader = self._generate_dataset_and_dataloader(
                self.test_indices, batch_size, collate_batch_fn, num_workers, infinite=False, n_samples=20,
            )

    def _generate_dataset_and_dataloader(self, indices, batch_size, collate_batch_fn, num_workers,
                                         infinite=True, n_samples=None, min_pos_to_include=0):

        return self._create_label_based_dataset_and_dataloader(
            indices=indices,
            labels=self.labels,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_batch_fn=collate_batch_fn,
            infinite=infinite,
            n_samples=n_samples,
            min_pos_to_include=min_pos_to_include,
            log_weighting=True,
        )
    
    def _create_vqa_dataset(self, q, a, indices, infinite):
        labels = self.finding_labels if self.use_merged_findings else self.labels
        return VinBigVQADataset(
            self.image_paths, self.transform, labels,
            question=q, answer=a, indices=indices,
            include_image=self.include_image,
            use_precomputed_visual_features=self.use_precomputed_visual_features,
            precomputed_visual_features=self.precomputed_visual_features,
            idx2visfeatidx = self.idx2visfeatidx,
            infinite=infinite,
        )

class VinBig_VisualModuleTrainer(VinBigTrainerBase):
    def __init__(self, collate_batch_fn, num_workers,
                training_data_mode=VinBigTrainingMode.ALL,
                use_training_set=True, use_validation_set=True,
                train_batch_size=None, val_batch_size=None,
                use_training_indices_for_validation=False,
                train_image_transform=None, val_image_transform=None,
                use_merged_findings=False, findings_remapper=None, n_findings=None,
                use_bounding_boxes=False, data_augmentation_enabled=False,
                use_yolov8=False, use_yolov11=False, class_id_offset=0,
                use_modified_labels=False,
        ):
        super().__init__(
            use_merged_findings=use_merged_findings,
            findings_remapper=findings_remapper,
            n_findings=n_findings,
            load_bouding_boxes=use_bounding_boxes,
            class_id_offset=class_id_offset,
            use_improved_labels=use_modified_labels,
        )

        if use_training_set:
            assert train_image_transform is not None
            assert train_batch_size is not None
        if use_validation_set:
            assert val_image_transform is not None
            assert val_batch_size is not None
        
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.training_data_mode = training_data_mode
        self.use_bounding_boxes = use_bounding_boxes
        self.data_augmentation_enabled = data_augmentation_enabled
        self.use_training_set = use_training_set
        self.use_validation_set = use_validation_set
        self.use_yolov8 = use_yolov8
        self.use_yolov11 = use_yolov11
        
        if training_data_mode == VinBigTrainingMode.TRAIN:
            train_indices = self.train_indices
        elif training_data_mode == VinBigTrainingMode.TEST:
            train_indices = self.test_indices
        elif training_data_mode == VinBigTrainingMode.ALL:
            train_indices = self.train_indices + self.test_indices
        else: assert False, f'Unknown training_data_mode = {training_data_mode}'
        
        if use_training_set:
            print('Generating train dataset and dataloader')
            self.train_dataset, self.train_dataloader = self._create_label_based_dataset_and_dataloader(
                indices=train_indices,
                labels=self.labels,
                label_names=self.label_names,
                templates=self.templates,
                batch_size=train_batch_size,
                num_workers=num_workers,
                collate_batch_fn=lambda batch: collate_batch_fn(batch, training_mode=True),
                infinite=True,
                min_pos_to_include=0,
                log_weighting=True,
                include_qa=False,
                create_dataset_kwargs={
                    'transform': self.train_image_transform,
                    'data_augmentation_enabled': self.data_augmentation_enabled,
                },
            )
            print(f'len(train_indices) = {len(train_indices)}')
        
        if use_validation_set:
            print('Generating val dataset and dataloader')
            if use_training_indices_for_validation:
                test_indices = self.train_indices
            else:
                test_indices = self.test_indices
            self.val_dataset = self._create_visual_dataset(test_indices, infinite=False,
                                                           transform=self.val_image_transform,
                                                           data_augmentation_enabled=False)
            self.val_dataloader = DataLoader(self.val_dataset,
                                             batch_size=val_batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=lambda batch: collate_batch_fn(batch, training_mode=False),
                                             pin_memory=True)
            print(f'len(test_indices) = {len(test_indices)}')
    
    def _create_visual_dataset(self, indices, infinite, transform, data_augmentation_enabled):
        labels = self.finding_labels if self.use_merged_findings else self.labels
        return VinBigVisualDataset(
            self.image_paths, transform, labels,
            indices=indices,
            infinite=infinite,
            use_bounding_boxes=self.use_bounding_boxes,
            bboxes=self.bboxes,
            data_augmentation_enabled=data_augmentation_enabled,
            use_yolov8=self.use_yolov8,
            use_yolov11=self.use_yolov11,
        )

class VinBigVQADataset(Dataset):
    
    def __init__(self, image_paths, transform, labels, question, answer, indices,
                include_image=True,
                suffle_indices=True,
                # precomputed visual features
                use_precomputed_visual_features=False,
                precomputed_visual_features=None,
                idx2visfeatidx=None,
                # infinite mode
                infinite = False,
        ):
        self.images = image_paths
        self.transform = transform
        self.labels = labels
        self.infinite = infinite
        self.question = question
        self.answer = answer
        self.indices = indices
        self.include_image = include_image
        
        if suffle_indices: np.random.shuffle(self.indices)
        self._len = INFINITE_DATASET_LENGTH if infinite else len(self.indices)

        if include_image:
            assert image_paths is not None

        self.use_precomputed_visual_features = use_precomputed_visual_features
        if use_precomputed_visual_features:
            assert precomputed_visual_features is not None
            assert idx2visfeatidx is not None
            self.precomputed_visual_features = precomputed_visual_features
            self.idx2visfeatidx = idx2visfeatidx
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        indices = self.indices
        if self.infinite:
            i %= len(indices)
        idx = indices[i]
        output = dict(
            idx=idx,
            l=self.labels[idx],
            q=self.question,
            a=self.answer,
        )
        if self.include_image:
            output['i'] = self.transform(Image.open(self.images[idx]).convert('RGB'))
        if self.use_precomputed_visual_features:
            output['vf'] = self.precomputed_visual_features[self.idx2visfeatidx[idx]]
        return output

class VinBigVisualDataset(Dataset):
    
    def __init__(
            self, image_paths, image_transform, labels, indices, suffle_indices=True, infinite=False,
            use_bounding_boxes=False, bboxes=None, use_yolov8=False, use_yolov11=False, data_augmentation_enabled=False,
        ):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.labels = labels
        self.infinite = infinite
        self.indices = indices
        self.use_bounding_boxes = use_bounding_boxes
        self.bboxes = bboxes
        self.use_yolov8 = use_yolov8
        self.use_yolov11 = use_yolov11
        self.data_augmentation_enabled = data_augmentation_enabled
        assert use_bounding_boxes == (use_yolov8 or use_yolov11), 'use_bounding_boxes must be True if use_yolov8 or use_yolov11 is True'
        if use_bounding_boxes:
            assert bboxes is not None
            if data_augmentation_enabled:
                self.albumentation_adapter = VinBigAlbumentationAdapter()
        
        if suffle_indices: np.random.shuffle(self.indices)
        self._len = INFINITE_DATASET_LENGTH if infinite else len(self.indices)
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        indices = self.indices
        if self.infinite:
            i %= len(indices)
        idx = indices[i]
        image_path = self.image_paths[idx]
        output = dict(
            idx=idx,
            l=self.labels[idx],
        )
        if self.use_bounding_boxes:
            bboxes, classes = self.bboxes[idx]
            if self.data_augmentation_enabled:
                tmp = self.image_transform(
                    image_path=image_path,
                    bboxes=bboxes,
                    classes=classes,
                    albumentation_adapter=self.albumentation_adapter,
                    return_image_size=True,
                )
                image, bboxes, classes, image_size_before, image_size_after = tmp
            else:
                tmp = self.image_transform(image_path, return_image_size=True)
                image, image_size_before, image_size_after = tmp
            
            # We need to adapt the output a little bit to make it compatible with YOLOV8 and YOLOV11
            output['im_file'] = image_path
            output['ori_shape'] = image_size_before
            output['resized_shape'] = image_size_after
            output['i'] = image
            output['bboxes'] = bboxes
            output['classes'] = classes
        else:
            output['i'] = self.image_transform(image_path)
        return output

class VinBig_MAE_Trainer(MAETrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers):

        self.transform = transform
        
        df_labels_train = pd.read_csv(VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH)
        df_labels_test = pd.read_csv(VINBIG_IMAGE_LABELS_TEST_CSV_PATH)

        assert len(df_labels_train) == N_IMAGES_TRAIN * 3
        assert len(df_labels_test) == N_IMAGES_TEST

        # Images ids        
        image_ids = [None] * (N_IMAGES_TRAIN + N_IMAGES_TEST)

        # train image ids
        train_image_ids = df_labels_train['image_id']
        for i in range(N_IMAGES_TRAIN):
            image_ids[i] = train_image_ids[i * 3]
            for j in range(1, 3):
                assert train_image_ids[i * 3 + j] == image_ids[i]
            assert image_ids[i] != image_ids[i-1]
        
        # test image ids
        test_images_ids = df_labels_test['image_id']
        for i in range(N_IMAGES_TEST):
            image_ids[N_IMAGES_TRAIN + i] = test_images_ids[i]

        # Image paths
        self.image_paths = [get_medium_size_image_path(img_id) for img_id in image_ids]

        # Labels
        labels = np.empty((N_IMAGES_TRAIN + N_IMAGES_TEST, len(VINBIG_LABELS)), dtype=np.int8)
        
        # train labels
        tmp = VINBIG_LABELS[:]
        tmp[tmp.index('Other disease')] = 'Other diseases' # HACK
        train_labels = df_labels_train[tmp].values
        for i in range(N_IMAGES_TRAIN):
            labels[i] = _merge_labels(
                train_labels[3 * i],
                train_labels[3 * i + 1],
                train_labels[3 * i + 2]
            )
        
        # test labels
        labels[N_IMAGES_TRAIN:] = df_labels_test[VINBIG_LABELS].values

        train_indices = list(range(len(labels)))
        labels_getter = lambda i: labels[i]
        super().__init__(train_indices, None, None, list(range(len(VINBIG_LABELS))),
                         labels_getter, batch_size, collate_batch_fn, num_workers, use_validation_set=False)
    
    def _create_mae_dataset(self, indices, shuffle=True, infinite=False):
        return BasicImageDataset(self.image_paths, self.transform, indices, shuffle, infinite)


# --- Dataset 1: Phrase Classification Only ---

class VinBig_PhraseClassificationDataset(Dataset):
    """
    Dataset for VinBig phrase classification task.
    Returns image, all phrase embeddings, and multi-label classification targets.
    """
    def __init__(self,
                 indices: List[int],
                 image_paths: List[str],
                 image_transform: Callable,
                 phrase_embeddings: np.ndarray,
                 phrase_classification_labels: np.ndarray,
                 infinite: bool = False,
                 shuffle_indices: bool = False):
        self.indices = list(indices) # Ensure it's a mutable list
        self.image_paths = image_paths
        self.image_transform = image_transform
        # Phrase embeddings are shared across all images
        self.phrase_embeddings = torch.tensor(phrase_embeddings, dtype=torch.float32)
        self.phrase_classification_labels = phrase_classification_labels
        self.infinite = infinite

        if shuffle_indices:
            random.shuffle(self.indices)

        if infinite:
            # Define a large length for infinite datasets
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if self.infinite:
            # Use modulo for infinite looping
            actual_idx = self.indices[i % len(self.indices)]
        else:
            actual_idx = self.indices[i]

        image_path = self.image_paths[actual_idx]
        # Apply image transformation (assuming it returns a tensor)
        # Modify this if your transform returns a dict (e.g., {'pixel_values': ...})
        image = self.image_transform(image_path)['pixel_values']

        # Get the classification labels for this image
        labels = torch.tensor(
            self.phrase_classification_labels[actual_idx], dtype=torch.float32
        )

        return {
            # 'i': Image tensor, e.g., (C, H, W)
            'i': image,
            # 'pe': All phrase embeddings, e.g., (Num Phrases, Embedding Size)
            'pe': self.phrase_embeddings,
            # 'pcl': Phrase classification labels (multi-label), e.g., (Num Phrases,)
            'pcl': labels,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for phrase classification."""
        images = torch.stack([item['i'] for item in batch])
        phrase_embeddings = torch.stack([item['pe'] for item in batch])
        labels = torch.stack([item['pcl'] for item in batch])

        return {
            'i': images,
            'pe': phrase_embeddings,
            'pcl': labels,
            'dataset_name': 'vinbig', # Used downstream in multi-dataset training
        }

# --- Dataset 2: Phrase Grounding Only ---

class VinBig_PhraseGroundingDataset(Dataset):
    """
    Dataset for VinBig phrase grounding task.
    Handles positive (image, phrase, bboxes) tuples.
    """
    def __init__(self,
                 image_paths: List[str],
                 image_transform: Callable,
                 phrase_idxs: List[int],
                 phrase_embeddings: np.ndarray,
                 phrase_bboxes: List[List[List[float]]], # List of bbox lists
                 phrase_prob_masks: List[np.ndarray],
                 indices: List[int],
                 feature_map_size: Tuple[int, int],
                 data_augmentation_enabled: bool = False,
                 shuffle_indices: bool = False,
                 for_training: bool = True,
                 bbox_format: str = 'xyxy',
                 infinite: bool = False):

        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_idxs = phrase_idxs
        self.phrase_embeddings = phrase_embeddings
        self.phrase_bboxes = phrase_bboxes
        self.phrase_prob_masks = phrase_prob_masks
        self.indices = list(indices) # Ensure mutable list
        self.feature_map_size = feature_map_size
        self.data_augmentation_enabled = data_augmentation_enabled
        self.for_training = for_training
        self.bbox_format = bbox_format
        self.infinite = infinite

        if shuffle_indices:
            random.shuffle(self.indices)

        if for_training:
            assert self.feature_map_size is not None
            assert self.phrase_prob_masks is not None

        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int) -> dict:
        if self.infinite:
            data_index = self.indices[i % len(self.indices)]
        else:
            data_index = self.indices[i]

        image_path = self.image_paths[data_index]
        phrase_idx = self.phrase_idxs[data_index]
        phrase_bboxes = self.phrase_bboxes[data_index] # BBoxes for this specific phrase
        phrase_embedding = torch.tensor(self.phrase_embeddings[phrase_idx], dtype=torch.float32)

        if self.for_training:
            prob_mask = self.phrase_prob_masks[data_index]
            if self.data_augmentation_enabled:
                out = self.image_transform(
                    image_path=image_path,
                    bboxes=phrase_bboxes,
                    bbox_labels=[0] * len(phrase_bboxes), # Dummy labels needed by Albumentations
                    masks=[prob_mask], # Pass mask for augmentation
                )
                # Extract results
                image = out['pixel_values']
                prob_mask = out['masks'][0]
                phrase_bboxes = out['bboxes']
            else:
                # Apply non-augmenting transform
                image = self.image_transform(image_path)['pixel_values']

            # Convert bboxes and mask to target tensors
            target_coords, target_presence, target_prob_mask = convert_bboxes_into_target_tensors(
                bboxes=phrase_bboxes,
                probabilistic_mask=prob_mask,
                feature_map_size=self.feature_map_size,
                bbox_format=self.bbox_format,
            )
            return {
                'i': image,
                'pe': phrase_embedding, # Single phrase embedding
                'tbc': target_coords,
                'tbp': target_presence,
                'tpm': target_prob_mask,
            }
        else: # For validation/testing
            # Apply non-augmenting transform
            image = self.image_transform(image_path)['pixel_values']
            return {
                'i': image,
                'pe': phrase_embedding, # Single phrase embedding
                'bboxes': phrase_bboxes, # Return original (or transformed) bboxes
            }

    def collate_fn(self, batch: List[dict]) -> dict:
        """Custom collate function for phrase grounding."""
        images = torch.stack([item['i'] for item in batch])
        phrase_embeddings = torch.stack([item['pe'] for item in batch])

        if self.for_training:
            target_coords = torch.stack([item['tbc'] for item in batch])
            target_presence = torch.stack([item['tbp'] for item in batch])
            target_prob_masks = torch.stack([item['tpm'] for item in batch])
            collated_batch = {
                'i': images,
                'pe': phrase_embeddings,
                'tbc': target_coords,
                'tbp': target_presence,
                'tpm': target_prob_masks,
            }
        else:
            # For validation/testing, bboxes might be lists of varying length
            bboxes = [item['bboxes'] for item in batch]
            collated_batch = {
                'i': images,
                'pe': phrase_embeddings,
                'bboxes': bboxes,
            }
        collated_batch['dataset_name'] = 'vinbig' # Used downstream in multi-dataset training
        return collated_batch


# --- Dataset 3: Phrase Grounding and Classification Combined ---

class VinBig_PhraseGroundingAndClassificationDataset(Dataset):
    """
    Dataset for combined VinBig phrase classification and grounding.
    - Training: Returns image, all phrase embeddings, classification labels,
      and grounding targets (tbc, tbp, tpm) ONLY for phrases with supervision,
      plus indices mapping these targets back to the full output tensor.
    - Inference: Returns image, all phrase embeddings, classification labels,
      and the actual bounding boxes with their corresponding phrase indices
      for phrases with supervision.
    """
    # TODO: review and test this dataset class
    def __init__(self,
                 indices: List[int],
                 image_paths: List[str],
                 image_transform: Callable, # Should handle augmentation of img, bboxes, masks
                 phrase_embeddings: np.ndarray,
                 phrase_classification_labels: np.ndarray,
                 # Lookup: image_idx -> phrase_idx -> {bboxes, prob_mask}
                 image_idx_to_grounding_info: Dict[int, Dict[int, Dict]],
                 feature_map_size: Tuple[int, int], # Needed for target generation (training)
                 bbox_format: str = 'xyxy',         # Needed for target generation (training)
                 infinite: bool = False,
                 shuffle_indices: bool = False,
                 data_augmentation_enabled: bool = False, # Control augmentation
                 for_training: bool = True, # <<< ADDED
                 ):
        self.indices = list(indices)
        self.image_paths = image_paths
        self.image_transform = image_transform # This callable handles augmentation logic internally
        self.phrase_embeddings = torch.tensor(phrase_embeddings, dtype=torch.float32)
        self.phrase_classification_labels = phrase_classification_labels
        self.image_idx_to_grounding_info = image_idx_to_grounding_info
        self.feature_map_size = feature_map_size
        self.bbox_format = bbox_format
        self.infinite = infinite
        self.data_augmentation_enabled = data_augmentation_enabled
        self.for_training = for_training # <<< STORED

        if for_training:
             assert self.feature_map_size is not None, "feature_map_size needed for training"
             assert self.bbox_format is not None, "bbox_format needed for training"

        if shuffle_indices:
            random.shuffle(self.indices)

        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int) -> Dict:
        if self.infinite:
            actual_idx = self.indices[i % len(self.indices)]
        else:
            actual_idx = self.indices[i]

        image_path = self.image_paths[actual_idx]
        labels = torch.tensor(self.phrase_classification_labels[actual_idx], dtype=torch.float32)

        # Get grounding info for this image
        grounding_info = self.image_idx_to_grounding_info.get(actual_idx, {})
        grounding_phrase_indices = sorted(list(grounding_info.keys())) # Original phrase indices (0-27)

        # --- Prepare FLAT lists for augmentation ---
        flat_bboxes_to_augment = []
        flat_bbox_labels_to_augment = [] # Will store the original phrase index (0-27)
        masks_to_augment = []
        bboxes_per_phrase_idx = {} # Track counts for reconstructing later if needed

        for pidx in grounding_phrase_indices:
            phrase_data = grounding_info[pidx]
            bboxes = phrase_data["bboxes"]
            mask = phrase_data["prob_mask"]
            count = 0
            for bbox in bboxes:
                flat_bboxes_to_augment.append(bbox)
                flat_bbox_labels_to_augment.append(pidx)
                count += 1
            masks_to_augment.append(mask) # Keep masks per phrase for now
            bboxes_per_phrase_idx[pidx] = count

        # --- Apply transform/augmentation ---
        augmented_image = None
        augmented_flat_bboxes = flat_bboxes_to_augment
        augmented_flat_labels = flat_bbox_labels_to_augment
        augmented_masks = masks_to_augment # List of masks, one per phrase index

        if self.data_augmentation_enabled and len(grounding_phrase_indices) > 0:
            # Pass FLAT lists to the transform. Assumes transform handles this format.
            # The transform should ideally also augment the masks if needed for training.
            transform_output = self.image_transform(
                image_path=image_path,
                bboxes=flat_bboxes_to_augment,   # Flat list of bboxes
                bbox_labels=flat_bbox_labels_to_augment, # Flat list of phrase indices
                masks=masks_to_augment,          # List of masks (one per phrase)
                # Add other args your transform might need, e.g., adapter
            )
            augmented_image = transform_output['pixel_values']
            augmented_flat_bboxes = transform_output['bboxes'] # Should return flat list
            augmented_flat_labels = transform_output['bbox_labels'] # Should return flat list
            augmented_masks = transform_output['masks'] # Should return list of masks
            # Sanity check (optional): Ensure label list length matches bbox list length
            assert len(augmented_flat_bboxes) == len(augmented_flat_labels)
        else:
            # No augmentation or no grounding phrases
            # Still need to apply basic image transform
            transform_output = self.image_transform(image_path=image_path)
            augmented_image = transform_output['pixel_values']
            # Use original flat lists if no augmentation applied
            augmented_flat_bboxes = flat_bboxes_to_augment
            augmented_flat_labels = flat_bbox_labels_to_augment
            augmented_masks = masks_to_augment


        # --- Prepare output based on mode ---
        output_dict = {
            'i': augmented_image,
            'pe': self.phrase_embeddings,
            'pcl': labels,
            'gpi': grounding_phrase_indices, # Still useful to know which phrases *could* have grounding
        }

        if self.for_training:
            # --- Generate target tensors for training ---
            tbc_list = []
            tbp_list = []
            tpm_list = []

            # We need to reconstruct per-phrase bboxes/masks from the flat augmented lists
            current_bbox_idx = 0
            current_mask_idx = 0
            for pidx in grounding_phrase_indices: # Iterate through the *original* sorted phrase indices
                num_bboxes_for_this_phrase = bboxes_per_phrase_idx[pidx]
                # Slice the flat augmented bboxes belonging to this phrase index
                # This assumes the transform preserves the *order* relative to the input flat list
                phrase_bboxes = augmented_flat_bboxes[current_bbox_idx : current_bbox_idx + num_bboxes_for_this_phrase]
                # Get the corresponding augmented mask
                # This assumes masks are returned one per original phrase index
                phrase_mask = augmented_masks[current_mask_idx]

                if len(phrase_bboxes) > 0: # Only generate targets if bboxes survived augmentation
                    tbc, tbp, tpm = convert_bboxes_into_target_tensors(
                        bboxes=phrase_bboxes,
                        probabilistic_mask=phrase_mask,
                        feature_map_size=self.feature_map_size,
                        bbox_format=self.bbox_format,
                    )
                    tbc_list.append(tbc)
                    tbp_list.append(tbp)
                    tpm_list.append(tpm)
                # else:
                    # Handle case where augmentation removed all bboxes for a phrase?
                    # Option 1: Skip target generation (lists will be shorter than gpi) - Requires careful collate
                    # Option 2: Generate empty/zero targets (might be simpler) - Let's assume this for now if needed
                    # Current convert_bboxes_into_target_tensors likely handles empty input gracefully.

                current_bbox_idx += num_bboxes_for_this_phrase
                current_mask_idx += 1

            # Add target lists to output
            output_dict['gt_tbc_list'] = tbc_list
            output_dict['gt_tbp_list'] = tbp_list
            output_dict['gt_tpm_list'] = tpm_list

        else:
            # --- Prepare output for inference ---
            # Return the flat lists of augmented bboxes and their corresponding phrase indices
            output_dict['bboxes'] = augmented_flat_bboxes # Flat list for this image
            output_dict['bbox_labels'] = augmented_flat_labels # Flat list of phrase indices (0-27)

        return output_dict


    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate function for combined classification and grounding.
        Handles both training and inference modes.
        """
        # Check mode based on keys in the first item
        is_training = 'gt_tbc_list' in batch[0]

        # Common batching
        images = torch.stack([item['i'] for item in batch])
        phrase_embeddings = batch[0]['pe'] # Shared
        labels = torch.stack([item['pcl'] for item in batch])
        num_phrases = phrase_embeddings.shape[0]

        collated_batch = {
            'i': images,
            'pe': phrase_embeddings,
            'pcl': labels,
            'dataset_name': 'vinbig', # Added dataset name
        }

        if is_training:
            # --- Collate for Training ---
            batched_grounding_indices = []
            batch_indices_for_grounding = []
            tbc_to_cat = []
            tbp_to_cat = []
            tpm_to_cat = []

            for batch_idx, item in enumerate(batch):
                # Use the *actual* generated target lists, their length might be
                # less than len(item['gpi']) if augmentation removed all bboxes for a phrase.
                num_targets_generated = len(item['gt_tbc_list'])
                original_phrase_indices_with_targets = item['gpi'][:num_targets_generated] # Assuming order is preserved

                if num_targets_generated > 0:
                    tbc_to_cat.extend(item['gt_tbc_list']) # Extend with list of tensors
                    tbp_to_cat.extend(item['gt_tbp_list'])
                    tpm_to_cat.extend(item['gt_tpm_list'])

                    for phrase_idx in original_phrase_indices_with_targets:
                        batched_grounding_indices.append(batch_idx * num_phrases + phrase_idx)
                        batch_indices_for_grounding.append(batch_idx)

            # Convert index lists to tensors
            gpi_flat_tensor = torch.tensor(batched_grounding_indices, dtype=torch.long)
            gpi_batch_idx_tensor = torch.tensor(batch_indices_for_grounding, dtype=torch.long)

            # Concatenate target tensors across the batch (stack the list of tensors)
            if tbc_to_cat:
                collated_gt_tbc = torch.stack(tbc_to_cat, dim=0)
                collated_gt_tbp = torch.stack(tbp_to_cat, dim=0)
                collated_gt_tpm = torch.stack(tpm_to_cat, dim=0)
            else:
                # Create empty tensors with correct trailing dimensions
                # Need feature_map_size info here - maybe infer from a non-empty item if possible
                # Or pass feature_map_size to collate_fn? Let's try inferring.
                fm_h, fm_w = batch[0]['gt_tbc_list'][0].shape[-2:] if batch and batch[0]['gt_tbc_list'] else (0, 4)
                fm_hw = batch[0]['gt_tbp_list'][0].shape[-1] if batch and batch[0]['gt_tbp_list'] else 0
                collated_gt_tbc = torch.empty((0, fm_h, fm_w), dtype=torch.float32)
                collated_gt_tbp = torch.empty((0, fm_hw), dtype=torch.bool) # Check dtype
                collated_gt_tpm = torch.empty((0, fm_hw), dtype=torch.float32)


            assert len(gpi_flat_tensor) == collated_gt_tbc.shape[0], \
                f"Training Collate Mismatch: {len(gpi_flat_tensor)} indices vs {collated_gt_tbc.shape[0]} target tensors"

            collated_batch.update({
                'gpi_flat': gpi_flat_tensor,
                'gpi_batch_idx': gpi_batch_idx_tensor,
                'gt_tbc': collated_gt_tbc,
                'gt_tbp': collated_gt_tbp,
                'gt_tpm': collated_gt_tpm,
                'task_mode': 'classification_and_grounding_train', # More specific mode
            })

        else:
            # --- Collate for Inference ---
            batched_bboxes = []
            batched_bbox_labels = []

            for item in batch:
                # Append the flat list of bboxes for the image
                batched_bboxes.append(item['bboxes'])
                # Append the flat list of corresponding phrase indices
                batched_bbox_labels.append(item['bbox_labels'])

            collated_batch.update({
                'bboxes': batched_bboxes,           # List[List[bbox]] - Outer list is batch
                'bbox_labels': batched_bbox_labels, # List[List[int]] - Outer list is batch
                'task_mode': 'classification_and_grounding_eval', # More specific mode
            })

        return collated_batch


class VinBigPhraseTrainer(VinBigTrainerBase):
    """
    Extends VinBigTrainerBase to handle phrase-based tasks like classification
    and grounding using precomputed phrase embeddings.

    This class orchestrates the loading of data (images, labels, bounding boxes),
    phrase embeddings, and sets up PyTorch DataLoaders tailored to specific
    tasks (classification, grounding, or both). It handles label/phrase
    reordering and dataset splitting based on configuration.

    Key Responsibilities:
    - Initializes the base class to load necessary data (including bboxes).
    - Loads phrase embeddings and corresponding phrases.
    - Reorders labels, phrases, and embeddings for consistency (bbox classes first).
    - Determines training and validation image indices based on configuration.
    - Sets up task-specific datasets and dataloaders (`train_dataloader`, `val_dataloader`).
    """

    def __init__(
        self,
        task_mode: str,  # Specify the task
        mask_height: int,  # Needed for grounding mask target resolution
        mask_width: int,  # Needed for grounding mask target resolution
        phrase_embeddings_filepath: str, # Path to precomputed embeddings/phrases
        max_images_per_batch: int, # Max images per batch for DataLoaders
        val_batch_size_factor: float = 1.0, # Multiplier for validation batch size
        training_data_mode: str = "train", # Dataset split(s) to use for training
        use_training_set: bool = True, # Whether to create a training dataloader
        use_validation_set: bool = True, # Whether to create a validation dataloader
        use_training_indices_for_validation: bool = False, # Use train split for validation?
        num_train_workers: Optional[int] = None, # Num workers for train DataLoader
        num_val_workers: Optional[int] = None, # Num workers for val DataLoader
        train_image_transform: Optional[Callable] = None, # Img transform for training
        val_image_transform: Optional[Callable] = None, # Img transform for validation
        data_augmentation_enabled: bool = False, # Enable/disable augmentation (grounding)
        replace_phrase_embeddings_with_random_vectors: bool = False, # Debug: use random embeddings
        use_modified_labels: bool = False, # Use modified label set?
        bbox_format: Literal["xyxy", "cxcywh"] = "xyxy", # Bbox format used/expected
    ):
        """
        Initializes the VinBigPhraseTrainer.

        Args:
            task_mode: The primary task ('classification', 'grounding', 'classification_and_grounding').
            mask_height: Target height for grounding masks.
            mask_width: Target width for grounding masks.
            phrase_embeddings_filepath: Path to the .pkl file containing
                'phrase_embeddings' (np.ndarray) and 'phrases' (List[str]).
            max_images_per_batch: The maximum number of images to include in a
                training batch. Validation batch size is derived from this.
            val_batch_size_factor: Multiplier applied to `max_images_per_batch`
                to determine the validation batch size. Defaults to 1.0.
            training_data_mode: Specifies which dataset split(s) to use for
                constructing the training set ('all', 'train_only', 'test_only').
                Defaults to 'train'.
            use_training_set: If True, sets up the training dataset and dataloader.
                Defaults to True.
            use_validation_set: If True, sets up the validation dataset and dataloader.
                Defaults to True.
            use_training_indices_for_validation: If True and `use_validation_set`
                is True, the validation set will use indices from the training split
                instead of the test split. Defaults to False.
            num_train_workers: Number of worker processes for the training DataLoader.
                Required if `use_training_set` is True. Defaults to None.
            num_val_workers: Number of worker processes for the validation DataLoader.
                Required if `use_validation_set` is True. Defaults to None.
            train_image_transform: Callable transformation function applied to
                training images. Required if `use_training_set` is True. Defaults to None.
            val_image_transform: Callable transformation function applied to
                validation images. Required if `use_validation_set` is True. Defaults to None.
            data_augmentation_enabled: If True, enables data augmentation within
                the grounding dataset (if applicable). Defaults to False.
            replace_phrase_embeddings_with_random_vectors: If True, replaces loaded
                phrase embeddings with normalized random vectors (for debugging/ablations).
                Defaults to False.
            use_modified_labels: If True, instructs the base class
                to load the modified/improved label set and corresponding bbox classes.
                Defaults to False.
            bbox_format: Specifies the format ('xyxy' or 'cxcywh') of the bounding
                boxes loaded by the base class and used for generating grounding targets.
                Defaults to 'xyxy'.
        """
        logger.info(f"Initializing VinBigPhraseTrainer (Task: {task_mode})")
        # --- Initialize Base Class ---
        # Force loading of bounding boxes as they are needed for grounding tasks
        # and potentially for label reordering even in classification.
        # Pass label modification flag and bbox format down.
        super().__init__(
            load_bounding_boxes=True, # Always load bboxes for phrase tasks
            use_improved_labels=use_modified_labels,
            bbox_format=bbox_format,
        )
        logger.info(f"Base trainer initialized. Bbox format: {bbox_format}")

        # --- Argument Validation ---
        assert use_training_set or use_validation_set, \
            "At least one of use_training_set or use_validation_set must be True"
        assert task_mode in VinBigPhraseTaskMode.get_choices(), f"Invalid task_mode: {task_mode}"

        if use_training_set:
            assert train_image_transform is not None, \
                "train_image_transform is required when use_training_set is True"
            assert num_train_workers is not None, \
                "num_train_workers is required when use_training_set is True"
        if use_validation_set:
            assert val_image_transform is not None, \
                "val_image_transform is required when use_validation_set is True"
            assert num_val_workers is not None, \
                "num_val_workers is required when use_validation_set is True"

        # --- Store Configuration ---
        self.task_mode = task_mode
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.training_data_mode = training_data_mode
        self.data_augmentation_enabled = data_augmentation_enabled
        self.use_validation_set = use_validation_set
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.bbox_format = bbox_format # Store for potential use in target generation
        self.max_images_per_batch = max_images_per_batch
        self.val_batch_size_factor = val_batch_size_factor
        self.num_train_workers = num_train_workers
        self.num_val_workers = num_val_workers

        # --- Load Phrase Embeddings ---
        logger.info(f'Loading phrase embeddings and phrases from {phrase_embeddings_filepath}...')
        embedding_data = get_cached_pickle_file(phrase_embeddings_filepath)
        phrase_embeddings = embedding_data['phrase_embeddings']
        phrases = embedding_data['phrases']
        assert phrase_embeddings.shape[0] == len(phrases), \
            "Mismatch between number of embeddings and phrases"
        logger.info(f'  Initial phrase_embeddings shape: {phrase_embeddings.shape}')
        logger.info(f'  Initial number of phrases: {len(phrases)}')

        # --- Optional: Replace Embeddings with Random Vectors ---
        if replace_phrase_embeddings_with_random_vectors:
            logger.warning('Replacing phrase embeddings with random vectors (DEBUG/ABLATION)')
            # Define a path to save/load the random vectors to ensure consistency across runs
            save_path = f'{phrase_embeddings_filepath}.random_vectors.pkl'
            if os.path.exists(save_path):
                logger.info(f'  Loading pre-generated random vectors from {save_path}')
                phrase_embeddings = load_pickle(save_path)['phrase_embeddings']
            else:
                logger.info('  Generating and saving new random vectors...')
                # Generate random vectors with the same shape and dtype
                phrase_embeddings = np.random.randn(*phrase_embeddings.shape).astype(phrase_embeddings.dtype)
                # Normalize the random vectors along the embedding dimension
                phrase_embeddings /= np.linalg.norm(phrase_embeddings, axis=1, keepdims=True)
                # Save for future runs
                save_dict = {'phrase_embeddings': phrase_embeddings}
                save_pickle(save_dict, save_path)
                logger.info(f'  Saved random vectors to {save_path}')
            logger.info(f'  Using random phrase_embeddings shape: {phrase_embeddings.shape}')

        # --- Reorder Labels, Phrases, and Embeddings ---
        # Goal: Ensure a consistent order where bounding box classes appear first,
        # followed by other image-level labels. This helps align grounding targets
        # with the initial part of the phrase/embedding list.
        logger.info("Reordering labels, phrases, and embeddings...")
        # Assumes self.bbox_class_names is populated by the base class init
        bbox_classes = self.bbox_class_names
        other_labels = [name for name in self.label_names if name not in bbox_classes]
        desired_label_order = bbox_classes + other_labels

        # Verify the reordering logic covers all labels exactly once
        assert len(desired_label_order) == len(self.label_names), "Label count mismatch after reorder planning"
        assert set(desired_label_order) == set(self.label_names), "Label set mismatch after reorder planning"
        logger.info(f'  Desired label order: {desired_label_order}')

        # Get the indices needed to reorder the original self.labels array
        current_label_indices = {name: i for i, name in enumerate(self.label_names)}
        sorted_label_idxs = [current_label_indices[name] for name in desired_label_order]

        # Apply reordering to labels and update self.label_names
        self.labels = self.labels[:, sorted_label_idxs] # Reorder columns
        self.label_names = desired_label_order # Update attribute to reflect new order
        self.phrase_classification_labels = self.labels # Keep a reference with a specific name if needed elsewhere
        logger.info(f'  Reordered phrase_classification_labels shape: {self.phrase_classification_labels.shape}')

        # Reorder phrases and embeddings to match the new label order
        # Assumes VINBIG_LABEL2PHRASE maps the *new* self.label_names to the desired phrases
        try:
            sorted_phrases = [VINBIG_LABEL2PHRASE[name] for name in self.label_names]
        except KeyError as e:
            logger.error(f"Missing phrase mapping for label: {e}. Check VINBIG_LABEL2PHRASE.")
            raise
        # Create mapping from original phrase list to index
        original_phrase2idx = {phrase: i for i, phrase in enumerate(phrases)}
        # Get indices to reorder original embeddings/phrases based on the desired phrase order
        try:
            sorted_phrase_idxs = [original_phrase2idx[phrase] for phrase in sorted_phrases]
        except KeyError as e:
            logger.error(f"Desired phrase '{e}' not found in loaded phrases. Check VINBIG_LABEL2PHRASE and embedding file.")
            raise

        # Apply reordering to phrases and embeddings
        self.phrases = [phrases[i] for i in sorted_phrase_idxs]
        assert self.phrases == sorted_phrases, "Phrase reordering failed sanity check"
        self.phrase_embeddings = phrase_embeddings[sorted_phrase_idxs]
        logger.info(f'  Reordered number of phrases: {len(self.phrases)}')
        logger.info(f'  Reordered phrase_embeddings shape: {self.phrase_embeddings.shape}')
        
        message_to_log = f'\nFinal phrases mapped to labels (in order):'
        for i, (phrase, label) in enumerate(zip(self.phrases, self.label_names)):
            message_to_log += f'\n\t{i}: "{phrase}" ({label})'
        logger.info(message_to_log)

        # --- Determine Training and Validation Indices ---
        logger.info("Determining training and validation indices...")
        # Select indices for the training dataloader based on the mode
        if use_training_set:
            if training_data_mode == VinBigTrainingMode.TRAIN.value:
                self.actual_train_indices = self.train_indices
                logger.info(f"  Using TRAIN split indices ({len(self.actual_train_indices)}) for training.")
            elif training_data_mode == VinBigTrainingMode.TEST.value:
                 # Note: Using test indices for training might be unusual, ensure intended.
                self.actual_train_indices = self.test_indices
                logger.warning(f"  Using TEST split indices ({len(self.actual_train_indices)}) for training.")
            elif training_data_mode == VinBigTrainingMode.ALL.value:
                self.actual_train_indices = self.train_indices + self.test_indices
                logger.warning(f"  Using ALL indices ({len(self.actual_train_indices)}) for training.")
            else:
                raise ValueError(f'Unknown training_data_mode: {training_data_mode}')
        else:
            self.actual_train_indices = [] # Use empty list if no training set
            logger.info("  Training set generation skipped (use_training_set=False).")

        # Select indices for the validation dataloader
        if use_validation_set:
            if use_training_indices_for_validation:
                # Use train indices for validation (e.g., for final eval on train set)
                self.actual_val_indices = self.train_indices
                logger.warning(f"  Using TRAIN split indices ({len(self.actual_val_indices)}) for validation.")
            else:
                # Default: use test indices for validation
                self.actual_val_indices = self.test_indices
                logger.info(f"  Using TEST split indices ({len(self.actual_val_indices)}) for validation.")
        else:
            self.actual_val_indices = [] # Use empty list if no validation set
            logger.info("  Validation set generation skipped (use_validation_set=False).")

        # --- Task-Specific Data Preparation ---
        # Dispatch to the appropriate setup method based on the task mode.
        logger.info(f"Setting up DataLoaders for task: {task_mode}...")
        if task_mode == VinBigPhraseTaskMode.CLASSIFICATION.value:
            self._setup_classification_dataloaders()
        elif task_mode == VinBigPhraseTaskMode.GROUNDING.value:
            # Grounding requires bounding boxes, which were loaded by super().__init__
            assert self.bboxes is not None, "Bounding boxes must be loaded for grounding task"
            self._setup_grounding_dataloaders()
        elif task_mode == VinBigPhraseTaskMode.CLASSIFICATION_AND_GROUNDING.value:
            assert self.bboxes is not None, "Bounding boxes must be loaded for combined task"
            self._setup_classification_and_grounding_dataloaders()
        else:
            # This case should ideally be caught by the initial assertion, but included for safety.
            raise ValueError(f"Unknown task_mode: {task_mode}")

        logger.info("VinBigPhraseTrainer initialization complete.")

    def _setup_classification_dataloaders(self):
        """
        Sets up datasets and dataloaders for the phrase classification task.

        Creates `self.train_dataset`, `self.train_dataloader` (if `use_training_set` is True)
        and `self.val_dataset`, `self.val_dataloader` (if `use_validation_set` is True).
        Training uses balanced sampling via `CompositeInfiniteDataset`.
        Validation uses a standard sequential dataset.
        """
        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_dataloader = None

        # --- Training Dataloader Setup ---
        if self.actual_train_indices: # Check if list is not empty
            logger.info(f'{ANSI_BOLD}Setting up TRAINING classification dataloader...{ANSI_RESET}')
            # Group indices for balanced sampling during training
            # This aims to mitigate class imbalance by sampling from groups of images
            # sharing less common labels more frequently.
            grouped_indices = group_indices_for_balanced_sampling(
                label_matrix=self.phrase_classification_labels, # Use the reordered labels
                indices=self.actual_train_indices,
                label_names=self.label_names, # Use the reordered label names
                min_group_size=50, # Parameter for grouping logic
            )
            train_datasets = []
            train_weights = []
            logger.info("  Creating balanced sampling groups for training:")
            for indices in grouped_indices:
                # Create a dataset for each group
                dataset = VinBig_PhraseClassificationDataset(
                    indices=indices,
                    image_paths=self.image_paths,
                    image_transform=self.train_image_transform,
                    phrase_embeddings=self.phrase_embeddings, # Use reordered embeddings
                    phrase_classification_labels=self.phrase_classification_labels, # Use reordered labels
                    infinite=True, # Make dataset effectively infinite for training sampling
                    shuffle_indices=True, # Shuffle within each group epoch
                )
                # Calculate a weight for each group (e.g., based on size) for sampling probability
                weight = math.log2(len(indices)) ** 3 # Example weighting scheme
                train_datasets.append(dataset)
                train_weights.append(weight)
                logger.info(f'    Group size: {len(indices):<6}, weight: {weight:.2f}')

            # Combine group datasets into a single dataset that samples based on weights
            self.train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.max_images_per_batch,
                num_workers=self.num_train_workers,
                collate_fn=VinBig_PhraseClassificationDataset.collate_fn, # Use dataset's collate_fn
                shuffle=False, # Shuffling is handled by the CompositeInfiniteDataset sampler
                pin_memory=True, # Optimization for GPU data transfer
            )
            logger.info(f"  Training classification DataLoader ready (Batch size: {self.max_images_per_batch}, Workers: {self.num_train_workers}).")

        # --- Validation Dataloader Setup ---
        if self.actual_val_indices: # Check if list is not empty
            logger.info(f'{ANSI_BOLD}Setting up VALIDATION classification dataloader...{ANSI_RESET}')
            # Validation dataset uses the selected validation indices directly
            self.val_dataset = VinBig_PhraseClassificationDataset(
                indices=self.actual_val_indices,
                image_paths=self.image_paths,
                image_transform=self.val_image_transform, # Use validation transform
                phrase_embeddings=self.phrase_embeddings, # Use reordered embeddings
                phrase_classification_labels=self.phrase_classification_labels, # Use reordered labels
                infinite=False, # Finite dataset for validation
                shuffle_indices=False, # No shuffling for validation
            )
            val_batch_size = int(self.max_images_per_batch * self.val_batch_size_factor)
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=self.num_val_workers,
                collate_fn=VinBig_PhraseClassificationDataset.collate_fn, # Use dataset's collate_fn
                shuffle=False, # No shuffling for validation
                pin_memory=True,
            )
            logger.info(f"  Validation classification DataLoader ready (Batch size: {val_batch_size}, Workers: {self.num_val_workers}).")

    def _setup_grounding_dataloaders(self):
        """
        Sets up datasets and dataloaders for the phrase grounding task.

        This involves preprocessing the bounding boxes loaded by the base class
        into probabilistic masks for each phrase associated with an image.
        It then creates `self.train_dataset`, `self.train_dataloader` (if applicable)
        and `self.val_dataset`, `self.val_dataloader` (if applicable) using
        `VinBig_PhraseGroundingDataset`.
        """
        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_dataloader = None

        logger.info(f'{ANSI_BOLD}Preprocessing data for grounding...{ANSI_RESET}')
        # --- Prepare Data Structures for Grounding Dataset ---
        # We need parallel lists containing information for each phrase-image pair
        # where a bounding box exists for that phrase in that image.
        phrase_idxs_for_grounding = [] # Index of the phrase (in the reordered list)
        image_idxs_for_grounding = [] # Index of the image (in self.image_ids)
        image_paths_for_grounding = [] # Path to the image file
        bboxes_for_grounding = [] # List of bboxes (in specified format) for the phrase in the image
        prob_masks_for_grounding = [] # Precomputed target mask for the phrase in the image

        # Iterate through all images and their associated bounding boxes (loaded by base class)
        num_bbox_processed = 0
        for image_idx, bbox_data in tqdm(enumerate(self.bboxes), total=len(self.bboxes),
                                         desc='Processing images for grounding', mininterval=0.5):
            bboxes, bbox_class_ids = bbox_data # Unpack tuple: (list_of_bboxes, list_of_class_ids)

            # Group bboxes by their class ID for the current image
            label_id_to_bboxes = {}
            for bbox, label_id in zip(bboxes, bbox_class_ids):
                if label_id not in label_id_to_bboxes:
                    label_id_to_bboxes[label_id] = []
                label_id_to_bboxes[label_id].append(bbox)
                num_bbox_processed += 1

            # For each class ID (which corresponds to a phrase index after reordering)
            # that has bounding boxes in this image, create an entry for the grounding dataset.
            for label_id, bboxes_for_label in label_id_to_bboxes.items():
                phrase_idx = label_id # Critical assumption: bbox class IDs directly map to reordered phrase indices
                assert 0 <= phrase_idx < len(self.phrases), \
                    f"Phrase index {phrase_idx} out of bounds for phrases list of length {len(self.phrases)}"

                # Calculate the probabilistic target mask from the bounding boxes for this phrase
                prob_mask = calculate_probabilistic_mask_from_bboxes(
                    bboxes=bboxes_for_label,
                    mask_resolution=(self.mask_height, self.mask_width),
                    bbox_format=self.bbox_format, # Use the specified format
                )

                # Append data to the parallel lists
                phrase_idxs_for_grounding.append(phrase_idx)
                image_idxs_for_grounding.append(image_idx) # Store original image index
                image_paths_for_grounding.append(self.image_paths[image_idx])
                bboxes_for_grounding.append(bboxes_for_label) # Store the list of bboxes
                prob_masks_for_grounding.append(prob_mask) # Store the generated mask

        num_grounding_samples = len(phrase_idxs_for_grounding)
        logger.info(f'  Processed {num_bbox_processed} bounding boxes.')
        logger.info(f'  Created {num_grounding_samples} phrase-image grounding samples.')
        assert num_grounding_samples > 0, \
            "No grounding samples were created. Check bounding box data and label mapping."

        # --- Filter Grounding Samples for Train/Val Splits ---
        # The grounding dataset works with indices into the *grounding samples* list created above.
        # We need to map the desired train/val *image* indices to these grounding sample indices.
        actual_train_image_indices_set = set(self.actual_train_indices or [])
        actual_val_image_indices_set = set(self.actual_val_indices or [])

        train_grounding_sample_indices = [
            i for i, img_idx in enumerate(image_idxs_for_grounding)
            if img_idx in actual_train_image_indices_set
        ]
        val_grounding_sample_indices = [
            i for i, img_idx in enumerate(image_idxs_for_grounding)
            if img_idx in actual_val_image_indices_set
        ]

        # --- Training Dataloader Setup ---
        if train_grounding_sample_indices: # Check if list is not empty
            logger.info(f'{ANSI_BOLD}Setting up TRAINING grounding dataloader...{ANSI_RESET}')
            logger.info(f'  Using {len(train_grounding_sample_indices)} grounding samples for training.')
            self.train_dataset = VinBig_PhraseGroundingDataset(
                # Provide the pre-processed parallel lists
                image_paths=image_paths_for_grounding,
                image_transform=self.train_image_transform,
                phrase_idxs=phrase_idxs_for_grounding,
                phrase_embeddings=self.phrase_embeddings, # Use reordered embeddings
                phrase_bboxes=bboxes_for_grounding, # Pass original bboxes
                phrase_prob_masks=prob_masks_for_grounding, # Pass generated masks
                # Provide the filtered indices for this split
                indices=train_grounding_sample_indices,
                feature_map_size=(self.mask_height, self.mask_width), # Target size
                data_augmentation_enabled=self.data_augmentation_enabled, # Control augmentation
                shuffle_indices=True, # Shuffle training samples each epoch
                for_training=True,
                bbox_format=self.bbox_format,
                infinite=False,
            )
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.max_images_per_batch,
                num_workers=self.num_train_workers,
                collate_fn=self.train_dataset.collate_fn, # Use dataset's collate_fn
                shuffle=True, # Shuffle batches each epoch (if dataset isn't infinite/doesn't shuffle internally)
                pin_memory=True,
            )
            logger.info(f"  Training grounding DataLoader ready (Batch size: {self.max_images_per_batch}, Workers: {self.num_train_workers}).")
        elif self.actual_train_indices:
            logger.warning("  Training indices specified, but no corresponding grounding samples found.")


        # --- Validation Dataloader Setup ---
        if val_grounding_sample_indices: # Check if list is not empty
            logger.info(f'{ANSI_BOLD}Setting up VALIDATION grounding dataloader...{ANSI_RESET}')
            logger.info(f'  Using {len(val_grounding_sample_indices)} grounding samples for validation.')
            self.val_dataset = VinBig_PhraseGroundingDataset(
                # Provide the pre-processed parallel lists
                image_paths=image_paths_for_grounding,
                image_transform=self.val_image_transform, # Use validation transform
                phrase_idxs=phrase_idxs_for_grounding,
                phrase_embeddings=self.phrase_embeddings, # Use reordered embeddings
                phrase_bboxes=bboxes_for_grounding,
                phrase_prob_masks=prob_masks_for_grounding,
                # Provide the filtered indices for this split
                indices=val_grounding_sample_indices,
                feature_map_size=(self.mask_height, self.mask_width),
                data_augmentation_enabled=False, # No augmentation for validation
                shuffle_indices=False, # No shuffling for validation
                for_training=False,
                bbox_format=self.bbox_format,
                infinite=False,
            )
            val_batch_size = int(self.max_images_per_batch * self.val_batch_size_factor)
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=self.num_val_workers,
                collate_fn=self.val_dataset.collate_fn, # Use dataset's collate_fn
                shuffle=False, # No shuffling for validation
                pin_memory=True,
            )
            logger.info(f"  Validation grounding DataLoader ready (Batch size: {val_batch_size}, Workers: {self.num_val_workers}).")
        elif self.actual_val_indices:
            logger.warning("  Validation indices specified, but no corresponding grounding samples found.")


    def _setup_classification_and_grounding_dataloaders(self):
        """
        Sets up datasets and dataloaders for the combined classification and
        grounding task.

        (This method is currently not implemented).
        """
        # Implementation Note: This would likely involve creating a dataset that returns
        # both classification targets (for all phrases) and grounding targets (masks)
        # for the phrases that have bounding boxes in the image. The collation function
        # would need to handle combining these potentially different structures.
        # It might reuse logic from the other two setup methods.
        logger.error("Combined classification and grounding dataloader setup is not yet implemented.")
        raise NotImplementedError("Classification and grounding combined dataloader setup not implemented yet.")