import os
import numpy as np
import math
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from medvqa.datasets.vinbig import (
    VINBIG_BBOX_NAMES__MODIFIED,
    VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH,
    VINBIG_IMAGE_LABELS_TEST_CSV_PATH,
    N_IMAGES_TRAIN,
    N_IMAGES_TEST,
    VINBIG_LABELS__MODIFIED,
    _merge_labels,
    compute_masks_and_binary_labels_from_bounding_boxes,
    get_medium_size_image_path,
    get_original_image_path,
    load_labels,
    load_test_image_id_2_bboxes,
    load_train_image_id_2_bboxes,
)
from medvqa.datasets.visual_module import BasicImageDataset, MAETrainerBase
from medvqa.utils.constants import VINBIG_BBOX_NAMES, VINBIG_DATASET_ID, VINBIG_LABEL2PHRASE, VINBIG_LABELS

from medvqa.datasets.dataloading_utils import (
    INFINITE_DATASET_LENGTH,
    CompositeInfiniteDataset,
    group_indices_for_balanced_sampling,
)
from medvqa.datasets.vqa import LabelBasedVQAClass, load_precomputed_visual_features
from medvqa.models.report_generation.templates.vinbig_v1 import TEMPLATES_VINBIG_v1
from medvqa.utils.files import get_cached_pickle_file, load_pickle, save_pickle
from medvqa.utils.logging import print_bold, print_orange

class VinBigTrainingMode:
    TRAIN_ONLY = 'train'
    TEST_ONLY = 'test'
    ALL = 'all'
    @staticmethod
    def get_all():
        return [
            VinBigTrainingMode.TRAIN_ONLY,
            VinBigTrainingMode.TEST_ONLY,
            VinBigTrainingMode.ALL,
        ]

class VinBigTrainerBase(LabelBasedVQAClass):
    
    def __init__(self, use_merged_findings=False, findings_remapper=None, n_findings=None, load_bouding_boxes=False,
                 class_id_offset=0, verbose=False, use_original_image_size=False, use_improved_labels=False):

        # Load labels
        train_image_id_to_labels, test_image_id_to_labels = load_labels(improve_labels=use_improved_labels)
        train_image_ids = list(train_image_id_to_labels.keys())
        test_image_ids = list(test_image_id_to_labels.keys())
        assert len(train_image_ids) == N_IMAGES_TRAIN
        assert len(test_image_ids) == N_IMAGES_TEST
        image_ids = train_image_ids + test_image_ids
        image_id_to_labels = {**train_image_id_to_labels, **test_image_id_to_labels}
        labels = np.array([image_id_to_labels[img_id] for img_id in image_ids], dtype=np.int8)
        assert labels.shape == (N_IMAGES_TRAIN + N_IMAGES_TEST, len(VINBIG_LABELS))
        self.image_ids = image_ids
        self.labels = labels

        # Image paths
        if use_original_image_size:
            self.image_paths = [get_original_image_path(img_id) for img_id in image_ids]
        else:
            self.image_paths = [get_medium_size_image_path(img_id) for img_id in image_ids]

        # Train/test indices
        self.train_indices = list(range(N_IMAGES_TRAIN))
        self.test_indices = list(range(N_IMAGES_TRAIN, N_IMAGES_TRAIN + N_IMAGES_TEST))

        # Bounding boxes
        if load_bouding_boxes:
            print('Loading bounding boxes')
            train_image_id_2_bboxes = load_train_image_id_2_bboxes(
                for_training=True, normalize=True, class_id_offset=class_id_offset,
                improve_labels=use_improved_labels)
            test_image_id_2_bboxes = load_test_image_id_2_bboxes(
                for_training=True, normalize=True, class_id_offset=class_id_offset,
                improve_labels=use_improved_labels)
            
            if use_improved_labels:
                num_bbox_classes = len(VINBIG_BBOX_NAMES__MODIFIED)
            else:
                num_bbox_classes = len(VINBIG_BBOX_NAMES)

            image_id_2_bboxes = {}
            for img_id in image_ids:
                if img_id in train_image_id_2_bboxes:
                    image_id_2_bboxes[img_id] = train_image_id_2_bboxes[img_id]
                elif img_id in test_image_id_2_bboxes:
                    image_id_2_bboxes[img_id] = test_image_id_2_bboxes[img_id]
                else:
                    image_id_2_bboxes[img_id] = ([], [])
            self.image_id_2_bboxes = image_id_2_bboxes
            self.bboxes = [image_id_2_bboxes[img_id] for img_id in image_ids]
            for x in self.bboxes: # sanity check
                assert len(x) == 2
                assert len(x[0]) == len(x[1])
                for y in x[0]:
                    assert len(y) == 4
                    for z in y:
                        assert 0 <= z <= 1, f'z: {z}, y: {y}, x: {x}'
                for y in x[1]:
                    assert 0 <= (y - class_id_offset) < num_bbox_classes
            print(f'  Loaded {len(image_id_2_bboxes)} bounding boxes')
            if verbose:
                # print 5 random bboxes
                import random
                for i in random.sample(range(len(self.bboxes)), 5):
                    print(f'  self.bboxes[{i}]: {self.bboxes[i]}')
        else:
            self.image_id_2_bboxes = None
            self.bboxes = None

        if use_improved_labels:
            label_names = VINBIG_LABELS__MODIFIED
        else:
            label_names = VINBIG_LABELS

        super().__init__(
            label_names=label_names,
            templates=TEMPLATES_VINBIG_v1,
            use_merged_findings=use_merged_findings,
            labels2mergedfindings=findings_remapper[VINBIG_DATASET_ID] if use_merged_findings else None,
            n_findings=n_findings,
            labels=self.labels,
        )

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
        if training_data_mode == VinBigTrainingMode.TRAIN_ONLY:
            train_indices = self.train_indices
        elif training_data_mode == VinBigTrainingMode.TEST_ONLY:
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
                use_vinbig_with_modified_labels=False,
        ):
        super().__init__(
            use_merged_findings=use_merged_findings,
            findings_remapper=findings_remapper,
            n_findings=n_findings,
            load_bouding_boxes=use_bounding_boxes,
            class_id_offset=class_id_offset,
            use_improved_labels=use_vinbig_with_modified_labels,
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
        
        if training_data_mode == VinBigTrainingMode.TRAIN_ONLY:
            train_indices = self.train_indices
        elif training_data_mode == VinBigTrainingMode.TEST_ONLY:
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

class _AlbumentationAdapter:

    def __init__(self):
        pass
    
    def encode(self, bbox_coords, bbox_classes):
        albumentation_bbox_coords = []
        for i in range(len(bbox_coords)):
            x_min = bbox_coords[i][0]
            y_min = bbox_coords[i][1]
            x_max = bbox_coords[i][2]
            y_max = bbox_coords[i][3]
            assert x_min <= x_max
            assert y_min <= y_max
            if x_min < x_max and y_min < y_max: # ignore invalid bboxes
                albumentation_bbox_coords.append([
                    bbox_coords[i][0],
                    bbox_coords[i][1],
                    bbox_coords[i][2],
                    bbox_coords[i][3],
                    bbox_classes[i],
                ])
        return albumentation_bbox_coords
    
    def decode(self, albumentation_bbox_coords):
        bbox_coords = []
        bbox_classes = []
        for i in range(len(albumentation_bbox_coords)):
            bbox_coords.append([
                albumentation_bbox_coords[i][0],
                albumentation_bbox_coords[i][1],
                albumentation_bbox_coords[i][2],
                albumentation_bbox_coords[i][3],
            ])
            bbox_classes.append(albumentation_bbox_coords[i][4])
        return bbox_coords, bbox_classes

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
                self.albumentation_adapter = _AlbumentationAdapter()
        
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

def _create_target_tensors(bboxes, classes, num_classes, feature_map_size):
    """
    Creates a target tensor for bounding boxes on a feature map.
    
    Args:
        bboxes: A tensor of shape (N, 4) representing bounding box coordinates.
        classes: A tensor of shape (N,) representing bounding box classes.
        num_classes: The number of classes.
        feature_map_size: Tuple (H, W) representing the size of the feature map.
    
    Returns:
        A tensor of shape (num_classes, H*W, 4) representing the target tensor with bounding box coordinates (x_min, y_min, x_max, y_max).
        A tensor of shape (num_classes, H*W) representing the target tensor with bounding box presence.
    """
    H, W = feature_map_size
    N = len(bboxes)
    assert len(classes) == N
    target_coords = torch.zeros(num_classes, H, W, 4)
    target_presence = torch.zeros(num_classes, H, W)
    
    for i in range(N):
        x_min, y_min, x_max, y_max = bboxes[i]
        cls = classes[i]
        x_min_scaled = math.floor(x_min * W)
        y_min_scaled = math.floor(y_min * H)
        x_max_scaled = math.ceil(x_max * W)
        y_max_scaled = math.ceil(y_max * H)
        target_coords[cls, y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled, :] = torch.tensor([x_min, y_min, x_max, y_max])
        target_presence[cls, y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled] = 1

    # Reshape target tensors
    target_coords = target_coords.view(num_classes, H*W, 4)
    target_presence = target_presence.view(num_classes, H*W)
    
    return target_coords, target_presence
    
class VinBigBboxGroundingDataset(Dataset):

    def __init__(self, indices, image_paths, image_transform, phrase_embeddings, phrase_classification_labels, 
                 predict_bboxes=False, predict_masks=False, num_bbox_classes=None, feature_map_size=None, bboxes=None,
                 phrase_grounding_masks=None, infinite=False, shuffle_indices=False, data_augmentation_enabled=False,
                 for_training=True, for_yolo=False):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_embeddings = phrase_embeddings
        self.phrase_classification_labels = phrase_classification_labels
        self.phrase_classification_labels_with_bbox = phrase_classification_labels[:, :len(VINBIG_BBOX_NAMES)]
        self.predict_bboxes = predict_bboxes
        self.predict_masks = predict_masks
        self.bboxes = bboxes
        self.indices = indices
        self.infinite = infinite
        self.data_augmentation_enabled = data_augmentation_enabled
        self.for_training = for_training
        self.for_yolo = for_yolo
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(indices)
        if shuffle_indices:
            np.random.shuffle(self.indices)
        if predict_bboxes:
            assert num_bbox_classes is not None
            assert feature_map_size is not None
            assert bboxes is not None
            self.num_bbox_classes = num_bbox_classes
            self.feature_map_size = feature_map_size
            self.bboxes = bboxes
            if data_augmentation_enabled:
                self.albumentation_adapter = _AlbumentationAdapter()
        elif predict_masks:
            assert phrase_grounding_masks is not None
            self.phrase_grounding_masks = phrase_grounding_masks
        if for_yolo:
            assert predict_bboxes

    def __len__(self):
        return self._len
    
    def __getitem__(self, i):
        if self.infinite:
            i %= len(self.indices)
        i = self.indices[i]
        image_path = self.image_paths[i]
        phrase_embeddings = self.phrase_embeddings
        phrase_classification_labels = self.phrase_classification_labels[i]

        if self.predict_bboxes:
            bboxes, classes = self.bboxes[i]
            if self.for_yolo:
                if self.data_augmentation_enabled:
                    image, bboxes, classes, image_size_before, image_size_after = self.image_transform(
                        image_path=image_path,
                        bboxes=bboxes,
                        classes=classes,
                        albumentation_adapter=self.albumentation_adapter,
                        return_image_size=True,
                    )
                else:
                    image, image_size_before, image_size_after = self.image_transform(image_path, return_image_size=True)
                return {
                    'i': image,
                    'pe': phrase_embeddings,
                    'pcl': phrase_classification_labels,
                    # For YOLO
                    'im_file': image_path,
                    'ori_shape': image_size_before,
                    'resized_shape': image_size_after,
                    'bboxes': bboxes,
                    'classes': classes,
                }
            else:
                if self.data_augmentation_enabled:
                    image, bboxes, classes = self.image_transform(
                        image_path=image_path,
                        bboxes=bboxes,
                        classes=classes,
                        albumentation_adapter=self.albumentation_adapter,
                    )
                else:
                    image = self.image_transform(image_path)
                if self.for_training:
                    bbox_target_coords, bbox_target_presence = _create_target_tensors(bboxes, classes, self.num_bbox_classes, self.feature_map_size)
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
                        'bboxes': bboxes,
                        'classes': classes,
                    }
        elif self.predict_masks:
            phrase_grounding_masks = self.phrase_grounding_masks[i]
            image, phrase_grounding_masks, phrase_classification_labels = self.image_transform(
                image_path, phrase_grounding_masks, phrase_classification_labels)
            return {
                'i': image,
                'pe': phrase_embeddings,
                'pgm': phrase_grounding_masks,
                'pcl': phrase_classification_labels,
            }
        else:
            image = self.image_transform(image_path)
            return {
                'i': image,
                'pe': phrase_embeddings,
                'pcl': phrase_classification_labels,
            }

class VinBigPhraseGroundingTrainer(VinBigTrainerBase):
    def __init__(self, collate_batch_fn,
                 mask_height, mask_width, phrase_embeddings_filepath,
                 max_images_per_batch, max_phrases_per_batch,
                 test_batch_size_factor=1,
                 training_data_mode=VinBigTrainingMode.ALL,
                 use_training_set=True, use_validation_set=True,
                 use_training_indices_for_validation=False,
                 num_train_workers=None, num_val_workers=None,
                 train_image_transform=None, val_image_transform=None,
                 data_augmentation_enabled=False,
                 do_visual_grounding_with_bbox_regression=False,
                 do_visual_grounding_with_segmentation=False,
                 for_yolo=False,
                 replace_phrase_embeddings_with_random_vectors=False,
                 use_vinbig_with_modified_labels=False,
                 ):
        super().__init__(
            load_bouding_boxes=True,
            use_improved_labels=use_vinbig_with_modified_labels,
        )

        assert use_training_set or use_validation_set
        
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.training_data_mode = training_data_mode
        self.data_augmentation_enabled = data_augmentation_enabled
        self.use_validation_set = use_validation_set

        print(f'Loding phrase_embeddings and phrases from {phrase_embeddings_filepath}...')
        tmp = get_cached_pickle_file(phrase_embeddings_filepath)
        phrase_embeddings = tmp['phrase_embeddings']
        
        if replace_phrase_embeddings_with_random_vectors:
            print_orange('NOTE: Replacing phrase embeddings with random vectors', bold=True)
            save_path = f'{phrase_embeddings_filepath}.random_vectors.pkl'
            if os.path.exists(save_path):
                print_orange(f'Random vectors already saved at {save_path}')
                phrase_embeddings = load_pickle(save_path)['phrase_embeddings']
            else:
                phrase_embeddings = np.random.randn(*phrase_embeddings.shape).astype(phrase_embeddings.dtype) # replace with random vectors
                phrase_embeddings /= np.linalg.norm(phrase_embeddings, axis=1, keepdims=True) # normalize
                save_dict = {'phrase_embeddings': phrase_embeddings}
                save_pickle(save_dict, save_path)
                print_orange(f'Saved random vectors at {save_path}')

        phrases = tmp['phrases']
        assert phrase_embeddings.shape[0] == len(phrases)
        print(f'phrase_embeddings.shape = {phrase_embeddings.shape}')
        print(f'phrase_embeddings.dtype = {phrase_embeddings.dtype}')
        print(f'len(phrases) = {len(phrases)}')
            
        print('Compute phrase grounding masks and labels')
        self.phrase_grounding_masks = [None] * len(self.bboxes)
        self.phrase_classification_labels = [None] * len(self.bboxes)
        if use_vinbig_with_modified_labels:
            vinbig_bbox_names = VINBIG_BBOX_NAMES__MODIFIED
            vinbig_labels = VINBIG_LABELS__MODIFIED
            num_bbox_classes = len(VINBIG_BBOX_NAMES__MODIFIED)
        else:
            vinbig_bbox_names = VINBIG_BBOX_NAMES
            vinbig_labels = VINBIG_LABELS
            num_bbox_classes = len(VINBIG_BBOX_NAMES)
        print(f'vinbig_bbox_names = {vinbig_bbox_names}')
        print(f'vinbig_labels = {vinbig_labels}')
        print(f'num_bbox_classes = {num_bbox_classes}')
        for i in range(len(self.bboxes)):
            self.phrase_grounding_masks[i], self.phrase_classification_labels[i] = compute_masks_and_binary_labels_from_bounding_boxes(
                mask_height=mask_height, mask_width=mask_width,
                bbox_coords=self.bboxes[i][0],
                bbox_classes=self.bboxes[i][1],
                num_bbox_classes=num_bbox_classes,
            )
        self.phrase_grounding_masks = np.array(self.phrase_grounding_masks)
        self.phrase_classification_labels = np.array(self.phrase_classification_labels)
        print(f'self.phrase_grounding_masks.shape = {self.phrase_grounding_masks.shape}')
        print(f'self.phrase_classification_labels.shape = {self.phrase_classification_labels.shape}')

        print('Append additional labels to phrase_classification_labels')
        # NOTE: only 22/23 of 28 classes have bounding boxes
        non_bbox_labels = [i for i, name in enumerate(vinbig_labels) if name not in vinbig_bbox_names]
        print(f'len(non_bbox_labels) = {len(non_bbox_labels)}')
        print(f'non_bbox_labels = {[vinbig_labels[i] for i in non_bbox_labels]}')
        assert len(self.labels) == len(self.phrase_classification_labels)
        self.phrase_classification_labels = np.concatenate([
            self.phrase_classification_labels,
            self.labels[:, non_bbox_labels],
        ], axis=1)
        print(f'self.phrase_classification_labels.shape = {self.phrase_classification_labels.shape}')
        assert self.phrase_classification_labels.shape == self.labels.shape
        
        # Reorder phrases and phrase_embeddings
        print('Reorder phrases and phrase_embeddings')
        actual_label_names = vinbig_bbox_names + [vinbig_labels[i] for i in non_bbox_labels]
        sorted_phrases = [VINBIG_LABEL2PHRASE[name] for name in actual_label_names]
        phrase2idx = {phrase: i for i, phrase in enumerate(phrases)}
        phrase_idxs = [phrase2idx[phrase] for phrase in sorted_phrases]
        phrases = [phrases[i] for i in phrase_idxs]
        assert phrases == sorted_phrases
        phrase_embeddings = phrase_embeddings[phrase_idxs]
        assert len(phrases) == len(sorted_phrases) == len(actual_label_names) == len(phrase_embeddings)
        print(f'len(phrases) = {len(phrases)}')
        print(f'len(sorted_phrases) = {len(sorted_phrases)}')
        print(f'phrase_embeddings.shape = {phrase_embeddings.shape}')

        for phrase, label in zip(phrases, actual_label_names):
            print(f'\t{phrase} ({label})')

        self.phrases = phrases
        self.actual_label_names = actual_label_names
        self.phrase_embeddings = phrase_embeddings

        if use_training_set:
            assert train_image_transform is not None
            assert num_train_workers is not None

            print_bold('Generating train dataset and dataloader')
            if training_data_mode == VinBigTrainingMode.TRAIN_ONLY:
                train_indices = self.train_indices
            elif training_data_mode == VinBigTrainingMode.TEST_ONLY:
                train_indices = self.test_indices
            elif training_data_mode == VinBigTrainingMode.ALL:
                train_indices = self.train_indices + self.test_indices
            else: assert False, f'Unknown training_data_mode = {training_data_mode}'
            print(f'len(train_indices) = {len(train_indices)}')

            grouped_indices = group_indices_for_balanced_sampling(label_matrix=self.phrase_classification_labels,
                                                                 indices=train_indices,
                                                                 label_names=actual_label_names,
                                                                 min_group_size=50)
            train_datasets = []
            train_weights = []
            for indices in grouped_indices:
                if do_visual_grounding_with_bbox_regression:
                    dataset = VinBigBboxGroundingDataset(
                        indices=indices,
                        image_paths=self.image_paths,
                        image_transform=self.train_image_transform,
                        phrase_embeddings=phrase_embeddings,
                        phrase_classification_labels=self.phrase_classification_labels,
                        predict_bboxes=True,
                        num_bbox_classes=num_bbox_classes,
                        feature_map_size=(mask_height, mask_width),
                        bboxes=self.bboxes,
                        data_augmentation_enabled=self.data_augmentation_enabled,
                        for_training=True,
                        infinite=True,
                        shuffle_indices=True,
                        for_yolo=for_yolo,
                    )
                elif do_visual_grounding_with_segmentation:
                    dataset = VinBigBboxGroundingDataset(
                        indices=indices,
                        image_paths=self.image_paths,
                        image_transform=self.train_image_transform,
                        phrase_embeddings=phrase_embeddings,
                        phrase_classification_labels=self.phrase_classification_labels,
                        predict_masks=True,
                        phrase_grounding_masks=self.phrase_grounding_masks,
                        data_augmentation_enabled=self.data_augmentation_enabled,
                        for_training=True,
                        infinite=True,
                        shuffle_indices=True,
                    )
                else:
                    dataset = VinBigBboxGroundingDataset(
                        indices=indices,
                        image_paths=self.image_paths,
                        image_transform=self.train_image_transform,
                        phrase_embeddings=phrase_embeddings,
                        phrase_classification_labels=self.phrase_classification_labels,
                        data_augmentation_enabled=self.data_augmentation_enabled,
                        for_training=True,
                        infinite=True,
                        shuffle_indices=True,
                    )
                weight = math.log2(len(indices)) ** 3
                train_datasets.append(dataset)
                train_weights.append(weight)
                print(f'  len(indices) = {len(indices)}, weight = {weight}')
            self.train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
            batch_size = max(min(max_images_per_batch, max_phrases_per_batch // len(phrases)), 1) # at least 1 image per batch
            print(f'batch_size = {batch_size}')
            self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_train_workers,
                                            collate_fn=lambda batch: collate_batch_fn(batch,
                                                                                      training_mode=True,
                                                                                      do_visual_grounding_with_bbox_regression=do_visual_grounding_with_bbox_regression,
                                                                                      do_visual_grounding_with_segmentation=do_visual_grounding_with_segmentation,
                                                                                      ),
                                            pin_memory=True)
        
        if use_validation_set:
            assert val_image_transform is not None
            assert num_val_workers is not None

            print_bold('Generating val dataset and dataloader')
            if use_training_indices_for_validation:
                test_indices = self.train_indices
            else:
                test_indices = self.test_indices
            print(f'len(test_indices) = {len(test_indices)}')
            if do_visual_grounding_with_bbox_regression:
                self.val_dataset = VinBigBboxGroundingDataset(
                    indices=test_indices,
                    image_paths=self.image_paths,
                    image_transform=self.val_image_transform,
                    phrase_embeddings=phrase_embeddings,
                    phrase_classification_labels=self.phrase_classification_labels,
                    predict_bboxes=True,
                    num_bbox_classes=num_bbox_classes,
                    feature_map_size=(mask_height, mask_width),
                    bboxes=self.bboxes,
                    for_training=False,
                    for_yolo=for_yolo,
                )
            elif do_visual_grounding_with_segmentation:
                self.val_dataset = VinBigBboxGroundingDataset(
                    indices=test_indices,
                    image_paths=self.image_paths,
                    image_transform=self.val_image_transform,
                    phrase_embeddings=phrase_embeddings,
                    phrase_classification_labels=self.phrase_classification_labels,
                    predict_masks=True,
                    phrase_grounding_masks=self.phrase_grounding_masks,
                    for_training=False,
                )
            else:
                self.val_dataset = VinBigBboxGroundingDataset(
                    indices=test_indices,
                    image_paths=self.image_paths,
                    image_transform=self.val_image_transform,
                    phrase_embeddings=phrase_embeddings,
                    phrase_classification_labels=self.phrase_classification_labels,
                    for_training=False,
                )
            batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // len(phrases)), 1) * test_batch_size_factor)
            print(f'batch_size = {batch_size}')
            self.val_dataloader = DataLoader(self.val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_val_workers,
                                             collate_fn=lambda batch: collate_batch_fn(batch,
                                                                                       training_mode=False,
                                                                                       do_visual_grounding_with_bbox_regression=do_visual_grounding_with_bbox_regression,
                                                                                       do_visual_grounding_with_segmentation=do_visual_grounding_with_segmentation,
                                                                                       ),
                                             pin_memory=True)
