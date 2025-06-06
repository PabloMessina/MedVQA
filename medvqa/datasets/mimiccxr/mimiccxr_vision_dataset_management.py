import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.augmentation import ChestImagenomeAlbumentationAdapter
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    get_anaxnet_bbox_sorted_indices,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    get_dicomId2gender,
    get_labels_per_anatomy_and_anatomy_group,
    load_chest_imagenome_dicom_ids,
    load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix,
    load_chest_imagenome_gold_bboxes,
    load_chest_imagenome_silver_bboxes,
    load_chest_imagenome_silver_bboxes_as_numpy_array,
    load_gold_attributes_relations_dicom_ids,
    load_gold_bbox_dicom_ids,
    load_gold_standard_related_dicom_ids,
    load_nongold_dicom_ids,
    load_chest_imagenome_label_names,
)
from medvqa.datasets.dataloading_utils import (
    INFINITE_DATASET_LENGTH,
    BatchedCompositeInfiniteDataset,
    CompositeInfiniteDataset,
)
from medvqa.datasets.image_processing import image_size_cache
from medvqa.datasets.utils import adapt_label_matrix_as_merged_findings
from medvqa.datasets.visual_module import (
    BasicImageDataset,
    MAETrainerBase,
)
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_IMAGE_ORIENTATIONS,
    MIMICCXR_SPLIT_NAMES,
    MIMICCXR_STUDY_REGEX,
    MIMICCXR_ImageSizeModes,
    MIMICCXR_ViewModes,
    get_broken_images,
    get_dicom_id_and_orientation_list,
    get_image_path_getter,
    get_image_views_dict,
    get_mimiccxr_medium_image_path,
    get_mimiccxr_small_image_path,
    get_split_dict,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.datasets.vqa import load_precomputed_visual_features
from medvqa.utils.constants import CHEXPERT_DATASET_ID, CHEXPERT_LABELS
from medvqa.utils.files_utils import get_cached_json_file, get_cached_pickle_file, load_pickle
from medvqa.utils.logging_utils import print_magenta

class _BalancedSamplingMode:
    BALANCED_CHEST_IMAGENOME_GLOBAL_LABELS = 'balanced_chest_imagenome_global_labels'
    BALANCED_CHEST_IMAGENOME_GLOBAL_LABELS_BATCHWISE = 'balanced_chest_imagenome_global_labels_batchwise'
    BALANCED_CHEXPERT_LABELS = 'balanced_chexpert_labels'
    BALANCED_CHEXPERT_LABELS_BATCHWISE = 'balanced_chexpert_labels_batchwise'

class MIMICCXR_Visual_Dataset(Dataset):
    def __init__(self, indices, report_ids,
                include_image=True,
                image_paths=None,
                image_transform=None,
                data_augmentation_enabled=False,
                shuffle=False,
                infinite=False,
                # aux task: medical tags
                classify_tags=False,
                rid2tags=None,
                # aux task: image orientation
                classify_orientation=False,
                orientations=None,
                # aux task: chexpert labels
                classify_chexpert=False,
                chexpert_labels=None,
                # aux task: question labels
                classify_questions=False,
                question_labels=None,
                # aux task: genders
                classify_gender=False,
                genders=None,
                # aux task: chest imagenome labels
                classify_chest_imagenome=False,
                chest_imagenome_labels=None,
                # aux task: chest imagenome bboxes
                predict_bboxes_chest_imagenome=False,
                dicom_idxs=None,
                gt_bbox_coords=None,
                gt_bbox_presence=None,
                flipped_gt_bbox_coords=None, # for data augmentation
                flipped_gt_bbox_presence=None, # for data augmentation
                use_anaxnet_bbox_subset=False,    
                pass_pred_bbox_coords_to_model=False,
                pred_bbox_coords=None,
                flipped_pred_bbox_coords=None, # for data augmentation
                horizontal_flip=False,
                # precomputed visual features
                use_precomputed_visual_features=False,
                precomputed_visual_features=None,
                idx2visfeatidx=None,
                # yolov8
                use_yolov8=False,
                # yolov11
                use_yolov11=False,
            ):
        self.indices = indices
        self.report_ids = report_ids
        self.include_image = include_image
        self.data_augmentation_enabled = data_augmentation_enabled    
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.classify_tags = classify_tags
        self.rid2tags = rid2tags
        self.classify_orientation = classify_orientation
        self.orientations = orientations
        self.classify_chexpert = classify_chexpert
        self.chexpert_labels = chexpert_labels
        self.classify_questions = classify_questions
        self.question_labels = question_labels
        self.classify_gender = classify_gender
        self.genders = genders
        self.classify_chest_imagenome = classify_chest_imagenome
        self.chest_imagenome_labels = chest_imagenome_labels
        self.predict_bboxes_chest_imagenome = predict_bboxes_chest_imagenome
        self.dicom_idxs = dicom_idxs
        self.gt_bbox_coords = gt_bbox_coords
        self.gt_bbox_presence = gt_bbox_presence
        self.pass_pred_bbox_coords_to_model = pass_pred_bbox_coords_to_model
        self.pred_bbox_coords = pred_bbox_coords
        self.use_precomputed_visual_features = use_precomputed_visual_features
        self.precomputed_visual_features = precomputed_visual_features
        self.idx2visfeatidx = idx2visfeatidx
        self.use_anaxnet_bbox_subset = use_anaxnet_bbox_subset
        self.horizontal_flip = horizontal_flip
        self.flipped_gt_bbox_coords = flipped_gt_bbox_coords
        self.flipped_gt_bbox_presence = flipped_gt_bbox_presence
        self.flipped_pred_bbox_coords = flipped_pred_bbox_coords
        self.use_yolov8 = use_yolov8
        self.use_yolov11 = use_yolov11

        if self.pass_pred_bbox_coords_to_model:
            assert self.pred_bbox_coords is not None

        if self.predict_bboxes_chest_imagenome:
            assert self.gt_bbox_coords is not None
            assert self.gt_bbox_presence is not None

        if self.predict_bboxes_chest_imagenome or self.pass_pred_bbox_coords_to_model:
            if self.data_augmentation_enabled:
                if horizontal_flip:
                    if self.predict_bboxes_chest_imagenome:
                        assert flipped_gt_bbox_coords is not None
                        assert flipped_gt_bbox_presence is not None
                    if self.pass_pred_bbox_coords_to_model:
                        assert flipped_pred_bbox_coords is not None
                if use_anaxnet_bbox_subset:
                    num_bbox_classes = CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES
                else:
                    num_bbox_classes = CHEST_IMAGENOME_NUM_BBOX_CLASSES
                self.albumentation_adapter = ChestImagenomeAlbumentationAdapter(num_bbox_classes)
        if shuffle:
            random.shuffle(self.indices) # shuffle in place            
        self.infinite = infinite
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self.infinite:
            i = i % len(self.indices)
        idx = self.indices[i]
        rid = self.report_ids[idx]
        output = { 'idx': idx }
        use_yolo = self.use_yolov8 or self.use_yolov11
        if self.include_image:
            # global _DEBUG
            image_path = self.image_paths[idx]
            # handle transform differently for chest imagenome bboxes
            if self.predict_bboxes_chest_imagenome and not self.pass_pred_bbox_coords_to_model: 
                dicom_idx = self.dicom_idxs[idx]
                gt_bbox_coords = self.gt_bbox_coords[dicom_idx]
                gt_bbox_presence = self.gt_bbox_presence[dicom_idx]
                if self.data_augmentation_enabled: # data augmentation with albumentations
                    if self.horizontal_flip:
                        tmp = self.image_transform(
                            image_path=image_path,
                            bboxes=gt_bbox_coords,
                            presence=gt_bbox_presence,
                            albumentation_adapter=self.albumentation_adapter,
                            flipped_bboxes=self.flipped_gt_bbox_coords[dicom_idx],
                            flipped_presence=self.flipped_gt_bbox_presence[dicom_idx],
                            return_image_size=use_yolo,
                        )
                    else:
                        tmp = self.image_transform(
                            image_path=image_path,
                            bboxes=gt_bbox_coords,
                            presence=gt_bbox_presence,
                            albumentation_adapter=self.albumentation_adapter,
                            return_image_size=use_yolo,
                        )
                    if use_yolo:
                        image, gt_bbox_coords, gt_bbox_presence, image_size_before, image_size_after = tmp
                    else:
                        image, gt_bbox_coords, gt_bbox_presence = tmp
                else: # no data augmentation
                    tmp = self.image_transform(image_path, return_image_size=use_yolo)
                    if use_yolo:
                        image, image_size_before, image_size_after = tmp
                    else:
                        image = tmp
                assert len(gt_bbox_coords.shape) == 2
                output['chest_imagenome_bbox_coords'] = gt_bbox_coords
                output['chest_imagenome_bbox_presence'] = gt_bbox_presence
                # if _DEBUG:
                #     print('Case 1: A and not B')
            elif not self.predict_bboxes_chest_imagenome and self.pass_pred_bbox_coords_to_model:
                assert not use_yolo
                dicom_idx = self.dicom_idxs[idx]
                pred_bbox_coords = self.pred_bbox_coords[dicom_idx]
                if self.data_augmentation_enabled:
                    if self.horizontal_flip:
                        image, pred_bbox_coords, _ = self.image_transform(
                            image_path=image_path,
                            bboxes=pred_bbox_coords,
                            albumentation_adapter=self.albumentation_adapter,
                            flipped_bboxes=self.flipped_pred_bbox_coords[dicom_idx],
                        )
                    else:
                        image, pred_bbox_coords, _ = self.image_transform(
                            image_path=image_path,
                            bboxes=pred_bbox_coords,
                            albumentation_adapter=self.albumentation_adapter,
                        )
                else:
                    image = self.image_transform(image_path)                
                output['pred_bbox_coords'] = pred_bbox_coords
                # if _DEBUG:
                #     print('Case 2: not A and B')
            elif self.predict_bboxes_chest_imagenome and self.pass_pred_bbox_coords_to_model:
                dicom_idx = self.dicom_idxs[idx]
                gt_bbox_coords = self.gt_bbox_coords[dicom_idx]
                gt_bbox_presence = self.gt_bbox_presence[dicom_idx]
                pred_bbox_coords = self.pred_bbox_coords[dicom_idx]
                if self.data_augmentation_enabled:
                    if self.horizontal_flip:
                        tmp = self.image_transform(
                            image_path=image_path,
                            bboxes=gt_bbox_coords,
                            presence=gt_bbox_presence,
                            albumentation_adapter=self.albumentation_adapter,
                            flipped_bboxes=self.flipped_gt_bbox_coords[dicom_idx],
                            flipped_presence=self.flipped_gt_bbox_presence[dicom_idx],
                            pred_bboxes=pred_bbox_coords,
                            flipped_pred_bboxes=self.flipped_pred_bbox_coords[dicom_idx],
                            return_image_size=use_yolo,
                        )
                    else:
                        tmp = self.image_transform(
                            image_path=image_path,
                            bboxes=gt_bbox_coords,
                            presence=gt_bbox_presence,
                            albumentation_adapter=self.albumentation_adapter,
                            pred_bboxes=pred_bbox_coords,
                            return_image_size=use_yolo,
                        )
                    if use_yolo:
                        image, gt_bbox_coords, gt_bbox_presence, pred_bbox_coords, image_size_before, image_size_after = tmp
                    else:
                        image, gt_bbox_coords, gt_bbox_presence, pred_bbox_coords = tmp
                else:
                    tmp = self.image_transform(image_path, return_image_size=use_yolo)
                    if use_yolo:
                        image, image_size_before, image_size_after = tmp
                    else:
                        image = tmp
                assert len(gt_bbox_coords.shape) == 2
                output['chest_imagenome_bbox_coords'] = gt_bbox_coords
                output['chest_imagenome_bbox_presence'] = gt_bbox_presence
                output['pred_bbox_coords'] = pred_bbox_coords
                # if _DEBUG:
                #     print('Case 3: A and B')
            else:
                assert not use_yolo
                image = self.image_transform(image_path)
                # if _DEBUG:
                #     print('Case 4: not A and not B')
            output['i'] = image
            # _DEBUG = False
        if self.use_precomputed_visual_features:
            visfeat_idx = self.idx2visfeatidx[idx]
            visfeat = self.precomputed_visual_features[visfeat_idx]
            output['vf'] = visfeat
        if self.classify_tags:
            output['tags'] = self.rid2tags[rid]
        if self.classify_orientation:
            output['orientation'] = self.orientations[idx]
        if self.classify_chexpert:
            output['chexpert'] = self.chexpert_labels[rid]
        if self.classify_questions:
            output['qlabels'] = self.question_labels[rid]
        if self.classify_gender:
            output['gender'] = self.genders[idx]
        if self.classify_chest_imagenome:
            output['chest_imagenome'] = self.chest_imagenome_labels[rid]
        if use_yolo:
            # We need to adapt the output a little bit to make it compatible with YOLOv8
            output['im_file'] = image_path
            output['ori_shape'] = image_size_before
            output['resized_shape'] = image_size_after
        return output

def _define_allowed_dicom_ids(view_mode, use_all_data, use_test_set, use_chest_imagenome_bbox_gold_set,
                              use_chest_imagenome_label_gold_set, use_decent_images_only):
    allowed_dicom_ids = None

    if use_all_data:
        if view_mode == MIMICCXR_ViewModes.CHEST_IMAGENOME:
            allowed_dicom_ids = set(load_chest_imagenome_dicom_ids(decent_images_only=use_decent_images_only))
    else:
        allowed_train_val_dicom_ids = None
        if view_mode == MIMICCXR_ViewModes.CHEST_IMAGENOME:
            if use_decent_images_only:
                decent_dicom_ids = set(load_chest_imagenome_dicom_ids(decent_images_only=True))
            if use_test_set:
                if use_decent_images_only:
                    allowed_test_dicom_ids = decent_dicom_ids
                else:
                    allowed_test_dicom_ids = set(load_chest_imagenome_dicom_ids())
            elif use_chest_imagenome_bbox_gold_set:
                allowed_test_dicom_ids = set(load_gold_bbox_dicom_ids())
                if use_decent_images_only:
                    allowed_test_dicom_ids &= decent_dicom_ids
            elif use_chest_imagenome_label_gold_set:
                allowed_test_dicom_ids = set(load_gold_attributes_relations_dicom_ids())
                if use_decent_images_only:
                    allowed_test_dicom_ids &= decent_dicom_ids
            else:
                allowed_train_val_dicom_ids = set(load_nongold_dicom_ids())
                if use_decent_images_only:
                    allowed_train_val_dicom_ids &= decent_dicom_ids
        else:
            assert use_decent_images_only is False
            assert use_chest_imagenome_bbox_gold_set is False
            assert use_chest_imagenome_label_gold_set is False
        allowed_dicom_ids = allowed_train_val_dicom_ids or allowed_test_dicom_ids
        print(f'Allowed dicom ids: {len(allowed_dicom_ids)}')
    
    return allowed_dicom_ids

def _load_chest_imagenome_labels(self, chest_imagenome_labels_filename, use_chest_imagenome_label_gold_set,
                                 chest_imagenome_label_names_filename):
    print('Loading Chest Imagenome labels...')
    assert chest_imagenome_labels_filename is not None
    assert chest_imagenome_labels_filename != 'gold_imageId2binaryLabels.pkl', \
        'This should be file used for training, not testing'
    if use_chest_imagenome_label_gold_set:
        _, self.chest_imagenome_labels = \
            load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix('gold_imageId2binaryLabels.pkl')
        print('Using gold labels for Chest Imagenome (not the labels used for training)')
        print(f'\tFile used in training: {chest_imagenome_labels_filename}')
        print('\tFile used for testing (right now): gold_imageId2binaryLabels.pkl')
        print(f'self.chest_imagenome_labels.shape = {self.chest_imagenome_labels.shape}')
        # sanity check
        used_rid_set = set()
        _hc, _unhc = 0, 0
        for i in self.test_indices:
            rid = self.report_ids[i]
            used_rid_set.add(rid)
            if self.chest_imagenome_labels[rid].max() == 1:
                _unhc += 1
            else:
                _hc += 1
        print('Sanity check: _hc =', _hc, '_unhc =', _unhc)
        assert _unhc > 0, 'No abnormal reports found in gold labels'
        # choose 100 report_ids not in used_rid_set
        for _ in range(100):
            while True:
                rid = random.randint(0, self.chest_imagenome_labels.shape[0] - 1)
                if rid not in used_rid_set:
                    break
            assert self.chest_imagenome_labels[rid].max() == 0
    else:
        _, self.chest_imagenome_labels = \
            load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix(chest_imagenome_labels_filename)

    if chest_imagenome_label_names_filename is not None:
        # We need to rearrange the labels to match the order in which the model will predict them
        print_magenta('Reordering Chest Imagenome labels according to anatomy-label relations...', bold=True)
        tmp = get_labels_per_anatomy_and_anatomy_group(chest_imagenome_label_names_filename, for_training=True)
        label_order = []
        for _, labels in tmp['anatomy_to_localized_labels']:
            label_order.extend(labels)
        for _, labels in tmp['anatomy_group_to_global_labels']:
            label_order.extend(labels)
        assert len(label_order) == len(self.chest_imagenome_label_names)
    elif use_chest_imagenome_label_gold_set:
        label_order = list(range(len(self.chest_imagenome_label_names)))
    else:
        label_order = None
    if label_order is not None:
        if use_chest_imagenome_label_gold_set:
            print('  Using gold labels for Chest Imagenome (not the labels used for training)')
            gold_label_names = load_chest_imagenome_label_names('gold_binary_labels.pkl')
            common_label_names = set(gold_label_names) & set(self.chest_imagenome_label_names)
            assert len(common_label_names) > 0
            _valid_label_indices__model_pov = [i for i, idx in enumerate(label_order)\
                                                if self.chest_imagenome_label_names[idx] in common_label_names]
            _valid_label_indices__gold_pov = [gold_label_names.index(self.chest_imagenome_label_names[idx]) for idx in label_order\
                                            if self.chest_imagenome_label_names[idx] in common_label_names]
            self.valid_chest_imagenome_label_indices = _valid_label_indices__model_pov
            self.chest_imagenome_labels = self.chest_imagenome_labels[:, _valid_label_indices__gold_pov]
            self.chest_imagenome_label_names = [gold_label_names[i] for i in _valid_label_indices__gold_pov]
            assert set(self.chest_imagenome_label_names) == set(common_label_names)
            print(f'  len(gold_label_names) = {len(gold_label_names)}')
            print(f'  len(label_order) = {len(label_order)}')
            print(f'  len(common_label_names) = {len(common_label_names)}')
            print(f'  len(self.valid_chest_imagenome_label_indices) = {len(self.valid_chest_imagenome_label_indices)}')
        else:
            assert self.chest_imagenome_labels.shape[1] == len(label_order)
            assert set(label_order) == set(range(len(label_order)))
            assert len(label_order) == len(self.chest_imagenome_label_names)
            self.chest_imagenome_labels = self.chest_imagenome_labels[:, label_order]
            self.chest_imagenome_label_names = [self.chest_imagenome_label_names[i] for i in label_order]
        print(f'  len(self.chest_imagenome_label_names) = {len(self.chest_imagenome_label_names)}')
        print(f'  self.chest_imagenome_labels.shape = {self.chest_imagenome_labels.shape}')
    else:
        self.chest_imagenome_labels = None

def _create_dataset(self, indices, shuffle=False, balanced_sampling_mode=None, **create_dataset_kwargs):
    if balanced_sampling_mode is not None:
        print(f'Balanced sampling mode: {balanced_sampling_mode}')
        datasets = []
        if balanced_sampling_mode == _BalancedSamplingMode.BALANCED_CHEST_IMAGENOME_GLOBAL_LABELS:
            assert (hasattr(self, 'use_chest_imagenome') and self.use_chest_imagenome) or\
                   (hasattr(self, 'classify_chest_imagenome') and self.classify_chest_imagenome)
            assert self.chest_imagenome_labels is not None
            assert self.chest_imagenome_label_names is not None
            global2idxs = {}
            without_global = []                
            print('Regrouping indices by Chest Imagenome labels for balanced sampling...')
            _labels, _rids = self.chest_imagenome_labels, self.report_ids
            for i, label_name in tqdm(enumerate(self.chest_imagenome_label_names)):
                if len(label_name) == 2:
                    global_name = label_name[-1]
                    global2idxs[global_name] = [idx for idx in indices if _labels[_rids[idx], i] == 1]
            without_global = [idx for idx in indices if _labels[_rids[idx]].max() == 0]
            lines = []
            for global_name, idxs in global2idxs.items():
                lines.append((len(idxs), f'Global: {global_name}, # images: {len(idxs)}'))
                dataset = self._create_dataset(idxs, shuffle=shuffle, infinite=True, **create_dataset_kwargs)
                datasets.append(dataset)
            lines.append((-1, f'# images without global: {len(without_global)}'))
            lines.sort(key=lambda x: x[0], reverse=True)
            for _, line in lines: print(line)
            if len(without_global) > 0:
                dataset = self._create_dataset(without_global, shuffle=shuffle, infinite=True, **create_dataset_kwargs)
                datasets.append(dataset)
            dataset = CompositeInfiniteDataset(datasets, [1] * len(datasets))
        elif balanced_sampling_mode == _BalancedSamplingMode.BALANCED_CHEXPERT_LABELS:
            assert (hasattr(self, 'use_chexpert') and self.use_chexpert) or\
                   (hasattr(self, 'classify_chexpert') and self.classify_chexpert)
            assert self.chexpert_labels is not None
            label2idxs = {}
            without_label = []
            print('Regrouping indices by CheXpert labels for balanced sampling...')
            for i in tqdm(indices, mininterval=2):
                rid = self.report_ids[i]
                labels = self.chexpert_labels[rid]
                has_label = False
                for j, label in enumerate(labels):
                    if label == 1:
                        label_name = CHEXPERT_LABELS[j]
                        try:
                            label2idxs[label_name].append(i)
                        except KeyError:
                            label2idxs[label_name] = [i]
                        has_label = True
                if not has_label:
                    without_label.append(i)
            lines = []
            for label_name, idxs in label2idxs.items():
                lines.append((len(idxs), f'Label: {label_name}, # images: {len(idxs)}'))
                dataset = self._create_dataset(idxs, shuffle=shuffle, infinite=True, **create_dataset_kwargs)
                datasets.append(dataset)
            lines.append((-1, f'# images without label: {len(without_label)}'))
            lines.sort(key=lambda x: x[0], reverse=True)
            for _, line in lines: print(line)
            if len(without_label) > 0:
                dataset = self._create_dataset(without_label, shuffle=shuffle, infinite=True, **create_dataset_kwargs)
                datasets.append(dataset)
            dataset = CompositeInfiniteDataset(datasets, [1] * len(datasets))
        elif balanced_sampling_mode == _BalancedSamplingMode.BALANCED_CHEST_IMAGENOME_GLOBAL_LABELS_BATCHWISE:
            assert hasattr(self, 'balanced_batch_size') and self.balanced_batch_size is not None
            # assert (hasattr(self, 'use_chest_imagenome') and self.use_chest_imagenome) or\
            #        (hasattr(self, 'classify_chest_imagenome') and self.classify_chest_imagenome)
            assert self.chest_imagenome_labels is not None
            assert self.chest_imagenome_label_names is not None
            global2idxs = {}
            print('Regrouping indices by Chest Imagenome labels for balanced sampling...')
            _labels, _rids = self.chest_imagenome_labels, self.report_ids
            for i, label_name in tqdm(enumerate(self.chest_imagenome_label_names)):
                if len(label_name) == 2:
                    global_name = label_name[-1]
                    global2idxs[global_name] = [idx for idx in indices if _labels[_rids[idx], i] == 1]
            max_name_len = max(len(global_name) for global_name in global2idxs.keys())
            lines = []
            for global_name, idxs in global2idxs.items():
                idxs_set = set(idxs)
                other_idxs = [i for i in indices if i not in idxs_set]
                global_name = global_name.ljust(max_name_len)
                lines.append((len(idxs), f'Global: {global_name}, # images: {len(idxs)} (other: {len(other_idxs)})'))
                assert len(idxs) > 0
                assert len(other_idxs) > 0
                dataset_pos = self._create_dataset(idxs, shuffle=shuffle, infinite=True, **create_dataset_kwargs)
                dataset_neg = self._create_dataset(other_idxs, shuffle=shuffle, infinite=True, **create_dataset_kwargs)
                dataset_pos_neg = CompositeInfiniteDataset([dataset_pos, dataset_neg], [0.6, 0.4])
                datasets.append(dataset_pos_neg)
            lines.sort(key=lambda x: x[0], reverse=True)
            for _, line in lines: print(line)
            dataset = BatchedCompositeInfiniteDataset(datasets, [1] * len(datasets), batch_size=self.balanced_batch_size)
        elif balanced_sampling_mode == _BalancedSamplingMode.BALANCED_CHEXPERT_LABELS_BATCHWISE:
            assert hasattr(self, 'balanced_batch_size') and self.balanced_batch_size is not None
            assert (hasattr(self, 'use_chexpert') and self.use_chexpert) or\
                   (hasattr(self, 'classify_chexpert') and self.classify_chexpert)
            assert self.chexpert_labels is not None
            label2idxs = {}
            print('Regrouping indices by CheXpert labels for balanced sampling...')
            for i in tqdm(indices, mininterval=2):
                rid = self.report_ids[i]
                labels = self.chexpert_labels[rid]
                for j, label in enumerate(labels):
                    if label == 1:
                        label_name = CHEXPERT_LABELS[j]
                        try:
                            label2idxs[label_name].append(i)
                        except KeyError:
                            label2idxs[label_name] = [i]
            max_name_len = max(len(label_name) for label_name in label2idxs.keys())
            lines = []
            for label_name, idxs in label2idxs.items():
                idxs_set = set(idxs)
                other_idxs = [i for i in indices if i not in idxs_set]
                label_name = label_name.ljust(max_name_len)
                lines.append((len(idxs), f'Label: {label_name}, # images: {len(idxs)} (other: {len(other_idxs)})'))
                assert len(idxs) > 0
                assert len(other_idxs) > 0
                dataset_pos = self._create_dataset(idxs, shuffle=shuffle, infinite=True, **create_dataset_kwargs)
                dataset_neg = self._create_dataset(other_idxs, shuffle=shuffle, infinite=True, **create_dataset_kwargs)
                dataset_pos_neg = CompositeInfiniteDataset([dataset_pos, dataset_neg], [0.6, 0.4])
                datasets.append(dataset_pos_neg)
            lines.sort(key=lambda x: x[0], reverse=True)
            for _, line in lines: print(line)
            dataset = BatchedCompositeInfiniteDataset(datasets, [1] * len(datasets), batch_size=self.balanced_batch_size)
        else:
            raise ValueError(f'Unexpected balanced sampling mode: {balanced_sampling_mode}')
    else:
        dataset = self._create_dataset(indices, **create_dataset_kwargs)
    return dataset

def _create_dataset_and_dataloader(self, indices, image_transform, collate_batch_fn, data_augmentation_enabled=False,
                                   shuffle=False, balanced_sampling_mode=None):
    dataset = _create_dataset(self, indices, shuffle, balanced_sampling_mode,
                              image_transform=image_transform,
                              data_augmentation_enabled=data_augmentation_enabled)

    dataloader = DataLoader(
        dataset,
        batch_size=self.batch_size,
        shuffle=shuffle and balanced_sampling_mode is None,
        num_workers=self.num_workers,
        collate_fn=collate_batch_fn,
        pin_memory=True,
    )
    return dataset, dataloader

class MIMICCXR_VisualModuleTrainer:

    def __init__(self, 
                batch_size, collate_batch_fn, num_workers,
                train_image_transform=None,
                val_image_transform=None, 
                use_test_set=False,
                use_all_data=False,
                use_chest_imagenome_bbox_gold_set=False,
                use_chest_imagenome_label_gold_set=False,
                use_val_set_only=False,
                test_image_transform=None,
                data_augmentation_enabled=False,
                horizontal_flip=False,
                include_image=True,
                source_image_size_mode=MIMICCXR_ImageSizeModes.SMALL_256x256,
                view_mode=MIMICCXR_ViewModes.ANY_SINGLE,
                use_decent_images_only=False,
                classify_tags=False,
                medical_tags_per_report_filename=None,
                classify_orientation=False,
                classify_gender=False,
                classify_chexpert=False,
                chexpert_labels_filename=None,
                classify_questions=False,
                question_labels_filename=None,
                classify_chest_imagenome=False,
                predict_bboxes_chest_imagenome=False,
                predict_labels_and_bboxes_chest_imagenome=False,
                clamp_bboxes_chest_imagenome=False,
                use_anaxnet_bbox_subset=False, # use the subset of bboxes that the AnaXNET paper used
                chest_imagenome_labels_filename=None,
                chest_imagenome_label_names_filename=None,
                use_precomputed_visual_features=False,
                precomputed_visual_features_path=None,
                use_merged_findings=False,
                findings_remapper=None,
                n_findings=None,
                use_detectron2=False,
                detectron2_cfg=None,
                balanced_sampling_mode=None,
                balanced_batch_size=None,
                pass_pred_bbox_coords_to_model=False,
                use_gt_bboxes_as_pred=False,
                use_yolov8=False,
                use_yolov11=False,
                **unused_kwargs,
            ):

        train_collate_batch_fn = lambda batch: collate_batch_fn(batch, training_mode=True)
        eval_collate_batch_fn = lambda batch: collate_batch_fn(batch, training_mode=False)

        if len(unused_kwargs) > 0:
            # Print warning in orange and bold
            print('\033[93m\033[1mWarning: unused kwargs in MIMICCXR_VisualModuleTrainer: {}\033[0m'.format(unused_kwargs))
        # Sanity checks
        assert sum([use_test_set, use_val_set_only,
                    use_chest_imagenome_bbox_gold_set,
                    use_chest_imagenome_label_gold_set,
                    use_all_data]) <= 1 # at most one of these can be true
        if use_test_set or use_chest_imagenome_bbox_gold_set or use_chest_imagenome_label_gold_set or use_all_data:
            assert test_image_transform is not None
            assert not data_augmentation_enabled
        else:
            assert val_image_transform is not None
            if not use_val_set_only:
                assert train_image_transform is not None
        if classify_gender:
            assert view_mode == MIMICCXR_ViewModes.CHEST_IMAGENOME

        self.use_detectron2 = use_detectron2
        self.use_anaxnet_bbox_subset = use_anaxnet_bbox_subset
        self.use_yolov8 = use_yolov8
        self.use_yolov11 = use_yolov11

        if chest_imagenome_label_names_filename is not None:
            self.chest_imagenome_label_names = load_chest_imagenome_label_names(chest_imagenome_label_names_filename)
        elif chest_imagenome_labels_filename is not None:
            self.chest_imagenome_label_names = load_chest_imagenome_label_names(
                chest_imagenome_labels_filename.replace('imageId2labels', 'labels'))
        else:
            self.chest_imagenome_label_names = None

        if use_detectron2:
            assert detectron2_cfg is not None
            assert source_image_size_mode == MIMICCXR_ImageSizeModes.MEDIUM_512
            assert use_decent_images_only
            assert clamp_bboxes_chest_imagenome
            
            if use_test_set:
                # Create test dataset and dataloader
                self.test_dataset = Detectron2AdaptedDataset(
                    split_name='test',
                    transform=test_image_transform,
                    source_image_size_mode=source_image_size_mode,
                    use_decent_images_only=use_decent_images_only,
                    clamp_bboxes_chest_imagenome=clamp_bboxes_chest_imagenome,
                    data_augmentation_enabled=False,
                    use_anaxnet_bbox_subset=use_anaxnet_bbox_subset,
                )
                self.test_dataloader = DataLoader(
                    dataset=self.test_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=eval_collate_batch_fn,
                    shuffle=False,
                )
            elif use_chest_imagenome_bbox_gold_set:
                # Create test dataset and dataloader using chest imagenome gold set
                self.test_dataset = Detectron2AdaptedDataset(
                    split_name=None,
                    use_chest_imagenome_gold_set=True,
                    transform=test_image_transform,
                    source_image_size_mode=source_image_size_mode,
                    use_decent_images_only=use_decent_images_only,
                    clamp_bboxes_chest_imagenome=clamp_bboxes_chest_imagenome,
                    data_augmentation_enabled=False,
                    use_anaxnet_bbox_subset=use_anaxnet_bbox_subset,
                )
                self.test_dataloader = DataLoader(
                    dataset=self.test_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=eval_collate_batch_fn,
                    shuffle=False,
                )
            else:
                assert not use_chest_imagenome_label_gold_set, 'Not implemented'
                # Create train dataset and dataloader
                if not use_val_set_only:
                    self.train_dataset = Detectron2AdaptedDataset(
                        split_name='train',
                        transform=train_image_transform,
                        source_image_size_mode=source_image_size_mode,
                        use_decent_images_only=use_decent_images_only,
                        clamp_bboxes_chest_imagenome=clamp_bboxes_chest_imagenome,
                        data_augmentation_enabled=data_augmentation_enabled,
                        use_anaxnet_bbox_subset=use_anaxnet_bbox_subset,
                    )
                    self.train_dataloader = DataLoader(
                        dataset=self.train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        collate_fn=train_collate_batch_fn,
                        shuffle=True,
                    )
                # self.train_dataloader = build_detection_train_loader(
                #     dataset=self.train_dataset,
                #     mapper=DatasetMapper(is_train=True, image_format=detectron2_cfg.INPUT.FORMAT, augmentations=[]),
                #     total_batch_size=batch_size,
                #     num_workers=num_workers,
                # )
                # Create validation dataset and dataloader
                self.val_dataset = Detectron2AdaptedDataset(
                    split_name='validate',
                    transform=val_image_transform,
                    source_image_size_mode=source_image_size_mode,
                    use_decent_images_only=use_decent_images_only,
                    clamp_bboxes_chest_imagenome=clamp_bboxes_chest_imagenome,
                    data_augmentation_enabled=False,
                    use_anaxnet_bbox_subset=use_anaxnet_bbox_subset,
                )
                self.val_dataloader = DataLoader(
                    dataset=self.val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=eval_collate_batch_fn,
                    shuffle=False,
                )
        else:
            self.train_image_transform = train_image_transform
            self.val_image_transform = val_image_transform
            self.test_image_transform = test_image_transform
            self.data_augmentation_enabled = data_augmentation_enabled
            self.horizontal_flip = horizontal_flip
            self.include_image = include_image
            self.batch_size = batch_size
            self.train_collate_batch_fn = train_collate_batch_fn
            self.eval_collate_batch_fn = eval_collate_batch_fn
            self.num_workers = num_workers

            BIG_ENOGUGH = 1000000
            dicom_ids = [None] * BIG_ENOGUGH
            image_paths = [None] * BIG_ENOGUGH
            report_ids = [None] * BIG_ENOGUGH
            orientations = [None] * BIG_ENOGUGH
            if use_all_data:
                all_indices = []
            elif use_test_set or use_chest_imagenome_bbox_gold_set or use_chest_imagenome_label_gold_set:
                test_indices = []
            else:
                train_indices = []
                val_indices = []
            idx = 0

            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            allowed_dicom_ids = _define_allowed_dicom_ids(view_mode, use_all_data, use_test_set,
                                      use_chest_imagenome_bbox_gold_set, use_chest_imagenome_label_gold_set,
                                      use_decent_images_only)
            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()

            max_idx_count = 0

            for rid, (part_id, subject_id, study_id, dicom_id_view_pairs, split) in \
                tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs'],
                    mimiccxr_metadata['splits']))):

                max_idx_count += len(dicom_id_view_pairs)

                for dicom_id, view in get_dicom_id_and_orientation_list(dicom_id_view_pairs, view_mode, allowed_dicom_ids):
                    dicom_ids[idx] = dicom_id
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    report_ids[idx] = rid
                    orientations[idx] = MIMICCXR_IMAGE_ORIENTATIONS.index(view)
                    if use_all_data:
                        all_indices.append(idx)
                    elif use_test_set or use_chest_imagenome_bbox_gold_set or use_chest_imagenome_label_gold_set:
                        if use_test_set:
                            if split == 'test':
                                test_indices.append(idx)
                        else:
                            test_indices.append(idx)
                    else:
                        if split == 'train':
                            train_indices.append(idx)
                        elif split == 'validate':
                            val_indices.append(idx)
                        elif split == 'test':
                            pass
                        else:
                            raise ValueError(f'Unknown split {split}')
                    idx += 1

            print('max_idx_count =', max_idx_count)
            print('actual_idx_count =', idx)
            if idx < max_idx_count:
                print(f'** NOTE: {max_idx_count - idx} images were skipped because they were not in the allowed DICOM IDs')
            
            self.dicom_ids = np.array(dicom_ids[:idx])
            self.image_paths = np.array(image_paths[:idx])
            self.report_ids = np.array(report_ids[:idx])
            self.orientations = np.array(orientations[:idx])
            if use_all_data:
                self.all_indices = np.array(all_indices)
                print(f'len(self.all_indices) = {len(self.all_indices)}')
            elif use_test_set or use_chest_imagenome_bbox_gold_set or use_chest_imagenome_label_gold_set:
                self.test_indices = np.array(test_indices)
                print(f'len(self.test_indices) = {len(self.test_indices)}')
            else:
                self.train_indices = np.array(train_indices)
                self.val_indices = np.array(val_indices)
                print(f'len(self.train_indices) = {len(self.train_indices)}')
                print(f'len(self.val_indices) = {len(self.val_indices)}')

            # Optional data to load
            self.classify_tags = classify_tags
            self.classify_orientation = classify_orientation
            self.classify_chexpert = classify_chexpert
            self.classify_questions = classify_questions
            self.classify_gender = classify_gender
            self.classify_chest_imagenome = classify_chest_imagenome
            self.predict_bboxes_chest_imagenome = predict_bboxes_chest_imagenome
            self.use_precomputed_visual_features = use_precomputed_visual_features
            self.use_merged_findings = use_merged_findings
            self.pass_pred_bbox_coords_to_model = pass_pred_bbox_coords_to_model
            self.use_gt_bboxes_as_pred = use_gt_bboxes_as_pred
            if pass_pred_bbox_coords_to_model:
                assert use_gt_bboxes_as_pred, 'Non-gt bboxes are not supported yet'
            
            if classify_tags:
                print('Loading medical tags per report...')
                assert medical_tags_per_report_filename is not None            
                medical_tags_per_report_path = os.path.join(MIMICCXR_CACHE_DIR, medical_tags_per_report_filename)
                self.rid2tags = load_pickle(medical_tags_per_report_path)
            else:
                self.rid2tags = None        
            
            if classify_chexpert:
                print('Loading CheXpert labels...')
                assert chexpert_labels_filename is not None
                chexpert_labels_path = os.path.join(MIMICCXR_CACHE_DIR, chexpert_labels_filename)
                self.chexpert_labels = get_cached_pickle_file(chexpert_labels_path)
                self.chexpert_labels = np.array(self.chexpert_labels)
            else:
                self.chexpert_labels = None
            
            if classify_questions:
                print('Loading question labels...')
                assert question_labels_filename is not None
                question_labels_path = os.path.join(MIMICCXR_CACHE_DIR, question_labels_filename)
                self.question_labels = get_cached_pickle_file(question_labels_path)
            else:
                self.question_labels = None

            if classify_gender:
                print('Loading Chest Imagenome genders...')
                dicomId2gender = get_dicomId2gender()
                def _get_gender_label(x):
                    if x == 'F': return 0
                    if x == 'M': return 1
                    assert np.isnan(x)
                    return 2
                self.genders = np.array([_get_gender_label(dicomId2gender[dicom_id]) for dicom_id in self.dicom_ids])
            else:
                self.genders = None

            if predict_labels_and_bboxes_chest_imagenome:
                assert classify_chest_imagenome
                assert chest_imagenome_label_names_filename is not None
                assert not use_anaxnet_bbox_subset # Not supported in this mode yet
                assert not use_chest_imagenome_bbox_gold_set # Not supported in this mode yet
            
            if classify_chest_imagenome:
                _load_chest_imagenome_labels(self, chest_imagenome_labels_filename, use_chest_imagenome_label_gold_set,
                                             chest_imagenome_label_names_filename)
            else:
                self.chest_imagenome_labels = None
            
            if predict_bboxes_chest_imagenome or (pass_pred_bbox_coords_to_model and use_gt_bboxes_as_pred):
                assert not use_chest_imagenome_bbox_gold_set, 'Not supported yet'
                print('Loading Chest Imagenome bounding boxes...')
                self.dicom_idxs, self.bbox_coords, self.bbox_presence =\
                    load_chest_imagenome_silver_bboxes_as_numpy_array(
                        self.dicom_ids, clamp_bboxes_chest_imagenome,
                        use_anaxnet_bbox_subset=use_anaxnet_bbox_subset)
                print(f'self.bbox_coords.shape = {self.bbox_coords.shape}')
                print(f'self.bbox_presence.shape = {self.bbox_presence.shape}')
                if horizontal_flip:
                    assert data_augmentation_enabled
                    print('Loading Chest Imagenome bounding boxes (flipped)...')
                    _, self.flipped_bbox_coords, self.flipped_bbox_presence =\
                        load_chest_imagenome_silver_bboxes_as_numpy_array(
                            self.dicom_ids, clamp_bboxes_chest_imagenome, flipped=True,
                            use_anaxnet_bbox_subset=use_anaxnet_bbox_subset)
                else:
                    self.flipped_bbox_coords = None
                    self.flipped_bbox_presence = None
                if pass_pred_bbox_coords_to_model and use_gt_bboxes_as_pred:
                    self.pred_bbox_coords = self.bbox_coords
                    if horizontal_flip:
                        self.flipped_pred_bbox_coords = self.flipped_bbox_coords
                    else:
                        self.flipped_pred_bbox_coords = None
                else:
                    self.pred_bbox_coords = None
                    self.flipped_pred_bbox_coords = None
            else:                
                self.dicom_idxs = None
                self.bbox_coords = None
                self.bbox_presence = None
                self.pred_bbox_coords = None
                self.flipped_bbox_coords = None
                self.flipped_bbox_presence = None
                self.flipped_pred_bbox_coords = None
            
            if use_precomputed_visual_features:
                print('Loading precomputed visual features...')
                assert precomputed_visual_features_path is not None
                features, idx2visfeatidx = load_precomputed_visual_features(
                    precomputed_visual_features_path, self.image_paths)
                self.precomputed_visual_features = features
                self.idx2visfeatidx = idx2visfeatidx
            else:
                self.precomputed_visual_features = None
                self.idx2visfeatidx = None
            
            if use_merged_findings:
                print('Loading merged findings...')
                assert findings_remapper is not None
                assert n_findings is not None
                assert classify_chexpert
                self.labels2mergedfindings = findings_remapper[CHEXPERT_DATASET_ID]
                self.finding_labels = adapt_label_matrix_as_merged_findings(
                    self.chexpert_labels, n_findings, self.labels2mergedfindings)
            else:
                self.labels2mergedfindings = None
                self.finding_labels = None

            if use_all_data:
                # Create all dataset and dataloader
                self.all_dataset, self.all_dataloader = self._create_dataset_and_dataloader(
                all_indices, test_image_transform, self.eval_collate_batch_fn)
            elif use_test_set or use_chest_imagenome_bbox_gold_set or use_chest_imagenome_label_gold_set:
                # Create test dataset and dataloader
                self.test_dataset, self.test_dataloader = self._create_dataset_and_dataloader(
                    self.test_indices, test_image_transform, self.eval_collate_batch_fn)
            else:
                if not use_val_set_only:
                    # Create train dataset and dataloader
                    self.balanced_batch_size = balanced_batch_size
                    self.train_dataset, self.train_dataloader = self._create_dataset_and_dataloader(
                        self.train_indices, train_image_transform, self.train_collate_batch_fn,
                        data_augmentation_enabled=self.data_augmentation_enabled,
                        shuffle=True, balanced_sampling_mode=balanced_sampling_mode)

                # Create validation dataset and dataloader
                self.val_dataset, self.val_dataloader = self._create_dataset_and_dataloader(
                    self.val_indices, val_image_transform, self.eval_collate_batch_fn)

    def _create_dataset(self, indices, image_transform, data_augmentation_enabled=False, shuffle=False, infinite=False):
        return MIMICCXR_Visual_Dataset(
            indices=indices,
            report_ids=self.report_ids,
            include_image=self.include_image,
            image_paths=self.image_paths,
            image_transform=image_transform,
            data_augmentation_enabled=data_augmentation_enabled,
            shuffle=shuffle,
            infinite=infinite,
            classify_tags=self.classify_tags,
            rid2tags=self.rid2tags,
            classify_orientation=self.classify_orientation,
            orientations=self.orientations,
            classify_chexpert=self.classify_chexpert,
            chexpert_labels=self.chexpert_labels,
            classify_questions=self.classify_questions,
            question_labels=self.question_labels,
            classify_gender=self.classify_gender,
            genders=self.genders,
            classify_chest_imagenome=self.classify_chest_imagenome,
            chest_imagenome_labels=self.chest_imagenome_labels,
            predict_bboxes_chest_imagenome=self.predict_bboxes_chest_imagenome,
            dicom_idxs=self.dicom_idxs,
            gt_bbox_coords=self.bbox_coords,
            gt_bbox_presence=self.bbox_presence,
            flipped_gt_bbox_coords=self.flipped_bbox_coords,
            flipped_gt_bbox_presence=self.flipped_bbox_presence,
            use_anaxnet_bbox_subset=self.use_anaxnet_bbox_subset,
            pass_pred_bbox_coords_to_model=self.pass_pred_bbox_coords_to_model,
            pred_bbox_coords=self.pred_bbox_coords,
            flipped_pred_bbox_coords=self.flipped_pred_bbox_coords,
            horizontal_flip=self.horizontal_flip,
            use_precomputed_visual_features=self.use_precomputed_visual_features,
            precomputed_visual_features=self.precomputed_visual_features,
            idx2visfeatidx=self.idx2visfeatidx,
            use_yolov8=self.use_yolov8,
            use_yolov11=self.use_yolov11,
        )
    
    def _create_dataset_and_dataloader(self, indices, image_transform, collate_batch_fn, data_augmentation_enabled=False,
                                       shuffle=False, balanced_sampling_mode=None):
        return _create_dataset_and_dataloader(self, indices, image_transform, collate_batch_fn, data_augmentation_enabled,
                                              shuffle, balanced_sampling_mode)

# MAE: Masked AutoEncoder
class MIMICCXR_MAE_Trainer(MAETrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers,
                qa_adapted_reports_filename, chexpert_labels_filename,
        ):
        self.transform = transform
        self.batch_size = batch_size
        self.collate_batch_fn = collate_batch_fn
        self.num_workers = num_workers
        
        chexpert_labels = load_pickle(os.path.join(MIMICCXR_CACHE_DIR, chexpert_labels_filename))
        qa_adapted_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))
        assert len(chexpert_labels) == len(qa_adapted_reports['reports'])

        image_views_dict = get_image_views_dict()
        broken_images = get_broken_images()
        split_dict = get_split_dict()

        report_ids = []
        image_paths = []
        train_indices = []
        val_indices = []
        test_indices = []
        idx = 0
        for rid, report in enumerate(qa_adapted_reports['reports']):
            part_id, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(report['filepath'])[0])
            views = image_views_dict[(subject_id, study_id)]
            report_split = None
            for dicom_id, _ in views:
                if (subject_id, study_id, dicom_id) not in broken_images:
                    report_ids.append(rid)
                    image_path = get_mimiccxr_small_image_path(part_id, subject_id, study_id, dicom_id)
                    image_paths.append(image_path)
                    split = split_dict[(subject_id, study_id, dicom_id)]
                    if report_split is None:
                        report_split = split
                    else:
                        assert report_split == split
                    if split == 'train':
                        train_indices.append(idx)
                    elif split == 'validate':
                        val_indices.append(idx)
                    elif split == 'test':
                        test_indices.append(idx)
                    else:
                        raise ValueError(f'Unknown split {split}')
                    idx += 1

        self.report_ids = np.array(report_ids, dtype=np.int32)
        self.image_paths = image_paths
        self.train_indices = np.array(train_indices, dtype=np.int32)
        self.val_indices = np.array(val_indices, dtype=np.int32)
        self.test_indices = np.array(test_indices, dtype=np.int32)

        labels_getter = lambda i: chexpert_labels[self.report_ids[i]]
        super().__init__(train_indices, val_indices, test_indices,
                        list(range(1, len(CHEXPERT_LABELS))), labels_getter,
                        batch_size, collate_batch_fn, num_workers)

    def _create_mae_dataset(self, indices, shuffle=True, infinite=False):
        return BasicImageDataset(self.image_paths, self.transform, indices, shuffle, infinite)

_detectron2_cache = {}

def get_detectron2_adapted_data(split_name,
    source_image_size_mode=MIMICCXR_ImageSizeModes.MEDIUM_512,
    use_chest_imagenome_gold_set=False,
    use_decent_images_only=True,
    clamp_bboxes_chest_imagenome=True,
    use_anaxnet_bbox_subset=False,
):
    if not use_chest_imagenome_gold_set:
        assert split_name in MIMICCXR_SPLIT_NAMES

    cache_name = f'{split_name}_{source_image_size_mode}_{use_chest_imagenome_gold_set}_{use_decent_images_only}_{clamp_bboxes_chest_imagenome}'
    if cache_name in _detectron2_cache:
        return _detectron2_cache[cache_name]

    if source_image_size_mode == MIMICCXR_ImageSizeModes.SMALL_256x256:
        image_path_getter = get_mimiccxr_small_image_path
    elif source_image_size_mode == MIMICCXR_ImageSizeModes.MEDIUM_512:
        image_path_getter = get_mimiccxr_medium_image_path
    else:
        raise ValueError(f'Unknown source image size mode: {source_image_size_mode}')
    print(f'Using source image size mode: {source_image_size_mode}')

    if use_chest_imagenome_gold_set:
        bboxes_dict = load_chest_imagenome_gold_bboxes()
    else:
        bboxes_dict = load_chest_imagenome_silver_bboxes()

    if use_chest_imagenome_gold_set:
        print('decent_images_only:', use_decent_images_only)
        allowed_dicom_ids = set(load_chest_imagenome_dicom_ids(decent_images_only=use_decent_images_only))
        print('len(allowed_dicom_ids) (before filtering by gold set):', len(allowed_dicom_ids))
        allowed_dicom_ids &= set(bboxes_dict.keys())
        print('len(allowed_dicom_ids) (after filtering by gold set):', len(allowed_dicom_ids))
    elif split_name == 'train' or split_name == 'validate':
        allowed_dicom_ids = set(load_chest_imagenome_dicom_ids(decent_images_only=use_decent_images_only))
        allowed_dicom_ids -= set(load_gold_standard_related_dicom_ids()) # remove gold standard images
    else:
        assert split_name == 'test'
        allowed_dicom_ids = set(load_chest_imagenome_dicom_ids(decent_images_only=use_decent_images_only))

    BIG_ENOUGH = 300000
    dataset_dicts = [None] * BIG_ENOUGH

    if use_anaxnet_bbox_subset:
        anaxnet_bbox_indices = get_anaxnet_bbox_sorted_indices()

    idx = 0
    max_idx_count = 0    
    mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()

    for part_id, subject_id, study_id, dicom_id_view_pairs, split in \
        tqdm(zip(mimiccxr_metadata['part_ids'],
            mimiccxr_metadata['subject_ids'],
            mimiccxr_metadata['study_ids'],
            mimiccxr_metadata['dicom_id_view_pos_pairs'],
            mimiccxr_metadata['splits']), mininterval=1, total=len(mimiccxr_metadata['part_ids'])):

        if not use_chest_imagenome_gold_set and split != split_name:
            continue

        max_idx_count += len(dicom_id_view_pairs)

        for dicom_id, _ in get_dicom_id_and_orientation_list(
                dicom_id_view_pairs, MIMICCXR_ViewModes.CHEST_IMAGENOME, allowed_dicom_ids):
            
            image_path = image_path_getter(part_id, subject_id, study_id, dicom_id)
            width, height = image_size_cache.get_image_size(image_path)
            bboxes = bboxes_dict[dicom_id]
            presence = bboxes['presence']
            coords = bboxes['coords']
            if clamp_bboxes_chest_imagenome:
                coords = np.clip(coords, 0, 1)
            # multiply by width and height to get absolute coordinates
            coords = coords.reshape(-1, 4)
            coords = coords * np.array([width, height, width, height], dtype=np.float32)
            if use_anaxnet_bbox_subset: # only use the 18 bboxes that AnaxNet uses
                coords = coords[anaxnet_bbox_indices]
                presence = presence[anaxnet_bbox_indices]
            dataset_dicts[idx] = {
                'file_name': image_path,
                'image_id': dicom_id,
                'height': height,
                'width': width,
                'coords': coords,
                'presence': presence,
            }
            idx += 1
    dataset_dicts = dataset_dicts[:idx]

    image_size_cache.update_cache_on_disk()

    print(f'max_idx_count: {max_idx_count}')
    print(f'idx: {idx}')
    if idx < max_idx_count:
        print(f'** NOTE: {max_idx_count - idx} images were skipped because they were not in the allowed DICOM IDs')

    _detectron2_cache[cache_name] = dataset_dicts

    return dataset_dicts

# DEBUG = True

class Detectron2AdaptedDataset(Dataset):
    def __init__(self, split_name, transform,
                source_image_size_mode=MIMICCXR_ImageSizeModes.MEDIUM_512,
                use_chest_imagenome_gold_set=False,
                use_decent_images_only=True,
                clamp_bboxes_chest_imagenome=True,
                data_augmentation_enabled=False,
                use_anaxnet_bbox_subset=False,
                ):
        from detectron2.structures import BoxMode
        from detectron2.data.detection_utils import annotations_to_instances
        self.box_mode = BoxMode.XYXY_ABS
        self.annotations_to_instances = annotations_to_instances
        self.split_name = split_name
        self.source_image_size_mode = source_image_size_mode
        self.use_chest_imagenome_gold_set = use_chest_imagenome_gold_set
        self.use_decent_images_only = use_decent_images_only
        self.clamp_bboxes_chest_imagenome = clamp_bboxes_chest_imagenome
        self.transform = transform
        self.data_augmentation_enabled = data_augmentation_enabled
        self.use_anaxnet_bbox_subset = use_anaxnet_bbox_subset
        self.dataset_dicts = get_detectron2_adapted_data(
            split_name, source_image_size_mode, use_chest_imagenome_gold_set, use_decent_images_only,
            clamp_bboxes_chest_imagenome, use_anaxnet_bbox_subset=use_anaxnet_bbox_subset)
        if self.data_augmentation_enabled:
            if use_anaxnet_bbox_subset:
                num_bbox_classes = CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES
            else:
                num_bbox_classes = CHEST_IMAGENOME_NUM_BBOX_CLASSES
            self.albumentation_adapter = _AlbumentationAdapter(num_bbox_classes)
    
    def __getitem__(self, idx):
        item = self.dataset_dicts[idx]
        width = item['width']
        height = item['height']
        image_path = item['file_name']
        bbox_coords = item['coords']
        bbox_presence = item['presence']
        # Albumentation section
        if self.data_augmentation_enabled:
            image, bbox_coords, bbox_presence = self.transform(
                image_path, bbox_coords, bbox_presence, self.albumentation_adapter)
        else:
            image = self.transform(image_path)
        assert image.shape[2] == width
        assert image.shape[1] == height
        # Detectron2 section
        annotations = []
        for i in range(len(bbox_presence)):
            if bbox_presence[i] == 1:
                annotations.append({
                    'bbox': bbox_coords[i].tolist(),
                    'bbox_mode': self.box_mode,
                    'category_id': i,
                })
        instances = self.annotations_to_instances(annotations, image.shape[-2:])
        output = {
            'image': image,
            'height': height,
            'width': width,
            'bbox_coords': bbox_coords,
            'bbox_presence': bbox_presence,
            'instances': instances,
        }
        # global DEBUG
        # if DEBUG:
        #     DEBUG = False
        #     print(output)
        return output

    # # Used for debugging
    # def __getitem__(self, idx):
    #     item = self.dataset_dicts[idx]
    #     width = item['width']
    #     height = item['height']
    #     image_path = item['file_name']
    #     bbox_coords = item['coords']
    #     bbox_presence = item['presence']
    #     # Detectron2 section
    #     annotations = []
    #     for i in range(len(bbox_presence)):
    #         if bbox_presence[i] == 1:
    #             x0, y0, x1, y1 = bbox_coords[4*i:4*i+4]
    #             x0 *= width
    #             x1 *= width
    #             y0 *= height
    #             y1 *= height
    #             annotations.append({
    #                 'bbox': [x0, y0, x1, y1],
    #                 'bbox_mode': BoxMode.XYXY_ABS,
    #                 'category_id': i,
    #             })
    #     return {
    #         'file_name': image_path,
    #         'image_id': item['image_id'],
    #         'height': height,
    #         'width': width,
    #         'annotations': annotations,
    #     }
    
    def __len__(self):
        return len(self.dataset_dicts)