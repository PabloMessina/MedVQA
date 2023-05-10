import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_NUM_BBOX_CLASSES
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    get_dicomId2gender,
    load_chest_imagenome_silver_bboxes_as_numpy_array,
    load_chest_imagenome_label_names,
)
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import (
    _AlbumentationAdapter,
    _BalancedSamplingMode,
    _create_dataset_and_dataloader,
    _define_allowed_dicom_ids,
    _load_chest_imagenome_labels,
)
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_IMAGE_ORIENTATIONS,
    MIMICCXR_ImageSizeModes,
    MIMICCXR_ViewModes,
    get_dicom_id_and_orientation_list,
    get_image_path_getter,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.files import get_cached_pickle_file
from medvqa.utils.logging import print_bold, print_magenta, print_orange

class MIMICCXR_Image2Report_Dataset(Dataset):
    def __init__(self, indices, reports, report_ids, image_paths, image_transform,
                data_augmentation_enabled=False,
                shuffle=False,
                infinite=False,
                # aux task: genders
                classify_gender=False,
                genders=None,
                # aux task: chexpert labels
                classify_chexpert=False,
                chexpert_labels=None,
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
                horizontal_flip=False,
                # yolov8
                use_yolov8=False,
            ):
        self.indices = indices
        self.reports = reports
        self.report_ids = report_ids
        self.data_augmentation_enabled = data_augmentation_enabled    
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.classify_gender = classify_gender
        self.genders = genders
        self.classify_chexpert = classify_chexpert
        self.chexpert_labels = chexpert_labels
        self.classify_chest_imagenome = classify_chest_imagenome
        self.chest_imagenome_labels = chest_imagenome_labels
        self.predict_bboxes_chest_imagenome = predict_bboxes_chest_imagenome
        self.dicom_idxs = dicom_idxs
        self.gt_bbox_coords = gt_bbox_coords
        self.gt_bbox_presence = gt_bbox_presence
        self.horizontal_flip = horizontal_flip
        self.flipped_gt_bbox_coords = flipped_gt_bbox_coords
        self.flipped_gt_bbox_presence = flipped_gt_bbox_presence
        self.use_yolov8 = use_yolov8

        if self.predict_bboxes_chest_imagenome:
            assert self.gt_bbox_coords is not None
            assert self.gt_bbox_presence is not None

        if self.predict_bboxes_chest_imagenome:
            if self.data_augmentation_enabled:
                if horizontal_flip:
                    if self.predict_bboxes_chest_imagenome:
                        assert flipped_gt_bbox_coords is not None
                        assert flipped_gt_bbox_presence is not None
                num_bbox_classes = CHEST_IMAGENOME_NUM_BBOX_CLASSES
                self.albumentation_adapter = _AlbumentationAdapter(num_bbox_classes)
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
        output = { 'idx': idx, 'report': self.reports[rid] }
        image_path = self.image_paths[idx]
        # handle transform differently for chest imagenome bboxes
        if self.predict_bboxes_chest_imagenome: 
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
                        return_image_size=self.use_yolov8,
                    )
                else:
                    tmp = self.image_transform(
                        image_path=image_path,
                        bboxes=gt_bbox_coords,
                        presence=gt_bbox_presence,
                        albumentation_adapter=self.albumentation_adapter,
                        return_image_size=self.use_yolov8,
                    )
                if self.use_yolov8:
                    image, gt_bbox_coords, gt_bbox_presence, image_size_before, image_size_after = tmp
                else:
                    image, gt_bbox_coords, gt_bbox_presence = tmp
            else: # no data augmentation
                tmp = self.image_transform(image_path, return_image_size=self.use_yolov8)
                if self.use_yolov8:
                    image, image_size_before, image_size_after = tmp
                else:
                    image = tmp
            assert len(gt_bbox_coords.shape) == 2
            output['chest_imagenome_bbox_coords'] = gt_bbox_coords
            output['chest_imagenome_bbox_presence'] = gt_bbox_presence
        elif self.use_yolov8:
            image, image_size_before, image_size_after = self.image_transform(image_path, return_image_size=True)
        else:
            image = self.image_transform(image_path)
        output['i'] = image
        if self.classify_gender:
            output['gender'] = self.genders[idx]
        if self.classify_chexpert:
            output['chexpert'] = self.chexpert_labels[rid]
        if self.classify_chest_imagenome:
            output['chest_imagenome'] = self.chest_imagenome_labels[rid]
        if self.use_yolov8:
            # We need to adapt the output a little bit to make it compatible with YOLOv8
            output['im_file'] = image_path
            output['ori_shape'] = image_size_before
            output['resized_shape'] = image_size_after
            output['img'] = output['i']
            del output['i']
        return output

class MIMICCXR_Image2ReportTrainer():

    def __init__(self, 
                qa_adapted_reports_filename, tokenizer, batch_size, collate_batch_fn, num_workers,
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
                source_image_size_mode=MIMICCXR_ImageSizeModes.SMALL_256x256,
                view_mode=MIMICCXR_ViewModes.ANY_SINGLE,
                use_decent_images_only=False,
                classify_gender=False,
                classify_chexpert=False,
                chexpert_labels_filename=None,
                classify_chest_imagenome=False,
                predict_bboxes_chest_imagenome=False,
                predict_labels_and_bboxes_chest_imagenome=False,
                clamp_bboxes_chest_imagenome=False,
                chest_imagenome_labels_filename=None,
                chest_imagenome_label_names_filename=None,
                balanced_sampling_mode=None,
                balanced_batch_size=None,
                use_yolov8=False,
                **unused_kwargs,
            ):

        train_collate_batch_fn = lambda batch: collate_batch_fn(batch, training_mode=True)
        eval_collate_batch_fn = lambda batch: collate_batch_fn(batch, training_mode=False)

        if len(unused_kwargs) > 0:
            # Print warning in orange and bold
            print_orange('Warning: unused kwargs in MIMICCXR_Image2ReportTrainer: {}'.format(unused_kwargs), bold=True)
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

        self.use_yolov8 = use_yolov8

        if chest_imagenome_label_names_filename is not None:
            self.chest_imagenome_label_names = load_chest_imagenome_label_names(chest_imagenome_label_names_filename)
        elif chest_imagenome_labels_filename is not None:
            self.chest_imagenome_label_names = load_chest_imagenome_label_names(
                chest_imagenome_labels_filename.replace('imageId2labels', 'labels'))
        else:
            self.chest_imagenome_label_names = None
        
        self.tokenizer = tokenizer
        self.qa_adapted_reports_filename = qa_adapted_reports_filename        
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.test_image_transform = test_image_transform
        self.data_augmentation_enabled = data_augmentation_enabled
        self.horizontal_flip = horizontal_flip
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

        mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata(qa_adapted_reports_filename, exclude_invalid_sentences=True)
        reports = [None] * len(mimiccxr_metadata['reports'])

        max_idx_count = 0

        if use_test_set or use_chest_imagenome_label_gold_set:
            tokenize_func = tokenizer.string2ids
        else:
            tokenize_func = tokenizer.tokenize

        for rid, (part_id, subject_id, study_id, report, dicom_id_view_pairs, split) in \
            tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                mimiccxr_metadata['subject_ids'],
                mimiccxr_metadata['study_ids'],
                mimiccxr_metadata['reports'],
                mimiccxr_metadata['dicom_id_view_pos_pairs'],
                mimiccxr_metadata['splits'])), mininterval=2):

            max_idx_count += len(dicom_id_view_pairs)
            reports[rid] = tokenize_func(report)

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

        # Print a random report to make sure the tokenization is correct
        random_report = random.choice(reports)
        print_bold('Random report:')
        print_magenta(tokenizer.ids2string(random_report), bold=True)
        
        self.reports = reports
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
        self.classify_gender = classify_gender
        self.classify_chexpert = classify_chexpert
        self.classify_chest_imagenome = classify_chest_imagenome
        self.predict_bboxes_chest_imagenome = predict_bboxes_chest_imagenome
        
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

        if classify_chexpert or balanced_sampling_mode in (
            _BalancedSamplingMode.BALANCED_CHEXPERT_LABELS,
            _BalancedSamplingMode.BALANCED_CHEXPERT_LABELS_BATCHWISE,
        ):
            print('Loading CheXpert labels...')
            assert chexpert_labels_filename is not None
            chexpert_labels_path = os.path.join(MIMICCXR_CACHE_DIR, chexpert_labels_filename)
            self.chexpert_labels = get_cached_pickle_file(chexpert_labels_path)
            self.chexpert_labels = np.array(self.chexpert_labels)
        else:
            assert chexpert_labels_filename is None
            self.chexpert_labels = None

        if predict_labels_and_bboxes_chest_imagenome:
            assert classify_chest_imagenome
            assert chest_imagenome_label_names_filename is not None
            assert not use_chest_imagenome_bbox_gold_set # Not supported in this mode yet
        
        if classify_chest_imagenome or balanced_sampling_mode in (
            _BalancedSamplingMode.BALANCED_CHEST_IMAGENOME_GLOBAL_LABELS,
            _BalancedSamplingMode.BALANCED_CHEST_IMAGENOME_GLOBAL_LABELS_BATCHWISE,
        ):
            _load_chest_imagenome_labels(self, chest_imagenome_labels_filename, use_chest_imagenome_label_gold_set,
                                            chest_imagenome_label_names_filename)
        else:
            self.chest_imagenome_labels = None
        
        if predict_bboxes_chest_imagenome:
            assert not use_chest_imagenome_bbox_gold_set, 'Not supported yet'
            print('Loading Chest Imagenome bounding boxes...')
            self.dicom_idxs, self.bbox_coords, self.bbox_presence =\
                load_chest_imagenome_silver_bboxes_as_numpy_array(self.dicom_ids, clamp_bboxes_chest_imagenome)
            print(f'self.bbox_coords.shape = {self.bbox_coords.shape}')
            print(f'self.bbox_presence.shape = {self.bbox_presence.shape}')
            if horizontal_flip:
                assert data_augmentation_enabled
                print('Loading Chest Imagenome bounding boxes (flipped)...')
                _, self.flipped_bbox_coords, self.flipped_bbox_presence =\
                    load_chest_imagenome_silver_bboxes_as_numpy_array(
                        self.dicom_ids, clamp_bboxes_chest_imagenome, flipped=True)
            else:
                self.flipped_bbox_coords = None
                self.flipped_bbox_presence = None
        else:                
            self.dicom_idxs = None
            self.bbox_coords = None
            self.bbox_presence = None
            self.flipped_bbox_coords = None
            self.flipped_bbox_presence = None

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
        return MIMICCXR_Image2Report_Dataset(
            indices=indices,
            reports=self.reports,
            report_ids=self.report_ids,
            image_paths=self.image_paths,
            image_transform=image_transform,
            data_augmentation_enabled=data_augmentation_enabled,
            shuffle=shuffle,
            infinite=infinite,
            classify_gender=self.classify_gender,
            genders=self.genders,
            classify_chexpert=self.classify_chexpert,
            chexpert_labels=self.chexpert_labels,
            classify_chest_imagenome=self.classify_chest_imagenome,
            chest_imagenome_labels=self.chest_imagenome_labels,
            predict_bboxes_chest_imagenome=self.predict_bboxes_chest_imagenome,
            dicom_idxs=self.dicom_idxs,
            gt_bbox_coords=self.bbox_coords,
            gt_bbox_presence=self.bbox_presence,
            flipped_gt_bbox_coords=self.flipped_bbox_coords,
            flipped_gt_bbox_presence=self.flipped_bbox_presence,
            horizontal_flip=self.horizontal_flip,
            use_yolov8=self.use_yolov8,
        )
    
    def _create_dataset_and_dataloader(self, indices, image_transform, collate_batch_fn, data_augmentation_enabled=False,
                                       shuffle=False, balanced_sampling_mode=None):
        return _create_dataset_and_dataloader(self, indices, image_transform, collate_batch_fn, data_augmentation_enabled,
                                              shuffle, balanced_sampling_mode)