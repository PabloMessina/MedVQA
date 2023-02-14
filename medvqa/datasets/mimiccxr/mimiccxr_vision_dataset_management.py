import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix,
    load_chest_imagenome_silver_bboxes_as_numpy_array,
    load_nongold_dicom_ids,
)
from medvqa.datasets.utils import adapt_label_matrix_as_merged_findings
from medvqa.datasets.visual_module import (
    BasicImageDataset,
    MAETrainerBase,
    VM_Evaluator,
)
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_IMAGE_ORIENTATIONS,
    MIMICCXR_STUDY_REGEX,
    MIMICCXR_ViewModes,
    get_broken_images,
    get_dicom_id_and_orientation_list,
    get_image_views_dict,
    get_mimiccxr_small_image_path,
    get_split_dict,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.datasets.vqa import load_precomputed_visual_features
from medvqa.utils.constants import CHEXPERT_DATASET_ID, CHEXPERT_LABELS
from medvqa.utils.files import get_cached_json_file, get_cached_pickle_file, load_pickle

class MIMICCXR_Visual_Dataset(Dataset):
    def __init__(self, indices, report_ids,
                include_image=True,
                image_paths=None,
                image_transform=None,
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
                # aux task: chest imagenome labels
                classify_chest_imagenome=False,
                chest_imagenome_labels=None,
                # aux task: chest imagenome bboxes
                predict_bboxes_chest_imagenome=False,
                dicom_idxs=None,
                bbox_coords=None,
                bbox_presence=None,
                # precomputed visual features
                use_precomputed_visual_features=False,
                precomputed_visual_features=None,
                idx2visfeatidx=None,
            ):
        self.indices = indices
        self.report_ids = report_ids
        self.include_image = include_image        
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
        self.classify_chest_imagenome = classify_chest_imagenome
        self.chest_imagenome_labels = chest_imagenome_labels
        self.predict_bboxes_chest_imagenome = predict_bboxes_chest_imagenome
        self.dicom_idxs = dicom_idxs
        self.bbox_coords = bbox_coords
        self.bbox_presence = bbox_presence
        self.use_precomputed_visual_features = use_precomputed_visual_features
        self.precomputed_visual_features = precomputed_visual_features
        self.idx2visfeatidx = idx2visfeatidx
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        rid = self.report_ids[idx]
        output = { 'idx': idx }
        if self.include_image:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            output['i'] = image
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
        if self.classify_chest_imagenome:
            output['chest_imagenome'] = self.chest_imagenome_labels[rid]
        if self.predict_bboxes_chest_imagenome:
            dicom_idx = self.dicom_idxs[idx]
            output['chest_imagenome_bbox_coords'] = self.bbox_coords[dicom_idx]
            output['chest_imagenome_bbox_presence'] = self.bbox_presence[dicom_idx]
        return output

class MIMICCXR_VisualModuleTrainer():

    def __init__(self, train_image_transform, val_image_transform, 
                batch_size, collate_batch_fn,
                num_workers,
                qa_adapted_reports_filename,
                include_image=True,
                view_mode = MIMICCXR_ViewModes.ANY_SINGLE,                
                classify_tags = False,
                medical_tags_per_report_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                classify_chest_imagenome = False,
                predict_bboxes_chest_imagenome = False,
                clamp_bboxes_chest_imagenome = False,
                chest_imagenome_labels_filename = None,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = None,
                use_merged_findings = False,
                findings_remapper = None,
                n_findings = None,                
            ):

        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.include_image = include_image
        self.batch_size = batch_size
        self.collate_batch_fn = collate_batch_fn
        self.num_workers = num_workers
        
        assert qa_adapted_reports_filename is not None
        mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata(qa_adapted_reports_filename)
        
        if view_mode == MIMICCXR_ViewModes.CHEST_IMAGENOME:
            chest_imagenome_nongold_dicom_ids = load_nongold_dicom_ids()
            chest_imagenome_nongold_dicom_ids = set(chest_imagenome_nongold_dicom_ids)
            print(f'Loaded {len(chest_imagenome_nongold_dicom_ids)} non-gold DICOM IDs from Chest Imagenome')
        else:
            chest_imagenome_nongold_dicom_ids = None

        BIG_ENOGUGH = 1000000
        dicom_ids = [None] * BIG_ENOGUGH
        image_paths = [None] * BIG_ENOGUGH
        report_ids = [None] * BIG_ENOGUGH
        orientations = [None] * BIG_ENOGUGH
        train_indices = []
        val_indices = []
        idx = 0

        for rid, (part_id, subject_id, study_id, dicom_id_view_pairs, split) in \
            tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                mimiccxr_metadata['subject_ids'],
                mimiccxr_metadata['study_ids'],
                mimiccxr_metadata['dicom_id_view_pos_pairs'],
                mimiccxr_metadata['splits']))):

            for dicom_id, view in get_dicom_id_and_orientation_list(
                    dicom_id_view_pairs, view_mode, chest_imagenome_nongold_dicom_ids):
                dicom_ids[idx] = dicom_id
                image_paths[idx] = get_mimiccxr_small_image_path(part_id, subject_id, study_id, dicom_id)
                report_ids[idx] = rid
                orientations[idx] = MIMICCXR_IMAGE_ORIENTATIONS.index(view)
                if split == 'train':
                    train_indices.append(idx)
                elif split == 'validate':
                    val_indices.append(idx)
                elif split == 'test':
                    pass
                else:
                    raise ValueError(f'Unknown split {split}')
                idx += 1
        
        self.dicom_ids = np.array(dicom_ids[:idx])
        self.image_paths = np.array(image_paths[:idx])
        self.report_ids = np.array(report_ids[:idx])
        self.orientations = np.array(orientations[:idx])
        self.train_indices = np.array(train_indices)
        self.val_indices = np.array(val_indices)

        # Optional data to load
        self.classify_tags = classify_tags
        self.classify_orientation = classify_orientation
        self.classify_chexpert = classify_chexpert
        self.classify_questions = classify_questions
        self.classify_chest_imagenome = classify_chest_imagenome
        self.predict_bboxes_chest_imagenome = predict_bboxes_chest_imagenome
        self.use_precomputed_visual_features = use_precomputed_visual_features
        self.use_merged_findings = use_merged_findings
        
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
        
        if classify_chest_imagenome:
            print('Loading Chest Imagenome labels...')
            assert chest_imagenome_labels_filename is not None
            assert qa_adapted_reports_filename is not None
            _, self.chest_imagenome_labels = \
                load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix(
                    chest_imagenome_labels_filename, qa_adapted_reports_filename)
        else:
            self.chest_imagenome_labels = None
        
        if predict_bboxes_chest_imagenome:
            print('Loading Chest Imagenome bounding boxes...')
            self.dicom_idxs, self.bbox_coords, self.bbox_presence =\
                load_chest_imagenome_silver_bboxes_as_numpy_array(
                    self.dicom_ids, clamp_bboxes_chest_imagenome)
        else:
            self.dicom_idxs = None
            self.bbox_coords = None
            self.bbox_presence = None
        
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
        
        # Create train dataset and dataloader
        self.train_dataset, self.train_dataloader = self._create_dataset_and_dataloader(
            self.train_indices, train_image_transform, shuffle=True)

        # Create validation dataset and dataloader
        self.val_dataset, self.val_dataloader = self._create_dataset_and_dataloader(
            self.val_indices, val_image_transform, shuffle=False)

    def _create_dataset_and_dataloader(self, indices, image_transform, shuffle=True):
        dataset = MIMICCXR_Visual_Dataset(
            indices=indices,
            report_ids=self.report_ids,
            include_image=self.include_image,
            image_paths=self.image_paths,
            image_transform=image_transform,
            classify_tags=self.classify_tags,
            rid2tags=self.rid2tags,
            classify_orientation=self.classify_orientation,
            orientations=self.orientations,
            classify_chexpert=self.classify_chexpert,
            chexpert_labels=self.chexpert_labels,
            classify_questions=self.classify_questions,
            question_labels=self.question_labels,
            classify_chest_imagenome=self.classify_chest_imagenome,
            chest_imagenome_labels=self.chest_imagenome_labels,
            predict_bboxes_chest_imagenome=self.predict_bboxes_chest_imagenome,
            dicom_idxs=self.dicom_idxs,
            bbox_coords=self.bbox_coords,
            bbox_presence=self.bbox_presence,
            use_precomputed_visual_features=self.use_precomputed_visual_features,
            precomputed_visual_features=self.precomputed_visual_features,
            idx2visfeatidx=self.idx2visfeatidx,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_batch_fn,
            pin_memory=True,
        )
        return dataset, dataloader

class MIMICCXR_VisualModuleEvaluator(VM_Evaluator):

    def __init__(self, transform, batch_size, collate_batch_fn,
                num_workers,
                preprocessed_data_filename,
                classify_tags = False,
                rid2tags_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                classify_chest_imagenome = False,
                chest_imagenome_labels_filename = None,
                qa_adapted_reports_filename = None,
                use_validation_indices = False,
                **unused_kwargs,
        ):
        preprocessed_data_path = os.path.join(MIMICCXR_CACHE_DIR, preprocessed_data_filename)
        
        if classify_chest_imagenome: # Chest-Imagenome specific logic
            assert chest_imagenome_labels_filename is not None
            assert qa_adapted_reports_filename is not None
            # Load Chest-Imagenome dicom_ids and labels
            self.chest_imagenome_dicom_ids, self.chest_imagenome_labels = \
                load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix(
                    chest_imagenome_labels_filename, qa_adapted_reports_filename)
            # Necessary hack so that parent classes can access chest_imagenome_labels
            other_tasks = [('chest_imagenome', lambda _, rid: self.chest_imagenome_labels[rid])]
        else:
            other_tasks = None

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessed_data_path,
                        MIMICCXR_CACHE_DIR,
                        num_workers,
                        classify_tags = classify_tags,
                        rid2tags_filename = rid2tags_filename,
                        classify_orientation = classify_orientation,
                        classify_chexpert = classify_chexpert,
                        chexpert_labels_filename = chexpert_labels_filename,
                        classify_questions = classify_questions,
                        question_labels_filename = question_labels_filename,
                        use_validation_indices = use_validation_indices,
                        other_tasks = other_tasks)

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



