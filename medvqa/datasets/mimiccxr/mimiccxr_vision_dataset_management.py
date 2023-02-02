import os
import numpy as np
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix
from medvqa.datasets.visual_module import BasicImageDataset, MAETrainerBase, VM_Trainer, VM_Evaluator
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_STUDY_REGEX,
    get_broken_images,
    get_image_views_dict,
    get_mimiccxr_small_image_path,
    get_split_dict,
)
from medvqa.utils.constants import CHEXPERT_LABELS
from medvqa.utils.files import get_cached_json_file, load_pickle

class MIMICCXR_VisualModuleTrainer(VM_Trainer):

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
                imbalance_reduction_coef = 1,
                validation_only = False,
                one_question_per_batch = False,
                question_balanced = False,
        ):
        preprocessed_data_path = os.path.join(MIMICCXR_CACHE_DIR, preprocessed_data_filename)
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
                        imbalance_reduction_coef = imbalance_reduction_coef,
                        validation_only = validation_only,
                        one_question_per_batch = one_question_per_batch,
                        question_balanced = question_balanced)

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



