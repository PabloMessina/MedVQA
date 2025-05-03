import os
from medvqa.datasets.iuxray import get_iuxray_image_path
from medvqa.datasets.visual_module import BasicImageDataset, MAETrainerBase, VM_Trainer
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR, IUXRAY_REPORTS_MIN_JSON_PATH, get_invalid_images
from medvqa.utils.constants import CHEXPERT_LABELS
from medvqa.utils.files_utils import get_cached_json_file, load_pickle

class IUXRAY_VisualModuleTrainer(VM_Trainer):

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
                **unused_kwargs,
        ):
        preprocessed_data_path = os.path.join(IUXRAY_CACHE_DIR, preprocessed_data_filename)
        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessed_data_path,
                        IUXRAY_CACHE_DIR,
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

class IUXRAY_MAE_Trainer(MAETrainerBase):
    def __init__(self, qa_adapted_reports_filename, chexpert_labels_filename, transform, batch_size, collate_batch_fn, num_workers):

        iuxray_qa_reports = get_cached_json_file(os.path.join(IUXRAY_CACHE_DIR, qa_adapted_reports_filename))
        iuxray_metadata = get_cached_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)
        chexpert_labels = load_pickle(os.path.join(IUXRAY_CACHE_DIR, chexpert_labels_filename))
        
        report_ids = []
        image_paths = []

        invalid_images = get_invalid_images()

        for ri in range(len(iuxray_qa_reports['reports'])):
            report = iuxray_qa_reports['reports'][ri]
            metadata = iuxray_metadata[report['filename']]
            images = metadata['images']

            for elem in images:
                image_name = f'{elem["id"]}.png'
                if image_name not in invalid_images:
                    image_path = get_iuxray_image_path(image_name)
                    report_ids.append(ri)
                    image_paths.append(image_path)

        train_indices = list(range(len(report_ids)))

        self.report_ids = report_ids
        self.image_paths = image_paths
        self.train_indices = train_indices
        self.transform = transform

        labels_getter = lambda i: chexpert_labels[report_ids[i]]
        super().__init__(train_indices, None, None, list(range(1, len(CHEXPERT_LABELS))),
                         labels_getter, batch_size, collate_batch_fn, num_workers,
                         use_validation_set=False)

    def _create_mae_dataset(self, indices, shuffle=True, infinite=False):
        return BasicImageDataset(self.image_paths, self.transform, indices, shuffle, infinite)