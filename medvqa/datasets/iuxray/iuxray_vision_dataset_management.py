import os
from medvqa.datasets.visual_module import VM_Trainer
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR

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
                        one_question_per_batch = one_question_per_batch)