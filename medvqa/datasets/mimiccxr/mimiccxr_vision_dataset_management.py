import os
from medvqa.datasets.visual_module import VM_Trainer, VM_Evaluator
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR

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
                **unused_kwargs,
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
                        question_labels_filename = question_labels_filename)