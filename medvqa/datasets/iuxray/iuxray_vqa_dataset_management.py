import os
import numpy as np
from medvqa.datasets.vqa import VQA_Trainer
from medvqa.datasets.iuxray import (
    IUXRAY_CACHE_DIR,
    IUXRAY_REPORTS_MIN_JSON_PATH,
    IUXRAY_IMAGE_INFO_JSON_PATH,
    IUXRAY_IMAGE_ORIENTATIONS,
    get_invalid_images,
    get_iuxray_image_path,
)
from medvqa.utils.files import (
    get_file_path_with_hashing_if_too_long,
    get_cached_json_file,
)

def get_iuxray_image_paths(report):
    filename = report['filename']
    iuxray_metadata = get_cached_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)
    metadata = iuxray_metadata[filename]
    images = metadata['images']
    image_paths = [get_iuxray_image_path(img["id"]) for img in images]
    return image_paths

def _get_train_preprocessing_save_path(qa_adapted_reports_filename, tokenizer,
                                       ignore_medical_tokenization=False):    
    
    tokenizer_string = f'{tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}'
    if tokenizer.medical_tokenization and not ignore_medical_tokenization:
        tokenizer_string += f',{tokenizer.medical_terms_frequency_filename}'
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'tokenizer={tokenizer_string}',
    ]
    return get_file_path_with_hashing_if_too_long(IUXRAY_CACHE_DIR, 'iuxray_preprocessed_train_data__', strings, 'pkl')

class IUXRAY_VQA_Trainer(VQA_Trainer):

    def __init__(self, transform, batch_size, collate_batch_fn,
                num_workers,
                qa_adapted_reports_filename,
                tokenizer,
                collate_batch_fn_chexpert_mode = None,
                verbose_question = True,
                classify_tags = False,
                medical_tags_per_report_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                iuxray_metadata = None,
                iuxray_image_info = None,
                iuxray_qa_reports = None,
                balanced_dataloading = False,
                balanced_metadata_filename = None,
                imbalance_reduction_coef = 1,
                ignore_medical_tokenization = False,
                allowed_questions = None,
                one_question_per_batch = False,
                include_chexpert_mode = False,
                use_chexpert_mode_only = False,
                chexpert_one_hot_offset = None,
                include_image = True,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = False,
                use_merged_findings = False,
                findings_remapper = None,
                n_findings = None,
                ):

        self.tokenizer = tokenizer
        self.iuxray_metadata = iuxray_metadata
        self.iuxray_image_info = iuxray_image_info
        self.iuxray_qa_reports = iuxray_qa_reports
        self.qa_adapted_reports_filename = qa_adapted_reports_filename
        self.ignore_medical_tokenization = ignore_medical_tokenization
        
        preprocessing_save_path = _get_train_preprocessing_save_path(
                        qa_adapted_reports_filename, tokenizer, ignore_medical_tokenization)

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessing_save_path,
                        IUXRAY_CACHE_DIR,
                        num_workers,
                        collate_batch_fn_chexpert_mode = collate_batch_fn_chexpert_mode,
                        verbose_question = verbose_question,
                        classify_tags = classify_tags,
                        rid2tags_filename = medical_tags_per_report_filename,
                        classify_orientation = classify_orientation,
                        classify_chexpert = classify_chexpert,
                        chexpert_labels_filename = chexpert_labels_filename,
                        classify_questions = classify_questions,
                        question_labels_filename = question_labels_filename,
                        balanced_dataloading = balanced_dataloading,
                        balanced_metadata_filename = balanced_metadata_filename,
                        imbalance_reduction_coef = imbalance_reduction_coef,
                        allowed_questions = allowed_questions,
                        qa_adapted_reports_filename = qa_adapted_reports_filename,
                        one_question_per_batch = one_question_per_batch,
                        include_chexpert_mode = include_chexpert_mode,
                        use_chexpert_mode_only = use_chexpert_mode_only,
                        chexpert_one_hot_offset = chexpert_one_hot_offset,
                        include_image = include_image,
                        use_precomputed_visual_features = use_precomputed_visual_features,
                        precomputed_visual_features_path = precomputed_visual_features_path,
                        use_merged_findings = use_merged_findings,
                        findings_remapper = findings_remapper,
                        n_findings = n_findings,
                        train_with_all=True,
                        )
        
    def _preprocess_data(self):

        tokenizer = self.tokenizer
        iuxray_metadata = self.iuxray_metadata
        iuxray_image_info = self.iuxray_image_info
        iuxray_qa_reports = self.iuxray_qa_reports

        if tokenizer.medical_tokenization and not self.ignore_medical_tokenization:
            answer_string2ids_func = tokenizer.string2medical_tag_ids
        else:
            answer_string2ids_func = tokenizer.string2ids

        if iuxray_qa_reports is None:
            file_path = os.path.join(IUXRAY_CACHE_DIR, self.qa_adapted_reports_filename)
            print(f'Loading {file_path}')
            iuxray_qa_reports = get_cached_json_file(file_path)
        if iuxray_metadata is None:
            print(f'Loading {IUXRAY_REPORTS_MIN_JSON_PATH}')
            iuxray_metadata = get_cached_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)
        if iuxray_image_info is None:
            print(f'Loading {IUXRAY_IMAGE_INFO_JSON_PATH}')
            iuxray_image_info = get_cached_json_file(IUXRAY_IMAGE_INFO_JSON_PATH)
        
        self.report_ids = []
        self.question_ids = []        
        self.images = []
        self.questions = []
        self.answers = []
        self.orientations = []

        print('Preprocessing IU X-ray vqa dataset ...')

        question_list = iuxray_qa_reports['questions']

        invalid_images = get_invalid_images()

        reports_ids = range(len(iuxray_qa_reports['reports']))

        for ri in reports_ids:
            report = iuxray_qa_reports['reports'][ri]
            metadata = iuxray_metadata[report['filename']]
            images = metadata['images']
            sentences = report['sentences']
            
            image_path = None

            for elem in images:
                image_name = f'{elem["id"]}.png'
                if image_name not in invalid_images and iuxray_image_info['classification'][image_name] == 'frontal':
                    image_path = get_iuxray_image_path(image_name)
                    break

            if image_path is None:
                for elem in images:
                    image_name = f'{elem["id"]}.png'
                    if image_name not in invalid_images:
                        image_path = get_iuxray_image_path(image_name)
                        break

            # print(f'ri={ri}, images={images}, image_path={image_path}')

            if image_path:
                orientation_id = IUXRAY_IMAGE_ORIENTATIONS.index(iuxray_image_info['classification'][image_name])                
                for qid, a_ids in report['qa'].items():
                    qid = int(qid)
                    question = question_list[qid]
                    answer = '. '.join(sentences[i] for i in a_ids)
                    self.report_ids.append(ri)
                    self.question_ids.append(qid)
                    self.images.append(image_path)
                    self.questions.append(tokenizer.string2ids(question.lower()))
                    self.answers.append(answer_string2ids_func(answer.lower()))
                    self.orientations.append(orientation_id)                
        
        self.report_ids = np.array(self.report_ids, dtype=int)
        self.question_ids = np.array(self.question_ids, dtype=int)
        self.images = np.array(self.images, dtype=str)
        self.questions = np.array(self.questions, dtype=object)
        self.answers = np.array(self.answers, dtype=object)
        self.orientations = np.array(self.orientations, dtype=int)