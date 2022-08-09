import os
import numpy as np
from medvqa.datasets.vqa import VQA_Trainer
from medvqa.datasets.iuxray import (
    IUXRAY_DATASET_DIR,
    IUXRAY_CACHE_DIR,
    IUXRAY_REPORTS_MIN_JSON_PATH,
    IUXRAY_IMAGE_INFO_JSON_PATH,
    IUXRAY_IMAGE_ORIENTATIONS,
)
from medvqa.utils.files import (
    load_pickle,
    save_to_pickle,
    get_cached_json_file,
    MAX_FILENAME_LENGTH,
)
from medvqa.utils.hashing import hash_string
from medvqa.utils.constants import ReportEvalMode


_IUXRAY_IMAGE_PATH_TEMPLATE = os.path.join(IUXRAY_DATASET_DIR, 'images', '{}')

def _get_iuxray_image_path(image_name):
    return _IUXRAY_IMAGE_PATH_TEMPLATE.format(image_name)

def get_iuxray_image_paths(report):
    filename = report['filename']
    iuxray_metadata = get_cached_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)
    metadata = iuxray_metadata[filename]
    images = metadata['images']
    image_paths = [_get_iuxray_image_path(f'{img["id"]}.png') for img in images]
    return image_paths

def _get_train_preprocessing_save_path(qa_adapted_reports_filename, split_kwargs, tokenizer,
                                       balanced_metadata_filename=None,
                                       chexpert_labels_filename=None,
                                       ignore_medical_tokenization=False):    
    
    split_params_string = f'({",".join(str(split_kwargs[k]) for k in sorted(list(split_kwargs.keys())))})'
    tokenizer_string = f'{tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}'
    if tokenizer.medical_tokenization and not ignore_medical_tokenization:
        tokenizer_string += f',{tokenizer.medical_terms_frequency_filename}'
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'split_params={split_params_string}',
        f'tokenizer={tokenizer_string}',
    ]
    if balanced_metadata_filename:
        strings.append(f'balanced_metadata={balanced_metadata_filename}')
    if chexpert_labels_filename:
        strings.append(f'chexpert_labels={chexpert_labels_filename}')
    merged_string = ";".join(strings)
    final_path = os.path.join(IUXRAY_CACHE_DIR, f'iuxray_preprocessed_train_data__({merged_string}).pkl')
    if len(final_path) > MAX_FILENAME_LENGTH:
        h = hash_string(merged_string)
        final_path = os.path.join(IUXRAY_CACHE_DIR, f'iuxray_preprocessed_train_data__(hash={h[0]},{h[1]}).pkl')
    return final_path

def _get_report_eval_mode_preprocessing_save_path(qa_adapted_reports_filename, load_split_from_path,
                                                  tokenizer, report_eval_mode):    
    tokenizer_string = f'{tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}'
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'load_split_from_path={load_split_from_path}',
        f'tokenizer={tokenizer_string}',
        f'report_eval_mode={report_eval_mode}'
    ]
    merged_string = ";".join(strings)
    final_path = os.path.join(IUXRAY_CACHE_DIR, f'report_eval_mode_preprocessed_data__({merged_string}).pkl')
    if len(final_path) > MAX_FILENAME_LENGTH:
        h = hash_string(merged_string)
        final_path = os.path.join(IUXRAY_CACHE_DIR, f'report_eval_mode_preprocessed_data__(hash={h[0]},{h[1]}).pkl')
    return final_path

def _precompute_questions_per_report(load_split_from_path, report_eval_mode, iuxray_qa_reports, report_ids):

    file_path = os.path.join(IUXRAY_CACHE_DIR,
        f'questions_per_report(load_split_from_path={load_split_from_path};report_eval_mode={report_eval_mode}).pkl')
    if len(file_path) > MAX_FILENAME_LENGTH:
        h = hash_string(file_path)
        file_path = os.path.join(IUXRAY_CACHE_DIR, f'questions_per_report(hash={h[0]},{h[1]}).pkl')

    data = load_pickle(file_path)
    if data is not None:
        print('questions per report data loaded from', file_path)
        return data

    data = {}

    if report_eval_mode == ReportEvalMode.GROUND_TRUTH.value:
        for ri in report_ids:
            report = iuxray_qa_reports['reports'][ri]
            data[ri] = report['question_ids']
    else:
        assert False, f'Unknown report_eval_mode = {report_eval_mode}'
    
    save_to_pickle(data, file_path)
    print('questions per report data saved to', file_path)
    
    return data

class IUXRAY_VQA_Trainer(VQA_Trainer):

    def __init__(self, transform, batch_size, collate_batch_fn,
                num_workers,
                qa_adapted_reports_filename,
                split_kwargs,
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
                balanced_split = False,
                balanced_dataloading = False,
                balanced_metadata_filename = None,
                imbalance_reduction_coef = 1,
                validation_only = False,
                report_eval_mode = None,
                ignore_medical_tokenization = False,
                allowed_questions = None,
                one_question_per_batch = False,
                include_chexpert_mode = False,
                use_chexpert_mode_only = False,
                chexpert_one_hot_offset = None,
                include_image = True,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = False):

        self.tokenizer = tokenizer
        self.iuxray_metadata = iuxray_metadata
        self.iuxray_image_info = iuxray_image_info
        self.iuxray_qa_reports = iuxray_qa_reports
        self.qa_adapted_reports_filename = qa_adapted_reports_filename
        self.ignore_medical_tokenization = ignore_medical_tokenization
        self.report_eval_mode = report_eval_mode

        if report_eval_mode is not None:
            assert validation_only            
            load_split_from_path = _get_train_preprocessing_save_path(
                            qa_adapted_reports_filename, split_kwargs, tokenizer, balanced_metadata_filename,
                            chexpert_labels_filename if balanced_split else None)
            preprocessing_save_path = _get_report_eval_mode_preprocessing_save_path(
                            qa_adapted_reports_filename, load_split_from_path, tokenizer, report_eval_mode)
            self.load_split_from_path = load_split_from_path
        else:
            if ignore_medical_tokenization:        
                preprocessing_save_path = _get_train_preprocessing_save_path(
                                qa_adapted_reports_filename, split_kwargs, tokenizer, balanced_metadata_filename,
                                chexpert_labels_filename if balanced_split else None,
                                ignore_medical_tokenization=True)
                load_split_from_path = _get_train_preprocessing_save_path(
                                qa_adapted_reports_filename, split_kwargs, tokenizer, balanced_metadata_filename,
                                chexpert_labels_filename if balanced_split else None)
            else:
                preprocessing_save_path = _get_train_preprocessing_save_path(
                                qa_adapted_reports_filename, split_kwargs, tokenizer, balanced_metadata_filename,
                                chexpert_labels_filename if balanced_split else None)
                load_split_from_path = None

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
                        dataset_name = 'IU X-Ray',
                        split_kwargs = split_kwargs,
                        load_split_from_path = load_split_from_path,
                        balanced_split = balanced_split,
                        balanced_dataloading = balanced_dataloading,
                        balanced_metadata_filename = balanced_metadata_filename,
                        imbalance_reduction_coef = imbalance_reduction_coef,
                        validation_only = validation_only,
                        include_answer = report_eval_mode == None,
                        use_report_eval_mode = report_eval_mode != None,
                        allowed_questions = allowed_questions,
                        qa_adapted_reports_filename = qa_adapted_reports_filename,
                        one_question_per_batch = one_question_per_batch,
                        include_chexpert_mode = include_chexpert_mode,
                        use_chexpert_mode_only = use_chexpert_mode_only,
                        chexpert_one_hot_offset = chexpert_one_hot_offset,
                        include_image = include_image,
                        use_precomputed_visual_features = use_precomputed_visual_features,
                        precomputed_visual_features_path = precomputed_visual_features_path,
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
        
        if self.report_eval_mode is not None:
            questions_per_report = _precompute_questions_per_report(self.load_split_from_path, self.report_eval_mode,
                                                                    iuxray_qa_reports, self.val_report_ids)
        else:
            questions_per_report = None

        print('Preprocessing IU X-ray vqa dataset ...')

        question_list = iuxray_qa_reports['questions']
        
        invalid_images = set()
        invalid_images.update(iuxray_image_info['marks']['wrong'])
        invalid_images.update(iuxray_image_info['marks']['broken'])

        reports_ids = questions_per_report.keys() if questions_per_report is not None else range(len(iuxray_qa_reports['reports']))

        for ri in reports_ids:
            report = iuxray_qa_reports['reports'][ri]
            metadata = iuxray_metadata[report['filename']]
            images = metadata['images']
            sentences = report['sentences']
            
            image_path = None
            
            if len(images) == 1:
                image_name = f'{images[0]["id"]}.png'
                if image_name not in invalid_images:
                    image_path = _get_iuxray_image_path(image_name)
            else:
                for elem in images:
                    image_name = f'{elem["id"]}.png'
                    if image_name not in invalid_images and iuxray_image_info['classification'][image_name] == 'frontal':
                        image_path = _get_iuxray_image_path(image_name)
                        break

            # print(f'ri={ri}, images={images}, image_path={image_path}')

            if image_path:
                orientation_id = IUXRAY_IMAGE_ORIENTATIONS.index(iuxray_image_info['classification'][image_name])
                if questions_per_report is None:
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
                else:
                    question_ids = questions_per_report[ri]
                    for qid in question_ids:
                        question = question_list[qid]
                        self.report_ids.append(ri)
                        self.question_ids.append(qid)
                        self.images.append(image_path)
                        self.questions.append(tokenizer.string2ids(question.lower()))
                        self.orientations.append(orientation_id)
        
        self.report_ids = np.array(self.report_ids, dtype=int)
        self.question_ids = np.array(self.question_ids, dtype=int)
        self.images = np.array(self.images, dtype=str)
        self.questions = np.array(self.questions, dtype=object)
        self.answers = np.array(self.answers, dtype=object)
        self.orientations = np.array(self.orientations, dtype=int)