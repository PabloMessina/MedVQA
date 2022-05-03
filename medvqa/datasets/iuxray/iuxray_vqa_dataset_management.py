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
from medvqa.utils.files import get_cached_json_file, MAX_FILENAME_LENGTH
from medvqa.utils.hashing import hash_string

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
    merged_string = ";".join(strings)
    final_path = os.path.join(IUXRAY_CACHE_DIR, f'iuxray_preprocessed_train_data__({merged_string}).pkl')
    if len(final_path) > MAX_FILENAME_LENGTH:
        h = hash_string(merged_string)
        final_path = os.path.join(IUXRAY_CACHE_DIR, f'iuxray_preprocessed_train_data__(hash={h[0]},{h[1]}).pkl')
    return final_path

class IUXRAY_VQA_Trainer(VQA_Trainer):

    def __init__(self, transform, batch_size, collate_batch_fn,
                num_workers,
                qa_adapted_reports_filename,
                split_kwargs,
                tokenizer,
                use_tags = False,
                medical_tags_per_report_filename = None,
                use_orientation = False,
                use_chexpert = False,
                chexpert_labels_filename = None,
                iuxray_metadata = None,
                iuxray_image_info = None,
                iuxray_qa_reports = None,
                balanced_split = False,
                balanced_dataloading = False,
                balanced_metadata_filename = None,
                validation_only = False,
                ignore_medical_tokenization = False):

        self.tokenizer = tokenizer
        self.iuxray_metadata = iuxray_metadata
        self.iuxray_image_info = iuxray_image_info
        self.iuxray_qa_reports = iuxray_qa_reports
        self.qa_adapted_reports_filename = qa_adapted_reports_filename

        if ignore_medical_tokenization:        
            preprocessing_save_path = _get_train_preprocessing_save_path(
                            qa_adapted_reports_filename, split_kwargs, tokenizer, balanced_metadata_filename,
                            ignore_medical_tokenization=True)
            load_split_from_path = _get_train_preprocessing_save_path(
                            qa_adapted_reports_filename, split_kwargs, tokenizer, balanced_metadata_filename)
        else:
            preprocessing_save_path = _get_train_preprocessing_save_path(
                            qa_adapted_reports_filename, split_kwargs, tokenizer, balanced_metadata_filename,
                            ignore_medical_tokenization=True)
            load_split_from_path = None

        rid2tags_path = os.path.join(IUXRAY_CACHE_DIR, medical_tags_per_report_filename) if use_tags else None
        chexpert_labels_path = os.path.join(IUXRAY_CACHE_DIR, chexpert_labels_filename) if use_chexpert else None
        balanced_metadata_path = os.path.join(IUXRAY_CACHE_DIR, balanced_metadata_filename) if balanced_split else None

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessing_save_path,
                        num_workers,
                        use_tags = use_tags,
                        rid2tags_path = rid2tags_path,
                        use_orientation = use_orientation,
                        use_chexpert = use_chexpert,
                        chexpert_labels_path = chexpert_labels_path,
                        dataset_name = 'IU X-Ray',
                        split_kwargs = split_kwargs,
                        load_split_from_path = load_split_from_path,
                        balanced_split = balanced_split,
                        balanced_dataloading = balanced_dataloading,
                        balanced_metadata_path = balanced_metadata_path,
                        validation_only = validation_only)
        
    def _preprocess_data(self):

        tokenizer = self.tokenizer
        iuxray_metadata = self.iuxray_metadata
        iuxray_image_info = self.iuxray_image_info
        iuxray_qa_reports = self.iuxray_qa_reports

        if tokenizer.medical_tokenization:
            answer_string2ids_func = tokenizer.strig2medical_tag_ids
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
        
        print('loading IU X-ray vqa dataset ...')

        question_list = iuxray_qa_reports['questions']
        
        invalid_images = set()
        invalid_images.update(iuxray_image_info['marks']['wrong'])
        invalid_images.update(iuxray_image_info['marks']['broken'])
        
        for ri, report in enumerate(iuxray_qa_reports['reports']):
            
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

            if image_path:
                orientation_id = IUXRAY_IMAGE_ORIENTATIONS.index(iuxray_image_info['classification'][image_name])
                for q_idx, a_idxs in report['qa'].items():
                    q_idx = int(q_idx)
                    question = question_list[q_idx]
                    answer = '. '.join(sentences[i] for i in a_idxs)
                    self.report_ids.append(ri)
                    self.question_ids.append(q_idx)
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