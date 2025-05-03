import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from medvqa.datasets.multimodal import MultiModal_Trainer
from medvqa.datasets.mimiccxr import (
    MIMICCXR_BROKEN_IMAGES,
    MIMICCXR_CACHE_DIR,
    MIMICCXR_IMAGE_REGEX,
    MIMICCXR_METADATA_CSV_PATH,
    MIMICCXR_SPLIT_CSV_PATH,
    MIMICCXR_IMAGE_ORIENTATIONS,
    MIMICCXR_STUDY_REGEX,
    choose_dicom_id_and_orientation,
    get_mimiccxr_image_path,
)
from medvqa.utils.files_utils import (
    get_cached_json_file,
    get_file_path_with_hashing_if_too_long,
)

def _get_train_preprocessed_data_path(qa_adapted_reports_filename, tokenizer):    
    strings = [
        f'dataset={qa_adapted_reports_filename}',
    ]
    if tokenizer is not None:
        tokenizer_string = f'{tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}'
        strings.append(f'tokenizer={tokenizer_string}')
    return get_file_path_with_hashing_if_too_long(MIMICCXR_CACHE_DIR, 'mimiccxr_preprocessed_multimodal_train_data__', strings)

def _get_orientation_id(orientation):
    try:
        return MIMICCXR_IMAGE_ORIENTATIONS.index(orientation)
    except ValueError:
        return 0

class MIMICCXR_Multimodal_Trainer(MultiModal_Trainer):

    def __init__(self, transform, batch_size, collate_batch_fn,
                num_workers,
                qa_adapted_reports_filename,
                use_text = True,
                tokenizer = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                mimiccxr_qa_reports = None,
                mimiccxr_metadata = None,
                mimiccxr_split = None,
                include_train = True,
                imbalance_reduction_coef = 0.4):
        
        if use_text:
            assert tokenizer is not None, 'If use_text is True, tokenizer must be provided'
        self.tokenizer = tokenizer
        self.mimiccxr_qa_reports = mimiccxr_qa_reports
        self.mimiccxr_metadata = mimiccxr_metadata
        self.mimiccxr_split = mimiccxr_split
        self.qa_adapted_reports_filename = qa_adapted_reports_filename        
        
        preprocessed_data_path = _get_train_preprocessed_data_path(qa_adapted_reports_filename, tokenizer)

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessed_data_path,
                        MIMICCXR_CACHE_DIR,
                        num_workers,
                        use_text=use_text,
                        classify_orientation = classify_orientation,
                        classify_chexpert = classify_chexpert,
                        chexpert_labels_filename = chexpert_labels_filename,
                        classify_questions = classify_questions,
                        question_labels_filename = question_labels_filename,                                                
                        imbalance_reduction_coef = imbalance_reduction_coef,
                        include_train = include_train)

    def _preprocess_data(self):
    
        tokenizer = self.tokenizer
        mimiccxr_qa_reports = self.mimiccxr_qa_reports
        mimiccxr_metadata = self.mimiccxr_metadata
        mimiccxr_split = self.mimiccxr_split
        qa_adapted_reports_filename = self.qa_adapted_reports_filename

        if mimiccxr_qa_reports is None:
            file_path = os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename)
            print(f'Loading {file_path}')
            mimiccxr_qa_reports = get_cached_json_file(file_path)
        if mimiccxr_metadata is None:
            print(f'Loading {MIMICCXR_METADATA_CSV_PATH}')
            mimiccxr_metadata = pd.read_csv(MIMICCXR_METADATA_CSV_PATH)
        if mimiccxr_split is None:
            print(f'Loading {MIMICCXR_SPLIT_CSV_PATH}')
            mimiccxr_split = pd.read_csv(MIMICCXR_SPLIT_CSV_PATH)
        
        print('Reading MIMIC-CXR splits ...')
        
        split_dict = { (sub_id, stud_id, dicom_id) : split for sub_id, stud_id, dicom_id, split in zip(mimiccxr_split['subject_id'],
                                                                                                        mimiccxr_split['study_id'],
                                                                                                        mimiccxr_split['dicom_id'],
                                                                                                        mimiccxr_split['split']) }        
        print('Reading MIMIC-CXR metadata ...')
        
        image_views_dict = dict()
        for subject_id, study_id, dicom_id, view_pos in zip(mimiccxr_metadata['subject_id'],
                                                            mimiccxr_metadata['study_id'],
                                                            mimiccxr_metadata['dicom_id'],
                                                            mimiccxr_metadata['ViewPosition']):
            key = (subject_id, study_id)
            try:
                views = image_views_dict[key]
            except KeyError:
                views = image_views_dict[key] = []
            views.append((dicom_id, view_pos))
            
        broken_images = set()
        for path in MIMICCXR_BROKEN_IMAGES:
            _, a, b, c = MIMICCXR_IMAGE_REGEX.findall(path)[0]
            broken_images.add((int(a), int(b), c))

        mimiccxr_qa_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))

        self.report_ids = []
        self.images = []
        self.orientations = []
        self.train_indices = []
        self.test_indices = []
        if self.use_text:
            self.backgrounds = []
        
        idx = 0

        for ri, report in tqdm(enumerate(mimiccxr_qa_reports['reports'])):

            part_id, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(report['filepath'])[0])
            views = image_views_dict[(subject_id, study_id)]
            
            dicom_id, orientation = choose_dicom_id_and_orientation(views)
                
            if (dicom_id is not None and (subject_id, study_id, dicom_id) not in broken_images):            
                
                image_path = get_mimiccxr_image_path(part_id, subject_id, study_id, dicom_id)
                orientation_id = _get_orientation_id(orientation)
                self.report_ids.append(ri)
                self.images.append(image_path)
                self.orientations.append(orientation_id)
                if self.use_text:
                    background = tokenizer.tokenize(report['background'])
                    self.backgrounds.append(background)

                if split_dict[(subject_id, study_id, dicom_id)] == 'test':
                    self.test_indices.append(idx)
                else:
                    self.train_indices.append(idx)

                idx += 1
            
        self.report_ids = np.array(self.report_ids, dtype=int)    
        self.images = np.array(self.images, dtype=str)
        self.orientations = np.array(self.orientations, dtype=int)
        self.train_indices = np.array(self.train_indices, dtype=int)
        self.test_indices = np.array(self.test_indices, dtype=int)
        if self.use_text:
            self.backgrounds = np.array(self.backgrounds, dtype=object)