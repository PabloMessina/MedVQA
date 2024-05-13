import os
import numpy as np
from medvqa.datasets.iuxray import get_iuxray_image_path
from medvqa.datasets.multimodal import MultiModal_Trainer
from medvqa.datasets.iuxray import (
    IUXRAY_IMAGE_INFO_JSON_PATH,
    IUXRAY_IMAGE_ORIENTATIONS,
    IUXRAY_REPORTS_MIN_JSON_PATH,
    IUXRAY_CACHE_DIR,
)
from medvqa.utils.files import (
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
    return get_file_path_with_hashing_if_too_long(IUXRAY_CACHE_DIR, 'iuxray_preprocessed_multimodal_train_data__', strings)

class IUXRAY_Multimodal_Trainer(MultiModal_Trainer):

    def __init__(self, transform, batch_size, collate_batch_fn,
                num_workers,
                qa_adapted_reports_filename,
                tokenizer = None,
                use_text = True,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                iuxray_qa_reports = None,
                imbalance_reduction_coef = 0.4,
                ):

        if use_text:
            assert tokenizer is not None, 'Tokenizer must be provided if use_text is True'
        self.tokenizer = tokenizer
        self.iuxray_qa_reports = iuxray_qa_reports
        self.qa_adapted_reports_filename = qa_adapted_reports_filename        
        
        preprocessed_data_path = _get_train_preprocessed_data_path(qa_adapted_reports_filename, tokenizer)

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessed_data_path,
                        IUXRAY_CACHE_DIR,
                        num_workers,
                        use_text = use_text,
                        classify_orientation = classify_orientation,
                        classify_chexpert = classify_chexpert,
                        chexpert_labels_filename = chexpert_labels_filename,
                        classify_questions = classify_questions,
                        question_labels_filename = question_labels_filename,                                                
                        imbalance_reduction_coef = imbalance_reduction_coef,
                        include_test=False)

    def _preprocess_data(self):

        tokenizer = self.tokenizer
        iuxray_qa_reports = self.iuxray_qa_reports

        print(f'Loading {IUXRAY_REPORTS_MIN_JSON_PATH}')
        iuxray_metadata = get_cached_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)
        
        print(f'Loading {IUXRAY_IMAGE_INFO_JSON_PATH}')
        iuxray_image_info = get_cached_json_file(IUXRAY_IMAGE_INFO_JSON_PATH)

        if iuxray_qa_reports is None:
            file_path = os.path.join(IUXRAY_CACHE_DIR, self.qa_adapted_reports_filename)
            print(f'Loading {file_path}')
            iuxray_qa_reports = get_cached_json_file(file_path)
        
        self.report_ids = []
        self.images = []
        self.backgrounds = []
        self.orientations = []
        self.train_indices = []

        print('Preprocessing IU X-ray dataset ...')
        
        invalid_images = set()
        invalid_images.update(iuxray_image_info['marks']['wrong'])
        invalid_images.update(iuxray_image_info['marks']['broken'])

        reports_ids = range(len(iuxray_qa_reports['reports']))

        for ri in reports_ids:
            report = iuxray_qa_reports['reports'][ri]
            metadata = iuxray_metadata[report['filename']]
            images = metadata['images']            
            
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

            if image_path:
                orientation_id = IUXRAY_IMAGE_ORIENTATIONS.index(iuxray_image_info['classification'][image_name])
                background = report['background']
                background = tokenizer.tokenize("" if background is None else background)
                self.report_ids.append(ri)                
                self.images.append(image_path)
                self.backgrounds.append(background)
                self.orientations.append(orientation_id)
        
        self.report_ids = np.array(self.report_ids, dtype=int)
        self.images = np.array(self.images, dtype=str)        
        self.backgrounds = np.array(self.backgrounds, dtype=object)
        self.orientations = np.array(self.orientations, dtype=int)
        self.train_indices = np.array(list(range(len(self.report_ids))), dtype=int)