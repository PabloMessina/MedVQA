import os
from medvqa.utils.files import load_json_file
from medvqa.datasets.vqa import VQA_Trainer
from medvqa.datasets.iuxray import (
    IUXRAY_DATASET_DIR,
    IUXRAY_CACHE_DIR,
    IUXRAY_REPORTS_MIN_JSON_PATH,
    IUXRAY_IMAGE_INFO_JSON_PATH,
)

_IUXRAY_IMAGE_PATH_TEMPLATE = os.path.join(IUXRAY_DATASET_DIR, 'images', '{}')

def _get_iuxray_image_path(image_name):
    return _IUXRAY_IMAGE_PATH_TEMPLATE.format(image_name)

def _get_train_preprocessing_save_path(qa_adapted_reports_filename, split_kwargs, tokenizer):
    m = split_kwargs['min_train_examples_per_question']
    n = split_kwargs['n_val_examples_per_question']
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'split_params=({m},{n})',
        f'tokenizer={tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}',
    ]
    return os.path.join('iuxray',
            f'iuxray_preprocessed_train_data__({";".join(strings)}).pkl')

class IUXRAY_VQA_Trainer(VQA_Trainer):

    def __init__(self, transform, batch_size, collate_batch_fn,
                qa_adapted_reports_filename,
                use_tags = False,
                medical_tags_per_report_filename = None,
                split_kwargs = None,
                tokenizer = None,
                iuxray_metadata = None,
                iuxray_image_info = None,
                iuxray_qa_reports = None):

        self.tokenizer = tokenizer
        self.iuxray_metadata = iuxray_metadata
        self.iuxray_image_info = iuxray_image_info
        self.iuxray_qa_reports = iuxray_qa_reports
        self.qa_adapted_reports_filename = qa_adapted_reports_filename
        
        preprocessing_save_path = _get_train_preprocessing_save_path(
                        qa_adapted_reports_filename, split_kwargs, tokenizer)

        rid2tags_path = os.path.join(IUXRAY_CACHE_DIR, medical_tags_per_report_filename) if use_tags else None

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessing_save_path,
                        use_tags = use_tags,
                        rid2tags_path = rid2tags_path,
                        dataset_name = 'IU X-Ray',
                        split_kwargs = split_kwargs)
        
    def _preprocess_data(self):

        tokenizer = self.tokenizer
        iuxray_metadata = self.iuxray_metadata
        iuxray_image_info = self.iuxray_image_info
        iuxray_qa_reports = self.iuxray_qa_reports

        if iuxray_qa_reports is None:
            file_path = os.path.join(IUXRAY_CACHE_DIR, self.qa_adapted_reports_filename)
            print(f'Loading {file_path}')
            iuxray_qa_reports = load_json_file(file_path)
        if iuxray_metadata is None:
            print(f'Loading {IUXRAY_REPORTS_MIN_JSON_PATH}')
            iuxray_metadata = load_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)
        if iuxray_image_info is None:
            print(f'Loading {IUXRAY_IMAGE_INFO_JSON_PATH}')
            iuxray_image_info = load_json_file(IUXRAY_IMAGE_INFO_JSON_PATH)
        
        self.report_ids = []
        self.question_ids = []        
        self.images = []
        self.questions = []
        self.answers = []                       
        
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
                for q_idx, a_idxs in report['qa'].items():
                    q_idx = int(q_idx)
                    question = question_list[q_idx]
                    answer = '. '.join(sentences[i] for i in a_idxs)
                    self.report_ids.append(ri)
                    self.question_ids.append(q_idx)
                    self.images.append(image_path)
                    self.questions.append(tokenizer.string2ids(question.lower()))
                    self.answers.append(tokenizer.string2ids(answer.lower()))