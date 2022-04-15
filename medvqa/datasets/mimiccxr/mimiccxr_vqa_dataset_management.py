import os
import re
import pandas as pd
from tqdm import tqdm
from medvqa.datasets.vqa import VQA_Evaluator, VQA_Trainer
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_JPG_IMAGES_SMALL_DIR,
    MIMICCXR_METADATA_CSV_PATH,
    MIMICCXR_SPLIT_CSV_PATH,
    MIMICCXR_IMAGE_ORIENTATIONS,
)
from medvqa.utils.files import load_json_file

_MIMICCXR_IMAGE_PATH_TEMPLATE = os.path.join(MIMICCXR_JPG_IMAGES_SMALL_DIR, 'p{}', 'p{}', 's{}', '{}.jpg')
_MIMICCXR_STUDY_REGEX = re.compile(r'/p(\d+)/p(\d+)/s(\d+)\.txt')
_MIMICCXR_IMAGE_REGEX = re.compile(r'p(\d+)/p(\d+)/s(\d+)/(.*)\.jpg$')
_MIMICCXR_BROKEN_IMAGES = set([
    'p11/p11285576/s54979966/03b2e67c-70631ff8-685825fb-6c989456-621ca64d.jpg',
    'p15/p15223781/s52459604/56b8afd3-5f6d4419-8699d79e-6913a2bd-35a08557.jpg',
    'p15/p15223781/s52459604/93020995-6b84ca33-2e41e00d-5d6e3bee-87cfe5c6.jpg',
    # Appears empty
    'p10/p10291098/s57194260/0539ee33-9d402e49-a9cc6d36-7aabc539-3d80a62b.jpg',
    # Blur empty images
    'p15/p15355458/s52423703/0b6f08b2-72deda00-d7ccc375-8278269f-b4e11c36.jpg',
    'p18/p18461911/s57183218/151abebe-2a750a5c-09c181bb-1a9016ef-92d8910e.jpg',
    'p19/p19839145/s54889255/f674e474-817bb713-8f16c90c-608cf869-2829cae7.jpg',
])

def _get_mimiccxr_image_path(part_id, subject_id, study_id, dicom_id):
    return _MIMICCXR_IMAGE_PATH_TEMPLATE.format(part_id, subject_id, study_id, dicom_id)

def _get_train_preprocessing_save_path(qa_adapted_reports_filename, split_kwargs, tokenizer,
                                       balanced_metadata_filename = None):
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'split_params={tuple(split_kwargs[k] for k in sorted(list(split_kwargs.keys())))}',
        f'tokenizer={tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}',
    ]
    if balanced_metadata_filename:
        strings.append(f'balanced_metadata={balanced_metadata_filename}')
    return os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_preprocessed_train_data__({";".join(strings)}).pkl')

def _get_test_preprocessing_save_path(qa_adapted_reports_filename, tokenizer):
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'tokenizer={tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}',
    ]
    return os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_preprocessed_test_data__({";".join(strings)}).pkl')

def _get_orientation_id(orientation):
    try:
        return MIMICCXR_IMAGE_ORIENTATIONS.index(orientation)
    except ValueError:
        return 0

def _preprocess_data(self, qa_adapted_reports_filename, split_lambda):

    tokenizer = self.tokenizer
    mimiccxr_qa_reports = self.mimiccxr_qa_reports
    mimiccxr_metadata = self.mimiccxr_metadata
    mimiccxr_split = self.mimiccxr_split

    if mimiccxr_qa_reports is None:
        file_path = os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename)
        print(f'Loading {file_path}')
        mimiccxr_qa_reports = load_json_file(file_path)
    if mimiccxr_metadata is None:
        print(f'Loading {MIMICCXR_METADATA_CSV_PATH}')
        mimiccxr_metadata = pd.read_csv(MIMICCXR_METADATA_CSV_PATH)
    if mimiccxr_split is None:
        print(f'Loading {MIMICCXR_SPLIT_CSV_PATH}')
        mimiccxr_split = pd.read_csv(MIMICCXR_SPLIT_CSV_PATH)
    
    self.report_ids = []
    self.question_ids = []
    self.images = []
    self.questions = []
    self.answers = []
    self.orientations = []
    
    print('reading MIMIC-CXR splits ...')
    
    split_dict = { (sub_id, stud_id, dicom_id) : split for sub_id, stud_id, dicom_id, split in zip(mimiccxr_split['subject_id'],
                                                                                                    mimiccxr_split['study_id'],
                                                                                                    mimiccxr_split['dicom_id'],
                                                                                                    mimiccxr_split['split']) }        
    print('reading MIMIC-CXR metadata ...')
    
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
    
    print('preprocessing MIMIC-CXR vqa dataset ...')

    broken_images = set()
    for path in _MIMICCXR_BROKEN_IMAGES:
        _, a, b, c = _MIMICCXR_IMAGE_REGEX.findall(path)[0]
        broken_images.add((int(a), int(b), c))
    
    question_list = mimiccxr_qa_reports['questions']
    
    for ri, report in tqdm(enumerate(mimiccxr_qa_reports['reports'])):
        
        sentences = report['sentences']
        part_id, subject_id, study_id = map(int, _MIMICCXR_STUDY_REGEX.findall(report['filepath'])[0])
        views = image_views_dict[(subject_id, study_id)]
        # images = glob.glob(f'/mnt/workspace/mimic-cxr-jpg/images-small/p{part_id}/p{subject_id}/s{study_id}/*.jpg')
        # assert len(views) == len(images)
        
        dicom_id = None
        if len(views) == 1:
            dicom_id = views[0][0]
            orientation = views[0][1]
        else:
            for view in views:
                if view[1] == 'PA' or view[1] == 'AP':
                    dicom_id = view[0]
                    orientation = view[1]
                    break
            
        if (dicom_id is not None and split_lambda(split_dict[(subject_id, study_id, dicom_id)]) and
                (subject_id, study_id, dicom_id) not in broken_images):
            image_path = _get_mimiccxr_image_path(part_id, subject_id, study_id, dicom_id)
            orientation_id = _get_orientation_id(orientation)
            for q_idx, a_idxs in report['qa'].items():
                q_idx = int(q_idx)
                question = question_list[q_idx]
                answer = '. '.join(sentences[i] for i in a_idxs)
                self.report_ids.append(ri)
                self.question_ids.append(q_idx)
                self.images.append(image_path)
                self.questions.append(tokenizer.string2ids(question.lower()))
                self.answers.append(tokenizer.string2ids(answer.lower()))
                self.orientations.append(orientation_id)

class MIMICCXR_VQA_Trainer(VQA_Trainer):

    def __init__(self, transform, batch_size, collate_batch_fn,
                qa_adapted_reports_filename,
                split_kwargs,
                tokenizer,
                use_tags = False,
                medical_tags_per_report_filename = None,
                use_orientation = False,
                use_chexpert = False,
                chexpert_labels_filename = None,
                mimiccxr_qa_reports = None,
                mimiccxr_metadata = None,
                mimiccxr_split = None,
                balanced_split = False,
                balanced_metadata_filename = None,
                debug = False):
        
        self.tokenizer = tokenizer
        self.mimiccxr_qa_reports = mimiccxr_qa_reports
        self.mimiccxr_metadata = mimiccxr_metadata
        self.mimiccxr_split = mimiccxr_split
        self.qa_adapted_reports_filename = qa_adapted_reports_filename
        
        preprocessing_save_path = _get_train_preprocessing_save_path(
                        qa_adapted_reports_filename, split_kwargs, tokenizer, balanced_metadata_filename)

        rid2tags_path = os.path.join(MIMICCXR_CACHE_DIR, medical_tags_per_report_filename) if use_tags else None
        chexpert_labels_path = os.path.join(MIMICCXR_CACHE_DIR, chexpert_labels_filename) if use_chexpert else None
        balanced_metadata_path = os.path.join(MIMICCXR_CACHE_DIR, balanced_metadata_filename) if balanced_split else None

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessing_save_path,
                        use_tags = use_tags,
                        rid2tags_path = rid2tags_path,
                        use_orientation = use_orientation,
                        use_chexpert = use_chexpert,
                        chexpert_labels_path = chexpert_labels_path,
                        dataset_name = 'MIMIC-CXR',
                        split_kwargs = split_kwargs,
                        balanced_split = balanced_split,
                        balanced_metadata_path = balanced_metadata_path,
                        debug = debug)

    def _preprocess_data(self):
        _preprocess_data(self, self.qa_adapted_reports_filename, lambda split : split != 'test')

class MIMICCXR_VQA_Evaluator(VQA_Evaluator):

    def __init__(self, transform, batch_size, collate_batch_fn,
                qa_adapted_reports_filename,
                use_tags = False,
                medical_tags_per_report_filename = None,
                use_orientation = False,
                use_chexpert = False,
                chexpert_labels_filename = None,
                tokenizer = None,
                mimiccxr_qa_reports = None,
                mimiccxr_metadata = None,
                mimiccxr_split = None,
                **unused_kwargs):
        
        self.tokenizer = tokenizer
        self.mimiccxr_qa_reports = mimiccxr_qa_reports
        self.mimiccxr_metadata = mimiccxr_metadata
        self.mimiccxr_split = mimiccxr_split
        self.qa_adapted_reports_filename = qa_adapted_reports_filename
        
        preprocessing_save_path = _get_test_preprocessing_save_path(
                        qa_adapted_reports_filename, tokenizer)
        
        rid2tags_path = os.path.join(MIMICCXR_CACHE_DIR, medical_tags_per_report_filename) if use_tags else None
        chexpert_labels_path = os.path.join(MIMICCXR_CACHE_DIR, chexpert_labels_filename) if use_chexpert else None

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessing_save_path,
                        use_tags = use_tags,
                        rid2tags_path = rid2tags_path,
                        use_orientation = use_orientation,
                        use_chexpert = use_chexpert,
                        chexpert_labels_path = chexpert_labels_path,
                        dataset_name = 'MIMIC-CXR')

    def _preprocess_data(self):
        _preprocess_data(self, self.qa_adapted_reports_filename, lambda split : split == 'test')        