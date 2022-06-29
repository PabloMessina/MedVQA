import os
import re
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from medvqa.datasets.vqa import VQA_Evaluator, VQA_Trainer
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_JPG_IMAGES_SMALL_DIR,
    MIMICCXR_METADATA_CSV_PATH,
    MIMICCXR_SPLIT_CSV_PATH,
    MIMICCXR_IMAGE_ORIENTATIONS,
)
from medvqa.datasets.image_processing import (
    classify_and_rank_questions,
    get_nearest_neighbors,
)
from medvqa.utils.files import (
    get_cached_json_file,
    load_pickle,
    save_to_pickle,
    MAX_FILENAME_LENGTH,
)
from medvqa.utils.hashing import hash_string
from medvqa.utils.constants import ReportEvalMode
from medvqa.datasets.preprocessing import (
    get_average_question_positions,
    get_question_frequencies,
)

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

def get_mimiccxr_image_paths(report):
    filepath = report['filepath']
    part_id, subject_id, study_id = map(int, _MIMICCXR_STUDY_REGEX.findall(filepath)[0])        
    images = glob.glob(_MIMICCXR_IMAGE_PATH_TEMPLATE.format(part_id, subject_id, study_id, '*'))
    return images

def _get_train_preprocessing_save_path(qa_adapted_reports_filename, split_kwargs, tokenizer,
                                       balanced_metadata_filename = None,
                                       chexpert_labels_filename = None):
    
    split_params_string = f'({",".join(str(split_kwargs[k]) for k in sorted(list(split_kwargs.keys())))})'
    tokenizer_string = f'{tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}'
    if tokenizer.medical_tokenization:
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
    final_path = os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_preprocessed_train_data__({merged_string}).pkl')
    if len(final_path) > MAX_FILENAME_LENGTH:
        h = hash_string(merged_string)
        final_path = os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_preprocessed_train_data__(hash={h[0]},{h[1]}).pkl')
    return final_path

def get_test_preprocessing_save_path(qa_adapted_reports_filename, tokenizer, report_eval_mode=None,
                                    pretrained_checkpoint_path=None, n_questions_per_report=None):
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'tokenizer={tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}',
    ]    
    if report_eval_mode is not None:        
        strings.append(f'report_eval_mode={report_eval_mode}')
        
        if report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION.value or\
           report_eval_mode == ReportEvalMode.NEAREST_NEIGHBOR.value:
            assert pretrained_checkpoint_path is not None
            strings.append(f'pretrained_checkpoint_path={pretrained_checkpoint_path}')
        
        if report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION.value or\
           report_eval_mode == ReportEvalMode.MOST_POPULAR.value:
            assert n_questions_per_report is not None
            strings.append(f'n_questions_per_report={n_questions_per_report}')

    file_path = os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_preprocessed_test_data__({";".join(strings)}).pkl')
    if len(file_path) > MAX_FILENAME_LENGTH:
        h = hash_string(file_path)
        file_path = os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_preprocessed_test_data__(hash={h[0]},{h[1]}).pkl')
    return file_path

def _get_orientation_id(orientation):
    try:
        return MIMICCXR_IMAGE_ORIENTATIONS.index(orientation)
    except ValueError:
        return 0

def _precompute_questions_per_report(split_name, split_data, report_eval_mode,
        n_questions_per_report=None, qa_adapted_reports_filename=None, image_transform=None,
        image_local_feat_size=None, n_questions=None, pretrained_weights=None,
        pretrained_checkpoint_path=None, train_split_data=None,
        batch_size=None):

    strings = [
        f'split={split_name}',
        f'report_eval_mode={report_eval_mode}',
        f'dataset={qa_adapted_reports_filename}',
    ]
    if n_questions_per_report is not None:
        strings.append(f'n_questions_per_report={n_questions_per_report}')
    file_path = os.path.join(MIMICCXR_CACHE_DIR, f'questions_per_report({";".join(strings)}).pkl')

    data = load_pickle(file_path)
    if data is not None:
        print('questions per report data loaded from', file_path)
        return data

    mimiccxr_qa_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))

    data = {}

    if report_eval_mode == ReportEvalMode.GROUND_TRUTH.value:        
        for ri in split_data['report_ids']:
            data[ri] = mimiccxr_qa_reports['reports'][ri]['question_ids']

    elif report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION.value:
        assert n_questions_per_report != None
        test_report_ids = set(split_data['report_ids'])
        train_report_ids = [i for i in range(len(mimiccxr_qa_reports['reports'])) if i not in test_report_ids]
        question_scores = get_average_question_positions(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename, train_report_ids)
        questions = classify_and_rank_questions(split_data['image_paths'],
                                    image_transform,
                                    image_local_feat_size,
                                    n_questions,
                                    pretrained_weights,
                                    batch_size,
                                    n_questions_per_report)
        assert len(questions) == len(split_data['report_ids'])
        question_scorer = lambda j : question_scores[j]
        for i, ri in enumerate(split_data['report_ids']):
            data[ri] = sorted(questions[i], key=question_scorer)

    elif report_eval_mode == ReportEvalMode.MOST_POPULAR.value:
        assert n_questions_per_report != None
        test_report_ids = set(split_data['report_ids'])
        train_report_ids = [i for i in range(len(mimiccxr_qa_reports['reports'])) if i not in test_report_ids]
        question_scores = get_question_frequencies(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename, train_report_ids)
        question_ids = list(range(len(mimiccxr_qa_reports['questions'])))
        assert len(question_scores) == len(question_ids)
        question_ids.sort(key=lambda i : question_scores[i], reverse=True)
        question_ids = question_ids[:n_questions_per_report]
        for ri in split_data['report_ids']:
            data[ri] = question_ids

    elif report_eval_mode == ReportEvalMode.NEAREST_NEIGHBOR.value:
        assert pretrained_weights is not None
        assert pretrained_checkpoint_path is not None
        assert train_split_data is not None
        nearest_neighbors = get_nearest_neighbors(target_images = split_data['image_paths'],
                                                  reference_images = train_split_data['image_paths'],
                                                  transform = image_transform,
                                                  pretrained_weights = pretrained_weights,
                                                  batch_size = batch_size,
                                                  cache_dir = MIMICCXR_CACHE_DIR,
                                                  pretrained_checkpoint_path = pretrained_checkpoint_path,
                                                  suffix = 'split=test')
        for i, ri in enumerate(split_data['report_ids']):
            nearest_ri = train_split_data['report_ids'][nearest_neighbors[i]]
            assert ri != nearest_ri
            data[ri] = mimiccxr_qa_reports['reports'][nearest_ri]['question_ids']
    
    else:
        assert False, f'Unknown report_eval_mode = {report_eval_mode}'
    
    save_to_pickle(data, file_path)
    print('questions per report data saved to', file_path)
    
    return data

def _get_split_data(qa_adapted_reports_filename, image_views_dict, split_dict, split_name, split_lambda):

    file_path = os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_{split_name}_split(dataset={qa_adapted_reports_filename}).pkl')
    
    data = load_pickle(file_path)
    if data is not None:
        print(f'{split_name} data loaded from', file_path)
        return data
    
    broken_images = set()
    for path in _MIMICCXR_BROKEN_IMAGES:
        _, a, b, c = _MIMICCXR_IMAGE_REGEX.findall(path)[0]
        broken_images.add((int(a), int(b), c))

    mimiccxr_qa_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))

    data = dict(
        report_ids = [],
        image_paths = [],
        orientation_ids = [],
    )
    
    for ri, report in tqdm(enumerate(mimiccxr_qa_reports['reports'])):

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
            
            data['report_ids'].append(ri)
            data['image_paths'].append(image_path)
            data['orientation_ids'].append(orientation_id)

    save_to_pickle(data, file_path)
    print(f'{split_name} data saved to', file_path)
    
    return data    

def _preprocess_data(self, qa_adapted_reports_filename, split_lambda, split_name):

    tokenizer = self.tokenizer
    mimiccxr_qa_reports = self.mimiccxr_qa_reports
    mimiccxr_metadata = self.mimiccxr_metadata
    mimiccxr_split = self.mimiccxr_split

    if tokenizer.medical_tokenization and split_name != 'test':
        answer_string2ids_func = tokenizer.string2medical_tag_ids
    else:
        answer_string2ids_func = tokenizer.string2ids

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
    
    split_data = _get_split_data(qa_adapted_reports_filename, image_views_dict, split_dict, split_name, split_lambda)

    if self.report_eval_mode is not None:
        assert split_name == 'test', 'report_eval_mode should not only be used with the test split'
        
        if self.report_eval_mode == ReportEvalMode.NEAREST_NEIGHBOR.value:
            train_split_data = _get_split_data(qa_adapted_reports_filename, image_views_dict, split_dict,
                                               'train', lambda split : split != 'test')
        else:
            train_split_data = None

        questions_per_report = _precompute_questions_per_report(split_name, split_data, self.report_eval_mode,
                        qa_adapted_reports_filename=qa_adapted_reports_filename,
                        image_transform=self.transform,
                        image_local_feat_size=self.image_local_feat_size,
                        n_questions=self.n_questions,
                        pretrained_weights=self.pretrained_weights,
                        batch_size=self.batch_size,
                        n_questions_per_report=self.n_questions_per_report,
                        pretrained_checkpoint_path=self.pretrained_checkpoint_path,
                        train_split_data=train_split_data)
    else:
        questions_per_report = None
    
    print('Preprocessing MIMIC-CXR vqa dataset ...')
       
    question_list = mimiccxr_qa_reports['questions']
    report_ids = split_data['report_ids']
    image_paths = split_data['image_paths']
    orientation_ids = split_data['orientation_ids']

    self.report_ids = []
    self.question_ids = []
    self.images = []
    self.questions = []
    self.answers = []
    self.orientations = []

    if questions_per_report is None: # Normal method
    
        for i in tqdm(range(len(report_ids))):

            ri = report_ids[i]
            report = mimiccxr_qa_reports['reports'][ri]
            sentences = report['sentences']
            image_path = image_paths[i]
            orientation_id = orientation_ids[i]

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

    else: # Report evaluation method -> we ignore the answers

        for i in tqdm(range(len(report_ids))):

            ri = report_ids[i]
            report = mimiccxr_qa_reports['reports'][ri]
            image_path = image_paths[i]
            orientation_id = orientation_ids[i]
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

class MIMICCXR_VQA_Trainer(VQA_Trainer):

    def __init__(self, transform, batch_size, collate_batch_fn,
                num_workers,
                qa_adapted_reports_filename,
                split_kwargs,
                tokenizer,
                verbose_question = True,
                classify_tags = False,
                medical_tags_per_report_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                mimiccxr_qa_reports = None,
                mimiccxr_metadata = None,
                mimiccxr_split = None,
                balanced_split = False,
                balanced_dataloading = False,
                balanced_metadata_filename = None,
                imbalance_reduction_coef = 1,
                allowed_questions = None,
                one_question_per_batch = False,
                debug = False):
        
        self.tokenizer = tokenizer
        self.mimiccxr_qa_reports = mimiccxr_qa_reports
        self.mimiccxr_metadata = mimiccxr_metadata
        self.mimiccxr_split = mimiccxr_split
        self.qa_adapted_reports_filename = qa_adapted_reports_filename
        self.report_eval_mode = None # Necessary hack
        
        preprocessing_save_path = _get_train_preprocessing_save_path(
                        qa_adapted_reports_filename, split_kwargs, tokenizer,
                        balanced_metadata_filename, chexpert_labels_filename if balanced_split else None)

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessing_save_path,
                        MIMICCXR_CACHE_DIR,
                        num_workers,
                        verbose_question = verbose_question,
                        classify_tags = classify_tags,
                        rid2tags_filename = medical_tags_per_report_filename,
                        classify_orientation = classify_orientation,
                        classify_chexpert = classify_chexpert,
                        chexpert_labels_filename = chexpert_labels_filename,
                        classify_questions = classify_questions,
                        question_labels_filename = question_labels_filename,
                        dataset_name = 'MIMIC-CXR',
                        split_kwargs = split_kwargs,
                        balanced_split = balanced_split,
                        balanced_dataloading = balanced_dataloading,
                        balanced_metadata_filename = balanced_metadata_filename,
                        imbalance_reduction_coef = imbalance_reduction_coef,
                        allowed_questions = allowed_questions,
                        qa_adapted_reports_filename = qa_adapted_reports_filename,
                        one_question_per_batch = one_question_per_batch,
                        debug = debug)

    def _preprocess_data(self):
        _preprocess_data(self, self.qa_adapted_reports_filename, lambda split : split != 'test', 'train')

class MIMICCXR_VQA_Evaluator(VQA_Evaluator):

    def __init__(self, transform, batch_size, collate_batch_fn,
                qa_adapted_reports_filename,
                num_workers,
                verbose_question = True,
                classify_tags = False,
                medical_tags_per_report_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                tokenizer = None,
                mimiccxr_qa_reports = None,
                mimiccxr_metadata = None,
                mimiccxr_split = None,
                report_eval_mode = None,
                image_local_feat_size = None,
                pretrained_checkpoint_path = None,
                pretrained_weights = None,
                n_questions = None,
                n_questions_per_report = None,                
                **unused_kwargs):

        print('report_eval_mode =', report_eval_mode)
        
        self.tokenizer = tokenizer
        self.mimiccxr_qa_reports = mimiccxr_qa_reports
        self.mimiccxr_metadata = mimiccxr_metadata
        self.mimiccxr_split = mimiccxr_split
        self.qa_adapted_reports_filename = qa_adapted_reports_filename
        
        # Args used in report_eval_mode
        self.report_eval_mode = report_eval_mode
        self.transform = transform
        self.image_local_feat_size = image_local_feat_size
        self.n_questions = n_questions
        self.pretrained_weights = pretrained_weights
        self.batch_size = batch_size
        self.n_questions_per_report = n_questions_per_report
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        if pretrained_weights is not None:
            assert pretrained_checkpoint_path is not None
        
        preprocessing_save_path = get_test_preprocessing_save_path(
                        qa_adapted_reports_filename, tokenizer, report_eval_mode,
                        pretrained_checkpoint_path, n_questions_per_report)

        super().__init__(transform, batch_size, collate_batch_fn,
                        preprocessing_save_path,
                        MIMICCXR_CACHE_DIR,
                        num_workers,
                        verbose_question = verbose_question,
                        include_answer = report_eval_mode == None,
                        classify_tags = classify_tags,
                        rid2tags_filename = medical_tags_per_report_filename,
                        classify_orientation = classify_orientation,
                        classify_chexpert = classify_chexpert,
                        chexpert_labels_filename = chexpert_labels_filename,
                        classify_questions = classify_questions,
                        question_labels_filename = question_labels_filename,
                        dataset_name = 'MIMIC-CXR')

    def _preprocess_data(self):
        _preprocess_data(self, self.qa_adapted_reports_filename, lambda split : split == 'test', 'test')

