import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from medvqa.datasets.vqa import VQA_Evaluator, VQA_Trainer
from medvqa.datasets.mimiccxr import (
    MIMICCXR_BROKEN_IMAGES,
    MIMICCXR_CACHE_DIR,
    MIMICCXR_IMAGE_PATH_TEMPLATE,
    MIMICCXR_IMAGE_REGEX,
    MIMICCXR_METADATA_CSV_PATH,
    MIMICCXR_SPLIT_CSV_PATH,
    MIMICCXR_IMAGE_ORIENTATIONS,
    MIMICCXR_STUDY_REGEX,
    get_mimiccxr_image_path,
)
from medvqa.datasets.image_processing import (
    classify_and_rank_questions,
    get_nearest_neighbors,
)
from medvqa.metrics.classification.multilabel_prf1 import MultiLabelPRF1
from medvqa.utils.files import (
    get_cached_json_file,
    get_file_path_with_hashing_if_too_long,
    load_pickle,
    save_to_pickle,
    MAX_FILENAME_LENGTH,
)
from medvqa.utils.hashing import hash_string
from medvqa.utils.constants import CHEXPERT_LABELS, VINBIG_DISEASES, ReportEvalMode
from medvqa.datasets.preprocessing import (
    get_average_question_positions,
    get_question_frequencies,
)
from medvqa.utils.logging import print_red

def get_mimiccxr_image_paths(report):
    filepath = report['filepath']
    part_id, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(filepath)[0])
    images = glob.glob(MIMICCXR_IMAGE_PATH_TEMPLATE.format(part_id, subject_id, study_id, '*'))
    return images

def _get_train_preprocessing_save_path(qa_adapted_reports_filename, split_kwargs, tokenizer,
                                       balanced_metadata_filename = None,
                                       chexpert_labels_filename = None,
                                       train_with_all = False):
    
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
    if train_with_all:
        strings.append('train_with_all')
    merged_string = ";".join(strings)
    final_path = os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_preprocessed_train_data__({merged_string}).pkl')
    if len(final_path) > MAX_FILENAME_LENGTH:
        h = hash_string(merged_string)
        final_path = os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_preprocessed_train_data__(hash={h[0]},{h[1]}).pkl')
    return final_path

def get_test_preprocessing_save_path(qa_adapted_reports_filename, tokenizer, report_eval_mode=None,
                                    pretrained_checkpoint_path=None, precomputed_question_probs_path=None,
                                    precomputed_question_thresholds_path=None,
                                    n_questions_per_report=None, qclass_threshold=None):
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'tokenizer={tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}',
    ]    
    if report_eval_mode is not None:        
        strings.append(f'report_eval_mode={report_eval_mode}')

        if report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION or\
           report_eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
            assert pretrained_checkpoint_path is not None or\
                   precomputed_question_probs_path is not None
            if precomputed_question_probs_path is not None:
                strings.append(f'precomputed_question_probs_path={precomputed_question_probs_path}')
                strings.append(f'timestamp={os.path.getmtime(precomputed_question_probs_path)}')
            else:              
                strings.append(f'pretrained_checkpoint_path={pretrained_checkpoint_path}')
        
        if report_eval_mode == ReportEvalMode.NEAREST_NEIGHBOR:
            assert pretrained_checkpoint_path is not None 
            strings.append(f'pretrained_checkpoint_path={pretrained_checkpoint_path}')
        
        if report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION or\
           report_eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION or\
           report_eval_mode == ReportEvalMode.MOST_POPULAR:
            assert n_questions_per_report is not None
            strings.append(f'n_questions_per_report={n_questions_per_report}')

        if report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION or\
           report_eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
            assert (qclass_threshold is not None) != (precomputed_question_thresholds_path is not None)
            if qclass_threshold:
                strings.append(f'qclass_threshold={qclass_threshold}')
            else:
                strings.append(f'precomputed_question_thresholds_path={precomputed_question_thresholds_path}')
        

    file_path = get_file_path_with_hashing_if_too_long(MIMICCXR_CACHE_DIR, 'mimiccxr_preprocessed_test_data__', strings)
    return file_path

def _get_orientation_id(orientation):
    try:
        return MIMICCXR_IMAGE_ORIENTATIONS.index(orientation)
    except ValueError:
        return 0

def _rank_questions(report_ids, precomputed_question_probs_path, n_questions, top_k,
        min_num_q_per_report=5, threshold=None, precomputed_question_thresholds_path=None):
    assert (threshold is None) != (precomputed_question_thresholds_path is None)
    print_red('_rank_questions(): loading question probabilities from', precomputed_question_probs_path)
    question_probs = load_pickle(precomputed_question_probs_path)
    if precomputed_question_thresholds_path is not None:
        print_red('_rank_questions(): loading question thresholds from', precomputed_question_thresholds_path)
        thresholds = load_pickle(precomputed_question_thresholds_path)
    question_ids = list(range(n_questions))
    questions = [None] * len(report_ids)
    K = min(min_num_q_per_report, top_k)
    for i, rid in enumerate(report_ids):
        probs = question_probs[rid]
        if threshold:
            probs -= threshold 
        else:
            probs -= thresholds
        question_ids.sort(key=lambda k:probs[k], reverse=True)
        questions[i] = [qid for k, qid in enumerate(question_ids) if k < top_k and probs[qid] >= 0]
        if len(questions[i]) < K:
            questions[i] = question_ids[:K]
            assert 0 < len(questions[i]) <= top_k
    print_red('_rank_questions(): average num of questions per report:', sum(len(q) for q in questions) / len(questions))
    return questions

def _sanity_check_questions(questions, report_ids, mimiccxr_qa_reports):
    n_q = len(mimiccxr_qa_reports['questions'])
    gt = np.zeros((len(report_ids), n_q))
    pred = np.zeros((len(report_ids), n_q))
    for i, ri in enumerate(report_ids):
        for j in mimiccxr_qa_reports['reports'][ri]['question_ids']:
            gt[i][j] = 1
        for j in questions[i]:
            pred[i][j] = 1
    met = MultiLabelPRF1(device='cpu')
    met.update((pred, gt))
    res = met.compute()
    print_red(f'_sanity_check_questions(): f1(macro)={res["f1_macro_avg"]}, f1(micro)={res["f1_micro_avg"]}')

def _precompute_questions_per_report(split_name, split_data, report_eval_mode,
        n_questions_per_report=None, qclass_threshold=None,
        qa_adapted_reports_filename=None, image_transform=None,
        image_local_feat_size=None, n_questions_aux_task=None, pretrained_weights=None,
        pretrained_checkpoint_path=None, precomputed_question_probs_path=None,
        precomputed_question_thresholds_path=None,
        train_split_data=None, chexpert_one_hot_offset=None, batch_size=None):

    print('_precompute_questions_per_report():')

    # Filepath
    strings = [
        f'split={split_name}',
        f'report_eval_mode={report_eval_mode}',
        f'dataset={qa_adapted_reports_filename}',
    ]
    if n_questions_per_report is not None:
        strings.append(f'n_questions_per_report={n_questions_per_report}')        
    if report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION or\
            report_eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
        assert pretrained_checkpoint_path is not None or precomputed_question_probs_path is not None
        assert qclass_threshold is not None or precomputed_question_thresholds_path is not None
        if precomputed_question_probs_path is not None:
            strings.append(f'precomputed_question_probs_path={precomputed_question_probs_path}')
            strings.append(f'timestamp={os.path.getmtime(precomputed_question_probs_path)}')
        else:
            strings.append(f'pretrained_checkpoint_path={pretrained_checkpoint_path}')
        if qclass_threshold:
            strings.append(f'qclass_threshold={qclass_threshold}')
        else:
            strings.append(f'precomputed_question_thresholds_path={precomputed_question_thresholds_path}')
    file_path = get_file_path_with_hashing_if_too_long(MIMICCXR_CACHE_DIR, 'questions_per_report', strings)

    data = load_pickle(file_path)
    if data is not None:
        print('   questions per report data loaded from', file_path)
        print('   len(data) =', len(data))
        return data

    mimiccxr_qa_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))

    data = {}

    if report_eval_mode == ReportEvalMode.GROUND_TRUTH:
        for ri in split_data['report_ids']:
            data[ri] = mimiccxr_qa_reports['reports'][ri]['question_ids']

    elif report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION or\
         report_eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
        assert n_questions_per_report is not None
        assert qclass_threshold is not None or precomputed_question_thresholds_path is not None
        assert chexpert_one_hot_offset is not None
        test_report_ids = set(split_data['report_ids'])
        train_report_ids = [i for i in range(len(mimiccxr_qa_reports['reports'])) if i not in test_report_ids]
        question_scores = get_average_question_positions(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename, train_report_ids)
        if precomputed_question_probs_path is not None:
            questions = _rank_questions(split_data['report_ids'],
                                        precomputed_question_probs_path,
                                        n_questions_aux_task,
                                        n_questions_per_report,
                                        threshold=qclass_threshold,
                                        precomputed_question_thresholds_path=precomputed_question_thresholds_path)
        elif pretrained_weights is not None:
            questions = classify_and_rank_questions(split_data['image_paths'],
                                        image_transform,
                                        image_local_feat_size,
                                        n_questions_aux_task,
                                        pretrained_weights,
                                        batch_size,
                                        n_questions_per_report,
                                        qclass_threshold)
        else: assert False        
        assert len(questions) == len(split_data['report_ids'])
        _sanity_check_questions(questions, split_data['report_ids'], mimiccxr_qa_reports)
        question_scorer = lambda j : question_scores[j]

        if report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION:            
            for i, ri in enumerate(split_data['report_ids']):            
                classified_question_ids = sorted(questions[i], key=question_scorer)
                data[ri] = classified_question_ids
        else: # hybrid method
            chexpert_question_ids = [chexpert_one_hot_offset + x for x in range(len(CHEXPERT_LABELS))]
            for i, ri in enumerate(split_data['report_ids']):
                classified_question_ids = sorted(questions[i], key=question_scorer)
                assert all(x not in chexpert_question_ids for x in classified_question_ids)
                data[ri] = chexpert_question_ids + classified_question_ids        

    elif report_eval_mode == ReportEvalMode.MOST_POPULAR:
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

    elif report_eval_mode == ReportEvalMode.NEAREST_NEIGHBOR:
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
    
    elif report_eval_mode == ReportEvalMode.CHEXPERT_LABELS:
        question_ids = list(range(len(CHEXPERT_LABELS)))
        for ri in split_data['report_ids']:
            data[ri] = question_ids        
    elif report_eval_mode == ReportEvalMode.VINBIG_DISEASES:
        question_ids = list(range(len(VINBIG_DISEASES)))
        for ri in split_data['report_ids']:
            data[ri] = question_ids
    else:
        assert False, f'Unknown report_eval_mode = {report_eval_mode}'
    
    save_to_pickle(data, file_path)
    print('   len(data) =', len(data))
    print('   questions per report data saved to', file_path)
    
    return data

def _get_split_data(qa_adapted_reports_filename, image_views_dict, split_dict, split_name, split_lambda):

    print(f'Obtaining {split_name} data ...')

    file_path = os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_{split_name}_split(dataset={qa_adapted_reports_filename}).pkl')
    
    data = load_pickle(file_path)
    if data is not None:
        print(f'   {split_name} data loaded from', file_path)
        return data
    
    broken_images = set()
    for path in MIMICCXR_BROKEN_IMAGES:
        _, a, b, c = MIMICCXR_IMAGE_REGEX.findall(path)[0]
        broken_images.add((int(a), int(b), c))

    mimiccxr_qa_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))

    data = dict(
        report_ids = [],
        image_paths = [],
        orientation_ids = [],
    )
    
    for ri, report in tqdm(enumerate(mimiccxr_qa_reports['reports'])):

        part_id, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(report['filepath'])[0])
        views = image_views_dict[(subject_id, study_id)]
        # images = glob.glob(f'/mnt/workspace/mimic-cxr-jpg/images-small/p{part_id}/p{subject_id}/s{study_id}/*.jpg')
        # assert len(views) == len(images)
        
        dicom_id = None
        for view in views:
            if view[1] == 'PA':
                dicom_id = view[0]
                orientation = view[1]
                break
        if dicom_id is None:
            for view in views:
                if view[1] == 'AP':
                    dicom_id = view[0]
                    orientation = view[1]
                    break
        if dicom_id is None and len(views) > 0:
            dicom_id = views[0][0]
            orientation = views[0][1]        
            
        if (dicom_id is not None and split_lambda(split_dict[(subject_id, study_id, dicom_id)]) and
                (subject_id, study_id, dicom_id) not in broken_images):
            image_path = get_mimiccxr_image_path(part_id, subject_id, study_id, dicom_id)
            orientation_id = _get_orientation_id(orientation)
            
            data['report_ids'].append(ri)
            data['image_paths'].append(image_path)
            data['orientation_ids'].append(orientation_id)

    save_to_pickle(data, file_path)
    print(f'   len(report_ids) =', len(data['report_ids']))
    print(f'   len(set(report_ids)) =', len(set(data['report_ids'])))
    print(f'   {split_name} data saved to', file_path)
    
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
        assert split_name == 'test', 'report_eval_mode should only be used with the test split'
        
        if self.report_eval_mode == ReportEvalMode.NEAREST_NEIGHBOR:
            train_split_data = _get_split_data(qa_adapted_reports_filename, image_views_dict, split_dict,
                                               'train', lambda split : split != 'test')
        else:
            train_split_data = None

        questions_per_report = _precompute_questions_per_report(split_name, split_data, self.report_eval_mode,
                        qa_adapted_reports_filename=qa_adapted_reports_filename,
                        image_transform=self.transform,
                        image_local_feat_size=self.image_local_feat_size,
                        n_questions_aux_task=self.n_questions_aux_task,
                        pretrained_weights=self.pretrained_weights,
                        batch_size=self.batch_size,
                        n_questions_per_report=self.n_questions_per_report,
                        qclass_threshold=self.qclass_threshold,
                        pretrained_checkpoint_path=self.pretrained_checkpoint_path,
                        precomputed_question_probs_path=self.precomputed_question_probs_path,
                        precomputed_question_thresholds_path=self.precomputed_question_thresholds_path,
                        chexpert_one_hot_offset=self.chexpert_one_hot_offset,
                        train_split_data=train_split_data)
    else:
        questions_per_report = None
    
    print('Preprocessing MIMIC-CXR vqa dataset ...')

    # Define question_list to use
    if self.report_eval_mode == ReportEvalMode.CHEXPERT_LABELS:
        question_list = CHEXPERT_LABELS
    elif self.report_eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
        question_list = CHEXPERT_LABELS + mimiccxr_qa_reports['questions']
    else:
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
            image_path = image_paths[i]
            orientation_id = orientation_ids[i]
            question_ids = questions_per_report[ri]

            # assert len(question_ids) > 0, mimiccxr_qa_reports['reports'][ri]

            for qid in question_ids:
                question = question_list[qid]
                self.report_ids.append(ri)
                self.question_ids.append(qid)
                self.images.append(image_path)
                self.questions.append(tokenizer.string2ids(question.lower()))
                self.orientations.append(orientation_id)

    # assert len(report_ids) == len(set(report_ids))
    # assert set(self.report_ids) == set(report_ids)
        
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
                collate_batch_fn_chexpert_mode = None,
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
                include_chexpert_mode = False,
                use_chexpert_mode_only = False,
                chexpert_one_hot_offset = None,
                include_image = True,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = None,
                use_merged_findings = False,
                findings_remapper = None,
                n_findings = None,
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
                        collate_batch_fn_chexpert_mode = collate_batch_fn_chexpert_mode,
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
                        include_chexpert_mode = include_chexpert_mode,
                        use_chexpert_mode_only = use_chexpert_mode_only,
                        chexpert_one_hot_offset = chexpert_one_hot_offset,
                        include_image = include_image,
                        use_precomputed_visual_features = use_precomputed_visual_features,
                        precomputed_visual_features_path = precomputed_visual_features_path,
                        use_merged_findings = use_merged_findings,
                        findings_remapper = findings_remapper,
                        n_findings = n_findings,
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
                precomputed_question_probs_path = None,
                precomputed_question_thresholds_path = None,
                n_questions_aux_task = None,
                n_questions_per_report = None,
                qclass_threshold = None,
                chexpert_one_hot_offset = None,
                include_image = True,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = None,
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
        self.n_questions_aux_task = n_questions_aux_task
        self.pretrained_weights = pretrained_weights
        self.batch_size = batch_size
        self.n_questions_per_report = n_questions_per_report
        self.qclass_threshold = qclass_threshold
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.precomputed_question_probs_path = precomputed_question_probs_path
        self.precomputed_question_thresholds_path = precomputed_question_thresholds_path
        self.chexpert_one_hot_offset = chexpert_one_hot_offset

        if pretrained_weights is not None:
            assert pretrained_checkpoint_path is not None
        
        preprocessing_save_path = get_test_preprocessing_save_path(
                        qa_adapted_reports_filename, tokenizer, report_eval_mode,
                        pretrained_checkpoint_path, precomputed_question_probs_path,
                        precomputed_question_thresholds_path,
                        n_questions_per_report, qclass_threshold)

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
                        include_image = include_image,
                        use_precomputed_visual_features = use_precomputed_visual_features,
                        precomputed_visual_features_path = precomputed_visual_features_path,
                        chexpert_one_hot_offset = chexpert_one_hot_offset,
                        dataset_name = 'MIMIC-CXR')

    def _preprocess_data(self):
        _preprocess_data(self, self.qa_adapted_reports_filename, lambda split : split == 'test', 'test')

