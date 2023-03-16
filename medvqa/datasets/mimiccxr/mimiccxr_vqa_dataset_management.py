from collections import Counter
import os
import numpy as np
from tqdm import tqdm
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_gold_standard_dicom_ids,
    load_chest_imagenome_label_names_and_templates,
    load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix,
    load_chest_imagenome_silver_bboxes_as_numpy_array,
)
from medvqa.datasets.utils import deduplicate_indices
from medvqa.datasets.vqa import VQA_Evaluator, VQA_Trainer
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_IMAGE_ORIENTATIONS,
    MIMICCXR_STUDY_REGEX,
    MIMICCXR_EvalViewModes,
    MIMICCXR_ViewModes,
    get_dicom_id_and_orientation_list,
    get_broken_images,
    get_image_views_dict,
    get_mimiccxr_small_image_path,
    get_split_dict,
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
)
from medvqa.utils.constants import CHEXPERT_LABELS, VINBIG_DISEASES, ReportEvalMode
from medvqa.datasets.preprocessing import (
    get_average_question_positions,
    get_question_frequencies,
)
from medvqa.utils.logging import print_red

def _get_train_preprocessing_save_path(qa_adapted_reports_filename, tokenizer, use_chest_imagenome_compatible_data):
    tokenizer_string = f'{tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}'
    if tokenizer.medical_tokenization:
        tokenizer_string += f',{tokenizer.medical_terms_frequency_filename}'
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'tokenizer={tokenizer_string}',
        f'use_chest_imagenome_compatible_data={use_chest_imagenome_compatible_data}',
    ]
    return get_file_path_with_hashing_if_too_long(MIMICCXR_CACHE_DIR, 'mimiccxr_preprocessed_train_data__', strings, 'pkl')

def get_test_preprocessing_save_path(qa_adapted_reports_filename, tokenizer, report_eval_mode=None,
                                    pretrained_checkpoint_path=None, precomputed_question_probs_path=None,
                                    precomputed_question_thresholds_path=None,
                                    n_questions_per_report=None, qclass_threshold=None, use_random_image=False,
                                    eval_view_mode=None):
    assert eval_view_mode is not None
    strings = [
        f'dataset={qa_adapted_reports_filename}',
        f'tokenizer={tokenizer.vocab_size},{tokenizer.hash[0]},{tokenizer.hash[1]}',
        f'eval_view_mode={eval_view_mode}',
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
        
        if use_random_image: strings.append('rand_img')

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

def _precompute_questions_per_report(basic_data, report_eval_mode,
        n_questions_per_report=None, qclass_threshold=None,
        qa_adapted_reports_filename=None, image_transform=None,
        image_local_feat_size=None, n_questions_aux_task=None, pretrained_weights=None,
        pretrained_checkpoint_path=None, precomputed_question_probs_path=None,
        precomputed_question_thresholds_path=None, batch_size=None):

    print('_precompute_questions_per_report():')

    # Filepath
    strings = [
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

    test_report_ids = [rid for rid, split in zip(basic_data['report_ids'], basic_data['splits']) if split == 'test']
    assert len(test_report_ids) > 0

    data = {}

    if report_eval_mode == ReportEvalMode.GROUND_TRUTH:
        for ri in test_report_ids:
            data[ri] = mimiccxr_qa_reports['reports'][ri]['question_ids']

    elif report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION or\
         report_eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
        assert n_questions_per_report is not None
        assert qclass_threshold is not None or precomputed_question_thresholds_path is not None        
        train_report_ids = [rid for rid, split in zip(basic_data['report_ids'], basic_data['splits']) if split == 'train']
        assert len(train_report_ids) > 0
        question_scores = get_average_question_positions(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename, train_report_ids)
        if precomputed_question_probs_path is not None:
            questions = _rank_questions(test_report_ids,
                                        precomputed_question_probs_path,
                                        n_questions_aux_task,
                                        n_questions_per_report,
                                        threshold=qclass_threshold,
                                        precomputed_question_thresholds_path=precomputed_question_thresholds_path)
        elif pretrained_weights is not None:
            test_image_paths = [image_path for image_path, split in zip(basic_data['image_paths'], basic_data['splits']) if split == 'test']
            questions = classify_and_rank_questions(test_image_paths,
                                        image_transform,
                                        image_local_feat_size,
                                        n_questions_aux_task,
                                        pretrained_weights,
                                        batch_size,
                                        n_questions_per_report,
                                        qclass_threshold)
        else: assert False        
        assert len(questions) == len(test_report_ids)
        _sanity_check_questions(questions, test_report_ids, mimiccxr_qa_reports)
        question_scorer = lambda j : question_scores[j]

        if report_eval_mode == ReportEvalMode.QUESTION_CLASSIFICATION:            
            for i, ri in enumerate(test_report_ids):            
                classified_question_ids = sorted(questions[i], key=question_scorer)
                data[ri] = classified_question_ids
        else: # hybrid method
            chexpert_question_ids = [x for x in range(len(CHEXPERT_LABELS))]
            for i, ri in enumerate(test_report_ids):
                classified_question_ids = sorted(questions[i], key=question_scorer)
                assert all(x not in chexpert_question_ids for x in classified_question_ids)
                data[ri] = chexpert_question_ids + classified_question_ids        

    elif report_eval_mode == ReportEvalMode.MOST_POPULAR:
        assert n_questions_per_report != None
        test_report_ids = set(test_report_ids)
        train_report_ids = [i for i in range(len(mimiccxr_qa_reports['reports'])) if i not in test_report_ids]
        question_scores = get_question_frequencies(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename, train_report_ids)
        question_ids = list(range(len(mimiccxr_qa_reports['questions'])))
        assert len(question_scores) == len(question_ids)
        question_ids.sort(key=lambda i : question_scores[i], reverse=True)
        question_ids = question_ids[:n_questions_per_report]
        for ri in test_report_ids:
            data[ri] = question_ids

    elif report_eval_mode == ReportEvalMode.NEAREST_NEIGHBOR:
        assert pretrained_weights is not None
        assert pretrained_checkpoint_path is not None
        test_image_paths = [image_path for image_path, split in zip(basic_data['image_paths'], basic_data['splits']) if split == 'test']
        train_image_paths = [image_path for image_path, split in zip(basic_data['image_paths'], basic_data['splits']) if split == 'train']
        nearest_neighbors = get_nearest_neighbors(target_images = test_image_paths,
                                                  reference_images = train_image_paths,
                                                  transform = image_transform,
                                                  pretrained_weights = pretrained_weights,
                                                  batch_size = batch_size,
                                                  cache_dir = MIMICCXR_CACHE_DIR,
                                                  pretrained_checkpoint_path = pretrained_checkpoint_path,
                                                  suffix = 'split=test')
        for i, ri in enumerate(test_report_ids):
            nearest_ri = train_report_ids[nearest_neighbors[i]]
            assert ri != nearest_ri
            data[ri] = mimiccxr_qa_reports['reports'][nearest_ri]['question_ids']
    
    elif report_eval_mode == ReportEvalMode.CHEXPERT_LABELS:
        question_ids = list(range(len(CHEXPERT_LABELS)))
        for ri in test_report_ids:
            data[ri] = question_ids        
    elif report_eval_mode == ReportEvalMode.VINBIG_DISEASES:
        question_ids = list(range(len(VINBIG_DISEASES)))
        for ri in test_report_ids:
            data[ri] = question_ids
    else:
        assert False, f'Unknown report_eval_mode = {report_eval_mode}'
    
    save_to_pickle(data, file_path)
    print('   len(data) =', len(data))
    print('   questions per report data saved to', file_path)
    
    return data

def _get_basic_data(qa_adapted_reports_filename, image_views_dict, split_dict,
                    view_mode=MIMICCXR_ViewModes.ANY_SINGLE,
                    use_chest_imagenome_compatible_data=False, chest_imagenome_dicom_ids=None):

    if use_chest_imagenome_compatible_data:
        assert chest_imagenome_dicom_ids is not None
        assert type(chest_imagenome_dicom_ids) == set
        assert view_mode == MIMICCXR_ViewModes.CHEST_IMAGENOME

    print(f'Obtaining basic data ...')

    file_path = os.path.join(MIMICCXR_CACHE_DIR,
        f'mimiccxr_basic_data(dataset={qa_adapted_reports_filename},view_mode={view_mode}).pkl')
    data = load_pickle(file_path)
    if data is not None:
        print(f'   basic data loaded from {file_path}')
        return data
    
    broken_images = get_broken_images()

    mimiccxr_qa_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))

    data = dict(
        report_ids = [],
        dicom_ids = [],
        image_paths = [],
        orientation_ids = [],
        splits = [],
    )
    
    for ri, report in tqdm(enumerate(mimiccxr_qa_reports['reports'])):

        part_id, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(report['filepath'])[0])
        views = image_views_dict[(subject_id, study_id)]
        # images = glob.glob(f'/mnt/workspace/mimic-cxr-jpg/images-small/p{part_id}/p{subject_id}/s{study_id}/*.jpg')
        # assert len(views) == len(images)

        dicom_id_orientation_pairs = get_dicom_id_and_orientation_list(views, view_mode, chest_imagenome_dicom_ids)

        for dicom_id, orientation in dicom_id_orientation_pairs:
            assert dicom_id is not None

            if (subject_id, study_id, dicom_id) not in broken_images:
                image_path = get_mimiccxr_small_image_path(part_id, subject_id, study_id, dicom_id)
                orientation_id = _get_orientation_id(orientation)
                data['report_ids'].append(ri)
                data['dicom_ids'].append(dicom_id)
                data['image_paths'].append(image_path)
                data['orientation_ids'].append(orientation_id)
                data['splits'].append(split_dict[(subject_id, study_id, dicom_id)])

    save_to_pickle(data, file_path)
    print(f'   len(report_ids) =', len(data['report_ids']))
    print(f'   len(set(report_ids)) =', len(set(data['report_ids'])))
    # Print the distribution of orientations
    counter = Counter(data["orientation_ids"])
    for oid, c in counter.items():
        print(f'   {MIMICCXR_IMAGE_ORIENTATIONS[oid]}, count = {c}')
    print(f'   basic data saved to {file_path}')
    
    return data

def _filter_report_orientation_pairs_for_test(report_ids, orientation_ids, view_mode):

    print(f'Filtering report orientation pairs for test ...')

    if view_mode == MIMICCXR_EvalViewModes.ALL:
        output = list(range(len(report_ids)))
    elif view_mode == MIMICCXR_EvalViewModes.FRONT_ALL:
        idxs = []
        PA_ID = MIMICCXR_IMAGE_ORIENTATIONS.index('PA')
        AP_ID = MIMICCXR_IMAGE_ORIENTATIONS.index('AP')
        valid_oids = [PA_ID, AP_ID]
        for i, (ri, oid) in enumerate(zip(report_ids, orientation_ids)):
            if oid in valid_oids:
                idxs.append(i)
        output = idxs
    elif view_mode == MIMICCXR_EvalViewModes.FRONT_SINGLE:
        rid2idx = {}
        PA_ID = MIMICCXR_IMAGE_ORIENTATIONS.index('PA')
        AP_ID = MIMICCXR_IMAGE_ORIENTATIONS.index('AP')
        valid_oids = [PA_ID, AP_ID]
        for i, (ri, oid) in enumerate(zip(report_ids, orientation_ids)):
            if oid in valid_oids:
                if ri not in rid2idx:
                    rid2idx[ri] = i
                else:
                    if oid == PA_ID:
                        rid2idx[ri] = i
        output = list(rid2idx.values())
    elif view_mode == MIMICCXR_EvalViewModes.ANY_SINGLE:
        rid2idx = {}
        PA_ID = MIMICCXR_IMAGE_ORIENTATIONS.index('PA')
        AP_ID = MIMICCXR_IMAGE_ORIENTATIONS.index('AP')
        for i, (ri, oid) in enumerate(zip(report_ids, orientation_ids)):
            if ri not in rid2idx:
                rid2idx[ri] = i
            else:
                if oid == PA_ID:
                    rid2idx[ri] = i
                elif oid == AP_ID:
                    if rid2idx[ri] != PA_ID:
                        rid2idx[ri] = i
        output = list(rid2idx.values())
    else:
        raise ValueError(f'Unknown view_mode = {view_mode}')

    print(f'   len(report_ids) = {len(report_ids)}')
    print(f'   len(output) = {len(output)}')
    return output

def _preprocess_data(self, split_name):

    assert split_name in ('train_val', 'test'), f'Unknown split_name = {split_name}'

    is_train_val = split_name == 'train_val'

    tokenizer = self.tokenizer
    mimiccxr_qa_reports = self.mimiccxr_qa_reports

    if tokenizer.medical_tokenization and split_name != 'test':
        answer_string2ids_func = tokenizer.string2medical_tag_ids
    else:
        answer_string2ids_func = tokenizer.string2ids

    if mimiccxr_qa_reports is None:
        file_path = os.path.join(MIMICCXR_CACHE_DIR, self.qa_adapted_reports_filename)
        print(f'Loading {file_path}')
        mimiccxr_qa_reports = get_cached_json_file(file_path)

    split_dict = get_split_dict()
    image_views_dict = get_image_views_dict()    
    basic_data = _get_basic_data(self.qa_adapted_reports_filename, image_views_dict, split_dict,
                                 view_mode=self.view_mode, use_chest_imagenome_compatible_data=self.use_chest_imagenome_compatible_data,
                                 chest_imagenome_dicom_ids=self.chest_imagenome_dicom_ids)

    if split_name == 'test' and self.report_eval_mode is not None:
        questions_per_report = _precompute_questions_per_report(basic_data, self.report_eval_mode,
                        qa_adapted_reports_filename=self.qa_adapted_reports_filename,
                        image_transform=self.transform,
                        image_local_feat_size=self.image_local_feat_size,
                        n_questions_aux_task=self.n_questions_aux_task,
                        pretrained_weights=self.pretrained_weights,
                        batch_size=self.batch_size,
                        n_questions_per_report=self.n_questions_per_report,
                        qclass_threshold=self.qclass_threshold,
                        pretrained_checkpoint_path=self.pretrained_checkpoint_path,
                        precomputed_question_probs_path=self.precomputed_question_probs_path,
                        precomputed_question_thresholds_path=self.precomputed_question_thresholds_path)
    else:
        questions_per_report = None
    
    print('Preprocessing MIMIC-CXR vqa dataset ...')

    # Define question_list to use
    question_list = mimiccxr_qa_reports['questions'] # default
    if split_name == 'test':
        if self.report_eval_mode == ReportEvalMode.CHEXPERT_LABELS:
            question_list = CHEXPERT_LABELS
        elif self.report_eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
            question_list = CHEXPERT_LABELS + mimiccxr_qa_reports['questions']
                                
    # Collect multiple fields considerng the split and mode
    if is_train_val:
        if self.use_chest_imagenome_compatible_data:
            # Use only instances whose dicom_id is not in Chest-Imagenome gold standard
            gold_dicom_ids = self.chest_imagenome_gold_dicom_ids
            assert type(gold_dicom_ids) is set
            report_ids = [rid for rid, s, did in zip(basic_data['report_ids'], basic_data['splits'], basic_data['dicom_ids'])\
                 if s != 'test' and did not in gold_dicom_ids]
            image_paths = [ip for ip, s, did in zip(basic_data['image_paths'], basic_data['splits'], basic_data['dicom_ids'])\
                    if s != 'test' and did not in gold_dicom_ids]
            orientation_ids = [oid for oid, s, did in zip(basic_data['orientation_ids'], basic_data['splits'], basic_data['dicom_ids'])\
                    if s != 'test' and did not in gold_dicom_ids]
            dicom_ids = [did for s, did in zip(basic_data['splits'], basic_data['dicom_ids']) if s != 'test' and did not in gold_dicom_ids]
            splits = [s for s, did in zip(basic_data['splits'], basic_data['dicom_ids']) if s != 'test' and did not in gold_dicom_ids]
        else:
            report_ids = [rid for rid, s in zip(basic_data['report_ids'], basic_data['splits']) if s != 'test']
            image_paths = [ip for ip, s in zip(basic_data['image_paths'], basic_data['splits']) if s != 'test']
            orientation_ids = [oid for oid, s in zip(basic_data['orientation_ids'], basic_data['splits']) if s != 'test']
            dicom_ids = [did for s, did in zip(basic_data['splits'], basic_data['dicom_ids']) if s != 'test']
            splits = [s for s in basic_data['splits'] if s != 'test']
    else:
        report_ids = [rid for rid, s in zip(basic_data['report_ids'], basic_data['splits']) if s == 'test']
        image_paths = [ip for ip, s in zip(basic_data['image_paths'], basic_data['splits']) if s == 'test']
        orientation_ids = [oid for oid, s in zip(basic_data['orientation_ids'], basic_data['splits']) if s == 'test']
        dicom_ids = [did for s, did in zip(basic_data['splits'], basic_data['dicom_ids']) if s == 'test']
    assert len(report_ids) == len(image_paths) == len(orientation_ids) == len(dicom_ids)
    assert len(report_ids) > 0

    # Collect data in VQA format (multiple questions per report)
    self.report_ids = []
    self.question_ids = []
    self.dicom_ids = []
    self.images = []
    self.questions = []
    self.orientations = []
    if is_train_val:
        self.train_indices = []
        self.val_indices = []

    if questions_per_report is None: # Normal method        
        self.answers = []
        idx = 0
    
        for i in tqdm(range(len(report_ids))):

            ri = report_ids[i]
            report = mimiccxr_qa_reports['reports'][ri]
            sentences = report['sentences']
            image_path = image_paths[i]
            orientation_id = orientation_ids[i]
            dicom_id = dicom_ids[i]
            split = splits[i]

            for qid, a_ids in report['qa'].items():
                qid = int(qid)
                question = question_list[qid]
                answer = '. '.join(sentences[i] for i in a_ids)
                self.report_ids.append(ri)
                self.question_ids.append(qid)
                self.dicom_ids.append(dicom_id)
                self.images.append(image_path)
                self.questions.append(tokenizer.string2ids(question.lower()))
                self.answers.append(answer_string2ids_func(answer.lower()))
                self.orientations.append(orientation_id)
                if is_train_val:
                    if split == 'train':
                        self.train_indices.append(idx)
                    elif split == 'validate':
                        self.val_indices.append(idx)
                    else:
                        raise ValueError(f'Unknown split: {split}')
                    idx += 1

    else: # Report evaluation method -> we ignore the answers

        filtered_idxs = _filter_report_orientation_pairs_for_test(report_ids, orientation_ids, self.eval_view_mode)

        for i in tqdm(filtered_idxs):
            ri = report_ids[i]
            image_path = image_paths[i]
            orientation_id = orientation_ids[i]
            dicom_id = dicom_ids[i]
            question_ids = questions_per_report[ri]

            # assert len(question_ids) > 0, mimiccxr_qa_reports['reports'][ri]

            for qid in question_ids:
                question = question_list[qid]
                self.report_ids.append(ri)
                self.question_ids.append(qid)
                self.dicom_ids.append(dicom_id)
                self.images.append(image_path)
                self.questions.append(tokenizer.string2ids(question.lower()))
                self.orientations.append(orientation_id)
        
    self.report_ids = np.array(self.report_ids, dtype=int)
    self.question_ids = np.array(self.question_ids, dtype=int)
    self.dicom_ids = np.array(self.dicom_ids, dtype=str)
    self.images = np.array(self.images, dtype=str)
    self.questions = np.array(self.questions, dtype=object)
    self.orientations = np.array(self.orientations, dtype=int)
    if is_train_val:
        self.train_indices = np.array(self.train_indices, dtype=int)
        self.val_indices = np.array(self.val_indices, dtype=int)
    if questions_per_report is None:
        self.answers = np.array(self.answers, dtype=object)

class MIMICCXR_VQA_Trainer(VQA_Trainer):

    def __init__(self, train_image_transform, val_image_transform, 
                batch_size, collate_batch_fn,
                num_workers,
                qa_adapted_reports_filename,
                tokenizer,
                view_mode = MIMICCXR_ViewModes.ANY_SINGLE,
                collate_batch_fn_chexpert_mode = None,
                verbose_question = True,
                classify_tags = False,
                medical_tags_per_report_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                classify_chest_imagenome = False,
                predict_bboxes_chest_imagenome = False,
                clamp_bboxes_chest_imagenome = False,
                question_labels_filename = None,
                mimiccxr_qa_reports = None,
                balanced_dataloading = False,
                balanced_metadata_filename = None,
                imbalance_reduction_coef = 1,
                allowed_questions = None,
                one_question_per_batch = False,
                include_mined_questions_mode = False,
                include_chexpert_mode = False,
                include_image = True,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = None,
                use_merged_findings = False,
                findings_remapper = None,
                n_findings = None,
                include_chest_imagenome_mode = False,
                chest_imagenome_labels_filename = None,
                chest_imagenome_label_names_filename = None,
                collate_batch_fn_chest_imagenome_mode = None,
                debug = False):
        
        self.tokenizer = tokenizer
        self.mimiccxr_qa_reports = mimiccxr_qa_reports
        self.qa_adapted_reports_filename = qa_adapted_reports_filename        
        self.view_mode = view_mode        
        self.include_chest_imagenome_mode = include_chest_imagenome_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classify_chest_imagenome = classify_chest_imagenome
        self.predict_bboxes_chest_imagenome = predict_bboxes_chest_imagenome
        self.chest_imagenome_label_names_filename = chest_imagenome_label_names_filename
        self.chest_imagenome_labels_filename = chest_imagenome_labels_filename
        self.use_chest_imagenome_compatible_data = (
            include_chest_imagenome_mode or classify_chest_imagenome or predict_bboxes_chest_imagenome
        )
        
        other_tasks = [] # Hack so that parent class can handle other tasks        
        
        preprocessing_save_path = _get_train_preprocessing_save_path(qa_adapted_reports_filename, tokenizer, self.use_chest_imagenome_compatible_data)

        print(f'MIMICCXR_VQA_Trainer: balanced_dataloading = {balanced_dataloading}')

        super().__init__(train_image_transform, val_image_transform,
                        batch_size, collate_batch_fn,
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
                        balanced_dataloading = balanced_dataloading,
                        balanced_metadata_filename = balanced_metadata_filename,
                        imbalance_reduction_coef = imbalance_reduction_coef,
                        allowed_questions = allowed_questions,
                        qa_adapted_reports_filename = qa_adapted_reports_filename,
                        one_question_per_batch = one_question_per_batch,
                        include_mined_questions_mode = include_mined_questions_mode,
                        include_chexpert_mode = include_chexpert_mode,
                        include_image = include_image,
                        use_precomputed_visual_features = use_precomputed_visual_features,
                        precomputed_visual_features_path = precomputed_visual_features_path,
                        use_merged_findings = use_merged_findings,
                        findings_remapper = findings_remapper,
                        n_findings = n_findings,
                        other_tasks=other_tasks,
                        debug = debug)

        # --------- Chest-Imagenome specific logic ---------
        if include_chest_imagenome_mode: 
            assert collate_batch_fn_chest_imagenome_mode is not None            
            self.collate_batch_fn_chest_imagenome_mode = collate_batch_fn_chest_imagenome_mode
            # Load Chest-Imagenome label names and templates
            assert chest_imagenome_label_names_filename is not None
            self.chest_imagenome_label_names, self.chest_imagenome_templates = \
                load_chest_imagenome_label_names_and_templates(chest_imagenome_label_names_filename)

        if include_chest_imagenome_mode or classify_chest_imagenome:
            assert chest_imagenome_labels_filename is not None
            # Load Chest-Imagenome dicom_ids and labels
            self.chest_imagenome_dicom_ids, self.chest_imagenome_labels = \
                load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix(chest_imagenome_labels_filename, qa_adapted_reports_filename)
            # Load Chest-Imagenome gold standard dicom_ids (they must be removed from training and validation sets)
            self.chest_imagenome_gold_dicom_ids = set(load_gold_standard_dicom_ids())            
            
        if classify_chest_imagenome:
            other_tasks.append(('chest_imagenome', lambda _, rid: self.chest_imagenome_labels[rid]))
        
        if predict_bboxes_chest_imagenome:
            self.dicom_idxs, self.bbox_coords, self.bbox_presence =\
                load_chest_imagenome_silver_bboxes_as_numpy_array(
                    self.dicom_ids, clamp_bboxes_chest_imagenome)
            other_tasks.append(('chest_imagenome_bbox_coords', lambda i, _: self.bbox_coords[self.dicom_idxs[i]]))
            other_tasks.append(('chest_imagenome_bbox_presence', lambda i, _: self.bbox_presence[self.dicom_idxs[i]]))

    def _preprocess_data(self):
        _preprocess_data(self, 'train_val')

    def _generate_dataset_and_dataloader__chest_imagenome_mode(
            self, indices, batch_size, collate_batch_fn, num_workers,
            infinite=True, n_pos_samples=None, n_neg_samples=None, min_pos_to_include=0):

        return self._create_label_based_dataset_and_dataloader(
            indices=indices,
            labels=self.chest_imagenome_labels,
            label_names=self.chest_imagenome_label_names,
            templates=self.chest_imagenome_templates,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_batch_fn=collate_batch_fn,
            infinite=infinite,
            n_pos_samples=n_pos_samples,
            n_neg_samples=n_neg_samples,
            min_pos_to_include=min_pos_to_include,
            report_ids=self.report_ids,
            print_every=30,
            # break_loop_at_i=40, # for debugging
            create_dataset_kwargs=dict(
                fixed_qa_pair=True,
            ))

    def _generate_train_dataset_and_dataloader__chest_imagenome_mode(self, batch_size, collate_batch_fn, num_workers):                
        print('Generating balanced train dataset in chest imagenome mode ...')        
        flattened_indices = []
        for _indices in self.train_indices.values():
            flattened_indices.extend(_indices)
        dedup_indices = deduplicate_indices(flattened_indices, self.report_ids)
        dataset, dataloader = self._generate_dataset_and_dataloader__chest_imagenome_mode(
            dedup_indices, batch_size, collate_batch_fn, num_workers, min_pos_to_include=20)
        self.train_dataset__chest_imagenome_mode = dataset
        self.train_dataloader__chest_imagenome_mode = dataloader
        print('len(self.train_dataset__chest_imagenome_mode) =', len(self.train_dataset__chest_imagenome_mode))

    def _generate_val_dataset_and_dataloader__chest_imagenome_mode(self, batch_size, collate_batch_fn, num_workers):
        print('Generating balanced validation dataset in chest imagenome mode ...')
        dedup_indices = deduplicate_indices(self.val_indices, self.report_ids)
        dataset, dataloader = self._generate_dataset_and_dataloader__chest_imagenome_mode(
            dedup_indices, batch_size, collate_batch_fn, num_workers, infinite=False,
            n_pos_samples=5, n_neg_samples=5)
        self.val_dataset__chest_imagenome_mode = dataset
        self.val_dataloader__chest_imagenome_mode = dataloader
        print('len(self.val_dataset__chest_imagenome_mode) =', len(self.val_dataset__chest_imagenome_mode))

    def _load_optional_datasets_and_dataloaders(self):
        if self.include_chest_imagenome_mode:
            if not self.validation_only:
                self._generate_train_dataset_and_dataloader__chest_imagenome_mode(
                    self.batch_size, self.collate_batch_fn_chest_imagenome_mode, self.num_workers)
            if not self.train_with_all:
                self._generate_val_dataset_and_dataloader__chest_imagenome_mode(
                    self.batch_size, self.collate_batch_fn_chest_imagenome_mode, self.num_workers)

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
                report_eval_mode = None,
                image_local_feat_size = None,
                pretrained_checkpoint_path = None,
                pretrained_weights = None,
                precomputed_question_probs_path = None,
                precomputed_question_thresholds_path = None,
                n_questions_aux_task = None,
                n_questions_per_report = None,
                qclass_threshold = None,
                include_image = True,
                use_random_image = False,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = None,
                view_mode = None,
                eval_view_mode = None,
                include_chest_imagenome_mode = False,
                classify_chest_imagenome = False,
                chest_imagenome_labels_filename = None,
                **unused_kwargs):

        print('report_eval_mode =', report_eval_mode)
        
        self.tokenizer = tokenizer
        self.mimiccxr_qa_reports = mimiccxr_qa_reports
        self.qa_adapted_reports_filename = qa_adapted_reports_filename
        self.view_mode = view_mode
        self.include_chest_imagenome_mode = include_chest_imagenome_mode
        
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
        self.eval_view_mode = eval_view_mode

        if pretrained_weights is not None:
            assert pretrained_checkpoint_path is not None

        if classify_chest_imagenome: # Chest-Imagenome specific logic
            assert chest_imagenome_labels_filename is not None
            # Load Chest-Imagenome dicom_ids and labels
            self.chest_imagenome_dicom_ids, self.chest_imagenome_labels = \
                load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix(chest_imagenome_labels_filename, qa_adapted_reports_filename)
            # Necessary hack so that parent classes can access chest_imagenome_labels
            other_tasks = [('chest_imagenome', lambda _, rid: self.chest_imagenome_labels[rid])]
        else:
            other_tasks = None
        
        preprocessing_save_path = get_test_preprocessing_save_path(
                        qa_adapted_reports_filename, tokenizer, report_eval_mode,
                        pretrained_checkpoint_path, precomputed_question_probs_path,
                        precomputed_question_thresholds_path,
                        n_questions_per_report, qclass_threshold, use_random_image,
                        eval_view_mode)

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
                        use_random_image = use_random_image,
                        use_precomputed_visual_features = use_precomputed_visual_features,
                        precomputed_visual_features_path = precomputed_visual_features_path,
                        other_tasks = other_tasks)

    def _preprocess_data(self):
        _preprocess_data(self, 'test')

