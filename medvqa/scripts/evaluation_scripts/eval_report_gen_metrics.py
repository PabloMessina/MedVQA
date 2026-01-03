import argparse
import random
import numpy as np
import multiprocessing as mp
import functools
import os
import time

import pandas as pd
from medvqa.datasets.nli import MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH, RADNLI_DEV_JSONL_PATH, RADNLI_TEST_JSONL_PATH

from medvqa.utils.text_data_utils import word_tokenize_texts_in_parallel
from medvqa.evaluation.plots import plot_correlation_matrix, plot_metric_lists
from medvqa.evaluation.ranking_evaluation_utils import load_mimiccxr_custom_radiologist_annotations
from medvqa.metrics.medical.chexbert import CheXbertLabeler
from medvqa.metrics.medical.chexpert import ChexpertLabeler
from medvqa.metrics.medical.fact_embedding import FactEmbeddingScorer
from medvqa.metrics.medical.radcliq import RADCLIQ_METRIC_NAMES
from medvqa.metrics.medical.radgraph import RadGraphLabeler, RadGraphLabelerOriginal
from medvqa.metrics.nlp.cider import CiderD
from medvqa.metrics.nlp.meteor import Meteor
from medvqa.metrics.nlp.rouge import RougeL
from medvqa.utils.hashing_utils import hash_string

from medvqa.utils.common import CACHE_DIR, get_timestamp, LARGE_FAST_CACHE_DIR
from medvqa.utils.files_utils import load_jsonl, load_pickle, save_pickle
from medvqa.utils.logging_utils import print_bold
from medvqa.metrics.nlp import Bleu
from medvqa.utils.metrics_utils import auc, f1_between_dicts, jaccard_between_dicts

def cache_matrix_to_file(alias, uses_sentences=True, uses_labels=False, uses_obs_anat_labels=False, use_checkpoint_folder_path=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if uses_sentences:
                sentences = args[0]
                n = len(sentences)
                lensum = sum(len(s) for s in sentences)
                if use_checkpoint_folder_path:
                    checkpoint_folder_path = args[1]
                    path_hash = hash_string(checkpoint_folder_path)
                    path_hash = f'{path_hash[0]},{path_hash[1]}'
                    cache_filepath = os.path.join(LARGE_FAST_CACHE_DIR, f'score_matrix({alias},{n},{lensum},{path_hash}).pkl')
                else:
                    cache_filepath = os.path.join(LARGE_FAST_CACHE_DIR, f'score_matrix({alias},{n},{lensum}).pkl')
            elif uses_labels:
                labels = args[0]
                n, m = labels.shape
                cache_filepath = os.path.join(LARGE_FAST_CACHE_DIR, f'score_matrix({alias},{n},{m}).pkl')
            elif uses_obs_anat_labels:
                obs_labels, anat_labels = args[0], args[1]
                n1, m1 = obs_labels.shape
                n2, m2 = anat_labels.shape
                cache_filepath = os.path.join(LARGE_FAST_CACHE_DIR, f'score_matrix({alias},{n1},{m1},{n2},{m2}).pkl')
            else:
                cache_filepath = os.path.join(LARGE_FAST_CACHE_DIR, f'score_matrix({alias}).pkl')
            matrix = load_pickle(cache_filepath)
            if matrix is None:
                start = time.time()
                matrix = func(*args, **kwargs)
                end = time.time()
                print(f'Computed {alias} matrix in {end - start} seconds')
                save_pickle(matrix,  cache_filepath)
                print(f'Saved {alias} matrix to {cache_filepath}')
            else:
                print(f'Loaded {alias} matrix from {cache_filepath}')
            return matrix
        return wrapper
    return decorator

_shared_sentences = None
_shared_labels = None
_shared_obs_labels = None
_shared_anat_labels = None
_shared_gold_relevance_matrix = None
_shared_gold_accuracy_matrix = None
_shared_gold_contradiction_matrix = None
_shared_score_matrix = None

def _compute_accuracy(args):
    i, j = args
    return np.sum(_shared_labels[i] == _shared_labels[j]) / len(_shared_labels[i])

def _compute_accuracy_matrix(labels):
    n = len(labels)
    args = [(i, j) for i in range(n) for j in range(i, n)]
    global _shared_labels
    _shared_labels = labels
    with mp.Pool(processes=mp.cpu_count()) as pool:
        accuracies = pool.map(_compute_accuracy, args)
    accuracy_matrix = np.zeros((n, n))
    for (i, j), acc in zip(args, accuracies):
        accuracy_matrix[i, j] = acc
        accuracy_matrix[j, i] = acc
    return accuracy_matrix

def _compute_f1(args):
    i, j = args
    tp = np.sum((_shared_labels[i] == 1) & (_shared_labels[j] == 1))
    fp = np.sum((_shared_labels[i] == 1) & (_shared_labels[j] == 0))
    fn = np.sum((_shared_labels[i] == 0) & (_shared_labels[j] == 1))
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1

def _compute_f1_matrix(labels):
    n = len(labels)
    args = [(i, j) for i in range(n) for j in range(n)]
    global _shared_labels
    _shared_labels = labels
    with mp.Pool(processes=mp.cpu_count()) as pool:
        f1s = pool.map(_compute_f1, args)
    f1_matrix = np.zeros((n, n))
    for (i, j), f1 in zip(args, f1s):
        f1_matrix[i, j] = f1
        f1_matrix[j, i] = f1
    return f1_matrix

def _a_anat_entails_b_anat(a_anat, b_anat):
    for i in range(len(b_anat)):
        if b_anat[i] and not a_anat[i]:
            return False
    return True

def _a_obs_entails_b_obs(a_obs, b_obs):
    for i in range(len(b_obs)):
        if b_obs[i] != -1 and a_obs[i] != b_obs[i]:
            return False
    return True

def _a_entails_b_custom(a, b):
    for i in range(len(b)):
        if b[i] == -2: # nan
            continue
        if a[i] != b[i]:
            return False
    return True

def _compute_relevance_v1(args):
    i, j = args
    return (_a_anat_entails_b_anat(_shared_anat_labels[i], _shared_anat_labels[j]) and \
              _a_obs_entails_b_obs(_shared_obs_labels[i], _shared_obs_labels[j])) or \
              (_a_anat_entails_b_anat(_shared_anat_labels[j], _shared_anat_labels[i]) and \
                _a_obs_entails_b_obs(_shared_obs_labels[j], _shared_obs_labels[i]))

def _compute_relevance_v2(args):
    i, j = args
    return _a_entails_b_custom(_shared_labels[i], _shared_labels[j]) or \
           _a_entails_b_custom(_shared_labels[j], _shared_labels[i])

def _compute_relevance_matrix_v1(obs_labels, anat_labels):
    n = len(obs_labels)
    args = [(i, j) for i in range(n) for j in range(i, n)]
    global _shared_obs_labels
    global _shared_anat_labels
    _shared_obs_labels = obs_labels
    _shared_anat_labels = anat_labels
    with mp.Pool(processes=mp.cpu_count()) as pool:
        relevances = pool.map(_compute_relevance_v1, args)
    relevance_matrix = np.zeros((n, n))
    for (i, j), rel in zip(args, relevances):
        relevance_matrix[i, j] = rel
        relevance_matrix[j, i] = rel
    return relevance_matrix

def _compute_relevance_matrix_v2(labels):
    n = len(labels)
    args = [(i, j) for i in range(n) for j in range(i, n)]
    global _shared_labels
    _shared_labels = labels
    with mp.Pool(processes=mp.cpu_count()) as pool:
        relevances = pool.map(_compute_relevance_v2, args)
    relevance_matrix = np.zeros((n, n))
    for (i, j), rel in zip(args, relevances):
        relevance_matrix[i, j] = rel
        relevance_matrix[j, i] = rel
    return relevance_matrix

@cache_matrix_to_file('chest_imagenome_gold_relevance', uses_sentences=False, uses_obs_anat_labels=True)
def compute_chest_imagenome_gold_relevance_matrix(obs_labels, anat_labels):
    print_bold('Computing gold labels relevance between each pair of sentences')
    return _compute_relevance_matrix_v1(obs_labels, anat_labels)

def _compute_jaccard(args):
    i, j = args
    a = _sets[i]
    b = _sets[j]
    lena = len(a)
    lenb = len(b)
    if lena == 0 and lenb == 0:
        return 1
    len_inters = len(a & b)
    return len_inters / (lena + lenb - len_inters)

@cache_matrix_to_file('chest_imagenome_gold_jaccard', uses_sentences=False, uses_obs_anat_labels=True)
def compute_chest_imagenome_gold_jaccard_matrix(obs_labels, anat_labels):
    print_bold('Computing gold labels jaccard index between each pair of sentences')
    n = len(obs_labels)
    args = [(i, j) for i in range(n) for j in range(i, n)]
    sets = [set() for _ in range(n)]
    for i in range(n):
        for j, x in enumerate(obs_labels[i]):
            if x != -1:
                sets[i].add((0, j, x))
        for j, x in enumerate(anat_labels[i]):
            if x:
                sets[i].add((1, j, x))
    global _sets
    _sets = sets
    with mp.Pool(processes=mp.cpu_count()) as pool:
        jaccards = pool.map(_compute_jaccard, args)
    jaccard_matrix = np.zeros((n, n))
    for (i, j), rel in zip(args, jaccards):
        jaccard_matrix[i, j] = rel
        jaccard_matrix[j, i] = rel
    return jaccard_matrix

def _compute_contradiction(args):
    i, j = args
    for k in range(len(_shared_obs_labels[i])):
        if _shared_obs_labels[i][k] != -1 and _shared_obs_labels[j][k] != -1 and _shared_obs_labels[i][k] != _shared_obs_labels[j][k]:
            return 1
    return 0

def _compute_contradiction_v2(args):
    i, j = args
    for k in range(len(_shared_labels[i])):
        if _shared_labels[i][k] != -2 and _shared_labels[j][k] != -2 and _shared_labels[i][k] != _shared_labels[j][k]:
            return 1
    return 0

@cache_matrix_to_file('chest_imagenome_gold_contradictions', uses_sentences=False, uses_labels=True)
def compute_chest_imagenome_gold_contradiction_matrix(obs_labels):
    n = len(obs_labels)
    contradiction_matrix = np.zeros((n, n))
    args = [(i, j) for i in range(n) for j in range(i, n)]
    global _shared_obs_labels
    _shared_obs_labels = obs_labels
    with mp.Pool(processes=mp.cpu_count()) as pool:
        contradictions = pool.map(_compute_contradiction, args)
    for (i, j), c in zip(args, contradictions):
        contradiction_matrix[i, j] = c
        contradiction_matrix[j, i] = c
    return contradiction_matrix

@cache_matrix_to_file('iuxray_gold_tags_jaccard', uses_sentences=False)
def compute_iuxray_gold_tags_jaccard_matrix(tag_sets):
    print_bold('Computing gold jaccard index between each pair of sentences')
    n = len(tag_sets)
    args = [(i, j) for i in range(n) for j in range(i, n)]
    global _sets
    _sets = tag_sets
    with mp.Pool(processes=mp.cpu_count()) as pool:
        jaccards = pool.map(_compute_jaccard, args)
    jaccard_matrix = np.zeros((n, n))
    for (i, j), rel in zip(args, jaccards):
        jaccard_matrix[i, j] = rel
        jaccard_matrix[j, i] = rel
    return jaccard_matrix

@cache_matrix_to_file('custom_mimiccxr_radiologist_annotations_relevance', uses_sentences=False, uses_labels=True)
def compute_custom_mimiccxr_radiologist_annotations_relevance_matrix(labels):
    print_bold('Computing custom MIMIC-CXR radiologist annotations relevance between each pair of sentences')
    return _compute_relevance_matrix_v2(labels)

@cache_matrix_to_file('custom_mimiccxr_radiologist_annotations_contradictions', uses_sentences=False, uses_labels=True)
def compute_custom_mimiccxr_radiologist_annotations_contradiction_matrix(labels):
    n = len(labels)
    contradiction_matrix = np.zeros((n, n))
    args = [(i, j) for i in range(n) for j in range(i, n)]
    global _shared_labels
    _shared_labels = labels
    with mp.Pool(processes=mp.cpu_count()) as pool:
        contradictions = pool.map(_compute_contradiction_v2, args)
    for (i, j), c in zip(args, contradictions):
        contradiction_matrix[i, j] = c
        contradiction_matrix[j, i] = c
    return contradiction_matrix

@cache_matrix_to_file('gold_accuracy', uses_sentences=False, uses_labels=True)
def compute_gold_accuracy_matrix(labels):
    print_bold('Computing gold labels accuracy between each pair of sentences')
    return _compute_accuracy_matrix(labels)    


def _compute_bleu_scores(gen_texts, gt_texts):
    bleu = Bleu(device='cpu', record_scores=True, using_ids=False)
    bleu.update((gen_texts, gt_texts))
    _, scores_by_instance = bleu.compute()
    assert len(scores_by_instance) == 4
    assert len(scores_by_instance[0]) == len(gt_texts)
    scores_by_instance = np.array(scores_by_instance)
    average_scores = np.mean(scores_by_instance, axis=0)
    assert average_scores.shape == (len(gt_texts),)
    return average_scores

def _compute_bleu_score_for_row(i):
    gt_texts = [_shared_sentences[i]] * len(_shared_sentences) # ground truth texts
    gen_texts = _shared_sentences # generated texts
    return _compute_bleu_scores(gen_texts, gt_texts)

def _compute_rougel_scores(gen_texts, gt_texts):
    rougel = RougeL(device='cpu', record_scores=True, using_ids=False)
    rougel.update((gen_texts, gt_texts))
    scores = rougel.compute()
    assert len(scores) == len(gt_texts)
    return scores

def _compute_rougel_score_for_row(i):
    gt_texts = [_shared_sentences[i]] * len(_shared_sentences) # ground truth texts
    gen_texts = _shared_sentences # generated texts
    return _compute_rougel_scores(gen_texts, gt_texts)

def _compute_meteor_scores(tokenized_gen_texts, tokenized_gt_texts):
    metric = Meteor(device='cpu', record_scores=True)
    metric.update((tokenized_gen_texts, tokenized_gt_texts))
    scores = metric.compute()
    assert len(scores) == len(tokenized_gt_texts)
    return scores

def _compute_meteor_score_for_row(i):
    gt_texts = [_shared_sentences[i]] * len(_shared_sentences) # ground truth texts
    gen_texts = _shared_sentences # generated texts
    return _compute_meteor_scores(gen_texts, gt_texts)

def _compute_ciderd_scores(gen_texts, gt_texts):
    metric = CiderD(device='cpu', record_scores=True, using_ids=False)
    metric.update((gen_texts, gt_texts))
    _, scores = metric.compute()
    assert len(scores) == len(gt_texts)
    return scores

def _compute_ciderd_score_for_row(i):
    gen_texts = [_shared_sentences[i]] * len(_shared_sentences) # ground truth texts
    gt_texts = _shared_sentences # generated texts
    return _compute_ciderd_scores(gen_texts, gt_texts)

def _compute_score_matrix(sentences, rowwise_score_computer):
    num_sentences = len(sentences)
    global _shared_sentences
    _shared_sentences = sentences
    with mp.Pool(processes=mp.cpu_count()) as pool:
        scores = pool.map(rowwise_score_computer, range(num_sentences))
    score_matrix = np.zeros((num_sentences, num_sentences))
    for i, row in enumerate(scores):
        for j, score in enumerate(row):
            score_matrix[i, j] = score
    return score_matrix

@cache_matrix_to_file('bleu')
def compute_bleu_score_matrix(sentences):
    print_bold('Computing BLEU score between each pair of sentences')
    return _compute_score_matrix(sentences, _compute_bleu_score_for_row)

@cache_matrix_to_file('rougel')
def compute_rougel_score_matrix(sentences):
    print_bold('Computing ROUGE-L score between each pair of sentences')
    return _compute_score_matrix(sentences, _compute_rougel_score_for_row)

@cache_matrix_to_file('meteor')
def compute_meteor_score_matrix(sentences):
    print_bold('Computing METEOR score between each pair of sentences')
    tokenized_sentences = word_tokenize_texts_in_parallel(sentences)
    # for s, ts in zip(sentences, tokenized_sentences):
    #     assert len(ts) > 0, f'Failed to tokenize sentence: {s} (tokenized: {ts})'
    return _compute_score_matrix(tokenized_sentences, _compute_meteor_score_for_row)

@cache_matrix_to_file('ciderd')
def compute_ciderd_score_matrix(sentences):
    print_bold('Computing CIDEr-D score between each pair of sentences')
    return _compute_score_matrix(sentences, _compute_ciderd_score_for_row)

def _compute_bertscore_scores(gen_texts, gt_texts):
    # from bert_score import score as bert_score
    from medvqa.metrics.nlp.bertscore import BertScore
    _, _, F1 = BertScore()(gt_texts, gen_texts)
    return F1.cpu().numpy()

@cache_matrix_to_file('bertscore')
def compute_bertscore_matrix(sentences):
    print_bold('Computing BERTScore between each pair of sentences')
    pairs = [(i, j) for i in range(len(sentences)) for j in range(i, len(sentences))]
    gen_texts = [sentences[i] for i, _ in pairs]
    gt_texts = [sentences[j] for _, j in pairs]
    F1 = _compute_bertscore_scores(gen_texts, gt_texts)
    bertscore_matrix = np.zeros((len(sentences), len(sentences)))
    for (i, j), f1 in zip(pairs, F1):
        bertscore_matrix[i, j] = f1
        bertscore_matrix[j, i] = f1
    return bertscore_matrix

def _get_chexpert_labels(gen_texts, gt_texts, max_processes=10):
    texts = gen_texts + gt_texts
    labeler = ChexpertLabeler(verbose=True)
    tmp_anticolission_code = f'_{get_timestamp()}_{random.random()}'
    labels = labeler.get_labels(texts, tmp_suffix=tmp_anticolission_code,
                                update_cache_on_disk=True, remove_tmp_files=True,
                                n_chunks=max_processes, max_processes=max_processes)
    gen_labels = labels[:len(gen_texts)]
    gt_labels = labels[len(gen_texts):]
    return gen_labels, gt_labels    

def _compute_chexpert_accuracy_scores(gen_texts, gt_texts, max_processes=10):
    gen_labels, gt_labels = _get_chexpert_labels(gen_texts, gt_texts, max_processes)
    acc_scores = np.zeros(len(gen_texts))
    for i in range(len(gen_texts)):
        acc_scores[i] = np.sum(gen_labels[i] == gt_labels[i]) / len(gen_labels[i])
    return acc_scores

def _compute_chexpert_f1_scores(gen_texts, gt_texts, max_processes=10):
    gen_labels, gt_labels = _get_chexpert_labels(gen_texts, gt_texts, max_processes)
    f1_scores = np.zeros(len(gen_texts))
    for i in range(len(gen_texts)):
        tp = np.sum((gen_labels[i] == 1) & (gt_labels[i] == 1))
        fp = np.sum((gen_labels[i] == 1) & (gt_labels[i] == 0))
        fn = np.sum((gen_labels[i] == 0) & (gt_labels[i] == 1))
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_scores[i] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1_scores

@cache_matrix_to_file('chexpert_accuracy')
def compute_chexpert_accuracy_matrix(sentences, max_processes=10):
    print_bold('Computing CheXpert labeler accuracy between each pair of sentences')
    tmp_anticolission_code = f'_{get_timestamp()}_{random.random()}'
    labeler = ChexpertLabeler(verbose=True)
    labels = labeler.get_labels(sentences, tmp_suffix=tmp_anticolission_code,
                                update_cache_on_disk=True, remove_tmp_files=True,
                                n_chunks=max_processes, max_processes=max_processes)
    return _compute_accuracy_matrix(labels)

@cache_matrix_to_file('chexpert_f1')
def compute_chexpert_f1_matrix(sentences, max_processes=10):
    print_bold('Computing CheXpert labeler f1 between each pair of sentences')
    tmp_anticolission_code = f'_{get_timestamp()}_{random.random()}'
    labeler = ChexpertLabeler(verbose=True)
    labels = labeler.get_labels(sentences, tmp_suffix=tmp_anticolission_code,
                                update_cache_on_disk=True, remove_tmp_files=True,
                                n_chunks=max_processes, max_processes=max_processes)
    return _compute_f1_matrix(labels)

def _get_chexbert_labels(gen_texts, gt_texts):
    texts = gen_texts + gt_texts
    labeler = CheXbertLabeler(verbose=True)
    labels = labeler.get_labels(texts, update_cache_on_disk=True)
    gen_labels = labels[:len(gen_texts)]
    gt_labels = labels[len(gen_texts):]
    return gen_labels, gt_labels

def _compute_chexbert_accuracy_scores(gen_texts, gt_texts):
    gen_labels, gt_labels = _get_chexbert_labels(gen_texts, gt_texts)
    acc_scores = np.zeros(len(gen_texts))
    for i in range(len(gen_texts)):
        acc_scores[i] = np.sum(gen_labels[i] == gt_labels[i]) / len(gen_labels[i])
    return acc_scores

def _compute_chexbert_f1_scores(gen_texts, gt_texts):
    gen_labels, gt_labels = _get_chexbert_labels(gen_texts, gt_texts)
    f1_scores = np.zeros(len(gen_texts))
    for i in range(len(gen_texts)):
        tp = np.sum((gen_labels[i] == 1) & (gt_labels[i] == 1))
        fp = np.sum((gen_labels[i] == 1) & (gt_labels[i] == 0))
        fn = np.sum((gen_labels[i] == 0) & (gt_labels[i] == 1))
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_scores[i] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1_scores

@cache_matrix_to_file('chexbert_accuracy')
def compute_chexbert_accuracy_matrix(sentences):
    print_bold('Computing CheXbert labeler accuracy between each pair of sentences')
    labeler = CheXbertLabeler(verbose=True)
    labels = labeler.get_labels(sentences, update_cache_on_disk=True)
    return _compute_accuracy_matrix(labels)

@cache_matrix_to_file('chexbert_f1')
def compute_chexbert_f1_matrix(sentences):
    print_bold('Computing CheXbert labeler f1 between each pair of sentences')
    labeler = CheXbertLabeler(verbose=True)
    labels = labeler.get_labels(sentences, update_cache_on_disk=True)
    return _compute_f1_matrix(labels)

def _compute_radgraph_jaccard_scores(gen_texts, gt_texts):
    labeler = RadGraphLabeler(verbose=True)
    texts = gen_texts + gt_texts
    labels = labeler.get_labels(texts, update_cache_on_disk=True)
    gen_labels = labels[:len(gen_texts)]
    gt_labels = labels[len(gen_texts):]
    jaccards = np.zeros(len(gen_texts))
    for i in range(len(gen_texts)):
        jaccards[i] = jaccard_between_dicts(gen_labels[i], gt_labels[i])
    return jaccards

@cache_matrix_to_file('radgraph_jaccard')
def compute_radgraph_jaccard_matrix(sentences):
    print_bold('Computing RadGraph jaccard index between each pair of sentences')
    labeler = RadGraphLabeler(verbose=True)
    labels = labeler.get_labels(sentences, update_cache_on_disk=True)
    matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            matrix[i, j] = matrix[j, i] = jaccard_between_dicts(labels[i], labels[j])
    return matrix

def _compute_radgraph_f1_scores(gen_texts, gt_texts):
    labeler = RadGraphLabeler(verbose=True)
    texts = gen_texts + gt_texts
    labels = labeler.get_labels(texts, update_cache_on_disk=True)
    gen_labels = labels[:len(gen_texts)]
    gt_labels = labels[len(gen_texts):]
    f1s = np.zeros(len(gen_texts))
    for i in range(len(gen_texts)):
        f1s[i] = f1_between_dicts(gt_labels[i], gen_labels[i])
    return f1s

def _compute_radgraph_f1_partial_scores(gen_texts, gt_texts):
    from medvqa.metrics.medical.radgraph import compute_reward
    labeler = RadGraphLabelerOriginal(verbose=True)
    texts = gen_texts + gt_texts
    labels = labeler(texts, update_cache_on_disk=True)
    gen_labels = labels[:len(gen_texts)]
    gt_labels = labels[len(gen_texts):]
    f1s = np.zeros(len(gen_texts))
    for i in range(len(gen_texts)):
        f1s[i] = compute_reward(gt_labels[i], gen_labels[i], "partial")
    return f1s

def _compute_radcliq_scores(gen_texts, gt_texts, device_id):
    from medvqa.metrics.medical.radcliq import invoke_radcliq_process
    out = invoke_radcliq_process(gt_texts, gen_texts, device_id=device_id)
    return out

@cache_matrix_to_file('radgraph_f1')
def compute_radgraph_f1_matrix(sentences):
    print_bold('Computing RadGraph F1 score between each pair of sentences')
    labeler = RadGraphLabeler(verbose=True)
    labels = labeler.get_labels(sentences, update_cache_on_disk=True)
    matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            matrix[i, j] = f1_between_dicts(labels[i], labels[j])
    return matrix

@cache_matrix_to_file('radgraph_f1_partial')
def compute_radgraph_f1_partial_matrix(sentences):
    print_bold('Computing RadGraph F1 partial score between each pair of sentences')
    from medvqa.metrics.medical.radgraph import compute_reward
    labeler = RadGraphLabelerOriginal(verbose=True)
    labels = labeler(sentences, update_cache_on_disk=True)
    matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            try:
                matrix[i, j] = compute_reward(labels[i], labels[j], "partial")
            except:
                print(f'Failed to compute reward for {i} and {j}')
                print(f'sentence1: {sentences[i]}')
                print(f'sentence2: {sentences[j]}')
                print(f'labels1: {labels[i]}')
                print(f'labels2: {labels[j]}')
                raise
    return matrix

def _compute_fact_embedding_scores(gen_texts, gt_texts, checkpoint_folder_path):
    scorer = FactEmbeddingScorer(verbose=True, fact_embedding_model_checkpoint_folder_path=checkpoint_folder_path)
    scores = scorer(gen_texts, gt_texts, update_cache_on_disk=True, return_avg_score=False, only_soft_score=True)
    return scores

@cache_matrix_to_file('fact_embedding_score', use_checkpoint_folder_path=True)
def compute_fact_embedding_score_matrix(sentences, checkpoint_folder_path):
    print_bold('Computing Fact Embedding Score between each pair of sentences')
    scorer = FactEmbeddingScorer(verbose=True, fact_embedding_model_checkpoint_folder_path=checkpoint_folder_path)
    return scorer.compute_pairwise_scores(sentences, update_cache_on_disk=True, only_soft_score=True)

def compute_AUC_for_row(i):
    return auc(_shared_score_matrix[i], _shared_gold_relevance_matrix[i])

def compute_AUC(gold_relevance_matrix, score_matrix):
    global _shared_gold_relevance_matrix
    global _shared_score_matrix
    _shared_gold_relevance_matrix = gold_relevance_matrix
    _shared_score_matrix = score_matrix
    with mp.Pool(processes=mp.cpu_count()) as pool:
        aucs = pool.map(compute_AUC_for_row, range(len(gold_relevance_matrix)))
    return np.mean(aucs)

def compute_average_accuracy_at_k_for_row(args):
    i, max_k = args
    top_k_indices = np.argsort(_shared_score_matrix[i])[::-1] # descending order
    average_accuracy_at_k = [0] * max_k
    acc_sum = 0
    for k in range(max_k):
        acc_sum += _shared_gold_accuracy_matrix[i][top_k_indices[k]]
        average_accuracy_at_k[k] = acc_sum / (k + 1)
    return average_accuracy_at_k

def compute_mean_average_accuracy_at_k(gold_accuracy_matrix, score_matrix, max_k):
    assert gold_accuracy_matrix.shape == score_matrix.shape
    assert gold_accuracy_matrix.shape[0] == gold_accuracy_matrix.shape[1]
    assert max_k <= gold_accuracy_matrix.shape[0]
    n = gold_accuracy_matrix.shape[0]
    global _shared_gold_accuracy_matrix
    global _shared_score_matrix
    _shared_gold_accuracy_matrix = gold_accuracy_matrix
    _shared_score_matrix = score_matrix
    with mp.Pool(processes=mp.cpu_count()) as pool:
        average_accuracy_at_k = pool.map(compute_average_accuracy_at_k_for_row, [(i, max_k) for i in range(n)])
    average_accuracy_at_k = np.mean(average_accuracy_at_k, axis=0)
    return average_accuracy_at_k

def compute_contradiction_at_k_for_row(args):
    i, max_k = args
    top_k_indices = np.argsort(_shared_score_matrix[i])[::-1] # descending order
    contradiction_at_k = [0] * max_k
    contradiction_sum = 0
    for k in range(max_k):
        contradiction_sum += _shared_gold_contradiction_matrix[i][top_k_indices[k]]
        contradiction_at_k[k] = contradiction_sum
    return contradiction_at_k

def compute_mean_contradictions_at_k(gold_contradiction_matrix, score_matrix, max_k):
    assert gold_contradiction_matrix.shape == score_matrix.shape
    assert gold_contradiction_matrix.shape[0] == gold_contradiction_matrix.shape[1]
    assert max_k <= gold_contradiction_matrix.shape[0]
    n = gold_contradiction_matrix.shape[0]
    global _shared_gold_contradiction_matrix
    global _shared_score_matrix
    _shared_gold_contradiction_matrix = gold_contradiction_matrix    
    _shared_score_matrix = score_matrix
    with mp.Pool(processes=mp.cpu_count()) as pool:
        contradictions_at_k = pool.map(compute_contradiction_at_k_for_row, [(i, max_k) for i in range(n)])
    mean_contradictions_at_k = np.mean(contradictions_at_k, axis=0)
    return mean_contradictions_at_k

def evaluate_metrics__chest_imagenome_gold(
        sentence2labels_gold_filepath,
        fact_embedding_model_checkpoint_folder_paths,
        fact_embedding_model_names,
    ):

    assert len(fact_embedding_model_checkpoint_folder_paths) == len(fact_embedding_model_names)
    assert len(fact_embedding_model_checkpoint_folder_paths) > 0
    
    print_bold(f'Loading sentence2labels_gold from {sentence2labels_gold_filepath}')
    sentence2labels_gold = load_pickle(sentence2labels_gold_filepath)
    sentences = sentence2labels_gold['phrases']
    observation_labels = sentence2labels_gold['observation_labels']
    anatomy_labels = sentence2labels_gold['anatomy_labels']
    labels =  np.concatenate([observation_labels, anatomy_labels], axis=1)
    print(f'Number of sentences: {len(sentences)}')
    print(f'label.shape: {labels.shape}')

    # Compute accuracy between each pair of sentences
    gold_accuracy_matrix = compute_gold_accuracy_matrix(labels)

    # Compute relevance between each pair of sentences
    gold_relevance_matrix = compute_chest_imagenome_gold_relevance_matrix(observation_labels, anatomy_labels)

    # Compute BLEU score between each pair of sentences
    bleu_score_matrix = compute_bleu_score_matrix(sentences)

    # Compute ROUGE-L score between each pair of sentences
    rougel_score_matrix = compute_rougel_score_matrix(sentences)

    # Compute METEOR score between each pair of sentences
    meteor_score_matrix = compute_meteor_score_matrix(sentences)

    # Compute CIDEr-D score between each pair of sentences
    ciderd_score_matrix = compute_ciderd_score_matrix(sentences)

    # Compute BERTScore between each pair of sentences
    bertscore_matrix = compute_bertscore_matrix(sentences)

    # Compute CheXpert labeler accuracy between each pair of sentences
    chexpert_accuracy_matrix = compute_chexpert_accuracy_matrix(sentences)

    # Compute CheXpert labeler f1 between each pair of sentences
    chexpert_f1_matrix = compute_chexpert_f1_matrix(sentences)

    # Compute CheXbert accuracy between each pair of sentences
    chexbert_accuracy_matrix = compute_chexbert_accuracy_matrix(sentences)

    # Compute CheXbert f1 between each pair of sentences
    chexbert_f1_matrix = compute_chexbert_f1_matrix(sentences)

    # # Compute RadGraph jaccard index between each pair of sentences
    # radgraph_jaccard_matrix = compute_radgraph_jaccard_matrix(sentences)

    # Compute RadGraph F1 score between each pair of sentences
    radgraph_f1_matrix = compute_radgraph_f1_matrix(sentences)

    # Compute RadGraph F1 partial score between each pair of sentences
    radgraph_f1_partial_matrix = compute_radgraph_f1_partial_matrix(sentences)

    # Compute Fact Embedding Score between each pair of sentences
    fact_embedding_score_matrix_list = [
        compute_fact_embedding_score_matrix(sentences, checkpoint_folder_path)
        for checkpoint_folder_path in fact_embedding_model_checkpoint_folder_paths
    ]

    # Plot a correlation matrix between all metrics
    scores_list = [
        # gold_accuracy_matrix,
        # gold_relevance_matrix,
        bleu_score_matrix,
        rougel_score_matrix,
        meteor_score_matrix,
        ciderd_score_matrix,
        # radgraph_jaccard_matrix,
        radgraph_f1_matrix,
        radgraph_f1_partial_matrix,
        bertscore_matrix,
        chexpert_accuracy_matrix,
        chexbert_accuracy_matrix,
        chexpert_f1_matrix,
        chexbert_f1_matrix,
    ]
    method_names = [
        # 'Gold Accuracy',
        # 'Gold Relevance',
        'BLEU',
        'ROUGE-L',
        'METEOR',
        'CIDEr-D',
        # 'RadGraph Jaccard',
        'RadGraph F1 Full',
        'RadGraph F1 Partial',
        'BERTScore',
        'CheXpert Accuracy',
        'CheXbert Accuracy',
        'CheXpert F1',
        'CheXbert F1',
    ]
    scores_list.extend(fact_embedding_score_matrix_list)
    method_names.extend(fact_embedding_model_names)
    assert len(scores_list) == len(method_names)
    flattend_scores_list = [score_matrix.flatten() for score_matrix in scores_list]
    plot_correlation_matrix(
        correlation_matrix=np.corrcoef(flattend_scores_list),
        method_names=method_names,
        title=f'Correlation Matrix between metrics (N={len(sentences)})',
    )

    # Compute AUC for each metric
    print_bold('Computing AUC for each metric')
    for method_name, score_matrix in zip(method_names, scores_list):
        print(f'{method_name} AUC: {compute_AUC(gold_relevance_matrix, score_matrix)}')

    # Compute mean average accuracy @k for each metric
    print_bold('Computing mean average accuracy @k for each metric')
    max_k = 100
    mean_average_accuracy_at_k_list = []
    for method_name, score_matrix in zip(method_names, scores_list):
        print_bold(f'Computing mean average accuracy @k for {method_name}')
        mean_average_accuracy_at_k = compute_mean_average_accuracy_at_k(gold_accuracy_matrix, score_matrix, max_k)
        mean_average_accuracy_at_k_list.append(mean_average_accuracy_at_k)
        for k in (1, 5, 10, 20, 50, 100):
            print(f'  mean_average_accuracy_at_{k}: {mean_average_accuracy_at_k[k - 1]}')
    # Plot mean average accuracy @k for each metric
    plot_metric_lists(
        metric_lists=mean_average_accuracy_at_k_list,
        method_names=method_names,
        title=f'Mean Average Accuracy @k for each metric (Chest ImaGenome Gold, sentence level, N={len(sentences)})',
        metric_name='Mean Average Accuracy @k',
        xlabel='k',
        ylabel='Mean Average Accuracy @k',
        figsize=(10, 6),
    )

    # Compute average number of contradictions @k for each metric
    print_bold('Computing average number of contradictions @k for each metric')

    # Compute relevance between each pair of sentences
    gold_contradiction_matrix = compute_chest_imagenome_gold_contradiction_matrix(observation_labels)
    max_k = 100
    mean_num_contradictions_at_k_list = []
    for method_name, score_matrix in zip(method_names, scores_list):
        print_bold(f'Computing average number of contradictions @k for {method_name}')
        mean_num_contradictions_at_k = compute_mean_contradictions_at_k(gold_contradiction_matrix, score_matrix, max_k)
        mean_num_contradictions_at_k_list.append(mean_num_contradictions_at_k)
        for k in (1, 5, 10, 20, 50, 100):
            print(f'  mean_num_contradictions_at_{k}: {mean_num_contradictions_at_k[k - 1]}')

    # Plot mean average number of contradictions @k for each metric
    plot_metric_lists(
        metric_lists=mean_num_contradictions_at_k_list,
        method_names=method_names,
        title=f'Average Number of Contradictions @k for each metric (Chest ImaGenome Gold, sentence level, N={len(sentences)})',
        metric_name='Average Number of Contradictions @k',
        xlabel='k',
        ylabel='Average Number of Contradictions @k',
        figsize=(10, 6),
    )

def evaluate_metrics__chest_imagenome_gold__report_level(
        report2labels_gold_filepath,
        fact_embedding_model_checkpoint_folder_paths,
        fact_embedding_model_names,
    ):

    assert len(fact_embedding_model_checkpoint_folder_paths) == len(fact_embedding_model_names)
    assert len(fact_embedding_model_checkpoint_folder_paths) > 0
    
    print_bold(f'Loading report2labels_gold from {report2labels_gold_filepath}')
    report2labels_gold = load_pickle(report2labels_gold_filepath)
    reports = report2labels_gold['reports']
    observation_labels = report2labels_gold['observation_labels']
    anatomy_labels = report2labels_gold['anatomy_labels']
    labels =  np.concatenate([observation_labels, anatomy_labels], axis=1)
    print(f'Number of reports: {len(reports)}')
    print(f'label.shape: {labels.shape}')

    # Compute accuracy between each pair of reports
    gold_accuracy_matrix = compute_gold_accuracy_matrix(labels)

    # Compute jaccard index between each pair of reports
    gold_jaccard_matrix = compute_chest_imagenome_gold_jaccard_matrix(observation_labels, anatomy_labels)

    # Compute BLEU score between each pair of reports
    bleu_score_matrix = compute_bleu_score_matrix(reports)

    # Compute ROUGE-L score between each pair of reports
    rougel_score_matrix = compute_rougel_score_matrix(reports)

    # Compute METEOR score between each pair of reports
    meteor_score_matrix = compute_meteor_score_matrix(reports)

    # Compute CIDEr-D score between each pair of reports
    ciderd_score_matrix = compute_ciderd_score_matrix(reports)

    # Compute BERTScore between each pair of reports
    bertscore_matrix = compute_bertscore_matrix(reports)

    # Compute CheXpert labeler accuracy between each pair of reports
    chexpert_accuracy_matrix = compute_chexpert_accuracy_matrix(reports)

    # Compute CheXpert labeler f1 between each pair of reports
    chexpert_f1_matrix = compute_chexpert_f1_matrix(reports)

    # Compute CheXbert accuracy between each pair of reports
    chexbert_accuracy_matrix = compute_chexbert_accuracy_matrix(reports)

    # Compute CheXbert f1 between each pair of reports
    chexbert_f1_matrix = compute_chexbert_f1_matrix(reports)

    # # Compute RadGraph jaccard index between each pair of reports
    # radgraph_jaccard_matrix = compute_radgraph_jaccard_matrix(reports)

    # Compute RadGraph F1 score between each pair of reports
    radgraph_f1_matrix = compute_radgraph_f1_matrix(reports)

    # Compute RadGraph F1 partial score between each pair of reports
    radgraph_f1_partial_matrix = compute_radgraph_f1_partial_matrix(reports)

    # Compute Fact Embedding Score between each pair of reports
    fact_embedding_score_matrix_list = [
        compute_fact_embedding_score_matrix(reports, checkpoint_folder_path)
        for checkpoint_folder_path in fact_embedding_model_checkpoint_folder_paths
    ]

    # Plot a correlation matrix between all metrics
    scores_list = [
        # gold_accuracy_matrix,
        gold_jaccard_matrix,
        bleu_score_matrix,
        rougel_score_matrix,
        meteor_score_matrix,
        ciderd_score_matrix,
        # radgraph_jaccard_matrix,
        radgraph_f1_matrix,
        radgraph_f1_partial_matrix,
        bertscore_matrix,
        chexpert_accuracy_matrix,
        chexbert_accuracy_matrix,
        chexpert_f1_matrix,
        chexbert_f1_matrix,
    ]
    method_names = [
        # 'Gold Accuracy',
        'Gold Jaccard',
        'BLEU',
        'ROUGE-L',
        'METEOR',
        'CIDEr-D',
        # 'RadGraph Jaccard',
        'RadGraph F1 Full',
        'RadGraph F1 Partial',
        'BERTScore',
        'CheXpert Accuracy',
        'CheXbert Accuracy',
        'CheXpert F1',
        'CheXbert F1',
    ]
    scores_list.extend(fact_embedding_score_matrix_list)
    method_names.extend(fact_embedding_model_names)
    assert len(scores_list) == len(method_names)
    flattend_scores_list = [score_matrix.flatten() for score_matrix in scores_list]
    plot_correlation_matrix(
        correlation_matrix=np.corrcoef(flattend_scores_list),
        method_names=method_names,
        title=f'Correlation Matrix between metrics (Chest ImaGenome Gold, report level, N={len(reports)})',
    )

    # remove gold metrics
    scores_list = scores_list[1:]
    method_names = method_names[1:]

    # Compute mean average accuracy @k for each metric
    print_bold('Computing mean average accuracy @k for each metric')
    max_k = 100
    mean_average_accuracy_at_k_list = []
    for method_name, score_matrix in zip(method_names, scores_list):
        print_bold(f'Computing mean average accuracy @k for {method_name}')
        mean_average_accuracy_at_k = compute_mean_average_accuracy_at_k(gold_accuracy_matrix, score_matrix, max_k)
        mean_average_accuracy_at_k_list.append(mean_average_accuracy_at_k)
        for k in (1, 5, 10, 20, 50, 100):
            print(f'  mean_average_accuracy_at_{k}: {mean_average_accuracy_at_k[k - 1]}')
    # Plot mean average accuracy @k for each metric
    plot_metric_lists(
        metric_lists=mean_average_accuracy_at_k_list,
        method_names=method_names,
        title=f'Mean Average Accuracy @k for each metric (Chest ImaGenome Gold, report level, N={len(reports)})',
        metric_name='Mean Average Accuracy @k',
        xlabel='k',
        ylabel='Mean Average Accuracy @k',
        figsize=(10, 6),
    )

    # Compute average jaccard index @k for each metric
    print_bold('Computing average jaccard index @k for each metric')
    max_k = 100
    mean_average_jaccard_at_k_list = []
    for method_name, score_matrix in zip(method_names, scores_list):
        print_bold(f'Computing average jaccard index @k for {method_name}')
        mean_average_jaccard_at_k = compute_mean_average_accuracy_at_k(gold_jaccard_matrix, score_matrix, max_k)
        mean_average_jaccard_at_k_list.append(mean_average_jaccard_at_k)
        for k in (1, 5, 10, 20, 50, 100):
            print(f'  mean_average_jaccard_at_{k}: {mean_average_jaccard_at_k[k - 1]}')
    # Plot mean average jaccard index @k for each metric
    plot_metric_lists(
        metric_lists=mean_average_jaccard_at_k_list,
        method_names=method_names,
        title=f'Mean Average Jaccard Index @k for each metric (Chest ImaGenome Gold, report level, N={len(reports)})',
        metric_name='Mean Average Jaccard Index @k',
        xlabel='k',
        ylabel='Mean Average Jaccard Index @k',
        figsize=(10, 6),
    )

class _NLI_Dataset_Name:
    MSCXRT_AND_RADNLI = 'MSCXRT_AND_RADNLI'
    CUSTOM_DATASET = 'CUSTOM_DATASET'
    @staticmethod
    def get_all():
        return [
            _NLI_Dataset_Name.MSCXRT_AND_RADNLI,
            _NLI_Dataset_Name.CUSTOM_DATASET,
        ]

def evaluate_metrics__NLI(
        fact_embedding_model_checkpoint_folder_paths,
        fact_embedding_model_names,
        use_radcliq=True,
        radcliq_device_id=None,
        dataset_name=_NLI_Dataset_Name.MSCXRT_AND_RADNLI,
        custom_dataset_csv_path=None,
        save_scores_to_disk=True,
        alias=None,
    ):

    assert len(fact_embedding_model_checkpoint_folder_paths) == len(fact_embedding_model_names)
    assert len(fact_embedding_model_checkpoint_folder_paths) > 0
    if use_radcliq:
        assert radcliq_device_id is not None

    premises = []
    hypotheses = []
    labels = []

    if dataset_name == _NLI_Dataset_Name.MSCXRT_AND_RADNLI:
    
        df = pd.read_csv(MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH)
        print(f'Number of MS_CXR_T samples: {len(df)}')
        for p, h, l in zip(df.sentence_1, df.sentence_2, df.category):
            if l == 'paraphrase':
                premises.append(p)
                hypotheses.append(h)
                labels.append(1)
            elif l == 'contradiction':
                premises.append(p)
                hypotheses.append(h)
                labels.append(0)
        mscxrt_count = len(labels)
        print(f'Number of entailment samples: {sum(labels)}')
        print(f'Number of contradiction samples: {len(labels) - sum(labels)}')

        rows = load_jsonl(RADNLI_DEV_JSONL_PATH)
        print(f'Number of RadNLI dev samples: {len(rows)}')
        for x in rows:
            p, h, l = x['sentence1'], x['sentence2'], x['gold_label']
            if l == 'entailment':
                premises.append(p)
                hypotheses.append(h)
                labels.append(1)
            elif l == 'contradiction':
                premises.append(p)
                hypotheses.append(h)
                labels.append(0)

        rows = load_jsonl(RADNLI_TEST_JSONL_PATH)
        print(f'Number of RadNLI test samples: {len(rows)}')
        for x in rows:
            p, h, l = x['sentence1'], x['sentence2'], x['gold_label']
            if l == 'entailment':
                premises.append(p)
                hypotheses.append(h)
                labels.append(1)
            elif l == 'contradiction':
                premises.append(p)
                hypotheses.append(h)
                labels.append(0)
        radnli_count = len(labels) - mscxrt_count
        print(f'Number of RadNLI samples: {radnli_count}')
        print(f'Number of entailment samples: {sum(labels[mscxrt_count:])}')
        print(f'Number of contradiction samples: {radnli_count - sum(labels[mscxrt_count:])}')

    elif dataset_name == _NLI_Dataset_Name.CUSTOM_DATASET:
        
        assert custom_dataset_csv_path is not None
        print(f'Loading custom dataset from {custom_dataset_csv_path}')
        df = pd.read_csv(custom_dataset_csv_path)
        print(f'Number of custom dataset samples: {len(df)}')
        for p, h, l in zip(df.reference, df.candidate, df.value):
            premises.append(p)
            hypotheses.append(h)
            if l == 'entailment':
                labels.append(1)
            elif l == 'contradiction':
                labels.append(0)
            else: assert False
        print(f'Number of entailment samples: {sum(labels)}')
        print(f'Number of contradiction samples: {len(labels) - sum(labels)}')

    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')

    # Compute BLEU score between each pair of sentences
    bleu_scores = _compute_bleu_scores(premises, hypotheses)

    # Compute ROUGE-L score between each pair of sentences
    rougel_scores = _compute_rougel_scores(premises, hypotheses)

    # Compute METEOR score between each pair of sentences
    tokenized_premises = word_tokenize_texts_in_parallel(premises)
    tokenized_hypotheses = word_tokenize_texts_in_parallel(hypotheses)
    meteor_scores = _compute_meteor_scores(tokenized_premises, tokenized_hypotheses)

    # Compute CIDEr-D score between each pair of sentences
    ciderd_scores = _compute_ciderd_scores(premises, hypotheses)

    # Compute BERTScore between each pair of sentences
    bertscore_scores = _compute_bertscore_scores(premises, hypotheses)

    # Compute CheXpert labeler accuracy between each pair of sentences
    chexpert_accuracy_scores = _compute_chexpert_accuracy_scores(premises, hypotheses)

    # Compute CheXpert labeler f1 between each pair of sentences
    chexpert_f1_scores = _compute_chexpert_f1_scores(premises, hypotheses)

    # Compute CheXbert accuracy between each pair of sentences
    chexbert_accuracy_scores = _compute_chexbert_accuracy_scores(premises, hypotheses)

    # Compute CheXbert f1 between each pair of sentences
    chexbert_f1_scores = _compute_chexbert_f1_scores(premises, hypotheses)

    # Compute RadGraph jaccard index between each pair of sentences
    radgraph_jaccard_scores = _compute_radgraph_jaccard_scores(premises, hypotheses)

    # Compute RadGraph F1 score between each pair of sentences
    radgraph_f1_scores = _compute_radgraph_f1_scores(premises, hypotheses)

    # Compute RadGraph F1 partial score between each pair of sentences
    radgraph_f1_partial_scores = _compute_radgraph_f1_partial_scores(premises, hypotheses)

    # Compute RadCliQ between each pair of sentences
    if use_radcliq:
        radcliq_scores = _compute_radcliq_scores(premises, hypotheses, device_id=radcliq_device_id)
        radcliq_scores['RadCliQ-v0'] = -1 * radcliq_scores['RadCliQ-v0'] # lower is better
        radcliq_scores['RadCliQ-v1'] = -1 * radcliq_scores['RadCliQ-v1'] # lower is better

    # Compute Fact Embedding Score between each pair of sentences
    fact_embedding_scores_list = [
        _compute_fact_embedding_scores(premises, hypotheses, checkpoint_folder_path)
        for checkpoint_folder_path in fact_embedding_model_checkpoint_folder_paths
    ]

    # Compute AUC for each metric
    scores_list = [
        bleu_scores,
        rougel_scores,
        meteor_scores,
        ciderd_scores,
        radgraph_jaccard_scores,
        radgraph_f1_scores,
        radgraph_f1_partial_scores,
        bertscore_scores,
        chexpert_accuracy_scores,
        chexbert_accuracy_scores,
        chexpert_f1_scores,
        chexbert_f1_scores,
    ]
    method_names = [
        'BLEU',
        'ROUGE-L',
        'METEOR',
        'CIDEr-D',
        'RadGraph Jaccard',
        'RadGraph F1',
        'RadGraph F1 Partial',
        'BERTScore',
        'CheXpert Accuracy',
        'CheXbert Accuracy',
        'CheXpert F1',
        'CheXbert F1',
    ]
    if use_radcliq:
        scores_list.extend([radcliq_scores[x] for x in RADCLIQ_METRIC_NAMES])
        method_names.extend([f'RadCliQ_{x}' for x in RADCLIQ_METRIC_NAMES])
    scores_list.extend(fact_embedding_scores_list)
    method_names.extend(fact_embedding_model_names)
    assert len(scores_list) == len(method_names)

    print_bold('Computing AUC for each metric')
    print()
    
    if dataset_name == _NLI_Dataset_Name.MSCXRT_AND_RADNLI:
        print_bold('MS-CXR-T:')
        for method_name, scores in zip(method_names, scores_list):
            print(f'{method_name} AUC: {auc(scores[:mscxrt_count], labels[:mscxrt_count])}')
        print()
        print_bold('RadNLI:')
        for method_name, scores in zip(method_names, scores_list):
            print(f'{method_name} AUC: {auc(scores[mscxrt_count:], labels[mscxrt_count:])}')
        print()
        print_bold('Overall:')
        for method_name, scores in zip(method_names, scores_list):
            print(f'{method_name} AUC: {auc(scores, labels)}')
    else:
        for method_name, scores in zip(method_names, scores_list):
            print(f'{method_name} AUC: {auc(scores, labels)}')

    if save_scores_to_disk:
        output = {
            'premises': premises,
            'hypotheses': hypotheses,
            'labels': labels,
        }
        for method_name, scores in zip(method_names, scores_list):
            output[method_name] = scores
        if dataset_name == _NLI_Dataset_Name.MSCXRT_AND_RADNLI:
            output['source'] = ['MS-CXR-T'] * mscxrt_count + ['RadNLI'] * radnli_count
            output_filepath = os.path.join(CACHE_DIR, f'report_gen_NLI_metrics_MSCXRT_AND_RADNLI.pkl')
        else:
            custom_filename = os.path.basename(custom_dataset_csv_path)
            if alias is not None:
                output_filepath = os.path.join(CACHE_DIR, f'report_gen_NLI_metrics_custom_{custom_filename}_{alias}.pkl')
            else:
                output_filepath = os.path.join(CACHE_DIR, f'report_gen_NLI_metrics_custom_{custom_filename}.pkl')
        print_bold(f'Saving scores to {output_filepath}')
        save_pickle(output, output_filepath)

def evaluate_metrics__custom_mimiccxr_radiologist_annotations():
    sentences, labels = load_mimiccxr_custom_radiologist_annotations()
    print(f'Number of sentences: {len(sentences)}')
    print(f'label.shape: {labels.shape}')

    # Compute accuracy between each pair of sentences
    gold_accuracy_matrix = compute_gold_accuracy_matrix(labels)

    # Compute relevance between each pair of sentences
    gold_relevance_matrix = compute_custom_mimiccxr_radiologist_annotations_relevance_matrix(labels)

    # Compute BLEU score between each pair of sentences
    bleu_score_matrix = compute_bleu_score_matrix(sentences)

    # Compute ROUGE-L score between each pair of sentences
    rougel_score_matrix = compute_rougel_score_matrix(sentences)

    # Compute METEOR score between each pair of sentences
    meteor_score_matrix = compute_meteor_score_matrix(sentences)

    # Compute CIDEr-D score between each pair of sentences
    ciderd_score_matrix = compute_ciderd_score_matrix(sentences)

    # Compute BERTScore between each pair of sentences
    bertscore_matrix = compute_bertscore_matrix(sentences)

    # Compute CheXpert labeler accuracy between each pair of sentences
    chexpert_accuracy_matrix = compute_chexpert_accuracy_matrix(sentences)

    # Compute CheXbert accuracy between each pair of sentences
    chexbert_accuracy_matrix = compute_chexbert_accuracy_matrix(sentences)

    # Compute RadGraph jaccard index between each pair of sentences
    radgraph_jaccard_matrix = compute_radgraph_jaccard_matrix(sentences)

    # Compute Fact Embedding Score between each pair of sentences
    fact_embedding_score_matrix = compute_fact_embedding_score_matrix(sentences)

    # Plot a correlation matrix between all metrics
    scores_list = [
        gold_accuracy_matrix,
        # gold_relevance_matrix,
        bleu_score_matrix,
        rougel_score_matrix,
        meteor_score_matrix,
        ciderd_score_matrix,
        radgraph_jaccard_matrix,
        bertscore_matrix,
        chexpert_accuracy_matrix,
        chexbert_accuracy_matrix,
        fact_embedding_score_matrix
    ]
    method_names = [
        'Gold Accuracy',
        # 'Gold Relevance',
        'BLEU',
        'ROUGE-L',
        'METEOR',
        'CIDEr-D',
        'RadGraph Jaccard',
        'BERTScore',
        'CheXpert Accuracy',
        'CheXbert Accuracy',
        'Fact Embedding Score'
    ]
    assert len(scores_list) == len(method_names)
    flattend_scores_list = [score_matrix.flatten() for score_matrix in scores_list]
    plot_correlation_matrix(
        correlation_matrix=np.corrcoef(flattend_scores_list),
        method_names=method_names,
        title=f'Correlation Matrix between metrics (N={len(sentences)})',
    )

    # Compute AUC for each metric
    print_bold('Computing AUC for each metric')
    for method_name, score_matrix in zip(method_names, scores_list):
        print(f'{method_name} AUC: {compute_AUC(gold_relevance_matrix, score_matrix)}')

    # Compute mean average accuracy @k for each metric
    print_bold('Computing mean average accuracy @k for each metric')
    max_k = 100
    mean_average_accuracy_at_k_list = []
    for method_name, score_matrix in zip(method_names, scores_list):
        print(f'Computing mean average accuracy @k for {method_name}')
        mean_average_accuracy_at_k = compute_mean_average_accuracy_at_k(gold_accuracy_matrix, score_matrix, max_k)
        mean_average_accuracy_at_k_list.append(mean_average_accuracy_at_k)
    # Plot mean average accuracy @k for each metric
    plot_metric_lists(
        metric_lists=mean_average_accuracy_at_k_list,
        method_names=method_names,
        title=f'Mean Average Accuracy @k for each metric (N={len(sentences)})',
        metric_name='Mean Average Accuracy @k',
        xlabel='k',
        ylabel='Mean Average Accuracy @k',
        figsize=(10, 6),
    )

    # Compute average number of contradictions @k for each metric
    print_bold('Computing average number of contradictions @k for each metric')

    # Compute relevance between each pair of sentences
    gold_contradiction_matrix = compute_custom_mimiccxr_radiologist_annotations_contradiction_matrix(labels)
    max_k = 100
    mean_num_contradictions_at_k_list = []
    for method_name, score_matrix in zip(method_names, scores_list):
        print(f'Computing average number of contradictions @k for {method_name}')
        mean_num_contradictions_at_k = compute_mean_contradictions_at_k(gold_contradiction_matrix, score_matrix, max_k)
        mean_num_contradictions_at_k_list.append(mean_num_contradictions_at_k)
    # Plot mean average number of contradictions @k for each metric
    plot_metric_lists(
        metric_lists=mean_num_contradictions_at_k_list,
        method_names=method_names,
        title=f'Average Number of Contradictions @k for each metric (N={len(sentences)})',
        metric_name='Average Number of Contradictions @k',
        xlabel='k',
        ylabel='Average Number of Contradictions @k',
        figsize=(10, 6),
    )

def evaluate_metrics__iuxray_tags__report_level(
        fact_embedding_model_checkpoint_folder_paths,
        fact_embedding_model_names,
    ):

    assert len(fact_embedding_model_checkpoint_folder_paths) == len(fact_embedding_model_names)
    assert len(fact_embedding_model_checkpoint_folder_paths) > 0

    from medvqa.datasets.iuxray import load_reports_and_tag_sets

    reports, tag_sets = load_reports_and_tag_sets()
    print(f'Number of reports: {len(reports)}')
    print(f'Number of tag_sets: {len(tag_sets)}')
    assert len(reports) == len(tag_sets)

    # Compute jaccard index between each pair of tag sets
    gold_jaccard_matrix = compute_iuxray_gold_tags_jaccard_matrix(tag_sets)

    # Compute BLEU score between each pair of reports
    bleu_score_matrix = compute_bleu_score_matrix(reports)

    # Compute ROUGE-L score between each pair of reports
    rougel_score_matrix = compute_rougel_score_matrix(reports)

    # Compute METEOR score between each pair of reports
    meteor_score_matrix = compute_meteor_score_matrix(reports)

    # Compute CIDEr-D score between each pair of reports
    ciderd_score_matrix = compute_ciderd_score_matrix(reports)

    # Compute BERTScore between each pair of reports
    bertscore_matrix = compute_bertscore_matrix(reports)

    # Compute CheXpert labeler accuracy between each pair of reports
    chexpert_accuracy_matrix = compute_chexpert_accuracy_matrix(reports)

    # Compute CheXpert labeler f1 between each pair of reports
    chexpert_f1_matrix = compute_chexpert_f1_matrix(reports)

    # Compute CheXbert accuracy between each pair of reports
    chexbert_accuracy_matrix = compute_chexbert_accuracy_matrix(reports)

    # Compute CheXbert f1 between each pair of reports
    chexbert_f1_matrix = compute_chexbert_f1_matrix(reports)

    # Compute RadGraph F1 score between each pair of reports
    radgraph_f1_matrix = compute_radgraph_f1_matrix(reports)

    # Compute RadGraph F1 partial score between each pair of reports
    radgraph_f1_partial_matrix = compute_radgraph_f1_partial_matrix(reports)

    # Compute Fact Embedding Score between each pair of reports
    fact_embedding_score_matrix_list = [
        compute_fact_embedding_score_matrix(reports, checkpoint_folder_path)
        for checkpoint_folder_path in fact_embedding_model_checkpoint_folder_paths
    ]

    # Plot a correlation matrix between all metrics
    scores_list = [
        gold_jaccard_matrix,
        bleu_score_matrix,
        rougel_score_matrix,
        meteor_score_matrix,
        ciderd_score_matrix,
        radgraph_f1_matrix,
        radgraph_f1_partial_matrix,
        bertscore_matrix,
        chexpert_accuracy_matrix,
        chexbert_accuracy_matrix,
        chexpert_f1_matrix,
        chexbert_f1_matrix,
    ]
    method_names = [
        'Gold Jaccard',
        'BLEU',
        'ROUGE-L',
        'METEOR',
        'CIDEr-D',
        'RadGraph F1 Full',
        'RadGraph F1 Partial',
        'BERTScore',
        'CheXpert Accuracy',
        'CheXbert Accuracy',
        'CheXpert F1',
        'CheXbert F1',
    ]
    scores_list.extend(fact_embedding_score_matrix_list)
    method_names.extend(fact_embedding_model_names)
    assert len(scores_list) == len(method_names)
    flattend_scores_list = [score_matrix.flatten() for score_matrix in scores_list]
    plot_correlation_matrix(
        correlation_matrix=np.corrcoef(flattend_scores_list),
        method_names=method_names,
        title=f'Correlation Matrix between metrics (IU X-ray, report level, N={len(reports)})',
    )

    # remove gold metrics
    scores_list = scores_list[1:]
    method_names = method_names[1:]

    # Compute average jaccard index @k for each metric
    print_bold('Computing average jaccard index @k for each metric')
    max_k = 100
    mean_average_jaccard_at_k_list = []
    for method_name, score_matrix in zip(method_names, scores_list):
        print_bold(f'Computing average jaccard index @k for {method_name}')
        mean_average_jaccard_at_k = compute_mean_average_accuracy_at_k(gold_jaccard_matrix, score_matrix, max_k)
        mean_average_jaccard_at_k_list.append(mean_average_jaccard_at_k)
        for k in (1, 5, 10, 20, 50, 100):
            print(f'  mean_average_jaccard_at_{k}: {mean_average_jaccard_at_k[k - 1]}')
    # Plot mean average jaccard index @k for each metric
    plot_metric_lists(
        metric_lists=mean_average_jaccard_at_k_list,
        method_names=method_names,
        title=f'Mean Average Jaccard Index @k for each metric (IU X-ray, report level, N={len(reports)})',
        metric_name='Mean Average Jaccard Index @k',
        xlabel='k',
        ylabel='Mean Average Jaccard Index @k',
        figsize=(10, 6),
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence2labels_gold_filepath', type=str)
    parser.add_argument('--task', type=str, required=True, choices=['chest_imagenome_gold', 'custom_mimiccxr_radiologist_annotations'])
    args = parser.parse_args()
    if args.task == 'chest_imagenome_gold':
        assert args.sentence2labels_gold_filepath is not None
        evaluate_metrics__chest_imagenome_gold(args.sentence2labels_gold_filepath)
    elif args.task == 'custom_mimiccxr_radiologist_annotations':
        evaluate_metrics__custom_mimiccxr_radiologist_annotations()
    else:
        raise ValueError(f'Unknown task: {args.task}')
