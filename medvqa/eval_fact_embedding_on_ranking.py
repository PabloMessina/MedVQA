import os
import argparse
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from tqdm import tqdm
from medvqa.datasets.iuxray import IUXRAY_REPORTS_MIN_JSON_PATH
from medvqa.eval_report_gen_metrics import (
    compute_AUC,
    compute_chest_imagenome_gold_contradiction_matrix,
    compute_chest_imagenome_gold_relevance_matrix,
    compute_gold_accuracy_matrix,
    compute_mean_average_accuracy_at_k,
    compute_mean_contradictions_at_k,
)
from medvqa.evaluation.ranking_evaluation_utils import load_mimiccxr_custom_radiologist_annotations
from medvqa.utils.logging import print_magenta, print_orange
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor, SupportedHuggingfaceMedicalBERTModels
from medvqa.utils.common import CACHE_DIR, parsed_args_to_dict
from medvqa.utils.files import load_json, load_pickle, save_pickle
from medvqa.utils.math import (
    rank_vectors_by_dot_product,
)
from medvqa.utils.metrics import jaccard_between_dicts

class _EvaluationModes:
    MIMICCXR_RADIOLOGIST_ANNOTATIONS = 'mimiccxr_radiologist_annotations'
    IUXRAY_WITH_AUTOMATIC_LABELERS = 'iuxray_with_automatic_labelers' # chexpert + chexbert + radgraph
    CHEST_IMAGENOME_GOLD = 'chest_imagenome_gold'
    @staticmethod
    def get_all():
        return [
            _EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS,
            _EvaluationModes.IUXRAY_WITH_AUTOMATIC_LABELERS,
            _EvaluationModes.CHEST_IMAGENOME_GOLD,
        ]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_mode', type=str, required=True, choices=_EvaluationModes.get_all())
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all() + ['CheXbert'])
    parser.add_argument('--device', type=str, default='GPU', choices=['CPU', 'GPU', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['cosine', 'dot_product'])
    parser.add_argument('--average_token_embeddings', action='store_true', default=False)
    parser.add_argument('--chest_imagenome_sentence2labels_gold_filepath', type=str, default=None)
    return parser.parse_args(args=args)

def _load_iuxray_sentences():
    print(f'Loading iuxray reports from {IUXRAY_REPORTS_MIN_JSON_PATH}')
    reports = load_json(IUXRAY_REPORTS_MIN_JSON_PATH)
    sentences = set()
    for r in reports.values():
        findings = r['findings']
        impression = r['impression']
        for x in (findings, impression):
            if x:
                for s in sent_tokenize(x):
                    s = ' '.join(s.split()) # Remove extra spaces
                    sentences.add(s)
    sentences = list(sentences)
    sentences.sort(key=lambda x: (len(x), x)) # Sort by length and then alphabetically
    sentences = [s for s in sentences if any(c.isalpha() for c in s)] # Remove sentences without any alphabetic character
    print(f'Number of sentences: {len(sentences)}')
    print(f'Shortest sentence: {sentences[0]}')
    print(f'Longest sentence: {sentences[-1]}')
    return sentences

def _load_mimiccxr_sentences_and_radiologist_based_relevant_sentences():
    save_path = os.path.join(CACHE_DIR, 'mimiccxr_sentences_and_relevant.pkl')
    if os.path.exists(save_path):
        print(f'Loading mimiccxr sentences and relevant sentences from {save_path}')
        return load_pickle(save_path)
    
    sentences, labels = load_mimiccxr_custom_radiologist_annotations()
    n = len(sentences)
    relevant_sentences = [set() for _ in range(n)]

    def _a_entails_b(a, b):
        for i in range(len(b)):
            if b[i] == -2: # nan
                continue
            if a[i] != b[i]:
                return False
        return True

    for i in tqdm(range(n), mininterval=2):
        for j in range(i+1, n):
            if _a_entails_b(labels[i], labels[j]) or _a_entails_b(labels[j], labels[i]):
                relevant_sentences[i].add(j)
                relevant_sentences[j].add(i)

    print(f'Saving mimiccxr sentences and relevant sentences to {save_path}')
    output = {
        'sentences': sentences,
        'relevant_sentences': relevant_sentences,
    }
    save_pickle(output, save_path)
    return output

def _load_iuxray_sentences_and_automatic_labeler_based_relevant_sentences(thr1=0.4, thr2=0.2):
    save_path = os.path.join(CACHE_DIR, f'iuxray_sentences_and_relevant(thr1={thr1},thr2={thr2}).pkl')
    if os.path.exists(save_path):
        print(f'Loading iuxray sentences and relevant sentences from {save_path}')
        return load_pickle(save_path)

    sentences = _load_iuxray_sentences()

    print('Loading CheXpert labeler')
    from medvqa.metrics.medical.chexpert import ChexpertLabeler
    chexpert_labeler = ChexpertLabeler()
    chexpert_labels = chexpert_labeler.get_labels(sentences, update_cache_on_disk=True)

    print('Loading CheXbert labeler')
    from medvqa.metrics.medical.chexbert import CheXbertLabeler
    chexbert_labeler = CheXbertLabeler()
    chexbert_labels = chexbert_labeler.get_labels(sentences, update_cache_on_disk=True)

    print('Loading RadGraph labeler')
    from medvqa.metrics.medical.radgraph import RadGraphLabeler
    radgraph_labeler = RadGraphLabeler()
    radgraph_labels = radgraph_labeler.get_labels(sentences, update_cache_on_disk=True)

    # Obtain relevant sentences for each sentence
    relevant_sentences = [set() for _ in range(len(sentences))]
    for i in tqdm(range(len(sentences)), mininterval=2):
        if len(radgraph_labels[i]) == 0: # Skip sentences without any radgraph labels
            continue
        for j in range(i+1, len(sentences)):
            js = jaccard_between_dicts(radgraph_labels[i], radgraph_labels[j])
            if js >= thr1 or (js >= thr2 and (np.all(chexpert_labels[i] == chexpert_labels[j]) or \
                np.all(chexbert_labels[i] == chexbert_labels[j]))):
                relevant_sentences[i].add(j)
                relevant_sentences[j].add(i)
    print(f'Saving iuxray sentences and relevant sentences to {save_path}')
    output = {
        'sentences': sentences,
        'relevant_sentences': relevant_sentences,
    }
    save_pickle(output, save_path)
    return output

def _compute_AUC(sorted_idxs, relevant_idxs, query_idx):
    # Compute AUC
    assert query_idx not in sorted_idxs
    assert query_idx not in relevant_idxs
    assert len(relevant_idxs) > 0
    assert len(sorted_idxs) > len(relevant_idxs)
    n_relevant = len(relevant_idxs)
    n_irrelevant = len(sorted_idxs) - n_relevant
    n_pairs = n_relevant * n_irrelevant
    n_wrong_pairs = 0
    n_irrelevant_below = 0
    relevant_found = 0
    for idx in sorted_idxs:
        if idx in relevant_idxs:
            n_wrong_pairs += n_irrelevant_below
            relevant_found += 1
        else:
            n_irrelevant_below += 1
    assert relevant_found == n_relevant
    AUC = 1 - (n_wrong_pairs / n_pairs)
    return AUC

def evaluate(
    evaluation_mode,
    model_name,
    device,
    batch_size,
    num_workers,
    model_checkpoint_folder_path,
    distance_metric,
    average_token_embeddings,
    chest_imagenome_sentence2labels_gold_filepath,
):
    if evaluation_mode == _EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS:
        tmp = _load_mimiccxr_sentences_and_radiologist_based_relevant_sentences()
        sentences, relevant_sentences = tmp['sentences'], tmp['relevant_sentences']
        print('len(sentences):', len(sentences))
        print('len(relevant_sentences):', len(relevant_sentences))
    elif evaluation_mode == _EvaluationModes.IUXRAY_WITH_AUTOMATIC_LABELERS:
        tmp = _load_iuxray_sentences_and_automatic_labeler_based_relevant_sentences()
        sentences, relevant_sentences = tmp['sentences'], tmp['relevant_sentences']
        print('len(sentences):', len(sentences))
        print('len(relevant_sentences):', len(relevant_sentences))
    elif evaluation_mode == _EvaluationModes.CHEST_IMAGENOME_GOLD:
        assert chest_imagenome_sentence2labels_gold_filepath is not None
        print(f'Loading sentence2labels_gold from {chest_imagenome_sentence2labels_gold_filepath}')
        sentence2labels_gold = load_pickle(chest_imagenome_sentence2labels_gold_filepath)
        sentences = sentence2labels_gold['phrases']
        observation_labels = sentence2labels_gold['observation_labels']
        anatomy_labels = sentence2labels_gold['anatomy_labels']
        labels =  np.concatenate([observation_labels, anatomy_labels], axis=1)
        print(f'Number of sentences: {len(sentences)}')
        print(f'label.shape: {labels.shape}')
        gold_relevance_matrix = compute_chest_imagenome_gold_relevance_matrix(observation_labels, anatomy_labels)
        gold_accuracy_matrix = compute_gold_accuracy_matrix(labels)
        gold_contradiction_matrix = compute_chest_imagenome_gold_contradiction_matrix(observation_labels)
    else:
        raise ValueError(f'Invalid evaluation_mode: {evaluation_mode}')
    
    n = len(sentences)
    
    # Obtain embeddings for each sentence
    if model_name in SupportedHuggingfaceMedicalBERTModels.get_all():
        embedding_extractor = CachedTextEmbeddingExtractor(
            model_name=model_name,
            device=device,
            model_checkpoint_folder_path=model_checkpoint_folder_path,
            batch_size=batch_size,
            num_workers=num_workers,
            average_token_embeddings=average_token_embeddings,
        )
        embeddings = embedding_extractor.compute_text_embeddings(sentences)
    elif model_name == 'CheXbert':
        from medvqa.metrics.medical.chexbert import CheXbertLabeler
        chexbert_labeler = CheXbertLabeler(device=device)
        embeddings = chexbert_labeler.get_embeddings(sentences)
    else:
        raise ValueError(f'Invalid model_name: {model_name}')
    print('Embeddings shape:', embeddings.shape)
    assert embeddings.shape[0] == n

    if evaluation_mode in (_EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS, _EvaluationModes.IUXRAY_WITH_AUTOMATIC_LABELERS):
        rank_vectors_func = None
        if distance_metric == 'cosine':
            print('Normalizing embeddings (for cosine similarity)')
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # Normalize embeddings
            rank_vectors_func = rank_vectors_by_dot_product # Cosine similarity is equivalent to dot product when embeddings are normalized
        elif distance_metric == 'dot_product':
            rank_vectors_func = rank_vectors_by_dot_product
        else:
            raise ValueError(f'Invalid distance_metric: {distance_metric}')
        
        # Evaluate embeddings on ranking task with each sentence as query
        print_orange('Evaluating embeddings on ranking task with each sentence as query', bold=True)
        mean_AUC = 0
        mean_relevant = 0
        count = 0
        for i in tqdm(range(n), mininterval=2):
            relevant_idxs = relevant_sentences[i]
            if len(relevant_idxs) == 0:
                continue
            sorted_idxs = rank_vectors_func(embeddings, embeddings[i])
            sorted_idxs = [idx for idx in sorted_idxs if idx != i] # Remove query_idx
            AUC = _compute_AUC(sorted_idxs, relevant_idxs, query_idx=i)
            mean_AUC += AUC
            mean_relevant += len(relevant_idxs)
            count += 1
        mean_AUC /= count
        mean_relevant /= count
        print_magenta(f'mean_AUC: {mean_AUC:.4f}', bold=True)
        print(f'mean_relevant: {mean_relevant:.4f}')
        print(f'count: {count} / {n} ({100 * count / n:.2f}%)')
    
    elif evaluation_mode == _EvaluationModes.CHEST_IMAGENOME_GOLD:

        if distance_metric == 'cosine':
            print('Normalizing embeddings (for cosine similarity)')
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # Normalize embeddings
        print('Computing score matrix')
        score_matrix = np.dot(embeddings, embeddings.T)
        print('score_matrix.shape:', score_matrix.shape)
        
        # Compute AUC
        print_magenta(f'mean_AUC: {compute_AUC(gold_relevance_matrix, score_matrix):.4f}', bold=True)

        # Compute mean average accuracy @k for each metric
        max_k = 200
        mean_average_accuracy_at_k = compute_mean_average_accuracy_at_k(gold_accuracy_matrix, score_matrix, max_k)
        for k in (1, 5, 10, 20, 50, 100, 200):
            print(f'mean_average_accuracy_at_{k}: {mean_average_accuracy_at_k[k-1]:.4f}')

        # Compute mean average contradiction @k
        mean_num_contradictions_at_k = compute_mean_contradictions_at_k(gold_contradiction_matrix, score_matrix, max_k)
        for k in (1, 5, 10, 20, 50, 100, 200):
            print(f'mean_num_contradictions_at_{k}: {mean_num_contradictions_at_k[k-1]:.4f}')
        

class SentenceRanker:
    def __init__(self, dataset_name):
        assert dataset_name in _EvaluationModes.get_all()
        if dataset_name == _EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS:
            tmp = _load_mimiccxr_sentences_and_radiologist_based_relevant_sentences()
            self.sentences, self.relevant_sentences = tmp['sentences'], tmp['relevant_sentences']
        elif dataset_name == _EvaluationModes.IUXRAY_WITH_AUTOMATIC_LABELERS:
            tmp = _load_iuxray_sentences_and_automatic_labeler_based_relevant_sentences()
            self.sentences, self.relevant_sentences = tmp['sentences'], tmp['relevant_sentences']
        else:
            raise ValueError(f'Invalid dataset_name: {dataset_name}')
        self.cache = {}
    
    def rank_sentences(self, query_idx, model_name, checkpoint_folder_path=None, average_token_embeddings=False,
                       top_k=10, batch_size=32, num_workers=4):
        assert isinstance(average_token_embeddings, bool)
        assert top_k > 0
        assert top_k <= len(self.sentences)
        assert 0 <= query_idx < len(self.sentences)
        if checkpoint_folder_path is None:
            key = (model_name, average_token_embeddings)
        else:
            key = (model_name, checkpoint_folder_path, average_token_embeddings)
        try:
            embedding_extractor = self.cache[key]
        except KeyError:
            embedding_extractor = self.cache[key] = CachedTextEmbeddingExtractor(
                model_name=model_name,
                device='GPU',
                model_checkpoint_folder_path=checkpoint_folder_path,
                batch_size=batch_size,
                num_workers=num_workers,
                average_token_embeddings=average_token_embeddings,
            )
        embeddings = embedding_extractor.compute_text_embeddings(self.sentences)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sorted_idxs = rank_vectors_by_dot_product(embeddings, embeddings[query_idx])
        sorted_sentences = [self.sentences[i] for i in sorted_idxs[:top_k]]
        is_relevant = [1 if i in self.relevant_sentences[query_idx] else 0 for i in sorted_idxs[:top_k]]
        return sorted_sentences, is_relevant

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)