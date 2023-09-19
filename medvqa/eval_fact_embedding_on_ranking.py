import os
import argparse
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from tqdm import tqdm
from medvqa.datasets.iuxray import IUXRAY_REPORTS_MIN_JSON_PATH
from medvqa.utils.logging import print_orange
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.common import CACHE_DIR, parsed_args_to_dict
from medvqa.utils.files import (
    get_checkpoint_folder_path,
    get_results_folder_path,
    load_json, load_pickle,
    save_pickle,
)
from medvqa.utils.math import (
    rank_vectors_by_dot_product,
    rank_vectors_by_cosine_similarity,
    rank_vectors_by_euclidean_distance,
)
from medvqa.utils.metrics import jaccard_between_dicts

class _EvaluationModes:
    MIMICCXR_RADIOLOGIST_ANNOTATIONS = 'mimiccxr_radiologist_annotations'
    IUXRAY_CHEXPERT_LABELER = 'iuxray_chexpert_labeler'
    IUXRAY_CHEXBERT_LABELER = 'iuxray_chexbert_labeler'
    IUXRAY_RADGRAPH_LABELER = 'iuxray_radgraph_labeler'
    @staticmethod
    def get_all():
        return [
            _EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS,
            _EvaluationModes.IUXRAY_CHEXPERT_LABELER,
            _EvaluationModes.IUXRAY_CHEXBERT_LABELER,
            _EvaluationModes.IUXRAY_RADGRAPH_LABELER,
        ]

class _Methods:
    CXR_BERT_SPECIALIZED = 'BiomedVLP-CXR-BERT-specialized'
    BIOVIL_T = 'BiomedVLP-BioViL-T'
    BIOLINKBERT_LARGE = 'BioLinkBERT-large'
    PUBMEDBERT_LARGE = 'BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    BIO_CLINICAL_BERT = 'Bio_ClinicalBERT'
    ORACLE = 'oracle'
    @staticmethod
    def get_all():
        return [
            _Methods.CXR_BERT_SPECIALIZED,
            _Methods.BIOVIL_T,
            _Methods.BIOLINKBERT_LARGE,
            _Methods.PUBMEDBERT_LARGE,
            _Methods.BIO_CLINICAL_BERT,
            _Methods.ORACLE,
        ]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_mode', type=str, required=True, choices=_EvaluationModes.get_all())
    parser.add_argument('--method', type=str, required=True, choices=_Methods.get_all())
    parser.add_argument('--device', type=str, default='GPU', choices=['CPU', 'GPU'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--save_embeddings', action='store_true', default=False)
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['cosine', 'euclidean', 'dot_product'])
    return parser.parse_args(args=args)

def _load_mimiccxr_radiologist_annotations():
    df1 = pd.read_csv("/home/pdpino/workspace-medical-ai/report_generation/nlp-chex-gold-sentences/cxr-sentence-assessment-expert1.csv")
    df2 = pd.read_csv("/home/pdpino/workspace-medical-ai/report_generation/nlp-chex-gold-sentences/cxr-sentence-assessment-expert2.csv")
    assert df1['Sentence'].equals(df2['Sentence'])
    
    finding_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Lung Opacity', 'Pleural Effusion', 'Any other finding']
    n_rows = len(df1)
    n_labels = len(finding_columns)
    labels = np.empty((n_rows, n_labels * 2), dtype=np.int8) # 2 experts
    
    for i, col in enumerate(finding_columns):
        # Map string values to integers
        # Abnormal: 1, Normal: 0, Uncertain: -1, nan: -2
        values = df1[col]
        values.fillna(-2, inplace=True)
        values.replace('Uncertain', -1, inplace=True)
        values.replace('Normal', 0, inplace=True)
        values.replace('Abnormal', 1, inplace=True)
        labels[:, i] = values.values
        
        values = df2[col]
        values.fillna(-2, inplace=True)
        values.replace('Uncertain', -1, inplace=True)
        values.replace('Normal', 0, inplace=True)
        values.replace('Abnormal', 1, inplace=True)
        labels[:, i + n_labels] = values.values

    assert set(labels.flatten()) == {-2, -1, 0, 1}
    sentences = df1['Sentence'].values
    return sentences, labels

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

def _load_iuxray_sentences_and_chexpert_labels():
    save_path = os.path.join(CACHE_DIR, 'iuxray_sentences_and_chexpert_labels.pkl')
    if os.path.exists(save_path):
        print(f'Loading iuxray sentences and chexpert labels from {save_path}')
        return load_pickle(save_path)

    sentences = _load_iuxray_sentences()

    print('Loading Chexpert labeler')
    from medvqa.metrics.medical.chexpert import ChexpertLabeler
    chexpert_labeler = ChexpertLabeler()
    labels = chexpert_labeler.get_labels(sentences, update_cache_on_disk=True)
    output = {
        'sentences': sentences,
        'labels': labels,
    }
    print(f'Saving iuxray sentences and chexpert labels to {save_path}')
    save_pickle(output, save_path)
    return output

def _load_iuxray_sentences_and_chexbert_labels():
    save_path = os.path.join(CACHE_DIR, 'iuxray_sentences_and_chexbert_labels.pkl')
    if os.path.exists(save_path):
        print(f'Loading iuxray sentences and chexbert labels from {save_path}')
        return load_pickle(save_path)

    sentences = _load_iuxray_sentences()
    
    print('Loading CheXbert labeler')
    from medvqa.metrics.medical.chexbert import CheXbertLabeler
    chexbert_labeler = CheXbertLabeler(verbose=True)
    labels = chexbert_labeler.get_labels(sentences, update_cache_on_disk=True)
    output = {
        'sentences': sentences,
        'labels': labels,
    }
    print(f'Saving iuxray sentences and chexpert labels to {save_path}')
    save_pickle(output, save_path)
    return output

def _load_iuxray_sentences_and_radgraph_labels():
    save_path = os.path.join(CACHE_DIR, 'iuxray_sentences_and_radgraph_labels.pkl')
    if os.path.exists(save_path):
        print(f'Loading iuxray sentences and radgraph labels from {save_path}')
        return load_pickle(save_path)

    sentences = _load_iuxray_sentences()
    
    print('Loading RadGraph labeler')
    from medvqa.metrics.medical.radgraph import RadGraphLabeler
    radgraph_labeler = RadGraphLabeler(verbose=True)
    labels = radgraph_labeler.get_labels(sentences, update_cache_on_disk=True)
    output = {
        'sentences': sentences,
        'labels': labels,
    }
    print(f'Saving iuxray sentences and radgraph labels to {save_path}')
    save_pickle(output, save_path)
    return output

def evaluate(
    evaluation_mode,
    method,
    device,
    batch_size,
    num_workers,
    model_checkpoint_folder_path,
    top_k,
    save_embeddings,
    distance_metric,
):
    use_accuracy = False
    use_jaccard = False
    if evaluation_mode == _EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS:
        sentences, labels = _load_mimiccxr_radiologist_annotations()
        dataset_name = 'mimiccxr_rad_annotations'
        use_accuracy = True
    elif evaluation_mode == _EvaluationModes.IUXRAY_CHEXPERT_LABELER:
        tmp = _load_iuxray_sentences_and_chexpert_labels()
        sentences, labels = tmp['sentences'], tmp['labels']
        dataset_name = 'iuxray_chexpert_labeler'
        use_accuracy = True
    elif evaluation_mode == _EvaluationModes.IUXRAY_CHEXBERT_LABELER:
        tmp = _load_iuxray_sentences_and_chexbert_labels()
        sentences, labels = tmp['sentences'], tmp['labels']
        dataset_name = 'iuxray_chexbert_labeler'
        use_accuracy = True
    elif evaluation_mode == _EvaluationModes.IUXRAY_RADGRAPH_LABELER:
        tmp = _load_iuxray_sentences_and_radgraph_labels()
        sentences, labels = tmp['sentences'], tmp['labels']
        dataset_name = 'iuxray_radgraph'
        use_jaccard = True
    else:
        raise ValueError(f'Invalid evaluation_mode: {evaluation_mode}')
    assert sum([use_accuracy, use_jaccard]) == 1 # Only one metric can be used at a time
    
    n = len(sentences)
    print('len(sentences):', len(sentences))
    print('len(labels):', len(labels))

    if top_k > n:
        print_orange(f'WARNING: top_k ({top_k}) is greater than the number of sentences ({n}). Setting top_k to {n}.', bold=True)
        top_k = n
    
    if method in [_Methods.CXR_BERT_SPECIALIZED, _Methods.BIOVIL_T, _Methods.BIOLINKBERT_LARGE,
                  _Methods.PUBMEDBERT_LARGE, _Methods.BIO_CLINICAL_BERT]:
        # Obtain embeddings for each sentence
        embedding_extractor = CachedTextEmbeddingExtractor(
            model_name=method,
            device=device,
            model_checkpoint_folder_path=model_checkpoint_folder_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        embeddings = embedding_extractor.compute_text_embeddings(sentences)
        print('Embeddings shape:', embeddings.shape)
        assert embeddings.shape[0] == n
        
        # Define results folder path
        if model_checkpoint_folder_path is not None:
            results_folder_path = get_results_folder_path(model_checkpoint_folder_path)
        else:
            results_folder_path = get_results_folder_path(get_checkpoint_folder_path('fact_embedding', dataset_name, method))

        rank_vectors_func = None
        if distance_metric == 'cosine':
            rank_vectors_func = rank_vectors_by_cosine_similarity
        elif distance_metric == 'euclidean':
            rank_vectors_func = rank_vectors_by_euclidean_distance
        elif distance_metric == 'dot_product':
            rank_vectors_func = rank_vectors_by_dot_product
        else:
            raise ValueError(f'Invalid distance_metric: {distance_metric}')
        
        # Evaluate embeddings on ranking
        if use_accuracy:
            mean_average_accuracy_up_to = [0] * top_k
            for i in tqdm(range(n), mininterval=2):
                sorted_idxs = rank_vectors_func(embeddings, embeddings[i])
                accsum = 0
                for k in range(top_k):
                    accsum += np.mean(labels[i] == labels[sorted_idxs[k]])
                    mean_average_accuracy_up_to[k] += accsum / (k + 1)
            for k in range(top_k):
                mean_average_accuracy_up_to[k] /= n
            metrics_to_save = mean_average_accuracy_up_to
            metrics_save_path = os.path.join(results_folder_path,
                                             f'mean_average_accuracy_up_to_{top_k}({dataset_name},{distance_metric}).pkl')
        elif use_jaccard:
            mean_average_jaccard_up_to = [0] * top_k
            for i in tqdm(range(n), mininterval=2):
                sorted_idxs = rank_vectors_func(embeddings, embeddings[i])
                accsum = 0
                li = labels[i]
                for k in range(top_k):
                    accsum += jaccard_between_dicts(li, labels[sorted_idxs[k]])
                    mean_average_jaccard_up_to[k] += accsum / (k + 1)
            for k in range(top_k):
                mean_average_jaccard_up_to[k] /= n
            metrics_to_save = mean_average_jaccard_up_to
            metrics_save_path = os.path.join(results_folder_path,
                                             f'mean_average_jaccard_up_to_{top_k}({dataset_name},{distance_metric}).pkl')
        else: assert False

        if save_embeddings:
            embeddings_save_path = os.path.join(results_folder_path, f'embeddings({dataset_name}).pkl')
            if os.path.exists(embeddings_save_path):
                print_orange(f'WARNING: embeddings save path {embeddings_save_path} already exists. Skipping.', bold=True)
            else:
                print(f'Saving embeddings to {embeddings_save_path}')
                save_pickle({
                    'sentences': sentences,
                    'embeddings': embeddings,
                }, embeddings_save_path)

    elif method == _Methods.ORACLE:

        # Define results folder path
        results_folder_path = get_results_folder_path(get_checkpoint_folder_path('fact_embedding', dataset_name, 'oracle'))
        
        # Evaluate oracle on ranking
        if use_accuracy:
            mean_average_accuracy_up_to = [0] * top_k
            for i in tqdm(range(n), mininterval=2):
                # Sort labels by accuracy
                sorted_idxs = np.argsort(np.mean(labels == labels[i], axis=1))[::-1]
                accsum = 0
                for k in range(top_k):
                    accsum += np.mean(labels[i] == labels[sorted_idxs[k]])
                    mean_average_accuracy_up_to[k] += accsum / (k + 1)
            for k in range(top_k):
                mean_average_accuracy_up_to[k] /= n
            metrics_to_save = mean_average_accuracy_up_to
            metrics_save_path = os.path.join(results_folder_path, f'mean_average_accuracy_up_to_{top_k}({dataset_name}).pkl')
        elif use_jaccard:
            mean_average_jaccard_up_to = [0] * top_k
            for i in tqdm(range(n), mininterval=2):
                # Sort labels by f1 score
                sorted_idxs = np.argsort([jaccard_between_dicts(labels[i], labels[j]) for j in range(n)])[::-1]
                accsum = 0
                for k in range(top_k):
                    accsum += jaccard_between_dicts(labels[i], labels[sorted_idxs[k]])
                    mean_average_jaccard_up_to[k] += accsum / (k + 1)
            for k in range(top_k):
                mean_average_jaccard_up_to[k] /= n
            metrics_to_save = mean_average_jaccard_up_to
            metrics_save_path = os.path.join(results_folder_path, f'mean_average_jaccard_up_to_{top_k}({dataset_name}).pkl')
        else: assert False
        
        if save_embeddings: # Save labels as embeddings in the case of oracle
            labels_save_path = os.path.join(results_folder_path, 'labels.pkl')
            if os.path.exists(labels_save_path):
                print_orange(f'WARNING: labels save path {labels_save_path} already exists. Skipping.', bold=True)
            else:
                print(f'Saving labels to {labels_save_path}')
                save_pickle({
                    'sentences': sentences,
                    'labels': labels,
                }, labels_save_path)

    else:
        raise ValueError(f'Invalid method: {method}')
    
    print(f'Saving metrics to {metrics_save_path}')
    save_pickle(metrics_to_save, metrics_save_path)

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)