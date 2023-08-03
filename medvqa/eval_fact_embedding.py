import os
import argparse
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from tqdm import tqdm
from medvqa.datasets.iuxray import IUXRAY_REPORTS_MIN_JSON_PATH
from medvqa.metrics.medical.chexpert import ChexpertLabeler

from medvqa.models.huggingface_utils import (
    compute_text_embeddings_with_BiomedVLP_BioVilT,
    compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized,
)
from medvqa.utils.common import CACHE_DIR, parsed_args_to_dict
from medvqa.utils.files import get_checkpoint_folder_path, get_results_folder_path, load_json, load_pickle, save_pickle
from medvqa.utils.math import rank_vectors_by_dot_product

class _EvaluationModes:
    MIMICCXR_RADIOLOGIST_ANNOTATIONS = 'mimiccxr_radiologist_annotations'
    IUXRAY_CHEXPERT_LABELER = 'iuxray_chexpert_labeler'
    @staticmethod
    def get_all():
        return [
            _EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS,
            _EvaluationModes.IUXRAY_CHEXPERT_LABELER,
        ]

class _Methods:
    CXR_BERT_SPECIALIZED = 'cxr-bert-specialized'
    BIOVIL_T = 'biovil-t'
    ORACLE = 'oracle'
    @staticmethod
    def get_all():
        return [
            _Methods.CXR_BERT_SPECIALIZED,
            _Methods.BIOVIL_T,
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
    return parser.parse_args(args=args)

def load_mimiccxr_radiologist_annotations():
    df1 = pd.read_csv("/home/pdpino/workspace-medical-ai/report_generation/nlp-chex-gold-sentences/cxr-sentence-assessment-expert1.csv")
    df2 = pd.read_csv("/home/pdpino/workspace-medical-ai/report_generation/nlp-chex-gold-sentences/cxr-sentence-assessment-expert2.csv")
    assert df1['Sentence'].equals(df2['Sentence'])
    
    _finding_columns = [ 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Lung Opacity', 'Pleural Effusion', 'Any other finding']
    n_rows = len(df1)
    n_labels = len(_finding_columns)
    labels = np.empty((n_rows, n_labels * 2), dtype=np.int8) # 2 experts
    
    for i, col in enumerate(_finding_columns):
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

def load_iuxray_sentences_and_chexpert_labels():
    save_path = os.path.join(CACHE_DIR, 'iuxray_sentences_and_chexpert_labels.pkl')
    if os.path.exists(save_path):
        print(f'Loading iuxray sentences and chexpert labels from {save_path}')
        return load_pickle(save_path)

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
    
    print('Loading Chexpert labeler')
    chexpert_labeler = ChexpertLabeler()
    labels = chexpert_labeler.get_labels(sentences, update_cache_on_disk=True)
    output = {
        'sentences': sentences,
        'labels': labels,
    }
    print(f'Saving iuxray sentences and chexpert labels to {save_path}')
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
):
    if evaluation_mode == _EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS:
        sentences, labels = load_mimiccxr_radiologist_annotations()
        dataset_name = 'mimiccxr_rad_annotations'
    elif evaluation_mode == _EvaluationModes.IUXRAY_CHEXPERT_LABELER:
        tmp = load_iuxray_sentences_and_chexpert_labels()
        sentences, labels = tmp['sentences'], tmp['labels']
        dataset_name = 'iuxray_chexpert_labeler'
    else:
        raise ValueError(f'Invalid evaluation_mode: {evaluation_mode}')
    
    n = len(sentences)
    mean_average_accuracy_up_to = [0] * top_k
    
    if method in [_Methods.CXR_BERT_SPECIALIZED, _Methods.BIOVIL_T]:
        # Obtain embeddings for each sentence
        if method == _Methods.CXR_BERT_SPECIALIZED:
            embeddings = compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized(
                texts=sentences, device=device, batch_size=batch_size, num_workers=num_workers,
                model_checkpoint_folder_path=model_checkpoint_folder_path)
        elif method == _Methods.BIOVIL_T:
            embeddings = compute_text_embeddings_with_BiomedVLP_BioVilT(
                texts=sentences, device=device, batch_size=batch_size, num_workers=num_workers,
                model_checkpoint_folder_path=model_checkpoint_folder_path)
        else: assert False
        print('Embeddings shape:', embeddings.shape)
        assert embeddings.shape[0] == n
        # Evaluate embeddings on ranking
        for i in tqdm(range(n), mininterval=2):
            sorted_idxs = rank_vectors_by_dot_product(embeddings, embeddings[i])
            accsum = 0
            for k in range(top_k):
                accsum += np.mean(labels[i] == labels[sorted_idxs[k]])
                mean_average_accuracy_up_to[k] += accsum / (k + 1)

        if model_checkpoint_folder_path is not None:
            results_folder_path = get_results_folder_path(model_checkpoint_folder_path)
        else:
            results_folder_path = get_results_folder_path(get_checkpoint_folder_path('fact_embedding', dataset_name, method))

        if save_embeddings:
            embeddings_save_path = os.path.join(results_folder_path, f'embeddings({dataset_name}).pkl')
            print(f'Saving embeddings to {embeddings_save_path}')
            save_pickle({
                'sentences': sentences,
                'embeddings': embeddings,
            }, embeddings_save_path)

    elif method == _Methods.ORACLE:
        # Evaluate oracle on ranking
        for i in tqdm(range(n), mininterval=2):
            # Sort labels by accuracy
            sorted_idxs = np.argsort(np.mean(labels == labels[i], axis=1))[::-1]
            accsum = 0
            for k in range(top_k):
                accsum += np.mean(labels[i] == labels[sorted_idxs[k]])
                mean_average_accuracy_up_to[k] += accsum / (k + 1)

        results_folder_path = get_results_folder_path(get_checkpoint_folder_path('fact_embedding', dataset_name, 'oracle'))
        if save_embeddings: # Save labels as embeddings in the case of oracle
            labels_save_path = os.path.join(results_folder_path, 'labels.pkl')
            print(f'Saving labels to {labels_save_path}')
            save_pickle({
                'sentences': sentences,
                'labels': labels,
            }, labels_save_path)

    else:
        raise ValueError(f'Invalid method: {method}')
    
    for k in range(top_k):
        mean_average_accuracy_up_to[k] /= n

    mean_average_acc_save_path = os.path.join(results_folder_path, f'mean_average_accuracy_up_to_{top_k}({dataset_name}).pkl')
    print(f'Saving mean average accuracy up to {top_k} to {mean_average_acc_save_path}')
    save_pickle(mean_average_accuracy_up_to, mean_average_acc_save_path)


if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)