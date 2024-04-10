import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import multiprocessing
from collections import Counter
from sklearn.metrics import f1_score
from medvqa.datasets.seq2seq.seq2seq_dataset_management import (
    load_gpt4_nli_examples_filepaths,
    load_ms_cxr_t_temporal_sentence_similarity_v1_data,
    load_radnli_dev_data,
    load_radnli_test_data,
    load_report_nli_examples_filepaths,
)
from medvqa.datasets.text_data_utils import sentence_tokenize_texts_in_parallel
from medvqa.evaluation.plots import plot_metrics
from medvqa.metrics.medical.chexbert import CheXbertLabeler
from medvqa.scripts.mimiccxr.generate_fact_based_report_nli_examples_with_openai import LABEL_BASED_FACTS
from medvqa.utils.common import FAST_CACHE_DIR, parsed_args_to_dict
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor, SupportedHuggingfaceMedicalBERTModels
from medvqa.utils.files import get_file_path_with_hashing_if_too_long, load_pickle, save_pickle
from medvqa.utils.logging import print_blue, print_bold
from medvqa.utils.metrics import best_threshold_and_f1_score

_NLI_LABEL2ID = {'entailment': 1, 'neutral': 0, 'contradiction': -1}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--huggingface_model_name', type=str, default=None, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--model_name', type=str, default=None, choices=['CheXbert'])
    parser.add_argument('--model_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--num_processes', type=int, required=True)
    parser.add_argument('--gpt4_nli_examples_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--report_nli_input_output_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument('--average_token_embeddings', action='store_true')
    return parser.parse_args(args=args)

_shared_premise_sentences = None
_shared_hypotheses = None
_shared_embeddings = None
_shared_sentence2idx = None
def _compute_premise_hypothesis_maxsim(i):
    premise_sentences = _shared_premise_sentences[i]
    if len(premise_sentences) == 0:
        return 0
    hypothesis = _shared_hypotheses[i]
    hypothesis_embedding = _shared_embeddings[_shared_sentence2idx[hypothesis]] # e.g. [768]
    premise_idxs = [_shared_sentence2idx[p] for p in premise_sentences]
    premise_embeddings = _shared_embeddings[premise_idxs] # e.g. [n, 768]
    premise_hypothesis_sim = np.sum(premise_embeddings * hypothesis_embedding, axis=1) # e.g. [n]
    max_sim = np.max(premise_hypothesis_sim)
    return max_sim

def _compute_confusion_matrix(scores, labels, ent_threshold, contr_threshold, sns_font_scale=1.8, font_size=25):
    if isinstance(ent_threshold, float):
        ent_threshold = np.full(len(labels), ent_threshold)
    if isinstance(contr_threshold, float):
        contr_threshold = np.full(len(labels), contr_threshold)
    assert np.all(ent_threshold >= contr_threshold)
    preds = np.zeros_like(scores) # neutral
    preds[scores >= ent_threshold] = 1 # entailment
    preds[scores <= contr_threshold] = -1 # contradiction
    # Compute accuracy
    acc = np.sum(preds == labels) / len(labels)
    print_blue(f"Accuracy: {acc:.4f}", bold=True)
    # Plot a confusion matrix
    cm = confusion_matrix(labels, preds, labels=[1, 0, -1])
    # Use label names instead of indices
    label_names = ['entailment', 'neutral', 'contradiction']
    plt.figure(figsize=(8, 6))
    sns.set_theme(font_scale=sns_font_scale)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted', fontsize=font_size)
    plt.ylabel('True', fontsize=font_size)
    plt.show()

def evaluate(
    model_name,
    huggingface_model_name,
    device,
    model_checkpoint_folder_path,
    batch_size,
    num_workers,
    num_processes,
    gpt4_nli_examples_filepaths,
    report_nli_input_output_jsonl_filepaths,
    average_token_embeddings,
    save_thresholds=False,
    thresholds_save_path=None,
    f1_figsize=(10, 35),
):    
    assert (model_name != None) != (huggingface_model_name != None), 'Exactly one of model_name or huggingface_model_name must be provided'
    if huggingface_model_name is not None:
        embedding_extractor = CachedTextEmbeddingExtractor(
            model_name=huggingface_model_name,
            device=device,
            model_checkpoint_folder_path=model_checkpoint_folder_path,
            batch_size=batch_size,
            num_workers=num_workers,
            average_token_embeddings=average_token_embeddings,
        )
        get_embeddings_func = embedding_extractor.compute_text_embeddings
    elif model_name is not None:
        if model_name == 'CheXbert':
            chexbert_labeler = CheXbertLabeler(device=device)
            get_embeddings_func = chexbert_labeler.get_embeddings
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
    else:
        raise ValueError('Exactly one of model_name or huggingface_model_name must be provided')

    radnli_dev_input_texts, radnli_dev_output_texts = load_radnli_dev_data(nli1_only=True)
    radnli_test_input_texts, radnli_test_output_texts = load_radnli_test_data(nli1_only=True)
    mscxrt_input_texts, mscxrt_output_texts = load_ms_cxr_t_temporal_sentence_similarity_v1_data(nli1_only=True)
    input_texts = radnli_dev_input_texts + radnli_test_input_texts + mscxrt_input_texts
    if gpt4_nli_examples_filepaths is not None:
        gpt4_nli_input_texts, gpt4_nli_output_texts = load_gpt4_nli_examples_filepaths(gpt4_nli_examples_filepaths, nli1_only=True)
        input_texts += gpt4_nli_input_texts
    if report_nli_input_output_jsonl_filepaths is not None:
        report_nli_input_texts, report_nli_output_texts, _ = load_report_nli_examples_filepaths(report_nli_input_output_jsonl_filepaths, nli1_only=True)
        input_texts += report_nli_input_texts
    premises = [None] * len(input_texts)
    hypotheses = [None] * len(input_texts)
    unique_sentences = set()
    for i in range(len(input_texts)):
        hidx = input_texts[i].index(' #H: ')
        input_text = input_texts[i]
        premises[i] = input_text[6:hidx]
        hypotheses[i] = input_text[hidx+5:]
        unique_sentences.add(hypotheses[i])
        if i < 2:
            print_bold('Example premise-hypothesis pair:')
            print(premises[i])
            print(hypotheses[i])
    premise_sentences = sentence_tokenize_texts_in_parallel(premises)
    for sentences in premise_sentences:
        for sentence in sentences:
            unique_sentences.add(sentence)

    sentences = list(unique_sentences)
    sentence2idx = {s: i for i, s in enumerate(sentences)}
    print(f"Total sentences: {len(sentences)}")
    embeddings = get_embeddings_func(sentences)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) # normalize for cosine similarity
    print(f"embeddings.shape: {embeddings.shape}")

    # Compute similarity scores
    global _shared_premise_sentences
    global _shared_hypotheses
    global _shared_embeddings
    global _shared_sentence2idx
    _shared_premise_sentences = premise_sentences
    _shared_hypotheses = hypotheses
    _shared_embeddings = embeddings
    _shared_sentence2idx = sentence2idx
    print(f"len(premise_sentences): {len(premise_sentences)}")
    print(Counter([len(p) for p in premise_sentences]))
    with multiprocessing.Pool(num_processes) as p:
        max_sims = p.map(_compute_premise_hypothesis_maxsim, range(len(premise_sentences)))
    max_sims = np.array(max_sims)
    print(f"max_sims: {max_sims[:5]}")
    print(f"len(max_sims): {len(max_sims)}")

    # Parse NLI labels
    labels = []
    for output_texts in [radnli_dev_output_texts, radnli_test_output_texts, mscxrt_output_texts]:
        for output_text in output_texts:
            labels.append(_NLI_LABEL2ID[output_text])
    if gpt4_nli_examples_filepaths is not None:
        for output_text in gpt4_nli_output_texts:
            labels.append(_NLI_LABEL2ID[output_text])
    if report_nli_input_output_jsonl_filepaths is not None:
        for output_text in report_nli_output_texts:
            labels.append(_NLI_LABEL2ID[output_text])
    labels = np.array(labels)

    # Calibrate thresholds and compute accuracy
    
    # 1) RadNLI dev
    offset = 0
    size = len(radnli_dev_output_texts)
    print_bold('--- RadNLI dev ---')
    radnli_dev_scores = max_sims[offset:offset+size]
    radnli_dev_labels = labels[offset:offset+size]
    ent_threshold, f1 = best_threshold_and_f1_score(radnli_dev_scores, radnli_dev_labels == 1)
    print(f"ent_threshold: {ent_threshold}, F1: {f1}")
    contr_threshold, f1 = best_threshold_and_f1_score(radnli_dev_scores, radnli_dev_labels != -1)
    print(f"contr_threshold: {contr_threshold}, F1: {f1}")
    assert ent_threshold > contr_threshold # higher similarity score should indicate entailment
    _compute_confusion_matrix(radnli_dev_scores, radnli_dev_labels, ent_threshold, contr_threshold)

    # 2) RadNLI test
    offset += size
    size = len(radnli_test_output_texts)
    print_bold('--- RadNLI test ---')
    radnli_test_scores = max_sims[offset:offset+size]
    radnli_test_labels = labels[offset:offset+size]
    ent_threshold, f1 = best_threshold_and_f1_score(radnli_test_scores, radnli_test_labels == 1)
    print(f"ent_threshold: {ent_threshold}, F1: {f1}")
    contr_threshold, f1 = best_threshold_and_f1_score(radnli_test_scores, radnli_test_labels != -1)
    print(f"contr_threshold: {contr_threshold}, F1: {f1}")
    assert ent_threshold >= contr_threshold # higher similarity score should indicate entailment
    _compute_confusion_matrix(radnli_test_scores, radnli_test_labels, ent_threshold, contr_threshold)

    # 3) MS-CXR-T
    offset += size
    size = len(mscxrt_output_texts)
    print_bold('--- MS-CXR-T ---')
    mscxrt_scores = max_sims[offset:offset+size]
    mscxrt_labels = labels[offset:offset+size]
    ent_threshold, f1 = best_threshold_and_f1_score(mscxrt_scores, mscxrt_labels == 1)
    print(f"ent_threshold: {ent_threshold}, F1: {f1}")
    contr_threshold, f1 = best_threshold_and_f1_score(mscxrt_scores, mscxrt_labels != -1)
    print(f"contr_threshold: {contr_threshold}, F1: {f1}")
    assert ent_threshold >= contr_threshold # higher similarity score should indicate entailment
    _compute_confusion_matrix(mscxrt_scores, mscxrt_labels, ent_threshold, contr_threshold)

    # 4) GPT-4 NLI
    if gpt4_nli_examples_filepaths is not None:
        offset += size
        size = len(gpt4_nli_output_texts)
        print_bold('--- GPT-4 NLI ---')
        gpt4_nli_scores = max_sims[offset:offset+size]
        gpt4_nli_labels = labels[offset:offset+size]
        ent_threshold, f1 = best_threshold_and_f1_score(gpt4_nli_scores, gpt4_nli_labels == 1)
        print(f"ent_threshold: {ent_threshold}, F1: {f1}")
        contr_threshold, f1 = best_threshold_and_f1_score(gpt4_nli_scores, gpt4_nli_labels != -1)
        print(f"contr_threshold: {contr_threshold}, F1: {f1}")
        assert ent_threshold >= contr_threshold # higher similarity score should indicate entailment
        _compute_confusion_matrix(gpt4_nli_scores, gpt4_nli_labels, ent_threshold, contr_threshold)

    # 5) Report NLI
    if report_nli_input_output_jsonl_filepaths is not None:
        offset += size
        size = len(report_nli_output_texts)
        print_bold('--- Report NLI ---')
        report_nli_scores = max_sims[offset:offset+size]
        report_nli_labels = labels[offset:offset+size]
        report_nli_hypotheses = hypotheses[offset:offset+size]
        ent_thresholds = np.empty(len(report_nli_hypotheses))
        contr_thresholds = np.empty(len(report_nli_hypotheses))

        fact2idxs = { fact: [] for fact in LABEL_BASED_FACTS }
        fact2idxs['#Other'] = []
        for i in range(size):
            fact = report_nli_hypotheses[i]
            try:
                fact2idxs[fact].append(i)
            except:
                fact2idxs['#Other'].append(i)
        
        if thresholds_save_path is not None:
            thresholds_dict = load_pickle(thresholds_save_path)
            print(f"Thresholds loaded from {thresholds_save_path}")
            for fact, idxs in fact2idxs.items():
                if len(idxs) == 0:
                    continue
                fact_scores = report_nli_scores[idxs]
                fact_labels = report_nli_labels[idxs]
                ent_threshold = thresholds_dict[fact]['ent_threshold']
                contr_threshold = thresholds_dict[fact]['contr_threshold']
                assert ent_threshold >= contr_threshold
                ent_thresholds[idxs] = ent_threshold
                contr_thresholds[idxs] = contr_threshold
        else:
            if save_thresholds:
                thresholds_dict = {}
            for fact, idxs in fact2idxs.items():
                if len(idxs) == 0:
                    continue
                fact_scores = report_nli_scores[idxs]
                fact_labels = report_nli_labels[idxs]
                ent_threshold, _ = best_threshold_and_f1_score(fact_scores, fact_labels == 1, verbose=False)
                contr_threshold, _ = best_threshold_and_f1_score(fact_scores, fact_labels != -1, verbose=False)
                if save_thresholds:
                    thresholds_dict[fact] = {'ent_threshold': ent_threshold, 'contr_threshold': contr_threshold}
                assert ent_threshold >= contr_threshold
                ent_thresholds[idxs] = ent_threshold
                contr_thresholds[idxs] = contr_threshold
            if save_thresholds:
                if thresholds_save_path is None:
                    thresholds_save_path = get_file_path_with_hashing_if_too_long(
                        folder_path=FAST_CACHE_DIR,
                        prefix='report_nli_thresholds',
                        strings=[
                            model_name,
                            huggingface_model_name,
                            model_checkpoint_folder_path,
                            *report_nli_input_output_jsonl_filepaths,
                        ],
                        force_hashing=True,
                    )
                save_pickle(thresholds_dict, thresholds_save_path)
                print(f"Thresholds saved to {thresholds_save_path}")
        
        _compute_confusion_matrix(report_nli_scores, report_nli_labels, ent_thresholds, contr_thresholds)

        # Plot F1 scores for each fact
        print_bold('--- Report NLI examples (by fact) ---')
        fact2stats = { fact: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for fact in LABEL_BASED_FACTS }
        other2stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        for i in range(size):
            fact = report_nli_hypotheses[i]
            gt_label = report_nli_labels[i] == 1 # binary
            pred_label = report_nli_scores[i] >= ent_thresholds[i]
            try:
                stats = fact2stats[fact]
            except:
                stats = other2stats
            if gt_label == pred_label:
                if gt_label:
                    stats['tp'] += 1
                else:
                    stats['tn'] += 1
            else:
                if gt_label:
                    stats['fn'] += 1
                else:
                    stats['fp'] += 1

        metric_names = [f'{fact} (tp: {stats["tp"]}, fp: {stats["fp"]}, tn: {stats["tn"]}, fn: {stats["fn"]})' for fact, stats in fact2stats.items()]
        metric_names.append(f'Other (tp: {other2stats["tp"]}, fp: {other2stats["fp"]}, tn: {other2stats["tn"]}, fn: {other2stats["fn"]})')
        
        f1s = [2 * stats["tp"] / max(stats["tp"] + stats["fp"] + stats["tp"] + stats["fn"], 1) for _, stats in fact2stats.items()]
        f1s.append(2 * other2stats["tp"] / max(other2stats["tp"] + other2stats["fp"] + other2stats["tp"] + other2stats["fn"], 1))
        plot_metrics(metric_names=metric_names, metric_values=f1s, title="F1",
                ylabel="Label", xlabel="F1", append_average_to_title=True, horizontal=True, sort_metrics=True,
                show_metrics_above_bars=True, draw_grid=True, figsize=f1_figsize)
        
        recalls = [stats["tp"] / max(stats["tp"] + stats["fn"], 1) for _, stats in fact2stats.items()]
        recalls.append(other2stats["tp"] / max(other2stats["tp"] + other2stats["fn"], 1))
        plot_metrics(metric_names=metric_names, metric_values=recalls, title="Recall",
                ylabel="Label", xlabel="Recall", append_average_to_title=True, horizontal=True, sort_metrics=True,
                show_metrics_above_bars=True, draw_grid=True, figsize=f1_figsize)
        
        accs = [(stats["tp"] + stats["tn"]) / max(stats["tp"] + stats["fp"] + stats["tn"] + stats["fn"], 1) for _, stats in fact2stats.items()]
        accs.append((other2stats["tp"] + other2stats["tn"]) / max(other2stats["tp"] + other2stats["fp"] + other2stats["tn"] + other2stats["fn"], 1))
        plot_metrics(metric_names=metric_names, metric_values=accs, title="Accuracy",
                ylabel="Label", xlabel="Accuracy", append_average_to_title=True, horizontal=True, sort_metrics=True,
                show_metrics_above_bars=True, draw_grid=True, figsize=f1_figsize)

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)