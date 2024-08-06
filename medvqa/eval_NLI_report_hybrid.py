import argparse
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from medvqa.datasets.dataloading_utils import embedding_based_nli_collate_batch_fn
from medvqa.datasets.nli.nli_dataset_management import EmbeddingNLIDatasetWrapper
from medvqa.datasets.seq2seq.seq2seq_dataset_management import (
    Seq2SeqDataset,
    load_ms_cxr_t_temporal_sentence_similarity_v1_data,
    load_radnli_test_data,
    load_report_nli_examples_filepaths,
)
from medvqa.datasets.text_data_utils import sentence_tokenize_texts_in_parallel
from medvqa.evaluation.plots import plot_metrics
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata
from medvqa.models.nlp.nli import EmbeddingBasedNLI
from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.scripts.mimiccxr.generate_fact_based_report_nli_examples_with_openai import LABEL_BASED_FACTS
from medvqa.utils.common import FAST_CACHE_DIR, parsed_args_to_dict
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.files import get_file_path_with_hashing_if_too_long, save_pickle, load_pickle
from medvqa.utils.logging import get_console_logger, print_blue, print_bold
from medvqa.utils.metrics import best_threshold_and_f1_score

logger = None

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu, cuda, gpu)')
    parser.add_argument('--fact_embedding_model_name', type=str, required=True,
                        help='Huggingface model name for fact embedding')
    parser.add_argument('--fact_embedding_model_checkpoint_folder_path', type=str, required=True,
                        help='Folder path containing the model checkpoint for fact embedding')
    parser.add_argument('--fact_embedding_batch_size', type=int, default=32, help='Batch size for fact embedding')
    parser.add_argument('--fact_embedding_num_workers', type=int, default=4, help='Number of workers for fact embedding')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--nli_checkpoint_folder_path', type=str, required=True, help='Folder path containing the model checkpoint for NLI')
    parser.add_argument('--gpt4_nli_examples_filepaths', type=str, nargs='+', default=None, help='Filepaths for GPT-4 NLI examples')
    parser.add_argument('--report_nli_input_output_jsonl_filepaths', type=str, nargs='+', default=None, help='Filepaths for Report NLI input-output jsonl files')
    parser.add_argument('--f1_figsize', type=int, nargs=2, default=(10, 35), help='F1 plot figsize')
    return parser.parse_args(args=args)

def _compute_confusion_matrix(pred_labels, gt_labels, include_undecided=False, sns_font_scale=1.8, font_size=25):
    # Compute accuracy
    acc = np.sum(pred_labels == gt_labels) / len(gt_labels)
    print_blue(f"Accuracy: {acc:.4f}", bold=True)
    # Compute accuracy ignoring undecided
    if include_undecided:
        non_undecided = pred_labels != 3
        acc = np.sum(pred_labels[non_undecided] == gt_labels[non_undecided]) / np.sum(non_undecided)
        print_blue(f"Accuracy (ignoring undecided): {acc:.4f}", bold=True)
        # Print percentage of undecided
        print_blue(f"Percentage of undecided: {np.sum(pred_labels == 3) / len(gt_labels):.4f}", bold=True)
    # Plot a confusion matrix
    if include_undecided:
        cm = confusion_matrix(gt_labels, pred_labels, labels=[0, 1, 2, 3])
    else:
        cm = confusion_matrix(gt_labels, pred_labels, labels=[0, 1, 2])
    # Use label names instead of indices
    if include_undecided:
        label_names = ['entailment', 'neutral', 'contradiction', 'undecided']
    else:
        label_names = ['entailment', 'neutral', 'contradiction']
    plt.figure(figsize=(8, 6))
    sns.set_theme(font_scale=sns_font_scale)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted', fontsize=font_size)
    plt.ylabel('True', fontsize=font_size)
    plt.show()

def compute_mlp_nli_softmaxes(
    device,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    mlp_batch_size,
    mlp_num_workers,
    unique_sentences,
    sentence2idx,
    whole_sentences,
    splittable_sentences,
    input_texts,
    output_texts,
    nli_checkpoint_folder_path,
    cache_output=False,
):
    if cache_output:
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=FAST_CACHE_DIR,
            prefix='report_nli_mlp_softmaxes',
            strings=[
                fact_embedding_model_name,
                fact_embedding_model_checkpoint_folder_path,
                nli_checkpoint_folder_path,
                f'len(input_texts)={len(input_texts)}',
                f'len(unique_sentences)={len(unique_sentences)}',
                f'len(whole_sentences)={len(whole_sentences)}',
            ],
            force_hashing=True,
        )
        if os.path.exists(save_path):
            print(f"Loading cached MLP NLI softmaxes from {save_path}")
            return load_pickle(save_path)

    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=device,
    )
    embeddings = embedding_extractor.compute_text_embeddings(unique_sentences)
    print(f'embeddings.shape: {embeddings.shape}')

    s2s_dataset = Seq2SeqDataset(
        indices=list(range(len(input_texts))),
        input_texts=input_texts,
        output_texts=output_texts,
    )
    nli_dataset = EmbeddingNLIDatasetWrapper(
        seq2seq_dataset=s2s_dataset,
        embeddings=embeddings,
        sentence2idx=sentence2idx,
        whole_sentences=whole_sentences,
        splittable_sentences=splittable_sentences,
    )
    dataloader = DataLoader(
        nli_dataset,
        batch_size=mlp_batch_size,
        shuffle=False,
        num_workers=mlp_num_workers,
        collate_fn=embedding_based_nli_collate_batch_fn,
        pin_memory=True,
    )

    # Load model metadata
    metadata = load_metadata(nli_checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and device in ['cuda', 'gpu', 'GPU'] else 'cpu')
    
    # Create model
    print("Creating model")
    model = EmbeddingBasedNLI(**model_kwargs)
    model = model.to(device)

    # Load model weights
    print(f"Loading model weights from {nli_checkpoint_folder_path}")
    checkpoint_path = get_checkpoint_filepath(nli_checkpoint_folder_path)
    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Set model to evaluation mode
    model.eval()

    # Evaluate
    all_softmaxes = np.empty((len(input_texts), 3))
    with torch.no_grad():
        i = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            h_embs = batch['h_embs'].to(device)
            p_most_sim_embs = batch['p_most_sim_embs'].to(device)
            p_least_sim_embs = batch['p_least_sim_embs'].to(device)
            p_max_embs = batch['p_max_embs'].to(device)
            p_avg_embs = batch['p_avg_embs'].to(device)
            logits = model(h_embs, p_most_sim_embs, p_least_sim_embs, p_max_embs, p_avg_embs)
            softmaxes = torch.softmax(logits, dim=1)
            bsize = len(softmaxes)
            all_softmaxes[i:i+bsize] = softmaxes.cpu().numpy()
            i += bsize
        assert i == len(input_texts)

    # Save output
    if cache_output:
        save_pickle(all_softmaxes, save_path)
        print(f"Saved MLP NLI softmaxes to {save_path}")

    # Return
    return all_softmaxes

def compute_BART_nli_predictions(
    seq2seq_checkpoint_folder_path,
    input_texts,
    logger,
    device,
    batch_size,
    num_workers,
    max_length,
    num_beams,
    cache_output=False,
):
    if cache_output:
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=FAST_CACHE_DIR,
            prefix='report_nli_BART_predictions',
            strings=[
                seq2seq_checkpoint_folder_path,
                f'len(input_texts)={len(input_texts)}',
            ],
            force_hashing=True,
        )
        if os.path.exists(save_path):
            print(f"Loading cached BART NLI predictions from {save_path}")
            return load_pickle(save_path)
    
    # Compute Seq2Seq NLI predictions
    seq2seq_nli_predictions, unprocessed_sentences = apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=seq2seq_checkpoint_folder_path,
        sentences=input_texts,
        logger=logger,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        num_beams=num_beams,
        postprocess_input_output_func=lambda _, output: output,
        save_outputs=False,
    )
    assert len(seq2seq_nli_predictions) == len(input_texts)
    assert len(unprocessed_sentences) == 0

    # Save output
    if cache_output:
        save_pickle(seq2seq_nli_predictions, save_path)
        print(f"Saved BART NLI predictions to {save_path}")

    return seq2seq_nli_predictions

def evaluate_mlp_nli(
    device,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    mlp_batch_size,
    mlp_num_workers,
    nli_checkpoint_folder_path,
    report_nli_input_output_jsonl_filepaths,
    logging_level='INFO',
    cache_output=False,
):
    global logger
    if logger is None:
        logger = get_console_logger(logging_level)
    
    whole_sentences = set()
    splittable_sentences = set()
    input_texts, output_texts, _ = load_report_nli_examples_filepaths(
        report_nli_input_output_jsonl_filepaths, nli1_only=True, whole_sentences=whole_sentences,
        splittable_sentences=splittable_sentences)
    
    radnli_test_input_texts, radnli_test_output_texts = load_radnli_test_data(nli1_only=True, whole_sentences=whole_sentences)
    input_texts.extend(radnli_test_input_texts)
    output_texts.extend(radnli_test_output_texts)

    mscxrt_input_texts, mscxrt_output_texts = load_ms_cxr_t_temporal_sentence_similarity_v1_data(
        nli1_only=True, whole_sentences=whole_sentences)
    input_texts.extend(mscxrt_input_texts)
    output_texts.extend(mscxrt_output_texts)
    
    print(f"Total input_texts: {len(input_texts)}")
    print(f"Total output_texts: {len(output_texts)}")
    print(f"Total whole_sentences: {len(whole_sentences)}")
    print(f"Total splittable_sentences: {len(splittable_sentences)}")

    unique_sentences = set()
    unique_sentences.update(whole_sentences)
    tokenized_sentences = sentence_tokenize_texts_in_parallel(splittable_sentences)
    for sentences in tokenized_sentences:
        unique_sentences.update(sentences)
    unique_sentences = list(unique_sentences)
    sentence2idx = {s: i for i, s in enumerate(unique_sentences)}
    
    print(f"Total unique_sentences: {len(unique_sentences)}")

    # Compute MLP NLI softmaxes
    mlp_nli_softmaxes = compute_mlp_nli_softmaxes(
        device=device,
        fact_embedding_model_name=fact_embedding_model_name,
        fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        fact_embedding_batch_size=fact_embedding_batch_size,
        fact_embedding_num_workers=fact_embedding_num_workers,
        mlp_batch_size=mlp_batch_size,
        mlp_num_workers=mlp_num_workers,
        unique_sentences=unique_sentences,
        sentence2idx=sentence2idx,
        whole_sentences=whole_sentences,
        splittable_sentences=splittable_sentences,
        input_texts=input_texts,
        output_texts=output_texts,
        nli_checkpoint_folder_path=nli_checkpoint_folder_path,
        cache_output=cache_output,
    )
    
    output2label = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    report_nli_gt_labels = np.array([output2label[output_text] for output_text in output_texts])

    print("Adjusting thresholds for MLP NLI")
    # Entailment
    best_et, best_et_f1 = best_threshold_and_f1_score(
        gt=report_nli_gt_labels == 0, # binarize (0 -> 1, 1 -> 0, 2 -> 0)
        probs=mlp_nli_softmaxes[:, 0],
    )
    # Neutral
    best_nt, best_nt_f1 = best_threshold_and_f1_score(
        gt=report_nli_gt_labels == 1, # binarize (0 -> 0, 1 -> 1, 2 -> 0)
        probs=mlp_nli_softmaxes[:, 1],
    )
    # Contradiction
    best_ct, best_ct_f1 = best_threshold_and_f1_score(
        gt=report_nli_gt_labels == 2, # binarize (0 -> 0, 1 -> 0, 2 -> 1)
        probs=mlp_nli_softmaxes[:, 2],
    )
    print_blue(f"Best entailment threshold: {best_et}, Best F1: {best_et_f1}", bold=True)
    print_blue(f"Best neutral threshold: {best_nt}, Best F1: {best_nt_f1}", bold=True)
    print_blue(f"Best contradiction threshold: {best_ct}, Best F1: {best_ct_f1}", bold=True)


def evaluate(
    device,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    mlp_batch_size,
    mlp_num_workers,
    nli_checkpoint_folder_path,
    seq2seq_checkpoint_folder_path,
    seq2seq_batch_size,
    seq2seq_num_workers,
    seq2seq_max_length,
    seq2seq_num_beams,
    report_nli_input_output_jsonl_filepaths,
    save_hybrid_metadata=False,
    cache_output=False,
    logging_level='INFO',
    hybrid_threshold_1=0.2,
    hybrid_threshold_2=0.4,
    f1_figsize=(10, 35),
    only_show_plots_for_hybrid=False,
    show_confusion_matrix=True,
    show_bar_plots=True,
):
    global logger
    if logger is None:
        logger = get_console_logger(logging_level)
    
    whole_sentences = set()
    splittable_sentences = set()
    input_texts, output_texts, _ = load_report_nli_examples_filepaths(
        report_nli_input_output_jsonl_filepaths, nli1_only=True, whole_sentences=whole_sentences,
        splittable_sentences=splittable_sentences)
    
    print(f"Total input_texts: {len(input_texts)}")
    print(f"Total output_texts: {len(output_texts)}")
    print(f"Total whole_sentences: {len(whole_sentences)}")
    print(f"Total splittable_sentences: {len(splittable_sentences)}")

    unique_sentences = set()
    unique_sentences.update(whole_sentences)
    tokenized_sentences = sentence_tokenize_texts_in_parallel(splittable_sentences)
    for sentences in tokenized_sentences:
        unique_sentences.update(sentences)
    unique_sentences = list(unique_sentences)
    sentence2idx = {s: i for i, s in enumerate(unique_sentences)}
    
    print(f"Total unique_sentences: {len(unique_sentences)}")

    # Compute MLP NLI softmaxes
    mlp_nli_softmaxes = compute_mlp_nli_softmaxes(
        device=device,
        fact_embedding_model_name=fact_embedding_model_name,
        fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        fact_embedding_batch_size=fact_embedding_batch_size,
        fact_embedding_num_workers=fact_embedding_num_workers,
        mlp_batch_size=mlp_batch_size,
        mlp_num_workers=mlp_num_workers,
        unique_sentences=unique_sentences,
        sentence2idx=sentence2idx,
        whole_sentences=whole_sentences,
        splittable_sentences=splittable_sentences,
        input_texts=input_texts,
        output_texts=output_texts,
        nli_checkpoint_folder_path=nli_checkpoint_folder_path,
        cache_output=cache_output,
    )

    # Compute Seq2Seq NLI predictions
    seq2seq_nli_predictions = compute_BART_nli_predictions(
        seq2seq_checkpoint_folder_path=seq2seq_checkpoint_folder_path,
        input_texts=input_texts,
        logger=logger,
        device=device,
        batch_size=seq2seq_batch_size,
        num_workers=seq2seq_num_workers,
        max_length=seq2seq_max_length,
        num_beams=seq2seq_num_beams,
        cache_output=cache_output,
    )
    
    output2label = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    report_nli_gt_labels = np.array([output2label[output_text] for output_text in output_texts])
    report_nli_hypotheses = [input_text.split(' #H: ')[1] for input_text in input_texts]

    h2idxs = {h:[] for h in LABEL_BASED_FACTS}
    h2idxs['other'] = []
    for i, h in enumerate(report_nli_hypotheses):
        if h in h2idxs:
            h2idxs[h].append(i)
        else:
            h2idxs['other'].append(i)

    print("Adjusting entailment threshold for MLP NLI")
    mlp_h2et = {} # entailment threshold
    mlp_h2ct = {} # contradiction threshold
    for h, idxs in h2idxs.items():
        assert len(idxs) > 0, f"len(idxs) == 0 for {h}"
        # Entailment
        best_et, best_f1 = best_threshold_and_f1_score(
            gt=report_nli_gt_labels[idxs] == 0, # binarize (0 -> 1, 1 -> 0, 2 -> 0)
            probs=mlp_nli_softmaxes[idxs, 0],
        )
        mlp_h2et[h] = best_et
        # print(f"{h}: Best ent. threshold: {best_et}, Best F1: {best_f1} (size: {len(idxs)})")
        # Contradiction
        best_ct, best_f1 = best_threshold_and_f1_score(
            gt=report_nli_gt_labels[idxs] == 2, # binarize (0 -> 0, 1 -> 0, 2 -> 1)
            probs=mlp_nli_softmaxes[idxs, 2],
        )
        mlp_h2ct[h] = best_ct
        # print(f"{h}: Best cont. threshold: {best_ct}, Best F1: {best_f1} (size: {len(idxs)})")

    # Plot metrics for each fact
    mlp_fact2stats = { fact: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for fact in LABEL_BASED_FACTS }
    mlp_fact2stats['other'] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    s2s_fact2stats = { fact: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for fact in LABEL_BASED_FACTS }
    s2s_fact2stats['other'] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    hyb_fact2stats = { fact: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for fact in LABEL_BASED_FACTS }
    hyb_fact2stats['other'] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    hyb_fact2stats_with_undecided = { fact: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'u': 0} for fact in LABEL_BASED_FACTS }
    hyb_fact2stats_with_undecided['other'] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'u': 0}

    mlp_pred_labels = np.empty(len(report_nli_hypotheses), dtype=int)
    s2s_pred_labels = np.empty(len(report_nli_hypotheses), dtype=int)
    hybrid_pred_labels = np.empty(len(report_nli_hypotheses), dtype=int)

    # Compute predictions

    for i in range(len(report_nli_hypotheses)):
        fact = report_nli_hypotheses[i]
        gt_label = report_nli_gt_labels[i] == 0 # binarize (0: entailment, 1: neutral, 2: contradiction)

        # MLP NLI
        et = mlp_h2et[fact] if fact in mlp_h2et else mlp_h2et['other']
        ct = mlp_h2ct[fact] if fact in mlp_h2ct else mlp_h2ct['other']
        if mlp_nli_softmaxes[i, 0] > et: mlp_pred_labels[i] = 0
        elif mlp_nli_softmaxes[i, 2] > ct: mlp_pred_labels[i] = 2
        else: mlp_pred_labels[i] = 1
        pred_label = mlp_pred_labels[i] == 0 # binarize
        try:
            stats = mlp_fact2stats[fact]
        except:
            stats = mlp_fact2stats['other']
        if gt_label == pred_label:
            if gt_label: stats['tp'] += 1
            else: stats['tn'] += 1
        else:
            if gt_label: stats['fn'] += 1
            else: stats['fp'] += 1

        # Seq2Seq NLI
        s2s_pred_labels[i] = output2label[seq2seq_nli_predictions[i]]
        pred_label = s2s_pred_labels[i] == 0 # binarize
        try:
            stats = s2s_fact2stats[fact]
        except:
            stats = s2s_fact2stats['other']
        if gt_label == pred_label:
            if gt_label: stats['tp'] += 1
            else: stats['tn'] += 1
        else:
            if gt_label: stats['fn'] += 1
            else: stats['fp'] += 1

    # Compute hybrid predictions
    calc_f1 = lambda h, m: 2 * m[h]['tp'] / max(2 * m[h]['tp'] + m[h]['fp'] + m[h]['fn'], 1)
    hybrid_metadata = {
        'hybrid_threshold_1': hybrid_threshold_1,
        'hybrid_threshold_2': hybrid_threshold_2,
    }
    for h, idxs in h2idxs.items():
        mlp_f1 = calc_f1(h, mlp_fact2stats)
        s2s_f1 = calc_f1(h, s2s_fact2stats)
        for i in idxs:
            if s2s_pred_labels[i] == mlp_pred_labels[i]: # if both agree
                hybrid_pred_labels[i] = s2s_pred_labels[i] # use agreed label
            elif abs(s2s_pred_labels[i] - mlp_pred_labels[i]) == 1:
                # if one is neutral and the other is not, use the label predicted by the model with higher F1
                if s2s_f1 > mlp_f1 + hybrid_threshold_1:
                    hybrid_pred_labels[i] = s2s_pred_labels[i]
                elif mlp_f1 > s2s_f1 + hybrid_threshold_1:
                    hybrid_pred_labels[i] = mlp_pred_labels[i]
                else:
                    hybrid_pred_labels[i] = 3 # undecided
            else:
                assert abs(s2s_pred_labels[i] - mlp_pred_labels[i]) == 2 # one is entailment and the other is contradiction
                # The models in large disagreement, use the label predicted by the model with higher F1
                # only if the F1 difference is greater than a threshold
                if abs(s2s_f1 - mlp_f1) > hybrid_threshold_2:
                    if s2s_f1 > mlp_f1:
                        hybrid_pred_labels[i] = s2s_pred_labels[i]
                    else:
                        hybrid_pred_labels[i] = mlp_pred_labels[i]
                else:
                    hybrid_pred_labels[i] = 3 # undecided
        stats = hyb_fact2stats[h]
        stats_u = hyb_fact2stats_with_undecided[h]
        for i in idxs:
            gt_label = report_nli_gt_labels[i] == 0
            pred_label = hybrid_pred_labels[i] == 0
            if gt_label == pred_label:
                if gt_label: stats['tp'] += 1
                else: stats['tn'] += 1
            else:
                if gt_label: stats['fn'] += 1
                else: stats['fp'] += 1
            if hybrid_pred_labels[i] == 3:
                stats_u['u'] += 1
            else:
                if gt_label == pred_label:
                    if gt_label: stats_u['tp'] += 1
                    else: stats_u['tn'] += 1
                else:
                    if gt_label: stats_u['fn'] += 1
                    else: stats_u['fp'] += 1
        hybrid_metadata[h] = {
            'mlp_et': mlp_h2et[h] if h in mlp_h2et else mlp_h2et['other'],
            'mlp_ct': mlp_h2ct[h] if h in mlp_h2ct else mlp_h2ct['other'],
            'mlp_f1': mlp_f1,
            's2s_f1': s2s_f1,
        }

    if show_confusion_matrix:
        for name, labels in zip(('MLP', 'Seq2Seq', 'Hybrid'),
                                (mlp_pred_labels, s2s_pred_labels, hybrid_pred_labels)):
            if only_show_plots_for_hybrid and name != 'Hybrid':
                continue
            print_bold(f'--- {name} NLI ---')
            if name == 'Hybrid':
                _compute_confusion_matrix(labels, report_nli_gt_labels, include_undecided=True)
            else:
                _compute_confusion_matrix(labels, report_nli_gt_labels)

    if show_bar_plots:
        for name, stats in zip(('MLP', 'Seq2Seq', 'Hybrid'),
                            (mlp_fact2stats, s2s_fact2stats, hyb_fact2stats)):
            if only_show_plots_for_hybrid and name != 'Hybrid':
                continue
            print_bold(f'--- {name} NLI (by fact) ---')
            metric_names = [f'{fact} (tp: {stats[fact]["tp"]}, fp: {stats[fact]["fp"]}, tn: {stats[fact]["tn"]}, fn: {stats[fact]["fn"]})' for fact in LABEL_BASED_FACTS]
            metric_names.append(f'Other (tp: {stats["other"]["tp"]}, fp: {stats["other"]["fp"]}, tn: {stats["other"]["tn"]}, fn: {stats["other"]["fn"]})')
            print(f'len(metric_names): {len(metric_names)}')
            
            f1s = [calc_f1(fact, stats) for fact in LABEL_BASED_FACTS]
            f1s.append(calc_f1('other', stats))
            accs = [(stats[fact]['tp'] + stats[fact]['tn']) / (stats[fact]['tp'] + stats[fact]['tn'] + stats[fact]['fp'] + stats[fact]['fn']) for fact in LABEL_BASED_FACTS]
            accs.append((stats['other']['tp'] + stats['other']['tn']) / (stats['other']['tp'] + stats['other']['tn'] + stats['other']['fp'] + stats['other']['fn']))            
            
            plot_metrics(metric_names=metric_names, metric_values=f1s, title="F1",
                    ylabel="Label", xlabel="F1", append_average_to_title=True, horizontal=True, sort_metrics=True,
                    show_metrics_above_bars=True, draw_grid=True, figsize=f1_figsize)
            
            plot_metrics(metric_names=metric_names, metric_values=accs, title="Accuracy",
                    ylabel="Label", xlabel="Accuracy", append_average_to_title=True, horizontal=True, sort_metrics=True,
                    show_metrics_above_bars=True, draw_grid=True, figsize=f1_figsize)
            
            if name == 'Hybrid':
                print_bold(f'--- {name} NLI (by fact, including undecided) ---')
                stats = hyb_fact2stats_with_undecided
                metric_names = [f'{fact} (tp: {stats[fact]["tp"]}, fp: {stats[fact]["fp"]}, tn: {stats[fact]["tn"]}, fn: {stats[fact]["fn"]}, u: {stats[fact]["u"]})' for fact in LABEL_BASED_FACTS]
                metric_names.append(f'Other (tp: {stats["other"]["tp"]}, fp: {stats["other"]["fp"]}, tn: {stats["other"]["tn"]}, fn: {stats["other"]["fn"]}, u: {stats["other"]["u"]})')
                print(f'len(metric_names): {len(metric_names)}')
                
                f1s = [calc_f1(fact, stats) for fact in LABEL_BASED_FACTS]
                f1s.append(calc_f1('other', stats))
                accs = [(stats[fact]['tp'] + stats[fact]['tn']) / (stats[fact]['tp'] + stats[fact]['tn'] + stats[fact]['fp'] + stats[fact]['fn']) for fact in LABEL_BASED_FACTS]
                accs.append((stats['other']['tp'] + stats['other']['tn']) / (stats['other']['tp'] + stats['other']['tn'] + stats['other']['fp'] + stats['other']['fn']))

                # sort based on fraction of undecided
                frac_undecided = [stats[fact]['u'] / (stats[fact]['tp'] + stats[fact]['tn'] + stats[fact]['fp'] + stats[fact]['fn'] + stats[fact]['u']) for fact in LABEL_BASED_FACTS]
                frac_undecided.append(stats['other']['u'] / (stats['other']['tp'] + stats['other']['tn'] + stats['other']['fp'] + stats['other']['fn'] + stats['other']['u']))
                idxs = np.argsort(frac_undecided)
                metric_names = [metric_names[i] for i in idxs]
                f1s = [f1s[i] for i in idxs]
                accs = [accs[i] for i in idxs]
                
                plot_metrics(metric_names=metric_names, metric_values=f1s, title="F1",
                        ylabel="Label", xlabel="F1", append_average_to_title=True, horizontal=True,
                        show_metrics_above_bars=True, draw_grid=True, figsize=f1_figsize)
                
                plot_metrics(metric_names=metric_names, metric_values=accs, title="Accuracy",
                        ylabel="Label", xlabel="Accuracy", append_average_to_title=True, horizontal=True,
                        show_metrics_above_bars=True, draw_grid=True, figsize=f1_figsize)
        
    if save_hybrid_metadata:
        hybrid_metadata_save_path = get_file_path_with_hashing_if_too_long(
            folder_path=FAST_CACHE_DIR,
            prefix='report_nli_hybrid_metadata',
            strings=[
                fact_embedding_model_name,
                fact_embedding_model_checkpoint_folder_path,
                nli_checkpoint_folder_path,
                seq2seq_checkpoint_folder_path,
                *report_nli_input_output_jsonl_filepaths,
                f'len(input_texts)={len(input_texts)}',
                f'threshold_1={hybrid_threshold_1}',
                f'threshold_2={hybrid_threshold_2}',
            ],
            force_hashing=True,
        )
        save_pickle(hybrid_metadata, hybrid_metadata_save_path)
        print(f"Saved hybrid metadata to {hybrid_metadata_save_path}")

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)