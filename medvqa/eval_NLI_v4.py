import argparse
import numpy as np
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
    load_gpt4_nli_examples_filepaths,
    load_ms_cxr_t_temporal_sentence_similarity_v1_data,
    load_radnli_dev_data,
    load_radnli_test_data,
    load_report_nli_examples_filepaths,
)
from medvqa.datasets.text_data_utils import sentence_tokenize_texts_in_parallel
from medvqa.evaluation.plots import plot_metrics
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata
from medvqa.models.nlp.nli import EmbeddingBasedNLI
from medvqa.scripts.mimiccxr.generate_fact_based_report_nli_examples_with_openai import LABEL_BASED_FACTS
from medvqa.utils.common import parsed_args_to_dict
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.logging_utils import print_blue, print_bold
from medvqa.utils.metrics_utils import best_threshold_and_f1_score

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

def _compute_confusion_matrix(pred_labels, gt_labels, sns_font_scale=1.8, font_size=25):
    # Compute accuracy
    acc = np.sum(pred_labels == gt_labels) / len(gt_labels)
    print_blue(f"Accuracy: {acc:.4f}", bold=True)
    # Plot a confusion matrix
    cm = confusion_matrix(gt_labels, pred_labels, labels=[0, 1, 2])
    # Use label names instead of indices
    label_names = ['entailment', 'neutral', 'contradiction']
    plt.figure(figsize=(8, 6))
    sns.set_theme(font_scale=sns_font_scale)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted', fontsize=font_size)
    plt.ylabel('True', fontsize=font_size)
    plt.show()

def evaluate(
    device,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    batch_size,
    num_workers,
    nli_checkpoint_folder_path,
    gpt4_nli_examples_filepaths,
    report_nli_input_output_jsonl_filepaths,
    adjust_entailment_threshold=False,
    f1_figsize=(10, 35),
):
    
    whole_sentences = set()
    splittable_sentences = set()

    radnli_dev_input_texts, radnli_dev_output_texts = load_radnli_dev_data(nli1_only=True, whole_sentences=whole_sentences)
    radnli_test_input_texts, radnli_test_output_texts = load_radnli_test_data(nli1_only=True, whole_sentences=whole_sentences)
    mscxrt_input_texts, mscxrt_output_texts = load_ms_cxr_t_temporal_sentence_similarity_v1_data(
        nli1_only=True, whole_sentences=whole_sentences)
    input_texts = radnli_dev_input_texts + radnli_test_input_texts + mscxrt_input_texts
    output_texts = radnli_dev_output_texts + radnli_test_output_texts + mscxrt_output_texts
    if gpt4_nli_examples_filepaths is not None:
        gpt4_nli_input_texts, gpt4_nli_output_texts = load_gpt4_nli_examples_filepaths(
            gpt4_nli_examples_filepaths, nli1_only=True, whole_sentences=whole_sentences)                               
        input_texts += gpt4_nli_input_texts
        output_texts += gpt4_nli_output_texts
    if report_nli_input_output_jsonl_filepaths is not None:
        report_nli_input_texts, report_nli_output_texts, _ = load_report_nli_examples_filepaths(
            report_nli_input_output_jsonl_filepaths, nli1_only=True, whole_sentences=whole_sentences,
            splittable_sentences=splittable_sentences)
        input_texts += report_nli_input_texts
        output_texts += report_nli_output_texts
    
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
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
    with torch.no_grad():
        all_pred_labels = []
        all_softmaxes = []
        all_gt_labels = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            h_embs = batch['h_embs'].to(device)
            p_most_sim_embs = batch['p_most_sim_embs'].to(device)
            p_least_sim_embs = batch['p_least_sim_embs'].to(device)
            p_max_embs = batch['p_max_embs'].to(device)
            p_avg_embs = batch['p_avg_embs'].to(device)
            labels = batch['labels']
            logits = model(h_embs, p_most_sim_embs, p_least_sim_embs, p_max_embs, p_avg_embs)
            softmaxes = torch.softmax(logits, dim=1)
            pred_labels = torch.argmax(logits, dim=1)
            all_softmaxes.extend(softmaxes.cpu().numpy())
            all_pred_labels.extend(pred_labels.cpu().numpy())
            all_gt_labels.extend(labels)
        all_pred_labels = np.array(all_pred_labels)
        all_gt_labels = np.array(all_gt_labels)
        all_softmaxes = np.array(all_softmaxes)

    # Calibrate thresholds and compute accuracy
    
    # 1) RadNLI dev
    offset = 0
    size = len(radnli_dev_output_texts)
    print_bold('--- RadNLI dev ---')
    radnli_dev_pred_labels = all_pred_labels[offset:offset+size]
    radnli_dev_gt_labels = all_gt_labels[offset:offset+size]
    _compute_confusion_matrix(radnli_dev_pred_labels, radnli_dev_gt_labels)

    # 2) RadNLI test
    offset += size
    size = len(radnli_test_output_texts)
    print_bold('--- RadNLI test ---')
    radnli_test_pred_labels = all_pred_labels[offset:offset+size]
    radnli_test_gt_labels = all_gt_labels[offset:offset+size]
    _compute_confusion_matrix(radnli_test_pred_labels, radnli_test_gt_labels)

    # 3) MS-CXR-T
    offset += size
    size = len(mscxrt_output_texts)
    print_bold('--- MS-CXR-T ---')
    mscxrt_pred_labels = all_pred_labels[offset:offset+size]
    mscxrt_gt_labels = all_gt_labels[offset:offset+size]
    _compute_confusion_matrix(mscxrt_pred_labels, mscxrt_gt_labels)

    # 4) GPT-4 NLI
    if gpt4_nli_examples_filepaths is not None:
        offset += size
        size = len(gpt4_nli_output_texts)
        print_bold('--- GPT-4 NLI ---')
        gpt4_nli_pred_labels = all_pred_labels[offset:offset+size]
        gpt4_nli_gt_labels = all_gt_labels[offset:offset+size]
        _compute_confusion_matrix(gpt4_nli_pred_labels, gpt4_nli_gt_labels)

    # 5) Report NLI
    if report_nli_input_output_jsonl_filepaths is not None:
        offset += size
        size = len(report_nli_output_texts)
        print_bold('--- Report NLI ---')
        report_nli_pred_labels = all_pred_labels[offset:offset+size]
        report_nli_gt_labels = all_gt_labels[offset:offset+size]
        report_nli_hypotheses = [input_text.split(' #H: ')[1] for input_text in report_nli_input_texts]

        if adjust_entailment_threshold:
            print("Adjusting entailment threshold")
            report_nli_softmaxes = all_softmaxes[offset:offset+size]
            report_nli_softmaxes = np.array(report_nli_softmaxes)
            best_thrs, best_f1 = best_threshold_and_f1_score(
                gt=report_nli_gt_labels == 0, # binarize (0 -> 1, 1 -> 0, 2 -> 0)
                probs=report_nli_softmaxes[:, 0],
            )
            print(f"Best threshold: {best_thrs}, Best F1: {best_f1}")
        
        _compute_confusion_matrix(report_nli_pred_labels, report_nli_gt_labels)

        # Plot metrics for each fact
        print_bold('--- Report NLI examples (by fact) ---')
        fact2stats = { fact: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for fact in LABEL_BASED_FACTS }
        other2stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        for i in range(size):
            fact = report_nli_hypotheses[i]
            gt_label = report_nli_gt_labels[i] == 0 # binarize (0: entailment, 1: neutral, 2: contradiction)
            if adjust_entailment_threshold:
                pred_label = report_nli_softmaxes[i][0] > best_thrs
            else:
                pred_label = report_nli_pred_labels[i] == 0
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