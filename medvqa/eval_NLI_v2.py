import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from medvqa.datasets.nli import RADNLI_TEST_JSONL_PATH, MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH
from medvqa.metrics.medical.chexbert import CheXbertLabeler
from medvqa.utils.common import parsed_args_to_dict
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor, SupportedHuggingfaceMedicalBERTModels
from medvqa.utils.files import load_jsonl
from medvqa.utils.logging import print_blue, print_bold, print_magenta

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--huggingface_model_name', type=str, default=None, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--model_name', type=str, default=None, choices=['CheXbert'])
    parser.add_argument('--model_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--integrated_nli_jsonl_filepath', type=str, default=None)
    parser.add_argument('--similarity_metric', type=str, default='cosine', choices=['cosine', 'dot_product'])
    parser.add_argument('--average_token_embeddings', action='store_true')
    parser.add_argument('--calibrate_on_test_set', action='store_true')
    return parser.parse_args(args=args)

def calibrate_threshold(train_entailment_pairs, train_contradiction_pairs, embeddings, s2i):
    print('--------------')
    print_blue('Calibrating threshold')
    train_p_ent_idxs = [s2i[p] for p, _ in train_entailment_pairs]
    train_h_ent_idxs = [s2i[h] for _, h in train_entailment_pairs]
    train_p_con_idxs = [s2i[p] for p, _ in train_contradiction_pairs]
    train_h_con_idxs = [s2i[h] for _, h in train_contradiction_pairs]
    train_p_ent_embeddings = embeddings[train_p_ent_idxs]
    train_h_ent_embeddings = embeddings[train_h_ent_idxs]
    train_p_con_embeddings = embeddings[train_p_con_idxs]
    train_h_con_embeddings = embeddings[train_h_con_idxs]
    print(f"train_p_ent_embeddings.shape: {train_p_ent_embeddings.shape}")
    print(f"train_h_ent_embeddings.shape: {train_h_ent_embeddings.shape}")
    print(f"train_p_con_embeddings.shape: {train_p_con_embeddings.shape}")
    print(f"train_h_con_embeddings.shape: {train_h_con_embeddings.shape}")
    train_ent_cosine_sim = np.sum(train_p_ent_embeddings * train_h_ent_embeddings, axis=1)
    train_con_cosine_sim = np.sum(train_p_con_embeddings * train_h_con_embeddings, axis=1)
    print(f"train_ent_cosine_sim.shape: {train_ent_cosine_sim.shape}")
    print(f"train_con_cosine_sim.shape: {train_con_cosine_sim.shape}")
    min_sim = min(np.min(train_ent_cosine_sim), np.min(train_con_cosine_sim))
    max_sim = max(np.max(train_ent_cosine_sim), np.max(train_con_cosine_sim))
    best_threshold = None
    best_acc = 0
    for threshold in tqdm(np.linspace(min_sim, max_sim, 1000)):
        entailment_preds = train_ent_cosine_sim >= threshold
        contradiction_preds = train_con_cosine_sim < threshold
        ent_acc = np.sum(entailment_preds) / len(train_ent_cosine_sim)
        con_acc = np.sum(contradiction_preds) / len(train_con_cosine_sim)
        acc = (ent_acc + con_acc) / 2
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            best_ent_acc = ent_acc
            best_con_acc = con_acc
    print(f"best_threshold: {best_threshold}")
    print(f"best_acc: {best_acc}")
    print(f"best_ent_acc: {best_ent_acc}")
    print(f"best_con_acc: {best_con_acc}")
    return best_threshold

def evaluate(
    model_name,
    huggingface_model_name,
    device,
    model_checkpoint_folder_path,
    batch_size,
    num_workers,
    integrated_nli_jsonl_filepath,
    similarity_metric,
    average_token_embeddings,
    calibrate_on_test_set=False,
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
    
    sentences = set()

    # Load train data
    if not calibrate_on_test_set:
        assert integrated_nli_jsonl_filepath is not None
        train_nli_rows = load_jsonl(integrated_nli_jsonl_filepath)
        train_entailment_pairs = []
        train_contradiction_pairs = []
        for row in train_nli_rows:
            label = row['label']
            if label == 'neutral':
                continue # skip neutral
            premise = row['premise']
            hypothesis = row['hypothesis']
            sentences.add(premise)
            sentences.add(hypothesis)
            if label == 'entailment':
                train_entailment_pairs.append((premise, hypothesis))
            elif label == 'contradiction':
                train_contradiction_pairs.append((premise, hypothesis))
            else: 
                raise ValueError(f"Unknown label: {label}")
        print(f"Train entailment pairs: {len(train_entailment_pairs)}")
        print(f"Train contradiction pairs: {len(train_contradiction_pairs)}")
        print_bold('Example train entailment pairs:')
        for pair in random.sample(train_entailment_pairs, 4):
            print(pair)
        print_bold('Example train contradiction pairs:')
        for pair in random.sample(train_contradiction_pairs, 4):
            print(pair)
    
    # Load test data
    
    # 1) RadNLI test set
    radnli_test_rows = load_jsonl(RADNLI_TEST_JSONL_PATH)
    test_radnli_entailment_pairs = []
    test_radnli_contradiction_pairs = []
    for row in radnli_test_rows:
        label = row['gold_label']
        if label == 'neutral':
            continue # skip neutral
        premise = row['sentence1']
        hypothesis = row['sentence2']
        sentences.add(premise)
        sentences.add(hypothesis)
        if label == 'entailment':
            test_radnli_entailment_pairs.append((premise, hypothesis))
        elif label == 'contradiction':
            test_radnli_contradiction_pairs.append((premise, hypothesis))
        else: 
            raise ValueError(f"Unknown label: {label}")
    print(f"Test radnli entailment pairs: {len(test_radnli_entailment_pairs)}")
    print(f"Test radnli contradiction pairs: {len(test_radnli_contradiction_pairs)}")
    print_bold('Example test radnli entailment pairs:')
    for pair in random.sample(test_radnli_entailment_pairs, 4):
        print(pair)
    print_bold('Example test radnli contradiction pairs:')
    for pair in random.sample(test_radnli_contradiction_pairs, 4):
        print(pair)

    # 2) MS-CXR-T Temporal Sentence Similarity v1
    df = pd.read_csv(MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH)
    test_mscxrt_entailment_pairs = []
    test_mscxrt_contradiction_pairs = []
    for premise, hypothesis, label in zip(df.sentence_1, df.sentence_2, df.category):
        sentences.add(premise)
        sentences.add(hypothesis)
        if label == 'paraphrase':
            test_mscxrt_entailment_pairs.append((premise, hypothesis))
        elif label == 'contradiction':
            test_mscxrt_contradiction_pairs.append((premise, hypothesis))
        else: 
            raise ValueError(f"Unknown label: {label}")
    print(f"Test mscxrt entailment pairs: {len(test_mscxrt_entailment_pairs)}")
    print(f"Test mscxrt contradiction pairs: {len(test_mscxrt_contradiction_pairs)}")
    print_bold('Example test mscxrt entailment pairs:')
    for pair in random.sample(test_mscxrt_entailment_pairs, 4):
        print(pair)
    print_bold('Example test mscxrt contradiction pairs:')
    for pair in random.sample(test_mscxrt_contradiction_pairs, 4):
        print(pair)
    
    sentences = list(sentences)
    s2i = {s: i for i, s in enumerate(sentences)}
    print(f"Total sentences: {len(sentences)}")
    # embeddings = embedding_extractor.compute_text_embeddings(sentences)
    embeddings = get_embeddings_func(sentences)
    if similarity_metric == 'cosine':
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) # normalize
    print(f"embeddings.shape: {embeddings.shape}")

    if not calibrate_on_test_set:
        best_threshold = calibrate_threshold(train_entailment_pairs, train_contradiction_pairs, embeddings, s2i)

    # Evaluate on RadNLI test set
    if calibrate_on_test_set:
        best_threshold = calibrate_threshold(test_radnli_entailment_pairs, test_radnli_contradiction_pairs, embeddings, s2i)
    print('--------------')
    print_blue('Evaluating on RadNLI test set')
    test_p_ent_idxs = [s2i[p] for p, _ in test_radnli_entailment_pairs]
    test_h_ent_idxs = [s2i[h] for _, h in test_radnli_entailment_pairs]
    test_p_con_idxs = [s2i[p] for p, _ in test_radnli_contradiction_pairs]
    test_h_con_idxs = [s2i[h] for _, h in test_radnli_contradiction_pairs]
    test_p_ent_embeddings = embeddings[test_p_ent_idxs]
    test_h_ent_embeddings = embeddings[test_h_ent_idxs]
    test_p_con_embeddings = embeddings[test_p_con_idxs]
    test_h_con_embeddings = embeddings[test_h_con_idxs]
    print(f"test_p_ent_embeddings.shape: {test_p_ent_embeddings.shape}")
    print(f"test_h_ent_embeddings.shape: {test_h_ent_embeddings.shape}")
    print(f"test_p_con_embeddings.shape: {test_p_con_embeddings.shape}")
    print(f"test_h_con_embeddings.shape: {test_h_con_embeddings.shape}")
    test_ent_cosine_sim = np.sum(test_p_ent_embeddings * test_h_ent_embeddings, axis=1)
    test_con_cosine_sim = np.sum(test_p_con_embeddings * test_h_con_embeddings, axis=1)
    print(f"test_ent_cosine_sim.shape: {test_ent_cosine_sim.shape}")
    print(f"test_con_cosine_sim.shape: {test_con_cosine_sim.shape}")
    entailment_preds = test_ent_cosine_sim >= best_threshold
    contradiction_preds = test_con_cosine_sim < best_threshold
    entailment_correct = entailment_preds.sum()
    contradiction_correct = contradiction_preds.sum()
    entailment_acc = entailment_correct / len(test_ent_cosine_sim)
    contradiction_acc = contradiction_correct / len(test_con_cosine_sim)
    print_magenta(f"entailment_acc: {entailment_acc}", bold=True)
    print_magenta(f"contradiction_acc: {contradiction_acc}", bold=True)
    print_magenta(f"test_acc: {(entailment_acc + contradiction_acc) / 2}", bold=True)
    print('--------------')

    # Evaluate on MS-CXR-T Temporal Sentence Similarity v1
    if calibrate_on_test_set:
        best_threshold = calibrate_threshold(test_mscxrt_entailment_pairs, test_mscxrt_contradiction_pairs, embeddings, s2i)
    print('--------------')
    print_blue('Evaluating on MS-CXR-T Temporal Sentence Similarity v1')
    test_p_ent_idxs = [s2i[p] for p, _ in test_mscxrt_entailment_pairs]
    test_h_ent_idxs = [s2i[h] for _, h in test_mscxrt_entailment_pairs]
    test_p_con_idxs = [s2i[p] for p, _ in test_mscxrt_contradiction_pairs]
    test_h_con_idxs = [s2i[h] for _, h in test_mscxrt_contradiction_pairs]
    test_p_ent_embeddings = embeddings[test_p_ent_idxs]
    test_h_ent_embeddings = embeddings[test_h_ent_idxs]
    test_p_con_embeddings = embeddings[test_p_con_idxs]
    test_h_con_embeddings = embeddings[test_h_con_idxs]
    print(f"test_p_ent_embeddings.shape: {test_p_ent_embeddings.shape}")
    print(f"test_h_ent_embeddings.shape: {test_h_ent_embeddings.shape}")
    print(f"test_p_con_embeddings.shape: {test_p_con_embeddings.shape}")
    print(f"test_h_con_embeddings.shape: {test_h_con_embeddings.shape}")
    test_ent_cosine_sim = np.sum(test_p_ent_embeddings * test_h_ent_embeddings, axis=1)
    test_con_cosine_sim = np.sum(test_p_con_embeddings * test_h_con_embeddings, axis=1)
    print(f"test_ent_cosine_sim.shape: {test_ent_cosine_sim.shape}")
    print(f"test_con_cosine_sim.shape: {test_con_cosine_sim.shape}")
    entailment_preds = test_ent_cosine_sim >= best_threshold
    contradiction_preds = test_con_cosine_sim < best_threshold
    entailment_correct = entailment_preds.sum()
    contradiction_correct = contradiction_preds.sum()
    entailment_acc = entailment_correct / len(test_ent_cosine_sim)
    contradiction_acc = contradiction_correct / len(test_con_cosine_sim)
    print_magenta(f"entailment_acc: {entailment_acc}", bold=True)
    print_magenta(f"contradiction_acc: {contradiction_acc}", bold=True)
    print_magenta(f"test_acc: {(entailment_acc + contradiction_acc) / 2}", bold=True)
    print('--------------')

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)