import argparse
import numpy as np
import pandas as pd
from medvqa.metrics.medical.chexbert import CheXbertLabeler
from medvqa.utils.common import parsed_args_to_dict
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor, SupportedHuggingfaceMedicalBERTModels
from medvqa.utils.logging_utils import print_blue
from medvqa.utils.metrics_utils import auc

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--huggingface_model_name', type=str, default=None, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--model_name', type=str, default=None, choices=['CheXbert'])
    parser.add_argument('--model_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--average_token_embeddings', action='store_true')
    parser.add_argument('--entcont_csv_filepath', type=str, required=True)
    return parser.parse_args(args=args)

def evaluate(
    model_name,
    huggingface_model_name,
    device,
    model_checkpoint_folder_path,
    batch_size,
    num_workers,
    average_token_embeddings,
    entcont_csv_filepath,
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

    print(f'Loading entailment and contradiction pairs from {entcont_csv_filepath}')
    df = pd.read_csv(entcont_csv_filepath)
    print(f'Number of samples: {len(df)}')
    premises, hypotheses, labels = [], [], []
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

    sentences = set()
    sentences.update(premises)
    sentences.update(hypotheses)    
    sentences = list(sentences)
    s2i = {s: i for i, s in enumerate(sentences)}
    print(f"Total sentences: {len(sentences)}")
    embeddings = get_embeddings_func(sentences)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) # normalize for cosine similarity
    print(f"embeddings.shape: {embeddings.shape}")

    # Evaluate on the provided entailment and contradiction pairs
    print('--------------')
    print_blue('Evaluating on the provided entailment and contradiction pairs')
    test_p_idxs = [s2i[p] for p in premises]
    test_h_idxs = [s2i[h] for h in hypotheses]
    test_p_embeddings = embeddings[test_p_idxs]
    test_h_embeddings = embeddings[test_h_idxs]
    print(f"test_p_embeddings.shape: {test_p_embeddings.shape}")
    print(f"test_h_embeddings.shape: {test_h_embeddings.shape}")
    test_cosine_sim = np.sum(test_p_embeddings * test_h_embeddings, axis=1)
    print(f"test_cosine_sim.shape: {test_cosine_sim.shape}")
    score = auc(test_cosine_sim, labels)
    print_blue(f"AUC: {score}", bold=True)

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)