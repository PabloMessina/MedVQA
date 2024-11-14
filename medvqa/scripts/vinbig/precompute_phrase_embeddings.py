import argparse

import numpy as np
from medvqa.models.huggingface_utils import (
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)
from medvqa.utils.files import (
    get_file_path_with_hashing_if_too_long,
    load_jsonl,
    save_pickle,
)
from medvqa.datasets.vinbig import VINBIG_LARGE_FAST_CACHE_DIR
from medvqa.utils.constants import VINBIG_LABELS, VINBIG_LABEL2PHRASE
from medvqa.utils.logging import print_blue, print_bold
from medvqa.utils.math import rank_vectors_by_dot_product

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='GPU', choices=['CPU', 'GPU'])
    parser.add_argument('--average_top_k_most_similar', action='store_true')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--integrated_fact_metadata_filepath', type=str, default=None)

    args = parser.parse_args()

    if args.average_top_k_most_similar:
        assert args.top_k > 0, 'top_k must be greater than 0 if average_top_k_most_similar is True'
        assert args.integrated_fact_metadata_filepath is not None, 'integrated_fact_metadata_filepath must be provided if average_top_k_most_similar is True'

    # Define  phrases
    phrases = [VINBIG_LABEL2PHRASE[label] for label in VINBIG_LABELS]
    
    # Create embedding extractor
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name, device=args.device,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )

    if args.average_top_k_most_similar:

        # Load integrated fact metadata
        print('Loading integrated fact metadata from:', args.integrated_fact_metadata_filepath)
        integrated_fact_metadata = load_jsonl(args.integrated_fact_metadata_filepath)

        # Extract facts
        facts = set()
        for x in integrated_fact_metadata:
            fact = x['fact']
            if fact: facts.add(fact)
        facts = list(facts)
        print(f'Loaded {len(facts)} facts.')

        # Extract embeddings
        print_bold('Extracting embeddings...')
        fact_embeddings = embedding_extractor.compute_text_embeddings(facts)
        phrase_embeddings = embedding_extractor.compute_text_embeddings(phrases)
        most_similar_facts = []
        most_similar_fact_embeddings = []
        most_similar_fact_similarities = []
        average_phrase_embeddings = np.zeros((len(phrases), phrase_embeddings.shape[1]), dtype=np.float32)
        for i in range(len(phrase_embeddings)):
            idxs = rank_vectors_by_dot_product(fact_embeddings, phrase_embeddings[i])
            top_k_idxs = idxs[:args.top_k]
            top_k_similar_facts = [facts[idx] for idx in top_k_idxs]
            top_k_similar_fact_embeddings = fact_embeddings[top_k_idxs]
            top_k_similarities = np.dot(top_k_similar_fact_embeddings, phrase_embeddings[i])
            for j in range(1, len(top_k_similarities)):
                try:
                    assert top_k_similarities[j] <= top_k_similarities[j-1]
                except:
                    print('top_k_similarities:', top_k_similarities)
                    print('top_k_similar_fact_embeddings:', top_k_similar_fact_embeddings)
                    print('phrase_embeddings[i]:', phrase_embeddings[i])
                    print('top_k_similar_facts:', top_k_similar_facts)
                    print('top_k_similarities[j]:', top_k_similarities[j])
                    print('top_k_similarities[j-1]:', top_k_similarities[j-1])
                    raise
            most_similar_facts.append(top_k_similar_facts)
            most_similar_fact_embeddings.append(top_k_similar_fact_embeddings)
            most_similar_fact_similarities.append(top_k_similarities)
            average_phrase_embeddings[i] = np.mean(top_k_similar_fact_embeddings, axis=0, dtype=np.float32)
        phrase_embeddings = average_phrase_embeddings

        # Define output to save
        output = {
            'phrases': phrases,
            'phrase_embeddings': phrase_embeddings,
            'most_similar_facts': most_similar_facts,
            'most_similar_fact_embeddings': most_similar_fact_embeddings,
            'most_similar_fact_similarities': most_similar_fact_similarities,
        }
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=VINBIG_LARGE_FAST_CACHE_DIR,
            prefix='label_phrase_embeddings',
            strings=[
                args.model_name,
                args.model_checkpoint_folder_path,
                args.integrated_fact_metadata_filepath,
                'average_top_k_most_similar',
                f'top_k={args.top_k}',
            ],
            force_hashing=True,
        )
    else:
        # Extract embeddings
        print_bold('Extracting embeddings...')
        phrase_embeddings = embedding_extractor.compute_text_embeddings(phrases)

        # Define output to save
        output = {
            'phrases': phrases,
            'phrase_embeddings': phrase_embeddings,
        }
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=VINBIG_LARGE_FAST_CACHE_DIR,
            prefix='label_phrase_embeddings',
            strings=[
                args.model_name,
                args.model_checkpoint_folder_path,
            ],
            force_hashing=True,
        )
    
    # Save output
    print_blue('Saving output to:', save_path, bold=True)
    save_pickle(output, save_path)
    print('Done!')

if __name__ == '__main__':
    main()