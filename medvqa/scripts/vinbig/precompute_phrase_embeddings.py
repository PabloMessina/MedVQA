import argparse

import numpy as np
from medvqa.models.huggingface_utils import (
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)
from medvqa.utils.files import (
    get_file_path_with_hashing_if_too_long,
    load_pickle,
    save_pickle,
)
from medvqa.datasets.vinbig import VINBIG_LABELS__MODIFIED, VINBIG_LARGE_FAST_CACHE_DIR
from medvqa.utils.constants import VINBIG_LABELS, VINBIG_LABEL2PHRASE
from medvqa.utils.logging import print_blue, print_bold
from medvqa.utils.math import rank_vectors_by_dot_product

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--average_top_k_most_similar', action='store_true')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--facts_relevant_to_anchor_facts_pickle_filepath', type=str, default=None)
    parser.add_argument('--use_modified_labels', action='store_true')

    args = parser.parse_args()

    if args.average_top_k_most_similar:
        assert args.top_k > 0, 'top_k must be greater than 0 if average_top_k_most_similar is True'
        assert args.facts_relevant_to_anchor_facts_pickle_filepath is not None, 'facts_relevant_to_anchor_facts_pickle_filepath must be provided if average_top_k_most_similar is True'

    # Define phrases
    if args.use_modified_labels:
        vinbig_labels = VINBIG_LABELS__MODIFIED
    else:
        vinbig_labels = VINBIG_LABELS
    phrases = [VINBIG_LABEL2PHRASE[label] for label in vinbig_labels]
    
    # Create embedding extractor
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name, device=args.device,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )

    if args.average_top_k_most_similar:

        # Load facts
        print(f'Loading facts_relevant_to_anchor_facts from {args.facts_relevant_to_anchor_facts_pickle_filepath}...')
        tmp = load_pickle(args.facts_relevant_to_anchor_facts_pickle_filepath)
        anchor_facts = tmp['anchor_facts']
        relevant_facts = tmp['relevant_facts']
        anchors_per_fact = tmp['anchors_per_fact']

        # Extract embeddings
        print_bold('Extracting embeddings...')
        relevant_fact_embeddings = embedding_extractor.compute_text_embeddings(relevant_facts)
        phrase_embeddings = embedding_extractor.compute_text_embeddings(phrases)
        most_similar_facts = []
        most_similar_fact_embeddings = []
        most_similar_fact_similarities = []
        average_phrase_embeddings = np.zeros((len(phrases), phrase_embeddings.shape[1]), dtype=np.float32)
        classes_to_skip = ('Other lesion', 'Other disease', 'No finding', 'Abnormal finding')
        for i in range(len(phrase_embeddings)):
            if vinbig_labels[i] in classes_to_skip:
                most_similar_facts.append(None)
                most_similar_fact_embeddings.append(None)
                most_similar_fact_similarities.append(None)
                average_phrase_embeddings[i] = phrase_embeddings[i]
                continue
            else:
                anchor_idx = anchor_facts.index(vinbig_labels[i])
                anchor_fact_idxs = [i for i, anchors in enumerate(anchors_per_fact) if anchor_idx in anchors]
                print(f'{vinbig_labels[i]}: len(anchor_fact_idxs)={len(anchor_fact_idxs)}')
                anchor_fact_embeddings = relevant_fact_embeddings[anchor_fact_idxs]
                idxs = rank_vectors_by_dot_product(anchor_fact_embeddings, phrase_embeddings[i])
                top_k_idxs = idxs[:args.top_k]
                top_k_idxs = [anchor_fact_idxs[j] for j in top_k_idxs]
                top_k_similar_facts = [relevant_facts[j] for j in top_k_idxs]
                top_k_similar_fact_embeddings = relevant_fact_embeddings[top_k_idxs]
                top_k_similarities = np.dot(top_k_similar_fact_embeddings, phrase_embeddings[i])
                for j in range(1, len(top_k_similarities)):
                    try:
                        assert ((top_k_similarities[j] <= top_k_similarities[j-1]) or 
                                abs(top_k_similarities[j] - top_k_similarities[j-1]) < 1e-6)
                    except:
                        print('top_k_similarities:', top_k_similarities)
                        print('top_k_similar_fact_embeddings:', top_k_similar_fact_embeddings)
                        print(f'phrase_embeddings[{i}]:', phrase_embeddings[i])
                        print('top_k_similar_facts:', top_k_similar_facts)
                        print(f'top_k_similarities[{j}]:', top_k_similarities[j])
                        print(f'top_k_similarities[{j-1}]:', top_k_similarities[j-1])
                        raise
                most_similar_facts.append(top_k_similar_facts)
                most_similar_fact_embeddings.append(top_k_similar_fact_embeddings)
                most_similar_fact_similarities.append(top_k_similarities)
                average_phrase_embeddings[i] = np.mean(top_k_similar_fact_embeddings, axis=0, dtype=np.float32)
                average_phrase_embeddings[i] = average_phrase_embeddings[i] / np.linalg.norm(average_phrase_embeddings[i])
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
            prefix='label_phrase_embeddings' if not args.use_modified_labels else 'label_phrase_embeddings__modified',
            strings=[
                args.model_name,
                args.model_checkpoint_folder_path,
                args.facts_relevant_to_anchor_facts_pickle_filepath,
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
            prefix='label_phrase_embeddings' if not args.use_modified_labels else 'label_phrase_embeddings__modified',
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