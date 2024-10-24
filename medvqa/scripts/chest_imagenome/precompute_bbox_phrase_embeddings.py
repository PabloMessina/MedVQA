import argparse

import numpy as np
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAMES
from medvqa.models.huggingface_utils import (
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)
from medvqa.utils.files import (
    get_file_path_with_hashing_if_too_long,
    load_jsonl,
    save_pickle,
)
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR
from medvqa.utils.logging import print_blue, print_bold
from medvqa.utils.math import rank_vectors_by_dot_product

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--device', type=str, default='GPU', choices=['CPU', 'GPU'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)
    parser.add_argument('--average_top_k_most_similar', action='store_true')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--integrated_fact_metadata_filepath', type=str, default=None)

    args = parser.parse_args()

    if args.average_top_k_most_similar:
        assert args.top_k > 0, 'top_k must be greater than 0 if average_top_k_most_similar is True'
        assert args.integrated_fact_metadata_filepath is not None, 'integrated_fact_metadata_filepath must be provided if average_top_k_most_similar is True'

    # Define bbox phrases
    bbox_phrases = CHEST_IMAGENOME_BBOX_NAMES

    # Create embedding extractor
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )

    if args.average_top_k_most_similar:
        # Load integrated fact metadata
        print('Loading integrated fact metadata from:', args.integrated_fact_metadata_filepath)
        integrated_fact_metadata = load_jsonl(args.integrated_fact_metadata_filepath)

        # Extract anatomical locations
        anatomical_locations = set()
        for x in integrated_fact_metadata:
            al = x['metadata']['anatomical location']
            if al: anatomical_locations.add(al)
        anatomical_locations = list(anatomical_locations)
        print(f'Loaded {len(anatomical_locations)} anatomical locations.')

        # Extract embeddings
        print_bold('Extracting embeddings...')
        anatomical_location_embeddings = embedding_extractor.compute_text_embeddings(anatomical_locations)
        bbox_phrase_embeddings = embedding_extractor.compute_text_embeddings(bbox_phrases)
        most_similar_anatomical_locations = []
        average_phrase_embeddings = np.zeros((len(bbox_phrases), bbox_phrase_embeddings.shape[1]), dtype=np.float32)
        for i in range(len(bbox_phrase_embeddings)):
            idxs = rank_vectors_by_dot_product(anatomical_location_embeddings, bbox_phrase_embeddings[i])
            most_similar_anatomical_locations.append([anatomical_locations[idx] for idx in idxs[:args.top_k]])
            average_phrase_embeddings[i] = np.mean(anatomical_location_embeddings[idxs[:args.top_k]], axis=0, dtype=np.float32)
        bbox_phrase_embeddings = average_phrase_embeddings

        # Define output to save
        output = {
            'bbox_phrases': bbox_phrases,
            'bbox_phrase_embeddings': bbox_phrase_embeddings,
            'most_similar_anatomical_locations': most_similar_anatomical_locations,
        }
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
            prefix='bbox_phrase_embeddings',
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
        bbox_phrase_embeddings = embedding_extractor.compute_text_embeddings(bbox_phrases)
        
        # Define output to save
        output = {
            'bbox_phrases': bbox_phrases,
            'bbox_phrase_embeddings': bbox_phrase_embeddings,
        }
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
            prefix='bbox_phrase_embeddings',
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