import argparse
import logging
from medvqa.datasets.padchest import (
    PADCHEST_LARGE_FAST_CACHE_DIR,
    get_padchest_gr_labels,
    get_padchest_gr_sentences_from_reports,
)
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor, SupportedHuggingfaceMedicalBERTModels
from medvqa.utils.files_utils import get_file_path_with_hashing_if_too_long, save_pickle
from medvqa.utils.logging_utils import ANSI_BLUE_BOLD, ANSI_RESET
from medvqa.utils.logging_utils import setup_logging
setup_logging()

logger =  logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)

    args = parser.parse_args()

    # Retrieve phrases
    phrases = get_padchest_gr_sentences_from_reports() + get_padchest_gr_labels()
    logger.info(f'len(phrases): {len(phrases)}')
    
    # Obtain embeddings for each phrase
    logger.info('Obtaining embeddings for each phrase...')
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )
    phrase_embeddings = embedding_extractor.compute_text_embeddings(phrases)
    phrase2embedding = dict(zip(phrases, phrase_embeddings))
    logger.info(f'phrase_embeddings.shape: {phrase_embeddings.shape}')
    logger.info(f'len(phrase2embedding): {len(phrase2embedding)}')
    
    # Save phrase2embedding
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=PADCHEST_LARGE_FAST_CACHE_DIR,
        prefix='padchest_gr_phrase2embedding',
        strings=[
            args.model_name,
            args.model_checkpoint_folder_path,
            *phrases,
        ],
        force_hashing=True,
    )
    logger.info(f'{ANSI_BLUE_BOLD}Saving output to: {save_path}{ANSI_RESET}')
    save_pickle(phrase2embedding, save_path)
    logger.info('Done!')

if __name__ == '__main__':
    main()