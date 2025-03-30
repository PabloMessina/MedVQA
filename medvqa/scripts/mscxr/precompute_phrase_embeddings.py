import argparse
from medvqa.datasets.ms_cxr import get_ms_cxr_phrases
from medvqa.models.huggingface_utils import (
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)
from medvqa.utils.files import (
    get_file_path_with_hashing_if_too_long,
    save_pickle,
)
from medvqa.datasets.mimiccxr import MIMICCXR_LARGE_FAST_CACHE_DIR
from medvqa.utils.logging import print_blue, print_bold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)

    args = parser.parse_args()

    # Retrieve phrases
    phrases = get_ms_cxr_phrases()
    print(f'len(phrases): {len(phrases)}')
    
    # Obtain embeddings for each phrase
    print_bold('Obtaining embeddings for each phrase...')
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )
    phrase_embeddings = embedding_extractor.compute_text_embeddings(phrases)
    phrase2embedding = dict(zip(phrases, phrase_embeddings))
    print(f'phrase_embeddings.shape: {phrase_embeddings.shape}')
    print(f'len(phrase2embedding): {len(phrase2embedding)}')
    
    # Save phrase2embedding
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
        prefix='mscxr_phrase2embedding',
        strings=[
            args.model_name,
            args.model_checkpoint_folder_path,
            *phrases,
        ],
        force_hashing=True,
    )
    print_blue('Saving output to:', save_path, bold=True)
    save_pickle(phrase2embedding, save_path)
    print('Done!')

if __name__ == '__main__':
    main()