import argparse
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAMES
from medvqa.models.huggingface_utils import (
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)
from medvqa.utils.files import (
    get_file_path_with_hashing_if_too_long,
    save_pickle,
)
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR
from medvqa.utils.logging import print_blue, print_bold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--device', type=str, default='GPU', choices=['CPU', 'GPU'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)

    args = parser.parse_args()

    # Define bbox phrases
    bbox_phrases = CHEST_IMAGENOME_BBOX_NAMES
    
    # Obtain embeddings for each sentence
    print_bold('Obtaining embeddings for each sentence...')
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )
    bbox_phrase_embeddings = embedding_extractor.compute_text_embeddings(bbox_phrases)
    
    # Save embeddings
    output = {
        'bbox_phrases': bbox_phrases,
        'bbox_phrase_embeddings': bbox_phrase_embeddings,
    }
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
        prefix='bbox_phrase_embeddings',
    )
    print_blue('Saving output to:', save_path, bold=True)
    save_pickle(output, save_path)
    print('Done!')

if __name__ == '__main__':
    main()