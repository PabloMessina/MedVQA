import argparse
from medvqa.models.huggingface_utils import (
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)
from medvqa.utils.files_utils import (
    get_file_path_with_hashing_if_too_long,
    save_pickle,
)
from medvqa.datasets.chexpert import CHEXPERT_LARGE_FAST_CACHE_DIR
from medvqa.utils.constants import CHEXPERT_LABELS, CHEXPERT_LABEL2PHRASE
from medvqa.utils.logging_utils import print_blue, print_bold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='GPU', choices=['CPU', 'GPU'])

    args = parser.parse_args()

    # Define phrases
    phrases = [CHEXPERT_LABEL2PHRASE[class_name] for class_name in CHEXPERT_LABELS]
    
    # Obtain embeddings for each sentence
    print_bold('Obtaining embeddings for each sentence...')
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name, device=args.device,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )
    phrase_embeddings = embedding_extractor.compute_text_embeddings(phrases)
    
    # Save embeddings
    output = {
        'class_phrases': phrases,
        'class_phrase_embeddings': phrase_embeddings,
    }
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=CHEXPERT_LARGE_FAST_CACHE_DIR,
        prefix='class_phrase_embeddings',
        strings=[
            args.model_name,
            args.model_checkpoint_folder_path,
            *phrases,
        ],
        force_hashing=True,
    )
    print_blue('Saving output to:', save_path, bold=True)
    save_pickle(output, save_path)
    print('Done!')

if __name__ == '__main__':
    main()