import argparse
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR
from medvqa.models.huggingface_utils import (
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)
from medvqa.utils.files_utils import (
    get_file_path_with_hashing_if_too_long,
    load_pickle,
    save_pickle,
)
from medvqa.utils.logging_utils import print_blue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=SupportedHuggingfaceMedicalBERTModels.get_all(),
    )
    parser.add_argument(
        "--device", type=str, default="GPU", choices=["CPU", "GPU"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_checkpoint_folder_path", type=str, required=True)
    parser.add_argument(
        "--chest_imagenome_augmented_groundings_filepath", type=str, required=True
    )

    args = parser.parse_args()

    # Load augmented groundings
    print("Loading augmented groundings from:", args.chest_imagenome_augmented_groundings_filepath)
    augmented_groundings = load_pickle(
        args.chest_imagenome_augmented_groundings_filepath
    )

    # Collect phrases from augmented groundings
    phrases = set()
    for item in augmented_groundings:
        phrases.update(item["phrase2locations"].keys())
    phrases = list(phrases)
    phrases.sort()
    print(f"Loaded {len(phrases)} phrases from augmented groundings.")

    # Create embedding extractor
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )

    # Extract embeddings
    embeddings = embedding_extractor.compute_text_embeddings(phrases)

    # Define output to save
    output = {
        "phrases": phrases,
        "embeddings": embeddings,
    }
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
        prefix="chest_imagenome_phrase_embeddings_",
        strings=[
            args.model_name,
            args.model_checkpoint_folder_path,
            args.chest_imagenome_augmented_groundings_filepath,
            f'len(phrases)={len(phrases)}',
        ],
        force_hashing=True,
    )

    # Save output
    print_blue("Saving output to:", save_path, bold=True)
    save_pickle(output, save_path)
    print("Done!")


if __name__ == "__main__":
    main()