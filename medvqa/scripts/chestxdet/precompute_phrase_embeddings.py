import argparse
import logging
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor, SupportedHuggingfaceMedicalBERTModels
from medvqa.utils.files_utils import get_file_path_with_hashing_if_too_long, load_json, save_pickle
from medvqa.utils.logging_utils import ANSI_BLUE_BOLD, ANSI_RESET
from medvqa.utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def load_labels_from_json(json_path):
    data = load_json(json_path)
    labels = set()
    for item in data:
        labels.update(item['syms'])
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, required=True, help='Path to ChestXDet train JSON')
    parser.add_argument('--test_json', type=str, required=True, help='Path to ChestXDet test JSON')
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save the label2embedding pickle')

    args = parser.parse_args()

    # Load and deduplicate labels
    train_labels = load_labels_from_json(args.train_json)
    test_labels = load_labels_from_json(args.test_json)
    all_labels = sorted(train_labels | test_labels)
    logger.info(f'Number of unique labels: {len(all_labels)}')
    logger.info(f'Labels: {all_labels}')

    # Obtain embeddings for each label
    logger.info('Obtaining embeddings for each label...')
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )
    label_embeddings = embedding_extractor.compute_text_embeddings(all_labels)
    label2embedding = dict(zip(all_labels, label_embeddings))
    logger.info(f'label_embeddings.shape: {label_embeddings.shape}')
    logger.info(f'len(label2embedding): {len(label2embedding)}')

    # Save label2embedding
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=args.output_dir,
        prefix='chestxdet_label2embedding',
        strings=[
            args.model_name,
            args.model_checkpoint_folder_path,
            *all_labels,
        ],
        force_hashing=True,
    )
    logger.info(f'{ANSI_BLUE_BOLD}Saving output to: {save_path}{ANSI_RESET}')
    save_pickle(label2embedding, save_path)
    logger.info('Done!')

if __name__ == '__main__':
    main()