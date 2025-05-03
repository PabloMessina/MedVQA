import os
import math
import argparse
import random
import numpy as np
import pandas as pd
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.constants import CXRLT2024_TASK1_CLASSES
from medvqa.utils.logging_utils import get_console_logger
from medvqa.datasets.mimiccxr import MIMIC_CXR_LT_2024_TASK1_TRAIN_CSV_PATH, MIMICCXR_LARGE_FAST_CACHE_DIR
from medvqa.utils.files_utils import save_pickle

def export_labels_for_fact_classification(fact_encoder_model_name, fact_encoder_checkpoint_folder_path,
                                          batch_size, num_workers, positive_fraction):
    """
    Export labels for fact classification.
    """

    # Load CXR-LT 2024 training data
    logger.info(f"Loading CXR-LT 2024 training data")
    train_df = pd.read_csv(MIMIC_CXR_LT_2024_TASK1_TRAIN_CSV_PATH)
    train_dicom_ids = train_df['dicom_id'].values
    train_labels = train_df[CXRLT2024_TASK1_CLASSES].values
    logger.info(f"train_labels.shape: {train_labels.shape}")

    # Compute class embeddings
    logger.info(f"Computing class embeddings")
    emb_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_encoder_model_name,
        model_checkpoint_folder_path=fact_encoder_checkpoint_folder_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    class_embeddings = emb_extractor.compute_text_embeddings(CXRLT2024_TASK1_CLASSES)
    logger.info(f"class_embeddings.shape: {class_embeddings.shape}")

    # Split the training data into train and validation sets, making sure at least 1% of each label is in the validation set
    logger.info(f"Splitting the training data into train and validation sets")
    val_indices = set()
    for i in range(train_labels.shape[1]): # For each class
        indices = np.where(train_labels[:, i] == 1)[0]
        assert len(indices) > 1
        random.shuffle(indices)
        val_indices.update(indices[:max(1, math.ceil(positive_fraction*len(indices)))])
    train_indices = [i for i in range(len(train_labels)) if i not in val_indices] # Remaining indices
    val_indices = list(val_indices) # Convert to list
    logger.info(f"len(train_indices): {len(train_indices)}")
    logger.info(f"len(val_indices): {len(val_indices)}")

    # Save output
    output = {
        "dicom_ids": train_dicom_ids,
        "labels": train_labels,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "class_names": CXRLT2024_TASK1_CLASSES,
        "class_embeddings": class_embeddings,
    }
    save_path = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR,
                             f'cxrlt2024_official_training_labels_for_fact_classification(n_train={len(train_indices)},n_val={len(val_indices)},pf={positive_fraction:.2f}).pkl')
    save_pickle(output, save_path)
    logger.info(f"Saved output to {save_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fact_encoder_model_name", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--fact_encoder_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--positive_fraction", type=float, default=0.03)
    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    # Export labels for fact classification
    export_labels_for_fact_classification(
        fact_encoder_model_name=args.fact_encoder_model_name,
        fact_encoder_checkpoint_folder_path=args.fact_encoder_checkpoint_folder_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_fraction=args.positive_fraction,
    )