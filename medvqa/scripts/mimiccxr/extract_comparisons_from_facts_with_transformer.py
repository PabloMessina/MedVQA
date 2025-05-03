from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import numpy as np
from tqdm import tqdm

from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.scripts.mimiccxr.extract_comparisons_from_facts_with_openai import _ALLOWED_CATEGORIES
from medvqa.utils.logging_utils import get_console_logger
from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.utils.files_utils import load_jsonl

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated_fact_metadata_filepath", type=str, required=True)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--checkpoint_folder_path", type=str, required=True)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--device", type=str, default='GPU', choices=['GPU', 'CPU'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    # Load preprocessed sentences to skip if they exist
    already_processed = set()
    if args.preprocessed_sentences_to_skip_filepaths is not None:
        for filepath in args.preprocessed_sentences_to_skip_filepaths:
            assert os.path.exists(filepath)
            logger.info(f"Loading preprocessed sentences to skip from {filepath}")
            rows = load_jsonl(filepath)
            for row in rows:
                already_processed.add(row['metadata']['sentence'])
            logger.info(f"Loaded {len(already_processed)} already processed sentences from {filepath}")

    # Collect sentences to process
    assert os.path.exists(args.integrated_fact_metadata_filepath)
    logger.info(f"Loading facts metadata from {args.integrated_fact_metadata_filepath}")
    integrated_fact_metadata = load_jsonl(args.integrated_fact_metadata_filepath)
    sentences_to_process = set()
    for i, row in tqdm(enumerate(integrated_fact_metadata), total=len(integrated_fact_metadata), mininterval=2):
        fact = row['fact']
        if fact in already_processed:
            continue
        metadata = row['metadata']
        comp = metadata['comparison status']
        psc = metadata['prev_study_comparison?']
        em = row['extraction_method']
        is_psc_invalid = psc not in ('yes', 'no')
        is_comp_inconsistent = (psc == 'yes') != (comp != '')
        if is_psc_invalid or is_comp_inconsistent:
            sentences_to_process.add(fact)
            continue
        if em == 't5-small-finetuned':
            sentences_to_process.add(fact)
        else:
            assert 'gpt' in em
            if (comp != '' and comp not in _ALLOWED_CATEGORIES):
                sentences_to_process.add(fact)
    logger.info(f"Found {len(sentences_to_process)} sentences to process")

    sentences_to_process = list(sentences_to_process)
    
    # Print example sentences
    logger.info(f"Example sentences to process:")
    indices = np.random.choice(len(sentences_to_process), min(10, len(sentences_to_process)), replace=False)
    for i in indices:
        logger.info(f"{i}: {sentences_to_process[i]}")
    
    # Extract comparisons
    save_dir = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface")
    save_filename_prefix = "extracted_comparisons"
    def _postprocess_input_output_func(sentence, output_text):
        return {
            'sentence': sentence,
            'comparison': output_text,
        }
    apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=args.checkpoint_folder_path,
        sentences=sentences_to_process,
        logger=logger,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        num_beams=args.num_beams,
        save_dir=save_dir,
        save_filename_prefix=save_filename_prefix,
        postprocess_input_output_func=_postprocess_input_output_func,
    )