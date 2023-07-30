from dotenv import load_dotenv

from medvqa.scripts.mimiccxr.extract_chest_imagenome_labels_from_sentences_with_openai import parse_openai_model_output
load_dotenv()

import os
import argparse
import numpy as np
from tqdm import tqdm

from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.utils.logging import get_console_logger
from medvqa.datasets.mimiccxr import MIMICCXR_LARGE_FAST_CACHE_DIR
from medvqa.utils.files import load_jsonl, load_pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated_facts_metadata_jsonl_filepath", type=str, default=None)
    parser.add_argument('--paraphrases_jsonl_filepaths', type=str, nargs='+', default=None)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--checkpoint_folder_path", type=str, required=True)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--device", type=str, default='GPU', choices=['GPU', 'CPU'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    # Load preprocessed sentences to skip if they exist
    sentences_to_skip = set()
    if args.preprocessed_sentences_to_skip_filepaths is not None:
        for filepath in args.preprocessed_sentences_to_skip_filepaths:
            assert os.path.exists(filepath)
            size_before = len(sentences_to_skip)
            if filepath.endswith(".jsonl"):
                rows = load_jsonl(filepath)
                sentences_to_skip.update(row['metadata']['query'] for row in rows)
            elif filepath.endswith(".pkl"):
                data = load_pickle(filepath)
                sentences_to_skip.update(data['phrases'])
            else:
                raise ValueError(f"Unknown file extension: {filepath}")
            logger.info(f"Loaded {len(sentences_to_skip) - size_before} sentences to skip from {filepath}")

    # Collect sentences from metadata and paraphrases
    
    # From metadata:
    unique_sentences = set()
    assert os.path.exists(args.integrated_facts_metadata_jsonl_filepath)
    integrated_fact_metadata = load_jsonl(args.integrated_facts_metadata_jsonl_filepath)
    logger.info(f"Loaded {len(integrated_fact_metadata)} facts metadata from {args.integrated_facts_metadata_jsonl_filepath}")
    for r in tqdm(integrated_fact_metadata, total=len(integrated_fact_metadata), mininterval=2):
        fact = r['fact']
        detailed_observation = r['metadata']['detailed observation']
        short_observation = r['metadata']['short observation']
        for x in (fact, detailed_observation, short_observation):
            if len(x) > 0 and any(c.isalpha() for c in x) and x not in sentences_to_skip:
                unique_sentences.add(x)

    # From paraphrases:
    assert args.paraphrases_jsonl_filepaths is not None
    for filepath in args.paraphrases_jsonl_filepaths:
        assert os.path.exists(filepath)
        rows = load_jsonl(filepath)
        logger.info(f"Loaded {len(rows)} paraphrases from {filepath}")
        for row in rows:
            p = row['parsed_response']
            s = next(iter(row['metadata'].values()))
            p.append(s)
            for x in p:
                if len(x) > 0 and any(c.isalpha() for c in x) and x not in sentences_to_skip:
                    unique_sentences.add(x)

    logger.info(f"Found {len(unique_sentences)} sentences to process")
    sentences_to_process = list(unique_sentences)
    
    # Print example sentences
    logger.info(f"Example sentences to process:")
    indices = np.random.choice(len(sentences_to_process), min(10, len(sentences_to_process)), replace=False)
    for i in indices:
        logger.info(f"{i}: {sentences_to_process[i]}")
    
    # Extract chest imagenome labels from sentences
    save_dir = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "huggingface")
    save_filename_prefix = "chest_imagenome_labels_from_sentences"
    def _postprocess_input_output_func(sentence, output_text):
        return {
            'sentence': sentence,
            'labels': parse_openai_model_output(output_text),
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