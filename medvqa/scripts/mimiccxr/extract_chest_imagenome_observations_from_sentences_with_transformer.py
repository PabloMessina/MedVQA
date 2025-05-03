from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import numpy as np
from tqdm import tqdm

from medvqa.scripts.mimiccxr.extract_chest_imagenome_observations_from_sentences_with_openai import parse_openai_model_output
from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.utils.logging_utils import get_console_logger
from medvqa.datasets.mimiccxr import MIMICCXR_LARGE_FAST_CACHE_DIR
from medvqa.utils.files_utils import load_jsonl, load_pickle

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
                if 'sentence' in rows[0]['metadata']:
                    sentences_to_skip.update(row['metadata']['sentence'] for row in rows) # backward compatibility
                else:
                    sentences_to_skip.update(row['metadata']['query'] for row in rows)
            elif filepath.endswith(".pkl"):
                data = load_pickle(filepath)
                sentences_to_skip.update(data['phrases'])
            else:
                raise ValueError(f"Unknown file extension: {filepath}")
            logger.info(f"Loaded {len(sentences_to_skip) - size_before} sentences to skip from {filepath}")
    logger.info(f"Found {len(sentences_to_skip)} sentences to skip in total")
    
    # Collect sentences from integrated_fact_metadata
    unique_sentences = set()
    assert os.path.exists(args.integrated_facts_metadata_jsonl_filepath)
    rows = load_jsonl(args.integrated_facts_metadata_jsonl_filepath)
    logger.info(f"Loaded {len(rows)} rows from {args.integrated_facts_metadata_jsonl_filepath}")
    for row in tqdm(rows, mininterval=2):
        fact = row['fact']
        metadata = row['metadata']
        short_obs = metadata['short observation']
        detailed_obs = metadata['detailed observation']
        for x in [fact, short_obs, detailed_obs]:
            if x:
                unique_sentences.add(x)

    # Collect sentences from paraphrases
    if args.paraphrases_jsonl_filepaths is not None:
        for filepath in args.paraphrases_jsonl_filepaths:
            assert os.path.exists(filepath)
            rows = load_jsonl(filepath)
            logger.info(f"Loaded {len(rows)} rows from {filepath}")
            for row in tqdm(rows, mininterval=2):
                s = next(iter(row['metadata'].values()))
                parsed_response = row['parsed_response']
                if type(parsed_response) == list:
                    p = parsed_response
                elif type(parsed_response) == dict:
                    assert 'positives' in parsed_response and 'negatives' in parsed_response
                    p = parsed_response['positives'] + parsed_response['negatives']
                else:
                    raise ValueError(f'Unknown type {type(parsed_response)}')
                p.append(s)
                for x in p:
                    if len(x) > 0 and any(c.isalpha() for c in x):
                        unique_sentences.add(x)
    
    logger.info(f"Found {len(unique_sentences)} unique sentences")
    assert len(unique_sentences) > 0
    
    sentences_to_process = [s for s in unique_sentences if s not in sentences_to_skip]
    logger.info(f"Found {len(sentences_to_process)} sentences to process")
    
    # Print example sentences
    logger.info(f"Example sentences to process:")
    indices = np.random.choice(len(sentences_to_process), min(10, len(sentences_to_process)), replace=False)
    for i in indices:
        logger.info(f"{i}: {sentences_to_process[i]}")
    
    # Extract chest imagenome observations from sentences
    save_dir = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "huggingface")
    save_filename_prefix = "chest_imagenome_observations_from_sentences"
    def _postprocess_input_output_func(sentence, output_text):
        return {
            'sentence': sentence,
            'observations': parse_openai_model_output(output_text),
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