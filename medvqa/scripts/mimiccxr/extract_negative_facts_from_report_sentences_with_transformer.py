import os
import re
import argparse
import sys
import json
import numpy as np
from tqdm import tqdm

from medvqa.datasets.text_data_utils import remove_consecutive_repeated_words_from_text
from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.utils.logging_utils import get_console_logger, print_red
from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.utils.files_utils import load_jsonl, save_jsonl

_VALID_JSON_OBJECT_REGEX = re.compile(
    r'\s*(?:"[^"]*"\s*:\s*\[\s*(?:"[^"]*"(?:\s*,\s*"[^"]*")*)?\s*\]\s*,?\s*)*',
    re.DOTALL
)

def parse_output(txt):
    output_str = _VALID_JSON_OBJECT_REGEX.search(txt).group()
    output = json.loads("{" + output_str + "}")
    return output

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_queries_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--integrated_sentence_facts_jsonl_filepath", type=str, default=None)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--device", type=str, default='gpu', choices=['cpu', 'gpu'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--integrate_files", action='store_true')
    parser.add_argument("--jsonl_filepaths_to_integrate", type=str, nargs="+", default=None)
    parser.add_argument("--extraction_methods", type=str, nargs="+", default=None)
    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.integrate_files:
        assert args.jsonl_filepaths_to_integrate is not None
        assert args.extraction_methods is not None
        assert len(args.jsonl_filepaths_to_integrate) == len(args.extraction_methods)
        
        s2ra = {}
        rows = []
        skip_count = 0
        for filepath, extraction_method in zip(
                args.jsonl_filepaths_to_integrate, args.extraction_methods):
            assert os.path.exists(filepath)
            print(f'Loading extracted negative facts from {filepath}...')

            tmp = load_jsonl(filepath)
            for x in tqdm(tmp, total=len(tmp), mininterval=2):
                try:
                    try:
                        s = x['metadata']['query']
                        ra = x['parsed_response']['ruled_out_abnormalities']
                    except KeyError:
                        s = x['sentence']
                        ra = x['ruled_out_abnormalities']
                except KeyError:
                    print(f'KeyError: {x}')
                    raise
                if s in s2ra:
                    if s2ra[s] != ra:
                        print_red(f'Warning: sentence "{s}" already found with different output. s2ra[s] = {s2ra[s]}, ra = {ra}')
                    skip_count += 1
                    continue # Skip
                clean_ra = {}
                for k, v in ra.items():
                    assert isinstance(v, list)
                    if len(v) > 0:
                        assert all(isinstance(x, str) for x in v)
                        v = [remove_consecutive_repeated_words_from_text(x) for x in v] # Remove consecutive repeated words
                        v = [x[3:] if x.lower().startswith("no ") else x for x in v] # Remove "no " from the beginning of the string if it exists
                        v = list(set(v)) # Remove duplicates
                        v.sort()
                        clean_ra[k] = v
                s2ra[s] = clean_ra
                rows.append({
                    'sentence': s,
                    'ruled_out_abnormalities': clean_ra,
                    'extraction_method': extraction_method,
                })
        logger.info(f"Skipped {skip_count} sentences with conflicting outputs")
        # Save rows
        total_length = 0
        for row in rows:
            total_length += len(json.dumps(row))
        logger.info(f"Total number of rows: {len(rows)}")
        logger.info(f"Total length: {total_length}")
        save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, f"integrated_sentence_to_negative_facts({len(rows)},{total_length}).jsonl")
        save_jsonl(rows, save_filepath)
        logger.info(f"Saved integrated negative facts to {save_filepath}")
        sys.exit(0)

    already_processed_queries = set()

    # Load preprocessed quries to skip if they exist
    if args.preprocessed_queries_to_skip_filepaths is not None:
        for filepath in args.preprocessed_queries_to_skip_filepaths:
            assert os.path.exists(filepath)
            logger.info(f"Loading preprocessed queries to skip from {filepath}")
            queries_to_skip = load_jsonl(filepath)
            logger.info(f"Loaded {len(queries_to_skip)} queries to skip")
            for row in queries_to_skip:
                try:
                    already_processed_queries.add(row['metadata']['query'])
                except KeyError:
                    already_processed_queries.add(row['sentence'])
            logger.info(f"Total number of queries to skip: {len(already_processed_queries)}")

    # Collect unprocessed sentences
    assert os.path.exists(args.integrated_sentence_facts_jsonl_filepath)
    logger.info(f"Loading sentences from {args.integrated_sentence_facts_jsonl_filepath}")
    rows = load_jsonl(args.integrated_sentence_facts_jsonl_filepath)
    sentences_to_process = set()
    for row in rows:
        s = row['sentence']
        if s in already_processed_queries:
            continue
        sentences_to_process.add(s)
    
    logger.info(f"Found {len(sentences_to_process)} sentences to process")
    if len(sentences_to_process) == 0:
        logger.info("Nothing to do. Exiting.")
        sys.exit(0)

    sentences_to_process = list(sentences_to_process) # Convert to list for indexing
    
    # Print examples of sentences to process
    logger.info(f"Example sentences to process:")
    indices = np.random.choice(len(sentences_to_process), min(10, len(sentences_to_process)), replace=False)
    for i in indices:
        logger.info(f"{i}: {sentences_to_process[i]}")

    # Process sentences
    save_dir = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface")
    save_filename_prefix = "negative_facts_from_report_sentences"
    def _postprocess_input_output_func(sentence, output_text):
        return {
            'sentence': sentence,
            'ruled_out_abnormalities': parse_output(output_text),
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