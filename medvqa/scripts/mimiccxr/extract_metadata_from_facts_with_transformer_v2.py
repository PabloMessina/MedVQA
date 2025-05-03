import os
import re
import argparse
import sys
import json
import numpy as np
from tqdm import tqdm

from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.utils.logging_utils import get_console_logger, print_red
from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.utils.files_utils import load_jsonl, save_jsonl

_VALID_JSON_OBJECT_REGEX = re.compile(
    r'\s*"reason"\s*:\s*"[^"]*"\s*,\s*"too_noisy_or_irrelevant"\s*:\s*"[^"]*"\s*,\s*"visually_observable"\s*:\s*"[^"]*"\s*,\s*"category"\s*:\s*"[^"]*"\s*,\s*"abnormality_status"\s*:\s*"[^"]*"\s*,\s*"anatomical_location"\s*:\s*"[^"]*"\s*,\s*"general_observation"\s*:\s*"[^"]*"\s*',
    re.DOTALL
)

def parse_metadata(txt):
    metadata_str = _VALID_JSON_OBJECT_REGEX.search(txt).group()
    metadata = json.loads("{" + metadata_str + "}")
    return metadata

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_facts_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--extracted_facts_jsonl_filepath", type=str, default=None)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--device", type=str, default='gpu', choices=['cpu', 'gpu'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--integrate_metadata_files", action='store_true')
    parser.add_argument("--metadata_jsonl_filepaths_to_integrate", type=str, nargs="+", default=None)
    parser.add_argument("--metadata_extraction_methods", type=str, nargs="+", default=None)
    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.integrate_metadata_files:
        assert args.metadata_jsonl_filepaths_to_integrate is not None
        assert args.metadata_extraction_methods is not None
        
        fact2metadata = {}
        fact_metadata_rows = []
        for extracted_metadata_filepath, metadata_extraction_method in zip(
                args.metadata_jsonl_filepaths_to_integrate, args.metadata_extraction_methods):
            assert os.path.exists(extracted_metadata_filepath)
            print(f'Loading extracted metadata from {extracted_metadata_filepath}...')

            extracted_metadata = load_jsonl(extracted_metadata_filepath)
            for x in tqdm(extracted_metadata, total=len(extracted_metadata), mininterval=2):
                try:
                    try:
                        f = x['metadata']['query']
                        m = x['parsed_response']
                    except KeyError:
                        f = x['fact']
                        m = x['metadata']
                except KeyError:
                    print(f'KeyError: {x}')
                    raise
                if f in fact2metadata and m != fact2metadata[f]:
                    print_red(f'Warning: fact "{f}" already found with different metadata. fact2metadata[f] = {fact2metadata[f]}, m = {m}')
                fact2metadata[f] = m
                fact_metadata_rows.append({
                    'fact': f,
                    'metadata': m,
                    'extraction_method': metadata_extraction_method
                })
        # Save fact metadata rows
        metadata_total_length = 0
        for row in fact_metadata_rows:
            for k, v in row['metadata'].items():
                metadata_total_length += len(v)
        logger.info(f"Total number of metadata rows: {len(fact_metadata_rows)}")
        logger.info(f"Total length of metadata: {metadata_total_length}")
        fact_metadata_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface", f"integrated_fact_metadata({len(fact_metadata_rows)},{metadata_total_length}).jsonl")
        save_jsonl(fact_metadata_rows, fact_metadata_filepath)
        logger.info(f"Saved fact metadata to {fact_metadata_filepath}")
        sys.exit(0)

    already_parsed_facts = set()

    # Load preprocessed facts to skip if they exist
    if args.preprocessed_facts_to_skip_filepaths is not None:
        for filepath in args.preprocessed_facts_to_skip_filepaths:
            assert os.path.exists(filepath)
            logger.info(f"Loading preprocessed facts to skip from {filepath}")
            facts_to_skip = load_jsonl(filepath)
            logger.info(f"Loaded {len(facts_to_skip)} facts to skip")
            for row in facts_to_skip:
                if 'fact' in row:
                    already_parsed_facts.add(row['fact'])
                else:
                    already_parsed_facts.add(row['metadata']['query'])
            logger.info(f"Total number of facts to skip: {len(already_parsed_facts)}")

    # Collect unprocessed facts from input files
    assert os.path.exists(args.extracted_facts_jsonl_filepath)
    logger.info(f"Loading facts from {args.extracted_facts_jsonl_filepath}")
    rows = load_jsonl(args.extracted_facts_jsonl_filepath)
    facts_to_process = set()
    for row in rows:
        for f in row['facts']:
            if f not in already_parsed_facts:
                facts_to_process.add(f)
    
    logger.info(f"Found {len(facts_to_process)} facts to process")
    if len(facts_to_process) == 0:
        logger.info("Nothing to do. Exiting.")
        sys.exit(0)

    facts_to_process = list(facts_to_process) # Convert to list for indexing
    
    # Print example facts
    logger.info(f"Example facts to process:")
    indices = np.random.choice(len(facts_to_process), min(10, len(facts_to_process)), replace=False)
    for i in indices:
        logger.info(f"{i}: {facts_to_process[i]}")

    # Extract metadata from facts
    save_dir = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface")
    save_filename_prefix = "extracted_metadata_v2"
    def _postprocess_input_output_func(sentence, output_text):
        return {
            'fact': sentence,
            'metadata': parse_metadata(output_text),
        }
    apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=args.checkpoint_folder_path,
        sentences=facts_to_process,
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