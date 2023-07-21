from dotenv import load_dotenv
load_dotenv()

import os
import re
import argparse
import sys
import json
import numpy as np

from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.utils.logging import get_console_logger
from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.utils.files import load_jsonl

_VALID_JSON_OBJECT_CONTENT_REGEX = re.compile(r"\s*\"anatomical location\"\s*:\s*\"[^\"]*\"\s*,\s*\"detailed observation\"\s*:\s*\"[^\"]*\"\s*,\s*\"short observation\"\s*:\s*\"[^\"]*\"\s*,\s*\"category\"\s*:\s*\"[^\"]*\"\s*,\s*\"health status\"\s*:\s*\"[^\"]*\"\s*,\s*\"prev_study_comparison\?\"\s*:\s*\"[^\"]*\"\s*,\s*\"comparison status\"\s*:\s*\"[^\"]*\"\s*")

def parse_metadata(txt):
    metadata_str = _VALID_JSON_OBJECT_CONTENT_REGEX.search(txt).group()
    metadata = json.loads("{" + metadata_str + "}")
    return metadata

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_facts_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--extracted_facts_jsonl_filepath", type=str, required=True)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--checkpoint_folder_path", type=str, required=True)
    parser.add_argument("--device", type=str, default='GPU', choices=['GPU', 'CPU'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    already_parsed_facts = set()
    facts_to_parse = set()

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
                    already_parsed_facts.add(row['metadata']['fact'])
            logger.info(f"Total number of facts to skip: {len(already_parsed_facts)}")

    # Collect unparsed facts from input files
    assert os.path.exists(args.extracted_facts_jsonl_filepath)
    logger.info(f"Loading facts from {args.extracted_facts_jsonl_filepath}")
    rows = load_jsonl(args.extracted_facts_jsonl_filepath)
    logger.info(f"Loaded {len(rows)} sentences with extracted facts from {args.extracted_facts_jsonl_filepath}")
    for row in rows:
        for f in row['facts']:
            if f not in already_parsed_facts:
                facts_to_parse.add(f)
    
    logger.info(f"Found {len(facts_to_parse)} facts to parse")
    if len(facts_to_parse) == 0:
        logger.info("Nothing to do. Exiting.")
        sys.exit(0)

    facts_to_parse = list(facts_to_parse) # Convert to list for indexing
    
    # Print example facts
    logger.info(f"Example facts to parse:")
    indices = np.random.choice(len(facts_to_parse), min(10, len(facts_to_parse)), replace=False)
    for i in indices:
        logger.info(f"{i}: {facts_to_parse[i]}")

    # Extract metadata from facts
    save_dir = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface")
    save_filename_prefix = "extracted_metadata"
    def _postprocess_input_output_func(sentence, output_text):
        return {
            'fact': sentence,
            'metadata': parse_metadata(output_text),
        }
    apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=args.checkpoint_folder_path,
        sentences=facts_to_parse,
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