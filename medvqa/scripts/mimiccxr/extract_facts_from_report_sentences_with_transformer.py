from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import sys
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from medvqa.utils.logging_utils import get_console_logger
from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.datasets.text_data_utils import parse_facts
from medvqa.utils.files_utils import load_json, load_jsonl

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_sentences_to_skip_filenames", nargs="+", default=None)
    parser.add_argument("--extracted_sentences_jsonl_filepaths", nargs="+", default=None)
    parser.add_argument("--preprocessed_reports_filename", type=str, default=None)
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

    already_parsed_sentences = set()
    sentences_to_parse = set()

    # Load preprocessed sentences to skip if they exist
    if args.preprocessed_sentences_to_skip_filenames is not None:
        for filename in args.preprocessed_sentences_to_skip_filenames:
            preprocessed_sentences_to_skip_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", filename)
            assert os.path.exists(preprocessed_sentences_to_skip_filepath)
            logger.info(f"Loading preprocessed sentences to skip from {preprocessed_sentences_to_skip_filepath}")
            sentences_to_skip = load_jsonl(preprocessed_sentences_to_skip_filepath)
            logger.info(f"Loaded {len(sentences_to_skip)} sentences to skip")
            for row in sentences_to_skip:
                try:
                    already_parsed_sentences.add(row['metadata']['query'])
                except KeyError:
                    already_parsed_sentences.add(row['metadata']['sentence']) # backward compatibility
            logger.info(f"Total number of sentences to skip: {len(already_parsed_sentences)}")

    # Collect unparsed sentences from input files
    if args.extracted_sentences_jsonl_filepaths is not None:
        for filepath in args.extracted_sentences_jsonl_filepaths:
            logger.info(f"Loading sentences from {filepath}")
            rows = load_jsonl(filepath)
            logger.info(f"Loaded {len(rows)} reports with extracted sentences from {filepath}")
            for row in rows:
                for s, _ in row['parsed_response']:
                    if s not in already_parsed_sentences:
                        sentences_to_parse.add(s)

    # Collect unparsed sentences from reports
    if args.preprocessed_reports_filename is not None:
        preprocessed_reports_filepath = os.path.join(MIMICCXR_CACHE_DIR, args.preprocessed_reports_filename)
        logger.info(f"Loading preprocessed reports from {preprocessed_reports_filepath}")
        reports = load_json(preprocessed_reports_filepath)
        for r in tqdm(reports, total=len(reports), mininterval=2):
            impression = r['impression']
            findings = r['findings']
            if len(impression) > 0:
                for s in sent_tokenize(impression):
                    if s not in already_parsed_sentences:
                        sentences_to_parse.add(s)
            if len(findings) > 0:
                for s in sent_tokenize(findings):
                    if s not in already_parsed_sentences:
                        sentences_to_parse.add(s)
        logger.info(f"Loaded {len(reports)} reports from {preprocessed_reports_filepath}")
    
    logger.info(f"Found {len(sentences_to_parse)} unique sentences to parse")
    if len(sentences_to_parse) == 0:
        logger.info("Nothing to do. Exiting.")
        sys.exit(0)

    sentences_to_parse = list(sentences_to_parse) # Convert to list for indexing
    
    # Print example sentences
    logger.info(f"Example sentences to parse:")
    indices = np.random.choice(len(sentences_to_parse), min(10, len(sentences_to_parse)), replace=False)
    for i in indices:
        logger.info(f"{i}: {sentences_to_parse[i]}")

    # Extract facts from sentences
    save_dir = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface")
    save_filename_prefix = "extracted_facts"
    def _postprocess_input_output_func(sentence, output_text):
        return {
            'sentence': sentence,
            'extracted_facts': parse_facts(output_text),
        }
    apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=args.checkpoint_folder_path,
        sentences=sentences_to_parse,
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