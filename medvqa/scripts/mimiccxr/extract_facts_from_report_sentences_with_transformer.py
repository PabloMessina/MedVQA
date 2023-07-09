from dotenv import load_dotenv

from medvqa.utils.common import get_timestamp

load_dotenv()

import torch
import os
import argparse
import sys
import json
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from medvqa.utils.logging import get_console_logger
from medvqa.models.checkpoint import load_metadata, get_checkpoint_filepath
from medvqa.models.nlp.seq2seq import Seq2SeqModel
from medvqa.datasets.text_data_utils import create_text_dataset_and_dataloader

from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.files import load_json, load_jsonl, save_jsonl

from transformers import AutoTokenizer

import re

_COMMA_SEPARATED_LIST_REGEX = re.compile(r'\[\s*(\".+?\"(\s*,\s*\".+?\")*)?\s*\]?')

def parse_facts(txt):
    facts_str = _COMMA_SEPARATED_LIST_REGEX.search(txt).group()
    if facts_str[-1] != ']': facts_str += ']'
    facts = json.loads(facts_str)
    seen = set()
    clean_facts = []
    for fact in facts:
        if fact not in seen:
            clean_facts.append(fact)
            seen.add(fact)
    return clean_facts

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
                already_parsed_sentences.add(row['metadata']['sentence'])
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
    
    # Load model metadata
    metadata = load_metadata(args.checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'GPU' else 'CPU')
    
    # Create model
    logger.info(f"Creating Seq2SeqModel")
    model = Seq2SeqModel(**model_kwargs)
    model = model.to(device)

    # Load model weights
    logger.info(f"Loading model weights from {args.checkpoint_folder_path}")
    checkpoint_path = get_checkpoint_filepath(args.checkpoint_folder_path)
    logger.info(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Create tokenizer
    logger.info(f"Creating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_kwargs['model_name'])

    # Create dataset and dataloader
    logger.info(f"Creating dataset and dataloader")
    tokenizer_func = lambda x: tokenizer(x, padding="longest", return_tensors="pt")
    dataset, dataloader = create_text_dataset_and_dataloader(
        texts=sentences_to_parse,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_func=tokenizer_func,
    )

    # Run inference
    logger.info(f"Running inference")
    model.eval()
    extracted_facts = [None] * len(sentences_to_parse)
    idx = 0
    unparsed_sentences = []
    output_text_batches = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), mininterval=2):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output_ids = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_len=args.max_length,
                num_beams=args.num_beams,
                mode='test',
            )
            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            output_text_batches.append(output_texts)
            for output_text in output_texts:
                try:
                    extracted_facts[idx] = {
                        'sentence': sentences_to_parse[idx],
                        'extracted_facts': parse_facts(output_text),
                    }
                    idx += 1
                    if (idx-1) % 5000 == 0:
                        logger.info(f"Processed {idx} sentences")
                        logger.info(f"Example extracted facts:")
                        logger.info(f"{extracted_facts[idx-1]}")
                except Exception as e:
                    logger.warning(f"Failed to parse output text: {output_text}, for sentence: {sentences_to_parse[idx]}, idx: {idx}")
                    logger.warning(f"Exception: {e}")
                    unparsed_sentences.append(sentences_to_parse[idx])
    
    extracted_facts = extracted_facts[:idx]
    assert all(f is not None for f in extracted_facts)
    
    if len(unparsed_sentences) > 0:
        logger.warning(f"Failed to parse {len(unparsed_sentences)} sentences")
    
    logger.info(f"Successfully processed {len(extracted_facts)} sentences")
    logger.info(f"Example extracted facts:")
    logger.info(f"{extracted_facts[-1]}")

    assert len(sentences_to_parse) == len(extracted_facts) + len(unparsed_sentences)
    
    # Save extracted facts
    timestamp = get_timestamp()
    filename = f"extracted_facts_{model.get_name()}_{args.max_length}_{args.num_beams}_{timestamp}.jsonl"
    save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface", filename)
    logger.info(f"Saving extracted facts to {save_filepath}")
    save_jsonl(extracted_facts, save_filepath)

    # Save unparsed sentences
    if len(unparsed_sentences) > 0:
        filename = f"unparsed_sentences_{model.get_name()}_{args.max_length}_{args.num_beams}_{timestamp}.jsonl"
        save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface", filename)
        logger.info(f"Saving unparsed sentences to {save_filepath}")
        save_jsonl(unparsed_sentences, save_filepath)

    logger.info(f"DONE")