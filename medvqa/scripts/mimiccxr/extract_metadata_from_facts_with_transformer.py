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

from medvqa.utils.logging import get_console_logger
from medvqa.models.checkpoint import load_metadata, get_checkpoint_filepath
from medvqa.models.nlp.seq2seq import Seq2SeqModel
from medvqa.datasets.text_data_utils import create_text_dataset_and_dataloader

from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.utils.files import load_jsonl, save_jsonl

from transformers import AutoTokenizer

import re

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
        texts=facts_to_parse,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_func=tokenizer_func,
    )

    # Run inference
    logger.info(f"Running inference")
    model.eval()
    extracted_metadata = [None] * len(facts_to_parse)
    i = 0
    idx = 0
    unparsed_facts = []

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
            for output_text in output_texts:
                try:
                    extracted_metadata[idx] = {
                        'fact': facts_to_parse[i],
                        'metadata': parse_metadata(output_text),
                    }
                    idx += 1
                    if (idx-1) % 5000 == 0:
                        logger.info(f"Processed {idx} facts")
                        logger.info(f"Example extracted metadata:")
                        logger.info(f"{extracted_metadata[idx-1]}")
                except Exception as e:
                    logger.warning(f"Failed to parse output text: {output_text}, for fact: {facts_to_parse[i]}, i: {i}, idx: {idx}")
                    logger.warning(f"Exception: {e}")
                    unparsed_facts.append(facts_to_parse[idx])
                i += 1
    
    extracted_metadata = extracted_metadata[:idx]
    assert all(x is not None for x in extracted_metadata)
    
    if len(unparsed_facts) > 0:
        logger.warning(f"Failed to parse {len(unparsed_facts)} facts")
    
    logger.info(f"Successfully processed {len(extracted_metadata)} facts")
    logger.info(f"Example extracted metadata:")
    logger.info(f"{extracted_metadata[-1]}")

    assert len(facts_to_parse) == len(extracted_metadata) + len(unparsed_facts)
    
    # Save extracted metadata
    timestamp = get_timestamp()
    filename = f"extracted_metadata_{model.get_name()}_{args.max_length}_{args.num_beams}_{timestamp}.jsonl"
    save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface", filename)
    logger.info(f"Saving extracted metadata to {save_filepath}")
    save_jsonl(extracted_metadata, save_filepath)

    # Save unparsed facts
    if len(unparsed_facts) > 0:
        filename = f"unparsed_facts_{model.get_name()}_{args.max_length}_{args.num_beams}_{timestamp}.jsonl"
        save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "huggingface", filename)
        logger.info(f"Saving unparsed facts to {save_filepath}")
        save_jsonl(unparsed_facts, save_filepath)

    logger.info(f"DONE")