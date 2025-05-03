from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import re
import sys
import json
import numpy as np
from tqdm import tqdm

from medvqa.utils.nlp import sort_sentences
from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api_utils import (
    GPT_IS_ACTING_WEIRD_REGEX,
    run_common_boilerplate_for_api_requests,
)
from medvqa.utils.logging_utils import get_console_logger
from medvqa.utils.files_utils import load_jsonl, load_pickle

INSTRUCTIONS = """Instructions:

Apply an annotation standard of anatomical locations to raw phrases extracted from chest X-ray reports.
The standard considers the following anatomical locations:

abdomen
aortic arch
cardiac silhouette
carina
cavoatrial junction
left apical zone
left arm
left breast
left chest wall
left clavicle
left costophrenic angle
left hemidiaphragm
left hilar structures
left lower lung zone
left lung
left mid lung zone
left shoulder
left upper lung zone
mediastinum
neck
right apical zone
right arm
right atrium
right breast
right chest wall
right clavicle
right costophrenic angle
right hemidiaphragm
right hilar structures
right lower lung zone
right lung
right mid lung zone
right shoulder
right upper lung zone
spine
svc
trachea
upper mediastinum

Output format:

JSON array of strings

Rules:

Only output the anatomical locations from the standard that best match the given phrase.
If none match the phrase, output an empty array ([]).

Examples:

cardiomediastinal silhouette
[
"cardiac silhouette",
"mediastinum"
]

A tracheostomy tube is in place
[
"neck",
"trachea"
]

bilateral pulmonary nodules
[
"left lung",
"right lung"
]

left IJV
[
"neck"
]"""

# Match a possibly truncated JSON array of strings, by matching as many strings as possible.
# This is useful because the OpenAI API sometimes truncates the output.
_JSON_STRING_ARRAY_REGEX = re.compile(r'^\[\s*(\".+?\"(\s*,\s*\".+?\")*)?\s*\]?')

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = _JSON_STRING_ARRAY_REGEX.search(text) # match a JSON list of strings
    if not match and GPT_IS_ACTING_WEIRD_REGEX.search(text):
        logger.warning(f"GPT is acting weird: {text}")
    assert match, f"Could not parse output: {text}"
    string = match.group(0)
    assert string[0] == "[", f"Could not parse output: {text}"
    if string[-1] != "]": string += "]" # add closing bracket
    list_of_strings = json.loads(string)
    assert isinstance(list_of_strings, list), f"Could not parse output: {text}"
    assert all(isinstance(x, str) for x in list_of_strings), f"Could not parse output: {text}"
    return list_of_strings

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--integrated_fact_metadata_filepath", type=str, default=None)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top_k_most_difficult_sentences", type=int, default=None)
    group.add_argument("--skip_top_k_most_difficult_sentences", type=int, default=None)

    parser.add_argument("--uniformly_sample_k_sentences", type=int, default=None)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_sentences", type=int, required=True)
    parser.add_argument("--process_kth_of_every_n_sentences", type=int, nargs=2, default=None,
                        help="If specified, only process the kth of every n sentences.")

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    args = parser.parse_args()
    
    processed_sentences_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_chest_imagenome_anatomies_from_sentences{args.alias}.jsonl")

    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None:
        
        # Load already processed sentences if they exist
        already_processed = set()
        if os.path.exists(processed_sentences_save_filepath):
            rows = load_jsonl(processed_sentences_save_filepath)
            for row in rows:
                already_processed.add(row['metadata']['query'])
            logger.info(f"Loaded {len(rows)} already processed sentences from {processed_sentences_save_filepath}")
        
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

        # Collect sentences from metadata
        
        # From metadata:
        unique_sentences = set()
        assert os.path.exists(args.integrated_fact_metadata_filepath)
        integrated_fact_metadata = load_jsonl(args.integrated_fact_metadata_filepath)
        logger.info(f"Loaded {len(integrated_fact_metadata)} facts metadata from {args.integrated_fact_metadata_filepath}")
        for r in tqdm(integrated_fact_metadata, total=len(integrated_fact_metadata), mininterval=2):
            fact = r['fact']
            anatloc = r['metadata']['anatomical location']
            for x in (fact, anatloc):
                if len(x) > 0 and any(c.isalpha() for c in x) and x not in sentences_to_skip:
                    unique_sentences.add(x)

        logger.info(f"Found {len(unique_sentences)} unique sentences")
        assert len(unique_sentences) > 0

        # Sort sentences by difficulty
        unique_sentences = list(unique_sentences)
        unique_sentences = sort_sentences(unique_sentences, logger, by_difficulty=True, cache_ranking=True)

        # Select sentences by difficulty
        if args.top_k_most_difficult_sentences is not None:
            logger.info(f"Selecting the top {args.top_k_most_difficult_sentences} most difficult sentences")
            unique_sentences = unique_sentences[:args.top_k_most_difficult_sentences]
        elif args.skip_top_k_most_difficult_sentences is not None:
            logger.info(f"Skipping the top {args.skip_top_k_most_difficult_sentences} most difficult sentences")
            unique_sentences = unique_sentences[args.skip_top_k_most_difficult_sentences:]
        else:
            assert False, "Must specify either --top_k_most_difficult_sentences or --skip_top_k_most_difficult_sentences"

        # Uniformly sample k sentences
        if args.uniformly_sample_k_sentences is not None:
            logger.info(f"Uniformly sampling {args.uniformly_sample_k_sentences} sentences")
            unique_sentences = [unique_sentences[i] for i in np.linspace(0, len(unique_sentences)-1, args.uniformly_sample_k_sentences, dtype=int)]

        # Filter sentences by kth of every n sentences if necessary
        if args.process_kth_of_every_n_sentences is not None:
            k, n = args.process_kth_of_every_n_sentences
            assert 0 <= k < n
            logger.info(f"Filtering sentences to the {k}-th of every {n} sentences")
            unique_sentences = [x for i, x in enumerate(unique_sentences) if i % n == k]
            logger.info(f"Found {len(unique_sentences)} sentences that are the {k}-th of every {n}")

        # Adjust number of sentences to process if necessary
        assert 0 <= args.offset < len(unique_sentences)
        if args.offset + args.num_sentences > len(unique_sentences):
            logger.warning(f"Requested {args.num_sentences} sentences but only {len(unique_sentences) - args.offset} are available."
                        f" Using {len(unique_sentences) - args.offset} instead.")
            args.num_sentences = len(unique_sentences) - args.offset
            assert args.num_sentences > 0

        # Collect sentences to process
        logger.info(f"Collecting the first {args.num_sentences} sentences from the {args.offset}-th sentence")
        sentences_to_process = [unique_sentences[i] for i in range(args.offset, args.offset + args.num_sentences)
                                    if unique_sentences[i] not in already_processed]
        if len(sentences_to_process) == 0:
            logger.info(f"All {args.num_sentences} sentences have already been processed. Nothing to do. Exiting.")
            sys.exit(0)

        logger.info(f"Total number of sentences to process: {len(sentences_to_process)}")
        
        # Print example sentences
        logger.info(f"Example sentences to process:")
        for i in np.linspace(0, len(sentences_to_process)-1, min(10, len(sentences_to_process)), dtype=int):
            logger.info(f"{i+1}. {sentences_to_process[i]}")

    else:
        assert os.path.exists(args.api_responses_filepath)
        sentences_to_process = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=sentences_to_process,
        system_instructions=INSTRUCTIONS,
        api_key_name=args.api_key_name,
        openai_model_name=args.openai_model_name,
        openai_request_url=args.openai_request_url,
        max_tokens_per_request=args.max_tokens_per_request,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        temperature=args.temperature,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logger=logger,
        logging_level=args.logging_level,
        parse_openai_output=parse_openai_model_output,
        tmp_dir=MIMICCXR_FAST_TMP_DIR,
        save_filepath=processed_sentences_save_filepath,
    )