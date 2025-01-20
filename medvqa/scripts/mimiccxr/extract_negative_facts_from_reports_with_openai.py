from medvqa.utils.logging import get_console_logger
from medvqa.utils.nlp import sort_sentences
import logging
import os
import argparse
import re
import json
import numpy as np

from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_LARGE_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files import load_jsonl

INSTRUCTIONS = """You will be given a list of factual statements extracted from a chest X-ray radiology report. Your task is to analyze them and produce a JSON object with the following fields:

1. "reason":  Provide a concise explanation (up to 4 brief sentences) outlining your reasoning.

2. "ruled_out_abnormalities": Output abnormalities (using standard radiological terminology) that are explicitly and definitively ruled out by the input. Pay attention to statements that clearly negate or rule out specific abnormalities, as well as descriptions such as "normal X", "clear X", "unremarkable X" as long as they unambiguously establish that a region X is 100% healthy. Output short, self-contained phrases (ideally using canonical names). Format: a JSON object where keys are anchor phrases from the report, and values are lists of strings. If an anchor rules out multiple abnormalities,  list the two most common examples. For example, "intact osseous structures" rules out ["bone fracture", "bone lesion"]. NOTE: each string value must be self-contained, which means you must include the full name of the abnormality, including anatomical location information if applicable. For example, "no left lower lobe collapse" rules out ["left lower lobe collapse"].

Ensure that your output is valid JSON and strictly adheres to the rules above."""

_VALID_JSON_OBJECT_REGEX = re.compile(
    r'\{\s*"reason"\s*:\s*"[^"]*"\s*,\s*"ruled_out_abnormalities"\s*:\s*\{\s*(?:"[^"]*"\s*:\s*\[\s*(?:"[^"]*"(?:\s*,\s*"[^"]*")*)?\s*\]\s*,?\s*)*\}\s*\}',
    re.DOTALL
)

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    text = text.replace("\\\"", "") # remove escaped quotes
    match = _VALID_JSON_OBJECT_REGEX.search(text) # match a JSON list of strings
    if not match:
        if GPT_IS_ACTING_WEIRD_REGEX.search(text):
            raise ValueError(f"GPT is acting weird: {text}")
        else:
            raise ValueError(f"Could not parse output: {text}")
    parsed_metadata = json.loads(match.group(0))
    assert isinstance(parsed_metadata, dict), f"Could not parse output: {text}"
    return parsed_metadata

def get_queries(
    offset: int,
    num_samples: int,
    integrated_report_facts_jsonl_filepath: str,
    already_processed_queries: set,
    logger: logging.Logger,
):
    """
    Sample queries to process.
    """
    rows = load_jsonl(integrated_report_facts_jsonl_filepath)
    fact_based_reports = set()
    for row in rows:
        fact_based_reports.add(row['fact_based_report'])
    fact_based_reports = list(fact_based_reports)

    logger.info(f"Total number of fact based reports: {len(fact_based_reports)}")
    
    fact_based_reports = sort_sentences(
        sentences=fact_based_reports,
        logger=logger,
        by_difficulty=True,
        cache_ranking=True,
    )
    
    queries = [fact_based_reports[i] for i in range(offset, offset+num_samples)
               if fact_based_reports[i] not in already_processed_queries]
    return queries


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--integrated_report_facts_jsonl_filepath", type=str, default=None)
    parser.add_argument("--queries_to_skip_filepaths", type=str, nargs="+", default=None)
    
    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_request", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    parser.add_argument("--not_delete_api_requests_and_responses", action="store_true", default=False)
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--use_batch_api", action="store_true", default=False)
    parser.add_argument("--batch_description", type=str, default=None)
    parser.add_argument("--batch_input_file_id", type=str, default=None)

    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    processed_queries_save_filepath = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_fact_based_report_to_negative_facts{args.alias}.jsonl")

    if args.api_responses_filepath is None and args.batch_input_file_id is None:

        # Load already processed queries if they exist
        already_processed = set()
        if os.path.exists(processed_queries_save_filepath):
            rows = load_jsonl(processed_queries_save_filepath)
            already_processed.update([x['metadata']['query'] for x in rows])
            logger.info(f"Loaded {len(rows)} already processed queries from {processed_queries_save_filepath}")

        # Load queries to skip
        if args.queries_to_skip_filepaths is not None:
            for queries_to_skip_filepath in args.queries_to_skip_filepaths:
                rows = load_jsonl(queries_to_skip_filepath)
                already_processed.update([x['metadata']['query'] for x in rows])
                logger.info(f"Loaded {len(rows)} queries to skip from {queries_to_skip_filepath}")

        # Sample queries
        queries_to_process = get_queries(
            offset=args.offset,
            num_samples=args.num_samples,
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            already_processed_queries=already_processed,
            logger=logger,
        )

        logger.info(f"Total number of queries to process: {len(queries_to_process)}")

        # Print example queries
        logger.info(f"Example queries to process:")
        for i in np.linspace(0, len(queries_to_process)-1, min(10, len(queries_to_process)), dtype=int):
            logger.info(f"{i+1}. {queries_to_process[i]}")

    else:
        if args.api_responses_filepath is not None:
            assert os.path.exists(args.api_responses_filepath)
        queries_to_process = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=queries_to_process,
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
        save_filepath=processed_queries_save_filepath,
        use_batch_api=args.use_batch_api,
        batch_description=args.batch_description,
        batch_input_file_id=args.batch_input_file_id,
        delete_api_requests_and_responses=not args.not_delete_api_requests_and_responses,
    )