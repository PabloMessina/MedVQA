from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
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

INSTRUCTIONS = """You will receive a factual statement extracted from a radiology report. Your task is to output a JSON object with the following 7 fields:
- "reason": A concise explanation (up to 3 brief sentences) supporting the outputs for the other fields.
- "too_noisy_or_irrelevant": Either "yes" or "no". Use "yes" if the statement is truncated, too noisy, too vague, incomprehensible, or not a medical observation from a radiology report. Otherwise, use "no".
- "visually_observable": Either "yes" or "no". Use "yes" if the statement describes something that can be observed visually on the chest X-ray. Otherwise, return "no".
- "category": Must be one of the following: "anatomical finding", "disease", "device", "tubes and lines", "foreign body", "symptom", "technical assessment", "procedure", or "does not apply".
- "anatomical_location": A short phrase explicitly extracting the anatomical location if mentioned. Infer the location only if sufficient information is available; otherwise, return "does not apply".
- "general_observation": A shorter, more standardized version of the statement (e.g., omitting the anatomical location) suitable for indexing purposes, or "does not apply" if not possible."""

_VALID_JSON_OBJECT_REGEX = re.compile(
    r'\{\s*"reason"\s*:\s*"[^"]*"\s*,\s*"too_noisy_or_irrelevant"\s*:\s*"[^"]*"\s*,\s*"visually_observable"\s*:\s*"[^"]*"\s*,\s*"category"\s*:\s*"[^"]*"\s*,\s*"abnormality_status"\s*:\s*"[^"]*"\s*,\s*"anatomical_location"\s*:\s*"[^"]*"\s*,\s*"general_observation"\s*:\s*"[^"]*"\s*\}',
    re.DOTALL
)

_EXPECTED_FIELDS = [
    "reason",
    "too_noisy_or_irrelevant",
    "visually_observable",
    "category",
    "abnormality_status",
    "anatomical_location",
    "general_observation",
]

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
    for x in _EXPECTED_FIELDS:
        assert x in parsed_metadata, f"Could not parse output: {text}"
        assert isinstance(parsed_metadata[x], str), f"Could not parse output: {text}"
    return parsed_metadata

def get_queries(
    offset: int,
    num_samples: int,
    integrated_sentence_facts_jsonl_filepath: str,
    already_processed_queries: set,
    logger: logging.Logger,
    fact_embedding_model_name: str,
    fact_embedding_model_checkpoint_folder_path: str,
    fact_embedding_batch_size: int = 32,
    fact_embedding_num_workers: int = 4,
    device: str = "cuda",
    num_clusters: int = 300,
):
    """
    Sample queries to process.
    """
    rows = load_jsonl(integrated_sentence_facts_jsonl_filepath)
    facts = set()
    for row in rows:
        facts.update(row['facts'])
    facts = list(facts)

    logger.info(f"Total number of unique facts: {len(facts)}")
    
    facts = sort_sentences(
        sentences=facts,
        logger=logger,
        by_difficulty=True,
        cache_ranking=True,
    )

    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=device,
    )
    cluster_labels = embedding_extractor.compute_kmeans_labels(facts, num_clusters=num_clusters)
    cluster_to_idxs = [[] for _ in range(num_clusters)]
    for i, c in enumerate(cluster_labels):
        cluster_to_idxs[c].append(i)
    cluster_to_idxs = [x for x in cluster_to_idxs if len(x) > 0]
    cluster_to_idxs.sort(key=lambda x: len(x))
    sorted_idxs = []
    i = 0
    for j in range(len(cluster_to_idxs[-1])):
        while len(cluster_to_idxs[i]) <= j:
            i += 1
        for k in range(i, len(cluster_to_idxs)):
            sorted_idxs.append(cluster_to_idxs[k][j])
    
    queries = [facts[i] for i in sorted_idxs[offset:offset+num_samples] if facts[i] not in already_processed_queries]
    return queries


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--integrated_sentence_facts_jsonl_filepath", type=str, default=None)
    parser.add_argument("--queries_to_skip_filepaths", type=str, nargs="+", default=None)
    parser.add_argument("--fact_embedding_model_name", type=str, default=None)
    parser.add_argument("--fact_embedding_model_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--fact_embedding_batch_size", type=int, default=32)
    parser.add_argument("--fact_embedding_num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_clusters", type=int, default=300)
    
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

    processed_queries_save_filepath = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_fact_to_metadata{args.alias}.jsonl")

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
            integrated_sentence_facts_jsonl_filepath=args.integrated_sentence_facts_jsonl_filepath,
            already_processed_queries=already_processed,
            logger=logger,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
            device=args.device,
            num_clusters=args.num_clusters,
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