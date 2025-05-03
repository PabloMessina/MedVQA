from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.logging_utils import get_console_logger
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
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import load_jsonl

INSTRUCTIONS = """You will be given a sentence extracted from a chest X-ray radiology report. Your task is to analyze it and produce a JSON object with the following fields:

1. "reason":  Provide a concise explanation (up to 2 brief sentences) outlining your reasoning.

2. "ruled_out_abnormalities": Output abnormalities (using standard radiological terminology) that are explicitly and definitively ruled out by the input. Pay attention to statements that clearly negate or rule out specific abnormalities, as well as descriptions such as "normal X", "clear X", "unremarkable X" that unambiguously establish that a region X is 100% healthy. Format: a JSON object where keys are atomic anchor statements from the input, and values are lists of strings. Break down complex sentences into atomic anchors if necessary. For example, the sentence "the heart and the lungs are normal" can be broken down into two anchors: "normal heart" and "normal lungs". If an anchor rules out multiple abnormalities, list the two most common ones. For example, "intact osseous structures" rules out ["bone fracture", "bone lesion"]. Each string value must be self-contained, which means you must include the full name of the abnormality, including anatomical location information if applicable. For example, "no left lower lobe collapse" rules out ["left lower lobe collapse"]. If a sentence is too vague, incomprehensible, or ambiguous, output an empty object. For example "left greater than right" returns {}. NOTE: No interval change of abnormality X does not rule out abnormalities. For example, "no change in consolidation" or "stable consolidation" return {}.

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
    sentences = set()
    for row in rows:
        sentences.add(row['sentence'])
        sentences.update(row['facts'])
    sentences = list(sentences)

    logger.info(f"Total number of unique sentences: {len(sentences)}")
    
    sentences = sort_sentences(
        sentences=sentences,
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
    cluster_labels = embedding_extractor.compute_kmeans_labels(sentences, num_clusters=num_clusters)
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
    
    queries = [sentences[i] for i in sorted_idxs[offset:offset+num_samples] if sentences[i] not in already_processed_queries]
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

    processed_queries_save_filepath = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_sentence_to_negative_facts{args.alias}.jsonl")

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