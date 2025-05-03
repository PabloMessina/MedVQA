import os
import argparse
import random
from medvqa.datasets.iuxray import IUXRAY_FAST_CACHE_DIR, IUXRAY_FAST_TMP_DIR, IUXRAY_REPORTS_MIN_JSON_PATH
from medvqa.utils.logging_utils import get_console_logger
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import load_json, load_jsonl

INSTRUCTIONS = """Given a tag extracted from a IU X-ray report, generate a short factual statement conveying the same information contained in the tag. Keep sentences simple. Avoid unnecessary verbosity.

Examples:

Input: Diaphragmatic Eventration/right
Output: Right diaphragmatic eventration

Input: Thoracic Vertebrae/degenerative/mild
Output: Mild degenerative thoracic vertebrae

Input: Airspace Disease/lung/bilateral/scattered/patchy/multiple/mild
Output: Mild scattered patchy bilateral airspace disease in the lungs"""

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    text = text.strip()
    assert len(text) > 0, f"Empty text: {text}"
    return text

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    args = parser.parse_args()

    processed_queries_save_filepath = os.path.join(IUXRAY_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_manual_tags_to_sentences(with_forward_slash){args.alias}.jsonl")
    
    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None:

        # Load already processed queries if they exist
        queries_to_skip = set()
        if os.path.exists(processed_queries_save_filepath):
            rows = load_jsonl(processed_queries_save_filepath)
            for row in rows:
                queries_to_skip.add(row['metadata']['query'])
            logger.info(f"Loaded {len(rows)} already processed queries from {processed_queries_save_filepath}")

        # Collect queries to make
        reports = load_json(IUXRAY_REPORTS_MIN_JSON_PATH)
        manual_tags = set()
        for report in reports.values():
            manual_tags.update(report['tags_manual'])
        manual_tags = list(manual_tags)
        manual_tags.sort()
        manual_tags = [x for x in manual_tags if '/' in x]  # Only keep tags with '/
        queries_to_make = manual_tags

        # Remove queries that have already been processed
        logger.info(f"Queries to make: {len(queries_to_make)} (before removing queries that have already been processed)")
        n_before = len(queries_to_make)
        queries_to_make = [x for x in queries_to_make if x not in queries_to_skip]
        logger.info(f"Queries to make: {len(queries_to_make)} (after removing queries that have already been processed)")
        n_after = len(queries_to_make)
        logger.info(f"Removed {n_before - n_after} queries that have already been processed")

        # Print some examples
        logger.info(f"Example queries to make:")
        for i in random.sample(range(len(queries_to_make)), k=3):
            logger.info(f"Query {i + 1}: {queries_to_make[i]}")

    else:
        assert os.path.exists(args.api_responses_filepath)
        queries_to_make = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=queries_to_make,
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
        tmp_dir=IUXRAY_FAST_TMP_DIR,
        save_filepath=processed_queries_save_filepath,
    )