import os
import argparse
import sys
import json
import numpy as np
from medvqa.utils.logging_utils import get_console_logger
from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
)
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import load_jsonl
from medvqa.datasets.nli import (
    RADNLI_DEV_JSONL_PATH,
    RADNLI_TEST_JSONL_PATH,
)

INSTRUCTIONS = """You will receive a NLI example from a chest X-ray dataset with premise, hypothesis and label ("entailment", "contradiction" or "neutral"). Generate 10 new NLI examples with the same label and following a similar writing style as the example provided. They have to be about chest X-ray reports. Output the examples as a JSON array of objects. If the premise or the hypothesis (or both) are complex (i.e. they say several things), create challenging examples following  a similar structure as the provided example that require a good understanding of logic in order to deduce the label.
Avoid verbosity and words such as "patient" or "X-ray". State observations succinctly."""

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    data = json.loads(text)
    assert isinstance(data, list), f"Could not parse output: {text}"
    for example in data:
        assert isinstance(example, dict), f"Could not parse output: {text}"
        for k in ['P', 'H', 'L']:
            assert k in example, f"Could not parse output: {text}"
            assert isinstance(example[k], str), f"Could not parse output: {text}"
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)

    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_texts", type=int, required=True)

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

    processed_texts_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_nli_examples_around_radnli{args.alias}.jsonl")
    
    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None:

        # Load already processed texts if they exist
        already_processed = set()
        if os.path.exists(processed_texts_save_filepath):
            rows = load_jsonl(processed_texts_save_filepath)
            for row in rows:
                already_processed.add(row['metadata']['query'])
            logger.info(f"Loaded {len(rows)} already processed texts from {processed_texts_save_filepath}")

        # Collect reference examples from RadNLI dataset
        texts = []
        radnli_dev_examples = load_jsonl(RADNLI_DEV_JSONL_PATH)
        radnli_test_examples = load_jsonl(RADNLI_TEST_JSONL_PATH)
        radnli_examples = radnli_dev_examples + radnli_test_examples
        for ex in radnli_examples:
            p = ex['sentence1'] # premise
            h = ex['sentence2'] # hypothesis
            l = ex['gold_label'] # label
            texts.append(json.dumps({'P': p, 'H': h, 'L': l}))
        logger.info(f"Loaded {len(radnli_examples)} RadNLI examples")
        logger.info(f"Loaded {len(texts)} texts from RadNLI examples")

        # Adjust number of texts to process if necessary
        assert 0 <= args.offset < len(texts)
        if args.offset + args.num_texts > len(texts):
            logger.warning(f"Requested {args.num_texts} texts but only {len(texts) - args.offset} are available."
                        f" Using {len(texts) - args.offset} instead.")
            args.num_texts = len(texts) - args.offset
            assert args.num_texts > 0

        # Apply offset and num_texts
        logger.info(f"Collecting the first {args.num_texts} texts starting from the {args.offset}-th text")
        texts = texts[args.offset:args.offset+args.num_texts]

        # Remove already processed texts
        logger.info(f"Removing {len(already_processed)} already processed texts")
        texts_to_process = [s for s in texts if s not in already_processed]
        if len(texts_to_process) == 0:
            logger.info(f"All {len(texts)} texts have already been processed. Nothing to do. Exiting.")
            sys.exit(0)

        logger.info(f"Total number of texts to process: {len(texts_to_process)}")

        # Print example texts
        logger.info(f"Example texts to process:")
        for i in np.linspace(0, len(texts_to_process)-1, min(20, len(texts_to_process)), dtype=int):
            logger.info(f"{i+1}. {texts_to_process[i]}")

    else:
        assert os.path.exists(args.api_responses_filepath)
        texts_to_process = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=texts_to_process,
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
        save_filepath=processed_texts_save_filepath,
    )