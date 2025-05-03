from dotenv import load_dotenv

from medvqa.utils.logging_utils import get_console_logger
load_dotenv()

import os
import argparse
import tiktoken
import re
import sys
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, process_api_requests_from_file
from medvqa.utils.files_utils import load_jsonl, save_jsonl
from medvqa.utils.common import get_timestamp

INSTRUCTIONS = """Given a statement extracted from a chest X-ray report, output a comparison category.  The category must be one item of the following list:

0. no comparison
1. new finding
2. resolved
3. improved
4. worsened
5. progressed
6. reappeared
7. larger
8. smaller
9. increase
10. decrease
11. position changed
12. stable/unchanged
13. unclear comparison
14. other

Examples:

right pleural effusion similar in size
12. stable/unchanged

pleural density on the left is somewhat increased
9. increase

new bibasal consolidations on the left
1. new finding

small foci of opacity in the left mid lung
0. no comparison

no ill-definition of right hemidiaphragm
0. no comparison

heart within upper limits of normal in size
0. no comparison"""

ALLOWED_GPT_CHAT_MODELS = ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4", "gpt-4-0613")

def generate_request(sentence, model_name, max_tokens, temperature=0.0):
    assert len(sentence) > 0, f"Sentence is empty: {sentence}"

    if model_name in ALLOWED_GPT_CHAT_MODELS:
        return {
            "model": model_name,
            "messages": [{
                "role": "system",
                "content": INSTRUCTIONS,
            }, {
                "role": "user",
                "content": sentence,
            }],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "metadata": {
                "sentence": sentence,
            },
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

_VALID_STRING = re.compile(r"^\d+\.\s+(.+)\b") # match a string like "0. no comparison"

_ALLOWED_CATEGORIES = set([
    "no comparison",
    "new finding",
    "resolved",
    "improved",
    "worsened",
    "progressed",
    "reappeared",
    "larger",
    "smaller",
    "increase",
    "decrease",
    "position changed",
    "stable/unchanged",
    "unclear comparison",
    "other",
])

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = _VALID_STRING.search(text) # match a string like "0. no comparison"
    if not match:
        if GPT_IS_ACTING_WEIRD_REGEX.search(text):
            raise ValueError(f"GPT is acting weird: {text}")
        else:
            raise ValueError(f"Could not parse output: {text}")
    category = match.group(1)
    assert category in _ALLOWED_CATEGORIES, f"Unexpected category: {category}"
    return category

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated_fact_metadata_filepath", type=str, required=True)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--sampling_threshold", type=int, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_sentences", type=int, required=True)
    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY_1")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    # Load already processed sentences if they exist
    comparisons_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_comparisons{args.alias}.jsonl")
    already_processed = set()
    if os.path.exists(comparisons_filepath):
        rows = load_jsonl(comparisons_filepath)
        for row in rows:
            already_processed.add(row['metadata']['sentence'])
        logger.info(f"Loaded {len(already_processed)} already processed sentences from {comparisons_filepath}")
    if args.preprocessed_sentences_to_skip_filepaths is not None:
        for filepath in args.preprocessed_sentences_to_skip_filepaths:
            assert os.path.exists(filepath)
            logger.info(f"Loading preprocessed sentences to skip from {filepath}")
            rows = load_jsonl(filepath)
            for row in rows:
                already_processed.add(row['metadata']['sentence'])
            logger.info(f"Loaded {len(already_processed)} already processed sentences from {filepath}")

    # Collect sentences to process
    assert os.path.exists(args.integrated_fact_metadata_filepath)
    logger.info(f"Loading facts metadata from {args.integrated_fact_metadata_filepath}")
    integrated_fact_metadata = load_jsonl(args.integrated_fact_metadata_filepath)
    tmp = Counter(x['extraction_method'] for x in integrated_fact_metadata)
    logger.info(f"Extraction methods: {tmp}")
    gpt_comparison2idxs = {}
    t5_comparison2idxs = {}
    inconsistent_count = 0
    for i, row in tqdm(enumerate(integrated_fact_metadata), total=len(integrated_fact_metadata), mininterval=2):
        metadata = row['metadata']
        comp = metadata['comparison status']
        psc = metadata['prev_study_comparison?']
        em = row['extraction_method']
        is_psc_invalid = psc not in ('yes', 'no')
        is_comp_inconsistent = (psc == 'yes') != (comp != '')
        if is_psc_invalid or is_comp_inconsistent:
            inconsistent_count += 1

        if em == 't5-small-finetuned':
            if comp not in t5_comparison2idxs:
                t5_comparison2idxs[comp] = []
            t5_comparison2idxs[comp].append(i)
        else:
            assert 'gpt' in em
            if is_psc_invalid or is_comp_inconsistent or (comp != '' and comp not in _ALLOWED_CATEGORIES):
                if comp not in gpt_comparison2idxs:
                    gpt_comparison2idxs[comp] = []
                gpt_comparison2idxs[comp].append(i)
    # Print stats
    if inconsistent_count > 0:
        logger.warning(f"Found {inconsistent_count} inconsistent comparison statuses")
    logger.info(f"Found {len(gpt_comparison2idxs)} unique GPT comparison statuses that need relabeling")
    logger.info(f"Found a total of {sum(len(x) for x in gpt_comparison2idxs.values())} GPT comparison statuses that need relabeling")
    logger.info(f"Found {len(t5_comparison2idxs)} unique T5 comparison statuses that need relabeling")
    logger.info(f"Found a total of {sum(len(x) for x in t5_comparison2idxs.values())} T5 comparison statuses that need relabeling")

    # Apply sampling threshold
    sentences_to_process = set()
    def _key(i):
        f = integrated_fact_metadata[i]['fact']
        return len(f), f # sort by length first, then alphabetically
    above_threshold = 0
    for comp, idxs in gpt_comparison2idxs.items():
        if len(idxs) > args.sampling_threshold:
            logger.info(f"GPT's comparison group \"{comp}\" has {len(idxs)} facts above threshold")
            idxs.sort(key=_key)
            idxs = [idxs[i] for i in np.linspace(0, len(idxs)-1, args.sampling_threshold, dtype=int)]
            above_threshold += 1
        sentences_to_process.update(integrated_fact_metadata[i]['fact'] for i in idxs)
    logger.info(f"Total number GPT's comparison groups above threshold: {above_threshold}")
    above_threshold = 0
    for comp, idxs in t5_comparison2idxs.items():
        if len(idxs) > args.sampling_threshold:
            logger.info(f"T5's comparison group \"{comp}\" has {len(idxs)} facts above threshold")
            idxs.sort(key=_key)
            idxs = [idxs[i] for i in np.linspace(0, len(idxs)-1, args.sampling_threshold, dtype=int)]
            above_threshold += 1
        sentences_to_process.update(integrated_fact_metadata[i]['fact'] for i in idxs)
    logger.info(f"Total number T5's comparison groups above threshold: {above_threshold}")
    logger.info(f"Total number of unique facts to process: {len(sentences_to_process)}")
    
    # Sort sentences
    sentences_to_process = list(sentences_to_process)
    sentences_to_process.sort(key=lambda x: (len(x), x)) # sort by length first, then alphabetically

    # Adjust number of sentences to paraphrase if necessary
    assert 0 <= args.offset < len(sentences_to_process)
    if args.offset + args.num_sentences > len(sentences_to_process):
        logger.warning(f"Requested {args.num_sentences} sentences but only {len(sentences_to_process) - args.offset} are available."
                       f" Using {len(sentences_to_process) - args.offset} instead.")
        args.num_sentences = len(sentences_to_process) - args.offset
        assert args.num_sentences > 0

    # Remove already processed sentences
    logger.info("Removing already processed sentences")
    sentences_to_process = [sentences_to_process[i] for i in range(args.offset, args.offset + args.num_sentences) if\
                               sentences_to_process[i] not in already_processed]
    if len(sentences_to_process) == 0:
        logger.info(f"All {args.num_sentences} sentences have already been processed. Nothing to do. Exiting.")
        sys.exit(0)

    logger.info(f"Total number of sentences to process: {len(sentences_to_process)}")
    
    # Print example sentences
    logger.info(f"Example sentences to process:")
    for i in np.linspace(0, len(sentences_to_process)-1, min(20, len(sentences_to_process)), dtype=int):
        logger.info(f"{i+1}. {sentences_to_process[i]}")

    # Prepare API requests
    jobs = []
    for sentence in sentences_to_process:
        jobs.append(generate_request(
            sentence=sentence,
            model_name=args.openai_model_name,
            max_tokens=args.max_tokens_per_request,
        ))
        assert 'metadata' in jobs[-1]
        assert jobs[-1]['metadata']['sentence'] == sentence
    
    timestamp = get_timestamp()
    api_requests_filepath = os.path.join(MIMICCXR_FAST_TMP_DIR, "openai", f"api_requests_{timestamp}.jsonl")
    api_responses_filepath = os.path.join(MIMICCXR_FAST_TMP_DIR, "openai", f"api_responses_{timestamp}.jsonl")
    logger.info(f"Saving API requests to {api_requests_filepath}")
    logger.info(f"Saving API responses to {api_responses_filepath}")
    save_jsonl(jobs, api_requests_filepath)

    # Send API requests
    process_api_requests_from_file(
        requests_filepath=api_requests_filepath,
        save_filepath=api_responses_filepath,
        request_url=args.openai_request_url,
        api_key=os.getenv(args.api_key_name),
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        token_encoding_name=tiktoken.encoding_for_model(args.openai_model_name).name,
        max_attempts=5,
        logging_level=args.logging_level,
        log_info_every_n_requests=50,
    )

    # Load and postprocess API responses
    logger.info(f"Loading API responses from {api_responses_filepath}")
    api_responses = load_jsonl(api_responses_filepath)
    assert len(api_responses) == len(jobs)

    postprocessed_responses = []
    for i in range(len(api_responses)):
        api_response = api_responses[i]
        assert len(api_response) == 3 # request, response, and metadata
        metadata = api_response[2]
        try:
            text = api_response[1]['choices'][0]['message']['content']
            parsed_output = parse_openai_model_output(text)
            postprocessed_responses.append({
                "metadata": metadata,
                "parsed_response": parsed_output,
            })
        except Exception as e:
            api_response_string = json.dumps(api_response)
            if len(api_response_string) > 300:
                api_response_string = api_response_string[:150] + "..." + api_response_string[-150:]
            logger.error(f"Error parsing response {api_response_string} for sentence \"{metadata['sentence']}\": {e}")
            continue

    # Delete API requests and responses
    logger.info(f"Deleting API requests and responses")
    os.remove(api_requests_filepath)
    os.remove(api_responses_filepath)

    if len(postprocessed_responses) == 0:
        logger.warning(f"None of the {len(api_responses)} API responses could be parsed. Exiting.")
    else:
        # Save processed sentences by appending to existing file
        n_processed = len(postprocessed_responses)
        n_total = len(api_responses)
        logger.info(f"""Succesfully processed {n_processed} of {n_total} API responses.
                    {n_total - n_processed} of {n_total} API responses could not be processed.
                    Saving processed sentences to {comparisons_filepath}""")
        save_jsonl(postprocessed_responses, comparisons_filepath, append=True)