from dotenv import load_dotenv

from medvqa.utils.logging import get_console_logger
load_dotenv()

import os
import sys
import argparse
import tiktoken
import re
import random
import json
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api import GPT_IS_ACTING_WEIRD_REGEX, process_api_requests_from_file
from medvqa.utils.files import load_jsonl, save_jsonl
from medvqa.utils.common import get_timestamp

INSTRUCTIONS = """Given a medical fact, output a JSON object with 7 fields:

1. "anatomical location"
2. "detailed observation"
3. "short observation"
4. "category"
5. "health status"
6. "prev_study_comparison?"
7. "comparison status"

Definitions:

1. "anatomical location" means the anatomical location of the observation. If not given, it should be "".

2. "detailed observation" means what was observed, with all the details provided (but excluding location, unless the location itself is the main observation).

3. "short observation" should be a brief summary of the previous field, without extra details, but logically consistent.

4. "category" can be "anatomical finding", "disease", "technical assessment", "tubes and lines" or "device".

5. "health status" can be "normal", "abnormal", "ambiguous" or "unknown".
"normal" means healthy.
"abnormal" means unhealthy.
"ambiguous" means it is not clear whether it is healthy or unhealthy, leaving room for interpretation.
"unknown" means the observation provides no information about abnormality/normality.

6. "prev_study_comparison?" is a boolean field. Can be either "yes" or "no". If the fact is comparing the current study with a previous study (something improved, worsened, changed or remained the same), return "yes". Otherwise, return "no".

7. "comparison status" can be "resolved", "new", "improved", "worsened", "larger", "smaller", "displaced", or "". If no comparison with a previous study is made, the default is "" (empty).

Examples:

small to moderate size left pleural effusion
{
"anatomical location": "left",
"detailed observation": "small to moderate size pleural effusion",
"short observation": "pleural effusion",
"category": "anatomical finding",
"health status": "abnormal",
"prev_study_comparison?": "no",
"comparison status": ""
}

new finding of mass in the abdomen
{
"anatomical location": "abdomen",
"detailed observation": "mass",
"short observation": "mass",
"category": "anatomical finding",
"health status": "abnormal",
"prev_study_comparison?": "yes",
"comparison status": "new"
}

clear lung fields
{
"anatomical location": "lung fields",
"detailed observation": "clear lung fields",
"short observation": "clear lung fields",
"category": "anatomical finding",
"health status": "normal",
"prev_study_comparison?": "no",
"comparison status": ""
}

no ill-definition of right hemidiaphragm
{
"anatomical location": "right hemidiaphragm",
"detailed observation": "no ill-definition of right hemidiaphragm
",
"short observation": "no ill-definition of right hemidiaphragm
",
"category": "anatomical finding",
"health status": "normal",
"prev_study_comparison?": "no",
"comparison status": ""
}"""

ALLOWED_GPT_CHAT_MODELS = ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-0613")

def generate_request(fact, model_name, max_tokens, temperature=0.0):
    assert len(fact) > 0, f"Fact is empty: {fact}"

    if model_name in ALLOWED_GPT_CHAT_MODELS:
        return {
            "model": model_name,
            "messages": [{
                "role": "system",
                "content": INSTRUCTIONS,
            }, {
                "role": "user",
                "content": fact,
            }],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "metadata": {
                "fact": fact,
            },
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

_VALID_JSON_OBJECT_REGEX = re.compile(r"\{\s*\"anatomical location\"\s*:\s*\"[^\"]*\"\s*,\s*\"detailed observation\"\s*:\s*\"[^\"]*\"\s*,\s*\"short observation\"\s*:\s*\"[^\"]*\"\s*,\s*\"category\"\s*:\s*\"[^\"]*\"\s*,\s*\"health status\"\s*:\s*\"[^\"]*\"\s*,\s*\"prev_study_comparison\?\"\s*:\s*\"[^\"]*\"\s*,\s*\"comparison status\"\s*:\s*\"[^\"]*\"\s*\}")

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = _VALID_JSON_OBJECT_REGEX.search(text) # match a JSON list of strings
    if not match:
        if GPT_IS_ACTING_WEIRD_REGEX.search(text):
            raise ValueError(f"GPT is acting weird: {text}")
        else:
            raise ValueError(f"Could not parse output: {text}")
    parsed_fact = json.loads(match.group(0))
    assert isinstance(parsed_fact, dict), f"Could not parse output: {text}"
    for x in ("anatomical location", "detailed observation", "short observation", "category", "health status", "prev_study_comparison?", "comparison status"):
        assert x in parsed_fact, f"Could not parse output: {text}"
        assert isinstance(parsed_fact[x], str), f"Could not parse output: {text}"
    return parsed_fact

def sort_facts(facts):
    assert type(facts) == list, f"Expected list, got {type(facts)}"
   
    logger.info("Sorting facts by difficulty...")
    tokenized_facts = [word_tokenize(x) for x in tqdm(facts, mininterval=2)]
    logger.info("Counting word frequencies...")
    vocab_freq = dict()        
    for tokens in tqdm(tokenized_facts, mininterval=2):
        for word in tokens:
            vocab_freq[word] = vocab_freq.get(word, 0) + 1
    def _difficulty(i):
        return sum(1 / vocab_freq[word] for word in tokenized_facts[i])
    ranked_indices = sorted(range(len(tokenized_facts)), key=_difficulty, reverse=True)
    ranked_facts = [facts[i] for i in ranked_indices]
    
    logger.info("Done sorting facts.")
    logger.info(f"First fact: {ranked_facts[0]}")
    logger.info(f"Last fact: {ranked_facts[-1]}")
    return ranked_facts

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_facts_jsonl_filepath", type=str, required=True)
    parser.add_argument("--preprocessed_facts_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_facts", type=int, required=True)
    parser.add_argument("--sample_facts_uniformly", action="store_true", default=False)
    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])    
    parser.add_argument("--alias", type=str, default="")
    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    # Load parsed facts if they exist
    parsed_facts_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_parsed_facts{args.alias}.jsonl")
    already_parsed_facts = set()
    if os.path.exists(parsed_facts_filepath):
        parsed_facts = load_jsonl(parsed_facts_filepath)
        for row in parsed_facts:
            already_parsed_facts.add(row['metadata']['fact'])
        logger.info(f"Loaded {len(already_parsed_facts)} already parsed facts from {parsed_facts_filepath}")

    # Load preprocessed facts to skip if they exist
    if args.preprocessed_facts_to_skip_filepaths is not None:
        for filepath in args.preprocessed_facts_to_skip_filepaths:
            assert os.path.exists(filepath), f"File {filepath} does not exist"
            facts_to_skip = load_jsonl(filepath)
            logger.info(f"Loaded {len(facts_to_skip)} facts to skip from {filepath}")
            for row in facts_to_skip:
                if 'fact' in row:
                    already_parsed_facts.add(row['fact'])
                else:
                    already_parsed_facts.add(row['metadata']['fact'])
            logger.info(f"Total number of facts to skip: {len(already_parsed_facts)}")

    # Collect facts from input file
    unique_facts = set()
    logger.info(f"Loading facts from {args.extracted_facts_jsonl_filepath}")
    rows = load_jsonl(args.extracted_facts_jsonl_filepath)
    logger.info(f"Loaded {len(rows)} rows from {args.extracted_facts_jsonl_filepath}")
    for row in rows:
        for f in row['facts']:
            unique_facts.add(f)
    
    logger.info(f"Found {len(unique_facts)} unique facts in {args.extracted_facts_jsonl_filepath}")
    assert len(unique_facts) > 0
    
    # Sort facts
    unique_facts = list(unique_facts)
    unique_facts = sort_facts(unique_facts)

    # Adjust number of facts to parse if necessary
    assert 0 <= args.offset < len(unique_facts)
    if args.offset + args.num_facts > len(unique_facts):
        logger.warning(f"Requested {args.num_facts} facts but only {len(unique_facts) - args.offset} are available."
                       f" Using {len(unique_facts) - args.offset} instead.")
        args.num_facts = len(unique_facts) - args.offset
        assert args.num_facts > 0

    # Sample facts to parse (if requested)
    if args.sample_facts_uniformly:
        indices = random.sample(range(args.offset, len(unique_facts)), args.num_facts)
    else: # First arg.num_facts facts
        indices = [i for i in range(args.offset, args.offset + args.num_facts)]

    # Remove already parsed facts
    facts_to_parse = [unique_facts[i] for i in indices if unique_facts[i] not in already_parsed_facts]
    if len(facts_to_parse) == 0:
        logger.info(f"All {len(indices)} facts have already been parsed. Nothing to do. Exiting.")
        sys.exit(0)

    logger.info(f"Total number of facts to parse: {len(facts_to_parse)}")
    
    # Print example facts to parse
    logger.info(f"Example facts to parse:")
    for i in np.linspace(0, len(facts_to_parse)-1, min(10, len(facts_to_parse)), dtype=int):
        logger.info(f"{i+1}. {facts_to_parse[i]}")

    # Prepare API requests
    jobs = []
    for fact in facts_to_parse:
        jobs.append(generate_request(
            fact=fact,
            model_name=args.openai_model_name,
            max_tokens=args.max_tokens_per_request,
        ))
        assert 'metadata' in jobs[-1]
        assert jobs[-1]['metadata']['fact'] == fact
    
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
        log_info_every_n_requests=20,
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
            logger.error(f"Error parsing response {api_response_string} for fact \"{metadata['fact']}\": {e}")
            continue

    # Delete API requests and responses
    logger.info(f"Deleting API requests and responses")
    os.remove(api_requests_filepath)
    os.remove(api_responses_filepath)

    if len(postprocessed_responses) == 0:
        logger.warning(f"None of the {len(api_responses)} API responses could be parsed. Exiting.")
    else:
        # Save parsed facts by appending to existing file
        n_parsed = len(postprocessed_responses)
        n_total = len(api_responses)
        logger.info(f"""Succesfully processed {n_parsed} of {n_total} API responses.
                    {n_total - n_parsed} of {n_total} API responses could not be processed.
                    Saving parsed facts to {parsed_facts_filepath}""")
        save_jsonl(postprocessed_responses, parsed_facts_filepath, append=True)