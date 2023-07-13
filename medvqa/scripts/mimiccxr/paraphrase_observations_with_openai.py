from dotenv import load_dotenv

from medvqa.utils.logging import get_console_logger
load_dotenv()

import os
import argparse
import tiktoken
import re
import sys
import json
import numpy as np
from tqdm import tqdm

from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api import process_api_requests_from_file
from medvqa.utils.files import load_jsonl, save_jsonl
from medvqa.utils.common import get_timestamp

INSTRUCTIONS = """Given a sentence describing a fact from a chest X-ray report, output a JSON array of strings paraphrasing it,
covering a wide diverse range of terminology, synonyms and abbreviations that radiologists commonly use to express the same idea.

Examples:

benign calcification
[
"non-cancerous calcification",
"harmless calcification",
"innocuous calcification",
"benign calcified lesion",
"non-malignant calcification",
"non-threatening calcification",
"not indicative of cancer calcification",
"safe calcification",
"non-dangerous calcification",
"non-metastatic calcification"
]

osteoporosis
[
"decreased bone density",
"brittle bones",
"low bone mass",
"thinning of the bones",
"weakening of the bones",
"porous bones",
"fragile bones",
"reduced bone strength",
"loss of bone density",
"degenerative bone disease"
]

no osteoporosis
[
"normal bone density",
"healthy bones",
"adequate bone mass",
"strong bones",
"normal bone strength",
"no signs of osteoporosis",
"absence of osteoporosis",
"no evidence of bone thinning",
"no indication of bone weakening",
"no osteoporotic changes",
"no degenerative bone disease"
]"""

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
            "frequency_penalty": 0.2, # 0.0 = no penalty, 2.0 = max penalty
            "presence_penalty": 0.2, # 0.0 = no penalty, 2.0 = max penalty
            "max_tokens": max_tokens,
            "metadata": {
                "observation": sentence,
            },
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# Match a possibly truncated JSON list of strings, by matching as many strings as possible.
# This is useful because the OpenAI API sometimes truncates the output.
_JSON_STRING_ARRAY_REGEX = re.compile(r'^\[\s*(\".+?\"(\s*,\s*\".+?\")*)?\s*\]?')

_GPT_IS_PROTESTING_REGEX = re.compile(r"\b(I'm sorry|Sorry|Could you|Can you|Please|please)\b")

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = _JSON_STRING_ARRAY_REGEX.search(text) # match a JSON list of strings
    if not match and _GPT_IS_PROTESTING_REGEX.search(text):
        logger.warning(f"GPT is protesting: {text}")
    assert match, f"Could not parse output: {text}"
    string = match.group(0)
    assert string[0] == "[", f"Could not parse output: {text}"
    if string[-1] != "]": string += "]" # add closing bracket
    list_of_strings = json.loads(string)
    assert isinstance(list_of_strings, list), f"Could not parse output: {text}"
    assert all(isinstance(fact, str) for fact in list_of_strings), f"Could not parse output: {text}"
    return list_of_strings

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated_fact_metadata_filepath", type=str, default=None)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)    
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num_words_per_sentence", type=int, default=None)
    group.add_argument("--min_num_words_per_sentence", type=int, default=None)

    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_sentences", type=int, required=True)
    parser.add_argument("--process_kth_of_every_n_sentences", type=int, nargs=2, default=None,
                        help="If specified, only process the kth of every n sentences.")
    parser.add_argument("--sample_equally_spaced_sentences", action="store_true",
                        help="If specified, sample sentences equally spaced apart.")

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

    # Load already paraphrased sentences if they exist
    paraphrased_observations_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_paraphrased_observations{args.alias}.jsonl")
    already_paraphrased = set()
    if os.path.exists(paraphrased_observations_filepath):
        rows = load_jsonl(paraphrased_observations_filepath)
        for row in rows:
            already_paraphrased.add(row['metadata']['observation'])
        logger.info(f"Loaded {len(already_paraphrased)} already paraphrased sentences from {paraphrased_observations_filepath}")
    if args.preprocessed_sentences_to_skip_filepaths is not None:
        for filepath in args.preprocessed_sentences_to_skip_filepaths:
            assert os.path.exists(filepath)
            logger.info(f"Loading paraphrased sentences to skip from {filepath}")
            rows = load_jsonl(filepath)
            for row in rows:
                already_paraphrased.add(row['metadata']['observation'])
            logger.info(f"Loaded {len(already_paraphrased)} already paraphrased sentences from {filepath}")

    # Collect observations from metadata
    unique_observations = set()
    assert os.path.exists(args.integrated_fact_metadata_filepath)
    logger.info(f"Loading facts metadata from {args.integrated_fact_metadata_filepath}")
    integrated_fact_metadata = load_jsonl(args.integrated_fact_metadata_filepath)
    for r in tqdm(integrated_fact_metadata, total=len(integrated_fact_metadata), mininterval=2):
        fact = r['fact']
        detailed_observation = r['metadata']['detailed observation']
        short_observation = r['metadata']['short observation']
        for x in (fact, detailed_observation, short_observation):
            if len(x) > 0 and any(c.isalpha() for c in x):
                unique_observations.add(x)
    
    logger.info(f"Found {len(unique_observations)} unique observations")
    assert len(unique_observations) > 0

    # Sort observations
    unique_observations = list(unique_observations)
    unique_observations.sort(key=lambda s: (len(s), s)) # sort by length and then alphabetically

    # Filter observations by length if necessary
    if args.num_words_per_sentence is not None:
        logger.info(f"Filtering observations to those with {args.num_words_per_sentence} words")
        unique_observations = [x for x in unique_observations if len(x.split()) == args.num_words_per_sentence]
        logger.info(f"Found {len(unique_observations)} observations with {args.num_words_per_sentence} words")
    
    if args.min_num_words_per_sentence is not None:
        logger.info(f"Filtering observations to those with at least {args.min_num_words_per_sentence} words")
        unique_observations = [x for x in unique_observations if len(x.split()) >= args.min_num_words_per_sentence]
        logger.info(f"Found {len(unique_observations)} observations with at least {args.min_num_words_per_sentence} words")

    # Filter observations by kth of every n sentences if necessary
    if args.process_kth_of_every_n_sentences is not None:
        k, n = args.process_kth_of_every_n_sentences
        assert 0 <= k < n
        logger.info(f"Filtering observations to the {k}-th of every {n} sentences")
        unique_observations = [x for i, x in enumerate(unique_observations) if i % n == k]
        logger.info(f"Found {len(unique_observations)} observations that are the {k}-th of every {n} sentences")

    # Adjust number of sentences to paraphrase if necessary
    assert 0 <= args.offset < len(unique_observations)
    if args.offset + args.num_sentences > len(unique_observations):
        logger.warning(f"Requested {args.num_sentences} sentences but only {len(unique_observations) - args.offset} are available."
                       f" Using {len(unique_observations) - args.offset} instead.")
        args.num_sentences = len(unique_observations) - args.offset
        assert args.num_sentences > 0

    # Collect sentences to paraphrase
    if args.sample_equally_spaced_sentences:
        logger.info(f"Sampling {args.num_sentences} equally spaced sentences")
        sentences_to_paraphrase = [unique_observations[i] for i in np.linspace(args.offset, len(unique_observations)-1, args.num_sentences, dtype=int)
                                   if unique_observations[i] not in already_paraphrased]
    else:
        logger.info(f"Sampling the first {args.num_sentences} sentences")
        sentences_to_paraphrase = [unique_observations[i] for i in range(args.offset, args.offset + args.num_sentences)
                                   if unique_observations[i] not in already_paraphrased]
    if len(sentences_to_paraphrase) == 0:
        logger.info(f"All {args.num_sentences} sentences have already been paraphrased. Nothing to do. Exiting.")
        sys.exit(0)

    logger.info(f"Total number of sentences to paraphrase: {len(sentences_to_paraphrase)}")
    
    # Print example sentences
    logger.info(f"Example sentences to paraphrase:")
    for i in np.linspace(0, len(sentences_to_paraphrase)-1, min(10, len(sentences_to_paraphrase)), dtype=int):
        logger.info(f"{i+1}. {sentences_to_paraphrase[i]}")
    
    # Prepare API requests
    jobs = []
    for sentence in sentences_to_paraphrase:
        jobs.append(generate_request(
            sentence=sentence,
            model_name=args.openai_model_name,
            max_tokens=args.max_tokens_per_request,
        ))
        assert 'metadata' in jobs[-1]
        assert jobs[-1]['metadata']['observation'] == sentence
    
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
            logger.error(f"Error parsing response {api_response_string} for sentence \"{metadata['observation']}\": {e}")
            continue

    # Delete API requests and responses
    logger.info(f"Deleting API requests and responses")
    os.remove(api_requests_filepath)
    os.remove(api_responses_filepath)

    if len(postprocessed_responses) == 0:
        logger.warning(f"None of the {len(api_responses)} API responses could be parsed. Exiting.")
    else:
        # Save parsed sentences by appending to existing file
        n_parsed = len(postprocessed_responses)
        n_total = len(api_responses)
        logger.info(f"""Succesfully processed {n_parsed} of {n_total} API responses.
                    {n_total - n_parsed} of {n_total} API responses could not be processed.
                    Saving paraphrased observations to {paraphrased_observations_filepath}""")
        save_jsonl(postprocessed_responses, paraphrased_observations_filepath, append=True)