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

from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, process_api_requests_from_file
from medvqa.utils.files_utils import load_jsonl, save_jsonl
from medvqa.utils.common import get_timestamp

INSTRUCTIONS = """Given a sentence referring to an anatomical location in the context of a chest X-ray, output a JSON array of strings paraphrasing it,
covering a wide diverse range of terminology, synonyms and abbreviations that radiologists commonly use to express the same idea in a chest X-ray report.

Examples:

gastroesophageal junction
[
"cardia",
"cardia of the stomach",
"cardiac orifice",
"esophagogastric junction",
"GE junction",
"esophagogastro junction",
"esophagogastric junctional region",
"gastroesophageal interface",
"gastroesophageal transition zone",
"lower esophageal junction",
"esophagogastric transition",
"stomach-esophagus interface",
"junction between stomach and esophagus",
"cardioesophageal junction",
"junction of the esophagus and stomach",
"transition between the esophagus and stomach",
"point where the esophagus meets the stomach",
"Z-line",
"distal end of the esophagus",
"proximal start of the gastric cardia"
]

costodiaphragmatic recess
[
"costophrenic angle",
"costophrenic recess",
"costophrenic sulcus",
"costophrenic notch",
"costophrenic sinus",
"phrenicocostal sinus",
"diaphragm-chest wall interface",
"junction of diaphragm and chest wall",
"area where the diaphragm meets the ribs",
"pleural reflection at the diaphragm and rib cage",
"costal margin of the diaphragm",
"angle formed by diaphragm and rib cage"
]

R>L
[
"right greater than left"
]

base
[
"lung base",
"base of the lung",
"lower lung area",
"basal lung region",
"basal portion of the lung",
"lung's basal part",
"bottom part of the lung",
"lower region of the lung",
"pulmonary base",
"base of the pulmonary structure",
"lower zone of the lung",
"lower pulmonary region",
"lower segment of the lung"
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
                "anatomical location": sentence,
            },
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# Match a possibly truncated JSON list of strings, by matching as many strings as possible.
# This is useful because the OpenAI API sometimes truncates the output.
_JSON_STRING_ARRAY_REGEX = re.compile(r'^\[\s*(\".+?\"(\s*,\s*\".+?\")*)?\s*\]?')

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = _JSON_STRING_ARRAY_REGEX.search(text) # match a JSON list of strings
    if not match and GPT_IS_ACTING_WEIRD_REGEX.search(text):
        logger.warning(f"GPT is protesting: {text}")
    assert match, f"Could not parse output: {text}"
    string = match.group(0)
    assert string[0] == "[", f"Could not parse output: {text}"
    if string[-1] != "]": string += "]" # add closing bracket
    list_of_facts = json.loads(string)
    assert isinstance(list_of_facts, list), f"Could not parse output: {text}"
    assert all(isinstance(fact, str) for fact in list_of_facts), f"Could not parse output: {text}"
    return list_of_facts

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated_fact_metadata_filepath", type=str, default=None)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_sentences", type=int, required=True)
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
    paraphrased_anatomical_locations_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_paraphrased_anatomical_locations{args.alias}.jsonl")
    already_paraphrased = set()
    if os.path.exists(paraphrased_anatomical_locations_filepath):
        rows = load_jsonl(paraphrased_anatomical_locations_filepath)
        for row in rows:
            already_paraphrased.add(row['metadata']['anatomical location'])
        logger.info(f"Loaded {len(already_paraphrased)} already paraphrased sentences from {paraphrased_anatomical_locations_filepath}")
    if args.preprocessed_sentences_to_skip_filepaths is not None:
        for filepath in args.preprocessed_sentences_to_skip_filepaths:
            assert os.path.exists(filepath)
            logger.info(f"Loading paraphrased sentences to skip from {filepath}")
            rows = load_jsonl(filepath)
            for row in rows:
                already_paraphrased.add(row['metadata']['anatomical location'])
            logger.info(f"Loaded {len(already_paraphrased)} already paraphrased sentences from {filepath}")

    # Collect anatomical locations from metadata
    unique_anatomical_locations = set()
    assert os.path.exists(args.integrated_fact_metadata_filepath)
    logger.info(f"Loading facts metadata from {args.integrated_fact_metadata_filepath}")
    integrated_fact_metadata = load_jsonl(args.integrated_fact_metadata_filepath)
    for r in tqdm(integrated_fact_metadata, total=len(integrated_fact_metadata), mininterval=2):
        anat_loc = r['metadata']['anatomical location']
        if len(anat_loc) > 0 and any(c.isalpha() for c in anat_loc):
            unique_anatomical_locations.add(anat_loc)
    
    logger.info(f"Found {len(unique_anatomical_locations)} unique anatomical locations")
    assert len(unique_anatomical_locations) > 0

    # Sort anatomical locations
    unique_anatomical_locations = list(unique_anatomical_locations)
    unique_anatomical_locations.sort(key=lambda s: (len(s), s)) # sort by length and then alphabetically

    # Adjust number of sentences to paraphrase if necessary
    assert 0 <= args.offset < len(unique_anatomical_locations)
    if args.offset + args.num_sentences > len(unique_anatomical_locations):
        logger.warning(f"Requested {args.num_sentences} sentences but only {len(unique_anatomical_locations) - args.offset} are available."
                       f" Using {len(unique_anatomical_locations) - args.offset} instead.")
        args.num_sentences = len(unique_anatomical_locations) - args.offset
        assert args.num_sentences > 0

    # Remove already paraphrased sentences
    sentences_to_paraphrase = [unique_anatomical_locations[i] for i in range(args.offset, args.offset + args.num_sentences) if\
                               unique_anatomical_locations[i] not in already_paraphrased]
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
        assert jobs[-1]['metadata']['anatomical location'] == sentence
    
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
            logger.error(f"Error parsing response {api_response_string} for sentence \"{metadata['anatomical location']}\": {e}")
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
                    Saving paraphrased anatomical locations to {paraphrased_anatomical_locations_filepath}""")
        save_jsonl(postprocessed_responses, paraphrased_anatomical_locations_filepath, append=True)