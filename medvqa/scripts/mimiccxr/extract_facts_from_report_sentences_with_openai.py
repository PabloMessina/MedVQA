from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import logging
import tiktoken
import re
import sys
import json
import numpy as np

from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.openai_api import process_api_requests_from_file
from medvqa.utils.files import load_jsonl, save_jsonl
from medvqa.utils.common import get_timestamp

INSTRUCTIONS = """Given a sentence from a chest x-ray report, generate a JSON list of strings.
Each string must be a fact extracted from the sentence. Facts must go from general to specific and should only include
positive observations of abnormalities, diseases, strange visual patterns, devices, and foreign bodies (observable things
that are meaningful for a radiologist). Negative facts (absences) or facts describing normal or healthy appearances should be excluded.
Facts should be stated as short phrases, almost like labels, and should avoid unnecessary verbosity.

Anatomical locations should not be standalone facts but can be included as descriptors within a fact about an observation.
For example, "Lower cervical spine" is wrong, but "Metallic hardware in the lower cervical spine" is fine.

If no positive facts meeting the criteria are found, return an empty list.

Easy examples:

Extensive pleural calcification is noted
[
"Calcification",
"Pleural calcification"
]

Support lines and tubes are unchanged
[
"Lines",
"Tubes",
"Support lines",
"Support tubes",
]

New left basilar opacity and small to moderate size left pleural effusion
[
"Opacity",
"Basilar opacity",
"Left basilar opacity",
"Pleural effusion",
"Left pleural effusion",
"Small to moderate size left pleural effusion"
]

Little overall change in interstitial appearance in right lung
[
"Interstitial appearance",
"Interstitial appearance in right lung"
]

Bilateral heterogeneous consolidations concerning for multi focal pneumonia
[
"Consolidations",
"Heterogeneous consolidations",
"Bilateral consolidations",
"Pneumonia",
"Multi focal pneumonia"
]

Examples of empty list (no positive observations)

Clearing of both bases
[]

Manometry device has been removed
[]

Unremarkable cardiac and mediastinal silhouettes
[]

No evidence of consolidation or pleural effusion
[]

No evidence of pneumothorax
[]

Normal cardiac silhouette
[]

Complex example (combining negative and positive facts, the negative is excluded):

No evidence of CHF except mild prominence of the cardiomediastinal silhouette
[
"Prominence of the cardiomediastinal silhouette",
"Mild prominence of the cardiomediastinal silhouette"
]"""

ALLOWED_GPT_CHAT_MODELS = ("gpt-3.5-turbo", "gpt-4", "gpt-4-0613")

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

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = re.search(r"\[\s*(\".+?\",?\s*)*\]", text) # match a JSON list of strings
    assert match, f"Could not parse output: {text}"
    list_of_facts = json.loads(match.group(0))
    assert isinstance(list_of_facts, list), f"Could not parse output: {text}"
    assert all(isinstance(fact, str) for fact in list_of_facts), f"Could not parse output: {text}"
    return list_of_facts

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_sentences_jsonl_filepaths", nargs="+", required=True)
    parser.add_argument("--preprocessed_sentences_to_skip_filename", type=str, default=None)
    parser.add_argument("--num_sentences", type=int, required=True)
    parser.add_argument("--sample_sentences_uniformly", action="store_true", default=False)
    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    # Set up logging
    logger = logging.getLogger()
    logging_level = logging.getLevelName(args.logging_level)
    logger.setLevel(logging_level)
    # configure a different color for each level
    logging.addLevelName(logging.DEBUG, "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.CRITICAL, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.CRITICAL))
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    # Load parsed sentences if they exist
    parsed_sentences_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_parsed_sentences.jsonl")
    already_parsed_sentences = set()
    if os.path.exists(parsed_sentences_filepath):
        parsed_sentences = load_jsonl(parsed_sentences_filepath)
        for row in parsed_sentences:
            already_parsed_sentences.add(row['metadata']['sentence'])
        logger.info(f"Loaded {len(already_parsed_sentences)} already parsed sentences from {parsed_sentences_filepath}")

    # Load preprocessed sentences to skip if they exist
    if args.preprocessed_sentences_to_skip_filename is not None:
        preprocessed_sentences_to_skip_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", args.preprocessed_sentences_to_skip_filename)
        assert os.path.exists(preprocessed_sentences_to_skip_filepath)
        logger.info(f"Loading preprocessed sentences to skip from {preprocessed_sentences_to_skip_filepath}")
        sentences_to_skip = load_jsonl(preprocessed_sentences_to_skip_filepath)
        logger.info(f"Loaded {len(sentences_to_skip)} sentences to skip")
        for row in sentences_to_skip:
            already_parsed_sentences.add(row['metadata']['sentence'])
        logger.info(f"Total number of sentences to skip: {len(already_parsed_sentences)}")

    # Collect unparsed sentences from input files
    assert len(args.extracted_sentences_jsonl_filepaths) > 0
    sentences_to_parse = set()
    for filepath in args.extracted_sentences_jsonl_filepaths:
        logger.info(f"Loading sentences from {filepath}")
        rows = load_jsonl(filepath)
        logger.info(f"Loaded {len(rows)} reports with extracted sentences from {filepath}")
        for row in rows:
            for s, _ in row['parsed_response']:
                if s not in already_parsed_sentences:
                    sentences_to_parse.add(s)
    logger.info(f"Found {len(sentences_to_parse)} unique sentences to parse")
    if len(sentences_to_parse) == 0:
        logger.info("Nothing to do. Exiting.")
        sys.exit(0)

    sentences_to_parse = list(sentences_to_parse)
    sentences_to_parse.sort(key=lambda s: len(s), reverse=True) # prioritize longer sentences

    if args.num_sentences > len(sentences_to_parse):
        logger.warning(f"Requested {args.num_sentences} sentences but only {len(sentences_to_parse)} are available. Using {len(sentences_to_parse)} sentences.")
        args.num_sentences = len(sentences_to_parse)
    
    # Print example sentences
    logger.info(f"Example sentences to parse:")
    # First 5 sentences
    for i in range(min(args.num_sentences, 5)):
        logger.info(f"{i+1}. {sentences_to_parse[i]}")
    # Last 5 sentences
    for i in range(max(5, args.num_sentences-5), args.num_sentences):
        logger.info(f"{i+1}. {sentences_to_parse[i]}")

    # Prepare API requests
    detailed_metadata = load_mimiccxr_reports_detailed_metadata()
    jobs = []
    if args.sample_sentences_uniformly:
        indices = np.linspace(0, len(sentences_to_parse)-1, args.num_sentences, dtype=int)
    else: # First arg.num_sentences sentences
        indices = [i for i in range(args.num_sentences)]

    for i in indices:
        sentence = sentences_to_parse[i]
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
        logging_level=logging_level,
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
            logger.error(f"Error parsing response {api_response_string} for sentence \"{metadata['sentence']}\": {e}")
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
                    Saving parsed sentences to {parsed_sentences_filepath}""")
        save_jsonl(postprocessed_responses, parsed_sentences_filepath, append=True)