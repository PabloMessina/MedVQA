from dotenv import load_dotenv

from medvqa.utils.logging import get_console_logger
load_dotenv()

import random
import os
import argparse
import tiktoken
import re
import sys
import json
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize

from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api import process_api_requests_from_file
from medvqa.utils.files import load_json, load_jsonl, save_jsonl
from medvqa.utils.common import get_timestamp

INSTRUCTIONS = """Relevant facts:

1. observations of abnormalities
2. observations of diseases
3. observations of strange visual patterns
4. observations of devices
5. observations of foreign bodies
6. observations of specific anatomical regions that look normal or healthy
7. absences of abnormalities (usually expressed with a negation)
8. comparisons with respect to a previous study (something changed or remained the same)

Task:

Given a sentence taken from a chest x-ray report, generate a JSON list of relevant facts.
Each fact should be about one observation. If a sentence mentions multiple observations,
each observation should be extracted as a separate fact.
Each fact should include the anatomical location where it was observed. If multiple facts
occur in the same location, repeat the location in each fact.

If no relevant facts are mentioned, return [] (an empty array).

Examples:

Opacity and density in the right lobe
[
"opacity in the right lobe",
"density in the right lobe"
]

Lungs are well inflated without evidence of focal airspace consolidation to suggest pneumonia.
[
"well inflated lungs",
"lungs without evidence of focal airspace consolidation",
"lungs without evidence of pneumonia"
]

Taken together, compared with less than 1 hr earlier, the findings are suggestive of worsening of CHF, with new or significantly increased left greater right pleural effusions and underlying bibasilar collapse and/or  consolidation, particularly on the left.
[
"worsening of CHF",
"new or significantly increased left pleural effusions",
"new or significantly increased right pleural effusions",
"underlying bibasilar collapse on the left",
"underlying consolidation on the left",
]

No acute cardiopulmonary abnormality
[
"no acute cardiopulmonary abnormality"
]"""

ALLOWED_GPT_CHAT_MODELS = ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-0613")

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

_JSON_STRING_ARRAY_REGEX = re.compile(r"\[\s*(\".+?\",?\s*)*\]")
_GPT_IS_PROTESTING_REGEX = re.compile(r"\b(I'm sorry|Sorry|Could you|Can you|Please|please)\b")

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = _JSON_STRING_ARRAY_REGEX.search(text) # match a JSON list of strings
    if not match:
        if _GPT_IS_PROTESTING_REGEX.search(text):
            logger.warning(f"GPT is protesting: {text}, returning []")
            return []
    assert match, f"Could not parse output: {text}"
    list_of_facts = json.loads(match.group(0))
    assert isinstance(list_of_facts, list), f"Could not parse output: {text}"
    assert all(isinstance(fact, str) for fact in list_of_facts), f"Could not parse output: {text}"
    return list_of_facts

def sort_sentences(sentences, by_difficulty=False):
    assert type(sentences) == list, f"Expected list, got {type(sentences)}"
    if by_difficulty:
        logger.info("Sorting sentences by difficulty...")
        tokenized_sentences = [word_tokenize(x) for x in tqdm(sentences, mininterval=2)]
        logger.info("Counting word frequencies...")
        vocab_freq = dict()        
        for tokens in tqdm(tokenized_sentences, mininterval=2):
            for word in tokens:
                vocab_freq[word] = vocab_freq.get(word, 0) + 1
        def _difficulty(i):
            return sum(1 / vocab_freq[word] for word in tokenized_sentences[i])
        ranked_indices = sorted(range(len(tokenized_sentences)), key=_difficulty, reverse=True)
        ranked_sentences = [sentences[i] for i in ranked_indices]
    else:
        logger.info("Sorting sentences by length...")
        ranked_sentences = sorted(sentences, key=len, reverse=True)
    logger.info("Done sorting sentences.")
    logger.info(f"First sentence: {ranked_sentences[0]}")
    logger.info(f"Last sentence: {ranked_sentences[-1]}")
    return ranked_sentences

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_sentences_to_skip_filenames", nargs="+", default=None)
    parser.add_argument("--preprocessed_reports_filename", type=str, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_sentences", type=int, required=True)
    parser.add_argument("--rank_sentences_by_difficulty", action="store_true", default=False)
    parser.add_argument("--sample_sentences_uniformly", action="store_true", default=False)
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

    # Load parsed sentences if they exist
    parsed_sentences_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_parsed_sentences{args.alias}.jsonl")
    already_parsed_sentences = set()
    if os.path.exists(parsed_sentences_filepath):
        parsed_sentences = load_jsonl(parsed_sentences_filepath)
        for row in parsed_sentences:
            already_parsed_sentences.add(row['metadata']['sentence'])
        logger.info(f"Loaded {len(already_parsed_sentences)} already parsed sentences from {parsed_sentences_filepath}")

    # Load preprocessed sentences to skip if they exist
    if args.preprocessed_sentences_to_skip_filenames is not None:
        for filename in args.preprocessed_sentences_to_skip_filenames:
            preprocessed_sentences_to_skip_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", filename)
            assert os.path.exists(preprocessed_sentences_to_skip_filepath)
            logger.info(f"Loading preprocessed sentences to skip from {preprocessed_sentences_to_skip_filepath}")
            sentences_to_skip = load_jsonl(preprocessed_sentences_to_skip_filepath)
            logger.info(f"Loaded {len(sentences_to_skip)} sentences to skip")
            for row in sentences_to_skip:
                already_parsed_sentences.add(row['metadata']['sentence'])
            logger.info(f"Total number of sentences to skip: {len(already_parsed_sentences)}")

    # Collect unparsed sentences from reports
    unique_sentences = set()
    preprocessed_reports_filepath = os.path.join(MIMICCXR_CACHE_DIR, args.preprocessed_reports_filename)
    assert os.path.exists(preprocessed_reports_filepath)
    logger.info(f"Loading preprocessed reports from {preprocessed_reports_filepath}")
    reports = load_json(preprocessed_reports_filepath)
    for r in tqdm(reports, total=len(reports), mininterval=2):
        impression = r['impression']
        findings = r['findings']
        if len(impression) > 0:
            for s in sent_tokenize(impression):
                unique_sentences.add(s)
        if len(findings) > 0:
            for s in sent_tokenize(findings):
                unique_sentences.add(s)
    logger.info(f"Loaded {len(reports)} reports from {preprocessed_reports_filepath}")
    
    logger.info(f"Found {len(unique_sentences)} unique sentences to parse")
    assert len(unique_sentences) > 0

    # Sort sentences
    unique_sentences = list(unique_sentences)
    unique_sentences = sort_sentences(unique_sentences, args.rank_sentences_by_difficulty)

    # Adjust number of sentences to parse if necessary
    assert 0 <= args.offset < len(unique_sentences)
    if args.offset + args.num_sentences > len(unique_sentences):
        logger.warning(f"Requested {args.num_sentences} sentences but only {len(unique_sentences) - args.offset} are available."
                       f" Using {len(unique_sentences) - args.offset} instead.")
        args.num_sentences = len(unique_sentences) - args.offset
        assert args.num_sentences > 0

    # Sample sentences to parse (if requested)
    if args.sample_sentences_uniformly:
        indices = random.sample(range(args.offset, len(unique_sentences)), args.num_sentences)
    else: # First arg.num_sentences sentences
        indices = [i for i in range(args.offset, args.offset + args.num_sentences)]

    # Remove already parsed sentences
    sentences_to_parse = [unique_sentences[i] for i in indices if unique_sentences[i] not in already_parsed_sentences]
    if len(sentences_to_parse) == 0:
        logger.info(f"All {len(indices)} sentences have already been parsed. Nothing to do. Exiting.")
        sys.exit(0)

    logger.info(f"Total number of sentences to parse: {len(sentences_to_parse)}")    
    
    # Print example sentences
    logger.info(f"Example sentences to parse:")
    for i in np.linspace(0, len(sentences_to_parse)-1, min(10, len(sentences_to_parse)), dtype=int):
        logger.info(f"{i+1}. {sentences_to_parse[i]}")

    # Prepare API requests
    jobs = []
    for sentence in sentences_to_parse:
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