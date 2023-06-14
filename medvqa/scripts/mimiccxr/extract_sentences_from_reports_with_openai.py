from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import logging
import tiktoken
import re
import sys
import json

from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_FAST_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.openai_api import process_api_requests_from_file
from medvqa.utils.files import load_json, load_jsonl, load_pickle, save_jsonl
from medvqa.utils.common import get_timestamp

INSTRUCTIONS = """Definitions:

Positive facts are observations of abnormalities, diseases, strange visual patterns, devices, and foreign bodies (observable things
that are meaningful for a radiologist).

Negative facts are absences (lack of something) or descriptions denoting normal or healthy appearances.

Task:

Convert the following radiological report into a JSON list of sentences, summarizing the essential information. Avoid long verbose sentences. Prefer short sentences.

If a sentence includes at least one positive fact, append the hashtag #pos. Otherwise, append the hashtag #neg.

Example of output:
[
"Cardiomegaly #pos",
"Wires are noted #pos",
"Tube is unchanged #pos",
"Normal heart size #neg",
"No pleural effusion #neg",
"Little change in air-fluid level #pos"
]"""

ALLOWED_GPT_CHAT_MODELS = ("gpt-3.5-turbo", "gpt-4", "gpt-4-0613")

def generate_request(report, report_idx, part_id, subject_id, study_id, model_name, max_tokens, temperature=0.0):
    report_sections = []
    if report['findings']:
        report_sections.append(report['findings'])
    if report['impression']:
        report_sections.append(report['impression'])
    input_text = "\n".join(report_sections)
    assert len(input_text) > 0, f"Could not find any text in report: {report}, report_idx: {report_idx}"

    if model_name in ALLOWED_GPT_CHAT_MODELS:
        return {
            "model": model_name,
            "messages": [{
                "role": "system",
                "content": INSTRUCTIONS,
            }, {
                "role": "user",
                "content": input_text,
            }],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "metadata": {
                "report_index": report_idx,
                "part_id": part_id,
                "subject_id": subject_id,
                "study_id": study_id,
                "report": report,
            }
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = re.search(r"\[\s*(\".+?\",?\s*)*\]", text) # match a JSON list of strings
    assert match, f"Could not parse output: {text}"
    sentences = json.loads(match.group(0))
    assert isinstance(sentences, list), f"Could not parse output: {text}"
    assert all(isinstance(s, str) for s in sentences), f"Could not parse output: {text}"
    assert all(re.match(r"^.+\s#(pos|neg)$", s) for s in sentences), f"Could not parse output: {text}"
    assert len(sentences) > 0, f"Empty output: {text}"
    pairs = [(s[:-4].strip(), 1 if s[-4:] == "#pos" else 0) for s in sentences] # each pair is (sentence, label)
    assert all(len(s) > 0 for s, _ in pairs), f"Empty sentence in output: {text}"
    return pairs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_reports_filename", type=str, required=True)
    parser.add_argument("--ranked_report_indices_filename", type=str, required=True)
    parser.add_argument("--preprocessed_reports_to_skip_filename", type=str, default=None)
    parser.add_argument("--num_reports", type=int, required=True)
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

    # Load preprocessed reports and ranked report indices
    preprocessed_reports_filepath = os.path.join(MIMICCXR_CACHE_DIR, args.preprocessed_reports_filename)
    logger.info(f"Loading preprocessed reports from {preprocessed_reports_filepath}")
    reports = load_json(preprocessed_reports_filepath)
    
    ranked_report_indices_filepath = os.path.join(MIMICCXR_CACHE_DIR, args.ranked_report_indices_filename)
    logger.info(f"Loading ranked report indices from {ranked_report_indices_filepath}")
    ranked_report_indices = load_pickle(ranked_report_indices_filepath)
    
    assert len(reports) == len(ranked_report_indices)
    assert 0 < args.num_reports <= len(reports)

    # Load parsed reports if they exist
    parsed_reports_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_parsed_reports.jsonl")
    parsed_report_indices = set()
    if os.path.exists(parsed_reports_filepath):
        parsed_reports = load_jsonl(parsed_reports_filepath)
        # determine which reports have already been parsed
        for parsed_report in parsed_reports:
            parsed_report_indices.add(parsed_report['metadata']['report_index'])
        logger.info(f"Loaded {len(parsed_report_indices)} parsed reports from {parsed_reports_filepath}")

    if args.preprocessed_reports_to_skip_filename is not None:
        preprocessed_reports_to_skip_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", args.preprocessed_reports_to_skip_filename)
        assert os.path.exists(preprocessed_reports_to_skip_filepath)
        logger.info(f"Loading preprocessed reports to skip from {preprocessed_reports_to_skip_filepath}")
        reports_to_skip = load_jsonl(preprocessed_reports_to_skip_filepath)
        logger.info(f"Loaded {len(reports_to_skip)} reports to skip")
        for report_to_skip in reports_to_skip:
            parsed_report_indices.add(report_to_skip['metadata']['report_index'])
        logger.info(f"Total number of reports to skip: {len(parsed_report_indices)}")

    # Prepare API requests
    detailed_metadata = load_mimiccxr_reports_detailed_metadata()
    jobs = []
    for i in range(args.num_reports):
        if ranked_report_indices[i] in parsed_report_indices:
            continue
        ridx = ranked_report_indices[i]
        report = reports[ridx]
        jobs.append(generate_request(
            report=report,
            report_idx=ridx,
            part_id=detailed_metadata['part_ids'][ridx],
            subject_id=detailed_metadata['subject_ids'][ridx],
            study_id=detailed_metadata['study_ids'][ridx],
            model_name=args.openai_model_name,
            max_tokens=args.max_tokens_per_request,
        ))
        assert 'metadata' in jobs[-1]
        assert jobs[-1]['metadata']['report_index'] == ridx

    if len(jobs) == 0:
        logger.info(f"All of the top {args.num_reports} reports were found in the cache. Exiting.")
        sys.exit(0)
    else:
        logger.info(f"{len(jobs)} of the top {args.num_reports} reports were not found in the cache. Parsing them now.")
    
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
            logger.error(f"Error parsing response {api_response_string} for report {metadata['report_index']}: {e}")
            continue

    # Delete API requests and responses
    logger.info(f"Deleting API requests and responses")
    os.remove(api_requests_filepath)
    os.remove(api_responses_filepath)

    if len(postprocessed_responses) == 0:
        logger.warning(f"None of the {len(api_responses)} API responses could be parsed. Exiting.")
    else:
        # Save parsed reports by appending to existing file
        n_parsed = len(postprocessed_responses)
        n_total = len(api_responses)
        logger.info(f"Succesfully parsed {n_parsed} of {n_total} API responses.\n"
                    f"{n_total - n_parsed} of {n_total} API responses could not be parsed.\n"
                    f"Saving parsed reports to {parsed_reports_filepath}")
        save_jsonl(postprocessed_responses, parsed_reports_filepath, append=True)
