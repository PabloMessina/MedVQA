from dotenv import load_dotenv

from medvqa.utils.logging import get_console_logger
load_dotenv()

import os
import argparse
import tiktoken
import re
import sys
import json

from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.openai_api import process_api_requests_from_file
from medvqa.utils.files import load_json, load_jsonl, load_pickle, save_jsonl
from medvqa.utils.common import get_timestamp

INSTRUCTIONS = """You are a function. Your input is the indication/history section of chest x-ray report.
Your output is a JSON list of facts. These facts must be very short. The topics to cover include demographics,
symptoms or conditions, and evaluation requests. Below are examples of the expected input/output format:

Input: "indication: 796.4 ABN CLINICAL FINDING NEC LOW RIB CAGE PAIN, R/O BODY/CHEST ETIOLOGY."
[
"low rib cage pain",
"rule out body/chest etiology"
]

Input: "indication: F with fall w/headstrike no loc // ICH? PNA?"
[
"female",
"fall",
"headstrike",
"no loss of consciousness",
"rule out intracranial hemorrhage",
"rule out pneunomia"
]

Input: "indication: M who blacked out and got in a fight last night, multiple cuts and bruises. // Fracture? Bleed? Forieng body in left hand."
[
"male",
"blacked out",
"fight",
"cuts",
"bruises",
"rule out fracture",
"rule out bleeding",
"rule out foreign body"
]

Input: "history: Abduction."
[
"abduction"
]

Note: if no facts can be extracted, return []."""

def generate_request(report, report_idx, part_id, subject_id, study_id, model_name, max_tokens, temperature=0.0):
    background = report["background"]
    assert len(background) > 0, f"Background is empty for report {report_idx}"

    if model_name == "gpt-3.5-turbo":
        return {
            "model": model_name,
            "messages": [{
                "role": "system",
                "content": INSTRUCTIONS,
            }, {
                "role": "user",
                "content": f'Input: "{background}"', # wrap in quotes to make it look like a string
            }],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "metadata": {
                "report_index": report_idx,
                "part_id": part_id,
                "subject_id": subject_id,
                "study_id": study_id,
                "background": background,
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
    list_of_facts = json.loads(match.group(0))
    assert isinstance(list_of_facts, list), f"Could not parse output: {text}"
    assert all(isinstance(fact, str) for fact in list_of_facts), f"Could not parse output: {text}"
    return list_of_facts

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_reports_filename", type=str, required=True)
    parser.add_argument("--ranked_background_indices_filename", type=str, required=True)
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
    logger = get_console_logger(args.logging_level)

    # Load preprocessed reports and ranked report indices
    preprocessed_reports_filepath = os.path.join(MIMICCXR_CACHE_DIR, args.preprocessed_reports_filename)
    logger.info(f"Loading preprocessed reports from {preprocessed_reports_filepath}")
    reports = load_json(preprocessed_reports_filepath)
    
    ranked_background_indices_filepath = os.path.join(MIMICCXR_CACHE_DIR, args.ranked_background_indices_filename)
    logger.info(f"Loading ranked background indices from {ranked_background_indices_filepath}")
    ranked_background_indices = load_pickle(ranked_background_indices_filepath)
    
    assert len(reports) == len(ranked_background_indices)
    assert 0 < args.num_reports <= len(reports)

    # Load parsed backgrounds if they exist
    parsed_backgrounds_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_parsed_backgrounds.jsonl")
    parsed_background_indices = set()
    if os.path.exists(parsed_backgrounds_filepath):
        parsed_backgrounds = load_jsonl(parsed_backgrounds_filepath)
        for parsed_background in parsed_backgrounds:
            parsed_background_indices.add(parsed_background['metadata']['report_index'])
        logger.info(f"Loaded {len(parsed_background_indices)} parsed backgrounds from {parsed_backgrounds_filepath}")        

    # Prepare API requests
    detailed_metadata = load_mimiccxr_reports_detailed_metadata()
    jobs = []
    for i in range(args.num_reports):
        if ranked_background_indices[i] in parsed_background_indices:
            continue
        ridx = ranked_background_indices[i]
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
            logger.error(f"Error parsing response {api_response_string} for report {metadata['report_index']}: {e}")
            continue

    # Delete API requests and responses
    logger.info(f"Deleting API requests and responses")
    os.remove(api_requests_filepath)
    os.remove(api_responses_filepath)

    if len(postprocessed_responses) == 0:
        logger.warning(f"None of the {len(api_responses)} API responses could be parsed. Exiting.")
    else:
        # Save parsed backgrouds by appending to existing file
        n_parsed = len(postprocessed_responses)
        n_total = len(api_responses)
        logger.info(f"Succesfully parsed {n_parsed} of {n_total} API responses.\n"
                    f"{n_total - n_parsed} of {n_total} API responses could not be parsed.\n"
                    f"Saving parsed backgrounds to {parsed_backgrounds_filepath}")
        save_jsonl(postprocessed_responses, parsed_backgrounds_filepath, append=True)