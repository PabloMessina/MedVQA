import os
import re
import argparse
import re
import sys
import json
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
from typing import Dict
from medvqa.datasets.chexpert import CHEXPERT_FAST_CACHE_DIR, CHEXPERT_FAST_TMP_DIR, CHEXPERT_PLUS_CSV_PATH
from medvqa.utils.text_data_utils import find_texts_matching_regex_in_parallel, sentence_tokenize_texts_in_parallel
from medvqa.utils.logging_utils import setup_logging
from medvqa.utils.text_data_utils import sort_sentences
from medvqa.utils.openai_api_utils import run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import load_jsonl


INSTRUCTIONS = """Relevant facts:

1. observations of abnormalities
2. observations of diseases
3. observations of strange visual patterns
4. observations of devices, tubes, lines, or similar objects
5. observations of foreign bodies
6. observations of specific anatomical regions that look normal or healthy
7. absences of abnormalities (usually expressed with a negation)
8. comparisons with respect to a previous study (something changed or remained the same)
9. suggestions or recommendations based on the observations

Task:

Given a sentence from a chest x-ray report, output a list of facts. Each fact should be about one observation. If a sentence mentions multiple observations, each observation should be extracted as a separate fact. Each fact should include the anatomical location where it was observed if provided. If multiple facts occur in the same location, repeat the location in each fact.

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

Taken together, compared with less than 1 hr earlier, the findings are suggestive of worsening of CHF, with new or significantly increased left greater right pleural effusions and underlying bibasilar collapse and/or consolidation, particularly on the left.
[
"worsening of CHF",
"new or significantly increased pleural effusions",
"the left pleural effusion is greater than the right pleural effusion",
"bibasilar lung collapse",
"bibasilar lung consolidation",
"left lung collapse",
"left lung consolidation",
]

No acute cardiopulmonary abnormality
[
"no acute cardiopulmonary abnormality"
]

Cutaneous clips.
[
"cutaneous clips"
]

If clinical presentation is not convincing for pneumonia, pulmonary embolism should be considered
[
"questionable evidence of pneumonia",
"possibility of pulmonary embolism",
]

Output format (JSON):
{ "reason": "{brief reasoning}", "facts": {list of facts} }

Note: correct any spelling errors, use lowercase except for names and abbreviations that are commonly written in uppercase, and only extract fully understandable facts (do not propagate errors or noise from the original sentence)."""


def parse_llm_output(llm_response_str: str) -> Dict[str, str]:
    """
    Parses the LLM response string to extract the reason and facts.
    We expect to find a JSON object in the following format:
    {
        "reason": "some brief reasoning",
        "facts": ["fact1", "fact2", ...]
    }
    Args:
        llm_response_str: The string response from the LLM.
    Returns:
        A dictionary containing the parsed reason and facts.
    """
    start = llm_response_str.find("{")
    end = llm_response_str.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"Could not find a valid JSON object in: {llm_response_str}"
        )
    json_str = llm_response_str[start : end + 1]
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError(f"Parsed data is not a dictionary: {data}")
    if "reason" not in data or "facts" not in data:
        raise ValueError(f"Missing expected keys in parsed data: {data}")
    if not isinstance(data["reason"], str) or not isinstance(data["facts"], list):
        raise ValueError(f"'reason' should be a string and 'facts' should be a list: {data}")
    assert all(isinstance(fact, str) for fact in data["facts"]), f"All facts should be strings: {data['facts']}"
    return {
        "reason": data["reason"],
        "facts": data["facts"],
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)    
    parser.add_argument("--num_processes", type=int, default=8)

    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--num_sentences", type=int, default=None)
    parser.add_argument("--only_special_sentences", action="store_true")

    parser.add_argument("--api_type", type=str, default="openai", choices=["openai", "gemini"],
                        help="Type of API to use. Default is 'openai'. For Gemini, use 'gemini'.")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_request", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alias", type=str, default="")
    parser.add_argument("--not_delete_api_requests_and_responses", action="store_true")
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--use_batch_api", action="store_true")
    parser.add_argument("--batch_description", type=str, default=None)
    parser.add_argument("--batch_input_file_id", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level. Default is INFO.")
    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce noise from httpx library
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level) # Update log level from args

    processed_sentences_save_filepath = os.path.join(CHEXPERT_FAST_CACHE_DIR, args.api_type,
                                                     f"{args.model_name}_facts_from_sentences{args.alias}.jsonl")

    if args.api_responses_filepath is None and args.batch_input_file_id is None:

        # Load already processed sentences if they exist
        already_processed = set()
        if os.path.exists(processed_sentences_save_filepath):
            rows = load_jsonl(processed_sentences_save_filepath)
            for row in rows:
                already_processed.add(row['metadata']['query'])
            logger.info(f"Loaded {len(rows)} already processed sentences from {processed_sentences_save_filepath}")

        # Collect report sections to process
        report_sections = []
        df = pd.read_csv(CHEXPERT_PLUS_CSV_PATH)
        logger.info(f"Loaded {len(df)} reports from {CHEXPERT_PLUS_CSV_PATH}")
        for findings in tqdm(df['section_findings'].dropna(), desc="Collecting findings"):
            findings = findings.strip()
            if findings:
                report_sections.append(findings)
        for impression in tqdm(df['section_impression'].dropna(), desc="Collecting impression"):
            impression = impression.strip()
            if impression:
                report_sections.append(impression)
        logger.info(f"Collected {len(report_sections)} report sections (findings and impression)")

        # Sentence tokenize the report sections
        logger.info(f"Sentence tokenizing the report sections")
        sentences_per_report = sentence_tokenize_texts_in_parallel(
            texts=report_sections,
            num_workers=args.num_processes,
            use_tqdm=True,
        )

        # Collect unique sentences
        logger.info(f"Collecting unique sentences")
        unique_sentences = set()
        for sentences in sentences_per_report:
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    unique_sentences.add(sentence)
        logger.info(f"Collected {len(unique_sentences)} unique sentences")

        # Sort sentences
        unique_sentences = list(unique_sentences)
        unique_sentences.sort(key=lambda x: (len(x), x)) # Sort by length and then alphabetically
        unique_sentences_lower = [s.lower() for s in unique_sentences]
        sorted_idxs = sort_sentences(sentences=unique_sentences_lower, by_difficulty=True,
                                     sort_indices=True, cache_ranking=True)
        unique_sorted_sentences = [unique_sentences[i] for i in sorted_idxs]
        
        # Print example sentences
        logger.info(f"Example sentences:")
        for i in range(10):
            logger.info(f"{i+1}. {unique_sorted_sentences[i]}")
        
        if args.only_special_sentences:
            logger.info(f"Filtering to only special sentences")
            regex = re.compile(
                # Conditionals
                r"\b(if|unless|when|whenever|wherever|whether|while|until|in case|as long as|provided that|"
                r"given that|although|though|whereas|as soon as|as long as|as much as|as often as|as far as|"
                r"as well as)\b|"
                # Greater than / less than
                r"\b(greater than|less than|more than|less than|gt|lt)\b|"
                # Concern
                r"concern|suggest|recommend|consider|questionable|represent|compatible|possible",
                re.IGNORECASE
            )
            unique_sorted_sentences = find_texts_matching_regex_in_parallel(
                unique_sorted_sentences, regex, num_workers=args.num_processes)
            logger.info(f"Found {len(unique_sorted_sentences)} special sentences")

            # Print example sentences
            logger.info(f"Example sentences (after filtering to only special sentences):")
            for i in range(10):
                logger.info(f"{i+1}. {unique_sorted_sentences[i]}")

        # Remove already processed sentences
        size_before = len(unique_sorted_sentences)
        unique_sorted_sentences = [s for s in unique_sorted_sentences if s not in already_processed]
        size_after = len(unique_sorted_sentences)
        logger.info(f"Removed {size_before - size_after} already processed sentences. {size_after} sentences left to process.")
        if size_after == 0:
            logger.info(f"All {len(unique_sorted_sentences)} sentences have already been processed. Nothing to do. Exiting.")
            sys.exit(0)

        # Adjust number of sentences to process if necessary
        assert 0 <= args.offset < len(unique_sorted_sentences)
        if args.offset + args.num_sentences > len(unique_sorted_sentences):
            logger.warning(
                f"Requested {args.num_sentences} sentences but only {len(unique_sorted_sentences) - args.offset} are available."
                f" Using {len(unique_sorted_sentences) - args.offset} instead.")
            args.num_sentences = len(unique_sorted_sentences) - args.offset
            assert args.num_sentences > 0

        # Apply offset and num_sentences
        logger.info(f"Collecting the first {args.num_sentences} sentences starting from the {args.offset}-th sentence")
        unique_sorted_sentences = unique_sorted_sentences[args.offset:args.offset + args.num_sentences]

        sentences_to_process = unique_sorted_sentences

        logger.info(f"Total number of sentences to process: {len(sentences_to_process)}")

        # Print example sentences
        logger.info(f"Example sentences to process:")
        for i in np.linspace(0, len(sentences_to_process)-1, min(10, len(sentences_to_process)), dtype=int):
            logger.info(f"{i+1}. {sentences_to_process[i]}")

    else:
        if args.api_responses_filepath is not None:
            assert os.path.exists(args.api_responses_filepath)
        sentences_to_process = None

    # Run API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=sentences_to_process,
        system_instructions=INSTRUCTIONS,
        api_key_name=args.api_key_name,
        model_name=args.model_name,        
        max_tokens_per_request=args.max_tokens_per_request,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        temperature=args.temperature,
        frequency_penalty=0.0,
        presence_penalty=0.0,        
        parse_output=parse_llm_output,
        tmp_dir=CHEXPERT_FAST_TMP_DIR,
        save_filepath=processed_sentences_save_filepath,
        use_batch_api=args.use_batch_api,
        batch_description=args.batch_description,
        batch_input_file_id=args.batch_input_file_id,
        api_type=args.api_type,
        delete_api_requests_and_responses=not args.not_delete_api_requests_and_responses,
    )