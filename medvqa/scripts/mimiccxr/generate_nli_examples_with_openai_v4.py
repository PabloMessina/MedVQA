import os
import argparse
import sys
import numpy as np
import pandas as pd
from medvqa.utils.logging import get_console_logger
from medvqa.datasets.nli import MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH, RADNLI_DEV_JSONL_PATH, RADNLI_TEST_JSONL_PATH
from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
)
from medvqa.utils.openai_api import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files import load_jsonl

# INSTRUCTIONS = """Context: natural language inference.

# Given a premise and a hypothesis, output "entailment", "contradiction", or "neutral".

# Use "entailment" when the facts stated by the premise necessarily entail the truth of the hypothesis.

# Use "contradiction" when premise and hypothesis are mutually exclusive/contradictory (both cannot be true at the same time).

# Use "neutral", if there is no contradiction (premise and hypothesis are compatible), but the premise does not entail the hypothesis (it's possible for the premise to be true and the hypothesis still be false). In other words, use "neutral" when neither "entailment" nor "contradiction" adequately fit."""


# INSTRUCTIONS = """Context: natural language inference.

# Given a premise (#P) and a hypothesis (#H), output "Reason: {reason}. Label: {label}" where {reason} is a short sentence and {label} is one of "entailment," "contradiction," or "neutral."

# Use "entailment" when the premise necessarily entails the truth of the hypothesis.

# Use "contradiction" when premise and hypothesis are mutually exclusive/contradictory. Pay attention to subtle contradictions such as contradictory degrees of certainty, expressions suggesting presence vs. absence, etc.

# Use "neutral" when there's no contradiction, but the premise doesn't necessarily entail the hypothesis.

# Examples:

# 1. #P: increased pulmonary edema. | #H: worsened pulmonary edema.
# Label: entailment

# 2. #P: No pulmonary edema, consolidation, or pneumothorax. | #H: No focal consolidation, pleural effusion, or pneumothorax is present.
# Label: neutral"""


INSTRUCTIONS = """Context: natural language inference.

Given a premise (#P) and a hypothesis (#H), output "Reason: {reason}. Label: {label}" where {reason} is a short sentence and {label} is one of "entailment," "contradiction," or "neutral."

Use "entailment" when the premise necessarily entails the truth of the hypothesis.

Use "contradiction" when premise and hypothesis are mutually exclusive/contradictory. Pay attention to logical inconsistencies, such as expressions suggesting presence vs. absence, etc.

Use "neutral" when there's no contradiction, but the premise doesn't necessarily entail the hypothesis.

Examples:

1. #P: increased pulmonary edema. | #H: worsened pulmonary edema.
Label: entailment

2. #P: No pulmonary edema, consolidation, or pneumothorax. | #H: No focal consolidation, pleural effusion, or pneumothorax is present.
Label: neutral"""


def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    text = text.lower()
    if 'label: entailment' in text:
        return 'entailment'
    if 'label: contradiction' in text:
        return 'contradiction'
    if 'label: neutral' in text:
        return 'neutral'
    raise RuntimeError(f"Could not parse output: {text}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    processed_texts_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_radnli_mscxrt_queries.jsonl")
    
    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None:

        # Load already processed queries if they exist
        already_processed = set()
        if os.path.exists(processed_texts_save_filepath):
            rows = load_jsonl(processed_texts_save_filepath)
            for row in rows:
                already_processed.add(row['metadata']['query'])
            logger.info(f"Loaded {len(rows)} already processed texts from {processed_texts_save_filepath}")
        
        nli_queries = []
        
        # Load RadNLI queries
        dev_rows = load_jsonl(RADNLI_DEV_JSONL_PATH)
        test_rows = load_jsonl(RADNLI_TEST_JSONL_PATH)
        for rows in [dev_rows, test_rows]:
            for row in rows:
                nli_queries.append(f'#P: {row["sentence1"]} | #H: {row["sentence2"]}')

        # Load MS-CXR-T queries
        df = pd.read_csv(MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH)
        n = len(df)
        for premise, hypothesis in zip(df.sentence_1, df.sentence_2):
            nli_queries.append(f'#P: {premise} | #H: {hypothesis}')

        # Remove already processed texts
        logger.info(f"Removing {len(already_processed)} already processed texts")
        texts_to_process = [s for s in nli_queries if s not in already_processed]
        if len(texts_to_process) == 0:
            logger.info(f"All {len(nli_queries)} texts have already been processed. Nothing to do. Exiting.")
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