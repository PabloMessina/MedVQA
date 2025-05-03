import os
import argparse
import random
from medvqa.utils.common import FAST_CACHE_DIR, FAST_TMP_DIR
from medvqa.utils.logging_utils import get_console_logger
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import load_jsonl, load_pickle

INSTRUCTIONS = """You will receive a Python object with the predictions from an Image Classifier over one or more views of a Chest X-ray. The object has the following format:
{
'num_images': int,
'yes':  list of strings,
'no': list of strings,
}
A string looks like this: "{label}: {pred_prob_str}, {prior_f1}, {prior_prob}".
pred_prob_str concatenates binary predictions and sigmoid probabilties from the classifier for each image.
prior_f1 is a priorly known f1 score of the classifier in a gold standard dataset (the classifier's reliability in that class).
prior_prob is the empirical probability of the class a priori (based on known statistics)

'yes' mostly contains 'yes' predictions.
'no' only contains 'no' predictions.

Your task is to generate a plausible chest X-ray radiological report matching the predictions from the image classifier, in order to preserve its more reliable predictions.

Format:

Output 2 sections:

FINDINGS: {write the findings section of the report}

IMPRESSION: {write the impression section of the report}

NOTE. The classifier gives you information about multiple classes. However, reports are more succinct. If most findings are negative, the report usually uses generic expressions conveying that most things look fine. Reports tend to focus more on the positive findings. Please write a realistic report. Avoid excessively long sentences. Review the following examples to get the idea. 

Example 1:

FINDINGS: The cardiomediastinal silhouette and pulmonary vasculature are unremarkable. There is opacification in the left base.  Though this may be atelectasis, pneumonia is not excluded.  Possible mild bronchial wall thickening is noted, particularly in the right lower lung field.  There is no pleural effusion or  pneumothorax.
 
IMPRESSION:  Possible left lower lobe pneumonia in the appropriate clinical context.

Example 2:

FINDINGS: There is a mild pectus deformity.  The heart size is normal.  The hilar and mediastinal contours are within normal limits.  There is no pneumothorax, focal consolidation, or pleural effusion.
 
IMPRESSION:  No acute intrathoracic process."""

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    findings_index = text.index("FINDINGS:")
    impression_index = text.index("IMPRESSION:")
    assert findings_index < impression_index, f"Could not find FINDINGS and IMPRESSION sections in the output: {text}"
    findings = text[findings_index + len("FINDINGS:"):impression_index].strip()
    impression = text[impression_index + len("IMPRESSION:"):].strip()
    assert findings, f"Could not find FINDINGS section in the output: {text}"
    assert impression, f"Could not find IMPRESSION section in the output: {text}"
    return {"findings": findings, "impression": impression}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--preprocessed_queries_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--template_based_reports_filepath", type=str, required=True)
    parser.add_argument("--first_n", type=int, default=None)

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    args = parser.parse_args()

    processed_queries_save_filepath = os.path.join(FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_templates_to_reports{args.alias}.jsonl")
    
    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None:

        # Load already processed queries if they exist
        queries_to_skip = set()
        if os.path.exists(processed_queries_save_filepath):
            rows = load_jsonl(processed_queries_save_filepath)
            for row in rows:
                queries_to_skip.add(row['metadata']['query'])
            logger.info(f"Loaded {len(rows)} already processed queries from {processed_queries_save_filepath}")
        if args.preprocessed_queries_to_skip_filepaths is not None:
            for filepath in args.preprocessed_queries_to_skip_filepaths:
                assert os.path.exists(filepath)
                rows = load_jsonl(filepath)
                queries_to_skip.update(row['metadata']['query'] for row in rows)
                logger.info(f"Loaded {len(rows)} queries to skip from {filepath}")

        # Collect queries to make
        assert os.path.exists(args.template_based_reports_filepath)
        logger.info(f"Loading template-based reports from {args.template_based_reports_filepath}")
        template_based_reports = load_pickle(args.template_based_reports_filepath)
        queries_to_make = [x['json_string'] for x in template_based_reports]

        # Remove queries that have already been processed
        logger.info(f"Queries to make: {len(queries_to_make)} (before removing queries that have already been processed)")
        n_before = len(queries_to_make)
        queries_to_make = [x for x in queries_to_make if x not in queries_to_skip]
        logger.info(f"Queries to make: {len(queries_to_make)} (after removing queries that have already been processed)")
        n_after = len(queries_to_make)
        logger.info(f"Removed {n_before - n_after} queries that have already been processed")

        # Limit the number of queries to make
        if args.first_n is not None:
            queries_to_make = queries_to_make[:args.first_n]

        # Print some examples
        logger.info(f"Example queries to make:")
        for i in random.sample(range(len(queries_to_make)), k=3):
            logger.info(f"Query {i + 1}: {queries_to_make[i]}")

    else:
        assert os.path.exists(args.api_responses_filepath)
        queries_to_make = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=queries_to_make,
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
        tmp_dir=FAST_TMP_DIR,
        save_filepath=processed_queries_save_filepath,
    )