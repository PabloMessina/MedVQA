import argparse
import json
import logging
import os
import random
from pathlib import Path

import pandas as pd

from medvqa.datasets.mimiccxr import get_study_id_to_report_text_dict
from medvqa.datasets.regular_expressions.cxr_patterns import (
    collect_reports_matching_class,
)
from medvqa.settings import (
    LLM_PROMPTS_DIR,
    MIMIC_CXR_LT_2024_LABELS_CSV_PATH,
    MIMIC_CXR_LT_2024_TASK2_TEST_CSV_PATH,
    MIMICCXR_CACHE_DIR,
    MIMICCXR_SPLIT_CSV_PATH,
)
from medvqa.utils.constants import (
    UNIFIED_CXRLT2024_VINDRCXR_CLASS_TO_REGEX_CLASSES,
    UNIFIED_CXRLT2024_VINDRCXR_CLASS_TO_VERBOSE_PHRASE_FOR_REPORT_NLI,
    UNIFIED_CXRLT2024_VINDRCXR_CLASSES,
)
from medvqa.utils.files_utils import load_jsonl, load_pickle, read_txt, save_pickle
from medvqa.utils.logging_utils import setup_logging
from medvqa.utils.openai_api_utils import (
    GPT_IS_ACTING_WEIRD_REGEX,
    orchestrate_api_calls,
)

setup_logging()
logger = logging.getLogger(__name__)

# Turn off annoying httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

_POSSIBLE_LABELS = [
    "definitely true",
    "likely true",
    "unknown",
    "likely false",
    "definitely false",
]

_LABEL_TO_BINARY = {
    "definitely true": 1,
    "likely true": 1,
    "unknown": 0,
    "likely false": 0,
    "definitely false": 0,
}

def parse_llm_model_output(text):
    """
    Parse the output of the LLM model.
    We expect to find a JSON object in the following format:
    {
        "reason": "some brief reasoning",
        "label": "one of {definitely true, likely true, unknown, likely false, definitely false}",
    }
    Args:
        text: The text response from the LLM.
    Returns:
        A dictionary containing the parsed reason and label.
    """
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    start_idx = text.index("{")
    end_idx = text.rindex("}")
    json_str = text[start_idx:end_idx + 1]
    json_obj = json.loads(json_str)
    assert "relevant_quote" in json_obj, f"No relevant quote found in output: {text}"
    assert "reasoning" in json_obj, f"No reasoning found in output: {text}"
    assert "uses_hedging_language" in json_obj, f"No uses_hedging_language found in output: {text}"
    assert "label" in json_obj, f"No label found in output: {text}"
    relevant_quote = json_obj["relevant_quote"]
    assert isinstance(relevant_quote, str), f"Relevant quote is not a string: {relevant_quote}"
    relevant_quote = relevant_quote.strip()
    reasoning = json_obj["reasoning"]
    assert isinstance(reasoning, str), f"Reasoning is not a string: {reasoning}"
    reasoning = reasoning.strip()
    assert len(reasoning) > 0, f"Empty reasoning: {reasoning}"
    uses_hedging_language = json_obj["uses_hedging_language"]
    assert isinstance(uses_hedging_language, str), f"Uses hedging language is not a string: {uses_hedging_language}"
    uses_hedging_language = uses_hedging_language.strip()
    assert len(uses_hedging_language) > 0, f"Empty uses hedging language: {uses_hedging_language}"
    label = json_obj["label"]
    assert isinstance(label, str), f"Label is not a string: {label}"
    label = label.strip()
    assert len(label) > 0, f"Empty label: {label}"
    label = label.lower()
    assert label in _POSSIBLE_LABELS, f"Invalid label: {label}"
    return {
        "relevant_quote": relevant_quote,
        "reasoning": reasoning,
        "uses_hedging_language": uses_hedging_language,
        "label": label,
    }

def define_mimiccxr_preliminary_splits(args):
    """
    Define a preliminary split of the MIMIC-CXR dataset that will be used to generate classification labels with LLMs. 
    Args:
        args: ArgumentParser object containing the following arguments:
            --min_num_samples_per_class: Minimum number of samples per class
            --total_samples: Total number of samples
    Returns:
        None. Saves the preliminary splits to a pickle file.
    """

    logger.info(f"Defining MIMIC-CXR preliminary splits with min_num_samples_per_class={args.min_num_samples_per_class} and total_samples={args.total_samples}")

    save_path = os.path.join(MIMICCXR_CACHE_DIR, f'mimiccxr_cxrlt2024+vindrcxr_preliminary_splits({args.min_num_samples_per_class},{args.total_samples}).pkl')
    if os.path.exists(save_path):
        logger.info(f"Preliminary splits already exist at {save_path}. Skipping...")
        return

    # Read the CSV that contains the MIMIC-CXR train/val/test split info
    mimiccxr_split_df = pd.read_csv(MIMICCXR_SPLIT_CSV_PATH)
    
    # Get original test study IDs
    mimiccxr_original_test_study_ids = set(mimiccxr_split_df[mimiccxr_split_df.split == 'test'].study_id)
    logger.info(f"Found {len(mimiccxr_original_test_study_ids)} MIMIC-CXR original test study IDs")
    
    # Get original validation study IDs
    mimiccxr_original_val_study_ids = set(mimiccxr_split_df[mimiccxr_split_df.split == 'validate'].study_id)
    logger.info(f"Found {len(mimiccxr_original_val_study_ids)} MIMIC-CXR original validation study IDs")

    # Get original train study IDs
    mimiccxr_original_train_study_ids = set(mimiccxr_split_df[mimiccxr_split_df.split == 'train'].study_id)
    logger.info(f"Found {len(mimiccxr_original_train_study_ids)} MIMIC-CXR original train study IDs")

    # Read CXR-LT 2024 test set
    cxrlt2024_test_df = pd.read_csv(MIMIC_CXR_LT_2024_TASK2_TEST_CSV_PATH)
    # Extract integer study IDs by removing the leading character ('s' in 's123456789')
    cxrlt2024_test_study_ids = set(int(x[1:]) for x in cxrlt2024_test_df.study_id)
    logger.info(f"Found {len(cxrlt2024_test_study_ids)} CXR-LT 2024 test study IDs")

    # Combine test and val as candidates (remove test set used by CXR-LT 2024)
    candidate_study_ids = mimiccxr_original_test_study_ids | mimiccxr_original_val_study_ids
    candidate_study_ids -= cxrlt2024_test_study_ids # Remove study IDs in the CXR-LT 2024 test set
    candidate_study_ids = list(candidate_study_ids)
    candidate_study_ids.sort()
    logger.info(f"Found {len(candidate_study_ids)} candidate study IDs")

    # Get study_id: report_text mapping for all studies
    study_id_to_report_text_dict = get_study_id_to_report_text_dict()
    # Select candidate report texts by study_id
    candidate_reports = [study_id_to_report_text_dict[study_id] for study_id in candidate_study_ids]
    # Lowercase and clean candidate report texts for easier regex matching
    candidate_reports_lower_clean = [' '.join(x.lower().split()) for x in candidate_reports]
    logger.info(f"Found {len(candidate_reports_lower_clean)} candidate reports")
    
    # Candidate set for each unified class
    unified_class_to_candidate_study_ids = {x: set() for x in UNIFIED_CXRLT2024_VINDRCXR_CLASSES}

    class_match_cache = {} # Cache for regex matches
    for unified_class in UNIFIED_CXRLT2024_VINDRCXR_CLASSES:
        regex_classes = UNIFIED_CXRLT2024_VINDRCXR_CLASS_TO_REGEX_CLASSES[unified_class]
        if regex_classes is None:
            logger.warning(f'Class {unified_class} has no regex classes. Skipping...')
            continue
        assert isinstance(regex_classes, list)
        for regex_class in regex_classes:
            # Find all candidate report indices matching the class
            matching_indices = collect_reports_matching_class(
                reports=candidate_reports_lower_clean,
                class_name=regex_class,
                class_match_cache=class_match_cache,
            )
            # Map indices back to study IDs and record matches
            for idx in matching_indices:
                study_id = candidate_study_ids[idx]
                unified_class_to_candidate_study_ids[unified_class].add(study_id)
        logger.info(f'Found {len(unified_class_to_candidate_study_ids[unified_class])} reports matching the regex classes'
                    f' {regex_classes} (unified class: {unified_class})')

    # Special handling for the "Normal" class: use explicit labels
    cxrlt2024_labels_df = pd.read_csv(MIMIC_CXR_LT_2024_LABELS_CSV_PATH)
    cxrlt2024_normal_study_ids = set(int(x[1:]) for x in cxrlt2024_labels_df[cxrlt2024_labels_df.Normal == 1].study_id)
    logger.info(f"Found {len(cxrlt2024_normal_study_ids)} CXR-LT 2024 normal study IDs")
    cxrlt2024_normal_study_ids &= set(candidate_study_ids) # Keep only study IDs that are in the candidate study IDs
    logger.info(f"Found {len(cxrlt2024_normal_study_ids)} CXR-LT 2024 normal study IDs that are in the candidate study IDs")
    unified_class_to_candidate_study_ids['Normal'] = cxrlt2024_normal_study_ids
    
    # Sort all unified classes by how many candidate reports exist for that class
    sorted_unified_classes = sorted(UNIFIED_CXRLT2024_VINDRCXR_CLASSES, key=lambda x: len(unified_class_to_candidate_study_ids[x]))

    chosen_extra_study_ids = set()  # Set of additional chosen study ids
    
    for unified_class in sorted_unified_classes:
        candidate_study_ids = unified_class_to_candidate_study_ids[unified_class]
        num_samples = min(args.min_num_samples_per_class, len(candidate_study_ids))
        assert num_samples > 0, f"Not enough samples for the class {unified_class}"
        # Prefer longer reports (sort by length of report text, descending)
        sorted_candidate_study_ids = list(candidate_study_ids)
        sorted_candidate_study_ids.sort(key=lambda x: len(study_id_to_report_text_dict[x]), reverse=True)
        # Add up to num_samples study_ids for this class
        chosen_extra_study_ids.update(sorted_candidate_study_ids[:num_samples])

    logger.info(f"Found {len(chosen_extra_study_ids)} extra study IDs (after regex matching, before filling up)")
    # Fill up with any remaining samples if required to reach total_extra
    total_extra = args.total_samples - len(cxrlt2024_test_study_ids)
    logger.info(f"Total extra samples needed: {total_extra}")
    if len(chosen_extra_study_ids) < total_extra:
        # Find unused studies and select randomly
        unused_study_ids = list(set(candidate_study_ids) - chosen_extra_study_ids)
        num_samples = min(total_extra - len(chosen_extra_study_ids), len(unused_study_ids))
        chosen_extra_study_ids.update(random.sample(unused_study_ids, num_samples))

    logger.info(f"Found {len(chosen_extra_study_ids)} extra study IDs (after filling up)")
    # Final val+test split = CXR-LT 2024 test + our extra study ids
    final_val_test_study_ids = cxrlt2024_test_study_ids | chosen_extra_study_ids
    final_val_test_study_ids = list(final_val_test_study_ids)
    final_val_test_study_ids.sort()
    # The remaining are train
    final_train_study_ids = mimiccxr_original_train_study_ids - set(final_val_test_study_ids)
    final_train_study_ids = list(final_train_study_ids)
    final_train_study_ids.sort()

    logger.info(f"Found {len(final_train_study_ids)} final train study IDs")
    logger.info(f"Found {len(final_val_test_study_ids)} final val_test study IDs")

    # Save splits
    splits = {
        'train_study_ids': final_train_study_ids,
        'val_test_study_ids': final_val_test_study_ids,
    }
    save_pickle(splits, save_path)
    logger.info(f"Saved splits to {save_path}")


def generate_classification_labels_with_llms(args):
    """
    Generate classification labels with LLMs for the MIMIC-CXR dataset.
    Args:
        args: ArgumentParser object containing the following arguments:
            --preliminary_splits_filepath: Path to the pickle file containing the preliminary splits
    Returns:
        None. Saves the classification labels to a pickle file.
    """

    logger.info(f"Generating classification labels with LLMs for the MIMIC-CXR dataset with preliminary splits from {args.preliminary_splits_filepath}")

    if args.processed_queries_save_filepath is not None:
        processed_queries_save_filepath = args.processed_queries_save_filepath
    else:
        processed_queries_save_filepath = os.path.join(MIMICCXR_CACHE_DIR, 'llm_annotations',
                                                    args.llm_model_name, 
                                                    f'mimiccxr_report_classification_labels{args.alias}.jsonl')
    processed_queries_save_filepath = Path(processed_queries_save_filepath)
    processed_queries_save_filepath.parent.mkdir(parents=True, exist_ok=True)

    if not args.api_responses_filepath:

        # --- 1. Load System Instructions ---
        try:
            system_instructions_path = (
                os.path.join(LLM_PROMPTS_DIR, args.system_instructions_relative_path)
            )
            system_instructions = read_txt(system_instructions_path).strip()
            logger.info(
                "Loaded system instructions (first 100 chars): "
                f"{system_instructions[:100]}..."
            )
        except FileNotFoundError:
            logger.error(
                f"System instructions file not found: {system_instructions_path}"
            )
            return

        # --- 2. Load already processed queries to avoid re-running ---
        already_processed_queries = set()
        if processed_queries_save_filepath.exists():
            processed_items = load_jsonl(processed_queries_save_filepath)

            for item in processed_items:
                if "metadata" in item and "query" in item["metadata"]:
                    already_processed_queries.add(item["metadata"]["query"])
            logger.info(
                f"Loaded {len(processed_items)} already processed items from {processed_queries_save_filepath}. "
                f"Found {len(already_processed_queries)} unique queries to skip."
            )

        # --- 3. Load MIMIC-CXR split data to filter reports ---
        preliminary_splits = load_pickle(args.preliminary_splits_filepath)
        study_id_to_report_text_dict = get_study_id_to_report_text_dict()
        queries_to_process_for_llm = set()

        if args.split_to_process == 'train':
            raise NotImplementedError("Training split is not supported yet")
        elif args.split_to_process == 'val_test':
            val_test_study_ids = preliminary_splits['val_test_study_ids']
            target_reports = [study_id_to_report_text_dict[study_id] for study_id in val_test_study_ids]
            target_reports_cleaned = [' '.join(x.split()) for x in target_reports]
            for report in target_reports_cleaned:
                for unified_class in UNIFIED_CXRLT2024_VINDRCXR_CLASSES:
                    phrase = UNIFIED_CXRLT2024_VINDRCXR_CLASS_TO_VERBOSE_PHRASE_FOR_REPORT_NLI[unified_class]
                    query = f'<report>{report}</report>\n<hypothesis>{phrase}</hypothesis>'
                    if query in already_processed_queries:
                        continue # Avoid duplicates
                    queries_to_process_for_llm.add(query)
                    if len(queries_to_process_for_llm) >= args.max_queries_to_process:
                        break
                if len(queries_to_process_for_llm) >= args.max_queries_to_process:
                    break
        else:
            raise ValueError(f"Invalid split to process: {args.split_to_process}")
        
        if not queries_to_process_for_llm:
            logger.info("No new queries to process. Exiting.")
            return

        queries_to_process_for_llm = list(queries_to_process_for_llm)
        logger.info(f"Prepared {len(queries_to_process_for_llm)} new queries for LLM.")

        # Print the first few queries for inspection
        logger.info("Example queries to process:")
        for i, query in enumerate(queries_to_process_for_llm[:5]):
            logger.info("---------------------")
            logger.info(f"{i + 1}. {query}")
            logger.info("---------------------")

    else:
        queries_to_process_for_llm = None
        system_instructions = None

    # --- 6. Run LLM API requests ---
    try:
        orchestrate_api_calls(
            api_responses_filepath=args.api_responses_filepath,
            texts=queries_to_process_for_llm,
            system_instructions=system_instructions,
            model_name=args.llm_model_name,
            api_key_name=args.api_key_name,
            api_type=args.api_type,
            max_requests_per_minute=args.max_requests_per_minute,
            max_tokens_per_minute=args.max_tokens_per_minute,
            max_tokens_per_request=args.max_tokens_per_request,
            temperature=args.temperature,
            parse_output=parse_llm_model_output,
            save_filepath=processed_queries_save_filepath,
            delete_api_requests_and_responses=not args.dont_delete_api_requests_and_responses,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            log_info_every_n_requests=args.log_info_every_n_requests,
        )
    except Exception as e:
        logger.error(f"Error during orchestrate_api_calls: {e}", exc_info=True)
        return

    logger.info("Processing complete.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: define_mimiccxr_splits
    parser_define_mimiccxr_preliminary_splits = subparsers.add_parser(
        "define_mimiccxr_preliminary_splits",
        help="Define a preliminar split of the MIMIC-CXR dataset that will be used to generate classification labels with LLMs"
    )
    parser_define_mimiccxr_preliminary_splits.add_argument("--min_num_samples_per_class", type=int, default=50)
    parser_define_mimiccxr_preliminary_splits.add_argument("--total_samples", type=int, default=1500)
    parser_define_mimiccxr_preliminary_splits.set_defaults(func=define_mimiccxr_preliminary_splits)

    # Sub-command: generate_classification_labels_with_llms
    parser_generate_classification_labels_with_llms = subparsers.add_parser(
        "generate_classification_labels_with_llms",
        help="Generate classification labels with LLMs for the MIMIC-CXR dataset"
    )
    parser_generate_classification_labels_with_llms.add_argument("--preliminary_splits_filepath", type=str, required=True)
    parser_generate_classification_labels_with_llms.add_argument("--max_queries_to_process", type=int, default=None)
    parser_generate_classification_labels_with_llms.add_argument("--api_responses_filepath", type=str, default=None)
    parser_generate_classification_labels_with_llms.add_argument("--dont_delete_api_requests_and_responses", action="store_true", default=False)
    parser_generate_classification_labels_with_llms.add_argument("--log_info_every_n_requests", type=int, default=100)
    parser_generate_classification_labels_with_llms.add_argument("--llm_model_name", type=str, default="gemini-flash-lite-latest")
    parser_generate_classification_labels_with_llms.add_argument("--system_instructions_relative_path", type=str, default="report_nli.txt")
    parser_generate_classification_labels_with_llms.add_argument("--api_key_name", type=str, default="GEMINI_API_KEY")
    parser_generate_classification_labels_with_llms.add_argument("--api_type", type=str, default="gemini")
    parser_generate_classification_labels_with_llms.add_argument("--max_requests_per_minute", type=int, default=None)
    parser_generate_classification_labels_with_llms.add_argument("--max_tokens_per_minute", type=int, default=None)
    parser_generate_classification_labels_with_llms.add_argument("--max_tokens_per_request", type=int, default=None)
    parser_generate_classification_labels_with_llms.add_argument("--temperature", type=float, default=0.0)
    parser_generate_classification_labels_with_llms.add_argument("--frequency_penalty", type=float, default=0.0)
    parser_generate_classification_labels_with_llms.add_argument("--presence_penalty", type=float, default=0.0)
    parser_generate_classification_labels_with_llms.add_argument("--alias", type=str, default="")
    parser_generate_classification_labels_with_llms.add_argument("--split_to_process", type=str, default="val_test", choices=["train", "val_test"])
    parser_generate_classification_labels_with_llms.add_argument("--processed_queries_save_filepath", type=str, default=None)
    parser_generate_classification_labels_with_llms.set_defaults(func=generate_classification_labels_with_llms)

    args = parser.parse_args()
    args.func(args)