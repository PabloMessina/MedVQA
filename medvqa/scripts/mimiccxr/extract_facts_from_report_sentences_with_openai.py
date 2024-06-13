import os
import re
import argparse
import re
import sys
import json
import numpy as np

from medvqa.datasets.text_data_utils import sentence_tokenize_texts_in_parallel
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.logging import get_console_logger
from medvqa.utils.nlp import sort_sentences
from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.openai_api import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files import load_json, load_jsonl

# INSTRUCTIONS = """Relevant facts:

# 1. observations of abnormalities
# 2. observations of diseases
# 3. observations of strange visual patterns
# 4. observations of devices
# 5. observations of foreign bodies
# 6. observations of specific anatomical regions that look normal or healthy
# 7. absences of abnormalities (usually expressed with a negation)
# 8. comparisons with respect to a previous study (something changed or remained the same)

# Task:

# Given a sentence taken from a chest x-ray report, generate a JSON list of relevant facts.
# Each fact should be about one observation. If a sentence mentions multiple observations,
# each observation should be extracted as a separate fact.
# Each fact should include the anatomical location where it was observed. If multiple facts
# occur in the same location, repeat the location in each fact.

# If no relevant facts are mentioned, return [] (an empty array).

# Examples:

# Opacity and density in the right lobe
# [
# "opacity in the right lobe",
# "density in the right lobe"
# ]

# Lungs are well inflated without evidence of focal airspace consolidation to suggest pneumonia.
# [
# "well inflated lungs",
# "lungs without evidence of focal airspace consolidation",
# "lungs without evidence of pneumonia"
# ]

# Taken together, compared with less than 1 hr earlier, the findings are suggestive of worsening of CHF, with new or significantly increased left greater right pleural effusions and underlying bibasilar collapse and/or  consolidation, particularly on the left.
# [
# "worsening of CHF",
# "new or significantly increased left pleural effusions",
# "new or significantly increased right pleural effusions",
# "underlying bibasilar collapse on the left",
# "underlying consolidation on the left",
# ]

# No acute cardiopulmonary abnormality
# [
# "no acute cardiopulmonary abnormality"
# ]"""

INSTRUCTIONS = """Relevant facts:

1. observations of abnormalities
2. observations of diseases
3. observations of strange visual patterns
4. observations of devices
5. observations of foreign bodies
6. observations of specific anatomical regions that look normal or healthy
7. absences of abnormalities (usually expressed with a negation)
8. comparisons with respect to a previous study (something changed or remained the same)
9. suggestions or recommendations based on the observations

Task:

Given a sentence from a chest x-ray report, output a JSON array of facts.
Each fact should be about one observation. If a sentence mentions multiple observations,
each observation should be extracted as a separate fact.
Each fact should include the anatomical location where it was observed. If multiple facts occur in the same location, repeat the location in each fact.

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
]

If clinical presentation is not convincing for pneumonia, pulmonary embolism should be considered
[
"clinical presentation may indicate pneumonia",
"clinical presentation may indicate pulmonary embolism",
]"""

_JSON_STRING_ARRAY_REGEX = re.compile(r"\[\s*(\".+?\",?\s*)*\]")

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = _JSON_STRING_ARRAY_REGEX.search(text) # match a JSON list of strings
    if not match:
        if GPT_IS_ACTING_WEIRD_REGEX.search(text):
            logger.warning(f"GPT is protesting: {text}, returning []")
            return []
    assert match, f"Could not parse output: {text}"
    list_of_facts = json.loads(match.group(0))
    assert isinstance(list_of_facts, list), f"Could not parse output: {text}"
    assert all(isinstance(fact, str) for fact in list_of_facts), f"Could not parse output: {text}"
    return list_of_facts

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--batch_input_file_id", type=str, default=None)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)    
    parser.add_argument("--preprocessed_reports_filepath", type=str)
    
    parser.add_argument("--cxr_bert_model_name", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--cxr_bert_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_clusters", type=int, default=200)
    parser.add_argument("--num_iterations", type=int, default=300)

    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--num_sentences", type=int, default=None)
    parser.add_argument("--rank_sentences_by_difficulty", action="store_true", default=False)
    parser.add_argument("--sample_sentences_uniformly", action="store_true", default=False)
    parser.add_argument("--process_kth_of_every_n_sentences", type=int, nargs=2, default=None,
                        help="If specified, only process the kth of every n sentences.")
    parser.add_argument("--only_conditional_sentences", action="store_true", default=False)
    parser.add_argument("--use_test_set_sentences", action="store_true", default=False)

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_request", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    parser.add_argument("--use_batch_api", action="store_true", default=False)
    parser.add_argument("--batch_description", type=str, default=None)
    args = parser.parse_args()

    processed_sentences_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_facts_from_sentences{args.alias}.jsonl")
    
    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None and args.batch_input_file_id is None:

        # Load already processed sentences if they exist
        already_processed = set()
        if os.path.exists(processed_sentences_save_filepath):
            rows = load_jsonl(processed_sentences_save_filepath)
            for row in rows:
                already_processed.add(row['metadata']['query'])
            logger.info(f"Loaded {len(rows)} already processed sentences from {processed_sentences_save_filepath}")

        # Collect sentences from reports
        unique_sentences = set()
        assert os.path.exists(args.preprocessed_reports_filepath)
        logger.info(f"Loading preprocessed reports from {args.preprocessed_reports_filepath}")
        if args.use_test_set_sentences:
            data = load_mimiccxr_reports_detailed_metadata(
                background_findings_and_impression_per_report_filepath=args.preprocessed_reports_filepath
            )
            reports = load_json(args.preprocessed_reports_filepath)
            texts = []
            for findings, impression, split in zip(data['findings'], data['impressions'], data['splits']):
                if split == 'test':
                    if len(impression) > 0:
                        texts.append(impression)
                    if len(findings) > 0:
                        texts.append(findings)
            logger.info(f"Loaded {len(texts)} texts from MIMIC-CXR test set")
        else:
            reports = load_json(args.preprocessed_reports_filepath)
            texts = []
            for r in reports:
                impression = r['impression']
                findings = r['findings']
                if len(impression) > 0:
                    texts.append(impression)
                if len(findings) > 0:
                    texts.append(findings)
            logger.info(f"Loaded {len(texts)} texts from reports")
        sentences_list = sentence_tokenize_texts_in_parallel(texts)
        for sentences in sentences_list:
            for sentence in sentences:
                unique_sentences.add(sentence)
        logger.info(f"Found {len(unique_sentences)} unique sentences in reports")
        assert len(unique_sentences) > 0

        # Sort sentences
        unique_sentences = list(unique_sentences)
        unique_sentences = sort_sentences(unique_sentences, logger, args.rank_sentences_by_difficulty, cache_ranking=True)

        if not args.use_test_set_sentences:
            # Obtain kmeans cluster labels for sentences
            emb_extractor = CachedTextEmbeddingExtractor(
                model_name=args.cxr_bert_model_name,
                model_checkpoint_folder_path=args.cxr_bert_checkpoint_folder_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            kmeans_labels = emb_extractor.compute_kmeans_labels(unique_sentences, num_clusters=args.num_clusters, num_iterations=args.num_iterations)
            assert len(kmeans_labels) == len(unique_sentences)
            label2idxs = {}
            for i, label in enumerate(kmeans_labels):
                if label not in label2idxs:
                    label2idxs[label] = []
                label2idxs[label].append(i)
            # sort clusters by size
            sorted_idx_clusters = sorted(list(label2idxs.values()), key=lambda x: len(x), reverse=True)
            # flatten clusters into a list of indices, alternating between clusters
            sorted_indices = []
            for i in range(len(sorted_idx_clusters[0])):
                for cluster in sorted_idx_clusters:
                    if i < len(cluster):
                        sorted_indices.append(cluster[i])
                    else:
                        break
            assert len(sorted_indices) == len(unique_sentences), f"len(sorted_indices)={len(sorted_indices)} != len(unique_sentences)={len(unique_sentences)}"
            unique_sentences = [unique_sentences[i] for i in sorted_indices]
        
        # Print example sentences
        logger.info(f"Example sentences (immediately after clustering-based sorting):")
        for i in range(10):
            logger.info(f"{i+1}. {unique_sentences[i]}")
        
        # Load sentences to skip if they exist
        sentences_to_skip = set()
        if args.preprocessed_sentences_to_skip_filepaths is not None:
            for filepath in args.preprocessed_sentences_to_skip_filepaths:
                assert os.path.exists(filepath)
                rows = load_jsonl(filepath)
                if 'sentence' in rows[0]['metadata']:
                    sentences_to_skip.update(row['metadata']['sentence'] for row in rows) # backward compatibility
                else:
                    sentences_to_skip.update(row['metadata']['query'] for row in rows)
                logger.info(f"Loaded {len(rows)} sentences to skip from {filepath}")
        
        # Remove sentences to skip
        unique_sentences = [s for s in unique_sentences if s not in sentences_to_skip]
        logger.info(f"Removed {len(sentences_to_skip)} sentences to skip. {len(unique_sentences)} sentences remaining.")

        # Filter sentences by conditional sentences if necessary
        if args.only_conditional_sentences:
            logger.info(f"Filtering sentences to only conditional sentences")
            cond_regex = re.compile(r"\b(if|unless|when|whenever|wherever|whether|while|until|in case|as long as|provided that|given that|although|though|whereas|as soon as|as long as|as much as|as often as|as far as|as well as)\b", re.IGNORECASE)
            unique_sentences = [s for s in unique_sentences if cond_regex.search(s)]
            logger.info(f"Found {len(unique_sentences)} conditional sentences")

            # Print example sentences
            logger.info(f"Example sentences (after filtering to only conditional sentences):")
            for i in range(10):
                logger.info(f"{i+1}. {unique_sentences[i]}")

        # Adjust number of sentences to process if necessary
        assert 0 <= args.offset < len(unique_sentences)
        if args.offset + args.num_sentences > len(unique_sentences):
            logger.warning(f"Requested {args.num_sentences} sentences but only {len(unique_sentences) - args.offset} are available."
                        f" Using {len(unique_sentences) - args.offset} instead.")
            args.num_sentences = len(unique_sentences) - args.offset
            assert args.num_sentences > 0

        # Apply offset, num_sentences, and sample_sentences_uniformly
        if args.sample_sentences_uniformly:
            logger.info(f"Uniformly sampling {args.num_sentences} sentences starting from the {args.offset}-th sentence")
            unique_sentences = [unique_sentences[i] for i in np.linspace(args.offset, len(unique_sentences)-1, args.num_sentences, dtype=int)]
        else:
            logger.info(f"Collecting the first {args.num_sentences} sentences starting from the {args.offset}-th sentence")
            unique_sentences = unique_sentences[args.offset:args.offset + args.num_sentences]

        # Filter sentences by kth of every n sentences if necessary
        if args.process_kth_of_every_n_sentences is not None:
            k, n = args.process_kth_of_every_n_sentences
            assert 0 <= k < n
            logger.info(f"Filtering sentences to the {k}-th of every {n} sentences")
            unique_sentences = [x for i, x in enumerate(unique_sentences) if i % n == k]
            logger.info(f"Found {len(unique_sentences)} sentences that are the {k}-th of every {n}")

        # Remove already processed sentences
        logger.info(f"Removing {len(already_processed)} already processed sentences")
        sentences_to_process = [s for s in unique_sentences if s not in already_processed]
        if len(sentences_to_process) == 0:
            logger.info(f"All {len(unique_sentences)} sentences have already been processed. Nothing to do. Exiting.")
            sys.exit(0)

        logger.info(f"Total number of sentences to process: {len(sentences_to_process)}")

        # Print example sentences
        logger.info(f"Example sentences to process:")
        for i in np.linspace(0, len(sentences_to_process)-1, min(20, len(sentences_to_process)), dtype=int):
            logger.info(f"{i+1}. {sentences_to_process[i]}")

    else:
        if args.api_responses_filepath is not None:
            assert os.path.exists(args.api_responses_filepath)
        sentences_to_process = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=sentences_to_process,
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
        save_filepath=processed_sentences_save_filepath,
        use_batch_api=args.use_batch_api,
        batch_description=args.batch_description,
        batch_input_file_id=args.batch_input_file_id,
    )