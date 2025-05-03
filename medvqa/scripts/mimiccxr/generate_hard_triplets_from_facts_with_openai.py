from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import re
import sys
import json
import numpy as np
from tqdm import tqdm

from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.logging_utils import get_console_logger
from medvqa.utils.nlp import sort_sentences
from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import load_jsonl

INSTRUCTIONS = """Given an anchor, output a list of positives and a list of negatives.
The anchor will come from a chest X-ray report. The positives must be 6 paraphrases of the anchor,
expressing the same idea with different terms and synonyms (cover a wide range of medical terminology).
The negatives must be 10 hard negatives, that is, similar to the anchor but semantically different,
for example, by expressing a different diagnosis (if applicable), by referring to a different anatomical
location (if applicable), etc.

Output format: a JSON object as follows
{
"positives": [ ... ],
"negatives": [ ... ]
}"""

# Regexes for parsing the output of the OpenAI API call
# Example output:
# { "positives": ["The heart is normal in size"], "negatives": ["The heart is enlarged"] }
_JSON_FORMAT_REGEX = re.compile(r'^\{\s*"positives"\s*:\s*\[\s*"[^"]+"(\s*,\s*"[^"]+")*\s*\]\s*,\s*"negatives"\s*:\s*\[\s*"[^"]+"(\s*,\s*"[^"]+")*\s*\]\s*\}$')

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = _JSON_FORMAT_REGEX.search(text) # match a JSON list of strings
    if not match:
        if GPT_IS_ACTING_WEIRD_REGEX.search(text):
            raise RuntimeError(f"GPT is protesting: {text}")
    assert match, f"Could not parse output: {text}"
    data = json.loads(match.group(0))
    assert isinstance(data, dict), f"Could not parse output: {text}"
    assert "positives" in data and "negatives" in data, f"Could not parse output: {text}"
    assert isinstance(data["positives"], list) and isinstance(data["negatives"], list), f"Could not parse output: {text}"
    assert len(data["positives"]) >= 0 and len(data["negatives"]) > 0, f"Could not parse output: {text}"
    assert all(isinstance(x, str) for x in data["positives"]), f"Could not parse output: {text}"
    assert all(isinstance(x, str) for x in data["negatives"]), f"Could not parse output: {text}"
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)    
    parser.add_argument("--integrated_fact_metadata_filepath", type=str, required=True)
    
    parser.add_argument("--cxr_bert_model_name", type=str, default="BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--cxr_bert_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_clusters", type=int, default=300)
    parser.add_argument("--num_iterations", type=int, default=300)

    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_sentences", type=int, required=True)
    parser.add_argument("--rank_sentences_by_difficulty", action="store_true", default=False)
    parser.add_argument("--sample_sentences_uniformly", action="store_true", default=False)
    parser.add_argument("--process_kth_of_every_n_sentences", type=int, nargs=2, default=None,
                        help="If specified, only process the kth of every n sentences.")

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

    processed_sentences_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_hard_triplets_from_facts{args.alias}.jsonl")
    
    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None:

        # Load already processed sentences if they exist
        already_processed = set()
        if os.path.exists(processed_sentences_save_filepath):
            rows = load_jsonl(processed_sentences_save_filepath)
            for row in rows:
                already_processed.add(row['metadata']['query'])
            logger.info(f"Loaded {len(rows)} already processed sentences from {processed_sentences_save_filepath}")

        # Collect sentences from integrated_fact_metadata
        unique_sentences = set()
        assert os.path.exists(args.integrated_fact_metadata_filepath)
        logger.info(f"Loading sentences from {args.integrated_fact_metadata_filepath}")
        rows = load_jsonl(args.integrated_fact_metadata_filepath)
        for row in tqdm(rows, mininterval=2):
            fact = row['fact']
            metadata = row['metadata']
            anatloc = metadata['anatomical location']
            short_obs = metadata['short observation']
            detailed_obs = metadata['detailed observation']
            for x in [fact, anatloc, short_obs, detailed_obs]:
                if x:
                    unique_sentences.add(x)
        logger.info(f"Loaded {len(rows)} rows from {args.integrated_fact_metadata_filepath}")
        logger.info(f"Found {len(unique_sentences)} unique sentences")
        assert len(unique_sentences) > 0

        # Sort sentences
        unique_sentences = list(unique_sentences)
        unique_sentences = sort_sentences(unique_sentences, logger, args.rank_sentences_by_difficulty, cache_ranking=True)

        # Obtain kmeans cluster labels for sentences
        emb_extractor = CachedTextEmbeddingExtractor(
            model_name=args.cxr_bert_model_name,
            model_checkpoint_folder_path=args.cxr_bert_checkpoint_folder_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        kmeans_labels = emb_extractor.compute_kmeans_labels(unique_sentences, n_clusters=args.num_clusters, num_iterations=args.num_iterations)
        assert len(kmeans_labels) == len(unique_sentences)
        label2idxs = {}
        for i, label in enumerate(kmeans_labels):
            if label not in label2idxs:
                label2idxs[label] = []
            label2idxs[label].append(i)
        logger.info(f"Found {len(label2idxs)} clusters")
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
    )