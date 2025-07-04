import os
import argparse
import sys
import json
import numpy as np
from tqdm import tqdm
from medvqa.utils.text_data_utils import sentence_tokenize_texts_in_parallel
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.logging_utils import get_console_logger
from medvqa.utils.nlp import sort_sentences
from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import load_json, load_jsonl

INSTRUCTIONS = """Given a premise from a Chest X-ray report, output 5 statements that explicitly logically contradict the premise, that is, the premise and any generated statement cannot both be true at the same time. Include at least one sentence that is very similar to the premise but with a very slight difference, thus creating a logical contradiction.
For example:
- "There is mid thoracic dextroscoliosis" vs. "There is no mid thoracic dextroscoliosis"
- "Left basal consolidation has slightly improved" vs. "Left basal consolidation has slightly increased"
Output format: a JSON array of strings."""

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    data = json.loads(text)
    assert isinstance(data, list), f"Could not parse output: {text}"
    assert all(isinstance(x, str) for x in data), f"Could not parse output: {text}"
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--preprocessed_reports_filepath", type=str, required=True)
    
    parser.add_argument("--cxr_bert_model_name", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--cxr_bert_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_clusters", type=int, default=200)
    parser.add_argument("--num_iterations", type=int, default=300)

    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_sentences", type=int, required=True)
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

    processed_sentences_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_nli_contradiction_examples{args.alias}.jsonl")
    
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

        # Collect sentences from reports        
        texts = []
        assert os.path.exists(args.preprocessed_reports_filepath)
        logger.info(f"Loading preprocessed reports from {args.preprocessed_reports_filepath}")
        reports = load_json(args.preprocessed_reports_filepath)
        for r in tqdm(reports, total=len(reports), mininterval=2):
            impression = r['impression']
            findings = r['findings']
            if len(impression) > 0:
                texts.append(impression)
            if len(findings) > 0:
                texts.append(findings)
        logger.info(f"Loaded {len(reports)} reports from {args.preprocessed_reports_filepath}")
        logger.info(f"Loaded {len(texts)} texts from reports")
        unique_sentences = set()
        sent_tokenized_texts = sentence_tokenize_texts_in_parallel(texts)
        for sentences in sent_tokenized_texts:
            unique_sentences.update(sentences)
        logger.info(f"Found {len(unique_sentences)} unique sentences in reports")
        assert len(unique_sentences) > 0

        # Sort sentences
        unique_sentences = list(unique_sentences)
        unique_sentences = sort_sentences(unique_sentences, logger, by_difficulty=True, cache_ranking=True)
        unique_sentences = unique_sentences[::-1] # reverse order -> easiest first

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
        for i in range(100):
            si = sorted_indices[i]
            logger.info(f"{i+1}. c={kmeans_labels[si]}, c.size={len(label2idxs[kmeans_labels[si]])}, s={unique_sentences[i]}")
        
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