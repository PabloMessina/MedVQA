import os
import argparse
import sys
import math
import random
import numpy as np
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from medvqa.datasets.text_data_utils import sentence_tokenize_texts_in_parallel
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.logging import get_console_logger
from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_LARGE_FAST_CACHE_DIR,
)
from medvqa.utils.nlp import sort_sentences
from medvqa.utils.openai_api import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files import get_file_path_with_hashing_if_too_long, load_json, load_jsonl, load_pickle, save_pickle

# INSTRUCTIONS = """Context: natural language inference.

# Given a premise and a hypothesis, output "entailment", "contradiction", or "neutral".

# Use "entailment" when the facts stated by the premise necessarily entail the truth of the hypothesis.

# Use "contradiction" when premise and hypothesis are mutually exclusive/contradictory (both cannot be true at the same time).

# Use "neutral", if there is no contradiction (premise and hypothesis are compatible), but the premise does not entail the hypothesis (it's possible for the premise to be true and the hypothesis still be false). In other words, use "neutral" when neither "entailment" nor "contradiction" adequately fit."""

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

print(INSTRUCTIONS)

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    text = text.lower()
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
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
    parser.add_argument("--preprocessed_texts_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--preprocessed_reports_filepath", type=str, required=True)
    
    parser.add_argument("--cxr_bert_model_name", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--cxr_bert_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_clusters", type=int, default=200)
    parser.add_argument("--num_iterations", type=int, default=300)
    parser.add_argument("--num_pairs_to_generate", type=int, default=100000)

    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_texts", type=int, required=True)
    parser.add_argument("--sample_texts_uniformly", action="store_true", default=False)
    parser.add_argument("--process_kth_of_every_n_texts", type=int, nargs=2, default=None,
                        help="If specified, only process the kth of every n texts.")

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    parser.add_argument("--not_delete_api_requests_and_responses", action="store_true", default=False)
    args = parser.parse_args()

    processed_texts_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_nli_queries_from_clusters{args.alias}.jsonl")
    
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

        nli_queries_filepath = get_file_path_with_hashing_if_too_long(
            folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
            prefix="nli_queries",
            strings=[
                f'{args.preprocessed_reports_filepath}',
                f'{args.cxr_bert_model_name}',
                f'{args.cxr_bert_checkpoint_folder_path}',
                f'{args.num_clusters}',
                f'{args.num_iterations}',           
                f'{args.num_pairs_to_generate}',
             ],
             force_hashing=True,
        )

        if os.path.exists(nli_queries_filepath): # load cached pairs if they exist
            nli_queries = load_pickle(nli_queries_filepath)
            logger.info(f"Loaded {len(nli_queries)} NLI pairs from {nli_queries_filepath}")
        else:

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

            # Obtain kmeans cluster labels for sentences
            emb_extractor = CachedTextEmbeddingExtractor(
                model_name=args.cxr_bert_model_name,
                model_checkpoint_folder_path=args.cxr_bert_checkpoint_folder_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            kmeans_labels = emb_extractor.compute_kmeans_labels(unique_sentences, num_clusters=args.num_clusters,
                                                                num_iterations=args.num_iterations)
            assert len(kmeans_labels) == len(unique_sentences)
            label2idxs = {}
            for i, label in enumerate(kmeans_labels):
                if label not in label2idxs:
                    label2idxs[label] = []
                label2idxs[label].append(i)
            logger.info(f"Found {len(label2idxs)} clusters")
            # sort clusters by size
            sorted_idx_clusters = sorted(list(label2idxs.values()), key=lambda x: len(x), reverse=True)

            # Generate pairs of sentences from clusters
            # 2 cases:
            # 1) pairs of sentences from the same cluster
            # 2) pairs of sentences from different clusters

            # 1) pairs of sentences from the same cluster
            n = int(args.num_pairs_to_generate * 0.8)
            n_per_cluster = math.ceil(n / len(sorted_idx_clusters))
            pairs = []
            seen = set()
            for cluster in tqdm(sorted_idx_clusters, total=len(sorted_idx_clusters), mininterval=2):
                s = int(len(cluster) * 0.7)
                e = int(len(cluster) * 0.95)
                for _ in range(n_per_cluster):
                    for _ in range(10):
                        idx1 = cluster[random.randint(s, e)]
                        idxs2 = random.sample(cluster, 100)
                        idx2 = None
                        min_dist = 1000
                        for idx in idxs2:
                            if idx == idx1:
                                continue
                            if (idx1, idx) in seen:
                                continue
                            dist = levenshtein_distance(unique_sentences[idx1], unique_sentences[idx])
                            if dist < min_dist:
                                min_dist = dist
                                idx2 = idx
                        if idx2 is None:
                            continue
                        p = (idx1, idx2)
                        if idx1 != idx2 and p not in seen:
                            seen.add(p)
                            pairs.append(p)
                            break
            # 2) pairs of sentences from different clusters
            n = args.num_pairs_to_generate - len(pairs)
            n_per_cluster = math.ceil(n / (len(sorted_idx_clusters) * (len(sorted_idx_clusters) - 1)))
            for i in tqdm(range(len(sorted_idx_clusters)), total=len(sorted_idx_clusters), mininterval=2):
                cluster1 = sorted_idx_clusters[i]
                s1 = int(len(cluster1) * 0.7)
                e1 = int(len(cluster1) * 0.95)
                for j in range(len(sorted_idx_clusters)):
                    if i == j:
                        continue
                    cluster2 = sorted_idx_clusters[j]
                    for _ in range(n_per_cluster):
                        for _ in range(10):
                            idx1 = cluster1[random.randint(s1, e1)]
                            idxs2 = random.sample(cluster2, 100)
                            idx2 = None
                            min_dist = 1000
                            for idx in idxs2:
                                if idx == idx1:
                                    continue
                                if (idx1, idx) in seen:
                                    continue
                                dist = levenshtein_distance(unique_sentences[idx1], unique_sentences[idx])
                                if dist < min_dist:
                                    min_dist = dist
                                    idx2 = idx
                            if idx2 is None:
                                continue
                            p = (idx1, idx2)
                            if idx1 != idx2 and p not in seen:
                                seen.add(p)
                                pairs.append(p)
                                break
            # shuffle pairs
            random.shuffle(pairs)
            logger.info(f"Generated {len(pairs)} pairs of sentences")
            nli_queries = [f'#P: {unique_sentences[i]} | #H: {unique_sentences[j]}' for i, j in pairs]

            # Save nli_queries
            logger.info(f"Saving {len(nli_queries)} NLI pairs to {nli_queries_filepath}")
            save_pickle(nli_queries, nli_queries_filepath)
        
        # Load texts to skip if they exist
        texts_to_skip = set()
        if args.preprocessed_texts_to_skip_filepaths is not None:
            for filepath in args.preprocessed_texts_to_skip_filepaths:
                assert os.path.exists(filepath)
                rows = load_jsonl(filepath)
                logger.info(f"Loaded {len(rows)} texts to skip from {filepath}")
                if 'metadata' in rows[0]:
                    texts_to_skip.update(row['metadata']['query'] for row in rows)
                else:
                    for row in rows:
                        text = f'#P: {row["premise"]} | #H: {row["hypothesis"]}'
                        texts_to_skip.add(text)
        
        # Remove texts to skip
        nli_queries = [q for q in nli_queries if q not in texts_to_skip]
        logger.info(f"Removed {len(texts_to_skip)} texts to skip. {len(nli_queries)} texts remaining.")

        # Adjust number of texts to process if necessary
        assert 0 <= args.offset < len(nli_queries)
        if args.offset + args.num_texts > len(nli_queries):
            logger.warning(f"Requested {args.num_texts} texts but only {len(nli_queries) - args.offset} are available."
                        f" Using {len(nli_queries) - args.offset} instead.")
            args.num_texts = len(nli_queries) - args.offset
            assert args.num_texts > 0

        # Apply offset and num_texts
        logger.info(f"Collecting the first {args.num_texts} texts starting from the {args.offset}-th sentence")
        nli_queries = nli_queries[args.offset:args.offset + args.num_texts]

        # Filter texts by kth of every n texts if necessary
        if args.process_kth_of_every_n_texts is not None:
            k, n = args.process_kth_of_every_n_texts
            assert 0 <= k < n
            logger.info(f"Filtering texts to the {k}-th of every {n} texts")
            nli_queries = [x for i, x in enumerate(nli_queries) if i % n == k]
            logger.info(f"Found {len(nli_queries)} texts that are the {k}-th of every {n}")

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
        delete_api_requests_and_responses=not args.not_delete_api_requests_and_responses,
    )