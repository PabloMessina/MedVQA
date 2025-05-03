from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import re
import sys
import json
import numpy as np
from tqdm import tqdm

from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api_utils import (
    GPT_IS_ACTING_WEIRD_REGEX,
    run_common_boilerplate_for_api_requests,
)
from medvqa.utils.logging_utils import get_console_logger
from medvqa.utils.files_utils import load_jsonl, load_pickle

# INSTRUCTIONS = """Given a sentence describing a fact from a chest X-ray report, output a JSON array of strings paraphrasing it,
# covering a wide diverse range of terminology, synonyms and abbreviations that radiologists commonly use to express the same idea.

# Examples:

# benign calcification
# [
# "non-cancerous calcification",
# "harmless calcification",
# "innocuous calcification",
# "benign calcified lesion",
# "non-malignant calcification",
# "non-threatening calcification",
# "not indicative of cancer calcification",
# "safe calcification",
# "non-dangerous calcification",
# "non-metastatic calcification"
# ]

# osteoporosis
# [
# "decreased bone density",
# "brittle bones",
# "low bone mass",
# "thinning of the bones",
# "weakening of the bones",
# "porous bones",
# "fragile bones",
# "reduced bone strength",
# "loss of bone density",
# "degenerative bone disease"
# ]

# no osteoporosis
# [
# "normal bone density",
# "healthy bones",
# "adequate bone mass",
# "strong bones",
# "normal bone strength",
# "no signs of osteoporosis",
# "absence of osteoporosis",
# "no evidence of bone thinning",
# "no indication of bone weakening",
# "no osteoporotic changes",
# "no degenerative bone disease"
# ]"""

INSTRUCTIONS = """Given a fact extracted from a Chest X-ray report, output a JSON array of strings.
Each string must be a paraphrased fact, with similar semantics, expressing the same finding/diagnosis.
These paraphrases will be used to train a fact embedding with triplet loss.

Generate multiple examples (at least 14), covering a wide variety and diversity of radiological
terms, synonyms, abbreviations and expressions that radiologists commonly use to communicate the same idea.

Output format:
[
"paraphrase 1",
"paraphrase 2",
...
"paraphrase 14"
]"""

# Match a possibly truncated JSON list of strings, by matching as many strings as possible.
# This is useful because the OpenAI API sometimes truncates the output.
_JSON_STRING_ARRAY_REGEX = re.compile(r'^\[\s*(\".+?\"(\s*,\s*\".+?\")*)?\s*\]?')

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    match = _JSON_STRING_ARRAY_REGEX.search(text) # match a JSON list of strings
    if not match and GPT_IS_ACTING_WEIRD_REGEX.search(text):
        logger.warning(f"GPT is protesting: {text}")
    assert match, f"Could not parse output: {text}"
    string = match.group(0)
    assert string[0] == "[", f"Could not parse output: {text}"
    if string[-1] != "]": string += "]" # add closing bracket
    list_of_strings = json.loads(string)
    assert isinstance(list_of_strings, list), f"Could not parse output: {text}"
    assert all(isinstance(fact, str) for fact in list_of_strings), f"Could not parse output: {text}"
    # remove ending punctuation
    list_of_strings = [fact[:-1] if fact[-1] in ".?!" else fact for fact in list_of_strings]
    return list_of_strings

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--integrated_fact_metadata_filepath", type=str, default=None)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)    
    parser.add_argument("--precomputed_sentence_embeddings_filepath", type=str, required=True)
    parser.add_argument("--precomputed_clusters_filepath", type=str, required=True)
    parser.add_argument("--num_sentences_per_cluster", type=int, required=True)
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num_words_per_sentence", type=int, default=None)
    group.add_argument("--min_num_words_per_sentence", type=int, default=None)

    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--num_sentences", type=int, required=True)
    parser.add_argument("--process_kth_of_every_n_sentences", type=int, nargs=2, default=None,
                        help="If specified, only process the kth of every n sentences.")

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    # Load already paraphrased sentences if they exist
    paraphrased_observations_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_paraphrased_observations{args.alias}.jsonl")

    if args.api_responses_filepath is None:
        
        already_paraphrased = set()
        if os.path.exists(paraphrased_observations_filepath):
            rows = load_jsonl(paraphrased_observations_filepath)
            for row in rows:
                already_paraphrased.add(row['metadata']['query'])
            logger.info(f"Loaded {len(already_paraphrased)} already paraphrased sentences from {paraphrased_observations_filepath}")
        
        paraphrases_to_skip = set()
        if args.preprocessed_sentences_to_skip_filepaths is not None:
            for filepath in args.preprocessed_sentences_to_skip_filepaths:
                assert os.path.exists(filepath)
                rows = load_jsonl(filepath)
                size_before = len(paraphrases_to_skip)
                for row in rows:
                    paraphrases_to_skip.add(row['metadata']['observation'])
                logger.info(f"Loaded {len(paraphrases_to_skip) - size_before} paraphrased sentences to skip from {filepath}")

        # Collect observations from metadata
        unique_observations = set()
        assert os.path.exists(args.integrated_fact_metadata_filepath)
        logger.info(f"Loading facts metadata from {args.integrated_fact_metadata_filepath}")
        integrated_fact_metadata = load_jsonl(args.integrated_fact_metadata_filepath)
        for r in tqdm(integrated_fact_metadata, total=len(integrated_fact_metadata), mininterval=2):
            fact = r['fact']
            detailed_observation = r['metadata']['detailed observation']
            short_observation = r['metadata']['short observation']
            for x in (fact, detailed_observation, short_observation):
                if len(x) > 0 and any(c.isalpha() for c in x) and x not in paraphrases_to_skip:
                    unique_observations.add(x)
        
        unique_observations = list(unique_observations)
        logger.info(f"Found {len(unique_observations)} unique observations")
        assert len(unique_observations) > 0

        # Filter observations by length if necessary
        if args.num_words_per_sentence is not None:
            logger.info(f"Filtering observations to those with {args.num_words_per_sentence} words")
            unique_observations = [x for x in unique_observations if len(x.split()) == args.num_words_per_sentence]
            logger.info(f"Found {len(unique_observations)} observations with {args.num_words_per_sentence} words")
        
        if args.min_num_words_per_sentence is not None:
            logger.info(f"Filtering observations to those with at least {args.min_num_words_per_sentence} words")
            unique_observations = [x for x in unique_observations if len(x.split()) >= args.min_num_words_per_sentence]
            logger.info(f"Found {len(unique_observations)} observations with at least {args.min_num_words_per_sentence} words")

        # Group observations by cluster
        logger.info(f"Loading clusters from {args.precomputed_clusters_filepath}")
        clusters = load_pickle(args.precomputed_clusters_filepath)
        logger.info(f"Loading sentence embeddings from {args.precomputed_sentence_embeddings_filepath}")
        sentence_embeddings = load_pickle(args.precomputed_sentence_embeddings_filepath)
        obs2cluster = {}
        for obs_idx, cluster in zip(clusters['observations']['indices'], clusters['observations']['clusters']):
            obs = sentence_embeddings['sentences'][obs_idx]
            obs2cluster[obs] = cluster
        cluster2observations = {}
        for obs in unique_observations:
            cid = obs2cluster[obs]
            if cid not in cluster2observations:
                cluster2observations[cid] = []
            cluster2observations[cid].append(obs)
        logger.info(f"Found {len(cluster2observations)} clusters")

        # Select observations from each cluster
        selected_observations = []
        for observations in cluster2observations.values():
            if len(observations) <= args.num_sentences_per_cluster:
                selected_observations.extend(observations) # no need to sample
                logger.warning(f"Cluster with {len(observations)} observations has fewer"
                            f" observations than requested {args.num_sentences_per_cluster}.")
                logger.warning(f"Example observations: {observations[:2]}")

            else:
                observations.sort(key=lambda s: (len(s), s)) # sort by length and then alphabetically
                selected_observations.extend(observations[i] for i in np.linspace(0, len(observations)-1,
                                                                                args.num_sentences_per_cluster, dtype=int))
        selected_observations.sort(key=lambda s: (len(s), s)) # sort by length and then alphabetically
        logger.info(f"Selected {len(selected_observations)} observations from {len(cluster2observations)} clusters")

        # Filter observations by kth of every n sentences if necessary
        if args.process_kth_of_every_n_sentences is not None:
            k, n = args.process_kth_of_every_n_sentences
            assert 0 <= k < n
            logger.info(f"Filtering observations to the {k}-th of every {n} sentences")
            selected_observations = [x for i, x in enumerate(selected_observations) if i % n == k]
            logger.info(f"Found {len(selected_observations)} observations that are the {k}-th of every {n} sentences")

        # Adjust number of sentences to paraphrase if necessary
        assert 0 <= args.offset < len(selected_observations)
        if args.offset + args.num_sentences > len(selected_observations):
            logger.warning(f"Requested {args.num_sentences} sentences but only {len(selected_observations) - args.offset} are available."
                        f" Using {len(selected_observations) - args.offset} instead.")
            args.num_sentences = len(selected_observations) - args.offset
            assert args.num_sentences > 0

        # Collect sentences to paraphrase
        logger.info(f"Collecting the first {args.num_sentences} sentence from the {args.offset}-th sentence")
        sentences_to_paraphrase = [selected_observations[i] for i in range(args.offset, args.offset + args.num_sentences)
                                    if selected_observations[i] not in already_paraphrased]
        if len(sentences_to_paraphrase) == 0:
            logger.info(f"All {args.num_sentences} sentences have already been paraphrased. Nothing to do. Exiting.")
            sys.exit(0)

        logger.info(f"Total number of sentences to paraphrase: {len(sentences_to_paraphrase)}")
        
        # Print example sentences
        logger.info(f"Example sentences to paraphrase:")
        for i in np.linspace(0, len(sentences_to_paraphrase)-1, min(10, len(sentences_to_paraphrase)), dtype=int):
            logger.info(f"{i+1}. {sentences_to_paraphrase[i]}")

    else:
        assert os.path.exists(args.api_responses_filepath)
        sentences_to_paraphrase = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=sentences_to_paraphrase,
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
        save_filepath=paraphrased_observations_filepath,
    )