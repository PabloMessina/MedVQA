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
from medvqa.utils.nlp import sort_sentences
from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_TMP_DIR,
    MIMICCXR_FAST_CACHE_DIR,
)
from medvqa.utils.openai_api import (
    GPT_IS_ACTING_WEIRD_REGEX,
    run_common_boilerplate_for_api_requests,
)
from medvqa.utils.logging import get_console_logger
from medvqa.utils.files import load_jsonl, load_pickle

INSTRUCTIONS = """Instructions:

Apply an annotation standard of observations to raw phrases extracted from chest X-ray reports.
The standard considers the following observations:

airspace opacity
atelectasis
bone lesion
bronchiectasis
calcified nodule
clavicle fracture
consolidation
costophrenic angle blunting
cyst/bullae
diaphragmatic eventration (benign)
elevated hemidiaphragm
enlarged cardiac silhouette
enlarged hilum
hernia
hydropneumothorax
hyperaeration
increased reticular markings/ild pattern
infiltration
linear/patchy atelectasis
lobar/segmental collapse
lung lesion
lung opacity
mass/nodule (not otherwise specified)
mediastinal displacement
mediastinal widening
multiple masses/nodules
pleural effusion
pleural/parenchymal scarring
pneumomediastinum
pneumothorax
pulmonary edema/hazy opacity
rib fracture
scoliosis
shoulder osteoarthritis
spinal degenerative changes
spinal fracture
sub-diaphragmatic air
subcutaneous air
superior mediastinal mass/enlargement
tortuous aorta
vascular calcification
vascular congestion
vascular redistribution
aortic graft/repair
cabg grafts
cardiac pacer and wires
prosthetic valve
alveolar hemorrhage
aspiration
copd/emphysema
fluid overload/heart failure
goiter
granulomatous disease
interstitial lung disease
lung cancer
pericardial effusion
pneumonia
artifact
breast/nipple shadows
low lung volumes
rotated
skin fold
chest port
chest tube
endotracheal tube
enteric tube
ij line
intra-aortic balloon pump
mediastinal drain
picc
pigtail catheter
subclavian line
swan-ganz catheter
tracheostomy tube

Output format:

JSON array of strings

Rules:

Only output the observations from the standard that best match or are supported by the phrase. If none match the phrase, output an empty array ([]).

Example:
Increased bibasilar opacities likely representing atelectasis, but cannot exclude aspiration or pneumonia in the
correct clinical setting
[
"lung opacity",
"atelectasis",
"aspiration",
"pneumonia"
]"""

# Match a possibly truncated JSON array of strings, by matching as many strings as possible.
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
    assert all(isinstance(x, str) for x in list_of_strings), f"Could not parse output: {text}"
    return list_of_strings

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--preprocessed_sentences_to_skip_filepaths", nargs="+", default=None)    
    parser.add_argument("--integrated_fact_metadata_filepath", type=str, required=True)
    parser.add_argument('--paraphrases_jsonl_filepaths', type=str, nargs='+', default=None)
    
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

    processed_sentences_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_chest_imagenome_observations_from_sentences{args.alias}.jsonl")
    
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
        rows = load_jsonl(args.integrated_fact_metadata_filepath)
        logger.info(f"Loaded {len(rows)} rows from {args.integrated_fact_metadata_filepath}")
        for row in tqdm(rows, mininterval=2):
            fact = row['fact']
            metadata = row['metadata']
            short_obs = metadata['short observation']
            detailed_obs = metadata['detailed observation']
            for x in [fact, short_obs, detailed_obs]:
                if x:
                    unique_sentences.add(x)

        # Collect sentences from paraphrases
        if args.paraphrases_jsonl_filepaths is not None:
            for filepath in args.paraphrases_jsonl_filepaths:
                assert os.path.exists(filepath)
                rows = load_jsonl(filepath)
                logger.info(f"Loaded {len(rows)} rows from {filepath}")
                for row in tqdm(rows, mininterval=2):
                    s = next(iter(row['metadata'].values()))
                    parsed_response = row['parsed_response']
                    if type(parsed_response) == list:
                        p = parsed_response
                    elif type(parsed_response) == dict:
                        assert 'positives' in parsed_response and 'negatives' in parsed_response
                        p = parsed_response['positives'] + parsed_response['negatives']
                    else:
                        raise ValueError(f'Unknown type {type(parsed_response)}')
                    p.append(s)
                    for x in p:
                        if len(x) > 0 and any(c.isalpha() for c in x):
                            unique_sentences.add(x)
        
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
                size_before = len(sentences_to_skip)
                if filepath.endswith(".jsonl"):
                    rows = load_jsonl(filepath)
                    if 'sentence' in rows[0]['metadata']:
                        sentences_to_skip.update(row['metadata']['sentence'] for row in rows) # backward compatibility
                    else:
                        sentences_to_skip.update(row['metadata']['query'] for row in rows)
                elif filepath.endswith(".pkl"):
                    data = load_pickle(filepath)
                    sentences_to_skip.update(data['phrases'])
                else:
                    raise ValueError(f"Unknown file extension: {filepath}")
                logger.info(f"Loaded {len(sentences_to_skip) - size_before} sentences to skip from {filepath}")
        
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