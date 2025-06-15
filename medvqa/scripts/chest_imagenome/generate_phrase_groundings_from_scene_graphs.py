import argparse
import functools
from multiprocessing import Pool
import os
import sys
import json
import logging
from typing import List
import numpy as np
import pandas as pd
import Levenshtein
import nltk
# import time
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAMES, CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_dicom_ids,
    load_scene_graph,
)
from medvqa.datasets.mimiccxr import MIMICCXR_METADATA_CSV_PATH
from medvqa.utils.files_utils import get_file_path_with_hashing_if_too_long, load_jsonl, load_pickle, save_jsonl, save_pickle
from medvqa.utils.logging_utils import setup_logging
from medvqa.utils.openai_api_utils import run_common_boilerplate_for_api_requests
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.metrics.bbox.utils import compute_bbox_union_iou

# Download NLTK data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt", quiet=True)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING) # Suppress httpx logs


SYSTEM_INSTRUCTIONS = """You have a predefined list of anatomical locations:

anatomical_locations = [
"right lung", "right upper lung zone", "right mid lung zone", "right lower lung zone",
"right hilar structures", "right apical zone", "right costophrenic angle", "right cardiophrenic angle",
"right hemidiaphragm", "left lung", "left upper lung zone", "left mid lung zone", "left lower lung zone",
"left hilar structures", "left apical zone", "left costophrenic angle", "left hemidiaphragm", "trachea",
"spine", "right clavicle", "left clavicle", "aortic arch", "mediastinum", "upper mediastinum", "svc",
"cardiac silhouette", "left cardiac silhouette", "right cardiac silhouette", "cavoatrial junction",
"right atrium", "descending aorta", "carina", "left upper abdomen", "right upper abdomen", "abdomen",
"left cardiophrenic angle"
]

Given a phrase and its annotated locations from the list above, output a JSON object with:

- reason: Brief explanation of your reasoning.
- has_localizable_abnormalities: "yes" or "no" — does the phrase describe one or more abnormalities localizable in the frontal chest X-ray?
- accurate (if has_localizable_abnormalities is "yes"): "yes" or "no" — are the phrase and locations accurate and not needing revision?
- too_verbose (if has_localizable_abnormalities is "yes"): "yes" or "no" — is the phrase too verbose and needs revision?
- improvements (if accurate is "no" or too_verbose is "yes"): List of objects, each with a revised or split phrase and its most accurate anatomical locations from the list, e.g.:
  {"phrase": "{revised_phrase}", "locations": ["{anatomical_location_1}", ...]}

Revised locations should be those most likely to overlap with the finding(s) (usually 1–3). Even if locations are correct, the phrase may need revision. Split complex phrases; simplify verbose ones. For example, "Right apical pneumothorax is still present as compared to previous chest x-rays with no significant interval change" can be simplified to "Right apical pneumothorax"; "IMPRESSION: Mild bibasilar atelectasis" can be simplified to "Mild bibasilar atelectasis".

Example Input:
{"phrase": "2. Increase in left perihilar and lower lobe opacities.", "locations": ["left hilar structures", "left lower lung zone", "left lung"]}

Example Output:
{"reason": "The phrase mentions two regions—perihilar and lower lobe—so it is split for clarity. 'Left perihilar opacities' maps to 'left hilar structures' and 'left lung'; 'Left lower lobe opacities' maps to 'left lower lung zone' and 'left lung'. This provides more granular, location-specific findings.",
"is_abnormal": "yes",
"accurate": "yes",
"too_verbose": "yes",
"improvements": [
{"phrase": "Left perihilar opacities.", "locations": ["left hilar structures", "left lung"]},
{"phrase": "Left lower lobe opacities.", "locations": ["left lower lung zone", "left lung"]}
]}"""

_VALID_ANATOMICAL_LOCATIONS = set(CHEST_IMAGENOME_BBOX_NAMES)

def parse_llm_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find a valid JSON object in: {text}")
    json_str = text[start:end+1]
    data = json.loads(json_str)
    # Ensure the output is a dictionary
    assert isinstance(data, dict), f"Output is not a dictionary: {data}"
    # Ensure the required keys are present
    required_keys = ["reason", "has_localizable_abnormalities"]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in output: {data}"
    # Ensure the values are of the expected types
    assert isinstance(data["reason"], str), f"Invalid type for 'reason': {data['reason']}"
    assert data["has_localizable_abnormalities"] in ["yes", "no"], \
        f"Invalid value for 'has_localizable_abnormalities': {data['has_localizable_abnormalities']}"
    if data["has_localizable_abnormalities"] == "yes":
        assert "accurate" in data, f"Missing key 'accurate' in output: {data}"
        assert data["accurate"] in ["yes", "no"], f"Invalid value for 'accurate': {data['accurate']}"
        assert "too_verbose" in data, f"Missing key 'too_verbose' in output: {data}"
        assert data["too_verbose"] in ["yes", "no"], f"Invalid value for 'too_verbose': {data['too_verbose']}"
        if data["accurate"] == "no" or data["too_verbose"] == "yes":
            assert "improvements" in data, f"Missing key 'improvements' in output: {data}"
            assert isinstance(data["improvements"], list), \
                f"Invalid type for 'improvements': {data['improvements']}"
            for improvement in data["improvements"]:
                assert isinstance(improvement, dict), f"Invalid type for improvement: {improvement}"
                assert "phrase" in improvement, f"Missing key 'phrase' in improvement: {improvement}"
                assert "locations" in improvement, f"Missing key 'locations' in improvement: {improvement}"
                assert isinstance(improvement["locations"], list), \
                    f"Invalid type for 'locations': {improvement['locations']}"
                for location in improvement["locations"]:
                    assert isinstance(location, str), f"Invalid type for location: {location}"
                    assert location in _VALID_ANATOMICAL_LOCATIONS, f"Invalid anatomical location: {location}"
    return data

def extract_abnormal_phrase_groundings_from_scene_graphs(args):
    """
    Extracts mappings from abnormal phrases to anatomical locations and their bounding boxes
    from Chest ImaGenome scene graphs.

    For each DICOM image:
        - Loads the scene graph and image dimensions.
        - For each phrase marked as 'abnormal' in the scene graph, records the anatomical
          locations (by bbox_name) it is associated with.
        - For each anatomical location, records the normalized bounding box coordinates.

    The result is a list of dictionaries, each containing:
        - 'dicom_id': The DICOM image identifier.
        - 'phrase2locations': Mapping from abnormal phrase text to a list of bbox_names.
        - 'location2bbox': Mapping from bbox_name to normalized (x1, y1, x2, y2) coordinates.

    The output is saved as a pickle file in the CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR.
    """
    
    logger.info("Extracting abnormal phrase-to-location mappings from scene graphs...")
    
    # Load metadata for image dimensions
    df = pd.read_csv(MIMICCXR_METADATA_CSV_PATH)
    dicom_id_to_width_height = {
        dicom_id: (width, height)
        for dicom_id, width, height in zip(df.dicom_id, df.Columns, df.Rows)
    }

    # Get all valid dicom_ids
    dicom_ids = load_chest_imagenome_dicom_ids()

    output_list = []

    for dicom_id in tqdm(dicom_ids, desc="Extracting mapping from scene graphs", mininterval=1, ncols=100):
        # Load scene graph for this dicom_id
        scene_graph = load_scene_graph(dicom_id)

        # Get image dimensions
        width, height = dicom_id_to_width_height[dicom_id]

        # Build phrase2locations and location2bbox
        phrase2locations = {}
        location2bbox = {}

        # Normalize and store bounding box coordinates for each object
        for obj in scene_graph["objects"]:
            location2bbox[obj["bbox_name"]] = (
                obj["original_x1"] / width,
                obj["original_y1"] / height,
                obj["original_x2"] / width,
                obj["original_y2"] / height,
            )

        # Map abnormal phrases to their anatomical locations
        for region_attribute_details in scene_graph["attributes"]:
            bbox_name = region_attribute_details["bbox_name"]
            phrases_in_region = region_attribute_details["phrases"]
            attributes_for_phrases_in_region = region_attribute_details["attributes"]

            assert len(phrases_in_region) == len(attributes_for_phrases_in_region), \
                f"Mismatch in phrases and attributes for {bbox_name}, dicom_id: {dicom_id}"

            for i, phrase_text in enumerate(phrases_in_region):
                assert isinstance(phrase_text, str), \
                    f"type(phrase_text) = {type(phrase_text)}, phrase_text = {phrase_text}, dicom_id: {dicom_id}"
                assert isinstance(attributes_for_phrases_in_region[i], list), \
                    (f"type(attributes_for_phrases_in_region[i]) = {type(attributes_for_phrases_in_region[i])}, "
                     f"attributes_for_phrases_in_region[i] = {attributes_for_phrases_in_region[i]}, "
                     f"dicom_id: {dicom_id}")
                
                if "nlp|yes|abnormal" in attributes_for_phrases_in_region[i]:  # Only add if abnormal
                    if phrase_text not in phrase2locations:
                        phrase2locations[phrase_text] = []
                    phrase2locations[phrase_text].append(bbox_name)

        output_list.append(
            {
                "dicom_id": dicom_id,
                "phrase2locations": phrase2locations,
                "location2bbox": location2bbox,
            }
        )

    # Write output to file
    save_path = os.path.join(CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR, "phrase_groundings_from_scene_graphs.pkl")
    save_pickle(output_list, save_path)
    logger.info(f"Saved phrase groundings to {save_path}")


def refine_phrase_groundings_with_llm(args):
    """
    Refines phrase-to-location mappings using a Large Language Model (LLM).

    This function:
    - Loads the initial phrase-to-location mappings (e.g., from scene graphs).
    - For each phrase and its associated locations, uses an LLM (e.g., OpenAI GPT or Gemini)
      to split, simplify, and correct the mapping.
    - Saves the refined mapping to disk.

    Args:
        args: argparse.Namespace with necessary arguments, such as input/output paths,
              LLM model name, API key, etc.
    """
    
    logger.info("Refining phrase groundings using LLM...")

    # Output file
    save_filepath = os.path.join(
        CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
        f"refined_mappings_{args.model_name}.jsonl"
    )

    # 1. Check if save_filepath already exists and load processed queries
    already_processed = set()
    if os.path.exists(save_filepath):
        items = load_jsonl(save_filepath)
        logger.info(f"Found {len(items)} already processed queries in {save_filepath}")
        for item in items:
            already_processed.add(item["metadata"]["query"])

    # 2. Load already processed queries from additional files if provided
    if args.already_processed_files:
        for file in args.already_processed_files:
            if os.path.exists(file):
                items = load_jsonl(file)
                logger.info(f"Found {len(items)} already processed queries in {file}")
                for item in items:
                    already_processed.add(item["metadata"]["query"])

    logger.info(f"Total already processed queries: {len(already_processed)}")

    # 3. Load the input file containing the extracted phrase groundings
    input_data = load_pickle(args.extracted_phrase_groundings)
    logger.info(f"Loaded {len(input_data)} entries from {args.extracted_phrase_groundings}")

    # 4. Prepare texts for LLM
    queries_to_process = []
    for entry in tqdm(input_data, desc="Preparing LLM inputs"):
        phrase2locations = entry["phrase2locations"]
        for phrase, locations in phrase2locations.items():
            # Prepare the input as required by your SYSTEM_INSTRUCTIONS
            input_obj = {
                "phrase": phrase,
                "locations": locations
            }
            query = json.dumps(input_obj)
            if query in already_processed:
                continue
            queries_to_process.append(query)
            already_processed.add(query) # Avoid duplicates
            if len(queries_to_process) >= args.max_num_queries:
                break
        if len(queries_to_process) >= args.max_num_queries:
            break

    if not queries_to_process:
        logger.info("No new queries to process. Exiting.")
        return
    
    logger.info(f"Total queries to process: {len(queries_to_process)}")

    # 4. Print the first 5 queries
    logger.info("First 5 queries to process:")
    for i, query in enumerate(queries_to_process[:5]):
        logger.info(f"Query {i}: {query}")

    # 5. Run LLM API requests
    run_common_boilerplate_for_api_requests(
        texts=queries_to_process,
        system_instructions=SYSTEM_INSTRUCTIONS,
        model_name=args.model_name,
        api_key_name=args.api_key_name,
        api_type=args.api_type,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        max_tokens_per_request=args.max_tokens_per_request,
        temperature=args.temperature,
        parse_output=parse_llm_output,
        save_filepath=save_filepath,
        delete_api_requests_and_responses=not args.dont_delete_api_requests_and_responses,
    )


def integrate_llm_revisions_into_groundings(args):
    """
    Integrates LLM-revised phrase groundings into the original annotations.
    Handles empty location lists and validates against bounding box data.
    """
    logger.info("Integrating LLM revisions into groundings...")

    # 1. Load LLM-revised phrase groundings
    llm_revision_map = {}
    for jsonl_path in args.llm_revised_jsonl_files:
        items = load_jsonl(jsonl_path)
        logger.info(f"Loaded {len(items)} entries from {jsonl_path}")
        for item in items:
            query = item["metadata"]["query"]
            llm_revision_map[query] = item["parsed_response"]
    logger.info(
        f"Loaded {len(llm_revision_map)} LLM revisions from {args.llm_revised_jsonl_files}"
    )

    # 2. Load original phrase groundings
    original_data = load_pickle(args.original_phrase_groundings)
    assert original_data, f"Original data is empty: {args.original_phrase_groundings}"

    # 3. Process and revise based on LLM output
    processed_data_after_llm = []
    revised_count = 0  # Counts original phrases that had at least one valid revision
    non_revised_count = 0 # Counts original phrases kept as is (either no LLM revision or LLM agreed without improvements)
    skipped_by_llm_no_abnormality_count = 0 # LLM says "has_localizable_abnormalities: no"
    
    # For new error type: empty locations
    empty_locations_anomalies = []
    skipped_due_to_empty_locations_count = 0


    for entry in tqdm(
        original_data, desc="Applying LLM revisions", mininterval=1, ncols=100
    ):
        dicom_id = entry["dicom_id"]
        original_phrase2locations_map = entry["phrase2locations"]
        location2bbox = entry["location2bbox"]
        
        current_dicom_revised_phrase2locations = {}

        for phrase, locations in original_phrase2locations_map.items():
            # CHECK 1: Original locations are empty or None
            if not locations: # Handles [] or None
                skipped_due_to_empty_locations_count += 1
                empty_locations_anomalies.append({
                    "dicom_id": dicom_id,
                    "original_phrase": phrase,
                    "status": "original_empty_locations",
                    "original_locations": locations if locations is not None else [],
                })
                logger.debug(f"DicomID {dicom_id}: Skipping phrase '{phrase}' due to originally empty locations.")
                continue # Skip to the next phrase in this dicom_id

            query_obj = {"phrase": phrase, "locations": locations}
            query_str = json.dumps(query_obj)

            if query_str in llm_revision_map:
                revision = llm_revision_map[query_str]
                if (
                    revision.get("has_localizable_abnormalities", "no")
                    == "no"
                ):
                    skipped_by_llm_no_abnormality_count += 1
                    # Optionally, log this to empty_locations_anomalies if desired,
                    # but keeping it separate for now as it's a different reason.
                    # empty_locations_anomalies.append({
                    #     "dicom_id": dicom_id,
                    #     "original_phrase": phrase,
                    #     "status": "llm_no_localizable_abnormality",
                    #     "llm_reason": revision.get("reason", "N/A")
                    # })
                    continue

                if revision.get("improvements"):
                    made_at_least_one_valid_revision_for_this_phrase = False
                    for imp in revision["improvements"]:
                        improved_phrase_text = imp["phrase"].strip()
                        improved_locations = imp["locations"]

                        if not improved_phrase_text:
                            logger.warning(f"DicomID {dicom_id}: LLM improvement for original phrase '{phrase}' "
                                           "has empty revised phrase text. Skipping this specific improvement.")
                            continue

                        # CHECK 2: LLM-suggested locations are empty or None
                        if not improved_locations: # Handles [] or None
                            skipped_due_to_empty_locations_count += 1
                            empty_locations_anomalies.append({
                                "dicom_id": dicom_id,
                                "original_phrase": phrase, # The phrase LLM was working on
                                "status": "llm_suggested_empty_locations",
                                "llm_revised_phrase": improved_phrase_text,
                                "llm_suggested_locations": improved_locations if improved_locations is not None else [],
                                "llm_reason": revision["reason"]
                            })
                            logger.debug(f"DicomID {dicom_id}: Skipping improved phrase '{improved_phrase_text}' (from original '{phrase}') "
                                           f"due to LLM-suggested empty locations. Reason: {revision.get('reason', 'N/A')}")
                            continue # Skip this specific improvement

                        # If both phrase and locations are valid from LLM
                        current_dicom_revised_phrase2locations[improved_phrase_text] = improved_locations
                        made_at_least_one_valid_revision_for_this_phrase = True
                    
                    if made_at_least_one_valid_revision_for_this_phrase:
                        revised_count += 1 # Original phrase successfully revised
                    # If no valid revision was made (e.g., all improvements had empty phrases/locations),
                    # the original phrase is effectively dropped for this dicom_id.
                else:
                    # No 'improvements' key, but 'has_localizable_abnormalities' is 'yes'.
                    # Original locations are guaranteed non-empty here due to CHECK 1.
                    non_revised_count += 1
                    current_dicom_revised_phrase2locations[phrase] = locations
            else:
                # No LLM revision found.
                # Original locations are guaranteed non-empty here due to CHECK 1.
                logger.warning(
                    f"No LLM revision found for phrase: '{phrase}' in dicom_id: {dicom_id}"
                )
                non_revised_count += 1
                current_dicom_revised_phrase2locations[phrase] = locations
        
        # Clean phrases (remove newlines, extra spaces, trailing periods)
        final_cleaned_phrases_for_dicom = {}
        if current_dicom_revised_phrase2locations:
            for phrase_text, locs in current_dicom_revised_phrase2locations.items():
                cleaned_phrase = phrase_text.replace("\\n", " ").strip()
                cleaned_phrase = " ".join(cleaned_phrase.split())
                if cleaned_phrase and cleaned_phrase[-1] == ".":
                    cleaned_phrase = cleaned_phrase[:-1]
                
                if cleaned_phrase: # Only add if phrase is not empty after cleaning
                    final_cleaned_phrases_for_dicom[cleaned_phrase] = locs
        
        if final_cleaned_phrases_for_dicom or location2bbox: # Only add entry if there's something to add
            processed_data_after_llm.append(
                {
                    "dicom_id": dicom_id,
                    "phrase2locations": final_cleaned_phrases_for_dicom,
                    "location2bbox": location2bbox,
                }
            )

    # Save empty location anomalies
    if empty_locations_anomalies:
        empty_loc_anomalies_save_path = os.path.join(
            CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
            "empty_locations_anomalies.jsonl",
        )
        save_jsonl(empty_locations_anomalies, empty_loc_anomalies_save_path)
        logger.info(
            f"Saved {len(empty_locations_anomalies)} anomalous cases (empty locations) to {empty_loc_anomalies_save_path}"
        )


    # 4. Intermediate Step: Validate phrase locations against location2bbox
    logger.info("Validating phrase locations against bounding box data...")
    final_revised_data = []
    dropped_phrases_due_to_missing_bbox_count = 0
    anomalous_bbox_cases = [] # For missing bboxes

    for entry in tqdm(
        processed_data_after_llm,
        desc="Validating location bboxes",
        mininterval=1,
        ncols=100,
    ):
        dicom_id = entry["dicom_id"]
        current_phrase2locations = entry["phrase2locations"]
        location2bbox = entry["location2bbox"]
        
        validated_phrase2locations_for_dicom = {}
        for phrase, loc_names_list in current_phrase2locations.items():
            # loc_names_list should already be a list from previous steps
            missing_bbox_for = [
                loc for loc in loc_names_list if loc not in location2bbox
            ]
            
            if not missing_bbox_for:
                validated_phrase2locations_for_dicom[phrase] = loc_names_list
            else:
                dropped_phrases_due_to_missing_bbox_count += 1
                anomalous_bbox_cases.append(
                    {
                        "dicom_id": dicom_id,
                        "dropped_phrase": phrase,
                        "original_locations_for_phrase": loc_names_list,
                        "locations_missing_bbox": missing_bbox_for,
                    }
                )
                logger.debug(
                    f"DicomID {dicom_id}: Dropping phrase '{phrase}' due to missing bbox for locations: {missing_bbox_for}"
                )
        
        if validated_phrase2locations_for_dicom or location2bbox:
            final_revised_data.append(
                {
                    "dicom_id": dicom_id,
                    "phrase2locations": validated_phrase2locations_for_dicom,
                    "location2bbox": location2bbox,
                }
            )

    if anomalous_bbox_cases: # Missing bbox anomalies
        bbox_anomalies_save_path = os.path.join(
            CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
            "dropped_phrases_missing_bbox_anomalies.jsonl",
        )
        save_jsonl(anomalous_bbox_cases, bbox_anomalies_save_path)
        logger.info(
            f"Saved {len(anomalous_bbox_cases)} anomalous cases (dropped phrases due to missing bboxes) to {bbox_anomalies_save_path}"
        )

    # 5. Log statistics
    logger.info(
        f"LLM processing stage: "
        f"Successfully revised {revised_count} original phrases. "
        f"Kept {non_revised_count} original phrases as non-revised. "
        f"Skipped {skipped_by_llm_no_abnormality_count} original phrases (LLM: no localizable abnormality)."
    )
    logger.info(
        f"Empty locations handling: Skipped {skipped_due_to_empty_locations_count} phrases/improvements due to empty location lists."
    )
    logger.info(
        f"Validation stage: Dropped {dropped_phrases_due_to_missing_bbox_count} phrases due to missing bounding box data for their locations."
    )
    
    logger.info(f"Final number of entries in integrated data: {len(final_revised_data)}")
    if final_revised_data:
        num_total_bboxes = sum(
            len(entry.get("location2bbox", {})) for entry in final_revised_data
        )
        avg_bboxes = num_total_bboxes / len(final_revised_data) if len(final_revised_data) > 0 else 0
        logger.info(
            f"Average number of unique bboxes (location2bbox) per dicom_id: {avg_bboxes:.2f}"
        )
        
        num_total_phrases = sum(
            len(entry.get("phrase2locations", {})) for entry in final_revised_data
        )
        avg_phrases = num_total_phrases / len(final_revised_data) if len(final_revised_data) > 0 else 0
        logger.info(
            f"Average number of phrases (phrase2locations) per dicom_id: {avg_phrases:.2f}"
        )
    else:
        logger.info("No data remaining after LLM processing and validation.")

    # 6. Save final revised and validated annotations
    save_path = os.path.join(
        CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
        "integrated_llm_revised_phrase_groundings.pkl",
    )
    save_pickle(final_revised_data, save_path)
    logger.info(f"Saved integrated and validated annotations to {save_path}")
    return save_path


def collect_phrase_groundings_from_padchest_gr_and_mscxr():
    from medvqa.datasets.padchest import get_padchest_gr_phrase_groundings
    from medvqa.datasets.ms_cxr import get_ms_cxr_dicom_id_2_phrases_and_bboxes
    padchest_gr_phrase_groundings = get_padchest_gr_phrase_groundings()
    ms_cxr_dicom_id_2_phrases_and_bboxes = get_ms_cxr_dicom_id_2_phrases_and_bboxes()
    output = []
    
    for item in padchest_gr_phrase_groundings:
        output.append({**item, "dataset": "padchest-gr"})
    num_phrases = len(output)
    logger.info(f"Collected {num_phrases} phrases from padchest-gr")
    
    for dicom_id, (phrases, bbox_lists) in ms_cxr_dicom_id_2_phrases_and_bboxes.items():
        for phrase, bboxes in zip(phrases, bbox_lists):
            output.append({
                "dicom_id": dicom_id,
                "phrase": phrase,
                "boxes": bboxes,
                "dataset": "ms-cxr"
            })
    num_phrases = len(output) - num_phrases # subtract the previous count
    logger.info(f"Collected {num_phrases} phrases from ms-cxr")
    
    return output


def bleu1_similarity(tok_sent_a: List[str], tok_sent_b: List[str]) -> float:
    return sentence_bleu([tok_sent_a], tok_sent_b, weights=(1.0,))

def levenshtein_similarity(sent_a: str, sent_b: str) -> float:
    lev_dist = Levenshtein.distance(sent_a, sent_b)
    max_len = max(len(sent_a), len(sent_b))
    if max_len == 0:
        return 1.0
    return 1 - lev_dist / max_len

def hybrid_score_similarity(
    original_tok_phrase: List[str],
    original_phrase: str,
    candidate_tok_phrase: List[str],
    candidate_phrase: str,
    semantic_similarity_score: float,  # Dot product of normalized embeddings
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    bleu1_sim = bleu1_similarity(original_tok_phrase, candidate_tok_phrase)
    lev_sim = levenshtein_similarity(original_phrase, candidate_phrase)
    return (
        alpha * semantic_similarity_score
        + beta * bleu1_sim
        + gamma * lev_sim
    )

def _calculate_union_area_from_bbox_list(bbox_list):
    shapely_boxes = []
    for x1, y1, x2, y2 in bbox_list:
        if x1 < x2 and y1 < y2:
            shapely_boxes.append(shapely_box(x1, y1, x2, y2))
    assert len(shapely_boxes) > 0, "No valid bounding boxes provided."
    return unary_union(shapely_boxes).area


# --- Global variables for multiprocessing worker ---
_worker_shared_data = {}

def _init_worker_augment_groundings(shared_data_package):
    global _worker_shared_data
    _worker_shared_data = shared_data_package


def _preprocess_locations(locations):
    to_remove = []
    if ('right upper lung zone' in locations or
        'right mid lung zone' in locations or
        'right lower lung zone' in locations or
        'right hilar structures' in locations or
        'right apical zone' in locations or
        'right costophrenic angle' in locations):
        to_remove.append('right lung')
    if ('left upper lung zone' in locations or
        'left mid lung zone' in locations or
        'left lower lung zone' in locations or
        'left hilar structures' in locations or
        'left apical zone' in locations or
        'left costophrenic angle' in locations):
        to_remove.append('left lung')
    if to_remove:
        clean_locations = [loc for loc in locations if loc not in to_remove]
    else:
        clean_locations = locations
    return clean_locations

# _time_step1 = 0
# _time_step2 = 0
# _time_step3 = 0
# _time_step4 = 0

def _process_single_entry_for_similar_groundings_worker(entry_tuple):
    entry_idx, entry_data = entry_tuple
    
    # Retrieve data from shared package
    revised_phrase_to_emb_map = _worker_shared_data['revised_phrase_to_emb_map']
    cluster_ids = _worker_shared_data['cluster_ids']
    cluster_centers = _worker_shared_data['cluster_centers']
    cluster_id_to_candidate_items_map = _worker_shared_data['cluster_id_to_candidate_items_map']
    
    args = argparse.Namespace(**_worker_shared_data['args_dict']) # Recreate Namespace
    
    entry_data["phrase2similar_grounding"] = {}
    entry_data["phrase2target_area"] = {}

    for original_phrase, original_loc_names in entry_data["phrase2locations"].items():
        original_phrase_embedding = revised_phrase_to_emb_map[original_phrase]
        assert len(original_loc_names) > 0, f"Original locations for phrase '{original_phrase}' are empty."

        original_tok_phrase = nltk.word_tokenize(original_phrase.lower())
        
        # Step 1: Find top 4 clusters based on semantic similarity
        # start_time = time.time()
        pairs = []
        for cluster_id, cluster_center in zip(cluster_ids, cluster_centers):
            # Semantic similarity part of hybrid score (dot product as embeddings are normalized)
            semantic_sim = np.dot(original_phrase_embedding, cluster_center)
            pairs.append((semantic_sim, cluster_id))
        
        pairs.sort(key=lambda x: x[0], reverse=True)
        assert len(pairs) >= 4
        top_cluster_ids = [cluster_id for _, cluster_id in pairs[:4]] # Top 4 clusters

        # end_time = time.time()
        # global _time_step1
        # _time_step1 += end_time - start_time
        
        # Step 2: Top K Hybrid Score Search (within the selected clusters)
        # start_time = time.time()
        all_candidate_hybrid_scores = []
        for cluster_id in top_cluster_ids:
            candidate_pool_for_search = cluster_id_to_candidate_items_map[cluster_id]
            for collected_item_dict, collected_item_embedding in candidate_pool_for_search:
                semantic_sim = np.dot(original_phrase_embedding, collected_item_embedding)
                hybrid_sim = hybrid_score_similarity(
                    original_tok_phrase,
                    original_phrase,
                    collected_item_dict['tok_phrase'],
                    collected_item_dict['phrase'],
                    semantic_sim,
                    args.hybrid_score_alpha,
                    args.hybrid_score_beta,
                    args.hybrid_score_gamma,
                )
                all_candidate_hybrid_scores.append((hybrid_sim, collected_item_dict))
                
        all_candidate_hybrid_scores.sort(key=lambda x: x[0], reverse=True)
        top_k_hybrid = all_candidate_hybrid_scores[:args.top_k_hybrid]

        # end_time = time.time()
        # global _time_step2
        # _time_step2 += end_time - start_time

        # Step 3: Re-rank based on IoU
        # start_time = time.time()
        original_bboxes_coords = [
            entry_data["location2bbox"][loc_name] for loc_name in _preprocess_locations(original_loc_names)
        ]
        assert len(original_bboxes_coords) > 0,\
            f"No valid bounding boxes found for locations: {original_loc_names} in dicom_id: {entry_data['dicom_id']}."
        
        iou_matches = []
        for _, item in top_k_hybrid:
            candidate_bboxes = item['boxes'] # List of bbox tuples
            assert len(candidate_bboxes) > 0
            iou = compute_bbox_union_iou(original_bboxes_coords, candidate_bboxes)
            if iou > 0: # Only consider positive IoU values
                iou_matches.append((iou, item))

        iou_matches.sort(key=lambda x: x[0], reverse=True)
        top_k_iou = iou_matches[:args.top_k_iou]

        # end_time = time.time()
        # global _time_step3
        # _time_step3 += end_time - start_time

        # Step 4: Final Selection and Target Area Calculation
        # start_time = time.time()
        final_similar_groundings_bboxes = []
        final_similar_groundings_items = []
        for iou, item in top_k_iou:
            if iou > 0 and item.get('boxes'): # Ensure item has boxes and IoU is positive
                final_similar_groundings_bboxes.append(item['boxes'])
                final_similar_groundings_items.append(item)

        entry_data["phrase2similar_grounding"][original_phrase] = final_similar_groundings_items
        
        if final_similar_groundings_bboxes:
            # Calculate the target area as the average union area of the bounding boxes
            target_area = sum(_calculate_union_area_from_bbox_list(bboxes) for bboxes\
                              in final_similar_groundings_bboxes) / len(final_similar_groundings_bboxes)
        else:
            # Fallback to original bboxes if no similar found
            target_area = _calculate_union_area_from_bbox_list(original_bboxes_coords)
        entry_data["phrase2target_area"][original_phrase] = target_area

        # end_time = time.time()
        # global _time_step4
        # _time_step4 += end_time - start_time
        
    return entry_idx, entry_data


def augment_groundings_with_similar_matches(args):
    logger.info("Augmenting groundings with similar matches from external datasets using clustering...")

    hybrid_sum = args.hybrid_score_alpha + args.hybrid_score_beta + args.hybrid_score_gamma
    if not np.isclose(hybrid_sum, 1.0):
        logger.error(f"Hybrid score weights (alpha, beta, gamma) must sum to 1.0. Current sum: {hybrid_sum}")
        sys.exit(1)

    revised_data = load_pickle(args.input_revised_groundings_file)
    logger.info(f"Loaded {len(revised_data)} entries from {args.input_revised_groundings_file}")

    collected_groundings = collect_phrase_groundings_from_padchest_gr_and_mscxr()
    logger.info(f"Collected {len(collected_groundings)} groundings from PadChest-GR and MS-CXR")

    if not revised_data or not collected_groundings:
        raise ValueError("No revised data or collected groundings found. Exiting.")

    logger.info("Preparing phrases for embedding extraction...")
    unique_revised_phrases = sorted(list(set(phrase for entry in revised_data for phrase in entry["phrase2locations"].keys())))
    unique_collected_phrases = sorted(list(set(item['phrase'] for item in collected_groundings)))
    all_unique_phrases = sorted(list(set(unique_revised_phrases + unique_collected_phrases)))
    phrase_to_idx_map = {phrase: i for i, phrase in enumerate(all_unique_phrases)}
    
    # Ensure these are not empty before proceeding, as per user's asserts
    if not unique_revised_phrases:
        raise ValueError("No unique revised phrases found after loading data.")
    if not unique_collected_phrases:
        raise ValueError("No unique collected phrases found from external datasets.")

    logger.info(f"Compute phrase embeddings for {len(all_unique_phrases)} unique phrases...")
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name='microsoft/BiomedVLP-CXR-BERT-specialized',
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        num_workers=args.embedding_num_workers,
    )
    cxr_bert_embeddings = embedding_extractor.compute_text_embeddings(all_unique_phrases, update_cache_on_disk=True)
    logger.info(f"cxr_bert_embeddings.shape: {cxr_bert_embeddings.shape}")

    from medvqa.metrics.medical.chexbert import CheXbertLabeler
    chexbert_labeler = CheXbertLabeler(verbose=True, default_batch_size=args.embedding_batch_size)
    chexbert_embeddings = chexbert_labeler.get_embeddings(all_unique_phrases, update_cache_on_disk=True)
    # Normalize the embeddings
    chexbert_embeddings = chexbert_embeddings / np.linalg.norm(chexbert_embeddings, axis=1, keepdims=True)
    logger.info(f"chexbert_embeddings.shape: {chexbert_embeddings.shape}")

    combined_embeddings = np.concatenate((cxr_bert_embeddings, chexbert_embeddings), axis=1)
    # Reduce dimensionality with PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.pca_components, random_state=0)
    reduced_embeddings = pca.fit_transform(combined_embeddings)
    # Normalize the reduced embeddings
    reduced_embeddings = reduced_embeddings / np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
    logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")

    revised_phrase_to_emb_map = {}
    for phrase in unique_revised_phrases:
        phrase_idx = phrase_to_idx_map[phrase]
        revised_phrase_to_emb_map[phrase] = reduced_embeddings[phrase_idx]

    # Apply KMeans clustering to the reduced embeddings
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=args.num_clusters, max_iter=args.kmeans_max_iter, random_state=0)
    kmeans.fit(reduced_embeddings)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Map cluster_id to list of (collected_item_dict, collected_item_embedding)
    # This maps the *original* collected items to their clusters.
    cluster_id_to_candidate_items_map = [[] for _ in range(args.num_clusters)]
    for item_dict in collected_groundings: # Iterate over the original list of dicts
        phrase = item_dict['phrase']
        tok_phrase = nltk.word_tokenize(phrase.lower())
        item_dict['tok_phrase'] = tok_phrase # Add tokenized phrase to the dict for later use
        phrase_idx = phrase_to_idx_map[phrase]
        cluster_id = cluster_labels[phrase_idx]
        embedding = reduced_embeddings[phrase_idx]
        cluster_id_to_candidate_items_map[cluster_id].append((item_dict, embedding))
    
    logger.info("Prepared cluster-to-candidate mapping for collected groundings.")
    cluster_sizes = [len(cluster) for cluster in cluster_id_to_candidate_items_map]
    cluster_sizes = np.array(cluster_sizes)
    logger.info(f"Cluster sizes: mean={np.mean(cluster_sizes):.2f}, "
                f"std={np.std(cluster_sizes):.2f}, min={np.min(cluster_sizes)}, max={np.max(cluster_sizes)}")
    
    # Collect non-empty cluster ids and centers
    non_empty_cluster_ids = [i for i, cluster in enumerate(cluster_id_to_candidate_items_map) if len(cluster) > 0]
    logger.info(f"Number of non-empty clusters: {len(non_empty_cluster_ids)} out of {args.num_clusters} total clusters")
    non_empty_cluster_centers = [cluster_centers[i] for i in non_empty_cluster_ids]

    # revised_data = revised_data[:10000] # For testing purposes, limit to 100 entries. TODO: Remove this line in production
    
    items_to_process = list(enumerate(revised_data))
    augmented_data_list = [None] * len(revised_data)

    worker_init_args = {
        'revised_phrase_to_emb_map': revised_phrase_to_emb_map,
        'cluster_ids': non_empty_cluster_ids,
        'cluster_centers': non_empty_cluster_centers,
        'cluster_id_to_candidate_items_map': cluster_id_to_candidate_items_map, # Items grouped by their cluster_id
        'args_dict': vars(args)
    }

    # global _time_step1
    # global _time_step2
    # global _time_step3
    # global _time_step4
    # _time_step1 = 0
    # _time_step2 = 0
    # _time_step3 = 0
    # _time_step4 = 0

    if args.num_similarity_workers > 0 and len(items_to_process) > 1:
        logger.info(f"Processing entries in parallel with {args.num_similarity_workers} workers...")
        # Use functools.partial to pass fixed arguments to the initializer
        initializer_func = functools.partial(_init_worker_augment_groundings, worker_init_args)
        with Pool(processes=args.num_similarity_workers, initializer=initializer_func) as pool:
            with tqdm(total=len(items_to_process), desc="Augmenting groundings", mininterval=2) as pbar:
                for entry_idx, processed_entry_data in pool.imap_unordered(
                    _process_single_entry_for_similar_groundings_worker, items_to_process
                ):
                    augmented_data_list[entry_idx] = processed_entry_data
                    pbar.update(1)
    else:
        logger.info("Processing entries sequentially...")
        _init_worker_augment_groundings(worker_init_args) # Initialize for sequential run
        for item_tuple in tqdm(items_to_process, desc="Augmenting groundings", mininterval=2):
            entry_idx, processed_entry_data = _process_single_entry_for_similar_groundings_worker(item_tuple)
            augmented_data_list[entry_idx] = processed_entry_data

    # Remove "tok_phrase" key from each entry
    for entry in augmented_data_list:
        for item in entry["phrase2similar_grounding"].values():
            for item_dict in item:
                item_dict.pop('tok_phrase', None)

    # # Log time taken for each step
    # logger.info(f"Time taken for Step 1 (Top 3 Cluster Search): {_time_step1:.2f} seconds")
    # logger.info(f"Time taken for Step 2 (Top K Hybrid Search): {_time_step2:.2f} seconds")
    # logger.info(f"Time taken for Step 3 (IoU Re-ranking): {_time_step3:.2f} seconds")
    # logger.info(f"Time taken for Step 4 (Final Selection): {_time_step4:.2f} seconds")
    
    save_path_strings = [
        args.input_revised_groundings_file,
        f'num_clusters={args.num_clusters}',
        f'top_k_hybrid={args.top_k_hybrid}',
        f'top_k_iou={args.top_k_iou}',
        f'alpha={args.hybrid_score_alpha}',
        f'beta={args.hybrid_score_beta}',
        f'gamma={args.hybrid_score_gamma}',
    ]
    output_filename = get_file_path_with_hashing_if_too_long(
        folder_path=CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
        prefix=f"augmented_groundings_",
        strings=save_path_strings,
        force_hashing=True,
    )
    save_pickle(augmented_data_list, output_filename)
    logger.info(f"Saved augmented groundings to {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-step phrase grounding annotation extractor"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Step 1: Extract abnormal phrase groundings
    parser_extract = subparsers.add_parser(
        "extract-abnormal-phrase-groundings",
        help="Extract abnormal phrase-to-location mappings from scene graphs"
    )
    parser_extract.set_defaults(func=extract_abnormal_phrase_groundings_from_scene_graphs)

    # Step 2: Refine phrase groundings with LLM
    parser_refine = subparsers.add_parser(
        "refine-phrase-groundings-with-llm",
        help="Refine phrase-to-location mappings using an LLM"
    )
    parser_refine.add_argument("--extracted-phrase-groundings", required=True, help="Input extracted phrase groundings file")
    parser_refine.add_argument("--model-name", required=True, help="LLM model name")
    parser_refine.add_argument("--api-key-name", required=True, help="API key environment variable name")
    parser_refine.add_argument("--api-type", default="openai", choices=["openai", "gemini"], help="API type")
    parser_refine.add_argument("--max-requests-per-minute", type=int, required=True)
    parser_refine.add_argument("--max-tokens-per-minute", type=int, required=True)
    parser_refine.add_argument("--max-tokens-per-request", type=int, required=True)
    parser_refine.add_argument("--temperature", type=float, default=0.0)
    parser_refine.add_argument("--max-num-queries", type=int, default=100)
    parser_refine.add_argument("--dont-delete-api-requests-and-responses", action="store_true", help="Do not delete API requests and responses")
    parser_refine.add_argument("--already-processed-files", nargs="*", help="List of already processed files")
    parser_refine.set_defaults(func=refine_phrase_groundings_with_llm)

    # Step 3: Integrate LLM revisions
    parser_integrate = subparsers.add_parser(
        "integrate-llm-revisions",
        help="Integrate LLM-based revisions into phrase grounding annotations"
    )
    parser_integrate.add_argument(
        "--llm-revised-jsonl-files", nargs='+', required=True,
        help="One or more JSONL files with LLM-revised phrase groundings"
    )
    parser_integrate.add_argument(
        "--original-phrase-groundings", required=True,
        help="Path to the phrase_groundings_from_scene_graphs.pkl (output of extract step)"
    )
    parser_integrate.set_defaults(func=integrate_llm_revisions_into_groundings)

    # Step 4: Augment groundings with similar matches
    parser_augment = subparsers.add_parser(
        "augment-groundings-with-matches",
        help="Augment groundings by finding similar matches from external datasets"
    )
    parser_augment.add_argument(
        "--input-revised-groundings-file", required=True,
        help="Input file: output of 'integrate-llm-revisions' step (e.g., integrated_llm_revised_phrase_groundings.pkl)"
    )
    # Embedding extractor args
    parser_augment.add_argument(
        "--embedding-device", type=str, default="cuda", choices=["cpu", "cuda"],
        help="Device for embedding extraction ('cpu' or 'cuda')."
    )
    parser_augment.add_argument(
        "--embedding-batch-size", type=int, default=128, # Increased default
        help="Batch size for embedding extraction."
    )
    parser_augment.add_argument(
        "--embedding-num-workers", type=int, default=4,
        help="Number of workers for DataLoader during embedding extraction."
    )
    # Similarity search and ranking args
    parser_augment.add_argument("--top-k-hybrid", type=int, default=25)
    parser_augment.add_argument("--top-k-iou", type=int, default=5)
    parser_augment.add_argument("--hybrid-score-alpha", type=float, default=0.5, help="Weight for semantic score in hybrid model.")
    parser_augment.add_argument("--hybrid-score-beta", type=float, default=0.3, help="Weight for BLEU-1 score in hybrid model.")
    parser_augment.add_argument("--hybrid-score-gamma", type=float, default=0.2, help="Weight for Levenshtein score in hybrid model.")
    parser_augment.add_argument(
        "--num-clusters", type=int, default=80,
        help="Number of clusters for KMeans clustering."
    )
    parser_augment.add_argument(
        "--kmeans-max-iter", type=int, default=300,
        help="Maximum number of iterations for KMeans clustering."
    )
    parser_augment.add_argument(
        "--num-similarity-workers", type=int, default=0, # Default 0 for sequential
        help="Number of workers for parallel processing of entries (0 for sequential)."
    )
    parser_augment.add_argument(
        "--pca-components", type=int, default=60,
        help="Number of PCA components to reduce the embedding dimensionality."
    )
    parser_augment.set_defaults(func=augment_groundings_with_similar_matches)    

    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
