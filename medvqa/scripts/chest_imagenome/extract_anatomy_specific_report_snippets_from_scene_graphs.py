import logging
import os

import pandas as pd
from tqdm import tqdm
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import load_chest_imagenome_dicom_ids, load_scene_graph
from medvqa.datasets.mimiccxr import MIMICCXR_METADATA_CSV_PATH, get_imageId2PartPatientStudy, get_mimiccxr_report_path
from medvqa.datasets.mimiccxr.report_utils import concatenate_sentences
from medvqa.utils.files_utils import read_txt, save_pickle
from medvqa.utils.logging_utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def _concatenate_phrases(phrases):
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
    return concatenate_sentences(phrases)

def extract_location_report_snippets_from_scene_graphs():
    """
    Extracts for each DICOM image:
    - The original full report.
    - An exhaustive mapping from bbox_name to normalized bounding box coordinates.
    - A mapping from bbox_name to a concatenated "mini-report" snippet of all
      phrases associated with that bounding box.

    The result is a list of dictionaries, where each dictionary contains:
        - 'dicom_id': The DICOM image identifier.
        - 'original_report': The full text of the report associated with the dicom_id.
        - 'location2bbox': Mapping from bbox_name to normalized (x1, y1, x2, y2) coordinates.
                          This mapping is exhaustive for all bounding boxes in the scene graph.
        - 'location2report_snippet': Mapping from bbox_name to a concatenated string of
                                     all phrases associated with that bbox.
                                     Entries only exist for locations that have associated phrases.

    The output is saved as a pickle file in the CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR.
    """

    logger.info("Extracting location-report snippets from scene graphs...")

    # Load metadata for image dimensions
    df = pd.read_csv(MIMICCXR_METADATA_CSV_PATH)
    dicom_id_to_width_height = {
        dicom_id: (width, height)
        for dicom_id, width, height in zip(df.dicom_id, df.Columns, df.Rows)
    }

    # Get all valid dicom_ids
    dicom_ids = load_chest_imagenome_dicom_ids()
    # dicom_ids = dicom_ids[:1000]  # Limit to first 1000 for testing; remove this line for full extraction

    output_list = []

    imageId2PartPatientStudy = get_imageId2PartPatientStudy()

    for dicom_id in tqdm(
        dicom_ids,
        desc="Extracting location report snippets",
        mininterval=1,
        ncols=100,
    ):
        scene_graph = load_scene_graph(dicom_id)
        if not scene_graph:
            logger.warning(f"No scene graph found for dicom_id: {dicom_id}, skipping.")
            continue

        part_id, patient_id, study_id = imageId2PartPatientStudy.get(dicom_id, (None, None, None))
        if part_id is None:
            logger.warning(f"DICOM ID {dicom_id} not found in imageId2PartPatientStudy, skipping.")
            continue

        # Get full report
        report_path = get_mimiccxr_report_path(part_id, patient_id, study_id)
        original_report = read_txt(report_path)

        # Get image dimensions
        width, height = dicom_id_to_width_height.get(dicom_id)
        if width is None or height is None:
            logger.warning(f"Dimensions not found for dicom_id: {dicom_id}, skipping.")
            continue

        # Construct location2bbox mapping
        location2bbox = {}
        if "objects" in scene_graph and isinstance(scene_graph["objects"], list):
            for obj in scene_graph["objects"]:
                if (
                    "bbox_name" in obj
                    and "original_x1" in obj
                    and "original_y1" in obj
                    and "original_x2" in obj
                    and "original_y2" in obj
                ):
                    location2bbox[obj["bbox_name"]] = (
                        obj["original_x1"] / width,
                        obj["original_y1"] / height,
                        obj["original_x2"] / width,
                        obj["original_y2"] / height,
                    )

        # Construct location2report_snippet from scene graph attributes
        location2report_snippet = {}
        if "attributes" in scene_graph and isinstance(
            scene_graph["attributes"], list
        ):
            for region_attribute_details in scene_graph["attributes"]:
                bbox_name = region_attribute_details["bbox_name"]
                phrases_in_region = region_attribute_details.get("phrases", [])

                if phrases_in_region:
                    assert bbox_name not in location2report_snippet
                    report = _concatenate_phrases(phrases_in_region)
                    if not report:
                        logger.warning(
                            f"No phrases found for bbox_name: {bbox_name} in dicom_id: {dicom_id}"
                        )
                        continue
                    location2report_snippet[bbox_name] = report

        output_list.append(
            {
                "dicom_id": dicom_id,
                "original_report": original_report,
                "location2bbox": location2bbox,
                "location2report_snippet": location2report_snippet,
            }
        )

    # Write output to file
    save_path = os.path.join(
        CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR, "location_report_snippets.pkl"
    )
    save_pickle(output_list, save_path)
    logger.info(f"Saved location report snippets to {save_path}")


if __name__ == "__main__":
    extract_location_report_snippets_from_scene_graphs()
    logger.info("Extraction completed successfully.")