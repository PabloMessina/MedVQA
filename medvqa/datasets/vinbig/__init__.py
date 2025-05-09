import random
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Literal
from dotenv import load_dotenv
from imagesize import get as get_image_size
from medvqa.datasets.segmentation_utils import compute_mask_from_bounding_box
from medvqa.utils.bbox_utils import xyxy_to_cxcywh
from medvqa.utils.common import CACHE_DIR, FAST_CACHE_DIR, LARGE_FAST_CACHE_DIR
from medvqa.utils.constants import VINBIG_BBOX_NAMES, VINBIG_LABELS
from medvqa.utils.logging_utils import ANSI_ORANGE_BOLD, ANSI_RESET

load_dotenv()

logger = logging.getLogger(__name__)

BboxDictType = Union[
    Dict[str, Dict[str, List[List[float]]]],
    Dict[str, Tuple[List[List[float]], List[int]]],
]

# VINBIG_ORIGINAL_IMAGES_FOLDER = os.environ['VINBIG_ORIGINAL_IMAGES_FOLDER']
# VINBIG_512x512_IMAGES_FOLDER = os.environ['VINBIG_512x512_IMAGES_FOLDER']
VINBIG_ORIGINAL_IMAGES_FOLDER = os.environ['VINBIG_ORIGINAL_HQ_IMAGES_FOLDER']
VINBIG_512x512_IMAGES_FOLDER = os.environ['VINBIG_512x512_HQ_IMAGES_FOLDER']
VINBIG_ALL_IMAGES_TXT_PATH  = os.environ['VINBIG_ALL_IMAGES_TXT_PATH']
VINBIG_TRAIN_VAL_IMAGES_TXT_PATH = os.environ['VINBIG_TRAIN_VAL_IMAGES_TXT_PATH']
VINBIG_TEST_IMAGES_TXT_PATH = os.environ['VINBIG_TEST_IMAGES_TXT_PATH']
VINBIG_LABELS_CSV_PATH = os.environ['VINBIG_LABELS_CSV_PATH']
VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH = os.environ['VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH']
VINBIG_IMAGE_LABELS_TEST_CSV_PATH = os.environ['VINBIG_IMAGE_LABELS_TEST_CSV_PATH']
VINBIG_ANNOTATIONS_TRAIN_CSV_PATH = os.environ['VINBIG_ANNOTATIONS_TRAIN_CSV_PATH']
VINBIG_ANNOTATIONS_TEST_CSV_PATH = os.environ['VINBIG_ANNOTATIONS_TEST_CSV_PATH']
VINBIG_YOLOV5_LABELS_DIR = os.environ['VINBIG_YOLOV5_LABELS_DIR']
VINBIG_CACHE_DIR = os.path.join(CACHE_DIR, 'vinbig')
VINBIG_FAST_CACHE_DIR = os.path.join(FAST_CACHE_DIR, 'vinbig')
VINBIG_LARGE_FAST_CACHE_DIR = os.path.join(LARGE_FAST_CACHE_DIR, 'vinbig')

N_IMAGES_TRAIN = 15000
N_IMAGES_TEST = 3000

VINBIG_LABELS__MODIFIED = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Clavicle fracture',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Enlarged PA',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Lung cavity',
    'Lung cyst',
    'Mediastinal shift',
    'Nodule/Mass',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'Rib fracture',
    'Other lesion',
    'COPD',
    'Lung tumor',
    'Pneumonia',
    'Tuberculosis',
    'Other disease',
    'Abnormal finding', # replaces "No finding"
]

VINBIG_BBOX_NAMES__MODIFIED = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Clavicle fracture',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Enlarged PA',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Lung cavity',
    'Lung cyst',
    'Mediastinal shift',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'Rib fracture',
    'Abnormal finding', # new class not present in the original dataset, it encompasses all the other classes
]

VINBIG_NUM_BBOX_CLASSES__MODIFIED = len(VINBIG_BBOX_NAMES__MODIFIED)

LUNG_OPACITY_CLASSES = [ # all classes that are subcategories of "Lung Opacity"
    'Lung Opacity',
    'Consolidation',
    'Nodule/Mass',
    'Atelectasis',
    'Pulmonary fibrosis',
    'Edema',
    'Infiltration',
    'ILD',
]
assert all(class_name in VINBIG_BBOX_NAMES for class_name in LUNG_OPACITY_CLASSES)


# source: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data
VINBIGDATA_CHALLENGE_CLASSES = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
]
assert all(x in VINBIG_BBOX_NAMES for x in VINBIGDATA_CHALLENGE_CLASSES)

VINBIGDATA_CHALLENGE_IOU_THRESHOLD = 0.4

# source: https://github.com/philip-mueller/chex/blob/main/conf/dataset/class_names/vindrcxr_loc_top15.yaml
VINBIG_CHEX_CLASSES = [
    'Aortic enlargement',
    'Atelectasis',
    'Cardiomegaly',
    'Calcification',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Mediastinal shift',
    'Nodule/Mass',
    'Pulmonary fibrosis',
    'Pneumothorax',
    'Pleural thickening',
    'Pleural effusion',
    'Other lesion',
]
assert all(x in VINBIG_BBOX_NAMES for x in VINBIG_CHEX_CLASSES)

VINBIG_CHEX_IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

VINBIG_RAD_DINO_CLASSES = [ # source: https://www.nature.com/articles/s42256-024-00965-w
    'Lung Opacity',
    'Cardiomegaly',
    'Pleural thickening',
    'Aortic enlargement',
    'Pulmonary fibrosis',
    'Tuberculosis',
    'Pleural effusion',
]

def get_unique_bbox_names():
    unique_bbox_names = set()
    for csv_path in [VINBIG_ANNOTATIONS_TRAIN_CSV_PATH, VINBIG_ANNOTATIONS_TEST_CSV_PATH]:
        df = pd.read_csv(csv_path)
        df = df[df['x_min'].notna()]
        for class_name in df['class_name']:
            unique_bbox_names.add(class_name)
    return sorted(list(unique_bbox_names))

def _load_image_id_2_bboxes(
    csv_path: str,
    for_training: bool = False,
    normalize: bool = False,
    class_id_offset: int = 0,
    improve_labels: bool = False,
    bbox_format: Literal["xyxy", "cxcywh"] = "xyxy",
) -> BboxDictType:
    """Loads bounding box annotations from a CSV file into a dictionary.

    Reads a CSV file containing bounding box data, processes it according
    to the specified arguments, and returns a dictionary mapping image IDs
    to their corresponding bounding boxes and potentially class information.

    The CSV file is expected to have columns: 'image_id', 'class_name',
    'x_min', 'y_min', 'x_max', 'y_max'.

    Args:
        csv_path: Path to the input CSV file.
        for_training: If True, the output dictionary maps image IDs to a
            tuple containing a list of all bounding boxes for that image and
            a corresponding list of class IDs. If False (default), it maps
            image IDs to a nested dictionary where keys are class names and
            values are lists of bounding boxes for that class.
        normalize: If True, bounding box coordinates (x_min, y_min, x_max, y_max)
            are normalized to the range [0, 1] by dividing by image width
            and height. Requires helper functions `get_original_image_path`
            and `get_image_size`. Bboxes outside [0, 1] after normalization
            are clamped and counted as anomalous. Defaults to False.
        class_id_offset: An integer offset added to the class IDs when
            `for_training` is True. Defaults to 0.
        improve_labels: If True, applies specific modifications to the class
            labels based on domain knowledge (e.g., merging subclasses into
            "Lung Opacity", creating "Abnormal finding"). Requires constants
            `LUNG_OPACITY_CLASSES` and `VINBIG_BBOX_NAMES__MODIFIED`.
            Defaults to False.
        bbox_format: The desired format for the output bounding boxes.
            Can be 'xyxy' (default) for [x_min, y_min, x_max, y_max] or
            'cxcywh' for [center_x, center_y, width, height]. Requires the
            helper function `xyxy_to_cxcywh` if set to 'cxcywh'.

    Returns:
        A dictionary containing the bounding box information.
        - If `for_training` is False:
          `Dict[image_id, Dict[class_name, List[bbox]]]`
          where bbox is `List[float]` in the specified `bbox_format`.
        - If `for_training` is True:
          `Dict[image_id, Tuple[List[bbox], List[class_id]]]`
          where bbox is `List[float]` in the specified `bbox_format`,
          and class_id is `int`.
    """
    # Validate bbox_format argument
    assert bbox_format in (
        "xyxy",
        "cxcywh",
    ), "bbox_format must be 'xyxy' or 'cxcywh'"

    # Load the dataframe
    df = pd.read_csv(csv_path)
    # Filter out rows where bounding box information is missing
    df = df[df["x_min"].notna()]

    # Initialize the main dictionary to store results
    image_id_2_bboxes: Union[
        Dict[str, Dict[str, List[List[float]]]],
        Dict[str, Tuple[List[List[float]], List[int]]],
    ] = {}
    anomalous_count = 0  # Counter for invalid or out-of-bounds boxes

    # Iterate through each annotation row
    for image_id, class_name, x_min, y_min, x_max, y_max in zip(
        df["image_id"],
        df["class_name"],
        df["x_min"],
        df["y_min"],
        df["x_max"],
        df["y_max"],
    ):
        is_anomalous = False
        bbox_xyxy: List[float]  # Store current bbox in xyxy format initially

        # --- BBox Processing and Normalization ---
        if normalize:
            # Get image dimensions for normalization
            image_path = get_original_image_path(image_id)
            w, h = get_image_size(image_path)

            # Prevent division by zero if image dimensions are invalid
            if w <= 0 or h <= 0:
                logger.warning(
                    f"Invalid image dimensions (w={w}, h={h}) for "
                    f"image_id={image_id}. Skipping bbox."
                )
                anomalous_count += 1
                continue

            # Normalize coordinates
            bbox_xyxy = [x_min / w, y_min / h, x_max / w, y_max / h]

            # Check if normalized coordinates are within [0, 1] and clamp if not
            for i in range(4):
                if not (0 <= bbox_xyxy[i] <= 1):
                    is_anomalous = True
                    # Clamp the coordinate to be within [0, 1]
                    bbox_xyxy[i] = max(0.0, min(1.0, bbox_xyxy[i]))
                    # Optional: Add more detailed logging here if needed
                    # print(f"Warning: Clamped bbox coordinate {i} for image {image_id}. Original value outside [0,1].")
        else:
            # Use absolute coordinates
            bbox_xyxy = [x_min, y_min, x_max, y_max]

        # --- BBox Validation (xyxy format) ---
        # Check if x_min < x_max and y_min < y_max.
        # This ensures the box has a positive width and height.
        is_valid_bbox = bbox_xyxy[0] < bbox_xyxy[2] and bbox_xyxy[1] < bbox_xyxy[3]

        if is_valid_bbox:
            # --- BBox Format Conversion ---
            # Convert to the desired format if necessary *after* validation
            if bbox_format == "cxcywh":
                final_bbox = xyxy_to_cxcywh(bbox_xyxy)
            else:  # bbox_format == 'xyxy'
                final_bbox = bbox_xyxy

            # --- Store Valid BBox ---
            # Use nested try-except blocks to handle dictionary initialization
            # This avoids checking if keys exist beforehand but can be less readable.
            # Structure: image_id -> class_name -> list_of_bboxes
            try:
                # Attempt to append to existing list for the class
                image_id_2_bboxes[image_id][class_name].append(final_bbox)
            except KeyError:
                try:
                    # Attempt to create the list for a new class
                    image_id_2_bboxes[image_id][class_name] = [final_bbox]
                except KeyError:
                    # Create the entry for a new image_id
                    image_id_2_bboxes[image_id] = {class_name: [final_bbox]}
        else:
            # Mark as anomalous if the basic xyxy validation failed
            is_anomalous = True

        # Increment count if anomalous for any reason (normalization or validation)
        if is_anomalous:
            anomalous_count += 1

    # Report the number of anomalous bboxes found during processing
    if anomalous_count > 0:
        logger.warning(f"Anomalous bboxes found/processed: {anomalous_count} of {len(df)}")

    # --- Optional Label Improvement ---
    if improve_labels:
        # Apply specific modifications to improve label quality/consistency
        logger.info(f"{ANSI_ORANGE_BOLD}NOTE: Improving VinDr-CXR bbox labels ...{ANSI_RESET}")

        # 1. Create an "Abnormal finding" class encompassing all findings
        for image_id in list(image_id_2_bboxes.keys()): # Iterate over copy of keys
            bboxes = image_id_2_bboxes[image_id]
            af_bboxes = []
            # Collect all bboxes present in the image, regardless of original class
            for class_name, bbox_list in bboxes.items():
                af_bboxes.extend(bbox_list)

            # If there are any bounding boxes, add the 'Abnormal finding' entry
            if len(af_bboxes) > 0:
                bboxes["Abnormal finding"] = af_bboxes

        # 2. Merge specific subclasses into a general "Lung Opacity" class
        for image_id in list(image_id_2_bboxes.keys()): # Iterate over copy of keys
            bboxes = image_id_2_bboxes[image_id]
            lo_bboxes = []
            # Collect all bboxes from classes considered as Lung Opacity
            for class_name in LUNG_OPACITY_CLASSES:
                if class_name in bboxes:
                    lo_bboxes.extend(bboxes[class_name])
            # If any such bboxes were found, add/replace the 'Lung Opacity' entry
            if len(lo_bboxes) > 0:
                bboxes["Lung Opacity"] = lo_bboxes # Overwrites if 'Lung Opacity' already existed

    # --- Format for Training ---
    if for_training:
        logger.info(f"class_id_offset: {class_id_offset}")
        # Select the appropriate list of class names
        if improve_labels:
            class_names = VINBIG_BBOX_NAMES__MODIFIED  # Use modified names
        else:
            class_names = VINBIG_BBOX_NAMES  # Use original names

        # Create a new dictionary for the training format
        training_formatted_bboxes: Dict[
            str, Tuple[List[List[float]], List[int]]
        ] = {}

        # Convert the dictionary structure
        for image_id, class_bboxes_map in image_id_2_bboxes.items():
            bbox_list_all = []
            class_id_list_all = []
            # Iterate through classes found in the image
            for class_name, bboxes_for_class in class_bboxes_map.items():
                # Find the index of the class name and add offset
                class_id = (
                    class_names.index(class_name) + class_id_offset
                )
                # Add all bboxes for this class and their corresponding class ID
                for bbox in bboxes_for_class:
                    bbox_list_all.append(bbox)
                    class_id_list_all.append(class_id)
            # Store the aggregated lists for the image
            training_formatted_bboxes[image_id] = (
                bbox_list_all,
                class_id_list_all,
            )
        # Replace the original dictionary with the training-formatted one
        image_id_2_bboxes = training_formatted_bboxes

    # Return the final dictionary (structure depends on for_training)
    return image_id_2_bboxes


def load_train_image_id_2_bboxes(
    for_training: bool = False,
    normalize: bool = False,
    class_id_offset: int = 0,
    improve_labels: bool = False,
    bbox_format: Literal["xyxy", "cxcywh"] = "xyxy",
) -> BboxDictType:
    """Loads training bounding box annotations from the predefined CSV path.

    This is a convenience wrapper around `_load_image_id_2_bboxes` that
    specifically loads data from the path defined by
    `VINBIG_ANNOTATIONS_TRAIN_CSV_PATH`.

    Args:
        for_training: If True, format output for training (list of bboxes
            and list of class IDs per image). See `_load_image_id_2_bboxes`.
            Defaults to False.
        normalize: If True, normalize bbox coordinates to [0, 1].
            See `_load_image_id_2_bboxes`. Defaults to False.
        class_id_offset: Offset added to class IDs if `for_training` is True.
            See `_load_image_id_2_bboxes`. Defaults to 0.
        improve_labels: If True, apply label merging/modification rules.
            See `_load_image_id_2_bboxes`. Defaults to False.
        bbox_format: The desired output format ('xyxy' or 'cxcywh').
            See `_load_image_id_2_bboxes`. Defaults to 'xyxy'.

    Returns:
        A dictionary containing the bounding box information, formatted
        according to the arguments. The exact structure depends on the
        `for_training` flag. See `_load_image_id_2_bboxes` return type
        for details.
    """
    return _load_image_id_2_bboxes(
        csv_path=VINBIG_ANNOTATIONS_TRAIN_CSV_PATH,
        for_training=for_training,
        normalize=normalize,
        class_id_offset=class_id_offset,
        improve_labels=improve_labels,
        bbox_format=bbox_format,  # Pass the format down
    )


def load_test_image_id_2_bboxes(
    for_training: bool = False,
    normalize: bool = False,
    class_id_offset: int = 0,
    improve_labels: bool = False,
    bbox_format: Literal["xyxy", "cxcywh"] = "xyxy",
) -> BboxDictType:
    """Loads test bounding box annotations from the predefined CSV path.

    This is a convenience wrapper around `_load_image_id_2_bboxes` that
    specifically loads data from the path defined by
    `VINBIG_ANNOTATIONS_TEST_CSV_PATH`.

    Args:
        for_training: If True, format output for training (list of bboxes
            and list of class IDs per image). See `_load_image_id_2_bboxes`.
            Defaults to False. (Typically False for test set evaluation).
        normalize: If True, normalize bbox coordinates to [0, 1].
            See `_load_image_id_2_bboxes`. Defaults to False.
        class_id_offset: Offset added to class IDs if `for_training` is True.
            See `_load_image_id_2_bboxes`. Defaults to 0.
        improve_labels: If True, apply label merging/modification rules.
            See `_load_image_id_2_bboxes`. Defaults to False. (Use with
            caution on test data, depending on evaluation needs).
        bbox_format: The desired output format ('xyxy' or 'cxcywh').
            See `_load_image_id_2_bboxes`. Defaults to 'xyxy'.

    Returns:
        A dictionary containing the bounding box information, formatted
        according to the arguments. The exact structure depends on the
        `for_training` flag. See `_load_image_id_2_bboxes` return type
        for details.
    """
    return _load_image_id_2_bboxes(
        csv_path=VINBIG_ANNOTATIONS_TEST_CSV_PATH,
        for_training=for_training,
        normalize=normalize,
        class_id_offset=class_id_offset,
        improve_labels=improve_labels,
        bbox_format=bbox_format,  # Pass the format down
    )


def compute_masks_and_binary_labels_from_bounding_boxes(mask_height, mask_width, bbox_coords, bbox_classes, num_bbox_classes, flatten_grid=True):
    mask = np.zeros((num_bbox_classes, mask_height, mask_width), dtype=np.float32)
    binary_labels = np.zeros((num_bbox_classes,), dtype=np.float32)
    for bbox, class_id in zip(bbox_coords, bbox_classes):
        x1, y1, x2, y2 = bbox
        compute_mask_from_bounding_box(mask_height, mask_width, x1, y1, x2, y2, mask=mask[class_id])
        binary_labels[class_id] = 1
    if flatten_grid:
        mask = mask.reshape((num_bbox_classes, -1))
    mask = (mask > 0).astype(np.float32) # binarize
    return mask, binary_labels

def _merge_labels(*labels_list):
    merged = np.zeros((len(VINBIG_LABELS),), np.int8)
    merged[-1] = 1
    
    # # First check: majority thinks it's healthy
    # healthy_count = 0
    # for labels in labels_list:
    #     if labels[-1] == 1:
    #         assert labels.sum() == 1
    #         healthy_count += 1
    #     else:
    #         assert labels.sum() > 0
    # if healthy_count >= len(labels_list) - 1:
    #     return merged

    # General case: union of labels
    for labels in labels_list:
        if labels[-1] == 0: # no findings
            merged[-1] = 0
        for i in range(0, len(VINBIG_LABELS)-1): # findings
            if labels[i] == 1:
                merged[i] = 1
    return merged

# def _sanity_check_train_labels(labels):
#     print('Sanity checking train labels ...')
#     df_labels = pd.read_csv(VINBIG_LABELS_CSV_PATH) # these are labels from kaggle's challenge
#     label_names = df_labels.columns[1:]
#     label_indices = [VINBIG_LABELS.index(x) for x in label_names]
#     gt_labels = df_labels[label_names].values
#     assert labels.shape[0] == gt_labels.shape[0]
#     assert labels.shape[1] > gt_labels.shape[1]
#     m = gt_labels.shape[1]
#     mismatches = 0
#     for i in range(N_IMAGES_TRAIN):
#         try:
#             assert all(labels[i][label_indices[j]] == gt_labels[i][j] for j in range(m))
#         except AssertionError:
#             mismatches += 1
#             if mismatches > 4: # tolerate no more than 4 mismatches (empirical heuristic)
#                 raise
#     print('Done!')

# def load_labels(sanity_check=True):
def load_labels(improve_labels=False):
    """
    Returns:
    - train_image_id_2_labels: dict, image_id -> labels
    - test_image_id_2_labels: dict, image_id -> labels
    """

    # Train & test label dataframes
    df_labels_train = pd.read_csv(VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH)
    df_labels_test = pd.read_csv(VINBIG_IMAGE_LABELS_TEST_CSV_PATH)

    # Images ids
    image_ids = [None] * (N_IMAGES_TRAIN + N_IMAGES_TEST)
    # train image ids
    train_image_ids = df_labels_train['image_id']
    for i in range(N_IMAGES_TRAIN):
        image_ids[i] = train_image_ids[i * 3]
        for j in range(1, 3):
            assert train_image_ids[i * 3 + j] == image_ids[i]
        assert image_ids[i] != image_ids[i-1]
    # test image ids
    test_images_ids = df_labels_test['image_id']
    for i in range(N_IMAGES_TEST):
        image_ids[N_IMAGES_TRAIN + i] = test_images_ids[i]

    # Labels
    train_image_id_2_labels = {}
    test_image_id_2_labels = {}
        
    # Train labels
    tmp = VINBIG_LABELS[:]
    tmp[tmp.index('Other disease')] = 'Other diseases' # HACK
    train_labels = df_labels_train[tmp].values
    for i in range(N_IMAGES_TRAIN):
        labels = _merge_labels(
            train_labels[3 * i],
            train_labels[3 * i + 1],
            train_labels[3 * i + 2]
        )
        train_image_id_2_labels[image_ids[i]] = labels
    
    # Test labels
    test_labels = df_labels_test[VINBIG_LABELS].values
    for i in range(N_IMAGES_TEST):
        test_image_id_2_labels[image_ids[N_IMAGES_TRAIN + i]] = test_labels[i]

    # # Sanity check train labels
    # if sanity_check:
    #     labels_matrix = np.zeros((N_IMAGES_TRAIN + N_IMAGES_TEST, len(VINBIG_LABELS)), np.int8)
    #     for i in range(N_IMAGES_TRAIN + N_IMAGES_TEST):
    #         labels_matrix[i] = image_id_2_labels[image_ids[i]]
    #     _sanity_check_train_labels(labels_matrix)

    if improve_labels:
        # We will apply some modifications to improve the labels
        logger.warning(f'NOTE: Improving VinDr-CXR classification labels ...')
        
        # 1. The class "Lung Opacity" should subsume the following classes as they are subcategories of it:
        #    - "Lung Opacity" (already included)
        #    - "Consolidation"
        #    - "Nodule/Mass"
        #    - "Atelectasis"
        #    - "Pulmonary fibrosis"
        #    - "Edema"
        #    - "Infiltration"
        #    - "ILD"
        lung_opacity_idx = VINBIG_LABELS.index('Lung Opacity')
        for class_name in LUNG_OPACITY_CLASSES:
            class_idx = VINBIG_LABELS.index(class_name)
            for labels in train_image_id_2_labels.values():
                if labels[class_idx] == 1:
                    labels[lung_opacity_idx] = 1
            for labels in test_image_id_2_labels.values():
                if labels[class_idx] == 1:
                    labels[lung_opacity_idx] = 1
        
        # 2. The class "No finding" will be converted to "Abnormal finding" (i.e., the logical negation of "No finding"),
        #    and all the bounding boxes present in an image will be assigned to this class.
        no_finding_idx = VINBIG_LABELS.index('No finding')
        for labels in train_image_id_2_labels.values():
            labels[no_finding_idx] = 1 - labels[no_finding_idx]
        for labels in test_image_id_2_labels.values():
            labels[no_finding_idx] = 1 - labels[no_finding_idx]

    # Return
    return train_image_id_2_labels, test_image_id_2_labels

def print_labels(labels):
    for i, label in enumerate(VINBIG_LABELS):
        if labels[i] == 1:
            print(f'{i}: {label} ({labels[i]})')

def get_original_image_path(image_id):
    return os.path.join(VINBIG_ORIGINAL_IMAGES_FOLDER, f'{image_id}.jpg')

def get_medium_size_image_path(image_id):
    return os.path.join(VINBIG_512x512_IMAGES_FOLDER, f'{image_id}.jpg')

def visualize_image_with_bounding_boxes(image_id, bbox_dict, figsize=(10, 10), denormalize=False, verbose=False,
                                        allowed_classes=None, class_to_draw_last=None, bbox_class_names=VINBIG_BBOX_NAMES):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    fig, ax = plt.subplots(1, figsize=figsize)
    image_path = get_original_image_path(image_id)
    print('image_path:', image_path)
    image = Image.open(image_path)
    image = image.convert('RGB')
    ax.imshow(image)
    if denormalize:
        w, h = get_image_size(image_path)
    class_names = list(bbox_dict.keys())
    class_names.sort()
    if class_to_draw_last is not None and class_to_draw_last in class_names:
        class_names.remove(class_to_draw_last)
        class_names.append(class_to_draw_last)
    for i, class_name in enumerate(class_names):
        if allowed_classes is not None and class_name not in allowed_classes:
            continue
        bbox_list = bbox_dict[class_name]
        if verbose:
            print(f'{i}: {class_name}')
            print(bbox_list)
        color_idx = bbox_class_names.index(class_name)
        for bbox in bbox_list:
            if denormalize:
                bbox = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=3, edgecolor=plt.cm.tab20(color_idx), facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], class_name, fontsize=10, bbox=dict(facecolor='white', alpha=0.3))
    plt.show()

class VinBigBBoxVisualizer:
    def __init__(self):
        # Load data
        self.train_image_id_2_bboxes = load_train_image_id_2_bboxes()
        self.test_image_id_2_bboxes = load_test_image_id_2_bboxes()
        self.train_image_id_to_labels, self.test_image_id_to_labels = load_labels()
        self.train_image_ids = list(self.train_image_id_to_labels.keys())
        self.test_image_ids = list(self.test_image_id_to_labels.keys())
        print(f'len(self.train_image_id_2_bboxes): {len(self.train_image_id_2_bboxes)}')
        print(f'len(self.test_image_id_2_bboxes): {len(self.test_image_id_2_bboxes)}')
        print(f'len(self.train_image_id_to_labels): {len(self.train_image_id_to_labels)}')
        print(f'len(self.test_image_id_to_labels): {len(self.test_image_id_to_labels)}')

    def visualize_train_image(self, image_id=None, class_name=None, excluded_class_name=None, exclude_other_classes=False,
                              figsize=(10, 10), verbose=False):
        self._visualize_image(image_ids=self.train_image_ids,
                              image_id_2_bboxes=self.train_image_id_2_bboxes,
                              image_id_to_labels=self.train_image_id_to_labels,
                              image_id=image_id,
                              class_name=class_name,
                              excluded_class_name=excluded_class_name,
                              exclude_other_classes=exclude_other_classes,
                              figsize=figsize,
                              verbose=verbose)
        
    def visualize_test_image(self, image_id=None, class_name=None, excluded_class_name=None, exclude_other_classes=False,
                             figsize=(10, 10), verbose=False):
        self._visualize_image(image_ids=self.test_image_ids,
                              image_id_2_bboxes=self.test_image_id_2_bboxes,
                              image_id_to_labels=self.test_image_id_to_labels,
                              image_id=image_id,
                              class_name=class_name,
                              excluded_class_name=excluded_class_name,
                              exclude_other_classes=exclude_other_classes,
                              figsize=figsize,
                              verbose=verbose)

    def _visualize_image(self, image_ids, image_id_2_bboxes, image_id_to_labels, image_id=None, class_name=None, excluded_class_name=None,
                         exclude_other_classes=False, figsize=(10, 10), verbose=False):
        if image_id is None:
            if class_name is not None:
                candidate_image_ids = []
                for image_id, bboxes in image_id_2_bboxes.items():
                    if class_name in bboxes:
                        candidate_image_ids.append(image_id)
                assert len(candidate_image_ids) > 0
                image_id = random.choice(candidate_image_ids)
            elif excluded_class_name is not None:
                candidate_image_ids = []
                for image_id in image_ids:
                    try:
                        bboxes = image_id_2_bboxes[image_id]
                        if excluded_class_name not in bboxes:
                            candidate_image_ids.append(image_id)
                    except KeyError:
                        candidate_image_ids.append(image_id)
                assert len(candidate_image_ids) > 0
                image_id = random.choice(candidate_image_ids)
            else:
                image_id = random.choice(image_ids)
        try:
            bbox_dict = image_id_2_bboxes[image_id]
        except KeyError:
            bbox_dict = {}
        print('Labels:', [VINBIG_LABELS[i] for i in range(len(VINBIG_LABELS)) if image_id_to_labels[image_id][i] == 1])
        if class_name is not None and exclude_other_classes:
            allowed_classes = [class_name]
        else:
            allowed_classes = None
        visualize_image_with_bounding_boxes(image_id, bbox_dict, figsize=figsize, verbose=verbose, allowed_classes=allowed_classes,
                                            class_to_draw_last=class_name)

def compute_label_frequencies(improve_labels=False):
    train_image_id_2_labels, test_image_id_2_labels = load_labels(improve_labels=improve_labels)

    # Train labels
    df = pd.read_csv(VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH)
    train_image_ids = df['image_id'].unique()
    train_label_frequencies = np.zeros((len(VINBIG_LABELS),), np.int32)
    for image_id in train_image_ids:
        labels = train_image_id_2_labels[image_id]
        train_label_frequencies += labels

    # Test labels
    df = pd.read_csv(VINBIG_IMAGE_LABELS_TEST_CSV_PATH)
    test_image_ids = df['image_id'].unique()
    test_label_frequencies = np.zeros((len(VINBIG_LABELS),), np.int32)
    for image_id in test_image_ids:
        labels = test_image_id_2_labels[image_id]
        test_label_frequencies += labels

    return {
        'train': { VINBIG_LABELS[i]: train_label_frequencies[i] for i in range(len(VINBIG_LABELS)) },
        'test': { VINBIG_LABELS[i]: test_label_frequencies[i] for i in range(len(VINBIG_LABELS)) }
    }


