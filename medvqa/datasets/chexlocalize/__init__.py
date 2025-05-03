from dotenv import load_dotenv
load_dotenv()

from medvqa.utils.common import LARGE_FAST_CACHE_DIR
from medvqa.utils.files_utils import load_json
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

CHEXLOCALIZE_DATA_DIR = os.environ['CHEXLOCALIZE_DATA_DIR']
CHEXLOCALIZE_WORKSPACE_DIR = os.environ['CHEXLOCALIZE_WORKSPACE_DIR']
CHEXLOCALIZE_CHEXPERT_TEST_LABELS_CSV_PATH = os.environ['CHEXLOCALIZE_CHEXPERT_TEST_LABELS_CSV_PATH']
CHEXLOCALIZE_CHEXPERT_VAL_LABELS_CSV_PATH = os.environ['CHEXLOCALIZE_CHEXPERT_VAL_LABELS_CSV_PATH']
CHEXLOCALIZE_TEST_GT_ANNOTATIONS_JSON_PATH = os.environ['CHEXLOCALIZE_TEST_GT_ANNOTATIONS_JSON_PATH']
CHEXLOCALIZE_VAL_GT_ANNOTATIONS_JSON_PATH = os.environ['CHEXLOCALIZE_VAL_GT_ANNOTATIONS_JSON_PATH']
CHEXLOCALIZE_LARGE_FAST_CACHE_DIR = os.path.join(LARGE_FAST_CACHE_DIR, 'chexlocalize')
CHEXLOCALIZE_IMAGE_DIR_512X512 = os.path.join(CHEXLOCALIZE_WORKSPACE_DIR, 'CheXpert_512x512')

CHEXLOCALIZE_CLASS_NAMES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Airspace Opacity",
                            "Edema", "Consolidation", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Support Devices"]

CHEXLOCALIZE_CLASS_NAME2PHRASE = {
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinum seen",
    "Cardiomegaly": "cardiomegaly seen",
    "Lung Lesion": "lung lesion seen",
    "Airspace Opacity": "airspace opacity seen",
    "Edema": "edema seen",
    "Consolidation": "consolidation seen",
    "Atelectasis": "atelectasis seen",
    "Pneumothorax": "pneumothorax seen",
    "Pleural Effusion": "pleural effusion seen",
    "Support Devices": "support devices seen"
}

def extract_images_segmentation_masks_and_binary_labels(mask_height, mask_width, image_dir=CHEXLOCALIZE_IMAGE_DIR_512X512,
                                                        target_image_size=(512, 512), flatten_masks=False, return_polygons=False):
    """
    Extracts images, segmentation masks and binary labels from CheXpert dataset (CheXlocalize subset).
    """
    test_labels_df = pd.read_csv(CHEXLOCALIZE_CHEXPERT_TEST_LABELS_CSV_PATH)
    val_labels_df = pd.read_csv(CHEXLOCALIZE_CHEXPERT_VAL_LABELS_CSV_PATH)
    test_gt_annotations = load_json(CHEXLOCALIZE_TEST_GT_ANNOTATIONS_JSON_PATH)
    val_gt_annotations = load_json(CHEXLOCALIZE_VAL_GT_ANNOTATIONS_JSON_PATH)

    val_labels = np.zeros((val_labels_df.shape[0], len(CHEXLOCALIZE_CLASS_NAMES)))
    test_labels = np.zeros((test_labels_df.shape[0], len(CHEXLOCALIZE_CLASS_NAMES)))
    val_masks = np.zeros((val_labels.shape[0], val_labels.shape[1], mask_height, mask_width))
    test_masks = np.zeros((test_labels.shape[0], test_labels.shape[1], mask_height, mask_width))
    image_paths = []
    if return_polygons:
        val_polygons = [[None] * len(CHEXLOCALIZE_CLASS_NAMES) for _ in range(val_labels_df.shape[0])]
        test_polygons = [[None] * len(CHEXLOCALIZE_CLASS_NAMES) for _ in range(test_labels_df.shape[0])]

    without_masks = 0
    
    for i in range(val_labels.shape[0]):
        path = val_labels_df.iloc[i]['Path'] 
        key = path[20:-4].replace('/', '_') # remove 'CheXpert-v1.0/valid/' prefix and '.jpg' suffix
        image_paths.append(os.path.join(image_dir, 'val', path[20:]))
        assert os.path.exists(image_paths[-1]), f"Image path {image_paths[-1]} does not exist."
        try:
            data = val_gt_annotations[key]
        except KeyError:
            without_masks += 1
            continue
        image_height, image_width = data['img_size']
        for j in range(val_labels.shape[1]):
            for k in data.keys():
                if k == 'img_size': continue                
                assert k in CHEXLOCALIZE_CLASS_NAMES, f"Invalid class name {k}"
            try:
                polygons = data[CHEXLOCALIZE_CLASS_NAMES[j]]
            except KeyError:
                    continue
            assert len(polygons) > 0, f"Empty polygon for {CHEXLOCALIZE_CLASS_NAMES[j]} in {key}"
            mask = Image.new('1', (mask_width, mask_height))
            for polygon in polygons:
                coords = [(point[0] * mask_width / image_width, point[1] * mask_height / image_height) for point in polygon]
                ImageDraw.Draw(mask).polygon(coords, outline=1, fill=1)
            val_masks[i, j] = np.array(mask, dtype=float)
            val_labels[i, j] = 1.0
            if return_polygons:
                val_polygons[i][j] = [[(p[0] * target_image_size[0] / image_width, p[1] * target_image_size[1] / image_height) for p in pol] for pol in polygons]

    for i in range(test_labels.shape[0]):
        path = test_labels_df.iloc[i]['Path']
        key = path[5:-4].replace('/', '_') # remove 'test/' prefix and '.jpg' suffix
        image_paths.append(os.path.join(image_dir, 'test', path[5:]))
        assert os.path.exists(image_paths[-1]), f"Image path {image_paths[-1]} does not exist."
        try:
            data = test_gt_annotations[key]
        except KeyError:
            without_masks += 1
            continue
        try:
            image_height, image_width = data['img_size']
        except KeyError:
            without_masks += 1
            continue
        for j in range(test_labels.shape[1]):
            for k in data.keys():
                if k == 'img_size': continue
                assert k in CHEXLOCALIZE_CLASS_NAMES, f"Invalid class name {k}"
            try:
                polygons = data[CHEXLOCALIZE_CLASS_NAMES[j]]
            except KeyError:
                continue
            assert len(polygons) > 0, f"Empty polygon for {CHEXLOCALIZE_CLASS_NAMES[j]} in {key}"
            mask = Image.new('1', (mask_width, mask_height))
            for polygon in polygons:
                coords = [(point[0] * mask_width / image_width, point[1] * mask_height / image_height) for point in polygon]
                ImageDraw.Draw(mask).polygon(coords, outline=1, fill=1)
            test_masks[i, j] = np.array(mask, dtype=float)
            test_labels[i, j] = 1.0
            if return_polygons:
                test_polygons[i][j] = [[(p[0] * target_image_size[0] / image_width, p[1] * target_image_size[1] / image_height) for p in pol] for pol in polygons]
                
    print(f"Without masks: {without_masks}/{val_labels.shape[0] + test_labels.shape[0]}")

    masks = np.concatenate((val_masks, test_masks), axis=0)
    labels = np.concatenate((val_labels, test_labels), axis=0)
    if flatten_masks:
        masks = masks.reshape(masks.shape[0], masks.shape[1], -1)
    if return_polygons:
        polygons = val_polygons + test_polygons
        polygon_names = [[] for _ in range(len(polygons))]
        for i in range(len(polygons)):
            for j in range(len(polygons[i])):
                if polygons[i][j] is not None:
                    polygon_names[i].append(CHEXLOCALIZE_CLASS_NAMES[j])
            polygons[i] = [p for p in polygons[i] if p is not None]
        return image_paths, masks, labels, polygons, polygon_names
    return image_paths, masks, labels