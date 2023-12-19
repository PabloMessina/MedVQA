from dotenv import load_dotenv
from imagesize import get as get_image_size
from medvqa.datasets.segmentation_utils import compute_mask_from_bounding_box

from medvqa.utils.constants import VINBIG_BBOX_NAMES, VINBIG_LABELS
load_dotenv()

import os
import numpy as np
import pandas as pd

VINBIG_ORIGINAL_IMAGES_FOLDER = os.environ['VINBIG_ORIGINAL_IMAGES_FOLDER']
VINBIG_512x512_IMAGES_FOLDER = os.environ['VINBIG_512x512_IMAGES_FOLDER']
VINBIG_ALL_IMAGES_TXT_PATH  = os.environ['VINBIG_ALL_IMAGES_TXT_PATH']
VINBIG_TRAIN_VAL_IMAGES_TXT_PATH = os.environ['VINBIG_TRAIN_VAL_IMAGES_TXT_PATH']
VINBIG_TEST_IMAGES_TXT_PATH = os.environ['VINBIG_TEST_IMAGES_TXT_PATH']
VINBIG_LABELS_CSV_PATH = os.environ['VINBIG_LABELS_CSV_PATH']
VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH = os.environ['VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH']
VINBIG_IMAGE_LABELS_TEST_CSV_PATH = os.environ['VINBIG_IMAGE_LABELS_TEST_CSV_PATH']
VINBIG_ANNOTATIONS_TRAIN_CSV_PATH = os.environ['VINBIG_ANNOTATIONS_TRAIN_CSV_PATH']
VINBIG_ANNOTATIONS_TEST_CSV_PATH = os.environ['VINBIG_ANNOTATIONS_TEST_CSV_PATH']

from medvqa.utils.common import CACHE_DIR, FAST_CACHE_DIR, LARGE_FAST_CACHE_DIR
VINBIG_CACHE_DIR = os.path.join(CACHE_DIR, 'vinbig')
VINBIG_FAST_CACHE_DIR = os.path.join(FAST_CACHE_DIR, 'vinbig')
VINBIG_LARGE_FAST_CACHE_DIR = os.path.join(LARGE_FAST_CACHE_DIR, 'vinbig')

N_IMAGES_TRAIN = 15000
N_IMAGES_TEST = 3000

def get_unique_bbox_names():
    unique_bbox_names = set()
    for csv_path in [VINBIG_ANNOTATIONS_TRAIN_CSV_PATH, VINBIG_ANNOTATIONS_TEST_CSV_PATH]:
        df = pd.read_csv(csv_path)
        df = df[df['x_min'].notna()]
        for class_name in df['class_name']:
            unique_bbox_names.add(class_name)
    return sorted(list(unique_bbox_names))

def _load_image_id_2_bboxes(csv_path, for_training=False, normalize=False, class_id_offset=0):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df['x_min'].notna()]
    image_id_2_bboxes = {}
    anomalous_count = 0
    for image_id, class_name, x_min, y_min, x_max, y_max in zip(
        df['image_id'], df['class_name'], df['x_min'], df['y_min'], df['x_max'], df['y_max']
    ):  
        is_anomalous = False
        if normalize:
            image_path = get_original_image_path(image_id)
            w, h = get_image_size(image_path)
            bbox = [x_min/w, y_min/h, x_max/w, y_max/h]
            for i in range(4):
                # assert -1e-1 <= bbox[i] <= 1 + 1e-1, f'bbox {bbox} is not normalized (image_id={image_id}, x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, w={w}, h={h})'
                if bbox[i] < 0 or bbox[i] > 1:
                    is_anomalous = True
                    bbox[i] = max(0, min(1, bbox[i]))
        else:
            bbox = [x_min, y_min, x_max, y_max]
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]: # x_min < x_max and y_min < y_max
            try:
                image_id_2_bboxes[image_id][class_name].append(bbox)
            except KeyError:
                try:
                    image_id_2_bboxes[image_id][class_name] = [bbox]
                except KeyError:
                    image_id_2_bboxes[image_id] = {class_name: [bbox]}
        else:
            is_anomalous = True
        if is_anomalous:
            anomalous_count += 1
    print(f'Anomalous bboxes found: {anomalous_count} of {len(df)}')
    if for_training:
        print('class_id_offset:', class_id_offset)
        for image_id, bboxes in image_id_2_bboxes.items():
            bbox_list = []
            class_list = []
            for class_name, bboxes in bboxes.items():
                class_id = VINBIG_BBOX_NAMES.index(class_name) + class_id_offset
                for bbox in bboxes:
                    bbox_list.append(bbox)
                    class_list.append(class_id)
            image_id_2_bboxes[image_id] = bbox_list, class_list
                    
    return image_id_2_bboxes

def load_train_image_id_2_bboxes(for_training=False, normalize=False, class_id_offset=0):
    return _load_image_id_2_bboxes(VINBIG_ANNOTATIONS_TRAIN_CSV_PATH, for_training, normalize, class_id_offset)

def load_test_image_id_2_bboxes(for_training=False, normalize=False, class_id_offset=0):
    return _load_image_id_2_bboxes(VINBIG_ANNOTATIONS_TEST_CSV_PATH, for_training, normalize, class_id_offset)

def compute_masks_and_binary_labels_from_bounding_boxes(mask_height, mask_width, bbox_coords, bbox_classes, flatten_grid=True):
    mask = np.zeros((len(VINBIG_BBOX_NAMES), mask_height, mask_width), dtype=np.float32)
    binary_labels = np.zeros((len(VINBIG_BBOX_NAMES),), dtype=np.float32)
    for bbox, class_id in zip(bbox_coords, bbox_classes):
        x1, y1, x2, y2 = bbox
        compute_mask_from_bounding_box(mask_height, mask_width, x1, y1, x2, y2, mask=mask[class_id])
        binary_labels[class_id] = 1
    if flatten_grid:
        mask = mask.reshape((len(VINBIG_BBOX_NAMES), -1))
    return mask, binary_labels

def _merge_labels(*labels_list):
    merged = np.zeros((len(VINBIG_LABELS),), np.int8)
    merged[-1] = 1
    
    # First check: majority thinks it's healthy
    healthy_count = 0
    for labels in labels_list:
        if labels[-1] == 1:
            assert labels.sum() == 1
            healthy_count += 1
        else:
            assert labels.sum() > 0
    if healthy_count >= len(labels_list) - 1:
        return merged

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
def load_labels():

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
    image_id_2_labels = {}
        
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
        image_id_2_labels[image_ids[i]] = labels
    
    # Test labels
    test_labels = df_labels_test[VINBIG_LABELS].values
    for i in range(N_IMAGES_TEST):
        image_id_2_labels[image_ids[N_IMAGES_TRAIN + i]] = test_labels[i]

    # # Sanity check train labels
    # if sanity_check:
    #     labels_matrix = np.zeros((N_IMAGES_TRAIN + N_IMAGES_TEST, len(VINBIG_LABELS)), np.int8)
    #     for i in range(N_IMAGES_TRAIN + N_IMAGES_TEST):
    #         labels_matrix[i] = image_id_2_labels[image_ids[i]]
    #     _sanity_check_train_labels(labels_matrix)

    # Return
    return image_id_2_labels

def print_labels(labels):
    for i, label in enumerate(VINBIG_LABELS):
        if labels[i] == 1:
            print(f'{i}: {label} ({labels[i]})')

def get_original_image_path(image_id):
    return os.path.join(VINBIG_ORIGINAL_IMAGES_FOLDER, f'{image_id}.jpg')

def get_medium_size_image_path(image_id):
    return os.path.join(VINBIG_512x512_IMAGES_FOLDER, f'{image_id}.jpg')

def visualize_image_with_bounding_boxes(image_id, bbox_dict, figsize=(10, 10), verbose=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    fig, ax = plt.subplots(1, figsize=figsize)
    image_path = get_original_image_path(image_id)
    print('image_path:', image_path)
    image = Image.open(image_path)
    image = image.convert('RGB')
    ax.imshow(image)
    class_names = list(bbox_dict.keys())
    class_names.sort()
    for i, class_name in enumerate(class_names):
        bbox_list = bbox_dict[class_name]
        if verbose:
            print(f'{i}: {class_name}')
            print(bbox_list)
        for bbox in bbox_list:
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=3, edgecolor=plt.cm.tab20(i), facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], class_name, fontsize=10, bbox=dict(facecolor='white', alpha=0.3))
    plt.show()