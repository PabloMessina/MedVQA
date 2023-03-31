import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_ATTRIBUTES_DICT,
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_CACHE_DIR,
    CHEST_IMAGENOME_GOLD_ATTRIBUTE_RELATIONS_TXT_PATH,
    CHEST_IMAGENOME_OTHER_REGIONS,
)
from medvqa.utils.files import save_to_pickle

if __name__ == '__main__':
    print(f'Reading {CHEST_IMAGENOME_GOLD_ATTRIBUTE_RELATIONS_TXT_PATH}')
    df = pd.read_csv(CHEST_IMAGENOME_GOLD_ATTRIBUTE_RELATIONS_TXT_PATH, sep='\t')

    # Collect labels
    image_id_2_raw_labels = {}
    localized_labels = set()
    global_labels = set()
    for image_id, bbox, category_id, label_name, ctx in tqdm(zip(df['image_id'], df['bbox'], 
                                                              df['categoryID'], df['label_name'], df['context'])):
        image_id = image_id[:-4]
        assert bbox in CHEST_IMAGENOME_BBOX_NAMES or bbox in CHEST_IMAGENOME_OTHER_REGIONS, bbox
        assert category_id in CHEST_IMAGENOME_ATTRIBUTES_DICT, category_id
        assert label_name in CHEST_IMAGENOME_ATTRIBUTES_DICT[category_id], label_name
        assert ctx in ['yes', 'no']
        value = 1 if ctx == 'yes' else 0
        try:
            image_id_2_raw_labels[image_id].append((bbox, category_id, label_name, value))
        except KeyError:
            image_id_2_raw_labels[image_id] = [(bbox, category_id, label_name, value)]
        if label_name == 'normal':
            label_name = 'abnormal'
            value = 1 - value
        localized_labels.add((bbox, category_id, label_name))
        global_labels.add((category_id, label_name))
    localized_labels = sorted(list(localized_labels))
    global_labels = sorted(list(global_labels))
    all_labels = localized_labels + global_labels
    print(f'Found {len(localized_labels)} localized labels')
    print(f'Found {len(global_labels)} global labels')
    print(f'Found {len(all_labels)} total labels')

    # Create a mapping from labels to indices
    label_to_index = {label: i for i, label in enumerate(all_labels)}

    # Create a mapping from image_id to binary labels
    imageId2binaryLabels = {}
    imageId2mask = {}
    imageId2contradictions = {}
    for image_id, raw_labels in tqdm(image_id_2_raw_labels.items()):
        binary_labels = np.zeros(len(all_labels), dtype=np.int8)
        mask = np.zeros(len(all_labels), dtype=np.int8)
        contradictions = np.zeros(len(all_labels), dtype=np.int8)
        seen = np.zeros(len(all_labels), dtype=np.bool)
        # Determine localized labels first
        localized_indices = []
        for a, b, c, d in raw_labels:
            if c == 'normal':
                c = 'abnormal'
                d = 1 - d
            idx = label_to_index[(a, b, c)]
            if seen[idx]:
                if binary_labels[idx] != d:
                    contradictions[idx] = 1
            else:
                seen[idx] = 1
                mask[idx] = 1
            binary_labels[idx] = max(binary_labels[idx], d)
            localized_indices.append(idx)
        # Determine global labels next
        for idx in localized_indices:
            a, b, c = all_labels[idx]
            glob_idx = label_to_index[(b, c)]
            mask[glob_idx] = 1
            binary_labels[glob_idx] = max(binary_labels[glob_idx], binary_labels[idx])
        # Save binary labels
        imageId2binaryLabels[image_id] = binary_labels
        imageId2mask[image_id] = mask
        imageId2contradictions[image_id] = contradictions

    # Save labels
    labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'gold_binary_labels.pkl')
    imageId2labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'gold_imageId2binaryLabels.pkl')
    imageId2mask_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'gold_imageId2mask.pkl')
    imageId2contradictions_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'gold_imageId2contradictions.pkl')
    print(f'Saving labels to {labels_path}')
    save_to_pickle(all_labels, labels_path)
    print(f'Saving imageId2labels to {imageId2labels_path}')
    save_to_pickle(imageId2binaryLabels, imageId2labels_path)
    print(f'Saving imageId2mask to {imageId2mask_path}')
    save_to_pickle(imageId2mask, imageId2mask_path)
    print(f'Saving imageId2contradictions to {imageId2contradictions_path}')
    save_to_pickle(imageId2contradictions, imageId2contradictions_path)
    print('Done')