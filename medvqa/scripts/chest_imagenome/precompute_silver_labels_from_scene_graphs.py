import argparse
import os
import numpy as np
from tqdm import tqdm
from medvqa.utils.hashing import hash_string
from medvqa.utils.files import load_pickle, save_pickle
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_CACHE_DIR
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    extract_labels_from_scene_graph,
    load_scene_graphs_in_parallel,  
)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--min_freq', type=int, default=100)
    parser.add_argument('--categories_to_skip', nargs='*', default=['temporal', 'severity', 'laterality'])
    parser.add_argument('--binarize_labels', action='store_true', default=False)
    args = parser.parse_args()

    # Extract labels from scene graphs
    print('Extracting labels from scene graphs')
    if len(args.categories_to_skip) > 0:
        skip_hash = hash_string(';'.join(args.categories_to_skip))
        skip_string = f'(skip_hash={skip_hash[0]},{skip_hash[1]})'
    else:
        skip_hash = None
        skip_string = ''
    cache_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'raw_labels_from_scene_graphs_{skip_string}.pkl')
    if os.path.exists(cache_path):
        image_id_2_raw_labels = load_pickle(cache_path)
        print(f'Labels loaded from cache ({cache_path})')
    else:
        # Load scene graphs
        print(f'Loading scene graphs with {args.num_workers} workers')
        scene_graphs = load_scene_graphs_in_parallel(num_workers=args.num_workers)
        print(f'Loaded {len(scene_graphs)} scene graphs')
        # Extract labels from scene graphs
        image_id_2_raw_labels = {}
        for i, scene_graph in tqdm(enumerate(scene_graphs), mininterval=2):
            image_id = scene_graph['image_id']
            image_id_2_raw_labels[image_id] = extract_labels_from_scene_graph(scene_graph, categories_to_skip=args.categories_to_skip)
        save_pickle(image_id_2_raw_labels, cache_path)
        print(f'Labels saved to cache ({cache_path})')
    
    # Count the number of times each label occurs
    print('Counting label frequencies')
    label_counts = {}
    for labels in image_id_2_raw_labels.values():
        for a, b, c, d in labels:
            if c == 'normal':
                c = 'abnormal'
                d = 1 - d
            label = (a, b, c)
            label_counts[label] = label_counts.get(label, 0) + 1

    # Filter labels by frequency
    print(f'Filtering labels by frequency >= {args.min_freq}')
    filtered_labels = sorted([label for label, count in label_counts.items() if count >= args.min_freq])
    print(f'Found {len(filtered_labels)} filtered (localized) labels')

    # Obtain non-localized labels from filtered labels
    nonlocalized_labels = sorted(list(set((b, c) for _, b, c in filtered_labels)))
    print(f'Found {len(nonlocalized_labels)} non-localized labels')

    # Final labels
    unique_labels = filtered_labels + nonlocalized_labels
    print(f'Found {len(unique_labels)} labels in total')

    # Create a mapping from labels to indices
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Create a mapping from image_id to labels
    imageId2labels = {}
    imageId2mask = {}
    imageId2contradictions = {}
    for image_id, raw_labels in tqdm(image_id_2_raw_labels.items()):
        if args.binarize_labels:
            labels = np.zeros(len(unique_labels), dtype=np.int8) # 0 means not present, 1 means present
        else:
            labels = np.full(len(unique_labels), -1, dtype=np.int8) # -1 means unknown, 0 means not present, 1 means present
        mask = np.zeros(len(unique_labels), dtype=np.int8)
        contradictions = np.zeros(len(unique_labels), dtype=np.int8)
        # Determine localized labels first
        localized_indices = []
        for a, b, c, d in raw_labels:
            if c == 'normal':
                c = 'abnormal'
                d = 1 - d
            try:
                idx = label_to_index[(a, b, c)]
            except KeyError:
                continue
            if mask[idx]:
                if labels[idx] != d:
                    contradictions[idx] = 1
            else:
                mask[idx] = 1
            labels[idx] = max(labels[idx], d)
            localized_indices.append(idx)
        # Determine non-localized labels next
        for idx in localized_indices:
            a, b, c = unique_labels[idx]
            nonloc_idx = label_to_index[(b, c)]
            mask[nonloc_idx] = 1
            labels[nonloc_idx] = max(labels[nonloc_idx], labels[idx])
        # Save labels
        imageId2labels[image_id] = labels
        imageId2mask[image_id] = mask
        imageId2contradictions[image_id] = contradictions        

    # Save labels
    _strings = [
        f'min_freq={args.min_freq}',
        f'num_labels={len(unique_labels)}',
    ]
    if skip_hash is not None:
        _strings.append(f'skip={skip_hash[0]}_{skip_hash[1]}')
    if args.binarize_labels:
        _strings.append('binarized')
    _joined_string = ','.join(_strings)
    labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'labels({_joined_string}).pkl')
    imageId2labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'imageId2labels({_joined_string}).pkl')
    imageId2mask_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'imageId2mask({_joined_string}).pkl')
    imageId2contradictions_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'imageId2contradictions({_joined_string}).pkl')
    print(f'Saving labels to {labels_path}')
    save_pickle(unique_labels, labels_path)
    print(f'Saving imageId2labels to {imageId2labels_path}')
    save_pickle(imageId2labels, imageId2labels_path)
    print(f'Saving imageId2mask to {imageId2mask_path}')
    save_pickle(imageId2mask, imageId2mask_path)
    print(f'Saving imageId2contradictions to {imageId2contradictions_path}')
    save_pickle(imageId2contradictions, imageId2contradictions_path)
    print('Done')