import argparse
import os
import numpy as np
from tqdm import tqdm
from medvqa.utils.files import load_pickle, save_to_pickle
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
    args = parser.parse_args()

    # Extract labels from scene graphs
    print('Extracting labels from scene graphs')
    cache_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'raw_labels_from_scene_graphs.pkl')
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
        for i, scene_graph in enumerate(tqdm(scene_graphs)):
            image_id = scene_graph['image_id']
            image_id_2_raw_labels[image_id] = extract_labels_from_scene_graph(scene_graph)
        save_to_pickle(image_id_2_raw_labels, cache_path)
        print(f'Labels saved to cache ({cache_path})')
    
    # Count the number of times each label occurs
    print('Counting label frequencies')
    label_counts = {}
    for labels in image_id_2_raw_labels.values():
        for a, b, c, d in labels:
            if c == 'normal':
                c = 'abnormal'
                d = 1 - d
            if d == 1:
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

    # Create a mapping from image_id to binary labels
    imageId2binaryLabels = {}
    for image_id, raw_labels in image_id_2_raw_labels.items():        
        binary_labels = np.zeros(len(unique_labels), dtype=np.int8)
        # Determine localized labels first
        localized_indices = []
        for a, b, c, d in raw_labels:
            if c == 'normal':
                c = 'abnormal'
                d = 1 - d
            try:
                idx = label_to_index[(a, b, c)]
                binary_labels[idx] = d
                localized_indices.append(idx)
            except KeyError:
                pass
        # Determine non-localized labels next
        for idx in localized_indices:
            a, b, c = unique_labels[idx]
            nonloc_idx = label_to_index[(b, c)]
            binary_labels[nonloc_idx] = max(binary_labels[nonloc_idx], binary_labels[idx])
        # Save binary labels
        imageId2binaryLabels[image_id] = binary_labels

    # Save labels
    labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'labels(min_freq={args.min_freq}).pkl')
    imageId2labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'imageId2labels(min_freq={args.min_freq}).pkl')
    print(f'Saving labels to {labels_path}')
    save_to_pickle(unique_labels, labels_path)
    print(f'Saving imageId2labels to {imageId2labels_path}')
    save_to_pickle(imageId2binaryLabels, imageId2labels_path)
    print('Done')