import argparse
import os
import numpy as np
from tqdm import tqdm
from medvqa.utils.files import save_to_pickle
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

    # Load scene graphs
    print(f'Loading scene graphs with {args.num_workers} workers')
    scene_graphs = load_scene_graphs_in_parallel(num_workers=args.num_workers)
    print(f'Loaded {len(scene_graphs)} scene graphs')

    # Extract labels
    labels_list = [None] * len(scene_graphs)
    for i, scene_graph in enumerate(tqdm(scene_graphs)):
        labels_list[i] = extract_labels_from_scene_graph(scene_graph)
    
    # Count the number of times each label occurs
    label_counts = {}
    for labels in labels_list:
        for a, b, c, d in labels:
            if c == 'normal':
                c = 'abnormal'
                d = 1 - d
            if d == 1:
                label = (a, b, c)
                label_counts[label] = label_counts.get(label, 0) + 1

    # Filter labels by frequency
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

    # Create a mapping from image_id to label indices
    imageId2labelIndices = {}
    for i, labels in enumerate(labels_list):
        image_id = scene_graphs[i]['image_id']
        label_indices = np.zeros(len(unique_labels), dtype=np.int8)
        for a, b, c, d in labels:
            if c == 'normal':
                c = 'abnormal'
                d = 1 - d
            # Localized label
            try:
                label_indices[label_to_index[(a, b, c)]] = d
            except KeyError:
                pass
            # Non-localized label
            try:
                idx = label_to_index[(b, c)]
                label_indices[idx] = max(label_indices[idx], d)
            except KeyError:
                pass
        imageId2labelIndices[image_id] = label_indices

    # Save labels
    labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'labels(min_freq={args.min_freq}).pkl')
    imageId2labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'imageId2labels(min_freq={args.min_freq}).pkl')
    print(f'Saving labels to {labels_path}')
    save_to_pickle(unique_labels, labels_path)
    print(f'Saving imageId2labels to {imageId2labels_path}')
    save_to_pickle(imageId2labelIndices, imageId2labels_path)
    print('Done')