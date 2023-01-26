from multiprocessing import Pool
import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from medvqa.utils.files import load_json_file, save_to_pickle
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR, CHEST_IMAGENOME_CACHE_DIR
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    extract_labels_from_scene_graph,
)

def _load_scene_graph(scene_graph_path):
    return load_json_file(scene_graph_path)

# Load scene graphs in parallel using multiprocessing
def load_scene_graphs(num_workers=4, first_k=None):
    # measure elapsed time
    start_time = time.time()
    filenames = os.listdir(CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR)
    if first_k is not None:
        filenames = filenames[:first_k]
    filepaths = [os.path.join(CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR, f) for f in filenames]
    with Pool(num_workers) as p:
        scene_graphs = p.map(_load_scene_graph, filepaths)
    elapsed_time = time.time() - start_time
    print(f'Elapsed time: {elapsed_time:.2f} seconds')
    return scene_graphs

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    # Load scene graphs
    print(f'Loading scene graphs from {CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR} with {args.num_workers} workers')
    scene_graphs = load_scene_graphs(num_workers=args.num_workers)
    print(f'Loaded {len(scene_graphs)} scene graphs')

    # Extract labels
    labels_list = [None] * len(scene_graphs)
    for i, scene_graph in enumerate(tqdm(scene_graphs)):
        labels_list[i] = extract_labels_from_scene_graph(scene_graph)
    
    # Create a unique set of labels
    labels_set = set()
    for labels in labels_list:
        labels_set.update((a, b, c) if c != 'normal' else (a, b, 'abnormal') for a, b, c, _ in labels)
    unique_labels = sorted(list(labels_set))
    print(f'Found {len(unique_labels)} unique labels')

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
            label_indices[label_to_index[(a, b, c)]] = d
        imageId2labelIndices[image_id] = label_indices

    # Save labels
    labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'labels.pkl')
    imageId2labels_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'imageId2labels.pkl')
    print(f'Saving labels to {labels_path}')
    save_to_pickle(unique_labels, labels_path)
    print(f'Saving imageId2labels to {imageId2labels_path}')
    save_to_pickle(imageId2labelIndices, imageId2labels_path)