import argparse
import os
import numpy as np
from tqdm import tqdm
from medvqa.utils.files import save_pickle
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_CACHE_DIR
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_scene_graphs_in_parallel,
)
from medvqa.utils.logging import print_orange

def _compute_phrases_to_labels(scene_graph, categories_to_skip):
    phrases2labels = {}
    for node in scene_graph['attributes']:
        assert len(node['attributes']) == len(node['phrases'])
        for a, p in zip(node['attributes'], node['phrases']):
            p = ' '.join(p.split()) # remove extra spaces
            if p not in phrases2labels:
                phrases2labels[p] = {}
            for x in a:
                category, value, name = x.split('|')
                if category in categories_to_skip:
                    continue
                value = int(value == 'yes') # convert to 0/1
                phrases2labels[p][f'{category}|{name}'] = value
    return phrases2labels

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--min_label_freq', type=int, default=50)
    parser.add_argument('--categories_to_skip', nargs='+', default=['temporal', 'severity', 'laterality'])
    args = parser.parse_args()

    # Load scene graphs
    print('Loading scene graphs...')
    scene_graphs = load_scene_graphs_in_parallel(num_workers=args.num_workers)
    print(f'Loaded {len(scene_graphs)} scene graphs')

    # Compute phrases to labels
    print('Computing phrases to labels...')
    phrases2labels = {}
    for scene_graph in tqdm(scene_graphs, mininterval=2):
        p2l =_compute_phrases_to_labels(scene_graph, categories_to_skip=args.categories_to_skip)
        for p, l in p2l.items():
            if p in phrases2labels:
                if phrases2labels[p] != l:
                    print_orange(f'Warning: "{p}" has different labels than before (old: {phrases2labels[p]}, new: {l})')
            else:
                phrases2labels[p] = l
    phrases = sorted(list(phrases2labels.keys()))
    print(f'Found {len(phrases)} phrases with labels')

    # Collect unique labels
    print('Collecting unique labels...')
    unique_labels = set()
    for labels in phrases2labels.values():
        unique_labels.update(labels.keys())
    unique_labels = sorted(list(unique_labels))
    print(f'Found {len(unique_labels)} unique labels')
    label2idx = {l: i for i, l in enumerate(unique_labels)}
    
    # Compute labels numpy array
    print('Computing labels numpy array...')
    labels = np.zeros((len(phrases), len(unique_labels)), dtype=np.int64)
    labels.fill(2) # prefill with 2 (unknown)
    for i, p in enumerate(phrases):
        for l, v in phrases2labels[p].items():
            labels[i, label2idx[l]] = v
    print(f'Labels shape: {labels.shape}')
    assert np.all(labels >= 0) and np.all(labels <= 2) # 0/1/2

    # Filter labels with low frequency
    print('Filtering labels with low frequency...')
    label_freq = (labels != 2).sum(axis=0)
    label_idxs_to_keep = np.where(label_freq >= args.min_label_freq)[0]
    labels = labels[:, label_idxs_to_keep]
    unique_labels = [unique_labels[i] for i in label_idxs_to_keep]
    print(f'Labels shape: {labels.shape}')
    
    # Save output
    print('Saving output...')
    output = {
        'phrases': phrases,
        'labels': labels,
        'label_names': unique_labels,
    }
    output_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'phrases2labels(num_labels={len(unique_labels)},num_phrases={len(phrases)}).pkl')
    save_pickle(output, output_path)
    print(f'Saved to {output_path}')