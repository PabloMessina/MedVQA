import argparse
import os
import numpy as np
from tqdm import tqdm
from medvqa.utils.files_utils import save_pickle
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_CACHE_DIR
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_scene_graphs_in_parallel,
)
from medvqa.utils.logging_utils import print_orange

def _compute_phrases_to_labels(scene_graph, categories_to_skip):
    phrases2observations = {}
    phrases2anatomies = {}
    for node in scene_graph['attributes']:
        assert len(node['attributes']) == len(node['phrases'])
        anatomy = node['bbox_name']
        for a, p in zip(node['attributes'], node['phrases']):
            p = ' '.join(p.split()) # remove extra spaces
            if p not in phrases2observations:
                phrases2observations[p] = {}
                phrases2anatomies[p] = set()
            phrases2anatomies[p].add(anatomy)
            for x in a:
                category, value, name = x.split('|')
                if category in categories_to_skip:
                    continue
                value = int(value == 'yes') # convert to 0/1
                phrases2observations[p][f'{category}|{name}'] = value
    return phrases2observations, phrases2anatomies

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
    print('Computing phrases to observations and anatomies...')
    phrases2observations = {}
    phrases2anatomies = {}
    for scene_graph in tqdm(scene_graphs, mininterval=2):
        p2o, p2a =_compute_phrases_to_labels(scene_graph, categories_to_skip=args.categories_to_skip)
        for p, o in p2o.items():
            if p in phrases2observations:
                if phrases2observations[p] != o:
                    print_orange(f'Warning: "{p}" has different observations than before (old: {phrases2observations[p]}, new: {o})')
            else:
                phrases2observations[p] = o
        for p, a in p2a.items():
            if p in phrases2anatomies:
                if phrases2anatomies[p] != a:
                    print_orange(f'Warning: "{p}" has different anatomies than before (old: {phrases2anatomies[p]}, new: {a})')
            else:
                phrases2anatomies[p] = a

    phrases_o = sorted(list(phrases2observations.keys()))
    phrases_a = sorted(list(phrases2anatomies.keys()))
    assert phrases_o == phrases_a
    print(f'Found {len(phrases_o)} unique phrases')

    # Collect unique observations
    print('Collecting unique observations...')
    unique_observations = set()
    for o in phrases2observations.values():
        unique_observations.update(o.keys())
    unique_observations = sorted(list(unique_observations))
    print(f'Found {len(unique_observations)} unique observations')
    o2idx = {o: i for i, o in enumerate(unique_observations)}

    # Collect unique anatomies
    print('Collecting unique anatomies...')
    unique_anatomies = set()
    for a in phrases2anatomies.values():
        unique_anatomies.update(a)
    unique_anatomies = sorted(list(unique_anatomies))
    print(f'Found {len(unique_anatomies)} unique anatomies')
    a2idx = {a: i for i, a in enumerate(unique_anatomies)}
    
    # Compute observation labels numpy array
    print('Computing observation labels numpy array...')
    labels_o = np.zeros((len(phrases_o), len(unique_observations)), dtype=np.int64)
    labels_o.fill(2) # prefill with 2 (unknown)
    for i, p in enumerate(phrases_o):
        for o, v in phrases2observations[p].items():
            labels_o[i, o2idx[o]] = v
    print(f'labels_o.shape: {labels_o.shape}')
    assert np.all(labels_o >= 0) and np.all(labels_o <= 2) # 0/1/2

    # Compute anatomy labels numpy array
    print('Computing anatomy labels numpy array...')
    labels_a = np.zeros((len(phrases_o), len(unique_anatomies)), dtype=np.int64)
    for i, p in enumerate(phrases_a):
        for a in phrases2anatomies[p]:
            labels_a[i, a2idx[a]] = 1
    print(f'labels_a.shape: {labels_a.shape}')
    assert np.all(labels_a >= 0) and np.all(labels_a <= 1) # 0/1

    # Filter observation labels with low frequency
    print('Filtering observation labels with low frequency...')
    label_freq = (labels_o != 2).sum(axis=0)
    label_idxs_to_keep = np.where(label_freq >= args.min_label_freq)[0]
    labels_o = labels_o[:, label_idxs_to_keep]
    unique_observations = [unique_observations[i] for i in label_idxs_to_keep]
    print(f'labels_o.shape: {labels_o.shape}')
    
    # Save output
    print('Saving output...')
    output = {
        'phrases': phrases_o,
        'observation_labels': labels_o,
        'observation_names': unique_observations,
        'anatomy_labels': labels_a,
        'anatomy_names': unique_anatomies,
    }
    output_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'phrases2labels_silver(num_obs={len(unique_observations)},num_anat={len(unique_anatomies)},num_phrases={len(phrases_o)}).pkl')
    save_pickle(output, output_path)
    print(f'Saved to {output_path}')