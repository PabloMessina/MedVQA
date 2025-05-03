import os
from tqdm import tqdm
from medvqa.utils.common import LARGE_FAST_CACHE_DIR
from medvqa.utils.files_utils import load_json, save_pickle

from medvqa.datasets.radgraph import (
    RADGRAPH_TRAIN_GRAPH_JSON_PATH,
    RADGRAPH_DEV_GRAPH_JSON_PATH,
    RADGRAPH_TEST_GRAPH_JSON_PATH,
    RADGRAPH_MIMICCXR_GRAPHS_JSON_PATH,
    RADGRAPH_CHEXPERT_GRAPHS_JSON_PATH,
    compute_label_set_per_sentence,
)

if __name__ == '__main__':

    print('Loading graphs...')
    train_graphs = load_json(RADGRAPH_TRAIN_GRAPH_JSON_PATH)
    print(f'Loaded train graphs from {RADGRAPH_TRAIN_GRAPH_JSON_PATH}')
    dev_graphs = load_json(RADGRAPH_DEV_GRAPH_JSON_PATH)
    print(f'Loaded dev graphs from {RADGRAPH_DEV_GRAPH_JSON_PATH}')
    test_graphs = load_json(RADGRAPH_TEST_GRAPH_JSON_PATH)
    print(f'Loaded test graphs from {RADGRAPH_TEST_GRAPH_JSON_PATH}')
    mimiccxr_graphs = load_json(RADGRAPH_MIMICCXR_GRAPHS_JSON_PATH)
    print(f'Loaded mimiccxr graphs from {RADGRAPH_MIMICCXR_GRAPHS_JSON_PATH}')
    chexpert_graphs = load_json(RADGRAPH_CHEXPERT_GRAPHS_JSON_PATH)
    print(f'Loaded chexpert graphs from {RADGRAPH_CHEXPERT_GRAPHS_JSON_PATH}')
    print(f'len(train_graphs) = {len(train_graphs)}')
    print(f'len(dev_graphs) = {len(dev_graphs)}')
    print(f'len(test_graphs) = {len(test_graphs)}')
    print(f'len(mimiccxr_graphs) = {len(mimiccxr_graphs)}')
    print(f'len(chexpert_graphs) = {len(chexpert_graphs)}')

    hash2string = {}
    output = {
        'hash2string': hash2string,
        'train': {},
        'dev': {},
        'test': {},
        'mimiccxr': {},
        'chexpert': {},
    }
    for graphs, category in zip([
        train_graphs,
        dev_graphs,
        test_graphs,
        mimiccxr_graphs,
        chexpert_graphs,
    ], [
        'train',
        'dev',
        'test',
        'mimiccxr',
        'chexpert',
    ]):
        print(f'Computing label sets for {category}...')
        category_dict = output[category]
        for graph in tqdm(graphs.values(), mininterval=2):
            for sentence, label_set in compute_label_set_per_sentence(graph, hash2string):
                if sentence in category_dict:
                    category_dict[sentence] |= label_set # Union
                else:
                    category_dict[sentence] = label_set
        print(f'len(hash2string) = {len(hash2string)}')
        print(f'len(output[{category}]) = {len(output[category])}')

    save_path = os.path.join(LARGE_FAST_CACHE_DIR, 'radgraph', 'label_sets_per_sentence.pkl')
    print(f'Saving label sets per sentence to {save_path}...')
    save_pickle(output, save_path)
    print('Done!')