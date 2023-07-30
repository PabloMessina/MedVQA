import argparse
import random
import numpy as np
import os
from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.utils.files import load_jsonl, load_pickle, save_pickle
from medvqa.utils.logging import print_orange

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--chest_imagenome_phrases2labels_filepath', type=str, required=True)
    parser.add_argument('--extracted_chest_imagenome_labels_with_t5_filepath', type=str, required=True)
    parser.add_argument('--extracted_chest_imagenome_labels_with_openai_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--extraction_methods', type=str, nargs='+', required=True)
    args = parser.parse_args()
    assert len(args.extracted_chest_imagenome_labels_with_openai_filepaths) + 2 == len(args.extraction_methods) # 2 for t5 and phrases2labels

    output = {}

    # Load chest_imagenome_phrases2labels
    chest_imagenome_phrases2labels = load_pickle(args.chest_imagenome_phrases2labels_filepath)
    print(f'Loaded chest_imagenome_phrases2labels from {args.chest_imagenome_phrases2labels_filepath}.')
    phrases = chest_imagenome_phrases2labels['phrases']
    labels = chest_imagenome_phrases2labels['labels']
    print(f'labels.shape: {labels.shape}')
    label_names = chest_imagenome_phrases2labels['label_names']
    # Remove nlp labels and prefixes
    _idxs = [i for i, x in enumerate(label_names) if not x.startswith('nlp|')]
    assert len(_idxs) + 2 == len(label_names) # 2 nlp labels
    labels = labels[:, _idxs]
    labels = (labels == 1).astype(np.int8)
    label_names = [label_names[i][label_names[i].index('|') + 1:] for i in _idxs]
    label_name_to_idx = {label_name: i for i, label_name in enumerate(label_names)}

    output['label_names'] = label_names
    output['groups'] = []
    output['groups'].append({
        'sentences': phrases,
        'labels': labels.astype(np.int8),
        'extraction_method': args.extraction_methods[0],
    })

    # Load extracted_chest_imagenome_labels_with_t5
    extracted_chest_imagenome_labels_with_t5 = load_jsonl(args.extracted_chest_imagenome_labels_with_t5_filepath)
    print(f'Loaded extracted_chest_imagenome_labels_with_t5 from {args.extracted_chest_imagenome_labels_with_t5_filepath}.')
    labels = np.zeros((len(extracted_chest_imagenome_labels_with_t5), len(label_names)), dtype=np.int8)
    print(f'labels.shape: {labels.shape}')
    unknown_labels = []
    sentences = []
    for i, row in enumerate(extracted_chest_imagenome_labels_with_t5):
        sentences.append(row['sentence'])
        for label_name in row['labels']:
            try:
                labels[i, label_name_to_idx[label_name]] = 1
            except KeyError:
                unknown_labels.append(label_name)
    if len(unknown_labels) > 0:
        print_orange(f'WARNING: {len(unknown_labels)} unknown labels in extracted_chest_imagenome_labels_with_t5.', bold=True)
        print_orange('Examples:', bold=True)
        for label_name in random.sample(unknown_labels, min(5, len(unknown_labels))):
            print_orange(f'  {label_name}', bold=True)

    output['groups'].append({
        'sentences': sentences,
        'labels': labels,
        'extraction_method': args.extraction_methods[1],
    })

    # Load extracted_chest_imagenome_labels_with_openai
    for i, filepath in enumerate(args.extracted_chest_imagenome_labels_with_openai_filepaths):
        rows = load_jsonl(filepath)
        print(f'Loaded extracted_chest_imagenome_labels_with_openai from {filepath}.')
        labels = np.zeros((len(rows), len(label_names)), dtype=np.int8)
        print(f'labels.shape: {labels.shape}')
        sentences = []
        unknown_labels = []
        for j, row in enumerate(rows):
            sentences.append(row['metadata']['query'])
            for label_name in row['parsed_response']:
                try:
                    labels[j, label_name_to_idx[label_name]] = 1
                except KeyError:
                    unknown_labels.append(label_name)
        if len(unknown_labels) > 0:
            print_orange(f'WARNING: {len(unknown_labels)} unknown labels found at {filepath}.', bold=True)
            print_orange('Examples:', bold=True)
            for label_name in random.sample(unknown_labels, min(5, len(unknown_labels))):
                print_orange(f'  {label_name}', bold=True)

        output['groups'].append({
            'sentences': sentences,
            'labels': labels,
            'extraction_method': args.extraction_methods[i + 2],
        })

    # Save integrated chest_imagenome labels
    integrated_chest_imagenome_labels_savepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, f'integrated_chest_imagenome_labels({len(output["groups"])}).pkl')
    print(f'Saving integrated chest_imagenome labels to {integrated_chest_imagenome_labels_savepath}...')
    save_pickle(output, integrated_chest_imagenome_labels_savepath)
    print('Done!')