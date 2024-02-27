import os
import pandas as pd
import numpy as np
from medvqa.utils.files import save_pickle
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_CACHE_DIR, CHEST_IMAGENOME_GOLD_ATTRIBUTE_RELATIONS_TXT_PATH

if __name__ == '__main__':

    # Load sentences and labels from dataframe
    df = pd.read_csv(CHEST_IMAGENOME_GOLD_ATTRIBUTE_RELATIONS_TXT_PATH, sep='\t')
    sentence2anatomies = {}
    sentence2observations = {}
    unique_anatomies = set()
    unique_observations = set()
    for i, row in df.iterrows():
        s = row['sentence']
        s = ' '.join(s.split()) # remove extra spaces
        bbox = row['bbox']
        label_name = row['label_name']
        context = row['context']
        unique_anatomies.add(bbox)
        unique_observations.add(label_name)
        try:
            regions = sentence2anatomies[s]
            observations = sentence2observations[s]
        except KeyError:
            regions = sentence2anatomies[s] = set()
            observations = sentence2observations[s] = set()
        regions.add(bbox)
        observations.add((label_name, context))
    unique_anatomies = sorted(list(unique_anatomies))
    unique_observations = sorted(list(unique_observations))
    anatomy2idx = {a: i for i, a in enumerate(unique_anatomies)}
    observation2idx = {o: i for i, o in enumerate(unique_observations)}
    sentences = sorted(list(sentence2anatomies.keys()))
    print(f'Found {len(sentences)} unique sentences')
    print(f'Found {len(unique_anatomies)} unique anatomies')
    print(f'Found {len(unique_observations)} unique observations')

    # Generate numpy arrays
    print('Generating numpy arrays...')
    labels_a = np.zeros((len(sentences), len(unique_anatomies)), dtype=np.int16)
    labels_o = np.empty((len(sentences), len(unique_observations)), dtype=np.int16)
    labels_o.fill(-1) # prefill with -1 (unknown)
    for i, s in enumerate(sentences):
        for a in sentence2anatomies[s]:
            labels_a[i, anatomy2idx[a]] = 1
        for o, c in sentence2observations[s]:
            assert c in ['yes', 'no']
            labels_o[i, observation2idx[o]] = int(c == 'yes')
    print(f'labels_a.shape: {labels_a.shape}')
    print(f'labels_o.shape: {labels_o.shape}')
    assert np.all(labels_a >= 0) and np.all(labels_a <= 1) # 0/1
    assert np.all(labels_o >= -1) and np.all(labels_o <= 1) # -1/0/1
    
    # Save output
    print('Saving output...')
    output = {
        'phrases': sentences,
        'observation_labels': labels_o,
        'observation_names': unique_observations,
        'anatomy_labels': labels_a,
        'anatomy_names': unique_anatomies,
    }
    output_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'phrases2labels_gold(num_obs={len(unique_observations)},num_anat={len(unique_anatomies)},num_phrases={len(sentences)}).pkl')
    save_pickle(output, output_path)
    print(f'Saved to {output_path}')