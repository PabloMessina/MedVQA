import os
import pandas as pd
import numpy as np
from collections import OrderedDict
from medvqa.utils.files import save_pickle
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_CACHE_DIR, CHEST_IMAGENOME_GOLD_ATTRIBUTE_RELATIONS_TXT_PATH

if __name__ == '__main__':

    # Load sentences and labels from dataframe
    df = pd.read_csv(CHEST_IMAGENOME_GOLD_ATTRIBUTE_RELATIONS_TXT_PATH, sep='\t')
    sentence2anatomies = {}
    sentence2observations = {}
    sid2sentences = {}
    unique_anatomies = set()
    unique_observations = set()
    for i, row in df.iterrows():
        s = row['sentence']
        s = ' '.join(s.split()) # remove extra spaces
        bbox = row['bbox']
        label_name = row['label_name']
        context = row['context']
        sid = row['study_id']
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
        try:
            sentences = sid2sentences[sid]
        except KeyError:
            sentences = sid2sentences[sid] = OrderedDict() # preserve order without duplicates
        sentences[s] = None
    unique_anatomies = sorted(list(unique_anatomies))
    unique_observations = sorted(list(unique_observations))
    anatomy2idx = {a: i for i, a in enumerate(unique_anatomies)}
    observation2idx = {o: i for i, o in enumerate(unique_observations)}
    sentences = sorted(list(sentence2anatomies.keys()))
    sentence2index = {s: i for i, s in enumerate(sentences)}
    sid2sentences = {sid: list(sentences) for sid, sentences in sid2sentences.items()}
    print(f'Found {len(sentences)} unique sentences')
    print(f'Found {len(unique_anatomies)} unique anatomies')
    print(f'Found {len(unique_observations)} unique observations')

    # Generate sentence-level numpy arrays
    print('Generating numpy arrays...')
    s_labels_a = np.zeros((len(sentences), len(unique_anatomies)), dtype=np.int16)
    s_labels_o = np.empty((len(sentences), len(unique_observations)), dtype=np.int16)
    s_labels_o.fill(-1) # prefill with -1 (unknown)
    for i, s in enumerate(sentences):
        for a in sentence2anatomies[s]:
            s_labels_a[i, anatomy2idx[a]] = 1
        for o, c in sentence2observations[s]:
            assert c in ['yes', 'no']
            s_labels_o[i, observation2idx[o]] = int(c == 'yes')
    print(f's_labels_a.shape: {s_labels_a.shape}')
    print(f's_labels_o.shape: {s_labels_o.shape}')
    assert np.all(s_labels_a >= 0) and np.all(s_labels_a <= 1) # 0/1
    assert np.all(s_labels_o >= -1) and np.all(s_labels_o <= 1) # -1/0/1

    # Generate report-level numpy arrays
    print('Generating report-level numpy arrays...')
    r_labels_a = np.zeros((len(sid2sentences), len(unique_anatomies)), dtype=np.int16)
    r_labels_o = np.empty((len(sid2sentences), len(unique_observations)), dtype=np.int16)
    r_labels_o.fill(-1) # prefill with -1 (unknown)
    reports = [None] * len(sid2sentences)
    for i, (sid, sentences) in enumerate(sid2sentences.items()):
        r_labels_a[i] = s_labels_a[[sentence2index[s] for s in sentences]].max(axis=0)
        r_labels_o[i] = s_labels_o[[sentence2index[s] for s in sentences]].max(axis=0)
        report = ''
        for s in sentences:
            if report:
                if report[-1] != '.':
                    report += '. '
                else:
                    report += ' '
            report += s
        reports[i] = report
    
    # Save output
    print('Saving output...')
    output = {
        'reports': reports,
        'observation_labels': r_labels_o,
        'observation_names': unique_observations,
        'anatomy_labels': r_labels_a,
        'anatomy_names': unique_anatomies,
    }
    output_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, f'reports2labels_gold(num_obs={len(unique_observations)},num_anat={len(unique_anatomies)},num_reports={len(reports)}).pkl')
    save_pickle(output, output_path)
    print(f'Saved to {output_path}')