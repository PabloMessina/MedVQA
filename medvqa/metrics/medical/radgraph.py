from dotenv import load_dotenv
load_dotenv()

from time import time
import os
import subprocess
from medvqa.utils.common import CACHE_DIR, SOURCE_DIR, TMP_DIR, get_timestamp
from medvqa.utils.files import get_cached_pickle_file, load_jsonl, save_pickle
from medvqa.utils.hashing import hash_string

RADGRAPH_MODEL_CHECKPOINT_PATH = os.environ['RADGRAPH_MODEL_CHECKPOINT_PATH']
DYGIE_PACKAGE_PARENT_FOLDER =  os.environ['DYGIE_PACKAGE_PARENT_FOLDER']

def compute_label_set(data, label2string):
    entities = data['entities']
    n = len(entities)
    e_strings = [None] * n
    hashes = set()
    for k, e in entities.items():
        i = int(k)-1
        e_strings[i] = f"{e['tokens']}|{e['label']}" # tokens|label
        h = hash_string(e_strings[i])
        hashes.add(h)
        if h in label2string:
            assert label2string[h] == e_strings[i]
        else:
            label2string[h] = e_strings[i]
    for k, e in entities.items():
        i = int(k)-1
        for r in e['relations']:
            j = int(r[1])-1
            rel_s1 = f"{e_strings[i]}|{r[0]}|{e_strings[j]}" # e1|rel|e2
            rel_s2 = f"{e_strings[i]}|{e_strings[j]}" # e1|e2
            h1 = hash_string(rel_s1)
            h2 = hash_string(rel_s2)
            hashes.add(h1)
            hashes.add(h2)
            if h1 in label2string:
                assert label2string[h1] == rel_s1
            else:
                label2string[h1] = rel_s1
            if h2 in label2string:
                assert label2string[h2] == rel_s2
            else:
                label2string[h2] = rel_s2
    return hashes

class RadGraphLabeler:

    def __init__(self, verbose=False):
        self.cache_path = os.path.join(CACHE_DIR, 'radgraph_labeler_cache.pkl')
        self.cache = get_cached_pickle_file(self.cache_path)
        self.verbose = verbose
        if self.cache is None:
            self.cache = dict(
                hash2labels=dict(),
                label2string=dict(),
            )
        elif verbose:
            print(f'Cache successfully loaded from {self.cache_path}')
        self.hash2labels = self.cache['hash2labels']
        self.label2string = self.cache['label2string']

    def get_labels(self, texts, update_cache_on_disk=False):

        if self.verbose:
            print(f'(*) RadGraph: labeling {len(texts)} texts ...')

        labels_list = [None] * len(texts)
        unlabeled_pairs = []
        unlabeled_hashes_set = set()
        unlabeled_hashes = []
        unlabeled_texts = []

        for i, text in enumerate(texts):
            hash = hash_string(text)
            labels_list[i] = self.hash2labels.get(hash, None)
            if labels_list[i] is None:
                unlabeled_pairs.append((i, hash))
                if hash not in unlabeled_hashes_set:
                    unlabeled_hashes_set.add(hash)
                    unlabeled_hashes.append(hash)
                    unlabeled_texts.append(text)

        if len(unlabeled_texts) > 0:
            if self.verbose:
                print(f'RadGraph: {len(unlabeled_texts)} texts not found in cache, invoking RadGraph labeler ...')
            
            # Create a temporary folder with one file per text
            timestamp = get_timestamp()
            temp_dir = os.path.join(TMP_DIR, 'radgraph', timestamp)
            os.makedirs(temp_dir, exist_ok=True)
            temp_files = []
            for i, text in enumerate(unlabeled_texts):
                temp_file = os.path.join(temp_dir, f'{i}.txt')
                with open(temp_file, 'w') as f:
                    f.write(text.lower()) # by lowercasing, we reduce noise in the labeling process
                temp_files.append(temp_file)
            
            # Invoke RadGraph labeler
            out_path = os.path.join(TMP_DIR, 'radgraph', 'results', f'{timestamp}.jsonl')
            temp_folder = os.path.join(TMP_DIR, 'radgraph', 'temp')
            command = (
                f'conda run -n dygiepp python3 {SOURCE_DIR}/medvqa/scripts/radgraph/radgraph_allennlp_inference.py '
                f'--model_path {RADGRAPH_MODEL_CHECKPOINT_PATH} '
                f'--dygie_package_parent_folder {DYGIE_PACKAGE_PARENT_FOLDER} '
                f'--data_path {temp_dir} '
                f'--out_path {out_path} '
                f'--temp_folder {temp_folder}'
            )
            time_before = time()
            ret = subprocess.call(command, shell=True)
            time_after = time()
            if self.verbose:
                print(f'RadGraph labeler took {time_after-time_before} seconds')
            if ret != 0:
                raise Exception('RadGraph labeler failed')
            assert os.path.exists(out_path), f'RadGraph labeler failed to generate output file {out_path}'

            # Read output file
            output = load_jsonl(out_path)
            assert len(output) == 1
            output = output[0]
            assert len(output) == len(unlabeled_texts)

            # Parse output file
            labels = [None] * len(unlabeled_texts)
            for input_path, data in output.items():
                filename = os.path.basename(input_path)
                idx = int(filename.split('.')[0])
                labels[idx] = compute_label_set(data, self.label2string)
            assert None not in labels

            # Update cache
            for hash, label in zip(unlabeled_hashes, labels):
                self.hash2labels[hash] = label
            if update_cache_on_disk:
                save_pickle(self.cache, self.cache_path)
                if self.verbose:
                    print(f'Cache successfully updated and saved to {self.cache_path}')
            
            # Update labels_list
            for i, hash in unlabeled_pairs:
                labels_list[i] = self.hash2labels[hash]

            # Delete temporary folder
            if self.verbose:
                print(f'RadGraph: deleting temporary folder {temp_dir} ...')
            subprocess.call(f'rm -rf {temp_dir}', shell=True)
            assert not os.path.exists(temp_dir)

            # Delete output file
            if self.verbose:
                print(f'RadGraph: deleting output file {out_path} ...')
            subprocess.call(f'rm {out_path}', shell=True)
            assert not os.path.exists(out_path)

        elif self.verbose:
            print('All labels found in cache, no need to invoke RadGraph labeler')
        
        assert None not in labels_list

        return labels_list