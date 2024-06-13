# from time import time
import os
# import subprocess
from medvqa.datasets.text_data_utils import split_text_into_chunks
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.files import get_cached_pickle_file, save_pickle
from medvqa.utils.hashing import hash_string
# from medvqa.datasets.radgraph import RADGRAPH_MODEL_CHECKPOINT_PATH, DYGIE_PACKAGE_PARENT_FOLDER
from radgraph import RadGraph
from radgraph.rewards import (
    exact_entity_token_match_reward,
    exact_entity_token_if_rel_exists_reward,
    exact_entity_token_if_all_match_reward,
)

def compute_label_dict(data, label2string=None):
    entities = data['entities']
    n = len(entities)
    e_strings = [None] * n
    hash2count = dict()
    for k, e in entities.items():
        i = int(k)-1
        e_strings[i] = f"{e['tokens']}|{e['label']}" # tokens|label
        e_strings[i] = e_strings[i].lower() # by lowercasing, we reduce noise in the labeling process
        h = hash_string(e_strings[i])
        hash2count[h] = hash2count.get(h, 0) + 1
        if label2string is not None:
            if h in label2string:
                assert label2string[h] == e_strings[i]
            else:
                label2string[h] = e_strings[i]
    for k, e in entities.items():
        i = int(k)-1
        if e['relations']: # if there are relations
            partial_rel = f"{e_strings[i]}|r" # e1|r -> denotes that e1 has relations
            h = hash_string(partial_rel)
            hash2count[h] = hash2count.get(h, 0) + 1
            if label2string is not None:
                if h in label2string:
                    assert label2string[h] == partial_rel
                else:
                    label2string[h] = partial_rel
        for r in e['relations']:
            j = int(r[1])-1
            rel_s1 = f"{e_strings[i]}|{r[0]}|{e_strings[j]}" # e1|rel|e2
            rel_s1 = rel_s1.lower() # by lowercasing, we reduce noise in the labeling process
            rel_s2 = f"{e_strings[i]}|{e_strings[j]}" # e1|e2
            rel_s2 = rel_s2.lower() # by lowercasing, we reduce noise in the labeling process
            h1 = hash_string(rel_s1)
            h2 = hash_string(rel_s2)
            hash2count[h1] = hash2count.get(h1, 0) + 1
            hash2count[h2] = hash2count.get(h2, 0) + 1
            if label2string is not None:
                if h1 in label2string:
                    assert label2string[h1] == rel_s1
                else:
                    label2string[h1] = rel_s1
                if h2 in label2string:
                    assert label2string[h2] == rel_s2
                else:
                    label2string[h2] = rel_s2
    return hash2count

class RadGraphLabeler:

    def __init__(self, verbose=False):
        self.cache_path = os.path.join(CACHE_DIR, 'radgraph_labeler_cache_.pkl')
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

    def __call__(self, texts, update_cache_on_disk=False):
        return self.get_labels(texts, update_cache_on_disk)

    def get_labels(self, texts, update_cache_on_disk=False):

        assert type(texts) == list or type(texts) == str
        if type(texts) == str:
            texts = [texts]

        if self.verbose:
            print(f'(*) RadGraph: labeling {len(texts)} texts ...')

        labels_list = [None] * len(texts)
        unlabeled_pairs = []
        unlabeled_hashes_set = set()
        unlabeled_hashes = []
        unlabeled_texts = []
        unlabeled_ranges = []

        offset = 0
        for i, text in enumerate(texts):
            if len(text) == 0: # empty text
                labels_list[i] = dict()
                continue
            hash = hash_string(text)
            labels_list[i] = self.hash2labels.get(hash, None)
            if labels_list[i] is None:
                unlabeled_pairs.append((i, hash))
                if hash not in unlabeled_hashes_set:
                    unlabeled_hashes_set.add(hash)
                    unlabeled_hashes.append(hash)
                    if len(text) > 1000:
                        chunks = split_text_into_chunks(text, max_length=1000)
                        unlabeled_texts.extend(chunks)
                        unlabeled_ranges.append((offset, offset+len(chunks)))
                        offset += len(chunks)
                    else:
                        unlabeled_texts.append(text)
                        unlabeled_ranges.append((offset, offset+1))
                        offset += 1

        if len(unlabeled_texts) > 0:
            if self.verbose:
                print(f'RadGraph: {len(unlabeled_texts)} texts not found in cache, invoking RadGraph labeler ...')
            
            # # Create a temporary folder with one file per text
            # timestamp = get_timestamp()
            # temp_dir = os.path.join(TMP_DIR, 'radgraph', timestamp)
            # os.makedirs(temp_dir, exist_ok=True)
            # temp_files = []
            # for i, text in enumerate(unlabeled_texts):
            #     temp_file = os.path.join(temp_dir, f'{i}.txt')
            #     with open(temp_file, 'w') as f:
            #         f.write(text.lower()) # by lowercasing, we reduce noise in the labeling process
            #     temp_files.append(temp_file)
            
            # # Invoke RadGraph labeler
            # out_path = os.path.join(TMP_DIR, 'radgraph', 'results', f'{timestamp}.jsonl')
            # temp_folder = os.path.join(TMP_DIR, 'radgraph', 'temp')
            # command = (
            #     f'conda run -n dygiepp python3 {SOURCE_DIR}/medvqa/scripts/radgraph/radgraph_allennlp_inference.py '
            #     f'--model_path {RADGRAPH_MODEL_CHECKPOINT_PATH} '
            #     f'--dygie_package_parent_folder {DYGIE_PACKAGE_PARENT_FOLDER} '
            #     f'--data_path {temp_dir} '
            #     f'--out_path {out_path} '
            #     f'--temp_folder {temp_folder} '
            #     f'--cuda_device {cuda_device}'
            # )
            # if self.verbose:
            #     print(f'RadGraph: invoking command {command} ...')
            # time_before = time()
            # ret = subprocess.call(command, shell=True)
            # time_after = time()
            # if self.verbose:
            #     print(f'RadGraph labeler took {time_after-time_before} seconds')
            # if ret != 0:
            #     raise Exception('RadGraph labeler failed')
            # assert os.path.exists(out_path), f'RadGraph labeler failed to generate output file {out_path}'

            # # Read output file
            # output = load_json(out_path)
            # assert len(output) == len(unlabeled_texts)

            # # Parse output file
            # labels = [None] * len(unlabeled_texts)
            # for input_path, data in output.items():
            #     filename = os.path.basename(input_path)
            #     idx = int(filename.split('.')[0])
            #     labels[idx] = compute_label_dict(data, self.label2string)
            # assert None not in labels
                
            # Create instance of RadGraph
            radgraph = RadGraph()
    
            # Compute RadGraph annotations
            annotations = radgraph(unlabeled_texts)

            # Parse output file
            labels = [None] * len(unlabeled_texts)
            for i in range(len(unlabeled_texts)):
                labels[i] = compute_label_dict(annotations[str(i)], self.label2string)

            # Update cache
            for hash, (s, e) in zip(unlabeled_hashes, unlabeled_ranges):
                assert s < e
                if s+1 == e:
                    label = labels[s]
                else:
                    label = dict()
                    for i in range(s, e):
                        for k, v in labels[i].items():
                            label[k] = label.get(k, 0) + v # sum counts
                self.hash2labels[hash] = label
            
            if update_cache_on_disk:
                save_pickle(self.cache, self.cache_path)
                if self.verbose:
                    print(f'Cache successfully updated and saved to {self.cache_path}')
            
            # Update labels_list
            for i, hash in unlabeled_pairs:
                labels_list[i] = self.hash2labels[hash]

            # # Delete temporary folder
            # if self.verbose:
            #     print(f'RadGraph: deleting temporary folder {temp_dir} ...')
            # subprocess.call(f'rm -rf {temp_dir}', shell=True)
            # assert not os.path.exists(temp_dir)

            # # Delete output file
            # if self.verbose:
            #     print(f'RadGraph: deleting output file {out_path} ...')
            # subprocess.call(f'rm {out_path}', shell=True)
            # assert not os.path.exists(out_path)

        elif self.verbose:
            print('All labels found in cache, no need to invoke RadGraph labeler')
        
        assert None not in labels_list

        return labels_list
    
class RadGraphLabelerOriginal:
    def __init__(self, verbose=False):
        self.cache_path = os.path.join(CACHE_DIR, 'radgraph_labeler_cache_original.pkl')
        self.cache = get_cached_pickle_file(self.cache_path)
        self.verbose = verbose
        if self.cache is None:
            self.cache = dict()
        elif verbose:
            print(f'Cache successfully loaded from {self.cache_path}')

    def __call__(self, texts, update_cache_on_disk=False):
        return self.get_labels(texts, update_cache_on_disk)

    def get_labels(self, texts, update_cache_on_disk=False):

        assert type(texts) == list or type(texts) == str
        if type(texts) == str:
            texts = [texts]

        if self.verbose:
            print(f'(*) RadGraph (Original): labeling {len(texts)} texts ...')

        labels_list = [None] * len(texts)
        unlabeled_pairs = []
        unlabeled_hashes_set = set()
        unlabeled_hashes = []
        unlabeled_texts = []
        
        for i, text in enumerate(texts):
            if len(text) == 0: # empty text
                labels_list[i] = dict()
                continue
            hash = hash_string(text)
            labels_list[i] = self.cache.get(hash, None)
            if labels_list[i] is None:
                unlabeled_pairs.append((i, hash))
                if hash not in unlabeled_hashes_set:
                    unlabeled_hashes_set.add(hash)
                    unlabeled_hashes.append(hash)
                    unlabeled_texts.append(text)

        if len(unlabeled_texts) > 0:
            if self.verbose:
                print(f'RadGraph (Original): {len(unlabeled_texts)} texts not found in cache, invoking RadGraph labeler ...')
                
            # Create instance of RadGraph
            radgraph = RadGraph()
    
            # Compute RadGraph annotations
            annotations = radgraph(unlabeled_texts)

            # Update cache
            for i, hash in enumerate(unlabeled_hashes):
                self.cache[hash] = annotations[str(i)]
            
            if update_cache_on_disk:
                save_pickle(self.cache, self.cache_path)
                if self.verbose:
                    print(f'Cache successfully updated and saved to {self.cache_path}')
            
            # Update labels_list
            for i, hash in unlabeled_pairs:
                labels_list[i] = self.cache[hash]

        elif self.verbose:
            print('All labels found in cache, no need to invoke RadGraph labeler')
        
        assert None not in labels_list

        return labels_list
    
def compute_reward(hypothetical_labels, true_labels, reward_level):
    assert type(hypothetical_labels) == dict
    assert type(true_labels) == dict
    if (
        len(hypothetical_labels) == 0 or
        len(hypothetical_labels["entities"].keys()) == 0 or
        len(true_labels) == 0 or
        len(true_labels["entities"].keys()) == 0
    ):
        return (0., 0., 0.) if reward_level == "all" else 0.
    if reward_level == "all":
        simple = exact_entity_token_match_reward(hypothetical_labels, true_labels)
        partial = exact_entity_token_if_rel_exists_reward(hypothetical_labels, true_labels)
        complete = exact_entity_token_if_all_match_reward(hypothetical_labels, true_labels)
        all = (simple, partial, complete)
        return all
    if reward_level == "simple":
        return exact_entity_token_match_reward(hypothetical_labels, true_labels)
    if reward_level == "partial":
        return exact_entity_token_if_rel_exists_reward(hypothetical_labels, true_labels)
    if reward_level == "complete":
        return exact_entity_token_if_all_match_reward(hypothetical_labels, true_labels)
    raise ValueError(f"Invalid reward level: {reward_level}")