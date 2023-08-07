import os
import numpy as np
from tqdm import tqdm
from f1chexbert import F1CheXbert
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.files import get_cached_pickle_file, save_pickle
from medvqa.utils.hashing import hash_string

class CheXbertLabeler(F1CheXbert):

    def __init__(self, device=None, verbose=False):
        super().__init__(device=device)
        self.cache_path = os.path.join(CACHE_DIR, 'chexbert_labeler_cache.pkl')
        self.cache = get_cached_pickle_file(self.cache_path)
        self.verbose = verbose
        if self.cache is None:
            self.cache = dict()
        elif verbose:
            print(f'Cache successfully loaded from {self.cache_path}')

    def get_labels(self, texts, update_cache_on_disk=False):

        if self.verbose:
            print(f'(*) Chexbert: labeling {len(texts)} texts ...')

        labels_list = [None] * len(texts)
        unlabeled_pairs = []
        unlabeled_hashes_set = set()
        unlabeled_hashes = []
        unlabeled_texts = []

        for i, text in enumerate(texts):
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
                print(f'Chexbert: {len(unlabeled_texts)} texts not found in cache, invoking chexbert labeler ...')
                labels = [self.get_label(text) for text in tqdm(unlabeled_texts, mininterval=2)]
            else:
                labels = [self.get_label(text) for text in unlabeled_texts]
            for hash, label in zip(unlabeled_hashes, labels):
                self.cache[hash] = label
            
            if update_cache_on_disk:
                save_pickle(self.cache, self.cache_path)
                if self.verbose:
                    print(f'Cache successfully updated and saved to {self.cache_path}')
            
            for i, hash in unlabeled_pairs:
                labels_list[i] = self.cache[hash]
        elif self.verbose:
            print('All labels found in cache, no need to invoke chexbert labeler')
        
        return np.array(labels_list)