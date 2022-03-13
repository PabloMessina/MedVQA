from dotenv import load_dotenv
load_dotenv()

from medvqa.utils.files import load_pickle, save_to_pickle
from medvqa.utils.common import CACHE_DIR, TMP_DIR
from medvqa.utils.constants import CHEXPERT_LABELS
from medvqa.utils.hashing import hash_string
import os
import subprocess
import pandas as pd
import csv
import numpy as np
import time

CHEXPERT_FOLDER = os.environ['CHEXPERT_FOLDER']
CHEXPERT_PYTHON = os.environ['CHEXPERT_PYTHON']
NEGBIO_PATH = os.environ['NEGBIO_PATH']
TMP_FOLDER = os.path.join(TMP_DIR, 'chexpert-labeler')

def _get_custom_env():
    custom_env = os.environ.copy()
    prev = custom_env.get('PYTHONPATH', '')
    custom_env['PYTHONPATH'] = f'{NEGBIO_PATH}:{prev}'
    return custom_env

class ChexpertLabeler:
    def __init__(self, verbose=True):        
        self.cache_path = os.path.join(CACHE_DIR, 'chexpert_labeler_cache.pkl')
        self.cache = load_pickle(self.cache_path)
        self.verbose = verbose
        if self.cache is None:
            self.cache = dict()
        elif verbose:
            print(f'Cache successfully loaded from {self.cache_path}')

    def get_labels(self, texts, fill_empty=0, fill_uncertain=1,
                   tmp_suffix='', update_cache_on_disk=False):

        if self.verbose:
            print(f'(*) Chexpert: labeling {len(texts)} texts ...')

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
            labels = self._invoke_chexpert_labeler_process(unlabeled_texts, tmp_suffix)
            for hash, label in zip(unlabeled_hashes, labels):
                self.cache[hash] = label
            
            if update_cache_on_disk:
                save_to_pickle(self.cache, self.cache_path)
                if self.verbose:
                    print(f'Cache successfully updated and saved to {self.cache_path}')
            
            for i, hash in unlabeled_pairs:
                labels_list[i] = self.cache[hash]
        elif self.verbose:
            print('All labels found in cache, no need to invoke chexpert labeler')
        
        out = np.array(labels_list)
        out = np.where(out == -2, fill_empty, out)
        out = np.where(out == -1, fill_uncertain, out)
        return out

    def _invoke_chexpert_labeler_process(self, texts, tmp_suffix=''):

        # Define input & output paths
        input_path = os.path.join(TMP_FOLDER, f'labeler-input{tmp_suffix}.csv')
        output_path = os.path.join(TMP_FOLDER, f'labeler-output{tmp_suffix}.csv')

        # Create input file
        os.makedirs(TMP_FOLDER, exist_ok=True)
        in_df = pd.DataFrame(texts)
        in_df.to_csv(input_path, header=False, index=False, quoting=csv.QUOTE_ALL)

        # Build command & call chexpert labeler process
        cmd_cd = f'cd {CHEXPERT_FOLDER}'
        cmd_call = f'{CHEXPERT_PYTHON} label.py --reports_path {input_path} --output_path {output_path}'
        cmd = f'{cmd_cd} && {cmd_call}'        
        try:            
            if self.verbose:
                print(f'Running chexpert labeler over {len(in_df)} texts ...')
                print(f'\tCommand = {cmd}')
                start = time.time()                
            subprocess.run(
                cmd, shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=_get_custom_env(),
            )
            if self.verbose:
                print(f'\tChexpert labeler process done. Elapsed seconds = {time.time() - start}')
        except subprocess.CalledProcessError as e:
            print('Labeler failed, stdout and stderr:')
            print(e.stdout)
            print(e.stderr)
            raise

        # Read chexpert-labeler output
        out_df = pd.read_csv(output_path)

        assert len(in_df) == len(out_df)

        # Mark nan as -2
        out_df = out_df.fillna(-2)

        return out_df[CHEXPERT_LABELS].to_numpy().astype(np.int8)


