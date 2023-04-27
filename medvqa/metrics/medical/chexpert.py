# from dotenv import load_dotenv
# load_dotenv()
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from sklearn.metrics import f1_score
from medvqa.utils.files import get_cached_pickle_file, save_to_pickle
from medvqa.utils.common import CACHE_DIR, TMP_DIR
from medvqa.utils.constants import CHEXPERT_LABELS
from medvqa.utils.hashing import hash_string
import os
import subprocess
import pandas as pd
import csv
import numpy as np
import time

# CHEXPERT_FOLDER = os.environ['CHEXPERT_FOLDER']
# CHEXPERT_PYTHON = os.environ['CHEXPERT_PYTHON']
# NEGBIO_PATH = os.environ['NEGBIO_PATH']
TMP_FOLDER = os.path.join(TMP_DIR, 'chexpert-labeler')

# def _get_custom_env():
#     custom_env = os.environ.copy()
#     prev = custom_env.get('PYTHONPATH', '')
#     custom_env['PYTHONPATH'] = f'{NEGBIO_PATH}:{prev}'
#     return custom_env

class ChexpertLabelsF1score(Metric):

    def __init__(self, output_transform=lambda x: x, device=None, record_scores=False):
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        if record_scores:
            self._scores = []
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()
        super().reset()

    def update(self, output):
        pred_labels, gt_labels = output
        n, m = pred_labels.shape
        for i in range(n):
            pred = pred_labels[i]
            gt = gt_labels[i]
            if all(pred[j] == gt[j] for j in range(m)):
                score = 1
            elif gt[0] == 1:
                score = 0
            else:
                score = f1_score(gt[1:], pred[1:])
            self._acc_score += score
            if self.record_scores:
                self._scores.append(score)
        self._count += n

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Chexpert F1score needs at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count

class ChexpertLabelerJob:
    def __init__(self, texts, input_filename, output_filename):
        
        self.texts = texts
        
        # Define input & output paths for chexpert labeler
        self.input_path = os.path.join(TMP_FOLDER, input_filename)
        self.output_path = os.path.join(TMP_FOLDER, output_filename)

        # Create input file
        os.makedirs(TMP_FOLDER, exist_ok=True)
        in_df = pd.DataFrame(texts)        
        in_df.to_csv(self.input_path, header=False, index=False, quoting=csv.QUOTE_ALL)

        # Build command
        self.cmd = (f'docker run --rm -v {TMP_FOLDER}:/data chexpert-labeler:latest '
        f'python label.py --reports_path /data/{input_filename} --output_path /data/{output_filename}')
    
    def remove_input_output_files(self):
        os.remove(self.input_path)
        os.remove(self.output_path)

def merge_raw_labels(labels_list):        
    merged = np.zeros((len(CHEXPERT_LABELS),), np.int8)
    merged[0] = 1
    for labels in labels_list:            
        labels = np.where(labels == -2, 0, labels)
        labels = np.where(labels == -1, 1, labels)
        if labels[0] == 0: # no findings
            merged[0] = 0
        for i in range(1, len(CHEXPERT_LABELS)): # abnormalities
            if labels[i] == 1:
                merged[i] = 1
    return merged

def invoke_chexpert_labeler_process(texts, tmp_suffix='', n_chunks=1, max_processes=1,
                                    verbose=True, remove_tmp_files=False):

    n = len(texts)
    chunk_size = n // n_chunks + (n % n_chunks > 0)
    if chunk_size < 70:
        chunk_size = 70
        n_chunks = n // chunk_size + (n % chunk_size > 0)
        chunk_size = n // n_chunks + (n % n_chunks > 0)

    processes = []
    
    if verbose:
        print(f'Chexpert labeler: running a maximum of {max_processes} '
                f'concurrent processes over {n_chunks} chunks')
    
    jobs = []
    for i in range(n_chunks):
        b = i * chunk_size
        e = b + chunk_size
        texts_chunk = texts[b:e]
        if verbose:
            print(f'chunk: i={i}, b={b}, e={e}, chunk_size={len(texts_chunk)}')
        input_filename = f'labeler-input{tmp_suffix}_{i}.csv'
        output_filename = f'labeler-output{tmp_suffix}_{i}.csv'
        jobs.append(ChexpertLabelerJob(texts_chunk, input_filename, output_filename))

    start = time.time()    
    idx = 1    
    job_idxs = list(range(len(jobs)))
    
    while len(job_idxs) > 0 or len(processes) > 0:
        
        if len(processes) == max_processes or len(job_idxs) == 0:
            
            next_processes = []
            
            for p in processes:
                p.wait()
                if verbose:
                    print(f'\t**** process {idx} finished, elapsed time = {time.time() - start}')
                idx += 1
                
                if len(job_idxs) > 0:
                    time.sleep(1)
                    i = job_idxs.pop(0)
                    if verbose:
                        print(f'\t#### process {i+1}: running chexpert labeler over {len(jobs[i].texts)} texts ...')
                        print(f'\tCommand = {jobs[i].cmd}')
                    next_processes.append(subprocess.Popen(jobs[i].cmd, shell=True))
                    
            
            processes.clear()
            processes = next_processes
        
        else:
            time.sleep(1)
            i = job_idxs.pop(0)
            if verbose:
                print(f'\t#### process {i+1}: running chexpert labeler over {len(jobs[i].texts)} texts ...')
                print(f'\tCommand = {jobs[i].cmd}')
            processes.append(subprocess.Popen(jobs[i].cmd, shell=True))    
       
    time.sleep(3)
    
    # Read chexpert-labeler output
    out_labels = np.empty((n, len(CHEXPERT_LABELS)), np.int8)
    offset = 0    
    for job in jobs:
        out_df = pd.read_csv(job.output_path)
        out_df = out_df.fillna(-2)
        assert len(out_df) == len(job.texts)
        out_labels[offset : offset + len(out_df)] = out_df[CHEXPERT_LABELS].to_numpy().astype(np.int8)
        offset += len(out_df)

    assert offset == n

    # Remove tmp files if required
    if remove_tmp_files:
        for job in jobs:
            job.remove_input_output_files()

    return out_labels

class ChexpertLabeler:
    def __init__(self, verbose=True):        
        self.cache_path = os.path.join(CACHE_DIR, 'chexpert_labeler_cache.pkl')
        self.cache = get_cached_pickle_file(self.cache_path)
        self.verbose = verbose
        if self.cache is None:
            self.cache = dict()
        elif verbose:
            print(f'Cache successfully loaded from {self.cache_path}')

    def get_labels(self, texts, fill_empty=0, fill_uncertain=1,
                    n_chunks=10, max_processes=10, tmp_suffix='',
                    update_cache_on_disk=False, remove_tmp_files=False):

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
                    unlabeled_texts.append(text if len(text) > 0 else " ") # HACK for corner case
                    # chexpert labeler explodes with the empty string

        if len(unlabeled_texts) > 0:
            labels = invoke_chexpert_labeler_process(unlabeled_texts, tmp_suffix,
                                                     n_chunks=n_chunks, max_processes=max_processes,
                                                     verbose=self.verbose,
                                                     remove_tmp_files=remove_tmp_files)
            for hash, label in zip(unlabeled_hashes, labels):
                self.cache[hash] = label
            
            if update_cache_on_disk and len(unlabeled_texts) > 10:
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


    # def _invoke_chexpert_labeler_process(self, texts, tmp_suffix=''):

    #     # Define input & output paths
    #     input_path = os.path.join(TMP_FOLDER, f'labeler-input{tmp_suffix}.csv')
    #     output_path = os.path.join(TMP_FOLDER, f'labeler-output{tmp_suffix}.csv')

    #     # Create input file
    #     os.makedirs(TMP_FOLDER, exist_ok=True)
    #     in_df = pd.DataFrame(texts)
    #     in_df.to_csv(input_path, header=False, index=False, quoting=csv.QUOTE_ALL)

    #     # Build command & call chexpert labeler process
    #     cmd_cd = f'cd {CHEXPERT_FOLDER}'
    #     cmd_call = f'{CHEXPERT_PYTHON} label.py --reports_path {input_path} --output_path {output_path}'
    #     cmd = f'{cmd_cd} && {cmd_call}'        
    #     try:            
    #         if self.verbose:
    #             print(f'Running chexpert labeler over {len(in_df)} texts ...')
    #             print(f'\tCommand = {cmd}')
    #             start = time.time()                
    #         subprocess.run(
    #             cmd, shell=True, check=True,
    #             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    #             env=_get_custom_env(),
    #         )
    #         if self.verbose:
    #             print(f'\tChexpert labeler process done. Elapsed seconds = {time.time() - start}')
    #     except subprocess.CalledProcessError as e:
    #         print('Labeler failed, stdout and stderr:')
    #         print(e.stdout)
    #         print(e.stderr)
    #         raise

    #     # Read chexpert-labeler output
    #     out_df = pd.read_csv(output_path)

    #     assert len(in_df) == len(out_df)

    #     # Mark nan as -2
    #     out_df = out_df.fillna(-2)

    #     return out_df[CHEXPERT_LABELS].to_numpy().astype(np.int8)

    # def _invoke_chexpert_labeler_process(self, texts, tmp_suffix='', n_processes = 1):

    #     n = len(texts)
    #     if n < 100:
    #         n_processes = 1

    #     chunk_size = n // n_processes
    #     processes = []
    #     output_paths = []

    #     if self.verbose:
    #         print(f'Chexpert labeler: running {n_processes} processes in parallel')

    #     start = time.time()
    #     custom_env = _get_custom_env()

    #     for i in range(n_processes):
    #         # Define chunk range
    #         b = i * chunk_size
    #         e = n if i + 1 == n_processes else b + chunk_size
    #         # print(f'i={i}, b={b}, e={e}, n={n}, chunk_size={chunk_size}')
            
    #         # Define input & output paths for i-th chunk
    #         input_path = os.path.join(TMP_FOLDER, f'labeler-input{tmp_suffix}_{i}.csv')
    #         output_path = os.path.join(TMP_FOLDER, f'labeler-output{tmp_suffix}_{i}.csv')
    #         output_paths.append(output_path)

    #         # Create input file
    #         os.makedirs(TMP_FOLDER, exist_ok=True)
    #         in_df = pd.DataFrame(texts[b:e])
    #         in_df.to_csv(input_path, header=False, index=False, quoting=csv.QUOTE_ALL)

    #         # Build command & call chexpert labeler process
    #         cmd_cd = f'cd {CHEXPERT_FOLDER}'
    #         cmd_call = f'{CHEXPERT_PYTHON} label.py --reports_path {input_path} --output_path {output_path}'
    #         cmd = f'{cmd_cd} && {cmd_call}'
    #         if self.verbose:
    #             print(f'\t{i+1}) Running chexpert labeler over {len(in_df)} texts ...')
    #             # print(f'\tCommand = {cmd}')
    #         processes.append(subprocess.Popen(cmd, shell=True, env=custom_env))
        
    #     out_labels = np.empty((n, len(CHEXPERT_LABELS)), np.int8)
        
    #     offset = 0        
    #     for i, p in enumerate(processes):
    #         # Wait for subprocess to finish
    #         if p.poll() is None:
    #             p.wait()                
    #         if self.verbose: print(f'\tprocess {i} finished, elapsed time = {time.time() - start}')
    #         # Read chexpert-labeler output
    #         out_df = pd.read_csv(output_paths[i])
    #         out_df = out_df.fillna(-2)
    #         out_labels[offset : offset + len(out_df)] = out_df[CHEXPERT_LABELS].to_numpy().astype(np.int8)
    #         offset += len(out_df)
        
    #     assert offset == n
        
    #     return out_labels