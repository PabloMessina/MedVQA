import argparse
import os
import numpy as np
from nltk.tokenize import word_tokenize

from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.files import load_json, save_pickle
from medvqa.utils.logging import print_bold, print_magenta
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports-filename', type=str, required=True)
    args = parser.parse_args()

    # Load reports
    reports_path = os.path.join(MIMICCXR_CACHE_DIR, args.reports_filename)
    print_bold(f'Loading reports from {reports_path}...')
    reports = load_json(reports_path)
    n_reports = len(reports)
    print(f'Number of reports: {n_reports}')

    # Collect vocabulary of all reports' background sections
    print_bold('Collecting vocabulary of all reports\' background sections...')
    vocab_freq = dict()
    tokenized_backgrounds = [word_tokenize(x['background']) for x in tqdm(reports, mininterval=2)]
    for tokens in tqdm(tokenized_backgrounds, mininterval=2):
        for word in tokens:
            vocab_freq[word] = vocab_freq.get(word, 0) + 1
    print(f'Vocabulary size: {len(vocab_freq)}')
    
    # Sort backgrounds by the inverse of the frequency of their words
    print_bold('Sorting backgrounds by the inverse of the frequency of their words...')
    mean_freq = np.mean(list(vocab_freq.values()))
    print(f'Mean word frequency: {mean_freq}')
    def _complexity(i):
        score = 0
        for word in tokenized_backgrounds[i]:
            score += mean_freq / vocab_freq[word]
        score /= max(len(tokenized_backgrounds[i]), 1) # Avoid division by zero
        return score
    ranked_background_idxs = sorted(range(n_reports), key=_complexity, reverse=True)

    # Save indices
    save_path = os.path.join(MIMICCXR_CACHE_DIR, f'ranked_background_idxs_({args.reports_filename}).pkl')
    save_pickle(ranked_background_idxs, save_path)
    print_magenta(f'Saved ranked background indices to {save_path}', bold=True)