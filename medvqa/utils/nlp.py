import os
import multiprocessing as mp
from nltk import word_tokenize
from tqdm import tqdm

from medvqa.utils.common import LARGE_FAST_CACHE_DIR
from medvqa.utils.files_utils import load_pickle, save_pickle

_shared_sentences = None

def indexes_to_string(indexes):
    return ' '.join(str(i) for i in indexes)

def sort_sentences(sentences, logger=None, by_difficulty=False, increasing=False, cache_ranking=False, num_workers=mp.cpu_count()):
    if logger is None:
        logger_info = print
    else:
        logger_info = logger.info
    if cache_ranking:
        len_sum = sum(len(x) for x in sentences)
        suffix = "_increasing" if increasing else ""
        save_path = os.path.join(LARGE_FAST_CACHE_DIR, f"sorted_sentences({len(sentences)},{len_sum}){suffix}.pkl")
        if os.path.exists(save_path):
            logger_info(f"Loading cached sorted sentences from {save_path}...")
            sorted_sentences = load_pickle(save_path)
            logger_info("Done loading cached sorted sentences.")
            logger_info(f"First sentence: {sorted_sentences[0]}")
            logger_info(f"Last sentence: {sorted_sentences[-1]}")
            return sorted_sentences
        
    logger_info(f"Sorting {len(sentences)} sentences...")
    assert type(sentences) == list, f"Expected list, got {type(sentences)}"
    if by_difficulty:
        logger_info("Sorting sentences by difficulty...")
        global _shared_sentences
        _shared_sentences = sentences
        with mp.Pool(num_workers) as pool:
            tokenized_sentences = pool.map(word_tokenize, _shared_sentences)
        logger_info("Counting word frequencies...")
        vocab_freq = dict()        
        for tokens in tqdm(tokenized_sentences, mininterval=2):
            for word in tokens:
                vocab_freq[word] = vocab_freq.get(word, 0) + 1
        def _difficulty(i):
            score = sum(1 / vocab_freq[word] for word in tokenized_sentences[i])
            if increasing:
                return score
            else:
                return -score
        ranked_indices = sorted(range(len(tokenized_sentences)), key=_difficulty)
        sorted_sentences = [sentences[i] for i in ranked_indices]
    else:
        logger_info("Sorting sentences by length and alphabetically...")
        sorted_sentences = sorted(sentences, key=lambda x: (len(x), x))

    logger_info("Done sorting sentences.")
    logger_info(f"First sentence: {sorted_sentences[0]}")
    logger_info(f"Last sentence: {sorted_sentences[-1]}")

    if cache_ranking:
        logger_info(f"Caching sorted sentences to {save_path}...")
        save_pickle(sorted_sentences, save_path)
        logger_info("Done caching sorted sentences.")

    return sorted_sentences
