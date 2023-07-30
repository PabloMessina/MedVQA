import os
from nltk import word_tokenize
from tqdm import tqdm

from medvqa.utils.common import LARGE_FAST_CACHE_DIR
from medvqa.utils.files import load_pickle, save_pickle

def indexes_to_string(indexes):
    return ' '.join(str(i) for i in indexes)

def sort_sentences(sentences, logger, by_difficulty=False, cache_ranking=False):
    if cache_ranking:
        len_sum = sum(len(x) for x in sentences)
        save_path = os.path.join(LARGE_FAST_CACHE_DIR, f"sorted_sentences({len(sentences)},{len_sum}).pkl")
        if os.path.exists(save_path):
            logger.info(f"Loading cached sorted sentences from {save_path}...")
            sorted_sentences = load_pickle(save_path)
            logger.info("Done loading cached sorted sentences.")
            logger.info(f"First sentence: {sorted_sentences[0]}")
            logger.info(f"Last sentence: {sorted_sentences[-1]}")
            return sorted_sentences
        
    logger.info(f"Sorting {len(sentences)} sentences...")
    assert type(sentences) == list, f"Expected list, got {type(sentences)}"
    if by_difficulty:
        logger.info("Sorting sentences by difficulty...")
        tokenized_sentences = [word_tokenize(x) for x in tqdm(sentences, mininterval=2)]
        logger.info("Counting word frequencies...")
        vocab_freq = dict()        
        for tokens in tqdm(tokenized_sentences, mininterval=2):
            for word in tokens:
                vocab_freq[word] = vocab_freq.get(word, 0) + 1
        def _difficulty(i):
            return sum(1 / vocab_freq[word] for word in tokenized_sentences[i])
        ranked_indices = sorted(range(len(tokenized_sentences)), key=_difficulty, reverse=True)
        sorted_sentences = [sentences[i] for i in ranked_indices]
    else:
        logger.info("Sorting sentences by length and alphabetically...")
        sorted_sentences = sorted(sentences, key=lambda x: (len(x), x))

    logger.info("Done sorting sentences.")
    logger.info(f"First sentence: {sorted_sentences[0]}")
    logger.info(f"Last sentence: {sorted_sentences[-1]}")

    if cache_ranking:
        logger.info(f"Caching sorted sentences to {save_path}...")
        save_pickle(sorted_sentences, save_path)
        logger.info("Done caching sorted sentences.")

    return sorted_sentences
