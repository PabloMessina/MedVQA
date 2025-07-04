import os
import re
import json
import logging
import multiprocessing as mp
from tqdm import tqdm
from typing import Any, Callable, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize, wordpunct_tokenize, word_tokenize
from Levenshtein import distance as levenshtein_distance

from medvqa.utils.common import LARGE_FAST_CACHE_DIR
from medvqa.utils.files_utils import load_pickle, save_pickle


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]

def get_text_collate_batch_fn(tokenizer_func):

    def collate_batch_fn(batch):
        assert type(batch) == list, type(batch)
        assert len(batch) > 0, len(batch)
        assert type(batch[0]) == str, type(batch[0])
        # encoding = tokenizer(
        #     batch,
        #     padding="longest",
        #     return_tensors="pt",
        # )
        encoding = tokenizer_func(batch)
        batch_dict = {}
        batch_dict['encoding'] = encoding
        return batch_dict
    
    return collate_batch_fn

def create_text_dataset_and_dataloader(texts, batch_size, num_workers, tokenizer_func):
    # Create collate batch function
    collate_batch_fn = get_text_collate_batch_fn(tokenizer_func)
    # Create dataset
    dataset = TextDataset(texts)
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch_fn,
        pin_memory=True,
    )
    return dataset, dataloader

def split_text_into_chunks(text, max_length):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ''
    for sentence in sentences:
        if len(chunk) + len(sentence) > max_length and chunk != '':
            chunks.append(chunk)
            chunk = ''
        if chunk != '':
            if chunk[-1] != '.':
                chunk += '. '
            else:
                chunk += ' '
        chunk += sentence
    if chunk != '':
        chunks.append(chunk)
    return chunks


def _parallel_map(
    func: Callable[[Any], Any],
    data: List[Any],
    num_workers: Optional[int] = None,
    use_tqdm: bool = False,
    desc: str = ""
) -> List[Any]:
    """
    Helper function to parallelize a function over a list of data with optional tqdm progress bar.

    Args:
        func: Function to apply to each element.
        data: List of data to process.
        num_workers: Number of worker processes to use.
        use_tqdm: Whether to display a tqdm progress bar.
        desc: Description for tqdm progress bar.

    Returns:
        List of results after applying func to each element in data.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = min(num_workers, mp.cpu_count())

    with mp.Pool(num_workers) as pool:
        if use_tqdm:
            imap = pool.imap(func, data)
            results = list(tqdm(imap, total=len(data), desc=desc))
        else:
            results = pool.map(func, data)
    return results


def sentence_tokenize_texts_in_parallel(
    texts: List[str],
    num_workers: Optional[int] = None,
    use_tqdm: bool = False
) -> List[List[str]]:
    """
    Tokenize a list of texts into sentences in parallel.

    Args:
        texts: List of input strings to tokenize.
        num_workers: Number of worker processes to use (default: all CPUs).
        use_tqdm: Whether to display a tqdm progress bar.

    Returns:
        List of lists, where each sublist contains the sentences of a text.
    """
    return _parallel_map(
        sent_tokenize, texts, num_workers, use_tqdm, desc="Sentence tokenizing"
    )


def wordpunct_tokenize_texts_in_parallel(
    texts: List[str],
    num_workers: Optional[int] = None,
    use_tqdm: bool = False
) -> List[List[str]]:
    """
    Tokenize a list of texts into word-punctuation tokens in parallel.

    Args:
        texts: List of input strings to tokenize.
        num_workers: Number of worker processes to use (default: all CPUs).
        use_tqdm: Whether to display a tqdm progress bar.

    Returns:
        List of lists, where each sublist contains the tokens of a text.
    """
    return _parallel_map(
        wordpunct_tokenize, texts, num_workers, use_tqdm, desc="Wordpunct tokenizing"
    )


def word_tokenize_texts_in_parallel(
    texts: List[str],
    num_workers: Optional[int] = None,
    use_tqdm: bool = False
) -> List[List[str]]:
    """
    Tokenize a list of texts into words in parallel.

    Args:
        texts: List of input strings to tokenize.
        num_workers: Number of worker processes to use (default: all CPUs).
        use_tqdm: Whether to display a tqdm progress bar.

    Returns:
        List of lists, where each sublist contains the tokens of a text.
    """
    return _parallel_map(
        word_tokenize, texts, num_workers, use_tqdm, desc="Word tokenizing"
    )


def _lower_tokenized_text(tokenized_text: List[str]) -> List[str]:
    """
    Convert all tokens in a tokenized text to lowercase.

    Args:
        tokenized_text: List of tokens.

    Returns:
        List of lowercase tokens.
    """
    return [t.lower() for t in tokenized_text]


def tokenized_texts_to_lower_in_parallel(
    tokenized_texts: List[List[str]],
    num_workers: Optional[int] = None,
    use_tqdm: bool = False
) -> List[List[str]]:
    """
    Convert all tokens in a list of tokenized texts to lowercase in parallel.

    Args:
        tokenized_texts: List of lists of tokens.
        num_workers: Number of worker processes to use (default: all CPUs).
        use_tqdm: Whether to display a tqdm progress bar.

    Returns:
        List of lists of lowercase tokens.
    """
    return _parallel_map(
        _lower_tokenized_text,
        tokenized_texts,
        num_workers,
        use_tqdm,
        desc="Lowercasing tokens"
    )


_shared_regex: Optional[re.Pattern] = None

def _find_matching_text(text):
    return text if _shared_regex.search(text) else None

def find_texts_matching_regex_in_parallel(
    texts: List[str],
    regex: re.Pattern,
    num_workers: Optional[int] = None,
    use_tqdm: bool = False
) -> List[str]:
    """
    Find texts that match a given regex pattern in parallel.

    Args:
        texts: List of input strings to search.
        regex: Compiled regex pattern to match.
        num_workers: Number of worker processes to use (default: all CPUs).
        use_tqdm: Whether to display a tqdm progress bar.

    Returns:
        List of strings that match the regex pattern.
    """
    global _shared_regex
    _shared_regex = regex  # Set the global regex pattern for use in the worker function    

    results = _parallel_map(
        _find_matching_text, texts, num_workers, use_tqdm, desc="Finding matching texts"
    )
    
    # Filter out None results
    return [text for text in results if text is not None]


def _levenshtein_to_query(
    reference: str, query: str
) -> Tuple[str, int]:
    """
    Compute Levenshtein distance between reference and query.
    """
    return reference, levenshtein_distance(query, reference)

def top_k_most_similar_by_levenshtein(
    query: str,
    reference_texts: List[str],
    k: int = 5,
    num_workers: Optional[int] = None,
    use_tqdm: bool = False
) -> List[Tuple[str, int]]:
    """
    Compute Levenshtein distance in parallel between a query and a list of references,
    and return the top k most similar (lowest distance).

    Args:
        query: The query string.
        reference_texts: List of reference strings to compare.
        k: Number of top similar references to return.
        num_workers: Number of worker processes to use.
        use_tqdm: Whether to display a tqdm progress bar.

    Returns:
        List of tuples: (reference_text, distance), sorted by ascending distance.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = min(num_workers, mp.cpu_count())

    # Prepare arguments for parallel processing
    args = [(ref, query) for ref in reference_texts]

    with mp.Pool(num_workers) as pool:
        if use_tqdm:
            results = list(
                tqdm(
                    pool.starmap(
                        _levenshtein_to_query, args
                    ),
                    total=len(reference_texts),
                    desc="Levenshtein"
                )
            )
        else:
            results = pool.starmap(_levenshtein_to_query, args)

    # Sort by distance (ascending) and return top k
    results.sort(key=lambda x: x[1])
    return results[:k]


_COMMA_SEPARATED_LIST_REGEX = re.compile(r'\[\s*(\".+?\"(\s*,\s*\".+?\")*)?\s*\]?')

def parse_facts(txt):
    facts_str = _COMMA_SEPARATED_LIST_REGEX.search(txt).group()
    if facts_str[-1] != ']': facts_str += ']'
    facts = json.loads(facts_str)
    seen = set()
    clean_facts = []
    for fact in facts:
        if fact not in seen:
            clean_facts.append(fact)
            seen.add(fact)
    return clean_facts


def is_s1_subsequence_of_s2(s1, s2):
    assert type(s1) == list
    assert type(s2) == list
    if len(s1) > len(s2):
        return False
    i = 0
    j = 0
    while i < len(s1) and j < len(s2):
        if s1[i] == s2[j]:
            i += 1
        j += 1
    return i == len(s1)

def _substrings_are_equal(text, i, j, k):
    for x in range(k):
        if text[i+x] != text[j+x]:
            return False
    return True

def remove_consecutive_repeated_words_from_text(text, ks=[1, 2, 3, 4, 5, 6, 7, 8]):
    # Sanity checks
    assert type(ks) == int or type(ks) == list
    if type(ks) == int:
        ks = [ks]
    else:
        assert len(ks) > 0
        assert all(type(x) == int for x in ks)

    tokens = text.split()
    lower_tokens = text.lower().split()
    dedup_tokens = []
    dedup_lower_tokens = []

    for k in ks:
        for i in range(len(lower_tokens)):
            # if current word is part of a k-word phrase that is repeated -> skip
            skip = False
            for j in range(k):
                s = i - j # start index
                e = s + k-1 # end index
                if s - k >= 0 and e < len(lower_tokens) and _substrings_are_equal(lower_tokens, s, s-k, k):
                    skip = True
                    break
            if skip:
                continue
            dedup_tokens.append(tokens[i])
            dedup_lower_tokens.append(lower_tokens[i])
        tokens = dedup_tokens
        lower_tokens = dedup_lower_tokens
        dedup_tokens = []
        dedup_lower_tokens = []
    return ' '.join(tokens)



# This global variable is used to share sentences across processes when sorting by difficulty.
_shared_sentences: List[str] = None

def sort_sentences(
    sentences: List[str],
    by_difficulty: bool = False,
    increasing: bool = False,
    cache_ranking: bool = False,
    sort_indices: bool = False,
    num_processes: int = mp.cpu_count(),
) -> Union[List[str], List[int]]:
    """Sorts a list of sentences based on specified criteria.

    Sentences can be sorted by length and alphabetically (default) or by difficulty,
    which is determined by word frequency. The function also supports caching the
    sorted results for faster retrieval on subsequent calls.

    Args:
        sentences: A list of strings, where each string is a sentence.
        by_difficulty: If True, sentences are sorted by a calculated difficulty score.
                       If False, sentences are sorted by length, then alphabetically.
                       Defaults to False.
        increasing: If True, sort in increasing order. If False, sort in decreasing order.
                    This applies to both length/alphabetical sort and difficulty sort.
                    Defaults to False.
        cache_ranking: If True, the sorted output (either sentences or indices) will be
                       cached to disk. If a cached version exists, it will be loaded
                       instead of re-sorting. Defaults to False.
        sort_indices: If True, the function returns a list of indices that represent
                      the sorted order of the original `sentences` list. If False,
                      the function returns the sorted list of sentences themselves.
                      Defaults to False.
        num_processes: The number of CPU processes to use when sorting by difficulty.
                       Defaults to the number of CPU cores available.

    Returns:
        A list of sentences (List[str]) if `sort_indices` is False,
        or a list of integers (List[int]) representing the sorted indices
        of the original `sentences` list if `sort_indices` is True.

    Raises:
        AssertionError: If `sentences` is not a list.
    """
    if cache_ranking:
        # Calculate a unique identifier for the cache file based on input parameters.
        len_sum = sum(len(x) for x in sentences)
        suffix = "_increasing" if increasing else ""
        save_path = os.path.join(
            LARGE_FAST_CACHE_DIR,
            f"sorted_sentences({len(sentences)},{len_sum},{sort_indices}){suffix}.pkl",
        )
        if os.path.exists(save_path):
            logger.info(f"Loading cached sorted sentences from {save_path}...")
            cached_output = load_pickle(save_path)
            logger.info("Done loading cached sorted sentences.")
            # Log the first and last items of the cached output for verification.
            if sort_indices:
                logger.info(f"First sentence: {sentences[cached_output[0]]}")
                logger.info(f"Last sentence: {sentences[cached_output[-1]]}")
            else:
                logger.info(f"First sentence: {cached_output[0]}")
                logger.info(f"Last sentence: {cached_output[-1]}")
            return cached_output

    logger.info(f"Sorting {len(sentences)} sentences...")
    # Validate input type.
    assert isinstance(sentences, list), f"Expected list, got {type(sentences)}"

    if by_difficulty:
        logger.info("Sorting sentences by difficulty...")
        # Assign sentences to a global variable for efficient sharing with worker processes.
        global _shared_sentences
        _shared_sentences = sentences

        # Tokenize sentences in parallel using multiprocessing.
        with mp.Pool(num_processes) as pool:
            tokenized_sentences = pool.map(word_tokenize, _shared_sentences)

        logger.info("Counting word frequencies...")
        # Calculate word frequencies across all tokenized sentences.
        vocab_freq = {}
        for tokens in tqdm(tokenized_sentences, mininterval=2):
            for word in tokens:
                vocab_freq[word] = vocab_freq.get(word, 0) + 1

        # Define the difficulty scoring function.
        def _difficulty(i: int) -> float:
            """Calculates the difficulty score for a sentence based on word frequencies.
            Lower scores indicate easier sentences (higher frequency words).
            """
            score = sum(1 / vocab_freq[word] for word in tokenized_sentences[i])
            return score

        # Sort sentence indices based on their difficulty scores.
        ranked_indices = sorted(
            range(len(tokenized_sentences)),
            key=_difficulty,
            reverse=not increasing,  # Apply reverse based on the 'increasing' flag
        )
    else:
        logger.info("Sorting sentences by length and alphabetically...")
        # Sort sentence indices based on length and then alphabetically.
        ranked_indices = sorted(
            range(len(sentences)),
            key=lambda i: (len(sentences[i]), sentences[i]),
            reverse=not increasing,  # Apply reverse based on the 'increasing' flag
        )

    logger.info("Done sorting sentences.")
    # Log the first and last sentences of the sorted output for verification.
    logger.info(f"First sentence: {sentences[ranked_indices[0]]}")
    logger.info(f"Last sentence: {sentences[ranked_indices[-1]]}")

    # Prepare the output based on `sort_indices` flag.
    if sort_indices:
        output_to_return = ranked_indices
    else:
        output_to_return = [sentences[i] for i in ranked_indices]

    if cache_ranking:
        logger.info(f"Caching ranking to {save_path}...")
        save_pickle(output_to_return, save_path)
        logger.info("Done caching ranking.")

    return output_to_return


def indexes_to_string(indexes):
    return ' '.join(str(i) for i in indexes)