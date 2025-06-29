import re
import json
import multiprocessing as mp
from tqdm import tqdm
from typing import Any, Callable, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize, wordpunct_tokenize, word_tokenize
from Levenshtein import distance as levenshtein_distance


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
        imap = pool.imap(func, data)
        if use_tqdm:
            results = list(tqdm(imap, total=len(data), desc=desc))
        else:
            results = list(imap)
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