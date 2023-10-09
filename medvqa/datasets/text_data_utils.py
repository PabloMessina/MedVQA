import re
import json
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize

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

def sentence_tokenize_texts_in_parallel(texts, num_workers=None):
    import multiprocessing as mp
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = min(num_workers, mp.cpu_count())
    with mp.Pool(num_workers) as pool:
        sentences = pool.map(sent_tokenize, texts)
    return sentences

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