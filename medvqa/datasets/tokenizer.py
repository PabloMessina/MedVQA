from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
from medvqa.utils.files import (
    load_pickle,
    save_to_pickle,
    load_json_file,
)
from medvqa.datasets.preprocessing import get_sentences
from medvqa.datasets.qa_pairs_extractor import REGULAR_EXPRESSIONS_FOLDER
from medvqa.utils.common import CACHE_DIR
from medvqa.metrics.medical.med_completeness import MEDICAL_TERMS_PATH
from medvqa.utils.hashing import hash_string
import os
import re

def _get_vocab_filepath(qa_adapted_filenames, min_freq):
    filename = f'vocab__min_freq={min_freq}__from({";".join(qa_adapted_filenames)}).pkl'
    return os.path.join(CACHE_DIR, filename)    

class Tokenizer:
    
    PAD_TOKEN = '<pad>'
    START_TOKEN = '<s>'
    END_TOKEN = '</s>'
    ignore_regex = re.compile(r'^(\d+(cm|mm|st|th|nd|rd)?|xxxx|jj|[()\[\]\-\\/+#*=><%?;!].*|[:,.].+)$')
    
    def __init__(self, qa_adapted_filenames, qa_adapted_datasets=None, min_freq=5, overwrite=False):

        assert type(qa_adapted_datasets) is list, type(qa_adapted_filenames)

        vocab_filepath = _get_vocab_filepath(qa_adapted_filenames, min_freq)

        if not overwrite:
            print(f'Loading {vocab_filepath} ...')
            self.id2token = load_pickle(vocab_filepath)

        if overwrite or self.id2token is None:
            # process Q&A datasets
            vocab = dict()
            for sentence in tqdm(get_sentences(qa_adapted_datasets)):
                for token in wordpunct_tokenize(sentence):
                    if self.ignore_regex.search(token):
                        continue
                    vocab[token] = vocab.get(token, 0) + 1
            # filter by frequency
            filtered_vocab = set(word for word, freq in vocab.items() if freq >= min_freq)

            # include questions' vocab
            questions = load_json_file(os.path.join(REGULAR_EXPRESSIONS_FOLDER, 'questions.json'))
            for item in questions:
                for token in item['question'][:-1].split():
                    filtered_vocab.add(token)

            # include medical terms
            with open(MEDICAL_TERMS_PATH) as f:
                for line in f.readlines():                    
                    filtered_vocab.add(line.strip())
            
            # sort
            filtered_vocab = sorted(list(filtered_vocab))

            self.id2token = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]
            self.id2token.extend(filtered_vocab)
            save_to_pickle(self.id2token, vocab_filepath)
            print (f'Vocabulary saved to {vocab_filepath}')
        
        self.token2id = {t:i for i,t in enumerate(self.id2token)}
        self.vocab_size = len(self.id2token)
        self.vocab = set(self.id2token)
        self._hash = None

    @property
    def hash(self):
        if self._hash is None:
            self._hash = hash_string(''.join(self.id2token))
        return self._hash

    def string2ids(self, s):
        ids = [self.token2id[self.START_TOKEN]]
        for token in wordpunct_tokenize(s):
            try:
                ids.append(self.token2id[token])
            except KeyError:
                pass
        ids.append(self.token2id[self.END_TOKEN])
        return ids

    def ids2string(self, ids):
        return ' '.join(self.id2token[i] for i in ids)

    def clean_sentence(self, sentence):
        clean = []
        for id in sentence:
            if not isinstance(id, int):
                id = id.item()
            if id == self.token2id[self.END_TOKEN]:
                break
            if id >= 3:
                clean.append(id)
        return clean
    
    def clean_batch(self, batch):
        clean_sentences = [None] * len(batch)
        for i in range(len(batch)):
            clean_sentences[i] = self.clean_sentence(batch[i])
        return clean_sentences