from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
from medvqa.utils.files import (
    load_pickle,
    save_to_pickle
)
from medvqa.datasets.preprocessing import get_sentences
from medvqa.utils.common import CACHE_DIR
import os
import re

class Tokenizer:
    
    PAD_TOKEN = '<pad>'
    START_TOKEN = '<s>'
    END_TOKEN = '</s>'
    ignore_regex = re.compile(r'^(\d+(cm|mm|st|th|nd|rd)?|xxxx|jj|[()\-\\/+#*=><%?;!].*|[:,.].+)$')
    
    def __init__(self, vocab_filepath, qa_adapted_datasets=None, min_freq=4):

        vocab_filepath = os.path.join(CACHE_DIR, vocab_filepath)
        self.id2token = load_pickle(vocab_filepath)
        if self.id2token is None:
            vocab = dict()        
            for sentence in tqdm(get_sentences(qa_adapted_datasets)):
                for token in wordpunct_tokenize(sentence):
                    if self.ignore_regex.search(token):
                        continue
                    vocab[token] = vocab.get(token, 0) + 1        
            filtered_vocab = [word for word, freq in vocab.items() if freq >= min_freq]
            filtered_vocab.sort()
            self.id2token = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]
            self.id2token.extend(filtered_vocab)
            save_to_pickle(self.id2token, vocab_filepath)
        
        self.token2id = {t:i for i,t in enumerate(self.id2token)}
        self.vocab_size = len(self.id2token)

    def string2ids(self, s):
        ids = [self.token2id[self.START_TOKEN]]
        for token in wordpunct_tokenize(s):
            try:
                ids.append(self.token2id[token])
            except KeyError:
                pass
        ids.append(self.token2id[self.END_TOKEN])
        return ids

    def ids2strings(self, ids):
        return ' '.join(self.id2token[i] for i in ids)

    def clean_sentence(self, sentence):
        clean = []
        for id in sentence:
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