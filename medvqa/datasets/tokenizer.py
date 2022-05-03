from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
from medvqa.datasets.medical_tags_extractor import MedicalTagsExtractor
from medvqa.utils.files import (
    get_cached_json_file,
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

_IGNORE_REGEX = re.compile(r'^(\d+(cm|mm|st|th|nd|rd)?|xxxx|jj|[()\[\]\-\\/+#*=><%?;!].*|[:,.].+)$')

def _get_vocab_filepath(qa_adapted_filenames, min_freq):
    filename = f'vocab__min_freq={min_freq}__from({";".join(qa_adapted_filenames)}).pkl'
    return os.path.join(CACHE_DIR, filename)

class Tokenizer:
    
    PAD_TOKEN = '<pad>'
    START_TOKEN = '<s>'
    END_TOKEN = '</s>'    
    
    def __init__(self, qa_adapted_dataset_paths, min_freq=5, overwrite=False,
                medical_terms_frequency_filename = None):

        assert type(qa_adapted_dataset_paths) is list, type(qa_adapted_dataset_paths)

        qa_adapted_filenames = [os.path.basename(x) for x in qa_adapted_dataset_paths]

        vocab_filepath = _get_vocab_filepath(qa_adapted_filenames, min_freq)
        
        if medical_terms_frequency_filename is not None:
            self.med_tags_extractor = MedicalTagsExtractor(medical_terms_frequency_filename)
            self.medical_terms_frequency_filename = medical_terms_frequency_filename
            self.medical_tokenization = True
        else:
            self.medical_tokenization = False

        if not overwrite:
            print(f'Loading {vocab_filepath} ...')
            self.id2token = load_pickle(vocab_filepath)

        if overwrite or self.id2token is None:
            # process Q&A datasets            
            vocab = dict()
            qa_adapted_datasets = [get_cached_json_file(path) for path in qa_adapted_dataset_paths]
            for sentence in tqdm(get_sentences(qa_adapted_datasets)):
                for token in wordpunct_tokenize(sentence):
                    if _IGNORE_REGEX.search(token):
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
    
    def strig2medical_tag_ids(self, s):
        tags = self.med_tags_extractor.extract_tags_sequence_with_punctuation(s)
        ids = [self.token2id[self.START_TOKEN]]
        for tag in tags:
            ids.append(self.token2id[tag])
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
            if id >= 3 and (len(clean) == 0 or clean[-1] != id):
                clean.append(id)
        return clean

    def clean_text(self, text):
        text = text.lower()
        ids = self.string2ids(text)
        ids = self.clean_sentence(ids)
        return self.ids2string(ids)
    
    def clean_batch(self, batch):
        clean_sentences = [None] * len(batch)
        for i in range(len(batch)):
            clean_sentences[i] = self.clean_sentence(batch[i])
        return clean_sentences