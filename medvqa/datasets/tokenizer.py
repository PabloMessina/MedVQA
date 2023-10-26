from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
from medvqa.datasets.medical_tags_extractor import MedicalTagsExtractor
from medvqa.datasets.text_data_utils import wordpunct_tokenize_texts_in_parallel
from medvqa.utils.files import (
    get_cached_json_file,
    load_pickle,
    save_pickle,
    load_json,
    get_file_path_with_hashing_if_too_long,
)
from medvqa.datasets.preprocessing import get_sentences
from medvqa.datasets.qa_pairs_extractor import REGULAR_EXPRESSIONS_FOLDER
from medvqa.utils.common import CACHE_DIR
from medvqa.metrics.medical.med_completeness import MEDICAL_TERMS_PATH
from medvqa.utils.hashing import hash_string, hash_string_list
import os
import re

_IGNORE_REGEX = re.compile(r'^(\d+(cm|mm|st|th|nd|rd)?|xxxx|jj|[()\[\]\-\\/+#*=><%?;!].*|[:,.].+)$')
_VALID_PUNCTUATIONS = ['.', ',', ':']

def _get_vocab_filepath(qa_adapted_filenames=None, min_freq=None, mode=None, other_vocab_generators_names=None):
    strings = []
    if qa_adapted_filenames is not None:
        assert type(qa_adapted_filenames) is list
        if len(qa_adapted_filenames) > 0:
            assert min_freq is not None
            assert mode is not None
            strings.append(f'min_freq={min_freq}')
            strings.append(f'mode={mode}')
            strings.extend(qa_adapted_filenames)
    if other_vocab_generators_names is not None:
        assert type(other_vocab_generators_names) is list
        strings.extend(other_vocab_generators_names)    
    assert len(strings) > 0, 'No strings to hash'
    prefix = 'vocab'
    return get_file_path_with_hashing_if_too_long(CACHE_DIR, prefix, strings, 'pkl')

class Tokenizer:
    
    PAD_TOKEN = '<pad>'
    START_TOKEN = '<s>'
    END_TOKEN = '</s>'
    
    def __init__(self, qa_adapted_dataset_paths=None, vocab_min_freq=5, overwrite=False,
                mode='report', use_medical_tokenization=False, medical_terms_frequency_filename=None,
                other_vocab_generators=None, other_vocab_generators_names=None,
                vocab_filepath=None):

        if vocab_filepath is None:
            if qa_adapted_dataset_paths is not None:
                assert type(qa_adapted_dataset_paths) is list, type(qa_adapted_dataset_paths)
                assert len(qa_adapted_dataset_paths) > 0, len(qa_adapted_dataset_paths)
                assert mode in ('report', 'background'), mode
                qa_adapted_filenames = [os.path.basename(x) for x in qa_adapted_dataset_paths]
            else:
                qa_adapted_filenames = None
            vocab_filepath = _get_vocab_filepath(qa_adapted_filenames, vocab_min_freq, mode,
                                            other_vocab_generators_names)
        self.vocab_filepath = vocab_filepath
        
        self.medical_tokenization = use_medical_tokenization
        if use_medical_tokenization:
            assert medical_terms_frequency_filename is not None
            self.med_tags_extractor = MedicalTagsExtractor(medical_terms_frequency_filename)
            self.medical_terms_frequency_filename = medical_terms_frequency_filename

        if not overwrite:
            print(f'Loading {vocab_filepath} ...')
            self.id2token = load_pickle(vocab_filepath)

        if overwrite or self.id2token is None:
            # process Q&A datasets
            if qa_adapted_dataset_paths is not None:
                vocab = dict()
                qa_adapted_datasets = [get_cached_json_file(path) for path in qa_adapted_dataset_paths]
                for sentence in tqdm(get_sentences(qa_adapted_datasets, mode=mode)):
                    for token in wordpunct_tokenize(sentence):
                        if _IGNORE_REGEX.search(token):
                            continue
                        vocab[token] = vocab.get(token, 0) + 1
                # filter by frequency
                filtered_vocab = set(word for word, freq in vocab.items() if freq >= vocab_min_freq)
            else:
                filtered_vocab = set()

            # valid punctuations
            filtered_vocab.update(_VALID_PUNCTUATIONS)

            if mode == 'report':
                # include questions' vocab
                questions = load_json(os.path.join(REGULAR_EXPRESSIONS_FOLDER, 'questions.json'))
                for item in questions:
                    for token in item['question'][:-1].split():
                        filtered_vocab.add(token)

                # include medical terms
                with open(MEDICAL_TERMS_PATH) as f:
                    for line in f.readlines():                    
                        filtered_vocab.add(line.strip())

            # include other vocab generators
            if other_vocab_generators is not None:
                for other_vocab_generator in other_vocab_generators:
                    for token in other_vocab_generator():
                        filtered_vocab.add(token)
            
            # sort
            filtered_vocab = sorted(list(filtered_vocab))

            self.id2token = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]
            self.id2token.extend(filtered_vocab)
            save_pickle(self.id2token, vocab_filepath)
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
        s = s.lower()
        ids = [self.token2id[self.START_TOKEN]]
        for token in wordpunct_tokenize(s):
            try:
                ids.append(self.token2id[token])
            except KeyError:
                pass
        ids.append(self.token2id[self.END_TOKEN])
        return ids
    
    def string2medical_tag_ids(self, s):
        s = s.lower()
        tags = self.med_tags_extractor.extract_tags_sequence_with_punctuation(s)
        ids = [self.token2id[self.START_TOKEN]]
        for tag in tags:
            ids.append(self.token2id[tag])
        ids.append(self.token2id[self.END_TOKEN])
        return ids
    
    def tokenize(self, s):
        s = s.lower()
        if self.medical_tokenization:
            return self.string2medical_tag_ids(s)
        return self.string2ids(s)

    def ids2string(self, ids, remove_special_tokens=False):
        if remove_special_tokens:
            ids = self.clean_sentence(ids)
        return ' '.join(self.id2token[i] for i in ids)

    def clean_sentence(self, sentence):
        clean = []
        for id in sentence:
            if not isinstance(id, int):
                try:
                    id = id.item()
                except ValueError:
                    print('sentence.shape=',sentence.shape)
                    print('sentence=',sentence)
                    print('id=',id)
                    print('type(id)=',type(id))
                    raise
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
        try:
            clean_sentences = [None] * len(batch)
            for i in range(len(batch)):
                clean_sentences[i] = self.clean_sentence(batch[i])
        except ValueError:
            print('batch.shape=',batch.shape)
            print('batch=',batch)
            raise
        return clean_sentences
    
class BasicTokenizer:
    
    PAD_TOKEN = '<PAD>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
    
    def __init__(self, vocab_filepath=None, texts=None, vocab_min_freq=5):

        build_vocab = True
        if vocab_filepath is not None:
            if os.path.exists(vocab_filepath):
                print(f'Loading {vocab_filepath} ...')
                self.id2token = load_pickle(vocab_filepath)
                build_vocab = False

        if build_vocab:
            assert texts is not None
            assert type(texts) is list
            print(f'Building vocabulary from {len(texts)} texts ...')
            token2freq = dict()
            tokens_per_text = wordpunct_tokenize_texts_in_parallel(texts)
            for tokens in tokens_per_text:
                for token in tokens:
                    token2freq[token] = token2freq.get(token, 0) + 1
            # filter by frequency
            filtered_vocab = set(word for word, freq in token2freq.items() if freq >= vocab_min_freq)
            # sort
            filtered_vocab = sorted(list(filtered_vocab))
            self.id2token = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]
            self.id2token.extend(filtered_vocab)
            # save
            if vocab_filepath is not None:
                save_pickle(self.id2token, vocab_filepath)
                print (f'Vocabulary saved to {vocab_filepath}')
        
        self.vocab_size = len(self.id2token)
        print(f'Vocabulary size: {self.vocab_size}')
        self.token2id = {t:i for i,t in enumerate(self.id2token)}
        self.vocab = set(self.id2token)
        self._hash = None

    @property
    def hash(self):
        if self._hash is None:
            self._hash = hash_string_list(self.id2token)
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

    def ids2string(self, ids, remove_special_tokens=False):
        if remove_special_tokens:
            ids = self.clean_ids(ids)
        return ' '.join(self.id2token[i] for i in ids)
    
    def batch_string2ids(self, batch, in_parallel=False, num_workers=None):
        if in_parallel:
            import multiprocessing
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()
            with multiprocessing.Pool(num_workers) as pool:
                ids_per_string = pool.map(self.string2ids, batch)
        else:
            ids_per_string = [self.string2ids(s) for s in batch]
        return ids_per_string

    def clean_ids(self, ids):
        clean_ids = []
        for id in ids:
            if not isinstance(id, int):
                id = id.item()
            if id == self.token2id[self.END_TOKEN]:
                break
            if id >= 3:
                clean_ids.append(id)
        return clean_ids

    def clean_string(self, s, remove_special_tokens=False):
        ids = self.string2ids(s)
        ids = self.clean_ids(ids)
        return self.ids2string(ids, remove_special_tokens=remove_special_tokens)
    
    def clean_batch(self, batch):
        clean_batch_ids = [None] * len(batch)
        for i in range(len(batch)):
            clean_batch_ids[i] = self.clean_ids(batch[i])
        return clean_batch_ids