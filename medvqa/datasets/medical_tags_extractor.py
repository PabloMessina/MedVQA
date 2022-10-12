from medvqa.metrics.medical.med_completeness import MEDICAL_TERMS_PATH, MEDICAL_SYNONYMS_PATH
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.files import read_lines_from_txt, load_pickle
from nltk.tokenize import wordpunct_tokenize
import os

_VALID_PUNCTUATIONS = ['.', ',', ':']

class MedicalTagsExtractor:

    def __init__(self, medical_terms_frequency_filename):
        self._load_medical_terms()
        self._load_medical_term_frequency(medical_terms_frequency_filename)
        self._load_medical_synonyms()
        self._compute_term2id()

    def _load_medical_terms(self):
        self.medical_terms = set(read_lines_from_txt(MEDICAL_TERMS_PATH))

    def _load_medical_term_frequency(self, filename):
        filepath = os.path.join(CACHE_DIR, filename)
        self.term2freq = load_pickle(filepath)

    def _load_medical_synonyms(self):
        self.medical_synonyms = [line.split() for line in read_lines_from_txt(MEDICAL_SYNONYMS_PATH)]
        self.term2synonym = { x : x for x in self.medical_terms }
        for row in self.medical_synonyms:
            _, max_i = max((self.term2freq[x], i) for i, x in enumerate(row))
            for i in range(len(row)):                
                self.term2synonym[row[i]] = row[max_i]

    def _compute_term2id(self):
        unique_synonyms = list(set(self.term2synonym.values()))
        unique_synonyms.sort()
        self.term2id = { term : unique_synonyms.index(self.term2synonym[term]) for term in self.medical_terms }
        self.tags = unique_synonyms        

    def extract_tags(self, text):
        ids = set()
        for token in wordpunct_tokenize(text.lower()):
            id = self.term2id.get(token, None)
            if id is not None: ids.add(id)
        return [self.tags[id] for id in ids]    

    def extract_tag_ids(self, text):
        ids = set()
        for token in wordpunct_tokenize(text.lower()):
            id = self.term2id.get(token, None)
            if id is not None: ids.add(id)
        return [id for id in ids]
    
    def extract_tags_sequence(self, text):
        tags = []
        for token in wordpunct_tokenize(text.lower()):
            id = self.term2id.get(token, None)
            if id is not None: tags.append(self.tags[id])
        return tags
    
    def extract_tags_sequence_with_punctuation(self, text):
        tags = []
        for token in wordpunct_tokenize(text.lower()):
            if token in _VALID_PUNCTUATIONS:
                tags.append(token)
            else: 
                id = self.term2id.get(token, None)
                if id is not None: tags.append(self.tags[id])
        return tags
