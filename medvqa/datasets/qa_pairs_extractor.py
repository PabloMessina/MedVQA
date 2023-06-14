import os
import re
import random
import pandas as pd

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from medvqa.utils.common import SOURCE_DIR
from medvqa.utils.files import load_json, read_lines_from_txt
from medvqa.metrics.medical.med_completeness import MEDICAL_TERMS_PATH

REGULAR_EXPRESSIONS_FOLDER = os.path.join(SOURCE_DIR, 'medvqa', 'datasets', 'regular_expressions')

_LINKING_WORDS = ['when', 'and', 'but', 'however', 'although', 'should', 'since', 'unless', 'nevertheless']
_aux = '('+'|'.join(f'\\b{x}\\b' for x in _LINKING_WORDS)+'|[,;.:])'
_aux = f'{_aux}+(\\s+{_aux}+)*'
_SEP_REGEX = re.compile(_aux, re.IGNORECASE)

class QuestionRule:
    def __init__(self, question, regex):
        self.question = question
        self.regex = regex
    def match(self, sentence):
        return self.regex.search(sentence)

class QuestionAnswerExtractor:
    
    def __init__(self, debug=False):

        self.stopwords = set(stopwords.words('english'))
        self.reload(debug)
    
    def reload(self, debug=False):
        # load question specifications
        questions = load_json(os.path.join(REGULAR_EXPRESSIONS_FOLDER, 'questions.json'))
        
        if debug: print(questions)
        
        # load question rules
        self.question_rules = []
        for item in questions:
            regex = self._load_regex_from_files(item['files'])
            self.question_rules.append(QuestionRule(item['question'], regex))
            
        self.questions = [rule.question for rule in self.question_rules]
        self.questions.sort()
        assert (len(self.questions) == len(set(self.questions)))
        self.question2index = {q:i for i,q in enumerate(self.questions)}
        
        # load invalid patterns
        self.invalid_regex = self._load_regex_from_files(['invalid_sentence_patterns.txt'])
        
        # load unknown tokens patterns
        self.unknown_regex = self._load_regex_from_files(['unknown_token_patterns.txt'])

        # load replacements
        self._load_replacements()

        # load medical terms
        self.medterms_regex = self._load_medical_terms_regex()
    
    def _load_regex_from_files(self, files):
        pattern = ''
        for file in files:
            with open(os.path.join(REGULAR_EXPRESSIONS_FOLDER, file)) as f:
                for line in f.readlines():
                    if len(pattern) > 0:
                        pattern += '|'
                    pattern += f'({line.strip()})'
        return re.compile(pattern, re.IGNORECASE)

    def _load_medical_terms_regex(self):
        medical_terms = read_lines_from_txt(MEDICAL_TERMS_PATH)
        pattern = f"\\b({'|'.join(medical_terms)})\\b"
        return re.compile(pattern, re.IGNORECASE)


    def _load_replacements(self):
        df = pd.read_csv(os.path.join(REGULAR_EXPRESSIONS_FOLDER, 'replacements.csv'), header=None)
        self.replacements = []
        for source, target in zip(df[0], df[1]):
            self.replacements.append((
                re.compile(source.strip()),
                target.strip()
            ))

    def _apply_replacements(self, sentence):
        for r in self.replacements:
            sentence = r[1].join(r[0].split(sentence))
        return sentence
    
    def get_matched_questions(self, sentence):
        for rule in self.question_rules:
            if rule.match(sentence):
                yield rule.question

    def clean_sentence(self, sentence):
        last_sep = None
        i = 0
        n = len(sentence)
        sentence = self._apply_replacements(sentence)
        clean_sentence = ''
        prepend_to_next_chunk = None
        for match in _SEP_REGEX.finditer(sentence):
            span = match.span()
            if i < span[0]:
                chunk = sentence[i:span[0]].strip()
                if prepend_to_next_chunk:
                    if len(clean_sentence) > 0:
                        chunk = prepend_to_next_chunk + chunk
                    prepend_to_next_chunk = None
                if self._is_relevant(chunk):
                    if len(clean_sentence) > 0:
                        if last_sep:
                            clean_sentence += ' ' + last_sep
                        clean_sentence += ' '                    
                    clean_sentence += chunk
            i = span[1]
            last_sep = sentence[span[0]:span[1]]
            if last_sep not in ('and', 'but', ':'):
                if last_sep == ', however,':
                    last_sep = None
                else:
                    done = False
                    for x in ('should', 'since', 'unless', 'when'):
                        if x in last_sep:
                            prepend_to_next_chunk = last_sep + ' '
                            last_sep = None
                            done = True
                            break
                    if not done:
                        last_sep = ','
        if i < n:
            chunk = sentence[i:].strip()
            if prepend_to_next_chunk:
                chunk = prepend_to_next_chunk + chunk
            if self._is_relevant(chunk):
                if len(clean_sentence) > 0:
                    if last_sep:
                        clean_sentence += ' ' + last_sep
                    clean_sentence += ' '
                clean_sentence += chunk
        return clean_sentence

    def _is_relevant(self, sentence):
        # invalid regex test
        if self.invalid_regex.search(sentence): return False
        # medical terms test
        if not self.medterms_regex.search(sentence): return False
        # unknown token test
        unknown_len = sum(len(x.group()) for x in self.unknown_regex.finditer(sentence))
        if unknown_len > 0.4 * len(sentence): return False
        # we are good to go
        return True

    def generate_qa_pairs(self, text, debug=False):        
        
        # clean sentences and remove duplicates while preserving
        # original order
        sentences = sent_tokenize(text)
        sentences_uniq = set()
        tmp = []
        for s in sentences:
            ss = self.clean_sentence(s)
            if len(ss) == 0:
                if debug:
                    print(f'---- invalid:', s)
                continue
            s = ss
            if s in sentences_uniq: continue
            tmp.append(s)
            sentences_uniq.add(s)
        sentences = tmp
        assert len(sentences_uniq) == len(sentences)
        
        # generate QA pairs
        qa_pairs = dict()
        for s in sentences:
            match = False
            for q in self.get_matched_questions(s):
                try:
                    answers = qa_pairs[q]
                except KeyError:
                    answers = qa_pairs[q] = []
                answers.append(s)
                match = True
            if debug and not match:
                print(f'** not captured:', s)
        return qa_pairs
    
    def generate_qa_pairs_compact_version(self, text):

        output = dict(sentences=[], invalid=[], unmatched=[], matched=[], qa=dict())
        
        # clean sentences and remove duplicates while preserving
        # original order
        sentences = sent_tokenize(text)
        sentences_uniq = set()
        tmp = output['sentences']
        valid_indices = []
        idx = 0
        for s in sentences:
            clean_s = self.clean_sentence(s)
            if len(clean_s) == 0: # invalid
                output['invalid'].append(idx)
                tmp.append(s)
                idx += 1
            else: # valid
                s = clean_s
                if s in sentences_uniq: continue
                sentences_uniq.add(s)
                tmp.append(s)
                valid_indices.append(idx)
                idx += 1
        sentences = tmp
        assert len(sentences) == idx
        
        # generate QA pairs
        question_ids = []
        for i in valid_indices:
            s = sentences[i]
            match = False
            for q in self.get_matched_questions(s):
                q_idx = self.question2index[q]
                if q_idx not in question_ids:
                    question_ids.append(q_idx)
                try:                        
                    answers = output['qa'][str(q_idx)]
                except KeyError:
                    answers = output['qa'][str(q_idx)] = []
                answers.append(i)
                match = True
            if match:
                output['matched'].append(i)
            else:
                output['unmatched'].append(i)
        output['question_ids'] = question_ids
        return output
    
    def get_unmatched_sentences(self, text):
        sentences = sent_tokenize(text)
        for s in sentences:
            s = self.clean_sentence(s)
            if len(s) == 0: continue
            unmatched = True
            for rule in self.question_rules:
                if rule.match(s):
                    unmatched = False
                    break
            if unmatched:
                yield s
    
    def remove_almost_duplicate_sentences(self, qa_pairs, debug=False):
        clean = dict()
        for k, v in qa_pairs.items():
            vv = []
            for i in range(len(v)):
                dup = False
                for j in range(len(v)):
                    if i == j: continue
                    if self.almost_same_sentence(v[i], v[j], i, j):
                        dup = True
                        if debug:
                            print('dups detected:')
                            print('s1:', v[i])
                            print('s2:', v[j])
                        break
                if not dup:
                    vv.append(v[i])
            assert len(vv) > 0
            clean[k] = vv
        return clean
    
    def almost_same_sentence(self, s1, s2, i1, i2):
        if len(s1) == len(s2):
            return s1 == s2 and i1 < i2
        if len(s1) > len(s2):
            return False
        if s1 in s2:
            return True
        if  len(s1) * 10 < 7 * len(s2):
            return False
        count = 0
        found = 0
        s2 = set(w for w in re.split(r",?\s+", s2) if w not in self.stopwords)
        for w in re.split(r",?\s+", s1):
            if w in self.stopwords:
                continue
            if w in s2:
                found += 1
            count += 1
        return found * 10 > 7 * count
    
    def count_unmatches(self, text, min_sent_len = 15, debug=False):
        sentences = sent_tokenize(text)
        count = 0
        for s in sentences:
            s = self.clean_sentence(s)
            if len(s) == 0:
                continue
            if len(s) < min_sent_len:
                continue                
            match = False
            for rule in self.question_rules:
                if rule.match(s):
                    match = True
                    break
            if not match:
                count += 1
                if debug:
                    print(s)
        return count
    
    def search_by_unmatched(self, k, reports, n_samples):
        indices = random.sample(range(len(reports)), n_samples)
        pairs = [(self.count_unmatches(reports[idx]), idx) for idx in indices]
        pairs.sort(reverse=True)
        return pairs[k]