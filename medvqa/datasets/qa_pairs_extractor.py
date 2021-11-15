import os
import re
import random

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from medvqa.utils.common import SOURCE_DIR
from medvqa.utils.files import load_json_file

REGULAR_EXPRESSIONS_FOLDER = os.path.join(SOURCE_DIR, 'medvqa', 'datasets', 'regular_expressions')

class QuestionRule:
    def __init__(self, question, regex):
        self.question = question
        self.regex = regex
    def match(self, sentence):
        return self.regex.search(sentence)

class QuestionAnswerExtractor:
    
    def __init__(self, debug=False):

        self.english_stopwords = set(stopwords.words('english'))
        self.reload(debug)
    
    def reload(self, debug=False):
        # load question specifications
        questions = load_json_file(os.path.join(REGULAR_EXPRESSIONS_FOLDER, 'questions.json'))
        
        if debug: print(questions)
        
        # load question rules
        self.question_rules = []
        for item in questions:
            regex = self.get_regex_from_files(item['files'])
            self.question_rules.append(QuestionRule(item['question'], regex))
            
        self.questions = [rule.question for rule in self.question_rules]
        self.questions.sort()
        assert (len(self.questions) == len(set(self.questions)))
        self.question2index = {q:i for i,q in enumerate(self.questions)}
        
        # load invalid patterns
        self.invalid_regex = self.get_regex_from_files(['invalid_sentence_patterns.txt'])
        
        # load unknown tokens patterns
        self.unknown_regex = self.get_regex_from_files(['unknown_token_patterns.txt'])
    
    def get_regex_from_files(self, files):
        pattern = ''
        for file in files:
            with open(os.path.join(REGULAR_EXPRESSIONS_FOLDER, file)) as f:
                for line in f.readlines():
                    if len(pattern) > 0:
                        pattern += '|'
                    pattern += f'({line.strip()})'
        return re.compile(pattern, re.IGNORECASE)
    
    def get_matched_questions(self, sentence):
        for rule in self.question_rules:
            if rule.match(sentence):
                yield rule.question

    def generate_qa_pairs(self, text, debug=False):        
        
        # remove duplicates while keeping original order
        sentences = sent_tokenize(text)
        sentences_uniq = set()
        tmp = []
        for s in sentences:
            if s in sentences_uniq: continue
            tmp.append(s)
            sentences_uniq.add(s)
        sentences = tmp
        assert len(sentences_uniq) == len(sentences)
        
        qa_pairs = dict()        
        for s in sentences:
            match = False
            valid = False
            if self.valid_sentence(s):
                valid = True
                for q in self.get_matched_questions(s):
                    try:
                        answers = qa_pairs[q]
                    except KeyError:
                        answers = qa_pairs[q] = []
                    answers.append(s)
                    match = True
            if debug and not match:
                if valid:
                    print(f'** not captured:', s)
                else:
                    print(f'------- invalid:', s)
        return qa_pairs
    
    def generate_qa_pairs_compact_version(self, text):        
        
        # remove duplicates while keeping original order
        sentences = sent_tokenize(text)
        sentences_uniq = set()
        tmp = []
        for s in sentences:
            if s in sentences_uniq:
                continue
            tmp.append(s)
            sentences_uniq.add(s)
        sentences = tmp
        assert len(sentences_uniq) == len(sentences)
        
        output = dict(sentences=sentences, invalid=[], unmatched=[], matched=[], qa=dict())
        for i, s in enumerate(sentences):
            match = False
            valid = False
            if self.valid_sentence(s):
                valid = True
                for q in self.get_matched_questions(s):
                    q_idx = self.question2index[q]
                    try:                        
                        answers = output['qa'][str(q_idx)]
                    except KeyError:
                        answers = output['qa'][str(q_idx)] = []
                    answers.append(i)
                    match = True
            if match:
                output['matched'].append(i)
            elif valid:
                output['unmatched'].append(i)
            else:
                output['invalid'].append(i)
        return output
    
    def get_unmatched_sentences(self, text):
        sentences = sent_tokenize(text)
        for s in sentences:            
            if self.valid_sentence(s):
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
        s2 = set(w for w in re.split(r",?\s+", s2) if w not in english_stopwords)
        for w in re.split(r",?\s+", s1):
            if w in self.english_stopwords:
                continue
            if w in s2:
                found += 1
            count += 1
        return found * 10 > 7 * count
    
    def valid_sentence(self, s):
        if self.invalid_regex.search(s): return False
        unknown_len = sum(len(x.group()) for x in self.unknown_regex.finditer(s))
        if unknown_len * 10 >= len(s): return False
        return True
    
    def count_unmatches(self, text, min_sent_len = 15, debug=False):
        sentences = sent_tokenize(text)
        count = 0
        for s in sentences:
            if self.valid_sentence(s):
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