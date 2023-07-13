import os
import random
import re
from pprint import pprint
from medvqa.utils.files import load_json, load_jsonl
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from medvqa.utils.logging import print_orange, print_red

_NEG_REGEX = re.compile(r'^\s*no(t|ne)?\s+', re.IGNORECASE)
_FACT_METADATA_FIELDS = ('anatomical location', 'detailed observation', 'short observation', 'category', 'health status', 'prev_study_comparison?', 'comparison status')

def _is_s1_subsequence_of_s2(s1, s2):
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

def _remove_consecutive_repeated_words_from_text(text, ks=[1, 2, 3, 4, 5, 6, 7, 8]):
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

class ReportFactsDisplayer:
    def __init__(self, preprocessed_reports_filepath, extracted_facts_filepaths, skip_negated_facts=True):
        assert type(preprocessed_reports_filepath) == str
        assert type(extracted_facts_filepaths) == list
        assert len(extracted_facts_filepaths) > 0
        assert all(type(fp) == str for fp in extracted_facts_filepaths)

        assert os.path.exists(preprocessed_reports_filepath)
        print(f'Loading preprocessed reports from {preprocessed_reports_filepath}...')
        self.preprocessed_reports = load_json(preprocessed_reports_filepath)

        self.sentence2facts = {}
        self.sentence2filename = {}
        for extracted_facts_filepath in extracted_facts_filepaths:
            assert os.path.exists(extracted_facts_filepath)
            print(f'Loading extracted facts from {extracted_facts_filepath}...')
            filename = os.path.basename(extracted_facts_filepath)
            extracted_facts = load_jsonl(extracted_facts_filepath)
            for x in extracted_facts:
                try:
                    if 'metadata' in x:
                        s = x['metadata']['sentence']
                        fs = x['parsed_response']
                    else:
                        s = x['sentence']
                        fs = x['extracted_facts']
                except KeyError:
                    print(f'KeyError: {x}')
                    raise
                assert s not in self.sentence2facts
                if skip_negated_facts:
                    fs = [f for f in fs if not _NEG_REGEX.match(f)]
                self.sentence2facts[s] = fs
                self.sentence2filename[s] = filename

        print('Building facts...')
        self.facts = set()
        for s in self.sentence2facts:
            for f in self.sentence2facts[s]:
                self.facts.add(f)
        self.facts = list(self.facts)

        print('Building facts to report ids...')
        self.facts2report_ids = {}
        for rid, report in tqdm(enumerate(self.preprocessed_reports), mininterval=2):
            for x in [report['findings'], report['impression']]:
                for s in sent_tokenize(x):
                    for f in self.sentence2facts[s]:
                        if f not in self.facts2report_ids:
                            self.facts2report_ids[f] = set()
                        self.facts2report_ids[f].add(rid)
        
        print(f'Loaded {len(self.preprocessed_reports)} preprocessed reports')
        print(f'Loaded {len(self.sentence2facts)} sentences with extracted facts')
        print(f'Loaded {len(self.facts)} unique facts')
        print(f'Loaded {len(self.facts2report_ids)} facts with report ids')

    def display(self, rid=None):
        if rid is None:
            rid = random.randint(0, len(self.preprocessed_reports)-1)
        report = self.preprocessed_reports[rid]
        print(f'RID: {rid}')
        print()
        pprint(report)
        print()
        print('Facts:')
        facts = []
        for x in [report['findings'], report['impression']]:
            for s in sent_tokenize(x):
                print(f'\t{s}')
                if s in self.sentence2facts:
                    for f in self.sentence2facts[s]:
                        print(f'\t\t{f}')
                        facts.append(f)
                else:
                    print('\t\tNo facts extracted')
        print()
        print('Paraphrased report:')
        facts_ = [f.lower().split() for f in facts]
        dedup_facts = []
        for i in range(len(facts)):
            duplicate = False
            for j in range(len(facts)):
                if i == j:
                    continue
                if _is_s1_subsequence_of_s2(facts_[i], facts_[j]):
                    if facts[i] == facts[j] and i > j:
                        continue
                    duplicate = True
                    break
            if not duplicate:
                dedup_facts.append(facts[i])
        for f in dedup_facts:
            print(f'\t{f}')

    def display_random_report_with_fact(self, fact):
        assert fact in self.facts2report_ids
        rid = random.choice(list(self.facts2report_ids[fact]))
        self.display(rid=rid)

    def get_facts_with_consecutive_repeated_words(self):
        facts = []
        for f in self.facts:
            f_ = f.lower().split()
            for i in range(len(f_)-1):
                if f_[i] == f_[i+1]:
                    facts.append(f)
                    break
        return facts
    
def integrate_reports_and_facts(preprocessed_reports_filepath, extracted_facts_filepaths, extraction_methods,
                                skip_negated_facts=True, remove_consecutive_repeated_words=True):

    assert type(preprocessed_reports_filepath) == str
    assert type(extracted_facts_filepaths) == list
    assert len(extracted_facts_filepaths) > 0
    assert all(type(fp) == str for fp in extracted_facts_filepaths)
    assert len(extracted_facts_filepaths) == len(extraction_methods)

    assert os.path.exists(preprocessed_reports_filepath)
    print(f'Loading preprocessed reports from {preprocessed_reports_filepath}...')
    preprocessed_reports = load_json(preprocessed_reports_filepath)

    sentence2facts = {}
    sentence_facts_rows = []
    for extracted_facts_filepath, extraction_method in zip(extracted_facts_filepaths, extraction_methods):
        assert os.path.exists(extracted_facts_filepath)
        print(f'Loading extracted facts from {extracted_facts_filepath}...')
        extracted_facts = load_jsonl(extracted_facts_filepath)
        for x in tqdm(extracted_facts, total=len(extracted_facts), mininterval=2):
            try:
                if 'metadata' in x:
                    s = x['metadata']['sentence']
                    fs = x['parsed_response']
                else:
                    s = x['sentence']
                    fs = x['extracted_facts']
            except KeyError:
                print(f'KeyError: {x}')
                raise
            assert s not in sentence2facts
            if skip_negated_facts:
                fs = [f for f in fs if not _NEG_REGEX.match(f)]
            if remove_consecutive_repeated_words:
                fs = [_remove_consecutive_repeated_words_from_text(f) for f in fs]
            sentence2facts[s] = fs
            sentence_facts_rows.append({
                'sentence': s,
                'facts': fs,
                'extraction_method': extraction_method
            })

    print('Integrating reports and facts...')
    report_facts_rows = [None] * len(preprocessed_reports)
    for rid, report in tqdm(enumerate(preprocessed_reports), total=len(preprocessed_reports), mininterval=2):
        dedup_facts = []
        seen_facts = set()
        for x in [report['findings'], report['impression']]:
            for s in sent_tokenize(x):
                for f in sentence2facts[s]:
                    if f not in seen_facts:
                        dedup_facts.append(f)
                        seen_facts.add(f)

        dedup_facts_ = [f.lower().split() for f in dedup_facts]
        nonredundant_facts = []
        for i in range(len(dedup_facts)):
            redundant = False
            for j in range(len(dedup_facts)):
                if i == j:
                    continue
                if _is_s1_subsequence_of_s2(dedup_facts_[i], dedup_facts_[j]):
                    if dedup_facts_[i] == dedup_facts_[j] and i > j:
                        continue
                    redundant = True
                    break
            if not redundant:
                nonredundant_facts.append(dedup_facts[i])
        fact_based_report = '. '.join(nonredundant_facts)
                    
        report_facts_rows[rid] = ({
            'report_idx': rid,
            **report,
            'facts': dedup_facts,
            'fact_based_report': fact_based_report,
        })

    return sentence_facts_rows, report_facts_rows

def integrate_reports_facts_and_metadata(
        preprocessed_reports_filepath,
        extracted_facts_filepaths, fact_extraction_methods,
        extracted_metadata_filepaths, metadata_extraction_methods,
        skip_negated_facts=True, remove_consecutive_repeated_words=True):

    assert type(preprocessed_reports_filepath) == str
    assert type(extracted_facts_filepaths) == list
    assert len(extracted_facts_filepaths) > 0
    assert all(type(fp) == str for fp in extracted_facts_filepaths)    
    assert len(extracted_facts_filepaths) == len(fact_extraction_methods)
    assert type(extracted_metadata_filepaths) == list
    assert len(extracted_metadata_filepaths) > 0
    assert all(type(fp) == str for fp in extracted_metadata_filepaths)
    assert len(extracted_metadata_filepaths) == len(metadata_extraction_methods)

    assert os.path.exists(preprocessed_reports_filepath)
    print(f'Loading preprocessed reports from {preprocessed_reports_filepath}...')
    preprocessed_reports = load_json(preprocessed_reports_filepath)

    print('Integrating sentences and facts...')
    sentence2facts = {}
    sentence_facts_rows = []
    for extracted_facts_filepath, extraction_method in zip(extracted_facts_filepaths, fact_extraction_methods):
        assert os.path.exists(extracted_facts_filepath)
        print(f'Loading extracted facts from {extracted_facts_filepath}...')
        extracted_facts = load_jsonl(extracted_facts_filepath)
        for x in tqdm(extracted_facts, total=len(extracted_facts), mininterval=2):
            try:
                try:
                    s = x['metadata']['sentence']
                    fs = x['parsed_response']
                except KeyError:
                    s = x['sentence']
                    fs = x['extracted_facts']
            except KeyError:
                print(f'KeyError: {x}')
                raise
            assert s not in sentence2facts
            if skip_negated_facts:
                fs = [f for f in fs if not _NEG_REGEX.match(f)]
            if remove_consecutive_repeated_words:
                fs = [_remove_consecutive_repeated_words_from_text(f) for f in fs]
            sentence2facts[s] = fs
            sentence_facts_rows.append({
                'sentence': s,
                'facts': fs,
                'extraction_method': extraction_method
            })

    print('Integrating facts and metadata...')
    fact2metadata = {}
    fact_metadata_rows = []
    for extracted_metadata_filepath, metadata_extraction_method in zip(extracted_metadata_filepaths, metadata_extraction_methods):
        assert os.path.exists(extracted_metadata_filepath)
        print(f'Loading extracted metadata from {extracted_metadata_filepath}...')
        extracted_metadata = load_jsonl(extracted_metadata_filepath)
        for x in tqdm(extracted_metadata, total=len(extracted_metadata), mininterval=2):
            try:
                try:
                    f = x['metadata']['fact']
                    m = x['parsed_response']
                except KeyError:
                    f = x['fact']
                    m = x['metadata']
            except KeyError:
                print(f'KeyError: {x}')
                raise
            if remove_consecutive_repeated_words:
                f = _remove_consecutive_repeated_words_from_text(f)
                for key in _FACT_METADATA_FIELDS:
                    m[key] = _remove_consecutive_repeated_words_from_text(m[key])
            if f in fact2metadata and m != fact2metadata[f]:
                print_red(f'Warning: fact "{f}" already found with different metadata. fact2metadata[f] = {fact2metadata[f]}, m = {m}')
            fact2metadata[f] = m
            fact_metadata_rows.append({
                'fact': f,
                'metadata': m,
                'extraction_method': metadata_extraction_method
            })

    print('Integrating reports, facts, and metadata...')
    report_facts_metadata_rows = [None] * len(preprocessed_reports)
    facts_without_metadata = []
    for rid, report in tqdm(enumerate(preprocessed_reports), total=len(preprocessed_reports), mininterval=2):
        dedup_facts = []
        seen_facts = set()
        for x in [report['findings'], report['impression']]:
            for s in sent_tokenize(x):
                for f in sentence2facts[s]:
                    try:
                        metadata = fact2metadata[f]
                    except KeyError:
                        facts_without_metadata.append(f)
                        print_orange(f'Warning: fact "{f}" has no metadata.')
                        continue
                    if len(metadata['detailed observation']) > 0 and len(metadata['short observation']) == 0:
                        print_orange(f'Warning: fact "{f}" has detailed observation but no short observation. metadata: {metadata}')
                    if len(metadata['detailed observation']) == 0 and len(metadata['short observation']) == 0:
                        continue # skip facts for which we don't have observation
                    if f not in seen_facts:
                        dedup_facts.append(f)
                        seen_facts.add(f)

        dedup_facts_ = [f.lower().split() for f in dedup_facts]
        nonredundant_facts = []
        for i in range(len(dedup_facts)):
            redundant = False
            for j in range(len(dedup_facts)):
                if i == j:
                    continue
                if _is_s1_subsequence_of_s2(dedup_facts_[i], dedup_facts_[j]):
                    if dedup_facts_[i] == dedup_facts_[j] and i > j:
                        continue
                    redundant = True
                    break
            if not redundant:
                nonredundant_facts.append(dedup_facts[i])
        fact_based_report = '. '.join(nonredundant_facts)
                    
        report_facts_metadata_rows[rid] = ({
            'report_idx': rid,
            **report,
            'facts': dedup_facts,
            'fact_based_report': fact_based_report,
        })

    return sentence_facts_rows, fact_metadata_rows, report_facts_metadata_rows, facts_without_metadata