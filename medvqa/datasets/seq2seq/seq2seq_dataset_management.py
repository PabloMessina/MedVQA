import numpy as np
import math
import random
import os
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.chexpert import CHEXPERT_V1_0_SMALL_DATASET_DIR
from medvqa.datasets.dataloading_utils import (
    INFINITE_DATASET_LENGTH,
    CompositeDataset,
    CompositeInfiniteDataset,
    group_indices_for_balanced_sampling,
    group_indices_into_bins_by_scores,
)
from medvqa.datasets.iuxray import get_iuxray_image_path
from medvqa.datasets.mimiccxr import get_imageId2PartPatientStudy, get_imageId2reportId, get_mimiccxr_medium_image_path
from medvqa.datasets.mimiccxr.report_utils import concatenate_sentences, concatenate_report_parts
from medvqa.datasets.nli import (
    ANLI_V1_DATASET_DIR,
    MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH,
    MULTI_NLI_DATASET_DIR,
    RADNLI_DEV_JSONL_PATH,
    RADNLI_TEST_JSONL_PATH,
    SNLI_DATASET_DIR,
)
from medvqa.datasets.interpret_cxr_challenge import (
    INTERPRET_CXR_TEST_PUBLIC_CSV_PATH,
    INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH,
)
from medvqa.utils.text_data_utils import tokenized_texts_to_lower_in_parallel, word_tokenize_texts_in_parallel
from medvqa.utils.files_utils import get_cached_jsonl_file, load_json, load_jsonl, load_pickle
from medvqa.utils.logging_utils import print_bold, print_magenta, print_orange
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAMES_WITH_TEXTUAL_GROUNDING

_ALLOWED_COMPARISONS = set([
    "no comparison",
    "new finding",
    "resolved",
    "improved",
    "worsened",
    "progressed",
    "reappeared",
    "larger",
    "smaller",
    "increase",
    "decrease",
    "position changed",
    "stable/unchanged",
    "unclear comparison",
    "other",
])

class Seq2SeqTaskNames:
    REPORT_TO_SENTENCES = 'report2sentences'
    REPORT_TO_NEGATIVE_FACTS = 'report2negative_facts'
    SENTENCE_TO_FACTS = 'sentence2facts'
    BACKGROUND_TO_FACTS = 'background2facts'
    FACT_TO_METADATA = 'fact2metadata'
    FACT_TO_METADATA_V2 = 'fact2metadata_v2'
    FACT_TO_COMPARISON = 'fact2comparison'
    SENTENCE_TO_CHEST_IMAGENOME_OBSERVATIONS = 'sentence2chestimagenome_observations'
    SENTENCE_TO_CHEST_IMAGENOME_ANATOMICAL_LOCATIONS = 'sentence2chestimagenome_anatomical_locations'
    NLI = 'nli'
    MLM = 'mlm' # masked language modeling
    FACT_CLASSIFIER_PREDICTIONS_TO_REPORT_SECTION = 'fact_classifier_predictions2report_section'
    MULTITASK = 'multitask'
    @staticmethod
    def get_all():
        return [
            Seq2SeqTaskNames.REPORT_TO_SENTENCES,
            Seq2SeqTaskNames.REPORT_TO_NEGATIVE_FACTS,
            Seq2SeqTaskNames.SENTENCE_TO_FACTS,
            Seq2SeqTaskNames.BACKGROUND_TO_FACTS,
            Seq2SeqTaskNames.FACT_TO_METADATA,
            Seq2SeqTaskNames.FACT_TO_METADATA_V2,
            Seq2SeqTaskNames.FACT_TO_COMPARISON,
            Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_OBSERVATIONS,
            Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_ANATOMICAL_LOCATIONS,
            Seq2SeqTaskNames.NLI,
            Seq2SeqTaskNames.MLM,
            Seq2SeqTaskNames.FACT_CLASSIFIER_PREDICTIONS_TO_REPORT_SECTION,
            Seq2SeqTaskNames.MULTITASK,
        ]

Task2Prefix = {
    Seq2SeqTaskNames.REPORT_TO_SENTENCES: 'R2S',
    Seq2SeqTaskNames.REPORT_TO_NEGATIVE_FACTS: 'R2NF',
    Seq2SeqTaskNames.SENTENCE_TO_FACTS: 'S2F',
    Seq2SeqTaskNames.BACKGROUND_TO_FACTS: 'B2F',
    Seq2SeqTaskNames.FACT_TO_METADATA: 'F2M',
    Seq2SeqTaskNames.FACT_TO_METADATA_V2: 'F2MV2',
    Seq2SeqTaskNames.FACT_TO_COMPARISON: 'F2C',
    Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_OBSERVATIONS: 'S2CO',
    Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_ANATOMICAL_LOCATIONS: 'S2CA',
    Seq2SeqTaskNames.NLI: 'NLI',
    Seq2SeqTaskNames.MLM: 'MLM',
}

class Seq2SeqDataset(Dataset):
    def __init__(self, indices, input_texts, output_texts, shuffle=False, infinite=False,
                 apply_uppercase_data_augmentation=False, word_tokenized_input_texts=None,
                 lowercase_indices_per_input_text=None):
        self.indices = indices
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.infinite = infinite
        self.apply_uppercase_data_augmentation = apply_uppercase_data_augmentation
        self.word_tokenized_input_texts = word_tokenized_input_texts
        self.lowercase_indices_per_input_text = lowercase_indices_per_input_text
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)
        if shuffle:
            random.shuffle(self.indices) # shuffle in place
        if apply_uppercase_data_augmentation:
            assert word_tokenized_input_texts is not None, 'word_tokenized_input_texts must be provided'
            assert lowercase_indices_per_input_text is not None, 'lowercase_indices_per_input_text must be provided'
            assert len(word_tokenized_input_texts) == len(input_texts), 'word_tokenized_input_texts must have the same length as input_texts'
            assert len(lowercase_indices_per_input_text) == len(input_texts), 'lowercase_indices_per_input_text must have the same length as input_texts'
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self.infinite:
            i = i % len(self.indices)
        idx = self.indices[i]
        if self.apply_uppercase_data_augmentation and random.random() < 0.5:
            lowercase_indices = self.lowercase_indices_per_input_text[idx]
            if len(lowercase_indices) > 0:
                n = random.randint(1, len(lowercase_indices))
                indices = random.sample(lowercase_indices, n)
                tokens = self.word_tokenized_input_texts[idx][:]
                for j in indices:
                    tokens[j] = tokens[j].upper() # uppercase
                input_text = ' '.join(tokens)
            else:
                input_text = self.input_texts[idx]
        else:
            input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]
        output = { 'input_text': input_text, 'output_text': output_text }
        return output
    
def _print_input_output_pair(input_text, output_text):
    print_bold(f'Input:')
    print_magenta(input_text, bold=True)
    print_bold(f'Output:')
    print_magenta(output_text, bold=True)

def _print_random_input_output_pair(input_texts, output_texts):
    i = random.randint(0, len(input_texts) - 1)
    _print_input_output_pair(input_texts[i], output_texts[i])

def _apply_input2paraphrases(input2paraphrases, input_texts, output_texts, verbose=True):
    if input2paraphrases is None:
        return # no paraphrases
    count = 0
    n = len(input_texts)
    for i in range(n):
        input_text = input_texts[i]
        output_text = output_texts[i]
        if input_text in input2paraphrases:
            paraphrases = input2paraphrases[input_text]
            paraphrases = set(paraphrases) # remove duplicates
            paraphrases.discard(input_text) # remove original input
            for p in paraphrases:
                input_texts.append(p)
                output_texts.append(output_text) # keep same output
                count += 1
    if verbose:
        print(f'Added {count} paraphrased inputs (total {len(input_texts)})')
        # print random paraphrased input/output pair
        i = random.randint(n, len(input_texts) - 1)
        _print_input_output_pair(input_texts[i], output_texts[i])

def _prepare_fact_to_comparison_data(
        integrated_facts_metadata_jsonl_filepath,
        input_output_jsonl_filepaths,
        input2paraphrases=None,
        apply_task_prefix=False,
        medical_sentences=None,
        verbose=True,
    ):
    assert integrated_facts_metadata_jsonl_filepath is not None, 'integrated_facts_metadata_jsonl_filepath must be provided'
    assert input_output_jsonl_filepaths is not None, 'input_output_jsonl_filepaths must be provided'

    # 1) input/output pairs from integrated_facts_metadata_jsonl_filepath
    integrated_facts_metadata = load_jsonl(integrated_facts_metadata_jsonl_filepath)
    input_texts = []
    output_texts = []
    for row in integrated_facts_metadata:
        metadata = row['metadata']
        comp = metadata['comparison status']
        psc = metadata['prev_study_comparison?']
        em = row['extraction_method']
        is_psc_invalid = psc not in ('yes', 'no')
        is_comp_inconsistent = (psc == 'yes') != (comp != '')
        if is_psc_invalid or is_comp_inconsistent:
            continue
        if 'gpt' in em:
            if comp == '':
                input_texts.append(row['fact'])
                output_texts.append('no comparison')
            elif comp in _ALLOWED_COMPARISONS:
                input_texts.append(row['fact'])
                output_texts.append(comp)
    print(f'Loaded {len(input_texts)} input/output pairs from {integrated_facts_metadata_jsonl_filepath}')
    
    # 2) input/output pairs from input_output_jsonl_filepaths
    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            fact = input_output['metadata']['sentence']
            input_text = fact
            output_text = input_output['parsed_response']
            input_texts.append(input_text)
            output_texts.append(output_text)

    # 3) add to medical_sentences
    if medical_sentences is not None:
        medical_sentences.update(input_texts)

    # 4) add paraphrased inputs
    _apply_input2paraphrases(input2paraphrases, input_texts, output_texts, verbose=verbose)

    # 5) apply task prefix
    if apply_task_prefix:
        input_texts = [f'F2C: {x}' for x in input_texts]

    if verbose:
        # Print random example
        print_bold('Input/output example:')
        _print_random_input_output_pair(input_texts, output_texts)

    return input_texts, output_texts

def _prepare_sentence_to_chest_imagenome_observations_data(
        chest_imagenome_phrases2observations_filepath,
        input_output_jsonl_filepaths,
        input2paraphrases=None,
        apply_task_prefix=False,
        medical_sentences=None,
        verbose=True):
    
    assert chest_imagenome_phrases2observations_filepath is not None, 'chest_imagenome_phrases2observations_filepath must be provided'
    assert input_output_jsonl_filepaths is not None, 'input_output_jsonl_filepaths must be provided'

    # 1) input/output pairs from chest_imagenome_phrases2observations_filepath
    chest_imagenome_phrases2observations = load_pickle(chest_imagenome_phrases2observations_filepath)
    phrases = chest_imagenome_phrases2observations['phrases']
    labels = chest_imagenome_phrases2observations['observation_labels']
    label_names = chest_imagenome_phrases2observations['observation_names']
    # Remove nlp labels and prefixes
    _idxs = [i for i, x in enumerate(label_names) if not x.startswith('nlp|')]
    assert len(_idxs) + 2 == len(label_names) # 2 nlp labels
    labels = labels[:, _idxs]
    label_names = [label_names[i][label_names[i].index('|') + 1:] for i in _idxs]

    input_texts = []
    output_texts = []
    n, m = labels.shape
    assert len(phrases) == n
    assert len(label_names) == m
    for i in tqdm(range(n), total=n, mininterval=2):
        input_texts.append(phrases[i])
        output_text = json.dumps([label_names[j] for j in range(m) if labels[i, j] == 1])
        output_texts.append(output_text)
    if verbose:
        print(f'Loaded {len(input_texts)} input/output pairs from {chest_imagenome_phrases2observations_filepath}')
        print('Examples:')
        for _ in range(3):
            _print_random_input_output_pair(input_texts, output_texts)
    
    # 2) input/output pairs from input_output_jsonl_filepaths
    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            sentence = input_output['metadata']['query']
            input_text = sentence
            output_text = json.dumps(input_output['parsed_response'])
            input_texts.append(input_text)
            output_texts.append(output_text)

    # 3) add to medical_sentences
    if medical_sentences is not None:
        medical_sentences.update(input_texts)
    
    # 4) add paraphrased inputs
    _apply_input2paraphrases(input2paraphrases, input_texts, output_texts, verbose=verbose)

    # 5) apply task prefix
    if apply_task_prefix:
        input_texts = [f'S2CO: {x}' for x in input_texts]
    
    return input_texts, output_texts, label_names

def _prepare_sentence_to_chest_imagenome_anatlocs_data(
        chest_imagenome_phrases2anatlocs_filepath,
        input_output_jsonl_filepaths,
        input2paraphrases=None,
        apply_task_prefix=False,
        medical_sentences=None,
        verbose=True):
    
    assert chest_imagenome_phrases2anatlocs_filepath is not None, 'chest_imagenome_phrases2anatlocs_filepath must be provided'
    assert input_output_jsonl_filepaths is not None, 'input_output_jsonl_filepaths must be provided'

    # 1) input/output pairs from chest_imagenome_phrases2anatlocs_filepath
    chest_imagenome_phrases2anatlocs = load_pickle(chest_imagenome_phrases2anatlocs_filepath)
    phrases = chest_imagenome_phrases2anatlocs['phrases']
    labels = chest_imagenome_phrases2anatlocs['anatomy_labels']
    label_names = chest_imagenome_phrases2anatlocs['anatomy_names']
    # Remove 'unknown' label
    _idxs = [i for i, x in enumerate(label_names) if x != 'unknown']
    assert len(_idxs) + 1 == len(label_names) # 1 unknown label
    labels = labels[:, _idxs]
    label_names = [label_names[i] for i in _idxs]
    assert set(label_names) == set(CHEST_IMAGENOME_BBOX_NAMES_WITH_TEXTUAL_GROUNDING), \
        (f'Expected {set(CHEST_IMAGENOME_BBOX_NAMES_WITH_TEXTUAL_GROUNDING)} but got {set(label_names)}\n'
            f'Difference (->): {set(label_names) - set(CHEST_IMAGENOME_BBOX_NAMES_WITH_TEXTUAL_GROUNDING)}\n'
            f'Difference (<-): {set(CHEST_IMAGENOME_BBOX_NAMES_WITH_TEXTUAL_GROUNDING) - set(label_names)}')

    input_texts = []
    output_texts = []
    n, m = labels.shape
    assert len(phrases) == n
    assert len(label_names) == m
    for i in tqdm(range(n), total=n, mininterval=2):
        input_texts.append(phrases[i])
        output_text = json.dumps([label_names[j] for j in range(m) if labels[i, j] == 1])
        output_texts.append(output_text)
    if verbose:
        print(f'Loaded {len(input_texts)} input/output pairs from {chest_imagenome_phrases2anatlocs_filepath}')
        print('Examples:')
        for _ in range(3):
            _print_random_input_output_pair(input_texts, output_texts)

    # 2) input/output pairs from input_output_jsonl_filepaths
    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            sentence = input_output['metadata']['query']
            input_text = sentence
            output_text = json.dumps(input_output['parsed_response'])
            input_texts.append(input_text)
            output_texts.append(output_text)

    # 3) add to medical_sentences
    if medical_sentences is not None:
        medical_sentences.update(input_texts)
    
    # 4) add paraphrased inputs
    _apply_input2paraphrases(input2paraphrases, input_texts, output_texts, verbose=verbose)

    # 5) apply task prefix
    if apply_task_prefix:
        input_texts = [f'S2CA: {x}' for x in input_texts]

    return input_texts, output_texts, label_names

def _prepare_reports_to_sentences_data(input_output_jsonl_filepaths, apply_task_prefix=False, verbose=True):
    input_texts = []
    output_texts = []
    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            report = input_output['metadata']['report']
            input_text = '\n'.join([report['findings'], report['impression']])
            output_text = json.dumps([f'{s} {"#pos" if p else "#neg"}' for s, p in input_output['parsed_response']])
            input_texts.append(input_text)
            output_texts.append(output_text)
    if apply_task_prefix:
        input_texts = [f'R2S: {x}' for x in input_texts]
    if verbose:
        # Print random example
        print_bold('Input/output example:')
        _print_random_input_output_pair(input_texts, output_texts)
    return input_texts, output_texts

class _RandomizedReportToNegativeFactsDataset(Dataset):
    def __init__(self, abnormal_sentences, normal_sentence_to_negative_facts, apply_task_prefix=False,
                 max_num_sentences=16, dataset_size=None):
        self.abnormal_sentences = abnormal_sentences
        self.normal_sentences = list(normal_sentence_to_negative_facts.keys())
        self.negative_facts = list(normal_sentence_to_negative_facts.values())
        self.normal_idxs = list(range(len(self.normal_sentences)))
        self.apply_task_prefix = apply_task_prefix
        self.max_num_sentences = max_num_sentences
        if dataset_size is not None:
            self._len = dataset_size
        else:
            self._len = INFINITE_DATASET_LENGTH
    
    def __len__(self):
        return self._len

    def __getitem__(self, i, verbose=False):
        mode = random.randint(1, 3) # 1: fully abnormal, 2: fully normal, 3: mixed
        if verbose:
            print(f'mode: {mode}')
        if mode == 1: # fully abnormal
            num_sentences = random.randint(1, self.max_num_sentences)
            sentences = random.sample(self.abnormal_sentences, num_sentences)
            input_text = concatenate_sentences(sentences)
            output_text = "{}" # empty object
        elif mode == 2: # fully normal
            num_sentences = random.randint(1, self.max_num_sentences)
            idxs = random.sample(self.normal_idxs, num_sentences)
            sentences = [self.normal_sentences[i] for i in idxs]
            negative_facts = [self.negative_facts[i] if len(self.negative_facts[i]) <= 2 else random.sample(self.negative_facts[i], 2) for i in idxs]
            input_text = concatenate_sentences(sentences)
            output_object = {s: nf for s, nf in zip(sentences, negative_facts)}
            output_text = json.dumps(output_object)
        elif mode == 3: # mixed
            num_sentences = random.randint(2, self.max_num_sentences)
            num_abnormal_sentences = random.randint(1, num_sentences - 1)
            num_normal_sentences = num_sentences - num_abnormal_sentences
            abnormal_sentences = random.sample(self.abnormal_sentences, num_abnormal_sentences)
            idxs = random.sample(self.normal_idxs, num_normal_sentences)
            normal_sentences = [self.normal_sentences[i] for i in idxs]
            normal_negative_facts = [self.negative_facts[i] if len(self.negative_facts[i]) <= 2 else random.sample(self.negative_facts[i], 2) for i in idxs]
            sentences = normal_sentences + abnormal_sentences
            rank = list(range(len(sentences)))
            random.shuffle(rank)
            normal_idxs = list(range(len(normal_sentences)))
            normal_idxs.sort(key=lambda i: rank[i])
            sentences_ = [None] * len(sentences)
            for i, r in enumerate(rank):
                sentences_[r] = sentences[i]
            input_text = concatenate_sentences(sentences_)
            output_object = {normal_sentences[i]: normal_negative_facts[i] for i in normal_idxs}
            output_text = json.dumps(output_object)
        else:
            raise ValueError(f'Unknown mode {mode}')
        if self.apply_task_prefix:
            input_text = f'R2NF: {input_text}'
        return { 'input_text': input_text, 'output_text': output_text }
    
class _ShuffledReportToNegativeFactsDataset(Dataset):
    def __init__(self, indices, input_sentences, input_texts, output_texts, infinite=False, shuffle=False,
                 apply_task_prefix=False):
        assert len(input_sentences) == len(input_texts)
        assert len(input_sentences) == len(output_texts)
        self.input_sentences = input_sentences
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.indices = indices
        self.apply_task_prefix = apply_task_prefix
        self.n = len(self.indices)
        self.infinite = infinite
        self._len = INFINITE_DATASET_LENGTH if infinite else self.n
        if shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self.infinite:
            i = i % self.n
        idx = self.indices[i]
        if random.random() < 0.7: # 70% chance of randomizing the order of sentences
            sentences = self.input_sentences[idx][:] # copy
            random.shuffle(sentences)
            input_text = concatenate_sentences(sentences)
        else:
            input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]
        if self.apply_task_prefix:
            input_text = f'R2NF: {input_text}'
        return { 'input_text': input_text, 'output_text': output_text }

def _prepare_reports_to_negative_facts_data(input_output_jsonl_filepaths, apply_task_prefix=False, verbose=True,
                                            val_size=500, filter_for_t5=False):
    from nltk.tokenize import sent_tokenize
    input_sentences = []
    abnormal_sentences = set()
    normal_sentence_to_negative_facts = dict()
    output_texts = []
    input_texts = []
    if filter_for_t5:
        skip_count = 0
    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in tqdm(input_output_jsonl, mininterval=2):
            input_query = input_output['metadata']['query']
            sentences = sent_tokenize(input_query)
            ruled_out_abnormalities = input_output['parsed_response']['ruled_out_abnormalities']
            output_count = 0
            for k, v in ruled_out_abnormalities.items():
                output_count += len(v)
                if len(v) > 0:
                    abnormal_sentences.update(v)
                    try:
                        counts = normal_sentence_to_negative_facts[k]
                    except KeyError:
                        counts = dict()
                        normal_sentence_to_negative_facts[k] = counts
                    for x in v:
                        counts[x] = counts.get(x, 0) + 1
                else:
                    abnormal_sentences.add(k)
            if output_count == 0:
                abnormal_sentences.update(sentences) # all sentences are abnormal
            if filter_for_t5:
                estimated_num_tokens = len(input_query) / 3.2
                if estimated_num_tokens > 512:
                    skip_count += 1
                    continue
            input_texts.append(input_query)
            input_sentences.append(sentences)
            output_text = json.dumps(ruled_out_abnormalities)
            output_texts.append(output_text)
    abnormal_sentences = list(abnormal_sentences)
    
    for k, v in normal_sentence_to_negative_facts.items():
        # Keep the two most common negative facts
        pairs = [(nf, c) for nf, c in v.items()]
        pairs.sort(key=lambda x: -x[1])
        normal_sentence_to_negative_facts[k] = [nf for nf, _ in pairs[:2]]

    print(f'Number of abnormal sentences: {len(abnormal_sentences)}')
    print(f'Number of normal sentences: {len(normal_sentence_to_negative_facts)}')
    if filter_for_t5:
        print(f'Skipped {skip_count} reports with estimated token count > 512')
    
    # Create datasets

    if filter_for_t5:
        max_num_sentences = 5
    else:
        max_num_sentences = 16
    
    # Randomized dataset up to 16 sentences
    train_dataset_1 = _RandomizedReportToNegativeFactsDataset(abnormal_sentences, normal_sentence_to_negative_facts,
                                                              apply_task_prefix=apply_task_prefix, max_num_sentences=max_num_sentences)
    val_dataset_1 = _RandomizedReportToNegativeFactsDataset(abnormal_sentences, normal_sentence_to_negative_facts,
                                                            apply_task_prefix=apply_task_prefix, max_num_sentences=max_num_sentences,
                                                            dataset_size=val_size)
    
    # Randomized dataset up to 2 sentences
    train_dataset_2 = _RandomizedReportToNegativeFactsDataset(abnormal_sentences, normal_sentence_to_negative_facts,
                                                              apply_task_prefix=apply_task_prefix, max_num_sentences=2)
    val_dataset_2 = _RandomizedReportToNegativeFactsDataset(abnormal_sentences, normal_sentence_to_negative_facts,
                                                            apply_task_prefix=apply_task_prefix, max_num_sentences=2,
                                                            dataset_size=val_size)
    
    # Shuffled dataset
    indices = list(range(len(input_texts)))
    indices.sort(key=lambda i: len(input_texts[i])) # sort by length        
    # Choose val_size random indices uniformly distributed across the dataset
    val_indices = np.linspace(0, len(indices) - 1, val_size, dtype=int)
    val_indices_set = set(val_indices)
    train_indices = [i for i in indices if i not in val_indices_set]

    train_dataset_3 = _ShuffledReportToNegativeFactsDataset(train_indices, input_sentences, input_texts, output_texts,
                                                            shuffle=True, infinite=True,
                                                            apply_task_prefix=apply_task_prefix)
    val_dataset_3 = _ShuffledReportToNegativeFactsDataset(val_indices, input_sentences, input_texts, output_texts,
                                                        apply_task_prefix=apply_task_prefix)

    merged_train_dataset = CompositeInfiniteDataset([train_dataset_1, train_dataset_2, train_dataset_3], [0.2, 0.1, 0.7])
    merged_val_dataset = CompositeDataset([val_dataset_1, val_dataset_2, val_dataset_3])

    print(f'len(train_dataset_1): {len(train_dataset_1)}')
    print(f'len(train_dataset_2): {len(train_dataset_2)}')
    print(f'len(train_dataset_3): {len(train_dataset_3)}')
    print(f'len(val_dataset_1): {len(val_dataset_1)}')
    print(f'len(val_dataset_2): {len(val_dataset_2)}')
    print(f'len(val_dataset_3): {len(val_dataset_3)}')
    print(f'len(merged_train_dataset): {len(merged_train_dataset)}')
    print(f'len(merged_val_dataset): {len(merged_val_dataset)}')
    
    if verbose:
        # Print random example
        print_bold('----- Input/output examples from train_dataset_1:')
        for _ in range(5):
            pair = train_dataset_1.__getitem__(random.randint(0, len(train_dataset_1) - 1), verbose=True)
            _print_input_output_pair(pair['input_text'], pair['output_text'])
        print_bold('----- Input/output examples from train_dataset_2:')
        for _ in range(10):
            pair = train_dataset_2.__getitem__(random.randint(0, len(train_dataset_2) - 1), verbose=True)
            _print_input_output_pair(pair['input_text'], pair['output_text'])
        print_bold('----- Input/output example from train_dataset_3:')
        for _ in range(2):
            pair = train_dataset_3[random.randint(0, len(train_dataset_3) - 1)]
            _print_input_output_pair(pair['input_text'], pair['output_text'])
        print_bold('----- Input/output example from val_dataset_1:')
        pair = val_dataset_1[random.randint(0, len(val_dataset_1) - 1)]
        _print_input_output_pair(pair['input_text'], pair['output_text'])
        print_bold('----- Input/output example from val_dataset_2:')
        pair = val_dataset_2[random.randint(0, len(val_dataset_2) - 1)]
        _print_input_output_pair(pair['input_text'], pair['output_text'])
        print_bold('----- Input/output example from val_dataset_3:')
        pair = val_dataset_3[random.randint(0, len(val_dataset_3) - 1)]
        _print_input_output_pair(pair['input_text'], pair['output_text'])

    return merged_train_dataset, merged_val_dataset

def _prepare_sentence_to_facts_data(input_output_jsonl_filepaths, apply_task_prefix=False, verbose=True,
                                    collect_input_output_for_nli=False, medical_sentences=None,
                                    concatenate_pairs=False):
    assert input_output_jsonl_filepaths is not None, 'input_output_jsonl_filepaths must be provided'
    sentence2facts = dict()
    if collect_input_output_for_nli:
        input_output_for_nli = []
    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            try:
                sentence = input_output['metadata']['query']
            except KeyError:
                sentence = input_output['metadata']['sentence'] # backward compatibility
            output = input_output['parsed_response']
            if isinstance(output, dict):
                assert 'reason' in output
                assert 'facts' in output
                facts = output['facts']
            else:
                assert isinstance(output, list), f'Expected list or dict but got {type(output)}'
                facts = output
            if collect_input_output_for_nli:
                input_output_for_nli.append((sentence, facts))
            sentence2facts[sentence] = facts
            if medical_sentences is not None:
                medical_sentences.add(sentence)
                for f in facts:
                    medical_sentences.add(f)
    input_texts = []
    output_texts = []
    sentences, facts_list = zip(*sentence2facts.items())
    for sentence, facts in zip(sentences, facts_list):
        input_texts.append(sentence)
        output_texts.append(json.dumps(facts))
    if concatenate_pairs:
        print_orange('Concatenating pairs of sentences for data augmentation...', bold=True)
        # Concatenate pairs of sentences and facts as a data augmentation technique:
        # e.g. "sentence1. sentence2" -> facts1 + facts2
        for i in range(len(sentences)):
            while True:
                j = random.randint(0, len(sentences) - 1)
                if i != j:
                    break
            if sentences[i][-1] == '.':
                input_text = f'{sentences[i]} {sentences[j]}'
            else:
                input_text = f'{sentences[i]}. {sentences[j]}'
            input_texts.append(input_text)
            output_texts.append(json.dumps(facts_list[i] + facts_list[j]))

    if apply_task_prefix:
        input_texts = [f'S2F: {x}' for x in input_texts]
    print(f'Number of sentence2facts pairs: {len(input_texts)}')
    if verbose:
        # Print random example
        print_bold('Input/output example:')
        _print_random_input_output_pair(input_texts, output_texts)
    if collect_input_output_for_nli:
        return input_texts, output_texts, input_output_for_nli
    return input_texts, output_texts

def _prepare_background_to_facts_data(input_output_jsonl_filepaths, apply_task_prefix=False, verbose=True):
    input_texts = []
    output_texts = []
    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            background = input_output['metadata']['background']
            input_text = background
            output_text = json.dumps(input_output['parsed_response'])
            input_texts.append(input_text)
            output_texts.append(output_text)
    if apply_task_prefix:
        input_texts = [f'B2F: {x}' for x in input_texts]
    if verbose:
        # Print random example
        print_bold('Input/output example:')
        _print_random_input_output_pair(input_texts, output_texts)
    return input_texts, output_texts

def _prepare_fact_to_metadata_data(input_output_jsonl_filepaths, apply_task_prefix=False, medical_sentences=None, verbose=True):
    assert input_output_jsonl_filepaths is not None, 'input_output_jsonl_filepaths must be provided'
    input_texts = []
    output_texts = []
    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            fact = input_output['metadata']['fact']
            metadata = input_output['parsed_response']
            input_text = fact
            output_text = json.dumps(metadata)
            input_texts.append(input_text)
            output_texts.append(output_text)
            if medical_sentences is not None:
                medical_sentences.add(input_text)
                al = metadata['anatomical location']
                do = metadata['detailed observation']
                so = metadata['short observation']
                if al: medical_sentences.add(al)
                if do: medical_sentences.add(do)
                if so: medical_sentences.add(so)
                
    if apply_task_prefix:
        input_texts = [f'F2M: {x}' for x in input_texts]
    if verbose:
        # Print random example
        print_bold('Input/output example:')
        _print_random_input_output_pair(input_texts, output_texts)
    return input_texts, output_texts

def _prepare_fact_to_metadata_v2_data(input_output_jsonl_filepaths, apply_task_prefix=False, medical_sentences=None, verbose=True):
    assert input_output_jsonl_filepaths is not None, 'input_output_jsonl_filepaths must be provided'
    input_texts = []
    output_texts = []
    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            fact = input_output['metadata']['query']
            metadata = input_output['parsed_response']
            input_text = fact
            output_text = json.dumps(metadata)
            input_texts.append(input_text)
            output_texts.append(output_text)
            if medical_sentences is not None:
                medical_sentences.add(input_text)
                al = metadata['anatomical_location']
                go = metadata['general_observation']
                if al: medical_sentences.add(al)
                if go: medical_sentences.add(go)
                
    if apply_task_prefix:
        input_texts = [f'F2M: {x}' for x in input_texts]
    if verbose:
        # Print random example
        print_bold('Input/output example:')
        _print_random_input_output_pair(input_texts, output_texts)
    return input_texts, output_texts

_ALLOWED_NLI_LABELS = [
    'entailment',
    'neutral',
    'contradiction',
]

_NLI_SHORT2LONG = {
    'e': 'entailment',
    'n': 'neutral',
    'c': 'contradiction',
}

def _get_nli_input_output_1(premise, hypothesis, label):
    assert label in _ALLOWED_NLI_LABELS, f'Unknown label {label}'
    # return f'NLI1: {premise} #Hypothesis: {hypothesis}', 'Most likely: ' + label
    return f'NLI1: {premise} #H: {hypothesis}', label

def _get_nli_input_output_2(premise, hypothesis, label):
    assert label in _ALLOWED_NLI_LABELS, f'Unknown label {label}'
    # return f'NLI2: {premise} #Generate {label}', hypothesis
    return f'NLI2: {premise} #G {label}', hypothesis

def load_radnli_dev_data(nli1_only=False, verbose=True, medical_sentences=None, whole_sentences=None):
    rows = load_jsonl(RADNLI_DEV_JSONL_PATH)
    if verbose:
        print(f'Number of RadNLI dev samples: {len(rows)}')
    input_texts = []
    output_texts = []
    for x in rows:
        p, h, l = x['sentence1'], x['sentence2'], x['gold_label']
        if medical_sentences is not None:
            medical_sentences.add(p)
            medical_sentences.add(h)
        if whole_sentences is not None:
            whole_sentences.add(p)
            whole_sentences.add(h)
        input_text, output_text = _get_nli_input_output_1(p, h, l)
        input_texts.append(input_text)
        output_texts.append(output_text)

        if not nli1_only:
            input_text, output_text = _get_nli_input_output_2(p, h, l)
            input_texts.append(input_text)
            output_texts.append(output_text)
    
    return input_texts, output_texts

def load_radnli_test_data(nli1_only=False, verbose=True, medical_sentences=None, whole_sentences=None):
    rows = load_jsonl(RADNLI_TEST_JSONL_PATH)
    if verbose:
        print(f'Number of RadNLI test samples: {len(rows)}')
    input_texts = []
    output_texts = []
    for x in rows:
        p, h, l = x['sentence1'], x['sentence2'], x['gold_label']
        if medical_sentences is not None:
            medical_sentences.add(p)
            medical_sentences.add(h)
        if whole_sentences is not None:
            whole_sentences.add(p)
            whole_sentences.add(h)
        input_text, output_text = _get_nli_input_output_1(p, h, l)
        input_texts.append(input_text)
        output_texts.append(output_text)

        if not nli1_only:
            input_text, output_text = _get_nli_input_output_2(p, h, l)
            input_texts.append(input_text)
            output_texts.append(output_text)
    
    return input_texts, output_texts

def load_ms_cxr_t_temporal_sentence_similarity_v1_data(nli1_only=False, verbose=True, medical_sentences=None,
                                                       whole_sentences=None):
    df = pd.read_csv(MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH)
    if verbose:
        print(f'Number of MS_CXR_T samples: {len(df)}')
    input_texts = []
    output_texts = []
    for p, h, l in zip(df.sentence_1, df.sentence_2, df.category):
        if l == 'paraphrase':
            l = 'entailment'
        elif l == 'contradiction':
            pass
        else:
            raise ValueError(f'Unknown label {l}')
        if medical_sentences is not None:
            medical_sentences.add(p)
            medical_sentences.add(h)
        if whole_sentences is not None:
            whole_sentences.add(p)
            whole_sentences.add(h)
        input_text, output_text = _get_nli_input_output_1(p, h, l)
        input_texts.append(input_text)
        output_texts.append(output_text)

        if not nli1_only:
            input_text, output_text = _get_nli_input_output_2(p, h, l)
            input_texts.append(input_text)
            output_texts.append(output_text)
    
    return input_texts, output_texts

def load_gpt4_nli_examples_filepaths(filepaths, nli1_only=False, verbose=True, medical_sentences=None, whole_sentences=None):
    input_texts = []
    output_texts = []
    for filepath in filepaths:
        rows = load_jsonl(filepath)
        if verbose:
            print(f'Loaded {len(rows)} rows from {filepath}')
        for row in rows:
            query = row['metadata']['query']
            p_idx = query.index('#P: ')
            h_idx = query.index('| #H: ')
            p = query[p_idx+4:h_idx].strip()
            h = query[h_idx+6:].strip()
            if medical_sentences is not None:
                medical_sentences.add(p)
                medical_sentences.add(h)
            if whole_sentences is not None:
                whole_sentences.add(p)
                whole_sentences.add(h)
            l = row['parsed_response']
            input_text, output_text = _get_nli_input_output_1(p, h, l)
            input_texts.append(input_text)
            output_texts.append(output_text)
            if not nli1_only:
                input_text, output_text = _get_nli_input_output_2(p, h, l)
                input_texts.append(input_text)
                output_texts.append(output_text)
    return input_texts, output_texts

def load_report_nli_examples_filepaths(filepaths, nli1_only=False, verbose=True, medical_sentences=None,
                                        whole_sentences=None, splittable_sentences=None):
    input_texts = []
    output_texts = []
    labels = []
    if type(filepaths) is str:
        filepaths = [filepaths] # make it a list
    for input_output_jsonl_filepath in filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            query = input_output['metadata']['query']
            report_start_idx = query.index("#F ") + 3
            report_end_idx = query.index(" | #H ")
            p = query[report_start_idx:report_end_idx]
            h = query[report_end_idx+6:]
            response = input_output['parsed_response']
            if type(response) is str:
                l = _REPORT_NLI_LABEL_TO_STANDARD_NLI_LABEL[response]
            else:
                assert type(response) is dict
                assert 'reason' in response
                assert 'label' in response
                l = _REPORT_NLI_LABEL_TO_STANDARD_NLI_LABEL[response['label']]

            if medical_sentences is not None:
                medical_sentences.add(p)
                medical_sentences.add(h)
            if splittable_sentences is not None:
                splittable_sentences.add(p)
            if whole_sentences is not None:
                whole_sentences.add(h)
            
            input_text, output_text = _get_nli_input_output_1(p, h, l)
            input_texts.append(input_text)
            output_texts.append(output_text)
            labels.append(l)

            if not nli1_only:
                input_text, output_text = _get_nli_input_output_2(p, h, l)
                input_texts.append(input_text)
                output_texts.append(output_text)
                labels.append(l)

    return input_texts, output_texts, labels

def load_raw_report_nli_examples_filepaths(nli_examples_filepaths, integrated_report_facts_jsonl_filepath, nli1_only=False, verbose=True,
                                           medical_sentences=None, whole_sentences=None, splittable_sentences=None):
    
    if verbose:
        print('----')
        print(f'Loading raw report NLI examples from {nli_examples_filepaths}...')
    
    integrated_report_facts = get_cached_jsonl_file(integrated_report_facts_jsonl_filepath)
    input_texts = []
    output_texts = []
    labels = []

    def _build_report(idx):
        background = integrated_report_facts[idx]['background']
        findings = integrated_report_facts[idx]['findings']
        impression = integrated_report_facts[idx]['impression']
        return concatenate_report_parts(background, findings, impression)
    
    report2idx = { _build_report(idx): idx for idx in range(len(integrated_report_facts)) }
    if verbose:
        print(f'len(report2idx): {len(report2idx)}')

    if type(nli_examples_filepaths) is str:
        nli_examples_filepaths = [nli_examples_filepaths] # make it a list

    debug = True

    for input_output_jsonl_filepath in nli_examples_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        for input_output in input_output_jsonl:
            query = input_output['metadata']['query']
            report_start_idx = query.index("#R: ") + 4
            report_end_idx = query.index(" | #H: ")
            p = query[report_start_idx:report_end_idx] # raw report
            if verbose and debug:
                print(f'Raw report: {p}')
            p = integrated_report_facts[report2idx[p]]['fact_based_report'] # use fact-based report instead
            if verbose and debug:
                print(f'Fact-based report: {p}')
                debug = False # print only once
            h = query[report_end_idx+7:]
            response = input_output['parsed_response']
            if type(response) is str:
                l = _REPORT_NLI_LABEL_TO_STANDARD_NLI_LABEL[response]
            else:
                assert type(response) is dict
                assert 'reason' in response
                assert 'label' in response
                l = _REPORT_NLI_LABEL_TO_STANDARD_NLI_LABEL[response['label']]

            if medical_sentences is not None:
                medical_sentences.add(p)
                medical_sentences.add(h)
            if splittable_sentences is not None:
                splittable_sentences.add(p)
            if whole_sentences is not None:
                whole_sentences.add(h)
            
            input_text, output_text = _get_nli_input_output_1(p, h, l)
            input_texts.append(input_text)
            output_texts.append(output_text)
            labels.append(l)

            if not nli1_only:
                input_text, output_text = _get_nli_input_output_2(p, h, l)
                input_texts.append(input_text)
                output_texts.append(output_text)
                labels.append(l)

    return input_texts, output_texts, labels

_REPORT_NLI_LABEL_TO_STANDARD_NLI_LABEL = {
    "definitely true": "entailment",
    "likely true": "entailment",
    "unknown": "neutral",
    "likely false": "contradiction",
    "definitely false": "contradiction",
}

def _prepare_nli_data(integrated_nli_jsonl_filepath, s2f_input_output_for_nli, use_anli=False, use_multinli=False,
                      use_snli=False, use_report_nli=False,
                      raw_report_nli_input_output_train_jsonl_filepaths=None,
                      report_nli_input_output_train_jsonl_filepaths=None,
                      report_nli_input_output_val_jsonl_filepaths=None,
                      use_report_nli_entailment_dataset=False, integrated_report_facts_jsonl_filepath=None,
                      use_report_nli_paraphrases_dataset=False, input_to_paraphrases=None,
                      verbose=True, medical_sentences=None, general_sentences=None, nli1_only_on_train=False, nli1_only_on_val=False,
                      whole_sentences=None, splittable_sentences=None):

    assert integrated_nli_jsonl_filepath, 'integrated_nli_jsonl_filepath must be provided'
    if nli1_only_on_train:
        assert nli1_only_on_val, 'nli1_only_on_val must be True if nli1_only_on_train is True'

    input_texts = []
    output_texts = []
    labels = []
    sources = []
    source2type = {}
    
    if verbose:
        print('----')
        print(f'Loading integrated NLI from {integrated_nli_jsonl_filepath}...')
    rows = load_jsonl(integrated_nli_jsonl_filepath)
    source2id = {}
    id2source = {}
    for row in rows:
        source = row['source']
        if source not in source2id:
            source2id[source] = len(source2id)
            id2source[source2id[source]] = source
            source2type[source] = 'medical'
    if verbose:
        print(f'Number of samples: {len(rows)}')
        print(f'Number of sources: {len(source2id)}')

    for row in rows:
        p, h, l, s = row['premise'], row['hypothesis'], row['label'], row['source']
        s = source2id[s]

        if medical_sentences is not None:
            medical_sentences.add(p)
            medical_sentences.add(h)
        if whole_sentences is not None:
            whole_sentences.add(p)
            whole_sentences.add(h)
        
        input_text, output_text = _get_nli_input_output_1(p, h, l)
        input_texts.append(input_text)
        output_texts.append(output_text)
        labels.append(l)
        sources.append(s)
        
        if not nli1_only_on_train:
            input_text, output_text = _get_nli_input_output_2(p, h, l)
            input_texts.append(input_text)
            output_texts.append(output_text)
            labels.append(l)
            sources.append(s)

    source_id = len(source2id) # to add a new source

    if s2f_input_output_for_nli is not None:
        if verbose:
            print('----')
            print_bold('Loading sentence2facts input/output pairs for NLI...')
        source2id['s2f'] = source_id
        id2source[source_id] = 's2f'
        source2type['s2f'] = 'medical'
        l = 'entailment'
        len_bef = len(input_texts)
        for (s, fs) in s2f_input_output_for_nli: # s: sentence, fs: facts
            if medical_sentences is not None:
                medical_sentences.add(s)
                for f in fs:
                    medical_sentences.add(f)
            if whole_sentences is not None:
                whole_sentences.add(s)
                for f in fs:
                    whole_sentences.add(f)
            for f in fs:
                if s != f:
                    p, h = s, f
                    
                    input_text, output_text = _get_nli_input_output_1(p, h, l)
                    input_texts.append(input_text)
                    output_texts.append(output_text)
                    labels.append(l)
                    sources.append(source_id)

                    if not nli1_only_on_train:
                        input_text, output_text = _get_nli_input_output_2(p, h, l)
                        input_texts.append(input_text)
                        output_texts.append(output_text)
                        labels.append(l)
                        sources.append(source_id)

        if verbose:
            print(f'Number of entailment samples added: {len(input_texts) - len_bef}')

        source_id += 1 # to add a new source

    if use_anli or use_multinli or use_snli:
        # General domain datasets
        if verbose:
            print('----')
            print_bold('Loading general domain datasets...')
        if use_anli:
            if verbose:
                print('Loading ANLI...')
            for r in ('R1', 'R2', 'R3'):
                anli_train_rows = load_jsonl(os.path.join(ANLI_V1_DATASET_DIR, r, 'train.jsonl'))
                anli_dev_rows = load_jsonl(os.path.join(ANLI_V1_DATASET_DIR, r, 'dev.jsonl'))
                anli_test_rows = load_jsonl(os.path.join(ANLI_V1_DATASET_DIR, r, 'test.jsonl'))
                if verbose:
                    print(f'Number of ANLI {r} samples: {len(anli_train_rows) + len(anli_dev_rows) + len(anli_test_rows)}')
                for rows in (anli_train_rows, anli_dev_rows, anli_test_rows):
                    for row in rows:
                        p, h, l = row['context'], row['hypothesis'], row['label']
                        l = _NLI_SHORT2LONG[l]

                        if general_sentences is not None:
                            general_sentences.add(p)
                            general_sentences.add(h)
                        if whole_sentences is not None:
                            whole_sentences.add(p)
                            whole_sentences.add(h)
                        
                        input_text, output_text = _get_nli_input_output_1(p, h, l)
                        input_texts.append(input_text)
                        output_texts.append(output_text)
                        labels.append(l)
                        sources.append(source_id)

                        if not nli1_only_on_train:
                            input_text, output_text = _get_nli_input_output_2(p, h, l)
                            input_texts.append(input_text)
                            output_texts.append(output_text)
                            labels.append(l)
                            sources.append(source_id)
            source2id['anli'] = source_id
            id2source[source_id] = 'anli'
            source2type['anli'] = 'general'
            source_id += 1 # to add a new source

        if use_multinli:
            if verbose:
                print('Loading MultiNLI...')
            for r in ('train', 'dev_matched', 'dev_mismatched'):
                rows = load_jsonl(os.path.join(MULTI_NLI_DATASET_DIR, f'multinli_1.0_{r}.jsonl'))
                if verbose:
                    print(f'Number of MultiNLI {r} samples: {len(rows)}')
                for row in rows:
                    p, h, l = row['sentence1'], row['sentence2'], row['gold_label']

                    if general_sentences is not None:
                        general_sentences.add(p)
                        general_sentences.add(h)
                    if whole_sentences is not None:
                        whole_sentences.add(p)
                        whole_sentences.add(h)

                    if l == '-':
                        continue # ignore samples with no label
                    input_text, output_text = _get_nli_input_output_1(p, h, l)
                    input_texts.append(input_text)
                    output_texts.append(output_text)
                    labels.append(l)
                    sources.append(source_id)

                    if not nli1_only_on_train:
                        input_text, output_text = _get_nli_input_output_2(p, h, l)
                        input_texts.append(input_text)
                        output_texts.append(output_text)
                        labels.append(l)
                        sources.append(source_id)
            source2id['multinli'] = source_id
            id2source[source_id] = 'multinli'
            source2type['multinli'] = 'general'
            source_id += 1 # to add a new source
                    
        if use_snli:
            if verbose:
                print('Loading SNLI...')
            for r in ('train', 'dev', 'test'):
                rows = load_jsonl(os.path.join(SNLI_DATASET_DIR, f'snli_1.0_{r}.jsonl'))
                if verbose:
                    print(f'Number of SNLI {r} samples: {len(rows)}')
                for row in rows:
                    p, h, l = row['sentence1'], row['sentence2'], row['gold_label']

                    if general_sentences is not None:
                        general_sentences.add(p)
                        general_sentences.add(h)
                    if whole_sentences is not None:
                        whole_sentences.add(p)
                        whole_sentences.add(h)

                    if l == '-':
                        continue # ignore samples with no label
                    input_text, output_text = _get_nli_input_output_1(p, h, l)
                    input_texts.append(input_text)
                    output_texts.append(output_text)
                    labels.append(l)
                    sources.append(source_id)

                    if not nli1_only_on_train:
                        input_text, output_text = _get_nli_input_output_2(p, h, l)
                        input_texts.append(input_text)
                        output_texts.append(output_text)
                        labels.append(l)
                        sources.append(source_id)
            source2id['snli'] = source_id
            id2source[source_id] = 'snli'
            source2type['snli'] = 'general'
            source_id += 1 # to add a new source

    if use_report_nli:
        # Report NLI datasets
        if verbose:
            print('----')
            print_bold('Loading report NLI datasets...')
        
        assert report_nli_input_output_train_jsonl_filepaths is not None, 'report_nli_input_output_train_jsonl_filepaths must be provided'
        assert len(report_nli_input_output_train_jsonl_filepaths) == 2
        if '(from_facts)' in report_nli_input_output_train_jsonl_filepaths[0]:
            assert '(from_labels)' in report_nli_input_output_train_jsonl_filepaths[1]
            from_facts_filepath = report_nli_input_output_train_jsonl_filepaths[0]
            from_labels_filepath = report_nli_input_output_train_jsonl_filepaths[1]
        elif '(from_facts)' in report_nli_input_output_train_jsonl_filepaths[1]:
            assert '(from_labels)' in report_nli_input_output_train_jsonl_filepaths[0]
            from_facts_filepath = report_nli_input_output_train_jsonl_filepaths[1]
            from_labels_filepath = report_nli_input_output_train_jsonl_filepaths[0]
        else:
            raise ValueError(f'Unknown report NLI input/output filepaths {report_nli_input_output_train_jsonl_filepaths}')

        from_facts_input_texts, from_facts_output_texts, from_facts_labels = load_report_nli_examples_filepaths(
            from_facts_filepath, nli1_only=nli1_only_on_train, verbose=verbose, medical_sentences=medical_sentences,
            whole_sentences=whole_sentences, splittable_sentences=splittable_sentences)

        input_texts.extend(from_facts_input_texts)
        output_texts.extend(from_facts_output_texts)
        labels.extend(from_facts_labels)
        sources.extend([source_id] * len(from_facts_input_texts))
        source2id['report_nli_from_facts'] = source_id
        id2source[source_id] = 'report_nli_from_facts'
        source2type['report_nli_from_facts'] = 'medical_report_from_facts'
        source_id += 1
        
        from_labels_input_texts, from_labels_output_texts, from_labels_labels = load_report_nli_examples_filepaths(
            from_labels_filepath, nli1_only=nli1_only_on_train, verbose=verbose, medical_sentences=medical_sentences,
            whole_sentences=whole_sentences, splittable_sentences=splittable_sentences)

        if raw_report_nli_input_output_train_jsonl_filepaths is not None:
            assert integrated_report_facts_jsonl_filepath is not None
            _input_texts, _output_texts, _labels = load_raw_report_nli_examples_filepaths(
                raw_report_nli_input_output_train_jsonl_filepaths, integrated_report_facts_jsonl_filepath,
                nli1_only=nli1_only_on_train, verbose=verbose, medical_sentences=medical_sentences,
                whole_sentences=whole_sentences, splittable_sentences=splittable_sentences)
            from_labels_input_texts.extend(_input_texts)
            from_labels_output_texts.extend(_output_texts)
            from_labels_labels.extend(_labels)
        
        input_texts.extend(from_labels_input_texts)
        output_texts.extend(from_labels_output_texts)
        labels.extend(from_labels_labels)
        sources.extend([source_id] * len(from_labels_input_texts))
        source2id['report_nli_from_labels'] = source_id
        id2source[source_id] = 'report_nli_from_labels'
        source2type['report_nli_from_labels'] = 'medical_report_from_labels'
        source_id += 1

    source_label_2_indices = {}
    unique_sources = set()
    unique_labels = set()
    for i, (source, label) in enumerate(zip(sources, labels)):
        unique_sources.add(source)
        unique_labels.add(label)
        key = (source, label)
        if key not in source_label_2_indices:
            source_label_2_indices[key] = []
        source_label_2_indices[key].append(i)
    unique_sources = sorted(list(unique_sources))
    unique_labels = sorted(list(unique_labels))
    
    # Train dataset
    if verbose:
        print('----')
        print_bold('Building train NLI dataset...')
    label_datasets = []
    for label in unique_labels:
        _general_datasets = []
        _general_weights = []
        _medical_datasets = []
        _medical_weights = []
        _medical_report_from_facts_datasets = []
        _medical_report_from_facts_weights = []
        _medical_report_from_labels_datasets = []
        _medical_report_from_labels_weights = []
        for source in unique_sources:
            key = (source, label)
            if key in source_label_2_indices:
                indices = source_label_2_indices[key]
                weight = math.log2(len(indices))**3 # weight by log2(N)^3
                dataset = Seq2SeqDataset(indices, input_texts, output_texts, shuffle=True, infinite=True)
                if source2type[id2source[source]] == 'general':
                    _general_datasets.append(dataset)
                    _general_weights.append(weight)
                elif source2type[id2source[source]] == 'medical':
                    _medical_datasets.append(dataset)
                    _medical_weights.append(weight)
                elif source2type[id2source[source]] == 'medical_report_from_facts':
                    _medical_report_from_facts_datasets.append(dataset)
                    _medical_report_from_facts_weights.append(weight)
                elif source2type[id2source[source]] == 'medical_report_from_labels':
                    _medical_report_from_labels_datasets.append(dataset)
                    _medical_report_from_labels_weights.append(weight)
                else:
                    raise ValueError(f'Unknown source type {source2type[id2source[source]]}')
                if verbose:
                    print_bold(f'Source: {id2source[source]} | Label: {label} -> {len(indices)} ({weight:.2f})')
                    print('Example:')
                    i = random.choice(indices)
                    _print_input_output_pair(input_texts[i], output_texts[i])
        
        if len(_general_datasets) > 0:
            label_datasets.append(CompositeInfiniteDataset(_general_datasets, _general_weights))
            print_orange(f'General datasets for label {label}: {len(_general_datasets)}')
        
        if len(_medical_datasets) > 0:
            label_datasets.append(CompositeInfiniteDataset(_medical_datasets, _medical_weights))
            print_orange(f'Medical datasets for label {label}: {len(_medical_datasets)}')
        
        if label == 'entailment':
            if use_report_nli_entailment_dataset:
                assert integrated_report_facts_jsonl_filepath is not None
                if splittable_sentences is not None or whole_sentences is not None or medical_sentences is not None:
                    reports = get_cached_jsonl_file(integrated_report_facts_jsonl_filepath)
                    print(f'Loaded {len(reports)} reports from {integrated_report_facts_jsonl_filepath}')
                    if splittable_sentences is not None:
                        for r in reports:
                            splittable_sentences.add(r['fact_based_report'])
                    if whole_sentences is not None:
                        for r in reports:
                            for f in r['facts']:
                                whole_sentences.add(f)
                    if medical_sentences is not None:
                        for r in reports:
                            medical_sentences.add(r['fact_based_report'])
                            for f in r['facts']:
                                medical_sentences.add(f)
                fbred = FactBasedReportEntailmentDataset(
                    integrated_report_facts_jsonl_filepath=integrated_report_facts_jsonl_filepath,
                    nli1_only=nli1_only_on_train, shuffle=True, infinite=True)
                weight = sum(_medical_report_from_facts_weights) / len(_medical_report_from_facts_weights) # average weight
                _medical_report_from_facts_datasets.append(fbred)
                _medical_report_from_facts_weights.append(weight)
            
            if use_report_nli_paraphrases_dataset:
                assert input_to_paraphrases is not None
                assert integrated_report_facts_jsonl_filepath is not None
                fbrpd = FactBasedReportParaphrasesDataset(
                    integrated_report_facts_jsonl_filepath=integrated_report_facts_jsonl_filepath,
                    input2paraphrases=input_to_paraphrases, nli1_only=nli1_only_on_train, infinite=True,
                    splitable_sentences=splittable_sentences, whole_sentences=whole_sentences,
                    medical_sentences=medical_sentences)
                weight = sum(_medical_report_from_labels_weights) / len(_medical_report_from_labels_weights) # average weight
                _medical_report_from_labels_datasets.append(fbrpd)
                _medical_report_from_labels_weights.append(weight)
                # print 3 examples
                for _ in range(3):
                    i = random.randint(0, len(fbrpd) - 1)
                    print_bold('Example:')
                    print(fbrpd[i])
        
        if len(_medical_report_from_facts_datasets) > 0:
            label_datasets.append(CompositeInfiniteDataset(_medical_report_from_facts_datasets, _medical_report_from_facts_weights))
            print_orange(f'Medical report from facts datasets for label {label}: {len(_medical_report_from_facts_datasets)}')
        
        if len(_medical_report_from_labels_datasets) > 0:
            label_datasets.append(CompositeInfiniteDataset(_medical_report_from_labels_datasets, _medical_report_from_labels_weights))
            print_orange(f'Medical report from labels datasets for label {label}: {len(_medical_report_from_labels_datasets)}')

    train_dataset = CompositeInfiniteDataset(label_datasets, [1] * len(label_datasets)) # equal weights
    
    # Val dataset
    if verbose:
        print('----')
    val_input_texts = []
    val_output_texts = []
    
    # RadNLI
    _input_texts, _output_texts = load_radnli_test_data(verbose=verbose, nli1_only=nli1_only_on_val,
                                                        medical_sentences=medical_sentences,
                                                        whole_sentences=whole_sentences)
    val_input_texts.extend(_input_texts)
    val_output_texts.extend(_output_texts)
        
    # MS_CXR_T
    _input_texts, _output_texts = load_ms_cxr_t_temporal_sentence_similarity_v1_data(verbose=verbose, nli1_only=nli1_only_on_val,
                                                                                    medical_sentences=medical_sentences,
                                                                                    whole_sentences=whole_sentences)
    val_input_texts.extend(_input_texts)
    val_output_texts.extend(_output_texts)

    # Report NLI
    if use_report_nli:
        if report_nli_input_output_val_jsonl_filepaths is not None:
            for input_output_jsonl_filepath in report_nli_input_output_val_jsonl_filepaths:
                _input_texts, _output_texts, _ = load_report_nli_examples_filepaths(
                    input_output_jsonl_filepath, nli1_only=nli1_only_on_val, verbose=verbose,
                    medical_sentences=medical_sentences, whole_sentences=whole_sentences)
                val_input_texts.extend(_input_texts)
                val_output_texts.extend(_output_texts)

    print(f'Number of val NLI samples: {len(val_input_texts)}')
    
    # Print val examples
    if verbose:
        for  _ in range(5):
            i = random.randint(0, len(val_input_texts) - 1)
            _print_input_output_pair(val_input_texts[i], val_output_texts[i])
    
    # Val NLI dataset and dataloader
    if verbose:
        print('----')
        print_bold('Building val NLI dataset...')
    val_dataset = Seq2SeqDataset(list(range(len(val_input_texts))), val_input_texts, val_output_texts, shuffle=False)

    return train_dataset, val_dataset

class FactBasedReportEntailmentDataset(Dataset):
    def __init__(self, integrated_report_facts_jsonl_filepath, nli1_only=True, shuffle=False, infinite=False):
        print('FactBasedReportEntailmentDataset')
        reports = get_cached_jsonl_file(integrated_report_facts_jsonl_filepath)
        print(f'Loaded {len(reports)} reports from {integrated_report_facts_jsonl_filepath}')
        indices = [i for i in range(len(reports)) if len(reports[i]['facts']) > 0]
        self.indices = indices
        self.reports = reports
        self.nli1_only = nli1_only
        self.shuffle = shuffle
        self.infinite = infinite
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)
        if shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self.infinite:
            i = i % len(self.indices)
        i = self.indices[i]
        r = self.reports[i]
        p = r['fact_based_report']
        h = random.choice(r['facts'])
        if self.nli1_only or random.random() < 0.5:
            input_text, output_text = _get_nli_input_output_1(p, h, 'entailment')
        else:
            input_text, output_text = _get_nli_input_output_2(p, h, 'entailment')
        return { 'input_text': input_text, 'output_text': output_text }

class FactBasedReportParaphrasesDataset(Dataset):
    def __init__(self, integrated_report_facts_jsonl_filepath, input2paraphrases, nli1_only=True, infinite=False,
                 splitable_sentences=None, whole_sentences=None, medical_sentences=None):
        print('FactBasedReportParaphrasesDataset')
        reports = get_cached_jsonl_file(integrated_report_facts_jsonl_filepath)
        print(f'Loaded {len(reports)} reports from {integrated_report_facts_jsonl_filepath}')
        
        fact2ridxs = {}
        for i, r in enumerate(reports):
            for f in r['facts']:
                if f not in fact2ridxs:
                    fact2ridxs[f] = []
                fact2ridxs[f].append(i)
        print(f'Number of facts: {len(fact2ridxs)}')
        
        inputs_with_reports = []
        inputs_without_reports = []
        for input_text, paraphrases in input2paraphrases.items():
            assert len(paraphrases) > 0
            if input_text in fact2ridxs:
                inputs_with_reports.append(input_text)
            else:
                inputs_without_reports.append(input_text)
        print(f'Number of inputs with reports: {len(inputs_with_reports)}')
        print(f'Number of inputs without reports: {len(inputs_without_reports)}')

        self.fact2ridxs = fact2ridxs
        self.inputs_with_reports = inputs_with_reports
        self.inputs_without_reports = inputs_without_reports
        self.input2paraphrases = input2paraphrases
        self.reports = reports
        self.nli1_only = nli1_only
        self.infinite = infinite
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.inputs_with_reports) + len(self.inputs_without_reports)

        if splitable_sentences is not None:
            for r in reports:
                splitable_sentences.add(r['fact_based_report'])
        if whole_sentences is not None:
            for r in reports:
                for f in r['facts']:
                    whole_sentences.add(f)
            for input_text, paraphrases in input2paraphrases.items():
                whole_sentences.add(input_text)
                whole_sentences.update(paraphrases)
        if medical_sentences is not None:
            for r in reports:
                medical_sentences.add(r['fact_based_report'])
                for f in r['facts']:
                    medical_sentences.add(f)
            for input_text, paraphrases in input2paraphrases.items():
                medical_sentences.add(input_text)
                for p in paraphrases:
                    medical_sentences.add(p)
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if random.random() < 0.5:
            input_text = random.choice(self.inputs_with_reports)
            ridxs = self.fact2ridxs[input_text]
            ridx = random.choice(ridxs)
            r = self.reports[ridx]
            p = r['fact_based_report']
            paraphrases = self.input2paraphrases[input_text]
            h = random.choice(paraphrases)
        else:
            input_text = random.choice(self.inputs_without_reports)
            paraphrases = self.input2paraphrases[input_text]
            p = input_text
            h = random.choice(paraphrases)
        if self.nli1_only or random.random() < 0.5:
            input_text, output_text = _get_nli_input_output_1(p, h, 'entailment')
        else:
            input_text, output_text = _get_nli_input_output_2(p, h, 'entailment')
        return { 'input_text': input_text, 'output_text': output_text }

class MaskedLMDataset(Dataset):
    def __init__(self, indices, sentences, tokenized_sentences, maskable_token_indices, masking_fraction,
                 apply_task_prefix=False, shuffle=False, infinite=False):
        self.indices = indices
        self.sentences = sentences
        self.tokenized_sentences = tokenized_sentences
        self.maskable_token_indices = maskable_token_indices
        self.masking_fraction = masking_fraction
        self.apply_task_prefix = apply_task_prefix
        self.shuffle = shuffle
        self.infinite = infinite
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)
        if shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self.infinite:
            i = i % len(self.indices)
        i = self.indices[i]
        tokens = self.tokenized_sentences[i][:] # copy
        maskable_token_indices = self.maskable_token_indices[i]
        n = math.ceil(len(maskable_token_indices) * self.masking_fraction)
        assert n > 0
        if n >= len(maskable_token_indices):
            n = len(maskable_token_indices) - 1
        assert n > 0
        indices = random.sample(maskable_token_indices, n)
        indices.sort()
        output_list = []
        for idx, j in enumerate(indices):
            original_token = tokens[j]
            masked_token = f'[tok{idx+1}]'
            output_list.append(masked_token)
            output_list.append(original_token)
            tokens[j] = masked_token
        input_text = ' '.join(tokens)
        if self.apply_task_prefix:
            input_text = f'MLM: {input_text}'
        output_text = ' '.join(output_list)
        output = { 'input_text': input_text, 'output_text': output_text }
        return output

def _prepare_masked_lm_dataset(sentences, apply_task_prefix=False, verbose=True, min_token_count=20, masking_fraction=0.2, num_bins=10):
    assert sentences is not None, 'sentences must be provided'
    print('Preparing masked LM dataset with masking')
    print(f'\tmin_token_count: {min_token_count}')
    print(f'\tmasking_fraction: {masking_fraction}')
    print(f'\tlen(sentences): {len(sentences)}')
    from nltk.corpus import stopwords
    from string import punctuation
    import re
    has_puctuation = re.compile(f'[{punctuation}]')
    has_letter = re.compile('[a-zA-Z]')
    stop_words = set(stopwords.words('english'))
    print('Tokenizing sentences...')
    tokenized_sentences = word_tokenize_texts_in_parallel(sentences)
    print('Lowercasing tokens...')
    lower_tokenized_sentences = tokenized_texts_to_lower_in_parallel(tokenized_sentences)
    token2count = {}
    for tokens in tqdm(lower_tokenized_sentences, total=len(tokenized_sentences), mininterval=2):
        for token in tokens:
            if token not in token2count:
                token2count[token] = 0
            token2count[token] += 1
    valid_tokens = [token for token, count in token2count.items() if count >= min_token_count and token not in stop_words \
                    and not has_puctuation.search(token) and has_letter.search(token)]
    valid_tokens = set(valid_tokens)
    print(f'\tlen(valid_tokens): {len(valid_tokens)}')
    valid_tokenized_sentences = [None] * len(tokenized_sentences)
    valid_maskable_token_indices = [None] * len(tokenized_sentences)
    valid_sentences = [None] * len(tokenized_sentences)
    valid_sentence_scores = [None] * len(tokenized_sentences)
    i = 0
    for tokens, lower_tokens in tqdm(zip(tokenized_sentences, lower_tokenized_sentences),
                                     total=len(tokenized_sentences), mininterval=2):
        token_indices = [j for j, token in enumerate(lower_tokens) if token in valid_tokens]
        if len(token_indices) == 0:
            continue
        n = math.ceil(len(token_indices) * masking_fraction)
        assert n > 0
        if n >= len(token_indices):
            n = len(token_indices) - 1
        if n == 0:
            continue
        valid_tokenized_sentences[i] = tokens
        valid_maskable_token_indices[i] = token_indices
        valid_sentences[i] = sentences[i]
        valid_sentence_scores[i] = sum(token2count[lower_tokens[j]] for j in token_indices) / len(token_indices)
        i += 1
    valid_tokenized_sentences = valid_tokenized_sentences[:i]
    valid_maskable_token_indices = valid_maskable_token_indices[:i]
    valid_sentences = valid_sentences[:i]
    valid_sentence_scores = valid_sentence_scores[:i]
    if verbose:
        print(f'Number of sentences: {len(valid_sentences)}')
    grouped_indices = group_indices_into_bins_by_scores(valid_sentence_scores, num_bins)
    print(f'Number of bins: {len(grouped_indices)}')
    datasets = []
    weights = []
    for indices in grouped_indices:
        weight = math.log2(len(indices))**3 # weight by log^3 of number of examples
        dataset = MaskedLMDataset(indices, valid_sentences, valid_tokenized_sentences, valid_maskable_token_indices,
                                masking_fraction, apply_task_prefix=apply_task_prefix, shuffle=True, infinite=True)
        datasets.append(dataset)
        weights.append(weight)
        print(f'Bin size: {len(indices)}, weight: {weight}')
    dataset = CompositeInfiniteDataset(datasets, weights)
    if verbose:
        print('Examples:')
        for _ in range(4):
            i = random.randint(0, len(dataset) - 1)
            output = dataset[i]
            print_bold('Input:')
            print_magenta(output['input_text'])
            print_bold('Output:')
            print_magenta(output['output_text'])
    return dataset

def _compute_input2paraphrases(paraphrased_inputs_jsonl_filepaths, verbose=True):
    if paraphrased_inputs_jsonl_filepaths is not None:
        assert type(paraphrased_inputs_jsonl_filepaths) == list
        input2paraphrases = {}
        empty_count = 0
        for filepath in paraphrased_inputs_jsonl_filepaths:
            paraphrased_inputs = load_jsonl(filepath)
            if verbose:
                print('--------')
                print_bold(f'Loaded {len(paraphrased_inputs)} paraphrased inputs from {filepath}')
                print_count = 0
            for row in paraphrased_inputs:
                input_text = next(iter(row['metadata'].values()))
                parsed_response = row['parsed_response']
                if type(parsed_response) == list:
                    paraphrases = parsed_response
                elif type(parsed_response) == dict:
                    assert 'positives' in parsed_response and 'negatives' in parsed_response
                    paraphrases = parsed_response['positives'] # only use positives
                else:
                    raise ValueError(f'Unknown type {type(parsed_response)}')
                assert len(input_text) > 0
                if len(paraphrases) == 0:
                    empty_count += 1
                    continue
                if verbose and print_count < 1:
                    print_count += 1
                    print_bold(f'Input:')
                    print_magenta(input_text, bold=True)
                    print_bold(f'Paraphrases:')
                    for p in paraphrases:
                        print_magenta(p, bold=True)
                if input_text not in input2paraphrases:
                    input2paraphrases[input_text] = []
                input2paraphrases[input_text].extend(paraphrases)
        if verbose:
            print('--------')
            print_bold(f'Number of unique inputs: {len(input2paraphrases)}')
            print_bold(f'Number of total paraphrases: {sum(len(x) for x in input2paraphrases.values())}')
            if empty_count > 0:
                print_orange(f'WARNING: Number of empty paraphrases: {empty_count}', bold=True)
        assert len(input2paraphrases) > 0
        assert all(len(x) > 0 for x in input2paraphrases.values())
        return input2paraphrases
    
def _get_fact_to_comparison_train_val_datasets(input_texts, output_texts, val_size, verbose=True, include_val=True):

    if verbose:
        from collections import Counter
        counter = Counter(output_texts)
        print('Counter:')
        print(counter)

    # Specific balanced train/val split for fact2comparison

    output2indices = {}
    for i, output_text in enumerate(output_texts):
        if output_text not in output2indices:
            output2indices[output_text] = []
        output2indices[output_text].append(i)

    if include_val:
        val_indices_set = set()
        val_sizes = [0]  * len(output2indices)
        count = 0
        while count < val_size:
            for i, indices in enumerate(output2indices.values()):
                if (val_sizes[i] + 1) * 10 <= len(indices):
                    val_sizes[i] += 1
                    count += 1
                    if count >= val_size:
                        break
    train_datasets = []
    train_weights = []
    train_size = 0
    for i, (output, indices) in enumerate(output2indices.items()):
        if include_val:
            assert val_sizes[i] * 10 <= len(indices), f'val_size {val_sizes[i]} is too large for output {output} (len {len(indices)})'
            indices.sort(key=lambda i: (len(output_texts[i]), output_texts[i])) # sort by output length, then alphabetically
            val_indices_set.update(indices[j] for j in np.linspace(0, len(indices) - 1, val_sizes[i], dtype=int))
            train_indices = [i for i in indices if i not in val_indices_set]
        else:
            train_indices = indices
        train_datasets.append(Seq2SeqDataset(train_indices, input_texts, output_texts, shuffle=True, infinite=True))
        train_weights.append(math.log2(len(train_indices))**3) # weight by log^3 of number of examples
        train_size += len(train_indices)
        if verbose:
            if include_val:
                print(f'Output: {output}, train size: {len(train_indices)}, val size: {val_sizes[i]}, weight: {train_weights[-1]}')
            else:
                print(f'Output: {output}, train size: {len(train_indices)}, weight: {train_weights[-1]}')

    if include_val:
        val_indices = list(val_indices_set)
    
    # Create datasets
    train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
    if include_val:
        val_dataset = Seq2SeqDataset(val_indices, input_texts, output_texts, shuffle=False)
    else:
        val_dataset = None

    if verbose:
        # Print stats
        print(f'Number of train examples: {train_size}')
        if include_val:
            print(f'Number of val examples: {len(val_indices)}')
            print(f'Number of total examples: {train_size + len(val_indices)}')

    return train_dataset, val_dataset

def _get_sentence_to_chest_imagenome_labels_datasets(input_texts, output_texts, val_size, label_names, verbose=True, include_val=True):
    
    label_name_2_idx = { label_name: i for i, label_name in enumerate(label_names) }
    assert len(label_name_2_idx) == len(label_names)
    label2idxs = [[] for _ in range(len(label_name_2_idx))]
    idxs_without_labels = []
    unknown_labels = set()
    unknown_count = 0
    
    for i, output_text in tqdm(enumerate(output_texts), total=len(output_texts), mininterval=2):
        labels = json.loads(output_text)
        clean_labels = []
        for label in labels:
            try:
                label2idxs[label_name_2_idx[label]].append(i)
                clean_labels.append(label)
            except KeyError:
                unknown_count += 1
                unknown_labels.add(label)
        if len(clean_labels) == 0:
            idxs_without_labels.append(i)
        if len(clean_labels) < len(labels):
            output_texts[i] = json.dumps(clean_labels) # remove unknown labels

    if verbose:
        print(f'Number of total examples: {len(output_texts)}')
        print(f'Number of examples without labels: {len(idxs_without_labels)}')
        if unknown_count > 0:
            print_orange(f'WARNING: Number of unknown labels: {unknown_count}', bold=True)
            unknown_labels = list(unknown_labels)
            print_orange('Some unknown labels:', bold=True)
            for x in random.sample(unknown_labels, min(10, len(unknown_labels))):
                print_orange(f'  {x}', bold=True)

    if include_val:
        # Split into train & val
        val_idxs_set = set()
        val_sizes = [0] * (len(label2idxs) + 1) # +1 for idxs_without_labels
        count = 0
        while count < val_size:
            for i, idxs in enumerate(label2idxs):
                if (val_sizes[i] + 1) * 10 <= len(idxs):
                    val_sizes[i] += 1
                    count += 1
                    if count >= val_size:
                        break
            if (val_sizes[-1] + 1) * 10 <= len(idxs_without_labels):
                val_sizes[-1] += 1
                count += 1
                if count >= val_size:
                    break
    train_datasets = []
    train_weights = []
    if verbose:
        print_messages = []
    train_size = 0
    if include_val:
        for i, idxs in enumerate(label2idxs):
            assert val_sizes[i] * 10 <= len(idxs), f'val_size {val_sizes[i]} is too large for label {label_names[i]} (len {len(idxs)})'
            val_idxs = random.sample(idxs, val_sizes[i])
            val_idxs_set.update(val_idxs)
        val_idxs_set.update(random.sample(idxs_without_labels, val_sizes[-1]))
    for i, idxs in enumerate(label2idxs):
        if include_val:
            train_idxs = [i for i in idxs if i not in val_idxs_set]
        else:
            train_idxs = idxs
        train_datasets.append(Seq2SeqDataset(train_idxs, input_texts, output_texts, shuffle=True, infinite=True))
        train_weights.append(math.log2(len(train_idxs))**3) # weight by log^3 of number of examples
        train_size += len(train_idxs)
        if verbose:
            if include_val:
                print_messages.append((
                    len(train_idxs),
                    f'Label: {label_names[i]}, train size: {len(train_idxs)}, val size: {val_sizes[i]}, weight: {train_weights[-1]}'
                ))
            else:
                print_messages.append((
                    len(train_idxs),
                    f'Label: {label_names[i]}, train size: {len(train_idxs)}, weight: {train_weights[-1]}'
                ))
    if include_val:
        train_idxs = [i for i in idxs_without_labels if i not in val_idxs_set]
    else:
        train_idxs = idxs_without_labels
    train_datasets.append(Seq2SeqDataset(train_idxs, input_texts, output_texts, shuffle=True, infinite=True))
    train_weights.append(math.log2(len(train_idxs))**3) # weight by log^3 of number of examples
    train_size += len(train_idxs)
    if verbose:
        if include_val:
            print_messages.append((
                len(train_idxs),
                f'Label: None, train size: {len(train_idxs)}, val size: {val_sizes[-1]}, weight: {train_weights[-1]}'
            ))
        else:
            print_messages.append((
                len(train_idxs),
                f'Label: None, train size: {len(train_idxs)}, weight: {train_weights[-1]}'
            ))
    if include_val:
        val_idxs = list(val_idxs_set)

    if verbose:
        # Print label stats
        print('--------')
        print_bold('Label stats:')
        print_messages.sort(key=lambda x: x[0], reverse=True)
        for _, msg in print_messages:
            print(msg)
        print('--------')
        print(f'Number of train examples: {train_size}')
        if include_val:
            print(f'Number of val examples: {len(val_idxs)}')
            print(f'Number of total examples: {train_size + len(val_idxs)}')
        print(f'Number of train datasets: {len(train_datasets)}')
        print('--------')
        print_bold('Examples:')
        for _ in range(4):
            _print_random_input_output_pair(input_texts, output_texts)

    # Create datasets
    train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
    if include_val:
        val_dataset = Seq2SeqDataset(val_idxs, input_texts, output_texts, shuffle=False)
    else:
        val_dataset = None

    return train_dataset, val_dataset

def _probs_and_preds_to_input_text(class_names, probs, preds, use_numeric_templates=False):
    assert probs.shape == preds.shape
    assert len(class_names) == probs.shape[1]
    n_views = probs.shape[0]
    assert n_views > 0
    if use_numeric_templates:
        lines = []
        lines.append(f'{n_views}')
        for i, cn in enumerate(class_names):
            strings = [f'{int(probs[j, i] * 10)}' for j in range(n_views)]
            lines.append(" ".join(strings))
        return ','.join(lines)
    else:    
        pred = preds.max(axis=0)
        lines = []
        lines.append(f'views: {n_views}')
        for i, cn in enumerate(class_names):
            if pred[i]:
                strings = []
                for j in range(n_views):
                    if preds[j, i]:
                        strings.append(f'yes {int(probs[j, i] * 10)}')
                    else:
                        strings.append(f'no {int(probs[j, i] * 10)}')
                lines.append(f'{cn}:{",".join(strings)}')
        return '\n'.join(lines)

def _get_fact_classifier_predictions_to_report_section_datasets__interpret_cxr_challenge(
    interpret_cxr__label_based_predictions_filepath,
    interpret_cxr_challenge_data_dir,
    mimiccxr_integrated_report_nli_data_filepath,
    section,
    lowercase_output_texts=True,
    include_public_test_in_train=False,
    verbose=True,
    best_k_classes=None,
    use_numeric_templates=False,
):
    assert section in ['findings', 'impression']
    if verbose:
        print(f'Loading {interpret_cxr__label_based_predictions_filepath}...')
    tmp = load_pickle(interpret_cxr__label_based_predictions_filepath)
    probs_filepath = tmp['probs_filepath']
    thresholds = tmp['thresholds']
    f1s = tmp['f1s']
    accs = tmp['accs']
    class_names = tmp['class_names']
    if verbose:
        print(f'Loading {probs_filepath}...')
    tmp2 = load_pickle(probs_filepath)
    probs = tmp2['probs']
    image_paths = tmp2['image_paths']
    image_path_2_idx = { image_path: i for i, image_path in enumerate(image_paths) }
    assert len(image_paths) == len(image_path_2_idx) # unique
    if verbose:
        print(f'thresholds.shape: {thresholds.shape}')
        print(f'f1s.shape: {f1s.shape}')
        print(f'accs.shape: {accs.shape}')
        print(f'probs.shape: {probs.shape}')
        print(f'len(image_paths): {len(image_paths)}')
        print(f'len(class_names): {len(class_names)}')

    # Sort by hybrid score
    hybrid_score = f1s + accs
    sorted_class_idxs = np.argsort(hybrid_score)[::-1]
    class_names = [class_names[i] for i in sorted_class_idxs]
    f1s = f1s[sorted_class_idxs]
    accs = accs[sorted_class_idxs]
    probs = probs[:, sorted_class_idxs]
    binary_predictions = (probs > thresholds).astype(int)

    # Select the first k classes
    if best_k_classes is not None:
        class_names = class_names[:best_k_classes]
        f1s = f1s[:best_k_classes]
        accs = accs[:best_k_classes]
        probs = probs[:, :best_k_classes]
        binary_predictions = binary_predictions[:, :best_k_classes]

    if verbose:
        print('Class names:')
        for i, class_name in enumerate(class_names):
            print(f'{i + 1}: {class_name}, f1: {f1s[i]:.3f}, acc: {accs[i]:.3f}')

    train_mimic_filepath = os.path.join(interpret_cxr_challenge_data_dir, 'train_mimic.json')
    val_mimic_filepath = os.path.join(interpret_cxr_challenge_data_dir, 'val_mimic.json')
    train_filepath = os.path.join(interpret_cxr_challenge_data_dir, 'train.csv')
    val_filepath = os.path.join(interpret_cxr_challenge_data_dir, 'val.csv')

    # 1. MIMIC-CXR inputs/outputs

    if verbose:
        print('-' * 100)
        print('Preparing MIMIC-CXR inputs/outputs...')

    train_mimic = load_json(train_mimic_filepath)
    val_mimic = load_json(val_mimic_filepath)
    mimiccxr_integrated_report_nli_data = load_pickle(mimiccxr_integrated_report_nli_data_filepath)
    extracted_facts = mimiccxr_integrated_report_nli_data['extracted_facts']
    reports = mimiccxr_integrated_report_nli_data['reports']
    imageId2reportId = get_imageId2reportId()
    imageId2PartPatientStudy = get_imageId2PartPatientStudy()
    
    def _collect_mimiccxr_input_output_texts(mimic_data):
        input_texts = []
        output_texts = []
        image_idxs_list = []
        skip_count = 0
        for item in tqdm(mimic_data, total=len(mimic_data), mininterval=2):
            image_idxs = []
            dicom_ids = []
            for image_path in item['images_path']:
                dicom_id = os.path.basename(image_path).split('.')[0] # remove extension
                part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
                actual_image_path = get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id)
                image_idx = image_path_2_idx[actual_image_path]
                image_idxs.append(image_idx)
                dicom_ids.append(dicom_id)
            # Build output text
            ridx = imageId2reportId[dicom_ids[0]]
            assert all(imageId2reportId[dicom_id] == ridx for dicom_id in dicom_ids) # same report
            report = reports[ridx]
            if section == 'findings':
                fact_idxs = report['findings_fact_idxs']
            elif section == 'impression':
                fact_idxs = report['impression_fact_idxs']
            else: assert False
            if len(fact_idxs) == 0:
                skip_count += 1
                continue # skip reports without facts
            facts = [extracted_facts[i] if extracted_facts[i][-1] == '.' else extracted_facts[i] + '.' for i in fact_idxs]
            output_text = ' '.join(facts) # a fact-based report
            output_texts.append(output_text)
            # Build input text
            image_probs = probs[image_idxs]
            image_preds = binary_predictions[image_idxs]
            input_text = _probs_and_preds_to_input_text(class_names, image_probs, image_preds, use_numeric_templates)
            input_texts.append(input_text)
            # Save image_idxs
            image_idxs_list.append(image_idxs)
        if skip_count > 0:
            if verbose:
                print_orange(f'WARNING: Skipped {skip_count}/{len(mimic_data)} reports without facts', bold=True)
        return input_texts, output_texts, image_idxs_list
        
    mimiccxr_train_input_texts, mimiccxr_train_output_texts, mimiccxr_train_image_idxs_list =\
        _collect_mimiccxr_input_output_texts(train_mimic)
    if verbose:
        print(f'len(mimiccxr_train_input_texts): {len(mimiccxr_train_input_texts)}')
    
    mimiccxr_val_input_texts, mimiccxr_val_output_texts, mimiccxr_val_image_idxs_list =\
        _collect_mimiccxr_input_output_texts(val_mimic)
    if verbose:
        print(f'len(mimiccxr_val_input_texts): {len(mimiccxr_val_input_texts)}')

    if verbose:
        print('Examples:')
        i = random.randint(0, len(mimiccxr_train_input_texts) - 1)
        print_bold('Train example:')
        print_bold('Input:')
        print_magenta(mimiccxr_train_input_texts[i])
        print_bold('Output:')
        print_magenta(mimiccxr_train_output_texts[i])
        print()
        
        i = random.randint(0, len(mimiccxr_val_input_texts) - 1)
        print_bold('Val example:')
        print_bold('Input:')
        print_magenta(mimiccxr_val_input_texts[i])
        print_bold('Output:')
        print_magenta(mimiccxr_val_output_texts[i])
        print()

    # 2. CheXpert inputs/outputs

    if verbose:
        print('-' * 100)
        print('Preparing CheXpert inputs/outputs...')
    
    train_df = pd.read_csv(train_filepath)
    val_df = pd.read_csv(val_filepath)
    
    # Replace NaN with empty string
    train_df.fillna('', inplace=True)
    val_df.fillna('', inplace=True)

    train_chexpert_df = train_df.loc[train_df.source == 'CheXpert']
    val_chexpert_df = val_df.loc[val_df.source == 'CheXpert']

    def _collect_chexpert_input_output_texts(chexpert_df):
        input_texts = []
        output_texts = []
        image_idxs_list = []
        skip_image_count = 0
        skip_text_count = 0
        for images_path, findings, impression in chexpert_df[['images_path_old', 'findings', 'impression']].values:
            # Convert images_path to image_idxs
            images_path_ = eval(images_path) # convert string to list
            image_idxs = []
            for image_path in images_path_:
                for x in image_path.split('.jpg'):
                    if x:
                        x = x[21:] + '.jpg'
                        actual_image_path = os.path.join(CHEXPERT_V1_0_SMALL_DATASET_DIR, x)
                        image_idx = image_path_2_idx[actual_image_path]
                        image_idxs.append(image_idx)
            if len(image_idxs) == 0:
                skip_image_count += 1
                continue
            # Build output text
            if section == 'findings':
                output_text = findings
            elif section == 'impression':
                output_text = impression
            else: assert False
            output_text = ' '.join([x for x in output_text.split() if x]) # remove whitespace
            if len(output_text) == 0:
                skip_text_count += 1
                continue
            output_texts.append(output_text)
            # Build input text
            image_probs = probs[image_idxs]
            image_preds = binary_predictions[image_idxs]
            input_text = _probs_and_preds_to_input_text(class_names, image_probs, image_preds, use_numeric_templates)
            input_texts.append(input_text)
            # Save image_idxs
            image_idxs_list.append(image_idxs)
        if verbose:
            if skip_image_count > 0:
                print_orange(f'WARNING: Skipped {skip_image_count}/{len(chexpert_df)} reports without images', bold=True)
            if skip_text_count > 0:
                print_orange(f'WARNING: Skipped {skip_text_count}/{len(chexpert_df)} reports with empty output', bold=True)

        return input_texts, output_texts, image_idxs_list

    chexpert_train_input_texts, chexpert_train_output_texts, chexpert_train_image_idxs_list =\
        _collect_chexpert_input_output_texts(train_chexpert_df)
    if verbose:
        print(f'len(chexpert_train_input_texts): {len(chexpert_train_input_texts)}')

    chexpert_val_input_texts, chexpert_val_output_texts, chexpert_val_image_idxs_list =\
        _collect_chexpert_input_output_texts(val_chexpert_df)
    if verbose:
        print(f'len(chexpert_val_input_texts): {len(chexpert_val_input_texts)}')

    if verbose:
        print('Examples:')
        i = random.randint(0, len(chexpert_train_input_texts) - 1)
        print_bold('Train example:')
        print_bold('Input:')
        print_magenta(chexpert_train_input_texts[i])
        print_bold('Output:')
        print_magenta(chexpert_train_output_texts[i])
        print()

        i = random.randint(0, len(chexpert_val_input_texts) - 1)
        print_bold('Val example:')
        print_bold('Input:')
        print_magenta(chexpert_val_input_texts[i])
        print_bold('Output:')
        print_magenta(chexpert_val_output_texts[i])
        print()

    # 3. OpenI inputs/outputs

    if verbose:
        print('-' * 100)
        print('Preparing OpenI inputs/outputs...')

    train_openi_df = train_df.loc[train_df.source == 'OpenI']
    val_openi_df = val_df.loc[val_df.source == 'OpenI']
    
    def _collect_openi_input_output_texts(openi_df):
        input_texts = []
        output_texts = []
        image_idxs_list = []
        skip_image_count = 0
        skip_text_count = 0
        for images_path, findings, impression in openi_df[['images_path_old', 'findings', 'impression']].values:
            # Convert images_path to image_idxs
            images_path_ = eval(images_path)
            image_idxs = []
            for image_path in images_path_:
                for x in image_path.split('.png'):
                    if x:
                        image_id = os.path.basename(x)
                        actual_image_path = get_iuxray_image_path(image_id)
                        image_idx = image_path_2_idx[actual_image_path]
                        image_idxs.append(image_idx)
            if len(image_idxs) == 0:
                skip_image_count += 1
                continue
            # Build output text
            if section == 'findings':
                output_text = findings
            elif section == 'impression':
                output_text = impression
            else: assert False
            output_text = ' '.join([x for x in output_text.split() if x]) # remove whitespace
            if len(output_text) == 0:
                skip_text_count += 1
                continue
            output_texts.append(output_text)
            # Build input text
            image_probs = probs[image_idxs]
            image_preds = binary_predictions[image_idxs]
            input_text = _probs_and_preds_to_input_text(class_names, image_probs, image_preds, use_numeric_templates)
            input_texts.append(input_text)
            # Save image_idxs
            image_idxs_list.append(image_idxs)
        if verbose:
            if skip_image_count > 0:
                print_orange(f'WARNING: Skipped {skip_image_count}/{len(openi_df)} reports without images', bold=True)
            if skip_text_count > 0:
                print_orange(f'WARNING: Skipped {skip_text_count}/{len(openi_df)} reports with empty output', bold=True)
        return input_texts, output_texts, image_idxs_list

    openi_train_input_texts, openi_train_output_texts, openi_train_image_idxs_list =\
        _collect_openi_input_output_texts(train_openi_df)
    if verbose:
        print(f'len(openi_train_input_texts): {len(openi_train_input_texts)}')

    openi_val_input_texts, openi_val_output_texts, openi_val_image_idxs_list =\
        _collect_openi_input_output_texts(val_openi_df)
    if verbose:
        print(f'len(openi_val_input_texts): {len(openi_val_input_texts)}')

    if verbose:
        print('Examples:')
        i = random.randint(0, len(openi_train_input_texts) - 1)
        print_bold('Train example:')
        print_bold('Input:')
        print_magenta(openi_train_input_texts[i])
        print_bold('Output:')
        print_magenta(openi_train_output_texts[i])
        print()

        i = random.randint(0, len(openi_val_input_texts) - 1)
        print_bold('Val example:')
        print_bold('Input:')
        print_magenta(openi_val_input_texts[i])
        print_bold('Output:')
        print_magenta(openi_val_output_texts[i])
        print()

    # 4. Interpret-CXR public test set inputs/outputs

    if verbose:
        print('-' * 100)
        print('Preparing Interpret-CXR public test set inputs/outputs...')

    df = pd.read_csv(INTERPRET_CXR_TEST_PUBLIC_CSV_PATH)
    df = df.replace(np.nan, '', regex=True) # replace nan with empty string
    public_test_input_texts = []
    public_test_output_texts = []
    public_test_image_idxs_list = []
    skip_count = 0
    for images_path, findings, impression in df[['images_path', 'findings', 'impression']].values:
        images_path = eval(images_path) # convert string to list
        image_idxs = []
        for image_path in images_path:
            actual_image_path = os.path.join(INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH, os.path.basename(image_path))
            image_idx = image_path_2_idx[actual_image_path]
            image_idxs.append(image_idx)
        if len(image_idxs) == 0:
            skip_count += 1
            continue
        # Build output text
        if section == 'findings':
            output_text = findings
        elif section == 'impression':
            output_text = impression
        else: assert False
        output_text = ' '.join([x for x in output_text.split() if x]) # remove whitespace
        if len(output_text) == 0:
            skip_count += 1
            continue
        public_test_output_texts.append(output_text)
        # Build input text
        image_probs = probs[image_idxs]
        image_preds = binary_predictions[image_idxs]
        input_text = _probs_and_preds_to_input_text(class_names, image_probs, image_preds, use_numeric_templates)
        public_test_input_texts.append(input_text)
        # Save image_idxs
        public_test_image_idxs_list.append(image_idxs)
    if skip_count > 0:
        if verbose:
            print_orange(f'WARNING: Skipped {skip_count}/{len(df)} reports without images or empty output', bold=True)

    if verbose:
        print(f'len(public_test_input_texts): {len(public_test_input_texts)}')

    if verbose:
        print('Examples:')
        i = random.randint(0, len(public_test_input_texts) - 1)
        print_bold('Public test example:')
        print_bold('Input:')
        print_magenta(public_test_input_texts[i])
        print_bold('Output:')
        print_magenta(public_test_output_texts[i])
        print()

    # Create training dataset
    
    if include_public_test_in_train:
        print('Including public test set in training dataset...')
        train_input_texts = (mimiccxr_train_input_texts + mimiccxr_val_input_texts +
                             chexpert_train_input_texts + chexpert_val_input_texts +
                             openi_train_input_texts + openi_val_input_texts +
                             public_test_input_texts)
        train_output_texts = (mimiccxr_train_output_texts + mimiccxr_val_output_texts +
                              chexpert_train_output_texts + chexpert_val_output_texts +
                              openi_train_output_texts + openi_val_output_texts +
                              public_test_output_texts)
        train_image_idxs_list = (mimiccxr_train_image_idxs_list + mimiccxr_val_image_idxs_list +
                                 chexpert_train_image_idxs_list + chexpert_val_image_idxs_list +
                                 openi_train_image_idxs_list + openi_val_image_idxs_list +
                                 public_test_image_idxs_list)
    else:
        train_input_texts = (mimiccxr_train_input_texts + mimiccxr_val_input_texts +
                             chexpert_train_input_texts + chexpert_val_input_texts +
                             openi_train_input_texts + openi_val_input_texts)
        train_output_texts = (mimiccxr_train_output_texts + mimiccxr_val_output_texts +
                              chexpert_train_output_texts + chexpert_val_output_texts +
                              openi_train_output_texts + openi_val_output_texts)
        train_image_idxs_list = (mimiccxr_train_image_idxs_list + mimiccxr_val_image_idxs_list +
                                 chexpert_train_image_idxs_list + chexpert_val_image_idxs_list +
                                 openi_train_image_idxs_list + openi_val_image_idxs_list)

    if lowercase_output_texts:
        train_output_texts = [x.lower() for x in train_output_texts]
    
    # Aggregate labels for each training example
    aggregated_train_labels = np.zeros((len(train_output_texts), len(class_names)), dtype=int)
    for i, image_idxs in enumerate(train_image_idxs_list):
        aggregated_train_labels[i] = binary_predictions[image_idxs].max(axis=0) # max pooling

    # Use agregated_train_labels to create a balanced dataset
    train_indices = list(range(len(train_output_texts)))
    grouped_indices = group_indices_for_balanced_sampling(label_matrix=aggregated_train_labels,
                                                          indices=train_indices,
                                                          label_names=class_names,
                                                          min_group_size=100, verbose=verbose)
    train_datasets = []
    train_weights = []
    for indices in grouped_indices:
        dataset = Seq2SeqDataset(indices, train_input_texts, train_output_texts, shuffle=True, infinite=True)
        weight = math.log2(len(indices)) ** 3
        train_datasets.append(dataset)
        train_weights.append(weight)
        if verbose:
            print(f'  len(indices) = {len(indices)}, weight = {weight}')
    train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)

    # Create validation dataset

    val_input_texts = public_test_input_texts
    val_output_texts = public_test_output_texts

    if lowercase_output_texts:
        val_output_texts = [x.lower() for x in val_output_texts]

    val_dataset = Seq2SeqDataset(list(range(len(val_output_texts))), val_input_texts, val_output_texts, shuffle=False)
    
    return train_dataset, val_dataset


def _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=True, include_val=True,
                                    apply_uppercase_data_augmentation=False):
    indices = list(range(len(input_texts)))
    
    if apply_uppercase_data_augmentation:
        import re
        word_tokenized_input_texts = word_tokenize_texts_in_parallel(input_texts)
        lowercase_alpha_regex = re.compile(r'^[a-z]+$')
        lowercase_indices_per_input_text = [None] * len(input_texts)
        for i, tokens in enumerate(word_tokenized_input_texts):
            lowercase_indices_per_input_text[i] = [j for j, token in enumerate(tokens) if lowercase_alpha_regex.match(token)]
    else:
        word_tokenized_input_texts = None
        lowercase_indices_per_input_text = None

    if include_val:
        # Split intro train & val
        indices.sort(key=lambda i: (len(output_texts[i]), output_texts[i])) # sort by output length, then alphabetically
        
        # Choose val_size random indices uniformly distributed across the dataset
        val_indices = np.linspace(0, len(indices) - 1, val_size, dtype=int)
        val_indices_set = set(val_indices)
        train_indices = [i for i in indices if i not in val_indices_set]

        # Create datasets
        train_dataset = Seq2SeqDataset(train_indices, input_texts, output_texts, shuffle=True, infinite=True,
                                       apply_uppercase_data_augmentation=apply_uppercase_data_augmentation,
                                       word_tokenized_input_texts=word_tokenized_input_texts,
                                       lowercase_indices_per_input_text=lowercase_indices_per_input_text)
        val_dataset = Seq2SeqDataset(val_indices, input_texts, output_texts, shuffle=False,
                                        apply_uppercase_data_augmentation=apply_uppercase_data_augmentation,
                                        word_tokenized_input_texts=word_tokenized_input_texts,
                                        lowercase_indices_per_input_text=lowercase_indices_per_input_text)
    else:
        # Create datasets
        train_indices = indices
        train_dataset = Seq2SeqDataset(train_indices, input_texts, output_texts, shuffle=True, infinite=True,
                                        apply_uppercase_data_augmentation=apply_uppercase_data_augmentation,
                                        word_tokenized_input_texts=word_tokenized_input_texts,
                                        lowercase_indices_per_input_text=lowercase_indices_per_input_text)
        val_dataset = None

    if verbose:
        # Print stats
        print(f'Number of train examples: {len(train_indices)}')
        if include_val:
            print(f'Number of val examples: {len(val_indices)}')
            print(f'Number of total examples: {len(indices)}')

        # Print examples from the datasets
        for _ in range(3):
            i = random.randint(0, len(train_indices) - 1)
            tmp = train_dataset[i]
            input_text = tmp['input_text']
            output_text = tmp['output_text']
            print('--------')
            print_bold('Train example:')
            print_bold('Input:')
            print_magenta(input_text)
            print_bold('Output:')
            print_magenta(output_text)
            print()

        if include_val:
            for _ in range(3):
                i = random.randint(0, len(val_indices) - 1)
                tmp = val_dataset[i]
                input_text = tmp['input_text']
                output_text = tmp['output_text']
                print('--------')
                print_bold('Val example:')
                print_bold('Input:')
                print_magenta(input_text)
                print_bold('Output:')
                print_magenta(output_text)
                print()

    return train_dataset, val_dataset

def get_seq2seq_datasets_and_dataloaders(task_name, batch_size, collate_batch_fn, num_workers,
                                        fact_to_comparison_input_output_jsonl_filepaths=None,
                                        chest_imagenome_obs_input_output_jsonl_filepaths=None,
                                        chest_imagenome_anatloc_input_output_jsonl_filepaths=None,
                                        report_to_sentences_input_output_jsonl_filepaths=None,
                                        report_to_negative_facts_input_output_jsonl_filepaths=None,
                                        sentence_to_facts_input_output_jsonl_filepaths=None,
                                        background_to_facts_input_output_jsonl_filepaths=None,
                                        fact_to_metadata_input_output_jsonl_filepaths=None,
                                        fact_to_metadata_v2_input_output_jsonl_filepaths=None,
                                        integrated_facts_metadata_jsonl_filepath=None,
                                        paraphrased_inputs_jsonl_filepaths=None,
                                        chest_imagenome_phrases2labels_filepath=None,
                                        multitask_name_list=None,
                                        task2weight=None,
                                        integrated_nli_jsonl_filepath=None,
                                        use_sentence2facts_for_nli=False,
                                        use_anli=False, use_multinli=False, use_snli=False,
                                        use_report_nli=False,
                                        raw_report_nli_input_output_train_jsonl_filepaths=None,
                                        report_nli_input_output_train_jsonl_filepaths=None,
                                        report_nli_input_output_val_jsonl_filepaths=None,
                                        use_fact_based_reports_in_mlm=False,
                                        use_report_nli_entailment_dataset=False,
                                        use_report_nli_paraphrases_dataset=False,
                                        integrated_report_facts_jsonl_filepath=None,
                                        mlm_min_token_count=20, mlm_masking_fraction=0.2,
                                        only_validate_nli=False, nli1_only_on_train=False, nli1_only_on_val=False,
                                        interpret_cxr__label_based_predictions_filepath=None,
                                        interpret_cxr_challenge_data_dir=None,
                                        mimiccxr_integrated_report_nli_data_filepath=None,
                                        report_section_to_generate=None,
                                        include_public_test_in_train=False,
                                        best_k_classes=None,
                                        use_numeric_templates=False,
                                        filter_for_t5=False,
                                        val_size=200, verbose=True):

    assert task_name in Seq2SeqTaskNames.get_all(), f'Unknown task name {task_name}'

    if use_sentence2facts_for_nli:
        assert task_name == Seq2SeqTaskNames.MULTITASK, 'use_sentence2facts_for_nli can only be used with multitask'
        assert Seq2SeqTaskNames.SENTENCE_TO_FACTS in multitask_name_list
        assert Seq2SeqTaskNames.NLI in multitask_name_list

    if only_validate_nli or nli1_only_on_val:
        assert task_name == Seq2SeqTaskNames.NLI or (
            task_name == Seq2SeqTaskNames.MULTITASK and Seq2SeqTaskNames.NLI in multitask_name_list), \
            'only_validate_nli and nli1_only_on_val can only be used with NLI or multitask with NLI'

    input2paraphrases = _compute_input2paraphrases(paraphrased_inputs_jsonl_filepaths, verbose=verbose)

    if task_name == Seq2SeqTaskNames.FACT_TO_COMPARISON:
        input_texts, output_texts = _prepare_fact_to_comparison_data(
            integrated_facts_metadata_jsonl_filepath, fact_to_comparison_input_output_jsonl_filepaths, input2paraphrases)
        train_dataset, val_dataset = _get_fact_to_comparison_train_val_datasets(
            input_texts, output_texts, val_size, verbose=verbose)

    elif task_name == Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_OBSERVATIONS:
        input_texts, output_texts, _label_names = _prepare_sentence_to_chest_imagenome_observations_data(
            chest_imagenome_phrases2labels_filepath, chest_imagenome_obs_input_output_jsonl_filepaths, input2paraphrases)
        train_dataset, val_dataset = _get_sentence_to_chest_imagenome_labels_datasets(
            input_texts, output_texts, val_size, label_names=_label_names, verbose=verbose)

    elif task_name == Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_ANATOMICAL_LOCATIONS:
        input_texts, output_texts, _label_names = _prepare_sentence_to_chest_imagenome_anatlocs_data(
            chest_imagenome_phrases2labels_filepath, chest_imagenome_anatloc_input_output_jsonl_filepaths, input2paraphrases)
        train_dataset, val_dataset = _get_sentence_to_chest_imagenome_labels_datasets(
            input_texts, output_texts, val_size, label_names=_label_names, verbose=verbose)

    elif task_name == Seq2SeqTaskNames.FACT_TO_METADATA:
        input_texts, output_texts = _prepare_fact_to_metadata_data(fact_to_metadata_input_output_jsonl_filepaths)
        train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose)

    elif task_name == Seq2SeqTaskNames.FACT_TO_METADATA_V2:
        input_texts, output_texts = _prepare_fact_to_metadata_v2_data(fact_to_metadata_v2_input_output_jsonl_filepaths)
        train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose)
    
    elif task_name == Seq2SeqTaskNames.REPORT_TO_SENTENCES:
        input_texts, output_texts = _prepare_reports_to_sentences_data(report_to_sentences_input_output_jsonl_filepaths)
        train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose)
                                                                     

    elif task_name == Seq2SeqTaskNames.REPORT_TO_NEGATIVE_FACTS:
        train_dataset, val_dataset = _prepare_reports_to_negative_facts_data(report_to_negative_facts_input_output_jsonl_filepaths,
                                                                             filter_for_t5=filter_for_t5)

    elif task_name == Seq2SeqTaskNames.SENTENCE_TO_FACTS:
        input_texts, output_texts = _prepare_sentence_to_facts_data(sentence_to_facts_input_output_jsonl_filepaths,
                                                                    concatenate_pairs=True)
        train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose,
                                                                     apply_uppercase_data_augmentation=True)

    elif task_name == Seq2SeqTaskNames.BACKGROUND_TO_FACTS:
        input_texts, output_texts = _prepare_background_to_facts_data(background_to_facts_input_output_jsonl_filepaths)
        train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose)

    elif task_name == Seq2SeqTaskNames.NLI:
        train_dataset, val_dataset = _prepare_nli_data(integrated_nli_jsonl_filepath, None,
                          use_anli, use_multinli, use_snli, use_report_nli,
                          raw_report_nli_input_output_train_jsonl_filepaths,
                          report_nli_input_output_train_jsonl_filepaths,
                          report_nli_input_output_val_jsonl_filepaths,
                          use_report_nli_entailment_dataset=use_report_nli_entailment_dataset,
                          use_report_nli_paraphrases_dataset=use_report_nli_paraphrases_dataset,
                          integrated_report_facts_jsonl_filepath=integrated_report_facts_jsonl_filepath,
                          input_to_paraphrases=input2paraphrases,
                          verbose=verbose, nli1_only_on_train=nli1_only_on_train, nli1_only_on_val=nli1_only_on_val)
        
    elif task_name == Seq2SeqTaskNames.FACT_CLASSIFIER_PREDICTIONS_TO_REPORT_SECTION:
        assert interpret_cxr__label_based_predictions_filepath is not None
        assert interpret_cxr_challenge_data_dir is not None
        assert mimiccxr_integrated_report_nli_data_filepath is not None
        assert report_section_to_generate in ['findings', 'impression']
        train_dataset, val_dataset = _get_fact_classifier_predictions_to_report_section_datasets__interpret_cxr_challenge(
            interpret_cxr__label_based_predictions_filepath=interpret_cxr__label_based_predictions_filepath,
            interpret_cxr_challenge_data_dir=interpret_cxr_challenge_data_dir,
            mimiccxr_integrated_report_nli_data_filepath=mimiccxr_integrated_report_nli_data_filepath,
            section=report_section_to_generate,
            lowercase_output_texts=True,
            include_public_test_in_train=include_public_test_in_train,
            verbose=verbose, best_k_classes=best_k_classes, use_numeric_templates=use_numeric_templates)
        
    elif task_name == Seq2SeqTaskNames.MULTITASK:
        assert multitask_name_list is not None, 'multitask_name_list must be provided'
        assert type(multitask_name_list) == list
        assert len(multitask_name_list) > 0
        assert len(set(multitask_name_list)) == len(multitask_name_list)
        assert all(task_name in Seq2SeqTaskNames.get_all() for task_name in multitask_name_list), \
            f'Unknown task name in multitask_name_list: {multitask_name_list}'
        assert task2weight is not None, 'task2weight must be provided'
        assert set(multitask_name_list) == set(task2weight.keys()), \
            f'Inconsistent task names in multitask_name_list and task2weight: {multitask_name_list}, {task2weight.keys()}'

        if use_sentence2facts_for_nli:
            # make sure SENTENCE_TO_FACTS is the first task
            multitask_name_list = [Seq2SeqTaskNames.SENTENCE_TO_FACTS] + [x for x in multitask_name_list if x != Seq2SeqTaskNames.SENTENCE_TO_FACTS]

        if Seq2SeqTaskNames.MLM in multitask_name_list:
            # make sure MLM is the last task
            multitask_name_list = [x for x in multitask_name_list if x != Seq2SeqTaskNames.MLM] + [Seq2SeqTaskNames.MLM]
            general_sentences = set()
            medical_sentences = set()

            if input2paraphrases is not None:
                for input_text, paraphrases in input2paraphrases.items():
                    medical_sentences.add(input_text)
                    for p in paraphrases:
                        medical_sentences.add(p)
                print(f'Number of medical sentences from paraphrases: {len(medical_sentences)}')

        else:
            general_sentences = None
            medical_sentences = None
        
        train_datasets = []
        train_weights = []
        val_datasets = []
        
        for task_name in multitask_name_list:
            print_bold('----------------------------------------')
            print_bold(f'Preparing {task_name} dataset')

            include_val = task_name != Seq2SeqTaskNames.MLM and (not only_validate_nli or task_name == Seq2SeqTaskNames.NLI)

            if task_name == Seq2SeqTaskNames.FACT_TO_COMPARISON:
                input_texts, output_texts = _prepare_fact_to_comparison_data(
                    integrated_facts_metadata_jsonl_filepath, fact_to_comparison_input_output_jsonl_filepaths, input2paraphrases,
                    apply_task_prefix=True, medical_sentences=medical_sentences)
                train_dataset, val_dataset = _get_fact_to_comparison_train_val_datasets(
                    input_texts, output_texts, val_size, verbose=verbose, include_val=include_val)

            elif task_name == Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_OBSERVATIONS:
                input_texts, output_texts, _label_names = _prepare_sentence_to_chest_imagenome_observations_data(
                    chest_imagenome_phrases2labels_filepath, chest_imagenome_obs_input_output_jsonl_filepaths, input2paraphrases,
                    apply_task_prefix=True, medical_sentences=medical_sentences)
                train_dataset, val_dataset = _get_sentence_to_chest_imagenome_labels_datasets(
                    input_texts, output_texts, val_size, label_names=_label_names, verbose=verbose, include_val=include_val)

            elif task_name == Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_ANATOMICAL_LOCATIONS:
                input_texts, output_texts, _label_names = _prepare_sentence_to_chest_imagenome_anatlocs_data(
                    chest_imagenome_phrases2labels_filepath, chest_imagenome_anatloc_input_output_jsonl_filepaths, input2paraphrases,
                    apply_task_prefix=True, medical_sentences=medical_sentences)
                train_dataset, val_dataset = _get_sentence_to_chest_imagenome_labels_datasets(
                    input_texts, output_texts, val_size, label_names=_label_names, verbose=verbose, include_val=include_val)

            elif task_name == Seq2SeqTaskNames.FACT_TO_METADATA:
                input_texts, output_texts = _prepare_fact_to_metadata_data(fact_to_metadata_input_output_jsonl_filepaths,
                                                                            apply_task_prefix=True, medical_sentences=medical_sentences)
                train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose,
                                                                             include_val=include_val)
                
            elif task_name == Seq2SeqTaskNames.FACT_TO_METADATA_V2:
                input_texts, output_texts = _prepare_fact_to_metadata_v2_data(fact_to_metadata_v2_input_output_jsonl_filepaths,
                                                                            apply_task_prefix=True, medical_sentences=medical_sentences)
                train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose,
                                                                             include_val=include_val)
            
            elif task_name == Seq2SeqTaskNames.REPORT_TO_SENTENCES:
                input_texts, output_texts = _prepare_reports_to_sentences_data(report_to_sentences_input_output_jsonl_filepaths,
                                                                                apply_task_prefix=True)
                train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose,
                                                                             include_val=include_val)
                
            elif task_name == Seq2SeqTaskNames.REPORT_TO_NEGATIVE_FACTS:
                train_dataset, val_dataset = _prepare_reports_to_negative_facts_data(report_to_negative_facts_input_output_jsonl_filepaths,
                                                                                apply_task_prefix=True, filter_for_t5=filter_for_t5)

            elif task_name == Seq2SeqTaskNames.SENTENCE_TO_FACTS:
                if use_sentence2facts_for_nli:
                    input_texts, output_texts, s2f_input_output_for_nli = _prepare_sentence_to_facts_data(
                        sentence_to_facts_input_output_jsonl_filepaths, apply_task_prefix=True, collect_input_output_for_nli=True,
                        medical_sentences=medical_sentences, concatenate_pairs=True)
                else:
                    input_texts, output_texts = _prepare_sentence_to_facts_data(sentence_to_facts_input_output_jsonl_filepaths,
                                                                                apply_task_prefix=True,
                                                                                medical_sentences=medical_sentences,
                                                                                concatenate_pairs=True)
                    
                train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose,
                                                                             include_val=include_val, apply_uppercase_data_augmentation=True)

            elif task_name == Seq2SeqTaskNames.BACKGROUND_TO_FACTS:
                input_texts, output_texts = _prepare_background_to_facts_data(background_to_facts_input_output_jsonl_filepaths,
                                                                                apply_task_prefix=True)
                train_dataset, val_dataset = _get_general_train_val_datasets(input_texts, output_texts, val_size, verbose=verbose,
                                                                             include_val=include_val)

            elif task_name == Seq2SeqTaskNames.NLI:
                if use_sentence2facts_for_nli:
                    s2f_aux = s2f_input_output_for_nli
                else:
                    s2f_aux = None
                train_dataset, val_dataset = _prepare_nli_data(integrated_nli_jsonl_filepath, s2f_aux,
                                                                use_anli, use_multinli, use_snli, verbose=verbose,
                                                                medical_sentences=medical_sentences,
                                                                general_sentences=general_sentences,
                                                                use_report_nli=use_report_nli,
                                                                use_report_nli_entailment_dataset=use_report_nli_entailment_dataset,
                                                                use_report_nli_paraphrases_dataset=use_report_nli_paraphrases_dataset,
                                                                input_to_paraphrases=input2paraphrases,
                                                                report_nli_input_output_train_jsonl_filepaths=report_nli_input_output_train_jsonl_filepaths,
                                                                report_nli_input_output_val_jsonl_filepaths=report_nli_input_output_val_jsonl_filepaths,
                                                                integrated_report_facts_jsonl_filepath=integrated_report_facts_jsonl_filepath,
                                                                nli1_only_on_train=nli1_only_on_train, nli1_only_on_val=nli1_only_on_val)
                    
            elif task_name == Seq2SeqTaskNames.MLM:
                mlm_datasets = []
                if general_sentences is not None:
                    general_sentences = list(general_sentences)
                    print(f'Number of general sentences: {len(general_sentences)}')
                    general_mlm_dataset = _prepare_masked_lm_dataset(
                        sentences=general_sentences, apply_task_prefix=True, verbose=verbose,
                        min_token_count=mlm_min_token_count, masking_fraction=mlm_masking_fraction)
                    mlm_datasets.append(general_mlm_dataset)
                if medical_sentences is not None:
                    if use_fact_based_reports_in_mlm:
                        assert integrated_report_facts_jsonl_filepath is not None
                        rows = load_jsonl(integrated_report_facts_jsonl_filepath)
                        print(f"Loaded {len(rows)} reports from {integrated_report_facts_jsonl_filepath}")
                        for row in rows:
                            medical_sentences.add(row['fact_based_report'])
                            medical_sentences.update(row['facts'])
                    medical_sentences = list(medical_sentences)
                    print(f'Number of medical sentences: {len(medical_sentences)}')
                    medical_mlm_dataset = _prepare_masked_lm_dataset(
                        sentences=medical_sentences, apply_task_prefix=True, verbose=verbose,
                        min_token_count=mlm_min_token_count, masking_fraction=mlm_masking_fraction)
                    mlm_datasets.append(medical_mlm_dataset)
                assert len(mlm_datasets) > 0
                val_dataset = None # no val dataset for MLM
                if len(mlm_datasets) == 1:
                    train_dataset = mlm_datasets[0]
                else:
                    train_dataset = CompositeInfiniteDataset(mlm_datasets, [1] * len(mlm_datasets))
            
            else:
                raise ValueError(f'Unknown task name {task_name}')
            
            train_datasets.append(train_dataset)
            train_weights.append(task2weight[task_name])
            assert include_val == (val_dataset is not None)
            if val_dataset is not None:
                val_datasets.append(val_dataset)

        assert len(train_datasets) > 0
        assert len(val_datasets) > 0
        print('----------------------------------------')
        print(f'Number of train datasets: {len(train_datasets)}')
        print(f'Number of val datasets: {len(val_datasets)}')
        train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
        val_dataset = CompositeDataset(val_datasets)
        print('----------------------------------------')
        print('Examples of val datasets:')
        for _ in range(4):
            i = random.randint(0, len(val_dataset) - 1)
            tmp = val_dataset[i]
            _print_input_output_pair(tmp['input_text'], tmp['output_text'])

    else:
        raise ValueError(f'Unknown task name {task_name}')

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch_fn,
        pin_memory=True,
    )

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
    }

class Seq2SeqTrainer():

    def __init__(self, batch_size, collate_batch_fn, num_workers, task_name, experiment_name,
                 integrated_facts_metadata_jsonl_filepath=None,
                 paraphrased_inputs_jsonl_filepaths=None,
                 chest_imagenome_phrases2labels_filepath=None,
                 sentence_to_facts_input_output_jsonl_filepaths=None,
                 fact_to_metadata_input_output_jsonl_filepaths=None,
                 fact_to_metadata_v2_input_output_jsonl_filepaths=None,
                 fact_to_comparison_input_output_jsonl_filepaths=None,
                 report_to_negative_facts_input_output_jsonl_filepaths=None,
                 chest_imagenome_obs_input_output_jsonl_filepaths=None,
                 chest_imagenome_anatloc_input_output_jsonl_filepaths=None,
                 integrated_nli_jsonl_filepath=None,
                 use_sentence2facts_for_nli=False,
                 use_anli=False, use_multinli=False, use_snli=False,
                 use_report_nli=False,
                 raw_report_nli_input_output_train_jsonl_filepaths=None,
                 report_nli_input_output_train_jsonl_filepaths=None,
                 report_nli_input_output_val_jsonl_filepaths=None,
                 multitask_name_list=None, task2weight=None,
                 use_fact_based_reports_in_mlm=False,
                 use_report_nli_entailment_dataset=False,
                 use_report_nli_paraphrases_dataset=False,
                 integrated_report_facts_jsonl_filepath=None,
                 mlm_min_token_count=20, mlm_masking_fraction=0.2,
                 only_validate_nli=False, nli1_only_on_train=False, nli1_only_on_val=False,
                 interpret_cxr__label_based_predictions_filepath=None,
                 interpret_cxr_challenge_data_dir=None,
                 mimiccxr_integrated_report_nli_data_filepath=None,
                 report_section_to_generate=None,
                 include_public_test_in_train=False,
                 best_k_classes=None,
                 use_numeric_templates=False,
                 filter_for_t5=False,
                 val_size=200, verbose=True):

        assert task_name in Seq2SeqTaskNames.get_all(), f'Unknown task name {task_name}'
        assert experiment_name is not None, 'experiment_name must be provided'
        
        self.batch_size = batch_size
        self.collate_batch_fn = collate_batch_fn
        self.num_workers = num_workers
        self.task_name = task_name
        self.experiment_name = experiment_name

        tmp = get_seq2seq_datasets_and_dataloaders(
            task_name, batch_size, collate_batch_fn, num_workers,
            integrated_facts_metadata_jsonl_filepath=integrated_facts_metadata_jsonl_filepath,
            paraphrased_inputs_jsonl_filepaths=paraphrased_inputs_jsonl_filepaths,
            chest_imagenome_phrases2labels_filepath=chest_imagenome_phrases2labels_filepath,
            sentence_to_facts_input_output_jsonl_filepaths=sentence_to_facts_input_output_jsonl_filepaths,
            fact_to_metadata_input_output_jsonl_filepaths=fact_to_metadata_input_output_jsonl_filepaths,
            fact_to_metadata_v2_input_output_jsonl_filepaths=fact_to_metadata_v2_input_output_jsonl_filepaths,
            report_to_negative_facts_input_output_jsonl_filepaths=report_to_negative_facts_input_output_jsonl_filepaths,
            fact_to_comparison_input_output_jsonl_filepaths=fact_to_comparison_input_output_jsonl_filepaths,
            chest_imagenome_obs_input_output_jsonl_filepaths=chest_imagenome_obs_input_output_jsonl_filepaths,
            chest_imagenome_anatloc_input_output_jsonl_filepaths=chest_imagenome_anatloc_input_output_jsonl_filepaths,
            integrated_nli_jsonl_filepath=integrated_nli_jsonl_filepath,
            use_sentence2facts_for_nli=use_sentence2facts_for_nli,
            use_anli=use_anli, use_multinli=use_multinli, use_snli=use_snli,
            use_report_nli=use_report_nli,
            raw_report_nli_input_output_train_jsonl_filepaths=raw_report_nli_input_output_train_jsonl_filepaths,
            report_nli_input_output_train_jsonl_filepaths=report_nli_input_output_train_jsonl_filepaths,
            report_nli_input_output_val_jsonl_filepaths=report_nli_input_output_val_jsonl_filepaths,
            multitask_name_list=multitask_name_list, task2weight=task2weight,
            use_fact_based_reports_in_mlm=use_fact_based_reports_in_mlm,
            use_report_nli_entailment_dataset=use_report_nli_entailment_dataset,
            use_report_nli_paraphrases_dataset=use_report_nli_paraphrases_dataset,
            integrated_report_facts_jsonl_filepath=integrated_report_facts_jsonl_filepath,
            mlm_min_token_count=mlm_min_token_count, mlm_masking_fraction=mlm_masking_fraction,
            only_validate_nli=only_validate_nli, nli1_only_on_train=nli1_only_on_train, nli1_only_on_val=nli1_only_on_val,
            interpret_cxr__label_based_predictions_filepath=interpret_cxr__label_based_predictions_filepath,
            interpret_cxr_challenge_data_dir=interpret_cxr_challenge_data_dir,
            mimiccxr_integrated_report_nli_data_filepath=mimiccxr_integrated_report_nli_data_filepath,
            report_section_to_generate=report_section_to_generate,
            include_public_test_in_train=include_public_test_in_train,
            filter_for_t5=filter_for_t5,
            val_size=val_size, verbose=verbose, best_k_classes=best_k_classes, use_numeric_templates=use_numeric_templates)
        
        self.train_dataset = tmp['train_dataset']
        self.val_dataset = tmp['val_dataset']
        self.train_dataloader = tmp['train_dataloader']
        self.val_dataloader = tmp['val_dataloader']

    @property
    def name(self):
        return f'{self.task_name}({self.experiment_name})'