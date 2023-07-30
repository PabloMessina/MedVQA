import numpy as np
import math
import random
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH, CompositeInfiniteDataset
from medvqa.utils.data_structures import UnionFind
from medvqa.utils.files import load_jsonl, load_pickle
from medvqa.utils.logging import print_bold, print_magenta, print_orange

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
    SENTENCE_TO_FACTS = 'sentence2facts'
    BACKGROUND_TO_FACTS = 'background2facts'
    FACT_TO_METADATA = 'fact2metadata'
    FACT_TO_COMPARISON = 'fact2comparison'
    SENTENCE_TO_CHEST_IMAGENOME_LABELS = 'sentence2chestimagenome_labels'
    @staticmethod
    def get_all():
        return [
            Seq2SeqTaskNames.REPORT_TO_SENTENCES,
            Seq2SeqTaskNames.SENTENCE_TO_FACTS,
            Seq2SeqTaskNames.BACKGROUND_TO_FACTS,
            Seq2SeqTaskNames.FACT_TO_METADATA,
            Seq2SeqTaskNames.FACT_TO_COMPARISON,
            Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_LABELS,
        ]

class Seq2SeqDataset(Dataset):
    def __init__(self, indices, input_texts, output_texts, shuffle=False, infinite=False):
        self.indices = indices
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.infinite = infinite
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)
        if shuffle:
            random.shuffle(self.indices) # shuffle in place
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self.infinite:
            i = i % len(self.indices)
        idx = self.indices[i]
        output = { 'idx': idx, 'input_text': self.input_texts[idx], 'output_text': self.output_texts[idx] }
        return output

def get_seq2seq_datasets_and_dataloaders(input_output_jsonl_filepaths, task_name, batch_size, collate_batch_fn, num_workers,
                                         integrated_facts_metadata_jsonl_filepath=None,
                                         paraphrased_inputs_jsonl_filepaths=None,
                                         chest_imagenome_phrases2labels_filepath=None,
                                         val_size=200, verbose=True):
    assert task_name in Seq2SeqTaskNames.get_all(), f'Unknown task name {task_name}'
    # Load input and output texts
    input_texts = []
    output_texts = []

    if task_name == Seq2SeqTaskNames.FACT_TO_COMPARISON:
        assert integrated_facts_metadata_jsonl_filepath is not None, 'integrated_facts_metadata_jsonl_filepath must be provided'
        integrated_facts_metadata = load_jsonl(integrated_facts_metadata_jsonl_filepath)
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

    if task_name == Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_LABELS:
        assert chest_imagenome_phrases2labels_filepath is not None, 'chest_imagenome_phrases2labels_filepath must be provided'
        chest_imagenome_phrases2labels = load_pickle(chest_imagenome_phrases2labels_filepath)
        phrases = chest_imagenome_phrases2labels['phrases']
        labels = chest_imagenome_phrases2labels['labels']
        label_names = chest_imagenome_phrases2labels['label_names']
        # Remove nlp labels and prefixes
        _idxs = [i for i, x in enumerate(label_names) if not x.startswith('nlp|')]
        assert len(_idxs) + 2 == len(label_names) # 2 nlp labels
        labels = labels[:, _idxs]
        label_names = [label_names[i][label_names[i].index('|') + 1:] for i in _idxs]
        n, m = labels.shape
        assert len(phrases) == n
        assert len(label_names) == m
        for i in tqdm(range(n), total=n, mininterval=2):
            input_texts.append(phrases[i])
            output_text = json.dumps([label_names[j] for j in range(m) if labels[i, j] == 1])
            output_texts.append(output_text)
        if verbose:
            print(f'Loaded {len(input_texts)} input/output pairs from {chest_imagenome_phrases2labels_filepath}')
            print('Examples:')
            for _ in range(3):
                i = random.randint(0, n - 1)
                print_bold(f'Input:')
                print_magenta(input_texts[i], bold=True)
                print_bold(f'Output:')
                print_magenta(output_texts[i], bold=True)

    for input_output_jsonl_filepath in input_output_jsonl_filepaths:
        input_output_jsonl = load_jsonl(input_output_jsonl_filepath)
        if verbose:
            print(f'Loaded {len(input_output_jsonl)} input/output pairs from {input_output_jsonl_filepath}')
        if task_name == Seq2SeqTaskNames.REPORT_TO_SENTENCES:
            for input_output in input_output_jsonl:
                report = input_output['metadata']['report']
                input_text = '\n'.join([report['findings'], report['impression']])
                output_text = json.dumps([f'{s} {"#pos" if p else "#neg"}' for s, p in input_output['parsed_response']])
                input_texts.append(input_text)
                output_texts.append(output_text)
        elif task_name == Seq2SeqTaskNames.SENTENCE_TO_FACTS:
            for input_output in input_output_jsonl:
                sentence = input_output['metadata']['sentence']
                input_text = sentence
                output_text = json.dumps(input_output['parsed_response'])
                input_texts.append(input_text)
                output_texts.append(output_text)
        elif task_name == Seq2SeqTaskNames.BACKGROUND_TO_FACTS:
            for input_output in input_output_jsonl:
                background = input_output['metadata']['background']
                input_text = background
                output_text = json.dumps(input_output['parsed_response'])
                input_texts.append(input_text)
                output_texts.append(output_text)
        elif task_name == Seq2SeqTaskNames.FACT_TO_METADATA:
            for input_output in input_output_jsonl:
                fact = input_output['metadata']['fact']
                input_text = fact
                output_text = json.dumps(input_output['parsed_response'])
                input_texts.append(input_text)
                output_texts.append(output_text)
        elif task_name == Seq2SeqTaskNames.FACT_TO_COMPARISON:
            for input_output in input_output_jsonl:
                fact = input_output['metadata']['sentence']
                input_text = fact
                output_text = input_output['parsed_response']
                input_texts.append(input_text)
                output_texts.append(output_text)
        elif task_name == Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_LABELS:
            for input_output in input_output_jsonl:
                sentence = input_output['metadata']['query']
                input_text = sentence
                output_text = json.dumps(input_output['parsed_response'])
                input_texts.append(input_text)
                output_texts.append(output_text)
        else:
            raise ValueError(f'Unknown task name {task_name}')
        
    # Add paraphrased inputs (if provided)
    if paraphrased_inputs_jsonl_filepaths is not None:
        assert type(paraphrased_inputs_jsonl_filepaths) == list
        input2paraphrases = {}
        for filepath in paraphrased_inputs_jsonl_filepaths:
            paraphrased_inputs = load_jsonl(filepath)
            if verbose:
                print('--------')
                print_bold(f'Loaded {len(paraphrased_inputs)} paraphrased inputs from {filepath}')
                print_count = 0
            for row in paraphrased_inputs:
                input_text = next(iter(row['metadata'].values()))
                paraphrases = row['parsed_response']
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
        count = 0
        print_count = 0
        for i in range(len(input_texts)):
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
                if verbose and len(paraphrases) > 0 and print_count < 20:
                    print_count += 1
                    print_bold(f'Input:')
                    print_magenta(input_texts[-1], bold=True)
                    print_bold(f'Output:')
                    print_magenta(output_texts[-1], bold=True)
        if verbose:
            print('--------')
            print(f'Added {count} paraphrased inputs')
            print_bold(f'Number of total examples: {len(input_texts)}')
    
    if task_name == Seq2SeqTaskNames.FACT_TO_COMPARISON:

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
            assert val_sizes[i] * 10 <= len(indices), f'val_size {val_sizes[i]} is too large for output {output} (len {len(indices)})'
            indices.sort(key=lambda i: (len(output_texts[i]), output_texts[i])) # sort by output length, then alphabetically
            val_indices_set.update(indices[j] for j in np.linspace(0, len(indices) - 1, val_sizes[i], dtype=int))
            train_indices = [i for i in indices if i not in val_indices_set]
            train_datasets.append(Seq2SeqDataset(train_indices, input_texts, output_texts, shuffle=True, infinite=True))
            train_weights.append(math.log2(len(train_indices))**2) # weight by log^2 of number of examples
            train_size += len(train_indices)
            if verbose:
                print(f'Output: {output}, train size: {len(train_indices)}, val size: {val_sizes[i]}, weight: {train_weights[-1]}')

        val_indices = list(val_indices_set)
        
        # Create datasets
        train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
        val_dataset = Seq2SeqDataset(val_indices, input_texts, output_texts, shuffle=False)

        if verbose:
            # Print stats
            print(f'Number of train examples: {train_size}')
            print(f'Number of val examples: {len(val_indices)}')
            print(f'Number of total examples: {train_size + len(val_indices)}')

    elif task_name == Seq2SeqTaskNames.SENTENCE_TO_CHEST_IMAGENOME_LABELS:

        label_name_2_idx = { label_name: i for i, label_name in enumerate(label_names) }
        assert len(label_name_2_idx) == len(label_names)
        label2idxs = [[] for _ in range(len(label_name_2_idx))]
        idxs_without_labels = []
        unknown_labels = set()
        unknown_count = 0
        for i, output_text in tqdm(enumerate(output_texts), total=len(output_texts), mininterval=2):
            labels = json.loads(output_text)
            label_found = False
            for label in labels:
                try:
                    label2idxs[label_name_2_idx[label]].append(i)
                    label_found = True
                except KeyError:
                    unknown_count += 1
                    unknown_labels.add(label)
            if not label_found:
                idxs_without_labels.append(i)

        if verbose:
            print(f'Number of total examples: {len(output_texts)}')
            print(f'Number of examples without labels: {len(idxs_without_labels)}')
            if unknown_count > 0:
                print_orange(f'WARNING: Number of unknown labels: {unknown_count}', bold=True)
                unknown_labels = list(unknown_labels)
                print_orange('Some unknown labels:', bold=True)
                for x in random.sample(unknown_labels, min(10, len(unknown_labels))):
                    print_orange(f'  {x}', bold=True)

        # Split intro train & val
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
        print_messages = []
        train_size = 0
        for i, idxs in enumerate(label2idxs):
            assert val_sizes[i] * 10 <= len(idxs), f'val_size {val_sizes[i]} is too large for label {label_names[i]} (len {len(idxs)})'
            val_idxs = random.sample(idxs, val_sizes[i])
            val_idxs_set.update(val_idxs)
        val_idxs_set.update(random.sample(idxs_without_labels, val_sizes[-1]))
        for i, idxs in enumerate(label2idxs):
            train_idxs = [i for i in idxs if i not in val_idxs_set]
            train_datasets.append(Seq2SeqDataset(train_idxs, input_texts, output_texts, shuffle=True, infinite=True))
            train_weights.append(math.log2(len(train_idxs))**3) # weight by log^3 of number of examples
            train_size += len(train_idxs)
            print_messages.append((
                len(train_idxs),
                f'Label: {label_names[i]}, train size: {len(train_idxs)}, val size: {val_sizes[i]}, weight: {train_weights[-1]}'
            ))
        train_idxs = [i for i in idxs_without_labels if i not in val_idxs_set]
        train_datasets.append(Seq2SeqDataset(train_idxs, input_texts, output_texts, shuffle=True, infinite=True))
        train_weights.append(math.log2(len(train_idxs))**3) # weight by log^3 of number of examples
        train_size += len(train_idxs)
        print_messages.append((
            len(train_idxs),
            f'Label: None, train size: {len(train_idxs)}, val size: {val_sizes[-1]}, weight: {train_weights[-1]}'
        ))
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
            print(f'Number of val examples: {len(val_idxs)}')
            print(f'Number of total examples: {train_size + len(val_idxs)}')
            print(f'Number of train datasets: {len(train_datasets)}')

        # Create datasets
        train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
        val_dataset = Seq2SeqDataset(val_idxs, input_texts, output_texts, shuffle=False)
        
    else:        
    
        # Split intro train & val
        indices = list(range(len(input_texts)))
        indices.sort(key=lambda i: (len(output_texts[i]), output_texts[i])) # sort by output length, then alphabetically

        # Choose val_size random indices uniformly distributed across the dataset
        val_indices = np.linspace(0, len(indices) - 1, val_size, dtype=int)
        val_indices_set = set(val_indices)
        train_indices = [i for i in indices if i not in val_indices_set]

        # Create datasets
        train_dataset = Seq2SeqDataset(train_indices, input_texts, output_texts, shuffle=True, infinite=True)
        val_dataset = Seq2SeqDataset(val_indices, input_texts, output_texts, shuffle=False)
    
        if verbose:
            # Print stats
            print_bold(f'Number of train examples: {len(train_indices)}')
            print_bold(f'Number of val examples: {len(val_indices)}')
            print_bold(f'Number of total examples: {len(indices)}')
        
    if verbose:
        # Print random example
        print_bold('Input/output example:')
        i = random.randint(0, len(input_texts) - 1)
        print_bold(f'Input:')
        print_magenta(input_texts[i], bold=True)
        print_bold(f'Output:')
        print_magenta(output_texts[i], bold=True)

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

    def __init__(self, batch_size, collate_batch_fn, num_workers, input_output_jsonl_filepaths, task_name,
                 integrated_facts_metadata_jsonl_filepath=None,
                 paraphrased_inputs_jsonl_filepaths=None,
                 chest_imagenome_phrases2labels_filepath=None,
                 val_size=200, verbose=True):

        assert task_name in Seq2SeqTaskNames.get_all(), f'Unknown task name {task_name}'
        assert input_output_jsonl_filepaths is not None, 'input_output_jsonl_filepaths must be provided'
        
        self.batch_size = batch_size
        self.collate_batch_fn = collate_batch_fn
        self.num_workers = num_workers
        self.input_output_jsonl_filepaths = input_output_jsonl_filepaths
        self.task_name = task_name

        tmp = get_seq2seq_datasets_and_dataloaders(
            input_output_jsonl_filepaths, task_name, batch_size, collate_batch_fn, num_workers,
            integrated_facts_metadata_jsonl_filepath=integrated_facts_metadata_jsonl_filepath,
            paraphrased_inputs_jsonl_filepaths=paraphrased_inputs_jsonl_filepaths,
            chest_imagenome_phrases2labels_filepath=chest_imagenome_phrases2labels_filepath,
            val_size=val_size, verbose=verbose)
        self.train_dataset = tmp['train_dataset']
        self.val_dataset = tmp['val_dataset']
        self.train_dataloader = tmp['train_dataloader']
        self.val_dataloader = tmp['val_dataloader']

    @property
    def name(self):
        filenames = [os.path.basename(filepath) for filepath in self.input_output_jsonl_filepaths]
        short_filenames = [filename[:filename.rfind('.')] for filename in filenames]
        for i in range(len(short_filenames)):
            x = short_filenames[i]
            if len(x) > 16:
                short_filenames[i] = x[:8] + '..' + x[-6:]
        filenames_str = ';'.join(short_filenames)
        return f'{self.task_name}({filenames_str})'