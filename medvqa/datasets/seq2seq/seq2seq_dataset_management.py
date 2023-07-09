import numpy as np
import random
import os
import json
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH
from medvqa.utils.files import load_jsonl
from medvqa.utils.logging import print_bold, print_magenta

class Seq2SeqTaskNames:
    REPORT_TO_SENTENCES = 'report2sentences'
    SENTENCE_TO_FACTS = 'sentence2facts'
    BACKGROUND_TO_FACTS = 'background2facts'
    FACT_TO_METADATA = 'fact2metadata'
    @staticmethod
    def get_all():
        return [
            Seq2SeqTaskNames.REPORT_TO_SENTENCES,
            Seq2SeqTaskNames.SENTENCE_TO_FACTS,
            Seq2SeqTaskNames.BACKGROUND_TO_FACTS,
            Seq2SeqTaskNames.FACT_TO_METADATA,
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
                                         val_size=200, verbose=True):
    assert task_name in Seq2SeqTaskNames.get_all(), f'Unknown task name {task_name}'
    # Load input and output texts
    input_texts = []
    output_texts = []
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
        else:
            raise ValueError(f'Unknown task name {task_name}')
    # Split intro train & val
    indices = list(range(len(input_texts)))
    indices.sort(key=lambda i: len(output_texts[i]))
    # Choose val_size random indices uniformly distributed across the dataset
    val_indices = np.linspace(0, len(indices) - 1, val_size, dtype=int)
    val_indices_set = set(val_indices)
    train_indices = [i for i in indices if i not in val_indices_set]
    
    if verbose:
        # Print stats
        print_bold(f'Number of train examples: {len(train_indices)}')
        print_bold(f'Number of val examples: {len(val_indices)}')
        print_bold(f'Number of total examples: {len(indices)}')
        # Print random example
        print_bold('Input/output example:')
        i = random.randint(0, len(input_texts) - 1)
        print_bold(f'Input:')
        print_magenta(input_texts[i], bold=True)
        print_bold(f'Output:')
        print_magenta(output_texts[i], bold=True)

    # Create datasets
    train_dataset = Seq2SeqDataset(train_indices, input_texts, output_texts, shuffle=True, infinite=True)
    val_dataset = Seq2SeqDataset(val_indices, input_texts, output_texts, shuffle=False)

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