import random
import math
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH, CompositeInfiniteDataset
from medvqa.datasets.nli import RADNLI_DEV_JSONL_PATH, RADNLI_TEST_JSONL_PATH
from medvqa.utils.files import load_jsonl
from medvqa.utils.logging import print_bold

_LABEL_TO_INDEX = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
}
_INDEX_TO_LABEL = { v: k for k, v in _LABEL_TO_INDEX.items() }

class NLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, shuffle=False, infinite=False, merged=False, indices=None):
        assert len(premises) == len(hypotheses) == len(labels)        
        if merged:
            self.texts = [f'{p} [SEP] {h}' for p, h in zip(premises, hypotheses)]
        else:            
            self.premises = premises
            self.hypotheses = hypotheses
        self.merged = merged
        self.labels = labels
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(self.labels)))
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
        l = self.labels[idx]
        if self.merged:
            t = self.texts[idx]
            output = { 't': t, 'l': l }
        else:
            p = self.premises[idx]
            h = self.hypotheses[idx]
            output = { 'p': p, 'h': h, 'l': l }
        return output

class EntailmentContradictionDataset(Dataset):
    def __init__(self, ent_premises, ent_hypotheses, con_premises, con_hypotheses, shuffle=False, infinite=False):
        print('EntailmentContradictionDataset: __init__')
        assert len(ent_premises) == len(ent_hypotheses)
        assert len(con_premises) == len(con_hypotheses)
        self.ent_premises = ent_premises
        self.ent_hypotheses = ent_hypotheses
        self.con_premises = con_premises
        self.con_hypotheses = con_hypotheses
        self.n_ent = len(ent_premises)
        self.n_con = len(con_premises)
        print(f'Number of entailment samples: {self.n_ent}')
        print(f'Number of contradiction samples: {self.n_con}')
        self.indices = list(range(self.n_con)) # iterate over contradiction samples and pick a random entailment sample
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
        con_idx = self.indices[i]
        ent_idx = random.randint(0, self.n_ent-1)
        return {
            'ent_p': self.ent_premises[ent_idx],
            'ent_h': self.ent_hypotheses[ent_idx],
            'con_p': self.con_premises[con_idx],
            'con_h': self.con_hypotheses[con_idx],
        } # sim(ent_p, ent_h) > sim(con_p, con_h) (in words: sentences that are entailed should have higher similarity
        # than sentences that are contradictory)

class NLI_Trainer:

    def __init__(self, batch_size, num_workers, collate_batch_fn, integrated_nli_jsonl_filepath=None,
                 integrated_sentence_facts_jsonl_filepath=None, merged=False,
                 train_mode=True, test_mode=False, dev_mode=False):

        self.batch_size = batch_size
        self.num_workers = num_workers
        assert sum([train_mode, dev_mode, test_mode]) == 1 # only one mode must be True

        if train_mode:
            assert integrated_nli_jsonl_filepath is not None
            # Load data from MedNLI. This will be used to train the NLI model.
            nli_train_rows = load_jsonl(integrated_nli_jsonl_filepath)
            train_premises = [x['premise'] for x in nli_train_rows]
            train_hypotheses = [x['hypothesis'] for x in nli_train_rows]
            train_labels = [_LABEL_TO_INDEX[x['label']] for x in nli_train_rows]
            source2id = {}
            for row in nli_train_rows:
                source = row['source']
                if source not in source2id:
                    source2id[source] = len(source2id)
            id2source = { v: k for k, v in source2id.items() }
            train_sources = [source2id[x['source']] for x in nli_train_rows]
            print(f'Number of train samples: {len(train_premises)}')
            print_bold('Example train text:')
            _i = random.randint(0, len(train_premises)-1)
            print(f'Premise: {train_premises[_i]}')
            print(f'Hypothesis: {train_hypotheses[_i]}')
            print(f'Label: {_INDEX_TO_LABEL[train_labels[_i]]}')
            print(f'Source: {train_sources[_i]}')

            if integrated_sentence_facts_jsonl_filepath is not None:
                print(f'Loading integrated sentence facts from {integrated_sentence_facts_jsonl_filepath}...')
                integrated_sentence_facts = load_jsonl(integrated_sentence_facts_jsonl_filepath)
                print(f'Number of integrated sentence facts: {len(integrated_sentence_facts)}')
                entailment_id = _LABEL_TO_INDEX['entailment']
                source_id = len(source2id) # add new source
                id2source[source_id] = 'integrated_sentence_facts'
                len_bef = len(train_premises)
                for row in integrated_sentence_facts:
                    s = row['sentence']
                    for f in row['facts']:
                        if s != f:
                            train_premises.append(s)
                            train_hypotheses.append(f)
                            train_sources.append(source_id)
                            train_labels.append(entailment_id)
                print(f'Number of train entailment samples added: {len(train_premises) - len_bef}')
        
        if train_mode or test_mode:
            radnli_test_rows = load_jsonl(RADNLI_TEST_JSONL_PATH)
            test_premises = [x['sentence1'] for x in radnli_test_rows]
            test_hypotheses = [x['sentence2'] for x in radnli_test_rows]
            test_labels = [_LABEL_TO_INDEX[x['gold_label']] for x in radnli_test_rows]
            print(f'Number of test samples: {len(test_premises)}')
            print_bold('Example test text:')
            _i = random.randint(0, len(test_premises)-1)
            print(f'Premise: {test_premises[_i]}')
            print(f'Hypothesis: {test_hypotheses[_i]}')
            print(f'Label: {_INDEX_TO_LABEL[test_labels[_i]]}')

        if dev_mode:
            radnli_dev_rows = load_jsonl(RADNLI_DEV_JSONL_PATH)
            dev_premises = [x['sentence1'] for x in radnli_dev_rows]
            dev_hypotheses = [x['sentence2'] for x in radnli_dev_rows]
            dev_labels = [_LABEL_TO_INDEX[x['gold_label']] for x in radnli_dev_rows]
            print(f'Number of dev samples: {len(dev_premises)}')
            print_bold('Example dev text:')
            _i = random.randint(0, len(dev_premises)-1)
            print(f'Premise: {dev_premises[_i]}')
            print(f'Hypothesis: {dev_hypotheses[_i]}')
            print(f'Label: {_INDEX_TO_LABEL[dev_labels[_i]]}')
        
        # Create datasets and dataloaders
        print('----')
        if train_mode:
            print_bold('Building train NLI dataset and dataloader...') 

            # Create a dataset for each label and source combination and weight them by log2(N)^3
            source_label_2_indices = {}
            unique_sources = set()
            unique_labels = set()
            for i, (source, label) in enumerate(zip(train_sources, train_labels)):
                unique_sources.add(source)
                unique_labels.add(label)
                key = (source, label)
                if key not in source_label_2_indices:
                    source_label_2_indices[key] = []
                source_label_2_indices[key].append(i)
            unique_sources = sorted(list(unique_sources))
            unique_labels = sorted(list(unique_labels))

            label_datasets = []
            label_weights = []
            label2weight = {
                'entailment': 1,
                'contradiction': 1,
                'neutral': 2, # 2 neutral samples will be sampled twice as much as the other labels
            }
            for label in unique_labels:
                _datasets = []
                _weights = []
                for source in unique_sources:
                    key = (source, label)
                    if key in source_label_2_indices:
                        indices = source_label_2_indices[key]
                        _datasets.append(NLIDataset(train_premises, train_hypotheses, train_labels, shuffle=True, infinite=True, indices=indices))
                        _weights.append(math.log2(len(indices))**3) # weight by log2(N)^3
                        # print(f'Source: {source} | Label: {label} -> {len(indices)} ({_weights[-1]:.2f})')
                        print(f'Label: {_INDEX_TO_LABEL[label]} | Source: {id2source[source]} -> {len(indices)} ({_weights[-1]:.2f})')
                label_datasets.append(CompositeInfiniteDataset(_datasets, _weights))
                label_weights.append(label2weight[_INDEX_TO_LABEL[label]])
            print(f'Number of label datasets: {len(label_datasets)}')
            print(f'Label weights: {label_weights}')
            
            self.train_dataset = CompositeInfiniteDataset(label_datasets, label_weights)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
        if dev_mode:
            print_bold('Building dev NLI dataset and dataloader...')
            self.dev_dataset = NLIDataset(dev_premises, dev_hypotheses, dev_labels, merged=merged)
            self.dev_dataloader = DataLoader(
                self.dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
        if train_mode or test_mode:
            print_bold('Building test NLI dataset and dataloader...')
            self.test_dataset = NLIDataset(test_premises, test_hypotheses, test_labels, merged=merged)
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )

    @property
    def name(self):
        return 'NLI(MedNLI+RadNLI+GPT-4)'