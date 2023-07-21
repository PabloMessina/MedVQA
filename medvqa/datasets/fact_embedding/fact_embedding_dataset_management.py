import random
import math
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH, CompositeDataset, CompositeInfiniteDataset
from medvqa.utils.common import DictWithDefault
from medvqa.utils.files import load_jsonl, load_pickle
from medvqa.utils.logging import print_bold, print_magenta

_GROUP_NAME_TO_SHORT = {
    'anatomical_locations': 'al',
    'observations': 'ob',
}

_CATEGORY_TO_LABEL = DictWithDefault(
    initial_values={
        'anatomical finding': 0,
        'disease': 1,
        'technical assessment': 2,
        'tubes and lines': 3,
        'device': 4,
        'other': 5,
    },
    default=5, # 'other'
)
_LABEL_TO_CATEGORY = {v: k for k, v in _CATEGORY_TO_LABEL.items()}

_HEALTH_STATUS_TO_LABEL = DictWithDefault(
    initial_values={
        'normal': 0,
        'abnormal': 1,
        'ambiguous': 2,
        'unknown': 3,
        'other': 4,
    },
    default=4, # 'other'
)
_LABEL_TO_HEALTH_STATUS = {v: k for k, v in _HEALTH_STATUS_TO_LABEL.items()}

_COMPARISON_STATUS_TO_LABEL = DictWithDefault(
    initial_values={
        'no comparison': 0,
        'new finding': 1,
        'resolved': 2,
        'improved': 3,
        'worsened': 4,
        'progressed': 5,
        'reappeared': 6,
        'larger': 7,
        'smaller': 8,
        'increase': 9,
        'decrease': 10,
        'position changed': 11,
        'stable/unchanged': 12,
        'unclear comparison': 13,
        'other': 14,
    },
    default=14, # 'other'
)
_LABEL_TO_COMPARISON_STATUS = {v: k for k, v in _COMPARISON_STATUS_TO_LABEL.items()}

class FactTripletDataset(Dataset):
    def __init__(self, facts, triplets, rule_id=None, shuffle=False, infinite=False):
        self.facts = facts
        self.triplets = triplets
        self.rule_id = rule_id
        self.infinite = infinite
        self.indices = list(range(len(self.triplets)))
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
        t = self.triplets[idx]
        output = {
            'a': self.facts[t[0]],
            'p': self.facts[t[1]],
            'n': self.facts[t[2]],
        }
        if self.rule_id is not None: # optional, useful for computing metrics per rule
            output['rule_id'] = self.rule_id
        return output
    
class FactClassificationDataset(Dataset):
    def __init__(self, facts, indices, categories, health_statuses, comparison_statuses, shuffle=False, infinite=False):
        self.facts = facts
        self.categories = categories
        self.health_statuses = health_statuses
        self.comparison_statuses = comparison_statuses
        self.indices = indices
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
        idx = self.indices[i]
        output = {
            'f': self.facts[idx],
            'c': self.categories[idx],
            'hs': self.health_statuses[idx],
            'cs': self.comparison_statuses[idx],
        }
        return output

class FactEmbeddingTrainer():

    def __init__(self, batch_size, num_workers, dataset_name,
                 # triplet ranking arguments
                 triplets_filepath,
                 triplet_rule_weights,
                 triplet_collate_batch_fn,
                 # classification arguments
                 integrated_facts_metadata_jsonl_filepath,
                 paraphrases_jsonl_filepaths,
                 classification_collate_batch_fn,
                 ):
        
        assert dataset_name, 'dataset_name must be provided'
        assert triplets_filepath, 'triplets_filepath must be provided'
        assert triplet_rule_weights, 'triplet_rule_weights must be provided'
        assert integrated_facts_metadata_jsonl_filepath, 'integrated_facts_metadata_jsonl_filepath must be provided'
        assert paraphrases_jsonl_filepaths, 'paraphrases_jsonl_filepaths must be provided'

        self.batch_size = batch_size
        self.triplet_collate_batch_fn = triplet_collate_batch_fn
        self.classification_collate_batch_fn = classification_collate_batch_fn
        self.num_workers = num_workers
        self.triplets_filepath = triplets_filepath
        self.triplet_rule_weights = triplet_rule_weights
        self.dataset_name = dataset_name

        print(f'Loading triplets from {triplets_filepath}...')
        triplets_data = load_pickle(triplets_filepath)
        self.facts = triplets_data['sentences']
        
        # Train triplet ranking dataset and dataloader
        print('----')
        print_bold('Building train triplet ranking dataset and dataloader...')
        _datasets = []
        _weights = []
        for group_name, rules in triplets_data['train'].items():
            assert len(triplet_rule_weights[group_name]) == len(rules), \
                f'Number of rules in {group_name} ({len(rules)}) does not match number of weights ({len(triplet_rule_weights[group_name])})'
            for i, rule in enumerate(rules):                
                w = triplet_rule_weights[group_name][i]
                rule_name = rule['rule']
                rule_triplets = rule['triplets']
                print(f'{group_name} -> rule {i}: "{rule_name}"')
                print(f'\tNumber of triplets: {len(rule_triplets)}')
                print(f'\tWeight: {w}')
                _datasets.append(FactTripletDataset(self.facts, rule_triplets, shuffle=True, infinite=True))
                _weights.append(w)
        self.train_triplet_dataset = CompositeInfiniteDataset(_datasets, _weights)
        self.train_triplet_dataloader = DataLoader(
            self.train_triplet_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=triplet_collate_batch_fn,
            pin_memory=True,
        )

        # Train classification dataset and dataloader
        print('----')
        print_bold('Building train classification dataset and dataloader...')
        sentence2paraphrases = {}
        for filepath in paraphrases_jsonl_filepaths:
            print(f'Loading paraphrases from {filepath}...')
            rows = load_jsonl(filepath)
            print_count = 0
            for row in rows:
                s = next(iter(row['metadata'].values()))
                p = row['parsed_response']
                if print_count < 1:
                    print_count += 1
                    print_bold(f'Input:')
                    print_magenta(s, bold=True)
                    print_bold(f'Paraphrases:')
                    for x in p:
                        print_magenta(x, bold=True)
                if s not in sentence2paraphrases:
                    sentence2paraphrases[s] = []
                sentence2paraphrases[s].extend(p)
        print(f'Number of sentences with paraphrases: {len(sentence2paraphrases)}')
        print(f'Number of paraphrases: {sum(len(x) for x in sentence2paraphrases.values())}')
        # 3 classification tasks: "category", "health status", "comparison status"
        sentences = []
        categories = []
        health_statuses = []
        comparison_statuses = []
        print('Loading integrated facts metadata...')
        integrated_facts_metadata = load_jsonl(integrated_facts_metadata_jsonl_filepath)
        improved_comparison_count = 0
        for row in integrated_facts_metadata:
            fact = row['fact']
            metadata = row['metadata']
            # category
            cat = _CATEGORY_TO_LABEL[metadata['category']]
            # health status
            hs = _HEALTH_STATUS_TO_LABEL[metadata['health status']]
            # comparison status
            if 'improved_comparison' in row:
                cs = _COMPARISON_STATUS_TO_LABEL[row['improved_comparison']['comparison']]
                improved_comparison_count += 1
            else:
                comp = metadata['comparison status']
                psc = metadata['prev_study_comparison?']
                assert psc in ('yes', 'no')
                assert (psc == 'yes') == (comp != '')
                if comp == '': comp = 'no comparison'
                cs = _COMPARISON_STATUS_TO_LABEL[comp]
            # paraphrases
            paraphrases = set()
            paraphrases.add(fact)
            if fact in sentence2paraphrases:
                paraphrases.update(sentence2paraphrases[fact])
            do = metadata['detailed observation']
            if len(do) > 0 and do != fact:
                paraphrases.add(do)
                if do in sentence2paraphrases:
                    paraphrases.update(sentence2paraphrases[do])
            for p in paraphrases:
                sentences.append(p)
                categories.append(cat)
                health_statuses.append(hs)
                comparison_statuses.append(cs)
        # print some stats
        print(f'Number of facts: {len(sentences)}')
        print(f'Number of improved comparisons: {improved_comparison_count}/{len(integrated_facts_metadata)}')
        for x, count in sorted(list(Counter(categories).items()), key=lambda x: x[1], reverse=True):
            print(f'Category: {_LABEL_TO_CATEGORY[x]} -> {count}')
        for x, count in sorted(list(Counter(health_statuses).items()), key=lambda x: x[1], reverse=True):
            print(f'Health status: {_LABEL_TO_HEALTH_STATUS[x]} -> {count}')
        for x, count in sorted(list(Counter(comparison_statuses).items()), key=lambda x: x[1], reverse=True):
            print(f'Comparison status: {_LABEL_TO_COMPARISON_STATUS[x]} -> {count}')
        for _ in range(10):
            i = random.randint(0, len(sentences)-1)
            print_bold(f'Example fact: {sentences[i]}')
            print(f'Category: {_LABEL_TO_CATEGORY[categories[i]]}')
            print(f'Health status: {_LABEL_TO_HEALTH_STATUS[health_statuses[i]]}')
            print(f'Comparison status: {_LABEL_TO_COMPARISON_STATUS[comparison_statuses[i]]}')

        cs2indices = {}
        for i, cs in enumerate(comparison_statuses):
            if cs not in cs2indices:
                cs2indices[cs] = []
            cs2indices[cs].append(i)
        _datasets = []
        _weights = []
        for cs, indices in cs2indices.items():
            print(f'Comparison status: {_LABEL_TO_COMPARISON_STATUS[cs]}')
            print(f'\tNumber of samples: {len(indices)}')
            _datasets.append(FactClassificationDataset(
                sentences, indices, categories, health_statuses, comparison_statuses,
                shuffle=True, infinite=True,
            ))
            _weights.append(math.log2(len(indices))**3) # weight by log2(N)^3
            print(f'\tWeight: {_weights[-1]}')
        self.train_classification_dataset = CompositeInfiniteDataset(_datasets, _weights)
        self.train_classification_dataloader = DataLoader(
            self.train_classification_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=classification_collate_batch_fn,
            pin_memory=True,
        )

        # Val dataset and dataloader
        print('----')
        print_bold('Building val dataset and dataloader...')
        val_rule_datasets = []
        self.rule_ids = []
        for group_name, rules in triplets_data['val'].items():
            for i, rule in enumerate(rules):
                rule_name = rule['rule']
                rule_triplets = rule['triplets']
                print(f'{group_name} -> rule {i}: "{rule_name}"')
                print(f'\tNumber of triplets: {len(rule_triplets)}')
                rule_id = f'{_GROUP_NAME_TO_SHORT[group_name]}{i}'
                print(f'\tRule ID: {rule_id}')
                self.rule_ids.append(rule_id)
                val_rule_datasets.append(FactTripletDataset(self.facts, rule_triplets, rule_id=rule_id))
        self.val_dataset = CompositeDataset(val_rule_datasets)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=triplet_collate_batch_fn,
            pin_memory=True,
        )

    @property
    def name(self):
        return self.dataset_name