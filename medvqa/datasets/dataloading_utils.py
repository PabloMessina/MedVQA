from tabnanny import verbose
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

INFINITE_DATASET_LENGTH = int(1e18)

def _get_balancedly_distributed_class_indices(class_weights):
    w_sum = sum(class_weights)
    ws = [w / w_sum for w in class_weights]
    w_min = min(ws)
    assert w_min > 0
    freqs = [max(int(w/w_min),1) for w in ws]
    count = sum(freqs)
    indices = [None] * count
    class_ids = list(range(len(class_weights)))
    class_ids.sort(key = lambda i : freqs[i], reverse=True)
    available_slots = list(range(count))
    for i in class_ids:
        assert len(available_slots) >= freqs[i]
        step = len(available_slots) / freqs[i]
        for j in range(freqs[i]):
            jj = int(j * step)
            indices[available_slots[jj]] = i
        available_slots = [s for s in available_slots if indices[s] is None]
    indices = [i for i in indices if i is not None]
    return np.array(indices, dtype=int)

class CompositeInfiniteDataset(Dataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self._init_indices(datasets, weights)
    
    def _init_indices(self, datasets, weights):
        assert len(datasets) == len(weights)
        
        indices = _get_balancedly_distributed_class_indices(weights)
        
        dataset_counts = np.zeros((len(datasets), len(indices)), dtype=int)
        for i in range(len(datasets)):
            for j in range(len(indices)):
                dataset_counts[i][j] = (indices[j] == i) + (dataset_counts[i][j-1] if j > 0 else 0)
            assert dataset_counts[i][-1] > 0, (i, dataset_counts[i], indices)

        self.indices = indices
        self.counts = dataset_counts
    
    def __len__(self):
        return INFINITE_DATASET_LENGTH
    
    def __getitem__(self, i):
        indices = self.indices        
        ii = i % len(indices)
        idx = indices[ii]
        # assert idx < len(self.datasets)
        counts = self.counts[idx]
        j = (i // len(indices)) * counts[-1] + (counts[ii - 1] if ii > 0 else 0)
        # assert j < len(self.datasets[idx])
        return self.datasets[idx][j]

class BatchedCompositeInfiniteDataset(Dataset):
    def __init__(self, datasets, weights, batch_size):
        self.datasets = datasets
        self._init_indices(datasets, weights)
        self.batch_size = batch_size
    
    def _init_indices(self, datasets, weights):
        assert len(datasets) == len(weights)
        
        dataset_indices = _get_balancedly_distributed_class_indices(weights)
        
        dataset_counts = np.zeros((len(datasets), len(dataset_indices)), dtype=int)
        for i in range(len(datasets)):
            for j in range(len(dataset_indices)):
                dataset_counts[i][j] = (dataset_indices[j] == i) + (dataset_counts[i][j-1] if j > 0 else 0)
            assert dataset_counts[i][-1] > 0, (i, dataset_counts[i], dataset_indices)

        self.indices = dataset_indices
        self.counts = dataset_counts
    
    def __len__(self):
        return INFINITE_DATASET_LENGTH
    
    def __getitem__(self, i):
        indices = self.indices
        batch_size = self.batch_size
        batch_i = i // batch_size
        dataset_i = batch_i % len(indices)
        dataset_id = indices[dataset_i]
        # assert dataset_id < len(self.datasets)
        counts = self.counts[dataset_id]
        j = (counts[-1] * (batch_i // len(indices)) +
            (counts[dataset_i - 1] if dataset_i > 0 else 0) - batch_i) * batch_size + i
        # assert j < len(self.datasets[dataset_id])
        return self.datasets[dataset_id][j]    

def get_imbalance_reduced_weights(ws, coef):
    assert 0 <= coef <= 1
    min_w = min(ws)    
    return [1 + math.log(w/min_w)**2 * coef for w in ws]

def cyclic_dataloader_generator(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def next_k_iterations_gen(cyclic_generator, k):
    for _ in range(k):
        yield next(cyclic_generator)

# def question_balanced_train_dataloader_generator(vqa_handler, min_max_ratio=1.8):
        
#     min_count = min(len(d) for d in vqa_handler.train_datasets)
#     max_count = max(len(d) for d in vqa_handler.train_datasets)
#     vqa_handler.freqs = []
#     vqa_handler.dataset_indices = []
#     for i, dataset in enumerate(vqa_handler.train_datasets):
#         freq = (min_max_ratio - 1) * (len(dataset) - min_count) / (max_count - min_count) + 1
#         freq = math.ceil(10 * freq * math.log2(len(dataset) / min_count + 1))
#         vqa_handler.freqs.append(freq)
#         for _ in range(freq):
#             vqa_handler.dataset_indices.append(i)
    
#     cyclic_dataloaders = [cyclic_dataloader_generator(d) for d in vqa_handler.train_dataloaders]
    
#     indices = vqa_handler.dataset_indices
#     while True:            
#         random.shuffle(indices)
#         for i in indices:
#             yield next(cyclic_dataloaders[i])

def balanced_dataloaders_generator(dataloaders, weights):
    assert len(dataloaders) == len(weights)
    indices = _get_balancedly_distributed_class_indices(weights)    
    cyclic_dataloaders = [cyclic_dataloader_generator(d) for d in dataloaders]
    while True:
        for i in indices:
            yield next(cyclic_dataloaders[i])

def multi_cyclic_dataloaders_generator(dataloaders):
    while True:
        for dataloader in dataloaders:
            for batch in dataloader:
                yield batch

# def collate_test_batch(batch):
#     indexes = sorted(range(len(batch)), key=lambda i : len(batch[i]['q']), reverse=True)
#     batch_dict = dict()
#     batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])
#     batch_dict['i'] = torch.stack([batch[i]['i'] for i in indexes])
#     batch_dict['q'] = nn.utils.rnn.pad_sequence(
#         sequences = [torch.tensor(batch[i]['q']) for i in indexes],
#         batch_first=True,
#         padding_value=0,
#     )
#     batch_dict['ql'] = torch.tensor([len(batch[i]['q']) for i in indexes])
#     return batch_dict

def get_vqa_collate_batch_fn(dataset_id, verbose_question=True, include_answer=True, use_tags=False, n_tags=None,
                         use_orientation=False, use_chexpert=False, classify_questions=False):

    if use_tags:
        mlb = MultiLabelBinarizer(list(range(n_tags)))

    def collate_batch_fn(batch):
        if verbose_question:
            indexes = sorted(range(len(batch)), key=lambda i : len(batch[i]['q']), reverse=True)
        else:
            indexes = list(range(len(batch)))
        
        batch_dict = dict()
        batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])
        batch_dict['i'] = torch.stack([batch[i]['i'] for i in indexes])
        
        if verbose_question:
            batch_dict['q'] = nn.utils.rnn.pad_sequence(
                sequences = [torch.tensor(batch[i]['q']) for i in indexes],
                batch_first=True,
                padding_value=0,
            )
            batch_dict['ql'] = torch.tensor([len(batch[i]['q']) for i in indexes])
        else:
            batch_dict['q'] = torch.tensor([batch[i]['q'] for i in indexes])
        
        if include_answer:
            batch_dict['a'] = nn.utils.rnn.pad_sequence(
                sequences = [torch.tensor(batch[i]['a']) for i in indexes],
                batch_first=True,
                padding_value=0,
            )
        # Auxiliary tasks
        if use_tags:
            batch_dict['tags'] = torch.tensor(mlb.fit_transform([batch[i]['tags'] for i in indexes]))
        if use_orientation:
            batch_dict['orientation'] = torch.tensor([batch[i]['orientation'] for i in indexes])
            batch_dict['dataset_id'] = dataset_id        
        if use_chexpert:
            batch_dict['chexpert'] = torch.tensor([batch[i]['chexpert'] for i in indexes])
        if classify_questions:
            batch_dict['qlabels'] = torch.tensor([batch[i]['qlabels'] for i in indexes])
            
        return batch_dict

    return collate_batch_fn

def qa_collate_batch_fn(batch):
    indexes = sorted(range(len(batch)), key=lambda i : len(batch[i]['q']), reverse=True)
    batch_dict = dict()
    batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])        
    batch_dict['q'] = nn.utils.rnn.pad_sequence(
        sequences = [torch.tensor(batch[i]['q']) for i in indexes],
        batch_first=True,
        padding_value=0,
    )
    batch_dict['ql'] = torch.tensor([len(batch[i]['q']) for i in indexes])
    batch_dict['a'] = nn.utils.rnn.pad_sequence(
        sequences = [torch.tensor(batch[i]['a']) for i in indexes],
        batch_first=True,
        padding_value=0,
    )            
    return batch_dict

def get_vision_collate_batch_fn(dataset_id,
                             classify_tags=False, n_tags=None,
                             classify_orientation=False,
                             classify_chexpert=False,
                             classify_questions=False):

    if classify_tags:
        mlb = MultiLabelBinarizer(list(range(n_tags)))

    def collate_batch_fn(batch):
        indexes = list(range(len(batch)))        
        batch_dict = dict()
        batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])
        batch_dict['i'] = torch.stack([batch[i]['i'] for i in indexes])
        # Auxiliary tasks
        if classify_tags:
            batch_dict['tags'] = torch.tensor(mlb.fit_transform([batch[i]['tags'] for i in indexes]))
        if classify_orientation:
            batch_dict['orientation'] = torch.tensor([batch[i]['orientation'] for i in indexes])
            batch_dict['dataset_id'] = dataset_id        
        if classify_chexpert:
            batch_dict['chexpert'] = torch.tensor([batch[i]['chexpert'] for i in indexes])
        if classify_questions:
            batch_dict['qlabels'] = torch.tensor([batch[i]['qlabels'] for i in indexes])
            
        return batch_dict

    return collate_batch_fn