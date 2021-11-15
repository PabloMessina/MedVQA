import random
import math
import torch
import torch.nn as nn

def cyclic_dataloader_generator(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def next_k_iterations_gen(cyclic_generator, k):
    for _ in range(k):
        yield next(cyclic_generator)

def question_balanced_train_dataloader_generator(vqa_handler, min_max_ratio=1.8):
        
    min_count = min(len(d) for d in vqa_handler.train_datasets)
    max_count = max(len(d) for d in vqa_handler.train_datasets)
    vqa_handler.freqs = []
    vqa_handler.dataset_indices = []
    for i, dataset in enumerate(vqa_handler.train_datasets):
        freq = (min_max_ratio - 1) * (len(dataset) - min_count) / (max_count - min_count) + 1
        freq = math.ceil(10 * freq * math.log2(len(dataset) / min_count + 1))
        vqa_handler.freqs.append(freq)
        for _ in range(freq):
            vqa_handler.dataset_indices.append(i)
    
    cyclic_dataloaders = [cyclic_dataloader_generator(d) for d in vqa_handler.train_dataloaders]
    
    indices = vqa_handler.dataset_indices
    while True:            
        random.shuffle(indices)
        for i in indices:
            yield next(cyclic_dataloaders[i])

def multi_cyclic_dataloader_sampler(dataloaders, frequencies=None, shuffle=False):
    if shuffle:
        indices = []
        for i, f in enumerate(frequencies):
            for _ in range(f):
                indices.append(i)    
        while True:
            random.shuffle(indices)
            for i in indices:
                yield next(dataloaders[i])
    else:
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

def collate_batch_fn(batch):
    indexes = sorted(range(len(batch)), key=lambda i : len(batch[i]['q']), reverse=True)
    batch_dict = dict()
    batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])
    batch_dict['i'] = torch.stack([batch[i]['i'] for i in indexes])
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