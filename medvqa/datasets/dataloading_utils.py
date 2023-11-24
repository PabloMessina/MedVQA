from collections import OrderedDict
import numpy as np
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAME_TO_SHORT
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import load_chest_imagenome_label_names
from medvqa.models.report_generation.templates.chexpert import TEMPLATES_CHEXPERT_v3

from medvqa.utils.constants import (
    CHEXPERT_LABELS,
    CXR14_DATASET_ID,
    CHEXPERT_DATASET_ID,
    IUXRAY_DATASET_ID__CHEXPERT_MODE,
    MIMICCXR_DATASET_ID,
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE,
    MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE,
    MIMICCXR_DATASET_ID__CHEXPERT_MODE,
    VINBIG_DATASET_ID,
    PADCHEST_DATASET_ID,    
)
from medvqa.utils.logging import print_bold, print_red

INFINITE_DATASET_LENGTH = int(1e18)

def _get_balancedly_distributed_class_indices(class_weights):
    if all(w == class_weights[0] for w in class_weights):
        return np.arange(len(class_weights), dtype=int) # all classes have the same weight
    w_sum = sum(class_weights)
    ws = [w / w_sum for w in class_weights]
    w_min = min(ws)
    assert w_min > 0
    freqs = [max(int(20*w/w_min),1) for w in ws]
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

class CompositeDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum(len(d) for d in datasets)
        self.i2di = [None] * self._len
        i = 0
        for di, d in enumerate(datasets):
            for j in range(len(d)):
                self.i2di[i] = (di, j)
                i += 1
        assert i == self._len
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, i):
        di, j = self.i2di[i]
        return self.datasets[di][j]

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

def multi_dataloaders_generator(dataloaders):
    for dataloader in dataloaders:
        for batch in dataloader:
            yield batch

def simple_yolov8_collate_batch_fn(batch):
    batch_dict = {}
    batch_dict['i'] = torch.stack([x['i'] for x in batch])
    batch_dict['resized_shape'] = [x['resized_shape'] for x in batch]
    return batch_dict

def get_vqa_collate_batch_fn(
        dataset_id, verbose_question=True, one_hot_question_offset=None, one_hot_question_offsets=None,
        include_image=True, include_visual_features=False, include_answer=True, use_visual_module_only=False,
        classify_tags=False, n_tags=None, classify_orientation=False, classify_gender=False,
        classify_chexpert=False, classify_questions=False, classify_chest_imagenome=False,
        predict_bboxes_chest_imagenome=False, pass_pred_bbox_coords_as_input=False, use_yolov8=False,
        predict_bboxes_vinbig=False,
    ):

    if classify_tags:
        mlb = MultiLabelBinarizer(list(range(n_tags)))

    if not use_visual_module_only:
        if not verbose_question:
            if one_hot_question_offset is None:
                one_hot_question_offset = one_hot_question_offsets[str(dataset_id)]
            print(f'get_vqa_collate_batch_fn(): dataset_id={dataset_id}, one_hot_question_offset={one_hot_question_offset}')  
    
    if dataset_id == MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE:
        def collate_batch_fn(batch):
            batch_dict = {
                'dataset_id': dataset_id,
                'batch': batch,
            }
            return batch_dict
        return collate_batch_fn
    
    if dataset_id in [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID,
                    IUXRAY_DATASET_ID__CHEXPERT_MODE, MIMICCXR_DATASET_ID__CHEXPERT_MODE,
                    MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE]:
        def collate_batch_fn(batch, training_mode=True):
            if not use_visual_module_only and verbose_question:
                indexes = sorted(range(len(batch)), key=lambda i : len(batch[i]['q']), reverse=True)
            else:
                indexes = list(range(len(batch)))
            
            batch_dict = {}
            batch_dict['dataset_id'] = dataset_id
            batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])
            
            if include_image:
                if use_yolov8:
                    batch_dict['img'] = torch.stack([batch[i]['img'] for i in indexes])
                else:
                    batch_dict['i'] = torch.stack([batch[i]['i'] for i in indexes])
                # print(f'collate_batch_fn(): batch_dict["i"].shape={batch_dict["i"].shape}')
            if include_visual_features:
                batch_dict['vf'] = torch.tensor([batch[i]['vf'] for i in indexes]).float()
            # Auxiliary tasks
            if classify_tags:
                batch_dict['tags'] = torch.tensor(mlb.fit_transform([batch[i]['tags'] for i in indexes]))
            if classify_orientation:
                batch_dict['orientation'] = torch.tensor([batch[i]['orientation'] for i in indexes])            
            if classify_chexpert:
                batch_dict['chexpert'] = torch.tensor([batch[i]['chexpert'] for i in indexes])
            if classify_questions:
                batch_dict['qlabels'] = torch.tensor([batch[i]['qlabels'] for i in indexes])
            if classify_gender:
                batch_dict['gender'] = torch.tensor([batch[i]['gender'] for i in indexes])
            if classify_chest_imagenome:
                batch_dict['chest_imagenome'] = torch.tensor([batch[i]['chest_imagenome'] for i in indexes])
            if predict_bboxes_chest_imagenome:
                if use_yolov8:
                    if training_mode:
                        batch_dict['im_file'] = [batch[i]['im_file'] for i in indexes]
                        batch_dict['ori_shape'] = [batch[i]['ori_shape'] for i in indexes]
                        batch_dict['resized_shape'] = [batch[i]['resized_shape'] for i in indexes]
                        bboxes_list, cls_list, batch_idx_list, count = None, None, None, 0
                        for i, idx in enumerate(indexes):
                            coords = batch[idx]['chest_imagenome_bbox_coords']
                            presence = batch[idx]['chest_imagenome_bbox_presence']
                            if bboxes_list is None:
                                bboxes_list = [None] * len(presence) * len(indexes)
                                cls_list = [None] * len(presence) * len(indexes)
                                batch_idx_list = [None] * len(presence) * len(indexes)
                            for cls in range(len(presence)):
                                if presence[cls]:
                                    # convert coords[cls] from xyxy to x_c, y_c, w, h
                                    bboxes_list[count] = torch.tensor([
                                        (coords[cls, 0] + coords[cls, 2]) / 2,
                                        (coords[cls, 1] + coords[cls, 3]) / 2,
                                        coords[cls, 2] - coords[cls, 0],
                                        coords[cls, 3] - coords[cls, 1],
                                    ])
                                    cls_list[count] = cls
                                    batch_idx_list[count] = i
                                    count += 1
                        batch_dict['bboxes'] = torch.stack(bboxes_list[:count])
                        assert batch_dict['bboxes'].shape == (count, 4)
                        batch_dict['cls'] = torch.tensor(cls_list[:count])
                        batch_dict['cls'] = batch_dict['cls'].view(-1, 1)
                        assert batch_dict['cls'].shape == (count, 1)
                        batch_dict['batch_idx'] = torch.tensor(batch_idx_list[:count])
                    else:
                        batch_dict['resized_shape'] = [batch[i]['resized_shape'] for i in indexes]
                        batch_dict['chest_imagenome_bbox_coords'] = torch.tensor([batch[i]['chest_imagenome_bbox_coords'] for i in indexes])
                        batch_dict['chest_imagenome_bbox_presence'] = torch.tensor([batch[i]['chest_imagenome_bbox_presence'] for i in indexes])
                else:
                    batch_dict['chest_imagenome_bbox_coords'] = torch.tensor([batch[i]['chest_imagenome_bbox_coords'] for i in indexes])
                    batch_dict['chest_imagenome_bbox_presence'] = torch.tensor([batch[i]['chest_imagenome_bbox_presence'] for i in indexes])
            if pass_pred_bbox_coords_as_input:
                batch_dict['pred_bbox_coords'] = torch.tensor([batch[i]['pred_bbox_coords'] for i in indexes])
            
            if not use_visual_module_only:
                if verbose_question:
                    batch_dict['q'] = nn.utils.rnn.pad_sequence(
                        sequences = [torch.tensor(batch[i]['q']) for i in indexes],
                        batch_first=True,
                        padding_value=0,
                    )
                    batch_dict['ql'] = torch.tensor([len(batch[i]['q']) for i in indexes])
                else:
                    batch_dict['q'] = torch.tensor([batch[i]['q'] + one_hot_question_offset for i in indexes])            
                if include_answer:
                    batch_dict['a'] = nn.utils.rnn.pad_sequence(
                        sequences = [torch.tensor(batch[i]['a']) for i in indexes],
                        batch_first=True,
                        padding_value=0,
                    )

            return batch_dict

    elif dataset_id in [CHEXPERT_DATASET_ID, CXR14_DATASET_ID]:        
        def collate_batch_fn(batch):            
            batch_dict = dict()
            batch_dict['dataset_id'] = dataset_id
            batch_dict['idx'] = torch.tensor([x['idx'] for x in batch])
            if classify_orientation:
                batch_dict['o'] = torch.tensor([x['o'] for x in batch])
            if classify_gender:
                batch_dict['g'] = torch.tensor([x['g'] for x in batch])
            if classify_chexpert or dataset_id == CXR14_DATASET_ID:
                batch_dict['l'] = torch.tensor([x['l'] for x in batch])            
            if include_image:
                batch_dict['i'] = torch.stack([x['i'] for x in batch])
            if include_visual_features:
                batch_dict['vf'] = torch.tensor([x['vf'] for x in batch]).float()
            if not use_visual_module_only:
                batch_dict['q'] = torch.tensor([x['q'] + one_hot_question_offset for x in batch])
                if include_answer:
                    batch_dict['a'] = nn.utils.rnn.pad_sequence(
                        sequences = [torch.tensor(x['a']) for x in batch],
                        batch_first=True,
                        padding_value=0,
                    )
            return batch_dict

    elif dataset_id == VINBIG_DATASET_ID:
        def collate_batch_fn(batch, training_mode):
            indexes = list(range(len(batch)))        
            batch_dict = dict()
            batch_dict['dataset_id'] = dataset_id
            batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])            
            batch_dict['l'] = torch.tensor([batch[i]['l'] for i in indexes])            
            if include_image:
                if use_yolov8:
                    batch_dict['img'] = torch.stack([batch[i]['img'] for i in indexes])
                else:
                    batch_dict['i'] = torch.stack([batch[i]['i'] for i in indexes])
            if include_visual_features:
                batch_dict['vf'] = torch.tensor([batch[i]['vf'] for i in indexes]).float()
            if predict_bboxes_vinbig:
                assert use_yolov8
                if training_mode:
                    batch_dict['im_file'] = [batch[i]['im_file'] for i in indexes]
                    batch_dict['ori_shape'] = [batch[i]['ori_shape'] for i in indexes]
                    batch_dict['resized_shape'] = [batch[i]['resized_shape'] for i in indexes]
                    bboxes_list, cls_list, batch_idx_list = [], [], []
                    for i, idx in enumerate(indexes):
                        bboxes = batch[idx]['bboxes']
                        classes = batch[idx]['classes']
                        for bbox, cls in zip(bboxes, classes):
                            # convert bbox from xyxy to x_c, y_c, w, h
                            bboxes_list.append(torch.tensor([
                                (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1],
                            ]))
                            cls_list.append(cls)
                            batch_idx_list.append(i)
                    batch_dict['bboxes'] = torch.stack(bboxes_list)
                    assert batch_dict['bboxes'].shape == (len(bboxes_list), 4)
                    batch_dict['cls'] = torch.tensor(cls_list)
                    batch_dict['cls'] = batch_dict['cls'].view(-1, 1)
                    assert batch_dict['cls'].shape == (len(cls_list), 1)
                    batch_dict['batch_idx'] = torch.tensor(batch_idx_list)
                else:
                    batch_dict['resized_shape'] = [batch[i]['resized_shape'] for i in indexes]
                    batch_dict['bboxes'] = [batch[i]['bboxes'] for i in indexes]
                    batch_dict['classes'] = [batch[i]['classes'] for i in indexes]
            if not use_visual_module_only:
                batch_dict['q'] = torch.tensor([batch[i]['q'] + one_hot_question_offset for i in indexes])
                if include_answer:
                    batch_dict['a'] = nn.utils.rnn.pad_sequence(
                        sequences = [torch.tensor(batch[i]['a']) for i in indexes],
                        batch_first=True,
                        padding_value=0,
                    )
            return batch_dict
    
    elif dataset_id == PADCHEST_DATASET_ID:            
        def collate_batch_fn(batch):
            batch_size = len(batch)
            batch_dict = dict()
            batch_dict['dataset_id'] = dataset_id
            batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in range(batch_size)])
            batch_dict['l'] = torch.tensor([batch[i]['l'] for i in range(batch_size)])
            batch_dict['loc'] = torch.tensor([batch[i]['loc'] for i in range(batch_size)])
            if classify_orientation:
                batch_dict['o'] = torch.tensor([batch[i]['proj'] for i in range(batch_size)])
            if classify_gender:
                batch_dict['g'] = torch.tensor([batch[i]['g'] for i in range(batch_size)])
            if include_image:
                batch_dict['i'] = torch.stack([batch[i]['i'] for i in range(batch_size)])
            if not use_visual_module_only:
                batch_dict['q'] = torch.tensor([batch[i]['q'] + one_hot_question_offset for i in range(batch_size)])
                if include_answer:
                    batch_dict['a'] = nn.utils.rnn.pad_sequence(
                        sequences = [torch.tensor(batch[i]['a']) for i in range(batch_size)],
                        batch_first=True,
                        padding_value=0,
                    )
            return batch_dict
    
    else: assert False, f'Unknown dataset_id {dataset_id}'

    return collate_batch_fn

def get_vision_collate_batch_fn(**kwargs):    
    # Use the same collate function used for VQA, but with the visual module only
    return get_vqa_collate_batch_fn(**kwargs, use_visual_module_only=True)

def get_multimodal_collate_batch_fn(dataset_id, use_text=True, classify_orientation=False,
                                    classify_chexpert=False, classify_questions=False):

    if dataset_id in [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID]:
        def collate_batch_fn(batch):
            if use_text:
                indexes = sorted(range(len(batch)), key=lambda i : len(batch[i]['t']), reverse=True)
            else:
                indexes = list(range(len(batch)))
            batch_dict = {}
            batch_dict['dataset_id'] = dataset_id
            batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])
            batch_dict['i'] = torch.stack([batch[i]['i'] for i in indexes])
            if use_text:
                batch_dict['t'] = nn.utils.rnn.pad_sequence(
                    sequences = [torch.tensor(batch[i]['t']) for i in indexes],
                    batch_first=True,
                    padding_value=0,
                )
                batch_dict['tl'] = torch.tensor([len(batch[i]['t']) for i in indexes])
            # Auxiliary tasks
            if classify_orientation:
                batch_dict['orientation'] = torch.tensor([batch[i]['orientation'] for i in indexes])            
            if classify_chexpert:
                batch_dict['chexpert'] = torch.tensor([batch[i]['chexpert'] for i in indexes])
            if classify_questions:
                batch_dict['qlabels'] = torch.tensor([batch[i]['qlabels'] for i in indexes])                
            return batch_dict

    elif dataset_id in [CHEXPERT_DATASET_ID, CXR14_DATASET_ID]:        
        def collate_batch_fn(batch):
            indexes = list(range(len(batch)))
            batch_dict = dict()
            batch_dict['dataset_id'] = dataset_id
            batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])            
            batch_dict['o'] = torch.tensor([batch[i]['o'] for i in indexes])
            batch_dict['g'] = torch.tensor([batch[i]['g'] for i in indexes])
            batch_dict['l'] = torch.tensor([batch[i]['l'] for i in indexes])
            batch_dict['i'] = torch.stack([batch[i]['i'] for i in indexes])
            return batch_dict

    elif dataset_id == VINBIG_DATASET_ID:
        def collate_batch_fn(batch):
            indexes = list(range(len(batch)))
            batch_dict = dict()
            batch_dict['dataset_id'] = dataset_id
            batch_dict['idx'] = torch.tensor([batch[i]['idx'] for i in indexes])            
            batch_dict['l'] = torch.tensor([batch[i]['l'] for i in indexes])
            batch_dict['i'] = torch.stack([batch[i]['i'] for i in indexes])
            return batch_dict
    
    else: assert False, f'Unknown dataset_id {dataset_id}'

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

def get_mae_collate_batch_fn(dataset_id):
    def mae_collate_batch_fn(batch):        
        batch_dict = dict()
        batch_dict['idx'] = torch.tensor([x['idx'] for x in batch])
        batch_dict['i'] = torch.stack([x['i'] for x in batch])
        batch_dict['dataset_id'] = dataset_id
        return batch_dict
    return mae_collate_batch_fn

class CheXpertPredictionsToText:

    def __init__(self):
        self.n_labels = len(CHEXPERT_LABELS)

    def __call__(self, vector, s, e):
        strings = []
        for i in range(s, e):
            if vector[i] == 1:
                strings.append(TEMPLATES_CHEXPERT_v3[CHEXPERT_LABELS[i - s]][1])
        return ', '.join(strings)

class ChestImaGenomePredictionsToText:

    def __init__(self, chest_imagenome_label_names_filename, apply_anatomy_reordering):
        assert chest_imagenome_label_names_filename is not None
        assert apply_anatomy_reordering is not None
        self.label_names = load_chest_imagenome_label_names(
            chest_imagenome_label_names_filename, apply_anatomy_reordering)
        self.n_labels = len(self.label_names)
        
    def __call__(self, vector, s, e):
        # create ordered dict
        d = OrderedDict()
        for i in range(s, e):
            if vector[i] == 1:
                label_name = self.label_names[i - s]
                if len(label_name) == 2: # global label
                    observation = label_name[1]
                    if observation not in d:
                        d[observation] = []
                elif len(label_name) == 3: # local label
                    observation = label_name[2]
                    anatomy = label_name[0]
                    if observation not in d:
                        d[observation] = []
                    d[observation].append(CHEST_IMAGENOME_BBOX_NAME_TO_SHORT[anatomy])
                else: assert False, label_name
        # create text
        strings = []
        for observation, anatomies in d.items():
            anatomies = sorted(anatomies)
            if len(anatomies) == 0:
                strings.append(f'{observation}')
            else:
                strings.append(f'{observation} ({", ".join(anatomies)})')
        return ', '.join(strings)
    
class BinaryPredictionsToText:

    def __init__(self, chexp_pred_to_text=None, chstimg_pred_to_text=None):
        self.chexp_pred_to_text = chexp_pred_to_text
        self.chstimg_pred_to_text = chstimg_pred_to_text
        self.n_labels = 0
        if chexp_pred_to_text is not None:
            self.n_labels += chexp_pred_to_text.n_labels
        if chstimg_pred_to_text is not None:
            self.n_labels += chstimg_pred_to_text.n_labels
        assert self.n_labels > 0

    def __call__(self, pred):
        if len(pred.shape) == 1:
            single = True
            if torch.is_tensor(pred):
                pred = pred.unsqueeze(0)
            else: # numpy
                pred = np.expand_dims(pred, axis=0)
        else:
            single = False
        assert len(pred.shape) == 2
        assert pred.shape[1] == self.n_labels
        batch_size = pred.shape[0]
        strings = [[] for _ in range(batch_size)]
        offset = 0
        if self.chexp_pred_to_text is not None:
            for i in range(batch_size):
                strings[i].append(self.chexp_pred_to_text(pred[i], offset, offset + self.chexp_pred_to_text.n_labels))
            offset += self.chexp_pred_to_text.n_labels
        if self.chstimg_pred_to_text is not None:
            for i in range(batch_size):
                strings[i].append(self.chstimg_pred_to_text(pred[i], offset, offset + self.chstimg_pred_to_text.n_labels))
            offset += self.chstimg_pred_to_text.n_labels
        for i in range(batch_size):
            strings[i] = '; '.join(x for x in strings[i] if len(x) > 0)
        if single:
            return strings[0]
        return strings

def get_labels2report_collate_batch_fn(dataset_id, use_report, use_gender, use_chexpert, use_chest_imagenome,
                                    use_ground_truth_as_prediction, is_second_label_source=False, flag=None,
                                    randomly_drop_labels=False, use_t5=False, t5_model_name=None,
                                    chest_imagenome_label_names_filename=None, apply_anatomy_reordering=None):
    if dataset_id == MIMICCXR_DATASET_ID:
        if randomly_drop_labels:
            assert use_ground_truth_as_prediction
        print_red(f'randomly_drop_labels = {randomly_drop_labels}', bold=True)

        if use_t5:
            assert t5_model_name is not None
            from transformers import T5Tokenizer
            t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            assert not use_gender # TODO: implement this
            if use_chexpert:
                _chexpert_predictions_to_text = CheXpertPredictionsToText()
                n_chexpert_labels = _chexpert_predictions_to_text.n_labels
            else:
                _chexpert_predictions_to_text = None
            if use_chest_imagenome:
                _chest_imagenome_predictions_to_text = ChestImaGenomePredictionsToText(
                    chest_imagenome_label_names_filename, apply_anatomy_reordering)
                n_chest_imagenome_labels = _chest_imagenome_predictions_to_text.n_labels
            else:
                _chest_imagenome_predictions_to_text = None
            _predictions_to_text = BinaryPredictionsToText(
                chexp_pred_to_text=_chexpert_predictions_to_text,
                chstimg_pred_to_text=_chest_imagenome_predictions_to_text,
            )
        
        DEBUG = False

        def collate_batch_fn(batch):
            nonlocal DEBUG

            batch_dict = dict()
            batch_dict['dataset_id'] = dataset_id
            batch_dict['is_second_label_source'] = is_second_label_source
            batch_dict['idx'] = torch.tensor([x['idx'] for x in batch])
            if flag is not None:
                batch_dict['flag'] = flag
            if use_report:
                if use_t5:
                    reports = [x['report'] for x in batch]
                    assert type(reports[0]) == str, type(reports[0])
                    target_encoding = t5_tokenizer(
                        reports,
                        padding="longest",
                        return_tensors="pt",
                    )
                    labels = target_encoding.input_ids
                    labels[labels == t5_tokenizer.pad_token_id] = -100
                    batch_dict['report'] = labels
                else:
                    batch_dict['report'] = nn.utils.rnn.pad_sequence(
                        sequences = [torch.tensor(x['report']) for x in batch],
                        batch_first=True,
                        padding_value=0,
                    )
            if use_ground_truth_as_prediction:
                to_concat = []
                if use_gender:
                    batch_dict['g'] = torch.tensor([x['g'] for x in batch])
                    to_concat.append(batch_dict['g'])
                if use_chexpert:
                    batch_dict['chexpert'] = torch.tensor([x['chexpert'] for x in batch])
                    to_concat.append(batch_dict['chexpert'])
                    if use_t5: assert batch_dict['chexpert'].shape[1] == n_chexpert_labels
                if use_chest_imagenome:
                    batch_dict['chest_imagenome'] = torch.tensor([x['chest_imagenome'] for x in batch])
                    to_concat.append(batch_dict['chest_imagenome'])
                    if use_t5: assert batch_dict['chest_imagenome'].shape[1] == n_chest_imagenome_labels
                assert len(to_concat) > 0
                batch_dict['predicted_binary_scores'] = torch.cat(to_concat, dim=1).float()
                if randomly_drop_labels:
                     x = batch_dict['predicted_binary_scores']
                     for i in range(x.shape[0]):
                        if random.random() < 0.5:
                            droppable_indices = [j for j in range(x.shape[1]) if x[i,j] == 1]
                            if len(droppable_indices) > 1:
                                n_to_drop = random.randint(1, len(droppable_indices)-1)
                                for j in random.sample(droppable_indices, n_to_drop):
                                    x[i,j] = 0
                if use_t5:
                    predicted_binary_scores = batch_dict['predicted_binary_scores']
                    input_strings = _predictions_to_text(predicted_binary_scores)
                    if DEBUG:
                        print_bold('input_string (gt):')
                        print(random.choice(input_strings))
                        DEBUG = False # only print once

                    input_encoding = t5_tokenizer(
                        input_strings,
                        padding="longest",
                        return_tensors="pt",
                    )
                    batch_dict['input_ids'] = input_encoding.input_ids
                    batch_dict['attention_mask'] = input_encoding.attention_mask
            else:
                if use_chest_imagenome:
                    batch_dict['chest_imagenome'] = torch.tensor([x['chest_imagenome'] for x in batch])
                if use_chexpert:
                    batch_dict['chexpert'] = torch.tensor([x['chexpert'] for x in batch])
                batch_dict['predicted_binary_scores'] = torch.tensor([x['ensemble_predictions'] for x in batch])
                if use_t5:
                    predicted_binary_scores = batch_dict['predicted_binary_scores']
                    input_strings = _predictions_to_text(predicted_binary_scores)
                    if DEBUG:
                        print_bold('input_string (pred):')
                        print(random.choice(input_strings))
                        DEBUG = False # only print once
                    
                    input_encoding = t5_tokenizer(
                        input_strings,
                        padding="longest",
                        return_tensors="pt",
                    )
                    batch_dict['input_ids'] = input_encoding.input_ids
                    batch_dict['attention_mask'] = input_encoding.attention_mask
            return batch_dict
    else: assert False, f'Unknown dataset_id {dataset_id}'
    return collate_batch_fn

def get_seq2seq_collate_batch_fn(use_t5=False, use_bart=False, model_name=None):

    assert use_t5 or use_bart # TODO: support other seq2seq models eventually

    if use_t5 or use_bart:
        assert model_name is not None
        if use_t5:
            from transformers import T5TokenizerFast
            tokenizer = T5TokenizerFast.from_pretrained(model_name)
        elif use_bart:
            from transformers import BartTokenizerFast
            tokenizer = BartTokenizerFast.from_pretrained(model_name)

    def collate_batch_fn(batch):

        batch_dict = dict()
        batch_dict['idx'] = torch.tensor([x['idx'] for x in batch])
        if use_t5 or use_bart:
            output_text = [x['output_text'] for x in batch]
            assert type(output_text[0]) == str, type(output_text[0])
            output_encoding = tokenizer(
                output_text,
                padding="longest",
                return_tensors="pt",
            )
            labels = output_encoding.input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            batch_dict['output_ids'] = labels
            input_text = [x['input_text'] for x in batch]
            input_encoding = tokenizer(
                input_text,
                padding="longest",
                return_tensors="pt",
            )
            batch_dict['input_ids'] = input_encoding.input_ids
            batch_dict['attention_mask'] = input_encoding.attention_mask
        return batch_dict
    
    return collate_batch_fn

def get_fact_embedding_collate_batch_fn(huggingface_model_name, for_triplet_ranking=False,
                                        for_metadata_classification=False,
                                        for_chest_imagenome_observation_classification=False,
                                        for_chest_imagenome_anatomical_location_classification=False,
                                        for_nli=False, for_entcon=False,
                                        for_sentence_autoencoder=False,
                                        ):

    assert sum([for_triplet_ranking,
                for_metadata_classification,
                for_chest_imagenome_observation_classification,
                for_chest_imagenome_anatomical_location_classification,
                for_nli, for_entcon, for_sentence_autoencoder]) == 1 # only one of these should be true
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name, trust_remote_code=True)

    if for_triplet_ranking:
        def collate_batch_fn(batch):
            batch_dict = dict(flag='t') # t for triplet ranking
            a_list = [x['a'] for x in batch]
            p_list = [x['p'] for x in batch]
            n_list = [x['n'] for x in batch]
            a_encoding = tokenizer(a_list, padding="longest", return_tensors="pt")
            p_encoding = tokenizer(p_list, padding="longest", return_tensors="pt")
            n_encoding = tokenizer(n_list, padding="longest", return_tensors="pt")
            batch_dict['a_input_ids'] = a_encoding.input_ids
            batch_dict['a_attention_mask'] = a_encoding.attention_mask
            batch_dict['p_input_ids'] = p_encoding.input_ids
            batch_dict['p_attention_mask'] = p_encoding.attention_mask
            batch_dict['n_input_ids'] = n_encoding.input_ids
            batch_dict['n_attention_mask'] = n_encoding.attention_mask
            if 'rule_id' in batch[0]:
                batch_dict['rule_id'] = [x['rule_id'] for x in batch]
            return batch_dict
    elif for_metadata_classification:
        def collate_batch_fn(batch):
            batch_dict = dict(flag='mc') # mc for metadata classification
            # facts
            facts = [x['f'] for x in batch]
            facts_encoding = tokenizer(facts, padding="longest", return_tensors="pt")
            batch_dict['input_ids'] = facts_encoding.input_ids
            batch_dict['attention_mask'] = facts_encoding.attention_mask
            # labels
            batch_dict['c'] = torch.tensor([x['c'] for x in batch]) # category
            batch_dict['hs'] = torch.tensor([x['hs'] for x in batch]) # health status
            batch_dict['cs'] = torch.tensor([x['cs'] for x in batch]) # comparison status
            return batch_dict
    elif for_chest_imagenome_observation_classification:
        def collate_batch_fn(batch):
            batch_dict = dict(flag='cioc') # cioc for chest imagenome observation classification
            # phrases
            phrases = [x['p'] for x in batch]
            phrases_encoding = tokenizer(phrases, padding="longest", return_tensors="pt")
            batch_dict['input_ids'] = phrases_encoding.input_ids
            batch_dict['attention_mask'] = phrases_encoding.attention_mask
            # labels
            batch_dict['l'] = torch.tensor([x['l'] for x in batch])
            return batch_dict
    elif for_chest_imagenome_anatomical_location_classification:
        def collate_batch_fn(batch):
            batch_dict = dict(flag='cialc') # cialc for chest imagenome anatomical location classification
            # phrases
            phrases = [x['p'] for x in batch]
            phrases_encoding = tokenizer(phrases, padding="longest", return_tensors="pt")
            batch_dict['input_ids'] = phrases_encoding.input_ids
            batch_dict['attention_mask'] = phrases_encoding.attention_mask
            # labels
            batch_dict['l'] = torch.tensor([x['l'] for x in batch])
            return batch_dict
    elif for_nli:
        def collate_batch_fn(batch):
            batch_dict = dict(flag='nli')
            premises = [x['p'] for x in batch]
            hypotheses = [x['h'] for x in batch]
            batch_dict['tokenized_premises'] = tokenizer(premises, padding="longest", return_tensors="pt")
            batch_dict['tokenized_hypotheses'] = tokenizer(hypotheses, padding="longest", return_tensors="pt")
            batch_dict['labels'] = torch.tensor([x['l'] for x in batch])
            return batch_dict
    elif for_entcon:
        def collate_batch_fn(batch):
            batch_dict = dict(flag='entcon')
            batch_dict['tokenized_ent_p'] = tokenizer([x['ent_p'] for x in batch], padding="longest", return_tensors="pt")
            batch_dict['tokenized_ent_h'] = tokenizer([x['ent_h'] for x in batch], padding="longest", return_tensors="pt")
            batch_dict['tokenized_con_p'] = tokenizer([x['con_p'] for x in batch], padding="longest", return_tensors="pt")
            batch_dict['tokenized_con_h'] = tokenizer([x['con_h'] for x in batch], padding="longest", return_tensors="pt")
            return batch_dict
    elif for_sentence_autoencoder:
        def collate_batch_fn(batch):
            batch_dict = dict(flag='sae')
            batch_dict['tokenized_sentences'] = tokenizer([x['s'] for x in batch], padding="longest", return_tensors="pt")
            batch_dict['decoder_ids'] = nn.utils.rnn.pad_sequence(
                        sequences = [torch.tensor(x['ids']) for x in batch],
                        batch_first=True,
                        padding_value=0,
                    )
            return batch_dict
    else: assert False
    
    return collate_batch_fn

def get_nli_collate_batch_fn(huggingface_model_name, merged_input=False):
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name, trust_remote_code=True)

    if merged_input:
        def collate_batch_fn(batch):
            batch_dict = dict()
            texts = [x['t'] for x in batch]
            batch_dict['tokenized_texts'] = tokenizer(texts, padding="longest", return_tensors="pt")
            batch_dict['labels'] = torch.tensor([x['l'] for x in batch])
            return batch_dict
    else:
        def collate_batch_fn(batch):
            batch_dict = dict()
            premises = [x['p'] for x in batch]
            hypotheses = [x['h'] for x in batch]
            batch_dict['tokenized_premises'] = tokenizer(premises, padding="longest", return_tensors="pt")
            batch_dict['tokenized_hypotheses'] = tokenizer(hypotheses, padding="longest", return_tensors="pt")
            batch_dict['labels'] = torch.tensor([x['l'] for x in batch])
            return batch_dict
    
    return collate_batch_fn

def get_image2report_collate_batch_fn(dataset_id, include_report=True, use_visual_module_only=False,
        classify_gender=False, classify_chexpert=False, classify_chest_imagenome=False,
        predict_bboxes_chest_imagenome=False, predict_local_feature_coords=False, use_yolov8=False,
    ):
    if dataset_id == MIMICCXR_DATASET_ID:
        def collate_batch_fn(batch, training_mode=True):
            batch_dict = dict()
            batch_dict['dataset_id'] = dataset_id
            batch_dict['idx'] = torch.tensor([x['idx'] for x in batch])
            if not use_visual_module_only:
                if include_report:
                    batch_dict['report'] = nn.utils.rnn.pad_sequence(
                        sequences = [torch.tensor(x['report']) for x in batch],
                        batch_first=True,
                        padding_value=0,
                    )
            if use_yolov8:
                batch_dict['img'] = torch.stack([x['img'] for x in batch])
            else:
                batch_dict['i'] = torch.stack([x['i'] for x in batch])
            # Auxiliary tasks
            if classify_gender:
                batch_dict['gender'] = torch.tensor([x['gender'] for x in batch])
            if classify_chexpert:
                batch_dict['chexpert'] = torch.tensor([x['chexpert'] for x in batch])
            if classify_chest_imagenome:
                batch_dict['chest_imagenome'] = torch.tensor([x['chest_imagenome'] for x in batch])
            if predict_local_feature_coords:
                batch_dict['local_feature_coords'] = torch.tensor([x['local_feature_coords'] for x in batch])
            if predict_bboxes_chest_imagenome:
                if use_yolov8:
                    if training_mode:
                        batch_dict['im_file'] = [x['im_file'] for x in batch]
                        batch_dict['ori_shape'] = [x['ori_shape'] for x in batch]
                        batch_dict['resized_shape'] = [x['resized_shape'] for x in batch]
                        bboxes_list, cls_list, batch_idx_list, count = None, None, None, 0
                        for i, x in enumerate(batch):
                            coords = x['chest_imagenome_bbox_coords']
                            presence = x['chest_imagenome_bbox_presence']
                            if bboxes_list is None:
                                bboxes_list = [None] * len(presence) * len(batch)
                                cls_list = [None] * len(presence) * len(batch)
                                batch_idx_list = [None] * len(presence) * len(batch)
                            for cls in range(len(presence)):
                                if presence[cls]:
                                    # convert coords[cls] from xyxy to x_c, y_c, w, h
                                    bboxes_list[count] = torch.tensor([
                                        (coords[cls, 0] + coords[cls, 2]) / 2,
                                        (coords[cls, 1] + coords[cls, 3]) / 2,
                                        coords[cls, 2] - coords[cls, 0],
                                        coords[cls, 3] - coords[cls, 1],
                                    ])
                                    cls_list[count] = cls
                                    batch_idx_list[count] = i
                                    count += 1
                        batch_dict['bboxes'] = torch.stack(bboxes_list[:count])
                        assert batch_dict['bboxes'].shape == (count, 4)
                        batch_dict['cls'] = torch.tensor(cls_list[:count])
                        batch_dict['cls'] = batch_dict['cls'].view(-1, 1)
                        assert batch_dict['cls'].shape == (count, 1)
                        batch_dict['batch_idx'] = torch.tensor(batch_idx_list[:count])
                    else:
                        batch_dict['resized_shape'] = [x['resized_shape'] for x in batch]
                        batch_dict['chest_imagenome_bbox_coords'] = torch.tensor([x['chest_imagenome_bbox_coords'] for x in batch])
                        batch_dict['chest_imagenome_bbox_presence'] = torch.tensor([x['chest_imagenome_bbox_presence'] for x in batch])
                else:
                    batch_dict['chest_imagenome_bbox_coords'] = torch.tensor([x['chest_imagenome_bbox_coords'] for x in batch])
                    batch_dict['chest_imagenome_bbox_presence'] = torch.tensor([x['chest_imagenome_bbox_presence'] for x in batch])
            
            return batch_dict
    else: assert False, f'Unknown dataset_id {dataset_id}'
    return collate_batch_fn

def get_phrase_grounding_collate_batch_fn(flag, use_yolov8=False):
    if flag == 'fg': # fact grounding
        def collate_batch_fn(batch):
            # We expect:
            # - 'i': images
            # - 'pe': positive phrase embeddings
            # - 'ne': negative phrase embeddings
            batch_dict = dict(flag=flag)
            if use_yolov8:
                batch_dict['img'] = torch.stack([x['i'] for x in batch])
            else:
                batch_dict['i'] = torch.stack([x['i'] for x in batch])
            batch_dict['pe'] = torch.tensor([x['pe'] for x in batch])
            batch_dict['ne'] = torch.tensor([x['ne'] for x in batch])
            return batch_dict
    elif flag == 'pg': # phrase grounding
        def collate_batch_fn(batch):
            # We expect:
            # - 'i': images
            # - 'pe': phrase embeddings
            # - 'pgm': phrase grounding masks
            batch_dict = dict(flag=flag)
            if use_yolov8:
                batch_dict['img'] = torch.stack([x['i'] for x in batch])
            else:
                batch_dict['i'] = torch.stack([x['i'] for x in batch])
            batch_dict['pe'] = torch.tensor([x['pe'] for x in batch])
            batch_dict['pgm'] = torch.tensor([x['pgm'] for x in batch])
            return batch_dict
    elif flag == 'cibg': # chest imagenome bbox grounding
        assert use_yolov8
        def collate_batch_fn(batch, training_mode=True):
            # We expect:
            # - 'i': images
            # - 'pe': phrase embeddings
            # - 'pgm': phrase grounding masks
            # - 'pcl': phrase classification labels
            # - 'bc': bounding boxes coordinates
            # - 'bp': bounding boxes presence
            batch_dict = dict(flag=flag)
            if use_yolov8:
                batch_dict['img'] = torch.stack([x['i'] for x in batch])
            else:
                batch_dict['i'] = torch.stack([x['i'] for x in batch])
            batch_dict['pe'] = torch.tensor([x['pe'] for x in batch])
            batch_dict['pcl'] = torch.tensor([x['pcl'] for x in batch])
            batch_dict['pgm'] = torch.tensor([x['pgm'] for x in batch])
            if training_mode:
                batch_dict['im_file'] = [x['im_file'] for x in batch]
                batch_dict['ori_shape'] = [x['ori_shape'] for x in batch]
                batch_dict['resized_shape'] = [x['resized_shape'] for x in batch]
                bboxes_list, cls_list, batch_idx_list, count = None, None, None, 0
                for i, x in enumerate(batch):
                    coords = x['bc']
                    presence = x['bp']
                    if bboxes_list is None:
                        bboxes_list = [None] * len(presence) * len(batch)
                        cls_list = [None] * len(presence) * len(batch)
                        batch_idx_list = [None] * len(presence) * len(batch)
                    for cls in range(len(presence)):
                        if presence[cls]:
                            # convert coords[cls] from xyxy to x_c, y_c, w, h
                            bboxes_list[count] = torch.tensor([
                                (coords[cls, 0] + coords[cls, 2]) / 2,
                                (coords[cls, 1] + coords[cls, 3]) / 2,
                                coords[cls, 2] - coords[cls, 0],
                                coords[cls, 3] - coords[cls, 1],
                            ])
                            cls_list[count] = cls
                            batch_idx_list[count] = i
                            count += 1
                batch_dict['bboxes'] = torch.stack(bboxes_list[:count])
                assert batch_dict['bboxes'].shape == (count, 4)
                batch_dict['cls'] = torch.tensor(cls_list[:count])
                batch_dict['cls'] = batch_dict['cls'].view(-1, 1)
                assert batch_dict['cls'].shape == (count, 1)
                batch_dict['batch_idx'] = torch.tensor(batch_idx_list[:count])
            else:
                batch_dict['resized_shape'] = [x['resized_shape'] for x in batch]
                batch_dict['bc'] = torch.tensor([x['bc'] for x in batch])
                batch_dict['bp'] = torch.tensor([x['bp'] for x in batch])
            return batch_dict
    else: assert False, f'Unknown flag {flag}'
    return collate_batch_fn
