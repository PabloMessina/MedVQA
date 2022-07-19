import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.utils import deduplicate_indices
from medvqa.utils.files import load_pickle
from medvqa.datasets.dataloading_utils import (
    CompositeInfiniteDataset,
    BatchedCompositeInfiniteDataset,
    get_imbalance_reduced_weights,
    INFINITE_DATASET_LENGTH,
)

class VisualModuleDataset(Dataset):
    
    def __init__(self, report_ids, images, indices, transform, source_dataset_name,
                shuffle_indices = True,
                # aux task: medical tags
                classify_tags = False, rid2tags = None,
                # aux task: image orientation
                classify_orientation = False, orientations = None,
                # aux task: chexpert labels
                classify_chexpert = False, chexpert_labels = None,
                # aux task: question labels
                classify_questions = False, question_labels = None,
                # infinite mode
                infinite = False,
        ):
        self.report_ids = report_ids
        self.images = images
        self.indices = indices
        self.transform = transform
        self.source_dataset_name = source_dataset_name
        self.infinite = infinite

        if shuffle_indices:
            np.random.shuffle(self.indices)
        
        # optional auxiliary tasks
        self.classify_tags = classify_tags
        self.rid2tags = rid2tags
        self.classify_orientation = classify_orientation
        self.orientations = orientations
        self.classify_chexpert = classify_chexpert
        self.chexpert_labels = chexpert_labels
        self.classify_questions = classify_questions
        self.question_labels = question_labels

        if infinite:
            self._length = INFINITE_DATASET_LENGTH
        else:
            self._length = len(indices)
    
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        indices = self.indices
        if self.infinite:
            i %= len(indices)
        idx = indices[i]
        output = dict(
            idx=idx,
            i=self.transform(Image.open(self.images[idx]).convert('RGB')),
        )
        rid = self.report_ids[idx]
        if self.classify_tags:
            output['tags'] = self.rid2tags[rid]
        if self.classify_orientation:
            output['orientation'] = self.orientations[idx]
        if self.classify_chexpert:
            output['chexpert'] = self.chexpert_labels[rid]
        if self.classify_questions:
            output['qlabels'] = self.question_labels[rid]
        return output

class VM_Base:

    def __init__(self, training, transform, batch_size, collate_batch_fn,
                preprocessed_data_path,                
                num_workers,
                classify_tags = False,
                rid2tags_path = None,
                classify_orientation = False,
                classify_chexpert = False,
                classify_questions = False,
                chexpert_labels_path = None,
                question_labels_path = None,
        ):
        assert preprocessed_data_path is not None
        self.preprocessed_data_path = preprocessed_data_path
        self.training = training
        self.transform = transform
        self.classify_tags = classify_tags
        self.classify_orientation = classify_orientation
        self.classify_chexpert = classify_chexpert
        self.classify_questions = classify_questions
        
        if classify_tags:
            assert rid2tags_path is not None
            self.rid2tags = load_pickle(rid2tags_path)

        if classify_chexpert:
            assert chexpert_labels_path is not None
            self.chexpert_labels = load_pickle(chexpert_labels_path)
        
        if classify_questions:
            assert question_labels_path is not None
            self.question_labels = load_pickle(question_labels_path)

        self._load_cached_data(preprocessed_data_path)
        print(f'batch_size = {batch_size}')
        self._generate_datasets_and_dataloaders(batch_size, collate_batch_fn, num_workers)

    def __len__(self):
        return len(self.report_ids)
    
    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers):
        raise NotImplementedError('Make sure your specialized class implements this function')
    
    def _load_cached_data(self, preprocessed_data_path):
        print (f'Loading data from path {preprocessed_data_path} ...')
        data = load_pickle(preprocessed_data_path)        
        self.dataset_name = data['dataset_name']
        self.report_ids = data['report_ids']
        self.images = data['images']
        self.question_ids = data['question_ids']
        if self.training:
            self.train_indices = data['train_indices']
            self.val_indices = deduplicate_indices(data['val_indices'], self.report_ids)
        if self.classify_orientation:
            self.orientations = data['orientations']
        print('\tDone!')

class VM_Trainer(VM_Base):
    
    def __init__(self, transform, batch_size, collate_batch_fn,
                preprocessed_data_path,
                cache_dir,
                num_workers,
                classify_tags = False,
                rid2tags_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                imbalance_reduction_coef = 1,
                validation_only = False,
                one_question_per_batch = False,
                question_balanced = False,
        ):

        rid2tags_path = os.path.join(cache_dir, rid2tags_filename) if classify_tags else None
        chexpert_labels_path = os.path.join(cache_dir, chexpert_labels_filename) if classify_chexpert else None
        question_labels_path = os.path.join(cache_dir, question_labels_filename) if classify_questions else None        
        
        self.validation_only = validation_only
        self.cache_dir = cache_dir
        self.imbalance_reduction_coef = imbalance_reduction_coef
        self.one_question_per_batch = one_question_per_batch
        self.question_balanced = question_balanced
    
        super().__init__(True, transform, batch_size, collate_batch_fn,
                preprocessed_data_path,
                num_workers,
                classify_tags = classify_tags,
                rid2tags_path = rid2tags_path,
                classify_orientation = classify_orientation,
                classify_chexpert = classify_chexpert,
                classify_questions = classify_questions,
                chexpert_labels_path = chexpert_labels_path,
                question_labels_path = question_labels_path)

    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers):
        print('generating training and validation datasets and dataloaders ...')
        print('num_workers =', num_workers)
        if not self.validation_only:
            if self.question_balanced:
                self._generate_train_dataset_and_dataloader__question_balanced(batch_size, collate_batch_fn, num_workers)
            else:
                self._generate_train_dataset_and_dataloader(batch_size, collate_batch_fn, num_workers)
        self._generate_val_dataset_and_dataloader(batch_size, collate_batch_fn, num_workers)

    def _get_composite_dataset(self, datasets, weights, batch_size):
        if self.one_question_per_batch:
            print('(***) Note: using BatchedCompositeInfiniteDataset (one question per batch)')
            return BatchedCompositeInfiniteDataset(datasets, weights, batch_size)
        return CompositeInfiniteDataset(datasets, weights)

    def _generate_train_dataset_and_dataloader(self, batch_size, collate_batch_fn, num_workers):
        print('len(self.train_indices) =', len(self.train_indices))
        question_datasets = []
        train_question_ids = list(self.train_indices.keys())
        train_question_ids.sort(key=lambda x : len(self.train_indices[x]))
        i = 0
        n = len(train_question_ids)
        while i < n:
            j = i
            acc_size = len(self.train_indices[train_question_ids[i]])
            while j+1 < n and acc_size < batch_size:
                j += 1
                acc_size += len(self.train_indices[train_question_ids[j]])
            if i == j:
                indices = self.train_indices[train_question_ids[i]]
            else:
                print(f' *** merging from i={i} to j={j}, acc_size = {acc_size}')
                indices = []
                for k in range(i, j+1):
                    indices.extend(self.train_indices[train_question_ids[k]])
                indices = np.array(indices, dtype=int)            
            
            question_datasets.append(self._get_visual_module_dataset(indices))
            i = j+1
        
        question_weights = get_imbalance_reduced_weights([len(d) for d in question_datasets], self.imbalance_reduction_coef)
        self.train_dataset = self._get_composite_dataset(question_datasets, question_weights, batch_size)
        print(f'\tlen(question_datasets) = {len(question_datasets)}')

        self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            collate_fn=collate_batch_fn,
                                            num_workers=num_workers,
                                            pin_memory=True)

    def _get_visual_module_dataset(self, indices, infinite=True, shuffle=True):
        return VisualModuleDataset(
            self.report_ids, self.images, 
            indices, self.transform, self.dataset_name,
            # aux task: medical tags
            classify_tags = self.classify_tags,
            rid2tags = self.rid2tags if self.classify_tags else None,
            # aux task: orientation
            classify_orientation = self.classify_orientation,
            orientations = self.orientations if self.classify_orientation else None,
            # aux task: chexpert labels
            classify_chexpert = self.classify_chexpert,
            chexpert_labels = self.chexpert_labels if self.classify_chexpert else None,
            # aux task: question labels
            classify_questions = self.classify_questions,
            question_labels = self.question_labels if self.classify_questions else None,
            # infinite mode
            infinite=infinite,
            shuffle_indices=shuffle,
        )

    def _generate_val_dataset_and_dataloader(self, batch_size, collate_batch_fn, num_workers):
        print('len(self.val_indices) =', len(self.val_indices))
        self.val_dataset = self._get_visual_module_dataset(self.val_indices, infinite=False, shuffle=False)
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=collate_batch_fn,
                                         num_workers=num_workers,
                                         pin_memory=True)

    def _generate_train_dataset_and_dataloader__question_balanced(self, batch_size, collate_batch_fn, num_workers):
        def _indices_generator():
            for indices in self.train_indices.values():
                for i in indices:
                    yield i
        indices = deduplicate_indices(_indices_generator(), self.report_ids)
        n_questions = self.question_labels.shape[1]
        print(f'len(indices) = {len(indices)}, n_questions = {n_questions}')

        question_datasets = []
        for i in range(n_questions):
            pos_indices = []
            neg_indices = []
            for j in indices:
                if self.question_labels[self.report_ids[j]][i] == 1:
                    pos_indices.append(j)
                else:
                    neg_indices.append(j)
            
            if i % 6 == 0:
                print(f'qlabel = {i}, len(pos_indices)={len(pos_indices)}, len(neg_indices)={len(neg_indices)}')
            
            if len(pos_indices) >= 5 and len(neg_indices) >= 5:            
                # positive
                pos_indices = np.array(pos_indices, dtype=int)
                pos_dataset = self._get_visual_module_dataset(pos_indices)

                # negative
                neg_indices = np.array(neg_indices, dtype=int)
                neg_dataset = self._get_visual_module_dataset(neg_indices)

                # merged
                comp_dataset = CompositeInfiniteDataset([pos_dataset, neg_dataset], [1, 1])
                question_datasets.append(comp_dataset)
        
        # final dataset
        self.train_dataset = CompositeInfiniteDataset(question_datasets, [1] * len(question_datasets))

        # dataloader
        self.train_dataloader = DataLoader(self.train_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        collate_fn=collate_batch_fn,
                                        pin_memory=True)            
    
class VM_Evaluator(VM_Base):
    
    def __init__(self, transform, batch_size, collate_batch_fn,
                preprocessed_data_path,
                cache_dir,
                num_workers,
                classify_tags = False,
                rid2tags_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
        ):

        rid2tags_path = os.path.join(cache_dir, rid2tags_filename) if classify_tags else None
        chexpert_labels_path = os.path.join(cache_dir, chexpert_labels_filename) if classify_chexpert else None        
        question_labels_path = os.path.join(cache_dir, question_labels_filename) if classify_questions else None
    
        super().__init__(False, transform, batch_size, collate_batch_fn,
                preprocessed_data_path,
                num_workers,
                classify_tags = classify_tags,
                rid2tags_path = rid2tags_path,
                classify_orientation = classify_orientation,
                classify_chexpert = classify_chexpert,
                classify_questions = classify_questions,
                chexpert_labels_path = chexpert_labels_path,
                question_labels_path = question_labels_path)

    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers):
        self.test_indices = list(range(len(self.report_ids)))
        self.test_indices = deduplicate_indices(self.test_indices, self.report_ids)
        self._generate_test_dataset()
        self._generate_test_dataloader(batch_size, collate_batch_fn, num_workers)

    def _generate_test_dataset(self):

        print('generating test dataset ...')       
        
        self.test_dataset = VisualModuleDataset(
            self.report_ids, self.images, self.test_indices,
            self.transform, self.dataset_name,
            # aux task: medical tags prediction
            classify_tags = self.classify_tags,
            rid2tags = self.rid2tags if self.classify_tags else None,
            # aux task: orientation
            classify_orientation = self.classify_orientation,
            orientations = self.orientations if self.classify_orientation else None,
            # aux task: chexpert labels
            classify_chexpert = self.classify_chexpert,
            chexpert_labels = self.chexpert_labels if self.classify_chexpert else None,
            # aux task: question labels
            classify_questions = self.classify_questions,
            question_labels = self.question_labels if self.classify_questions else None,
        )
        
            
    def _generate_test_dataloader(self, batch_size, collate_batch_fn, num_workers):

        print('generating test dataloader ...')
        
        self.test_dataloader = DataLoader(self.test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         collate_fn=collate_batch_fn,
                                         pin_memory=True)
