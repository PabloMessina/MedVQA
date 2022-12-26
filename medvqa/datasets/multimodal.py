import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from medvqa.utils.files import load_pickle, save_to_pickle
from medvqa.datasets.dataloading_utils import (
    CompositeInfiniteDataset,
    get_imbalance_reduced_weights,
    INFINITE_DATASET_LENGTH,
)

class MultimodalDataset(Dataset):
    
    def __init__(self, report_ids, images, texts, indices, transform,
                 shuffle_indices = True, use_text = True,
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
        self.texts = texts
        self.indices = indices
        self.transform = transform        
        self.infinite = infinite
        self.use_text = use_text

        if shuffle_indices:
            np.random.shuffle(self.indices)
        
        # optional auxiliary tasks
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
        rid = self.report_ids[idx]
        output = dict(
            idx=idx,
            i=self.transform(Image.open(self.images[idx]).convert('RGB')),
        )
        if self.use_text:
            output['t'] = self.texts[idx]
        if self.classify_orientation:
            output['orientation'] = self.orientations[idx]
        if self.classify_chexpert:
            output['chexpert'] = self.chexpert_labels[rid]
        if self.classify_questions:
            output['qlabels'] = self.question_labels[rid]
        return output

class MultiModal_Trainer():
    
    def __init__(self, transform, batch_size, collate_batch_fn,
                preprocessed_data_path,
                cache_dir,
                num_workers,
                use_text=True,
                classify_orientation=False,
                classify_chexpert=False,
                chexpert_labels_filename=None,
                classify_questions=False,
                question_labels_filename=None,
                imbalance_reduction_coef=0.4,
                include_train=True,
                include_test=True,
        ):
        
        self.cache_dir = cache_dir
        self.imbalance_reduction_coef = imbalance_reduction_coef
        self.preprocessed_data_path = preprocessed_data_path
        self.transform = transform
        self.classify_orientation = classify_orientation
        self.classify_chexpert = classify_chexpert
        self.classify_questions = classify_questions
        self.include_train = include_train
        self.include_test = include_test
        self.use_text = use_text

        if (not self._load_cached_data(preprocessed_data_path)):
                self._preprocess_data()
                self._save_data(preprocessed_data_path)
            
        assert chexpert_labels_filename is not None
        self.chexpert_labels = load_pickle(os.path.join(cache_dir, chexpert_labels_filename))
        
        assert question_labels_filename is not None
        self.question_labels = load_pickle(os.path.join(cache_dir, question_labels_filename))
                
        self._generate_datasets_and_dataloaders(batch_size, collate_batch_fn, num_workers)

    def _preprocess_data(self):
        raise NotImplementedError('Make sure your specialized class implements this method')
    
    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers):
        print('generating datasets and dataloaders ...')
        print('num_workers =', num_workers)
        if self.include_train:
            self._generate_train_dataset_and_dataloader__question_balanced(batch_size, collate_batch_fn, num_workers)
            print('_generate_train_dataset_and_dataloader__question_balanced()')
            self._generate_train_dataset_and_dataloader__chexpert_balanced(batch_size, collate_batch_fn, num_workers)
        if self.include_test:
            self._generate_test_dataset_and_dataloader(batch_size, collate_batch_fn, num_workers)

    def _load_cached_data(self, preprocessed_data_path):
        print(f'Checking if data is already cached in path {preprocessed_data_path} ...')
        data = load_pickle(preprocessed_data_path)
        if data is None:
            print('\tNo, it isn\'t :(')
            return False
        self.report_ids = data['report_ids']
        self.images = data['images']
        print('_generate_train_dataset_and_dataloader__chexpert_balanced()')
        self.backgrounds = data['backgrounds']
        self.orientations = data['orientations']
        if self.include_train:
            self.train_indices = data['train_indices']
        if self.include_test:
            self.test_indices = data['test_indices']
        print('\tYes, it is, data successfully loaded :)')
        return True

    def _save_data(self, preprocessed_data_path):
        print('Saving data to', preprocessed_data_path)
        data = dict(
            report_ids = self.report_ids,
            images = self.images,
            backgrounds = self.backgrounds,
            orientations = self.orientations,
            train_indices = self.train_indices,
        )
        if self.include_train:
            data['train_indices'] = self.train_indices
        if self.include_test:
            data['test_indices'] = self.test_indices
        save_to_pickle(data, preprocessed_data_path)
        print('\tDone!')

    def _create_multimodal_dataset(self, indices, infinite=True, shuffle=True):
        return MultimodalDataset(
            report_ids = self.report_ids,
            images = self.images,
            texts = self.backgrounds,
            indices = indices,
            transform = self.transform,
            use_text = self.use_text,
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

    def _generate_test_dataset_and_dataloader(self, batch_size, collate_batch_fn, num_workers):
        print('len(self.test_indices) =', len(self.test_indices))
        self.test_dataset = self._create_multimodal_dataset(self.test_indices, infinite=False, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=collate_batch_fn,
                                         num_workers=num_workers,
                                         pin_memory=True)

    def _generate_train_dataset_and_dataloader__label_balanced(
            self, labels, batch_size, collate_batch_fn, num_workers, print_every=1, log_weighting=False):
        n_labels = labels.shape[1]
        print(f'len(self.train_indices) = {len(self.train_indices)}, n_labels = {n_labels}')

        datasets = []
        pos_counts = []
        for i in range(n_labels):
            pos_indices = []
            neg_indices = []
            for j in self.train_indices:
                if labels[self.report_ids[j]][i] == 1:
                    pos_indices.append(j)
                else:
                    neg_indices.append(j)
            
            if i % print_every == 0:
                print(f'label = {i}, len(pos_indices)={len(pos_indices)}, len(neg_indices)={len(neg_indices)}')
            
            if len(pos_indices) >= 5 and len(neg_indices) >= 5:
                
                pos_counts.append(len(pos_indices))

                # positive
                pos_indices = np.array(pos_indices, dtype=int)
                pos_dataset = self._create_multimodal_dataset(pos_indices)

                # negative
                neg_indices = np.array(neg_indices, dtype=int)
                neg_dataset = self._create_multimodal_dataset(neg_indices)

                # merged
                comp_dataset = CompositeInfiniteDataset([pos_dataset, neg_dataset], [1, 1])
                datasets.append(comp_dataset)
        
        # final dataset
        if log_weighting:
            weights = get_imbalance_reduced_weights(pos_counts, self.imbalance_reduction_coef)
        else: # uniform weights
            weights = [1] * len(datasets)
        dataset = CompositeInfiniteDataset(datasets, weights)

        # dataloader
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=collate_batch_fn,
                                pin_memory=True)

        return dataset, dataloader

    def _generate_train_dataset_and_dataloader__question_balanced(self, batch_size, collate_batch_fn, num_workers):
        print('_generate_train_dataset_and_dataloader__question_balanced()')
        dataset, dataloader = self._generate_train_dataset_and_dataloader__label_balanced(
            labels = self.question_labels,
            batch_size = batch_size,
            collate_batch_fn = collate_batch_fn,
            num_workers = num_workers,
            print_every = 6,
            log_weighting = True,
        )
        self.train_dataset__question_balanced = dataset
        self.train_dataloader__question_balanced = dataloader

    def _generate_train_dataset_and_dataloader__chexpert_balanced(self, batch_size, collate_batch_fn, num_workers):
        print('_generate_train_dataset_and_dataloader__chexpert_balanced()')
        dataset, dataloader = self._generate_train_dataset_and_dataloader__label_balanced(
            labels = self.chexpert_labels,
            batch_size = batch_size,
            collate_batch_fn = collate_batch_fn,
            num_workers = num_workers,
            print_every = 1,
            log_weighting = False,
        )
        self.train_dataset__chexpert_balanced = dataset
        self.train_dataloader__chexpert_balanced = dataloader