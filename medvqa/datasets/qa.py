import numpy as np
from torch.utils.data import Dataset, DataLoader
from medvqa.utils.files_utils import load_pickle
from medvqa.datasets.dataloading_utils import (
    BatchedCompositeInfiniteDataset,
    get_imbalance_reduced_weights,
    INFINITE_DATASET_LENGTH,
)

class QADataset(Dataset):
    
    def __init__(self, report_ids, questions, answers, indices,
                suffle_indices = True, infinite = False,
        ):
        self.report_ids = report_ids
        self.questions = questions
        self.answers = answers
        self.indices = indices
        self.infinite = infinite

        if suffle_indices:
            np.random.shuffle(self.indices)

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
            q=self.questions[idx],
            a=self.answers[idx],
        )
        return output

class QA_Base:

    def __init__(self, training, batch_size, collate_batch_fn,
                preprocessed_data_path,                
                num_workers,
        ):
        assert preprocessed_data_path is not None        
        self.preprocessed_data_path = preprocessed_data_path
        self.training = training
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
        self.questions = data['questions']
        if 'question_ids' in data: # backward compatible hack
            self.question_ids = data['question_ids']
        self.answers = data['answers']
        if self.training:
            self.train_indices = data['train_indices']
            self.val_indices = data['val_indices']
        print ('\tDone!')

class QA_Trainer(QA_Base):
    
    def __init__(self, batch_size, collate_batch_fn, preprocessed_data_path, num_workers,
                imbalance_reduction_coef = 1,
                validation_only=False):

        self.imbalance_reduction_coef = imbalance_reduction_coef
        self.validation_only = validation_only

        super().__init__(True, batch_size, collate_batch_fn, preprocessed_data_path, num_workers)

    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers):
        if not self.validation_only:
            self._generate_train_dataset(batch_size)
        self._generate_val_dataset()
        self._generate_train_val_dataloaders(batch_size, collate_batch_fn, num_workers)

    def _generate_train_dataset(self, batch_size):
        
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
            question_datasets.append(QADataset(
                self.report_ids, self.questions, self.answers, indices, infinite=True,
            ))
            i = j+1
        
        question_weights = get_imbalance_reduced_weights([len(d) for d in question_datasets], self.imbalance_reduction_coef)
        self.train_dataset = BatchedCompositeInfiniteDataset(question_datasets, question_weights, batch_size)
        print(f'\tlen(question_datasets) = {len(question_datasets)}')

    def _generate_val_dataset(self):
        # validation dataset
        self.val_dataset = QADataset(
            self.report_ids, self.questions, self.answers, self.val_indices,
        )
            
    def _generate_train_val_dataloaders(self, batch_size, collate_batch_fn, num_workers):

        print('generating training and validation dataloaders ...')
        print('num_workers =', num_workers)
        
        if not self.validation_only:
            self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            collate_fn=collate_batch_fn,
                                            num_workers=num_workers,
                                            pin_memory=True)
        
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=collate_batch_fn,
                                         num_workers=num_workers,
                                         pin_memory=True)

class QA_Evaluator(QA_Base):
    
    def __init__(self, batch_size, collate_batch_fn, preprocessed_data_path, num_workers):
    
        super().__init__(False, batch_size, collate_batch_fn, preprocessed_data_path, num_workers)

    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers):
        self.test_indices = list(range(len(self.report_ids)))
        self._generate_test_dataset()
        self._generate_test_dataloader(batch_size, collate_batch_fn, num_workers)

    def _generate_test_dataset(self):
        print('generating test dataset ...')        
        self.test_dataset = QADataset(
            self.report_ids, self.questions, self.answers, self.test_indices)
            
    def _generate_test_dataloader(self, batch_size, collate_batch_fn, num_workers):
        print('generating test dataloader ...')        
        self.test_dataloader = DataLoader(self.test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=collate_batch_fn,
                                         num_workers=num_workers,
                                         pin_memory=True)
