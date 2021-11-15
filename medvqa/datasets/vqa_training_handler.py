import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from medvqa.utils.files import (
    load_pickle,
    save_to_pickle
)
from medvqa.utils.common import CACHE_DIR

def _split_data_train_val(question_ids, answers, n_vals_per_question=10, min_question_count=100):
    
    tmp = dict()
    for i, (qi, a) in enumerate(zip(question_ids, answers)):
        try:
            list_ = tmp[qi]
        except KeyError:
            list_ = tmp[qi] = []
        list_.append((len(a), i))
    train_indices = {k:[] for k in tmp.keys()}
    val_indices = []
    for k, list_ in tmp.items():
        list_.sort()
        if len(list_) >= min_question_count:
            chunk_size = len(list_) // n_vals_per_question
            offset = 0
            while offset < len(list_):
                min_i = offset
                max_i = min(offset + chunk_size, len(list_)) - 1
                if max_i - min_i + 1 == chunk_size:
                    val_i = random.randint(min_i, max_i)
                else:
                    val_i = None
                for i in range(min_i, max_i+1):
                    if i == val_i:
                        val_indices.append(list_[val_i][1])
                    else:
                        train_indices[k].append(list_[i][1])
                offset += chunk_size
        else:
            train_indices[k].extend(e[1] for e in list_)

    return train_indices, val_indices

class VQADataset(Dataset):
    
    def __init__(self, images, questions, answers, indices, transform, source_dataset_name):
        self.images = images
        self.questions = questions
        self.answers = answers
        self.indices = indices
        self.transform = transform
        self.source_dataset_name = source_dataset_name
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return dict(
            idx=idx,
            i=self.transform(Image.open(self.images[idx]).convert('RGB')),
            q=self.questions[idx],
            a=self.answers[idx],
        )

class VQATrainingHandler:
    
    def __init__(self, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                dataset_name = None,
                split_kwargs = None,
        ):
    
        self.transform = transform

        # create absolute path from relative path
        preprocessing_save_path = os.path.join(CACHE_DIR, preprocessing_save_path)

        first_time = not self._load_cached_data(preprocessing_save_path)

        if first_time:
            
            self.dataset_name = dataset_name            
            self._preprocess_data()
            self._split_data_train_val(**split_kwargs)
            self._save_data(preprocessing_save_path)
        
        self._generate_train_val_datasets()
        self._generate_train_val_dataloaders(batch_size, collate_batch_fn)
        print('done!')

    def __len__(self):
        return len(self.report_ids)
    
    def _preprocess_data(self):
        raise NotImplementedError('Make sure your especialized class implements this function')

    def _load_cached_data(self, preprocessing_save_path):
        print (f'checking if data is already cached in path {preprocessing_save_path} ...')
        data = load_pickle(preprocessing_save_path)
        if data is None:
            print('\tNo, it wasn\'t :(')
            return False        
        self.dataset_name = data['dataset_name']
        self.report_ids = data['report_ids']
        self.images = data['images']
        self.questions = data['questions']
        self.answers = data['answers']
        self.train_indices = data['train_indices']
        self.val_indices = data['val_indices']
        print ('\tYes, it was, data successfully loaded :)')
        return True
    
    def _save_data(self, preprocessing_save_path):
        print('saving data to', preprocessing_save_path)
        data = dict(
            dataset_name = self.dataset_name,
            report_ids = self.report_ids,
            images = self.images,
            questions = self.questions,
            answers = self.answers,
            train_indices = self.train_indices,
            val_indices = self.val_indices,
        )
        save_to_pickle(data, preprocessing_save_path)
        print('\tdone!')
        return True
    
    def _split_data_train_val(self,
                              n_val_examples_per_question=10,
                              min_train_examples_per_question=100):
        
        print('splitting data into training and validation ...')
        
        train_indices, val_indices = _split_data_train_val(self.question_ids,
                                                          self.answers,
                                                          n_val_examples_per_question,
                                                          min_train_examples_per_question)                
        self.train_indices = train_indices
        self.val_indices = val_indices

    def _generate_train_val_datasets(self):

        print('generating training and validation datasets ...')
        
        self.train_datasets = []
        for indices in self.train_indices.values():
            self.train_datasets.append(VQADataset(
                self.images, self.questions, self.answers, indices, self.transform, self.dataset_name))
        
        self.val_dataset = VQADataset(
            self.images, self.questions, self.answers, self.val_indices, self.transform, self.dataset_name)
        
            
    def _generate_train_val_dataloaders(self, batch_size, collate_batch_fn):

        print('generating training and validation dataloaders ...')
        
        self.train_dataloaders = []
        for dataset in self.train_datasets:
            self.train_dataloaders.append(DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     collate_fn=collate_batch_fn))
        
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=collate_batch_fn)