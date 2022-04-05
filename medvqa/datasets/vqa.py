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
    """Splits questions and answers into train and val splits.

    Question-answer pairs are sampled for validation in a stratified manner, by sorting answers
    by length (for each question id), splitting answers into bins and then sampling randomly
    from each bin.

    Args:
        question_ids (list of ints): list of question ids, each id can be mapped to the original question
        answers (list of tokenized answers): each answer is a list of tokens
        n_vals_per_question (int, optional): number of validation instances to sample per question. Defaults to 10.
        min_question_count (int, optional): minimun number of examples for a question id to be considered for validation. Defaults to 100.

    Returns:
        pair of dicts: each dict maps question ids to list of indices. The actual questions
        and answers can be recovered from these indices.
    """
    
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
    
    def __init__(self, report_ids, images, questions, answers, indices, transform, source_dataset_name,
                # aux task: medical tags
                use_tags = False, rid2tags = None,
                # aux task: image orientation
                use_orientation = False, orientations = None,
                # aux task: chexpert labels
                use_chexpert = False, chexpert_labels = None,
        ):
        self.report_ids = report_ids
        self.images = images
        self.questions = questions
        self.answers = answers
        self.indices = indices
        self.transform = transform
        self.source_dataset_name = source_dataset_name
        
        # optional auxiliary tasks
        self.use_tags = use_tags
        self.rid2tags = rid2tags
        self.use_orientation = use_orientation
        self.orientations = orientations
        self.use_chexpert = use_chexpert
        self.chexpert_labels = chexpert_labels
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        output = dict(
            idx=idx,
            i=self.transform(Image.open(self.images[idx]).convert('RGB')),
            q=self.questions[idx],
            a=self.answers[idx],
        )
        if self.use_tags:
            output['tags'] = self.rid2tags[self.report_ids[idx]]
        if self.use_orientation:
            output['orientation'] = self.orientations[idx]
        if self.use_chexpert:
            output['chexpert'] = self.chexpert_labels[self.report_ids[idx]]
        return output

class VQA_Base:

    def __init__(self, training, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                use_tags = False,
                rid2tags_path = None,
                use_orientation = False,
                use_chexpert = False,
                chexpert_labels_path = None,
                dataset_name = None,
                split_kwargs = None,
                debug = False,
        ):
    
        self.training = training
        self.transform = transform
        self.use_tags = use_tags
        self.use_orientation = use_orientation
        self.use_chexpert = use_chexpert
        
        if use_tags:
            assert rid2tags_path is not None
            self.rid2tags = load_pickle(rid2tags_path)

        if use_chexpert:
            assert chexpert_labels_path is not None
            self.chexpert_labels = load_pickle(chexpert_labels_path)

        # create absolute path from relative path
        if not debug:
            preprocessing_save_path = os.path.join(CACHE_DIR, preprocessing_save_path)
            first_time = not self._load_cached_data(preprocessing_save_path)

        if debug or first_time:
            
            self.dataset_name = dataset_name
            self._preprocess_data()

            if training:
                self._split_data_train_val(**split_kwargs)
            
            if not debug:
                self._save_data(preprocessing_save_path)
        
        self._generate_datasets_and_dataloaders(batch_size, collate_batch_fn)        
        print('done!')

    def __len__(self):
        return len(self.report_ids)
    
    def _preprocess_data(self):
        raise NotImplementedError('Make sure your specialized class implements this function')
    
    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn):
        raise NotImplementedError('Make sure your specialized class implements this function')

    def _load_cached_data(self, preprocessing_save_path):
        print (f'Checking if data is already cached in path {preprocessing_save_path} ...')
        data = load_pickle(preprocessing_save_path)
        if data is None:
            print('\tNo, it isn\'t :(')
            return False        
        self.dataset_name = data['dataset_name']
        self.report_ids = data['report_ids']
        self.images = data['images']
        self.questions = data['questions']
        self.answers = data['answers']
        if self.training:
            self.train_indices = data['train_indices']
            self.val_indices = data['val_indices']
        if self.use_orientation:
            self.orientations = data['orientations']
        print ('\tYes, it is, data successfully loaded :)')
        return True
    
    def _save_data(self, preprocessing_save_path):
        print('saving data to', preprocessing_save_path)
        data = dict(
            dataset_name = self.dataset_name,
            report_ids = self.report_ids,
            images = self.images,
            questions = self.questions,
            answers = self.answers,
        )
        if self.training:
            data['train_indices'] = self.train_indices
            data['val_indices'] = self.val_indices
        if self.use_orientation:
            data['orientations'] = self.orientations
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

class VQA_Trainer(VQA_Base):
    
    def __init__(self, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                use_tags = False,
                rid2tags_path = None,
                use_orientation = False,
                use_chexpert = False,
                chexpert_labels_path = None,
                dataset_name = None,
                split_kwargs = None,
                debug = False,
        ):
    
        super().__init__(True, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                use_tags = use_tags,
                rid2tags_path = rid2tags_path,
                use_orientation = use_orientation,
                use_chexpert = use_chexpert,
                chexpert_labels_path = chexpert_labels_path,
                dataset_name = dataset_name,
                split_kwargs = split_kwargs,
                debug = debug)

    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn):        
        self._generate_train_val_datasets(batch_size)
        self._generate_train_val_dataloaders(batch_size, collate_batch_fn)

    def _generate_train_val_datasets(self, batch_size):

        print('generating training and validation datasets ...')
        
        # train datasets: we define a different dataset per question, or combine questions
        # if they are too few for the given batch_size
        self.train_datasets = []
        train_question_ids = list(self.train_indices.keys())
        train_question_ids.sort(key=lambda x : len(self.train_indices[x]))
        acc_indices = []
        for i, qid in enumerate(train_question_ids):
            indices = self.train_indices[qid]
            if len(acc_indices) > 0 or len(indices) < batch_size:
                assert len(acc_indices) < batch_size
                acc_indices += indices
                indices = acc_indices
            if len(indices) >= batch_size or i+1 == len(train_question_ids):
                self.train_datasets.append(VQADataset(
                    self.report_ids, self.images, self.questions, self.answers,
                    indices, self.transform, self.dataset_name,
                    # aux task: medical tags
                    use_tags = self.use_tags,
                    rid2tags = self.rid2tags if self.use_tags else None,
                    # aux task: orientation
                    use_orientation = self.use_orientation,
                    orientations = self.orientations if self.use_orientation else None,
                    # aux task: chexpert labels
                    use_chexpert = self.use_chexpert,
                    chexpert_labels = self.chexpert_labels if self.use_chexpert else None,
                ))
                if len(acc_indices) > 0:
                    acc_indices = []
        print(f'\tlen(self.train_datasets) = {len(self.train_datasets)}')
        
        # validation dataset
        self.val_dataset = VQADataset(
            self.report_ids, self.images, self.questions, self.answers, self.val_indices,
            self.transform, self.dataset_name,
            # aux task: medical tags
            use_tags = self.use_tags,
            rid2tags = self.rid2tags if self.use_tags else None,
            # aux task: orientation
            use_orientation = self.use_orientation,
            orientations = self.orientations if self.use_orientation else None,
            # aux task: chexpert labels
            use_chexpert = self.use_chexpert,
            chexpert_labels = self.chexpert_labels if self.use_chexpert else None,
        )
        
            
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

class VQA_Evaluator(VQA_Base):
    
    def __init__(self, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,                
                use_tags = False,
                rid2tags_path = None,
                use_orientation = False,
                use_chexpert = False,
                chexpert_labels_path = None,
                dataset_name = None,
                debug = False,
        ):
    
        super().__init__(False, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                use_tags = use_tags,
                rid2tags_path = rid2tags_path,
                use_orientation = use_orientation,
                use_chexpert = use_chexpert,
                chexpert_labels_path = chexpert_labels_path,
                dataset_name = dataset_name,
                debug = debug)

    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn):
        self.test_indices = list(range(len(self.report_ids)))
        self._generate_test_dataset()
        self._generate_test_dataloader(batch_size, collate_batch_fn)

    def _generate_test_dataset(self):

        print('generating test dataset ...')
        
        self.test_dataset = VQADataset(
            self.report_ids, self.images, self.questions, self.answers, self.test_indices,
            self.transform, self.dataset_name,
            # aux task: medical tags prediction
            use_tags = self.use_tags,
            rid2tags = self.rid2tags if self.use_tags else None,
            # aux task: orientation
            use_orientation = self.use_orientation,
            orientations = self.orientations if self.use_orientation else None,
            # aux task: chexpert labels
            use_chexpert = self.use_chexpert,
            chexpert_labels = self.chexpert_labels if self.use_chexpert else None,
        )
        
            
    def _generate_test_dataloader(self, batch_size, collate_batch_fn):

        print('generating test dataloader ...')
        
        self.test_dataloader = DataLoader(self.test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=collate_batch_fn)
