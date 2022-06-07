import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.utils.files import (
    load_pickle,
    get_cached_json_file,
    save_to_pickle,
)
from medvqa.utils.files import MAX_FILENAME_LENGTH
from medvqa.utils.hashing import hash_string
from medvqa.datasets.dataloading_utils import (
    CompositeInfiniteDataset,
    BatchedCompositeInfiniteDataset,
    get_imbalance_reduced_weights,
    INFINITE_DATASET_LENGTH,
)

def _split_data_train_val(report_ids, question_ids, answers, n_vals_per_question=4, min_question_count=100):
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
    
    val_rids = set()
    forbidden_rids = set()
    for list_ in tmp.values():
        list_.sort()
        if len(list_) >= min_question_count:
            chunk_size = len(list_) // n_vals_per_question
            offset = 0
            while offset < len(list_):
                min_i = offset
                max_i = min(offset + chunk_size, len(list_)) - 1
                if max_i - min_i + 1 == chunk_size:
                    val_i = random.randint(min_i, max_i)
                    val_rids.add(report_ids[list_[val_i][1]])
                offset += chunk_size
        else:
            forbidden_rids.update(report_ids[e[1]] for e in list_)
    
    # split indices into train and validation
    train_indices = { qid:[] for qid in tmp.keys() }
    val_indices = []
    for qid, list_ in tmp.items():
        for _, i in list_:
            if report_ids[i] in val_rids and report_ids[i] not in forbidden_rids:
                val_indices.append(i)
            else:
                train_indices[qid].append(i)

    # convert to numpy arrays
    for qid in train_indices.keys():
        train_indices[qid] = np.array(train_indices[qid], dtype=int)
    val_indices = np.array(val_indices, dtype=int)

    return train_indices, val_indices

def _get_diverse_samples(k, indices, class_getter, max_tries=200):
    samples = []
    classes = []
    tries = 0
    while len(samples) < k and tries < max_tries:
        tries += 1
        i = random.choice(indices)
        c = class_getter(i)
        if c in classes: continue
        samples.append(i)
        classes.append(c)
    assert len(samples) > 0
    return samples

def _split_data_train_val__balanced(report_ids, question_ids, balanced_metadata,
        n_healthy_per_question=2, n_unhealthy_per_question=3, min_question_count=50):    

    assert n_healthy_per_question < min_question_count
    assert n_unhealthy_per_question < min_question_count

    health_metadata = balanced_metadata['healthy']
    tags_based_class_metadata = balanced_metadata['tags_based_class']
    # split indices by question
    n = len(report_ids)
    qid2indices = dict()
    for i in range(n):
        qid = question_ids[i]
        try:
            tmp = qid2indices[qid]
        except KeyError:
            tmp = qid2indices[qid] = []
        tmp.append(i)
    # choose report ids for validation
    val_rids = set()
    forbidden_rids = set()
    for qid, indices in qid2indices.items():
        # split indices into healthy and unhealthy
        h_indices = []
        unh_indices = []
        for i in indices:
            healthy = health_metadata[report_ids[i]][str(qid)]
            if healthy: h_indices.append(i)
            else: unh_indices.append(i)
        
        # get diverse samples from each category         
        class_getter = lambda i : tags_based_class_metadata[report_ids[i]][str(qid)]
        
        if len(h_indices) >= min_question_count:
            h_samples = _get_diverse_samples(n_healthy_per_question, h_indices, class_getter)
            val_rids.update(report_ids[i] for i in h_samples)
        else:
            forbidden_rids.update(report_ids[i] for i in h_indices)

        if len(unh_indices) >= min_question_count:
            unh_samples = _get_diverse_samples(n_unhealthy_per_question, unh_indices, class_getter)
            val_rids.update(report_ids[i] for i in unh_samples)
        else:
            forbidden_rids.update(report_ids[i] for i in unh_indices)
    
    # split indices into train and validation
    train_indices = { qid:[] for qid in qid2indices.keys() }
    val_indices = []
    for qid, indices in qid2indices.items():    
        for i in indices:
            if report_ids[i] in val_rids and report_ids[i] not in forbidden_rids:
                val_indices.append(i)
            else:
                train_indices[qid].append(i)
    
    # convert to numpy arrays
    for qid in train_indices.keys():
        train_indices[qid] = np.array(train_indices[qid], dtype=int)
    val_indices = np.array(val_indices, dtype=int)

    print(f'balanced splitting: len(train_indices) = {sum(len(x) for x in train_indices.values())},'
          f' len(val_indices) = {len(val_indices)}')

    return train_indices, val_indices

class VQADataset(Dataset):
    
    def __init__(self, report_ids, images, questions, answers, indices, transform, source_dataset_name,
                include_answer = True,
                suffle_indices = True,
                # aux task: medical tags
                use_tags = False, rid2tags = None,
                # aux task: image orientation
                use_orientation = False, orientations = None,
                # aux task: chexpert labels
                use_chexpert = False, chexpert_labels = None,
                # aux task: question labels
                classify_questions = False, question_labels = None,
                # infinite mode
                infinite = False,
        ):
        self.report_ids = report_ids
        self.images = images
        self.questions = questions
        self.answers = answers
        self.indices = indices
        self.transform = transform
        self.source_dataset_name = source_dataset_name
        self.infinite = infinite
        self.include_answer = include_answer

        if suffle_indices:
            np.random.shuffle(self.indices)
        
        # optional auxiliary tasks
        self.use_tags = use_tags
        self.rid2tags = rid2tags
        self.use_orientation = use_orientation
        self.orientations = orientations
        self.use_chexpert = use_chexpert
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
            q=self.questions[idx],
        )
        if self.include_answer:
            output['a'] = self.answers[idx]
        rid = self.report_ids[idx]
        if self.use_tags:
            output['tags'] = self.rid2tags[rid]
        if self.use_orientation:
            output['orientation'] = self.orientations[idx]
        if self.use_chexpert:
            output['chexpert'] = self.chexpert_labels[rid]
        if self.classify_questions:
            output['qlabels'] = self.question_labels[rid]
        return output

class VQA_Base:

    def __init__(self, training, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,                
                num_workers,
                use_tags = False,
                rid2tags_path = None,
                use_orientation = False,
                use_chexpert = False,
                classify_questions = False,
                chexpert_labels_path = None,
                question_labels_path = None,
                dataset_name = None,
                split_kwargs = None,
                load_split_from_path = None,
                balanced_split = False,
                balanced_metadata_path = None,
                use_report_eval_mode = False,
                debug = False,
        ):
        
        self.preprocessing_save_path = preprocessing_save_path
        self.training = training
        self.transform = transform
        self.use_tags = use_tags
        self.use_orientation = use_orientation
        self.use_chexpert = use_chexpert
        self.classify_questions = classify_questions
        self.balanced_split = balanced_split
        self.dataset_name = dataset_name
        self.use_report_eval_mode = use_report_eval_mode
        
        if use_tags:
            assert rid2tags_path is not None
            self.rid2tags = load_pickle(rid2tags_path)

        if use_chexpert:
            assert chexpert_labels_path is not None
            self.chexpert_labels = load_pickle(chexpert_labels_path)
        
        if classify_questions:
            assert question_labels_path is not None
            self.question_labels = load_pickle(question_labels_path)
        
        if balanced_split:
            assert balanced_metadata_path is not None
            self.balanced_metadata = load_pickle(balanced_metadata_path)

        if use_report_eval_mode:
            assert load_split_from_path is not None
            self._load_split_from_path(load_split_from_path)
            if (not self._load_cached_data(preprocessing_save_path)):
                self._preprocess_data()
                self._save_data(preprocessing_save_path)
            self.val_indices = np.array(list(range(len(self.report_ids))), dtype=int)
        else:        
            if not debug:            
                first_time = not self._load_cached_data(preprocessing_save_path)
                if not first_time and load_split_from_path is not None:
                    self._load_split_from_path(load_split_from_path)

            if debug or first_time:                
                self._preprocess_data()
                if training:
                    if load_split_from_path is not None:
                        self._load_split_from_path(load_split_from_path)
                    elif balanced_split:
                        self._split_data_train_val__balanced(**split_kwargs)
                    else:
                        self._split_data_train_val(**split_kwargs)                
                if not debug:
                    self._save_data(preprocessing_save_path)        
        
        print(f'batch_size = {batch_size}')
        self._generate_datasets_and_dataloaders(batch_size, collate_batch_fn, num_workers)
        print('done!')

    def __len__(self):
        return len(self.report_ids)
    
    def _preprocess_data(self):
        raise NotImplementedError('Make sure your specialized class implements this function')
    
    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers):
        raise NotImplementedError('Make sure your specialized class implements this function')

    def _load_split_from_path(self, preprocessing_save_path):
        data = load_pickle(preprocessing_save_path)
        if self.use_report_eval_mode:
            report_ids = data['report_ids']
            self.train_report_ids = list(set(report_ids[i] for i in data['train_indices']))
            self.val_report_ids = list(set(report_ids[i] for i in data['val_indices']))
        else:
            self.train_indices = data['train_indices']
            self.val_indices = data['val_indices']
        print(f'Train-val split indices successfully loaded from {preprocessing_save_path}')
    
    def _load_cached_data(self, preprocessing_save_path):
        print(f'Checking if data is already cached in path {preprocessing_save_path} ...')
        data = load_pickle(preprocessing_save_path)
        if data is None:
            print('\tNo, it isn\'t :(')
            return False        
        self.dataset_name = data['dataset_name']
        self.report_ids = data['report_ids']
        self.images = data['images']
        self.questions = data['questions']
        if 'question_ids' in data: # backward compatible hack
            self.question_ids = data['question_ids']
        self.answers = data['answers']
        if self.training:
            self.train_indices = data['train_indices']
            self.val_indices = data['val_indices']
        if self.use_orientation:
            self.orientations = data['orientations']
        print('\tYes, it is, data successfully loaded :)')
        return True
    
    def _save_data(self, preprocessing_save_path):
        print('saving data to', preprocessing_save_path)
        data = dict(
            dataset_name = self.dataset_name,
            report_ids = self.report_ids,
            images = self.images,
            questions = self.questions,
            question_ids = self.question_ids,
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
        
        print('Splitting data into training and validation ...')
        
        train_indices, val_indices = _split_data_train_val(self.question_ids,
                                                          self.answers,
                                                          n_val_examples_per_question,
                                                          min_train_examples_per_question)                
        self.train_indices = train_indices
        self.val_indices = val_indices
    
    def _split_data_train_val__balanced(self,
                                        n_healthy_per_question=2,
                                        n_unhealthy_per_question=3,
                                        min_question_count=100):

        print('Splitting data into training and validation in balanced mode ...')

        train_indices, val_indices = _split_data_train_val__balanced(self.report_ids,
                                                                     self.question_ids,
                                                                     self.balanced_metadata,
                                                                     n_healthy_per_question,
                                                                     n_unhealthy_per_question,
                                                                     min_question_count)
        self.train_indices = train_indices
        self.val_indices = val_indices

class BalancedTreeNode:
    
    def __init__(self, indices=None, children=None, weights=None):
        # print(f'BalancedTreeNode()')
        # if indices is not None: print(f'\tlen(indices)={len(indices)}')
        # if children is not None: print(f'\tlen(children)={len(children)}')
        # if weights is not None: print(f'\tlen(weights)={len(weights)}')
        self.indices = indices
        self.weights = weights
        self.children = children
    
    def get_dataset(self, vqa_trainer):
        if self.children is None: # this is a leaf node
            assert len(self.indices) > 0
            questions = vqa_trainer.questions if vqa_trainer.verbose_question else vqa_trainer.question_ids
            return VQADataset(
                vqa_trainer.report_ids, vqa_trainer.images, questions, vqa_trainer.answers,
                self.indices, vqa_trainer.transform, vqa_trainer.dataset_name,
                # aux task: medical tags
                use_tags = vqa_trainer.use_tags,
                rid2tags = vqa_trainer.rid2tags if vqa_trainer.use_tags else None,
                # aux task: orientation
                use_orientation = vqa_trainer.use_orientation,
                orientations = vqa_trainer.orientations if vqa_trainer.use_orientation else None,
                # aux task: chexpert labels
                use_chexpert = vqa_trainer.use_chexpert,
                chexpert_labels = vqa_trainer.chexpert_labels if vqa_trainer.use_chexpert else None,
                # aux task: question labels
                classify_questions = vqa_trainer.classify_questions,
                question_labels = vqa_trainer.question_labels if vqa_trainer.classify_questions else None,
                # infinite mode
                infinite=True,
            )
        else:
            assert len(self.children) == len(self.weights)
            datasets = [child.get_dataset(vqa_trainer) for child in self.children]
            return CompositeInfiniteDataset(datasets, self.weights)


def _get_cached_balanced_train_datasets_path(preprocessing_save_path, batch_size, allowed_question_ids=None):    
    if allowed_question_ids is not None:
        aqi_string = '(' + ','.join(map(str, allowed_question_ids)) + ')'
        filepath = f'{preprocessing_save_path}.balanced_train_data(bs={batch_size},aqi={aqi_string}).pkl'
    else:
        filepath = f'{preprocessing_save_path}.balanced_train_data(bs={batch_size}).pkl'

    if len(filepath) > MAX_FILENAME_LENGTH:
        h = hash_string(filepath)
        filepath = os.path.join(os.path.dirname(preprocessing_save_path), f'balanced_train_data(hash={h[0]},{h[1]}).pkl')
    return filepath

class VQA_Trainer(VQA_Base):
    
    def __init__(self, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                cache_dir,
                num_workers,
                verbose_question = True,
                use_tags = False,
                rid2tags_filename = None,
                use_orientation = False,
                use_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                dataset_name = None,
                split_kwargs = None,
                load_split_from_path = None,
                balanced_split = False,
                balanced_dataloading = False,
                balanced_metadata_filename = None,
                imbalance_reduction_coef = 1,
                validation_only = False,
                include_answer = True,
                use_report_eval_mode = False,
                allowed_questions = None,
                qa_adapted_reports_filename = None,
                one_question_per_batch = False,
                debug = False,
        ):

        rid2tags_path = os.path.join(cache_dir, rid2tags_filename) if use_tags else None
        chexpert_labels_path = os.path.join(cache_dir, chexpert_labels_filename) if use_chexpert else None
        balanced_metadata_path = os.path.join(cache_dir, balanced_metadata_filename) if balanced_split else None
        question_labels_path = os.path.join(cache_dir, question_labels_filename) if classify_questions else None

        if balanced_dataloading:
            assert balanced_split
            assert balanced_metadata_path is not None

        self.verbose_question = verbose_question
        self.balanced_dataloading = balanced_dataloading
        self.validation_only = validation_only
        self.cache_dir = cache_dir
        self.imbalance_reduction_coef = imbalance_reduction_coef
        self.one_question_per_batch = one_question_per_batch
        self.include_answer = include_answer
        
        if allowed_questions is not None:
            assert qa_adapted_reports_filename is not None
            allowed_questions = [q.strip() for q in allowed_questions.split('?')]
            allowed_questions = [q + '?' for q in allowed_questions if len(q) > 0]
            qa_adapted_reports_path = os.path.join(cache_dir, qa_adapted_reports_filename)
            qa_adapted_reports = get_cached_json_file(qa_adapted_reports_path)
            assert all(q in qa_adapted_reports['questions'] for q in allowed_questions)
            self.allowed_question_ids = [qa_adapted_reports['questions'].index(q) for q in allowed_questions]
        else:
            self.allowed_question_ids = None
    
        super().__init__(True, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                num_workers,
                use_tags = use_tags,
                rid2tags_path = rid2tags_path,
                use_orientation = use_orientation,
                use_chexpert = use_chexpert,
                classify_questions = classify_questions,
                chexpert_labels_path = chexpert_labels_path,
                question_labels_path = question_labels_path,
                dataset_name = dataset_name,
                split_kwargs = split_kwargs,
                load_split_from_path = load_split_from_path,
                balanced_split = balanced_split,
                balanced_metadata_path = balanced_metadata_path,
                use_report_eval_mode = use_report_eval_mode,
                debug = debug)

    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers):
        if not self.validation_only:
            if self.balanced_dataloading:
                self._generate_train_dataset__balanced(batch_size)
            else:
                self._generate_train_dataset(batch_size)
        self._generate_val_dataset()
        self._generate_train_val_dataloaders(batch_size, collate_batch_fn, num_workers)

    def _get_composite_dataset(self, datasets, weights, batch_size):
        if self.one_question_per_batch:
            print('(***) Note: using BatchedCompositeInfiniteDataset (one question per batch)')
            return BatchedCompositeInfiniteDataset(datasets, weights, batch_size)
        return CompositeInfiniteDataset(datasets, weights)

    def _generate_train_dataset(self, batch_size):

        question_datasets = []
        if self.allowed_question_ids is not None:
            train_question_ids = self.allowed_question_ids
        else:
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
            
            questions = self.questions if self.verbose_question else self.question_ids
            question_datasets.append(VQADataset(
                self.report_ids, self.images, questions, self.answers,
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
                # aux task: question labels
                classify_questions = self.classify_questions,
                question_labels = self.question_labels if self.classify_questions else None,
                # infinite mode
                infinite=True,
            ))
            i = j+1
        
        question_weights = get_imbalance_reduced_weights([len(d) for d in question_datasets], self.imbalance_reduction_coef)
        self.train_dataset = self._get_composite_dataset(question_datasets, question_weights, batch_size)
        print(f'\tlen(question_datasets) = {len(question_datasets)}')
    
    def _generate_train_dataset__balanced(self, batch_size):

        cached_balanced_train_datasets_path = _get_cached_balanced_train_datasets_path(
            self.preprocessing_save_path, batch_size, self.allowed_question_ids)
        cached_data = load_pickle(cached_balanced_train_datasets_path)
        if cached_data is not None:
            print(f'Balanced train data loaded from {cached_balanced_train_datasets_path}')
            question_datasets = [node.get_dataset(self) for node in cached_data['question_nodes']]
            question_weights = cached_data['question_weights']
            self.train_dataset = self._get_composite_dataset(question_datasets, question_weights, batch_size)
            print(f'\tlen(question_datasets) = {len(question_datasets)}')
            return

        print('Computing balanced datasets from scratch ...')

        health_metadata = self.balanced_metadata['healthy']
        tags_based_class_metadata = self.balanced_metadata['tags_based_class']
        report_ids = self.report_ids
        question_ids = self.question_ids

        train_nodes = []
        if self.allowed_question_ids is not None:
            train_question_ids = self.allowed_question_ids
        else:
            train_question_ids = list(self.train_indices.keys())
        train_question_ids.sort(key=lambda x : len(self.train_indices[x]))
        
        for qid in tqdm(train_question_ids):
            indices = self.train_indices[qid]
            healthy_indices = [j for j in indices if health_metadata[report_ids[j]][str(question_ids[j])] == 1]
            unhealthy_indices = [j for j in indices if health_metadata[report_ids[j]][str(question_ids[j])] == 0]
            assert len(healthy_indices) + len(unhealthy_indices) == len(indices)

            nodes_by_health = []
            weights_by_health = []

            for indices in [healthy_indices, unhealthy_indices]:

                if len(indices) == 0:
                    continue
            
                indices_by_class = dict()
                for j in indices:
                    class_id = tags_based_class_metadata[report_ids[j]][str(question_ids[j])]
                    try:
                        tmp = indices_by_class[class_id]
                    except KeyError:
                        tmp = indices_by_class[class_id] = []
                    tmp.append(j)

                nodes = []
                weights = []

                for sub_indices in indices_by_class.values():

                    assert len(sub_indices) > 0

                    nodes.append(BalancedTreeNode(indices=sub_indices))
                    weights.append(len(sub_indices))

                if len(nodes) > 1:
                    weights = get_imbalance_reduced_weights(weights, self.imbalance_reduction_coef)
                    nodes_by_health.append(BalancedTreeNode(children=nodes, weights=weights))
                else:
                    assert len(nodes) == 1
                    nodes_by_health.append(nodes[0])
                weights_by_health.append(len(indices))
            
            if len(nodes_by_health) > 1:
                weights_by_health = get_imbalance_reduced_weights(weights_by_health, self.imbalance_reduction_coef)
                train_nodes.append(BalancedTreeNode(children=nodes_by_health, weights=weights_by_health))
            else:
                assert len(nodes_by_health) == 1
                train_nodes.append(nodes_by_health[0])

        i = 0
        n = len(train_nodes)        
        merged_train_nodes = []
        dataset_weights = []
        while i < n:
            acc_size = len(self.train_indices[train_question_ids[i]])
            j = i
            while j + 1 < n and acc_size < batch_size:
                j += 1
                acc_size += len(self.train_indices[train_question_ids[j]])
            if i == j:
                merged_train_nodes.append(train_nodes[i])
            else:
                sub_nodes = [train_nodes[k] for k in range(i, j+1)]
                weights = [len(self.train_indices[train_question_ids[k]]) for k in range(i, j+1)]
                weights = get_imbalance_reduced_weights(weights, self.imbalance_reduction_coef)
                merged_train_nodes.append(BalancedTreeNode(children=sub_nodes, weights=weights))
                print(f' *** merging from i={i} to j={j}, acc_size = {acc_size}')
            i = j + 1
            dataset_weights.append(acc_size)
        
        cached_data = {
            'question_nodes': merged_train_nodes,
            'question_weights': get_imbalance_reduced_weights(dataset_weights, self.imbalance_reduction_coef),
        }
        save_to_pickle(cached_data, cached_balanced_train_datasets_path)
        print(f'Balanced train data saved to {cached_balanced_train_datasets_path}')
        question_datasets = [node.get_dataset(self) for node in cached_data['question_nodes']]
        question_weights = cached_data['question_weights']
        self.train_dataset = self._get_composite_dataset(question_datasets, question_weights, batch_size)
        print(f'\tlen(question_datasets) = {len(question_datasets)}')

    def _generate_val_dataset(self):
        if self.allowed_question_ids is not None:
            val_indices = np.array([i for i in self.val_indices if self.question_ids[i] in self.allowed_question_ids], dtype=int)
        else:
            val_indices = self.val_indices
        print('len(self.val_indices) =', len(self.val_indices))
        print('len(val_indices) =', len(val_indices))
        questions = self.questions if self.verbose_question else self.question_ids
        self.val_dataset = VQADataset(
            self.report_ids, self.images, questions, self.answers, val_indices,
            self.transform, self.dataset_name,
            include_answer = self.include_answer,
            # aux task: medical tags
            use_tags = self.use_tags,
            rid2tags = self.rid2tags if self.use_tags else None,
            # aux task: orientation
            use_orientation = self.use_orientation,
            orientations = self.orientations if self.use_orientation else None,
            # aux task: chexpert labels
            use_chexpert = self.use_chexpert,
            chexpert_labels = self.chexpert_labels if self.use_chexpert else None,
            # aux task: question labels
            classify_questions = self.classify_questions,
            question_labels = self.question_labels if self.classify_questions else None,
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

class VQA_Evaluator(VQA_Base):
    
    def __init__(self, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                cache_dir,
                num_workers,
                verbose_question = True,
                include_answer = True,
                use_tags = False,
                rid2tags_filename = None,
                use_orientation = False,
                use_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                dataset_name = None,
                debug = False,
        ):

        rid2tags_path = os.path.join(cache_dir, rid2tags_filename) if use_tags else None
        chexpert_labels_path = os.path.join(cache_dir, chexpert_labels_filename) if use_chexpert else None        
        question_labels_path = os.path.join(cache_dir, question_labels_filename) if classify_questions else None

        self.verbose_question = verbose_question
        self.include_answer = include_answer
    
        super().__init__(False, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                num_workers,
                use_tags = use_tags,
                rid2tags_path = rid2tags_path,
                use_orientation = use_orientation,
                use_chexpert = use_chexpert,
                classify_questions = classify_questions,
                chexpert_labels_path = chexpert_labels_path,
                question_labels_path = question_labels_path,
                dataset_name = dataset_name,
                debug = debug)

    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers):
        self.test_indices = list(range(len(self.report_ids)))
        self._generate_test_dataset()
        self._generate_test_dataloader(batch_size, collate_batch_fn, num_workers)

    def _generate_test_dataset(self):

        print('generating test dataset ...')
        
        questions = self.questions if self.verbose_question else self.question_ids
        self.test_dataset = VQADataset(
            self.report_ids, self.images, questions, self.answers, self.test_indices,
            self.transform, self.dataset_name,
            include_answer = self.include_answer,
            # aux task: medical tags prediction
            use_tags = self.use_tags,
            rid2tags = self.rid2tags if self.use_tags else None,
            # aux task: orientation
            use_orientation = self.use_orientation,
            orientations = self.orientations if self.use_orientation else None,
            # aux task: chexpert labels
            use_chexpert = self.use_chexpert,
            chexpert_labels = self.chexpert_labels if self.use_chexpert else None,
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
