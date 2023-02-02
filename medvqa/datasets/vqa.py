import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.utils import adapt_label_matrix_as_merged_findings, deduplicate_indices
from medvqa.utils.files import (
    get_cached_pickle_file,
    load_pickle,
    get_cached_json_file,
    save_to_pickle,
)
from medvqa.utils.files import MAX_FILENAME_LENGTH
from medvqa.utils.hashing import hash_string
from medvqa.datasets.dataloading_utils import (
    CompositeDataset,
    CompositeInfiniteDataset,
    BatchedCompositeInfiniteDataset,
    get_imbalance_reduced_weights,
    INFINITE_DATASET_LENGTH,
)
from medvqa.utils.constants import CHEXPERT_DATASET_ID, CHEXPERT_LABELS
from medvqa.models.report_generation.templates.chex_v1 import TEMPLATES_CHEXPERT_v1

# def _split_data_train_val(report_ids, question_ids, answers, n_vals_per_question=4, min_question_count=100):
#     """Splits questions and answers into train and val splits.

#     Question-answer pairs are sampled for validation in a stratified manner, by sorting answers
#     by length (for each question id), splitting answers into bins and then sampling randomly
#     from each bin.

#     Args:
#         question_ids (list of ints): list of question ids, each id can be mapped to the original question
#         answers (list of tokenized answers): each answer is a list of tokens
#         n_vals_per_question (int, optional): number of validation instances to sample per question. Defaults to 10.
#         min_question_count (int, optional): minimun number of examples for a question id to be considered for validation. Defaults to 100.

#     Returns:
#         pair of dicts: each dict maps question ids to list of indices. The actual questions
#         and answers can be recovered from these indices.
#     """
    
#     tmp = dict()
#     for i, (qi, a) in enumerate(zip(question_ids, answers)):
#         try:
#             list_ = tmp[qi]
#         except KeyError:
#             list_ = tmp[qi] = []
#         list_.append((len(a), i))
    
#     val_rids = set()
#     forbidden_rids = set()
#     for list_ in tmp.values():
#         list_.sort()
#         if len(list_) >= min_question_count:
#             chunk_size = len(list_) // n_vals_per_question
#             offset = 0
#             while offset < len(list_):
#                 min_i = offset
#                 max_i = min(offset + chunk_size, len(list_)) - 1
#                 if max_i - min_i + 1 == chunk_size:
#                     val_i = random.randint(min_i, max_i)
#                     val_rids.add(report_ids[list_[val_i][1]])
#                 offset += chunk_size
#         else:
#             forbidden_rids.update(report_ids[e[1]] for e in list_)
    
#     # split indices into train and validation
#     train_indices = { qid:[] for qid in tmp.keys() }
#     val_indices = []
#     for qid, list_ in tmp.items():
#         for _, i in list_:
#             if report_ids[i] in val_rids and report_ids[i] not in forbidden_rids:
#                 val_indices.append(i)
#             else:
#                 train_indices[qid].append(i)

#     # convert to numpy arrays
#     for qid in train_indices.keys():
#         train_indices[qid] = np.array(train_indices[qid], dtype=int)
#     val_indices = np.array(val_indices, dtype=int)

#     return train_indices, val_indices

# def _get_diverse_samples(k, indices, class_getter, max_tries=200):
#     samples = []
#     classes = []
#     tries = 0
#     while len(samples) < k and tries < max_tries:
#         tries += 1
#         i = random.choice(indices)
#         c = class_getter(i)
#         if c in classes: continue
#         samples.append(i)
#         classes.append(c)
#     assert len(samples) > 0
#     return samples

def _group_indices_by_question_id(indices, question_ids, convert_to_numpy=True):
    # group by question id
    tmp = dict()
    for i in indices:
        qid = question_ids[i]
        try:
            tmp[qid].append(i)
        except KeyError:
            tmp[qid] = [i]
    # convert to numpy arrays
    if convert_to_numpy:
        for qid in tmp.keys(): tmp[qid] = np.array(tmp[qid], dtype=int)
    return tmp

def _assign_all_data_to_train(question_ids):    
    n = len(question_ids)
    train_indices = _group_indices_by_question_id(range(n), question_ids)
    print(f'All data assigned to train: len(train_indices) = {sum(len(x) for x in train_indices.values())},'
          f' len(val_indices) = 0')
    return train_indices

# def _split_data_train_val__balanced(report_ids, question_ids, balanced_metadata,
#         n_healthy_per_question=2, n_unhealthy_per_question=3, min_question_count=50,
#         chexpert_labels=None, n_positive_per_chexpert_label=7):

#     assert n_healthy_per_question < min_question_count
#     assert n_unhealthy_per_question < min_question_count

#     health_metadata = balanced_metadata['healthy']
#     tags_based_class_metadata = balanced_metadata['tags_based_class']
    
#     # split indices by question
#     n = len(report_ids)
#     qid2indices = dict()
#     for i in range(n):
#         qid = question_ids[i]
#         try:
#             tmp = qid2indices[qid]
#         except KeyError:
#             tmp = qid2indices[qid] = []
#         tmp.append(i)
    
#     # choose report ids for validation
#     val_rids = set()
#     forbidden_rids = set()
#     for qid, indices in qid2indices.items():
#         # split indices into healthy and unhealthy
#         h_indices = []
#         unh_indices = []
#         for i in indices:
#             healthy = health_metadata[report_ids[i]][str(qid)]
#             if healthy: h_indices.append(i)
#             else: unh_indices.append(i)
        
#         # get diverse samples from each category         
#         class_getter = lambda i : tags_based_class_metadata[report_ids[i]][str(qid)]
        
#         if len(h_indices) >= min_question_count:
#             h_samples = _get_diverse_samples(n_healthy_per_question, h_indices, class_getter)
#             val_rids.update(report_ids[i] for i in h_samples)
#         else:
#             forbidden_rids.update(report_ids[i] for i in h_indices)

#         if len(unh_indices) >= min_question_count:
#             unh_samples = _get_diverse_samples(n_unhealthy_per_question, unh_indices, class_getter)
#             val_rids.update(report_ids[i] for i in unh_samples)
#         else:
#             forbidden_rids.update(report_ids[i] for i in unh_indices)
    
#     # bin report ids by chexpert label
#     if chexpert_labels is not None:
#         print('_split_data_train_val__balanced(): sampling from chexpert labels')
#         binned_report_ids = [[] for _ in range(len(CHEXPERT_LABELS))]
#         for rid in report_ids:
#             if rid in forbidden_rids: continue
#             labels = chexpert_labels[rid]
#             for i, label in enumerate(labels):
#                 if label == 1:
#                     binned_report_ids[i].append(rid)
#         # add examples from each label to validation set
#         for bin in binned_report_ids:
#             bin = list(set(bin))
#             if len(bin) > 0:
#                 print('len(bin)=', len(bin), ', label=', chexpert_labels[bin[0]])
#             else:
#                 print('len(bin)=', len(bin))
#             if len(bin) > n_positive_per_chexpert_label:
#                 val_rids.update(random.sample(bin, n_positive_per_chexpert_label))
    
#     # split indices into train and validation
#     train_indices = { qid:[] for qid in qid2indices.keys() }
#     val_indices = []
#     for qid, indices in qid2indices.items():    
#         for i in indices:
#             if report_ids[i] in val_rids and report_ids[i] not in forbidden_rids:
#                 val_indices.append(i)
#             else:
#                 train_indices[qid].append(i)
    
#     # convert to numpy arrays
#     for qid in train_indices.keys():
#         train_indices[qid] = np.array(train_indices[qid], dtype=int)
#     val_indices = np.array(val_indices, dtype=int)

#     print(f'balanced splitting: len(train_indices) = {sum(len(x) for x in train_indices.values())},'
#           f' len(val_indices) = {len(val_indices)}')

#     return train_indices, val_indices

def load_precomputed_visual_features(precomputed_visual_features_path, images):
    print(f'Loading precomputed visual features from {precomputed_visual_features_path} ...')
    features_data = load_pickle(precomputed_visual_features_path)
    print('  File loaded!')
    image_paths = features_data['image_paths']
    features = features_data['features']
    print (f'  features.shape = {features.shape}, len(image_paths) = {len(image_paths)}')
    mean = np.nanmean(features, 0)
    std = np.nanstd(features, 0) + 1e-9
    features -= mean
    features /= std
    features[np.isnan(features)] = 0
    print('  Feature normalization done')
    image2idx = { img:idx for idx, img in enumerate(image_paths) }
    idx2visfeatidx = np.empty((len(images),), dtype=int)
    for idx, img in enumerate(images):
        idx2visfeatidx[idx] = image2idx[img]    
    print('  Done!')
    return features, idx2visfeatidx

class VQADataset(Dataset):
    
    def __init__(self, report_ids, indices,
                images=None, transform=None,
                questions=None, answers=None,
                question=None, answer=None,
                fixed_qa_pair=False,
                include_answer=True,
                include_image=True,
                use_random_image=False,
                shuffle_indices=True,
                # aux task: medical tags
                classify_tags=False, rid2tags=None,
                # aux task: image orientation
                classify_orientation=False, orientations=None,
                # aux task: chexpert labels
                classify_chexpert=False, chexpert_labels=None,
                # aux task: question labels
                classify_questions=False, question_labels=None,
                # precomputed visual features
                use_precomputed_visual_features=False,
                precomputed_visual_features=None,
                idx2visfeatidx=None,
                # infinite mode
                infinite=False,
                # other tasks
                other_tasks=None,
        ):
        self.report_ids = report_ids
        self.images = images
        self.questions = questions
        self.answers = answers
        self.indices = indices
        self.transform = transform
        self.infinite = infinite
        self.include_answer = include_answer
        self.include_image = include_image
        self.fixed_qa_pair = fixed_qa_pair
        self.use_precomputed_visual_features = use_precomputed_visual_features
        self.use_random_image = use_random_image
        self.other_tasks = other_tasks

        if include_image:
            assert images is not None

        if fixed_qa_pair:
            assert question is not None
            assert answer is not None
            self.question = question
            self.answer = answer

        if use_precomputed_visual_features:
            assert precomputed_visual_features is not None
            assert idx2visfeatidx is not None
            self.precomputed_visual_features = precomputed_visual_features
            self.idx2visfeatidx = idx2visfeatidx

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
        rid = self.report_ids[idx]        
        output = dict(
            idx=idx,            
            q=self.question if self.fixed_qa_pair else self.questions[idx],
        )        
        if self.use_precomputed_visual_features:
            output['vf'] = self.precomputed_visual_features[self.idx2visfeatidx[idx]]
        if self.include_image:
            if self.use_random_image:
                img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3)*255))
            else:
                img = Image.open(self.images[idx]).convert('RGB')
            output['i'] = self.transform(img)
        if self.include_answer:
            output['a'] = self.answer if self.fixed_qa_pair else self.answers[idx]
        if self.classify_tags:
            output['tags'] = self.rid2tags[rid]
        if self.classify_orientation:
            output['orientation'] = self.orientations[idx]
        if self.classify_chexpert:
            output['chexpert'] = self.chexpert_labels[rid]
        if self.classify_questions:
            output['qlabels'] = self.question_labels[rid]
        # other tasks
        if self.other_tasks is not None:
            for task in self.other_tasks:
                output[task[0]] = task[1](i, rid)
        return output

class LabelBasedVQAClass:

    def __init__(self, label_names, templates, use_merged_findings=False,
                labels2mergedfindings=None, n_findings=None, labels=None):        
        self.label_names = label_names
        self.templates = templates
        self.use_merged_findings = use_merged_findings        
        if use_merged_findings:
            assert labels2mergedfindings is not None
            assert n_findings is not None
            assert labels is not None
            self.labels2mergedfindings = labels2mergedfindings
            self.finding_labels = adapt_label_matrix_as_merged_findings(
                labels, n_findings, self.labels2mergedfindings)

    def _create_label_based_dataset_and_dataloader(self, indices, labels, label_names, templates,
            batch_size, num_workers, collate_batch_fn, tokenizer=None, report_ids=None, infinite=True,
            n_pos_samples=None, n_neg_samples=None, min_pos_to_include=0, log_weighting=False, create_dataset_kwargs={}, include_qa=True,
            print_every=1, break_loop_at_i=None):
        
        disease_datasets = []
        if log_weighting:
            pos_counts = []

        for i in range(len(label_names)):
            
            if i == break_loop_at_i:
                break

            should_print = i % print_every == 0
            pos_indices = [None] * len(indices)
            neg_indices = [None] * len(indices)
            i_pos = 0
            i_neg = 0
            for j in indices:
                jj = j if report_ids is None else report_ids[j]
                if labels[jj][i] == 1:
                    pos_indices[i_pos] = j
                    i_pos += 1
                else:
                    neg_indices[i_neg] = j
                    i_neg += 1
            pos_indices = pos_indices[:i_pos]
            neg_indices = neg_indices[:i_neg]

            if n_pos_samples is not None:
                if len(pos_indices) > n_pos_samples:
                    pos_indices = random.sample(pos_indices, n_pos_samples)
            if n_neg_samples is not None:
                if len(neg_indices) > n_neg_samples:
                    neg_indices = random.sample(neg_indices, n_neg_samples)                

            if len(pos_indices) < min_pos_to_include:
                print(f'    ignoring label = {label_names[i]}, reason: too few positive examples ({len(pos_indices)})')
                continue

            if include_qa:
                if self.use_merged_findings:
                    q_id = self.labels2mergedfindings[i]
                else:
                    q_id = i
                if should_print:
                    print(f'    label = {i}, onehot={q_id}, len(pos_indices)={len(pos_indices)}, len(neg_indices)={len(neg_indices)}')
            else:
                if should_print:
                    print(f'    label = {i}, len(pos_indices)={len(pos_indices)}, len(neg_indices)={len(neg_indices)}')

            if log_weighting:
                pos_counts.append(len(pos_indices))
            
            # positive
            if len(pos_indices) > 0:
                pos_indices = np.array(pos_indices, dtype=int)
                if include_qa:
                    pos_answer = tokenizer.string2ids(templates[label_names[i]][1].lower())
                    pos_dataset = self._create_vqa_dataset(q=q_id, a=pos_answer, indices=pos_indices,
                                                        infinite=infinite, **create_dataset_kwargs)
                else:
                    pos_dataset = self._create_visual_dataset(indices=pos_indices, infinite=infinite, **create_dataset_kwargs)
            else:
                pos_dataset = None

            # negative
            if len(neg_indices) > 0:
                neg_indices = np.array(neg_indices, dtype=int)
                if include_qa:
                    neg_answer = tokenizer.string2ids(templates[label_names[i]][0].lower())
                    neg_dataset = self._create_vqa_dataset(q=q_id, a=neg_answer, indices=neg_indices,
                                                        infinite=infinite, **create_dataset_kwargs)
                else:
                    neg_dataset = self._create_visual_dataset(indices=neg_indices, infinite=infinite, **create_dataset_kwargs)
            else:
                neg_dataset = None

            # merging
            assert pos_dataset or neg_dataset
            if pos_dataset and neg_dataset: # merge
                if infinite:
                    comp_dataset = CompositeInfiniteDataset([pos_dataset, neg_dataset], [1, 1])
                else:
                    comp_dataset = CompositeDataset([pos_dataset, neg_dataset])
                disease_datasets.append(comp_dataset)
            else:
                disease_datasets.append(pos_dataset if pos_dataset else neg_dataset)
            assert disease_datasets[-1] is not None            
        
        # final dataset
        if infinite:
            if log_weighting:
                weights = get_imbalance_reduced_weights(pos_counts, 0.4)
            else: # uniform weights
                weights = [1] * len(disease_datasets)
            dataset = CompositeInfiniteDataset(disease_datasets, weights)
        else:
            dataset = CompositeDataset(disease_datasets)        

        # dataloader
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=collate_batch_fn,
                                pin_memory=True)
        
        return dataset, dataloader

    def _create_vqa_dataset(self, q, a, indices, **kwargs):
        raise NotImplementedError('Make sure your specialized class implements this function')
    
    def _create_visual_dataset(self, indices, **kwargs):
        raise NotImplementedError('Make sure your specialized class implements this function')


class VQA_Base(LabelBasedVQAClass):

    def __init__(self, training, train_image_transform, val_image_transform,
                batch_size, collate_batch_fn,
                preprocessing_save_path,                
                num_workers,
                collate_batch_fn_chexpert_mode = None,
                classify_tags = False,
                rid2tags_path = None,
                classify_orientation = False,
                classify_chexpert = False,
                classify_questions = False,
                chexpert_labels_path = None,
                question_labels_path = None,
                balanced_dataloading = False,
                balanced_metadata_path = None,
                train_with_all = False,
                use_report_eval_mode = False,
                verbose_question = True,
                include_image = True,
                use_random_image = False,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = None,
                use_merged_findings=False,
                findings_remapper=None,
                n_findings=None,
                other_tasks = None,
                debug = False,
        ):
        
        self.preprocessing_save_path = preprocessing_save_path
        self.training = training
        if self.training:
            self.image_transform = train_image_transform
        else:
            self.image_transform = val_image_transform
        self.val_image_transform = val_image_transform
        self.classify_tags = classify_tags
        self.classify_orientation = classify_orientation
        self.classify_chexpert = classify_chexpert
        self.classify_questions = classify_questions
        self.balanced_dataloading = balanced_dataloading
        self.use_report_eval_mode = use_report_eval_mode
        self.verbose_question = verbose_question
        self.include_image = include_image
        self.use_random_image = use_random_image
        self.use_precomputed_visual_features = use_precomputed_visual_features
        self.precomputed_visual_features_path = precomputed_visual_features_path
        self.train_with_all = train_with_all
        self.use_merged_findings = use_merged_findings
        self.other_tasks = other_tasks

        self.chexpert_labels = None
        
        if classify_tags:
            assert rid2tags_path is not None
            self.rid2tags = load_pickle(rid2tags_path)

        if classify_chexpert:
            assert chexpert_labels_path is not None
            self.chexpert_labels = get_cached_pickle_file(chexpert_labels_path)
        
        if classify_questions:
            assert question_labels_path is not None
            self.question_labels = load_pickle(question_labels_path)
        
        if balanced_dataloading:
            assert balanced_metadata_path is not None
            self.balanced_metadata = load_pickle(balanced_metadata_path)
            self.chexpert_labels = get_cached_pickle_file(chexpert_labels_path) \
                if chexpert_labels_path is not None else None

        if self.chexpert_labels is not None:
            self.chexpert_labels = np.array(self.chexpert_labels)

        if use_report_eval_mode:
            if (not self._load_cached_data(preprocessing_save_path)):
                self._preprocess_data()
                self._save_data(preprocessing_save_path)
                # self.train_report_ids = list(set(report_ids[i] for i in data['train_indices']))
                # self.val_report_ids = list(set(report_ids[i] for i in data['val_indices']))
            # self.val_indices = np.array(list(range(len(self.report_ids))), dtype=int)
        else:
            first_time = not self._load_cached_data(preprocessing_save_path)
            if debug or first_time:
                self._preprocess_data()
                if not debug: self._save_data(preprocessing_save_path)
            if training:
                if train_with_all: self._assign_all_data_to_train()
                else: self.train_indices = _group_indices_by_question_id(self.train_indices, self.question_ids)

        self._sanity_check_train_val_split()

        self._preprocess_noncachable_data()

        super().__init__(
            label_names=CHEXPERT_LABELS,
            templates=TEMPLATES_CHEXPERT_v1,
            use_merged_findings=use_merged_findings,
            labels2mergedfindings=findings_remapper[CHEXPERT_DATASET_ID] if use_merged_findings else None,
            n_findings=n_findings,
            labels = self.chexpert_labels,
        )
        
        print(f'batch_size = {batch_size}')
        print(f'len(self.report_ids) = {len(self.report_ids)}, len(set(self.report_ids)) = {len(set(self.report_ids))}')
        self._generate_datasets_and_dataloaders(batch_size, collate_batch_fn, num_workers, collate_batch_fn_chexpert_mode)
        print('done!')

    def __len__(self):
        return len(self.report_ids)
    
    def _preprocess_data(self):
        raise NotImplementedError('Make sure your specialized class implements this function')

    def _sanity_check_train_val_split(self):
        if hasattr(self, 'train_indices') and hasattr(self, 'val_indices'):
            assert type(self.train_indices) == dict
            val_indices_set = set(self.val_indices)
            for indices in self.train_indices.values():
                assert len(set(indices).intersection(val_indices_set)) == 0
            assert sum(len(indices) for indices in self.train_indices.values()) +\
                     len(self.val_indices) == len(self.report_ids)
    
    def _preprocess_noncachable_data(self):
        if self.use_precomputed_visual_features:
            features, idx2visfeatidx = load_precomputed_visual_features(
                self.precomputed_visual_features_path,
                self.images,
            )
            self.precomputed_visual_features = features
            self.idx2visfeatidx = idx2visfeatidx
    
    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers, collate_batch_fn_chexpert_mode=None):
        raise NotImplementedError('Make sure your specialized class implements this function')

    def _create_vqa_dataset(self, indices, shuffle_indices=True, include_answer=True, fixed_qa_pair=False,
                            q=None, a=None, infinite=True):
        
        if self.classify_chexpert:
            labels = self.finding_labels if self.use_merged_findings else self.chexpert_labels
        else:
            labels = None

        kwargs = dict(
            report_ids=self.report_ids,
            images=self.images,
            indices=indices,
            transform=self.image_transform,
            include_answer=include_answer,
            include_image=self.include_image,
            shuffle_indices=shuffle_indices,
            use_random_image=self.use_random_image,
            # aux task: medical tags
            classify_tags = self.classify_tags,
            rid2tags = self.rid2tags if self.classify_tags else None,
            # aux task: orientation
            classify_orientation = self.classify_orientation,
            orientations = self.orientations if self.classify_orientation else None,
            # aux task: chexpert labels
            classify_chexpert = self.classify_chexpert,
            chexpert_labels = labels,
            # aux task: question labels
            classify_questions = self.classify_questions,
            question_labels = self.question_labels if self.classify_questions else None,
            # infinite mode
            infinite=infinite,
            # other tasks
            other_tasks = self.other_tasks,
        )

        if include_answer:            
            kwargs['answers']=self.answers
        
        if fixed_qa_pair:
            kwargs['fixed_qa_pair'] = True
            kwargs['question'] = q
            kwargs['answer'] = a
        else:
            kwargs['questions'] = self.questions if self.verbose_question else self.question_ids        
        
        if self.use_precomputed_visual_features:
            kwargs['use_precomputed_visual_features'] = True
            kwargs['precomputed_visual_features'] = self.precomputed_visual_features
            kwargs['idx2visfeatidx'] = self.idx2visfeatidx
        
        return VQADataset(**kwargs)

    # def _load_split_from_path(self, preprocessing_save_path):
    #     data = load_pickle(preprocessing_save_path)
    #     if self.use_report_eval_mode:
    #         report_ids = data['report_ids']
    #         self.train_report_ids = list(set(report_ids[i] for i in data['train_indices']))
    #         self.val_report_ids = list(set(report_ids[i] for i in data['val_indices']))
    #     else:
    #         self.train_indices = data['train_indices']
    #         self.val_indices = data['val_indices']
    #     print(f'Train-val split indices successfully loaded from {preprocessing_save_path}')
    
    def _load_cached_data(self, preprocessing_save_path):
        print(f'Checking if data is already cached in path {preprocessing_save_path} ...')
        data = load_pickle(preprocessing_save_path)
        if data is None:
            print('\tNo, it isn\'t :(')
            return False
        self.report_ids = data['report_ids']
        self.images = data['images']
        self.questions = data['questions']
        if 'question_ids' in data: # backward compatible hack
            self.question_ids = data['question_ids']
        if 'answers' in data:
            self.answers = data['answers']
        if self.training:
            if 'train_indices' in data:
                self.train_indices = data['train_indices']
            if not self.train_with_all:
                self.val_indices = data['val_indices']
        if self.classify_orientation:
            self.orientations = data['orientations']
        print('\tYes, it is, data successfully loaded :)')
        return True
    
    def _save_data(self, preprocessing_save_path):
        print('saving data to', preprocessing_save_path)
        data = dict(
            report_ids = self.report_ids,
            images = self.images,
            questions = self.questions,
            question_ids = self.question_ids,
        )
        if hasattr(self, 'answers'):
            data['answers'] = self.answers
        if hasattr(self, 'train_indices'):
            data['train_indices'] = self.train_indices
        if hasattr(self, 'val_indices'):
            data['val_indices'] = self.val_indices
        if hasattr(self, 'orientations'):
            data['orientations'] = self.orientations
        save_to_pickle(data, preprocessing_save_path)
        print('\tdone!')
        return True

    def _assign_all_data_to_train(self):

        print('Assigning all data to training ...')        
        self.train_indices = _assign_all_data_to_train(self.question_ids)
    
    # def _split_data_train_val(self,
    #                           n_val_examples_per_question=10,
    #                           min_train_examples_per_question=100):
        
    #     print('Splitting data into training and validation ...')
        
    #     train_indices, val_indices = _split_data_train_val(self.question_ids,
    #                                                       self.answers,
    #                                                       n_val_examples_per_question,
    #                                                       min_train_examples_per_question)                
    #     self.train_indices = train_indices
    #     self.val_indices = val_indices
    
    # def _split_data_train_val__balanced(self,
    #                                     n_healthy_per_question=2,
    #                                     n_unhealthy_per_question=3,
    #                                     min_question_count=100,
    #                                     n_positive_per_chexpert_label=7):

    #     print('Splitting data into training and validation in balanced mode ...')

    #     train_indices, val_indices = _split_data_train_val__balanced(self.report_ids,
    #                                                                  self.question_ids,
    #                                                                  self.balanced_metadata,
    #                                                                  n_healthy_per_question,
    #                                                                  n_unhealthy_per_question,
    #                                                                  min_question_count,
    #                                                                  self.chexpert_labels,
    #                                                                  n_positive_per_chexpert_label)
    #     self.train_indices = train_indices
    #     self.val_indices = val_indices

class BalancedTreeNode:
    
    def __init__(self, indices=None, children=None, weights=None):
        self.indices = indices
        self.weights = weights
        self.children = children
    
    def get_dataset(self, vqa_trainer):
        if self.children is None: # this is a leaf node
            assert len(self.indices) > 0
            return vqa_trainer._create_vqa_dataset(self.indices)
        else:
            assert len(self.children) == len(self.weights)
            datasets = [child.get_dataset(vqa_trainer) for child in self.children]
            return CompositeInfiniteDataset(datasets, self.weights)

def _get_cached_balanced_train_datasets_path(preprocessing_save_path, batch_size, imbalance_reduction_coef, allowed_question_ids=None):
    strings = []
    strings.append(f'bs={batch_size}')
    strings.append(f'imb_redu_coef={imbalance_reduction_coef}')
    if allowed_question_ids is not None:
        aqi_string = '(' + ','.join(map(str, allowed_question_ids)) + ')'
        strings.append(f'aqi={aqi_string}')
    filepath = f'{preprocessing_save_path}.balanced_train_data({",".join(strings)}).pkl'
    if len(filepath) > MAX_FILENAME_LENGTH:
        h = hash_string(filepath)
        filepath = os.path.join(os.path.dirname(preprocessing_save_path), f'balanced_train_data(hash={h[0]},{h[1]}).pkl')
    return filepath

class VQA_Trainer(VQA_Base):
    
    def __init__(self, train_image_transform, val_image_transform,
                batch_size, collate_batch_fn,
                preprocessing_save_path,
                cache_dir,
                num_workers,
                collate_batch_fn_chexpert_mode = None,
                verbose_question = True,
                classify_tags = False,
                rid2tags_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                balanced_dataloading = False,
                balanced_metadata_filename = None,
                imbalance_reduction_coef = 1,
                validation_only = False,
                train_with_all = False,
                include_answer = True,
                allowed_questions = None,
                qa_adapted_reports_filename = None,
                one_question_per_batch = False,
                include_mined_questions_mode = True,
                include_chexpert_mode = False,                
                include_image = True,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = None,
                use_merged_findings=False,
                findings_remapper=None,
                n_findings=None,
                other_tasks=None,
                debug = False,
        ):

        rid2tags_path = os.path.join(cache_dir, rid2tags_filename) if classify_tags else None
        chexpert_labels_path = os.path.join(cache_dir, chexpert_labels_filename) \
            if chexpert_labels_filename is not None else None
        balanced_metadata_path = os.path.join(cache_dir, balanced_metadata_filename) if balanced_dataloading else None        
        question_labels_path = os.path.join(cache_dir, question_labels_filename) if classify_questions else None

        if balanced_dataloading:
            assert balanced_metadata_path is not None
        
        self.balanced_dataloading = balanced_dataloading
        print(f'self.balanced_dataloading = {self.balanced_dataloading}')

        self.validation_only = validation_only
        self.train_with_all = train_with_all
        self.cache_dir = cache_dir
        self.imbalance_reduction_coef = imbalance_reduction_coef
        self.one_question_per_batch = one_question_per_batch
        self.include_answer = include_answer
        self.include_chexpert_mode = include_chexpert_mode
        self.include_mined_questions_mode = include_mined_questions_mode
        
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
    
        super().__init__(True, train_image_transform, val_image_transform,
                batch_size, collate_batch_fn,
                preprocessing_save_path,
                num_workers,
                collate_batch_fn_chexpert_mode = collate_batch_fn_chexpert_mode,
                classify_tags = classify_tags,
                rid2tags_path = rid2tags_path,
                classify_orientation = classify_orientation,
                classify_chexpert = classify_chexpert,
                classify_questions = classify_questions,
                chexpert_labels_path = chexpert_labels_path,
                question_labels_path = question_labels_path,
                balanced_dataloading = balanced_dataloading,
                balanced_metadata_path = balanced_metadata_path,
                train_with_all = train_with_all,
                verbose_question = verbose_question,
                include_image = include_image,
                use_precomputed_visual_features = use_precomputed_visual_features,
                precomputed_visual_features_path = precomputed_visual_features_path,
                use_merged_findings = use_merged_findings,
                findings_remapper = findings_remapper,
                n_findings = n_findings,
                other_tasks = other_tasks,
                debug = debug)

    def _load_optional_datasets_and_dataloaders(self):
        raise NotImplementedError('This method should be implemented in the child class')
    
    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers, collate_batch_fn_chexpert_mode=None):
        print('_generate_datasets_and_dataloaders()')
        
        if self.include_mined_questions_mode:
            if not self.validation_only:
                print(f'self.balanced_dataloading = {self.balanced_dataloading}')
                if self.balanced_dataloading:
                    self._generate_train_dataset__balanced(batch_size)
                else:
                    self._generate_train_dataset(batch_size)
            if not self.train_with_all:
                self._generate_val_dataset()
            self._generate_train_val_dataloaders(batch_size, collate_batch_fn, num_workers)

        if self.include_chexpert_mode:
            if not self.validation_only:
                self._generate_train_dataset_and_dataloader__chexpert_mode(batch_size, collate_batch_fn_chexpert_mode, num_workers)
            if not self.train_with_all:
                self._generate_val_dataset_and_dataloader__chexpert_mode(batch_size, collate_batch_fn_chexpert_mode, num_workers)

        self._load_optional_datasets_and_dataloaders()


    def _get_composite_dataset(self, datasets, weights, batch_size):
        if self.one_question_per_batch:
            print('(***) Note: using BatchedCompositeInfiniteDataset (one question per batch)')
            return BatchedCompositeInfiniteDataset(datasets, weights, batch_size)
        return CompositeInfiniteDataset(datasets, weights)
    
    def _generate_train_dataset(self, batch_size):

        print('_generate_train_dataset()')

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
            question_datasets.append(self._create_vqa_dataset(indices, infinite=True))
            i = j+1
        
        question_weights = get_imbalance_reduced_weights([len(d) for d in question_datasets], self.imbalance_reduction_coef)
        self.train_dataset = self._get_composite_dataset(question_datasets, question_weights, batch_size)
        print(f'\tlen(question_datasets) = {len(question_datasets)}')
    
    def _generate_train_dataset__balanced(self, batch_size):

        print('_generate_train_dataset__balanced()')

        cached_balanced_train_datasets_path = _get_cached_balanced_train_datasets_path(
            self.preprocessing_save_path, batch_size, self.imbalance_reduction_coef, self.allowed_question_ids)
        cached_data = load_pickle(cached_balanced_train_datasets_path)
        if cached_data is not None:
            print(f'Balanced train data loaded from {cached_balanced_train_datasets_path}')
            question_datasets = [node.get_dataset(self) for node in cached_data['question_nodes']]
            question_weights = cached_data['question_weights']
            self.train_dataset = self._get_composite_dataset(question_datasets, question_weights, batch_size)
            print(f'\tlen(question_datasets) = {len(question_datasets)}')
            return

        print('Computing balanced datasets from scratch ...')
        print(f'self.imbalance_reduction_coef = {self.imbalance_reduction_coef}')

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
    
    def _generate_dataset_and_dataloader__chexpert_mode(self, indices, batch_size, collate_batch_fn,
                                                        num_workers, infinite=True, n_samples=None):

        return self._create_label_based_dataset_and_dataloader(
            indices=indices,
            labels=self.chexpert_labels,
            label_names=CHEXPERT_LABELS,
            templates=TEMPLATES_CHEXPERT_v1,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_batch_fn=collate_batch_fn,
            infinite=infinite,
            n_pos_samples=n_samples,
            n_neg_samples=n_samples,
            report_ids=self.report_ids,
            create_dataset_kwargs=dict(
                fixed_qa_pair=True,
            ))

    def _generate_train_dataset_and_dataloader__chexpert_mode(self, batch_size, collate_batch_fn, num_workers):                
        print('Generating perfectly balanced train dataset in chexpert mode ...')        
        flattened_indices = []
        for _indices in self.train_indices.values():
            flattened_indices.extend(_indices)
        dedup_indices = deduplicate_indices(flattened_indices, self.report_ids)
        dataset, dataloader = self._generate_dataset_and_dataloader__chexpert_mode(
            dedup_indices, batch_size, collate_batch_fn, num_workers)        
        self.train_dataset__chexpert_mode = dataset
        self.train_dataloader__chexpert_mode = dataloader
        print('len(self.train_dataset__chexpert_mode) =', len(self.train_dataset__chexpert_mode))

    def _generate_val_dataset_and_dataloader__chexpert_mode(self, batch_size, collate_batch_fn, num_workers):                
        print('Generating balanced validation dataset in chexpert mode ...')
        dedup_indices = deduplicate_indices(self.val_indices, self.report_ids)
        dataset, dataloader = self._generate_dataset_and_dataloader__chexpert_mode(
            dedup_indices, batch_size, collate_batch_fn, num_workers, infinite=False, n_samples=40)
        self.val_dataset__chexpert_mode = dataset
        self.val_dataloader__chexpert_mode = dataloader
        print('len(self.val_dataset__chexpert_mode) =', len(self.val_dataset__chexpert_mode))

    def _generate_val_dataset(self, max_num_examples_per_question=60):
        if self.allowed_question_ids is not None:
            val_indices = [i for i in self.val_indices if self.question_ids[i] in self.allowed_question_ids]
        else:
            val_indices = self.val_indices
        grouped_val_indices = _group_indices_by_question_id(val_indices, self.question_ids, convert_to_numpy=False)
        val_indices = []
        for indices in grouped_val_indices.values():
            if len(indices) > max_num_examples_per_question:
                indices = random.sample(indices, max_num_examples_per_question)
            val_indices.extend(indices)
        print('len(self.val_indices) =', len(self.val_indices))
        print('len(val_indices) =', len(val_indices))
        self.val_dataset = self._create_vqa_dataset(val_indices, shuffle_indices=False,
                                                    include_answer=self.include_answer, infinite=False)
            
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
        if not self.train_with_all:
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
                classify_tags = False,
                rid2tags_filename = None,
                classify_orientation = False,
                classify_chexpert = False,
                chexpert_labels_filename = None,
                classify_questions = False,
                question_labels_filename = None,
                include_image = True,
                use_random_image = False,
                use_precomputed_visual_features = False,
                precomputed_visual_features_path = None,
                other_tasks = None,
                debug = False,
        ):

        rid2tags_path = os.path.join(cache_dir, rid2tags_filename) if classify_tags else None
        chexpert_labels_path = os.path.join(cache_dir, chexpert_labels_filename) if classify_chexpert else None        
        question_labels_path = os.path.join(cache_dir, question_labels_filename) if classify_questions else None
        
        self.include_answer = include_answer
    
        super().__init__(False, None, transform, batch_size, collate_batch_fn,
                preprocessing_save_path,
                num_workers,
                classify_tags = classify_tags,
                rid2tags_path = rid2tags_path,
                classify_orientation = classify_orientation,
                classify_chexpert = classify_chexpert,
                classify_questions = classify_questions,
                chexpert_labels_path = chexpert_labels_path,
                question_labels_path = question_labels_path,
                verbose_question = verbose_question,
                include_image = include_image,
                use_random_image = use_random_image,
                use_precomputed_visual_features = use_precomputed_visual_features,
                precomputed_visual_features_path = precomputed_visual_features_path,
                other_tasks = other_tasks,
                debug = debug)

    def _generate_datasets_and_dataloaders(self, batch_size, collate_batch_fn, num_workers, *unused_args):
        print('VQA_Evaluator():')
        self.test_indices = list(range(len(self.report_ids)))
        print(f'  len(self.test_indices) = {len(self.test_indices)}, '
              f'len(set(self.report_ids)) = {len(set(self.report_ids))}')
        self._generate_test_dataset()
        self._generate_test_dataloader(batch_size, collate_batch_fn, num_workers)

    def _generate_test_dataset(self):

        print('generating test dataset ...')

        self.test_dataset = self._create_vqa_dataset(self.test_indices,
            shuffle_indices=False,
            include_answer=self.include_answer,
            infinite=False)
            
    def _generate_test_dataloader(self, batch_size, collate_batch_fn, num_workers):

        print('generating test dataloader ...')
        
        self.test_dataloader = DataLoader(self.test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         collate_fn=collate_batch_fn,
                                         pin_memory=True)
