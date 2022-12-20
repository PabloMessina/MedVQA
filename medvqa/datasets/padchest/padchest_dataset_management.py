from medvqa.utils.files import read_lines_from_txt
from medvqa.datasets.padchest import (
    PADCHEST_LABELS_CSV_PATH,
    PADCHEST_IMAGES_SMALL_DIR,
    PADCHEST_BROKEN_IMAGES_TXT_PATH,
)
from medvqa.utils.constants import (
    PADCHEST_PROJECTIONS,
    PADCHEST_NUM_LABELS,
    PADCHEST_NUM_LOCALIZATIONS,
)
from medvqa.utils.constants import PADCHEST_GENDERS
from medvqa.datasets.dataloading_utils import (
    INFINITE_DATASET_LENGTH,
    CompositeInfiniteDataset,
    CompositeDataset,
    get_imbalance_reduced_weights,
)

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import pandas as pd
import os
import numpy as np
import random
import re
import os
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

def _labels_localizations_by_sentence_to_answer_string(x):
    x = x.lower()
    x = re.sub(r'\bnan\b', '[]', x)
    x = eval(x)
    assert type(x) is list
    if type(x[0]) is list:
        sentences = [' '.join(y) for y in x]
    else:
        sentences = [' '.join(x)]
    dedup_sentences = []
    for sentence in sentences:
        if sentence not in dedup_sentences:
            dedup_sentences.append(sentence)
    concat_sentences = ', '.join(dedup_sentences)
    return concat_sentences

class PadChestVocabGenerator():
    def __init__(self):
        self.name = PADCHEST_LABELS_CSV_PATH.replace(os.sep, '_')

    def __call__(self):
        df = pd.read_csv(PADCHEST_LABELS_CSV_PATH)        
        for labels in df['Labels'].dropna():
            labels = eval(labels)
            for label in labels:
                label = label.strip().lower()
                if label:
                    for token in wordpunct_tokenize(label):
                        yield token

        for localizations in df['Localizations'].dropna():
            localizations = eval(localizations)
            for localization in localizations:
                localization = localization.strip().lower()
                if localization:
                    for token in wordpunct_tokenize(localization):
                        yield token

        for labels_localizations_by_sentence in df['LabelsLocalizationsBySentence'].dropna():
            x = _labels_localizations_by_sentence_to_answer_string(labels_localizations_by_sentence)
            for token in wordpunct_tokenize(x):
                yield token

class PadChestTrainingDataMode:
    TRAIN_ONLY = 'train'
    ALL = 'all'

def _65535_to_255(x):
    return x * 255 / 65535

class PadChestVQADataset(Dataset):
    def __init__(self, image_paths, transform, labels, localizations,
                 projections, genders, indices, question,
                 answer=None, answers=None, include_image=True,
                 shuffle_indices=True, infinite=False):
        assert (answer is None) != (answers is None), 'Must provide either answer or answers'
        self.image_paths = image_paths
        self.transform = transform
        self.labels = labels
        self.localizations = localizations
        self.projections = projections
        self.genders = genders
        self.indices = indices
        self.question = question
        self.answer = answer
        self.answers = answers
        self.include_image = include_image
        self.infinite = infinite

        if include_image:
            assert image_paths is not None, 'Must provide image paths if include_image is True'

        if shuffle_indices: np.random.shuffle(self.indices)
        self._len = INFINITE_DATASET_LENGTH if infinite else len(self.indices)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.infinite: idx %= len(self.indices)
        idx = self.indices[idx]
        output = dict(
            idx=idx,
            l=self.labels[idx],
            loc=self.localizations[idx],
            proj=self.projections[idx],
            g=self.genders[idx],
            q=self.question,
            a=self.answer if self.answer is not None else self.answers[idx],
        )
        if self.include_image:
            # Convert from 65535 L to 255 RGB
            output['i'] = self.transform(Image.open(self.image_paths[idx]).point(_65535_to_255).convert('RGB'))

        return output

class PadChest_VQA_Trainer:

    def __init__(self, transform, batch_size, collate_batch_fn, num_workers, tokenizer,
                 train_study_ids_path, val_study_ids_path, test_study_ids_path,
                 training_data_mode=PadChestTrainingDataMode.ALL,
                 use_validation_set=True, include_image=True):

        assert train_study_ids_path is not None, 'Must provide train_study_ids_path'
        assert val_study_ids_path is not None, 'Must provide val_study_ids_path'
        assert test_study_ids_path is not None, 'Must provide test_study_ids_path'

        self.transform = transform
        self.batch_size = batch_size
        self.collate_batch_fn = collate_batch_fn
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.use_validation_set = use_validation_set
        self.include_image = include_image

        # Load study ids
        train_study_ids = read_lines_from_txt(train_study_ids_path)
        val_study_ids = read_lines_from_txt(val_study_ids_path)
        test_study_ids = read_lines_from_txt(test_study_ids_path)
        all_study_ids = train_study_ids + val_study_ids + test_study_ids
        all_study_ids_set = set(all_study_ids)
        train_study_ids_set = set(train_study_ids)
        val_study_ids_set = set(val_study_ids)
        test_study_ids_set = set(test_study_ids)

        # Load the labels CSV file
        labels_df = pd.read_csv(PADCHEST_LABELS_CSV_PATH)
        print(f'Number of rows before filtering: {len(labels_df)}')

        # Keep only the following columns: ImageID, StudyID, PatientSex_DICOM, Projection,
        # Labels, Localizations, LabelsLocalizationsBySentence
        labels_df = labels_df[['ImageID', 'StudyID', 'PatientSex_DICOM', 'Projection',
                                 'Labels', 'Localizations', 'LabelsLocalizationsBySentence']]

        # Drop nan rows
        labels_df = labels_df.dropna()
        print(f'Number of rows after filtering: {len(labels_df)} (dropped nan rows)')

        # Filter the labels to only include the study ids we want
        labels_df = labels_df[labels_df['StudyID'].isin(all_study_ids_set)]
        print(f'Number of rows after filtering: {len(labels_df)} (dropped study ids not in all_study_ids_set)')

        # Filter the labels to only include the sexes we want
        labels_df = labels_df[labels_df['PatientSex_DICOM'].isin(PADCHEST_GENDERS)]
        print(f'Number of rows after filtering: {len(labels_df)} (dropped rows with unexpected PatientSex_DICOM)')

        # Filter the labels to only include the projections we want
        labels_df = labels_df[labels_df['Projection'].isin(PADCHEST_PROJECTIONS)]
        print(f'Number of rows after filtering: {len(labels_df)} (dropped rows with unexpected Projection)')

        # Load broken images
        broken_images = read_lines_from_txt(PADCHEST_BROKEN_IMAGES_TXT_PATH)
        broken_image_names = [os.path.basename(x) for x in broken_images]
        broken_image_names_set = set(broken_image_names)
        # Filter the labels to only include non-broken images
        labels_df = labels_df[~labels_df['ImageID'].isin(broken_image_names_set)]
        print(f'Number of rows after filtering: {len(labels_df)} (dropped rows with broken images)')

        # For each study, keep only one row according to their projection, as follows:
        # PA > AP > AP_horizontal > L > COSTAL
        labels_df['Projection_Rank'] = labels_df['Projection'].apply(lambda x: PADCHEST_PROJECTIONS.index(x))
        labels_df = labels_df.sort_values(by=['StudyID', 'Projection_Rank'], ascending=True)
        labels_df = labels_df.drop_duplicates(subset=['StudyID'], keep='first')
        labels_df = labels_df.drop(columns=['Projection_Rank'])        
        print(f'Number of rows after filtering: {len(labels_df)} (dropped rows with duplicate StudyID)')

        # Image paths
        self.image_paths = [os.path.join(PADCHEST_IMAGES_SMALL_DIR, image_id) for image_id in labels_df['ImageID']]

        # Gender labels
        self.genders = labels_df['PatientSex_DICOM'].apply(lambda x: PADCHEST_GENDERS.index(x)).values

        # Projection labels
        self.projections = labels_df['Projection'].apply(lambda x: PADCHEST_PROJECTIONS.index(x)).values

        def parse_and_clean_labels(labels):
            labels = eval(labels)
            labels = [label.strip().lower() for label in labels]
            labels = [label for label in labels if label != '']
            return labels

        # Labels
        labels_set = set()
        for labels in labels_df['Labels']:
            labels = parse_and_clean_labels(labels)
            labels_set.update(labels)
        labels_list = list(labels_set)
        labels_list.sort()
        label2idx = {label: idx for idx, label in enumerate(labels_list)}
        print(f'Number of labels: {len(labels_list)}')        
        labels_matrix = np.zeros((len(labels_df), len(labels_list)), dtype=np.int8)
        for i, labels in enumerate(labels_df['Labels']):
            labels = parse_and_clean_labels(labels)
            for label in labels:
                labels_matrix[i, label2idx[label]] = 1
        self.labels_list = labels_list
        self.labels = labels_matrix
        assert len(self.labels_list) == PADCHEST_NUM_LABELS

        # Localizations
        localizations_set = set()
        for localizations in labels_df['Localizations']:
            localizations = eval(localizations)
            for localization in localizations:
                localization = localization.strip().lower()
                localizations_set.add(localization)
        localizations_list = list(localizations_set)
        localizations_list.sort()
        localization2idx = {localization: idx for idx, localization in enumerate(localizations_list)}
        print(f'Number of localizations: {len(localizations_list)}')
        localizations_matrix = np.zeros((len(labels_df), len(localizations_list)), dtype=np.int8)
        for i, localizations in enumerate(labels_df['Localizations']):
            localizations = eval(localizations)
            for localization in localizations:
                localization = localization.strip().lower()
                localizations_matrix[i, localization2idx[localization]] = 1
        self.localizations_list = localizations_list
        self.localizations = localizations_matrix
        assert len(self.localizations_list) == PADCHEST_NUM_LOCALIZATIONS

        # Answers based on Labels
        print('Generating answers based on labels...')
        labels_answers = []
        for i, labels in tqdm(enumerate(labels_df['Labels']), total=len(labels_df)):
            labels = parse_and_clean_labels(labels)
            answer = ', '.join(labels)
            labels_answers.append(tokenizer.string2ids(answer))
        self.labels_answers = labels_answers
        print(f'Done. Example answer: {tokenizer.ids2string(random.choice(labels_answers))}')
        # Answers based on Localizations
        print('Generating answers based on localizations...')
        localizations_answers = []
        for i, localizations in tqdm(enumerate(labels_df['Localizations']), total=len(labels_df)):
            localizations = eval(localizations)
            localizations = [localization.strip().lower() for localization in localizations]
            answer = ', '.join(localizations)
            localizations_answers.append(tokenizer.string2ids(answer))
        self.localization_answers = localizations_answers    
        print(f'Done. Example answer: {tokenizer.ids2string(random.choice(localizations_answers))}')
        print(f'Number of non-empty answers: {sum(len(answer) > 8 for answer in localizations_answers)}')
        
        # Answers based on LabelsLocalizationsBySentence
        print('Generating answers based on labels localizations by sentence...')
        labels_localizations_answers = []        
        for labels_localizations_by_sentence in tqdm(labels_df['LabelsLocalizationsBySentence'], total=len(labels_df)):
            x = _labels_localizations_by_sentence_to_answer_string(labels_localizations_by_sentence)
            labels_localizations_answers.append(tokenizer.string2ids(x))
        self.labels_localizations_answers = labels_localizations_answers
        print(f'Done. Example answer: {tokenizer.ids2string(random.choice(labels_localizations_answers))}')

        # Train, val, test indices
        train_indices = []
        val_indices = []
        test_indices = []
        for i, study_id in enumerate(labels_df['StudyID']):
            if study_id in train_study_ids_set:
                train_indices.append(i)
            elif study_id in val_study_ids_set:
                val_indices.append(i)
            elif study_id in test_study_ids_set:
                test_indices.append(i)
            else:
                raise ValueError(f'Study ID {study_id} is not in train, val, or test study IDs.')
        
        if training_data_mode == PadChestTrainingDataMode.ALL:
            self.train_indices = train_indices + val_indices + test_indices
        elif training_data_mode == PadChestTrainingDataMode.TRAIN_ONLY:
            self.train_indices = train_indices
        else:
            raise ValueError(f'Invalid training data mode: {training_data_mode}')        
        self.val_indices = val_indices
        self.test_indices = test_indices

        # Create datasets and dataloaders
        self.train_dataset, self.train_dataloader = self._create_dataset_and_dataloader(
            indices=self.train_indices,
            batch_size=batch_size,
            collate_batch_fn=collate_batch_fn,
            num_workers=num_workers,
            shuffle=True,
            infinite=True,
            log_weighting=True,
            min_pos_to_include=50,
        )
        print(f'len(self.train_dataset) = {len(self.train_dataset)}')
        if self.use_validation_set:
            self.val_dataset, self.val_dataloader = self._create_dataset_and_dataloader(
                indices=self.val_indices,
                batch_size=batch_size,
                collate_batch_fn=collate_batch_fn,
                num_workers=num_workers,
                shuffle=False,
                infinite=False,
                log_weighting=False,
                n_samples=1000,
                n_samples_per_label=4,
            )
            print(f'len(self.val_dataset) = {len(self.val_dataset)}')

    def _create_vqa_dataset(self, indices, question, answer=None, answers=None, shuffle=True, infinite=True):
        return PadChestVQADataset(
            image_paths=self.image_paths,
            transform=self.transform,
            labels=self.labels,
            localizations=self.localizations,
            projections=self.projections,
            genders=self.genders,
            indices=indices,
            question=question,
            answer=answer,
            answers=answers,
            include_image=self.include_image,
            shuffle_indices=shuffle,
            infinite=infinite,
        )

    def _create_dataset_and_dataloader(self, indices, batch_size, collate_batch_fn, num_workers,
            shuffle, infinite, log_weighting, min_pos_to_include=0, n_samples=None, n_samples_per_label=None):

        # Question ID meaning:
        # 0, 1, ..., len(self.labels_list) - 1: specific labels
        # len(self.labels_list): all labels
        # len(self.labels_list) + 1: all localizations
        # len(self.labels_list) + 2: all labels and localizations

        # Labels answers vqa dataset
        q_id = len(self.labels_list)
        indices_ = random.sample(indices, n_samples) if n_samples is not None and\
                                                        n_samples < len(indices) else indices
        labels_answers_vqa_dataset = self._create_vqa_dataset(indices_, q_id,
            answers=self.labels_answers, shuffle=shuffle, infinite=infinite)
        
        # Localizations answers vqa dataset
        q_id = len(self.labels_list) + 1
        indices_ = random.sample(indices, n_samples) if n_samples is not None and\
                                                        n_samples < len(indices) else indices
        localizations_answers_vqa_dataset = self._create_vqa_dataset(indices_, q_id,
            answers=self.localization_answers, shuffle=shuffle, infinite=infinite)
        
        # Labels and localizations answers vqa dataset
        q_id = len(self.labels_list) + 2
        indices_ = random.sample(indices, n_samples) if n_samples is not None and\
                                                        n_samples < len(indices) else indices
        labels_localizations_answers_vqa_dataset = self._create_vqa_dataset(indices_, q_id,
            answers=self.labels_localizations_answers, shuffle=shuffle, infinite=infinite)
        
        # Label-specific vqa datasets
        label_specific_vqa_datasets = []
        pos_counts = []
        for i, label in enumerate(self.labels_list):
            q_id = i
            
            # split indices into positive and negative
            pos_indices = []
            neg_indices = []
            for j in indices:
                if self.labels[j][i] == 1:
                    pos_indices.append(j)
                else:
                    neg_indices.append(j)

            # if there are not enough positive indices, skip this label
            if len(pos_indices) < min_pos_to_include:
                continue

            # if n_samples_per_label is not None, randomly sample n_samples_per_label from pos_indices and neg_indices
            if n_samples_per_label is not None:
                if len(pos_indices) > n_samples_per_label:
                    pos_indices = random.sample(pos_indices, n_samples_per_label)
                if len(neg_indices) > n_samples_per_label:
                    neg_indices = random.sample(neg_indices, n_samples_per_label)
            pos_counts.append(len(pos_indices))
            
            # positive answer vqa dataset
            if len(pos_indices) > 0:
                pos_indices = np.array(pos_indices, dtype=np.int32)            
                pos_answer = self.tokenizer.string2ids(label)
                pos_vqa_dataset = self._create_vqa_dataset(pos_indices, q_id,
                    answer=pos_answer, shuffle=shuffle, infinite=infinite)
            else:
                pos_vqa_dataset = None

            # negative answer vqa dataset
            if len(neg_indices) > 0:
                neg_indices = np.array(neg_indices, dtype=np.int32)
                neg_answer = self.tokenizer.string2ids('no')
                neg_vqa_dataset = self._create_vqa_dataset(neg_indices, q_id,
                    answer=neg_answer, shuffle=shuffle, infinite=infinite)
            else:
                neg_vqa_dataset = None
            
            # combine positive and negative answer vqa datasets
            if pos_vqa_dataset is not None and neg_vqa_dataset is not None:
                if infinite:
                    comp_vqa_dataset = CompositeInfiniteDataset([pos_vqa_dataset, neg_vqa_dataset], [1, 1])
                else:
                    comp_vqa_dataset = CompositeDataset([pos_vqa_dataset, neg_vqa_dataset])
            elif pos_vqa_dataset is not None:
                comp_vqa_dataset = pos_vqa_dataset
            elif neg_vqa_dataset is not None:
                comp_vqa_dataset = neg_vqa_dataset
            else:
                raise ValueError('Both pos_vqa_dataset and neg_vqa_dataset are None')
            label_specific_vqa_datasets.append(comp_vqa_dataset)
        
        # Combine all label-specific vqa datasets
        if infinite:
            if log_weighting:
                weights = get_imbalance_reduced_weights(pos_counts, 0.4)
            else: # uniform weights
                weights = [1] * len(label_specific_vqa_datasets)
            label_specific_vqa_dataset = CompositeInfiniteDataset(label_specific_vqa_datasets, weights)
        else:
            label_specific_vqa_dataset = CompositeDataset(label_specific_vqa_datasets)
        
        # Combine all vqa datasets
        if infinite:
            final_vqa_dataset = CompositeInfiniteDataset([
                labels_answers_vqa_dataset,
                localizations_answers_vqa_dataset,
                labels_localizations_answers_vqa_dataset,
                label_specific_vqa_dataset,
            ], [1, 1, 1, 1])
        else:
            final_vqa_dataset = CompositeDataset([
                labels_answers_vqa_dataset,
                localizations_answers_vqa_dataset,
                labels_localizations_answers_vqa_dataset,
                label_specific_vqa_dataset,
            ])

        # Data loader
        final_vqa_dataloader = DataLoader(final_vqa_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch_fn,            
            pin_memory=True,
        )

        return final_vqa_dataset, final_vqa_dataloader

    def print_dataset_instance(self, dataset, i):
        row = dataset[i]
        idx = row['idx']
        # Print idx
        print(f'Idx: {idx}')
        # Display image
        print(f'Image path: {self.image_paths[idx]}')
        img = Image.open(self.image_paths[idx]).point(_65535_to_255).convert('RGB')
        plt.imshow(img)
        plt.show()
        # Display projection
        proj = row['proj']
        print(f'Projection: {PADCHEST_PROJECTIONS[proj]}')
        # Display gender
        gender = row['g']
        print(f'Gender: {PADCHEST_GENDERS[gender]}')
        # Display question
        question = row['q']
        if question < len(self.labels_list):
            question = f'(#{question}) {self.labels_list[question]}'
        else:
            if question == len(self.labels_list):
                question = f'**(#{question}) Labels question'
            elif question == len(self.labels_list) + 1:
                question = f'**(#{question}) Localizations question'
                # question = 'Localizations question'
            elif question == len(self.labels_list) + 2:
                question = f'**(#{question}) Labels and localizations question'
            else:
                raise ValueError(f'Invalid question index: {question}')
        print(f'Question: {question}')
        # Display answer
        answer = row['a']
        answer = self.tokenizer.ids2string(answer)
        print(f'Answer: {answer}')
        # Display labels
        labels = row['l']
        labels_str = ', '.join(self.labels_list[i] for i in range(len(labels)) if labels[i] == 1)
        print(f'Labels: {labels_str}')
        # Display localizations
        localizations = row['loc']
        localizations_str = ', '.join(self.localizations_list[i] for i in range(len(localizations)) if localizations[i] == 1)
        print(f'Localizations: {localizations_str}')