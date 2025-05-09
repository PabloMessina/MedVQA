import os
import re
import random
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Literal
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from medvqa.datasets.image_transforms_factory import create_image_transforms
from medvqa.datasets.visual_module import BasicImageDataset, MAETrainerBase
from medvqa.utils.bbox_utils import (
    calculate_probabilistic_mask_from_bboxes,
    convert_bboxes_into_target_tensors,
    xyxy_to_cxcywh,
)
from medvqa.utils.files_utils import load_json, load_pickle, read_lines_from_txt
from medvqa.datasets.padchest import (
    PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH,
    PADCHEST_GR_JPG_DIR,
    PADCHEST_GR_MASTER_TABLE_CSV_PATH,
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

logger = logging.getLogger(__name__)


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

_65535_to_255_COEF = 255. / 65535.
def _65535_to_255(x):
    return x * _65535_to_255_COEF

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

class PadChestImageDataset(BasicImageDataset):
    def __init__(self, image_paths, transform, indices, shuffle_indices=True, infinite=False):
        super().__init__(image_paths, transform, indices, shuffle_indices, infinite)
    def _get_image(self, idx): # Override BasicImageDataset._get_image
        return self.transform(Image.open(self.image_paths[idx]).point(_65535_to_255).convert('RGB'))

def _parse_and_clean_labels(labels):
    labels = eval(labels)
    labels = [label.strip().lower() for label in labels]
    labels = [label for label in labels if label != '']
    return labels

def _PadChest_common_init(self, transform, batch_size, collate_batch_fn, num_workers,
    train_study_ids_path, val_study_ids_path, test_study_ids_path,
    training_data_mode=PadChestTrainingDataMode.TRAIN_ONLY,
    use_validation_set=True, include_image=True,
    keep_one_projection_per_study=True):

    assert train_study_ids_path is not None, 'Must provide train_study_ids_path'
    assert val_study_ids_path is not None, 'Must provide val_study_ids_path'
    assert test_study_ids_path is not None, 'Must provide test_study_ids_path'
    if training_data_mode == PadChestTrainingDataMode.ALL:
        # Prevent data leakage
        assert not use_validation_set, 'Cannot use validation set if training_data_mode is ALL'

    self.transform = transform
    self.batch_size = batch_size
    self.collate_batch_fn = collate_batch_fn
    self.num_workers = num_workers
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

    # Filter the labels to only include the genders we want
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
    if keep_one_projection_per_study:
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

    # Labels
    labels_set = set()
    for labels in labels_df['Labels']:
        labels = _parse_and_clean_labels(labels)
        labels_set.update(labels)
    labels_list = list(labels_set)
    labels_list.sort()
    label2idx = {label: idx for idx, label in enumerate(labels_list)}
    print(f'Number of labels: {len(labels_list)}')        
    labels_matrix = np.zeros((len(labels_df), len(labels_list)), dtype=np.int8)
    for i, labels in enumerate(labels_df['Labels']):
        labels = _parse_and_clean_labels(labels)
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

    self.labels_df = labels_df


class PadChest_VQA_Trainer:

    def __init__(self, transform, batch_size, collate_batch_fn, num_workers, tokenizer,
                 train_study_ids_path, val_study_ids_path, test_study_ids_path,
                 training_data_mode=PadChestTrainingDataMode.TRAIN_ONLY,
                 use_validation_set=True, include_image=True):

        _PadChest_common_init(self, transform, batch_size, collate_batch_fn, num_workers,
                            train_study_ids_path, val_study_ids_path, test_study_ids_path,
                            training_data_mode, use_validation_set, include_image)

        self.tokenizer = tokenizer        
        labels_df = self.labels_df
        
        # Answers based on Labels
        print('Generating answers based on labels...')
        labels_answers = []
        for i, labels in tqdm(enumerate(labels_df['Labels']), total=len(labels_df)):
            labels = _parse_and_clean_labels(labels)
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
        print(f'Number of non-empty answers: {sum(len(answer) > 2 for answer in localizations_answers)}')
        
        # Answers based on LabelsLocalizationsBySentence
        print('Generating answers based on labels localizations by sentence...')
        labels_localizations_answers = []        
        for labels_localizations_by_sentence in tqdm(labels_df['LabelsLocalizationsBySentence'], total=len(labels_df)):
            x = _labels_localizations_by_sentence_to_answer_string(labels_localizations_by_sentence)
            labels_localizations_answers.append(tokenizer.string2ids(x))
        self.labels_localizations_answers = labels_localizations_answers
        print(f'Done. Example answer: {tokenizer.ids2string(random.choice(labels_localizations_answers))}')

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

        print('Creating dataset and dataloader...')

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
        pos_indices_ = [i for i in indices_ if len(self.localization_answers[i]) > 2]
        neg_indices_ = [i for i in indices_ if len(self.localization_answers[i]) <= 2]
        datasets_ = []
        if len(pos_indices_) > 0:
            pos_loc_answers_vqa_dataset = self._create_vqa_dataset(pos_indices_, q_id,
                answers=self.localization_answers, shuffle=shuffle, infinite=infinite)
            ans = self.tokenizer.ids2string(self.localization_answers[random.choice(pos_indices_)])
            print(f'Example of positive localization answer: {ans}')
            datasets_.append(pos_loc_answers_vqa_dataset)
        if len(neg_indices_) > 0:
            neg_loc_answers_vqa_dataset = self._create_vqa_dataset(neg_indices_, q_id,
                answers=self.localization_answers, shuffle=shuffle, infinite=infinite)
            ans = self.tokenizer.ids2string(self.localization_answers[random.choice(neg_indices_)])
            print(f'Example of negative localization answer: {ans}')
            datasets_.append(neg_loc_answers_vqa_dataset)
        assert len(datasets_) > 0
        if len(datasets_) == 1:
            localizations_answers_vqa_dataset = datasets_[0]
        else:
            if infinite:
                localizations_answers_vqa_dataset = CompositeInfiniteDataset(datasets_, [1, 1])
            else:
                localizations_answers_vqa_dataset = CompositeDataset(datasets_)
        
        # Labels and localizations answers vqa dataset
        q_id = len(self.labels_list) + 2
        indices_ = random.sample(indices, n_samples) if n_samples is not None and\
                                                        n_samples < len(indices) else indices
        labels_localizations_answers_vqa_dataset = self._create_vqa_dataset(indices_, q_id,
            answers=self.labels_localizations_answers, shuffle=shuffle, infinite=infinite)
        
        # Label-specific vqa datasets
        label_specific_vqa_datasets = []
        pos_counts = []
        for i, label in tqdm(enumerate(self.labels_list)):
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

        print('Done!')
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

class PadChest_MAE_Trainer(MAETrainerBase):

    def __init__(self, transform, batch_size, collate_batch_fn, num_workers,
                 train_study_ids_path, val_study_ids_path, test_study_ids_path,
                 training_data_mode=PadChestTrainingDataMode.TRAIN_ONLY,
                 use_validation_set=True):

        _PadChest_common_init(self, transform, batch_size, collate_batch_fn, num_workers,
                            train_study_ids_path, val_study_ids_path, test_study_ids_path,
                            training_data_mode, use_validation_set, keep_one_projection_per_study=False)

        labels_getter = lambda i: self.labels[i]
        super().__init__(self.train_indices, self.val_indices, self.test_indices,
                         list(range(len(self.labels_list))),
                         labels_getter, batch_size, collate_batch_fn, num_workers,
                         use_validation_set=use_validation_set)

    def _create_mae_dataset(self, indices, shuffle, infinite):
        return PadChestImageDataset(
            image_paths=self.image_paths,
            transform=self.transform,
            indices=indices,
            shuffle_indices=shuffle,
            infinite=infinite,
        )
    

class PadChestGRPhraseGroundingDataset(Dataset):
    """
    PyTorch Dataset for Phrase Grounding using the PadChest-GR dataset.

    This dataset adapts PadChest-GR for phrase grounding tasks. Each sample
    consists of an image paired with a single phrase (sentence) that has
    associated bounding box annotations within that image.

    Args:
        image_transforms_kwargs: Dictionary of keyword arguments used to build the
            image transformation function via `create_image_transforms`.
            The created function is expected to handle images, bounding boxes,
            and potentially masks (if `data_augmentation_enabled` is True).
        phrase2embedding: A dictionary mapping cleaned sentence text (str) to
            its corresponding embedding (e.g., numpy array).
        feature_map_size: A tuple (height, width) representing the target
            feature map size for generating ground truth tensors during training.
            Required if `for_training` is True.
        json_path: Path to the PadChest-GR JSON file containing report findings
            and bounding boxes (xyxy format assumed).
        csv_path: Path to the PadChest-GR master CSV file containing study
            metadata and split information.
        img_dir: Path to the directory containing the primary JPG/PNG images.
        split: The dataset split to load ('train', 'validation', 'test', 'all').
        language: The language of the reports to use ('en' or 'es').
        bbox_format: The bounding box format expected by `image_transforms`
            and used for target tensor generation ('cxcywh' or 'xyxy').
            Coordinates are always normalized [0, 1].
        image_format: The file extension of the images ('jpg' or 'png').
        data_augmentation_enabled: If True, data augmentation (via the function
            created from `image_transforms_kwargs`) will be applied to images,
            bounding boxes, and probabilistic masks (if `for_training`).
        for_training: If True, returns target tensors suitable for training.
            If False, returns the image, phrase embedding, and original
            bounding boxes for inference.
    """
    def __init__(
        self,
        image_transforms_kwargs: Dict[str, Any],
        phrase2embedding: Dict[str, np.ndarray],
        feature_map_size: Optional[Tuple[int, int]] = None,
        json_path: str = PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH,
        csv_path: str = PADCHEST_GR_MASTER_TABLE_CSV_PATH,
        img_dir: str = PADCHEST_GR_JPG_DIR,
        split: Literal["train", "validation", "test", "all"] = "train",
        language: Literal["en", "es"] = "en",
        bbox_format: str = "cxcywh",
        image_format: Literal["jpg", "png"] = "jpg",
        data_augmentation_enabled: bool = False,
        for_training: bool = True,
    ):
        super().__init__()

        # --- Input Validation ---
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.isdir(img_dir):
            raise NotADirectoryError(f"Image directory not found: {img_dir}")
        assert split in ["train", "validation", "test", "all"], \
            f"Invalid split '{split}'. Must be one of ['train', 'validation', 'test', 'all']."
        assert language in ["en", "es"], \
            f"Invalid language '{language}'. Must be one of ['en', 'es']."
        assert bbox_format in ["cxcywh", "xyxy"], \
            f"Invalid bbox_format '{bbox_format}'. Must be one of ['cxcywh', 'xyxy']."
        if for_training:
            assert feature_map_size is not None, \
                "feature_map_size must be provided when for_training=True."
            assert len(feature_map_size) == 2, \
                "feature_map_size should be a tuple (height, width)."
        if not phrase2embedding:
            raise ValueError("phrase2embedding dictionary cannot be empty.")

        self.img_dir = img_dir
        self.image_format = image_format
        self.language = language
        self.bbox_format = bbox_format
        self.feature_map_size = feature_map_size
        self.data_augmentation_enabled = data_augmentation_enabled
        self.for_training = for_training

        # Create image transform function
        self.image_transforms = create_image_transforms(**image_transforms_kwargs)

        # --- Load and Filter Metadata ---
        logger.info(f"Loading master CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Filtering dataset for split: {split}")
        if split != "all":
            df = df[df['split'] == split]
        if df.empty:
            raise ValueError(f"No data found for split '{split}' in {csv_path}")
        study_ids_in_split = df['StudyID'].unique().tolist()
        logger.info(
            f"Found {len(study_ids_in_split)} unique studies for split '{split}'"
        )

        # --- Load Reports Data ---
        logger.info(f"Loading reports JSON from: {json_path}")
        reports_json_list = load_json(json_path)
        reports_data = {item['StudyID']: item for item in reports_json_list}
        logger.info(f"Loaded {len(reports_data)} reports from JSON.")

        # --- Pre-process Data into Image-Phrase Pairs ---
        logger.info("Processing data into image-phrase pairs...")
        self.image_paths: List[str] = []
        self.phrase_embeddings: List[torch.Tensor] = [] # Store as tensors directly
        self.phrase_bboxes: List[List[List[float]]] = [] # List of lists of boxes per phrase
        self.study_ids: List[str] = [] # Keep track of original study ID
        self.image_ids: List[str] = [] # Keep track of original image ID
        self.phrase_texts: List[str] = [] # Keep track of the phrase text

        lang_key = f'sentence_{self.language}'
        num_missing_embeddings = 0
        num_grounded_phrases = 0

        for study_id in tqdm(study_ids_in_split, desc="Processing Studies"):

            report_info = reports_data[study_id]
            image_id = report_info['ImageID']
            base_name, _ = os.path.splitext(image_id)
            image_id_with_ext = f"{base_name}.{self.image_format}"
            image_path = os.path.join(self.img_dir, image_id_with_ext)

            assert os.path.exists(image_path), f"Image not found: {image_path}"

            findings = report_info.get('findings', [])
            for finding in findings:
                finding_phrases = []
                # Clean sentence
                sentence_text = finding.get(lang_key, "")
                sentence_text = sentence_text.strip()
                if sentence_text.endswith('.'):
                    sentence_text = sentence_text[:-1]
                finding_phrases.append(sentence_text)
                # Collect labels
                finding_phrases.extend(finding.get('labels', []))

                # Check for bounding boxes
                original_boxes_xyxy = finding.get('boxes')
                if original_boxes_xyxy: # Must have boxes
                    # Convert boxes to the required format (once)
                    if self.bbox_format == "cxcywh":
                        processed_boxes = [xyxy_to_cxcywh(box) for box in original_boxes_xyxy]
                    else: # xyxy
                        processed_boxes = original_boxes_xyxy
                    
                    for phrase in finding_phrases:
                        # Check if embedding exists for this phrase
                        if phrase not in phrase2embedding:
                            logger.warning(
                                f"Phrase '{phrase}' not found in phrase2embedding dictionary. "
                                f"Skipping this phrase."
                            )
                            num_missing_embeddings += 1
                            continue # Skip if no embedding is available
                        
                        # Add data point (image-phrase pair)
                        self.image_paths.append(image_path)
                        embedding = phrase2embedding[phrase]
                        self.phrase_embeddings.append(torch.tensor(embedding, dtype=torch.float32))
                        self.phrase_bboxes.append(processed_boxes)
                        self.study_ids.append(study_id)
                        self.image_ids.append(image_id)
                        self.phrase_texts.append(phrase)
                        num_grounded_phrases += 1

        if num_missing_embeddings > 0:
            logger.warning(
                f"Skipped {num_missing_embeddings} phrases due to missing entries "
                f"in phrase2embedding dictionary."
            )
        if not self.image_paths:
            raise ValueError(
                f"No grounded phrases found for split '{split}' with language "
                f"'{language}' and available embeddings. Check data and "
                f"phrase2embedding keys."
            )

        logger.info(f"Created {len(self.image_paths)} image-phrase grounding pairs.")

        # Store embedding size
        self.embedding_size = self.phrase_embeddings[0].shape[0]


    def __len__(self) -> int:
        """Returns the total number of image-phrase pairs."""
        return len(self.image_paths)

    def __getitem__(self, i: int) -> dict:
        """
        Retrieves a single image-phrase grounding sample.

        Args:
            i: The index of the sample to retrieve.

        Returns:
            A dictionary containing the data for the requested sample.
            Structure depends on the `for_training` flag.
        """
        image_path = self.image_paths[i]
        phrase_embedding = self.phrase_embeddings[i] # Already a tensor
        # Get the list of boxes for this specific phrase
        bboxes_for_phrase = self.phrase_bboxes[i]

        # --- Apply Image Transformations / Augmentation ---
        image: Optional[torch.Tensor] = None
        augmented_bboxes = bboxes_for_phrase # Default to original if no aug
        augmented_prob_mask: Optional[np.ndarray] = None

        if self.for_training:
            # 1. Calculate probabilistic mask from original boxes BEFORE augmentation
            # Ensure mask_resolution matches feature_map_size for consistency
            prob_mask = calculate_probabilistic_mask_from_bboxes(
                bboxes=bboxes_for_phrase,
                mask_resolution=self.feature_map_size, # Use feature map size for mask
                bbox_format=self.bbox_format,
            )

            if self.data_augmentation_enabled:
                # 2. Apply augmentation to image, boxes, and mask
                # Assume self.image_transforms handles this structure
                transform_input = {
                    'image_path': image_path,
                    'bboxes': bboxes_for_phrase,
                    'bbox_labels': [0] * len(bboxes_for_phrase), # Dummy labels
                    'masks': [prob_mask], # Pass mask in a list
                }
                transform_output = self.image_transforms(**transform_input)
                image = transform_output['pixel_values']
                augmented_bboxes = transform_output['bboxes']
                augmented_prob_mask = transform_output['masks'][0]
            else:
                # 2. Apply transform to image only (no augmentation)
                # Assume transform takes path and returns dict or tensor
                transform_output = self.image_transforms(image_path)['pixel_values']
                augmented_prob_mask = prob_mask # Use un-augmented mask

            # 3. Convert augmented boxes and mask to target tensors
            target_coords, target_presence, target_prob_mask = \
                convert_bboxes_into_target_tensors(
                    bboxes=augmented_bboxes,
                    probabilistic_mask=augmented_prob_mask,
                    feature_map_size=self.feature_map_size,
                    bbox_format=self.bbox_format,
                )

            output = {
                'i': image,                 # Image tensor
                'pe': phrase_embedding,     # Phrase embedding tensor
                'tbc': target_coords,       # Target bbox coordinates tensor
                'tbp': target_presence,     # Target bbox presence tensor
                'tpm': target_prob_mask,    # Target probabilistic mask tensor
            }

        else: # Inference mode
            # Apply transform to image only (no augmentation expected)
            transform_output = self.image_transforms(image_path)['pixel_values']

            # Return image, embedding, and original (processed) bboxes
            output = {
                'i': image,
                'pe': phrase_embedding,
                'bboxes': bboxes_for_phrase, # Return original boxes for eval
            }

        return output

    def collate_fn(self, batch: List[dict]) -> dict:
        """
        Custom collate function for DataLoader.

        Args:
            batch: A list of dictionaries, each from __getitem__.

        Returns:
            A dictionary containing the batched data.
        """
        if self.for_training:
            # Collate the batch for training
            images = torch.stack([item['i'] for item in batch])
            phrase_embeddings = torch.stack([item['pe'] for item in batch])
            target_bboxes = torch.stack([item['tbc'] for item in batch])
            target_presence = torch.stack([item['tbp'] for item in batch])
            target_prob_masks = torch.stack([item['tpm'] for item in batch])
            collated_batch = {
                'i': images,
                'pe': phrase_embeddings,
                'tbc': target_bboxes,
                'tbp': target_presence,
                'tpm': target_prob_masks,
            }
        else:
            # Collate the batch for inference
            images = torch.stack([item['i'] for item in batch])
            phrase_embeddings = torch.stack([item['pe'] for item in batch])
            # BBoxes are lists of lists, keep as a list of lists
            bboxes = [item['bboxes'] for item in batch]
            collated_batch = {
                'i': images,
                'pe': phrase_embeddings,
                'bboxes': bboxes,
            }

        # Add dataset identifier
        collated_batch['dataset_name'] = 'padchest_gr'
        return collated_batch
    

class PadChestGRPhraseTrainer:
    """
    A wrapper class to facilitate the creation of DataLoaders for phrase
    grounding tasks using the PadChestGRPhraseGroundingDataset.

    This class handles:
    - Loading phrase embeddings.
    - Instantiating the PadChestGRPhraseGroundingDataset for train,
      validation, and/or test splits based on configuration.
    - Creating PyTorch DataLoaders for the instantiated datasets.
    """

    def __init__(
        self,

        # --- Task & Target Configuration ---
        mask_height: int,
        mask_width: int,

        # --- Data Source Configuration ---
        phrase_embeddings_filepath: str,
        json_path: str = PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH,
        csv_path: str = PADCHEST_GR_MASTER_TABLE_CSV_PATH,
        img_dir: str = PADCHEST_GR_JPG_DIR,
        language: Literal["en", "es"] = "en",
        image_format: Literal["jpg", "png"] = "jpg",
        bbox_format: Literal["xyxy", "cxcywh"] = "cxcywh",

        # --- Split & DataLoader Configuration ---
        use_training_set: bool = False,
        use_validation_set: bool = False,
        use_test_set: bool = False,
        training_split: Literal["train", "all"] = "train",
        max_images_per_batch: int = 32,
        val_batch_size_factor: float = 1.5,
        test_batch_size_factor: float = 1.5,
        num_train_workers: Optional[int] = 2,
        num_val_workers: Optional[int] = 2,
        num_test_workers: Optional[int] = 2,

        # --- Transforms & Augmentation ---
        # Pass the *full kwargs dict* for create_image_transforms
        train_image_transforms_kwargs: Optional[Dict[str, Any]] = None,
        val_image_transforms_kwargs: Optional[Dict[str, Any]] = None,
        test_image_transforms_kwargs: Optional[Dict[str, Any]] = None,
        data_augmentation_enabled: bool = True, # Only affects training set
    ):
        """
        Initializes the PadChestGRPhraseTrainer.

        Args:
            mask_height: Target height for grounding masks/feature maps.
            mask_width: Target width for grounding masks/feature maps.
            phrase_embeddings_filepath: Path to the .pkl file containing the
                phrase-to-embedding dictionary. Expected key: 'phrase2embedding'.
            json_path: Path to the PadChest-GR JSON file.
            csv_path: Path to the PadChest-GR master CSV file.
            img_dir: Path to the directory containing PadChest-GR images.
            language: Language for report sentences ('en' or 'es').
            image_format: Image file extension ('jpg' or 'png').
            bbox_format: Bounding box format ('xyxy' or 'cxcywh').
            use_training_set: If True, create training dataset and dataloader.
            use_validation_set: If True, create validation dataset and dataloader.
            use_test_set: If True, create test dataset and dataloader.
            training_split: Which split(s) to use for training ('train' or 'all').
            max_images_per_batch: Batch size for the training dataloader.
            val_batch_size_factor: Multiplier for training batch size to get
                validation batch size.
            test_batch_size_factor: Multiplier for training batch size to get
                test batch size.
            num_train_workers: Number of workers for training DataLoader.
            num_val_workers: Number of workers for validation DataLoader.
            num_test_workers: Number of workers for test DataLoader.
            train_image_transforms_kwargs: Keyword arguments dictionary for
                `create_image_transforms` for the training set. Required if `use_training_set`.
            val_image_transforms_kwargs: Keyword arguments dictionary for
                `create_image_transforms` for the validation set. Required if `use_validation_set`.
            test_image_transforms_kwargs: Keyword arguments dictionary for
                `create_image_transforms` for the test set. Required if `use_test_set`.
            data_augmentation_enabled: If True, enable data augmentation in the
                training dataset.
        """
        logger.info("Initializing PadChestGRPhraseTrainer...")

        # --- Argument Validation ---
        if use_training_set:
            assert train_image_transforms_kwargs is not None, \
                "train_image_transforms_kwargs is required when use_training_set=True"
            assert num_train_workers is not None, \
                "num_train_workers must be specified when use_training_set=True"
        if use_validation_set:
            assert val_image_transforms_kwargs is not None, \
                "val_image_transforms_kwargs is required when use_validation_set=True"
            assert num_val_workers is not None, \
                "num_val_workers must be specified when use_validation_set=True"
        if use_test_set:
            assert test_image_transforms_kwargs is not None, \
                "test_image_transforms_kwargs is required when use_test_set=True"
            assert num_test_workers is not None, \
                "num_test_workers must be specified when use_test_set=True"
        assert training_split in ["train", "all"], \
            f"Invalid training_split '{training_split}'. Must be 'train' or 'all'."

        self.feature_map_size = (mask_height, mask_width)

        # --- Load Phrase Embeddings ---
        logger.info(f"Loading phrase embeddings from: {phrase_embeddings_filepath}")
        if not os.path.exists(phrase_embeddings_filepath):
            raise FileNotFoundError(f"Phrase embedding file not found: {phrase_embeddings_filepath}")
        self.phrase2embedding = load_pickle(phrase_embeddings_filepath)
        if not isinstance(self.phrase2embedding, dict):
            raise TypeError(f"Expected 'phrase2embedding' to be a dict, got {type(self.phrase2embedding)}")
        if not self.phrase2embedding:
            raise ValueError("'phrase2embedding' dictionary is empty.")
        logger.info(f"Loaded {len(self.phrase2embedding)} phrase embeddings.")
        # Extract details for potential external use
        self.phrases = sorted(list(self.phrase2embedding.keys()))
        # Note: self.phrase_embeddings array isn't directly stored here,
        # as the dataset handles embedding lookup internally.

        # --- Initialize Datasets and DataLoaders ---
        self.train_dataset: Optional[PadChestGRPhraseGroundingDataset] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataset: Optional[PadChestGRPhraseGroundingDataset] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.test_dataset: Optional[PadChestGRPhraseGroundingDataset] = None
        self.test_dataloader: Optional[DataLoader] = None

        # Common dataset args
        common_dataset_args = {
            "phrase2embedding": self.phrase2embedding,
            "json_path": json_path,
            "csv_path": csv_path,
            "img_dir": img_dir,
            "language": language,
            "bbox_format": bbox_format,
            "image_format": image_format,
        }

        # --- Training Set ---
        if use_training_set:
            logger.info(f"Setting up TRAINING dataset (split: '{training_split}')...")
            self.train_dataset = PadChestGRPhraseGroundingDataset(
                image_transforms_kwargs=train_image_transforms_kwargs,
                feature_map_size=self.feature_map_size,
                split=training_split,
                data_augmentation_enabled=data_augmentation_enabled,
                for_training=True,
                **common_dataset_args
            )
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=max_images_per_batch,
                shuffle=True, # Shuffle training data
                num_workers=num_train_workers,
                collate_fn=self.train_dataset.collate_fn,
                pin_memory=True,
            )
            logger.info(f"  Training DataLoader ready (Batch size: {max_images_per_batch}, Workers: {num_train_workers}, Augmentation: {data_augmentation_enabled})")
            logger.info(f"  Training dataset size: {len(self.train_dataset)}")

        # --- Validation Set ---
        if use_validation_set:
            logger.info("Setting up VALIDATION dataset (split: 'validation')...")
            self.val_dataset = PadChestGRPhraseGroundingDataset(
                image_transforms_kwargs=val_image_transforms_kwargs,
                feature_map_size=self.feature_map_size, # Needed even if for_training=False if mask calc happens
                split="validation",
                data_augmentation_enabled=False, # No augmentation for validation
                for_training=False, # Return format for inference/evaluation
                **common_dataset_args
            )
            val_batch_size = int(max_images_per_batch * val_batch_size_factor)
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                shuffle=False, # No shuffling for validation
                num_workers=num_val_workers,
                collate_fn=self.val_dataset.collate_fn,
                pin_memory=True,
            )
            logger.info(f"  Validation DataLoader ready (Batch size: {val_batch_size}, Workers: {num_val_workers})")
            logger.info(f"  Validation dataset size: {len(self.val_dataset)}")


        # --- Test Set ---
        if use_test_set:
            logger.info("Setting up TEST dataset (split: 'test')...")
            self.test_dataset = PadChestGRPhraseGroundingDataset(
                image_transforms_kwargs=test_image_transforms_kwargs,
                feature_map_size=self.feature_map_size, # Needed even if for_training=False
                split="test",
                data_augmentation_enabled=False, # No augmentation for test
                for_training=False, # Return format for inference/evaluation
                **common_dataset_args
            )
            test_batch_size = int(max_images_per_batch * test_batch_size_factor)
            self.test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=test_batch_size,
                shuffle=False, # No shuffling for test
                num_workers=num_test_workers,
                collate_fn=self.test_dataset.collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            logger.info(f"  Test DataLoader ready (Batch size: {test_batch_size}, Workers: {num_test_workers})")
            logger.info(f"  Test dataset size: {len(self.test_dataset)}")


        logger.info("PadChestGRPhraseTrainer initialization complete.")