import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from medvqa.datasets.chexlocalize import CHEXLOCALIZE_IMAGE_DIR_512X512
from medvqa.datasets.chexpert import (
    CHEXPERT_DATASET_DIR,
    CHEXPERT_TRAIN_VAL_CSV_PATH,
    CHEXPERT_TEST_LABELS_CSV_PATH,
    CHEXPERT_TRAIN_VISUALCHEXBERT_CSV_PATH,
    CHEXPERT_VALID_CSV_PATH,
)
from medvqa.datasets.dataloading_utils import (
    INFINITE_DATASET_LENGTH,
    CompositeInfiniteDataset,
    group_indices_for_balanced_sampling,
)
from medvqa.datasets.image_processing import ImageFactBasedMultilabelClassificationDataset
from medvqa.datasets.visual_module import BasicImageDataset, MAETrainerBase
from medvqa.datasets.vqa import LabelBasedVQAClass, load_precomputed_visual_features
from medvqa.models.report_generation.templates.chexpert import TEMPLATES_CHEXPERT_v1
from medvqa.utils.constants import (
    CHEXPERT_DATASET_ID,
    CHEXPERT_GENDER2ID,
    CHEXPERT_LABEL2PHRASE,
    CHEXPERT_LABELS,
    CHEXPERT_ORIENTATION2ID,
)
from medvqa.utils.files_utils import get_cached_pickle_file, load_pickle
from medvqa.utils.logging_utils import print_bold

class CheXpertTrainingMode:
    TRAIN_VAL = 'train_val'
    VAL_TEST = 'val_test'
    TEST = 'test'
    ALL = 'all'
    @staticmethod
    def get_all():
        return [
            CheXpertTrainingMode.TRAIN_VAL,
            CheXpertTrainingMode.VAL_TEST,
            CheXpertTrainingMode.TEST,
            CheXpertTrainingMode.ALL,
        ]

class CheXpertTrainerBase(LabelBasedVQAClass):

    def __init__(self, use_merged_findings=False, findings_remapper=None, n_findings=None, load_test_set=False,
                 use_visualchexbert=True, apply_filtering=True, load_orientation=True, load_gender=True):
        
        if use_visualchexbert:
            print(f'Loading dataframe from {CHEXPERT_TRAIN_VISUALCHEXBERT_CSV_PATH}')
            df_train_visualchexbert = pd.read_csv(CHEXPERT_TRAIN_VISUALCHEXBERT_CSV_PATH)
            print(f'len(df_train_visualchexbert) = {len(df_train_visualchexbert)}')
            df_train_visualchexbert['Path'] = df_train_visualchexbert['Path'].\
                apply(lambda x: x.replace('CheXpert-v1.0', 'CheXpert-v1.0-small')) # use small images
            print(f'Loading dataframe from {CHEXPERT_VALID_CSV_PATH}')
            df_valid = pd.read_csv(CHEXPERT_VALID_CSV_PATH)
            print(f'len(df_valid) = {len(df_valid)}')
            # verify that the same columns are present in both dataframes
            assert set(df_train_visualchexbert.columns) == set(df_valid.columns)
            df = pd.concat([df_train_visualchexbert, df_valid], ignore_index=True)
            print(f'len(df) = {len(df)}')
        else:        
            print(f'Loading dataframe from {CHEXPERT_TRAIN_VAL_CSV_PATH}')
            df = pd.read_csv(CHEXPERT_TRAIN_VAL_CSV_PATH)
        
        df_orien = df['Frontal/Lateral'] + df['AP/PA'].fillna('')
        df_gender = df['Sex']
        df_paths = df['Path']
        df_labels = df[CHEXPERT_LABELS]

        if apply_filtering:
            valid_rows = (df_orien != 'FrontalLL') & (df_orien != 'FrontalRL') & (df_gender != 'Unknown')
            df_orien = df_orien[valid_rows]
            df_gender = df_gender[valid_rows]
            df_labels = df_labels[valid_rows]
            df_paths = df_paths[valid_rows]
            print(f'num invalid rows removes = {len(df) - len(df_paths)}')
        
        n = len(df_paths)

        # images
        print('Loading images')
        image_paths = (CHEXPERT_DATASET_DIR + os.path.sep + df_paths).to_numpy()
        self.image_paths = image_paths
        
        # orientations
        if load_orientation:
            print('Loading orientations')
            orientations = np.empty((n,), dtype=int)
            for i, x in enumerate(df_orien):
                orientations[i] = CHEXPERT_ORIENTATION2ID[x]
            self.orientations = orientations
        
        # gender
        if load_gender:
            print('Loading genders')
            genders = np.empty((n,), dtype=int)
            for i, x in enumerate(df_gender):
                genders[i] = CHEXPERT_GENDER2ID[x]
            self.genders = genders

        # chexpert labels
        print('Loading chexpert labels')
        labels = df_labels.fillna(0).to_numpy().astype(np.int8)
        labels = np.where(labels == -1, 1, labels)
        self.labels = labels

        # train & valid indices
        self.train_indices = [i for i, x in enumerate(image_paths) if '/train/' in x]
        self.valid_indices = [i for i, x in enumerate(image_paths) if '/valid/' in x]
        assert len(self.train_indices) + len(self.valid_indices) == len(image_paths)

        super().__init__(
            label_names=CHEXPERT_LABELS,
            templates=TEMPLATES_CHEXPERT_v1,
            use_merged_findings=use_merged_findings,
            labels2mergedfindings=findings_remapper[CHEXPERT_DATASET_ID] if use_merged_findings else None,
            n_findings=n_findings,
            labels=self.labels,
        )

        if load_test_set:
            print('Loading test set')
            df_test_labels = pd.read_csv(CHEXPERT_TEST_LABELS_CSV_PATH)
            labels = df_test_labels[CHEXPERT_LABELS].to_numpy().astype(np.int8)
            assert labels.min() == 0 and labels.max() == 1 # binary labels
            self.test_labels = labels
            self.test_image_paths = [os.path.join(CHEXLOCALIZE_IMAGE_DIR_512X512, 'test', path[5:]) for path in df_test_labels['Path']]
            assert all(os.path.exists(path) for path in self.test_image_paths)
            assert len(self.test_image_paths) == len(self.test_labels)
            print(f'len(self.test_image_paths) = {len(self.test_image_paths)}')
            print(f'self.test_labels.shape = {self.test_labels.shape}')

def load_all_chexpert_image_paths_and_labels():
    df_train = pd.read_csv(CHEXPERT_TRAIN_VISUALCHEXBERT_CSV_PATH)
    df_train['Path'] = df_train['Path'].\
        apply(lambda x: x.replace('CheXpert-v1.0', 'CheXpert-v1.0-small')) # use small images
    train_image_paths = (CHEXPERT_DATASET_DIR + os.path.sep + df_train['Path']).to_list()
    train_labels = df_train[CHEXPERT_LABELS].to_numpy().astype(np.int8)
            
    df_valid = pd.read_csv(CHEXPERT_VALID_CSV_PATH)
    valid_image_paths = (CHEXPERT_DATASET_DIR + os.path.sep + df_valid['Path']).to_list()
    valid_labels = df_valid[CHEXPERT_LABELS].to_numpy().astype(np.int8)

    df_test_labels = pd.read_csv(CHEXPERT_TEST_LABELS_CSV_PATH)
    test_image_paths = [os.path.join(CHEXLOCALIZE_IMAGE_DIR_512X512, 'test', path[5:]) for path in df_test_labels['Path']]
    test_labels = df_test_labels[CHEXPERT_LABELS].to_numpy().astype(np.int8)

    all_image_paths = np.concatenate([train_image_paths, valid_image_paths, test_image_paths])
    all_labels = np.concatenate([train_labels, valid_labels, test_labels])
    print(f'len(all_image_paths) = {len(all_image_paths)}')
    print(f'all_labels.shape = {all_labels.shape}')
    assert len(all_image_paths) == len(all_labels)
    assert 0 == all_labels.min() and all_labels.max() == 1
    return all_image_paths, all_labels


class CheXpert_VisualModuleTrainer(CheXpertTrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers):        
        super().__init__()
        self.transform = transform
        self.dataset, self.dataloader = self._create_label_based_dataset_and_dataloader(
            indices= np.arange(len(self.labels)),
            labels=self.labels,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_batch_fn=collate_batch_fn,
            infinite=True,
            include_qa=False,
        )

    def _create_visual_dataset(self, indices, infinite=True):
        return CheXpertImageDataset(
            self.image_paths, self.transform, self.orientations, self.genders, self.labels,
            indices=indices, infinite=infinite
        )

class CheXpert_VQA_Trainer(CheXpertTrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers, tokenizer,
                include_image=True,
                use_precomputed_visual_features=False,
                precomputed_visual_features_path=None,
                use_merged_findings=False, findings_remapper=None, n_findings=None,
        ):
        super().__init__(
            use_merged_findings=use_merged_findings,
            findings_remapper=findings_remapper,
            n_findings=n_findings,
        )
        
        self.transform = transform
        self.include_image = include_image
        self.use_precomputed_visual_features = use_precomputed_visual_features

        if use_precomputed_visual_features:
            assert precomputed_visual_features_path is not None
            features, idx2visfeatidx = load_precomputed_visual_features(
                precomputed_visual_features_path,
                self.image_paths,
            )
            self.precomputed_visual_features = features
            self.idx2visfeatidx = idx2visfeatidx
        else:
            self.precomputed_visual_features = None
            self.idx2visfeatidx = None

        self.dataset, self.dataloader = self._create_label_based_dataset_and_dataloader(
            indices=list(range(len(self.labels))),
            labels=self.labels,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_batch_fn=collate_batch_fn,
            infinite=True,
        )

    def _create_vqa_dataset(self, q, a, indices, infinite=True):
        labels = self.finding_labels if self.use_merged_findings else self.labels
        return CheXpertVQADataset(
            self.image_paths, self.transform, self.orientations, self.genders, labels,
            question=q, answer=a, indices=indices,
            include_image=self.include_image,
            use_precomputed_visual_features=self.use_precomputed_visual_features,
            precomputed_visual_features=self.precomputed_visual_features,
            idx2visfeatidx = self.idx2visfeatidx,
            infinite=infinite
        )

class CheXpertImageDataset(Dataset):
    
    def __init__(self, image_paths, transform, orientations, genders, chexpert_labels, indices,
                suffle_indices = True,
                # infinite mode
                infinite = False,
        ):
        self.images = image_paths
        self.transform = transform
        self.orientations = orientations
        self.genders = genders
        self.labels = chexpert_labels
        self.infinite = infinite
        self.indices = indices
        if suffle_indices: np.random.shuffle(self.indices)
        self._len = INFINITE_DATASET_LENGTH if infinite else len(self.indices)        
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        indices = self.indices
        if self.infinite:
            i %= len(indices)
        idx = indices[i]
        return dict(
            idx=idx,
            i=self.transform(Image.open(self.images[idx]).convert('RGB')),
            o=self.orientations[idx],
            g=self.genders[idx],
            l=self.labels[idx],
        )

class CheXpertVQADataset(Dataset):
    
    def __init__(self, image_paths, transform, orientations, genders, chexpert_labels,
                question, answer, indices,
                include_image = True,
                suffle_indices = True,
                # precomputed visual features
                use_precomputed_visual_features=False,
                precomputed_visual_features=None,
                idx2visfeatidx=None,
                # infinite mode
                infinite = False,
        ):
        self.images = image_paths
        self.transform = transform
        self.orientations = orientations
        self.genders = genders
        self.labels = chexpert_labels
        self.infinite = infinite
        self.question = question
        self.answer = answer
        self.indices = indices
        self.include_image = include_image
        
        if suffle_indices: np.random.shuffle(self.indices)
        self._len = INFINITE_DATASET_LENGTH if infinite else len(self.indices)

        if include_image:
            assert image_paths is not None

        self.use_precomputed_visual_features = use_precomputed_visual_features
        if use_precomputed_visual_features:
            assert precomputed_visual_features is not None
            assert idx2visfeatidx is not None
            self.precomputed_visual_features = precomputed_visual_features
            self.idx2visfeatidx = idx2visfeatidx
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        indices = self.indices
        if self.infinite:
            i %= len(indices)
        idx = indices[i]
        output = dict(
            idx=idx,
            o=self.orientations[idx],
            g=self.genders[idx],
            l=self.labels[idx],
            q=self.question,
            a=self.answer,
        )
        if self.include_image:
            output['i'] = self.transform(Image.open(self.images[idx]).convert('RGB'))
        if self.use_precomputed_visual_features:
            output['vf'] = self.precomputed_visual_features[self.idx2visfeatidx[idx]]
        return output

class CheXpert_MAE_Trainer(MAETrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers):

        self.transform = transform

        print(f'Loading dataframe from {CHEXPERT_TRAIN_VAL_CSV_PATH}')
        df = pd.read_csv(CHEXPERT_TRAIN_VAL_CSV_PATH)
        df_labels = df[CHEXPERT_LABELS]
        df_paths = df['Path']

        # images
        print('Loading images')
        image_paths = (CHEXPERT_DATASET_DIR + os.path.sep + df_paths).to_numpy()
        self.image_paths = image_paths

        # chexpert labels
        print('Loading chexpert labels')
        labels = df_labels.fillna(0).to_numpy().astype(np.int8)
        labels = np.where(labels == -1, 1, labels)
        self.labels = labels

        train_indices = list(range(len(labels)))
        labels_getter = lambda i: labels[i]
        super().__init__(train_indices, None, None, list(range(1, len(CHEXPERT_LABELS))),
                         labels_getter, batch_size, collate_batch_fn, num_workers, use_validation_set=False)
    
    def _create_mae_dataset(self, indices, shuffle=True, infinite=False):
        return BasicImageDataset(self.image_paths, self.transform, indices, shuffle, infinite)

class CheXpertPhraseGroundingTrainer(CheXpertTrainerBase):
    def __init__(self, train_image_transform, val_image_transform, collate_batch_fn, num_train_workers, num_val_workers,
                 phrase_embeddings_filepath, max_images_per_batch, max_phrases_per_batch, test_batch_size_factor,
                 training_data_mode=CheXpertTrainingMode.ALL, use_training_set=True, use_validation_set=True,
                 use_interpret_cxr_challenge_split=False, interpret_cxr_challenge_split_filepath=None):
        super().__init__(
            load_test_set=True,
            use_visualchexbert=True,
            apply_filtering=False,
            load_orientation=False,
            load_gender=False,
        )

        assert use_training_set or use_validation_set
        
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.training_data_mode = training_data_mode
        self.use_validation_set = use_validation_set

        # phrase embeddings
        print(f'Loding phrase_embeddings and phrases from {phrase_embeddings_filepath}...')
        tmp = get_cached_pickle_file(phrase_embeddings_filepath)
        phrase_embeddings = tmp['class_phrase_embeddings']
        phrases = tmp['class_phrases']
        assert phrase_embeddings.shape[0] == len(phrases)
        assert phrases == [CHEXPERT_LABEL2PHRASE[label] for label in CHEXPERT_LABELS]
        print(f'phrase_embeddings.shape = {phrase_embeddings.shape}')
        print(f'len(phrases) = {len(phrases)}')
        for phrase in phrases:
            print('\t', phrase)

        # phrase classification labels
        all_image_paths = np.concatenate([self.image_paths, self.test_image_paths])
        all_labels = np.concatenate([self.labels, self.test_labels])
        print(f'len(all_image_paths) = {len(all_image_paths)}')
        print(f'all_labels.shape = {all_labels.shape}')

        if use_interpret_cxr_challenge_split:
            assert interpret_cxr_challenge_split_filepath is not None
            print_bold(f'Using split from {interpret_cxr_challenge_split_filepath}')
            challenge_split = load_pickle(interpret_cxr_challenge_split_filepath)
            all_image_partial_paths = []
            for ip in all_image_paths:
                if 'train/' in ip:
                    all_image_partial_paths.append(ip[ip.index('train/'):])
                elif 'valid/' in ip:
                    all_image_partial_paths.append(ip[ip.index('valid/'):])
                elif 'test/' in ip:
                    all_image_partial_paths.append(ip[ip.index('test/'):])
                else:
                    raise ValueError(f'Unknown image path: {ip}')
            image_partial_path_2_idx = {p: i for i, p in enumerate(all_image_partial_paths)}

        if use_training_set:
            print('Generating train dataset and dataloader')
            if use_interpret_cxr_challenge_split:
                # print('DEBUG: all_image_partial_paths[:10]')
                # print(all_image_partial_paths[:10])
                train_indices = [image_partial_path_2_idx[ip] for ip in challenge_split['train']]
            else:
                if training_data_mode == CheXpertTrainingMode.TRAIN_VAL:
                    train_indices = self.train_indices + self.valid_indices # train + val set
                elif training_data_mode == CheXpertTrainingMode.VAL_TEST:
                    train_indices = self.valid_indices + list(range(len(self.labels), len(all_labels))) # val + test set
                elif training_data_mode == CheXpertTrainingMode.TEST:
                    train_indices = list(range(len(self.labels), len(all_labels))) # only test set
                elif training_data_mode == CheXpertTrainingMode.ALL:
                    train_indices = list(range(len(all_labels)))
                else: assert False, f'Unknown training_data_mode = {training_data_mode}'
            assert len(train_indices) == len(set(train_indices)) # no duplicates
            print(f'len(train_indices) = {len(train_indices)}')

            grouped_indices = group_indices_for_balanced_sampling(label_matrix=all_labels,
                                                                 indices=train_indices,
                                                                 label_names=CHEXPERT_LABELS,
                                                                 min_group_size=50)
            train_datasets = []
            train_weights = []
            for indices in grouped_indices:
                dataset = ImageFactBasedMultilabelClassificationDataset(
                    image_paths=all_image_paths,
                    image_transform=self.train_image_transform,
                    phrase_embeddings=phrase_embeddings,
                    phrase_classification_labels=all_labels,
                    indices=indices,
                    infinite=True,
                    shuffle_indices=True,
                )
                weight = math.log2(len(indices)) ** 3
                train_datasets.append(dataset)
                train_weights.append(weight)
                print(f'  len(indices) = {len(indices)}, weight = {weight}')
            self.train_dataset = CompositeInfiniteDataset(train_datasets, train_weights)
            batch_size = max(min(max_images_per_batch, max_phrases_per_batch // len(phrases)), 1) # at least 1 image per batch
            self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_train_workers,
                                            collate_fn=collate_batch_fn,
                                            pin_memory=True)
        
        if use_validation_set:
            if use_interpret_cxr_challenge_split:
                test_indices = [image_partial_path_2_idx[ip] for ip in challenge_split['val']]
            else:
                test_indices = [i for i in range(len(self.labels), len(all_labels))]
            if not use_interpret_cxr_challenge_split:
                assert len(test_indices) == len(self.test_labels)
            print('Generating val dataset and dataloader')
            print(f'len(test_indices) = {len(test_indices)}')
            self.val_dataset = ImageFactBasedMultilabelClassificationDataset(
                image_paths=all_image_paths,
                image_transform=self.val_image_transform,
                phrase_embeddings=phrase_embeddings,
                phrase_classification_labels=all_labels,
                indices=test_indices,
            )
            batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // len(phrases)), 1) * test_batch_size_factor)
            self.val_dataloader = DataLoader(self.val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_val_workers,
                                             collate_fn=collate_batch_fn,
                                             pin_memory=True)