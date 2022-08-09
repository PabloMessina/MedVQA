import os
import numpy as np
import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from medvqa.datasets.vinbig import (
    VINBIG_LABELS_CSV_PATH,
    VINBIG_256x256_IMAGES_FOLDER,
    VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH,
    VINBIG_IMAGE_LABELS_TEST_CSV_PATH,
    N_IMAGES_TRAIN,
    N_IMAGES_TEST,
)
from medvqa.utils.constants import VINBIG_DISEASES

from medvqa.datasets.dataloading_utils import (
    INFINITE_DATASET_LENGTH,
    CompositeDataset,
    CompositeInfiniteDataset,
    get_imbalance_reduced_weights,
)
from medvqa.datasets.vqa import load_precomputed_visual_features
from medvqa.models.report_generation.templates.vinbig_v1 import TEMPLATES_VINBIG_v1

def merge_labels(*labels_list):
    merged = np.zeros((len(VINBIG_DISEASES),), np.int8)
    merged[-1] = 1
    for labels in labels_list:
        if labels[-1] == 0: # no findings
            merged[-1] = 0
        for i in range(0, len(VINBIG_DISEASES)-1): # findings
            if labels[i] == 1:
                merged[i] = 1
    return merged

class VinBigTrainerBase:
    def __init__(self):

        print('Loading dataframe from', VINBIG_LABELS_CSV_PATH)
        df_labels_train = pd.read_csv(VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH)
        df_labels_test = pd.read_csv(VINBIG_IMAGE_LABELS_TEST_CSV_PATH)

        assert len(df_labels_train) == N_IMAGES_TRAIN * 3
        assert len(df_labels_test) == N_IMAGES_TEST

        # Images ids        
        image_ids = [None] * (N_IMAGES_TRAIN + N_IMAGES_TEST)
        self.image_ids = image_ids
        # train image ids
        train_image_ids = df_labels_train['image_id']
        for i in range(N_IMAGES_TRAIN):
            image_ids[i] = train_image_ids[i * 3]
            for j in range(1, 3):
                assert train_image_ids[i * 3 + j] == image_ids[i]
            assert image_ids[i] != image_ids[i-1]
        self.train_indices = [i for i in range(N_IMAGES_TRAIN)]        
        # test image ids
        test_images_ids = df_labels_test['image_id']
        for i in range(N_IMAGES_TEST):
            image_ids[N_IMAGES_TRAIN + i] = test_images_ids[i]
        self.test_indices = [i + N_IMAGES_TRAIN for i in range(N_IMAGES_TEST)]

        # Image paths
        self.image_paths = [os.path.join(VINBIG_256x256_IMAGES_FOLDER, f'{img_id}.png') for img_id in image_ids]

        # Labels
        labels = np.empty((N_IMAGES_TRAIN + N_IMAGES_TEST, len(VINBIG_DISEASES)), dtype=np.int8)
        self.labels = labels
        # train labels
        tmp = VINBIG_DISEASES[:]
        tmp[tmp.index('Other disease')] = 'Other diseases' # HACK
        train_labels = df_labels_train[tmp].values
        for i in range(N_IMAGES_TRAIN):
            labels[i] = merge_labels(
                train_labels[3 * i],
                train_labels[3 * i + 1],
                train_labels[3 * i + 2]
            )
        # test labels
        labels[N_IMAGES_TRAIN:] = df_labels_test[VINBIG_DISEASES].values

class VinBig_VQA_Trainer(VinBigTrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers, tokenizer,
                include_image=True,
                use_precomputed_visual_features=False,
                precomputed_visual_features_path=None,
                train_with_everything = True,
        ):
        super().__init__()
        
        self.transform = transform
        self.include_image = include_image
        self.use_precomputed_visual_features = use_precomputed_visual_features
        self.tokenizer = tokenizer
        self.train_with_everything = train_with_everything

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
        
        print('Generating train dataset and dataloader')
        train_indices = self.train_indices + self.test_indices if train_with_everything else self.train_indices
        self.train_dataset, self.train_dataloader = self._generate_dataset_and_dataloader(
            train_indices, batch_size, collate_batch_fn, num_workers, infinite=True
        )
        print('Generating val dataset and dataloader')
        self.val_dataset, self.val_dataloader = self._generate_dataset_and_dataloader(
            self.test_indices, batch_size, collate_batch_fn, num_workers, infinite=False, n_samples=20,
        )

    def _generate_dataset_and_dataloader(self, indices, batch_size, collate_batch_fn, num_workers,
                                         infinite=True, n_samples=None):
        # balanced dataset
        disease_datasets = []
        pos_counts = []
        for i in range(0, len(VINBIG_DISEASES)):
            pos_indices = []
            neg_indices = []
            for j in indices:
                if self.labels[j][i] == 1:
                    pos_indices.append(j)
                elif self.labels[j][i] == 0:
                    neg_indices.append(j)
                else: assert False

            if n_samples is not None:
                if len(pos_indices) > n_samples:
                    pos_indices = random.sample(pos_indices, n_samples)
                if len(neg_indices) > n_samples:
                    neg_indices = random.sample(neg_indices, n_samples)
            
            print(f'label = {i}, len(pos_indices)={len(pos_indices)}, len(neg_indices)={len(neg_indices)}')
            
            pos_counts.append(len(pos_indices))

            # positive            
            if len(pos_indices) > 0:
                pos_indices = np.array(pos_indices, dtype=int)
                pos_answer = self.tokenizer.string2ids(TEMPLATES_VINBIG_v1[VINBIG_DISEASES[i]][1].lower())
                pos_dataset = self._create_vqa_dataset(q=i, a=pos_answer, indices=pos_indices, infinite=infinite)
            else:
                pos_dataset = None

            # negative
            if len(neg_indices) > 0:
                neg_indices = np.array(neg_indices, dtype=int)
                neg_answer = self.tokenizer.string2ids(TEMPLATES_VINBIG_v1[VINBIG_DISEASES[i]][0].lower())
                neg_dataset = self._create_vqa_dataset(q=i, a=neg_answer, indices=neg_indices, infinite=infinite)
            else:
                neg_dataset = None
            
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
            weights = get_imbalance_reduced_weights(pos_counts, 0.2)
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
    
    def _create_vqa_dataset(self, q, a, indices, infinite):
        return VinBigVQADataset(
            self.image_paths, self.transform, self.labels,
            question=q, answer=a, indices=indices,
            include_image=self.include_image,
            use_precomputed_visual_features=self.use_precomputed_visual_features,
            precomputed_visual_features=self.precomputed_visual_features,
            idx2visfeatidx = self.idx2visfeatidx,
            infinite=infinite,
        )

class VinBigVQADataset(Dataset):
    
    def __init__(self, image_paths, transform, labels, question, answer, indices,
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
        self.labels = labels
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
            l=self.labels[idx],
            q=self.question,
            a=self.answer,
        )
        if self.include_image:
            output['i'] = self.transform(Image.open(self.images[idx]).convert('RGB'))
        if self.use_precomputed_visual_features:
            output['vf'] = self.precomputed_visual_features[self.idx2visfeatidx[idx]]
        return output
