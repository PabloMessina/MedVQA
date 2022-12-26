import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from medvqa.datasets.vinbig import (
    VINBIG_LABELS_CSV_PATH,
    VINBIG_ORIGINAL_IMAGES_FOLDER,
    VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH,
    VINBIG_IMAGE_LABELS_TEST_CSV_PATH,
    N_IMAGES_TRAIN,
    N_IMAGES_TEST,
)
from medvqa.utils.constants import VINBIG_DATASET_ID, VINBIG_DISEASES

from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH
from medvqa.datasets.vqa import LabelBasedVQAClass, load_precomputed_visual_features
from medvqa.models.report_generation.templates.vinbig_v1 import TEMPLATES_VINBIG_v1

def merge_labels(*labels_list):
    merged = np.zeros((len(VINBIG_DISEASES),), np.int8)
    merged[-1] = 1
    
    # First check: majority thinks it's healthy
    healthy_count = 0
    for labels in labels_list:
        if labels[-1] == 1:
            assert labels.sum() == 1
            healthy_count += 1
        else:
            assert labels.sum() > 0
    if healthy_count >= len(labels_list) - 1:
        return merged

    # General case: union of labels
    for labels in labels_list:
        if labels[-1] == 0: # no findings
            merged[-1] = 0
        for i in range(0, len(VINBIG_DISEASES)-1): # findings
            if labels[i] == 1:
                merged[i] = 1
    return merged

class VinBigTrainingData:
    TRAIN_ONLY = 'train'
    TEST_ONLY = 'test'
    ALL = 'all'

class VinBigTrainerBase(LabelBasedVQAClass):
    
    def __init__(self, use_merged_findings=False, findings_remapper=None, n_findings=None):

        print('Loading dataframe from:'
                f'\n  {VINBIG_IMAGE_LABELS_TRAIN_CSV_PATH}'
                f'\n  {VINBIG_IMAGE_LABELS_TEST_CSV_PATH}')
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
        self.image_paths = [os.path.join(VINBIG_ORIGINAL_IMAGES_FOLDER, f'{img_id}.png') for img_id in image_ids]

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

        # sanity check train labels
        self._sanity_check_train_labels()

        super().__init__(
            label_names=VINBIG_DISEASES,
            templates=TEMPLATES_VINBIG_v1,
            labels_offset=0,
            use_merged_findings=use_merged_findings,
            labels2mergedfindings=findings_remapper[VINBIG_DATASET_ID] if use_merged_findings else None,
            n_findings=n_findings,
            labels=self.labels,
        )

    def _sanity_check_train_labels(self):
        print('Sanity checking train labels ...')
        df_labels = pd.read_csv(VINBIG_LABELS_CSV_PATH) # these are labels from kaggle's challenge
        label_names = df_labels.columns[1:]
        label_indices = [VINBIG_DISEASES.index(x) for x in label_names]
        gt_labels = df_labels[label_names].values
        assert self.labels.shape[0] == gt_labels.shape[0]
        assert self.labels.shape[1] > gt_labels.shape[1]
        m = gt_labels.shape[1]
        mismatches = 0
        for i in range(N_IMAGES_TRAIN):
            try:
                assert all(self.labels[i][label_indices[j]] == gt_labels[i][j] for j in range(m))
            except AssertionError:
                mismatches += 1
                if mismatches > 4: # tolerate no more than 4 mismatches (empirical heuristic)
                    raise
        print('Done!')

class VinBig_VQA_Trainer(VinBigTrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers, tokenizer,
                include_image=True,
                use_precomputed_visual_features=False,
                precomputed_visual_features_path=None,
                training_data = VinBigTrainingData.ALL,
                use_validation = True,
                use_merged_findings=False,
                findings_remapper=None,
                n_findings=None,
        ):
        super().__init__(
            use_merged_findings=use_merged_findings,
            findings_remapper=findings_remapper,
            n_findings=n_findings,
        )
        
        self.transform = transform
        self.include_image = include_image
        self.use_precomputed_visual_features = use_precomputed_visual_features
        self.tokenizer = tokenizer
        self.training_data = training_data
        self.use_validation = use_validation

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
        if training_data == VinBigTrainingData.TRAIN_ONLY:
            train_indices = self.train_indices
        elif training_data == VinBigTrainingData.TEST_ONLY:
            train_indices = self.test_indices
        elif training_data == VinBigTrainingData.ALL:
            train_indices = self.train_indices + self.test_indices
        else: assert False, f'Unknown training_data = {training_data}'         
        
        self.train_dataset, self.train_dataloader = self._generate_dataset_and_dataloader(
            train_indices, batch_size, collate_batch_fn, num_workers, infinite=True, min_pos_to_include=10,
        )

        if use_validation:
            print('Generating val dataset and dataloader')
            self.val_dataset, self.val_dataloader = self._generate_dataset_and_dataloader(
                self.test_indices, batch_size, collate_batch_fn, num_workers, infinite=False, n_samples=20,
            )

    def _generate_dataset_and_dataloader(self, indices, batch_size, collate_batch_fn, num_workers,
                                         infinite=True, n_samples=None, min_pos_to_include=0):

        return self._create_label_based_dataset_and_dataloader(
            indices=indices,
            labels=self.labels,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_batch_fn=collate_batch_fn,
            infinite=infinite,
            n_samples=n_samples,
            min_pos_to_include=min_pos_to_include,
            log_weighting=True,
        )
    
    def _create_vqa_dataset(self, q, a, indices, infinite):
        labels = self.finding_labels if self.use_merged_findings else self.labels
        return VinBigVQADataset(
            self.image_paths, self.transform, labels,
            question=q, answer=a, indices=indices,
            include_image=self.include_image,
            use_precomputed_visual_features=self.use_precomputed_visual_features,
            precomputed_visual_features=self.precomputed_visual_features,
            idx2visfeatidx = self.idx2visfeatidx,
            infinite=infinite,
        )

class VinBig_VisualModuleTrainer(VinBigTrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers,
                training_data = VinBigTrainingData.ALL,
                use_merged_findings=False, findings_remapper=None, n_findings=None,
        ):
        super().__init__(
            use_merged_findings=use_merged_findings,
            findings_remapper=findings_remapper,
            n_findings=n_findings,
        )
        
        self.transform = transform
        self.training_data = training_data
        
        print('Generating train dataset and dataloader')
        if training_data == VinBigTrainingData.TRAIN_ONLY:
            train_indices = self.train_indices
        elif training_data == VinBigTrainingData.TEST_ONLY:
            train_indices = self.test_indices
        elif training_data == VinBigTrainingData.ALL:
            train_indices = self.train_indices + self.test_indices
        else: assert False, f'Unknown training_data = {training_data}'
        
        self.train_dataset, self.train_dataloader = self._create_label_based_dataset_and_dataloader(
            indices=train_indices,
            labels=self.labels,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_batch_fn=collate_batch_fn,
            infinite=True,
            min_pos_to_include=0,
            log_weighting=True,
            include_qa=False,
        )
    
    def _create_visual_dataset(self, indices, infinite):
        labels = self.finding_labels if self.use_merged_findings else self.labels
        return VinBigVisualDataset(
            self.image_paths, self.transform, labels,
            indices=indices,
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

class VinBigVisualDataset(Dataset):
    
    def __init__(self, image_paths, transform, labels, indices, suffle_indices=True, infinite=False):
        self.images = image_paths
        self.transform = transform
        self.labels = labels
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
            l=self.labels[idx],
        )