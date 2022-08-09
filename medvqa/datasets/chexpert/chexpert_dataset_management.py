import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from medvqa.datasets.chexpert import CHEXPERT_DATASET_DIR, CHEXPERT_TRAIN_VAL_CSV_PATH
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH
from medvqa.datasets.vqa import CompositeInfiniteDataset, load_precomputed_visual_features
from medvqa.models.report_generation.templates.chex_v1 import TEMPLATES_CHEXPERT_v1
from medvqa.utils.constants import CHEXPERT_LABELS

_orientation2id = {
    'FrontalAP': 0,
    'Lateral': 1,
    'FrontalPA': 2
}
_gender2id = {
    'Female': 0,
    'Male': 1,
}

class ChexpertTrainerBase:
    def __init__(self):

        print('Loading dataframe from', CHEXPERT_TRAIN_VAL_CSV_PATH)
        df = pd.read_csv(CHEXPERT_TRAIN_VAL_CSV_PATH)
        df_orien = df['Frontal/Lateral'] + df['AP/PA'].fillna('')
        df_gender = df['Sex']
        df_labels = df[CHEXPERT_LABELS]        
        valid_rows = (df_orien != 'FrontalLL') & (df_orien != 'FrontalRL') & (df_gender != 'Unknown')

        df_orien = df_orien[valid_rows]
        df_gender = df_gender[valid_rows]
        df_labels = df_labels[valid_rows]
        df_paths = df['Path'][valid_rows]
        n = len(df_paths)

        # images
        print('Loading images')
        image_paths = (CHEXPERT_DATASET_DIR + os.path.sep + df_paths).to_numpy()
        self.image_paths = image_paths
        
        # orientations
        print('Loading orientations')
        orientations = np.empty((n,), dtype=int)
        for i, x in enumerate(df_orien):
            orientations[i] = _orientation2id[x]
        self.orientations = orientations
        
        # gender
        print('Loading genders')
        genders = np.empty((n,), dtype=int)
        for i, x in enumerate(df_gender):
            genders[i] = _gender2id[x]
        self.genders = genders

        # chexpert labels
        print('Loading chexpert labels')
        labels = df_labels.fillna(0).to_numpy().astype(np.int8)
        labels = np.where(labels == -1, 1, labels)
        self.labels = labels

class Chexpert_VisualModuleTrainer(ChexpertTrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers):        
        super().__init__()
        # dataset
        self.dataset = ChexpertImageDataset(self.image_paths, transform, self.orientations,
                                       self.genders, self.labels, infinite=True)        
        # dataloader
        self.dataloader = DataLoader(self.dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=collate_batch_fn,
                                    pin_memory=True)

class Chexpert_VQA_Trainer(ChexpertTrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers, tokenizer,
                include_image=True,
                use_precomputed_visual_features=False,
                precomputed_visual_features_path=None,
        ):
        super().__init__()
        
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
        
        # balanced dataset
        
        n = len(self.labels)
        disease_datasets = []
        for i in range(len(CHEXPERT_LABELS)):
            pos_indices = []
            neg_indices = []
            for j in range(n):
                if self.labels[j][i] == 1:
                    pos_indices.append(j)
                else:
                    neg_indices.append(j)
            
            print(f'label = {i}, len(pos_indices)={len(pos_indices)}, len(neg_indices)={len(neg_indices)}')
            
            # positive
            pos_indices = np.array(pos_indices, dtype=int)
            pos_answer = tokenizer.string2ids(TEMPLATES_CHEXPERT_v1[CHEXPERT_LABELS[i]][1].lower())
            pos_dataset = self._create_vqa_dataset(q=i, a=pos_answer, indices=pos_indices)

            # negative
            neg_indices = np.array(neg_indices, dtype=int)
            neg_answer = tokenizer.string2ids(TEMPLATES_CHEXPERT_v1[CHEXPERT_LABELS[i]][0].lower())
            neg_dataset = self._create_vqa_dataset(q=i, a=neg_answer, indices=neg_indices)
            
            # merged
            comp_dataset = CompositeInfiniteDataset([pos_dataset, neg_dataset], [1, 1])
            disease_datasets.append(comp_dataset)
        
        self.dataset = CompositeInfiniteDataset(disease_datasets, [1] * len(disease_datasets))

        # dataloader
        self.dataloader = DataLoader(self.dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=collate_batch_fn,
                                    pin_memory=True)

    def _create_vqa_dataset(self, q, a, indices):
        return ChexpertVQADataset(
            self.image_paths, self.transform, self.orientations, self.genders, self.labels,
            question=q, answer=a, indices=indices,
            include_image=self.include_image,
            use_precomputed_visual_features=self.use_precomputed_visual_features,
            precomputed_visual_features=self.precomputed_visual_features,
            idx2visfeatidx = self.idx2visfeatidx,
            infinite=True
        )

class ChexpertImageDataset(Dataset):
    
    def __init__(self, image_paths, transform, orientations, genders, chexpert_labels,
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
        self.indices = np.arange(len(self.images))
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

class ChexpertVQADataset(Dataset):
    
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
