import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from medvqa.datasets.chexpert import (
    CHEXPERT_DATASET_DIR,
    CHEXPERT_TRAIN_VAL_CSV_PATH,
)
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH
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

class Chexpert_VisualModuleTrainer:
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers):

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
        
        # orientations
        print('Loading orientations')
        orientations = np.empty((n,), dtype=int)
        for i, x in enumerate(df_orien):
            orientations[i] = _orientation2id[x]
        
        # gender
        print('Loading genders')
        genders = np.empty((n,), dtype=int)
        for i, x in enumerate(df_gender):
            genders[i] = _gender2id[x]

        # chexpert labels
        print('Loading chexpert labels')
        labels = df_labels.fillna(0).to_numpy().astype(np.int8)
        labels = np.where(labels == -1, 1, labels)

        # dataset
        self.dataset = ChexpertDataset(image_paths, transform, orientations, genders, labels, infinite=True)
        
        # dataloader
        self.dataloader = DataLoader(self.dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=collate_batch_fn,
                                    pin_memory=True)        

class ChexpertDataset(Dataset):
    
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