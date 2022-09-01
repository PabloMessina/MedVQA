import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from medvqa.datasets.cxr14 import CXR14_IMAGES_DIR_PATH, CXR14_METADATA_CSV_PATH
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH
from medvqa.datasets.vqa import LabelBasedVQAClass
from medvqa.models.report_generation.templates.cxr14_v1 import TEMPLATES_CXR14_v1
from medvqa.utils.constants import CXR14_DATASET_ID, CXR14_GENDER2ID, CXR14_LABELS, CXR14_ORIENTATION2ID

class CXR14TrainerBase(LabelBasedVQAClass):
    
    def __init__(self, use_merged_findings=False, findings_remapper=None, n_findings=None):

        print('Loading dataframe from', CXR14_METADATA_CSV_PATH)
        df = pd.read_csv(CXR14_METADATA_CSV_PATH)
        n = len(df)
        df_orien = df['View Position']
        df_gender = df['Patient Gender']
        df_labels = df['Finding Labels']
        df_images = df['Image Index']

        # images
        print('Loading images')
        image_paths = (CXR14_IMAGES_DIR_PATH + os.path.sep + df_images).to_numpy()
        self.image_paths = image_paths
        
        # orientations
        print('Loading orientations')
        orientations = np.empty((n,), dtype=int)
        for i, x in enumerate(df_orien):
            orientations[i] = CXR14_ORIENTATION2ID[x]
        self.orientations = orientations
        
        # gender
        print('Loading genders')
        genders = np.empty((n,), dtype=int)
        for i, x in enumerate(df_gender):
            genders[i] = CXR14_GENDER2ID[x]
        self.genders = genders

        # chexpert labels
        print('Loading chexpert labels')
        labels = np.zeros((n, len(CXR14_LABELS)), dtype=np.int8)
        for i, x in enumerate(df_labels):
            for label in x.split('|'):
                labels[i][CXR14_LABELS.index(label)] = 1
        self.labels = labels

        super().__init__(
            label_names=CXR14_LABELS,
            templates=TEMPLATES_CXR14_v1,
            labels_offset=0,
            use_merged_findings=use_merged_findings,
            labels2mergedfindings=findings_remapper[CXR14_DATASET_ID] if use_merged_findings else None,
            n_findings=n_findings,
            labels=self.labels,
        )


class CXR14_VQA_Trainer(CXR14TrainerBase):
    def __init__(self, transform, batch_size, collate_batch_fn, num_workers, tokenizer,
                use_merged_findings=False, findings_remapper=None, n_findings=None):
        
        super().__init__(
            use_merged_findings=use_merged_findings,
            findings_remapper=findings_remapper,
            n_findings=n_findings,
        )
        
        self.transform = transform
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
        return CXR14VQADataset(
            self.image_paths, self.transform, self.orientations, self.genders, labels,
            question=q, answer=a, indices=indices, infinite=infinite
        )

class CXR14VQADataset(Dataset):
    
    def __init__(self, image_paths, transform, orientations, genders, labels,
                question, answer, indices,
                suffle_indices = True,
                # infinite mode
                infinite = False,
        ):
        self.images = image_paths
        self.transform = transform
        self.orientations = orientations
        self.genders = genders
        self.labels = labels
        self.infinite = infinite
        self.question = question
        self.answer = answer
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
        output = dict(
            idx=idx,
            o=self.orientations[idx],
            g=self.genders[idx],
            l=self.labels[idx],
            q=self.question,
            a=self.answer,
            i=self.transform(Image.open(self.images[idx]).convert('RGB')),
        )
        return output
