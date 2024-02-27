import numpy as np
import pandas as pd

from medvqa.datasets.mimiccxr import (
    MIMICCXR_CUSTOM_RADIOLOGIST_ANNOTATIONS_CSV_1_PATH,
    MIMICCXR_CUSTOM_RADIOLOGIST_ANNOTATIONS_CSV_2_PATH,
)

def load_mimiccxr_custom_radiologist_annotations():
    df1 = pd.read_csv(MIMICCXR_CUSTOM_RADIOLOGIST_ANNOTATIONS_CSV_1_PATH)
    df2 = pd.read_csv(MIMICCXR_CUSTOM_RADIOLOGIST_ANNOTATIONS_CSV_2_PATH)
    assert df1['Sentence'].equals(df2['Sentence'])
    
    finding_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Lung Opacity', 'Pleural Effusion', 'Any other finding']
    n_rows = len(df1)
    n_labels = len(finding_columns)
    labels = np.empty((n_rows, n_labels * 2), dtype=np.int8) # 2 experts
    
    for i, col in enumerate(finding_columns):
        # Map string values to integers
        # Abnormal: 1, Normal: 0, Uncertain: -1, nan: -2
        values = df1[col]
        values.fillna(-2, inplace=True)
        values.replace('Uncertain', -1, inplace=True)
        values.replace('Normal', 0, inplace=True)
        values.replace('Abnormal', 1, inplace=True)
        labels[:, i] = values.values
        
        values = df2[col]
        values.fillna(-2, inplace=True)
        values.replace('Uncertain', -1, inplace=True)
        values.replace('Normal', 0, inplace=True)
        values.replace('Abnormal', 1, inplace=True)
        labels[:, i + n_labels] = values.values

    assert set(labels.flatten()) == {-2, -1, 0, 1}
    sentences = df1['Sentence'].values
    return sentences, labels