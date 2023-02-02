from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from medvqa.utils.common import CACHE_DIR

CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR = os.environ['CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR']
CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH = os.environ['CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH']
CHEST_IMAGENOME_CACHE_DIR = os.path.join(CACHE_DIR, 'chest_imagenome')

CHEST_IMAGENOME_ATTRIBUTES_DICT = {
    'anatomicalfinding': [
        'lung opacity',
        'airspace opacity',
        'consolidation',
        'infiltration',
        'atelectasis',
        'linear/patchy atelectasis',
        'lobar/segmental collapse',
        'pulmonary edema/hazy opacity',
        'vascular congestion',
        'vascular redistribution',
        'increased reticular markings/ild pattern',
        'pleural effusion',
        'costophrenic angle blunting',
        'pleural/parenchymal scarring',
        'bronchiectasis',
        'enlarged cardiac silhouette',
        'mediastinal displacement',
        'mediastinal widening',
        'enlarged hilum',
        'tortuous aorta',
        'vascular calcification',
        'pneumomediastinum',
        'pneumothorax',
        'hydropneumothorax',
        'lung lesion',
        'mass/nodule (not otherwise specified)',
        'multiple masses/nodules',
        'calcified nodule',
        'superior mediastinal mass/enlargement',
        'rib fracture',
        'clavicle fracture',
        'spinal fracture',
        'hyperaeration',
        'cyst/bullae',
        'elevated hemidiaphragm',
        'diaphragmatic eventration (benign)',
        'sub-diaphragmatic air',
        'subcutaneous air',
        'hernia',
        'scoliosis',
        'spinal degenerative changes',
        'shoulder osteoarthritis',
        'bone lesion',
    ],
    'nlp': [
        'abnormal',
        'normal',
    ],
    'disease': [
        'pneumonia',
        'fluid overload/heart failure',
        'copd/emphysema',
        'granulomatous disease',
        'interstitial lung disease',
        'goiter',
        'lung cancer',
        'aspiration',
        'alveolar hemorrhage',
        'pericardial effusion',
    ],
    'technicalassessment': [
        'low lung volumes',
        'rotated',
        'artifact',
        'breast/nipple shadows',
        'skin fold',
    ],
    'tubesandlines': [
        'chest tube',
        'mediastinal drain',
        'pigtail catheter',
        'endotracheal tube',
        'tracheostomy tube',
        'picc',
        'ij line',
        'chest port',
        'subclavian line',
        'swan-ganz catheter',
        'intra-aortic balloon pump',
        'enteric tube',
    ],
    'device': [
        'sternotomy wires',
        'cabg grafts',
        'aortic graft/repair',
        'prosthetic valve',
        'cardiac pacer and wires',
    ],
    'texture': [
        'opacity',
        'alveolar',
        'interstitial',
        'calcified',
        'lucency',
    ],
}

def load_gold_standard_dicom_ids():
    df = pd.read_csv(CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH)
    return df['dicom_id'].tolist()