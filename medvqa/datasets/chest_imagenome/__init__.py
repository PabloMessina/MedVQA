from dotenv import load_dotenv
load_dotenv()

import os
from medvqa.utils.common import CACHE_DIR

CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR = os.environ['CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR']
CHEST_IMAGENOME_SILVER_DATASET_DIR = os.environ['CHEST_IMAGENOME_SILVER_DATASET_DIR']
CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH = os.environ['CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH']
CHEST_IMAGENOME_GOLD_BBOX_COORDINATE_ANNOTATIONS_CSV_PATH = os.environ['CHEST_IMAGENOME_GOLD_BBOX_COORDINATE_ANNOTATIONS_CSV_PATH']
CHEST_IMAGENOME_CACHE_DIR = os.path.join(CACHE_DIR, 'chest_imagenome')
CHEST_IMAGENOME_SILVER_BBOXES_FILEPATH = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'bboxes.pkl')
CHEST_IMAGENOME_HORIZONTALLY_FLIPPED_SILVER_BBOXES_FILEPATH = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'horizontally_flipped_bboxes.pkl')

CHEST_IMAGENOME_NUM_BBOX_CLASSES = 36
CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES = 26

CHEST_IMAGENOME_BBOX_NAMES = [
    'right lung',
    'right upper lung zone',
    'right mid lung zone',
    'right lower lung zone',
    'right hilar structures',
    'right apical zone',
    'right costophrenic angle',
    'right cardiophrenic angle',
    'right hemidiaphragm',
    'left lung',
    'left upper lung zone',
    'left mid lung zone',
    'left lower lung zone',
    'left hilar structures',
    'left apical zone',
    'left costophrenic angle',
    'left hemidiaphragm',
    'trachea',
    'spine',
    'right clavicle',
    'left clavicle',
    'aortic arch',
    'mediastinum',
    'upper mediastinum',
    'svc',
    'cardiac silhouette',
    'left cardiac silhouette',
    'right cardiac silhouette',
    'cavoatrial junction',
    'right atrium',
    'descending aorta',
    'carina',
    'left upper abdomen',
    'right upper abdomen',
    'abdomen',
    'left cardiophrenic angle',
]

CHEST_IMAGENOME_BBOX_NAME_TO_SHORT = {
    'right lung': 'RL',
    'right upper lung zone': 'RULZ',
    'right mid lung zone': 'RMLZ',
    'right lower lung zone': 'RLLZ',
    'right hilar structures': 'RHS',
    'right apical zone': 'RAZ',
    'right costophrenic angle': 'RCOA',
    'right cardiophrenic angle': 'RCAA',
    'right hemidiaphragm': 'RHD',
    'left lung': 'LL',
    'left upper lung zone': 'LULZ',
    'left mid lung zone': 'LMLZ',
    'left lower lung zone': 'LLLZ',
    'left hilar structures': 'LHS',
    'left apical zone': 'LAZ',
    'left costophrenic angle': 'LCOA',
    'left hemidiaphragm': 'LHD',
    'trachea': 'T',
    'spine': 'S',
    'right clavicle': 'RC',
    'left clavicle': 'LC',
    'aortic arch': 'AA',
    'mediastinum': 'M',
    'upper mediastinum': 'UM',
    'svc': 'SVC',
    'cardiac silhouette': 'CS',
    'left cardiac silhouette': 'LCS',
    'right cardiac silhouette': 'RCS',
    'cavoatrial junction': 'CAJ',
    'right atrium': 'RA',
    'descending aorta': 'DA',
    'carina': 'C',
    'left upper abdomen': 'LUA',
    'right upper abdomen': 'RUA',
    'abdomen': 'A',
    'left cardiophrenic angle': 'LCAA',
}

CHEST_IMAGENOME_SHORT_TO_BBOX_NAME = {}
for k, v in CHEST_IMAGENOME_BBOX_NAME_TO_SHORT.items():
    CHEST_IMAGENOME_SHORT_TO_BBOX_NAME[v] = k

assert set(CHEST_IMAGENOME_SHORT_TO_BBOX_NAME.keys()) == set(CHEST_IMAGENOME_BBOX_NAME_TO_SHORT.values())
assert set(CHEST_IMAGENOME_SHORT_TO_BBOX_NAME.values()) == set(CHEST_IMAGENOME_BBOX_NAME_TO_SHORT.keys())

CHEST_IMAGENOME_GOLD_BBOX_NAMES = [
    'aortic arch',
    'cardiac silhouette',
    'carina',
    'cavoatrial junction',
    'left apical zone',
    'left clavicle',
    'left costophrenic angle',
    'left hemidiaphragm',
    'left hilar structures',
    'left lower lung zone',
    'left lung',
    'left mid lung zone',
    'left upper lung zone',
    'right apical zone',
    'right atrium',
    'right clavicle',
    'right costophrenic angle',
    'right hemidiaphragm',
    'right hilar structures',
    'right lower lung zone',
    'right lung',
    'right mid lung zone',
    'right upper lung zone',
    'svc',
    'trachea',
    'upper mediastinum',
]

CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED = []
for name in CHEST_IMAGENOME_BBOX_NAMES:
    if name in CHEST_IMAGENOME_GOLD_BBOX_NAMES:
        CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED.append(name)
assert len(CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED) == len(CHEST_IMAGENOME_GOLD_BBOX_NAMES)

CHEST_IMAGENOME_BBOX_SYMMETRY_PAIRS = [
    ('right lung', 'left lung'),
    ('right upper lung zone', 'left upper lung zone'),
    ('right mid lung zone', 'left mid lung zone'),
    ('right lower lung zone', 'left lower lung zone'),
    ('right hilar structures', 'left hilar structures'),
    ('right apical zone', 'left apical zone'),
    ('right costophrenic angle', 'left costophrenic angle'),
    ('right cardiophrenic angle', 'left cardiophrenic angle'),
    ('right hemidiaphragm', 'left hemidiaphragm'),
    ('right clavicle', 'left clavicle'),
    ('right cardiac silhouette', 'left cardiac silhouette'),
    ('right upper abdomen', 'left upper abdomen'),
]

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