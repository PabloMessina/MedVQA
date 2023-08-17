from dotenv import load_dotenv
load_dotenv()

import os
from medvqa.utils.common import CACHE_DIR, FAST_CACHE_DIR

CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR = os.environ['CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR']
CHEST_IMAGENOME_SILVER_DATASET_DIR = os.environ['CHEST_IMAGENOME_SILVER_DATASET_DIR']
CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH = os.environ['CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH']
CHEST_IMAGENOME_GOLD_BBOX_COORDINATE_ANNOTATIONS_CSV_PATH = os.environ['CHEST_IMAGENOME_GOLD_BBOX_COORDINATE_ANNOTATIONS_CSV_PATH']
CHEST_IMAGENOME_GOLD_ATTRIBUTE_RELATIONS_TXT_PATH = os.environ['CHEST_IMAGENOME_GOLD_ATTRIBUTE_RELATIONS_TXT_PATH']
CHEST_IMAGENOME_CACHE_DIR = os.path.join(CACHE_DIR, 'chest_imagenome')
CHEST_IMAGENOME_FAST_CACHE_DIR = os.path.join(FAST_CACHE_DIR, 'chest_imagenome')
CHEST_IMAGENOME_SILVER_BBOXES_FILEPATH = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'bboxes.pkl')
CHEST_IMAGENOME_HORIZONTALLY_FLIPPED_SILVER_BBOXES_FILEPATH = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'horizontally_flipped_bboxes.pkl')
CHEST_IMAGENOME_YOLOV5_LABELS_DIR = os.environ['CHEST_IMAGENOME_YOLOV5_LABELS_DIR']

CHEST_IMAGENOME_NUM_BBOX_CLASSES = 36
CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES = 26
CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES = 18

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

# print('CHEST_IMAGENOME_BBOX_NAMES:', CHEST_IMAGENOME_BBOX_NAMES)
# print('len(CHEST_IMAGENOME_BBOX_NAMES):', len(CHEST_IMAGENOME_BBOX_NAMES))
# print('CHEST_IMAGENOME_NUM_BBOX_CLASSES:', CHEST_IMAGENOME_NUM_BBOX_CLASSES)

CHEST_IMAGENOME_OTHER_REGIONS = [ # these do not come with bounding boxes
    'neck',
    'left chest wall',
    'right chest wall',
    'left shoulder',
    'right shoulder',
    'left arm',
    'right arm',
    'left breast',
    'right breast',
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
    'right atrium': 'RAT',
    'descending aorta': 'DA',
    'carina': 'C',
    'left upper abdomen': 'LUA',
    'right upper abdomen': 'RUA',
    'abdomen': 'A',
    'left cardiophrenic angle': 'LCAA',
    'neck': 'N',
    'left chest wall': 'LCW',
    'right chest wall': 'RCW',
    'left shoulder': 'LS',
    'right shoulder': 'RS',
    'left arm': 'LA',
    'right arm': 'RAR',
    'left breast': 'LB',
    'right breast': 'RB',
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

# from AnaXNET paper: https://arxiv.org/pdf/2105.09937.pdf
ANAXNET_BBOX_NAMES = [
    'right lung',
    'right apical zone',
    'right upper lung zone',
    'right mid lung zone',
    'right lower lung zone',
    'right hilar structures',
    'right costophrenic angle',
    'left lung',
    'left apical zone',
    'left upper lung zone',
    'left mid lung zone',
    'left lower lung zone',
    'left hilar structures',
    'left costophrenic angle',
    'mediastinum',
    'upper mediastinum',
    'cardiac silhouette',
    'trachea',
]
assert len(set(ANAXNET_BBOX_NAMES)) == len(ANAXNET_BBOX_NAMES) # no duplicates
assert all(name in CHEST_IMAGENOME_BBOX_NAMES for name in ANAXNET_BBOX_NAMES) # all names are in the original list
assert len(ANAXNET_BBOX_NAMES) == CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES # same number of classes

CHEST_IMAGENOME_BBOX_NAMES_WITH_TEXTUAL_GROUNDING = [
    'abdomen',
    'aortic arch',
    'cardiac silhouette',
    'carina',
    'cavoatrial junction',
    'left apical zone',
    'left arm',
    'left breast',
    'left chest wall',
    'left clavicle',
    'left costophrenic angle',
    'left hemidiaphragm',
    'left hilar structures',
    'left lower lung zone',
    'left lung',
    'left mid lung zone',
    'left shoulder',
    'left upper lung zone',
    'mediastinum',
    'neck',
    'right apical zone',
    'right arm',
    'right atrium',
    'right breast',
    'right chest wall',
    'right clavicle',
    'right costophrenic angle',
    'right hemidiaphragm',
    'right hilar structures',
    'right lower lung zone',
    'right lung',
    'right mid lung zone',
    'right shoulder',
    'right upper lung zone',
    'spine',
    'svc',
    'trachea',
    'upper mediastinum',
]

def get_anaxnet_bbox_sorted_indices():
    indices = [i for i, name in enumerate(CHEST_IMAGENOME_BBOX_NAMES) if name in ANAXNET_BBOX_NAMES]
    return indices

def get_anaxnet_bbox_coords_and_presence_sorted_indices(check_intersection_with_gold=False, for_model_output=False):
    coords_indices = []
    presence_indices = []
    if for_model_output:
        for i, idx in enumerate(get_anaxnet_bbox_sorted_indices()):
            if check_intersection_with_gold:
                bbox_name = CHEST_IMAGENOME_BBOX_NAMES[idx]
                assert bbox_name in ANAXNET_BBOX_NAMES
                if bbox_name not in CHEST_IMAGENOME_GOLD_BBOX_NAMES:
                    continue
            presence_indices.append(i)
            for j in range(4):
                coords_indices.append(i*4 + j)
    else:
        for i, name in enumerate(CHEST_IMAGENOME_BBOX_NAMES):
            if check_intersection_with_gold:
                if name not in CHEST_IMAGENOME_GOLD_BBOX_NAMES:
                    continue
            if name in ANAXNET_BBOX_NAMES:            
                presence_indices.append(i)
                for j in range(4):
                    coords_indices.append(i*4 + j)
    return coords_indices, presence_indices

def get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices():
    coords_indices = []
    presence_indices = []
    for i, name in enumerate(CHEST_IMAGENOME_BBOX_NAMES):
        if name in CHEST_IMAGENOME_GOLD_BBOX_NAMES:            
            presence_indices.append(i)
            for j in range(4):
                coords_indices.append(i*4 + j)
    return coords_indices, presence_indices

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

assert len(CHEST_IMAGENOME_BBOX_NAMES) == CHEST_IMAGENOME_NUM_BBOX_CLASSES