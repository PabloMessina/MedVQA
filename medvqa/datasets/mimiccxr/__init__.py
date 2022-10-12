from dotenv import load_dotenv
load_dotenv()

from medvqa.utils.common import CACHE_DIR

import os
import re

MIMICCXR_DATASET_DIR = os.environ['MIMICCXR_DATASET_DIR']
MIMICCXR_DATASET_AUX_DIR = os.environ['MIMICCXR_DATASET_AUX_DIR']
MIMICCXR_JPG_IMAGES_SMALL_DIR = os.environ['MIMICCXR_JPG_IMAGES_SMALL_DIR']
MIMICCXR_JPG_DIR = os.environ['MIMICCXR_JPG_DIR']
MIMICCXR_METADATA_CSV_PATH = os.path.join(MIMICCXR_JPG_DIR, 'mimic-cxr-2.0.0-metadata.csv')
MIMICCXR_SPLIT_CSV_PATH = os.path.join(MIMICCXR_JPG_DIR, 'mimic-cxr-2.0.0-split.csv')
MIMICCXR_CACHE_DIR = os.path.join(CACHE_DIR, 'mimiccxr')
MIMICCXR_REPORTS_TXT_PATHS = os.path.join(MIMICCXR_CACHE_DIR, 'reports_txt_paths.pkl')
MIMICCXR_IMAGE_ORIENTATIONS = ['UNKNOWN', 'PA', 'AP']

MIMICCXR_IMAGE_PATH_TEMPLATE = os.path.join(MIMICCXR_JPG_IMAGES_SMALL_DIR, 'p{}', 'p{}', 's{}', '{}.jpg')
MIMICCXR_STUDY_REGEX = re.compile(r'/p(\d+)/p(\d+)/s(\d+)\.txt')
MIMICCXR_IMAGE_REGEX = re.compile(r'p(\d+)/p(\d+)/s(\d+)/(.*)\.jpg$')
MIMICCXR_BROKEN_IMAGES = set([
    'p11/p11285576/s54979966/03b2e67c-70631ff8-685825fb-6c989456-621ca64d.jpg',
    'p15/p15223781/s52459604/56b8afd3-5f6d4419-8699d79e-6913a2bd-35a08557.jpg',
    'p15/p15223781/s52459604/93020995-6b84ca33-2e41e00d-5d6e3bee-87cfe5c6.jpg',
    # Appears empty
    'p10/p10291098/s57194260/0539ee33-9d402e49-a9cc6d36-7aabc539-3d80a62b.jpg',
    # Blur empty images
    'p15/p15355458/s52423703/0b6f08b2-72deda00-d7ccc375-8278269f-b4e11c36.jpg',
    'p18/p18461911/s57183218/151abebe-2a750a5c-09c181bb-1a9016ef-92d8910e.jpg',
    'p19/p19839145/s54889255/f674e474-817bb713-8f16c90c-608cf869-2829cae7.jpg',
])

def get_mimiccxr_image_path(part_id, subject_id, study_id, dicom_id):
    return MIMICCXR_IMAGE_PATH_TEMPLATE.format(part_id, subject_id, study_id, dicom_id)
