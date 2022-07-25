from dotenv import load_dotenv
load_dotenv()

from medvqa.utils.common import CACHE_DIR

import os

MIMICCXR_DATASET_DIR = os.environ['MIMICCXR_DATASET_DIR']
MIMICCXR_DATASET_AUX_DIR = os.environ['MIMICCXR_DATASET_AUX_DIR']
MIMICCXR_JPG_IMAGES_SMALL_DIR = os.environ['MIMICCXR_JPG_IMAGES_SMALL_DIR']
MIMICCXR_JPG_DIR = os.environ['MIMICCXR_JPG_DIR']
MIMICCXR_METADATA_CSV_PATH = os.path.join(MIMICCXR_JPG_DIR, 'mimic-cxr-2.0.0-metadata.csv')
MIMICCXR_SPLIT_CSV_PATH = os.path.join(MIMICCXR_JPG_DIR, 'mimic-cxr-2.0.0-split.csv')
MIMICCXR_CACHE_DIR = os.path.join(CACHE_DIR, 'mimiccxr')
MIMICCXR_IMAGE_ORIENTATIONS = ['UNKNOWN', 'PA', 'AP']
