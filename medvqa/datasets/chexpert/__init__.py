from dotenv import load_dotenv
load_dotenv()

from medvqa.utils.common import CACHE_DIR
import os

CHEXPERT_DATASET_DIR = os.environ['CHEXPERT_DATASET_DIR']
CHEXPERT_DATASET_AUX_DIR = os.environ['CHEXPERT_DATASET_AUX_DIR']
CHEXPERT_CACHE_DIR = os.path.join(CACHE_DIR, 'chexpert')
CHEXPERT_TRAIN_VAL_CSV_PATH = os.path.join(CHEXPERT_DATASET_DIR, 'CheXpert-v1.0-small', 'train-val.csv')