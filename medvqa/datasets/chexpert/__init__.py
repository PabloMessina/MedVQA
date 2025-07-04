from dotenv import load_dotenv
load_dotenv()

from medvqa.utils.common import CACHE_DIR, FAST_TMP_DIR, LARGE_FAST_CACHE_DIR, FAST_CACHE_DIR
import os

CHEXPERT_DATASET_DIR = os.environ['CHEXPERT_DATASET_DIR']
CHEXPERT_V1_0_SMALL_DATASET_DIR = os.environ['CHEXPERT_V1.0_SMALL_DATASET_DIR']
CHEXPERT_DATASET_AUX_DIR = os.environ['CHEXPERT_DATASET_AUX_DIR']
CHEXPERT_CACHE_DIR = os.path.join(CACHE_DIR, 'chexpert')
CHEXPERT_FAST_CACHE_DIR = os.path.join(FAST_CACHE_DIR, 'chexpert')
CHEXPERT_FAST_TMP_DIR = os.path.join(FAST_TMP_DIR, 'chexpert')
CHEXPERT_LARGE_FAST_CACHE_DIR = os.path.join(LARGE_FAST_CACHE_DIR, 'chexpert')
CHEXPERT_TRAIN_CSV_PATH = os.path.join(CHEXPERT_DATASET_DIR, 'CheXpert-v1.0-small', 'train.csv')
CHEXPERT_VALID_CSV_PATH = os.path.join(CHEXPERT_DATASET_DIR, 'CheXpert-v1.0-small', 'valid.csv')
CHEXPERT_TRAIN_VISUALCHEXBERT_CSV_PATH = os.path.join(CHEXPERT_DATASET_DIR, 'CheXpert-v1.0-small', 'train_visualCheXbert.csv')
CHEXPERT_TRAIN_VAL_CSV_PATH = os.path.join(CHEXPERT_DATASET_DIR, 'CheXpert-v1.0-small', 'train-val.csv')
CHEXPERT_TEST_LABELS_CSV_PATH = os.path.join(CHEXPERT_DATASET_DIR, 'CheXpert-v1.0-small', 'test_labels.csv')
CHEXPERT_PLUS_CSV_PATH = os.environ['CHEXPERT_PLUS_CSV_PATH']