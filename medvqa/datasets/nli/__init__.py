from dotenv import load_dotenv
load_dotenv()

import os

MEDNLI_TRAIN_JSONL_PATH = os.environ['MEDNLI_TRAIN_JSONL_PATH']
MEDNLI_DEV_JSONL_PATH = os.environ['MEDNLI_DEV_JSONL_PATH']
MEDNLI_TEST_JSONL_PATH = os.environ['MEDNLI_TEST_JSONL_PATH']
RADNLI_DEV_JSONL_PATH = os.environ['RADNLI_DEV_JSONL_PATH']
RADNLI_TEST_JSONL_PATH = os.environ['RADNLI_TEST_JSONL_PATH']
MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH = os.environ['MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH']
ANLI_V1_DATASET_DIR = os.environ['ANLI_V1_DATASET_DIR']
MULTI_NLI_DATASET_DIR = os.environ['MULTI_NLI_DATASET_DIR']
SNLI_DATASET_DIR = os.environ['SNLI_DATASET_DIR']