from dotenv import load_dotenv
load_dotenv()

from medvqa.utils.common import CACHE_DIR

import os

IUXRAY_DATASET_DIR = os.environ['IUXRAY_DATASET_DIR']
IUXRAY_IMAGE_INFO_JSON_PATH = os.path.join(IUXRAY_DATASET_DIR, 'info.json')
IUXRAY_REPORTS_JSON_PATH = os.path.join(IUXRAY_DATASET_DIR, 'reports/reports.json')
IUXRAY_REPORTS_MIN_JSON_PATH = os.path.join(IUXRAY_DATASET_DIR, 'reports/reports.min.json')
IUXRAY_CACHE_DIR = os.path.join(CACHE_DIR, 'iuxray')