from dotenv import load_dotenv

from medvqa.utils.files import get_cached_json_file
load_dotenv()

from medvqa.utils.common import CACHE_DIR

import os

IUXRAY_DATASET_DIR = os.environ['IUXRAY_DATASET_DIR']
IUXRAY_DATASET_AUX_DIR = os.environ['IUXRAY_DATASET_AUX_DIR']
IUXRAY_IMAGE_INFO_JSON_PATH = os.path.join(IUXRAY_DATASET_DIR, 'info.json')
IUXRAY_REPORTS_JSON_PATH = os.path.join(IUXRAY_DATASET_DIR, 'reports/reports.json')
IUXRAY_REPORTS_MIN_JSON_PATH = os.path.join(IUXRAY_DATASET_DIR, 'reports/reports.min.json')
IUXRAY_CACHE_DIR = os.path.join(CACHE_DIR, 'iuxray')
IUXRAY_IMAGE_ORIENTATIONS = ['lateral-left', 'lateral-right', 'frontal']


def get_invalid_images():
    iuxray_image_info = get_cached_json_file(IUXRAY_IMAGE_INFO_JSON_PATH)
    invalid_images = set()
    invalid_images.update(iuxray_image_info['marks']['wrong'])
    invalid_images.update(iuxray_image_info['marks']['broken'])
    return invalid_images