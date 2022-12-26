from dotenv import load_dotenv
load_dotenv()

import os

from medvqa.utils.common import CACHE_DIR

PADCHEST_DATASET_DIR = os.environ['PADCHEST_DATASET_DIR']
PADCHEST_IMAGES_SMALL_DIR = os.environ['PADCHEST_IMAGES_SMALL_DIR']
PADCHEST_LABELS_CSV_PATH = os.path.join(PADCHEST_DATASET_DIR, 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
PADCHEST_CACHE_DIR = os.path.join(CACHE_DIR, 'padchest')
PADCHEST_BROKEN_IMAGES_TXT_PATH = os.path.join(PADCHEST_CACHE_DIR, 'broken_images.txt')