from dotenv import load_dotenv

from medvqa.utils.files_utils import load_json
load_dotenv()

import os

from medvqa.utils.common import CACHE_DIR, LARGE_FAST_CACHE_DIR

PADCHEST_DATASET_DIR = os.environ['PADCHEST_DATASET_DIR']
PADCHEST_IMAGES_SMALL_DIR = os.environ['PADCHEST_IMAGES_SMALL_DIR']
PADCHEST_LABELS_CSV_PATH = os.path.join(PADCHEST_DATASET_DIR, 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
PADCHEST_CACHE_DIR = os.path.join(CACHE_DIR, 'padchest')
PADCHEST_LARGE_FAST_CACHE_DIR = os.path.join(LARGE_FAST_CACHE_DIR, 'padchest')
PADCHEST_BROKEN_IMAGES_TXT_PATH = os.path.join(PADCHEST_CACHE_DIR, 'broken_images.txt')
PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH = os.environ['PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH']
PADCHEST_GR_MASTER_TABLE_CSV_PATH = os.environ['PADCHEST_GR_MASTER_TABLE_CSV_PATH']
PADCHEST_GR_JPG_DIR = os.environ['PADCHEST_GR_JPG_DIR']


def get_padchest_gr_sentences_from_reports(language: str = 'en'):
    lang_key = f'sentence_{language}'
    reports_json_list = load_json(PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH)
    unique_sentences = set()
    for report_info in reports_json_list:
        findings = report_info['findings']
        for finding in findings:
            sentence = finding.get(lang_key, "")
            # Clean sentence
            sentence = sentence.strip()
            if sentence.endswith('.'):
                sentence = sentence[:-1]
            # Skip empty sentences
            if not sentence:
                continue
            # Add to set
            unique_sentences.add(sentence)
    # Sort the sentences
    unique_sentences = sorted(list(unique_sentences))
    return unique_sentences

            
def get_padchest_gr_labels():
    reports_json_list = load_json(PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH)
    unique_labels = set()
    for report_info in reports_json_list:
        findings = report_info['findings']
        for finding in findings:
            labels = finding.get('labels', [])
            unique_labels.update(labels)
    # Sort the labels
    unique_labels = sorted(list(unique_labels))
    return unique_labels
