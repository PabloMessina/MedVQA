from dotenv import load_dotenv
load_dotenv()

from medvqa.utils.common import CACHE_DIR, FAST_CACHE_DIR, FAST_TMP_DIR, LARGE_FAST_CACHE_DIR
from medvqa.utils.files_utils import get_cached_json_file

import os

IUXRAY_DATASET_DIR = os.environ['IUXRAY_DATASET_DIR']
IUXRAY_DATASET_AUX_DIR = os.environ['IUXRAY_DATASET_AUX_DIR']
IUXRAY_IMAGE_INFO_JSON_PATH = os.path.join(IUXRAY_DATASET_DIR, 'info.json')
IUXRAY_REPORTS_JSON_PATH = os.path.join(IUXRAY_DATASET_DIR, 'reports/reports.json')
IUXRAY_REPORTS_MIN_JSON_PATH = os.path.join(IUXRAY_DATASET_DIR, 'reports/reports.min.json')
IUXRAY_CACHE_DIR = os.path.join(CACHE_DIR, 'iuxray')
IUXRAY_LARGE_FAST_CACHE_DIR = os.path.join(LARGE_FAST_CACHE_DIR, 'iuxray')
IUXRAY_FAST_CACHE_DIR = os.path.join(FAST_CACHE_DIR, 'iuxray')
IUXRAY_FAST_TMP_DIR = os.path.join(FAST_TMP_DIR, 'iuxray')
IUXRAY_IMAGE_ORIENTATIONS = ['lateral-left', 'lateral-right', 'frontal']

_IUXRAY_IMAGE_PATH_TEMPLATE = os.path.join(IUXRAY_DATASET_DIR, 'images', '{}')

def get_invalid_images():
    iuxray_image_info = get_cached_json_file(IUXRAY_IMAGE_INFO_JSON_PATH)
    invalid_images = set()
    invalid_images.update(iuxray_image_info['marks']['wrong'])
    invalid_images.update(iuxray_image_info['marks']['broken'])
    return invalid_images

def get_iuxray_image_path(image_name):
    if image_name.endswith('.png'):
        return _IUXRAY_IMAGE_PATH_TEMPLATE.format(image_name)
    return _IUXRAY_IMAGE_PATH_TEMPLATE.format(f'{image_name}.png')

def get_iuxray_all_image_ids():
    reports = get_cached_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)
    image_ids = []
    for report in reports.values():
        image_ids.extend(image['id'] for image in report['images'])
    return image_ids

def load_reports_and_tag_sets():
    reports = get_cached_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)
    report_texts = []
    tag_sets = []
    for report in reports.values():
        tags_manual = report['tags_manual']
        tags_auto = report['tags_auto']
        tag_set = set()
        for tag in tags_manual + tags_auto:
            tag = tag.lower()
            if '/' in tag or ',' in tag:
                tag_parts = tag.replace(',', '/').split('/')
            else:
                tag_parts = [tag]
            for tag_part in tag_parts:
                tag_part = tag_part.strip()
                tag_set.add(tag_part)
                if ' ' in tag_part:
                    # add individual words
                    words = tag_part.split()
                    for word in words:
                        tag_set.add(word)
        assert len(tag_set) > 0
        tag_sets.append(tag_set)
        findings = report['findings']
        impression = report['impression']
        text = ""
        if findings:
            text += findings
        if impression:
            if text:
                if text[-1] != '.':
                    text += '.'
                text += ' '
            text += impression
        report_texts.append(text)
    return report_texts, tag_sets