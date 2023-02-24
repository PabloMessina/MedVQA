from dotenv import load_dotenv
from pathlib import Path
from medvqa.utils.files import get_cached_json_file, get_cached_pickle_file, load_pickle, save_to_pickle
load_dotenv()

from medvqa.utils.common import CACHE_DIR

import os
import re
import glob
import pandas as pd
from tqdm import tqdm

MIMICCXR_DATASET_DIR = os.environ['MIMICCXR_DATASET_DIR']
MIMICCXR_DATASET_AUX_DIR = os.environ['MIMICCXR_DATASET_AUX_DIR']
MIMICCXR_JPG_IMAGES_SMALL_DIR = os.environ['MIMICCXR_JPG_IMAGES_SMALL_DIR']
MIMICCXR_JPG_IMAGES_MEDIUM_DIR = os.environ['MIMICCXR_JPG_IMAGES_MEDIUM_DIR']
MIMICCXR_JPG_IMAGES_LARGE_DIR = os.environ['MIMICCXR_JPG_IMAGES_LARGE_DIR']
MIMICCXR_JPG_DIR = os.environ['MIMICCXR_JPG_DIR']
MIMICCXR_METADATA_CSV_PATH = os.path.join(MIMICCXR_JPG_DIR, 'mimic-cxr-2.0.0-metadata.csv')
MIMICCXR_SPLIT_CSV_PATH = os.path.join(MIMICCXR_JPG_DIR, 'mimic-cxr-2.0.0-split.csv')
MIMICCXR_CACHE_DIR = os.path.join(CACHE_DIR, 'mimiccxr')
MIMICCXR_REPORTS_TXT_PATHS = os.path.join(MIMICCXR_CACHE_DIR, 'reports_txt_paths.pkl')
MIMICCXR_IMAGE_ORIENTATIONS = ['UNKNOWN', 'PA', 'AP']

MIMICCXR_IMAGE_SMALL_PATH_TEMPLATE = os.path.join(MIMICCXR_JPG_IMAGES_SMALL_DIR, 'p{}', 'p{}', 's{}', '{}.jpg')
MIMICCXR_IMAGE_MEDIUM_PATH_TEMPLATE = os.path.join(MIMICCXR_JPG_IMAGES_MEDIUM_DIR, 'p{}', 'p{}', 's{}', '{}.jpg')
MIMICCXR_IMAGE_LARGE_PATH_TEMPLATE = os.path.join(MIMICCXR_JPG_IMAGES_LARGE_DIR, 'p{}', 'p{}', 's{}', '{}.jpg')
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

class MIMICCXR_ImageSizeModes:
    SMALL_256x256 = 'small_256x256'
    MEDIUM_512 = 'medium_512'

def get_mimiccxr_small_image_path(part_id, subject_id, study_id, dicom_id):
    return MIMICCXR_IMAGE_SMALL_PATH_TEMPLATE.format(part_id, subject_id, study_id, dicom_id)

def get_mimiccxr_medium_image_path(part_id, subject_id, study_id, dicom_id):
    return MIMICCXR_IMAGE_MEDIUM_PATH_TEMPLATE.format(part_id, subject_id, study_id, dicom_id)

def get_mimiccxr_large_image_path(part_id, subject_id, study_id, dicom_id):
    return MIMICCXR_IMAGE_LARGE_PATH_TEMPLATE.format(part_id, subject_id, study_id, dicom_id)

def get_mimiccxr_report_path(part_id, subject_id, study_id):    
    return os.path.join(MIMICCXR_DATASET_DIR, 'files', 'p{}'.format(part_id), 'p{}'.format(subject_id), 's{}.txt'.format(study_id))

class MIMICCXR_ViewModes:
    ALL = 'all'
    FRONT_ALL = 'front_all'
    FRONT_SINGLE = 'front_single'
    ANY_SINGLE = 'any_single'
    CHEST_IMAGENOME = 'chest_imagenome'

class MIMICCXR_EvalViewModes:
    ALL = 'all'
    FRONT_ALL = 'front_all'
    FRONT_SINGLE = 'front_single'
    ANY_SINGLE = 'any_single'

def get_dicom_id_and_orientation_list(views, view_mode=MIMICCXR_ViewModes.ANY_SINGLE, chest_imagenome_dicom_ids=None):
    output = []
    if view_mode == MIMICCXR_ViewModes.ALL:
        for view in views:
            output.append((view[0], view[1]))
    elif view_mode == MIMICCXR_ViewModes.FRONT_ALL:
        for view in views:
            if view[1] == 'PA' or view[1] == 'AP':
                output.append((view[0], view[1]))
    elif view_mode == MIMICCXR_ViewModes.FRONT_SINGLE:
        dicom_id = None
        for view in views:
            if view[1] == 'PA':
                dicom_id = view[0]
                orientation = view[1]
                break
        if dicom_id is None:
            for view in views:
                if view[1] == 'AP':
                    dicom_id = view[0]
                    orientation = view[1]
                    break
        if dicom_id is not None:
            output.append((dicom_id, orientation))
    elif view_mode == MIMICCXR_ViewModes.ANY_SINGLE:
        dicom_id = None
        for view in views:
            if view[1] == 'PA':
                dicom_id = view[0]
                orientation = view[1]
                break
        if dicom_id is None:
            for view in views:
                if view[1] == 'AP':
                    dicom_id = view[0]
                    orientation = view[1]
                    break
        if dicom_id is None and len(views) > 0:
            dicom_id = views[0][0]
            orientation = views[0][1]
        if dicom_id is not None:
            output.append((dicom_id, orientation))
    elif view_mode == MIMICCXR_ViewModes.CHEST_IMAGENOME:
        assert chest_imagenome_dicom_ids is not None
        for view in views:
            if view[0] in chest_imagenome_dicom_ids:
                output.append((view[0], view[1]))
    else:
        raise ValueError('Unknown view mode: {}'.format(view_mode))
    return output

def get_image_views_dict():
    mimiccxr_metadata = pd.read_csv(MIMICCXR_METADATA_CSV_PATH)
    image_views_dict = dict()
    for subject_id, study_id, dicom_id, view_pos in zip(mimiccxr_metadata['subject_id'],
                                                        mimiccxr_metadata['study_id'],
                                                        mimiccxr_metadata['dicom_id'],
                                                        mimiccxr_metadata['ViewPosition']):
        key = (subject_id, study_id)
        try:
            views = image_views_dict[key]
        except KeyError:
            views = image_views_dict[key] = []
        views.append((dicom_id, view_pos))
    return image_views_dict

def get_broken_images():
    broken_images = set()
    for path in MIMICCXR_BROKEN_IMAGES:
        _, a, b, c = MIMICCXR_IMAGE_REGEX.findall(path)[0]
        broken_images.add((int(a), int(b), c))
    return broken_images

def get_split_dict():
    mimiccxr_split = pd.read_csv(MIMICCXR_SPLIT_CSV_PATH)        
    split_dict = { (sub_id, stud_id, dicom_id) : split for sub_id, stud_id, dicom_id, split in zip(mimiccxr_split['subject_id'],
                                                                                                    mimiccxr_split['study_id'],
                                                                                                    mimiccxr_split['dicom_id'],
                                                                                                    mimiccxr_split['split']) }
    return split_dict

def get_mimiccxr_image_paths(report):
    filepath = report['filepath']
    part_id, subject_id, study_id = MIMICCXR_STUDY_REGEX.findall(filepath)[0]
    images = glob.glob(MIMICCXR_IMAGE_SMALL_PATH_TEMPLATE.format(part_id, subject_id, study_id, '*'))
    return images

def get_reports_txt_paths():
    # if cached
    report_paths = load_pickle(MIMICCXR_REPORTS_TXT_PATHS)
    if report_paths is not None:
        return report_paths
    # if not cached
    report_paths = [None] * 300000
    for i, rp in tqdm(enumerate(report_paths_generator())):
        report_paths[i] = rp.as_posix()
    report_paths = report_paths[:i+1]
    save_to_pickle(report_paths, MIMICCXR_REPORTS_TXT_PATHS)
    print('reports txt paths saved to', MIMICCXR_REPORTS_TXT_PATHS)
    return report_paths

def report_paths_generator():
    for x in range(10, 20):
        for filepath in Path(os.path.join(MIMICCXR_DATASET_DIR, f'files/p{x}/')).rglob("s*.txt"):
            yield filepath

def image_paths_generator():
    for x in range(10, 20):
        for filepath in Path(os.path.join(MIMICCXR_JPG_IMAGES_SMALL_DIR, f'p{x}/')).rglob("*.jpg"):
            yield filepath

def get_imageId2partId():
    cache_path = os.path.join(MIMICCXR_CACHE_DIR, 'imageId2partId.pkl')
    if os.path.exists(cache_path):
        return get_cached_pickle_file(cache_path)
    imageId2partId = {}
    for image_path in tqdm(image_paths_generator()):
        image_path = str(image_path)
        partId, _, _, imageId = MIMICCXR_IMAGE_REGEX.findall(image_path)[0]
        imageId2partId[imageId] = partId
    save_to_pickle(imageId2partId, cache_path)
    return imageId2partId

def get_imageId2PartPatientStudy():
    cache_path = os.path.join(MIMICCXR_CACHE_DIR, 'imageId2PartPatientStudy.pkl')
    if os.path.exists(cache_path):
        return get_cached_pickle_file(cache_path)
    imageId2partId = {}
    for image_path in tqdm(image_paths_generator()):
        image_path = str(image_path)
        partId, patientId, studyId, imageId = MIMICCXR_IMAGE_REGEX.findall(image_path)[0]
        imageId2partId[imageId] = (partId, patientId, studyId)
    save_to_pickle(imageId2partId, cache_path)
    return imageId2partId

def load_mimiccxr_reports_detailed_metadata(qa_adapted_reports_filename=None):

    if qa_adapted_reports_filename is None:
        filename = 'detailed_metadata.pkl'
    else:
        filename = f'{qa_adapted_reports_filename}__detailed_metadata.pkl'
    cache_path = os.path.join(MIMICCXR_CACHE_DIR, filename)
    if os.path.exists(cache_path):
        print(f'Loading cached detailed metadata from {cache_path}')
        return get_cached_pickle_file(cache_path)

    if qa_adapted_reports_filename is not None:
        qa_adapted_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))
        n_reports = len(qa_adapted_reports['reports'])
    else:
        report_paths = get_reports_txt_paths()
        n_reports = len(report_paths)
    
    image_views_dict = get_image_views_dict()
    split_dict = get_split_dict()        
    part_ids = [None] * n_reports
    subject_ids = [None] * n_reports
    study_ids = [None] * n_reports
    dicom_id_view_pos_pairs = [None] * n_reports
    splits = [None] * n_reports
    filepaths = [None] * n_reports
    
    report_metadata = dict(
        part_ids=part_ids,
        subject_ids=subject_ids,
        study_ids=study_ids,
        dicom_id_view_pos_pairs=dicom_id_view_pos_pairs,
        splits=splits,
        filepaths=filepaths,
    )
    
    if qa_adapted_reports_filename is not None:
        backgrounds = [None] * n_reports
        reports = [None] * n_reports
        report_metadata['backgrounds'] = backgrounds
        report_metadata['reports'] = reports
        for i, report in tqdm(enumerate(qa_adapted_reports['reports'])):
            filepath = report['filepath']
            part_id, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(filepath)[0])            
            backgrounds[i] = report['background']
            reports[i] = '.\n '.join(report['sentences'])
            part_ids[i] = part_id
            subject_ids[i] = subject_id
            study_ids[i] = study_id
            dicom_id_view_pos_pairs[i] = image_views_dict[(subject_id, study_id)]
            splits[i] = split_dict[(subject_id, study_id, dicom_id_view_pos_pairs[i][0][0])]
            for j in range(1, len(dicom_id_view_pos_pairs[i])):
                assert split_dict[(subject_id, study_id, dicom_id_view_pos_pairs[i][j][0])] == splits[i]
            filepaths[i] = filepath
    else:
        for i, report_path in tqdm(enumerate(report_paths)):
            report_path = str(report_path)
            part_id, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(report_path)[0])
            part_ids[i] = part_id
            subject_ids[i] = subject_id
            study_ids[i] = study_id
            dicom_id_view_pos_pairs[i] = image_views_dict[(subject_id, study_id)]
            splits[i] = split_dict[(subject_id, study_id, dicom_id_view_pos_pairs[i][0][0])]
            for j in range(1, len(dicom_id_view_pos_pairs[i])):
                assert split_dict[(subject_id, study_id, dicom_id_view_pos_pairs[i][j][0])] == splits[i]
            filepaths[i] = report_path

    save_to_pickle(report_metadata, cache_path)
    print(f'Saved detailed metadata to {cache_path}')
    return report_metadata

def get_detailed_metadata_for_dicom_id(dicom_id, qa_adapted_reports_filename):
    detailed_metadata = load_mimiccxr_reports_detailed_metadata(qa_adapted_reports_filename)    
    output = []
    for i in range(len(detailed_metadata['dicom_id_view_pos_pairs'])):
        for dicom_id_view_pos_pair in detailed_metadata['dicom_id_view_pos_pairs'][i]:
            if dicom_id_view_pos_pair[0] == dicom_id:
                output.append({
                    'background': detailed_metadata['backgrounds'][i],
                    'report': detailed_metadata['reports'][i],
                    'part_id': detailed_metadata['part_ids'][i],
                    'subject_id': detailed_metadata['subject_ids'][i],
                    'study_id': detailed_metadata['study_ids'][i],
                    'dicom_id': dicom_id,
                    'view_pos': dicom_id_view_pos_pair[1],
                    'split': detailed_metadata['splits'][i],                    
                    'filepath': detailed_metadata['filepaths'][i],
                    'dicom_id_view_pos_pairs': detailed_metadata['dicom_id_view_pos_pairs'][i],
                })
    return output

def get_mimimiccxr_test_dicom_ids():
    return [x[0][2] for x in get_split_dict().items() if x[1] == 'test']