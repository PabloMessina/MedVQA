from dotenv import load_dotenv
from pathlib import Path
from medvqa.utils.constants import CHEXPERT_LABELS
from medvqa.utils.files import get_cached_json_file, get_cached_pickle_file, load_pickle, save_pickle
load_dotenv()

from medvqa.utils.common import CACHE_DIR, FAST_CACHE_DIR, LARGE_FAST_CACHE_DIR, FAST_TMP_DIR

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
MIMICCXR_FAST_CACHE_DIR = os.path.join(FAST_CACHE_DIR, 'mimiccxr')
MIMICCXR_LARGE_FAST_CACHE_DIR = os.path.join(LARGE_FAST_CACHE_DIR, 'mimiccxr')
MIMICCXR_FAST_TMP_DIR = os.path.join(FAST_TMP_DIR, 'mimiccxr')
MIMICCXR_REPORTS_TXT_PATHS = os.path.join(MIMICCXR_CACHE_DIR, 'reports_txt_paths.pkl')
MIMICCXR_CUSTOM_RADIOLOGIST_ANNOTATIONS_CSV_1_PATH = os.environ['MIMICCXR_CUSTOM_RADIOLOGIST_ANNOTATIONS_CSV_1_PATH']
MIMICCXR_CUSTOM_RADIOLOGIST_ANNOTATIONS_CSV_2_PATH = os.environ['MIMICCXR_CUSTOM_RADIOLOGIST_ANNOTATIONS_CSV_2_PATH']

MIMICCXR_IMAGE_ORIENTATIONS__RAW = ['PA', 'LATERAL', 'LL', 'AP', 'UNKNOWN', 'LAO', 'RAO', 
                                    'AP LLD', 'AP AXIAL', 'SWIMMERS', 'PA LLD', 'XTABLE LATERAL',
                                    'PA RLD', 'AP RLD', 'LPO']
MIMICCXR_IMAGE_ORIENTATIONS = ['UNKNOWN', 'PA', 'AP', 'LATERAL']

def get_mimiccxr_image_orientation_id(o):
    if type(o) != str:
        o = 'UNKNOWN'
    assert o in MIMICCXR_IMAGE_ORIENTATIONS__RAW
    if o == 'LL':
        o = 'LATERAL'
    try:
        return MIMICCXR_IMAGE_ORIENTATIONS.index(o)    
    except ValueError:
        return 0

MIMICCXR_SPLIT_NAMES = ['train', 'validate', 'test']

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

_cache = {}

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

def get_image_path_getter(image_size_mode, verbose=False):
    if image_size_mode == MIMICCXR_ImageSizeModes.SMALL_256x256:
        image_path_getter = get_mimiccxr_small_image_path
    elif image_size_mode == MIMICCXR_ImageSizeModes.MEDIUM_512:
        image_path_getter = get_mimiccxr_medium_image_path
    else:
        raise ValueError(f'Unknown source image size mode: {image_size_mode}')
    if verbose:
        print(f'Using image size mode: {image_size_mode}')
    return image_path_getter

class MIMICCXR_ViewModes:
    ALL = 'all'
    FRONT_ALL = 'front_all'
    FRONT_SINGLE = 'front_single'
    ANY_SINGLE = 'any_single'
    CHEST_IMAGENOME = 'chest_imagenome'

    @staticmethod
    def get_all_modes():
        return [
            MIMICCXR_ViewModes.ALL,
            MIMICCXR_ViewModes.FRONT_ALL,
            MIMICCXR_ViewModes.FRONT_SINGLE,
            MIMICCXR_ViewModes.ANY_SINGLE,
            MIMICCXR_ViewModes.CHEST_IMAGENOME,
        ]

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
    if 'get_split_dict()' in _cache:
        return _cache['get_split_dict()']
    mimiccxr_split = pd.read_csv(MIMICCXR_SPLIT_CSV_PATH)        
    split_dict = { (sub_id, stud_id, dicom_id) : split for sub_id, stud_id, dicom_id, split in zip(mimiccxr_split['subject_id'],
                                                                                                    mimiccxr_split['study_id'],
                                                                                                    mimiccxr_split['dicom_id'],
                                                                                                    mimiccxr_split['split']) }
    _cache['get_split_dict()'] = split_dict
    return split_dict

def get_mimiccxr_image_paths(report=None, filepath=None, image_size='small'):
    assert report is not None or filepath is not None
    assert image_size in ['small', 'medium', 'large']
    if filepath is None:
        filepath = report['filepath']
    part_id, subject_id, study_id = MIMICCXR_STUDY_REGEX.findall(filepath)[0]
    if image_size == 'small':
        template = MIMICCXR_IMAGE_SMALL_PATH_TEMPLATE
    elif image_size == 'medium':
        template = MIMICCXR_IMAGE_MEDIUM_PATH_TEMPLATE
    elif image_size == 'large':
        template = MIMICCXR_IMAGE_LARGE_PATH_TEMPLATE
    images = glob.glob(template.format(part_id, subject_id, study_id, '*'))
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
    save_pickle(report_paths, MIMICCXR_REPORTS_TXT_PATHS)
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
    save_pickle(imageId2partId, cache_path)
    return imageId2partId

def get_imageId2reportId():
    cache_path = os.path.join(MIMICCXR_CACHE_DIR, 'imageId2reportId.pkl')
    if os.path.exists(cache_path):
        return get_cached_pickle_file(cache_path)    
    metadata = load_mimiccxr_reports_detailed_metadata()
    imageId2reportId = {}
    for rid, dicom_id_view_pos_pairs in enumerate(metadata['dicom_id_view_pos_pairs']):
        for dicom_id, _ in dicom_id_view_pos_pairs:
            imageId2reportId[dicom_id] = rid
    save_pickle(imageId2reportId, cache_path)
    return imageId2reportId

def get_imageId2PartPatientStudy():
    cache_path = os.path.join(MIMICCXR_CACHE_DIR, 'imageId2PartPatientStudy.pkl')
    if os.path.exists(cache_path):
        return get_cached_pickle_file(cache_path)
    imageId2partpatstud = {}
    for image_path in tqdm(image_paths_generator()):
        image_path = str(image_path)
        partId, patientId, studyId, imageId = MIMICCXR_IMAGE_REGEX.findall(image_path)[0]
        imageId2partpatstud[imageId] = (partId, patientId, studyId)
    save_pickle(imageId2partpatstud, cache_path)
    return imageId2partpatstud

def visualize_image_report_and_other_images(dicom_id, figsize=(8, 8)):
    metadata = get_detailed_metadata_for_dicom_id(dicom_id)
    assert len(metadata) == 1
    metadata = metadata[0]
    part_id = metadata['part_id']
    patient_id = metadata['subject_id']
    study_id = metadata['study_id']
    image_path = get_mimiccxr_large_image_path(part_id, patient_id, study_id, dicom_id)
    # Plot main image
    from PIL import Image
    from matplotlib import pyplot as plt
    image = Image.open(image_path)
    image = image.convert('RGB')
    fig, ax = plt.subplots(1, figsize=figsize)
    print(f'dicom_id = {dicom_id}')
    print(f'image_path = {image_path}')
    # set title
    view_pos = metadata['view_pos']
    title = f'View position: {view_pos}, dicom_id: {dicom_id}'
    ax.set_title(title)
    ax.imshow(image)
    plt.show()
    # Print original report
    print()
    print('-' * 80)
    print('Original report:')
    report_path = get_mimiccxr_report_path(part_id=part_id, subject_id=patient_id, study_id=study_id)
    with open(report_path, 'r') as f:
        print(f.read())
    # Find and plot other images
    print()
    print('-' * 80)
    if len(metadata['dicom_id_view_pos_pairs']) == 1:
        print('No other images')
        return
    print('Other images:')
    for _dicom_id, _view_pos in metadata['dicom_id_view_pos_pairs']:
        if _dicom_id == dicom_id:
            continue
        _image_path = get_mimiccxr_large_image_path(part_id, patient_id, study_id, _dicom_id)
        _image = Image.open(_image_path)
        _image = _image.convert('RGB')
        fig, ax = plt.subplots(1, figsize=figsize)
        print(f'dicom_id = {_dicom_id}')
        print(f'image_path = {_image_path}')
        # set title
        title = f'View position: {_view_pos}, dicom_id: {_dicom_id}'
        ax.set_title(title)
        ax.imshow(_image)
        plt.show()

def save_report_image_and_other_images_as_pdf(dicom_id, pdf_path):
    import pdfkit
    metadata = get_detailed_metadata_for_dicom_id(dicom_id)
    assert len(metadata) == 1
    metadata = metadata[0]
    part_id = metadata['part_id']
    patient_id = metadata['subject_id']
    study_id = metadata['study_id']
    image_path = get_mimiccxr_large_image_path(part_id, patient_id, study_id, dicom_id)
    # Generate html
    html = f'''
    <html>
    <head>
    <style>
    .image {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 85%;
    }}
    </style>
    </head>
    <body>
    '''
    # Add main image
    html += f'''
    <h2>View position: {metadata['view_pos']}, dicom_id: {dicom_id}</h2>
    <img src="{image_path}" class="image">
    '''
    # Add original report
    html += f'''
    <h2>Original report</h2>
    '''
    report_path = get_mimiccxr_report_path(part_id=part_id, subject_id=patient_id, study_id=study_id)
    with open(report_path, 'r') as f:
        report = f.read()
    # we will use white-space: pre-wrap; to preserve newlines
    html += f'''
    <pre style="white-space: pre-wrap;">{report}</pre>
    '''
    # Add other images
    html += f'''
    <h2>Other images</h2>
    '''
    if len(metadata['dicom_id_view_pos_pairs']) == 1:
        html += f'''
        <p>No other images</p>
        '''
    else:
        for _dicom_id, _view_pos in metadata['dicom_id_view_pos_pairs']:
            if _dicom_id == dicom_id:
                continue
            _image_path = get_mimiccxr_large_image_path(part_id, patient_id, study_id, _dicom_id)
            html += f'''
            <h2>View position: {_view_pos}, dicom_id: {_dicom_id}</h2>
            <img src="{_image_path}" class="image">
            '''
    html += '''
    </body>
    </html>
    '''
    # Save html to pdf
    # NOTE: to avoid OSError: wkhtmltopdf reported an error:
    #       Exit with code 1 due to network error: ProtocolUnknownError
    # we add the following option: '--enable-local-file-access ""'
    pdfkit.from_string(html, pdf_path, options={'enable-local-file-access': ''})

def load_mimiccxr_reports_detailed_metadata(qa_adapted_reports_filename=None, exclude_invalid_sentences=False,
                                            background_findings_and_impression_per_report_filepath=None):
    
    assert qa_adapted_reports_filename is None or background_findings_and_impression_per_report_filepath is None

    if qa_adapted_reports_filename is None and background_findings_and_impression_per_report_filepath is None:
        filename = 'detailed_metadata.pkl'
    elif qa_adapted_reports_filename is not None:
        if exclude_invalid_sentences:
            filename = f'{qa_adapted_reports_filename}(invalid_excluded)__detailed_metadata.pkl'
        else:
            filename = f'{qa_adapted_reports_filename}__detailed_metadata.pkl'
    elif background_findings_and_impression_per_report_filepath is not None:
        bfaipr_filename = os.path.basename(background_findings_and_impression_per_report_filepath)
        filename = f'{bfaipr_filename}__detailed_metadata.pkl'
    else: assert False
    cache_path = os.path.join(MIMICCXR_CACHE_DIR, filename)
    if os.path.exists(cache_path):
        return get_cached_pickle_file(cache_path)
    
    print('Computing detailed metadata...')

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
        for i, report in tqdm(enumerate(qa_adapted_reports['reports']), mininterval=2):
            filepath = report['filepath']
            part_id, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(filepath)[0])            
            backgrounds[i] = report['background']
            if exclude_invalid_sentences:
                invalid_set = set(report['invalid'])
                sentences = report['sentences']
                reports[i] = '.\n '.join(sentences[i] for i in range(len(sentences)) if i not in invalid_set)
            else:
                reports[i] = '.\n '.join(report['sentences'])
            part_ids[i] = part_id
            subject_ids[i] = subject_id
            study_ids[i] = study_id
            dicom_id_view_pos_pairs[i] = image_views_dict[(subject_id, study_id)]
            splits[i] = split_dict[(subject_id, study_id, dicom_id_view_pos_pairs[i][0][0])]
            for j in range(1, len(dicom_id_view_pos_pairs[i])):
                assert split_dict[(subject_id, study_id, dicom_id_view_pos_pairs[i][j][0])] == splits[i]
            filepaths[i] = filepath
    elif background_findings_and_impression_per_report_filepath is not None:
        backgrounds = [None] * n_reports
        findings = [None] * n_reports
        impressions = [None] * n_reports
        report_metadata['backgrounds'] = backgrounds
        report_metadata['findings'] = findings
        report_metadata['impressions'] = impressions
        bfaipr_data = get_cached_json_file(background_findings_and_impression_per_report_filepath)
        assert len(bfaipr_data) == n_reports
        for i, row in tqdm(enumerate(bfaipr_data), mininterval=2):
            filepath = row['path']
            backgrounds[i] = row['background']
            findings[i] = row['findings']
            impressions[i] = row['impression']
            part_id, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(filepath)[0])
            part_ids[i] = part_id
            subject_ids[i] = subject_id
            study_ids[i] = study_id
            dicom_id_view_pos_pairs[i] = image_views_dict[(subject_id, study_id)]
            splits[i] = split_dict[(subject_id, study_id, dicom_id_view_pos_pairs[i][0][0])]
            for j in range(1, len(dicom_id_view_pos_pairs[i])):
                assert split_dict[(subject_id, study_id, dicom_id_view_pos_pairs[i][j][0])] == splits[i]
            filepaths[i] = filepath
    else:
        for i, report_path in tqdm(enumerate(report_paths), mininterval=2):
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

    save_pickle(report_metadata, cache_path)
    print(f'Saved detailed metadata to {cache_path}')
    return report_metadata

def get_number_of_reports():
    return len(load_mimiccxr_reports_detailed_metadata()['part_ids'])

def get_detailed_metadata_for_dicom_id(dicom_id, qa_adapted_reports_filename=None):
    detailed_metadata = load_mimiccxr_reports_detailed_metadata(qa_adapted_reports_filename)    
    output = []
    for i in range(len(detailed_metadata['dicom_id_view_pos_pairs'])):
        for dicom_id_view_pos_pair in detailed_metadata['dicom_id_view_pos_pairs'][i]:
            if dicom_id_view_pos_pair[0] == dicom_id:
                output.append({
                    'report_index': i,
                    'part_id': detailed_metadata['part_ids'][i],
                    'subject_id': detailed_metadata['subject_ids'][i],
                    'study_id': detailed_metadata['study_ids'][i],
                    'dicom_id': dicom_id,
                    'view_pos': dicom_id_view_pos_pair[1],
                    'split': detailed_metadata['splits'][i],                    
                    'filepath': detailed_metadata['filepaths'][i],
                    'dicom_id_view_pos_pairs': detailed_metadata['dicom_id_view_pos_pairs'][i],
                })
                if qa_adapted_reports_filename is not None:
                    output[-1]['background'] = detailed_metadata['backgrounds'][i]
                    output[-1]['report'] = detailed_metadata['reports'][i]
    return output

def _get_mimiccxr_split_dicom_ids(split_name):
    key = f'get_mimiccxr_{split_name}_dicom_ids()'
    if key in _cache: return _cache[key]
    output = [x[0][2] for x in get_split_dict().items() if x[1] == split_name]
    _cache[key] = output
    return output
def get_mimiccxr_test_dicom_ids():
    return _get_mimiccxr_split_dicom_ids('test')
def get_mimiccxr_train_dicom_ids():
    return _get_mimiccxr_split_dicom_ids('train')
def get_mimiccxr_val_dicom_ids():
    return _get_mimiccxr_split_dicom_ids('validate')

def get_train_val_test_stats_per_chexpert_label(chexpert_labels_filename):
    cache_path = os.path.join(MIMICCXR_CACHE_DIR, f'train_val_test_chexpert_stats_per_label({chexpert_labels_filename}).pkl')
    if os.path.exists(cache_path): return get_cached_pickle_file(cache_path)

    chexpert_labels_path = os.path.join(MIMICCXR_CACHE_DIR, chexpert_labels_filename)
    chexpert_labels = get_cached_pickle_file(chexpert_labels_path)

    metadata = load_mimiccxr_reports_detailed_metadata()
    splits = metadata['splits']
    assert len(splits) == len(chexpert_labels)
    n = len(splits)

    label2stats = {}
    for i, label_name in tqdm(enumerate(CHEXPERT_LABELS)):
        train_pos_count, train_neg_count = 0, 0
        val_pos_count, val_neg_count = 0, 0
        test_pos_count, test_neg_count = 0, 0
        for j in range(n):
            if splits[j] == 'train':
                if chexpert_labels[j][i] == 1: train_pos_count += 1
                elif chexpert_labels[j][i] == 0: train_neg_count += 1
                else: assert False
            elif splits[j] == 'validate':
                if chexpert_labels[j][i] == 1: val_pos_count += 1
                elif chexpert_labels[j][i] == 0: val_neg_count += 1
                else: assert False
            elif splits[j] == 'test':
                if chexpert_labels[j][i] == 1: test_pos_count += 1
                elif chexpert_labels[j][i] == 0: test_neg_count += 1
                else: assert False
            else: assert False
        label2stats[label_name] = {
            'train': {'positive': train_pos_count, 'negative': train_neg_count},
            'val': {'positive': val_pos_count, 'negative': val_neg_count},
            'test': {'positive': test_pos_count, 'negative': test_neg_count},
        }

    save_pickle(label2stats, cache_path)
    return label2stats

def get_train_val_test_summary_text_for_chexpert_label(label, labels_filename):
    label2stats = get_train_val_test_stats_per_chexpert_label(labels_filename)
    # Get actual label
    label_idx = CHEXPERT_LABELS.index(label)
    assert label_idx != -1, f'Could not find label {label}'
    # Retrieve summary statistics
    stats = label2stats[label]
    # Return summary text
    train_pos_count = stats['train']['positive']
    train_fraction  = train_pos_count / (train_pos_count + stats['train']['negative'])
    val_pos_count   = stats['val']['positive']
    val_fraction    = val_pos_count / (val_pos_count + stats['val']['negative'])
    test_pos_count  = stats['test']['positive']
    test_fraction   = test_pos_count / (test_pos_count + stats['test']['negative'])
    summary_text = (f'Label: {label}\n'
                    f'Train: {train_pos_count} ({train_fraction:.2%})\n'
                    f'Val: {val_pos_count} ({val_fraction:.2%})\n'
                    f'Test: {test_pos_count} ({test_fraction:.2%})')
    return summary_text