import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset
from PIL import Image
import imagesize
import torch

from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH,
    CHEST_IMAGENOME_SILVER_BBOXES_FILEPATH,
    CHEST_IMAGENOME_CACHE_DIR,
    CHEST_IMAGENOME_GOLD_BBOX_COORDINATE_ANNOTATIONS_CSV_PATH,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR,
    CHEST_IMAGENOME_ATTRIBUTES_DICT,
)
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_STUDY_REGEX,
    get_image_views_dict as get_mimiccxr_image_views_dict,    
    get_mimiccxr_large_image_path,
    get_mimiccxr_report_path,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.datasets.mimiccxr import get_imageId2PartPatientStudy, get_imageId2partId
from medvqa.utils.files import get_cached_json_file, get_cached_pickle_file, load_json_file, load_pickle, save_to_pickle
from medvqa.metrics.bbox.utils import compute_iou

def _load_scene_graph(scene_graph_path):
    return load_json_file(scene_graph_path)

def _load_scene_graphs(scene_graphs_dir, k=None, offset=0):    
    filenames = os.listdir(scene_graphs_dir)
    scene_graphs = [None] * len(filenames)
    for i in tqdm(range(offset, len(filenames)), desc='Loading scene graphs'):
        f = filenames[i]
        if i == offset + k:
            i -= 1
            break
        scene_graphs[i-offset] = _load_scene_graph(os.path.join(scene_graphs_dir, f))
    i -= offset
    if k is None or k > 0:
        assert scene_graphs[i] is not None
    assert scene_graphs[i+1] is None
    scene_graphs = scene_graphs[:i+1]
    return scene_graphs

def load_scene_graphs_in_parallel(num_workers=4, first_k=None):
    start_time = time.time()
    filenames = os.listdir(CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR)
    if first_k is not None:
        filenames = filenames[:first_k]
    filepaths = [os.path.join(CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR, f) for f in filenames]
    with Pool(num_workers) as p:
        scene_graphs = p.map(_load_scene_graph, filepaths)
    elapsed_time = time.time() - start_time
    print(f'Elapsed time: {elapsed_time:.2f} seconds')
    return scene_graphs

def load_silver_scene_graphs(k, offset):
    return _load_scene_graphs(CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR, k=k, offset=offset)

def get_silver_scene_graphs_paths_for_gold_dataset():
    df = pd.read_csv(CHEST_IMAGENOME_GOLD_BBOX_COORDINATE_ANNOTATIONS_CSV_PATH)
    gold_dicom_ids = [x[:-4] for x in df.image_id.unique()]
    output = []
    for dicom_id in gold_dicom_ids:
        scene_graph_path = os.path.join(CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR, f'{dicom_id}_SceneGraph.json')
        output.append(scene_graph_path)
    return output

def load_postprocessed_label_names(label_names_filename):
    label_names = load_pickle(os.path.join(CHEST_IMAGENOME_CACHE_DIR, label_names_filename))
    assert label_names is not None, label_names_filename
    return label_names

def load_postprocessed_labels(labels_filename):
    labels = load_pickle(os.path.join(CHEST_IMAGENOME_CACHE_DIR, labels_filename))
    assert labels is not None, labels_filename
    return labels

def load_chest_imagenome_silver_bboxes():
    chest_imagenome_bboxes = get_cached_pickle_file(CHEST_IMAGENOME_SILVER_BBOXES_FILEPATH)
    assert chest_imagenome_bboxes is not None, CHEST_IMAGENOME_SILVER_BBOXES_FILEPATH
    return chest_imagenome_bboxes

def load_gold_standard_dicom_ids():
    df = pd.read_csv(CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH)
    return df['dicom_id'].tolist()

def load_nongold_dicom_ids():
    cache_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'nongold_dicom_ids.pkl')
    if os.path.exists(cache_path):
        return load_pickle(cache_path)
    dicom_ids_set = set(load_chest_imagenome_silver_bboxes().keys())
    gold_ids_set = set(load_gold_standard_dicom_ids())
    dicom_ids = list(dicom_ids_set - gold_ids_set)
    save_to_pickle(dicom_ids, cache_path)
    return dicom_ids

def load_chest_imagenome_silver_bboxes_as_numpy_array(dicom_ids_list, clamp=False):
    bboxes = load_chest_imagenome_silver_bboxes()
    bbox_coords = np.empty((len(bboxes), 4 * CHEST_IMAGENOME_NUM_BBOX_CLASSES), dtype=float)
    bbox_presence = np.empty((len(bboxes), CHEST_IMAGENOME_NUM_BBOX_CLASSES), dtype=float)
    did2idx = {did: i for i, did in enumerate(bboxes.keys())}
    idxs = np.array([did2idx[did] for did in dicom_ids_list], dtype=int)
    for did in bboxes.keys():
        idx = did2idx[did]
        bbox = bboxes[did]
        bbox_coords[idx] = bbox['coords']
        bbox_presence[idx] = bbox['presence']
    if clamp:
        bbox_coords.clip(0, 1, out=bbox_coords) # Clip to [0, 1] in-place
    return idxs, bbox_coords, bbox_presence

def load_chest_imagenome_dicom_ids():
    chest_imagenome_bboxes = load_chest_imagenome_silver_bboxes()
    return list(chest_imagenome_bboxes.keys())

def load_chest_imagenome_gold_bboxes():
    cache_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'chest_imagenome_gold_bboxes.pkl')
    if os.path.exists(cache_path):
        return get_cached_pickle_file(cache_path)
    df = pd.read_csv(CHEST_IMAGENOME_GOLD_BBOX_COORDINATE_ANNOTATIONS_CSV_PATH)
    imageId2PartPatientStudy = get_imageId2PartPatientStudy()
    imageId2bboxes = {}
    imageId2widthHeight = {}
    for image_id, x1, y1, x2, y2, bbox_name in zip(
        df.image_id, df.original_x1, df.original_y1,
        df.original_x2, df.original_y2, df.bbox_name,
    ):
        image_id = image_id[:-4]
        try:
            bboxes = imageId2bboxes[image_id]
        except KeyError:
            bboxes = {
                'coords': np.zeros((CHEST_IMAGENOME_NUM_BBOX_CLASSES * 4,)),
                'presence': np.zeros((CHEST_IMAGENOME_NUM_BBOX_CLASSES,)),
            }
            imageId2bboxes[image_id] = bboxes
        idx = CHEST_IMAGENOME_BBOX_NAMES.index(bbox_name)
        try:
            width, height = imageId2widthHeight[image_id]
        except KeyError:
            part_id, patient_id, study_id = imageId2PartPatientStudy[image_id]
            image_path = get_mimiccxr_large_image_path(part_id, patient_id, study_id, image_id)
            imageId2widthHeight[image_id] = imagesize.get(image_path)
            width, height = imageId2widthHeight[image_id]        
        bboxes['coords'][idx*4+0] = x1 / width
        bboxes['coords'][idx*4+1] = y1 / height
        bboxes['coords'][idx*4+2] = x2 / width
        bboxes['coords'][idx*4+3] = y2 / height
        bboxes['presence'][idx] = 1
    save_to_pickle(imageId2bboxes, cache_path)
    return imageId2bboxes

def get_chest_imagenome_gold_bbox_names():
    cache_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'chest_imagenome_gold_bbox_names.pkl')
    if os.path.exists(cache_path):
        return get_cached_pickle_file(cache_path)
    gold_bboxes = load_chest_imagenome_gold_bboxes()
    bbox_names = set()
    for bbox in gold_bboxes.values():
        for i, presence in enumerate(bbox['presence']):
            if presence == 1:
                bbox_names.add(CHEST_IMAGENOME_BBOX_NAMES[i])
    bbox_names = list(bbox_names)
    bbox_names.sort()
    save_to_pickle(bbox_names, cache_path)    
    return bbox_names

def load_chest_imagenome_train_average_bbox_coords(mimiccxr_qa_adapted_reports_filename, clamp_bbox_coords=True):
    # Define output path
    output_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR,
        f'chest_imagenome_train_average_bbox_coords({mimiccxr_qa_adapted_reports_filename},{"clamped" if clamp_bbox_coords else "unclamped"}).pkl')    
    # Load from cache if possible
    if os.path.exists(output_path):
        print(f'Loading {output_path}...')
        return load_pickle(output_path)
    # Compute the average bbox for each class from the training set    
    bboxes_dict = load_chest_imagenome_silver_bboxes()
    mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata(mimiccxr_qa_adapted_reports_filename)
    train_idxs = [i for i, split in enumerate(mimiccxr_detailed_metadata['splits']) if split == 'train']
    avg_bbox_coords = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES * 4)
    bbox_counts = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES * 4)
    for idx in train_idxs:
        dicom_id_view_pairs = mimiccxr_detailed_metadata['dicom_id_view_pos_pairs'][idx]
        for dicom_id, _ in dicom_id_view_pairs:
            if dicom_id in bboxes_dict:
                bbox = bboxes_dict[dicom_id]
                bbox_coords = bbox['coords']
                if clamp_bbox_coords:
                    bbox_coords.clip(0, 1, out=bbox_coords)
                bbox_presence = bbox['presence']
                for i in range(CHEST_IMAGENOME_NUM_BBOX_CLASSES):
                    if bbox_presence[i]:
                        s = i*4
                        e = (i+1)*4
                        avg_bbox_coords[s:e] += bbox_coords[s:e]
                        bbox_counts[s:e] += 1
    avg_bbox_coords /= bbox_counts
    # Save the average bbox coords
    save_to_pickle(avg_bbox_coords, output_path)
    print(f'Saved {output_path}')
    # Return the average bbox coords
    return avg_bbox_coords

def load_chest_imagenome_average_bbox_coords(clamp_bbox_coords=True):
    # Define output path
    output_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR,
        f'chest_imagenome_train_average_bbox_coords({"clamped" if clamp_bbox_coords else "unclamped"}).pkl')
    # Load from cache if possible
    if os.path.exists(output_path):
        print(f'Loading {output_path}...')
        return load_pickle(output_path)
    # Compute the average bbox for each class from the training set    
    bboxes_dict = load_chest_imagenome_silver_bboxes()
    mimiccxr_detailed_metadata = load_mimiccxr_reports_detailed_metadata()
    avg_bbox_coords = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES * 4)
    bbox_counts = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES * 4)
    for dicom_id_view_pairs in mimiccxr_detailed_metadata['dicom_id_view_pos_pairs']:
        for dicom_id, _ in dicom_id_view_pairs:
            if dicom_id in bboxes_dict:
                bbox = bboxes_dict[dicom_id]
                bbox_coords = bbox['coords']
                if clamp_bbox_coords:
                    bbox_coords.clip(0, 1, out=bbox_coords)
                bbox_presence = bbox['presence']
                for i in range(CHEST_IMAGENOME_NUM_BBOX_CLASSES):
                    if bbox_presence[i]:
                        s = i*4
                        e = (i+1)*4
                        avg_bbox_coords[s:e] += bbox_coords[s:e]
                        bbox_counts[s:e] += 1
    avg_bbox_coords /= bbox_counts
    # Save the average bbox coords
    save_to_pickle(avg_bbox_coords, output_path)
    print(f'Saved {output_path}')
    # Return the average bbox coords
    return avg_bbox_coords
        
# Extract labels from a scene graph
def extract_labels_from_scene_graph(scene_graph):
    labels = []
    for node in scene_graph['attributes']:
        bbox_name = node['bbox_name']
        for x in node['attributes']:
            for y in x:
                category, value, name = y.split('|')
                if category == 'temporal' or\
                   category == 'severity' or\
                   category == 'laterality':
                    continue
                assert category in CHEST_IMAGENOME_ATTRIBUTES_DICT, category
                assert name in CHEST_IMAGENOME_ATTRIBUTES_DICT[category], name
                assert value == 'yes' or value == 'no', value
                value = int(value == 'yes') # convert to 0/1
                labels.append((bbox_name, category, name, value))    
    return labels

def load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix(
        chest_imagenome_labels_filename,
        qa_adapted_reports_filename,
    ):    
    # Load chest_imagenome_labels    
    chest_imagenome_labels = load_postprocessed_labels(chest_imagenome_labels_filename)

    # Obtain dicom_ids
    dicom_ids = set(chest_imagenome_labels.keys())
    
    # Adapt chest_imagenome_labels so that they can be indexed by report_id            
    mimiccxr_qa_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))
    n_reports = len(mimiccxr_qa_reports['reports'])
    n_labels = len(next(iter(chest_imagenome_labels.values())))
    adapted_chest_imagenome_labels = np.zeros((n_reports, n_labels), dtype=np.int8)    
    # map dicom_id to report_id
    image_views_dict = get_mimiccxr_image_views_dict()
    did2rid = {}
    for i, report in enumerate(mimiccxr_qa_reports['reports']):
        _, subject_id, study_id = map(int, MIMICCXR_STUDY_REGEX.findall(report['filepath'])[0])            
        views = image_views_dict[(subject_id, study_id)]
        for view in views:
            did2rid[view[0]] = i            
    # use did2rid to map dicom_id to report_id to get the label
    for dicom_id, label in chest_imagenome_labels.items():
        adapted_chest_imagenome_labels[did2rid[dicom_id]] = label
        
    return dicom_ids, adapted_chest_imagenome_labels

def load_chest_imagenome_label_names_and_templates(chest_imagenome_label_names_filename):
    # Load chest_imagenome_label_names and compute templates for each label
    chest_imagenome_label_names = load_postprocessed_label_names(chest_imagenome_label_names_filename)
    chest_imagenome_templates = {}
    for label_name in chest_imagenome_label_names:
        if len(label_name) == 3:
            anomaly = label_name[2]
            anomaly = anomaly.replace('/', ' or ')
            anatomy = label_name[0]
            positive_answer = f'{anomaly} in {anatomy}' # anomaly observed in anatomy
        elif len(label_name) == 2:
            anomaly = label_name[1]
            anomaly = anomaly.replace('/', ' or ')
            positive_answer = anomaly # anomaly observed
        else:
            raise ValueError(f'Unexpected label_name: {label_name}')
        chest_imagenome_templates[label_name] = {
            0: '', # no anomaly
            1: positive_answer # anomaly observed
        }        
    return chest_imagenome_label_names, chest_imagenome_templates

# Visualize a scene graph
def visualize_scene_graph(scene_graph, figsize=(10, 10)):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    imageId2partId = get_imageId2partId()

    # Extract labels
    labels = extract_labels_from_scene_graph(scene_graph)

    # Load image
    image_path = get_mimiccxr_large_image_path(
        part_id=imageId2partId[scene_graph['image_id']],
        subject_id=scene_graph['patient_id'],
        study_id=scene_graph['study_id'],
        dicom_id=scene_graph['image_id'],
    )
    image = Image.open(image_path)
    image = image.convert('RGB')

    # Print image path
    print('Image path:', image_path)

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=figsize)

    # Display the image
    ax.imshow(image)

    # Create a Rectangle patch for each object
    for i, object in enumerate(scene_graph['objects']):
        # x, y, w, h = object['bbox']
        x = object['original_x1']
        y = object['original_y1']
        w = object['original_x2'] - object['original_x1']
        h = object['original_y2'] - object['original_y1']
        assert abs(w - object['original_width']) < 2, (w, object['original_width'])
        assert abs(h - object['original_height']) < 2, (h, object['original_height'])
        # make bounding box transparent if it is not in labels
        in_labels = any([label[0] == object['bbox_name'] for label in labels])
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=plt.cm.tab20(i), facecolor='none', alpha=0.3 if not in_labels else 1.0)
        ax.add_patch(rect)
        # add a label (make it transparent if it is not in labels)
        ax.text(x, y-3, object['bbox_name'], fontsize=16, color=plt.cm.tab20(i), alpha=0.3 if not in_labels else 1.0)
        print('Object:', object['bbox_name'], (x, y, w, h))

    print('Num objects:', len(scene_graph['objects']))

    # Show the image
    plt.show()

    # Print labels
    print('Labels:')
    for label in labels:
        print(label)

    # Print original report
    print()
    print('-' * 80)
    print('Original report:')
    report_path = get_mimiccxr_report_path(
        part_id=imageId2partId[scene_graph['image_id']],
        subject_id=scene_graph['patient_id'],
        study_id=scene_graph['study_id'],
    )
    with open(report_path, 'r') as f:
        print(f.read())

def visualize_ground_truth_bounding_boxes(dicom_id):
    bboxes_dict = load_chest_imagenome_silver_bboxes()
    imageId2PartPatientStudy = get_imageId2PartPatientStudy()
    bbox = bboxes_dict[dicom_id]
    coords = bbox['coords']
    presence = bbox['presence']
    part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
    
    # Large image
    image_path = get_mimiccxr_large_image_path(part_id, patient_id, study_id, dicom_id)
    image = Image.open(image_path)
    image = image.convert('RGB')
    width = image.size[0]
    height = image.size[1]
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    print(f'Image path: {image_path}')
    for i in range(len(presence)):
        if presence[i] == 1:
            x1 = coords[i * 4 + 0] * width
            y1 = coords[i * 4 + 1] * height
            x2 = coords[i * 4 + 2] * width
            y2 = coords[i * 4 + 3] * height
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=plt.cm.tab20(i), facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-3, CHEST_IMAGENOME_BBOX_NAMES[i], fontsize=16, color=plt.cm.tab20(i))
            print(f'Object: {CHEST_IMAGENOME_BBOX_NAMES[i]} ({x1:.1f}, {y1:.1f}, {x2-x1:.1f}, {y2-y1:.1f})')
    plt.show()

def visualize_predicted_bounding_boxes(dicom_id, pred_coords, pred_presence,
                                        gt_coords=None, gt_presence=None, figsize=(10, 10),
                                        verbose=False, bbox_names=CHEST_IMAGENOME_BBOX_NAMES):
    if gt_coords is None or gt_presence is None:
        bboxes_dict = load_chest_imagenome_silver_bboxes()        
        bbox = bboxes_dict[dicom_id]
        gt_coords = bbox['coords']
        gt_presence = bbox['presence']

    imageId2PartPatientStudy = get_imageId2PartPatientStudy()    
    part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
    
    # Large image
    image_path = get_mimiccxr_large_image_path(part_id, patient_id, study_id, dicom_id)
    image = Image.open(image_path)
    image = image.convert('RGB')
    width = image.size[0]
    height = image.size[1]
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    print(f'Image path: {image_path}')
    for i in range(len(gt_presence)):
        if gt_presence[i] == 1:
            x1 = gt_coords[i * 4 + 0] * width
            y1 = gt_coords[i * 4 + 1] * height
            x2 = gt_coords[i * 4 + 2] * width
            y2 = gt_coords[i * 4 + 3] * height
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-3, bbox_names[i], fontsize=16, color='red')
            if verbose:
                print(f'Object: {bbox_names[i]} ({x1:.1f}, {y1:.1f}, {x2-x1:.1f}, {y2-y1:.1f})')
    for i in range(len(pred_presence)):
        if pred_presence[i] > 0:
            x1 = pred_coords[i * 4 + 0] * width
            y1 = pred_coords[i * 4 + 1] * height
            x2 = pred_coords[i * 4 + 2] * width
            y2 = pred_coords[i * 4 + 3] * height
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='blue', facecolor='none', linestyle='dashed')
            ax.add_patch(rect)
            ax.text(x1, y1-3, bbox_names[i], fontsize=16, color='blue')
    plt.show()

_RIGHT_LEFT_PAIRS = []
for _reg_r, _reg_l in [
    ('right lung', 'left lung'),
    ('right upper lung zone', 'left upper lung zone'),
    ('right mid lung zone', 'left mid lung zone'),
    ('right lower lung zone', 'left lower lung zone'),
    ('right hilar structures', 'left hilar structures'),
    ('right apical zone', 'left apical zone'),
    ('right costophrenic angle', 'left costophrenic angle'),
    ('right cardiophrenic angle', 'left cardiophrenic angle'),
    ('right hemidiaphragm', 'left hemidiaphragm'),
    ('right clavicle', 'left clavicle'),
    ('right cardiac silhouette', 'left cardiac silhouette'),
    ('right upper abdomen', 'left upper abdomen'),
]:
    _RIGHT_LEFT_PAIRS.append((
        CHEST_IMAGENOME_BBOX_NAMES.index(_reg_r),
        CHEST_IMAGENOME_BBOX_NAMES.index(_reg_l),
    ))

def determine_if_image_is_decent(dicom_id, average_bboxes, detailed_output=False, debug=False):
    bboxes_dict = load_chest_imagenome_silver_bboxes()
    bbox = bboxes_dict[dicom_id]
    coords = bbox['coords']
    presence = bbox['presence']

    if debug:
        imageId2PartPatientStudy = get_imageId2PartPatientStudy()
        part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
        image_path = get_mimiccxr_large_image_path(part_id, patient_id, study_id, dicom_id)
        width, height = imagesize.get(image_path)
        print(f'Image path: {image_path}')
        print(f'Image size: {width} x {height}')
        print(f'Image ID: {dicom_id}')
        print(f'bbox: {bbox}')

    # Heuristic 1: if the average distance vector between pairs of bboxes of symmetric objects is roughly horizontal,
    # and the standard deviation of the distance vectors is small, then the image is decent
    vector_list = []
    for r_idx, l_idx in _RIGHT_LEFT_PAIRS:
        if presence[r_idx] == 1 and presence[l_idx] == 1:
            r_x1 = coords[r_idx * 4 + 0]
            r_y1 = coords[r_idx * 4 + 1]
            r_x2 = coords[r_idx * 4 + 2]
            r_y2 = coords[r_idx * 4 + 3]
            l_x1 = coords[l_idx * 4 + 0]
            l_y1 = coords[l_idx * 4 + 1]
            l_x2 = coords[l_idx * 4 + 2]
            l_y2 = coords[l_idx * 4 + 3]
            r_center = np.array([(r_x1 + r_x2) / 2, (r_y1 + r_y2) / 2])
            l_center = np.array([(l_x1 + l_x2) / 2, (l_y1 + l_y2) / 2])
            vector = l_center - r_center            
            vector /= np.linalg.norm(vector) # normalize
            vector_list.append(vector)
            if debug:
                print(f'{CHEST_IMAGENOME_BBOX_NAMES[r_idx]} - {CHEST_IMAGENOME_BBOX_NAMES[l_idx]} = {vector}')            
    if len(vector_list) > 0.4 * len(_RIGHT_LEFT_PAIRS): # if more than 40% of symmetric objects are present
        # Sub-heuristic: if all vectors point to the same direction except for one, then
        # remove the outlier vector
        for i in range(2): # do this twice because i=0 may be the outlier
            outlier_idxs = []
            for j in range(len(vector_list)):
                if np.dot(vector_list[i], vector_list[j]) < 0.5:
                    outlier_idxs.append(j)
            if len(outlier_idxs) == 1:                
                vector_list.pop(outlier_idxs[0])
                if debug:
                    print(f'Remove outlier vector {outlier_idxs[0]}')
                break

        avg_vector = np.mean(vector_list, axis=0)
        std_vector = np.std(vector_list, axis=0)
        if debug:
            print('avg_vector:', avg_vector)
            print('std_vector:', std_vector)
        if abs(avg_vector[0]) * 0.4 > abs(avg_vector[1]) and avg_vector[0] > 0\
            and std_vector[0] + std_vector[1] < 0.65:

            if detailed_output:
                return True, vector_list
            return True

    # Heuristic 2: if the IoU between these bboxes and the average bboxes across all images
    # is high, then the image is decent
    mean_iou = 0
    for i in range(len(presence)):
        if presence[i] == 1:
            iou = compute_iou(coords[i * 4: i * 4 + 4], average_bboxes[i * 4: i * 4 + 4])
            mean_iou += iou
    den = sum(presence)
    if den > 0:
        mean_iou /= den
    if debug:
        print('mean_iou:', mean_iou)
    if mean_iou > 0.75:
        if detailed_output:
            return True, vector_list
        return True

    if detailed_output:
        return False, vector_list
    return False

def determine_if_image_is_rotated(dicom_id, average_bboxes, debug=False):

    # Heuristic 1: if image is decent, then it is not rotated.
    ans, vector_list = determine_if_image_is_decent(dicom_id, average_bboxes, detailed_output=True, debug=debug)
    if ans: return False

    # Heuristic 2: if few pairs are present, then the image is not rotated.
    if len(vector_list) < len(_RIGHT_LEFT_PAIRS) * 0.5:
        return False

    # Heuristic 3: if for some reason the image is still roughly horizontal, then the image is not rotated.
    avg_vector = np.mean(vector_list, axis=0)
    if debug:
        print('avg_vector:', avg_vector)
    if abs(avg_vector[0]) * 0.5 > abs(avg_vector[1]) and avg_vector[0] > 0:
        return False

    # Heuristic 4: if the standard deviation of the vectors is not small, then the image is not rotated.
    std_vector = np.std(vector_list, axis=0)
    if debug:
        print('std_vector:', std_vector)
    if std_vector[0] + std_vector[1] > 0.25:
        return False
    
    # It probably is rotated
    return True

def visualize_symmetric_ground_truth_bounding_boxes(dicom_id):
    bboxes_dict = load_chest_imagenome_silver_bboxes()
    imageId2PartPatientStudy = get_imageId2PartPatientStudy()
    bbox = bboxes_dict[dicom_id]
    coords = bbox['coords']
    presence = bbox['presence']
    part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
    # Large image
    image_path = get_mimiccxr_large_image_path(part_id, patient_id, study_id, dicom_id)
    image = Image.open(image_path)
    image = image.convert('RGB')
    width = image.size[0]
    height = image.size[1]
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    print(f'Image path: {image_path}')
    unused_idxs = set(range(len(CHEST_IMAGENOME_BBOX_NAMES)))
    # Plot symmetric bboxes
    i = 0
    for r_idx, l_idx in _RIGHT_LEFT_PAIRS:
        if presence[r_idx] == 1 and presence[l_idx] == 1:
            r_x1 = coords[r_idx * 4 + 0] * width
            r_y1 = coords[r_idx * 4 + 1] * height
            r_x2 = coords[r_idx * 4 + 2] * width
            r_y2 = coords[r_idx * 4 + 3] * height
            l_x1 = coords[l_idx * 4 + 0] * width
            l_y1 = coords[l_idx * 4 + 1] * height
            l_x2 = coords[l_idx * 4 + 2] * width
            l_y2 = coords[l_idx * 4 + 3] * height
            r_center = np.array([(r_x1 + r_x2) / 2, (r_y1 + r_y2) / 2])
            l_center = np.array([(l_x1 + l_x2) / 2, (l_y1 + l_y2) / 2])
            # right bbox
            rect = patches.Rectangle((r_x1, r_y1), r_x2-r_x1, r_y2-r_y1, linewidth=3, edgecolor=plt.cm.tab20(i), facecolor='none')
            ax.add_patch(rect)
            ax.text(r_x1, r_y1, CHEST_IMAGENOME_BBOX_NAMES[r_idx], fontsize=16, color=plt.cm.tab20(i))
            # left bbox
            rect = patches.Rectangle((l_x1, l_y1), l_x2-l_x1, l_y2-l_y1, linewidth=3, edgecolor=plt.cm.tab20(i), facecolor='none')
            ax.add_patch(rect)
            ax.text(l_x1, l_y1, CHEST_IMAGENOME_BBOX_NAMES[l_idx], fontsize=16, color=plt.cm.tab20(i))
            # line
            ax.plot([r_center[0], l_center[0]], [r_center[1], l_center[1]], color=plt.cm.tab20(i), linewidth=3)
            unused_idxs.remove(r_idx)
            unused_idxs.remove(l_idx)
            i += 1
    # Plot unused bboxes
    for idx in unused_idxs:
        if presence[idx] == 1:
            x1 = coords[idx * 4 + 0] * width
            y1 = coords[idx * 4 + 1] * height
            x2 = coords[idx * 4 + 2] * width
            y2 = coords[idx * 4 + 3] * height
            # make the bbox a bit transparent unless it is the spine
            is_spine = CHEST_IMAGENOME_BBOX_NAMES[idx] == 'spine'
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='y', facecolor='none', alpha=1.0 if is_spine else 0.3)
            ax.add_patch(rect)
            ax.text(x1, y1, CHEST_IMAGENOME_BBOX_NAMES[idx], fontsize=16, color='y', alpha=1.0 if is_spine else 0.3)
    # Show
    plt.show()

class ChestImagenomeBboxDataset(Dataset):
    def __init__(self, image_paths, image_transform, bbox_coords, bbox_presences):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.bbox_coords = bbox_coords
        self.bbox_presences = bbox_presences

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.image_transform(image)
        bbox_coords = self.bbox_coords[i]
        bbox_presences = self.bbox_presences[i]
        return image, bbox_coords, bbox_presences

def chest_imagenome_bbox_collate_fn(batch):
    images = []
    bbox_coords = []
    bbox_presences = []
    for image, bbox_coord, bbox_presence in batch:
        images.append(image)
        bbox_coords.append(bbox_coord)
        bbox_presences.append(bbox_presence)
    images = torch.stack(images)
    bbox_coords = torch.stack(bbox_coords)
    bbox_presences = torch.stack(bbox_presences)
    return images, bbox_coords, bbox_presences