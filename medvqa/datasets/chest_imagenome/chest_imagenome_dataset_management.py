import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_CACHE_DIR,
    CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR,
    CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH,
    CHEST_IMAGENOME_ATTRIBUTES_DICT,
)
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_IMAGE_REGEX,
    MIMICCXR_STUDY_REGEX,
    get_image_views_dict as get_mimiccxr_image_views_dict,
    get_mimiccxr_large_image_path,
    get_mimiccxr_report_path,
)
from medvqa.datasets.mimiccxr.preprocessing import image_paths_generator
from medvqa.utils.files import get_cached_json_file, load_pickle

# Load scene graphs
def _load_scene_graphs(scene_graphs_dir, k=None, offset=0):    
    filenames = os.listdir(scene_graphs_dir)
    scene_graphs = [None] * len(filenames)
    for i in tqdm(range(offset, len(filenames)), desc='Loading scene graphs'):
        f = filenames[i]
        if i == offset + k:
            i -= 1
            break
        if f.endswith('.json'):
            with open(os.path.join(scene_graphs_dir, f), 'r') as f:
                scene_graphs[i-offset] = json.load(f)
    i -= offset
    if k is None or k > 0:
        assert scene_graphs[i] is not None
    assert scene_graphs[i+1] is None
    scene_graphs = scene_graphs[:i+1]
    return scene_graphs

def load_silver_scene_graphs(k, offset):
    return _load_scene_graphs(CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR, k=k, offset=offset)

def get_gold_scene_graphs_paths():
    df = pd.read_csv(CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH)
    dicom_ids_to_avoid = df['dicom_id'].tolist()
    output = []
    for dicom_id in dicom_ids_to_avoid:
        scene_graph_path = os.path.join(CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR, f'{dicom_id}_SceneGraph.json')        
        if os.path.exists(scene_graph_path):
            output.append(scene_graph_path)
    return output

def load_postprocessed_label_names(labels_filename):
    labels = load_pickle(os.path.join(CHEST_IMAGENOME_CACHE_DIR, labels_filename))
    assert labels is not None, labels_filename
    return labels
        
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

def get_imageId2partId():
    imageId2partId = {}
    for image_path in tqdm(image_paths_generator()):
        image_path = str(image_path)
        partId, _, _, imageId = MIMICCXR_IMAGE_REGEX.findall(image_path)[0]
        imageId2partId[imageId] = partId
    return imageId2partId

def load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix(
        chest_imagenome_labels_filename,
        qa_adapted_reports_filename,
    ):    
    # Load chest_imagenome_labels    
    chest_imagenome_labels = load_pickle(os.path.join(CHEST_IMAGENOME_CACHE_DIR, chest_imagenome_labels_filename))
    assert chest_imagenome_labels is not None, chest_imagenome_labels_filename

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
    chest_imagenome_label_names = load_pickle(os.path.join(CHEST_IMAGENOME_CACHE_DIR, chest_imagenome_label_names_filename))
    assert chest_imagenome_label_names is not None, chest_imagenome_label_names_filename
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
def visualize_scene_graph(scene_graph, imageId2partId, figsize=(10, 10)):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

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
        assert w == object['original_width']
        assert h == object['original_height']
        # make bounding box transparent if it is not in labels
        in_labels = any([label[0] == object['bbox_name'] for label in labels])
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=plt.cm.tab20(i), facecolor='none', alpha=0.3 if not in_labels else 1.0)
        ax.add_patch(rect)
        # add a label (make it transparent if it is not in labels)
        ax.text(x, y-3, object['bbox_name'], fontsize=16, color=plt.cm.tab20(i), alpha=0.3 if not in_labels else 1.0)

    # Show the image
    plt.show()

    # Print labels
    print('Labels:')
    labels = extract_labels_from_scene_graph(scene_graph)
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