import json
import os
import pandas as pd
from tqdm import tqdm

from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_SILVER_SCENE_GRAPHS_DIR,
    CHEST_IMAGENOME_IMAGES_TO_AVOID_CSV_PATH,
    CHEST_IMAGENOME_ATTRIBUTES_DICT,
)
from medvqa.datasets.mimiccxr import (
    MIMICCXR_IMAGE_REGEX,
    get_mimiccxr_large_image_path,
    get_mimiccxr_report_path,
)
from medvqa.datasets.mimiccxr.preprocessing import image_paths_generator

# import reload # Python 3


# Load scene graphs
def _load_scene_graphs(scene_graphs_dir, k=None, offset=0):    
    filepaths = os.listdir(scene_graphs_dir)
    scene_graphs = [None] * len(filepaths)
    for i in tqdm(range(offset, len(filepaths)), desc='Loading scene graphs'):
        f = filepaths[i]
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
        
# # Load scene graphs in parallel using multiprocessing
# def _load_scene_graphs_parallel(scene_graphs_dir, n_workers=4), k=None:
#     import multiprocessing as mp
#     from functools import partial

#     # Load scene graphs
#     pool = mp.Pool(mp.cpu_count())
#     scene_graphs = pool.map(partial(_load_scene_graphs, scene_graphs_dir, k=k), range(mp.cpu_count()))
#     pool.close()
#     pool.join()

#     # Merge scene graphs
#     scene_graphs = [scene_graph for scene_graphs_ in scene_graphs for scene_graph in scene_graphs_]
#     return scene_graphs

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