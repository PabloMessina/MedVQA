import argparse
import imagesize
import numpy as np
from tqdm import tqdm

from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    get_imageId2partId,
    load_scene_graphs_in_parallel,
)
from medvqa.datasets.mimiccxr import get_mimiccxr_large_image_path
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_BBOXES_FILEPATH,
)
from medvqa.utils.files import save_to_pickle

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    # Load scene graphs
    print(f'Loading scene graphs with {args.num_workers} workers')
    scene_graphs = load_scene_graphs_in_parallel(num_workers=args.num_workers)
    print(f'Loaded {len(scene_graphs)} scene graphs')

    # Load imageId2partId
    imageId2partId = get_imageId2partId()

    # Map bbox names to indices
    bboxName2index = {bbox_name: i for i, bbox_name in enumerate(CHEST_IMAGENOME_BBOX_NAMES)}

    # Obtain bbox coords and presence    
    def _obtain_bbox_coords_and_presence(scene_graph):
        image_path = get_mimiccxr_large_image_path(
            part_id=imageId2partId[scene_graph['image_id']],
            subject_id=scene_graph['patient_id'],
            study_id=scene_graph['study_id'],
            dicom_id=scene_graph['image_id'],
        )
        width, height = imagesize.get(image_path)
        assert len(scene_graph['objects']) <= CHEST_IMAGENOME_NUM_BBOX_CLASSES, scene_graph
        bbox_coords = np.zeros((CHEST_IMAGENOME_NUM_BBOX_CLASSES * 4,), dtype=np.float32) # x1, y1, x2, y2
        bbox_presence = np.zeros((CHEST_IMAGENOME_NUM_BBOX_CLASSES,), dtype=np.float32) # 0 or 1
        for bbox in scene_graph['objects']:
            i = bboxName2index[bbox['bbox_name']]
            # Normalize bbox coords and clamp to [0, 1]
            x1 = max(0, min(1, bbox['original_x1'] / width))
            y1 = max(0, min(1, bbox['original_y1'] / height))
            x2 = max(0, min(1, bbox['original_x2'] / width))
            y2 = max(0, min(1, bbox['original_y2'] / height))
            bbox_coords[i * 4 + 0] = x1
            bbox_coords[i * 4 + 1] = y1
            bbox_coords[i * 4 + 2] = x2
            bbox_coords[i * 4 + 3] = y2
            bbox_presence[i] = 1
        return {
            'coords': bbox_coords,
            'presence': bbox_presence,
        }
    
    imageId2bboxcoords = {}
    for scene_graph in tqdm(scene_graphs):
        imageId2bboxcoords[scene_graph['image_id']] = _obtain_bbox_coords_and_presence(scene_graph)

    # Save imageId2bboxcoords
    print(f'Saving imageId2bboxcoords to {CHEST_IMAGENOME_BBOXES_FILEPATH}')
    save_to_pickle(imageId2bboxcoords, CHEST_IMAGENOME_BBOXES_FILEPATH)