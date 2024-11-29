import os
import random
import subprocess
import argparse
from tqdm import tqdm
from medvqa.datasets.vinbig import VINBIG_FAST_CACHE_DIR, VINBIG_LARGE_FAST_CACHE_DIR, VINBIG_ORIGINAL_IMAGES_FOLDER, VINBIG_YOLOV5_LABELS_DIR
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBigTrainerBase
from medvqa.utils.logging import print_blue
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_FAST_CACHE_DIR,
    CHEST_IMAGENOME_YOLOV5_LABELS_DIR,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_dicom_ids,
    load_gold_standard_related_dicom_ids,
    load_chest_imagenome_silver_bboxes,
)
from medvqa.datasets.mimiccxr import (
    MIMICCXR_JPG_IMAGES_MEDIUM_DIR,
    get_mimiccxr_test_dicom_ids,
    get_mimiccxr_train_dicom_ids,
    get_mimiccxr_val_dicom_ids,
    get_imageId2PartPatientStudy,
    get_mimiccxr_medium_image_path,
)
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.constants import DATASET_NAMES, VINBIG_BBOX_NAMES, VINBIG_NUM_BBOX_CLASSES
from medvqa.utils.common import (
    YOLOv5_PYTHON_PATH,
    YOLOv5_TRAIN_SCRIPT_PATH,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default=DATASET_NAMES.MIMICCXR_CHEST_IMAGENOME_MODE)
    parser.add_argument('--decent-images-only', action='store_true', default=False)
    parser.add_argument('--validation-only', action='store_true', default=False)
    parser.add_argument('--image-size', type=int, default=416)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--cache-images', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()

def prepare_train_val_test_data__chest_imagenome(decent_images_only=False, validation_only=False, debug=False):

    if debug:
        debug_images_txt_path = os.path.join(CHEST_IMAGENOME_FAST_CACHE_DIR, 'yolov5', 'images', f'train_debug.txt')
        if os.path.exists(debug_images_txt_path):
            print(f'Found {debug_images_txt_path}')
        else:
            debug_size = 100
            print(f'Writing {debug_size} images to {debug_images_txt_path}')
            debug_dicom_ids = random.sample(load_chest_imagenome_dicom_ids(decent_images_only=True), debug_size)
            imageId2PartPatientStudy = get_imageId2PartPatientStudy()
            with open(debug_images_txt_path, 'w') as f:
                for dicom_id in debug_dicom_ids:
                    part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
                    image_path = get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id)
                    assert os.path.exists(image_path)
                    f.write(f'{image_path}\n')
        
        train_images_txt_path = debug_images_txt_path
        val_images_txt_path = debug_images_txt_path
        test_images_txt_path = debug_images_txt_path
    else:
        train_images_txt_path  = os.path.join(CHEST_IMAGENOME_FAST_CACHE_DIR, 'yolov5', 'images', f'train{"(decent_images_only)" if decent_images_only else ""}.txt')
        val_images_txt_path    = os.path.join(CHEST_IMAGENOME_FAST_CACHE_DIR, 'yolov5', 'images', f'val{"(decent_images_only)" if decent_images_only else ""}.txt')
        test_images_txt_path   = os.path.join(CHEST_IMAGENOME_FAST_CACHE_DIR, 'yolov5', 'images', f'test{"(decent_images_only)" if decent_images_only else ""}.txt')

        if os.path.exists(train_images_txt_path):
            assert os.path.exists(val_images_txt_path)
            assert os.path.exists(test_images_txt_path)
            print(f'Found {train_images_txt_path}')
            print(f'Found {val_images_txt_path}')
            print(f'Found {test_images_txt_path}')
        else:
            os.makedirs(os.path.dirname(train_images_txt_path), exist_ok=True)
            os.makedirs(os.path.dirname(val_images_txt_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_images_txt_path), exist_ok=True)
            
            mimiccxr_train_dicom_ids = set(get_mimiccxr_train_dicom_ids())
            mimiccxr_val_dicom_ids = set(get_mimiccxr_val_dicom_ids())
            mimiccxr_test_dicom_ids = set(get_mimiccxr_test_dicom_ids())
            allowed_dicom_ids = set(load_chest_imagenome_dicom_ids(decent_images_only=decent_images_only))
            gold_dicom_ids = set(load_gold_standard_related_dicom_ids())
            actual_train_dicom_ids = (mimiccxr_train_dicom_ids & allowed_dicom_ids) - gold_dicom_ids
            actual_val_dicom_ids = (mimiccxr_val_dicom_ids & allowed_dicom_ids) - gold_dicom_ids
            actual_test_dicom_ids = mimiccxr_test_dicom_ids & allowed_dicom_ids
            
            imageId2PartPatientStudy = get_imageId2PartPatientStudy()

            for txt_path, dicom_ids in [
                (train_images_txt_path, actual_train_dicom_ids),
                (val_images_txt_path, actual_val_dicom_ids),
                (test_images_txt_path, actual_test_dicom_ids),
            ]:
                print(f'Writing {len(dicom_ids)} images to {txt_path}')
                with open(txt_path, 'w') as f:
                    for dicom_id in dicom_ids:
                        part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
                        image_path = get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id)
                        assert os.path.exists(image_path)
                        f.write(f'{image_path}\n')

        if validation_only:
            train_images_txt_path = val_images_txt_path
            test_images_txt_path = val_images_txt_path
    
    if os.path.exists(CHEST_IMAGENOME_YOLOV5_LABELS_DIR):
        print(f'Found {CHEST_IMAGENOME_YOLOV5_LABELS_DIR}')
    else:
        print(f'Writing {CHEST_IMAGENOME_YOLOV5_LABELS_DIR}/*.txt')
        os.makedirs(CHEST_IMAGENOME_YOLOV5_LABELS_DIR, exist_ok=True)
        bbox_dict = load_chest_imagenome_silver_bboxes()
        imageId2PartPatientStudy = get_imageId2PartPatientStudy()
        for dicom_id, bboxes in tqdm(bbox_dict.items()):
            part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
            image_path = get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id)
            labels_path = image_path.replace(MIMICCXR_JPG_IMAGES_MEDIUM_DIR, CHEST_IMAGENOME_YOLOV5_LABELS_DIR).replace('.jpg', '.txt')
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            # assert False
            coords = bboxes['coords']
            presence = bboxes['presence']            
            with open(labels_path, 'w') as f:
                for i in range(len(presence)):
                    if presence[i]:
                        x1 = coords[i * 4 + 0]
                        y1 = coords[i * 4 + 1]
                        x2 = coords[i * 4 + 2]
                        y2 = coords[i * 4 + 3]
                        x1 = max(min(x1, 1), 0)
                        y1 = max(min(y1, 1), 0)
                        x2 = max(min(x2, 1), 0)
                        y2 = max(min(y2, 1), 0)                        
                        x_mid = (x1 + x2) / 2
                        y_mid = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        assert 0 <= x_mid <= 1
                        assert 0 <= y_mid <= 1
                        assert 0 <= w <= 1
                        assert 0 <= h <= 1
                        if w > 0 and h > 0:
                            f.write(f'{i} {x_mid} {y_mid} {w} {h}\n')

    # Write YAML config file
    strings = []
    if debug:
        strings.append('debug')
    else:
        if decent_images_only:
            strings.append('decent_images_only')
        else:
            strings.append('all_images')
        if validation_only:
            strings.append('validation_only')
    assert len(strings) > 0
    yaml_config_filename = f'config({",".join(strings)}).yaml'
    yaml_config_path = os.path.join(CHEST_IMAGENOME_FAST_CACHE_DIR, 'yolov5', yaml_config_filename)
    if os.path.exists(yaml_config_path):
        print(f'Found {yaml_config_path}')
    else:
        print(f'Writing {yaml_config_path}')
        with open(yaml_config_path, 'w') as f:
            f.write('# Train/val/test sets\n')
            f.write(f'path: {CHEST_IMAGENOME_FAST_CACHE_DIR}/yolov5\n')
            f.write(f'train: images/{os.path.basename(train_images_txt_path)}\n')
            f.write(f'val: images/{os.path.basename(val_images_txt_path)}\n')
            f.write(f'test: images/{os.path.basename(test_images_txt_path)}\n')
            f.write('\n# From image to labels\n')
            f.write(f'source_pattern: {MIMICCXR_JPG_IMAGES_MEDIUM_DIR}\n')
            f.write(f'target_pattern: {CHEST_IMAGENOME_YOLOV5_LABELS_DIR}\n')
            f.write('\n# Classes\n')
            f.write(f'nc: {len(CHEST_IMAGENOME_BBOX_NAMES)}\n')
            f.write('names:\n')
            for i, name in enumerate(CHEST_IMAGENOME_BBOX_NAMES):
                f.write(f'  {i}: {name}\n')

    output = {
        'train_images_txt_path': train_images_txt_path,
        'val_images_txt_path': val_images_txt_path,
        'test_images_txt_path': test_images_txt_path,
        'yaml_config_path': yaml_config_path,
    }
    return output

def prepare_train_val_test_data__vindrcxr(validation_only=False):
    
    train_images_txt_path  = os.path.join(VINBIG_FAST_CACHE_DIR, 'yolov5', 'images', f'train.txt')
    val_images_txt_path    = os.path.join(VINBIG_FAST_CACHE_DIR, 'yolov5', 'images', f'val.txt')
    test_images_txt_path   = os.path.join(VINBIG_FAST_CACHE_DIR, 'yolov5', 'images', f'test.txt')

    vinbigdata = VinBigTrainerBase(load_bouding_boxes=True)

    if os.path.exists(train_images_txt_path):
        assert os.path.exists(val_images_txt_path)
        assert os.path.exists(test_images_txt_path)
        print(f'Found {train_images_txt_path}')
        print(f'Found {val_images_txt_path}')
        print(f'Found {test_images_txt_path}')
    else:
        os.makedirs(os.path.dirname(train_images_txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(val_images_txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_images_txt_path), exist_ok=True)
        
        train_indices, test_indices = vinbigdata.train_indices, vinbigdata.test_indices
        val_indices = test_indices # there is no official validation set, so we use the test set as validation set

        for txt_path, indices in [
            (train_images_txt_path, train_indices),
            (val_images_txt_path, val_indices),
            (test_images_txt_path, test_indices),
        ]:
            print(f'Writing {len(indices)} images to {txt_path}')
            with open(txt_path, 'w') as f:
                for idx in indices:
                    image_path = vinbigdata.image_paths[idx]
                    assert os.path.exists(image_path)
                    f.write(f'{image_path}\n')

    if validation_only:
        train_images_txt_path = val_images_txt_path
        test_images_txt_path = val_images_txt_path
    
    if os.path.exists(VINBIG_YOLOV5_LABELS_DIR):
        print(f'Found {VINBIG_YOLOV5_LABELS_DIR}')
    else:
        print(f'Writing {VINBIG_YOLOV5_LABELS_DIR}/*.txt')
        os.makedirs(VINBIG_YOLOV5_LABELS_DIR, exist_ok=True)
        for image_id, image_path in tqdm(zip(vinbigdata.image_ids, vinbigdata.image_paths)):
            bboxes = vinbigdata.image_id_2_bboxes[image_id]
            labels_path = image_path.replace(VINBIG_ORIGINAL_IMAGES_FOLDER, VINBIG_YOLOV5_LABELS_DIR).replace('.jpg', '.txt')
            assert os.path.exists(image_path)
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            bbox_list, class_list = bboxes           
            with open(labels_path, 'w') as f:
                for bbox, class_id in zip(bbox_list, class_list):
                    x1, y1, x2, y2 = bbox
                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    assert 0 <= x_mid <= 1
                    assert 0 <= y_mid <= 1
                    assert 0 < w <= 1
                    assert 0 < h <= 1
                    f.write(f'{class_id} {x_mid} {y_mid} {w} {h}\n')

    # Write YAML config file
    strings = []
    if validation_only:
        strings.append('validation_only')
    yaml_config_filename = f'config({",".join(strings)}).yaml' if len(strings) > 0 else 'config.yaml'
    yaml_config_path = os.path.join(VINBIG_LARGE_FAST_CACHE_DIR, 'yolov5', yaml_config_filename)
    if os.path.exists(yaml_config_path):
        print(f'Found {yaml_config_path}')
    else:
        os.makedirs(os.path.dirname(yaml_config_path), exist_ok=True)
        print(f'Writing {yaml_config_path}')
        with open(yaml_config_path, 'w') as f:
            f.write('# Train/val/test sets\n')
            f.write(f'path: {VINBIG_FAST_CACHE_DIR}/yolov5\n')
            f.write(f'train: images/{os.path.basename(train_images_txt_path)}\n')
            f.write(f'val: images/{os.path.basename(val_images_txt_path)}\n')
            f.write(f'test: images/{os.path.basename(test_images_txt_path)}\n')
            f.write('\n# From image to labels\n')
            f.write(f'source_pattern: {VINBIG_ORIGINAL_IMAGES_FOLDER}\n')
            f.write(f'target_pattern: {VINBIG_YOLOV5_LABELS_DIR}\n')
            f.write('\n# Classes\n')
            f.write(f'nc: {VINBIG_NUM_BBOX_CLASSES}\n')
            f.write('names:\n')
            for i, name in enumerate(VINBIG_BBOX_NAMES):
                f.write(f'  {i}: {name}\n')

    output = {
        'train_images_txt_path': train_images_txt_path,
        'val_images_txt_path': val_images_txt_path,
        'test_images_txt_path': test_images_txt_path,
        'yaml_config_path': yaml_config_path,
    }
    return output

def train_yolov5(
    dataset_name,
    decent_images_only,
    validation_only,
    image_size,
    batch_size,
    epochs,
    weights,
    cache_images,
    debug,
):
    print_blue('-' * 50, bold=True)
    print_blue('Training YOLOv5', bold=True)
    if dataset_name == DATASET_NAMES.MIMICCXR_CHEST_IMAGENOME_MODE:
        # Prepare train, val and test data for YOLOv5
        print_blue('\n1) Preparing train, val and test data for YOLOv5', bold=True)
        data_info = prepare_train_val_test_data__chest_imagenome(
            decent_images_only=decent_images_only,
            validation_only=validation_only,
            debug=debug,
        )
        # Run YOLOv5 train.py script
        print_blue('\n2) Running YOLOv5 train.py script', bold=True)
        cmd = [
            YOLOv5_PYTHON_PATH,
            YOLOv5_TRAIN_SCRIPT_PATH,
            '--img', str(image_size),
            '--batch', str(batch_size),
            '--epochs', str(epochs),
            '--data', data_info['yaml_config_path'],
            '--weights', weights,
        ]
        if cache_images:
            cmd.append('--cache')
        print(' '.join(cmd))
        print('\n')
        subprocess.call(cmd)
    else:
        raise ValueError(f'Unknown or unsupported dataset: {dataset_name}')

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    train_yolov5(**args)