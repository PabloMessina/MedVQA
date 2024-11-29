import os
import numpy as np
import argparse
import subprocess
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_FAST_CACHE_DIR,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_dicom_ids,
    load_chest_imagenome_gold_bboxes,
    load_chest_imagenome_silver_bboxes,
)
from medvqa.datasets.mimiccxr import (
    get_imageId2PartPatientStudy,
    get_mimiccxr_medium_image_path,
    get_mimiccxr_test_dicom_ids,
)
from medvqa.metrics.bbox.utils import (
    compute_mae_per_class__yolov5,
    compute_mean_iou_per_class__yolov5,
    compute_multiple_prf1_scores__yolov5,
)
from medvqa.train_yolov5 import YOLOv5_PYTHON_PATH

from medvqa.utils.common import (
    YOLOv5_DETECT_SCRIPT_PATH,
    YOLOv5_RUNS_DETECT_DIR,
    parsed_args_to_dict,
)
from medvqa.utils.files import get_checkpoint_folder_path, get_results_folder_path, save_pickle
from medvqa.utils.logging import print_blue, print_red, print_bold

class EvalDatasets:
    MIMICCXR_TEST_SET = 'chest_imagenome__mimiccxr_test_set'
    CHEST_IMAGENOME_GOLD = 'chest_imagenome_gold'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dataset-name', type=str, required=True,
                        choices=[EvalDatasets.MIMICCXR_TEST_SET, EvalDatasets.CHEST_IMAGENOME_GOLD])
    parser.add_argument('--weights-path', type=str, required=True)
    parser.add_argument('--decent-images-only', default=False, action='store_true')
    parser.add_argument('--image-size', type=int, default=416)
    parser.add_argument('--conf', type=float, default=0.1)
    parser.add_argument('--max-det', type=int, default=36, help='maximum number of detections per image')
    parser.add_argument('--print-inference-time', default=False, action='store_true')
    return parser.parse_args()

def get_eval_image_paths_txt_filepath__chest_imagenome(eval_dataset_name, decent_images_only=False):
    eval_images_txt_path = os.path.join(CHEST_IMAGENOME_FAST_CACHE_DIR, 'yolov5', 'images', f'{eval_dataset_name}{"(decent_images_only)" if decent_images_only else ""}.txt')
    if os.path.exists(eval_images_txt_path):
        print(f'Found {eval_images_txt_path}')
    else:
        os.makedirs(os.path.dirname(eval_images_txt_path), exist_ok=True)
        if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
            eval_dicom_ids = set(get_mimiccxr_test_dicom_ids())
            eval_dicom_ids &= set(load_chest_imagenome_dicom_ids(decent_images_only=decent_images_only))
        elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
            eval_dicom_ids = set(load_chest_imagenome_gold_bboxes().keys())
            if decent_images_only:
                eval_dicom_ids &= set(load_chest_imagenome_dicom_ids(decent_images_only=decent_images_only))
        else:
            raise ValueError(f'Unknown eval_dataset_name: {eval_dataset_name}')
        
        imageId2PartPatientStudy = get_imageId2PartPatientStudy()

        print(f'Writing {len(eval_dicom_ids)} images to {eval_images_txt_path}')
        with open(eval_images_txt_path, 'w') as f:
            for dicom_id in eval_dicom_ids:
                part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
                image_path = get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id)
                assert os.path.exists(image_path)
                f.write(f'{image_path}\n')

    return eval_images_txt_path

def _load_predictions_from_folder(model_predictions_folder, dicom_ids):
    n = len(dicom_ids)
    pred_boxes = [None] * n
    pred_classes = [None] * n
    skipped = 0
    for i in range(n):
        dicom_id = dicom_ids[i]
        pred_boxes_path = os.path.join(model_predictions_folder, f'{dicom_id}.txt')
        if not os.path.exists(pred_boxes_path):
            pred_boxes[i] = []
            pred_classes[i] = []
            skipped += 1
        else:
            pred_labels = np.loadtxt(pred_boxes_path, delimiter=' ', dtype=object)
            assert 1 <= len(pred_labels.shape) <= 2
            if len(pred_labels.shape) == 1:
                pred_labels = pred_labels.reshape(1, -1)
            pred_classes[i] = pred_labels[:, 0].astype(int)
            pred_boxes[i] = pred_labels[:, 1:].astype(float)
            # adapt from (x_center, y_center, w, h) to (x_min, y_min, x_max, y_max)
            pred_boxes[i][:, 0] = pred_boxes[i][:, 0] - pred_boxes[i][:, 2] / 2
            pred_boxes[i][:, 1] = pred_boxes[i][:, 1] - pred_boxes[i][:, 3] / 2
            pred_boxes[i][:, 2] = pred_boxes[i][:, 0] + pred_boxes[i][:, 2]
            pred_boxes[i][:, 3] = pred_boxes[i][:, 1] + pred_boxes[i][:, 3]

    if skipped > 0:
        print_red(f'WARNING: {skipped} images without predictions', bold=True)
    
    return pred_boxes, pred_classes

def _load_ground_truth(dicom_ids, gt_bbox_dict, n_bbox_classes):
    n = len(dicom_ids)
    test_bbox_coords = np.zeros((n, n_bbox_classes, 4), dtype=float)
    test_bbox_presences = np.zeros((n, n_bbox_classes), dtype=int)
    for i in range(n):
        dicom_id = dicom_ids[i]
        gt_bboxes = gt_bbox_dict[dicom_id]            
        test_bbox_coords[i] = gt_bboxes['coords'].reshape(-1, 4)
        test_bbox_presences[i] = gt_bboxes['presence']
    # clip coords to [0, 1]
    test_bbox_coords = np.clip(test_bbox_coords, 0, 1)
    return test_bbox_coords, test_bbox_presences

def _compute_bbox_metrics(pred_boxes, pred_classes, test_bbox_coords, test_bbox_presences, valid_classes, bbox_names):
    metrics = {}
    # Mean Absolute Error
    print('Computing Mean Absolute Error (MAE) ...')
    metrics['mae'] = compute_mae_per_class__yolov5(pred_boxes, pred_classes, test_bbox_coords, test_bbox_presences, valid_classes,
                                                   abs_fn=np.abs)
    metrics['mean_mae'] = np.mean(metrics['mae'])
    
    # Mean Intersection Over Union (IOU)
    print('Computing Mean Intersection Over Union (IOU) ...')
    metrics['iou'] = compute_mean_iou_per_class__yolov5(pred_boxes, pred_classes, test_bbox_coords, test_bbox_presences, valid_classes)
    metrics['mean_iou'] = np.mean(metrics['iou'])
    
    # Precision, Recall, and F1 Score
    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f'Computing Precision, Recall, and F1 Score at IOU thresholds {iou_thresholds} ...')    
    scores = compute_multiple_prf1_scores__yolov5(
        pred_boxes=pred_boxes,
        pred_classes=pred_classes,
        gt_coords=test_bbox_coords,
        gt_presences=test_bbox_presences,
        iou_thresholds=iou_thresholds,
        valid_classes=valid_classes,
        num_workers=7,
    )
    n_valid_classes = len(bbox_names) if valid_classes is None else sum(valid_classes)
    assert scores.shape == (len(iou_thresholds), n_valid_classes, 3)
    mean_p = 0
    mean_r = 0
    mean_f1 = 0
    for i, iou_thrs in enumerate(iou_thresholds):
        metrics[f'p@{iou_thrs}'] = scores[i, :, 0]
        metrics[f'r@{iou_thrs}'] = scores[i, :, 1]
        metrics[f'f1@{iou_thrs}'] = scores[i, :, 2]
        metrics[f'mean_p@{iou_thrs}'] = metrics[f'p@{iou_thrs}'].mean()
        metrics[f'mean_r@{iou_thrs}'] = metrics[f'r@{iou_thrs}'].mean()
        metrics[f'mean_f1@{iou_thrs}'] = metrics[f'f1@{iou_thrs}'].mean()
        mean_p += metrics[f'mean_p@{iou_thrs}']
        mean_r += metrics[f'mean_r@{iou_thrs}']
        mean_f1 += metrics[f'mean_f1@{iou_thrs}']
    mean_p /= len(iou_thresholds)
    mean_r /= len(iou_thresholds)
    mean_f1 /= len(iou_thresholds)
    metrics['mean_p'] = mean_p
    metrics['mean_r'] = mean_r
    metrics['mean_f1'] = mean_f1
    
    # Save class names
    if valid_classes is None:
        metrics['bbox_names'] = bbox_names
    else:
        assert len(valid_classes) == len(bbox_names)
        metrics['bbox_names'] = [bbox_names[i] for i in range(len(bbox_names)) if valid_classes[i]]

    return metrics


def compute_and_save_bbox_metrics(model_predictions_folder, cmd_args, dicom_ids, gt_bbox_dict, bbox_names, eval_dataset_name,
                                  results_folder_path, decent_images_only, valid_classes=None):

    # Load predictions
    pred_boxes, pred_classes = _load_predictions_from_folder(model_predictions_folder, dicom_ids)
    
    # Load ground truth
    test_bbox_coords, test_bbox_presences = _load_ground_truth(dicom_ids, gt_bbox_dict, len(bbox_names))

    # Compute metrics
    metrics = _compute_bbox_metrics(pred_boxes, pred_classes, test_bbox_coords, test_bbox_presences, valid_classes, bbox_names)
    
    # Save metrics
    save_path = os.path.join(results_folder_path,
        (f'{eval_dataset_name}__bbox_metrics({"decent" if decent_images_only else ""}).pkl'))
    save_pickle(metrics, save_path)
    print('Saved bbox metrics to: ', end='')
    print_bold(save_path)
    
    # Save additional info to be able to trace back original model and predictions
    save_path = os.path.join(results_folder_path,
        (f'{eval_dataset_name}__metadata({"decent" if decent_images_only else ""}).pkl'))
    save_pickle({
        'cmd_args': cmd_args,
        'model_predictions_folder': model_predictions_folder,
    }, save_path)
    print('Saved metadata to: ', end='')
    print_bold(save_path)

def get_dicom_ids_from_eval_txt_file(eval_images_txt_path):
    dicom_ids = []
    with open(eval_images_txt_path, 'r') as f:
        for line in f:
            image_path = line.strip()
            if len(image_path) == 0:
                continue
            dicom_id = os.path.basename(image_path).split('.')[0]
            dicom_ids.append(dicom_id)
    return dicom_ids

def eval_yolov5(
        eval_dataset_name,
        decent_images_only,
        weights_path,
        image_size,
        conf,
        max_det,
        print_inference_time,
    ):
    print_blue('-' * 50, bold=True)
    print_blue('Evaluating YOLOv5', bold=True)

    # Prepare eval data for YOLOv5
    print_blue('\n1) Preparing eval data for YOLOv5', bold=True)
    eval_images_txt_path = get_eval_image_paths_txt_filepath__chest_imagenome(
        eval_dataset_name=eval_dataset_name,
        decent_images_only=decent_images_only,
    )
    
    # Run YOLOv5 detect.py script
    print_blue('\n2) Running YOLOv5 detect.py script', bold=True)
    cmd = [
        YOLOv5_PYTHON_PATH,
        YOLOv5_DETECT_SCRIPT_PATH,
        '--weights', weights_path,
        '--img', str(image_size),
        '--conf', str(conf),
        '--source', eval_images_txt_path,
        '--save-txt',
        '--nosave',
        '--max-det', str(max_det),
    ]
    if not print_inference_time:
        cmd.append('--dont-print-inference-time')
    print(' '.join(cmd))
    print('\n')
    subprocess.call(cmd)
    
    # Find specific folder where YOLOv5 saved the predicted labels
    print_blue('\n3) Finding folder where YOLOv5 saved the predicted labels', bold=True)
    exp_folders = os.listdir(YOLOv5_RUNS_DETECT_DIR)
    _, exp_folder = max((int(x[3:] if len(x) > 3 else 0), x) for x in exp_folders)
    model_predictions_folder = os.path.join(YOLOv5_RUNS_DETECT_DIR, exp_folder, 'labels')
    print(f'Found folder: {model_predictions_folder}')
    
    # Compute and save metrics
    print_blue('\n4) Computing and saving metrics', bold=True)
    dicom_ids = get_dicom_ids_from_eval_txt_file(eval_images_txt_path)
    print(f'Found {len(dicom_ids)} dicom ids')
    if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
        gt_bbox_dict = load_chest_imagenome_silver_bboxes()
        valid_classes = None
    elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
        gt_bbox_dict = load_chest_imagenome_gold_bboxes()
        valid_classes = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES, dtype=np.bool)
        _, _gt_presence_indices = get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices()
        valid_classes[_gt_presence_indices] = True
    else:
        raise ValueError(f'Unknown eval dataset name: {eval_dataset_name}')
    results_folder_path = get_results_folder_path(get_checkpoint_folder_path('visual_module', 'chest-imagenome', 'yolov5'))
    compute_and_save_bbox_metrics(
        model_predictions_folder=model_predictions_folder,
        cmd_args=cmd,
        dicom_ids=dicom_ids,
        gt_bbox_dict=gt_bbox_dict,
        bbox_names=CHEST_IMAGENOME_BBOX_NAMES,
        eval_dataset_name=eval_dataset_name,
        results_folder_path=results_folder_path,
        decent_images_only=decent_images_only,
        valid_classes=valid_classes,
    )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    eval_yolov5(**args)