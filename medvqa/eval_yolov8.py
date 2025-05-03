import os
import numpy as np
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_gold_bboxes,
    load_chest_imagenome_silver_bboxes,
)
from medvqa.eval_yolov5 import _compute_bbox_metrics, _load_ground_truth, prepare_eval_data__chest_imagenome
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.files_utils import get_checkpoint_folder_path, get_results_folder_path, read_lines_from_txt, save_pickle
from medvqa.utils.logging_utils import print_blue, print_bold

class EvalDatasets:
    MIMICCXR_TEST_SET = 'chest_imagenome__mimiccxr_test_set'
    CHEST_IMAGENOME_GOLD = 'chest_imagenome_gold'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dataset-name', type=str, required=True,
                        choices=[EvalDatasets.MIMICCXR_TEST_SET, EvalDatasets.CHEST_IMAGENOME_GOLD])
    parser.add_argument('--weights-path', type=str, required=True)
    parser.add_argument('--decent-images-only', default=False, action='store_true')
    parser.add_argument('--conf', type=float, default=0.1)
    parser.add_argument('--max-det', type=int, default=36, help='maximum number of detections per image')
    parser.add_argument('--save-predictions', default=False, action='store_true')
    return parser.parse_args()

def compute_and_save_bbox_metrics(pred_boxes, pred_classes, metadata, dicom_ids, gt_bbox_dict, bbox_names, eval_dataset_name,
                                  results_folder_path, decent_images_only, valid_classes=None, save_predictions=False):
    
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
    save_pickle(metadata, save_path)
    print('Saved metadata to: ', end='')
    print_bold(save_path)

    # Save predictions
    if save_predictions:
        save_path = os.path.join(results_folder_path,
            (f'{eval_dataset_name}__predictions({"decent" if decent_images_only else ""}).pkl'))
        data = {
            'pred_boxes': pred_boxes,
            'pred_classes': pred_classes,
            'dicom_ids': dicom_ids,
        }
        save_pickle(data, save_path)
        print('Saved predictions to: ', end='')
        print_bold(save_path)

def eval_yolov8(
        eval_dataset_name,
        decent_images_only,
        weights_path,
        conf,
        max_det,
        save_predictions,
    ):
    print_blue('-' * 50, bold=True)
    print_blue('Evaluating YOLOv8', bold=True)

    # Prepare eval data for YOLOv8
    print_blue('\n1) Preparing eval data for YOLOv8', bold=True)
    eval_images_txt_path = prepare_eval_data__chest_imagenome(
        eval_dataset_name=eval_dataset_name,
        decent_images_only=decent_images_only,
    )
    image_paths = read_lines_from_txt(eval_images_txt_path)
    dicom_ids = [os.path.basename(x).split('.')[0] for x in image_paths]
    
    # Run YOLOv8 in eval mode
    print_blue('\n2) Running YOLOv8 in eval mode', bold=True)
    model = YOLO(model=weights_path, task='detect')
    n = len(image_paths)
    pred_boxes = [None] * n
    pred_classes = [None] * n
    # results = model(image_paths, conf=conf, max_det=max_det, stream=True, verbose=False)
    for i, image_path in tqdm(enumerate(image_paths)):
        results = model([image_path], conf=conf, max_det=max_det, verbose=False)
        result = results[0]
        pred_boxes[i] = result.boxes.xyxyn.cpu().numpy()
        pred_classes[i] = result.boxes.cls.cpu().numpy().astype(int)
    # for i, result in tqdm(enumerate(results)):
    #     pred_boxes[i] = result.boxes.xyxyn.cpu().numpy()
    #     pred_classes[i] = result.boxes.cls.cpu().numpy().astype(int)
    
    # Compute and save metrics
    print_blue('\n3) Computing and saving metrics', bold=True)
    if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
        gt_bbox_dict = load_chest_imagenome_silver_bboxes()
        valid_classes = None
    elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
        gt_bbox_dict = load_chest_imagenome_gold_bboxes()
        valid_classes = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES, dtype=bool)
        _, _gt_presence_indices = get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices()
        valid_classes[_gt_presence_indices] = True
    else:
        raise ValueError(f'Unknown eval dataset name: {eval_dataset_name}')
    
    results_folder_path = get_results_folder_path(get_checkpoint_folder_path('visual_module', 'chest-imagenome', 'yolov8'))
    metadata = {
        'weights_path': weights_path,
        'conf': conf,
        'max_det': max_det,
    }
    compute_and_save_bbox_metrics(
        pred_boxes=pred_boxes,
        pred_classes=pred_classes,
        metadata=metadata,
        dicom_ids=dicom_ids,
        gt_bbox_dict=gt_bbox_dict,
        bbox_names=CHEST_IMAGENOME_BBOX_NAMES,
        eval_dataset_name=eval_dataset_name,
        results_folder_path=results_folder_path,
        decent_images_only=decent_images_only,
        valid_classes=valid_classes,
        save_predictions=save_predictions,
    )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    eval_yolov8(**args)