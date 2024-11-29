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
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBigTrainerBase
from medvqa.eval_yolov5 import get_eval_image_paths_txt_filepath__chest_imagenome
from medvqa.metrics.bbox.utils import compute_mAP__yolov11, compute_mean_iou_per_class__yolov11
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.constants import VINBIG_BBOX_NAMES, VINBIG_NUM_BBOX_CLASSES
from medvqa.utils.files import get_checkpoint_folder_path, get_results_folder_path, read_lines_from_txt, save_pickle
from medvqa.utils.logging import print_blue, print_bold

class EvalDatasets:
    CHEST_IMAGENOME_MIMICCXR_TEST_SET = 'chest_imagenome__mimiccxr_test_set'
    CHEST_IMAGENOME_GOLD = 'chest_imagenome_gold'
    VINDRCXR_TEST_SET = 'vindrcxr_test_set'
    VINDRCXR_TRAIN_SET = 'vindrcxr_train_set'

    @staticmethod
    def get_all():
        return [
            EvalDatasets.CHEST_IMAGENOME_MIMICCXR_TEST_SET,
            EvalDatasets.CHEST_IMAGENOME_GOLD,
            EvalDatasets.VINDRCXR_TEST_SET,
            EvalDatasets.VINDRCXR_TRAIN_SET,
        ]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dataset_name', type=str, required=True, choices=EvalDatasets.get_all())
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--decent_images_only', action='store_true')
    parser.add_argument('--conf_threshold', type=float, default=0.1)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--max_det', type=int, default=36, help='maximum number of detections per image')
    parser.add_argument('--save_predictions', action='store_true')
    return parser.parse_args()

def _compute_bbox_metrics(pred_boxes, pred_classes, pred_confs, gt_bbox_coords, valid_classes, bbox_names):
    metrics = {}

    # Mean Intersection Over Union (IOU)
    print('Computing Mean Intersection Over Union (IOU) ...')
    metrics['iou'] = compute_mean_iou_per_class__yolov11(pred_boxes, pred_classes, gt_bbox_coords, valid_classes)
    metrics['mean_iou'] = np.mean(metrics['iou'])
    
    # Compute mean Average Precision (mAP) at different IOU thresholds
    iou_thresholds = [0.1, 0.3, 0.4, 0.5, 0.7]
    print(f'Computing mAP at IOU thresholds: {iou_thresholds} ...')
    scores = compute_mAP__yolov11(
        pred_boxes=pred_boxes,
        pred_classes=pred_classes,
        pred_confs=pred_confs,
        gt_coords=gt_bbox_coords,
        iou_thresholds=iou_thresholds,
        valid_classes=valid_classes,
        num_workers=7,
    )
    n_valid_classes = len(bbox_names) if valid_classes is None else sum(valid_classes)
    assert scores.shape == (len(iou_thresholds), n_valid_classes)
    for i, iou_thrs in enumerate(iou_thresholds):
        metrics[f'AP@{iou_thrs}'] = scores[i, :]
        metrics[f'mAP@{iou_thrs}'] = scores[i, :].mean()

    return metrics

def compute_and_save_bbox_metrics(pred_boxes, pred_classes, pred_confs, metadata, gt_bbox_coords, bbox_names, eval_dataset_name,
                                  results_folder_path, valid_classes=None, save_predictions=False, save_path_strings=None,
                                  image_paths=None):

    # Compute metrics
    metrics = _compute_bbox_metrics(pred_boxes, pred_classes, pred_confs, gt_bbox_coords, valid_classes, bbox_names)

    if save_path_strings is not None:
        parenthetical_str = f'({",".join(save_path_strings)})'
    else:
        parenthetical_str = ''
    
    # Save metrics
    save_path = os.path.join(results_folder_path,
        (f'{eval_dataset_name}__bbox_metrics{parenthetical_str}.pkl'))
    save_pickle(metrics, save_path)
    print('Saved bbox metrics to: ', end='')
    print_bold(save_path)
    
    # Save additional info to be able to trace back original model and predictions
    save_path = os.path.join(results_folder_path,
        (f'{eval_dataset_name}__metadata{parenthetical_str}.pkl'))
    save_pickle(metadata, save_path)
    print('Saved metadata to: ', end='')
    print_bold(save_path)

    # Save predictions
    if save_predictions:
        assert image_paths is not None
        save_path = os.path.join(results_folder_path,
            (f'{eval_dataset_name}__predictions{parenthetical_str}.pkl'))
        data = {
            'pred_boxes': pred_boxes,
            'pred_classes': pred_classes,
            'pred_confs': pred_confs,
            'gt_bbox_coords': gt_bbox_coords,
            'image_paths': image_paths,
        }
        save_pickle(data, save_path)
        print('Saved predictions to: ', end='')
        print_bold(save_path)

def eval_yolov11(
        eval_dataset_name,
        decent_images_only,
        weights_path,
        conf_threshold,
        iou_threshold,
        max_det,
        save_predictions,
    ):
    print_blue('-' * 50, bold=True)
    print_blue('Evaluating YOLOv11', bold=True)

    # Prepare eval data for YOLOv11
    print_blue('\n1) Preparing eval data for YOLOv11', bold=True)
    if eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_MIMICCXR_TEST_SET or\
        eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:
        eval_images_txt_path = get_eval_image_paths_txt_filepath__chest_imagenome(
            eval_dataset_name=eval_dataset_name,
            decent_images_only=decent_images_only,
        )
        image_paths = read_lines_from_txt(eval_images_txt_path)
        dicom_ids = [os.path.basename(x).split('.')[0] for x in image_paths]
    
    elif eval_dataset_name == EvalDatasets.VINDRCXR_TEST_SET:
        vinbigdata = VinBigTrainerBase(load_bouding_boxes=True)
        test_indices = vinbigdata.test_indices
        image_paths = [vinbigdata.image_paths[i] for i in test_indices]

    elif eval_dataset_name == EvalDatasets.VINDRCXR_TRAIN_SET:
        vinbigdata = VinBigTrainerBase(load_bouding_boxes=True)
        train_indices = vinbigdata.train_indices
        image_paths = [vinbigdata.image_paths[i] for i in train_indices]
    
    else:
        raise ValueError(f'Unknown eval dataset name: {eval_dataset_name}')
    
    # Run YOLOv11 in eval mode
    print_blue('\n2) Running YOLOv11 in eval mode', bold=True)
    model = YOLO(model=weights_path, task='detect')
    n = len(image_paths)
    pred_boxes = [None] * n
    pred_classes = [None] * n
    pred_confs = [None] * n
    for i, image_path in tqdm(enumerate(image_paths)):
        results = model.predict(source=image_path, conf=conf_threshold, iou=iou_threshold,
                                max_det=max_det, verbose=False)
        result = results[0]
        pred_boxes[i] = result.boxes.xyxyn.cpu().numpy()
        pred_classes[i] = result.boxes.cls.cpu().numpy().astype(int)
        pred_confs[i] = result.boxes.conf.cpu().numpy()
    
    # Compute and save metrics
    print_blue('\n3) Computing and saving metrics', bold=True)
    
    valid_classes = None
    
    if eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_MIMICCXR_TEST_SET:
        
        gt_bbox_dict = load_chest_imagenome_silver_bboxes()
        gt_bbox_coords = [[None] * len(CHEST_IMAGENOME_BBOX_NAMES) for _ in dicom_ids]
        for i, dicom_id in enumerate(dicom_ids):
            gt_bboxes = gt_bbox_dict[dicom_id]
            coords = gt_bboxes['coords'].reshape(-1, 4)
            presence = gt_bboxes['presence']
            for j in range(len(CHEST_IMAGENOME_BBOX_NAMES)):
                if presence[j]:
                    gt_bbox_coords[i][j] = coords[j].reshape(1, 4)

        results_folder_path = get_results_folder_path(get_checkpoint_folder_path('visual_module', 'chest-imagenome', 'yolov11'))
        bbox_names = CHEST_IMAGENOME_BBOX_NAMES

    elif eval_dataset_name == EvalDatasets.CHEST_IMAGENOME_GOLD:

        gt_bbox_dict = load_chest_imagenome_gold_bboxes()
        gt_bbox_coords = [[None] * len(CHEST_IMAGENOME_BBOX_NAMES) for _ in dicom_ids]
        for i, dicom_id in enumerate(dicom_ids):
            gt_bboxes = gt_bbox_dict[dicom_id]
            coords = gt_bboxes['coords'].reshape(-1, 4)
            presence = gt_bboxes['presence']
            for j in range(len(CHEST_IMAGENOME_BBOX_NAMES)):
                if presence[j]:
                    gt_bbox_coords[i][j] = coords[j].reshape(1, 4)

        valid_classes = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES, dtype=bool)
        _, _gt_presence_indices = get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices()
        valid_classes[_gt_presence_indices] = True

        results_folder_path = get_results_folder_path(get_checkpoint_folder_path('visual_module', 'chest-imagenome', 'yolov11'))
        bbox_names = CHEST_IMAGENOME_BBOX_NAMES
    
    elif eval_dataset_name == EvalDatasets.VINDRCXR_TEST_SET or\
        eval_dataset_name == EvalDatasets.VINDRCXR_TRAIN_SET:

        indices = vinbigdata.test_indices if eval_dataset_name == EvalDatasets.VINDRCXR_TEST_SET else vinbigdata.train_indices

        gt_bbox_coords = [[None] * VINBIG_NUM_BBOX_CLASSES for _ in indices]
        for i, idx in enumerate(indices):
            coords_list, class_list = vinbigdata.bboxes[idx]
            for coords, cls in zip(coords_list, class_list):
                assert len(coords) == 4
                if gt_bbox_coords[i][cls] is None:
                    gt_bbox_coords[i][cls] = [coords]
                else:
                    gt_bbox_coords[i][cls].append(coords)
            for j in range(VINBIG_NUM_BBOX_CLASSES):
                if gt_bbox_coords[i][j] is not None:
                    gt_bbox_coords[i][j] = np.array(gt_bbox_coords[i][j])
                    assert gt_bbox_coords[i][j].shape[1] == 4

        results_folder_path = get_results_folder_path(get_checkpoint_folder_path('visual_module', 'vinbig', 'yolov11'))
        bbox_names = VINBIG_BBOX_NAMES

    else:
        raise ValueError(f'Unknown eval dataset name: {eval_dataset_name}')
    
    metadata = {
        'weights_path': weights_path,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'max_det': max_det,
    }
    save_path_strings = [
        f'conf_{conf_threshold:.2f}',
        f'iou_{iou_threshold:.2f}',
        f'max_det_{max_det}',
    ]
    if decent_images_only:
        save_path_strings.append('decent_images_only')
    compute_and_save_bbox_metrics(
        pred_boxes=pred_boxes,
        pred_classes=pred_classes,
        pred_confs=pred_confs,
        metadata=metadata,
        gt_bbox_coords=gt_bbox_coords,
        bbox_names=bbox_names,
        eval_dataset_name=eval_dataset_name,
        results_folder_path=results_folder_path,
        valid_classes=valid_classes,
        save_predictions=save_predictions,
        save_path_strings=save_path_strings,
        image_paths=image_paths,
    )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    eval_yolov11(**args)