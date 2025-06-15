import itertools
from sklearn.metrics import average_precision_score
import time
import torch
import numpy as np
from torchvision.ops.boxes import nms
from multiprocessing import Pool
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from tqdm import tqdm

from medvqa.utils.logging_utils import print_bold

_VALID_TYPES = [np.ndarray, list, torch.Tensor, tuple]

def compute_bbox_union_iou(bboxes_A, bboxes_B, bbox_format='xyxy'):
    """
    Calculates the exact IoU (Intersection over Union) between two sets of bounding boxes.
    
    Args:
        bboxes_A (list, np.ndarray, or torch.Tensor): First set of bounding boxes (N, 4) or (4,).
        bboxes_B (list, np.ndarray, or torch.Tensor): Second set of bounding boxes (M, 4) or (4,).
        bbox_format (str): Format of bounding boxes, either 'xyxy' or 'cxcywh'. Default is 'xyxy'.
    
    Returns:
        float: IoU value between the two sets of bounding boxes.
    """
    assert type(bboxes_A) in _VALID_TYPES, f"bboxes_A is of type {type(bboxes_A)}"
    assert type(bboxes_B) in _VALID_TYPES, f"bboxes_B is of type {type(bboxes_B)}"
    assert bbox_format in ['xyxy', 'cxcywh'], f"Invalid bbox_format: {bbox_format}. Must be 'xyxy' or 'cxcywh'."

    if len(bboxes_A) == 0 or len(bboxes_B) == 0:
        return 0.0
    
    # Convert bounding boxes to 'xyxy' format if necessary
    if bbox_format == 'cxcywh':
        if type(bboxes_A[0]) in _VALID_TYPES:
            bboxes_A = [cxcywh_to_xyxy_basic(bbox) for bbox in bboxes_A]
        else:
            bboxes_A = cxcywh_to_xyxy_basic(bboxes_A)
        
        if type(bboxes_B[0]) in _VALID_TYPES:
            bboxes_B = [cxcywh_to_xyxy_basic(bbox) for bbox in bboxes_B]
        else:
            bboxes_B = cxcywh_to_xyxy_basic(bboxes_B)
    
    # bboxes_A: (N, 4) or (4,)
    if type(bboxes_A[0]) in _VALID_TYPES:
        polygons_A = [shapely_box(*bbox) for bbox in bboxes_A]
        union_A = unary_union(polygons_A)
    else:
        union_A = shapely_box(*bboxes_A)
    
    # bboxes_B: (N, 4) or (4,)
    if type(bboxes_B[0]) in _VALID_TYPES:
        polygons_B = [shapely_box(*bbox) for bbox in bboxes_B]
        union_B = unary_union(polygons_B)
    else:
        union_B = shapely_box(*bboxes_B)
    
    intersection = union_A.intersection(union_B)
    inter_area = intersection.area
    union_area = union_A.area + union_B.area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def compute_mean_bbox_union_iou(pred_bbox_coords_list, gt_bbox_coords_list, bbox_format='xyxy'):
    assert len(pred_bbox_coords_list) == len(gt_bbox_coords_list)
    n = len(pred_bbox_coords_list)
    mean_iou = 0
    for i in range(n):
        mean_iou += compute_bbox_union_iou(pred_bbox_coords_list[i], gt_bbox_coords_list[i], bbox_format)
    mean_iou /= n
    return mean_iou

def compute_iou(pred, gt):
    assert len(pred) == 4
    assert len(gt) == 4
    # compute intersection over union (IoU)
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(pred[0], gt[0])
    yA = max(pred[1], gt[1])
    xB = min(pred[2], gt[2])
    yB = min(pred[3], gt[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    boxBArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    # return the intersection over union value
    if torch.is_tensor(iou): iou = iou.item()
    return iou

# HACK to make multiprocessing work. Based on https://stackoverflow.com/a/37746961/2801404
_shared_pred_presences = None
_shared_pred_coords = None
_shared_gt_presences = None
_shared_gt_coords = None
_shared_gt_classes = None

def _compute_mean_iou(j):
    mean_iou = 0
    count = 0
    for i in range(_shared_pred_coords.shape[0]):
        if _shared_gt_presences[i, j] == 1:
            mean_iou += compute_iou(_shared_pred_coords[i, j], _shared_gt_coords[i, j])
            count += 1
    mean_iou /= count
    return mean_iou
def compute_mean_iou_per_class(pred_coords, gt_coords, gt_presences, num_workers=5):
    assert len(pred_coords.shape) == 3
    assert len(gt_coords.shape) == 3
    m = pred_coords.shape[1]
    global _shared_pred_coords, _shared_gt_coords, _shared_gt_presences
    _shared_pred_coords = pred_coords
    _shared_gt_coords = gt_coords
    _shared_gt_presences = gt_presences
    task_args = [i for i in range(m)]
    with Pool(num_workers) as p:
        mean_ious = p.map(_compute_mean_iou, task_args)
    return mean_ious

def compute_mean_iou_per_class__detectron2(pred_boxes, pred_classes, scores, gt_coords, gt_presences, valid_classes=None):
    assert len(gt_coords.shape) == 3
    assert len(gt_presences.shape) == 2
    assert gt_coords.shape[-1] == 4 # each bounding box is represented by 4 coordinates
    m = gt_coords.shape[1]
    mean_ious = np.zeros((m,), dtype=np.float32)
    counts = np.zeros((m,), dtype=np.int32)
    n = len(gt_presences)
    for i in range(n):
        for j in range(len(pred_boxes[i])):
            if scores is not None and scores[i][j] < 0.5:
                continue
            cls = pred_classes[i][j].item()
            if gt_presences[i][cls] == 1:
                mean_ious[cls] += compute_iou(pred_boxes[i][j], gt_coords[i][cls])
                counts[cls] += 1
    for i in range(m):
        if counts[i] > 0:
            mean_ious[i] /= counts[i]
    if valid_classes is not None:
        mean_ious = mean_ious[valid_classes]
    return mean_ious

def compute_mean_iou_per_class__yolov8(pred_boxes, pred_classes, gt_coords, gt_presences, valid_classes=None):
    return compute_mean_iou_per_class__detectron2(pred_boxes, pred_classes, None, gt_coords, gt_presences, valid_classes)

def compute_mean_iou_per_class__yolov5(pred_boxes, pred_classes, gt_coords, gt_presences, valid_classes=None):
    return compute_mean_iou_per_class__detectron2(pred_boxes, pred_classes, None, gt_coords, gt_presences, valid_classes)

def compute_mean_iou_per_class__yolov11(pred_boxes, pred_classes, gt_coords, valid_classes=None, compute_iou_per_sample=False,
                                        compute_micro_average_iou=False, return_counts=False):
    assert len(pred_boxes) == len(gt_coords)
    assert len(pred_boxes) == len(pred_classes)
    m = len(gt_coords[0]) # number of classes
    n = len(gt_coords) # number of samples
    class_ious = np.zeros((m,), dtype=np.float32)
    class_counts = np.zeros((m,), dtype=np.int32)
    if compute_iou_per_sample:
        sample_ious = np.zeros((n,), dtype=np.float32)
        sample_counts = np.zeros((n,), dtype=np.int32)
    if compute_micro_average_iou:
        micro_iou = 0
        micro_count = 0
    for i in range(n):
        if pred_classes is None:
            coords_per_class = pred_boxes[i]
        else: # group boxes by class
            coords_per_class = [[] for _ in range(m)]
            assert len(pred_boxes[i]) == len(pred_classes[i])
            for j in range(len(pred_boxes[i])):
                coords_per_class[pred_classes[i][j]].append(pred_boxes[i][j])
        for j in range(m):
            if gt_coords[i][j] is not None and len(gt_coords[i][j]) > 0:
                assert gt_coords[i][j].ndim == 2 # (N, 4)
                iou = compute_bbox_union_iou(gt_coords[i][j], coords_per_class[j])
                class_ious[j] += iou
                class_counts[j] += 1
                if compute_iou_per_sample:
                    sample_ious[i] += iou
                    sample_counts[i] += 1
                if compute_micro_average_iou:
                    micro_iou += iou
                    micro_count += 1

    for i in range(m):
        if class_counts[i] > 0:
            class_ious[i] /= class_counts[i]
    if valid_classes is not None:
        class_ious = class_ious[valid_classes]
    output = { 'class_ious': class_ious }
    if compute_iou_per_sample:
        for i in range(n):
            if sample_counts[i] > 0:
                sample_ious[i] /= sample_counts[i]
        output['sample_ious'] = sample_ious
    if compute_micro_average_iou:
        micro_iou /= micro_count
        output['micro_iou'] = micro_iou
    if return_counts:
        output['class_counts'] = class_counts
        if compute_iou_per_sample:
            output['sample_counts'] = sample_counts
    if len(output) == 1:
        return output['class_ious']
    return output

def compute_mAP__yolov11(gt_coords, pred_boxes=None, pred_classes=None, pred_confs=None,
                         adapted_pred_boxes=None, adapted_pred_confs=None, classifier_confs=None,
                         valid_classes=None, iou_thresholds=[0.5], compute_micro_average=False,
                         use_multiprocessing=True, num_processes=5):
    
    m = len(gt_coords[0]) # number of classes
    n = len(gt_coords) # number of samples
    
    if pred_boxes is not None:
        assert pred_classes is not None
        assert pred_confs is not None
        assert len(pred_boxes) == len(gt_coords)
        assert len(pred_boxes) == len(pred_classes)
        assert len(pred_boxes) == len(pred_confs)

        # Adapt pred_boxes and pred_confs
        adapted_pred_boxes = [[None] * m for _ in range(n)]
        adapted_pred_confs = np.zeros((n, m), dtype=np.float32)
        counts = np.zeros((n, m), dtype=np.int32)
        for i in range(n):
            for j in range(len(pred_boxes[i])):
                c = pred_classes[i][j]
                if adapted_pred_boxes[i][c] is None:
                    adapted_pred_boxes[i][c] = [pred_boxes[i][j]]
                else:
                    adapted_pred_boxes[i][c].append(pred_boxes[i][j])
                adapted_pred_confs[i,c] += pred_confs[i][j] # sum confidences
                counts[i,c] += 1
        counts[counts == 0] = 1 # avoid division by zero
        adapted_pred_confs /= counts
        if classifier_confs is not None:
            assert classifier_confs.shape == (n, m)
            adapted_pred_confs += classifier_confs
            adapted_pred_confs /= 2 # average
    else:
        assert adapted_pred_boxes is not None
        assert adapted_pred_confs is not None
        assert len(adapted_pred_boxes) == len(gt_coords)
        assert len(adapted_pred_boxes) == len(adapted_pred_confs)

    global _shared_pred_boxes, _shared_pred_confs, _shared_gt_coords
    _shared_pred_boxes = adapted_pred_boxes
    _shared_pred_confs = adapted_pred_confs
    _shared_gt_coords = gt_coords

    task_args = []
    for iou_thr in iou_thresholds:
        for c in range(m):
            if valid_classes is not None and valid_classes[c] == 0:
                continue
            task_args.append((iou_thr, c, n))
    if use_multiprocessing:
        with Pool(num_processes) as p:
            aps = p.map(_compute_ap__yolov11, task_args)
    else:
        aps = [_compute_ap__yolov11(task) for task in task_args]
    if valid_classes is not None:
        aps = np.array(aps).reshape((len(iou_thresholds), np.sum(valid_classes)))
    else:
        aps = np.array(aps).reshape((len(iou_thresholds), m))

    if compute_micro_average:
        task_args = [(iou_thr, n, m) for iou_thr in iou_thresholds]
        if use_multiprocessing:
            with Pool(num_processes) as p:
                micro_aps = p.map(_compute_ap_micro__yolov11, task_args)
        else:
            micro_aps = [_compute_ap_micro__yolov11(task) for task in task_args]
        micro_aps = np.array(micro_aps)

        return {
            'class_aps': aps,
            'micro_aps': micro_aps,
        }

    return aps

def _compute_ap__yolov11(task):
    iou_thr, c, n = task
    y_true = np.zeros((n,), dtype=np.int32)
    y_scores = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        if _shared_gt_coords[i][c] is not None and len(_shared_gt_coords[i][c]) > 0:
            y_true[i] = 1
        if _shared_pred_boxes[i][c] is not None:
            if y_true[i]:
                if compute_bbox_union_iou(_shared_pred_boxes[i][c], _shared_gt_coords[i][c]) >= iou_thr:
                    y_scores[i] = _shared_pred_confs[i][c]
            else:
                y_scores[i] = _shared_pred_confs[i][c]
    return average_precision_score(y_true, y_scores)

def _compute_ap_micro__yolov11(task):
    iou_thr, num_samples, num_classes = task
    y_true = np.zeros((num_samples, num_classes), dtype=np.int32)
    y_scores = np.zeros((num_samples, num_classes), dtype=np.float32)
    for i in range(num_samples):
        for j in range(num_classes):
            if _shared_gt_coords[i][j] is not None and len(_shared_gt_coords[i][j]) > 0:
                y_true[i, j] = 1
            if _shared_pred_boxes[i][j] is not None:
                if y_true[i, j]:
                    if compute_bbox_union_iou(_shared_pred_boxes[i][j], _shared_gt_coords[i][j]) >= iou_thr:
                        y_scores[i, j] = _shared_pred_confs[i][j]
                else:
                    y_scores[i, j] = _shared_pred_confs[i][j]
    return average_precision_score(y_true.ravel(), y_scores.ravel()) # micro average

def compute_mae_per_class(pred_coords, gt_coords, gt_presences):
    assert len(pred_coords.shape) == 3
    assert len(gt_coords.shape) == 3
    m = pred_coords.shape[1]
    mae = np.zeros((m,), dtype=np.float32)
    for i in range(m):
        if gt_presences[:, i].sum() > 0:
            mae[i] = np.abs(pred_coords[:, i] - gt_coords[:, i])[gt_presences[:, i] == 1].mean()
    return mae

def compute_mae_per_class__detectron2(pred_boxes, pred_classes, scores, gt_coords, gt_presences, valid_classes=None,
                                      abs_fn=torch.abs):
    assert len(gt_coords.shape) == 3
    assert len(gt_presences.shape) == 2
    assert gt_coords.shape[-1] == 4 # each bounding box is represented by 4 coordinates
    m = gt_coords.shape[1]
    mae = np.zeros((m,), dtype=np.float32)
    counts = np.zeros((m,), dtype=np.int32)
    n = len(gt_presences)
    for i in range(n):
        for j in range(len(pred_boxes[i])):
            if scores is not None and scores[i][j] < 0.5:
                continue
            cls = pred_classes[i][j].item()
            if gt_presences[i][cls] == 1:
                ae = abs_fn(pred_boxes[i][j] - gt_coords[i][cls])
                mae[cls] += ae.mean()
                counts[cls] += 1
    for i in range(m):
        if counts[i] > 0:
            mae[i] /= counts[i]
    if valid_classes is not None:
        mae = mae[valid_classes]
    return mae

def compute_mae_per_class__yolov8(pred_boxes, pred_classes, gt_coords, gt_presences, valid_classes=None, abs_fn=np.abs):
    return compute_mae_per_class__detectron2(pred_boxes, pred_classes, None, gt_coords, gt_presences, valid_classes, abs_fn)

def compute_mae_per_class__yolov5(pred_boxes, pred_classes, gt_coords, gt_presences, valid_classes=None, abs_fn=np.abs):
    return compute_mae_per_class__detectron2(pred_boxes, pred_classes, None, gt_coords, gt_presences, valid_classes, abs_fn)

def _compute_score(task, metric_fn):
    iou_thrs, c, n = task
    tp, fp, fn = 0, 0, 0
    if type(_shared_pred_presences) == list:
        for i in range(n):
            if _shared_pred_presences[i][c] > 0:
                if _shared_gt_presences[i][c] == 1:
                    if compute_iou(_shared_pred_coords[i][c], _shared_gt_coords[i][c]) >= iou_thrs:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                else:
                    fp += 1
            else:
                if _shared_gt_presences[i][c] == 1:
                    fn += 1
    else: # numpy array or torch tensor -> use comma indexing
        for i in range(n):
            if _shared_pred_presences[i, c] > 0:
                if _shared_gt_presences[i, c] == 1:
                    if compute_iou(_shared_pred_coords[i, c], _shared_gt_coords[i, c]) >= iou_thrs:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                else:
                    fp += 1
            else:
                if _shared_gt_presences[i, c] == 1:
                    fn += 1
    return metric_fn(tp, fp, fn)

def _precision_score(tp, fp, fn):
    return tp / (tp + fp) if tp + fp > 0 else 0
def _recall_score(tp, fp, fn):
    return tp / (tp + fn) if tp + fn > 0 else 0
def _f1_score(tp, fp, fn):
    p = _precision_score(tp, fp, fn)
    r = _recall_score(tp, fp, fn)
    return 2 * p * r / (p + r) if p + r > 0 else 0
def _prf1_scores(tp, fp, fn):
    p = _precision_score(tp, fp, fn)
    r = _recall_score(tp, fp, fn)
    f1 = _f1_score(tp, fp, fn)
    return p, r, f1

def _compute_f1(task):
    return _compute_score(task, _f1_score)
def _compute_precision(task):
    return _compute_score(task, _precision_score)
def _compute_recall(task):
    return _compute_score(task, _recall_score)
def _compute_prf1(task):
    return _compute_score(task, _prf1_scores)

def _compute_multiple_scores(pred_coords, pred_presences, gt_coords, gt_presences, iou_thresholds, num_workers, metric_fn):
    global _shared_pred_coords, _shared_pred_presences, _shared_gt_coords, _shared_gt_presences
    _shared_pred_coords = pred_coords
    _shared_pred_presences = pred_presences
    _shared_gt_coords = gt_coords
    _shared_gt_presences = gt_presences
    num_classes = len(gt_presences[0])
    num_samples = len(gt_presences)
    task_args = []
    for iou_thr in iou_thresholds:
        for c in range(num_classes):
            task_args.append((iou_thr, c, num_samples))
    with Pool(num_workers) as p:
        scores = p.map(metric_fn, task_args)
    if type(scores[0]) == tuple:
        scores = np.array(scores).reshape((len(iou_thresholds), num_classes, len(scores[0])))
    else:
        scores = np.array(scores).reshape((len(iou_thresholds), num_classes))
    return scores

def compute_multiple_f1_scores(pred_coords, pred_presences, gt_coords, gt_presences, iou_thresholds, num_workers=5):
    return _compute_multiple_scores(pred_coords, pred_presences, gt_coords, gt_presences,
                                    iou_thresholds, num_workers, _compute_f1)

def compute_multiple_precision_scores(pred_coords, pred_presences, gt_coords, gt_presences, iou_thresholds, num_workers=5):
    return _compute_multiple_scores(pred_coords, pred_presences, gt_coords, gt_presences,
                                    iou_thresholds, num_workers, _compute_precision)

def compute_multiple_recall_scores(pred_coords, pred_presences, gt_coords, gt_presences, iou_thresholds, num_workers=5):
    return _compute_multiple_scores(pred_coords, pred_presences, gt_coords, gt_presences,
                                    iou_thresholds, num_workers, _compute_recall)

def compute_multiple_prf1_scores(pred_coords, pred_presences, gt_coords, gt_presences, iou_thresholds, num_workers=5):
    return _compute_multiple_scores(pred_coords, pred_presences, gt_coords, gt_presences,
                                    iou_thresholds, num_workers, _compute_prf1)

_shared_pred_boxes = None
_shared_pred_classes = None
_shared_pred_confs = None
_shared_scores = None

def _compute_score__detectron2(task, metric_fn):
    iou_thr, c, num_samples = task
    tp, fp, fn = 0, 0, 0
    for i in range(num_samples):
        match_found = False
        for j in range(len(_shared_pred_classes[i])):
            if _shared_pred_classes[i][j] == c:
                if _shared_scores is None or _shared_scores[i][j] > 0.5:
                    if _shared_gt_presences[i][c] == 1:
                        # try:
                        if compute_iou(_shared_pred_boxes[i][j], _shared_gt_coords[i][c]) >= iou_thr:
                            tp += 1
                            match_found = True
                        else:
                            fp += 1
                        # except RuntimeError:
                        #     print('_shared_pred_boxes[i]', _shared_pred_boxes[i])
                        #     print('_shared_gt_coords[i][a:b]', _shared_gt_coords[i][c])
                        #     print('_shared_pred_classes[i]', _shared_pred_classes[i])
                        #     print('_shared_scores[i]', _shared_scores[i])
                        #     print('_shared_gt_presences[i]', _shared_gt_presences[i])
                        #     raise
                    else:
                        fp += 1
        if _shared_gt_presences[i][c] == 1 and not match_found:
            fn += 1
    return metric_fn(tp, fp, fn)

def _compute_f1__detectron2(task):
    return _compute_score__detectron2(task, _f1_score)
def _compute_precision__detectron2(task):
    return _compute_score__detectron2(task, _precision_score)
def _compute_recall__detectron2(task):
    return _compute_score__detectron2(task, _recall_score)
def _compute_prf1__detectron2(task):
    return _compute_score__detectron2(task, _prf1_scores)

def _compute_multiple_scores__detectron2(pred_boxes, pred_classes, scores, gt_coords, gt_presences, iou_thresholds,
                                         valid_classes, num_workers, metric_fn):
    global _shared_pred_boxes, _shared_pred_classes, _shared_scores, _shared_gt_coords, _shared_gt_presences
    _shared_pred_boxes = pred_boxes
    _shared_pred_classes = pred_classes
    _shared_scores = scores
    _shared_gt_coords = gt_coords
    _shared_gt_presences = gt_presences
    num_classes = len(gt_presences[0])
    num_samples = len(gt_presences)
    task_args = []
    for iou_thr in iou_thresholds:
        for c in range(num_classes):
            if valid_classes is not None and valid_classes[c] == 0:
                continue
            task_args.append((iou_thr, c, num_samples))
    with Pool(num_workers) as p:
        scores = p.map(metric_fn, task_args)
    if valid_classes is not None:
        actual_num_classes = np.sum(valid_classes)
    else:
        actual_num_classes = num_classes
    if type(scores[0]) == tuple:
        scores = np.array(scores).reshape((len(iou_thresholds), actual_num_classes, len(scores[0])))
    else:
        scores = np.array(scores).reshape((len(iou_thresholds), actual_num_classes))
    return scores

def compute_multiple_f1_scores__detectron2(
    pred_boxes, pred_classes, scores, gt_coords, gt_presences, iou_thresholds, num_workers=5):
    return _compute_multiple_scores__detectron2(
        pred_boxes, pred_classes, scores, gt_coords, gt_presences, iou_thresholds, None, num_workers, _compute_f1__detectron2)

def compute_multiple_precision_scores__detectron2(
    pred_boxes, pred_classes, scores, gt_coords, gt_presences, iou_thresholds, num_workers=5):
    return _compute_multiple_scores__detectron2(
        pred_boxes, pred_classes, scores, gt_coords, gt_presences, iou_thresholds, None, num_workers, _compute_precision__detectron2)

def compute_multiple_recall_scores__detectron2(
    pred_boxes, pred_classes, scores, gt_coords, gt_presences, iou_thresholds, num_workers=5):
    return _compute_multiple_scores__detectron2(
        pred_boxes, pred_classes, scores, gt_coords, gt_presences, iou_thresholds, None, num_workers, _compute_recall__detectron2)

def compute_multiple_prf1_scores__detectron2(
    pred_boxes, pred_classes, scores, gt_coords, gt_presences, iou_thresholds, valid_classes=None, num_workers=5):
    return _compute_multiple_scores__detectron2(
        pred_boxes, pred_classes, scores, gt_coords, gt_presences, iou_thresholds, valid_classes, num_workers, _compute_prf1__detectron2)

def compute_multiple_prf1_scores__yolov5(
    pred_boxes, pred_classes, gt_coords, gt_presences, iou_thresholds, valid_classes=None, num_workers=5):
    return compute_multiple_prf1_scores__detectron2(
        pred_boxes, pred_classes, None, gt_coords, gt_presences, iou_thresholds, valid_classes, num_workers)

def compute_multiple_prf1_scores__yolov8(
    pred_boxes, pred_classes, gt_coords, gt_presences, iou_thresholds, valid_classes=None, num_workers=5):
    return compute_multiple_prf1_scores__detectron2(
        pred_boxes, pred_classes, None, gt_coords, gt_presences, iou_thresholds, valid_classes, num_workers)

def compute_multiple_f1_scores__yolov8(
    pred_boxes, pred_classes, gt_coords, iou_thresholds, gt_presences=None, gt_classes=None, num_classes=None, num_workers=5):
    assert (gt_presences is None) != (gt_classes is None) # xor
    if gt_presences is not None:
        return _compute_multiple_scores__detectron2(
            pred_boxes, pred_classes, None, gt_coords, gt_presences, iou_thresholds, None, num_workers, _compute_f1__detectron2)
    else:
        assert num_classes is not None
        return _compute_multiple_scores__v2(
            pred_boxes, pred_classes, gt_coords, gt_classes, num_classes, iou_thresholds, num_workers, _compute_f1__v2)

def _compute_multiple_scores__v2(pred_boxes, pred_classes, gt_coords, gt_classes, num_classes, iou_thresholds,
                                 num_workers, metric_fn):
    global _shared_pred_boxes, _shared_pred_classes, _shared_gt_coords, _shared_gt_classes
    _shared_pred_boxes = pred_boxes
    _shared_pred_classes = pred_classes
    _shared_gt_coords = gt_coords
    _shared_gt_classes = gt_classes
    num_samples = len(gt_classes)
    task_args = []
    for iou_thr in iou_thresholds:
        for c in range(num_classes):
            task_args.append((iou_thr, c, num_samples))
    with Pool(num_workers) as p:
        scores = p.map(metric_fn, task_args)
    if type(scores[0]) == tuple:
        scores = np.array(scores).reshape((len(iou_thresholds), num_classes, len(scores[0])))
    else:
        scores = np.array(scores).reshape((len(iou_thresholds), num_classes))
    return scores

def _compute_f1__v2(task):
    return _compute_score__v2(task, _f1_score)

def _compute_score__v2(task, metric_fn):
    iou_thr, c, num_samples = task
    tp, fp, fn = 0, 0, 0
    for i in range(num_samples):
        pred_idxs = [j for j in range(len(_shared_pred_classes[i])) if _shared_pred_classes[i][j] == c]
        gt_idxs = [j for j in range(len(_shared_gt_classes[i])) if _shared_gt_classes[i][j] == c]
        for a in pred_idxs:
            match_found = False
            for b in gt_idxs:
                if compute_iou(_shared_pred_boxes[i][a], _shared_gt_coords[i][b]) >= iou_thr:
                    tp += 1
                    match_found = True
                    break
            if not match_found:
                fp += 1
        for b in gt_idxs:
            match_found = False
            for a in pred_idxs:
                if compute_iou(_shared_pred_boxes[i][a], _shared_gt_coords[i][b]) >= iou_thr:
                    match_found = True
                    break
            if not match_found:
                fn += 1
    return metric_fn(tp, fp, fn)


def _nms_method1(bbox_coords, bbox_probs, conf_th, iou_th, max_det_per_class): # slower
    coords_list = []
    probs_list = []
    classes_list = []
    for j in range(bbox_coords.size(0)): # for each class
        coords = bbox_coords[j] # (num_boxes, 4)
        probs = bbox_probs[j] # (num_boxes,)
        mask = probs > conf_th
        coords = coords[mask]
        probs = probs[mask]
        if coords.size(0) == 0:
            continue
        if coords.size(0) > max_det_per_class:
            probs, idxs = torch.topk(probs, max_det_per_class)
            coords = coords[idxs]
        keep = nms(coords, probs, iou_th)
        coords_list.append(coords[keep].cpu().numpy())
        probs_list.append(probs[keep].cpu().numpy())
        classes_list.append(np.full_like(probs_list[-1], j, dtype=np.int32))
    pred_boxes = np.concatenate(coords_list, axis=0) if len(coords_list) > 0 else np.empty((0, 4), dtype=np.float32)
    pred_confs = np.concatenate(probs_list, axis=0) if len(probs_list) > 0 else np.empty((0,), dtype=np.float32)
    pred_classes = np.concatenate(classes_list, axis=0) if len(classes_list) > 0 else np.empty((0,), dtype=np.int32)
    return pred_boxes, pred_confs, pred_classes

def _nms_method2(bbox_coords, bbox_probs, class_ids, conf_th, iou_th,
                 max_det_per_class, sort_confidence=False, bbox_format='xyxy'):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes with an optional sorting
    based on confidence scores.

    This function limits the number of detections per class before thresholding and
    applying NMS. After NMS, the output detections can optionally be sorted in the
    order of decreasing confidence.

    Parameters:
        bbox_coords (torch.Tensor): Tensor of bounding box coordinates with shape
            (num_batches, num_detections, 4).
        bbox_probs (torch.Tensor): Tensor of detection confidence scores with shape
            (num_batches, num_detections).
        class_ids (torch.Tensor): Tensor of class IDs for each detection with shape
            (num_batches, num_detections).
        conf_th (float): Confidence threshold. Boxes with confidence lower than this value
            will be discarded.
        iou_th (float): Intersection-over-Union (IoU) threshold used for NMS.
        max_det_per_class (int): Maximum number of detections to keep per class before
            thresholding and NMS.
        sort_confidence (bool, optional): If True, the final output detections are sorted
            by confidence in decreasing order. Default is False.
        bbox_format (str, optional): Format of the bounding boxes. Can be 'xyxy' or 'xywh'.

    Returns:
        tuple: A tuple containing:
            - pred_boxes (numpy.ndarray): The filtered bounding boxes, shape (N, 4).
            - pred_confs (numpy.ndarray): The confidence scores for the final boxes, shape (N,).
            - pred_classes (numpy.ndarray): The class IDs corresponding to the final boxes,
              shape (N,).
    """
    # Limit detections per class by taking the top-k detections based on confidence.
    if bbox_coords.size(1) > max_det_per_class:
        bbox_probs, idxs = torch.topk(bbox_probs, max_det_per_class, dim=1)
        # Gather the corresponding bounding boxes using index expansion.
        bbox_coords = torch.gather(bbox_coords, 1, idxs.unsqueeze(-1).expand(-1, -1, 4))
        # Gather the corresponding class IDs.
        class_ids = torch.gather(class_ids, 1, idxs)

    # Apply confidence threshold filtering on the detections.
    mask = bbox_probs > conf_th
    bbox_coords = bbox_coords[mask]
    bbox_probs = bbox_probs[mask]
    class_ids = class_ids[mask]

    # Run Non-Maximum Suppression (NMS) if there are any remaining detections.
    if bbox_coords.size(0) > 0:
        # Shift bounding box coordinates based on class IDs to separate boxes
        # of different classes during NMS.
        if bbox_format == 'cxcywh':
            bbox_coords_xyxy = cxcywh_to_xyxy_tensor(bbox_coords) # Convert to 'xyxy' format
        else: # 'xyxy'
            bbox_coords_xyxy = bbox_coords # Already in 'xyxy' format
        shifted_bbox_coords = bbox_coords_xyxy + class_ids.unsqueeze(-1).float() * 10.0
        keep = nms(shifted_bbox_coords, bbox_probs, iou_th)
        # Retrieve the final detections after NMS.
        pred_boxes = bbox_coords[keep].cpu().numpy()
        pred_confs = bbox_probs[keep].cpu().numpy()
        pred_classes = class_ids[keep].cpu().numpy()
    else:
        # No detections remained; return empty arrays.
        pred_boxes = np.empty((0, 4), dtype=np.float32)
        pred_confs = np.empty((0,), dtype=np.float32)
        pred_classes = np.empty((0,), dtype=np.int32)

    # Optionally sort the output detections by confidence in decreasing order.
    if sort_confidence and pred_confs.size > 0:
        sort_idx = pred_confs.argsort()[::-1]
        pred_boxes = pred_boxes[sort_idx]
        pred_confs = pred_confs[sort_idx]
        pred_classes = pred_classes[sort_idx]

    return pred_boxes, pred_confs, pred_classes

def find_optimal_conf_iou_thresholds(gt_coords_list, yolo_predictions_list=None, is_fact_conditioned_yolo=False,
                                     resized_shape_list=None, pred_boxes_list=None, pred_confs_list=None, classifier_confs=None,
                                     iou_thresholds=np.arange(0.05, 0.6, 0.05), conf_thresholds=np.arange(0.05, 0.5, 0.05),
                                     max_det=100, max_det_per_class=20, num_classes=None, verbose=True):
    n = len(gt_coords_list)
    use_yolo_predictions = yolo_predictions_list is not None
    if use_yolo_predictions:
        assert resized_shape_list is not None # needed to normalize the boxes from yolo predictions
        assert pred_boxes_list is None
        assert pred_confs_list is None
    else:
        assert pred_boxes_list is not None
        assert pred_confs_list is not None
        assert len(pred_boxes_list) == n
        assert len(pred_confs_list) == n
    if is_fact_conditioned_yolo:
        assert num_classes is not None

    import itertools

    if verbose:
        print_bold("Finding optimal conf and iou thresholds")

    best_score = -1e9
    best_iou_threshold = None
    best_conf_threshold = None
    best_pred_boxes_list = None
    best_pred_classes_list = None
    best_pred_confs_list = None
    
    if use_yolo_predictions:
        
        from ultralytics.utils.ops import non_max_suppression
        
        # Consider all pairs of conf_threshold and iou_threshold
        for conf_th, iou_th in itertools.product(conf_thresholds, iou_thresholds):
            start = time.time()
            BIG_ENOUGH = 1000000
            pred_boxes_list = [None] * BIG_ENOUGH
            pred_confs_list = [None] * BIG_ENOUGH
            pred_classes_list = [None] * BIG_ENOUGH
            idx = 0
            if is_fact_conditioned_yolo:
                for yolo_predictions, resized_shapes in zip(yolo_predictions_list, resized_shape_list):
                    assert yolo_predictions.ndim == 3, yolo_predictions.shape # (batch_size * num_classes, 1 + 4, num_boxes)
                    yolo_predictions = yolo_predictions.detach().clone()
                    preds = non_max_suppression(yolo_predictions, conf_thres=conf_th, iou_thres=iou_th, max_det=max_det)
                    assert isinstance(preds, list) and len(preds) % num_classes == 0
                    batch_size = len(preds) // num_classes
                    assert batch_size == len(resized_shapes)
                    for i  in range(batch_size):
                        preds_i = preds[i*num_classes:(i+1)*num_classes]
                        rs = resized_shapes[i]
                        boxes_list = []
                        confs_list = []
                        classes_list = []
                        for j in range(num_classes):
                            pred = preds_i[j]
                            assert pred.ndim == 2 # (num_boxes, 6)
                            assert pred.size(1) == 6
                            pred = pred.cpu().numpy()
                            boxes_list.append(pred[:, :4] / np.array([rs[1], rs[0], rs[1], rs[0]], dtype=np.float32))
                            confs_list.append(pred[:, 4])
                            classes_list.append(np.full_like(pred[:, 5], j, dtype=np.int32))
                        pred_boxes_list[idx] = np.concatenate(boxes_list, axis=0)
                        pred_confs_list[idx] = np.concatenate(confs_list, axis=0)
                        pred_classes_list[idx] = np.concatenate(classes_list, axis=0)
                        idx += 1
            else:
                for yolo_predictions, resized_shapes in zip(yolo_predictions_list, resized_shape_list):
                    assert isinstance(yolo_predictions, torch.Tensor)
                    assert yolo_predictions.ndim == 3 # (batch_size, num_classes + 4, num_boxes)
                    yolo_predictions = yolo_predictions.detach().clone()
                    preds = non_max_suppression(yolo_predictions, conf_thres=conf_th, iou_thres=iou_th, max_det=max_det)
                    for pred, rs in zip(preds, resized_shapes):
                        assert pred.ndim == 2 # (num_boxes, 6)
                        assert pred.size(1) == 6
                        assert len(rs) == 2 # (height, width)
                        pred = pred.cpu().numpy()
                        pred_boxes_list[idx] = pred[:, :4] / np.array([rs[1], rs[0], rs[1], rs[0]], dtype=np.float32)
                        pred_confs_list[idx] = pred[:, 4]
                        pred_classes_list[idx] = pred[:, 5].astype(int)
                        idx += 1
            pred_boxes_list = pred_boxes_list[:idx]
            pred_confs_list = pred_confs_list[:idx]
            pred_classes_list = pred_classes_list[:idx]
            assert idx == n
            # score = compute_mean_iou_per_class__yolov11(pred_boxes=pred_boxes_list,
            #                                             pred_classes=pred_classes_list, gt_coords=gt_coords_list).mean()
            time_before_map = time.time()
            score = compute_mAP__yolov11(gt_coords=gt_coords_list, pred_boxes=pred_boxes_list, pred_classes=pred_classes_list,
                                            pred_confs=pred_confs_list, classifier_confs=classifier_confs,
                                            iou_thresholds=[0.05, 0.4], num_processes=1).mean()
            time_after_map = time.time()
            if verbose:
                print(f"conf_th={conf_th}, iou_th={iou_th}, mAP={score} (time_input_processing={time_before_map - start}, time_map={time_after_map - time_before_map})")
            if score > best_score:
                best_score = score
                best_iou_threshold = iou_th
                best_conf_threshold = conf_th
                best_pred_boxes_list = pred_boxes_list
                best_pred_classes_list = pred_classes_list
                best_pred_confs_list = pred_confs_list

    else:

        classes_tensor = None

        # Consider all pairs of conf_threshold and iou_threshold
        for conf_th, iou_th in itertools.product(conf_thresholds, iou_thresholds):
            start = time.time()
            pred_boxes_list_ = [None] * n
            pred_confs_list_ = [None] * n
            pred_classes_list_ = [None] * n
            for i, (bbox_coords, bbox_probs) in enumerate(zip(pred_boxes_list, pred_confs_list)):

                if classes_tensor is None:
                    classes_tensor = torch.arange(bbox_coords.size(0), device=bbox_coords.device).unsqueeze(-1).expand(-1, bbox_coords.size(1)).contiguous()

                class_ids = classes_tensor

                assert bbox_coords.ndim == 3 # (num_classes, num_boxes, 4)
                assert bbox_probs.ndim == 3 # (num_classes, num_boxes, 1)
                bbox_probs = bbox_probs.squeeze(-1) # (num_classes, num_boxes)

                pred_boxes, pred_confs, pred_classes = _nms_method2(bbox_coords, bbox_probs, class_ids, conf_th, iou_th, max_det_per_class)
                pred_boxes_list_[i] = pred_boxes
                pred_confs_list_[i] = pred_confs
                pred_classes_list_[i] = pred_classes

            # score = compute_mean_iou_per_class__yolov11(pred_boxes=pred_boxes_list_,
            #                                             pred_classes=pred_classes_list_, gt_coords=gt_coords_list).mean()
            time_before_map = time.time()
            score = compute_mAP__yolov11(gt_coords=gt_coords_list, pred_boxes=pred_boxes_list_, pred_classes=pred_classes_list_,
                                         pred_confs=pred_confs_list_, classifier_confs=classifier_confs,
                                         iou_thresholds=[0.05, 0.4], num_processes=1).mean()
            time_after_map = time.time()
            
            if verbose:
                print(f"conf_th={conf_th}, iou_th={iou_th}, mAP={score} (time_input_processing={time_before_map - start}, time_map={time_after_map - time_before_map})")
            if score > best_score:
                best_score = score
                best_iou_threshold = iou_th
                best_conf_threshold = conf_th
                best_pred_boxes_list = pred_boxes_list_
                best_pred_classes_list = pred_classes_list_
                best_pred_confs_list = pred_confs_list_
        
    return {
        'best_iou_threshold': best_iou_threshold,
        'best_conf_threshold': best_conf_threshold,
        'pred_boxes_list': best_pred_boxes_list,
        'pred_classes_list': best_pred_classes_list,
        'pred_confs_list': best_pred_confs_list
    }

def find_optimal_conf_iou_max_det_thresholds__single_class(
    gt_coords_list, pred_boxes_list, pred_confs_list, 
    bbox_format='xyxy',
    iou_thresholds=np.arange(0.05, 0.6, 0.05),
    conf_thresholds=np.arange(0.1, 0.9, 0.05), 
    post_nms_max_dets=[1, 2, 3, 4],
    pre_nms_max_dets=[20, 50, 100],
    verbose=True
):
    """
    Finds the optimal confidence and IoU thresholds for a single class by maximizing the mean IoU score.

    Args:
        gt_coords_list (list): List of ground-truth bounding box coordinates per sample.
        pred_boxes_list (list): List of predicted bounding box coordinates per sample.
        pred_confs_list (list): List of confidence scores for predicted bounding boxes per sample.
        bbox_format (str): Format of the bounding box coordinates. Either 'xyxy' or 'cxcywh'.
        iou_thresholds (numpy.ndarray): Array of IoU thresholds to test.
        conf_thresholds (numpy.ndarray): Array of confidence thresholds to test.
        post_nms_max_dets (list): List of maximum detections to keep after NMS.
        pre_nms_max_dets (list): List of maximum detections to keep before NMS.
        verbose (bool): If True, prints progress information.

    Returns:
        dict: A dictionary containing the best IoU threshold, confidence threshold, and the corresponding filtered predictions.
    """
    n = len(gt_coords_list)
    
    # Ensure the predicted boxes and confidence lists have the same length as ground truth
    assert len(pred_boxes_list) == n
    assert len(pred_confs_list) == n
    
    # Ensure the bounding box format is valid
    assert bbox_format in ['xyxy', 'cxcywh']

    # Convert to PyTorch tensors if not already
    if not isinstance(pred_boxes_list[0], torch.Tensor):
        pred_boxes_list = [torch.tensor(pred_boxes, dtype=torch.float32) for pred_boxes in pred_boxes_list]
    if not isinstance(pred_confs_list[0], torch.Tensor):
        pred_confs_list = [torch.tensor(pred_confs, dtype=torch.float32) for pred_confs in pred_confs_list]

    if verbose:
        print("Finding optimal confidence and IoU thresholds")

    best_score = -float("inf")  # Initialize best IoU score
    best_iou_threshold = None
    best_conf_threshold = None
    best_pre_nms_max_det = None
    best_post_nms_max_det = None
    best_pred_boxes_list = None
    best_pred_confs_list = None

    classes_tensor = None  # To store class indices

    # Iterate over all combinations of confidence and IoU thresholds
    if verbose:
        iterable = itertools.product(conf_thresholds, iou_thresholds, pre_nms_max_dets)
    else:
        iterable = tqdm(itertools.product(conf_thresholds, iou_thresholds, pre_nms_max_dets),
                        total=len(conf_thresholds) * len(iou_thresholds) * len(pre_nms_max_dets),
                        desc="Optimizing thresholds", mininterval=2.0)
    for conf_th, iou_th, pre_nms_max_det in iterable:
        
        # Store filtered predictions per sample
        pred_boxes_list_ = [None] * n
        pred_confs_list_ = [None] * n

        for i, (bbox_coords, bbox_probs) in enumerate(zip(pred_boxes_list, pred_confs_list)):
            assert bbox_coords.ndim == 3  # Expected shape: (H, W, 4)
            assert bbox_probs.ndim == 2  # Expected shape: (H, W)
            
            H, W = bbox_probs.shape
            
            # Flatten spatial dimensions and add a class dimension
            bbox_coords = bbox_coords.view(1, H * W, 4)  # Shape: (1, H*W, 4)
            bbox_probs = bbox_probs.view(1, H * W)  # Shape: (1, H*W)

            # Initialize class tensor once
            if classes_tensor is None:
                classes_tensor = torch.arange(bbox_coords.size(0), device=bbox_coords.device)
                classes_tensor = classes_tensor.unsqueeze(-1).expand(-1, bbox_coords.size(1)).contiguous()

            class_ids = classes_tensor

            # Apply NMS filtering
            pred_boxes, pred_confs, _ = _nms_method2(bbox_coords, bbox_probs, class_ids, conf_th, iou_th, pre_nms_max_det,
                                                     sort_confidence=True, bbox_format=bbox_format)
            pred_boxes_list_[i] = pred_boxes  # Filtered boxes (num_boxes, 4)
            pred_confs_list_[i] = pred_confs  # Filtered confidence scores (num_boxes,)
        
        # Compute mean IoU score for the current threshold combination
        for post_nms_max_det in post_nms_max_dets:
            truncated_pred_boxes_list = [pred_boxes[:post_nms_max_det] for pred_boxes in pred_boxes_list_]
            score = compute_mean_bbox_union_iou(truncated_pred_boxes_list, gt_coords_list, bbox_format=bbox_format)
            
            # Update the best parameters if the current score is better
            if score > best_score:
                truncated_pred_confs_list = [pred_confs[:post_nms_max_det] for pred_confs in pred_confs_list_]
                best_score = score
                best_iou_threshold = iou_th
                best_conf_threshold = conf_th
                best_pre_nms_max_det = pre_nms_max_det
                best_post_nms_max_det = post_nms_max_det
                best_pred_boxes_list = truncated_pred_boxes_list
                best_pred_confs_list = truncated_pred_confs_list

            if verbose:
                print(f"conf_th={conf_th:.2f}, iou_th={iou_th:.2f}, pre_nms_max_det={pre_nms_max_det}, "
                      f"post_nms_max_det={post_nms_max_det}, mIoU={score:.4f}")

    # Return best thresholds and corresponding filtered predictions
    return {
        'best_iou_threshold': best_iou_threshold,
        'best_conf_threshold': best_conf_threshold,
        'best_pre_nms_max_det': best_pre_nms_max_det,
        'best_post_nms_max_det': best_post_nms_max_det,
        'pred_boxes_list': best_pred_boxes_list,
        'pred_confs_list': best_pred_confs_list
    }

def compute_probability_map_iou(prob_map, gt_bboxes, conf_th, bbox_format='xyxy'):
    """
    Compute the Intersection over Union (IoU) between the predicted probability map and ground truth bounding boxes.

    Parameters:
    - prob_map (numpy.ndarray): A 2D array (height, width) representing the probability map.
    - gt_bboxes (numpy.ndarray): An array of shape (num_boxes, 4) containing normalized ground truth bounding boxes.
    - conf_th (float): Confidence threshold for binarizing the probability map.
    - bbox_format (str): Format of the bounding box coordinates. Either 'xyxy' or 'cxcywh'.

    Returns:
    - float: The computed IoU between the binary probability map and the ground truth bounding boxes.
    """
    assert prob_map.ndim == 2, "Probability map must be a 2D array"
    height, width = prob_map.shape
    binary_map = prob_map > conf_th
    
    # Generate predicted bounding boxes based on the binary probability map
    pred_boxes = []
    cell_w, cell_h = 1 / width, 1 / height
    for y in range(height):
        for x in range(width):
            if binary_map[y, x]:
                x_min, y_min = x * cell_w, y * cell_h
                x_max, y_max = (x + 1) * cell_w, (y + 1) * cell_h
                pred_boxes.append(shapely_box(x_min, y_min, x_max, y_max))
    
    if not pred_boxes:
        return 0.0  # No predicted bounding boxes, IoU is 0
    
    # Compute union of predicted bounding boxes
    pred_union = unary_union(pred_boxes)
    
    # Compute union of ground truth bounding boxes
    if bbox_format == 'cxcywh':
        gt_bboxes = [(
            cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        ) for cx, cy, w, h in gt_bboxes]
    elif bbox_format != 'xyxy':
        raise ValueError("Invalid bbox_format. Use 'xyxy' or 'cxcywh'.")
    gt_boxes = [shapely_box(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in gt_bboxes]
    gt_union = unary_union(gt_boxes)
    
    # Compute intersection
    intersection = pred_union.intersection(gt_union).area
    union = pred_union.area + gt_union.area - intersection
    
    return intersection / union if union > 0 else 0.0


def find_optimal_probability_map_conf_threshold(prob_maps, gt_bboxes_list, bbox_format='xyxy',
                                                 conf_thresholds=np.arange(0.1, 0.9, 0.1)):
    """
    Find the optimal confidence threshold for binarizing probability maps that maximizes the average IoU with ground truth.

    Parameters:
    - prob_maps (numpy.ndarray): A 3D array (num_samples, height, width) representing multiple probability maps.
    - gt_bboxes_list (list of arrays/lists/tuples): A list containing the ground truth bounding boxes for each sample.
    - bbox_format (str): Format of the bounding box coordinates. Either 'xyxy' or 'cxcywh'.
    - conf_thresholds (numpy.ndarray or list): A list of confidence thresholds to evaluate.

    Returns:
    - dict: A dictionary containing:
        - 'best_iou' (float): The highest average IoU achieved across samples.
        - 'best_conf_th' (float): The confidence threshold that achieved this IoU.
    """
    assert prob_maps.ndim == 3, "prob_maps must have shape (num_samples, height, width)"
    assert len(gt_bboxes_list) == prob_maps.shape[0], "Mismatch between number of samples in prob_maps and gt_bboxes_list"
    assert isinstance(gt_bboxes_list, list), "gt_bboxes_list must be a list"
    assert all(isinstance(b, (np.ndarray, list, tuple)) for b in gt_bboxes_list), "Each entry in gt_bboxes_list must be an array, list, or tuple"
    assert all(len(bbox) == 4 for bboxes in gt_bboxes_list for bbox in bboxes), "Each bounding box must be in (x_min, y_min, x_max, y_max) format"
    assert all(0 <= x <= 1 for bboxes in gt_bboxes_list for bbox in bboxes for x in bbox), "Bounding box coordinates must be normalized"

    best_iou = -1.0
    best_conf_th = None

    # Iterate over confidence thresholds to find the best one
    for conf_th in tqdm(conf_thresholds, desc="Optimizing threshold", mininterval=2.0):
        total_iou = 0.0
        
        # Compute IoU for each probability map and its corresponding ground truth
        for prob_map, gt_bboxes in zip(prob_maps, gt_bboxes_list):
            total_iou += compute_probability_map_iou(prob_map, gt_bboxes, float(conf_th),
                                                     bbox_format=bbox_format)  # Ensure conf_th is a float
        
        avg_iou = total_iou / len(prob_maps)  # Compute mean IoU over all samples
        
        # Track the best threshold
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_conf_th = conf_th

    return {
        'best_iou': best_iou,
        'best_conf_th': best_conf_th
    }

def compute_iou_with_nms(gt_bboxes, pred_bbox_coords, pred_bbox_probs, iou_th, conf_th, pre_nms_max_det,
                         post_nms_max_det=None, bbox_format='xyxy'):
    """
    Computes the Intersection over Union (IoU) between ground truth bounding boxes and predicted bounding boxes 
    after applying confidence thresholding and Non-Maximum Suppression (NMS).

    Parameters:
    - gt_bboxes (list or np.ndarray): List or array of ground truth bounding boxes (N_gt, 4).
    - pred_bbox_coords (np.ndarray or torch.Tensor): Predicted bounding box coordinates (N_pred, 4).
    - pred_bbox_probs (np.ndarray or torch.Tensor): Confidence scores for predicted bounding boxes (N_pred,).
    - iou_th (float): IoU threshold for NMS.
    - conf_th (float): Confidence threshold for filtering predictions.
    - pre_nms_max_det (int): Maximum number of detections to keep before NMS.
    - post_nms_max_det (int, optional): Maximum number of detections to keep after NMS.
    - bbox_format (str): Format of the bounding box coordinates. Either 'xyxy' or 'cxcywh'.

    Returns:
    - float: IoU score computed between remaining predicted boxes and ground truth boxes.
    """
    assert isinstance(gt_bboxes, (list, np.ndarray)), "gt_bboxes must be a list or np.ndarray"
    assert isinstance(pred_bbox_coords, (np.ndarray, torch.Tensor)), "pred_bbox_coords must be a np.ndarray or torch.Tensor"
    assert isinstance(pred_bbox_probs, (np.ndarray, torch.Tensor)), "pred_bbox_probs must be a np.ndarray or torch.Tensor"
    assert pred_bbox_coords.ndim == 2 and pred_bbox_coords.shape[1] == 4, "pred_bbox_coords must have shape (num_boxes, 4)"
    assert pred_bbox_probs.ndim == 1, "pred_bbox_probs must have shape (num_boxes,)"
    assert bbox_format in ['xyxy', 'cxcywh'], "bbox_format must be 'xyxy' or 'cxcywh'"
    
    # Convert inputs to tensors if needed
    if isinstance(pred_bbox_coords, np.ndarray):
        pred_bbox_coords = torch.tensor(pred_bbox_coords, dtype=torch.float32)
    if isinstance(pred_bbox_probs, np.ndarray):
        pred_bbox_probs = torch.tensor(pred_bbox_probs, dtype=torch.float32)
    
    # Apply pre-NMS filtering
    if pred_bbox_coords.shape[0] > pre_nms_max_det:
        pred_bbox_probs, idxs = torch.topk(pred_bbox_probs, pre_nms_max_det)
        pred_bbox_coords = pred_bbox_coords[idxs]
    
    # Apply confidence threshold
    mask = pred_bbox_probs > conf_th
    pred_bbox_coords = pred_bbox_coords[mask]
    pred_bbox_probs = pred_bbox_probs[mask]
    
    # Apply Non-Maximum Suppression (NMS)
    if pred_bbox_coords.shape[0] > 0:
        if bbox_format == 'cxcywh':
            pred_bbox_coords_xyxy = cxcywh_to_xyxy_tensor(pred_bbox_coords) # Convert to 'xyxy' format
        else: # 'xyxy'
            pred_bbox_coords_xyxy = pred_bbox_coords # Already in 'xyxy' format
        keep = nms(pred_bbox_coords_xyxy, pred_bbox_probs, iou_th)
        pred_bbox_coords = pred_bbox_coords[keep]
        pred_bbox_probs = pred_bbox_probs[keep]
        
    # Apply post-NMS filtering (if specified)
    if post_nms_max_det is not None and pred_bbox_coords.shape[0] > post_nms_max_det:
        pred_bbox_probs, idxs = torch.topk(pred_bbox_probs, post_nms_max_det)
        pred_bbox_coords = pred_bbox_coords[idxs]

    # Compute IoU between predicted and ground truth boxes
    iou = compute_bbox_union_iou(pred_bbox_coords, gt_bboxes, bbox_format)
    
    return iou

def get_grid_centers(grid_height, grid_width, device=None):
    """
    Computes normalized grid cell centers for the given grid dimensions.

    Args:
        grid_height (int): Number of rows in the grid.
        grid_width (int): Number of columns in the grid.
        device (torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: Tensor of shape (grid_height, grid_width, 2) containing cell centers [cx, cy].
    """    
    centers = torch.empty((grid_height, grid_width, 2), device=device)
    w = 1.0 / grid_width
    h = 1.0 / grid_height
    for i in range(grid_height):
        for j in range(grid_width):
            centers[i, j, 0] = (j + 0.5) * w
            centers[i, j, 1] = (i + 0.5) * h
    return centers

def cxcywh_to_xyxy_tensor(boxes):
    """
    Convert bounding boxes from (cx, cy, w, h) format to (x_min, y_min, x_max, y_max) format.

    Args:
        boxes (torch.Tensor): Bounding boxes in (cx, cy, w, h) format.

    Returns:
        torch.Tensor: Bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack((x_min, y_min, x_max, y_max), dim=-1)

def xyxy_to_cxcywh_tensor(boxes):
    """
    Convert bounding boxes from (x_min, y_min, x_max, y_max) format to (cx, cy, w, h) format.

    Args:
        boxes (torch.Tensor): Bounding boxes in (x_min, y_min, x_max, y_max) format.

    Returns:
        torch.Tensor: Bounding boxes in (cx, cy, w, h) format.
    """
    x_min, y_min, x_max, y_max = boxes.unbind(-1)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    w = x_max - x_min
    h = y_max - y_min
    return torch.stack((cx, cy, w, h), dim=-1)

def cxcywh_to_xyxy_basic(bbox):
    """
    Converts bounding box from 'cxcywh' format to 'xyxy' format.
    
    Args:
        bbox (list, np.ndarray, or torch.Tensor): Bounding box in 'cxcywh' format (cx, cy, w, h).
    
    Returns:
        list: Bounding box in 'xyxy' format (x1, y1, x2, y2).
    """
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def xyxy_to_cxcywh_basic(bbox):
    """
    Converts bounding box from 'xyxy' format to 'cxcywh' format.
    
    Args:
        bbox (list, np.ndarray, or torch.Tensor): Bounding box in 'xyxy' format (x1, y1, x2, y2).
    
    Returns:
        list: Bounding box in 'cxcywh' format (cx, cy, w, h).
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]