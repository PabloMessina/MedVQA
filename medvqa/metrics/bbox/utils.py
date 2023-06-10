import torch
import numpy as np
from multiprocessing import Pool

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