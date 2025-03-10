import numpy as np
import time
from sklearn.utils import resample
from tqdm import tqdm
from medvqa.datasets.vinbig import (
    VINBIG_CHEX_CLASSES,
    VINBIG_CHEX_IOU_THRESHOLDS,
    VINBIGDATA_CHALLENGE_CLASSES,
    VINBIGDATA_CHALLENGE_IOU_THRESHOLD,
)
from medvqa.utils.logging import print_bold

def stratified_multilabel_bootstrap_metrics(gt_labels, pred_probs, metric_fn, num_bootstraps=500):
    """
    Perform stratified bootstrapping for multi-label classification.
    
    Args:
        gt_labels (numpy.ndarray): Shape (N, C), ground truth binary labels.
        pred_probs (numpy.ndarray): Shape (N, C), predicted probabilities.
        metric_fn (callable): Metric function that accepts two arguments: y_true and y_score.
        num_bootstraps (int): Number of bootstrap iterations.

    Returns:
        dict: Dictionary containing mean and standard deviation of per-class metrics and macro-average.
    """
    pred_probs, gt_labels = np.array(pred_probs), np.array(gt_labels)
    N, C = gt_labels.shape
    per_class_metrics = np.zeros((num_bootstraps, C))  # Store per-class metric per bootstrap
    micro_metrics = np.zeros(num_bootstraps)  # Store micro-average metric per bootstrap

    # Precompute positive and negative indices for each class
    pos_indices_per_class = [np.where(gt_labels[:, c] == 1)[0] for c in range(C)]
    neg_indices_per_class = [np.where(gt_labels[:, c] == 0)[0] for c in range(C)]
    assert all(len(pos_indices) > 0 for pos_indices in pos_indices_per_class), "At least one positive sample per class is required."
    assert all(len(neg_indices) > 0 for neg_indices in neg_indices_per_class), "At least one negative sample per class is required."
    all_indices = np.arange(N)

    for b in tqdm(range(num_bootstraps), mininterval=2.0, desc="Bootstrapping"):
        # Step 1: Select one positive and one negative index per class
        selected_indices = []
        for pos_indices in pos_indices_per_class:
            selected_indices.append(np.random.choice(pos_indices))
        for neg_indices in neg_indices_per_class:
            selected_indices.append(np.random.choice(neg_indices))

        # Step 2: Deduplicate selected indices
        unique_selected_indices = np.unique(selected_indices)

        # Step 3: Perform normal bootstrapping on remaining indices
        remaining_indices = np.setdiff1d(all_indices, unique_selected_indices, assume_unique=True)
        boot_indices = resample(remaining_indices, replace=True)

        # Step 4: Combine both sets
        final_boot_indices = np.hstack([unique_selected_indices, boot_indices])
        assert len(final_boot_indices) == N

        # Step 5: Compute metric (e.g., ROCAUC) per class
        boot_gt_labels = gt_labels[final_boot_indices]
        boot_pred_probs = pred_probs[final_boot_indices]
        for c in range(C):
            per_class_metrics[b, c] = metric_fn(boot_gt_labels[:, c], boot_pred_probs[:, c])

        # Step 6: Compute micro-average metric
        micro_metrics[b] = metric_fn(boot_gt_labels.ravel(), boot_pred_probs.ravel())

    # Compute mean and standard deviation per class
    mean_per_class = per_class_metrics.mean(axis=0)
    std_per_class = per_class_metrics.std(axis=0)

    # Compute mean and standard deviation of macro-average
    macro_avg_per_bootstrap = per_class_metrics.mean(axis=1)
    mean_macro_avg = macro_avg_per_bootstrap.mean()
    std_macro_avg = macro_avg_per_bootstrap.std()

    # Compute mean and standard deviation of micro-average
    mean_micro_avg = micro_metrics.mean()
    std_micro_avg = micro_metrics.std()

    return dict(
        mean_per_class=mean_per_class,
        std_per_class=std_per_class,
        mean_macro_avg=mean_macro_avg,
        std_macro_avg=std_macro_avg,
        mean_micro_avg=mean_micro_avg,
        std_micro_avg=std_micro_avg,
    )

_shared_pred_boxes_list = None
_shared_pred_classes_list = None
_shared_pred_confs_list = None
_shared_classifier_confs = None
_shared_gt_coords_list = None
_shared_vinbig_bbox_names = None
_shared_map_iou_thresholds = None
_shared_compute_mean_iou_per_class_fn = None
_shared_compute_mAP_fn = None
_shared_all_indices = None
_shared_pos_indices_per_class = None
_shared_neg_indices_per_class = None

def _compute_vinbig_iou_map_metrics(*unused_args):

    # Step 0: seed random number generator
    np.random.seed() # seed with random value

    # Step 1: Select one positive and one negative index per class
    selected_indices = []
    for pos_indices in _shared_pos_indices_per_class:
        selected_indices.append(np.random.choice(pos_indices))
    for neg_indices in _shared_neg_indices_per_class:
        selected_indices.append(np.random.choice(neg_indices))

    # Step 2: Deduplicate selected indices
    unique_selected_indices = np.unique(selected_indices)

    # Step 3: Perform normal bootstrapping on remaining indices
    remaining_indices = np.setdiff1d(_shared_all_indices, unique_selected_indices, assume_unique=True)

    # Step 4: Combine both sets
    boot_indices = np.hstack([unique_selected_indices, resample(remaining_indices, replace=True)])

    boot_pred_boxes_list = [_shared_pred_boxes_list[i] for i in boot_indices]
    boot_pred_classes_list = [_shared_pred_classes_list[i] for i in boot_indices]
    boot_pred_confs_list = [_shared_pred_confs_list[i] for i in boot_indices]
    boot_classifier_confs = _shared_classifier_confs[boot_indices] if _shared_classifier_confs is not None else None
    boot_gt_coords_list = [_shared_gt_coords_list[i] for i in boot_indices]

    # Step 5: Compute metrics
    
    # Compute IoU
    tmp_iou = _shared_compute_mean_iou_per_class_fn(
        pred_boxes=boot_pred_boxes_list,
        pred_classes=boot_pred_classes_list,
        gt_coords=boot_gt_coords_list,
        compute_micro_average_iou=True,
    )
    metrics = dict()
    metrics["class_ious"] = tmp_iou['class_ious']
    metrics["micro_iou"] = tmp_iou['micro_iou']
    metrics["macro_iou"]= tmp_iou['class_ious'].mean()
    
    # Compute mAP
    class_aps = _shared_compute_mAP_fn(
        pred_boxes=boot_pred_boxes_list,
        pred_classes=boot_pred_classes_list,
        pred_confs=boot_pred_confs_list,
        classifier_confs=boot_classifier_confs,
        gt_coords=boot_gt_coords_list,
        iou_thresholds=_shared_map_iou_thresholds,
        use_multiprocessing=False, # Avoid nested multiprocessing -> it triggers "AssertionError: daemonic processes are not allowed to have children"
    )
    metrics["class_aps"] = class_aps
    
    # Compute VinBigData challenge mAP
    class_idxs_vbdc = [_shared_vinbig_bbox_names.index(x) for x in VINBIGDATA_CHALLENGE_CLASSES]
    iou_idx_vbdc = _shared_map_iou_thresholds.index(VINBIGDATA_CHALLENGE_IOU_THRESHOLD)
    vbdc_mAP = class_aps[iou_idx_vbdc, class_idxs_vbdc].mean()
    metrics["vbdc_mAP"] = vbdc_mAP
    
    # Compute CheX mAP
    class_idxs_chex = [_shared_vinbig_bbox_names.index(x) for x in VINBIG_CHEX_CLASSES]
    iou_idxs_chex = [_shared_map_iou_thresholds.index(x) for x in VINBIG_CHEX_IOU_THRESHOLDS]
    chex_mAP = class_aps[iou_idxs_chex][:, class_idxs_chex].mean()
    metrics["chex_mAP"]= chex_mAP

    return metrics

def stratified_vinbig_bootstrap_iou_map(
    pred_boxes_list, pred_classes_list, pred_confs_list, classifier_confs,
    gt_coords_list, vinbig_bbox_names, map_iou_thresholds,
    compute_mean_iou_per_class_fn, compute_mAP_fn,
    num_bootstraps=200,
    num_processes=8,
):
    """
    Perform bootstrapping to compute mean and std of IoU and mAP metrics.
    
    Args:
        pred_boxes_list (list): List of predicted bounding boxes.
        pred_classes_list (list): List of predicted class labels.
        pred_confs_list (list): List of predicted confidences.
        classifier_confs (list): Classifier confidences.
        gt_coords_list (list): List of ground truth bounding boxes.
        vinbig_bbox_names (list): List of class names.
        map_iou_thresholds (list): List of IoU thresholds for mAP computation.
        compute_mean_iou_per_class_fn (callable): Function to compute mean IoU per class.
        compute_mAP_fn (callable): Function to compute mAP.
        num_bootstraps (int): Number of bootstrap iterations.
    
    Returns:
        dict: Dictionary containing mean and std of IoU and mAP metrics.
    """
    N = len(pred_boxes_list)
    metric_names = ["class_ious", "micro_iou", "macro_iou", "class_aps", "vbdc_mAP", "chex_mAP"]

    # Precompute positive and negative indices for each class
    pos_indices_per_class = [[] for _ in range(len(vinbig_bbox_names))]
    neg_indices_per_class = [[] for _ in range(len(vinbig_bbox_names))]
    for i, gt_coords in enumerate(gt_coords_list):
        assert len(gt_coords) == len(vinbig_bbox_names)
        for c, coords in enumerate(gt_coords):
            if len(coords) > 0:
                pos_indices_per_class[c].append(i)
            else:
                neg_indices_per_class[c].append(i)
    assert all(len(pos_indices) > 0 for pos_indices in pos_indices_per_class), "At least one positive sample per class is required."
    assert all(len(neg_indices) > 0 for neg_indices in neg_indices_per_class), "At least one negative sample per class is required."
    all_indices = np.arange(N)

    # Perform bootstrapping with multiprocessing
    import multiprocessing as mp
    global _shared_pred_boxes_list
    global _shared_pred_classes_list
    global _shared_pred_confs_list
    global _shared_classifier_confs
    global _shared_gt_coords_list
    global _shared_vinbig_bbox_names
    global _shared_map_iou_thresholds
    global _shared_compute_mean_iou_per_class_fn
    global _shared_compute_mAP_fn
    global _shared_all_indices
    global _shared_pos_indices_per_class
    global _shared_neg_indices_per_class
    _shared_pred_boxes_list = pred_boxes_list
    _shared_pred_classes_list = pred_classes_list
    _shared_pred_confs_list = pred_confs_list
    _shared_classifier_confs = classifier_confs
    _shared_gt_coords_list = gt_coords_list
    _shared_vinbig_bbox_names = vinbig_bbox_names
    _shared_map_iou_thresholds = map_iou_thresholds
    _shared_compute_mean_iou_per_class_fn = compute_mean_iou_per_class_fn
    _shared_compute_mAP_fn = compute_mAP_fn
    _shared_all_indices = all_indices
    _shared_pos_indices_per_class = pos_indices_per_class
    _shared_neg_indices_per_class = neg_indices_per_class

    num_processes = min(num_processes, mp.cpu_count())

    print_bold(f"Performing bootstrapping with {num_processes} processes...")
    start = time.time()
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(_compute_vinbig_iou_map_metrics, range(num_bootstraps)), total=num_bootstraps))
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")
    
    # Compute mean and std
    result = {}
    for key in metric_names:
        values = np.array([x[key] for x in results])
        result[key] = dict(
            mean=values.mean(),
            std=values.std(),
        )
    
    return result

_shared_probs_list = None
_shared_gt_labels_list = None
_shared_metric_fn = None
_shared_pos_indices_list = None
_shared_neg_indices_list = None
_shared_all_indices_list = None

def _compute_pos_neg_fact_classification_metric(*unused_args):

    # Step 0: seed random number generator
    np.random.seed() # seed with random value

    metric_sum = 0

    n = len(_shared_probs_list)

    for i in range(n):
        pos_indices = _shared_pos_indices_list[i]
        neg_indices = _shared_neg_indices_list[i]
        all_indices = _shared_all_indices_list[i]

        # Step 1: Select one positive and one negative index
        selected_indices = [np.random.choice(pos_indices), np.random.choice(neg_indices)]

        # Step 2: Perform normal bootstrapping on remaining indices
        remaining_indices = np.setdiff1d(all_indices, selected_indices, assume_unique=True)
        boot_indices = resample(remaining_indices, replace=True)

        # Step 3: Combine both sets
        final_boot_indices = np.hstack([selected_indices, boot_indices])
        assert len(final_boot_indices) == len(_shared_probs_list[i])

        # Step 4: Compute metric (e.g., ROCAUC)
        metric_sum += _shared_metric_fn(_shared_gt_labels_list[i][final_boot_indices],
                                        _shared_probs_list[i][final_boot_indices])

    metric_avg = metric_sum / n

    return metric_avg


def stratified_bootstrap_pos_neg_fact_classification_metrics(items, metric_fn, num_bootstraps=500, num_processes=8):
    """
    Perform stratified bootstrapping for positive-negative classification.
    
    Args:
        items (list): List of items with ground truth and predicted scores.
        metric_fn (callable): Metric function that accepts two arguments: y_true and y_score.
        num_bootstraps (int): Number of bootstrap iterations.
        num_processes (int): Number of processes for multiprocessing.

    Returns:
        dict: Dictionary containing mean and standard deviation of metrics.
    """
    items = np.array(items)

    # Precompute positive and negative indices for each item
    probs_list = []
    pos_indices_list = []
    neg_indices_list = []
    all_indices_list = []
    gt_labels_list = []
    for item in items:
        pos_probs = item['pos_probs']
        neg_probs = item['neg_probs']
        assert len(pos_probs) > 0, "At least one positive sample is required."
        assert len(neg_probs) > 0, "At least one negative sample is required."
        probs_list.append(np.hstack([pos_probs, neg_probs]))
        pos_indices_list.append(np.arange(len(pos_probs)))
        neg_indices_list.append(np.arange(len(pos_probs), len(pos_probs) + len(neg_probs)))
        all_indices_list.append(np.arange(len(pos_probs) + len(neg_probs)))
        gt_labels = np.zeros(len(pos_probs) + len(neg_probs))
        gt_labels[:len(pos_probs)] = 1
        gt_labels_list.append(gt_labels)

    # Perform bootstrapping with multiprocessing
    import multiprocessing as mp
    global _shared_probs_list
    global _shared_gt_labels_list
    global _shared_metric_fn
    global _shared_pos_indices_list
    global _shared_neg_indices_list
    global _shared_all_indices_list
    _shared_probs_list = probs_list
    _shared_gt_labels_list = gt_labels_list
    _shared_metric_fn = metric_fn
    _shared_pos_indices_list = pos_indices_list
    _shared_neg_indices_list = neg_indices_list
    _shared_all_indices_list = all_indices_list

    num_processes = min(num_processes, mp.cpu_count())

    print_bold(f"Performing bootstrapping with {num_processes} processes...")
    start = time.time()
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(_compute_pos_neg_fact_classification_metric, range(num_bootstraps)), total=num_bootstraps))
    end = time.time()

    print(f"Elapsed time: {end - start:.2f} seconds")

    # Compute mean and std
    mean_metric = np.mean(results)
    std_metric = np.std(results)

    return dict(
        mean=mean_metric,
        std=std_metric
    )