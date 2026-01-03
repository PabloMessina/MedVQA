import cv2
import numpy as np
from tqdm import tqdm


def average_ignoring_nones_and_nans(values):
    acc = 0
    count = 0
    for x in values:
        if x is not None and not np.isnan(x):
            acc += x
            count += 1
    return acc / count if count > 0 else 0

def f1_between_sets(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    if len1 == 0 and len2 == 0:
        return 1 # both sets are empty -> perfect match
    len_inters = len(s1 & s2)
    p = len_inters / len1 if len1 > 0 else 0
    r = len_inters / len2 if len2 > 0 else 0
    return 2 * p * r / (p + r) if p + r > 0 else 0

def jaccard_between_sets(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    if len1 == 0 and len2 == 0:
        return 1 # both sets are empty -> perfect match
    len_inters = len(s1 & s2)
    return len_inters / (len1 + len2 - len_inters)

def jaccard_between_dicts(d1, d2):
    if len(d1) == 0 and len(d2) == 0:
        return 1 # both dicts are empty -> perfect match
    inters_size = 0
    for k, c in d1.items():
        inters_size += min(c, d2.get(k, 0))
    union_size = sum(d1.values()) + sum(d2.values()) - inters_size
    assert union_size > 0
    return inters_size / union_size

def f1_between_dicts(gt_dict, pred_dict):
    if len(gt_dict) == 0 and len(pred_dict) == 0:
        return 1 # both dicts are empty -> perfect match
    inters_size = 0
    for k, c in gt_dict.items():
        inters_size += min(c, pred_dict.get(k, 0))
    pred_sum = sum(pred_dict.values())
    gt_sum = sum(gt_dict.values())
    p = inters_size / pred_sum if pred_sum > 0 else 0
    r = inters_size / gt_sum if gt_sum > 0 else 0
    return 2 * p * r / (p + r) if p + r > 0 else 0

def precision_between_dicts(gt_dict, pred_dict):
    if len(gt_dict) == 0 and len(pred_dict) == 0:
        return 1 # both dicts are empty -> perfect match
    inters_size = 0
    for k, c in gt_dict.items():
        inters_size += min(c, pred_dict.get(k, 0))
    pred_sum = sum(pred_dict.values())
    return inters_size / pred_sum if pred_sum > 0 else 0

def recall_between_dicts(gt_dict, pred_dict):
    if len(gt_dict) == 0 and len(pred_dict) == 0:
        return 1 # both dicts are empty -> perfect match
    inters_size = 0
    for k, c in gt_dict.items():
        inters_size += min(c, pred_dict.get(k, 0))
    gt_sum = sum(gt_dict.values())
    return inters_size / gt_sum if gt_sum > 0 else 0
            
def auc(scores, binary_labels):
    assert len(scores) == len(binary_labels)
    order = np.argsort(scores)
    n = len(scores)
    pos = 0
    count = 0
    for i in range(n):
        if binary_labels[order[i]] == 1:
            count += i - pos
            pos += 1
    return count / (pos * (n - pos)) if pos > 0 and pos < n else 0.5

def best_threshold_and_f1_score(gt, probs):
    idxs = np.argsort(probs)
    best_thrs = 0
    tp = gt.sum()
    fp = len(gt) - tp
    fn = 0
    best_f1 = 2 * tp / (2 * tp + fp + fn)
    for i in idxs:
        if gt[i]:
            tp -= 1
            fn += 1
        else:
            fp -= 1
        if tp == 0:
            break
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1 = f1
            best_thrs = probs[i]
    return best_thrs, best_f1

def best_threshold_and_precision_score(gt, probs):
    idxs = np.argsort(probs)
    best_thrs = 0
    tp = gt.sum()
    fp = len(gt) - tp
    best_precision = tp / (tp + fp)
    for i in idxs:
        if gt[i]:
            tp -= 1
        else:
            fp -= 1
        if tp == 0:
            break
        precision = tp / (tp + fp)
        if precision > best_precision:
            best_precision = precision
            best_thrs = probs[i]
    return best_thrs, best_precision

def best_threshold_and_accuracy_score(gt, probs):
    idxs = np.argsort(probs)
    best_thrs = 0
    tp = gt.sum()
    tn = 0
    fp = len(gt) - tp
    fn = 0
    best_acc = (tp + tn) / (tp + tn + fp + fn)
    for i in idxs:
        if gt[i]:
            tp -= 1
            fn += 1
        else:
            tn += 1
            fp -= 1
        acc = (tp + tn) / (tp + tn + fp + fn)
        if acc > best_acc:
            best_acc = acc
            best_thrs = probs[i]
    return best_thrs, best_acc


def calculate_cnr(mask: np.ndarray, prob_map: np.ndarray) -> float:
    """
    Computes the Contrast-to-Noise Ratio (CNR) between a foreground and
    background region defined by a binary mask.

    The formula is defined as:
        CNR = mu_A - mu_A_bar / sqrt(sigma_A^2 + sigma_A_bar^2)

    where:
    - A is the interior (foreground) region (mask == 1).
    - A_bar is the exterior (background) region (mask == 0).
    - mu is the mean of probability values in a region.
    - sigma^2 is the variance of probability values in a region.

    Args:
        mask (np.ndarray): A 2D binary numpy array (H, W) where 1s represent
            the interior (foreground) and 0s represent the exterior (background).
            The dtype should be integer-like (e.g., uint8, int).
        prob_map (np.ndarray): A 2D numpy array (Hp, Wp) with probability
            values (0 to 1). This map will be resized to match the mask's
            dimensions if they differ.

    Returns:
        float: The calculated CNR value. Returns np.nan if either the
            foreground or background region is empty. Returns np.inf if
            there is contrast but zero variance (perfect, noiseless separation).
    """
    # --- 1. Validate Inputs ---
    if mask.ndim != 2 or prob_map.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must be binary (containing only 0s and 1s).")

    # --- 2. Resize Probability Map to Match Mask Dimensions ---
    # Ensure the mask is boolean for indexing later
    binary_mask = mask.astype(bool)

    if binary_mask.shape != prob_map.shape:
        # Get target dimensions from the mask (H, W)
        target_height, target_width = binary_mask.shape
        # Resize prob_map. Note: cv2.resize expects (width, height) for dsize.
        # We use bilinear interpolation as it's a good default for down/upsampling
        # continuous-valued maps like probability maps.
        resized_prob_map = cv2.resize(
            prob_map,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        resized_prob_map = prob_map

    # --- 3. Separate Interior (A) and Exterior (A_bar) Regions ---
    # Use the boolean mask to select values from the resized probability map.
    interior_probs = resized_prob_map[binary_mask]
    exterior_probs = resized_prob_map[~binary_mask]

    # --- 4. Handle Edge Cases ---
    # If either region is empty, CNR is undefined.
    if interior_probs.size == 0 or exterior_probs.size == 0:
        # This can happen if the mask is all 1s or all 0s.
        return np.nan

    # --- 5. Calculate Statistics (Mean and Variance) ---
    mu_A = np.mean(interior_probs)
    mu_A_bar = np.mean(exterior_probs)

    # Use ddof=0 for population variance, as we have all pixels in the region.
    sigma_A_sq = np.var(interior_probs, ddof=0)
    sigma_A_bar_sq = np.var(exterior_probs, ddof=0)

    # --- 6. Compute CNR ---
    numerator = mu_A - mu_A_bar
    denominator = np.sqrt(sigma_A_sq + sigma_A_bar_sq)

    # Handle division by zero: this occurs if both regions have zero variance
    # (i.e., all pixels in the interior are identical, and all in the exterior
    # are identical).
    if denominator < 1e-9:  # Use a small epsilon for floating point comparison
        if numerator < 1e-9:
            # No contrast and no noise, CNR is 0.
            return 0.0
        else:
            # Finite contrast but zero noise, CNR is infinite.
            return np.inf

    cnr = numerator / denominator
    return cnr


def calculate_soft_dice(mask: np.ndarray, prob_map: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Computes the soft Dice score between a binary mask and a probability map.

    Args:
        mask (np.ndarray): 2D binary array (H, W), 1 for foreground, 0 for background.
        prob_map (np.ndarray): 2D array (Hp, Wp) with values in [0, 1].
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: Soft Dice score (0 to 1).
    """
    if mask.ndim != 2 or prob_map.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must be binary (containing only 0s and 1s).")

    # Resize prob_map if needed
    if mask.shape != prob_map.shape:
        prob_map = cv2.resize(
            prob_map,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    mask = mask.astype(np.float32)
    prob_map = prob_map.astype(np.float32)

    intersection = np.sum(mask * prob_map)
    denominator = np.sum(mask) + np.sum(prob_map)
    soft_dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return soft_dice


def calculate_dice(
    mask: np.ndarray,
    prob_map: np.ndarray,
    threshold: float = 0.5,
    epsilon: float = 1e-6
) -> float:
    """
    Computes the Dice score between a binary mask and a thresholded probability map.

    Args:
        mask (np.ndarray): 2D binary array (H, W), 1 for foreground, 0 for background.
        prob_map (np.ndarray): 2D array (Hp, Wp) with values in [0, 1].
        threshold (float): Threshold to binarize the probability map.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: Dice score (0 to 1).
    """
    if mask.ndim != 2 or prob_map.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must be binary (containing only 0s and 1s).")

    # Resize prob_map if needed
    if mask.shape != prob_map.shape:
        prob_map = cv2.resize(
            prob_map,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Binarize the probability map
    pred_mask = (prob_map >= threshold).astype(np.float32)
    mask = mask.astype(np.float32)

    intersection = np.sum(mask * pred_mask)
    denominator = np.sum(mask) + np.sum(pred_mask)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return dice


def calculate_segmentation_iou(
    gt_mask: np.ndarray,
    prob_map: np.ndarray,
    threshold: float = 0.5,
    epsilon: float = 1e-6
) -> float:
    """
    Computes the Intersection over Union (IoU) between a binary mask and a thresholded probability map.

    Args:
        gt_mask (np.ndarray): 2D binary array (H, W), 1 for foreground, 0 for background.
        prob_map (np.ndarray): 2D array (Hp, Wp) with values in [0, 1].
        threshold (float): Threshold to binarize the probability map.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: IoU score (0 to 1).
    """
    if gt_mask.ndim != 2 or prob_map.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional.")
    if not np.all((gt_mask == 0) | (gt_mask == 1)):
        raise ValueError("Mask must be binary (containing only 0s and 1s).")

    # Resize prob_map if needed
    if gt_mask.shape != prob_map.shape:
        prob_map = cv2.resize(
            prob_map,
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Binarize the probability map
    pred_mask = (prob_map >= threshold).astype(np.float32)
    gt_mask = gt_mask.astype(np.float32)

    intersection = np.sum(gt_mask * pred_mask)
    union = np.sum(gt_mask) + np.sum(pred_mask) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou


def calculate_rand_index(
    mask: np.ndarray,
    prob_map: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Computes the Rand Index between a binary mask and a thresholded probability map.

    Args:
        mask (np.ndarray): 2D binary array (H, W).
        prob_map (np.ndarray): 2D array (Hp, Wp) with values in [0, 1].
        threshold (float): Threshold to binarize the probability map.

    Returns:
        float: Rand Index (0 to 1).
    """
    if mask.shape != prob_map.shape:
        prob_map = cv2.resize(
            prob_map,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    pred_mask = (prob_map >= threshold).astype(np.uint8)
    mask = mask.astype(np.uint8)

    TP = np.sum((mask == 1) & (pred_mask == 1))
    TN = np.sum((mask == 0) & (pred_mask == 0))
    FP = np.sum((mask == 0) & (pred_mask == 1))
    FN = np.sum((mask == 1) & (pred_mask == 0))

    total = TP + TN + FP + FN
    assert total > 0, "Total count of pixels must be greater than zero."
    rand_index = (TP + TN) / total
    return rand_index


def calculate_segmentation_precision(
    mask: np.ndarray,
    prob_map: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Computes the precision between a binary mask and a thresholded probability map.

    Args:
        mask (np.ndarray): 2D binary array (H, W).
        prob_map (np.ndarray): 2D array (Hp, Wp) with values in [0, 1].
        threshold (float): Threshold to binarize the probability map.

    Returns:
        float: Precision (0 to 1).
    """
    if mask.shape != prob_map.shape:
        prob_map = cv2.resize(
            prob_map,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    pred_mask = (prob_map >= threshold).astype(np.uint8)
    mask = mask.astype(np.uint8)

    TP = np.sum((mask == 1) & (pred_mask == 1))
    FP = np.sum((mask == 0) & (pred_mask == 1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return precision


def calculate_segmentation_recall(
    mask: np.ndarray,
    prob_map: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Computes the recall between a binary mask and a thresholded probability map.

    Args:
        mask (np.ndarray): 2D binary array (H, W).
        prob_map (np.ndarray): 2D array (Hp, Wp) with values in [0, 1].
        threshold (float): Threshold to binarize the probability map.

    Returns:
        float: Recall (0 to 1).
    """
    if mask.shape != prob_map.shape:
        prob_map = cv2.resize(
            prob_map,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    pred_mask = (prob_map >= threshold).astype(np.uint8)
    mask = mask.astype(np.uint8)

    TP = np.sum((mask == 1) & (pred_mask == 1))
    FN = np.sum((mask == 1) & (pred_mask == 0))
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return recall


def find_optimal_probability_map_conf_threshold_for_masks(
    prob_maps: np.ndarray,
    gt_masks: list,
    conf_thresholds: np.ndarray = np.arange(0.1, 0.9, 0.1),
):
    """
    Find the optimal confidence threshold for binarizing probability maps that maximizes the average IoU with ground truth masks.

    Parameters:
    - prob_maps (numpy.ndarray): A 3D array (num_samples, height, width) representing multiple probability maps.
    - gt_masks (list of np.ndarray): A list of 2D binary arrays (H, W) for each sample.
    - conf_thresholds (numpy.ndarray or list): A list of confidence thresholds to evaluate.

    Returns:
    - dict: A dictionary containing:
        - 'best_iou' (float): The highest average IoU achieved across samples.
        - 'best_conf_th' (float): The confidence threshold that achieved this IoU.
    """
    assert prob_maps.ndim == 3, "prob_maps must have shape (num_samples, height, width)"
    assert len(gt_masks) == prob_maps.shape[0], "Mismatch between number of samples in prob_maps and gt_masks"
    assert all(isinstance(m, np.ndarray) for m in gt_masks), "Each entry in gt_masks must be a numpy array"
    assert all(m.ndim == 2 for m in gt_masks), "Each mask must be 2D"

    best_iou = -1.0
    best_conf_th = None

    for conf_th in tqdm(conf_thresholds, desc="Optimizing threshold", mininterval=2.0):
        conf_th = float(conf_th)
        total_iou = 0.0
        for prob_map, gt_mask in zip(prob_maps, gt_masks):
            iou = calculate_segmentation_iou(gt_mask, prob_map, threshold=conf_th)
            total_iou += iou
        avg_iou = total_iou / len(prob_maps)
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_conf_th = conf_th

    return {
        'best_iou': best_iou,
        'best_conf_th': best_conf_th
    }