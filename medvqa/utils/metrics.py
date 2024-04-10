import numpy as np

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

def best_threshold_and_f1_score(probs, gt):
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

def best_threshold_and_precision_score(probs, gt):
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

def best_threshold_and_accuracy_score(probs, gt):
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