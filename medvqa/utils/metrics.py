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
            
