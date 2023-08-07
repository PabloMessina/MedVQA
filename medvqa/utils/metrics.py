import numpy as np

def average_ignoring_nones_and_nans(values):
    acc = 0
    count = 0
    for x in values:
        if x is not None and not np.isnan(x):
            acc += x
            count += 1
    return acc / count if count > 0 else 0

def f1_score_between_sets(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    if len1 == 0 and len2 == 0:
        return 1 # both sets are empty -> perfect match
    len_inters = len(s1 & s2)
    p = len_inters / len1 if len1 > 0 else 0
    r = len_inters / len2 if len2 > 0 else 0
    return 2 * p * r / (p + r) if p + r > 0 else 0

def jaccard_score_between_sets(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    if len1 == 0 and len2 == 0:
        return 1 # both sets are empty -> perfect match
    len_inters = len(s1 & s2)
    return len_inters / (len1 + len2 - len_inters)