import numpy as np

def average_ignoring_nones_and_nans(values):
    acc = 0
    count = 0
    for x in values:
        if x is not None and not np.isnan(x):
            acc += x
            count += 1
    return acc / count if count > 0 else 0