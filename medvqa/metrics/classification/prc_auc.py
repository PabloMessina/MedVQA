import multiprocessing as mp
import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from medvqa.utils.metrics import average_ignoring_nones

def _prc_auc(probs, gt):
    precision, recall, _ = precision_recall_curve(gt, probs)
    return auc(recall, precision)

_shared_probs = None
_shared_gt = None
def _prc_auc_task(idx):
    probs = _shared_probs.T[idx]
    gt = _shared_gt.T[idx]
    return _prc_auc(probs, gt)

def prc_auc_fn(probs, gt, num_workers=6):
    # Compute micro-average AUC by flattening the arrays.
    micro_avg = _prc_auc(probs.flatten(), gt.flatten())
    # Compute macro-average AUC by averaging over classes.
    n_classes = probs.shape[1]
    global _shared_probs, _shared_gt
    _shared_probs = probs
    _shared_gt = gt
    with mp.Pool(num_workers) as pool:
        per_class = pool.map(_prc_auc_task, range(n_classes))
    per_class = [x if not np.isnan(x) else None for x in per_class]
    macro_avg = average_ignoring_nones(per_class)
    return {
        'micro_avg': micro_avg,
        'macro_avg': macro_avg,
        'per_class': per_class,
    }