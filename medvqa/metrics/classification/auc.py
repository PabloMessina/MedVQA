import numpy as np
import multiprocessing as mp

from medvqa.utils.metrics_utils import auc
# import time

_shared_probs = None
_shared_gt = None
def _auc_task(idx):
    probs = _shared_probs.T[idx]
    gt = _shared_gt.T[idx]
    return auc(probs, gt)

def auc_fn(probs, gt, num_workers=6):
    # start = time.time()
    # Compute micro-average AUC by flattening the arrays.
    micro_avg = auc(probs.flatten(), gt.flatten())
    # print(f'auc_fn(): elapsed time: {time.time() - start:.2f} s (after micro), probs.shape: {probs.shape}, gt.shape: {gt.shape}, num_workers: {num_workers}')
    # Compute macro-average AUC by averaging over classes.
    n_classes = probs.shape[1]
    global _shared_probs, _shared_gt
    _shared_probs = probs
    _shared_gt = gt
    with mp.Pool(num_workers) as pool:
        per_class = pool.map(_auc_task, range(n_classes))
    macro_avg = np.mean(per_class)
    # print(f'auc_fn(): elapsed time: {time.time() - start:.2f} s (after macro), probs.shape: {probs.shape}, gt.shape: {gt.shape}, num_workers: {num_workers}')
    return {
        'micro_avg': micro_avg,
        'macro_avg': macro_avg,
        'per_class': per_class,
    }