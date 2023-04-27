import numpy as np
import multiprocessing as mp
# import time

_shared_probs = None
_shared_gt = None
def _auc_task(idx):
    probs = _shared_probs.T[idx]
    gt = _shared_gt.T[idx]
    return _auc(probs, gt)

def _auc(probs, gt):
    idxs = np.argsort(probs)
    n = len(gt)
    pos = 0
    count = 0
    for i in range(n):
        if gt[idxs[i]]:
            count += i - pos
            pos += 1
    return count / (pos * (n - pos)) if pos > 0 and pos < n else 0.5

def auc_fn(probs, gt, num_workers=6):
    # start = time.time()
    # Compute micro-average AUC by flattening the arrays.
    micro_avg = _auc(probs.flatten(), gt.flatten())
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