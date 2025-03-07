import multiprocessing as mp
import numpy as np
# import time
from sklearn.metrics import auc, precision_recall_curve

from medvqa.metrics.condition_aware_metric import ConditionAwareMetric
from medvqa.utils.metrics import average_ignoring_nones_and_nans

def prc_auc_score(gt, probs):
    assert ((gt == 0) | (gt == 1)).all() # gt must be binary
    assert ((probs >= 0) & (probs <= 1)).all() # probs must be in [0, 1]
    precision, recall, _ = precision_recall_curve(gt, probs)
    return auc(recall, precision)

_shared_probs = None
_shared_gt = None
def _prc_auc_task(idx):
    probs = _shared_probs.T[idx]
    gt = _shared_gt.T[idx]
    return prc_auc_score(gt, probs)

def prc_auc_fn(gt, probs, num_workers=6):
    # start = time.time()
    # Compute micro-average AUC by flattening the arrays.
    micro_avg = prc_auc_score(gt.flatten(), probs.flatten())
    # Compute macro-average AUC by averaging over classes.
    macro_avg, per_class = prc_auc_macro_avg_fn(gt, probs, num_workers=num_workers, return_per_class=True)
    # print(f'prc_auc_fn(): elapsed time: {time.time() - start:.2f} s, probs.shape: {probs.shape}, gt.shape: {gt.shape}, num_workers: {num_workers}')
    return {
        'micro_avg': micro_avg,
        'macro_avg': macro_avg,
        'per_class': per_class,
    }

def prc_auc_macro_avg_fn(gt, probs, num_workers=6, return_per_class=False):
    # Compute macro-average AUC by averaging over classes.
    n_classes = probs.shape[1]
    global _shared_probs, _shared_gt
    _shared_probs = probs
    _shared_gt = gt
    with mp.Pool(num_workers) as pool:
        per_class = pool.map(_prc_auc_task, range(n_classes))
    per_class = [x if not np.isnan(x) else None for x in per_class]
    macro_avg = average_ignoring_nones_and_nans(per_class)
    if return_per_class:
        return macro_avg, per_class
    return macro_avg


class ConditionAwareClassAveragedPRCAUC(ConditionAwareMetric):

    def __init__(self, output_transform, condition_function=lambda _: True, use_indices=False):
        super().__init__(output_transform, condition_function)
        self.use_indices = use_indices
        if use_indices:
            self._update = self._update_with_indices
            self._compute = self._compute_with_indices
            self._class2gt = {}
            self._class2probs = {}
        else:
            self._update = self._update_without_indices
            self._compute = self._compute_without_indices
            self._gt = []
            self._probs = []

    def reset(self):
        if self.use_indices:
            self._class2gt.clear()
            self._class2probs.clear()
        else:
            self._gt.clear()
            self._probs.clear()

    def _update_with_indices(self, output):
        probs, gt, class_indices = output
        assert probs.ndim == gt.ndim == 1
        assert len(probs) == len(gt) == len(class_indices)
        probs = probs.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        for i, class_idx in enumerate(class_indices):
            if class_idx not in self._class2gt:
                self._class2gt[class_idx] = []
                self._class2probs[class_idx] = []
            self._class2gt[class_idx].append(gt[i])
            self._class2probs[class_idx].append(probs[i])

    def _update_without_indices(self, output):
        probs, gt = output
        assert probs.ndim == gt.ndim == 2
        assert probs.shape == gt.shape
        probs = probs.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        self._gt.append(gt)
        self._probs.append(probs)

    def update(self, output):
        self._update(output)

    def _compute_with_indices(self):
        avg = 0
        class_ids = sorted(self._class2gt.keys())
        for class_id in class_ids:
            gt = np.array(self._class2gt[class_id])
            probs = np.array(self._class2probs[class_id])
            auc = prc_auc_score(gt, probs)
            avg += auc
        avg /= len(class_ids)
        return avg
    
    def _compute_without_indices(self):
        gt = np.concatenate(self._gt, axis=0)
        probs = np.concatenate(self._probs, axis=0)
        return prc_auc_macro_avg_fn(gt, probs)
        
    def compute(self):
        return self._compute()