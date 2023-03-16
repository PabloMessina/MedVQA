import numpy as np
import heapq
import random
import itertools
from tqdm import tqdm

from medvqa.utils.files import load_json_file, load_pickle
from medvqa.metrics.classification.multilabel_prf1 import MultiLabelPRF1
from medvqa.utils.logging import print_blue

def _apply_noise(weights, coef):
    # coef is the standard deviation of the noise
    weights = weights + np.random.normal(0, coef, weights.shape)
    w_min = weights.min()
    if w_min <= 0:
        weights = weights - w_min + 1e-3
    return weights


class _Item:
    def __init__(self, weights, threshold, score):
        self.weights = weights
        self.threshold = threshold
        self.score = score
    
    def __lt__(self, other):
        return self.score < other.score

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

class MultilabelOptimalEnsembleSearcher:
    def __init__(self, probs, gt, topk=10):
    
        assert len(probs.shape) == 3
        assert len(gt.shape) == 2
        assert probs[0].shape == gt.shape
        
        self.gt = gt
        self.probs = probs
        self.n = gt.shape[0]
        self.m = gt.shape[1]
        self.k = probs.shape[0]
        self.topk = topk
        self.minheaps = [[] for _ in range(self.m)]
        
    def _update_minheap(self, i, weights, threshold, score):
        item = _Item(weights, threshold, score)
        h = self.minheaps[i]
        if len(h) < self.topk:
            heapq.heappush(h, item)
        else:
            heapq.heappushpop(h, item)
        
    def try_basic_weight_heuristics(self):
        print('Trying basic weight heuristics...')
        # For each model, try 1 for that model and 0 for the others
        # This is equivalent to trying the model alone
        print('  1) Try each model alone')
        for i in tqdm(range(self.k)):
            weights = np.zeros(self.k)
            weights[i] = 1
            merged_probs = np.average(self.probs, 0, weights=weights)
            for j in range(self.m):
                threshold, score = best_threshold_and_f1_score(merged_probs.T[j], self.gt.T[j])
                self._update_minheap(j, weights, threshold, score)
        # Try 1 for two models and 0 for the others, for each pair of models
        print('  2) Try each pair of models together')
        for (i, j) in tqdm(itertools.combinations(range(self.k), 2)):
            weights = np.zeros(self.k)
            weights[i] = 1
            weights[j] = 1
            weights /= weights.sum()
            merged_probs = np.average(self.probs, 0, weights=weights)
            for k in range(self.m):
                threshold, score = best_threshold_and_f1_score(merged_probs.T[k], self.gt.T[k])
                self._update_minheap(k, weights, threshold, score)
        # Try the average of all models
        print('  3) Try the average of all models')
        weights = np.ones(self.k)
        weights /= weights.sum()
        merged_probs = np.average(self.probs, 0, weights=weights)
        for j in range(self.m):
            threshold, score = best_threshold_and_f1_score(merged_probs.T[j], self.gt.T[j])
            self._update_minheap(j, weights, threshold, score)
        print('  Done')
    
    def sample_weights(self, n_tries):
        for _ in tqdm(range(n_tries)):
            weights = np.random.rand(self.k)
            weights /= weights.sum()
            merged_probs = np.average(self.probs, 0, weights=weights)
            for i in range(self.m):
                threshold, score = best_threshold_and_f1_score(merged_probs.T[i], self.gt.T[i])
                self._update_minheap(i, weights, threshold, score)
                
    def sample_weights_from_previous_ones(self, n_tries, noise_coef=0.05):
        probs = self.probs.transpose(1, 2, 0)
        for _ in tqdm(range(n_tries)):
            weights_array = np.empty((self.m, self.k))
            for i in range(self.m):
                item = random.choice(self.minheaps[i])
                weights_array[i] = _apply_noise(item.weights, noise_coef)
                weights_array[i] /= weights_array[i].sum()                
            merged_probs = (probs * weights_array).sum(-1)            
            for i in range(self.m):
                threshold, score = best_threshold_and_f1_score(merged_probs.T[i], self.gt.T[i])
                self._update_minheap(i, weights_array[i], threshold, score)
                
    def _compute_best_merged_probs_and_thresholds(self):
        probs = self.probs.transpose(1, 2, 0)
        weights_array = np.empty((self.m, self.k))
        threshold_array = np.empty((self.m))
        for i in range(self.m):
            _, item = max((item.score, item) for item in self.minheaps[i])
            weights_array[i] = item.weights
            threshold_array[i] = item.threshold
        merged_probs = (probs * weights_array).sum(-1)
        return merged_probs, threshold_array, weights_array

    def _evaluate(self, merged_probs, thresholds):
        pred = merged_probs > thresholds
        met = MultiLabelPRF1(device='cpu')
        met.update((pred, self.gt))
        res = met.compute()
        f1_macro = res["f1_macro_avg"]
        f1_micro = res["f1_micro_avg"]
        score = f1_macro + f1_micro
        print_blue(f'f1(macro)={f1_macro}, f1(micro)={f1_micro}, score={score}')
        return score

    def compute_best_merged_probs_and_thresholds(self):
        merged_probs, thresholds, weights = self._compute_best_merged_probs_and_thresholds()
        score = self._evaluate(merged_probs, thresholds)
        return dict(
            merged_probs=merged_probs,
            thresholds=thresholds,
            weights=weights,
            score=score,
        )

    def evaluate_best_predictions(self):
        merged_probs, thresholds, _ = self._compute_best_merged_probs_and_thresholds()
        return self._evaluate(merged_probs, thresholds)

class QuestionClassificationEnsembleSearcher(MultilabelOptimalEnsembleSearcher):
    def __init__(self, probs_paths, qa_adapted_reports_path, topk=6):
        
        probs_list = [load_pickle(x) for x in probs_paths]
        for i in range(1, len(probs_list)):
            assert probs_list[0].keys() == probs_list[i].keys()
        
        report_ids = list(probs_list[0].keys())

        qa_reports = load_json_file(qa_adapted_reports_path)
        nq = len(qa_reports['questions'])
        gt = np.zeros((len(report_ids), nq))
        for i, ri in enumerate(report_ids):
            for j in qa_reports['reports'][ri]['question_ids']:
                gt[i][j] = 1

        n, m = gt.shape
        probs = np.empty((len(probs_list), n, m), dtype=float)
        for i in range(len(probs_list)):
            probs[i] = np.array([probs_list[i][rid].numpy() for rid in report_ids])

        assert len(report_ids) == n
        assert len(next(iter(probs_list[0].values()))) == m

        self.report_ids = report_ids

        super().__init__(probs, gt, topk=topk)

    def compute_best_merged_probs_and_thresholds(self):
        merged_probs, thresholds = self._compute_best_merged_probs_and_thresholds()
        score = self._evaluate(merged_probs, thresholds)
        merged_probs = { rid : merged_probs[i] for i, rid in enumerate(self.report_ids) }
        return dict(
            merged_probs=merged_probs,
            thresholds=thresholds,
            score=score,
        )