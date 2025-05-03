import numpy as np
import heapq
import random
import itertools
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from medvqa.metrics.classification.prc_auc import prc_auc_score
from medvqa.utils.files_utils import load_json, load_pickle
from medvqa.utils.logging_utils import print_blue
from medvqa.utils.metrics_utils import (
    best_threshold_and_accuracy_score,
    best_threshold_and_f1_score,
    best_threshold_and_precision_score,
)

def _apply_noise(weights, coef):
    # coef is the standard deviation of the noise
    weights = weights + np.random.normal(0, coef, weights.shape)
    w_min = weights.min()
    if w_min <= 0:
        weights = weights - w_min + 1e-3
    return weights

class _Item:
    def __init__(self, weights, score, threshold):
        self.weights = weights
        self.score = score
        self.threshold = threshold
    
    def __lt__(self, other):
        return self.score < other.score

class MultilabelOptimalEnsembleSearcher:
    def __init__(self, probs, gt, topk=10, score_name='f1'):
        
        assert probs.ndim == 3 # (k, n, m)
        assert gt.ndim == 2 # (n, m)
        assert probs[0].shape == gt.shape
        assert score_name in ['f1', 'precision', 'accuracy', 'prc_auc', 'roc_auc']
        self.use_threshold = score_name in ['f1', 'precision', 'accuracy'] # whether to use threshold or not
        if score_name == 'f1':
            self.score_func = best_threshold_and_f1_score
        elif score_name == 'precision':
            self.score_func = best_threshold_and_precision_score
        elif score_name == 'accuracy':
            self.score_func = best_threshold_and_accuracy_score
        elif score_name == 'prc_auc':
            self.score_func = prc_auc_score
        elif score_name == 'roc_auc':
            self.score_func = roc_auc_score
        else: assert False # should not reach here

        self.gt = gt
        self.probs = probs
        self.n = gt.shape[0] # number of samples
        self.m = gt.shape[1] # number of classes
        self.k = probs.shape[0] # number of models
        self.topk = topk # number of best weights to keep
        self.minheaps = [[] for _ in range(self.m)] # keep track of the topk best weights for each class
        
    def _update_minheap(self, i, weights, score, threshold=None):
        item = _Item(weights, score, threshold)
        h = self.minheaps[i]
        if len(h) < self.topk:
            heapq.heappush(h, item)
        else:
            heapq.heappushpop(h, item)
        
    def try_basic_weight_heuristics(self):
        print('Trying basic weight heuristics...')
        
        # For each model, try 1 for that model and 0 for the others
        # This is equivalent to trying the model alone
        print('1) Try each model alone:')
        for i in tqdm(range(self.k)):
            weights = np.zeros(self.k)
            weights[i] = 1
            merged_probs = np.average(self.probs, 0, weights=weights)
            for j in range(self.m):
                if self.use_threshold:
                    threshold, score = self.score_func(self.gt.T[j], merged_probs.T[j])
                else:
                    score = self.score_func(self.gt.T[j], merged_probs.T[j])
                    threshold = None
                self._update_minheap(j, weights, score, threshold)
        
        # Try 1 for two models and 0 for the others, for each pair of models
        print('2) Try pairs of models:')
        pairs = list(itertools.combinations(range(self.k), 2))
        if len(pairs) > 50:
            pairs = random.sample(pairs, 50)
        for (i, j) in tqdm(pairs):
            weights = np.zeros(self.k)
            weights[i] = 1
            weights[j] = 1
            weights /= weights.sum()
            merged_probs = np.average(self.probs, 0, weights=weights)
            for k in range(self.m):
                if self.use_threshold:
                    threshold, score = self.score_func(self.gt.T[k], merged_probs.T[k])
                else:
                    score = self.score_func(self.gt.T[k], merged_probs.T[k])
                    threshold = None
                self._update_minheap(k, weights, score, threshold)
        
        # Try the average of all models
        print('3) Try the average of all models')
        weights = np.ones(self.k)
        weights /= weights.sum()
        merged_probs = np.average(self.probs, 0, weights=weights)
        for j in range(self.m):
            if self.use_threshold:
                threshold, score = self.score_func(self.gt.T[j], merged_probs.T[j])
            else:
                score = self.score_func(self.gt.T[j], merged_probs.T[j])
                threshold = None
            self._update_minheap(j, weights, score, threshold)
        
        # Try weights proportional to the scores of the models
        print('4) Try weights proportional to the scores of the individual models')
        for j in range(self.m): # for each class
            model_scores = []
            for i in range(self.k): # for each model
                model_probs = self.probs[i].T[j]
                if self.use_threshold:
                    _, score = self.score_func(self.gt.T[j], model_probs)
                else:
                    score = self.score_func(self.gt.T[j], model_probs)
                model_scores.append(score + 1e-6) # add a small value to avoid division by zero
            weights = np.array(model_scores)
            weights /= weights.sum() # normalize
            merged_probs = np.average(self.probs[:, :, j], 0, weights=weights) # (k, n) -> (n)
            assert merged_probs.shape == self.gt.T[j].shape
            if self.use_threshold:
                threshold, score = self.score_func(self.gt.T[j], merged_probs)
            else:
                score = self.score_func(self.gt.T[j], merged_probs)
                threshold = None
            self._update_minheap(j, weights, score, threshold)
        print('Done!')
    
    def sample_weights(self, n_tries):
        """
        Randomly sample weights and update the ensemble for each label
        """
        print('Sampling weights...')
        for _ in tqdm(range(n_tries)):
            weights = np.random.rand(self.k)
            weights /= weights.sum() # normalize
            merged_probs = np.average(self.probs, 0, weights=weights)
            for i in range(self.m):
                if self.use_threshold:
                    threshold, score = self.score_func(self.gt.T[i], merged_probs.T[i])
                else:
                    score = self.score_func(self.gt.T[i], merged_probs.T[i])
                    threshold = None
                self._update_minheap(i, weights, score, threshold)
                
    def sample_weights_from_previous_ones(self, n_tries, noise_coef=0.05):
        """
        Randomly sample weights around the best weights found so far
        """
        print('Sampling weights around the best weights found so far...')
        probs = self.probs.transpose(1, 2, 0)
        for _ in tqdm(range(n_tries)):
            weights_array = np.empty((self.m, self.k))
            for i in range(self.m):
                item = random.choice(self.minheaps[i])
                weights_array[i] = _apply_noise(item.weights, noise_coef)
                weights_array[i] /= weights_array[i].sum()                
            merged_probs = (probs * weights_array).sum(-1)            
            for i in range(self.m):
                if self.use_threshold:
                    threshold, score = self.score_func(self.gt.T[i], merged_probs.T[i])
                else:
                    score = self.score_func(self.gt.T[i], merged_probs.T[i])
                    threshold = None
                self._update_minheap(i, weights_array[i], score, threshold)
                
    def _compute_best_merged_probs_thresholds_weights(self):
        probs = self.probs.transpose(1, 2, 0) # (n, m, k)
        weights_array = np.empty((self.m, self.k))
        threshold_array = np.empty((self.m))
        for i in range(self.m): # for each label
            _, item = max((item.score, item) for item in self.minheaps[i])
            weights_array[i] = item.weights
            threshold_array[i] = item.threshold
        merged_probs = (probs * weights_array).sum(-1) # (n, m, k) * (m, k) -> (n, m, k) -> (n, m)
        return merged_probs, threshold_array, weights_array
    
    def _compute_best_merged_probs_weights(self):
        probs = self.probs.transpose(1, 2, 0) # (n, m, k)
        weights_array = np.empty((self.m, self.k))
        for i in range(self.m): # for each label
            _, item = max((item.score, item) for item in self.minheaps[i])
            weights_array[i] = item.weights
        merged_probs = (probs * weights_array).sum(-1) # (n, m, k) * (m, k) -> (n, m, k) -> (n, m)
        return merged_probs, weights_array

    def _evaluate(self, merged_probs, thresholds=None, verbose=True):
        if self.use_threshold:
            pred = merged_probs > thresholds
            score = np.mean([self.score_func(self.gt.T[i], pred.T[i]) for i in range(self.m)])
            if verbose:
                print_blue(f'score={score}')
            return score
        else:
            score = np.mean([self.score_func(self.gt.T[i], merged_probs.T[i]) for i in range(self.m)])
            if verbose:
                print_blue(f'score={score}')
            return score

    def compute_best_merged_probs_thresholds_weights(self, verbose=True):
        assert self.use_threshold, 'This method is only for threshold-based metrics'
        merged_probs, thresholds, weights = self._compute_best_merged_probs_thresholds_weights()
        score = self._evaluate(merged_probs, thresholds, verbose=verbose)
        return dict(
            merged_probs=merged_probs,
            thresholds=thresholds,
            weights=weights,
            score=score,
        )
    
    def compute_best_merged_probs_weights(self, verbose=True):
        merged_probs, weights = self._compute_best_merged_probs_weights()
        score = self._evaluate(merged_probs, verbose=verbose)
        return dict(
            merged_probs=merged_probs,
            weights=weights,
            score=score,
        )

    def evaluate_best_predictions(self):
        if self.use_threshold:
            merged_probs, thresholds, _ = self._compute_best_merged_probs_thresholds_weights()
            return self._evaluate(merged_probs, thresholds)
        else:
            merged_probs, _ = self._compute_best_merged_probs_weights()
            return self._evaluate(merged_probs)

class QuestionClassificationEnsembleSearcher(MultilabelOptimalEnsembleSearcher):
    def __init__(self, probs_paths, qa_adapted_reports_path, topk=6):
        
        probs_list = [load_pickle(x) for x in probs_paths]
        for i in range(1, len(probs_list)):
            assert probs_list[0].keys() == probs_list[i].keys()
        
        report_ids = list(probs_list[0].keys())

        qa_reports = load_json(qa_adapted_reports_path)
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
        merged_probs, thresholds, _ = self._compute_best_merged_probs_thresholds_weights()
        score = self._evaluate(merged_probs, thresholds)
        merged_probs = { rid : merged_probs[i] for i, rid in enumerate(self.report_ids) }
        return dict(
            merged_probs=merged_probs,
            thresholds=thresholds,
            score=score,
        )