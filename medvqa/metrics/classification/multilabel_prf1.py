import numpy as np
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.engine import Events
from sklearn.metrics import f1_score
from medvqa.metrics.condition_aware_metric import ConditionAwareMetric
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric

class MultiLabelF1score(Metric):

    def __init__(self, output_transform=lambda x: x, device=None, record_scores=False):
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        if record_scores:
            self._scores = []
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()
        super().reset()

    def update(self, output):
        pred_tags, gt_tags = output
        # assert pred_tags.shape == gt_tags.shape
        # assert pred_tags.size(1) == 1956, pred_tags.shape
        # assert gt_tags.size(1) == 1956, gt_tags.shape
        # assert len(pred_tags.shape) == 2
        n = pred_tags.size(0)
        for i in range(n):
            pred = pred_tags[i]
            gt = gt_tags[i]            
            score = f1_score(gt, pred)
            self._acc_score += score
            if self.record_scores:
                self._scores.append(score)
        self._count += n

    def compute(self):
        if self._count == 0:
            raise NotComputableError('MultiLabel f1score needs at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count

class DatasetAwareMultilabelF1score:

    def __init__(self, output_transform, allowed_dataset_ids, record_scores=False):
        self.allowed_dataset_ids = allowed_dataset_ids
        self.output_transform = output_transform
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        if record_scores:
            self._scores = []
    
    def reset(self):
        self._acc_score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()

    def update(self, output):
        pred_labels, gt_labels = output
        n = pred_labels.size(0)
        for i in range(n):
            pred = pred_labels[i]
            gt = gt_labels[i]            
            score = f1_score(gt, pred)
            self._acc_score += score
            # if (i == 0):
            #     print('pred =', pred)
            #     print('gt =', gt)
            #     print('score =', score)
            #     print('self._acc_score =', self._acc_score)
            if self.record_scores:
                self._scores.append(score)
        self._count += n
    
    def attach(self, engine, metric_alias):
        
        def epoch_started_handler(unused_engine):
            self.reset()

        def iteration_completed_handler(engine):
            output = engine.state.output
            dataset_id = output['dataset_id'] # make sure your step_fn returns this
            if dataset_id in self.allowed_dataset_ids:
                self.update(self.output_transform(output))

        def epoch_completed_handler(engine):
            engine.state.metrics[metric_alias] = self.compute()

        engine.add_event_handler(Events.EPOCH_STARTED, epoch_started_handler)
        engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_completed_handler)
        engine.add_event_handler(Events.EPOCH_COMPLETED, epoch_completed_handler)

    def compute(self):
        if self._count == 0:
            raise NotComputableError('DatasetAwareMultilabelF1score needs at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count

class MultiLabelMicroAvgF1(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0
        super().reset()

    def update(self, output):
        pred_labels, gt_labels = output
        n, m = pred_labels.shape
        for i in range(n):
            for j in range(m):
                pred = pred_labels[i][j]
                gt = gt_labels[i][j]
                if pred:
                    if gt: self._tp += 1
                    else: self._fp += 1
                else:
                    if gt: self._fn += 1
                    else: self._tn += 1

    def compute(self):
        prec = self._tp / max(self._tp + self._fp, 1)
        rec = self._tp / max(self._tp + self._fn, 1)
        return (2 * prec * rec) / max(prec + rec, 1)

class MultiLabelMacroAvgF1(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._tp = []
        self._tn = []
        self._fp = []
        self._fn = []
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._tp.clear()
        self._tn.clear()
        self._fp.clear()
        self._fn.clear()
        super().reset()

    def update(self, output):
        pred_labels, gt_labels = output
        n, m = pred_labels.shape
        if len(self._tp) == 0:
            self._tp = [0] * m
            self._tn = [0] * m
            self._fp = [0] * m
            self._fn = [0] * m
        for i in range(n):
            for j in range(m):
                pred = pred_labels[i][j]
                gt = gt_labels[i][j]
                if pred:
                    if gt: self._tp[j] += 1
                    else: self._fp[j] += 1
                else:
                    if gt: self._fn[j] += 1
                    else: self._tn[j] += 1

    def compute(self):
        m = len(self._tp)
        mean_f1 = 0
        for j in range(m):            
            prec = self._tp[j] / max(self._tp[j] + self._fp[j], 1)
            rec = self._tp[j] / max(self._tp[j] + self._fn[j], 1)            
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            mean_f1 += f1
        mean_f1 /= m
        return mean_f1

class DatasetAwareMultiLabelMicroAvgF1(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids):
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0
        super().__init__(output_transform, allowed_dataset_ids)
    
    def reset(self):
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0

    def update(self, output):
        pred_labels, gt_labels = output
        n, m = pred_labels.shape
        for i in range(n):
            for j in range(m):
                pred = pred_labels[i][j]
                gt = gt_labels[i][j]
                if pred:
                    if gt: self._tp += 1
                    else: self._fp += 1
                else:
                    if gt: self._fn += 1
                    else: self._tn += 1

    def compute(self):
        prec = self._tp / max(self._tp + self._fp, 1)
        rec = self._tp / max(self._tp + self._fn, 1)
        return (2 * prec * rec) / max(prec + rec, 1)

class DatasetAwareMultiLabelMacroAvgF1(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids):
        self._tp = []
        self._tn = []
        self._fp = []
        self._fn = []
        super().__init__(output_transform, allowed_dataset_ids)
    
    def reset(self):
        self._tp.clear()
        self._tn.clear()
        self._fp.clear()
        self._fn.clear()

    def update(self, output):
        pred_labels, gt_labels = output
        n, m = pred_labels.shape
        if len(self._tp) == 0:
            self._tp = [0] * m
            self._tn = [0] * m
            self._fp = [0] * m
            self._fn = [0] * m
        for i in range(n):
            for j in range(m):
                pred = pred_labels[i][j]
                gt = gt_labels[i][j]
                if pred:
                    if gt: self._tp[j] += 1
                    else: self._fp[j] += 1
                else:
                    if gt: self._fn[j] += 1
                    else: self._tn[j] += 1

    def compute(self):
        m = len(self._tp)
        mean_f1 = 0
        for j in range(m):            
            prec = self._tp[j] / max(self._tp[j] + self._fp[j], 1)
            rec = self._tp[j] / max(self._tp[j] + self._fn[j], 1)            
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            mean_f1 += f1
        mean_f1 /= m
        return mean_f1

class MultiLabelPRF1(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._tp = []
        self._tn = []
        self._fp = []
        self._fn = []
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._tp.clear()
        self._tn.clear()
        self._fp.clear()
        self._fn.clear()
        super().reset()

    def update(self, output):
        pred_labels, gt_labels = output
        n, m = pred_labels.shape
        if len(self._tp) == 0:
            self._tp = [0] * m
            self._tn = [0] * m
            self._fp = [0] * m
            self._fn = [0] * m
        for i in range(n):
            for j in range(m):
                pred = pred_labels[i][j]
                gt = gt_labels[i][j]
                if pred:
                    if gt: self._tp[j] += 1
                    else: self._fp[j] += 1
                else:
                    if gt: self._fn[j] += 1
                    else: self._tn[j] += 1

    def compute(self):
        m = len(self._tp)
        mean_f1 = 0
        for j in range(m):            
            prec = self._tp[j] / max(self._tp[j] + self._fp[j], 1)
            rec = self._tp[j] / max(self._tp[j] + self._fn[j], 1)
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            mean_f1 += f1
        mean_f1 /= m
        p = [self._tp[i] / max(self._tp[i] + self._fp[i], 1) for i in range(m)]
        r = [self._tp[i] / max(self._tp[i] + self._fn[i], 1) for i in range(m)]
        f1 = [(2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0 for prec, rec in zip(p, r)]
        p_micro_avg = sum(self._tp) / max(sum(self._tp) + sum(self._fp), 1)
        r_micro_avg = sum(self._tp) / max(sum(self._tp) + sum(self._fn), 1)
        f1_micro_avg = (2 * p_micro_avg * r_micro_avg) / (p_micro_avg + r_micro_avg) if (p_micro_avg + r_micro_avg) > 0 else 0
        p_macro_avg = sum(p) / len(p)
        r_macro_avg = sum(r) / len(r)
        f1_macro_avg = sum(f1) / len(f1)
        return {
            'p': p,
            'r': r,
            'f1': f1,
            'p_micro_avg': p_micro_avg,
            'r_micro_avg': r_micro_avg,
            'f1_micro_avg': f1_micro_avg,
            'p_macro_avg': p_macro_avg,
            'r_macro_avg': r_macro_avg,
            'f1_macro_avg': f1_macro_avg,
        }
    
class ConditionAwareMultiLabelMultiClassMacroAvgF1(ConditionAwareMetric):

    def __init__(self, output_transform, condition_function):
        self._pred_labels_batches = []
        self._gt_labels_batches = []
        super().__init__(output_transform, condition_function)
    
    def reset(self):
        self._pred_labels_batches.clear()
        self._gt_labels_batches.clear()

    def update(self, output):
        pred_labels, gt_labels = output
        self._pred_labels_batches.append(pred_labels.cpu().numpy())
        self._gt_labels_batches.append(gt_labels.cpu().numpy())

    def compute(self):
        pred_labels = np.concatenate(self._pred_labels_batches, axis=0)
        gt_labels = np.concatenate(self._gt_labels_batches, axis=0)
        assert pred_labels.shape == gt_labels.shape
        m = pred_labels.shape[1]
        score = 0
        for j in range(m):
            y_true = gt_labels[:, j]
            y_pred = pred_labels[:, j]
            score += f1_score(y_true, y_pred, average='macro')
        score /= m
        return score


class ConditionAwareMultiLabelMacroAvgF1(ConditionAwareMetric):

    def __init__(self, output_transform, condition_function):
        self._tp = []
        self._tn = []
        self._fp = []
        self._fn = []
        super().__init__(output_transform, condition_function)
    
    def reset(self):
        self._tp.clear()
        self._tn.clear()
        self._fp.clear()
        self._fn.clear()

    def update(self, output):
        pred_labels, gt_labels = output
        
        n, m = pred_labels.shape
        if len(self._tp) == 0:
            self._tp = [0] * m
            self._tn = [0] * m
            self._fp = [0] * m
            self._fn = [0] * m
        for i in range(n):
            for j in range(m):
                pred = pred_labels[i][j]
                gt = gt_labels[i][j]
                if pred:
                    if gt: self._tp[j] += 1
                    else: self._fp[j] += 1
                else:
                    if gt: self._fn[j] += 1
                    else: self._tn[j] += 1

    def compute(self):
        m = len(self._tp)
        mean_f1 = 0
        for j in range(m):            
            prec = self._tp[j] / max(self._tp[j] + self._fp[j], 1)
            rec = self._tp[j] / max(self._tp[j] + self._fn[j], 1)            
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            mean_f1 += f1
        mean_f1 /= m
        return mean_f1