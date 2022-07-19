from medvqa.utils.constants import METRIC2SHORT
import numbers

def _default_metric_getter(metrics_dict, key):
    return metrics_dict[key]

def _weighted_average(metrics_dict, metric_names, metric_weights, metric_getter):
    num = sum(metric_getter(metrics_dict, name) * metric_weights[name] for name in metric_names)
    den = sum(metric_weights[name] for name in metric_names)
    score = num / den
    return score

def get_merge_metrics_fn(train_metric_names, val_metric_names, metric_weights, w_train, w_val,
        metric_getter=_default_metric_getter):

    def merge_metrics_fn(train_metrics, val_metrics):
        # print('train_metric_names =', train_metric_names)
        # print('val_metric_names =', val_metric_names)
        # print('train_metrics =', train_metrics)
        # print('val_metrics =', val_metrics)
        train_score = _weighted_average(train_metrics, train_metric_names, metric_weights, metric_getter)
        val_score = _weighted_average(val_metrics, val_metric_names, metric_weights, metric_getter)
        score =  (train_score * w_train + val_score * w_val) / (w_train + w_val)
        assert isinstance(score, numbers.Number), type(score)
        return score

    return merge_metrics_fn

def get_hybrid_score_name(metric_names):
    return '+'.join(METRIC2SHORT[m] for m in metric_names)
