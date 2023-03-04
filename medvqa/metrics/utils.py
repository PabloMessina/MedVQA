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

    assert val_metric_names is not None and len(val_metric_names) > 0 and w_val > 0, \
        f'val_metric_names = {val_metric_names}, w_val = {w_val}'

    if train_metric_names is None or len(train_metric_names) == 0 or w_train == 0:
        print('NOTE: Using only validation metrics')
        # Only use validation metrics
        def merge_metrics_fn(val_metrics):
            return _weighted_average(val_metrics, val_metric_names, metric_weights, metric_getter)
    else:
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

def get_hybrid_score_name(*metric_names):
    unique = set()
    for names in metric_names:
        assert type(names) is list
        unique.update(names)
    sorted_unique = sorted(list(unique))
    short_names = [METRIC2SHORT[m] for m in sorted_unique]
    hybrid_name = '+'.join(short_names)
    if len(hybrid_name) > 50:
        short_names = [x[:2]+x[-2:] if len(x) > 4 else x for x in short_names]
        hybrid_name = '+'.join(short_names)
    return hybrid_name
