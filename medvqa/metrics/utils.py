from medvqa.utils.constants import METRIC2SHORT
import numbers

from medvqa.utils.logging_utils import print_orange

def _default_metric_getter(metrics_dict, key):
    return metrics_dict[key]

def _weighted_average(metrics_dict, metric_names, metric_weights, metric_getter, verbose=False):
    if verbose:
        for name in metric_names:
            print_orange(f'{name}: {metric_getter(metrics_dict, name):.4f}, weight = {metric_weights[name]}')
    num = sum(metric_getter(metrics_dict, name) * metric_weights[name] for name in metric_names)
    den = sum(metric_weights[name] for name in metric_names)
    score = num / den
    if verbose:
        print_orange(f'num = {num:.4f}, den = {den:.4f}, score = {score:.4f}')
    return score

def get_merge_metrics_fn(train_metric_names, val_metric_names, metric_weights, w_train, w_val,
        metric_getter=_default_metric_getter, verbose_the_first_time=True):

    assert val_metric_names is not None and len(val_metric_names) > 0 and w_val > 0, \
        f'val_metric_names = {val_metric_names}, w_val = {w_val}'

    if train_metric_names is None or len(train_metric_names) == 0 or w_train == 0:
        print('NOTE: Using only validation metrics')
        # Only use validation metrics
        def merge_metrics_fn(val_metrics):
            nonlocal verbose_the_first_time
            if verbose_the_first_time:
                print_orange('Val metrics:')
                score = _weighted_average(val_metrics, val_metric_names, metric_weights, metric_getter,
                                          verbose=True)
                print_orange(f'Final val score = {score:.4f}')
                verbose_the_first_time = False # Print only once
                return score
            else:
                return _weighted_average(val_metrics, val_metric_names, metric_weights, metric_getter)
    else:
        def merge_metrics_fn(train_metrics, val_metrics):
            nonlocal verbose_the_first_time
            if verbose_the_first_time:
                print_orange('Train metrics:')
                train_score = _weighted_average(train_metrics, train_metric_names, metric_weights, metric_getter, verbose=True)
                print_orange('Val metrics:')
                val_score = _weighted_average(val_metrics, val_metric_names, metric_weights, metric_getter, verbose=True)
                score =  (train_score * w_train + val_score * w_val) / (w_train + w_val)
                print_orange(f'Train score = {train_score:.4f}, Val score = {val_score:.4f}, Final score = {score:.4f}')
                print_orange(f'w_train = {w_train}, w_val = {w_val}')
                verbose_the_first_time = False
            else:
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
    short_names = [METRIC2SHORT.get(m, m) for m in sorted_unique]
    hybrid_name = '+'.join(short_names)
    if len(hybrid_name) > 50:
        short_names = [x[:2]+x[-2:] if len(x) > 4 else x for x in short_names]
        hybrid_name = '+'.join(short_names)
    return hybrid_name
