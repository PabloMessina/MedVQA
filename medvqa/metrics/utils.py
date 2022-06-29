from medvqa.utils.constants import METRIC2SHORT
import numbers

def _default_metric_getter(metrics_dict, key):
    return metrics_dict[key]

def get_merge_metrics_fn(metric_names, metric_weights, w_train, w_val,
        metric_getter=_default_metric_getter):

    def merge_metrics_fn(train_metrics, val_metrics):
        # print('train_metrics =', train_metrics)
        # print('val_metrics =', val_metrics)
        train_value = 0
        val_value = 0
        train_weight_sum = 0
        val_weight_sum = 0
        for met in metric_names:
            w = metric_weights[met]
            try:
                train_value += metric_getter(train_metrics, met) * w
                train_weight_sum += w
            except KeyError:
                pass
            try:
                val_value += metric_getter(val_metrics, met) * w
                val_weight_sum += w
            except KeyError:
                pass
        train_score = train_value / train_weight_sum
        val_score = val_value / val_weight_sum
        score =  (train_score * w_train + val_score * w_val) / (w_train + w_val)
        assert isinstance(score, numbers.Number), type(score)
        return score

    return merge_metrics_fn

def get_hybrid_score_name(metric_names):
    return '+'.join(METRIC2SHORT[m] for m in metric_names)
