from medvqa.utils.constants import METRIC2SHORT

def get_merge_metrics_fn(metric_names, metric_weights, w_train, w_val):

    def merge_metrics_fn(train_metrics, val_metrics):
        train_value = 0
        val_value = 0
        weight_sum = 0
        for met in metric_names:
            w = metric_weights[met]
            train_value += train_metrics[met] * w
            val_value += val_metrics[met] * w
            weight_sum += w
        return (train_value * w_train + val_value * w_val) / (weight_sum * (w_train + w_val))

    return merge_metrics_fn

def get_hybrid_score_name(metric_names):
    return '+'.join(METRIC2SHORT[m] for m in metric_names)
