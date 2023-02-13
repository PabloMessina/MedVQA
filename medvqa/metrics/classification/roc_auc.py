from sklearn.metrics import roc_auc_score
from medvqa.utils.metrics import average_ignoring_nones

def roc_auc_fn(y_pred, y_true):
    try:
        micro_avg = roc_auc_score(y_true, y_pred, average='micro')
    except ValueError:
        print(y_true, y_pred)
        raise
    n_classes = y_pred.shape[1]
    per_class = [None] * n_classes
    for i in range(n_classes):
        try:
            per_class[i] = roc_auc_score(y_true.T[i], y_pred.T[i])
        except ValueError:
            pass
    macro_avg = average_ignoring_nones(per_class)
    return {
        'micro_avg': micro_avg,
        'macro_avg': macro_avg,
        'per_class': per_class,
    }