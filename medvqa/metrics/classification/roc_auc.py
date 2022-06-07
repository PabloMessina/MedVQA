from sklearn.metrics import roc_auc_score

def roc_auc_fn(y_pred, y_true):
    return roc_auc_score(y_true, y_pred, average='micro')