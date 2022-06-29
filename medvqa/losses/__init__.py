import torch.nn as nn
from .dataset_aware_loss import DatasetAwareLoss
from .wbce import WeigthedBCEByClassLoss

_BINARY_MULTILABEL_LOSSES = {
    'bce': nn.BCEWithLogitsLoss,
    'wbce-c': WeigthedBCEByClassLoss,
}

def get_binary_multilabel_loss(loss_name):
    if loss_name not in _BINARY_MULTILABEL_LOSSES:
        raise Exception(f'Loss not found: {loss_name}')
    return _BINARY_MULTILABEL_LOSSES[loss_name]()
