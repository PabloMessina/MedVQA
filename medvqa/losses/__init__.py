import torch.nn as nn
from .wbce import WeigthedByClassBCELoss
from .focal_wbce import FocalWeigthedByClassBCELoss
from .dataset_aware_loss import DatasetAwareLoss

__all__ = [
    'get_binary_multilabel_loss',
    'DatasetAwareLoss',
]

_BINARY_MULTILABEL_LOSSES = {
    'bce': nn.BCEWithLogitsLoss,
    'wbce-c': WeigthedByClassBCELoss,
    'focal-wbce-c': FocalWeigthedByClassBCELoss,
}

def get_binary_multilabel_loss(loss_name, **kwargs):
    if loss_name not in _BINARY_MULTILABEL_LOSSES:
        raise Exception(f'Loss not found: {loss_name}')
    return _BINARY_MULTILABEL_LOSSES[loss_name](**kwargs)