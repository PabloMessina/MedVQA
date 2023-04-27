import torch.nn as nn
from .wbce import WeigthedByClassBCELoss
from .focal_loss import FocalLoss
from .dataset_aware_loss import DatasetAwareLoss

__all__ = [
    'get_binary_multilabel_loss',
    'DatasetAwareLoss',
]

class Focal_BCE_WBCE_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, focal_weight=1.0, bce_weight=1.0, wbce_weight=1.0, adaptively_rescale_losses=True):
        super().__init__()        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.wbce_loss = WeigthedByClassBCELoss()
        tot = focal_weight + bce_weight + wbce_weight
        self.focal_weight = focal_weight / tot
        self.bce_weight = bce_weight / tot
        self.wbce_weight = wbce_weight / tot
        self.adaptively_rescale_losses = adaptively_rescale_losses
        print('Focal_BCE_WBCE_Loss(): focal_weight =',
               self.focal_weight, 'bce_weight =', self.bce_weight, 'wbce_weight =', self.wbce_weight)

    def forward(self, output, target):
        loss1 = self.focal_loss(output, target)
        loss2 = self.bce_loss(output, target)
        loss3 = self.wbce_loss(output, target)
        if self.adaptively_rescale_losses:
            tot = (loss1 + loss2 + loss3).detach().item()
            w1 = tot / (loss1.detach().item() + 1e-6)
            w2 = tot / (loss2.detach().item() + 1e-6)
            w3 = tot / (loss3.detach().item() + 1e-6)
            w1 *= self.focal_weight
            w2 *= self.bce_weight
            w3 *= self.wbce_weight
        else:
            w1 = self.focal_weight
            w2 = self.bce_weight
            w3 = self.wbce_weight
        return w1 * loss1 + w2 * loss2 + w3 * loss3

_BINARY_MULTILABEL_LOSSES = {
    'bce': nn.BCEWithLogitsLoss,
    'wbce-c': WeigthedByClassBCELoss,
    'focal+bce+wbce-c': Focal_BCE_WBCE_Loss,
    'focal-loss': FocalLoss,
}

def get_binary_multilabel_loss(loss_name, **kwargs):
    if loss_name not in _BINARY_MULTILABEL_LOSSES:
        raise Exception(f'Loss not found: {loss_name}')
    return _BINARY_MULTILABEL_LOSSES[loss_name](**kwargs)