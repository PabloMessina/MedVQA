import torch.nn as nn
import torch

class FocalWeigthedByClassBCELoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma

    def forward(self, logits, target):
        target = target.float()
        not_target = 1 - target
        positive = target.sum(0)
        negative = not_target.sum(0)
        total = target.size(0)
        pos_weight = negative / positive.clamp(1, total)
        neg_weight = positive / negative.clamp(1, total)
        pos_weight.clamp_(1, total)
        neg_weight.clamp_(1, total)
        bce_loss = self.bce(logits, target)
        w_loss = (target * pos_weight + not_target * neg_weight) * bce_loss
        # focal loss part based on https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * not_target
        fw_loss = ((1 - p_t) ** self.gamma) * w_loss
        loss = fw_loss.mean()
        return loss