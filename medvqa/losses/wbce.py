import torch.nn as nn

class WeigthedBCEByClassLoss(nn.Module):
    def __init__(self, max_factor=8):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.max_factor = max_factor

    def forward(self, output, target):
        target = target.float()
        not_target = 1 - target
        positive = target.sum(0)
        negative = not_target.sum(0)
        total = target.size(0)
        pos_weight = negative / positive.clamp(1, total)
        neg_weight = positive / negative.clamp(1, total)
        pos_weight.clamp_(1, self.max_factor)
        neg_weight.clamp_(1, self.max_factor)
        loss = (target * pos_weight + not_target * neg_weight) * self.bce(output, target)
        return loss.mean()