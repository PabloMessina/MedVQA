import torch.nn as nn

class WeigthedByClassBCELoss(nn.Module):
    def __init__(self, classes_mask=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.classes_mask = classes_mask
        if classes_mask is not None:
            self.n_classes = classes_mask.sum()
            print(f'WeightedBCEByClassLoss(): self.classes_mask = {self.classes_mask}, self.n_classes = {self.n_classes}')

    def forward(self, output, target):
        target = target.float()
        not_target = 1 - target
        positive = target.sum(0)
        negative = not_target.sum(0)
        total = target.size(0)
        pos_weight = negative / positive.clamp(1, total)
        neg_weight = positive / negative.clamp(1, total)
        pos_weight.clamp_(1, total)
        neg_weight.clamp_(1, total)
        loss = (target * pos_weight + not_target * neg_weight) * self.bce(output, target)
        if self.classes_mask is not None:
            loss = loss * self.classes_mask
            loss = loss.sum() / (self.n_classes * total)
        else:
            loss = loss.mean()
        return loss