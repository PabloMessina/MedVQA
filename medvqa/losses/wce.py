import torch
import torch.nn as nn

class WeigthedByClassCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        n_classes = output.size(1)
        # Get class counts in the target batch
        counts = torch.bincount(target, minlength=n_classes).float()
        n_actual_classes = (counts > 0).sum().item()
        # Get the weight for each class
        weights = (counts.sum() / (n_actual_classes * counts.clamp(1))) * counts.clamp(0, 1)
        # Get the weight for each sample
        weights = weights[target]
        # Compute the loss
        loss = self.ce(output, target) * weights
        return loss.mean()