import torch
import torch.nn as nn

class ThresholdLoss(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, values):
        loss = torch.where(values < self.threshold, self.threshold - values, torch.zeros_like(values))
        return loss.mean()