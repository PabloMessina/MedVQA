import torch

class NTXentLoss(torch.nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    """
    def __init__(self, temperature, device):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = device

    def forward(self, pos_sim, neg_sim):
        # Create matrix of shape (K_pos, 1 + K_neg), where the first column
        # is the positive similarity and the remaining columns are the
        # negative similarity values.
        sim = torch.empty((len(pos_sim), 1 + len(neg_sim)))
        sim[:, 0] = pos_sim
        sim[:, 1:] = neg_sim
        # Scale the similarities by the temperature parameter.
        sim /= self.temperature
        # Compute cross entropy loss.
        labels = torch.zeros(len(pos_sim), dtype=torch.long).to(self.device) # 0 is the positive pair
        loss = self.criterion(sim, labels)
        return loss
