import torch.nn as nn

class LinearFiLM(nn.Module):
    def __init__(self, in_dim, condition_dim):
        super(LinearFiLM, self).__init__()        
        self.gamma = nn.Linear(condition_dim, in_dim)
        self.beta = nn.Linear(condition_dim, in_dim)
    
    def forward(self, x, condition):
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return gamma * x + beta