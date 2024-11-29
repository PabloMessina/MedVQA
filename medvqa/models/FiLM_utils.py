import torch.nn as nn

class LinearFiLM(nn.Module):
    def __init__(self, in_dim, condition_dim):
        super(LinearFiLM, self).__init__()
        self.in_dim = in_dim
        self.condition_dim = condition_dim
        self.gamma = nn.Linear(condition_dim, in_dim)
        self.beta = nn.Linear(condition_dim, in_dim)
    
    def forward(self, x, condition):
        assert x.size(-1) == self.in_dim, "x.shape = {}, self.in_dim = {}".format(x.shape, self.in_dim)
        assert condition.size(-1) == self.condition_dim, "condition.shape = {}, self.condition_dim = {}".format(condition.shape, self.condition_dim)
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return gamma * x + beta