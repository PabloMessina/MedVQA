from torch import nn

# Taken from https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims = None,
        dropout = 0,
        activation = nn.ReLU,
        normalization = None,
    ):
        super().__init__()

        layers = nn.ModuleList()

        if hidden_dims is None:
            hidden_dims = []

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.hidden_dims = hidden_dims

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)