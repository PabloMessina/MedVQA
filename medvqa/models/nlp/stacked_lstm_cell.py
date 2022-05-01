import torch.nn as nn

class StackedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_size=input_size if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                        bias=True) for i in range(n_layers)
        ])
    
    def forward(self, x, h0, c0):
        assert len(h0) == self.n_layers
        assert len(c0) == self.n_layers
        h = h0[:]
        c = c0[:]
        for i, lstm_cell in enumerate(self.lstm_cells):
            h_i, c_i = lstm_cell(x, (h[i], c[i]))
            x = h_i
            h[i] = h_i
            c[i] = c_i
        return h, c