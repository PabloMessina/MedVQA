import torch
import torch.nn as nn
import math

class PositionalEncodingMode:
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"

    @staticmethod
    def values():
        return [PositionalEncodingMode.SINUSOIDAL, PositionalEncodingMode.LEARNED]

class PositionalEncoding(nn.Module):
  
    def __init__(self, d_model, dropout=0.1, max_len=5000, mode=PositionalEncodingMode.SINUSOIDAL):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.mode = mode

        if mode == PositionalEncodingMode.SINUSOIDAL:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
            self.register_buffer('pe', pe)
        elif mode == PositionalEncodingMode.LEARNED:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x):
        if self.mode == PositionalEncodingMode.SINUSOIDAL:
            x = x + self.pe[:x.size(0), :]
        elif self.mode == PositionalEncodingMode.LEARNED:
            # print('x.shape', x.shape)
            pos_enc = self.pe(torch.arange(x.size(0), device=x.device)).unsqueeze(1) # (seq_len, 1, d_model)
            # print('pos_enc.shape', pos_enc.shape)
            x = x + pos_enc
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self.dropout(x)