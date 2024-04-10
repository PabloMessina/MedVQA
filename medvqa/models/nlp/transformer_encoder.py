from typing import Optional
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from medvqa.models.nlp.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_ff: int, nlayers: int, dropout: float = 0.):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    # def forward(self, src: Tensor, src_key_padding_mask: Tensor) -> Tensor:
    def forward(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_key_padding_mask: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, d_model]``
        """
        src = self.pos_encoder(src)
        # print('src:', src)
        # print('src.shape:', src.shape)
        # print('src_key_padding_mask:', src_key_padding_mask)
        # print('src_key_padding_mask.shape:', src_key_padding_mask.shape)
        output = self.transformer_encoder(src=src, src_key_padding_mask=src_key_padding_mask)
        # print('output:', output)
        # print('output.shape:', output.shape)
        return output