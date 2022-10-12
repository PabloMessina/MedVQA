import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class BiLSTMBasedTextEncoder(nn.Module):
    
    def __init__(self, embedding_table, embed_size, hidden_size, output_size, device):
        super().__init__()
        self.device = device
        self.embedding_table = embedding_table
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.W_out = nn.Linear(hidden_size * 4, output_size)
  
    def forward(self, texts, lengths):
        embedded_texts = self.embedding_table(texts)
        packed_texts = pack_padded_sequence(embedded_texts, lengths, batch_first=True)
        batch_size, max_question_length = texts.shape

        assert max_question_length == lengths.max()

        h0 = self.init_hidden_state(batch_size)
        c0 = self.init_cell_state(batch_size)
        _, (hidden_states, cell_states) = self.lstm(packed_texts, (h0, c0))
        # all_hidden_states, _ = pad_packed_sequence(output,
        #                                            batch_first=True,
        #                                            padding_value=0,
        #                                            total_length=max_question_length)
        # return all_hidden_states, hidden_states, cell_states
        return self.W_out(torch.cat((hidden_states[0], hidden_states[1], cell_states[0], cell_states[1]), 1))

    def init_hidden_state(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=self.device)
  
    def init_cell_state(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size, device=self.device)