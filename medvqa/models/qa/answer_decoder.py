import torch
import torch.nn as nn
from medvqa.models.nlp.stacked_lstm_cell import StackedLSTMCell

class AnswerDecoder(nn.Module):
  
    def __init__(
        self,
        embedding_table,
        question_vec_size,
        embed_size,
        hidden_size,
        n_lstm_layers,
        start_idx,
        vocab_size,
        dropout_prob
    ):
        super().__init__()
        self.embedding_table = embedding_table
        self.question_vec_size = question_vec_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size    
        self.register_buffer('start_idx', torch.tensor(start_idx))
        self.vocab_size = vocab_size
        self.W_h = nn.Linear(question_vec_size, hidden_size, bias=False)
        self.W_c = nn.Linear(question_vec_size, hidden_size, bias=False)
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size, bias=True)
        self.n_lstm_layers = n_lstm_layers
        if n_lstm_layers > 1:
            self.stacked_lstm_cell = StackedLSTMCell(hidden_size, hidden_size, n_lstm_layers-1)
        self.dropout = nn.Dropout(dropout_prob)
        self.W_vocab = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        question_vectors,
        answers=None,
        max_answer_length=None,
        mode='train',
    ):
        if mode == 'train':
            batch_size, max_answer_length = answers.shape
            assert answers is not None
        else:
            batch_size = question_vectors.size(0)
            assert max_answer_length is not None

        y = self.embedding_table(self.start_idx).expand(batch_size, -1)
        h = self.W_h(question_vectors)
        c = self.W_c(question_vectors)
        multiple_layers = self.n_lstm_layers > 1
        if multiple_layers:
            h_stacked = [h for _ in range(self.n_lstm_layers-1)]
            c_stacked = [c for _ in range(self.n_lstm_layers-1)]

        if mode == 'train':
            answer_embeddings = self.embedding_table(answers.permute(1,0))

        output = []

        for t in range(max_answer_length):
            h, c = self.lstm_cell(y, (h, c))
            if multiple_layers:
                h_stacked, c_stacked = self.stacked_lstm_cell(h, h_stacked, c_stacked)
                h_final = h_stacked[-1]
            else:
                h_final = h
            output.append(self.W_vocab(h_final))
            if mode == 'train':
                y = answer_embeddings[t] # teacher-forcing
            else:
                y = self.embedding_table(torch.argmax(output[t], 1)) # greedy decoding

        output = torch.stack(output, 1)
        return output