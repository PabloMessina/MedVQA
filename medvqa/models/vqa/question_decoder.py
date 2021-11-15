import torch
import torch.nn as nn

class QuestionDecoder(nn.Module):

    def __init__(self, embedding_table, embed_size, hidden_size, encoded_size, vocab_size, start_idx):
        super().__init__()
        self.embedding_table = embedding_table
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encoded_size = encoded_size
        self.vocab_size = vocab_size
        self.register_buffer('start_idx', torch.tensor(start_idx))
        self.lstm_cell = nn.LSTMCell(input_size=embed_size + hidden_size + encoded_size,
                                    hidden_size=hidden_size,
                                    bias=True)
        self.W_o = nn.Linear(encoded_size, hidden_size)
        self.W_h = nn.Linear(encoded_size, hidden_size)
        self.W_c = nn.Linear(encoded_size, hidden_size)
        self.W_vocab = nn.Linear(hidden_size, vocab_size)
  
    def forward(self, question_vectors, questions, lengths):
        batch_size, max_length = questions.shape    
        y = self.embedding_table(self.start_idx).expand(batch_size, -1)
        o = self.W_o(question_vectors)
        h = self.W_h(question_vectors)
        c = self.W_c(question_vectors)
        embedded_questions = self.embedding_table(questions.permute(1,0))
        assert embedded_questions.shape == (max_length, batch_size, self.embed_size)

        output = []
        for t in range(max_length):
            y_bar = torch.cat((y, o, question_vectors),1)
            h, c = self.lstm_cell(y_bar, (h, c))
            output.append(self.W_vocab(h))
            y = embedded_questions[t]

        output = torch.stack(output, 1)
        return output