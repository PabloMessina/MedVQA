import torch
import torch.nn as nn

class AnswerDecoder(nn.Module):
  
    def __init__(
        self,
        embedding_table,
        image_local_feat_size,
        question_vec_size,
        embed_size,
        hidden_size,
        start_idx,
        vocab_size,
        dropout_prob
    ):
        super().__init__()
        self.embedding_table = embedding_table
        self.image_local_feat_size = image_local_feat_size
        self.question_vec_size = question_vec_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size    
        self.register_buffer('start_idx', torch.tensor(start_idx))
        self.vocab_size = vocab_size
        self.W_g2o = nn.Linear(image_local_feat_size * 2, hidden_size)
        self.W_h = nn.Linear(question_vec_size, hidden_size, bias=False)
        self.W_c = nn.Linear(question_vec_size, hidden_size, bias=False)
        self.lstm_cell = nn.LSTMCell(input_size=embed_size + hidden_size,
                                     hidden_size=hidden_size,
                                     bias=True)
        self.W_attn = nn.Linear(image_local_feat_size, hidden_size, bias=False)
        self.W_u = nn.Linear(image_local_feat_size + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.W_vocab = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        image_local_features,
        image_global_features,
        question_vectors,
        answers=None,
        max_answer_length=None,
        mode='train',
    ):
        if mode == 'train':
            batch_size, max_answer_length = answers.shape
            assert answers is not None
        else:
            batch_size = image_local_features.size(0)
            assert max_answer_length is not None

        y = self.embedding_table(self.start_idx).expand(batch_size, -1)
        # o = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        o = self.W_g2o(image_global_features)
        h = self.W_h(question_vectors)
        c = self.W_c(question_vectors)

        if mode == 'train':
            answer_embeddings = self.embedding_table(answers.permute(1,0))
#             assert answer_embeddings.shape == (max_answer_length, batch_size, self.embed_size)

        output = []

        for t in range(max_answer_length):
            y_bar = torch.cat((y,o),1)
#             assert y_bar.shape == (batch_size, self.embed_size + self.hidden_size)
#             assert h.shape == (batch_size, self.hidden_size)
#             assert c.shape == (batch_size, self.hidden_size)
            h, c = self.lstm_cell(y_bar, (h, c))
            e = (self.W_attn(image_local_features) * h.unsqueeze(1)).sum(-1)
            att = torch.softmax(e,-1)
            a = (image_local_features * att.unsqueeze(2)).sum(1)
#             assert a.shape == (batch_size, self.image_local_feat_size)
            u = torch.cat((a,h),1)
#             assert u.shape == (batch_size, self.hidden_size + self.image_local_feat_size)
            v = self.W_u(u)
            o = self.dropout(torch.tanh(v))
#             assert o.shape == (batch_size, self.hidden_size)
            output.append(self.W_vocab(o))
            if mode == 'train':
                y = answer_embeddings[t] # teacher-forcing
            else:
                y = self.embedding_table(torch.argmax(output[t], 1)) # greedy search
#             assert y.shape == (batch_size, self.embed_size)

        output = torch.stack(output, 1)
#         assert output.shape == (batch_size, max_answer_length, self.vocab_size)
        return output