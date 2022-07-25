import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
  
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerAnswerDecoder(nn.Module):
  
    def __init__(
        self,
        embedding_table,
        embed_size,
        hidden_size,
        question_vec_size,
        image_local_feat_size,
        nhead,
        dim_feedforward,
        num_layers,
        start_idx,
        vocab_size,
        dropout_prob,
    ):
        assert embed_size == hidden_size
        super().__init__()
        self.embedding_table = embedding_table
        self.embed_size = embed_size
        self.hidden_size = hidden_size    
        self.register_buffer('start_idx', torch.tensor(start_idx))
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_prob)
        self.decoder = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward
        ), num_layers=num_layers)
        self.W_vocab = nn.Linear(hidden_size, vocab_size)
        self.W_local_feat = nn.Linear(image_local_feat_size, hidden_size)    
        self.W_global_feat = nn.Linear(image_local_feat_size * 2, hidden_size)
        self.W_q = nn.Linear(question_vec_size, hidden_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_image_question_memory(self, local_feat, global_feat, question_vectors):
        # merge image and question
        batch_size = local_feat.size(0)
        image_question_memory = torch.cat((
            self.W_local_feat(local_feat),
            self.W_global_feat(global_feat).view(batch_size, 1, -1),
            self.W_q(question_vectors).view(batch_size, 1, -1),
        ), 1)
        return image_question_memory
    
    def teacher_forcing_decoding(self, local_feat, global_feat, question_vectors, answers, device):
        batch_size, max_answer_length = answers.shape
        answer_embeddings = self.pos_encoder(self.embedding_table(answers.permute(1,0)))
        assert answer_embeddings.shape == (max_answer_length, batch_size, self.embed_size)
        tgt_mask = self.generate_square_subsequent_mask(max_answer_length).to(device)
        image_question_memory = self.get_image_question_memory(local_feat, global_feat, question_vectors)
        image_question_memory = image_question_memory.permute(1,0,2)
        # print('device=',device)
        # print('answer_embeddings.get_device()=', answer_embeddings.get_device())
        # print('image_question_memory.get_device()=', image_question_memory.get_device())
        # print('tgt_mask.get_device()=', tgt_mask.get_device())
        decoded = self.decoder(answer_embeddings,
                            image_question_memory,
                            tgt_mask=tgt_mask)        
        vocab_logits = self.W_vocab(decoded)
        vocab_logits = vocab_logits.permute(1, 0, 2)
        assert vocab_logits.shape == (batch_size, max_answer_length, self.vocab_size)
        return vocab_logits

    def greedy_search_decoding(self, local_feat, global_feat, question_vectors, max_answer_length, device):
        batch_size = local_feat.size(0)
        image_question_memory = self.get_image_question_memory(local_feat, global_feat, question_vectors)
        image_question_memory = image_question_memory.permute(1,0,2)
        decoded_tokens = self.start_idx.expand(batch_size).unsqueeze(0)
        output = []

        # generation loop
        while len(output) < max_answer_length:
            decoded_embedding = self.pos_encoder(self.embedding_table(decoded_tokens))
            tgt_mask = self.generate_square_subsequent_mask(decoded_embedding.size(0)).to(device)
            decoded = self.decoder(decoded_embedding,
                                image_question_memory,
                                tgt_mask=tgt_mask)
            vocab_logits = self.W_vocab(decoded[-1])
            output.append(torch.argmax(vocab_logits, 1))
            assert vocab_logits.shape == (batch_size, self.vocab_size)
            next_tokens = vocab_logits.argmax(1)
            next_tokens = next_tokens.unsqueeze(0)
            assert next_tokens.shape == (1, batch_size)
            decoded_tokens = torch.cat((decoded_tokens, next_tokens), 0)

        output = torch.stack(output, 1)
        assert output.shape == (batch_size, max_answer_length)
        return output

    def forward(
        self, local_feat, global_feat, question_vectors, device,
        answers=None, max_answer_length=None, mode='train',
    ):
        if mode == 'train':
            assert answers is not None
            return self.teacher_forcing_decoding(
                    local_feat, global_feat, question_vectors, answers, device)
        else:
            assert max_answer_length is not None
            return self.greedy_search_decoding(
                    local_feat, global_feat, question_vectors,
                    max_answer_length, device)