import torch
import torch.nn as nn

from medvqa.models.nlp.positional_encoding import PositionalEncoding

class LSTMBasedTextDecoder(nn.Module):

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
  
    def forward(self, text_vectors, texts, lengths):
        batch_size, max_length = texts.shape    
        y = self.embedding_table(self.start_idx).expand(batch_size, -1)
        o = self.W_o(text_vectors)
        h = self.W_h(text_vectors)
        c = self.W_c(text_vectors)
        embedded_texts = self.embedding_table(texts.permute(1,0))
        assert embedded_texts.shape == (max_length, batch_size, self.embed_size)

        output = []
        for t in range(max_length):
            y_bar = torch.cat((y, o, text_vectors),1)
            h, c = self.lstm_cell(y_bar, (h, c))
            output.append(self.W_vocab(h))
            y = embedded_texts[t]

        output = torch.stack(output, 1)
        return output


class TransformerTextDecoder(nn.Module):
  
    def __init__(
        self,
        embedding_table,
        embed_size,
        hidden_size,
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
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.register_buffer('start_idx', torch.tensor(start_idx))
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_prob)
        self.decoder = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward
        ), num_layers=num_layers)
        self.W_vocab = nn.Linear(hidden_size, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def teacher_forcing_decoding(self, input_memory, texts, device):
        batch_size, max_text_length = texts.shape
        text_embeddings = self.pos_encoder(self.embedding_table(texts.permute(1,0)))
        assert text_embeddings.shape == (max_text_length, batch_size, self.embed_size)
        tgt_mask = self.generate_square_subsequent_mask(max_text_length).to(device)
        input_memory = input_memory.permute(1,0,2)
        decoded = self.decoder(text_embeddings, input_memory, tgt_mask=tgt_mask)
        vocab_logits = self.W_vocab(decoded)
        vocab_logits = vocab_logits.permute(1, 0, 2)
        assert vocab_logits.shape == (batch_size, max_text_length, self.vocab_size)
        return vocab_logits

    def greedy_search_decoding(self, input_memory, max_text_length, device):
        # print('DEBUG: greedy_search_decoding')
        batch_size = input_memory.size(0)
        input_memory = input_memory.permute(1,0,2)
        decoded_tokens = self.start_idx.expand(batch_size).unsqueeze(0)
        output = []

        # generation loop
        while len(output) < max_text_length:
            decoded_embedding = self.pos_encoder(self.embedding_table(decoded_tokens))
            tgt_mask = self.generate_square_subsequent_mask(decoded_embedding.size(0)).to(device)
            decoded = self.decoder(decoded_embedding, input_memory, tgt_mask=tgt_mask)
            vocab_logits = self.W_vocab(decoded[-1])
            output.append(torch.argmax(vocab_logits, 1))
            assert vocab_logits.shape == (batch_size, self.vocab_size)
            next_tokens = vocab_logits.argmax(1)
            next_tokens = next_tokens.unsqueeze(0)
            assert next_tokens.shape == (1, batch_size)
            decoded_tokens = torch.cat((decoded_tokens, next_tokens), 0)

        output = torch.stack(output, 1)
        assert output.shape == (batch_size, max_text_length)
        return output

    def forward(self, input_memory, device, texts=None, max_text_length=None, mode='train'):
        if mode == 'train':
            assert texts is not None
            return self.teacher_forcing_decoding(input_memory, texts, device)
        else:
            assert max_text_length is not None
            return self.greedy_search_decoding(input_memory, max_text_length, device)
        
    def get_name(self):
        return (f'TransfTextDec(es={self.embed_size},hs={self.hidden_size},'
                f'nl={self.num_layers},nh={self.nhead},dff={self.dim_feedforward})')