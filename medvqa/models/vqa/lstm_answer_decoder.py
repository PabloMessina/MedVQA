import torch
import torch.nn as nn
from medvqa.models.nlp.stacked_lstm_cell import StackedLSTMCell

# def _expand_tensor_for_beamsearch(x, k=3):
#     x = x.unsqueeze(1)
#     expanded_shape = list(x.shape)
#     expanded_shape[1] = k
#     final_shape = (-1, *expanded_shape[2:])
#     x = x.expand(expanded_shape).reshape(final_shape)
#     return x

class LSTMAnswerDecoder(nn.Module):
  
    def __init__(
        self,
        embedding_table,
        image_local_feat_size,
        question_vec_size,
        embed_size,
        hidden_size,
        n_lstm_layers,
        start_idx,
        vocab_size,
        dropout_prob,
        eos_idx=None,
        padding_idx=None,
    ):
        super().__init__()
        self.embedding_table = embedding_table
        self.image_local_feat_size = image_local_feat_size
        self.question_vec_size = question_vec_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size    
        self.register_buffer('start_idx', torch.tensor(start_idx))
        if eos_idx is not None: self.eos_idx = eos_idx
        if padding_idx is not None: self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.W_g2o = nn.Linear(image_local_feat_size * 2, hidden_size)
        self.W_h = nn.Linear(question_vec_size, hidden_size, bias=False)
        self.W_c = nn.Linear(question_vec_size, hidden_size, bias=False)
        self.lstm_cell = nn.LSTMCell(input_size=embed_size + hidden_size,
                                     hidden_size=hidden_size,
                                     bias=True)
        self.n_lstm_layers = n_lstm_layers
        if n_lstm_layers > 1:
            self.stacked_lstm_cell = StackedLSTMCell(hidden_size, hidden_size, n_lstm_layers-1)
        self.W_attn = nn.Linear(image_local_feat_size, hidden_size, bias=False)
        self.W_u = nn.Linear(image_local_feat_size + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.W_vocab = nn.Linear(hidden_size, vocab_size)

    def teacher_forcing_decoding(
        self,
        image_local_features,
        image_global_features,
        question_vectors,
        answers,
    ):
        batch_size, max_answer_length = answers.shape

        y = self.embedding_table(self.start_idx).expand(batch_size, -1)
        # o = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        o = self.W_g2o(image_global_features)
        h = self.W_h(question_vectors)
        c = self.W_c(question_vectors)
        multiple_layers = self.n_lstm_layers > 1
        if multiple_layers:
            h_stacked = [h for _ in range(self.n_lstm_layers-1)]
            c_stacked = [c for _ in range(self.n_lstm_layers-1)]
        
        answer_embeddings = self.embedding_table(answers.permute(1,0))
        # assert answer_embeddings.shape == (max_answer_length, batch_size, self.embed_size)

        output = []

        for t in range(max_answer_length):
            y_bar = torch.cat((y,o),1)
            # assert y_bar.shape == (batch_size, self.embed_size + self.hidden_size)
            # assert h.shape == (batch_size, self.hidden_size)
            # assert c.shape == (batch_size, self.hidden_size)
            h, c = self.lstm_cell(y_bar, (h, c))
            if multiple_layers:
                h_stacked, c_stacked = self.stacked_lstm_cell(h, h_stacked, c_stacked)
                h_final = h_stacked[-1]
            else:
                h_final = h

            e = (self.W_attn(image_local_features) * h_final.unsqueeze(1)).sum(-1)
            att = torch.softmax(e,-1)
            a = (image_local_features * att.unsqueeze(2)).sum(1)
            # assert a.shape == (batch_size, self.image_local_feat_size)
            u = torch.cat((a,h_final),1)
            # assert u.shape == (batch_size, self.hidden_size + self.image_local_feat_size)
            v = self.W_u(u)
            o = self.dropout(torch.tanh(v))
            # assert o.shape == (batch_size, self.hidden_size)
            output.append(self.W_vocab(o))
            y = answer_embeddings[t] # teacher-forcing
            # assert y.shape == (batch_size, self.embed_size)

        output = torch.stack(output, 1)
        # assert output.shape == (batch_size, max_answer_length, self.vocab_size)
        return output

    def greedy_search_decoding(
        self,
        image_local_features,
        image_global_features,
        question_vectors,
        max_answer_length,
    ):
        batch_size = image_local_features.size(0)

        y = self.embedding_table(self.start_idx).expand(batch_size, -1)
        # o = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        o = self.W_g2o(image_global_features)
        h = self.W_h(question_vectors)
        c = self.W_c(question_vectors)
        multiple_layers = self.n_lstm_layers > 1
        if multiple_layers:
            h_stacked = [h for _ in range(self.n_lstm_layers-1)]
            c_stacked = [c for _ in range(self.n_lstm_layers-1)]

        output = []

        for t in range(max_answer_length):
            y_bar = torch.cat((y,o),1)
            # assert y_bar.shape == (batch_size, self.embed_size + self.hidden_size)
            # assert h.shape == (batch_size, self.hidden_size)
            # assert c.shape == (batch_size, self.hidden_size)
            h, c = self.lstm_cell(y_bar, (h, c))
            if multiple_layers:
                h_stacked, c_stacked = self.stacked_lstm_cell(h, h_stacked, c_stacked)
                h_final = h_stacked[-1]
            else:
                h_final = h

            e = (self.W_attn(image_local_features) * h_final.unsqueeze(1)).sum(-1)
            att = torch.softmax(e,-1)
            a = (image_local_features * att.unsqueeze(2)).sum(1)
            # assert a.shape == (batch_size, self.image_local_feat_size)
            u = torch.cat((a,h_final),1)
            # assert u.shape == (batch_size, self.hidden_size + self.image_local_feat_size)
            v = self.W_u(u)
            o = self.dropout(torch.tanh(v))
            # assert o.shape == (batch_size, self.hidden_size)
            vocab_logits = self.W_vocab(o)
            output.append(torch.argmax(vocab_logits, 1))
            y = self.embedding_table(output[t]) # greedy search
            # assert y.shape == (batch_size, self.embed_size)

        output = torch.stack(output, 1)
        assert output.shape == (batch_size, max_answer_length)
        return output

    def beam_search_decoding(
        self,
        image_local_features,
        image_global_features,
        question_vectors,
        max_answer_length,
        device,
        k=3,
    ):
        batch_size = image_local_features.size(0)

        y0 = self.embedding_table(self.start_idx).expand(batch_size, -1)
        # o = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        o0 = self.W_g2o(image_global_features)
        h0 = self.W_h(question_vectors)
        c0 = self.W_c(question_vectors)
        multiple_layers = self.n_lstm_layers > 1
        if multiple_layers:
            h_stacked0 = [h0] * (self.n_lstm_layers-1)
            c_stacked0 = [c0] * (self.n_lstm_layers-1)
        
        y_list = [y0] * k
        o_list = [o0] * k
        h_list = [h0] * k
        c_list = [c0] * k
        if multiple_layers:
            h_stacked_list = [h_stacked0] * k
            c_stacked_list = [c_stacked0] * k

        o_list_aux = [o0.clone() for _ in range(k)]
        h_list_aux = [h0.clone() for _ in range(k)]
        c_list_aux = [c0.clone() for _ in range(k)]
        if multiple_layers:
            h_stacked_list_aux = [[h0.clone() for _ in range(self.n_lstm_layers-1)] for _ in range(k)]
            c_stacked_list_aux = [[c0.clone() for _ in range(self.n_lstm_layers-1)] for _ in range(k)]
        
        log_prob_sums = torch.zeros((k, batch_size), dtype=float, device=device)
        topk_list = [None] * k
        
        output_list = [torch.full((batch_size, max_answer_length),
                         self.padding_idx, dtype=int, device=device) for _ in range(k)]
        output_list_aux = [torch.full((batch_size, max_answer_length),
                         self.padding_idx, dtype=int, device=device) for _ in range(k)]
        
        EOS_found = [[False] * batch_size for _ in range(k)]
        EOS_found_aux = [[False] * batch_size for _ in range(k)]

        for t in range(max_answer_length):
            
            for ki in range(k if t > 0 else 1):
                
                y = y_list[ki]
                o = o_list[ki]
                h = h_list[ki]
                c = c_list[ki]
                if multiple_layers:
                    h_stacked = h_stacked_list[ki]
                    c_stacked = c_stacked_list[ki]

                y_bar = torch.cat((y,o),1)
                # assert y_bar.shape == (batch_size, self.embed_size + self.hidden_size)
                # assert h.shape == (batch_size, self.hidden_size)
                # assert c.shape == (batch_size, self.hidden_size)
                h, c = self.lstm_cell(y_bar, (h, c))
                if multiple_layers:
                    h_stacked, c_stacked = self.stacked_lstm_cell(h, h_stacked, c_stacked)
                    h_final = h_stacked[-1]
                else:
                    h_final = h

                e = (self.W_attn(image_local_features) * h_final.unsqueeze(1)).sum(-1)
                att = torch.softmax(e,-1)
                a = (image_local_features * att.unsqueeze(2)).sum(1)
                # assert a.shape == (batch_size, self.image_local_feat_size)
                u = torch.cat((a,h_final),1)
                # assert u.shape == (batch_size, self.hidden_size + self.image_local_feat_size)
                v = self.W_u(u)
                o = self.dropout(torch.tanh(v))
                # assert o.shape == (batch_size, self.hidden_size)
                
                vocab_logits = self.W_vocab(o)
                # assert (vocab_logits.shape == (batch_size, self.vocab_size))
                vocab_softmax = torch.softmax(vocab_logits, -1)
                # assert (vocab_softmax.shape == (batch_size, self.vocab_size))
                topk = vocab_softmax.topk(k)
                # assert topk.values.shape == (batch_size, k), f'topk.values.shape = {topk.values.shape}'
                
                topk_list[ki] = topk
                o_list[ki] = o
                h_list[ki] = h
                c_list[ki] = c
                if multiple_layers:
                    h_stacked_list[ki] = h_stacked
                    c_stacked_list[ki] = c_stacked
            
            for bi in range(batch_size):
                
                tuples = []
                for ki in range(k if t > 0 else 1):
                    if not EOS_found[ki][bi]:
                        for kj in range(k):
                            log_prob_sum = log_prob_sums[ki][bi] + torch.log(topk_list[ki].values[bi][kj])
                            # assert not torch.isnan(log_prob_sum), (log_prob_sums[ki][bi], topk_list[ki].values[bi][kj], torch.log(topk_list[ki].values[bi][kj]))
                            tuples.append((log_prob_sum, ki, kj))
                    else:
                        log_prob_sum = log_prob_sums[ki][bi].clone()
                        # assert not torch.isnan(log_prob_sum)
                        tuples.append((log_prob_sum, ki, 0))
                # assert len(tuples) == k * k if t > 0 else k
                tuples.sort(reverse=True)
                for i in range(k):
                    log_prob_sum, ki, kj = tuples[i]
                    # assert not torch.isnan(log_prob_sum)
                    log_prob_sums[i][bi] = log_prob_sum
                    output_list_aux[i][bi] = output_list[ki][bi]
                    output_list_aux[i][bi][t] = topk_list[ki].indices[bi][kj]
                    EOS_found_aux[i][bi] = EOS_found[ki][bi]
                    if output_list_aux[i][bi][t] == self.eos_idx:
                        EOS_found_aux[i][bi] = True
                    o_list_aux[i][bi] = o_list[ki][bi]
                    h_list_aux[i][bi] = h_list[ki][bi]
                    c_list_aux[i][bi] = c_list[ki][bi]
                    if multiple_layers:
                        for j in range(self.n_lstm_layers-1):
                            h_stacked_list_aux[i][j][bi] = h_stacked_list[ki][j][bi]
                            c_stacked_list_aux[i][j][bi] = c_stacked_list[ki][j][bi]
            
            output_list, output_list_aux = output_list_aux, output_list
            EOS_found, EOS_found_aux = EOS_found_aux, EOS_found
            o_list, o_list_aux = o_list_aux, o_list
            h_list, h_list_aux = h_list_aux, h_list
            c_list, c_list_aux = c_list_aux, c_list
            if multiple_layers:
                h_stacked_list, h_stacked_list_aux = h_stacked_list_aux, h_stacked_list
                c_stacked_list, c_stacked_list_aux = c_stacked_list_aux, c_stacked_list

            # print()    
            # print('****** t =', t)
            # print()
            for ki in range(k):
                # print('\nki =', ki)
                # print('topk_list[ki]:')
                # print(topk_list[ki])
                # print('output_list[ki]:')
                # print(output_list[ki])
                # print('EOS_found[ki]:')
                # print(EOS_found[ki])
                # print('log_prob_sums[ki]:')
                # print(log_prob_sums[ki])
                y_list[ki] = self.embedding_table(output_list[ki][:, t])
                # assert y_list[ki].shape == (batch_size, self.embed_size)

        output = output_list[0]
        # assert output.shape == (batch_size, max_answer_length)
        # return output_list, log_prob_sums, EOS_found
        return output

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
            assert answers is not None
            return self.teacher_forcing_decoding(
                image_local_features,
                image_global_features,
                question_vectors,
                answers,
            )
        else:
            assert max_answer_length is not None
            return self.greedy_search_decoding(
                image_local_features,
                image_global_features,
                question_vectors,
                max_answer_length,
            )