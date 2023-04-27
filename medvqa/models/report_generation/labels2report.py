import torch
import torch.nn as nn

from medvqa.models.nlp.positional_encoding import PositionalEncoding

class GenerationMode:
    GROUND_TRUTH_LABELS_2_REPORT = 'gt2report'
    ENSEMBLE_PREDICTIONS_2_REPORT = 'ensemble2report'
    HYBRID = 'hybrid'

class TransformerReportDecoder(nn.Module):
  
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
    
    def teacher_forcing_decoding(self, input_memory, reports, device):
        # print('DEBUG: teacher_forcing_decoding')
        batch_size, max_report_length = reports.shape
        report_embeddings = self.pos_encoder(self.embedding_table(reports.permute(1,0)))
        assert report_embeddings.shape == (max_report_length, batch_size, self.embed_size)
        tgt_mask = self.generate_square_subsequent_mask(max_report_length).to(device)
        input_memory = input_memory.permute(1,0,2)
        decoded = self.decoder(report_embeddings, input_memory, tgt_mask=tgt_mask)
        vocab_logits = self.W_vocab(decoded)
        vocab_logits = vocab_logits.permute(1, 0, 2)
        assert vocab_logits.shape == (batch_size, max_report_length, self.vocab_size)
        return vocab_logits

    def greedy_search_decoding(self, input_memory, max_report_length, device):
        # print('DEBUG: greedy_search_decoding')
        batch_size = input_memory.size(0)
        input_memory = input_memory.permute(1,0,2)
        decoded_tokens = self.start_idx.expand(batch_size).unsqueeze(0)
        output = []

        # generation loop
        while len(output) < max_report_length:
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
        assert output.shape == (batch_size, max_report_length)
        return output

    def forward(self, input_memory, device, reports=None, max_report_length=None, mode='train'):
        if mode == 'train':
            assert reports is not None
            return self.teacher_forcing_decoding(input_memory, reports, device)
        else:
            assert max_report_length is not None
            return self.greedy_search_decoding(input_memory, max_report_length, device)
        
    def get_name(self):
        return (f'TransfReportDec(es={self.embed_size},hs={self.hidden_size},'
                f'nl={self.num_layers},nh={self.nhead},dff={self.dim_feedforward})')

class Labels2ReportModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_input_labels, labels_hidden_dim, num_output_labels,
                 transf_dec_num_memory_vecs, transf_dec_hidden_dim, transf_dec_nhead, transf_dec_dim_forward,
                 transf_dec_num_layers, start_idx, dropout_prob, **unused_kwargs):
        super().__init__()

        print('Labels2ReportModel')
        print(f'  vocab_size: {vocab_size}')
        print(f'  embedding_dim: {embedding_dim}')
        print(f'  num_input_labels: {num_input_labels}')
        print(f'  labels_hidden_dim: {labels_hidden_dim}')
        print(f'  num_output_labels: {num_output_labels}')
        print(f'  transf_dec_num_memory_vecs: {transf_dec_num_memory_vecs}')
        print(f'  transf_dec_hidden_dim: {transf_dec_hidden_dim}')
        print(f'  transf_dec_nhead: {transf_dec_nhead}')
        print(f'  transf_dec_dim_forward: {transf_dec_dim_forward}')
        print(f'  transf_dec_num_layers: {transf_dec_num_layers}')
        print(f'  start_idx: {start_idx}')
        print(f'  dropout_prob: {dropout_prob}')

        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.report_decoder = TransformerReportDecoder(
            embedding_table=self.embedding_table,
            embed_size=embedding_dim,
            hidden_size=transf_dec_hidden_dim,
            nhead=transf_dec_nhead,
            dim_feedforward=transf_dec_dim_forward,
            num_layers=transf_dec_num_layers,
            start_idx=start_idx,
            vocab_size=vocab_size,
            dropout_prob=dropout_prob,
        )

        self.binary_scores2memory = nn.Linear(num_output_labels, transf_dec_hidden_dim * transf_dec_num_memory_vecs)
        self.input2output_labels_1 = nn.Linear(num_input_labels, labels_hidden_dim)
        self.input2output_labels_2 = nn.Linear(labels_hidden_dim, num_output_labels)
        self.num_input_labels = num_input_labels
        self.labels_hidden_dim = labels_hidden_dim
        self.num_output_labels = num_output_labels
        self.transf_dec_num_memory_vecs = transf_dec_num_memory_vecs

    def forward(self, predicted_binary_scores, device, reports=None, max_report_length=None, mode='train', use_gt_as_pred=False):
        batch_size = predicted_binary_scores.size(0)
        output = {}
        if use_gt_as_pred: # use ground truth labels to generate report -> skip binary score prediction
            assert predicted_binary_scores.shape == (batch_size, self.num_output_labels)
            decoder_input_memory = self.binary_scores2memory(predicted_binary_scores).view(batch_size, -1, self.report_decoder.hidden_size)
        else: # use predicted labels to generate report
            assert predicted_binary_scores.shape == (batch_size, self.num_input_labels)
            output_logits = self.input2output_labels_2(torch.relu(self.input2output_labels_1(predicted_binary_scores)))
            output_probs = torch.sigmoid(output_logits)
            decoder_input_memory = self.binary_scores2memory(output_probs).view(batch_size, -1, self.report_decoder.hidden_size)
            output['pred_label_logits'] = output_logits
            output['pred_label_probs'] = output_probs
        assert decoder_input_memory.shape == (batch_size, self.transf_dec_num_memory_vecs, self.report_decoder.hidden_size)
        if mode == 'train':
            pred_reports = self.report_decoder(input_memory=decoder_input_memory, device=device,
                                               reports=reports, mode=mode)
        else:
            pred_reports = self.report_decoder(input_memory=decoder_input_memory, device=device,
                                               max_report_length=max_report_length, mode=mode)
        output['pred_reports'] = pred_reports
        return output

    def get_name(self):
        return (f'Labels2ReportModel({self.transf_dec_num_memory_vecs}'
                f',{self.num_input_labels}->{self.labels_hidden_dim}->{self.num_output_labels}'
                f',{self.report_decoder.get_name()})')