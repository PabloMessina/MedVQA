import torch
import torch.nn as nn

from medvqa.models.nlp.text_decoder import TransformerTextDecoder

class GenerationMode:
    GROUND_TRUTH_LABELS_2_REPORT = 'gt2report'
    ENSEMBLE_PREDICTIONS_2_REPORT = 'ensemble2report'
    HYBRID = 'hybrid'

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

        self.report_decoder = TransformerTextDecoder(
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
                                               texts=reports, mode=mode)
        else:
            pred_reports = self.report_decoder(input_memory=decoder_input_memory, device=device,
                                               max_text_length=max_report_length, mode=mode)
        output['pred_reports'] = pred_reports
        return output

    def get_name(self):
        return (f'Labels2ReportModel({self.transf_dec_num_memory_vecs}'
                f',{self.num_input_labels}->{self.labels_hidden_dim}->{self.num_output_labels}'
                f',{self.report_decoder.get_name()})')