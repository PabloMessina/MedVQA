import torch
import torch.nn as nn

from transformers import T5ForConditionalGeneration

from medvqa.models.nlp.text_decoder import TransformerTextDecoder
from medvqa.utils.logging import print_orange

class GenerationMode:
    PREDICTIONS_2_REFINED_PREDICTIONS_2_REPORT = 'pred2refpred2rep'
    PREDICTIONS_2_REPORT = 'pred2rep'
    VERBOSE_PREDICTIONS_2_REPORT = 'verbose_pred2rep'
    @staticmethod
    def get_all_modes():
        return [
            GenerationMode.PREDICTIONS_2_REFINED_PREDICTIONS_2_REPORT,
            GenerationMode.PREDICTIONS_2_REPORT,
            GenerationMode.VERBOSE_PREDICTIONS_2_REPORT,
        ]

class NLG_Models:
    PYTORCH_TRANSFORMER = 'pytorch_transformer'
    T5 = 't5'
    @staticmethod
    def get_all_models():
        return [
            NLG_Models.PYTORCH_TRANSFORMER,
            NLG_Models.T5,
        ]

class Labels2ReportModel(nn.Module):

    def __init__(self, gen_mode, nlg_model, vocab_size, embedding_dim, num_input_labels, labels_hidden_dim,
                 transf_dec_num_memory_vecs, transf_dec_hidden_dim, transf_dec_nhead, transf_dec_dim_forward,
                 transf_dec_num_layers, start_idx, dropout_prob, num_output_labels=None,
                 support_two_label_sources=False, num_input_labels_2=None, labels_hidden_dim_2=None,
                 num_output_labels_2=None, t5_model_name=None, **unused_kwargs):
        super().__init__()

        print('Labels2ReportModel')
        print(f'  gen_mode: {gen_mode}')
        print(f'  nlg_model: {nlg_model}')
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
        print(f'  support_two_label_sources: {support_two_label_sources}')
        print(f'  num_input_labels_2: {num_input_labels_2}')
        print(f'  labels_hidden_dim_2: {labels_hidden_dim_2}')
        print(f'  num_output_labels_2: {num_output_labels_2}')
        print(f'  t5_model_name: {t5_model_name}')

        if len(unused_kwargs) > 0:
            print_orange(f'WARNING: Unused kwargs: {unused_kwargs}')

        # Sanity checks
        if nlg_model == NLG_Models.T5:
            assert gen_mode == GenerationMode.VERBOSE_PREDICTIONS_2_REPORT, \
                f'For T5, gen_mode must be {GenerationMode.VERBOSE_PREDICTIONS_2_REPORT}'
        elif nlg_model == NLG_Models.PYTORCH_TRANSFORMER:
            assert gen_mode in [
                GenerationMode.PREDICTIONS_2_REFINED_PREDICTIONS_2_REPORT,
                GenerationMode.PREDICTIONS_2_REPORT,
            ]

        self.gen_mode = gen_mode
        self.nlg_model = nlg_model
        self.num_input_labels = num_input_labels
        self.labels_hidden_dim = labels_hidden_dim
        self.num_output_labels = num_output_labels
        self.transf_dec_num_memory_vecs = transf_dec_num_memory_vecs
        self.support_two_label_sources = support_two_label_sources
        self.num_input_labels_2 = num_input_labels_2
        self.labels_hidden_dim_2 = labels_hidden_dim_2
        self.num_output_labels_2 = num_output_labels_2
        self.t5_model_name = t5_model_name

        if nlg_model == NLG_Models.PYTORCH_TRANSFORMER:

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
            
            if gen_mode == GenerationMode.PREDICTIONS_2_REFINED_PREDICTIONS_2_REPORT:
                assert num_output_labels is not None
                self.input2output_labels_1 = nn.Linear(num_input_labels, labels_hidden_dim)
                self.input2output_labels_2 = nn.Linear(labels_hidden_dim, num_output_labels)
                self.binary_scores2memory = nn.Linear(num_output_labels, transf_dec_hidden_dim * transf_dec_num_memory_vecs)
                if support_two_label_sources:
                    assert num_input_labels_2 is not None
                    assert labels_hidden_dim_2 is not None
                    assert num_output_labels_2 is not None
                    self.input2output_labels_1_2 = nn.Linear(num_input_labels_2, labels_hidden_dim_2)
                    self.input2output_labels_2_2 = nn.Linear(labels_hidden_dim_2, num_output_labels_2)
                    self.binary_scores2memory_2 = nn.Linear(num_output_labels_2, transf_dec_hidden_dim * transf_dec_num_memory_vecs)
            elif gen_mode == GenerationMode.PREDICTIONS_2_REPORT:
                self.binary_scores2memory = nn.Linear(num_input_labels, transf_dec_hidden_dim * transf_dec_num_memory_vecs)
                if support_two_label_sources:
                    assert num_input_labels_2 is not None
                    self.binary_scores2memory_2 = nn.Linear(num_input_labels_2, transf_dec_hidden_dim * transf_dec_num_memory_vecs)
            else:
                raise ValueError(f'Unsupported generation mode: {gen_mode}')
        elif nlg_model == NLG_Models.T5:
            assert t5_model_name is not None
            self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        else:
            raise ValueError(f'Unsupported NLG network: {nlg_model}')

    def forward(self, predicted_binary_scores=None, device=None, reports=None, max_report_length=None, mode='train',
                is_second_label_source=False, input_ids=None, attention_mask=None, labels=None, num_beams=1):

        if self.nlg_model == NLG_Models.PYTORCH_TRANSFORMER:

            assert predicted_binary_scores is not None
            assert device is not None

            batch_size = predicted_binary_scores.size(0)
            assert predicted_binary_scores.shape == (batch_size, self.num_input_labels)
            output = {}
            if self.gen_mode == GenerationMode.PREDICTIONS_2_REFINED_PREDICTIONS_2_REPORT:
                if is_second_label_source:
                    assert self.support_two_label_sources
                    assert self.input2output_labels_1_2 is not None
                    assert self.input2output_labels_2_2 is not None
                    assert self.binary_scores2memory_2 is not None
                    output_logits = self.input2output_labels_2_2(torch.relu(self.input2output_labels_1_2(predicted_binary_scores)))
                    output_probs = torch.sigmoid(output_logits)
                    decoder_input_memory = self.binary_scores2memory_2(output_probs).view(batch_size, -1, self.report_decoder.hidden_size)
                else:
                    output_logits = self.input2output_labels_2(torch.relu(self.input2output_labels_1(predicted_binary_scores)))
                    output_probs = torch.sigmoid(output_logits)
                    decoder_input_memory = self.binary_scores2memory(output_probs).view(batch_size, -1, self.report_decoder.hidden_size)
                output['pred_label_logits'] = output_logits
                output['pred_label_probs'] = output_probs
            elif self.gen_mode == GenerationMode.PREDICTIONS_2_REPORT:
                if is_second_label_source:
                    assert self.support_two_label_sources
                    assert self.binary_scores2memory_2 is not None
                    decoder_input_memory = self.binary_scores2memory_2(predicted_binary_scores).view(batch_size, -1, self.report_decoder.hidden_size)
                else:
                    decoder_input_memory = self.binary_scores2memory(predicted_binary_scores).view(batch_size, -1, self.report_decoder.hidden_size)
            else:
                raise ValueError(f'Unknown generation mode: {self.gen_mode}')
            assert decoder_input_memory.shape == (batch_size, self.transf_dec_num_memory_vecs, self.report_decoder.hidden_size)
            if mode == 'train':
                assert reports is not None
                pred_reports = self.report_decoder(input_memory=decoder_input_memory, device=device,
                                                texts=reports, mode=mode)
            else:
                assert max_report_length is not None
                pred_reports = self.report_decoder(input_memory=decoder_input_memory, device=device,
                                                max_text_length=max_report_length, mode=mode)
            output['pred_reports'] = pred_reports
            return output

        elif self.nlg_model == NLG_Models.T5:
            
            assert input_ids is not None
            assert attention_mask is not None
            if mode == 'train' or mode == 'val':
                assert labels is not None
                output = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            elif mode == 'test':
                assert labels is not None or max_report_length is not None
                if labels is not None:
                    max_report_length = labels.size(1)
                output = self.t5_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                max_new_tokens=max_report_length, num_beams=num_beams)
            else:
                raise ValueError(f'Unknown mode: {mode}')

            return output
        
        else: assert False

    def get_name(self):
        strings = []
        if self.gen_mode == GenerationMode.PREDICTIONS_2_REFINED_PREDICTIONS_2_REPORT:
            strings.append(f'{self.num_input_labels}->{self.labels_hidden_dim}->{self.num_output_labels}')
            if self.support_two_label_sources:
                strings.append(f'(2nd:{self.num_input_labels_2}->{self.labels_hidden_dim_2}->{self.num_output_labels_2})')
            strings.append(f'{self.transf_dec_num_memory_vecs}')
            strings.append(f'{self.report_decoder.get_name()})')
        elif self.gen_mode == GenerationMode.PREDICTIONS_2_REPORT:
            strings.append(f'{self.num_input_labels}')
            if self.support_two_label_sources:
                strings.append(f'(2nd:{self.num_input_labels_2})')
            strings.append(f'{self.transf_dec_num_memory_vecs}')
            strings.append(f'{self.report_decoder.get_name()})')
        elif self.gen_mode == GenerationMode.VERBOSE_PREDICTIONS_2_REPORT:
            assert self.t5_model_name is not None
            strings.append(self.t5_model_name)
        else:
            raise ValueError(f'Unknown generation mode: {self.gen_mode}')
        return f'Labels2ReportModel({",".join(strings)})'