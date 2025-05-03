import torch.nn as nn

from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoModelForSeq2SeqLM,
)

from medvqa.utils.logging_utils import print_orange

class Seq2SeqModels:
    T5 = 't5'
    FLAN_T5 = 'flan-t5'
    BART = 'bart'
    @staticmethod
    def get_all_models():
        return [
            Seq2SeqModels.T5,
            Seq2SeqModels.FLAN_T5,
            Seq2SeqModels.BART,
        ]

class Seq2SeqModel(nn.Module):

    def __init__(self, seq2seq_model_name, model_name=None, **unused_kwargs):
        super().__init__()
        print('Seq2Seq model:')
        print(f'  model_name: {model_name}')
        if unused_kwargs:
            print_orange(f'WARNING: unused kwargs: {unused_kwargs}', bold=True)

        self.seq2seq_model_name = seq2seq_model_name
        self.model_name = model_name
        self.use_t5 = seq2seq_model_name == Seq2SeqModels.T5
        self.use_flan_t5 = seq2seq_model_name == Seq2SeqModels.FLAN_T5
        self.use_bart = seq2seq_model_name == Seq2SeqModels.BART

        if self.use_t5:
            assert model_name is not None
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif self.use_flan_t5:
            assert model_name is not None
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif self.use_bart:
            assert model_name is not None
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            raise ValueError(f'Unsupported Seq2Seq model: {seq2seq_model_name}')

    def forward(self, input_ids=None, attention_mask=None, labels=None, max_len=None, num_beams=1, mode='train'):

        if self.use_t5 or self.use_flan_t5 or self.use_bart:
            assert input_ids is not None
            assert attention_mask is not None
            if mode == 'train' or mode == 'val':
                assert labels is not None
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            elif mode == 'test':
                assert labels is not None or max_len is not None
                if labels is not None:
                    max_len = labels.size(1)
                output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                max_new_tokens=max_len, num_beams=num_beams)
            else:
                raise ValueError(f'Unknown mode: {mode}')
        
        else: assert False

        return output
    
    def get_name(self):
        strings = []
        if self.use_t5 or self.use_flan_t5 or self.use_bart:
            assert self.model_name is not None
            strings.append(self.model_name)
        else:
            raise ValueError(f'Unsupported Seq2Seq model: {self.seq2seq_model_name}')
        return f'Seq2Seq({",".join(strings)})'
