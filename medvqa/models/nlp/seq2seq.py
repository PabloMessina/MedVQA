import torch.nn as nn

from transformers import T5ForConditionalGeneration, BartForConditionalGeneration

class Seq2SeqModels:
    T5 = 't5'
    BART = 'bart'
    @staticmethod
    def get_all_models():
        return [
            Seq2SeqModels.T5,
            Seq2SeqModels.BART,
        ]

class Seq2SeqModel(nn.Module):

    def __init__(self, seq2seq_model_name, model_name=None, **unused_kwargs):
        super().__init__()
        print('Seq2Seq model:')
        print(f'  model_name: {model_name}')

        self.seq2seq_model_name = seq2seq_model_name
        self.model_name = model_name

        if seq2seq_model_name == Seq2SeqModels.T5:
            assert model_name is not None
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif seq2seq_model_name == Seq2SeqModels.BART:
            assert model_name is not None
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            raise ValueError(f'Unsupported Seq2Seq model: {seq2seq_model_name}')

    def forward(self, input_ids=None, attention_mask=None, labels=None, max_len=None, num_beams=1, mode='train'):

        if self.seq2seq_model_name == Seq2SeqModels.T5 or self.seq2seq_model_name == Seq2SeqModels.BART:
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
        if self.seq2seq_model_name == Seq2SeqModels.T5 or self.seq2seq_model_name == Seq2SeqModels.BART:
            assert self.model_name is not None
            strings.append(self.model_name)
        else:
            raise ValueError(f'Unknown seq2seq_model_name: {self.seq2seq_model_name}')
        return f'Seq2Seq({",".join(strings)})'
