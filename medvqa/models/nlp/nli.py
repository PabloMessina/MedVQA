import torch
import torch.nn as nn
from transformers import AutoModel
from medvqa.models.common import freeze_parameters
from medvqa.models.huggingface_utils import SupportedHuggingfaceMedicalBERTModels

from medvqa.utils.logging import print_orange

class BertBasedNLI(nn.Module):

    def __init__(self,
                 huggingface_model_name,
                 merged_input=False,
                 hidden_size=None,
                 freeze_huggingface_model=False,
                 **unused_kwargs):
        super().__init__()
        print('BertBasedNLI')
        print(f'  huggingface_model_name: {huggingface_model_name}')
        print(f'  merged_input: {merged_input}')
        print(f'  hidden_size: {hidden_size}')
        print(f'  freeze_huggingface_model: {freeze_huggingface_model}')
        if len(unused_kwargs) > 0:
            print_orange(f'WARNING: unused_kwargs: {unused_kwargs}', bold=True)

        if not merged_input:
            assert hidden_size is not None, 'hidden_size must be provided if merged_input is False'

        self.huggingface_model_name = huggingface_model_name
        self.hidden_size = hidden_size
        self.merged_input = merged_input

        if huggingface_model_name in SupportedHuggingfaceMedicalBERTModels.get_all():
            self.model = AutoModel.from_pretrained(huggingface_model_name, trust_remote_code=True)
            if huggingface_model_name in SupportedHuggingfaceMedicalBERTModels.get_models_with_pooler_output():
                self._text_to_embedding_func = lambda x: self.model(**x).pooler_output
            elif huggingface_model_name in SupportedHuggingfaceMedicalBERTModels.get_all_cxr_bert_variants():
                self._text_to_embedding_func = lambda x: self.model.get_projected_text_embeddings(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
            else:
                raise ValueError(f'Unexpected huggingface_model_name: {huggingface_model_name}')
        else:
            raise ValueError(f'Unsupported huggingface_model_name: {huggingface_model_name}')
        
        if freeze_huggingface_model:
            print('Freezing huggingface model')
            freeze_parameters(self.model)

        self.embedding_size = SupportedHuggingfaceMedicalBERTModels.get_embedding_size(huggingface_model_name)
        if merged_input:
            self.nli_classifier = nn.Linear(self.embedding_size, 3)
        else:
            self.nli_hidden_layer = nn.Linear(self.embedding_size * 3, hidden_size) # 3 for p, h, p*h
            self.nli_classifier = nn.Linear(hidden_size, 3)

    def _forward_with_merged_input(self, tokenized_text):
        return self.nli_classifier(self._text_to_embedding_func(tokenized_text))
    
    def _forward_without_merged_input(self, tokenized_premises, tokenized_hypotheses):
        p_vectors = self._text_to_embedding_func(tokenized_premises)
        h_vectors = self._text_to_embedding_func(tokenized_hypotheses)
        element_wise_product = p_vectors * h_vectors
        # concatenate the vectors
        concat_vectors = torch.cat([p_vectors, h_vectors, element_wise_product], dim=-1)
        return self.nli_classifier(self.nli_hidden_layer(concat_vectors))
    
    def forward_with_precomputed_embeddings(self, p_vectors, h_vectors):
        assert not self.merged_input
        element_wise_product = p_vectors * h_vectors
        # concatenate the vectors
        concat_vectors = torch.cat([p_vectors, h_vectors, element_wise_product], dim=-1)
        return self.nli_classifier(self.nli_hidden_layer(concat_vectors))

    def forward(self, *args):
        if self.merged_input:
            return self._forward_with_merged_input(*args)
        else:
            return self._forward_without_merged_input(*args)
    
    def get_name(self):
        if self.merged_input:
            return f'BerBasedNLI({self.huggingface_model_name},{self.embedding_size},merged)'
        else:
            return f'BerBasedNLI({self.huggingface_model_name},{self.embedding_size},{self.hidden_size})'
