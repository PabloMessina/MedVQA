import torch
import torch.nn as nn
from transformers import AutoModel
from medvqa.models.common import freeze_parameters

from medvqa.utils.logging import print_orange

class HuggingfaceModels:
    MICROSOFT_BIOMEDVLP_CXR_BERT_SPECIALIZED = 'microsoft/BiomedVLP-CXR-BERT-specialized'
    MICROSOFT_BIOMEDVLP_BIOVIL_T = 'microsoft/BiomedVLP-BioViL-T'
    @staticmethod
    def get_all():
        return [
            HuggingfaceModels.MICROSOFT_BIOMEDVLP_CXR_BERT_SPECIALIZED,
            HuggingfaceModels.MICROSOFT_BIOMEDVLP_BIOVIL_T,
        ]

class FactEncoder(nn.Module):

    def __init__(self,
                 huggingface_model_name,
                 embedding_size,
                 freeze_huggingface_model=False,
                 # Auxiliary tasks
                 classify_category=False,
                 n_categories=None,
                 classify_health_status=False,
                 n_health_statuses=None,
                 classify_comparison_status=False,
                 n_comparison_statuses=None,
                 classify_chest_imagenome_obs=False,
                 n_chest_imagenome_observations=None,
                 classify_chest_imagenome_anatloc=False,
                 n_chest_imagenome_anatomical_locations=None,
                 use_aux_task_hidden_layer=False,
                 aux_task_hidden_layer_size=None,
                 do_nli=False,
                 nli_hidden_layer_size=None,
                 **unused_kwargs):
        super().__init__()
        print('Fact encoder')
        print(f'  huggingface_model_name: {huggingface_model_name}')
        print(f'  embedding_size: {embedding_size}')
        print(f'  classify_category: {classify_category}')
        print(f'  n_categories: {n_categories}')
        print(f'  classify_health_status: {classify_health_status}')
        print(f'  n_health_statuses: {n_health_statuses}')
        print(f'  classify_comparison_status: {classify_comparison_status}')
        print(f'  n_comparison_statuses: {n_comparison_statuses}')
        print(f'  classify_chest_imagenome_obs: {classify_chest_imagenome_obs}')
        print(f'  n_chest_imagenome_observations: {n_chest_imagenome_observations}')
        print(f'  classify_chest_imagenome_anatloc: {classify_chest_imagenome_anatloc}')
        print(f'  n_chest_imagenome_anatomical_locations: {n_chest_imagenome_anatomical_locations}')
        print(f'  use_aux_task_hidden_layer: {use_aux_task_hidden_layer}')
        print(f'  aux_task_hidden_layer_size: {aux_task_hidden_layer_size}')
        print(f'  do_nli: {do_nli}')
        print(f'  nli_hidden_layer_size: {nli_hidden_layer_size}')

        if len(unused_kwargs) > 0:
            print_orange(f'WARNING: unused_kwargs: {unused_kwargs}', bold=True)

        self.huggingface_model_name = huggingface_model_name

        if huggingface_model_name in [
            HuggingfaceModels.MICROSOFT_BIOMEDVLP_CXR_BERT_SPECIALIZED,
            HuggingfaceModels.MICROSOFT_BIOMEDVLP_BIOVIL_T,
        ]:
            self.model = AutoModel.from_pretrained(huggingface_model_name, trust_remote_code=True)
        else:
            raise ValueError(f'Unsupported huggingface_model_name: {huggingface_model_name}')
        
        if freeze_huggingface_model:
            print('Freezing huggingface model')
            freeze_parameters(self.model)
        
        self.embedding_size = embedding_size
        self.classify_category = classify_category
        self.n_categories = n_categories
        self.classify_health_status = classify_health_status
        self.n_health_statuses = n_health_statuses
        self.classify_comparison_status = classify_comparison_status
        self.n_comparison_statuses = n_comparison_statuses
        self.classify_chest_imagenome_obs = classify_chest_imagenome_obs
        self.n_chest_imagenome_observations = n_chest_imagenome_observations
        self.classify_chest_imagenome_anatloc = classify_chest_imagenome_anatloc
        self.n_chest_imagenome_anatomical_locations = n_chest_imagenome_anatomical_locations
        self.use_aux_task_hidden_layer = use_aux_task_hidden_layer
        self.do_nli = do_nli
        self.nli_hidden_layer_size = nli_hidden_layer_size
        
        # Auxiliary tasks
        if use_aux_task_hidden_layer:
            assert aux_task_hidden_layer_size is not None
            self.aux_task_hidden_layer = nn.Linear(self.embedding_size, aux_task_hidden_layer_size)
            aux_task_input_size = aux_task_hidden_layer_size
        else:
            aux_task_input_size = self.embedding_size
        if self.classify_category:
            self.category_classifier = nn.Linear(aux_task_input_size, self.n_categories)
        if self.classify_health_status:
            self.health_status_classifier = nn.Linear(aux_task_input_size, self.n_health_statuses)
        if self.classify_comparison_status:
            self.comparison_status_classifier = nn.Linear(aux_task_input_size, self.n_comparison_statuses)
        if classify_chest_imagenome_obs:
            self.chest_imagenome_obs_classifier = nn.Linear(aux_task_input_size, self.n_chest_imagenome_observations)
        if classify_chest_imagenome_anatloc:
            self.chest_imagenome_anatloc_classifier = nn.Linear(aux_task_input_size, self.n_chest_imagenome_anatomical_locations)
        if do_nli:
            assert nli_hidden_layer_size is not None
            self.nli_hidden_layer = nn.Linear(self.embedding_size * 3, nli_hidden_layer_size)
            self.nli_classifier = nn.Linear(nli_hidden_layer_size, 3)

    def nli_forward(self, p_input_ids, p_attention_mask, h_input_ids, h_attention_mask):
        p_embeddings = self.model.get_projected_text_embeddings(input_ids=p_input_ids, attention_mask=p_attention_mask)
        h_embeddings = self.model.get_projected_text_embeddings(input_ids=h_input_ids, attention_mask=h_attention_mask)
        elementwise_product = p_embeddings * h_embeddings
        nli_input = torch.cat([p_embeddings, h_embeddings, elementwise_product], dim=-1)
        nli_input = torch.relu(self.nli_hidden_layer(nli_input))
        nli_logits = self.nli_classifier(nli_input)
        return nli_logits

    def forward(self, input_ids, attention_mask, run_metadata_auxiliary_tasks=False, run_chest_imagenome_obs_task=False,
                run_chest_imagenome_anatloc_task=False):
        text_embeddings = self.model.get_projected_text_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        output = { 'text_embeddings': text_embeddings }
        if run_metadata_auxiliary_tasks or run_chest_imagenome_obs_task or run_chest_imagenome_anatloc_task:
            # Run auxiliary tasks
            if self.use_aux_task_hidden_layer:
                aux_task_input = self.aux_task_hidden_layer(text_embeddings)
                aux_task_input = torch.relu(aux_task_input)
            else:
                aux_task_input = text_embeddings
            if run_metadata_auxiliary_tasks:
                if self.classify_category:
                    output['category_logits'] = self.category_classifier(aux_task_input)
                if self.classify_health_status:
                    output['health_status_logits'] = self.health_status_classifier(aux_task_input)
                if self.classify_comparison_status:
                    output['comparison_status_logits'] = self.comparison_status_classifier(aux_task_input)
            if run_chest_imagenome_obs_task:
                output['chest_imagenome_obs_logits'] = self.chest_imagenome_obs_classifier(aux_task_input)
            if run_chest_imagenome_anatloc_task:
                output['chest_imagenome_anatloc_logits'] = self.chest_imagenome_anatloc_classifier(aux_task_input)
        return output
    
    def get_name(self):
        return f'FactEncoder({self.huggingface_model_name})'
