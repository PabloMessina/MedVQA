import torch.nn as nn
from transformers import AutoModel

class HuggingfaceModels:
    MICROSOFT_BIOMEDVLP_CXR_BERT_SPECIALIZED = 'microsoft/BiomedVLP-CXR-BERT-specialized'
    @staticmethod
    def get_all():
        return [HuggingfaceModels.MICROSOFT_BIOMEDVLP_CXR_BERT_SPECIALIZED]

class FactEncoder(nn.Module):

    def __init__(self,
                 huggingface_model_name,
                 embedding_size,
                 # Auxiliary tasks
                 classify_category=False,
                 n_categories=None,
                 classify_health_status=False,
                 n_health_statuses=None,
                 classify_comparison_status=False,
                 n_comparison_statuses=None,
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

        self.huggingface_model_name = huggingface_model_name

        if huggingface_model_name == HuggingfaceModels.MICROSOFT_BIOMEDVLP_CXR_BERT_SPECIALIZED:
            self.model = AutoModel.from_pretrained(huggingface_model_name, trust_remote_code=True)
        else:
            raise ValueError(f'Unsupported huggingface_model_name: {huggingface_model_name}')
        
        self.embedding_size = embedding_size
        self.classify_category = classify_category
        self.n_categories = n_categories
        self.classify_health_status = classify_health_status
        self.n_health_statuses = n_health_statuses
        self.classify_comparison_status = classify_comparison_status
        self.n_comparison_statuses = n_comparison_statuses
        
        # Auxiliary tasks
        if self.classify_category:
            self.category_classifier = nn.Linear(self.embedding_size, self.n_categories)
        if self.classify_health_status:
            self.health_status_classifier = nn.Linear(self.embedding_size, self.n_health_statuses)
        if self.classify_comparison_status:
            self.comparison_status_classifier = nn.Linear(self.embedding_size, self.n_comparison_statuses)

    def forward(self, input_ids, attention_mask, run_auxiliary_tasks=False):
        text_embeddings = self.model.get_projected_text_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        output = { 'text_embeddings': text_embeddings }
        if run_auxiliary_tasks:
            if self.classify_category:
                output['category_logits'] = self.category_classifier(text_embeddings)
            if self.classify_health_status:
                output['health_status_logits'] = self.health_status_classifier(text_embeddings)
            if self.classify_comparison_status:
                output['comparison_status_logits'] = self.comparison_status_classifier(text_embeddings)
        return output
    
    def get_name(self):
        return f'FactEncoder({self.huggingface_model_name})'
