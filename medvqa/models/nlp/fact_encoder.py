import torch
import torch.nn as nn
from transformers import AutoModel
from medvqa.models.common import freeze_parameters
from medvqa.models.nlp.text_decoder import TransformerTextDecoder

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

def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])
    
def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask

def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


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
                 use_spert=False,
                 spert_size_embedding=None,
                 spert_relation_types=None,
                 spert_entity_types=None,
                 spert_max_pairs=None,
                 spert_prop_drop=None,
                 spert_cls_token=None,
                 use_fact_decoder=False,
                 fact_decoder_embed_size=None,
                 fact_decoder_hidden_size=None,
                 fact_decoder_nhead=None,
                 fact_decoder_dim_feedforward=None,
                 fact_decoder_num_layers=None,
                 fact_decoder_start_idx=None,
                 fact_decoder_vocab_size=None,
                 fact_decoder_dropout_prob=0,
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
        print(f'  use_spert: {use_spert}')
        print(f'  spert_size_embedding: {spert_size_embedding}')
        print(f'  spert_relation_types: {spert_relation_types}')
        print(f'  spert_entity_types: {spert_entity_types}')
        print(f'  spert_max_pairs: {spert_max_pairs}')
        print(f'  spert_prop_drop: {spert_prop_drop}')
        print(f'  spert_cls_token: {spert_cls_token}')
        print(f'  use_fact_decoder: {use_fact_decoder}')
        print(f'  fact_decoder_embed_size: {fact_decoder_embed_size}')
        print(f'  fact_decoder_hidden_size: {fact_decoder_hidden_size}')
        print(f'  fact_decoder_nhead: {fact_decoder_nhead}')
        print(f'  fact_decoder_dim_feedforward: {fact_decoder_dim_feedforward}')
        print(f'  fact_decoder_num_layers: {fact_decoder_num_layers}')
        print(f'  fact_decoder_start_idx: {fact_decoder_start_idx}')
        print(f'  fact_decoder_vocab_size: {fact_decoder_vocab_size}')
        print(f'  fact_decoder_dropout_prob: {fact_decoder_dropout_prob}')

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
        self.use_fact_decoder = use_fact_decoder
        self.fact_decoder_embed_size = fact_decoder_embed_size
        
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
        
        if use_spert:
            hidden_size = self.model.config.hidden_size
            print(f'self.model.config.hidden_size: {hidden_size}')
            self.spert_rel_classifier = nn.Linear(hidden_size * 3 + spert_size_embedding * 2, spert_relation_types)
            self.spert_entity_classifier = nn.Linear(hidden_size * 2 + spert_size_embedding, spert_entity_types)
            self.spert_size_embeddings = nn.Embedding(100, spert_size_embedding)
            self.spert_dropout = nn.Dropout(spert_prop_drop)

            self._spert_cls_token = spert_cls_token
            self._spert_relation_types = spert_relation_types
            self._spert_entity_types = spert_entity_types
            self._spert_max_pairs = spert_max_pairs

        if use_fact_decoder:
            assert fact_decoder_embed_size is not None
            assert fact_decoder_hidden_size is not None
            assert fact_decoder_nhead is not None
            assert fact_decoder_dim_feedforward is not None
            assert fact_decoder_num_layers is not None
            assert fact_decoder_start_idx is not None
            assert fact_decoder_vocab_size is not None
            assert fact_decoder_dropout_prob is not None
            self.fact_decoder_embedding_table = nn.Embedding(
                num_embeddings=fact_decoder_vocab_size,
                embedding_dim=fact_decoder_embed_size,
                padding_idx=0,
            )
            self.fact_decoder = TransformerTextDecoder(
                embedding_table=self.fact_decoder_embedding_table,
                embed_size=fact_decoder_embed_size,
                hidden_size=fact_decoder_hidden_size,
                nhead=fact_decoder_nhead,
                dim_feedforward=fact_decoder_dim_feedforward,
                num_layers=fact_decoder_num_layers,
                start_idx=fact_decoder_start_idx,
                vocab_size=fact_decoder_vocab_size,
                dropout_prob=fact_decoder_dropout_prob,
                apply_pos_encoding_to_input=False,
                input_pos_encoding_mode="sinusoidal",
            )
            self.fact_decoder_input_layer = nn.Linear(self.embedding_size, fact_decoder_embed_size)

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
    
    def spert_forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.model(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.spert_size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._spert_classify_entities(encodings, h, entity_masks, size_embeddings)

        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._spert_max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._spert_relation_types]).to(
            self.spert_rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._spert_max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._spert_classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._spert_max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf
    
    def spert_forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.model(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entities
        size_embeddings = self.spert_size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._spert_classify_entities(encodings, h, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._spert_filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._spert_max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._spert_relation_types]).to(
            self.spert_rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._spert_max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._spert_classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._spert_max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations
    
    def _spert_filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.spert_rel_classifier.weight.device
        batch_relations = padded_stack(batch_relations).to(device)
        batch_rel_masks = padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks
    
    def _spert_classify_entities(self, encodings, h, entity_masks, size_embeddings):
        # max pool entity candidate spans
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._spert_cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.spert_dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.spert_entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool
    
    def _spert_classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._spert_max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._spert_max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._spert_max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.spert_dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.spert_rel_classifier(rel_repr)
        return chunk_rel_logits
    
    def fact_decoder_forward_teacher_forcing(self, input_ids, attention_mask, decoder_input_ids):
        text_embeddings = self.model.get_projected_text_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self.fact_decoder_input_layer(text_embeddings)
        text_embeddings = text_embeddings.unsqueeze(1) # [batch_size, 1, fact_decoder_embed_size]
        assert text_embeddings.shape == (input_ids.shape[0], 1, self.fact_decoder_embed_size)
        decoder_logits = self.fact_decoder.teacher_forcing_decoding(
            input_memory=text_embeddings,
            texts=decoder_input_ids,
            device=input_ids.device,
        )
        return decoder_logits
    
    def fact_decoder_forward_greedy_decoding(self, input_ids, attention_mask, max_length=100):
        text_embeddings = self.model.get_projected_text_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self.fact_decoder_input_layer(text_embeddings)
        text_embeddings = text_embeddings.unsqueeze(1)
        assert text_embeddings.shape == (input_ids.shape[0], 1, self.fact_decoder_embed_size)
        decoded_ids = self.fact_decoder.greedy_decoding(
            input_memory=text_embeddings,
            max_length=max_length,
            device=input_ids.device,
        )
        return decoded_ids
    
    def get_name(self):
        return f'FactEncoder({self.huggingface_model_name})'
