import torch
from torch import nn
from medvqa.models.nlp.positional_encoding import compute_2D_positional_encoding, PositionalEncoding
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule
from medvqa.utils.logging import print_orange

class PhraseGroundingMode:
    SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER = 'sigmoid_attention_plus_custom_classifier'
    TRANSFORMER_ENCODER = 'transformer_encoder'
    
    @staticmethod
    def get_all_modes():
        return [
            PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER,
            PhraseGroundingMode.TRANSFORMER_ENCODER,
        ]
    
_mode2shortname = {
    PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER: 'SigmoidAttention',
    PhraseGroundingMode.TRANSFORMER_ENCODER: 'TransformerEncoder',
}

class PhraseGrounder(MultiPurposeVisualModule):

    def __init__(
            self,
            # Image Encoder
            raw_image_encoding,
            freeze_image_encoder,
            image_local_feat_size,
            image_encoder_pretrained_weights_path,
            num_regions,
            yolov8_model_name_or_path,
            yolov8_model_alias,
            yolov8_use_one_detector_per_dataset,
            # Auxiliary tasks
            predict_bboxes_chest_imagenome,
            predict_bboxes_vinbig,
            # Other
            apply_positional_encoding,
            phrase_embedding_size,
            regions_width,
            regions_height,
            qkv_size, # query, key, value size
            phrase_classifier_hidden_size,
            phrase_grounding_mode=PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER,
            # Transformer Encoder
            transf_d_model=None,
            transf_nhead=None,
            transf_dim_feedforward=None,
            transf_dropout=None,
            transf_num_layers=None,
            **unused_kwargs,
            ):

        if len(unused_kwargs) > 0:
            print_orange(f'WARNING: Unused kwargs: {unused_kwargs}')
        
        assert regions_width * regions_height == num_regions, f'width * height ({regions_width * regions_height}) != num_regions ({num_regions})'

        # Init MultiPurposeVisualModule components (for image encoder and auxiliary tasks)
        super().__init__(
            # Image Encoder kwargs
            raw_image_encoding=raw_image_encoding,
            freeze_image_encoder=freeze_image_encoder,
            image_local_feat_size=image_local_feat_size,
            image_encoder_pretrained_weights_path=image_encoder_pretrained_weights_path,
            num_regions=num_regions,
            yolov8_model_name_or_path=yolov8_model_name_or_path,
            yolov8_model_alias=yolov8_model_alias,
            # Auxiliary tasks kwargs
            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
            predict_bboxes_vinbig=predict_bboxes_vinbig,
            yolov8_use_one_detector_per_dataset=yolov8_use_one_detector_per_dataset,
        )

        self.regions_width = regions_width
        self.regions_height = regions_height
        self.phrase_grounding_mode = phrase_grounding_mode

        if self.phrase_grounding_mode == PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER:
            assert qkv_size is not None
            assert phrase_classifier_hidden_size is not None

            # Init PhraseGrounder components
            self.phrase_embedding_size = phrase_embedding_size
            self.qkv_size = qkv_size
            self.q_proj = nn.Linear(phrase_embedding_size, qkv_size)
            self.k_proj = nn.Linear(image_local_feat_size, qkv_size)
            self.v_proj = nn.Linear(image_local_feat_size, qkv_size)
            self.W = nn.Linear(qkv_size, phrase_embedding_size)
            self.att_proj = nn.Linear(qkv_size, 1)

            # Init phrase classifier (for auxiliary task: true or false)
            self.phrase_classifier_1 = nn.Linear(phrase_embedding_size * 3 + num_regions ,
                                                phrase_classifier_hidden_size) # (phrase, grounding, element-wise mult, attention map) -> hidden size
            self.phrase_classifier_2 = nn.Linear(phrase_classifier_hidden_size, 1) # hidden size -> true or false (binary classification)

            # Init positional encoding
            self.apply_positional_encoding = apply_positional_encoding
            if self.apply_positional_encoding:
                pe = compute_2D_positional_encoding(h=regions_height, w=regions_width, d=image_local_feat_size)
                assert pe.shape == (regions_height, regions_width, image_local_feat_size)
                pe = pe.reshape(1, regions_height * regions_width, image_local_feat_size) # (1, num_regions, image_local_feat_size)
                self.register_buffer('pe', pe)

        elif self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER:
            assert transf_d_model is not None
            assert transf_nhead is not None
            assert transf_dim_feedforward is not None
            assert transf_dropout is not None
            assert transf_num_layers is not None

            self.transf_d_model = transf_d_model
            self.transf_nhead = transf_nhead
            self.transf_dim_feedforward = transf_dim_feedforward
            self.transf_dropout = transf_dropout
            self.transf_num_layers = transf_num_layers

            # Init PhraseGrounder components
            self.phrase_embedding_size = phrase_embedding_size
            self.image_proj = nn.Linear(image_local_feat_size, transf_d_model)
            self.phrase_proj = nn.Linear(phrase_embedding_size, transf_d_model)
            self.att_proj = nn.Linear(transf_d_model, 1)

            # Init transformer encoder
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=transf_d_model,
                    nhead=transf_nhead,
                    dim_feedforward=transf_dim_feedforward,
                    dropout=transf_dropout,
                    batch_first=True, # (batch_size, seq_len, d_model)
                ),
                num_layers=transf_num_layers,
            )

            # Init phrase classifier (for auxiliary task: true or false)
            self.phrase_classifier = nn.Linear(transf_d_model, 1) # hidden size -> true or false (binary classification)

            # Init positional encoding
            self.apply_positional_encoding = apply_positional_encoding
            if self.apply_positional_encoding:
                self.pe = PositionalEncoding(d_model=transf_d_model, dropout=transf_dropout)

        else:
            raise ValueError(f'Unknown phrase_grounding_mode: {phrase_grounding_mode}')

    def get_name(self):
        if self.phrase_grounding_mode == PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER:
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]},'
                    f'{self.phrase_embedding_size},{self.qkv_size},{self.phrase_classifier_1.out_features},'
                    f'{self.phrase_classifier_2.out_features})')
        elif self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER:
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]},'
                    f'{self.phrase_embedding_size},{self.transf_d_model},{self.transf_nhead},'
                    f'{self.transf_dim_feedforward},{self.transf_num_layers})')
        else:
            raise ValueError(f'Unknown phrase_grounding_mode: {self.phrase_grounding_mode}')

    def forward(
        self,
        raw_images, # (batch_size, 3, H, W)
        phrase_embeddings, # (batch_size, K, phrase_embedding_size)
        only_compute_features=False,
        skip_phrase_classifier=False,
        yolov8_detection_layer_index=None,
        mimiccxr_forward=False,
        vinbig_forward=False,
        return_normalized_average_v=False,
    ):  
        assert mimiccxr_forward or vinbig_forward or only_compute_features
        # Visual Component
        output = super().forward(
            raw_images=raw_images,
            return_local_features=True,
            only_compute_features=only_compute_features,
            mimiccxr_forward=mimiccxr_forward,
            vinbig_forward=vinbig_forward,
            yolov8_detection_layer_index=yolov8_detection_layer_index,
        )
        local_feat = output['local_feat'] # (batch_size, num_regions, image_local_feat_size)

        if self.phrase_grounding_mode == PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER:
            if self.apply_positional_encoding:
                local_feat = local_feat + self.pe # apply positional encoding
            assert local_feat.shape == (raw_images.shape[0], self.num_regions, self.image_local_feat_size)
            # Queries
            q = self.q_proj(phrase_embeddings) # (batch_size, K, qkv_size)
            # Keys and Values
            k = self.k_proj(local_feat) # (batch_size, num_regions, qkv_size)
            v = self.v_proj(local_feat) # (batch_size, num_regions, qkv_size)
            # Attention
            # attention = torch.matmul(q, k.transpose(1,2)) # (batch_size, K, num_regions)
            attention = self.att_proj(q.unsqueeze(2) * k.unsqueeze(1)).squeeze(3) # (batch_size, K, num_regions)
            sigmoid_attention = torch.sigmoid(attention) # (batch_size, K, num_regions)
            # Weighted average
            weighted_sum = torch.matmul(sigmoid_attention, v) # (batch_size, K, qkv_size)
            weighted_avg = weighted_sum / (sigmoid_attention.sum(dim=-1, keepdim=True) + 1e-8) # (batch_size, K, qkv_size)
            # Grounding vector
            grounding_vector = self.W(weighted_avg) # (batch_size, K, phrase_embedding_size)
            grounding_vector = torch.nn.functional.normalize(grounding_vector, p=2, dim=-1) # (batch_size, K, phrase_embedding_size)
            # Phrase classifier
            element_wise_mult = phrase_embeddings * grounding_vector # (batch_size, K, phrase_embedding_size)
            if not skip_phrase_classifier:
                phrase_classifier_input = torch.cat([phrase_embeddings, grounding_vector, element_wise_mult, sigmoid_attention], dim=-1)
                phrase_classifier_logits = self.phrase_classifier_1(phrase_classifier_input)
                phrase_classifier_logits = torch.relu(phrase_classifier_logits)
                phrase_classifier_logits = self.phrase_classifier_2(phrase_classifier_logits)
            # Phrase-grounding similarity
            phrase_grounding_similarity = element_wise_mult.sum(dim=-1)
            # Output
            output['sigmoid_attention'] = sigmoid_attention
            output['phrase_grounding_similarity'] = phrase_grounding_similarity
            if not skip_phrase_classifier:
                output['phrase_classifier_logits'] = phrase_classifier_logits.squeeze(-1) # (batch_size, K, 1) -> (batch_size, K)
            if return_normalized_average_v:
                average_v = v.mean(dim=1) # (batch_size, qkv_size)
                average_v = self.W(average_v) # (batch_size, phrase_embedding_size)
                normalized_average_v = torch.nn.functional.normalize(average_v, p=2, dim=-1) # (batch_size, phrase_embedding_size)
                output['normalized_average_v'] = normalized_average_v
        
        elif self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER:

            batch_size, K, _ = phrase_embeddings.shape

            X_phrase = self.phrase_proj(phrase_embeddings) # (batch_size, K, transf_d_model)
            X_image = self.image_proj(local_feat) # (batch_size, num_regions, transf_d_model)
            X_image = X_image.unsqueeze(1).expand(-1, X_phrase.shape[1], -1, -1) # (batch_size, K, num_regions, transf_d_model)
            X_phrase = X_phrase.unsqueeze(2) # (batch_size, K, 1, transf_d_model)
            X = torch.cat([X_phrase, X_image], dim=2) # (batch_size, K, num_regions + 1, transf_d_model)
            X = X.view(-1, self.num_regions + 1, self.transf_d_model) # (batch_size * K, num_regions + 1, transf_d_model)

            if self.apply_positional_encoding:
                X = self.pe(X)

            X = self.transformer_encoder(X) # (batch_size * K, num_regions + 1, transf_d_model)

            # Phrase grounding attention
            image_representation = X[:, 1:, :] # (batch_size * K, num_regions, transf_d_model)
            attention_logits = self.att_proj(image_representation) # (batch_size * K, num_regions, 1)
            sigmoid_attention = torch.sigmoid(attention_logits) # (batch_size * K, num_regions, 1)
            sigmoid_attention = sigmoid_attention.view(batch_size, K, self.num_regions) # (batch_size, K, num_regions)
            
            # Phrase classifier
            # (use the first token as the phrase representation)
            if not skip_phrase_classifier:
                phrase_representation = X[:, 0, :] # (batch_size * K, transf_d_model)
                phrase_classifier_logits = self.phrase_classifier(phrase_representation) # (batch_size * K, 1)
                phrase_classifier_logits = phrase_classifier_logits.view(batch_size, K) # (batch_size, K)

            # Output
            output['sigmoid_attention'] = sigmoid_attention
            if not skip_phrase_classifier:
                output['phrase_classifier_logits'] = phrase_classifier_logits

            
        return output