import torch
from torch import nn
from medvqa.models.nlp.positional_encoding import compute_2D_positional_encoding
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule
from medvqa.utils.logging import print_orange

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

    def get_name(self):
        return f'PhraseGrounder({super().get_name()},{self.phrase_embedding_size},{self.qkv_size})'

    def forward(
        self,
        raw_images, # (batch_size, 3, H, W)
        only_compute_features=False,
        pos_phrase_embeddings=None, # (batch_size, K_pos, phrase_embedding_size)
        neg_phrase_embeddings=None, # (batch_size, K_neg, phrase_embedding_size)
        phrase_embeddings=None, # (batch_size, K, phrase_embedding_size)
        skip_phrase_classifier=False,
        yolov8_detection_layer_index=None,
        mimiccxr_forward=False,
        vinbig_forward=False,
    ):  
        assert (pos_phrase_embeddings is not None and neg_phrase_embeddings is not None) or \
            phrase_embeddings is not None, 'Either (pos_phrase_embeddings and neg_phrase_embeddings) or phrase_embeddings must be provided'
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
        if self.apply_positional_encoding:
            local_feat = local_feat + self.pe # apply positional encoding
        assert local_feat.shape == (raw_images.shape[0], self.num_regions, self.image_local_feat_size)
        
        if pos_phrase_embeddings is not None and neg_phrase_embeddings is not None:
            # Queries
            q_pos = self.q_proj(pos_phrase_embeddings) # (batch_size, K_pos, qkv_size)
            q_neg = self.q_proj(neg_phrase_embeddings) # (batch_size, K_neg, qkv_size)
            # Keys and Values
            k = self.k_proj(local_feat) # (batch_size, num_regions, qkv_size)
            v = self.v_proj(local_feat) # (batch_size, num_regions, qkv_size)
            # Attention
            # attention_pos = torch.matmul(q_pos, k.transpose(1,2)) # (batch_size, K_pos, num_regions)
            attention_pos = self.att_proj(q_pos.unsqueeze(2) * k.unsqueeze(1)).squeeze(3) # (batch_size, K_pos, num_regions)
            sigmoid_attention_pos = torch.sigmoid(attention_pos) # (batch_size, K_pos, num_regions)
            # attention_neg = torch.matmul(q_neg, k.transpose(1,2)) # (batch_size, K_neg, num_regions)
            attention_neg = self.att_proj(q_neg.unsqueeze(2) * k.unsqueeze(1)).squeeze(3) # (batch_size, K_neg, num_regions)
            sigmoid_attention_neg = torch.sigmoid(attention_neg) # (batch_size, K_neg, num_regions)
            # Weighted average
            weighted_sum_pos = torch.matmul(sigmoid_attention_pos, v) # (batch_size, K_pos, qkv_size)
            weighted_avg_pos = weighted_sum_pos / (sigmoid_attention_pos.sum(dim=-1, keepdim=True) + 1e-8) # (batch_size, K_pos, qkv_size)
            weighted_sum_neg = torch.matmul(sigmoid_attention_neg, v) # (batch_size, K_neg, qkv_size)
            weighted_avg_neg = weighted_sum_neg / (sigmoid_attention_neg.sum(dim=-1, keepdim=True) + 1e-8) # (batch_size, K_neg, qkv_size)
            # Grounding vector
            grounding_vector_pos = self.W(weighted_avg_pos) # (batch_size, K_pos, phrase_embedding_size)
            grounding_vector_pos = torch.nn.functional.normalize(grounding_vector_pos, p=2, dim=-1) # (batch_size, K_pos, phrase_embedding_size)
            grounding_vector_neg = self.W(weighted_avg_neg) # (batch_size, K_neg, phrase_embedding_size)
            grounding_vector_neg = torch.nn.functional.normalize(grounding_vector_neg, p=2, dim=-1) # (batch_size, K_neg, phrase_embedding_size)
            # Phrase classifier
            element_wise_mult_pos = pos_phrase_embeddings * grounding_vector_pos # (batch_size, K_pos, phrase_embedding_size)
            phrase_classifier_input_pos = torch.cat([pos_phrase_embeddings, grounding_vector_pos,
                                                     element_wise_mult_pos, sigmoid_attention_pos], dim=-1)
            phrase_classifier_output_pos = self.phrase_classifier_1(phrase_classifier_input_pos)
            phrase_classifier_output_pos = torch.relu(phrase_classifier_output_pos)
            phrase_classifier_output_pos = self.phrase_classifier_2(phrase_classifier_output_pos)

            element_wise_mult_neg = neg_phrase_embeddings * grounding_vector_neg
            phrase_classifier_input_neg = torch.cat([neg_phrase_embeddings, grounding_vector_neg,
                                                     element_wise_mult_neg, sigmoid_attention_neg], dim=-1)
            phrase_classifier_output_neg = self.phrase_classifier_1(phrase_classifier_input_neg)
            phrase_classifier_output_neg = torch.relu(phrase_classifier_output_neg)
            phrase_classifier_output_neg = self.phrase_classifier_2(phrase_classifier_output_neg)
            # Phrase-grounding similarity
            phrase_grounding_similarity_pos = element_wise_mult_pos.sum(dim=-1) # (batch_size, K_pos)
            phrase_grounding_similarity_neg = element_wise_mult_neg.sum(dim=-1) # (batch_size, K_neg)
            # Output
            output['sigmoid_attention_pos'] = sigmoid_attention_pos
            output['sigmoid_attention_neg'] = sigmoid_attention_neg
            output['phrase_classifier_output_pos'] = phrase_classifier_output_pos
            output['phrase_classifier_output_neg'] = phrase_classifier_output_neg
            output['phrase_grounding_similarity_pos'] = phrase_grounding_similarity_pos
            output['phrase_grounding_similarity_neg'] = phrase_grounding_similarity_neg
        elif phrase_embeddings is not None:
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
            if not skip_phrase_classifier:
                element_wise_mult = phrase_embeddings * grounding_vector # (batch_size, K, phrase_embedding_size)
                phrase_classifier_input = torch.cat([phrase_embeddings, grounding_vector, element_wise_mult, sigmoid_attention], dim=-1)
                phrase_classifier_output = self.phrase_classifier_1(phrase_classifier_input)
                phrase_classifier_output = torch.relu(phrase_classifier_output)
                phrase_classifier_output = self.phrase_classifier_2(phrase_classifier_output)
            # Output
            output['sigmoid_attention'] = sigmoid_attention
            if not skip_phrase_classifier:
                output['phrase_classifier_output'] = phrase_classifier_output
        return output