import logging
import torch
import torch.nn.functional as F
from torch import nn
from medvqa.models.FiLM_utils import LinearFiLM
from medvqa.models.mlp import MLP
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, Summer
from medvqa.models.vision.bbox_regression import MultiClassBoundingBoxRegressor
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule, RawImageEncoding
from medvqa.utils.common import ChoiceEnum
from medvqa.utils.logging_utils import ANSI_MAGENTA_BOLD, ANSI_RESET

logger = logging.getLogger(__name__)

class PhraseGroundingMode(ChoiceEnum):
    SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER = 'sigmoid_attention_plus_custom_classifier'
    TRANSFORMER_ENCODER_WITH_SEGMENTATION = 'transformer_encoder_with_segmentation'
    TRANSFORMER_ENCODER_WITH_BBOX_REGRESSION = 'transformer_encoder_with_bbox_regression'
    TRANSFORMER_ENCODER__NO_GROUNDING = 'transformer_encoder__no_grounding'
    FILM_LAYERS_PLUS_SIGMOID_ATTENTION_AND_CUSTOM_CLASSIFIER = 'film_layers_plus_sigmoid_attention_and_custom_classifier'
    FILM_LAYERS_SIGMOID_ATTENTION_CUSTOM_CLASSIFIER_OBJECT_DETECTION = 'film_layers_sigmoid_attention_custom_classifier_object_detection'
    GLOBAL_POOLING_CONCAT_MLP__NO_GROUNDING = 'global_pooling_concat_mlp__no_grounding'
    GLOBAL_POOLING_FILM_MLP__NO_GROUNDING = 'global_pooling_film_mlp__no_grounding'
    ADAPTIVE_FILM_BASED_POOLING_MLP__NO_GROUNDING = 'adaptive_film_based_pooling_mlp__no_grounding'
    ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION = 'adaptive_film_based_pooling_mlp_with_bbox_regression'
    ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_YOLOV11 = 'adaptive_film_based_pooling_mlp_with_yolov11'
    
_mode2shortname = {
    PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER.value: 'SigmoidAttention',
    PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_SEGMENTATION.value: 'TransformerEncoder_Segmentation',
    PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_BBOX_REGRESSION.value: 'TransformerEncoder_BBoxRegression',
    PhraseGroundingMode.TRANSFORMER_ENCODER__NO_GROUNDING.value: 'TransformerEncoder_NoGrounding',
    PhraseGroundingMode.FILM_LAYERS_PLUS_SIGMOID_ATTENTION_AND_CUSTOM_CLASSIFIER.value: 'FiLM_SigmoidAttention',
    PhraseGroundingMode.FILM_LAYERS_SIGMOID_ATTENTION_CUSTOM_CLASSIFIER_OBJECT_DETECTION.value: 'FiLM_SigmoidAttention_ObjectDetection',
    PhraseGroundingMode.GLOBAL_POOLING_CONCAT_MLP__NO_GROUNDING.value: 'GlobalPoolingConcatMLP',
    PhraseGroundingMode.GLOBAL_POOLING_FILM_MLP__NO_GROUNDING.value: 'GlobalPoolingFiLMMLP',
    PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP__NO_GROUNDING.value: 'AdaptiveFiLM_MLP',
    PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value: 'AdaptiveFiLM_MLP_BBoxRegression',
    PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_YOLOV11.value: 'AdaptiveFiLM_MLP_YOLOv11',
}

class PhraseGrounder(MultiPurposeVisualModule):

    def __init__(
            self,
            # Image Encoder
            raw_image_encoding,
            freeze_image_encoder,
            image_local_feat_size,
            image_encoder_pretrained_weights_path,
            image_size,
            num_regions,
            yolov8_model_name_or_path,
            yolov8_model_alias,
            yolov8_use_one_detector_per_dataset,
            yolov11_model_name_or_path,
            yolov11_model_alias,
            # Auxiliary tasks
            predict_bboxes_chest_imagenome,
            predict_bboxes_vinbig,
            predict_global_alignment, # Align the global features with the phrase embeddings
            # Other
            apply_positional_encoding,
            phrase_embedding_size,
            regions_width,
            regions_height,
            qkv_size, # query, key, value size
            phrase_classifier_hidden_size,
            phrase_grounding_mode=PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER.value,
            image_encoder_dropout_p=0,
            huggingface_model_name=None,
            alignment_proj_size=None,
            device=None,
            predict_relative_bbox_coords=False,
            bbox_format='xyxy',
            # FiLM-based approach's hyperparameters
            visual_feature_proj_size=None,
            visual_grounding_hidden_size=None,
            phrase_mlp_hidden_dims=None,
            # Transformer Encoder
            transf_d_model=None,
            transf_nhead=None,
            transf_dim_feedforward=None,
            transf_dropout=None,
            transf_num_layers=None,
            **unused_kwargs,
            ):

        if len(unused_kwargs) > 0:
            logger.warning(f'Unused kwargs: {unused_kwargs}')
        
        assert regions_width * regions_height == num_regions, f'width * height ({regions_width * regions_height}) != num_regions ({num_regions})'

        if phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_YOLOV11.value:
            assert raw_image_encoding == RawImageEncoding.YOLOV11_FACT_CONDITIONED

        # Init MultiPurposeVisualModule components (for image encoder and auxiliary tasks)
        super().__init__(
            # Image Encoder kwargs
            raw_image_encoding=raw_image_encoding,
            huggingface_model_name=huggingface_model_name,
            freeze_image_encoder=freeze_image_encoder,
            image_local_feat_size=image_local_feat_size,
            image_encoder_pretrained_weights_path=image_encoder_pretrained_weights_path,
            image_encoder_dropout_p=image_encoder_dropout_p,
            image_size=image_size,
            num_regions=num_regions,
            yolov8_model_name_or_path=yolov8_model_name_or_path,
            yolov8_model_alias=yolov8_model_alias,
            yolov11_model_name_or_path=yolov11_model_name_or_path,
            yolov11_model_alias=yolov11_model_alias,
            query_embed_size=phrase_embedding_size,
            local_attention_hidden_size=visual_grounding_hidden_size,
            classification_mlp_hidden_dims=phrase_mlp_hidden_dims,
            device=device,
            # Auxiliary tasks kwargs
            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
            predict_bboxes_vinbig=predict_bboxes_vinbig,
            yolov8_use_one_detector_per_dataset=yolov8_use_one_detector_per_dataset,
        )

        self.regions_width = regions_width
        self.regions_height = regions_height
        self.phrase_embedding_size = phrase_embedding_size
        self.phrase_grounding_mode = phrase_grounding_mode
        self.use_global_features = False # false by default, but can be set to true in the FiLM-based approach
        self.predict_global_alignment = predict_global_alignment
        self.alignment_proj_size = alignment_proj_size
        self.visual_feature_proj_size = visual_feature_proj_size

        if predict_global_alignment:
            assert self.phrase_grounding_mode == PhraseGroundingMode.FILM_LAYERS_PLUS_SIGMOID_ATTENTION_AND_CUSTOM_CLASSIFIER.value, \
                'predict_global_alignment is only supported in the FiLM-based approach'

        if self.phrase_grounding_mode == PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER.value:
            assert qkv_size is not None
            assert phrase_classifier_hidden_size is not None

            # Init PhraseGrounder components
            self.qkv_size = qkv_size
            self.q_proj = nn.Linear(phrase_embedding_size, qkv_size)
            self.k_proj = nn.Linear(image_local_feat_size, qkv_size)
            self.v_proj = nn.Linear(image_local_feat_size, qkv_size)
            self.W = nn.Linear(qkv_size, phrase_embedding_size)
            self.att_proj = nn.Linear(qkv_size, 1)

            # Init phrase classifier (for auxiliary task: true or false)
            self.phrase_classifier_1 = nn.Linear(phrase_embedding_size * 3 + num_regions,
                                                phrase_classifier_hidden_size) # (phrase, grounding, element-wise mult, attention map) -> hidden size
            self.phrase_classifier_2 = nn.Linear(phrase_classifier_hidden_size, 1) # hidden size -> true or false (binary classification)

            # Init positional encoding
            self.apply_positional_encoding = apply_positional_encoding
            if self.apply_positional_encoding:
                self.pos_encoding = Summer(PositionalEncoding2D(image_local_feat_size))

        elif (self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_SEGMENTATION.value or
              self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_BBOX_REGRESSION.value or
              self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER__NO_GROUNDING.value):
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

            # Init input projection layers
            self.image_proj = nn.Linear(image_local_feat_size, transf_d_model)
            self.phrase_proj = nn.Linear(phrase_embedding_size, transf_d_model)            

            # Init transformer encoder
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=transf_d_model,
                    nhead=transf_nhead,
                    dim_feedforward=transf_dim_feedforward,
                    dropout=transf_dropout,
                    batch_first=True, # (batch_size, seq_len, d_model)
                    activation=F.gelu,
                ),
                num_layers=transf_num_layers,
            )

            # Init phrase classifier (for auxiliary task: true or false)
            self.phrase_classifier = nn.Linear(transf_d_model, 1) # hidden size -> true or false (binary classification)

            # Init positional encoding
            self.pos_encoding = Summer(PositionalEncoding1D(image_local_feat_size))

            if self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_SEGMENTATION.value:
                self.att_proj = nn.Linear(transf_d_model, 1) # hidden size -> scalar (segmentation score)
            elif self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_BBOX_REGRESSION.value:
                self.visual_grounding_bbox_regressor = MultiClassBoundingBoxRegressor(local_feat_dim=transf_d_model) # hidden size -> bbox

        elif self.phrase_grounding_mode == PhraseGroundingMode.FILM_LAYERS_PLUS_SIGMOID_ATTENTION_AND_CUSTOM_CLASSIFIER.value:
            assert visual_feature_proj_size is not None
            assert visual_grounding_hidden_size is not None
            assert phrase_mlp_hidden_dims is not None

            self.global_proj = nn.Linear(self.global_feat_size, visual_feature_proj_size)
            self.local_proj = nn.Linear(image_local_feat_size, visual_feature_proj_size)
            self.global_film = LinearFiLM(visual_feature_proj_size, phrase_embedding_size)
            self.local_film = LinearFiLM(visual_feature_proj_size, phrase_embedding_size)
            self.visual_grounding_layer_1 = nn.Linear(2 * visual_feature_proj_size, visual_grounding_hidden_size)
            self.visual_grounding_layer_2 = nn.Linear(visual_grounding_hidden_size, 1) # hidden size -> scalar (visual grounding score)
            self.classifier_mlp = MLP(in_dim=phrase_embedding_size + visual_feature_proj_size * 2 + num_regions, # phrase, global, local, attention
                                      out_dim=1, # true or false (binary classification)
                                      hidden_dims=phrase_mlp_hidden_dims)
            self.use_global_features = True
            
            # Init positional encoding
            self.apply_positional_encoding = apply_positional_encoding
            if self.apply_positional_encoding:
                self.pos_encoding = Summer(PositionalEncoding2D(image_local_feat_size))
                logger.info(f'{ANSI_MAGENTA_BOLD}Using positional encoding for local features{ANSI_RESET}')

            # Init alignment projection layers
            if predict_global_alignment:
                assert alignment_proj_size is not None
                self.global_vf_align_proj = nn.Linear(self.global_feat_size, alignment_proj_size)
                self.phrase_emb_align_proj = nn.Linear(phrase_embedding_size, alignment_proj_size)

        elif self.phrase_grounding_mode == PhraseGroundingMode.FILM_LAYERS_SIGMOID_ATTENTION_CUSTOM_CLASSIFIER_OBJECT_DETECTION.value:
            assert visual_feature_proj_size is not None
            assert visual_grounding_hidden_size is not None
            assert phrase_mlp_hidden_dims is not None
            
            self.global_film = LinearFiLM(self.global_feat_size, phrase_embedding_size)
            self.local_film_1 = LinearFiLM(self.local_feat_size, phrase_embedding_size)
            self.local_film_2 = LinearFiLM(self.local_feat_size, phrase_embedding_size)
            self.visual_grounding_hidden_layer = nn.Linear(self.local_feat_size, visual_grounding_hidden_size)
            self.visual_grounding_bbox_regressor = MultiClassBoundingBoxRegressor(local_feat_dim=visual_grounding_hidden_size)
            self.classifier_mlp = MLP(in_dim=self.global_feat_size + self.local_feat_size + num_regions, # global, local, attention
                                      out_dim=1, # true or false (binary classification)
                                      activation=nn.GELU,
                                      hidden_dims=phrase_mlp_hidden_dims)
            self.use_global_features = True
            
            # Init positional encoding
            self.apply_positional_encoding = apply_positional_encoding
            if self.apply_positional_encoding:
                self.pos_encoding = Summer(PositionalEncoding2D(image_local_feat_size))
                logger.info(f'{ANSI_MAGENTA_BOLD}Using positional encoding for local features{ANSI_RESET}')

            # Init alignment projection layers
            if predict_global_alignment:
                assert alignment_proj_size is not None
                self.global_vf_align_proj = nn.Linear(self.global_feat_size, alignment_proj_size)
                self.phrase_emb_align_proj = nn.Linear(phrase_embedding_size, alignment_proj_size)

        elif self.phrase_grounding_mode == PhraseGroundingMode.GLOBAL_POOLING_CONCAT_MLP__NO_GROUNDING.value:
            assert phrase_mlp_hidden_dims is not None
            self.use_global_features = True
            self.classifier_mlp = MLP(in_dim=self.global_feat_size + self.phrase_embedding_size, # global, phrase
                                        out_dim=1, # true or false (binary classification)
                                        hidden_dims=phrase_mlp_hidden_dims,
                                        activation=nn.GELU)
            
        elif self.phrase_grounding_mode == PhraseGroundingMode.GLOBAL_POOLING_FILM_MLP__NO_GROUNDING.value:
            assert phrase_mlp_hidden_dims is not None
            self.use_global_features = True
            self.global_film = LinearFiLM(self.global_feat_size, phrase_embedding_size)
            self.classifier_mlp = MLP(in_dim=self.global_feat_size, # global
                                        out_dim=1, # true or false (binary classification)
                                        hidden_dims=phrase_mlp_hidden_dims,
                                        activation=nn.GELU)
            
        elif self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP__NO_GROUNDING.value or \
                self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value:
            
            self.local_film = LinearFiLM(self.local_feat_size, phrase_embedding_size)
            self.local_attention_hidden_layer = nn.Linear(self.local_feat_size, visual_grounding_hidden_size)
            self.classifier_mlp = MLP(in_dim=self.local_feat_size + num_regions, # local, attention
                                      out_dim=1, # true or false (binary classification)
                                      activation=nn.GELU,
                                      hidden_dims=phrase_mlp_hidden_dims)
            
            # Init positional encoding
            self.apply_positional_encoding = apply_positional_encoding
            if self.apply_positional_encoding:
                self.pos_encoding = Summer(PositionalEncoding2D(image_local_feat_size))
                logger.info(f'{ANSI_MAGENTA_BOLD}Using positional encoding for local features{ANSI_RESET}')

            if phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value:
                self.visual_grounding_bbox_regressor = MultiClassBoundingBoxRegressor(
                    local_feat_dim=visual_grounding_hidden_size, bbox_format=bbox_format,
                    predict_relative=predict_relative_bbox_coords)
            else:
                self.local_attention_final_layer = nn.Linear(visual_grounding_hidden_size, 1) # hidden size -> scalar (local attention score)

        elif self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_YOLOV11.value:
            pass

        else:
            raise ValueError(f'Unknown phrase_grounding_mode: {phrase_grounding_mode}')

    def get_name(self):
        if self.phrase_grounding_mode == PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER.value:
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]},'
                    f'{self.phrase_embedding_size},{self.qkv_size},{self.phrase_classifier_1.out_features},'
                    f'{self.phrase_classifier_2.out_features})')
        elif (self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_SEGMENTATION.value or
                self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_BBOX_REGRESSION.value or
                self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER__NO_GROUNDING.value):
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]},'
                    f'{self.phrase_embedding_size},{self.transf_d_model},{self.transf_nhead},'
                    f'{self.transf_dim_feedforward},{self.transf_num_layers})')
        elif self.phrase_grounding_mode == PhraseGroundingMode.FILM_LAYERS_PLUS_SIGMOID_ATTENTION_AND_CUSTOM_CLASSIFIER.value:
            mlp_hidden_dims_str = '-'.join(map(str, self.classifier_mlp.hidden_dims))
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]},'
                    f'{self.phrase_embedding_size},{self.global_proj.out_features},{self.local_proj.out_features},'
                    f'{self.visual_grounding_layer_1.out_features},{self.visual_grounding_layer_2.out_features},'
                    f'{mlp_hidden_dims_str})')
        elif self.phrase_grounding_mode == PhraseGroundingMode.FILM_LAYERS_SIGMOID_ATTENTION_CUSTOM_CLASSIFIER_OBJECT_DETECTION.value:
            mlp_hidden_dims_str = '-'.join(map(str, self.classifier_mlp.hidden_dims))
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]},'
                    f'{self.phrase_embedding_size},{self.visual_grounding_hidden_layer.out_features},'
                    f'{self.visual_grounding_bbox_regressor.local_feat_dim},{mlp_hidden_dims_str})')
        elif self.phrase_grounding_mode == PhraseGroundingMode.GLOBAL_POOLING_CONCAT_MLP__NO_GROUNDING.value:
            mlp_hidden_dims_str = '-'.join(map(str, self.classifier_mlp.hidden_dims))
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]},'
                    f'{self.phrase_embedding_size},{mlp_hidden_dims_str})')
        elif self.phrase_grounding_mode == PhraseGroundingMode.GLOBAL_POOLING_FILM_MLP__NO_GROUNDING.value:
            mlp_hidden_dims_str = '-'.join(map(str, self.classifier_mlp.hidden_dims))
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]},'
                    f'{self.phrase_embedding_size},{mlp_hidden_dims_str})')
        elif (self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP__NO_GROUNDING.value or
              self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value):
            mlp_hidden_dims_str = '-'.join(map(str, self.classifier_mlp.hidden_dims))
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]},'
                    f'{self.phrase_embedding_size},{self.local_attention_hidden_layer.out_features},'
                    f'{mlp_hidden_dims_str})')
        elif self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_YOLOV11.value:
            return (f'PhraseGrounder({super().get_name()},{_mode2shortname[self.phrase_grounding_mode]})')
        else:
            raise ValueError(f'Unknown phrase_grounding_mode: {self.phrase_grounding_mode}')

    def forward(
        self,
        raw_images, # (batch_size, 3, H, W)
        phrase_embeddings, # (batch_size, K, phrase_embedding_size)
        only_compute_features=False,
        skip_phrase_classifier=False,
        compute_global_alignment=False,
        yolov8_detection_layer_index=None,
        mimiccxr_forward=False,
        vinbig_forward=False,
        return_normalized_average_v=False,
        predict_bboxes=False,
        apply_nms=False,
        iou_threshold=0.1,
        conf_threshold=0.1,
        max_det_per_class=20,
        max_det=10, # for YOLOv11
        batch=None, # for YOLOv11
        use_first_n_facts_for_detection=None, # for YOLOv11
        return_sigmoid_attention=False,
    ):  
        # Visual Component
        output = super().forward(
            raw_images=raw_images,
            return_local_features=True,
            return_global_features=self.use_global_features,
            only_compute_features=only_compute_features,
            mimiccxr_forward=mimiccxr_forward,
            vinbig_forward=vinbig_forward,
            yolov8_detection_layer_index=yolov8_detection_layer_index,
            # For YOLOv11 fact-conditioned
            apply_nms=apply_nms,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            max_det=max_det,
            fact_embeddings=phrase_embeddings,
            batch=batch,
            use_first_n_facts_for_detection=use_first_n_facts_for_detection,
        )

        if self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_YOLOV11.value:
            return output # YOLOv11 fact-conditioned

        local_feat = output['local_feat'] # (batch_size, num_regions, image_local_feat_size)

        if self.phrase_grounding_mode == PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER.value:
            if self.apply_positional_encoding:
                local_feat = local_feat.view(-1, self.num_regions_sqrt, self.num_regions_sqrt, self.image_local_feat_size)
                local_feat = self.pos_encoding(local_feat) # apply positional encoding
                local_feat = local_feat.view(-1, self.num_regions, self.image_local_feat_size)

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
        
        elif (self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_SEGMENTATION.value or
                self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_BBOX_REGRESSION.value or
                self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER__NO_GROUNDING.value):

            batch_size, K, _ = phrase_embeddings.shape

            X_phrase = self.phrase_proj(phrase_embeddings) # (batch_size, K, transf_d_model)
            X_phrase = X_phrase.unsqueeze(2) # (batch_size, K, 1, transf_d_model)
            
            X_image = self.image_proj(local_feat) # (batch_size, num_regions, transf_d_model)
            X_image = X_image.unsqueeze(1).expand(-1, K, -1, -1) # (batch_size, K, num_regions, transf_d_model)
            
            X = torch.cat([X_phrase, X_image], dim=2) # (batch_size, K, num_regions + 1, transf_d_model)
            X = X.view(-1, self.num_regions + 1, self.transf_d_model) # (batch_size * K, num_regions + 1, transf_d_model)
            X = self.pos_encoding(X) # apply positional encoding
            X = self.transformer_encoder(X) # (batch_size * K, num_regions + 1, transf_d_model)

            if self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_SEGMENTATION.value:
                # Phrase grounding attention
                image_representation = X[:, 1:, :] # (batch_size * K, num_regions, transf_d_model)
                attention_logits = self.att_proj(image_representation) # (batch_size * K, num_regions, 1)
                sigmoid_attention = torch.sigmoid(attention_logits) # (batch_size * K, num_regions, 1)
                sigmoid_attention = sigmoid_attention.view(batch_size, K, self.num_regions) # (batch_size, K, num_regions)
            elif self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_BBOX_REGRESSION.value:
                # Phrase grounding bbox regression
                image_representation = X[:, 1:, :] # (batch_size * K, num_regions, transf_d_model)
                image_representation = image_representation.view(batch_size, K, self.num_regions, self.transf_d_model) # (batch_size, K, num_regions, transf_d_model)
                if apply_nms:
                    predicted_bboxes, visual_grounding_binary_logits = self.visual_grounding_bbox_regressor(
                        image_representation, apply_nms=apply_nms, iou_threshold=iou_threshold, conf_threshold=conf_threshold)
                else:
                    visual_grounding_bbox_logits, visual_grounding_binary_logits = self.visual_grounding_bbox_regressor(image_representation)
                sigmoid_attention = torch.sigmoid(visual_grounding_binary_logits) # (batch_size, K, num_regions, 1)
                sigmoid_attention = sigmoid_attention.squeeze(-1) # (batch_size, K, num_regions)
            
            # Phrase classifier
            # (use the first token as the phrase representation)
            if not skip_phrase_classifier:
                phrase_representation = X[:, 0, :] # (batch_size * K, transf_d_model)
                phrase_classifier_logits = self.phrase_classifier(phrase_representation) # (batch_size * K, 1)
                phrase_classifier_logits = phrase_classifier_logits.view(batch_size, K) # (batch_size, K)

            # Output
            if self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_SEGMENTATION.value:
                output['sigmoid_attention'] = sigmoid_attention # (batch_size, K, num_regions)
            elif self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER_WITH_BBOX_REGRESSION.value:
                output['sigmoid_attention'] = sigmoid_attention # (batch_size, K, num_regions)
                if apply_nms:
                    output['predicted_bboxes'] = predicted_bboxes # (batch_size, K, list of bboxes)
                else:
                    output['visual_grounding_binary_logits'] = visual_grounding_binary_logits # (batch_size, K, num_regions, 1)
                    output['visual_grounding_bbox_logits'] = visual_grounding_bbox_logits # (batch_size, K, num_regions, 4)
            if not skip_phrase_classifier:
                output['phrase_classifier_logits'] = phrase_classifier_logits

        elif self.phrase_grounding_mode == PhraseGroundingMode.FILM_LAYERS_PLUS_SIGMOID_ATTENTION_AND_CUSTOM_CLASSIFIER.value:

            orig_global_feat = output['global_feat'] # (batch_size, global_feat_size)
            global_feat = self.global_proj(orig_global_feat) # (batch_size, visual_feature_proj_size)
            
            if self.apply_positional_encoding:
                local_feat = local_feat.view(-1, self.num_regions_sqrt, self.num_regions_sqrt, self.image_local_feat_size)
                local_feat = self.pos_encoding(local_feat) # apply positional encoding
                local_feat = local_feat.view(-1, self.num_regions, self.image_local_feat_size)
            local_feat = self.local_proj(local_feat) # (batch_size, num_regions, visual_feature_proj_size)

            # FiLM layers
            num_facts = phrase_embeddings.shape[1] # K
            global_feat = global_feat.unsqueeze(1).expand(-1, num_facts, -1) # (batch_size, K, visual_feature_proj_size)
            global_feat = self.global_film(global_feat, phrase_embeddings) # (batch_size, K, visual_feature_proj_size)

            local_feat = local_feat.unsqueeze(1).expand(-1, num_facts, -1, -1) # (batch_size, K, num_regions, visual_feature_proj_size)
            phrase_embeddings_ = phrase_embeddings.unsqueeze(2).expand(-1, -1, self.num_regions, -1) # (batch_size, K, num_regions, phrase_embedding_size)
            local_feat = self.local_film(local_feat, phrase_embeddings_) # (batch_size, K, num_regions, visual_feature_proj_size)

            # Visual grounding
            global_feat_ = global_feat.unsqueeze(2).expand(-1, -1, self.num_regions, -1) # (batch_size, K, num_regions, visual_feature_proj_size)
            visual_grounding_input = torch.cat([global_feat_, local_feat], dim=-1) # (batch_size, K, num_regions, 2 * visual_feature_proj_size)
            visual_grounding_logits = self.visual_grounding_layer_1(visual_grounding_input) # (batch_size, K, num_regions, visual_grounding_hidden_size)
            visual_grounding_logits = torch.relu(visual_grounding_logits) # (batch_size, K, num_regions, visual_grounding_hidden_size)
            visual_grounding_logits = self.visual_grounding_layer_2(visual_grounding_logits) # (batch_size, K, num_regions, 1)
            sigmoid_attention = torch.sigmoid(visual_grounding_logits) # (batch_size, K, num_regions, 1)

            # Weighted average of local features
            weighted_sum = (sigmoid_attention * local_feat).sum(dim=-2) # (batch_size, K, visual_feature_proj_size)
            weighted_avg = weighted_sum / (sigmoid_attention.sum(dim=-2) + 1e-8)  # (batch_size, K, visual_feature_proj_size)
            sigmoid_attention = sigmoid_attention.squeeze(-1) # (batch_size, K, num_regions)

            # Phrase classifier
            if not skip_phrase_classifier:
                mlp_input = torch.cat([phrase_embeddings, global_feat, weighted_avg, sigmoid_attention], dim=-1)
                phrase_classifier_logits = self.classifier_mlp(mlp_input) # (batch_size, K, 1)

            # Global alignment
            if compute_global_alignment:
                assert self.predict_global_alignment
                global_feat_align = self.global_vf_align_proj(orig_global_feat) # (batch_size, alignment_proj_size)
                phrase_emb_align = self.phrase_emb_align_proj(phrase_embeddings) # (batch_size, K, alignment_proj_size)
                # normalize
                global_feat_align = torch.nn.functional.normalize(global_feat_align, p=2, dim=-1) # (batch_size, alignment_proj_size)
                phrase_emb_align = torch.nn.functional.normalize(phrase_emb_align, p=2, dim=-1) # (batch_size, K, alignment_proj_size)
                # cosine similarity
                element_wise_mult = global_feat_align.unsqueeze(1) * phrase_emb_align # (batch_size, K, alignment_proj_size)
                global_alignment_similarity = element_wise_mult.sum(dim=-1) # (batch_size, K)

            # Output
            output['sigmoid_attention'] = sigmoid_attention
            if not skip_phrase_classifier:
                output['phrase_classifier_logits'] = phrase_classifier_logits.squeeze(-1) # (batch_size, K, 1) -> (batch_size, K)
            if compute_global_alignment:
                output['global_alignment_similarity'] = global_alignment_similarity

        elif self.phrase_grounding_mode == PhraseGroundingMode.FILM_LAYERS_SIGMOID_ATTENTION_CUSTOM_CLASSIFIER_OBJECT_DETECTION.value:

            orig_global_feat = output['global_feat'] # (batch_size, global_feat_size)
            
            if self.apply_positional_encoding:
                local_feat = local_feat.view(-1, self.num_regions_sqrt, self.num_regions_sqrt, self.image_local_feat_size)
                local_feat = self.pos_encoding(local_feat) # apply positional encoding
                local_feat = local_feat.view(-1, self.num_regions, self.image_local_feat_size)

            # FiLM layer 1 (local features)
            num_facts = phrase_embeddings.shape[1] # K
            local_feat = local_feat.unsqueeze(1).expand(-1, num_facts, -1, -1) # (batch_size, K, num_regions, local_feat_size)
            local_feat_before_film = local_feat # remember the local features before applying FiLM
            phrase_embeddings_ = phrase_embeddings.unsqueeze(2).expand(-1, -1, self.num_regions, -1) # (batch_size, K, num_regions, phrase_embedding_size)
            local_feat = self.local_film_1(local_feat, phrase_embeddings_) # (batch_size, K, num_regions, local_feat_size)

            # Visual grounding
            visual_grounding_logits = self.visual_grounding_hidden_layer(local_feat) # (batch_size, K, num_regions, visual_grounding_hidden_size)
            visual_grounding_logits = F.gelu(visual_grounding_logits) # (batch_size, K, num_regions, visual_grounding_hidden_size)
            if predict_bboxes:
                if apply_nms:
                    predicted_bboxes, visual_grounding_binary_logits = self.visual_grounding_bbox_regressor(
                        visual_grounding_logits, apply_nms=apply_nms, iou_threshold=iou_threshold, conf_threshold=conf_threshold)
                else:
                    visual_grounding_bbox_logits, visual_grounding_binary_logits = self.visual_grounding_bbox_regressor(visual_grounding_logits)
            else:
                visual_grounding_binary_logits  = self.visual_grounding_bbox_regressor(visual_grounding_logits, predict_coords=False)
            sigmoid_attention = torch.sigmoid(visual_grounding_binary_logits) # (batch_size, K, num_regions, 1)

            # Weighted average of local features
            weighted_sum = (sigmoid_attention * local_feat_before_film).sum(dim=-2) # (batch_size, K, local_feat_size)
            weighted_avg = weighted_sum / (sigmoid_attention.sum(dim=-2) + 1e-8)  # (batch_size, K, local_feat_size)
            sigmoid_attention = sigmoid_attention.squeeze(-1) # (batch_size, K, num_regions)

            # Phrase classifier
            if not skip_phrase_classifier:
                global_feat = orig_global_feat.unsqueeze(1).expand(-1, num_facts, -1) # (batch_size, K, global_feat_size)
                global_feat = self.global_film(global_feat, phrase_embeddings) # (batch_size, K, global_feat_size)

                weighted_avg = self.local_film_2(weighted_avg, phrase_embeddings) # (batch_size, K, local_feat_size)
                mlp_input = torch.cat([global_feat, weighted_avg, sigmoid_attention], dim=-1)
                phrase_classifier_logits = self.classifier_mlp(mlp_input) # (batch_size, K, 1)

            # Global alignment
            if compute_global_alignment:
                assert self.predict_global_alignment
                global_feat_align = self.global_vf_align_proj(orig_global_feat) # (batch_size, alignment_proj_size)
                phrase_emb_align = self.phrase_emb_align_proj(phrase_embeddings) # (batch_size, K, alignment_proj_size)
                # normalize
                global_feat_align = torch.nn.functional.normalize(global_feat_align, p=2, dim=-1) # (batch_size, alignment_proj_size)
                phrase_emb_align = torch.nn.functional.normalize(phrase_emb_align, p=2, dim=-1) # (batch_size, K, alignment_proj_size)
                # cosine similarity
                element_wise_mult = global_feat_align.unsqueeze(1) * phrase_emb_align # (batch_size, K, alignment_proj_size)
                global_alignment_similarity = element_wise_mult.sum(dim=-1) # (batch_size, K)

            # Output
            output['sigmoid_attention'] = sigmoid_attention # (batch_size, K, num_regions)
            if predict_bboxes:
                if apply_nms:
                    output['predicted_bboxes'] = predicted_bboxes # (batch_size, K, list of bboxes)
                else:
                    output['visual_grounding_binary_logits'] = visual_grounding_binary_logits # (batch_size, K, num_regions, 1)
                    output['visual_grounding_bbox_logits'] = visual_grounding_bbox_logits # (batch_size, K, num_regions, 4)
            if not skip_phrase_classifier:
                output['phrase_classifier_logits'] = phrase_classifier_logits.squeeze(-1) # (batch_size, K, 1) -> (batch_size, K)
            if compute_global_alignment:
                output['global_alignment_similarity'] = global_alignment_similarity

        elif self.phrase_grounding_mode == PhraseGroundingMode.GLOBAL_POOLING_CONCAT_MLP__NO_GROUNDING.value:

            global_feat = output['global_feat'] # (batch_size, global_feat_size)
            global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=-1) # (batch_size, global_feat_size)
            global_feat = global_feat.unsqueeze(1).expand(-1, phrase_embeddings.shape[1], -1) # (batch_size, K, global_feat_size)

            # Phrase classifier
            mlp_input = torch.cat([phrase_embeddings, global_feat], dim=-1) # (batch_size, K, phrase_embedding_size + global_feat_size)
            phrase_classifier_logits = self.classifier_mlp(mlp_input) # (batch_size, K, 1)
            phrase_classifier_logits = phrase_classifier_logits.squeeze(-1) # (batch_size, K)

            # Output
            output['phrase_classifier_logits'] = phrase_classifier_logits

        elif self.phrase_grounding_mode == PhraseGroundingMode.GLOBAL_POOLING_FILM_MLP__NO_GROUNDING.value:

            global_feat = output['global_feat'] # (batch_size, global_feat_size)
            global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=-1) # (batch_size, global_feat_size)
            global_feat = global_feat.unsqueeze(1).expand(-1, phrase_embeddings.shape[1], -1) # (batch_size, K, global_feat_size)
            
            # FiLM layer
            global_feat = self.global_film(global_feat, phrase_embeddings) # (batch_size, K, global_feat_size)

            # Phrase classifier
            phrase_classifier_logits = self.classifier_mlp(global_feat) # (batch_size, K, 1)
            phrase_classifier_logits = phrase_classifier_logits.squeeze(-1) # (batch_size, K)

            # Output
            output['phrase_classifier_logits'] = phrase_classifier_logits

        elif (self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP__NO_GROUNDING.value or
                self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value):              

            if self.apply_positional_encoding:
                local_feat = local_feat.view(-1, self.num_regions_sqrt, self.num_regions_sqrt, self.image_local_feat_size)
                local_feat = self.pos_encoding(local_feat)
                local_feat = local_feat.view(-1, self.num_regions, self.image_local_feat_size)

            # FiLM layer (local features)
            num_facts = phrase_embeddings.shape[1] # K
            phrase_embeddings = phrase_embeddings.unsqueeze(2).expand(-1, -1, self.num_regions, -1) # (batch_size, K, num_regions, phrase_embedding_size)
            local_feat = local_feat.unsqueeze(1).expand(-1, num_facts, -1, -1) # (batch_size, K, num_regions, local_feat_size)
            local_feat_after_film = self.local_film(local_feat, phrase_embeddings) # (batch_size, K, num_regions, local_feat_size)

            # Local attention logits
            local_attention_logits = self.local_attention_hidden_layer(local_feat_after_film) # (batch_size, K, num_regions, visual_grounding_hidden_size)
            local_attention_logits = F.gelu(local_attention_logits) # (batch_size, K, num_regions, visual_grounding_hidden_size)

            if self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP__NO_GROUNDING.value:
                # Adaptive attention
                local_attention_logits = self.local_attention_final_layer(local_attention_logits) # (batch_size, K, num_regions, 1)
                sigmoid_attention = torch.sigmoid(local_attention_logits) # (batch_size, K, num_regions, 1)
            elif self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value:
                # Adaptive bbox regression
                if apply_nms:
                    predicted_bboxes, visual_grounding_confidence_logits = self.visual_grounding_bbox_regressor(
                        local_attention_logits, apply_nms=apply_nms, iou_threshold=iou_threshold, conf_threshold=conf_threshold,
                        max_det_per_class=max_det_per_class)
                else:
                    visual_grounding_bbox_logits, visual_grounding_confidence_logits = self.visual_grounding_bbox_regressor(local_attention_logits)
                sigmoid_attention = torch.sigmoid(visual_grounding_confidence_logits) # (batch_size, K, num_regions, 1)

            # Weighted average of local features after FiLM layer
            weighted_sum = (sigmoid_attention * local_feat_after_film).sum(dim=-2) # (batch_size, K, local_feat_size)
            weighted_avg = weighted_sum / (sigmoid_attention.sum(dim=-2) + 1e-8) # (batch_size, K, local_feat_size)
            sigmoid_attention = sigmoid_attention.squeeze(-1) # (batch_size, K, num_regions)

            # Phrase classifier
            mlp_input = torch.cat([weighted_avg, sigmoid_attention], dim=-1) # (batch_size, K, local_feat_size + num_regions)
            phrase_classifier_logits = self.classifier_mlp(mlp_input) # (batch_size, K, 1)
            phrase_classifier_logits = phrase_classifier_logits.squeeze(-1) # (batch_size, K)

            # Output
            output['phrase_classifier_logits'] = phrase_classifier_logits
            if self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value:
                if apply_nms:
                    output['predicted_bboxes'] = predicted_bboxes # (batch_size, (coords, conf, class))
                else:
                    output['visual_grounding_confidence_logits'] = visual_grounding_confidence_logits # (batch_size, K, num_regions, 1)
                    output['visual_grounding_bbox_logits'] = visual_grounding_bbox_logits # (batch_size, K, num_regions, 4)
            if return_sigmoid_attention:
                output['sigmoid_attention'] = sigmoid_attention

        else:
            raise ValueError(f'Unknown phrase_grounding_mode: {self.phrase_grounding_mode}')

        return output
    
    def compute_image_features(self, raw_images, only_global_alignment_features=False):
        # Visual Component
        output = super().forward(
            raw_images=raw_images,
            return_local_features=True,
            return_global_features=self.use_global_features,
            only_compute_features=True,
        )
        
        if only_global_alignment_features:
            assert self.predict_global_alignment
            global_feat = output['global_feat']
            global_feat = self.global_vf_align_proj(global_feat) # (batch_size, alignment_proj_size)
            global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=-1) # (batch_size, alignment_proj_size)
            return { 'global_feat': global_feat }

        local_feat = output['local_feat'] # (batch_size, num_regions, image_local_feat_size)

        if self.phrase_grounding_mode == PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER.value:
            if self.apply_positional_encoding:
                local_feat = local_feat + self.pe # apply positional encoding
            assert local_feat.shape == (raw_images.shape[0], self.num_regions, self.image_local_feat_size)
            return { 'local_feat': local_feat }
        
        # elif self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER.value:
        #     local_feat = self.image_proj(local_feat) # (batch_size, num_regions, transf_d_model)
        #     return { 'local_feat': local_feat }

        elif self.phrase_grounding_mode == PhraseGroundingMode.FILM_LAYERS_PLUS_SIGMOID_ATTENTION_AND_CUSTOM_CLASSIFIER.value:
            global_feat = output['global_feat'] # (batch_size, global_feat_size)
            global_feat = self.global_proj(global_feat) # (batch_size, visual_feature_proj_size)
            if self.apply_positional_encoding:
                local_feat = local_feat.view(-1, self.num_regions_sqrt, self.num_regions_sqrt, self.image_local_feat_size)
                local_feat = self.pos_encoding(local_feat) # apply positional encoding
                local_feat = local_feat.view(-1, self.num_regions, self.image_local_feat_size)
            local_feat = self.local_proj(local_feat) # (batch_size, num_regions, visual_feature_proj_size)
            return { 'local_feat': local_feat, 'global_feat': global_feat }
        
        elif self.phrase_grounding_mode in [
            PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP__NO_GROUNDING.value,
            PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value,
        ]:
            if self.apply_positional_encoding:
                local_feat = local_feat.view(-1, self.num_regions_sqrt, self.num_regions_sqrt, self.image_local_feat_size)
                local_feat = self.pos_encoding(local_feat)
                local_feat = local_feat.view(-1, self.num_regions, self.image_local_feat_size)
            return { 'local_feat': local_feat }

        else:
            raise ValueError(f'Unknown phrase_grounding_mode: {self.phrase_grounding_mode}')
        
    def forward_with_precomputed_image_features(
        self,
        local_feat, # (batch_size, num_regions, image_local_feat_size)
        phrase_embeddings, # (batch_size, K, phrase_embedding_size)
        global_feat=None, # (batch_size, global_feat_size)
        skip_phrase_classifier=False,
    ): 
        assert local_feat.ndim == 3
        assert phrase_embeddings.ndim == 3
        assert local_feat.shape[0] == phrase_embeddings.shape[0]
        assert local_feat.shape[1] == self.num_regions
        assert phrase_embeddings.shape[2] == self.phrase_embedding_size
        if global_feat is not None:
            assert global_feat.ndim == 2
            assert global_feat.shape[0] == local_feat.shape[0]
        
        output = {}

        if self.phrase_grounding_mode == PhraseGroundingMode.SIGMOID_ATTENTION_PLUS_CUSTOM_CLASSIFIER.value:

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
        
        # elif self.phrase_grounding_mode == PhraseGroundingMode.TRANSFORMER_ENCODER.value:

        #     batch_size, K, _ = phrase_embeddings.shape

        #     X_phrase = self.phrase_proj(phrase_embeddings) # (batch_size, K, transf_d_model)
        #     X_image = local_feat.unsqueeze(1).expand(-1, X_phrase.shape[1], -1, -1) # (batch_size, K, num_regions, transf_d_model)
        #     X_phrase = X_phrase.unsqueeze(2) # (batch_size, K, 1, transf_d_model)
        #     X = torch.cat([X_phrase, X_image], dim=2) # (batch_size, K, num_regions + 1, transf_d_model)
        #     X = X.view(-1, self.num_regions + 1, self.transf_d_model) # (batch_size * K, num_regions + 1, transf_d_model)

        #     if self.apply_positional_encoding:
        #         X = self.pe(X)

        #     X = self.transformer_encoder(X) # (batch_size * K, num_regions + 1, transf_d_model)

        #     # Phrase grounding attention
        #     image_representation = X[:, 1:, :] # (batch_size * K, num_regions, transf_d_model)
        #     attention_logits = self.att_proj(image_representation) # (batch_size * K, num_regions, 1)
        #     sigmoid_attention = torch.sigmoid(attention_logits) # (batch_size * K, num_regions, 1)
        #     sigmoid_attention = sigmoid_attention.view(batch_size, K, self.num_regions) # (batch_size, K, num_regions)
            
        #     # Phrase classifier
        #     # (use the first token as the phrase representation)
        #     if not skip_phrase_classifier:
        #         phrase_representation = X[:, 0, :] # (batch_size * K, transf_d_model)
        #         phrase_classifier_logits = self.phrase_classifier(phrase_representation) # (batch_size * K, 1)
        #         phrase_classifier_logits = phrase_classifier_logits.view(batch_size, K) # (batch_size, K)

        #     # Output
        #     output['sigmoid_attention'] = sigmoid_attention
        #     if not skip_phrase_classifier:
        #         output['phrase_classifier_logits'] = phrase_classifier_logits

        elif self.phrase_grounding_mode == PhraseGroundingMode.FILM_LAYERS_PLUS_SIGMOID_ATTENTION_AND_CUSTOM_CLASSIFIER.value:
            
            assert global_feat is not None

            # FiLM layers
            num_facts = phrase_embeddings.shape[1] # K
            global_feat = global_feat.unsqueeze(1).expand(-1, num_facts, -1) # (batch_size, K, visual_feature_proj_size)
            global_feat = self.global_film(global_feat, phrase_embeddings) # (batch_size, K, visual_feature_proj_size)

            local_feat = local_feat.unsqueeze(1).expand(-1, num_facts, -1, -1) # (batch_size, K, num_regions, visual_feature_proj_size)
            phrase_embeddings_ = phrase_embeddings.unsqueeze(2).expand(-1, -1, self.num_regions, -1) # (batch_size, K, num_regions, phrase_embedding_size)
            local_feat = self.local_film(local_feat, phrase_embeddings_) # (batch_size, K, num_regions, visual_feature_proj_size)

            # Visual grounding
            global_feat_ = global_feat.unsqueeze(2).expand(-1, -1, self.num_regions, -1) # (batch_size, K, num_regions, visual_feature_proj_size)
            visual_grounding_input = torch.cat([global_feat_, local_feat], dim=-1) # (batch_size, K, num_regions, 2 * visual_feature_proj_size)
            visual_grounding_logits = self.visual_grounding_layer_1(visual_grounding_input) # (batch_size, K, num_regions, visual_grounding_hidden_size)
            visual_grounding_logits = torch.relu(visual_grounding_logits) # (batch_size, K, num_regions, visual_grounding_hidden_size)
            visual_grounding_logits = self.visual_grounding_layer_2(visual_grounding_logits) # (batch_size, K, num_regions, 1)
            sigmoid_attention = torch.sigmoid(visual_grounding_logits) # (batch_size, K, num_regions, 1)

            # Weighted average of local features
            weighted_sum = (sigmoid_attention * local_feat).sum(dim=-2) # (batch_size, K, visual_feature_proj_size)
            weighted_avg = weighted_sum / (sigmoid_attention.sum(dim=-2) + 1e-8)  # (batch_size, K, visual_feature_proj_size)
            sigmoid_attention = sigmoid_attention.squeeze(-1) # (batch_size, K, num_regions)

            # Phrase classifier
            if not skip_phrase_classifier:
                mlp_input = torch.cat([phrase_embeddings, global_feat, weighted_avg, sigmoid_attention], dim=-1)
                phrase_classifier_logits = self.classifier_mlp(mlp_input) # (batch_size, K, 1)

            # Output
            output['sigmoid_attention'] = sigmoid_attention
            # assert sigmoid_attention.shape == (raw_images.shape[0], phrase_embeddings.shape[1], self.num_regions), \
            #     f'sigmoid_attention.shape: {sigmoid_attention.shape}'
            if not skip_phrase_classifier:
                output['phrase_classifier_logits'] = phrase_classifier_logits.squeeze(-1) # (batch_size, K, 1) -> (batch_size, K)
                # assert output['phrase_classifier_logits'].shape == (raw_images.shape[0], phrase_embeddings.shape[1])

        elif self.phrase_grounding_mode in [
            PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP__NO_GROUNDING.value,
            PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value,
        ]:

            # FiLM layer (local features)
            num_facts = phrase_embeddings.shape[1] # K
            phrase_embeddings = phrase_embeddings.unsqueeze(2).expand(-1, -1, self.num_regions, -1) # (batch_size, K, num_regions, phrase_embedding_size)
            local_feat = local_feat.unsqueeze(1).expand(-1, num_facts, -1, -1) # (batch_size, K, num_regions, local_feat_size)
            local_feat_after_film = self.local_film(local_feat, phrase_embeddings) # (batch_size, K, num_regions, local_feat_size)

            # Local attention logits
            local_attention_logits = self.local_attention_hidden_layer(local_feat_after_film) # (batch_size, K, num_regions, visual_grounding_hidden_size)
            local_attention_logits = F.gelu(local_attention_logits) # (batch_size, K, num_regions, visual_grounding_hidden_size)

            if self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP__NO_GROUNDING.value:
                local_attention_logits = self.local_attention_final_layer(local_attention_logits) # (batch_size, K, num_regions, 1)
                sigmoid_attention = torch.sigmoid(local_attention_logits) # (batch_size, K, num_regions, 1)
            elif self.phrase_grounding_mode == PhraseGroundingMode.ADAPTIVE_FILM_BASED_POOLING_MLP_WITH_BBOX_REGRESSION.value:
                visual_grounding_binary_logits = self.visual_grounding_bbox_regressor(local_attention_logits, predict_coords=False)
                sigmoid_attention = torch.sigmoid(visual_grounding_binary_logits) # (batch_size, K, num_regions, 1)

            # Weighted average of local features after FiLM layer
            weighted_sum = (sigmoid_attention * local_feat_after_film).sum(dim=-2) # (batch_size, K, local_feat_size)
            weighted_avg = weighted_sum / (sigmoid_attention.sum(dim=-2) + 1e-8) # (batch_size, K, local_feat_size)
            sigmoid_attention = sigmoid_attention.squeeze(-1) # (batch_size, K, num_regions)

            # Phrase classifier
            mlp_input = torch.cat([weighted_avg, sigmoid_attention], dim=-1) # (batch_size, K, local_feat_size + num_regions)
            phrase_classifier_logits = self.classifier_mlp(mlp_input) # (batch_size, K, 1)
            phrase_classifier_logits = phrase_classifier_logits.squeeze(-1) # (batch_size, K)

            # Output
            output['phrase_classifier_logits'] = phrase_classifier_logits

        else:
            raise ValueError(f'Unsupported phrase_grounding_mode: {self.phrase_grounding_mode}')

        return output
    
    def compute_global_alignment_similarity_with_precomputed_features(
        self,
        global_feat, # (batch_size, alignment_proj_size) -> it's assumed to be already projected and normalized
        phrase_embeddings, # (batch_size, K, phrase_embedding_size)
    ):
        assert global_feat.ndim == 2
        assert phrase_embeddings.ndim == 3
        assert global_feat.shape[0] == phrase_embeddings.shape[0]
        assert global_feat.shape[1] == self.alignment_proj_size
        assert phrase_embeddings.shape[2] == self.phrase_embedding_size

        phrase_emb_align = self.phrase_emb_align_proj(phrase_embeddings) # (batch_size, K, alignment_proj_size)
        phrase_emb_align = torch.nn.functional.normalize(phrase_emb_align, p=2, dim=-1) # (batch_size, K, alignment_proj_size)
        # cosine similarity
        element_wise_mult = global_feat.unsqueeze(1) * phrase_emb_align
        global_alignment_similarity = element_wise_mult.sum(dim=-1)
        return global_alignment_similarity # (batch_size, K)