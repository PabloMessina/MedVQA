import math
import copy
import torch
import torch.nn as nn
import torchvision.models as models
import torchxrayvision as xrv
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES,
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    get_anaxnet_bbox_sorted_indices,
)
from medvqa.datasets.mimiccxr import MIMICCXR_IMAGE_ORIENTATIONS
from medvqa.datasets.iuxray import IUXRAY_IMAGE_ORIENTATIONS
from medvqa.datasets.vinbig import VINBIG_BBOX_NAMES__MODIFIED, VINBIG_LABELS__MODIFIED
from medvqa.models.checkpoint import load_model_state_dict
from medvqa.models.common import freeze_parameters
from medvqa.models.mlp import MLP
from medvqa.models.vision.bbox_regression import (
    BBoxRegressorVersion,
    BoundingBoxRegressor_v1,
    BoundingBoxRegressor_v2,
    BoundingBoxRegressor_v3,
    BoundingBoxRegressorAndMultiLabelClassifier_v4,
    BoundingBoxRegressorAndMultiLabelClassifier_v4_1,
    BoundingBoxRegressorAndMultiLabelClassifier_v5,
    BoundingBoxRegressorAndMultiLabelClassifier_v6,
)
from medvqa.models.vision.multilabel_classification import (
    MLCVersion, MultilabelClassifier_v1, MultilabelClassifier_v2, MultilabelClassifier_v3,
)

from medvqa.utils.constants import (
    CHEST_IMAGENOME_GENDERS,    
    CHEXPERT_LABELS,
    CHEXPERT_GENDERS,
    CHEXPERT_ORIENTATIONS,
    CXR14_LABELS,
    PADCHEST_NUM_LABELS,
    PADCHEST_NUM_LOCALIZATIONS,
    PADCHEST_PROJECTIONS,
    VINBIG_BBOX_NAMES,
    VINBIG_LABELS,
)
from ultralytics.utils.ops import non_max_suppression
from ultralytics.nn.tasks import DetectionModel
import re
import logging

logger = logging.getLogger(__name__)

class RawImageEncoding:
    DENSENET_121 = 'densenet-121'
    DENSENET_121__TORCHXRAYVISION = 'densenet-121-torchxrayvision'
    RESNET__TORCHXRAYVISION = 'resnet-torchxrayvision'
    RESNET_AUTOENCODER__TORCHXRAYVISION = 'resnet-autoencoder-torchxrayvision'
    CLIP_RESNET = 'clip-resnet'
    CLIP_VIT = 'clip-vit'
    CLIP_VIT__HUGGINGFACE = 'clip-vit-huggingface'
    CLIP_VIT_LARGE__HUGGINGFACE = 'clip-vit-large-huggingface'
    CLIP_RESNET__HUGGINGFACE = 'clip-resnet-huggingface'
    VITMODEL__HUGGINGFACE = 'vitmodel-huggingface'
    VITMODEL_LARGE__HUGGINGFACE = 'vitmodel-huggingface-large'
    CONVNEXTMODEL__HUGGINGFACE = 'convnextmodel-huggingface'
    RAD_DINO__HUGGINGFACE = 'rad-dino-huggingface'
    CXRMATE_RRG24_UNIFORMER__HUGGINGFACE = 'cxrmate-rrg24-uniformer-huggingface'
    UNIFORMER_BASE_TL_384__HUGGINGFACE = 'uniformer-base-tl-384-huggingface'
    SIGLIP_HUGGINGFACE = 'siglip-huggingface'
    DETECTRON2 = 'detectron2'
    YOLOV8 = 'yolov8'
    YOLOV11_FOR_DET_MLC = 'yolov11-for-det-mlc'
    YOLOV11_FACT_CONDITIONED = 'yolov11-fact-conditioned'
    YOLOV11_FEATURE_EXTRACTOR = 'yolov11-feature-extractor'
    MEDSAM_FEATURE_EXTRACTOR__HUGGINGFACE = 'medsam-feature-extractor-huggingface'

class VisualInputMode:
    RAW_IMAGE = 'raw-image'
    PRECOMP_FEAT = 'precomp-feat' # precomputed visual features
    HYBRID = 'hybrid'

def does_include_image(visual_input_mode):
    return visual_input_mode in (VisualInputMode.RAW_IMAGE, VisualInputMode.HYBRID)

def does_include_visual_features(visual_input_mode):
    return visual_input_mode in (VisualInputMode.PRECOMP_FEAT, VisualInputMode.HYBRID)

def comes_with_positional_encoding(raw_image_encoding):
    return raw_image_encoding in [
        RawImageEncoding.CLIP_VIT,
        RawImageEncoding.CLIP_VIT__HUGGINGFACE,
        RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE,
        RawImageEncoding.VITMODEL__HUGGINGFACE,
        RawImageEncoding.VITMODEL_LARGE__HUGGINGFACE,
        RawImageEncoding.RAD_DINO__HUGGINGFACE,
        RawImageEncoding.CXRMATE_RRG24_UNIFORMER__HUGGINGFACE,
        RawImageEncoding.UNIFORMER_BASE_TL_384__HUGGINGFACE,
        RawImageEncoding.SIGLIP_HUGGINGFACE,
        RawImageEncoding.MEDSAM_FEATURE_EXTRACTOR__HUGGINGFACE,
    ]

def inject_mean_std_for_image_normalization(kwargs, raw_image_encoding):
    assert 'image_mean' not in kwargs
    assert 'image_std' not in kwargs
    if raw_image_encoding in [
        RawImageEncoding.DENSENET_121,
        RawImageEncoding.UNIFORMER_BASE_TL_384__HUGGINGFACE,
        RawImageEncoding.CXRMATE_RRG24_UNIFORMER__HUGGINGFACE,
        RawImageEncoding.MEDSAM_FEATURE_EXTRACTOR__HUGGINGFACE,
        RawImageEncoding.CONVNEXTMODEL__HUGGINGFACE,
    ]:
        kwargs['image_mean'] = [0.485, 0.456, 0.406]
        kwargs['image_std'] = [0.229, 0.224, 0.225]
    elif raw_image_encoding == RawImageEncoding.RAD_DINO__HUGGINGFACE:
        kwargs['image_mean'] = [0.5307, 0.5307, 0.5307]
        kwargs['image_std'] = [0.2583, 0.2583, 0.2583]
    elif raw_image_encoding in [
        RawImageEncoding.YOLOV8,
        RawImageEncoding.YOLOV11_FOR_DET_MLC,
        RawImageEncoding.YOLOV11_FACT_CONDITIONED,
        RawImageEncoding.YOLOV11_FEATURE_EXTRACTOR,
    ]:
        kwargs['image_mean'] = [0.0, 0.0, 0.0]
        kwargs['image_std'] = [1.0, 1.0, 1.0]
    else:
        raise ValueError(f'Unknown raw_image_encoding: {raw_image_encoding}')


class MultiPurposeVisualModule(nn.Module):

    def __init__(self,
                # Image Encoder kwargs
                visual_input_mode=VisualInputMode.RAW_IMAGE,
                raw_image_encoding=RawImageEncoding.DENSENET_121,
                image_local_feat_size=None,
                freeze_image_encoder=False,
                only_compute_features=False,
                image_encoder_pretrained_weights_path=None,
                imagenet_pretrained=False,
                visual_features_mlp_in_dim=None,
                visual_features_mlp_out_dim=None,
                visual_features_mlp_hidden_dims=None,
                classification_mlp_hidden_dims=None,
                clip_version=None,
                num_regions=None,
                huggingface_model_name=None,
                torchxrayvision_weights_name=None,
                detectron2_model_yaml=None,
                roi_heads_batch_size_per_image=None,
                rpn_batch_size_per_image=None,
                roi_align_output_size=None,
                yolov8_model_name_or_path=None,
                yolov8_model_alias=None,
                yolov8_use_one_detector_per_dataset=False,
                yolov11_model_name_or_path=None,
                yolov11_model_alias=None,
                query_embed_size=None,
                local_attention_hidden_size=None,
                image_size=None,
                image_encoder_dropout_p=0,
                # Auxiliary tasks kwargs
                use_mimiccxr=False,
                use_iuxray=False,
                use_chexpert=False,
                use_cxr14=False,
                use_vinbig=False,
                use_padchest=False,
                classify_tags=False,
                classify_orientation=False,
                classify_gender=False,
                classify_chexpert=False,
                classify_questions=False,
                classify_chest_imagenome=False,
                classify_labels_vinbig=False,
                chexpert_mlc_version=MLCVersion.DEFAULT,
                chexpert_mlc_hidden_size=None,
                predict_bboxes_chest_imagenome=False,
                chest_imagenome_train_average_bbox_coords=None,
                predict_labels_and_bboxes_chest_imagenome=False,
                chest_imagenome_anatomy_to_labels=None,
                chest_imagenome_anatomy_group_to_labels=None,
                n_medical_tags=None,
                n_questions_aux_task=None,
                n_chest_imagenome_labels=None,
                n_chest_imagenome_bboxes=None,
                chest_imagenome_bbox_hidden_size=None,
                chest_imagenome_bbox_regressor_version=None,
                chest_imagenome_mlc_version=MLCVersion.DEFAULT,
                chest_imagenome_mlc_hidden_size=None,
                use_anaxnet_bbox_subset=False,
                predict_bboxes_vinbig=False,
                vinbig_mlc_hidden_size=None,
                merge_findings=False,
                n_findings=None,
                device=None,
                use_linear_head_for_classification=False,
                use_vinbig_with_modified_labels=False,
                # Other kwargs
                **unused_kwargs,
                ): 
        super().__init__()

        logger.info('MultiPurposeVisualModule()')
        
        self.visual_input_mode = visual_input_mode
        self.raw_image_encoding = raw_image_encoding
        self.clip_version = clip_version
        self.huggingface_model_name = huggingface_model_name
        self.torchxrayvision_weights_name = torchxrayvision_weights_name
        self.detectron2_model_yaml = detectron2_model_yaml
        self.classify_chexpert = classify_chexpert
        self.classify_questions = classify_questions
        self.classify_tags = classify_tags
        self.classify_orientation = classify_orientation
        self.classify_gender = classify_gender
        self.classify_chest_imagenome = classify_chest_imagenome
        self.classify_labels_vinbig = classify_labels_vinbig
        self.use_mimiccxr = use_mimiccxr
        self.use_iuxray = use_iuxray
        self.use_chexpert = use_chexpert
        self.use_cxr14 = use_cxr14
        self.use_vinbig = use_vinbig
        self.use_padchest = use_padchest
        self.n_medical_tags = n_medical_tags
        self.n_questions_aux_task = n_questions_aux_task
        self.merge_findings = merge_findings
        self.n_findings = n_findings
        self.n_chest_imagenome_labels = n_chest_imagenome_labels
        self.n_chest_imagenome_bboxes = n_chest_imagenome_bboxes
        self.image_encoder_pretrained_weights_path = image_encoder_pretrained_weights_path
        self.imagenet_pretrained = imagenet_pretrained
        self.freeze_image_encoder = freeze_image_encoder
        self.only_compute_features = only_compute_features
        self.image_local_feat_size = image_local_feat_size
        self.visual_features_mlp_in_dim = visual_features_mlp_in_dim
        self.visual_features_mlp_out_dim = visual_features_mlp_out_dim
        self.visual_features_mlp_hidden_dims = visual_features_mlp_hidden_dims
        self.classification_mlp_hidden_dims = classification_mlp_hidden_dims
        self.chexpert_mlc_version = chexpert_mlc_version
        self.chexpert_mlc_hidden_size = chexpert_mlc_hidden_size
        self.predict_bboxes_chest_imagenome = predict_bboxes_chest_imagenome
        self.chest_imagenome_bbox_hidden_size = chest_imagenome_bbox_hidden_size
        self.chest_imagenome_bbox_regressor_version = chest_imagenome_bbox_regressor_version
        self.chest_imagenome_train_average_bbox_coords = chest_imagenome_train_average_bbox_coords
        self.predict_labels_and_bboxes_chest_imagenome = predict_labels_and_bboxes_chest_imagenome
        self.chest_imagenome_anatomy_to_labels = chest_imagenome_anatomy_to_labels
        self.chest_imagenome_anatomy_group_to_labels = chest_imagenome_anatomy_group_to_labels
        self.use_anaxnet_bbox_subset = use_anaxnet_bbox_subset
        self.num_regions = num_regions
        self.roi_heads_batch_size_per_image = roi_heads_batch_size_per_image
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.roi_align_output_size = roi_align_output_size
        self.yolov8_model_name_or_path = yolov8_model_name_or_path
        self.yolov8_model_alias = yolov8_model_alias
        self.yolov8_use_one_detector_per_dataset = yolov8_use_one_detector_per_dataset
        self.yolov11_model_name_or_path = yolov11_model_name_or_path
        self.yolov11_model_alias = yolov11_model_alias
        self.chest_imagenome_mlc_version = chest_imagenome_mlc_version
        self.chest_imagenome_mlc_hidden_size = chest_imagenome_mlc_hidden_size
        self.predict_bboxes_vinbig = predict_bboxes_vinbig
        self.vinbig_mlc_hidden_size = vinbig_mlc_hidden_size
        self.image_encoder_dropout_p = image_encoder_dropout_p
        self.query_embed_size = query_embed_size
        self.local_attention_hidden_size = local_attention_hidden_size
        self.image_size = image_size
        self.device = device
        self.use_linear_head_for_classification = use_linear_head_for_classification
        self.use_vinbig_with_modified_labels = use_vinbig_with_modified_labels

        # Check that num_regions is a square number
        if self.num_regions is not None:
            self.num_regions_sqrt = math.isqrt(self.num_regions)
            assert self.num_regions_sqrt ** 2 == self.num_regions
        
        if raw_image_encoding in [
            RawImageEncoding.VITMODEL__HUGGINGFACE,
            RawImageEncoding.VITMODEL_LARGE__HUGGINGFACE,
            RawImageEncoding.CLIP_VIT__HUGGINGFACE,
            RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE,
            RawImageEncoding.CLIP_RESNET__HUGGINGFACE,
            RawImageEncoding.CONVNEXTMODEL__HUGGINGFACE,
            RawImageEncoding.RAD_DINO__HUGGINGFACE,
            RawImageEncoding.CXRMATE_RRG24_UNIFORMER__HUGGINGFACE,
            RawImageEncoding.UNIFORMER_BASE_TL_384__HUGGINGFACE,
            RawImageEncoding.SIGLIP_HUGGINGFACE,
            RawImageEncoding.MEDSAM_FEATURE_EXTRACTOR__HUGGINGFACE,
        ]:
            assert huggingface_model_name is not None
        if raw_image_encoding in [
            RawImageEncoding.DENSENET_121__TORCHXRAYVISION,
            RawImageEncoding.RESNET__TORCHXRAYVISION,
            RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION,
        ]:
            assert torchxrayvision_weights_name is not None
        if raw_image_encoding == RawImageEncoding.YOLOV8:
            assert yolov8_model_name_or_path is not None
            assert yolov8_model_alias is not None
        if raw_image_encoding in [
            RawImageEncoding.YOLOV11_FOR_DET_MLC,
            RawImageEncoding.YOLOV11_FACT_CONDITIONED,
            RawImageEncoding.YOLOV11_FEATURE_EXTRACTOR,
        ]:
            assert yolov11_model_name_or_path is not None
            assert yolov11_model_alias is not None

        if use_anaxnet_bbox_subset:
            assert predict_bboxes_chest_imagenome
            if raw_image_encoding != RawImageEncoding.DETECTRON2:
                assert chest_imagenome_train_average_bbox_coords is not None
                assert len(chest_imagenome_train_average_bbox_coords) == 4 * CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES

        if classify_labels_vinbig:
            assert use_vinbig
        
        self._init_visual_backbone()
        self._init_auxiliary_tasks()

    def _init_visual_backbone(self):

        skip_local_global = self.raw_image_encoding in [
            RawImageEncoding.DETECTRON2,
            RawImageEncoding.YOLOV11_FOR_DET_MLC,
            RawImageEncoding.YOLOV11_FACT_CONDITIONED,
        ]

        if not skip_local_global:
            global_feat_size = 0
        
        if does_include_image(self.visual_input_mode):
            if self.clip_version is not None:
                model_name = self.clip_version
            elif self.huggingface_model_name is not None:
                model_name = self.huggingface_model_name
            elif self.torchxrayvision_weights_name is not None:
                model_name = self.torchxrayvision_weights_name
            elif self.detectron2_model_yaml is not None:
                model_name = self.detectron2_model_yaml
            elif self.yolov8_model_name_or_path is not None:
                model_name = self.yolov8_model_name_or_path
            else:
                model_name = None
            self._init_raw_image_encoder(self.image_encoder_pretrained_weights_path,
                                         self.imagenet_pretrained, model_name, self.freeze_image_encoder,
                                         self.image_encoder_dropout_p)
            if not skip_local_global:
                global_feat_size += self._get_raw_image_encoder_global_feat_size(self.image_local_feat_size)
        
        if does_include_visual_features(self.visual_input_mode):
            self._init_mlp_visual_feat_encoder(
                self.visual_features_mlp_in_dim, self.visual_features_mlp_out_dim,
                 self.visual_features_mlp_hidden_dims, self.freeze_image_encoder)
            global_feat_size += self.visual_features_mlp_out_dim
        
        if not skip_local_global:
            assert global_feat_size > 0
            self.local_feat_size = self.image_local_feat_size
            self.global_feat_size = global_feat_size
            logger.info(f'  self.global_feat_size = {self.global_feat_size}')
            logger.info(f'  self.local_feat_size = {self.local_feat_size}')

    def _get_raw_image_encoder_global_feat_size(self, image_local_feat_size):
        if self.raw_image_encoding in [
            RawImageEncoding.DENSENET_121,
            RawImageEncoding.DENSENET_121__TORCHXRAYVISION,
            RawImageEncoding.RESNET__TORCHXRAYVISION,
            RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION,
            RawImageEncoding.YOLOV8,
            RawImageEncoding.YOLOV11_FEATURE_EXTRACTOR,
            RawImageEncoding.CXRMATE_RRG24_UNIFORMER__HUGGINGFACE,
            RawImageEncoding.UNIFORMER_BASE_TL_384__HUGGINGFACE,
            RawImageEncoding.MEDSAM_FEATURE_EXTRACTOR__HUGGINGFACE,
        ]:
            return 2 * image_local_feat_size
        if self.raw_image_encoding == RawImageEncoding.CLIP_VIT:
            return CLIP_VIT_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.CLIP_RESNET:
            return CLIP_RESNET_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE:
            return HUGGINGFACE_CLIP_VIT_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE:
            return HUGGINGFACE_CLIP_VIT_LARGE_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.CLIP_RESNET__HUGGINGFACE:
            return CLIP_RESNET_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE:
            return HUGGINGFACE_VITMODEL_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.VITMODEL_LARGE__HUGGINGFACE:
            return HUGGINGFACE_VITMODEL_LARGE_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.CONVNEXTMODEL__HUGGINGFACE:
            return HUGGINGFACE_CONVNEXTMODEL_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.RAD_DINO__HUGGINGFACE:
            return HUGGINGFACE_RAD_DINO_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.SIGLIP_HUGGINGFACE:
            return HUGGINGFACE_SIGLIP_GLOBAL_FEAT_SIZE[self.huggingface_model_name]
        raise ValueError(f'Unknown raw_image_encoding: {self.raw_image_encoding}')
    
    def _init_raw_image_encoder(self, pretrained_weights_path, imagenet_pretrained, model_name, freeze_image_encoder,
                                dropout_p=0):
        logger.info(f'  Initializing raw_image_encoder: {self.raw_image_encoding}')
        ignore_name_regex = None
        if self.raw_image_encoding == RawImageEncoding.DENSENET_121:
            self.raw_image_encoder = create_densenet121_feature_extractor(
                pretrained_weights_path, imagenet_pretrained, drop_rate=dropout_p)
        elif self.raw_image_encoding == RawImageEncoding.DENSENET_121__TORCHXRAYVISION:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_torchxrayvision_densenet121_feature_extractor(model_name)
        elif self.raw_image_encoding == RawImageEncoding.RESNET__TORCHXRAYVISION:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_torchxrayvision_resnet_feature_extractor(model_name)
        elif self.raw_image_encoding == RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_torchxrayvision_resnet_autoencoder_feature_extractor(model_name)
        elif self.raw_image_encoding == RawImageEncoding.CLIP_RESNET:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_clip_resnet_feature_extractor(model_name, pretrained_weights_path)
        elif self.raw_image_encoding == RawImageEncoding.CLIP_VIT:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_clip_vit_feature_extractor(model_name, pretrained_weights_path)
        elif self.raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE or \
                self.raw_image_encoding == RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_huggingface_clip_vit_feature_extractor(model_name, pretrained_weights_path)
        elif self.raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE or \
                self.raw_image_encoding == RawImageEncoding.VITMODEL_LARGE__HUGGINGFACE:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_huggingface_vitmodel_feature_extractor(model_name, pretrained_weights_path)
            ignore_name_regex = HUGGINGFACE_VITMODEL_UNFROZEN_PARAM_NAMES_REGEX
        elif self.raw_image_encoding == RawImageEncoding.DETECTRON2:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            if self.predict_bboxes_chest_imagenome:
                if self.use_anaxnet_bbox_subset:
                    num_classes = CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES
                else:
                    num_classes = CHEST_IMAGENOME_NUM_BBOX_CLASSES
            else:
                assert False, 'We only support predicting bboxes for chest_imagenome at the moment'
            assert self.rpn_batch_size_per_image is not None
            self.raw_image_encoder = create_detectron2_model(model_name, num_classes=num_classes,
                                                             roi_heads_batch_size_per_image=self.roi_heads_batch_size_per_image,
                                                             rpn_batch_size_per_image=self.rpn_batch_size_per_image)
        elif self.raw_image_encoding == RawImageEncoding.YOLOV8:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model', bold=True)
            assert self.predict_bboxes_chest_imagenome or self.predict_bboxes_vinbig
            if self.yolov8_use_one_detector_per_dataset:
                assert self.predict_bboxes_chest_imagenome and self.predict_bboxes_vinbig # at least two datasets
                class_names_list = []
                nc_list = []
                self.num_bbox_classes = []
                if self.predict_bboxes_chest_imagenome:
                    class_names = {}
                    if self.use_anaxnet_bbox_subset:
                        for i, idx in enumerate(get_anaxnet_bbox_sorted_indices()):
                            class_names[i] = CHEST_IMAGENOME_BBOX_NAMES[idx]
                    else:
                        for i, name in enumerate(CHEST_IMAGENOME_BBOX_NAMES):
                            class_names[i] = name
                    class_names_list.append(class_names)
                    nc_list.append(len(class_names))
                    self.num_bbox_classes.append(len(class_names))
                if self.predict_bboxes_vinbig:
                    class_names = {}
                    for i, name in enumerate(VINBIG_BBOX_NAMES):
                        class_names[i] = name
                    class_names_list.append(class_names)
                    nc_list.append(len(class_names))
                    self.num_bbox_classes.append(len(class_names))
                logger.info(f'  nc_list: {nc_list}')
                self.raw_image_encoder = create_yolov8_model_for_multiple_datasets(
                    model_name_or_path=model_name, nc_list=nc_list, class_names_list=class_names_list
                )
            else:
                offset = 0
                class_names = {}
                if self.predict_bboxes_chest_imagenome:
                    if self.use_anaxnet_bbox_subset:
                        for i, idx in enumerate(get_anaxnet_bbox_sorted_indices()):
                            class_names[i+offset] = CHEST_IMAGENOME_BBOX_NAMES[idx]
                        offset += CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES
                    else:
                        for i, name in enumerate(CHEST_IMAGENOME_BBOX_NAMES):
                            class_names[i+offset] = name
                        offset += CHEST_IMAGENOME_NUM_BBOX_CLASSES
                if self.predict_bboxes_vinbig:
                    for i, name in enumerate(VINBIG_BBOX_NAMES):
                        class_names[i+offset] = name
                    offset += len(VINBIG_BBOX_NAMES)
                logger.info(f'  num_bbox_classes: {offset}')
                self.num_bbox_classes = offset
                self.raw_image_encoder = create_yolov8_model(model_name_or_path=model_name, nc=offset, class_names=class_names)
        elif self.raw_image_encoding == RawImageEncoding.YOLOV11_FOR_DET_MLC:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            assert self.predict_bboxes_chest_imagenome or self.predict_bboxes_vinbig
            self.raw_image_encoder = create_yolov11_model_for_det_mlc(
                predict_bboxes_chest_imagenome=self.predict_bboxes_chest_imagenome,
                predict_bboxes_vinbig=self.predict_bboxes_vinbig,
                predict_labels_vinbig=self.classify_labels_vinbig,
                use_vinbig_with_modified_labels=self.use_vinbig_with_modified_labels,
                query_embed_size=self.query_embed_size,
                mlp_hidden_dims=self.classification_mlp_hidden_dims,
                local_attention_hidden_size=self.local_attention_hidden_size,
                image_size=self.image_size,
                model_name_or_path=self.yolov11_model_name_or_path,
                model_alias=self.yolov11_model_alias,
                device=self.device,
            )
        elif self.raw_image_encoding == RawImageEncoding.YOLOV11_FACT_CONDITIONED:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_yolov11_fact_conditioned(
                fact_embed_size=self.query_embed_size,
                mlp_hidden_dims=self.classification_mlp_hidden_dims,
                local_attention_hidden_size=self.local_attention_hidden_size,
                image_size=self.image_size,
                model_name_or_path=self.yolov11_model_name_or_path,
                yolo_alias=self.yolov11_model_alias,
                device=self.device,
                yolov11_pretrained_weights_path=pretrained_weights_path,
            )
        elif self.raw_image_encoding == RawImageEncoding.YOLOV11_FEATURE_EXTRACTOR:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_yolov11_feature_extractor(
                model_name_or_path=self.yolov11_model_name_or_path,
                yolo_alias=self.yolov11_model_alias,
                yolov11_pretrained_weights_path=pretrained_weights_path,
            )
        elif self.raw_image_encoding == RawImageEncoding.CONVNEXTMODEL__HUGGINGFACE:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_huggingface_convnextmodel_feature_extractor(model_name, pretrained_weights_path)
        elif self.raw_image_encoding == RawImageEncoding.RAD_DINO__HUGGINGFACE:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_huggingface_rad_dino_feature_extractor(model_name, pretrained_weights_path)
        elif self.raw_image_encoding == RawImageEncoding.CXRMATE_RRG24_UNIFORMER__HUGGINGFACE:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_huggingface_cxrmate_rrg24_uniformer_feature_extractor(model_name, pretrained_weights_path)
        elif self.raw_image_encoding == RawImageEncoding.UNIFORMER_BASE_TL_384__HUGGINGFACE:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_uniformer_base_tl_384_feature_extractor(model_name, pretrained_weights_path)
        elif self.raw_image_encoding == RawImageEncoding.SIGLIP_HUGGINGFACE:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_huggingface_siglip_feature_extractor(model_name, pretrained_weights_path)
        elif self.raw_image_encoding == RawImageEncoding.MEDSAM_FEATURE_EXTRACTOR__HUGGINGFACE:
            if dropout_p:
                logger.warning('dropout_p is not implemented yet for this model')
            self.raw_image_encoder = create_huggingface_medsam_feature_extractor(model_name, pretrained_weights_path)
        else: raise ValueError(f'Unknown raw_image_encoding: {self.raw_image_encoding}')
        if freeze_image_encoder: freeze_parameters(self.raw_image_encoder, ignore_name_regex)

    def _init_mlp_visual_feat_encoder(self, mlp_in_dim, mlp_out_dim, mlp_hidden_dims, freeze_image_encoder):
        logger.info(f'  Initializing mlp_visual_feat_encoder: {mlp_in_dim} -> {mlp_out_dim}, hidden_dims={mlp_hidden_dims}')
        self.mlp_vf_encoder = MLP(in_dim=mlp_in_dim, out_dim=mlp_out_dim, hidden_dims=mlp_hidden_dims)
        if freeze_image_encoder: freeze_parameters(self.mlp_vf_encoder)

    def _init_auxiliary_tasks(self):
        
        logger.info('  Initializing auxiliary tasks')
        # Optional auxiliary tasks

        if self.raw_image_encoding in [
            RawImageEncoding.YOLOV11_FOR_DET_MLC,
            RawImageEncoding.YOLOV11_FACT_CONDITIONED,
        ]:
            logger.info('    Skipping auxiliary tasks initialization for YOLOv11 models')
            return
        
        # 1) medical tags classification
        if self.classify_tags:
            logger.info(f'    Initializing medical tags classification task (n_medical_tags={self.n_medical_tags})')
            assert self.n_medical_tags is not None
            self.W_tags = nn.Linear(self.global_feat_size, self.n_medical_tags)
        
        # 2) orientation classifiction
        if self.classify_orientation:
            logger.info(f'    Initializing orientation classification task')
            if self.use_mimiccxr:
                self.W_ori_mimiccxr = nn.Linear(self.global_feat_size, len(MIMICCXR_IMAGE_ORIENTATIONS))
            if self.use_iuxray:
                self.W_ori_iuxray = nn.Linear(self.global_feat_size, len(IUXRAY_IMAGE_ORIENTATIONS))
            if self.use_chexpert or self.use_cxr14: # weight sharing among chexpert & CRX14
                self.W_ori_chexpert = nn.Linear(self.global_feat_size, len(CHEXPERT_ORIENTATIONS))

        # 3) questions classification
        if self.classify_questions:
            logger.info(f'    Initializing questions classification task (n_questions_aux_task={self.n_questions_aux_task})')
            self.W_q = nn.Linear(self.global_feat_size, self.n_questions_aux_task)

        # 4) gender classification
        if self.classify_gender:
            logger.info(f'    Initializing gender classification task')   
            self.W_gender_chexpert = nn.Linear(self.global_feat_size, len(CHEXPERT_GENDERS))
            self.W_gender_chstimgn = nn.Linear(self.global_feat_size, len(CHEST_IMAGENOME_GENDERS))

        if self.merge_findings:
            logger.info(f'    Initializing merged findings classification task (n_findings={self.n_findings})')
            assert self.n_findings is not None
            self.W_findings = nn.Linear(self.global_feat_size, self.n_findings)
        else:        
            # 6) chexpert classifiction
            if self.classify_chexpert:
                logger.info(f'    Initializing chexpert classification task')
                logger.info(f'    chexpert_mlc_version: {self.chexpert_mlc_version}')
                if self.chexpert_mlc_version == MLCVersion.DEFAULT:
                    self.W_chx = nn.Linear(self.global_feat_size, len(CHEXPERT_LABELS))
                elif self.chexpert_mlc_version == MLCVersion.V3:
                    self.MLC_chx = MultilabelClassifier_v3(
                        local_feat_dim=self.local_feat_size,
                        global_feat_dim=self.global_feat_size,
                        hidden_dim=self.chexpert_mlc_hidden_size,
                        num_regions=self.num_regions,
                        num_labels=len(CHEXPERT_LABELS),
                    )
                else: raise ValueError(f'Unknown chexpert_mlc_version: {self.chexpert_mlc_version}')

            # 7) CXR14 specific labels
            if self.use_cxr14:
                logger.info(f'    Initializing CXR14 classification task')
                self.W_cxr14 = nn.Linear(self.global_feat_size, len(CXR14_LABELS))

            # 8) VinBig specific labels
            if self.use_vinbig:
                if self.classify_labels_vinbig:
                    logger.info(f'    Initializing VinBig classification task')
                    if self.use_linear_head_for_classification:
                        self.W_vinbig = nn.Linear(self.global_feat_size, len(VINBIG_LABELS))
                    else:
                        self.MLC_vinbig = MultilabelClassifier_v3(
                            local_feat_dim=self.local_feat_size,
                            global_feat_dim=self.global_feat_size,
                            hidden_dim=self.vinbig_mlc_hidden_size,
                            num_regions=self.num_regions,
                            num_labels=len(VINBIG_LABELS),
                        )

        # 9) PadChest specific tasks
        if self.use_padchest:
            logger.info(f'    Initializing PadChest classification tasks')
            self.W_padchest_labels = nn.Linear(self.global_feat_size, PADCHEST_NUM_LABELS)
            self.W_padchest_loc = nn.Linear(self.global_feat_size, PADCHEST_NUM_LOCALIZATIONS)
            self.W_padchest_ori = nn.Linear(self.global_feat_size, len(PADCHEST_PROJECTIONS))

        # 10) Chest ImaGenome specific tasks
        if self.predict_labels_and_bboxes_chest_imagenome:
            # We predict both labels and bboxes for Chest ImaGenome using a combined approach
            assert self.classify_chest_imagenome
            assert not self.use_anaxnet_bbox_subset # Not supported yet for this combined approach
            assert self.chest_imagenome_anatomy_to_labels is not None
            assert self.chest_imagenome_bbox_hidden_size is not None
            assert self.chest_imagenome_anatomy_group_to_labels is not None
            assert len(self.chest_imagenome_anatomy_to_labels) >= CHEST_IMAGENOME_NUM_BBOX_CLASSES, \
                f'len(self.chest_imagenome_anatomy_to_labels)={len(self.chest_imagenome_anatomy_to_labels)}' \
                f' != CHEST_IMAGENOME_NUM_BBOX_CLASSES={CHEST_IMAGENOME_NUM_BBOX_CLASSES}'
            assert self.n_chest_imagenome_bboxes is not None
            logger.info(f'    Initializing Chest ImaGenome classification and bounding box regression'
                  f' tasks (n_chest_imagenome_labels={self.n_chest_imagenome_labels})')
            if self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V4:
                assert self.chest_imagenome_train_average_bbox_coords is not None
                self.bbox_regressor_and_classifier = BoundingBoxRegressorAndMultiLabelClassifier_v4(
                    local_feat_dim=self.local_feat_size,
                    global_feat_dim=self.global_feat_size,
                    hidden_dim=self.chest_imagenome_bbox_hidden_size,
                    num_bboxes=self.n_chest_imagenome_bboxes,
                    num_bboxes_to_supervise=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
                    num_regions=self.num_regions,
                    train_average_bbox_coords=self.chest_imagenome_train_average_bbox_coords,
                    bbox_to_labels=self.chest_imagenome_anatomy_to_labels,
                    bbox_group_to_labels=self.chest_imagenome_anatomy_group_to_labels,
                )
            elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V4_1:
                self.bbox_regressor_and_classifier = BoundingBoxRegressorAndMultiLabelClassifier_v4_1(
                    local_feat_dim=self.local_feat_size,
                    hidden_dim=self.chest_imagenome_bbox_hidden_size,
                    num_bboxes=self.n_chest_imagenome_bboxes,
                    num_regions=self.num_regions,
                    bbox_to_labels=self.chest_imagenome_anatomy_to_labels,
                    bbox_group_to_labels=self.chest_imagenome_anatomy_group_to_labels,
                )
            elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V5:
                assert self.roi_align_output_size is not None
                self.bbox_regressor_and_classifier = BoundingBoxRegressorAndMultiLabelClassifier_v5(
                    local_feat_dim=self.local_feat_size,                    
                    input_size=self.num_regions_sqrt,
                    roi_align_output_size=self.roi_align_output_size,
                    roi_align_spatial_scale=self.num_regions_sqrt, # [0, 1] -> [0, num_regions_sqrt]
                    hidden_dim=self.chest_imagenome_bbox_hidden_size,
                    num_boxes=self.n_chest_imagenome_bboxes,
                    num_boxes_to_supervise=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
                    bbox_to_labels=self.chest_imagenome_anatomy_to_labels,
                    bbox_group_to_labels=self.chest_imagenome_anatomy_group_to_labels,
                )
            elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V6:
                assert self.chest_imagenome_train_average_bbox_coords is not None
                self.bbox_regressor_and_classifier = BoundingBoxRegressorAndMultiLabelClassifier_v6(
                    global_feat_dim=self.global_feat_size,
                    local_feat_dim=self.local_feat_size,
                    input_size=self.num_regions_sqrt,
                    roi_align_output_size=self.roi_align_output_size,
                    roi_align_spatial_scale=self.num_regions_sqrt, # [0, 1] -> [0, num_regions_sqrt]
                    hidden_dim=self.chest_imagenome_bbox_hidden_size,
                    num_boxes=self.n_chest_imagenome_bboxes,
                    num_boxes_to_supervise=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
                    bbox_to_labels=self.chest_imagenome_anatomy_to_labels,
                    bbox_group_to_labels=self.chest_imagenome_anatomy_group_to_labels,
                    train_average_bbox_coords=self.chest_imagenome_train_average_bbox_coords,
                )
            else:
                raise NotImplementedError(f'Unsupported Chest ImaGenome bbox regressor version: {self.chest_imagenome_bbox_regressor_version}')
        else:
            if self.classify_chest_imagenome:
                logger.info(f'    Initializing Chest ImaGenome classification task (n_chest_imagenome_labels={self.n_chest_imagenome_labels})')
                if self.chest_imagenome_mlc_version == MLCVersion.DEFAULT:
                    assert self.n_chest_imagenome_labels is not None
                    self.W_chst_imgn = nn.Linear(self.global_feat_size, self.n_chest_imagenome_labels)
                elif self.chest_imagenome_mlc_version == MLCVersion.V1:
                    assert self.chest_imagenome_mlc_hidden_size is not None
                    assert self.chest_imagenome_anatomy_to_labels is not None
                    assert self.chest_imagenome_anatomy_group_to_labels is not None
                    assert self.n_chest_imagenome_bboxes is not None
                    self.MLC_chst_imgn = MultilabelClassifier_v1(
                        local_feat_dim=self.local_feat_size,
                        global_feat_dim=self.global_feat_size,
                        hidden_dim=self.chest_imagenome_mlc_hidden_size,
                        num_bboxes=self.n_chest_imagenome_bboxes,
                        num_regions=self.num_regions,
                        bbox_to_labels=self.chest_imagenome_anatomy_to_labels,
                        bbox_group_to_labels=self.chest_imagenome_anatomy_group_to_labels,
                    )
                elif self.chest_imagenome_mlc_version == MLCVersion.V2:
                    self.MLC_chst_imgn = MultilabelClassifier_v2(
                        global_feat_dim=self.global_feat_size,
                        local_feat_dim=self.local_feat_size,
                        input_size=self.num_regions_sqrt,
                        roi_align_output_size=self.roi_align_output_size,
                        roi_align_spatial_scale=self.num_regions_sqrt, # [0, 1] -> [0, num_regions_sqrt]
                        hidden_dim=self.chest_imagenome_mlc_hidden_size,
                        num_boxes=self.n_chest_imagenome_bboxes,
                        num_annotated_boxes=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
                        bbox_to_labels=self.chest_imagenome_anatomy_to_labels,
                        bbox_group_to_labels=self.chest_imagenome_anatomy_group_to_labels,
                    )
                else:
                    raise NotImplementedError(f'Unsupported Chest ImaGenome MLC version: {self.chest_imagenome_mlc_version}')
            if self.predict_bboxes_chest_imagenome and not self.raw_image_encoding in [
                    RawImageEncoding.DETECTRON2, RawImageEncoding.YOLOV8]:
                logger.info(f'    Initializing Chest ImaGenome bounding box regression task')
                # Don't need to initialize these custom bbox regressors if using Detectron2
                assert self.chest_imagenome_bbox_hidden_size is not None
                assert self.chest_imagenome_bbox_regressor_version is not None
                if self.use_anaxnet_bbox_subset:
                    num_classes = CHEST_IMAGENOME_ANAXNET_NUM_BBOX_CLASSES
                else:
                    num_classes = CHEST_IMAGENOME_NUM_BBOX_CLASSES
                if self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V1:
                    self.bbox_regressor_chst_imgn = BoundingBoxRegressor_v1(
                        local_feat_dim=self.local_feat_size,
                        global_feat_dim=self.global_feat_size,
                        hidden_dim=self.chest_imagenome_bbox_hidden_size,
                        num_classes=num_classes,
                        train_average_bbox_coords=self.chest_imagenome_train_average_bbox_coords,
                    )
                elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V2:
                    assert self.num_regions is not None
                    self.bbox_regressor_chst_imgn = BoundingBoxRegressor_v2(
                        local_feat_dim=self.local_feat_size,
                        global_feat_dim=self.global_feat_size,
                        hidden_dim=self.chest_imagenome_bbox_hidden_size,
                        num_classes=num_classes,
                        train_average_bbox_coords=self.chest_imagenome_train_average_bbox_coords,
                        num_regions=self.num_regions,
                    )
                elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V3:
                    assert self.num_regions is not None
                    self.bbox_regressor_chst_imgn = BoundingBoxRegressor_v3(
                        local_feat_dim=self.local_feat_size,
                        global_feat_dim=self.global_feat_size,
                        hidden_dim=self.chest_imagenome_bbox_hidden_size,
                        num_classes=num_classes,
                        train_average_bbox_coords=self.chest_imagenome_train_average_bbox_coords,
                        num_regions=self.num_regions,                    
                    )
                else:
                    raise ValueError(f'Unknown bbox regressor version: {self.chest_imagenome_bbox_regressor_version}')

    def get_name(self):
        if self.raw_image_encoding == RawImageEncoding.DENSENET_121:
            img_str = 'dn121'
        elif self.raw_image_encoding == RawImageEncoding.DENSENET_121__TORCHXRAYVISION:
            img_str = f'dn121-txv({self.torchxrayvision_weights_name})'
        elif self.raw_image_encoding == RawImageEncoding.RESNET__TORCHXRAYVISION:
            img_str = f'resnet-txv({self.torchxrayvision_weights_name})'
        elif self.raw_image_encoding == RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION:
            img_str = f'resnet-ae-txv({self.torchxrayvision_weights_name})'
        elif self.raw_image_encoding in (RawImageEncoding.CLIP_VIT,
                                         RawImageEncoding.CLIP_RESNET):
            img_str = f'clip-{self.clip_version}'
        elif self.raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE or \
                self.raw_image_encoding == RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE:
            img_str = HUGGINGFACE_CLIP_VIT_NAMES_2_SHORT[self.clip_version]
        elif self.raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE or \
                self.raw_image_encoding == RawImageEncoding.VITMODEL_LARGE__HUGGINGFACE:
            img_str = HUGGINGFACE_VITMODEL_NAMES_2_SHORT[self.huggingface_model_name]
        elif self.raw_image_encoding == RawImageEncoding.DETECTRON2:
            img_str = DETECTRON2_YAML_2_SHORT[self.detectron2_model_yaml]
        elif self.raw_image_encoding == RawImageEncoding.YOLOV8:
            img_str = self.yolov8_model_alias
        elif self.raw_image_encoding in [RawImageEncoding.YOLOV11_FOR_DET_MLC,
                                            RawImageEncoding.YOLOV11_FACT_CONDITIONED,
                                            RawImageEncoding.YOLOV11_FEATURE_EXTRACTOR]:
            img_str = str(self.raw_image_encoder)
        elif self.raw_image_encoding == RawImageEncoding.CONVNEXTMODEL__HUGGINGFACE:
            img_str = HUGGINGFACE_CONVNEXTMODEL_NAMES_2_SHORT[self.huggingface_model_name]
        elif self.raw_image_encoding == RawImageEncoding.RAD_DINO__HUGGINGFACE:
            img_str = HUGGINGFACE_RAD_DINO_NAMES_2_SHORT[self.huggingface_model_name]
        elif self.raw_image_encoding == RawImageEncoding.CXRMATE_RRG24_UNIFORMER__HUGGINGFACE:
            img_str = HUGGINGFACE_CXRMATE_RRG24_UNIFORMER_NAMES_2_SHORT[self.huggingface_model_name]
        elif self.raw_image_encoding == RawImageEncoding.UNIFORMER_BASE_TL_384__HUGGINGFACE:
            img_str = HUGGINGFACE_UNIFORMER_BASE_TL_384_NAMES_2_SHORT[self.huggingface_model_name]
        elif self.raw_image_encoding == RawImageEncoding.SIGLIP_HUGGINGFACE:
            img_str = HUGGINGFACE_SIGLIP_NAMES_2_SHORT[self.huggingface_model_name]
        elif self.raw_image_encoding == RawImageEncoding.MEDSAM_FEATURE_EXTRACTOR__HUGGINGFACE:
            img_str = HUGGINGFACE_MEDSAM_NAMES_2_SHORT[self.huggingface_model_name]
        else: assert False, f'Unknown raw image encoding {self.raw_image_encoding}'
        vf_str = 'mlp(vf)'
        if self.visual_input_mode == VisualInputMode.HYBRID:
            vm_str = f'{img_str}+{vf_str}'
        elif self.visual_input_mode == VisualInputMode.PRECOMP_FEAT:
            vm_str = vf_str
        elif self.visual_input_mode == VisualInputMode.RAW_IMAGE:
            vm_str = img_str
        else: assert False, f'Unknown visual input mode {self.visual_input_mode}'
        return vm_str

    def forward(
        self,
        raw_images=None,
        visual_features=None,
        iuxray_forward=False,
        mimiccxr_forward=False,
        chexpert_forward=False,
        cxr14_forward=False,
        vinbig_forward=False,
        padchest_forward=False,
        detectron2_forward=False,
        detectron2_input=None,
        pred_bbox_coords=None,
        refine_bbox_coords=False,
        return_local_features=False,
        return_global_features=False,
        skip_mlc=False,
        yolov8_detection_layer_index=None,
        yolov11_classification_tasks=None,
        yolov11_detection_tasks=None,
        only_compute_features=False,
        apply_nms=True,
        batch=None, # Used by YOLOv11_FOR_DET_MLC
        conf_thres=None, # Used by YOLOv11_FOR_DET_MLC
        iou_thres=None, # Used by YOLOv11_FOR_DET_MLC
        max_det=None, # Used by YOLOv11_FOR_DET_MLC
        fact_embeddings=None, # Used by YOLOv11_FACT_CONDITIONED
        use_first_n_facts_for_detection=None, # Used by YOLOv11_FACT_CONDITIONED
        **unused_kwargs,
    ):  
        # Forward pass for YOLOV11_FOR_DET_MLC
        if self.raw_image_encoding == RawImageEncoding.YOLOV11_FOR_DET_MLC:
            assert yolov11_classification_tasks is not None or yolov11_detection_tasks is not None
            if apply_nms:
                assert conf_thres is not None
                assert iou_thres is not None
                assert max_det is not None
            return self.raw_image_encoder(
                x=raw_images,
                detection_task_names=yolov11_detection_tasks,
                classification_task_names=yolov11_classification_tasks,
                batch=batch,
                apply_nms=apply_nms,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
            )
        # Forward pass for YOLOV11_FACT_CONDITIONED
        elif self.raw_image_encoding == RawImageEncoding.YOLOV11_FACT_CONDITIONED:
            assert fact_embeddings is not None
            if apply_nms:
                assert conf_thres is not None
                assert iou_thres is not None
                assert max_det is not None
            return self.raw_image_encoder(
                images=raw_images,
                fact_embeddings=fact_embeddings,
                batch=batch,
                apply_nms=apply_nms,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                use_first_n_facts_for_detection=use_first_n_facts_for_detection,
            )

        # Detectron2-specific forward pass
        if detectron2_forward:
            assert detectron2_input is not None
            return self.raw_image_encoder(detectron2_input)

        # General forward pass
        assert (raw_images is not None) or (visual_features is not None)

        only_compute_features = only_compute_features or self.only_compute_features

        if only_compute_features:
            assert return_global_features or return_local_features

        permute_and_flatten_local_feat = False
        compute_global_features = False

        if return_global_features:
            compute_global_features = True
        if return_local_features:
            permute_and_flatten_local_feat = True

        if self.predict_labels_and_bboxes_chest_imagenome:
            if self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V4:
                compute_global_features = True
                permute_and_flatten_local_feat = True
            elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V4_1:
                permute_and_flatten_local_feat = True
            elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V6:
                compute_global_features = True
        
        if self.classify_chest_imagenome:
            if self.raw_image_encoding == RawImageEncoding.YOLOV8:
                compute_global_features = True
            if self.chest_imagenome_mlc_version == MLCVersion.V1:
                permute_and_flatten_local_feat = True
        
        if vinbig_forward:
            compute_global_features = True
            if not self.use_linear_head_for_classification:
                permute_and_flatten_local_feat = True

        if compute_global_features:
            global_list = []
        
        if raw_images is not None:

            use_default_method = False
            if self.raw_image_encoding == RawImageEncoding.DENSENET_121 or \
                    self.raw_image_encoding == RawImageEncoding.CXRMATE_RRG24_UNIFORMER__HUGGINGFACE:
                local_feat_NxCxHxW = self.raw_image_encoder(raw_images)
                use_default_method = True
            elif self.raw_image_encoding == RawImageEncoding.DENSENET_121__TORCHXRAYVISION:
                local_feat_NxCxHxW = self.raw_image_encoder.features(raw_images)
                use_default_method = True
            elif self.raw_image_encoding == RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION:
                local_feat_NxCxHxW = self.raw_image_encoder.encode(raw_images)
                use_default_method = True
            elif self.raw_image_encoding == RawImageEncoding.YOLOV11_FEATURE_EXTRACTOR:
                local_feat_NxCxHxW = self.raw_image_encoder(raw_images)[-1] # take the last layer out of the list
                use_default_method = True
            elif self.raw_image_encoding == RawImageEncoding.MEDSAM_FEATURE_EXTRACTOR__HUGGINGFACE:
                local_feat_NxCxHxW = self.raw_image_encoder(raw_images).last_hidden_state
                use_default_method = True
            if use_default_method:
                # compute local features
                batch_size = raw_images.size(0)
                feat_size = local_feat_NxCxHxW.size(1)
                if permute_and_flatten_local_feat:
                    local_feat_NxRxC = local_feat_NxCxHxW.permute(0,2,3,1).view(batch_size, -1, feat_size)
                # compute global features
                if compute_global_features:
                    aux = local_feat_NxCxHxW.view(batch_size, feat_size, -1)
                    global_avg_pool = aux.mean(2)
                    global_max_pool = aux.max(2)[0]
                    global_list.append(global_avg_pool)
                    global_list.append(global_max_pool)

            elif self.raw_image_encoding == RawImageEncoding.CLIP_RESNET:
                global_feat, local_feat_NxCxHxW = self.raw_image_encoder(raw_images, return_local_features=True)
                batch_size = raw_images.size(0)
                feat_size = local_feat_NxCxHxW.size(1)
                if permute_and_flatten_local_feat:
                    local_feat_NxRxC = local_feat_NxCxHxW.permute(0,2,3,1).view(batch_size, -1, feat_size)
                if compute_global_features:
                    global_list.append(global_feat)
            
            elif self.raw_image_encoding == RawImageEncoding.CLIP_VIT:
                global_feat, local_feat_NxRxC = self.raw_image_encoder(raw_images, return_local_features=True)
                if compute_global_features:
                    global_list.append(global_feat)
            
            elif self.raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE or \
                    self.raw_image_encoding == RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE or \
                    self.raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE or \
                    self.raw_image_encoding == RawImageEncoding.VITMODEL_LARGE__HUGGINGFACE or \
                    self.raw_image_encoding == RawImageEncoding.RAD_DINO__HUGGINGFACE:
                tmp = self.raw_image_encoder(raw_images)
                global_feat, local_feat_NxRxC = tmp.pooler_output, tmp.last_hidden_state[:, 1:] # remove CLS token
                if compute_global_features:
                    global_list.append(global_feat)

            elif self.raw_image_encoding == RawImageEncoding.SIGLIP_HUGGINGFACE:
                tmp = self.raw_image_encoder(raw_images)
                global_feat, local_feat_NxRxC = tmp.pooler_output, tmp.last_hidden_state
                if compute_global_features:
                    global_list.append(global_feat)

            elif self.raw_image_encoding == RawImageEncoding.CONVNEXTMODEL__HUGGINGFACE:
                tmp = self.raw_image_encoder(raw_images)
                global_feat, local_feat_NxCxHxW = tmp.pooler_output, tmp.last_hidden_state
                if permute_and_flatten_local_feat:
                    batch_size = raw_images.size(0)
                    feat_size = local_feat_NxCxHxW.size(1)
                    local_feat_NxRxC = local_feat_NxCxHxW.permute(0,2,3,1).view(batch_size, -1, feat_size)
                if compute_global_features:
                    global_list.append(global_feat)

            elif self.raw_image_encoding == RawImageEncoding.UNIFORMER_BASE_TL_384__HUGGINGFACE:
                tmp = self.raw_image_encoder(raw_images)
                local_feat_NxRxC = tmp.last_hidden_state
                if compute_global_features:
                    global_avg_pool = local_feat_NxRxC.mean(1)
                    global_max_pool = local_feat_NxRxC.max(1)[0]
                    global_list.append(global_avg_pool)
                    global_list.append(global_max_pool)

            elif self.raw_image_encoding == RawImageEncoding.YOLOV8:
                batch_size = raw_images.size(0)
                if only_compute_features:
                    local_feat_NxCxHxW = self.raw_image_encoder.custom_forward(raw_images, only_return_features=True)
                else:
                    if yolov8_detection_layer_index is None:
                        local_feat_NxCxHxW, detection_output = self.raw_image_encoder.custom_forward(raw_images)
                    else:
                        local_feat_NxCxHxW, detection_output = self.raw_image_encoder.custom_forward(
                            x=raw_images,
                            detection_layer_index=yolov8_detection_layer_index,
                        )
                    assert local_feat_NxCxHxW.shape == (batch_size, self.local_feat_size,
                                                        self.num_regions_sqrt, self.num_regions_sqrt), \
                    f'local_feat_NxCxHxW.shape = {local_feat_NxCxHxW.shape}, but expected {(batch_size, self.local_feat_size, self.num_regions_sqrt, self.num_regions_sqrt)}'
                    assert type(detection_output) == list or type(detection_output) == tuple
                    assert len(detection_output) == 3 or len(detection_output) == 2
                    if len(detection_output) == 2: # this is the case when the model is in evaluation mode
                        # logger.info('YOLOv8 output in evaluation mode')
                        yolov8_features = detection_output[1]
                        yolov8_predictions = detection_output[0]
                        # logger.info(f'yolov8_predictions.shape = {yolov8_predictions.shape}')
                        if yolov8_detection_layer_index is None:
                            num_bbox_classes = self.num_bbox_classes
                        else:
                            num_bbox_classes = self.num_bbox_classes[yolov8_detection_layer_index]
                        yolov8_predictions = non_max_suppression(yolov8_predictions.detach(),
                                                                conf_thres=0.1, iou_thres=0.1,
                                                                max_det=num_bbox_classes)
                        # logger.info(f'len(yolov8_predictions) (after NMS) = {len(yolov8_predictions)}')
                    else: # this is the case when the model is in training mode
                        # logger.info('YOLOv8 output in training mode')
                        yolov8_predictions = None
                        yolov8_features = detection_output
                if permute_and_flatten_local_feat:
                    local_feat_NxRxC = local_feat_NxCxHxW.permute(0,2,3,1).view(batch_size, -1, self.local_feat_size)
                # compute global features
                if compute_global_features:
                    aux = local_feat_NxCxHxW.view(batch_size, self.local_feat_size, -1)
                    global_avg_pool = aux.mean(2)
                    global_max_pool = aux.max(2)[0]
                    global_list.append(global_avg_pool)
                    global_list.append(global_max_pool)
            
            else: assert False, f'Unknown raw image encoding {self.raw_image_encoding}'

        if visual_features is not None:
            if compute_global_features:
                global_vf  = self.mlp_vf_encoder(visual_features)
                global_list.append(global_vf)

        if compute_global_features:
            if len(global_list) > 1:
                global_feat = torch.cat(global_list, 1)
            else:
                global_feat = global_list[0]

        output = {}

        if return_global_features:
            output['global_feat'] = global_feat
        if return_local_features:
            output['local_feat'] = local_feat_NxRxC

        if not only_compute_features:
            if self.merge_findings:
                output['pred_findings'] = self.W_findings(global_feat)
                output['pred_findings_probs'] = torch.sigmoid(output['pred_findings'])
            if chexpert_forward:
                if self.classify_orientation:
                    output['pred_orientation'] = self.W_ori_chexpert(global_feat)
                if self.classify_gender:
                    output['pred_gender'] = self.W_gender_chexpert(global_feat)
                if not self.merge_findings and self.classify_chexpert:
                    output['pred_chexpert'] = self.W_chx(global_feat)
                    output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
            elif cxr14_forward:
                if self.classify_orientation:            
                    output['pred_orientation'] = self.W_ori_chexpert(global_feat) # weight sharing with chexpert
                if self.classify_gender:
                    output['pred_gender'] = self.W_gender_chexpert(global_feat) # weight sharing with chexpert
                if not self.merge_findings:
                    output['pred_cxr14'] = self.W_cxr14(global_feat)
                    output['pred_cxr14_probs'] = torch.sigmoid(output['pred_cxr14'])
            elif vinbig_forward:
                if self.classify_labels_vinbig:
                    if not self.merge_findings:
                        if self.use_linear_head_for_classification:
                            output['pred_vinbig'] = self.W_vinbig(global_feat)
                        else:
                            output['pred_vinbig'] = self.MLC_vinbig(local_feat_NxRxC, global_feat)
                        output['pred_vinbig_probs'] = torch.sigmoid(output['pred_vinbig'])
                if self.predict_bboxes_vinbig:
                    assert self.raw_image_encoding == RawImageEncoding.YOLOV8
                    output['yolov8_features'] = yolov8_features
                    if yolov8_predictions is not None:
                        output['yolov8_predictions'] = yolov8_predictions
            elif padchest_forward:
                if self.classify_orientation:
                    output['pred_orientation'] = self.W_padchest_ori(global_feat)
                if self.classify_gender:
                    output['pred_gender'] = self.W_gender_chexpert(global_feat) # weight sharing with chexpert
                output['pred_padchest_labels'] = self.W_padchest_labels(global_feat)
                output['pred_padchest_labels_probs'] = torch.sigmoid(output['pred_padchest_labels'])
                output['pred_padchest_loc'] = self.W_padchest_loc(global_feat)
                output['pred_padchest_loc_probs'] = torch.sigmoid(output['pred_padchest_loc'])
            elif iuxray_forward:
                if self.classify_tags:
                    output['pred_tags'] = self.W_tags(global_feat)
                if self.classify_orientation:
                    output['iuxray_pred_orientation'] = self.W_ori_iuxray(global_feat)
                if self.classify_questions:
                    output['pred_qlabels'] = self.W_q(global_feat)
                if not self.merge_findings and self.classify_chexpert:
                    output['pred_chexpert'] = self.W_chx(global_feat)
                    output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
                if self.classify_chest_imagenome:
                    output['pred_chest_imagenome'] = self.W_chst_imgn(global_feat)
                    output['pred_chest_imagenome_probs'] = torch.sigmoid(output['pred_chest_imagenome'])
            elif mimiccxr_forward:
                if self.classify_tags:
                    output['pred_tags'] = self.W_tags(global_feat)
                if self.classify_orientation:
                    output['mimiccxr_pred_orientation'] = self.W_ori_iuxray(global_feat)
                if self.classify_questions:
                    output['pred_qlabels'] = self.W_q(global_feat)
                if self.classify_gender:
                    output['pred_gender'] = self.W_gender_chstimgn(global_feat)
                if not self.merge_findings and self.classify_chexpert:
                    if self.chexpert_mlc_version == MLCVersion.DEFAULT:
                        output['pred_chexpert'] = self.W_chx(global_feat)
                    elif self.chexpert_mlc_version == MLCVersion.V3:
                        output['pred_chexpert'] = self.MLC_chx(local_feat_NxRxC, global_feat)
                    else: assert False
                    output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
                if self.predict_labels_and_bboxes_chest_imagenome:
                    if self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V4:
                        pred_bbox_coords, pred_bbox_presence, mlc_scores = self.bbox_regressor_and_classifier(local_feat_NxRxC, global_feat)
                        output['pred_chest_imagenome'] = mlc_scores
                        output['pred_chest_imagenome_probs'] = torch.sigmoid(output['pred_chest_imagenome'])
                        output['pred_chest_imagenome_bbox_coords'] = pred_bbox_coords
                        output['pred_chest_imagenome_bbox_presence'] = pred_bbox_presence
                    elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V4_1:
                        mlc_scores = self.bbox_regressor_and_classifier(local_feat_NxRxC)
                        output['pred_chest_imagenome'] = mlc_scores
                        output['pred_chest_imagenome_probs'] = torch.sigmoid(output['pred_chest_imagenome'])
                    elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V5:
                        assert pred_bbox_coords is not None
                        if refine_bbox_coords:
                            pred_bbox_coords, pred_bbox_presence, mlc_scores = self.bbox_regressor_and_classifier(
                                local_feat_NxCxHxW, pred_bbox_coords, refine_bbox_coords)
                            output['pred_chest_imagenome'] = mlc_scores
                            output['pred_chest_imagenome_probs'] = torch.sigmoid(output['pred_chest_imagenome'])
                            output['pred_chest_imagenome_bbox_coords'] = pred_bbox_coords
                            output['pred_chest_imagenome_bbox_presence'] = pred_bbox_presence
                        else:
                            mlc_scores  = self.bbox_regressor_and_classifier(
                                local_feat_NxCxHxW, pred_bbox_coords, refine_bbox_coords)
                            output['pred_chest_imagenome'] = mlc_scores
                            output['pred_chest_imagenome_probs'] = torch.sigmoid(output['pred_chest_imagenome'])
                    elif self.chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V6:
                        assert pred_bbox_coords is not None
                        pred_bbox_coords, pred_bbox_presence, mlc_scores = self.bbox_regressor_and_classifier(
                            global_feat, local_feat_NxCxHxW, pred_bbox_coords)
                        output['pred_chest_imagenome'] = mlc_scores
                        output['pred_chest_imagenome_probs'] = torch.sigmoid(output['pred_chest_imagenome'])
                        output['pred_chest_imagenome_bbox_coords'] = pred_bbox_coords
                        output['pred_chest_imagenome_bbox_presence'] = pred_bbox_presence
                else:
                    if self.classify_chest_imagenome and not skip_mlc:
                        if self.chest_imagenome_mlc_version == MLCVersion.DEFAULT:
                            output['pred_chest_imagenome'] = self.W_chst_imgn(global_feat)
                        elif self.chest_imagenome_mlc_version == MLCVersion.V1:
                            output['pred_chest_imagenome'] = self.MLC_chst_imgn(local_feat_NxRxC, global_feat)
                        elif self.chest_imagenome_mlc_version == MLCVersion.V2:
                            assert pred_bbox_coords is not None
                            output['pred_chest_imagenome'] = self.MLC_chst_imgn(local_feat_NxCxHxW, global_feat, pred_bbox_coords)
                        else: assert False
                        output['pred_chest_imagenome_probs'] = torch.sigmoid(output['pred_chest_imagenome'])
                    if self.predict_bboxes_chest_imagenome:
                        if self.raw_image_encoding == RawImageEncoding.YOLOV8:
                            output['yolov8_features'] = yolov8_features
                            if yolov8_predictions is not None:
                                output['yolov8_predictions'] = yolov8_predictions
                        else:
                            pred_bbox_coords, pred_bbox_presence = self.bbox_regressor_chst_imgn(local_feat_NxRxC, global_feat)
                            output['pred_chest_imagenome_bbox_coords'] = pred_bbox_coords
                            output['pred_chest_imagenome_bbox_presence'] = pred_bbox_presence
            else: assert False, f'Unknown forward pass mode'

        return output

class ImageQuestionClassifier(nn.Module):

    def __init__(self, image_local_feat_size, n_questions):
        super().__init__()
        densenet = models.densenet121(pretrained=False)
        self.image_encoder = densenet.features
        self.W_q = nn.Linear(image_local_feat_size * 2, n_questions)

    def forward(self, images):
        # cnn local features
        batch_size = images.size(0)
        local_feat = self.image_encoder(images)
        feat_size = local_feat.size(1)
        local_feat = local_feat.permute(0,2,3,1).view(batch_size, -1, feat_size)
        # compute global features
        global_avg_pool = local_feat.mean(1)
        global_max_pool = local_feat.max(1)[0]
        global_feat = torch.cat((global_avg_pool, global_max_pool), 1)
        # classify questions
        return self.W_q(global_feat)

class ImageFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        densenet = models.densenet121(pretrained=False)
        self.image_encoder = densenet.features

    def forward(self, images):
        # cnn local features
        batch_size = images.size(0)
        local_feat = self.image_encoder(images)
        feat_size = local_feat.size(1)
        local_feat = local_feat.permute(0,2,3,1).view(batch_size, -1, feat_size)
        # compute global features
        global_avg_pool = local_feat.mean(1)
        global_max_pool = local_feat.max(1)[0]
        global_feat = torch.cat((global_avg_pool, global_max_pool), 1)
        # return global features
        return global_feat

class DensenetVisualModule(nn.Module):

    def __init__(self, image_local_feat_size,
                pretrained=True,
                densenet_pretrained_weights_path=None,
                classify_tags=False,
                classify_orientation=False,
                classify_chexpert=False,
                classify_questions=False,
                classify_chest_imagenome=False,
                n_medical_tags=None,
                n_questions_aux_task=None,
                n_chest_imagenome_labels=None,
                use_chexpert_forward=False,
                merge_findings=False,
                n_findings=False,
                chexpert_indices=None,
                freeze_cnn=False,
                **unused_kwargs,
        ):
        super().__init__()
        self.name = 'densenet121'
        self.raw_image_encoder = create_densenet121_feature_extractor(densenet_pretrained_weights_path, pretrained)
        if freeze_cnn: freeze_parameters(self.raw_image_encoder)

        self.global_feat_size = 2 * image_local_feat_size        
        self.merge_findings = merge_findings

        # Optional auxiliary tasks
        
        # 1) medical tags classification
        if classify_tags:
            assert n_medical_tags is not None
            self.W_tags = nn.Linear(self.global_feat_size, n_medical_tags)
            self.tags_aux_task = True
        else:
            self.tags_aux_task = False
        
        # 2) orientation classifiction
        if classify_orientation:
            self.W_ori_mimiccxr = nn.Linear(self.global_feat_size, len(MIMICCXR_IMAGE_ORIENTATIONS))
            self.W_ori_iuxray = nn.Linear(self.global_feat_size, len(IUXRAY_IMAGE_ORIENTATIONS))
            self.orien_aux_task = True
        else:
            self.orien_aux_task = False

        # 3) questions classification
        if classify_questions:
            self.W_q = nn.Linear(self.global_feat_size, n_questions_aux_task)
            self.q_aux_task = True
        else:
            self.q_aux_task = False

        if merge_findings:
            assert n_findings is not None
            assert chexpert_indices is not None
            self.W_findings = nn.Linear(self.global_feat_size, n_findings)
            self.chexpert_indices = chexpert_indices
            if classify_chexpert:
                self.chx_aux_task = True
        else:        
            # 5) chexpert classifiction
            if classify_chexpert:
                self.W_chx = nn.Linear(self.global_feat_size, len(CHEXPERT_LABELS))
                self.chx_aux_task = True
            else:
                self.chx_aux_task = False
            
            # 6) chest imagenome classifiction
            if classify_chest_imagenome:
                self.W_chst_imgn = nn.Linear(self.global_feat_size, n_chest_imagenome_labels)
                self.chst_imgn_label_aux_task = True
            else:
                self.chst_imgn_label_aux_task = False

        if use_chexpert_forward:
            self.W_gender_chexpert = nn.Linear(self.global_feat_size, len(CHEXPERT_GENDERS))
            self.W_ori_chexpert = nn.Linear(self.global_feat_size, len(CHEXPERT_ORIENTATIONS))

    def forward(self, images, iuxray_foward=False, mimiccxr_foward=False, chexpert_forward=False):
        
        # local features
        batch_size = images.size(0)
        local_feat = self.raw_image_encoder(images)
        feat_size = local_feat.size(1)
        local_feat = local_feat.permute(0,2,3,1).view(batch_size, -1, feat_size)
        
        # global features
        global_avg_pool = local_feat.mean(1)
        global_max_pool = local_feat.max(1)[0]
        global_feat = torch.cat((global_avg_pool, global_max_pool), 1)

        output = { 'global_feat': global_feat }

        if chexpert_forward:
            output['pred_chexpert'] = self.W_chx(global_feat)
            output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
            output['pred_orientation'] = self.W_ori_chexpert(global_feat)
            output['pred_gender'] = self.W_gender_chexpert(global_feat)
        else:
            # auxiliary tasks (optional)        
            if self.tags_aux_task:
                output['pred_tags'] = self.W_tags(global_feat)        
            if self.orien_aux_task:            
                if iuxray_foward:
                    output['iuxray_pred_orientation'] = self.W_ori_iuxray(global_feat)
                if mimiccxr_foward:
                    output['mimiccxr_pred_orientation'] = self.W_ori_mimiccxr(global_feat)        
            if self.chx_aux_task:
                if self.merge_findings:                    
                    output['pred_chexpert'] = self.W_findings(global_feat)[:, self.chexpert_indices]
                else:
                    output['pred_chexpert'] = self.W_chx(global_feat)
                output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
            if self.q_aux_task:
                output['pred_qlabels'] = self.W_q(global_feat)
            if self.chst_imgn_label_aux_task:
                output['pred_chest_imagenome'] = self.W_chst_imgn(global_feat)
                output['pred_chest_imagenome_probs'] = torch.sigmoid(output['pred_chest_imagenome'])
        
        return output

def create_densenet121_feature_extractor(
    pretrained_weights_path=None,
    imagenet_pretrained=False,
    drop_rate=0.0,
):
    logger.info('create_densenet121_feature_extractor()')
    logger.info(f'   drop_rate: {drop_rate}')
    # Load pre-trained CNN weights
    if pretrained_weights_path:
        densenet = models.densenet121(drop_rate=drop_rate)
        pretrained_weights = torch.load(pretrained_weights_path, map_location='cuda')
        densenet.load_state_dict(pretrained_weights, strict=False)
        logger.info("DenseNet121's pretrained weights loaded from", pretrained_weights_path)
    elif imagenet_pretrained:
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1, drop_rate=drop_rate)
        logger.info("DenseNet121's pretrained weights loaded from ImageNet")
    else:
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT, drop_rate=drop_rate)
        logger.info("DenseNet121's default weights loaded")
    return densenet.features

def create_torchxrayvision_densenet121_feature_extractor(weights_name):
    logger.info('create_torchxrayvision_densenet121_feature_extractor()')
    model = xrv.models.DenseNet(weights=weights_name)
    return model

def create_torchxrayvision_resnet_feature_extractor(weights_name):
    logger.info('create_torchxrayvision_resnet_feature_extractor()')
    model = xrv.models.ResNet(weights=weights_name)
    model.forward = _torchxrayvision_resnet_modified_forward.__get__(model) # HACK: monkey patching
    return model

def create_torchxrayvision_resnet_autoencoder_feature_extractor(weights_name):
    logger.info('create_torchxrayvision_resnet_autoencoder_feature_extractor()')
    model = xrv.autoencoders.ResNetAE(weights=weights_name)
    return model

def _torchxrayvision_resnet_modified_forward(self, x):
    # x = fix_resolution(x, 512, self)
    # warn_normalization(x)

    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)

    x = self.model.layer1(x)
    x = self.model.layer2(x)
    x = self.model.layer3(x)
    x = self.model.layer4(x)
    local_features = x

    x = self.model.avgpool(x)
    x = torch.flatten(x, 1)
    global_features = x
    
    return global_features, local_features

_CLIP_VIT_VERSIONS = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
_CLIP_RESNET_VERSIONS = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']
_HUGGINGFACE_CLIP_VIT_VERSIONS = [
    'CenIA/clip-vit-bio-clinical-bert-finetuned',
    'CenIA/clip-vit-bio-clinical-bert-finetuned-frozen-text',
    'CenIA/vte-vit-large-patch16-bio-clinical-bert-finetuned',
    'CenIA/vte-vit-base-patch16-bio-clinical-bert-finetuned',
    'CenIA/vte-vit-large-patch16-bio-clinical-bert-finetuned-v3',
    'CenIA/vte-vit-large-patch16-bio-clinical-bert-finetuned-v2',
    'CenIA/vte-vit-base-patch16-bio-clinical-bert-finetuned-v3',
    'CenIA/vte-vit-base-patch16-bio-clinical-bert-finetuned-v2',
]
_HUGGINGFACE_VITMODEL_VERSIONS = [
    'facebook/vit-mae-base',
    'facebook/vit-mae-large',
    'CenIA/vit-mae-base-finetuned-mimic',
    'CenIA/vit-mae-large-finetuned-mimic',
]
_HUGGINGFACE_RAD_DINO_VERSIONS = [
    'microsoft/rad-dino',
    'microsoft/rad-dino-maira-2',
]
_HUGGINGFACE_CXRMATE_RRG24_UNIFORMER_VERSIONS = [
    'aehrc/cxrmate-rrg24',
]
_HUGGINGFACE_UNIFORMER_BASE_TL_384_VERSIONS = [
    'aehrc/uniformer_base_tl_384',
]
_HUGGINGFACE_SIGLIP_VERSIONS = [
    'google/siglip-base-patch16-224',
    'google/siglip-so400m-patch14-384',
]

CLIP_DEFAULT_IMAGE_MEAN_STD = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
CLIP_VERSION_2_IMAGE_MEAN_STD = {}
_tmp = _CLIP_VIT_VERSIONS + _CLIP_RESNET_VERSIONS
_tmp.append('CenIA/clip-vit-bio-clinical-bert-finetuned')
_tmp.append('CenIA/clip-vit-bio-clinical-bert-finetuned-frozen-text')
for _k in _tmp:
    CLIP_VERSION_2_IMAGE_MEAN_STD[_k] = CLIP_DEFAULT_IMAGE_MEAN_STD
for _k in _HUGGINGFACE_CLIP_VIT_VERSIONS:
    if _k not in _tmp:
        CLIP_VERSION_2_IMAGE_MEAN_STD[_k] = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

HUGGINGFACE_CLIP_VIT_NAMES_2_SHORT = {
    'CenIA/clip-vit-bio-clinical-bert-finetuned': 'CenIA/clip-vit-bcbf',
    'CenIA/clip-vit-bio-clinical-bert-finetuned-frozen-text': 'CenIA/clip-vit-bcbfft',
    'CenIA/vte-vit-large-patch16-bio-clinical-bert-finetuned': 'CenIA/clip-vte-vit-lp16bcbf',
    'CenIA/vte-vit-base-patch16-bio-clinical-bert-finetuned': 'CenIA/clip-vte-vit-bp16bcbf',
    'CenIA/vte-vit-large-patch16-bio-clinical-bert-finetuned-v3': 'CenIA/clip-vte-vit-lp16bcbf-v3',
    'CenIA/vte-vit-large-patch16-bio-clinical-bert-finetuned-v2': 'CenIA/clip-vte-vit-lp16bcbf-v2',
    'CenIA/vte-vit-base-patch16-bio-clinical-bert-finetuned-v3': 'CenIA/clip-vte-vit-bp16bcbf-v3',
    'CenIA/vte-vit-base-patch16-bio-clinical-bert-finetuned-v2': 'CenIA/clip-vte-vit-bp16bcbf-v2',
}

HUGGINGFACE_VITMODEL_NAMES_2_SHORT = {
    'facebook/vit-mae-base': 'facebook/vit-mae-base',
    'facebook/vit-mae-large': 'facebook/vit-mae-large',
    'CenIA/vit-mae-base-finetuned-mimic': 'CenIA/vit-mae-base-ft-mimic',
    'CenIA/vit-mae-large-finetuned-mimic': 'CenIA/vit-mae-large-ft-mimic',
}

HUGGINGFACE_CONVNEXTMODEL_NAMES_2_SHORT = {
    'facebook/convnext-small-224': 'facebook/convnext-small-224',
}

HUGGINGFACE_RAD_DINO_NAMES_2_SHORT = {
    'microsoft/rad-dino': 'microsoft/rad-dino',
    'microsoft/rad-dino-maira-2': 'microsoft/rad-dino-maira-2',
}

HUGGINGFACE_CXRMATE_RRG24_UNIFORMER_NAMES_2_SHORT = {
    'aehrc/cxrmate-rrg24': 'aehrc/cxrmate-rrg24-uniformer',
}

HUGGINGFACE_UNIFORMER_BASE_TL_384_NAMES_2_SHORT = {
    'aehrc/uniformer_base_tl_384': 'aehrc/uniformer-base-tl-384',
}

HUGGINGFACE_SIGLIP_NAMES_2_SHORT = {
    'google/siglip-base-patch16-224': 'google/siglip-base-p16-224',
    'google/siglip-so400m-patch14-384': 'google/siglip-so400m-p14-384',
}

HUGGINGFACE_MEDSAM_NAMES_2_SHORT = {
    'wanglab/medsam-vit-base': 'wanglab/medsam-vit-base',
}

DETECTRON2_YAML_2_SHORT = {
    'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml': 'D2-CocoDet-faster-rcnn-R-50-FPN-3x',
    'COCO-Detection/retinanet_R_50_FPN_1x.yaml': 'D2-CocoDet-retinanet-R-50-FPN-1x',
}

DETECTRON2_HAS_RPN = {
    'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml': True,
    'COCO-Detection/retinanet_R_50_FPN_1x.yaml': False,
}

def _get_clip_vit_modified_forward(dtype):    
    def forward(self, x: torch.Tensor, return_local_features=False):
        x = x.type(dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        if return_local_features:
            global_features = self.ln_post(x[:, 0, :])        
            if self.proj is not None:
                global_features = global_features @ self.proj
            local_features = x[:, 1:, :]
            return global_features, local_features
        else:
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj        
            return x
    return forward

def _clip_resnet_modified_forward(self, x, return_local_features=False):
    x = x.type(self.conv1.weight.dtype)
    x = self.relu1(self.bn1(self.conv1(x)))
    x = self.relu2(self.bn2(self.conv2(x)))
    x = self.relu3(self.bn3(self.conv3(x)))
    x = self.avgpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x_pooled = self.attnpool(x)
    if return_local_features:
        return x_pooled, x
    return x_pooled

CLIP_VIT_GLOBAL_FEAT_SIZE = 512
CLIP_VIT_LOCAL_FEAT_SIZE = 768
CLIP_RESNET_GLOBAL_FEAT_SIZE = 1024
HUGGINGFACE_CLIP_VIT_GLOBAL_FEAT_SIZE = 768
HUGGINGFACE_CLIP_VIT_LARGE_GLOBAL_FEAT_SIZE = 1024
HUGGINGFACE_VITMODEL_GLOBAL_FEAT_SIZE = 768
HUGGINGFACE_VITMODEL_LARGE_GLOBAL_FEAT_SIZE = 1024
HUGGINGFACE_CONVNEXTMODEL_GLOBAL_FEAT_SIZE = 768
HUGGINGFACE_RAD_DINO_GLOBAL_FEAT_SIZE = 768
HUGGINGFACE_SIGLIP_GLOBAL_FEAT_SIZE = {
    'google/siglip-base-patch16-224': 768,
    'google/siglip-so400m-patch14-384': 1152,
}


HUGGINGFACE_VITMODEL_UNFROZEN_PARAM_NAMES_REGEX = re.compile(r'\bpooler\b')

def _load_pretrained_model_state_dict(model, pretrained_weights_path, key_adaptation_fn=None):
    data = torch.load(pretrained_weights_path)
    if 'model' in data: data = data['model']
    if 'state_dict' in data: data = data['state_dict']
    if key_adaptation_fn:
        data = {key_adaptation_fn(k): v for k, v in data.items()}
    load_model_state_dict(model, data)
    logger.info(f'Pre-trained weights successfully loaded from {pretrained_weights_path}')

def create_clip_vit_feature_extractor(clip_vit_version, pretrained_weights_path):
    import clip
    assert clip_vit_version in _CLIP_VIT_VERSIONS, f'Unknown CLIP ViT version {clip_vit_version}'
    model, _ = clip.load(clip_vit_version)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    vit = model.visual.float()
    vit.forward = _get_clip_vit_modified_forward(model.dtype).__get__(vit) # HACK
    return vit

def create_clip_resnet_feature_extractor(clip_resnet_version, pretrained_weights_path):
    import clip
    assert clip_resnet_version in _CLIP_RESNET_VERSIONS, f'Unknown CLIP ResNet version {clip_resnet_version}'
    model, _ = clip.load(clip_resnet_version)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    resnet = model.visual.float()
    resnet.forward = _clip_resnet_modified_forward.__get__(resnet) # HACK
    return resnet

def create_huggingface_clip_vit_feature_extractor(clip_vit_version, pretrained_weights_path):
    from transformers import AutoModel
    assert clip_vit_version in _HUGGINGFACE_CLIP_VIT_VERSIONS, f'Unknown Huggingface CLIP ViT version {clip_vit_version}'
    model = AutoModel.from_pretrained(clip_vit_version, use_auth_token=True)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    vit = model.vision_model.float()
    return vit

def create_huggingface_vitmodel_feature_extractor(vitmodel_version, pretrained_weights_path):
    from transformers import ViTModel
    assert vitmodel_version in _HUGGINGFACE_VITMODEL_VERSIONS, f'Unknown Huggingface ViTModel version {vitmodel_version}'
    model = ViTModel.from_pretrained(vitmodel_version, use_auth_token=True)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    return model

def create_huggingface_convnextmodel_feature_extractor(convnextmodel_version, pretrained_weights_path):
    from transformers import ConvNextModel
    model = ConvNextModel.from_pretrained(convnextmodel_version)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    return model

def create_huggingface_rad_dino_feature_extractor(rad_dino_version, pretrained_weights_path):
    from transformers import AutoModel
    assert rad_dino_version in _HUGGINGFACE_RAD_DINO_VERSIONS, f'Unknown Huggingface RadDINO version {rad_dino_version}' 
    model = AutoModel.from_pretrained(rad_dino_version)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    return model

def create_huggingface_cxrmate_rrg24_uniformer_feature_extractor(version, pretrained_weights_path):
    from transformers import AutoModel
    assert version in _HUGGINGFACE_CXRMATE_RRG24_UNIFORMER_VERSIONS, f'Unknown Huggingface CxRMate RRG24 Uniformer version {version}'
    model = AutoModel.from_pretrained(version, trust_remote_code=True)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    model = model.encoder.uniformer # HACK to get the uniformer from the model
    return model

def create_uniformer_base_tl_384_feature_extractor(version, pretrained_weights_path):
    from transformers import AutoModel
    assert version in _HUGGINGFACE_UNIFORMER_BASE_TL_384_VERSIONS, f'Unknown Huggingface Uniformer Base TL 384 version {version}'
    model = AutoModel.from_pretrained(version, trust_remote_code=True)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    return model

def create_huggingface_siglip_feature_extractor(version, pretrained_weights_path):
    from transformers import AutoModel
    assert version in _HUGGINGFACE_SIGLIP_VERSIONS, f'Unknown Huggingface SigLIP version {version}'
    model = AutoModel.from_pretrained(version)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    model = model.vision_model # HACK to get the vision model from the model
    return model

def create_huggingface_medsam_feature_extractor(version, pretrained_weights_path):
    from transformers import SamModel
    model = SamModel.from_pretrained(version)
    model = model.vision_encoder # HACK to get the vision encoder from the model
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    return model

def create_detectron2_model(
        model_yaml, num_classes=None,
        roi_heads_batch_size_per_image=None,
        rpn_batch_size_per_image=None,
        load_model_zoo_weights=True,
        verbose=False,
    ):
    from detectron2.modeling import build_model as build_detectron2_model
    from detectron2.config import get_cfg as get_detectron2_cfg
    from detectron2 import model_zoo as detectron2_model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    assert model_yaml is not None
    assert model_yaml.endswith('.yaml')
    cfg = get_detectron2_cfg()
    cfg.merge_from_file(detectron2_model_zoo.get_config_file(model_yaml))
    # Relevant reading on how to update the config:
    # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
    logger.info('Building Detectron2 model')
    if num_classes is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        logger.info(f'cfg.MODEL.ROI_HEADS.NUM_CLASSES overriden to {num_classes}')
        logger.info(f'cfg.MODEL.RETINANET.NUM_CLASSES overriden to {num_classes}')
    if roi_heads_batch_size_per_image is not None:
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_heads_batch_size_per_image
        logger.info(f'cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE overriden to {roi_heads_batch_size_per_image}')
    if rpn_batch_size_per_image is not None:
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = rpn_batch_size_per_image
        logger.info(f'cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE overriden to {rpn_batch_size_per_image}')
    if verbose:
        logger.info(cfg)
        
    model = build_detectron2_model(cfg)
    logger.info('Detectron2 model successfully built')
    # Load weights from the official Detectron2 model zoo
    if load_model_zoo_weights:
        checkpoint_url = detectron2_model_zoo.get_checkpoint_url(model_yaml)
        logger.info(f'Loading weights from {checkpoint_url}')
        DetectionCheckpointer(model).load(checkpoint_url)
        logger.info('Weights successfully loaded')
    return model

class YOLOv8DetectionAndFeatureExtractorModel(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True, detection_layers=None):
        
        super().__init__(cfg, ch, nc, verbose)
        
        if detection_layers is not None:
            assert type(detection_layers) == list
            assert len(detection_layers) > 0
            # Create a module list containing only the detection layers
            self.detection_layers = nn.ModuleList(detection_layers)
            self.using_multiple_detection_layers = True
            # Remove the original detection layer (last layer) from the model
            # NOTE: self.model is a nn.Sequential object
            self.model = self.model[:-1]
        else:
            self.detection_layers = None
            self.using_multiple_detection_layers = False
    
    def custom_forward(self, x, detection_layer_index=None, detection_layer_indexes=None, only_return_features=False):
        """
        This is a modified version of the original _forward_once() method in BaseModel,
        found in ultralytics/nn/tasks.py.
        The original method returns only the detection output, while this method returns
        both the detection output and the features extracted by the last convolutional layer.
        We also added the option to return the detection output of multiple detection layers,
        each one with a different number of classes. This can be useful for example when
        training a model with multiple datasets, each one with a different number of classes.
        """

        if only_return_features:
            y = []
            features = None
            for m in self.model:
                # print('----')
                # print(m)
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if torch.is_tensor(x):
                    features = x # keep the last tensor as features
                x = m(x)  # run
                if torch.is_tensor(x):
                    features = x # keep the last tensor as features
                y.append(x if m.i in self.save else None)  # save output
            if torch.is_tensor(x):
                features = x # keep the last tensor as features
            return features # return features
        
        if self.using_multiple_detection_layers:
            assert detection_layer_index is not None or detection_layer_indexes is not None
            if detection_layer_index is not None:
                assert type(detection_layer_index) == int
                assert detection_layer_index >= 0 and detection_layer_index < len(self.detection_layers)
                detection_layer_indexes = [detection_layer_index]
                return_list = False
            else:
                assert type(detection_layer_indexes) == list
                assert len(detection_layer_indexes) > 0
                assert all([type(i) == int for i in detection_layer_indexes])
                assert all([i >= 0 and i < len(self.detection_layers) for i in detection_layer_indexes])
                return_list = True
            
            # Run the model up to the last convolutional layer
            count = 0
            debug_list = []
            y = []
            features = None
            for m in self.model:
                # print(f'---- {count}')
                # print(m)
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if torch.is_tensor(x):
                    features = x # keep the last tensor as features
                    # print(f'features.shape: {features.shape}')
                    # print(f'features: {features}')
                    debug_list.append((count, features.shape, features.view(-1)[0].item(), features.sum().item()))
                x = m(x) # run
                if torch.is_tensor(x):
                    features = x # keep the last tensor as features
                    # print(f'features.shape: {features.shape}')
                    # print(f'features: {features}')
                    debug_list.append((count, features.shape, features.view(-1)[0].item(), features.sum().item()))
                y.append(x if m.i in self.save else None)  # save output
                count += 1
            if torch.is_tensor(x):
                features = x # keep the last tensor as features
                # print(f'features.shape: {features.shape}')
                # print(f'features: {features}')
                debug_list.append((count, features.shape, features.view(-1)[0].item(), features.sum().item()))
            print('debug_list')
            for row in debug_list:
                print(row)
            # Run the detection layers
            detection_output = []
            xx = x
            for i in detection_layer_indexes:
                m = self.detection_layers[i]
                # print('$$$$$$$$$$$$$$$$$$$$$$$')
                # print(m)
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [xx if j == -1 else y[j] for j in m.f]  # from earlier layers
                else:
                    x = xx
                x = m(x) # run
                detection_output.append(x)
            if not return_list:
                detection_output = detection_output[0]
            return features, detection_output # return features and detection output
        else:
            y = []
            features = None
            for m in self.model:
                # print('----')
                # print(m)
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if torch.is_tensor(x):
                    features = x # keep the last tensor as features
                x = m(x)  # run
                if torch.is_tensor(x):
                    features = x # keep the last tensor as features
                y.append(x if m.i in self.save else None)  # save output
            if torch.is_tensor(x):
                features = x # keep the last tensor as features
            return features, x # return features and detection output

def create_yolov8_model(model_name_or_path, nc, class_names):
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.yolo.cfg import get_cfg
    ckpt = None
    if str(model_name_or_path).endswith('.pt'):
        weights, ckpt = attempt_load_one_weight(model_name_or_path)
        cfg = ckpt['model'].yaml
    else:
        cfg = model_name_or_path
    model = YOLOv8DetectionAndFeatureExtractorModel(cfg, nc=nc, verbose=True)
    if weights:
        model.load(weights)
    model.nc = nc
    model.names = class_names  # attach class names to model
    args = get_cfg(overrides={'model': model_name_or_path})
    model.args = args  # attach hyperparameters to model
    return model

def create_yolov8_model_for_multiple_datasets(model_name_or_path, nc_list, class_names_list, verbose=True):
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.yolo.cfg import get_cfg
    import time
    
    assert type(nc_list) == list
    assert type(class_names_list) == list
    assert len(nc_list) == len(class_names_list)

    logger.info('Creating YOLOv8 model for multiple datasets')
    
    # Load weights and config
    ckpt = None
    if str(model_name_or_path).endswith('.pt'):
        weights, ckpt = attempt_load_one_weight(model_name_or_path)
        cfg = ckpt['model'].yaml
    else:
        cfg = model_name_or_path

    # Create one DetectionModel for each dataset and extract its last Detection layer
    detection_layers = []
    for i, nc in enumerate(nc_list):
        # (cfg, ch, nc, verbose)
        if verbose:
            logger.info(f'   {i+1}. Creating DetectionModel for {nc} classes')
            time.sleep(0.2) # to avoid colliding prints
        # clone cfg
        cfg_ = copy.deepcopy(cfg)
        model = DetectionModel(cfg_, ch=3, nc=nc, verbose=verbose)
        if weights:
            model.load(weights)
        detection_layers.append(model.model[-1])

    # Create a new model with the detection layers from the previous models
    if verbose:
        logger.info(f'   Creating final YOLOv8 model with {len(detection_layers)} detection layers')
        time.sleep(0.2) # to avoid colliding prints
    model = YOLOv8DetectionAndFeatureExtractorModel(cfg, nc=nc_list[0], verbose=verbose, detection_layers=detection_layers)
    if weights:
        model.load(weights)

    model.nc = nc_list[0]
    model.names = class_names_list[0]  # attach class names to model
    args = get_cfg(overrides={'model': model_name_or_path})
    model.args = args  # attach hyperparameters to model
    return model

def create_yolov11_model_for_det_mlc(
        predict_bboxes_chest_imagenome,
        predict_bboxes_vinbig,
        predict_labels_vinbig,
        use_vinbig_with_modified_labels,
        # TODO: support more tasks
        query_embed_size,
        mlp_hidden_dims,
        local_attention_hidden_size,
        image_size,
        model_name_or_path,
        model_alias,
        device,
):
    assert device is not None

    from medvqa.models.vision.yolov11_modified import (
        YOLOv11MultiClassifierDetector,
        ClassificationTaskDescriptor,
        DetectionTaskDescriptor,
    )

    classification_tasks = []
    if predict_labels_vinbig:
        classification_tasks.append(
            ClassificationTaskDescriptor(
                task_name='vinbig',
                label_names=VINBIG_LABELS if not use_vinbig_with_modified_labels else VINBIG_LABELS__MODIFIED,
                class_names=['absent', 'present'],
            )
        )

    detection_tasks = []
    if predict_bboxes_chest_imagenome:
        detection_tasks.append(
            DetectionTaskDescriptor(
                task_name='cig',
                class_names=CHEST_IMAGENOME_BBOX_NAMES,
            )
        )
    if predict_bboxes_vinbig:
        detection_tasks.append(
            DetectionTaskDescriptor(
                task_name='vinbig',
                class_names=VINBIG_BBOX_NAMES if not use_vinbig_with_modified_labels else VINBIG_BBOX_NAMES__MODIFIED,
            )
        )

    model = YOLOv11MultiClassifierDetector(
        classification_tasks=classification_tasks,
        detection_tasks=detection_tasks,
        query_embed_size=query_embed_size,
        mlp_hidden_dims=mlp_hidden_dims,
        local_attention_hidden_size=local_attention_hidden_size,
        image_size=image_size,
        model_name_or_path=model_name_or_path,
        alias=model_alias,
        device=device,
    )

    return model

def create_yolov11_fact_conditioned(
        fact_embed_size,
        mlp_hidden_dims,
        local_attention_hidden_size,
        image_size,
        model_name_or_path,
        yolo_alias,
        device,
        yolov11_pretrained_weights_path=None,
):
    from medvqa.models.vision.yolov11_modified import YOLOv11FactConditionedClassifierDetector
    model = YOLOv11FactConditionedClassifierDetector(
        fact_embed_size=fact_embed_size,
        mlp_hidden_dims=mlp_hidden_dims,
        local_attention_hidden_size=local_attention_hidden_size,
        image_size=image_size,
        model_name_or_path=model_name_or_path,
        yolo_alias=yolo_alias,
        device=device,
    )
    if yolov11_pretrained_weights_path:
        def _key_adaptation_fn(k):
            if k.startswith('raw_image_encoder.'):
                return k[len('raw_image_encoder.'):]
            return k
        _load_pretrained_model_state_dict(model, yolov11_pretrained_weights_path, key_adaptation_fn=_key_adaptation_fn)
    return model

def create_yolov11_feature_extractor(
        model_name_or_path,
        yolo_alias,
        yolov11_pretrained_weights_path=None,
):
    from medvqa.models.vision.yolov11_modified import YOLOv11FeatureExtractor
    model = YOLOv11FeatureExtractor(
        model_name_or_path=model_name_or_path,
        yolo_alias=yolo_alias,
    )
    if yolov11_pretrained_weights_path:
        def _key_adaptation_fn(k):
            if k.startswith('raw_image_encoder.'):
                return k[len('raw_image_encoder.'):]
            return k
        _load_pretrained_model_state_dict(model, yolov11_pretrained_weights_path, key_adaptation_fn=_key_adaptation_fn)
    return model