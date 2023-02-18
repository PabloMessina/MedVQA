import torch
import torch.nn as nn
import torchvision.models as models
import torchxrayvision as xrv
import clip
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_NUM_BBOX_CLASSES
from medvqa.datasets.mimiccxr import MIMICCXR_IMAGE_ORIENTATIONS
from medvqa.datasets.iuxray import IUXRAY_IMAGE_ORIENTATIONS
from medvqa.models.common import freeze_parameters, load_model_state_dict
from medvqa.models.mlp import MLP
from medvqa.models.vision.bbox_regression import (
    BBoxRegressorVersion,
    BoundingBoxRegressor_v1,
    BoundingBoxRegressor_v2,
    BoundingBoxRegressor_v3,
)
from medvqa.utils.constants import (
    CHEXPERT_LABELS,
    CHEXPERT_GENDERS,
    CHEXPERT_ORIENTATIONS,
    CXR14_LABELS,
    PADCHEST_NUM_LABELS,
    PADCHEST_NUM_LOCALIZATIONS,
    PADCHEST_PROJECTIONS,
    VINBIG_DISEASES,
)
from transformers import AutoModel, ViTModel
import re

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

class VisualInputMode:
    RAW_IMAGE = 'raw-image'
    PRECOMP_FEAT = 'precomp-feat' # precomputed visual features
    HYBRID = 'hybrid'

def does_include_image(visual_input_mode):
    return visual_input_mode in (VisualInputMode.RAW_IMAGE, VisualInputMode.HYBRID)

def does_include_visual_features(visual_input_mode):
    return visual_input_mode in (VisualInputMode.PRECOMP_FEAT, VisualInputMode.HYBRID)


class MultiPurposeVisualModule(nn.Module):

    def __init__(self,
                # Image Encoder kwargs
                visual_input_mode=VisualInputMode.RAW_IMAGE,
                raw_image_encoding=RawImageEncoding.DENSENET_121,
                image_local_feat_size=None,
                freeze_image_encoder=False,
                image_encoder_pretrained_weights_path=None,
                imagenet_pretrained=True,
                mlp_in_dim=None,
                mlp_out_dim=None,
                mlp_hidden_dims=None,
                clip_version=None,
                num_regions=None,
                huggingface_model_name=None,
                torchxrayvision_weights_name=None,
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
                predict_bboxes_chest_imagenome=False,
                chest_imagenome_train_average_bbox_coords=None,
                n_medical_tags=None,
                n_questions_aux_task=None,
                n_chest_imagenome_labels=None,
                chest_imagenome_bbox_hidden_size=None,
                chest_imagenome_bbox_regressor_version=None,
                merge_findings=False,
                n_findings=None,
                # Other kwargs
                **unused_kwargs,
                ): 
        super().__init__()
        
        self.visual_input_mode = visual_input_mode
        self.clip_version = clip_version
        self.huggingface_model_name = huggingface_model_name
        self.torchxrayvision_weights_name = torchxrayvision_weights_name
        if raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE:
            assert huggingface_model_name is not None
        if raw_image_encoding in [
            RawImageEncoding.DENSENET_121__TORCHXRAYVISION,
            RawImageEncoding.RESNET__TORCHXRAYVISION,
            RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION,
        ]:
            assert torchxrayvision_weights_name is not None
        
        # Init visual backbone
        self._init_visual_backbone(
            visual_input_mode=visual_input_mode,
            raw_image_encoding=raw_image_encoding,
            image_encoder_pretrained_weights_path=image_encoder_pretrained_weights_path,
            imagenet_pretrained=imagenet_pretrained,
            image_local_feat_size=image_local_feat_size,
            mlp_in_dim=mlp_in_dim,
            mlp_out_dim=mlp_out_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            freeze_image_encoder=freeze_image_encoder,
        )
            
        # Init auxiliary tasks
        self._init_auxiliary_tasks(
            use_mimiccxr=use_mimiccxr,
            use_iuxray=use_iuxray,
            use_chexpert=use_chexpert,
            use_cxr14=use_cxr14,
            use_vinbig=use_vinbig,
            use_padchest=use_padchest,
            classify_tags=classify_tags,
            classify_orientation=classify_orientation,
            classify_gender=classify_gender,
            classify_chexpert=classify_chexpert,
            classify_questions=classify_questions,
            classify_chest_imagenome=classify_chest_imagenome,
            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
            n_medical_tags=n_medical_tags,
            n_questions_aux_task=n_questions_aux_task,
            n_chest_imagenome_labels=n_chest_imagenome_labels,
            chest_imagenome_bbox_hidden_size=chest_imagenome_bbox_hidden_size,
            chest_imagenome_train_average_bbox_coords=chest_imagenome_train_average_bbox_coords,
            chest_imagenome_bbox_regressor_version=chest_imagenome_bbox_regressor_version,
            num_regions=num_regions,
            merge_findings=merge_findings,
            n_findings=n_findings,
        )

        print(f'MultiPurposeVisualModule: self.name={self.name}')

    def _init_visual_backbone(self, visual_input_mode, raw_image_encoding, image_encoder_pretrained_weights_path,
                            imagenet_pretrained, image_local_feat_size, mlp_in_dim, mlp_out_dim, mlp_hidden_dims,
                            freeze_image_encoder):
        global_feat_size = 0
        
        if does_include_image(visual_input_mode):
            if self.clip_version is not None:
                model_name = self.clip_version
            elif self.huggingface_model_name is not None:
                model_name = self.huggingface_model_name
            elif self.torchxrayvision_weights_name is not None:
                model_name = self.torchxrayvision_weights_name
            else:
                model_name = None
            self._init_raw_image_encoder(raw_image_encoding, image_encoder_pretrained_weights_path,
                                         imagenet_pretrained, model_name, freeze_image_encoder)
            global_feat_size += self._get_raw_image_encoder_global_feat_size(image_local_feat_size)
        
        if does_include_visual_features(visual_input_mode):
            self._init_mlp_visual_feat_encoder(mlp_in_dim, mlp_out_dim, mlp_hidden_dims, freeze_image_encoder)
            global_feat_size += mlp_out_dim
        
        assert global_feat_size > 0
        self.local_feat_size = image_local_feat_size
        self.global_feat_size = global_feat_size
        print('  self.global_feat_size =', self.global_feat_size)

    def _get_raw_image_encoder_global_feat_size(self, image_local_feat_size):
        if self.raw_image_encoding in [
            RawImageEncoding.DENSENET_121,
            RawImageEncoding.DENSENET_121__TORCHXRAYVISION,
            RawImageEncoding.RESNET__TORCHXRAYVISION,
            RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION,
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
        raise ValueError(f'Unknown raw_image_encoding: {self.raw_image_encoding}')
    
    def _init_raw_image_encoder(self, raw_image_encoding, pretrained_weights_path,
                                imagenet_pretrained, model_name, freeze_image_encoder):
        self.raw_image_encoding = raw_image_encoding
        ignore_name_regex = None
        if raw_image_encoding == RawImageEncoding.DENSENET_121:
            self.raw_image_encoder = create_densenet121_feature_extractor(pretrained_weights_path, imagenet_pretrained)
        elif raw_image_encoding == RawImageEncoding.DENSENET_121__TORCHXRAYVISION:
            self.raw_image_encoder = create_torchxrayvision_densenet121_feature_extractor(model_name)
        elif raw_image_encoding == RawImageEncoding.RESNET__TORCHXRAYVISION:
            self.raw_image_encoder = create_torchxrayvision_resnet_feature_extractor(model_name)
        elif raw_image_encoding == RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION:
            self.raw_image_encoder = create_torchxrayvision_resnet_autoencoder_feature_extractor(model_name)
        elif raw_image_encoding == RawImageEncoding.CLIP_RESNET:
            self.raw_image_encoder = create_clip_resnet_feature_extractor(model_name, pretrained_weights_path)
        elif raw_image_encoding == RawImageEncoding.CLIP_VIT:
            self.raw_image_encoder = create_clip_vit_feature_extractor(model_name, pretrained_weights_path)
        elif raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE or \
                raw_image_encoding == RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE:
            self.raw_image_encoder = create_huggingface_clip_vit_feature_extractor(model_name, pretrained_weights_path)
        elif raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE or \
                raw_image_encoding == RawImageEncoding.VITMODEL_LARGE__HUGGINGFACE:
            self.raw_image_encoder = create_huggingface_vitmodel_feature_extractor(model_name, pretrained_weights_path)
            ignore_name_regex = HUGGINGFACE_VITMODEL_UNFROZEN_PARAM_NAMES_REGEX
        else: raise ValueError(f'Unknown raw_image_encoding: {raw_image_encoding}')
        if freeze_image_encoder: freeze_parameters(self.raw_image_encoder, ignore_name_regex)

    def _init_mlp_visual_feat_encoder(self, mlp_in_dim, mlp_out_dim, mlp_hidden_dims, freeze_image_encoder):
        self.mlp_vf_encoder = MLP(in_dim=mlp_in_dim, out_dim=mlp_out_dim, hidden_dims=mlp_hidden_dims)
        if freeze_image_encoder: freeze_parameters(self.mlp_vf_encoder)

    def _init_auxiliary_tasks(self, use_mimiccxr, use_iuxray, use_chexpert, use_cxr14, use_vinbig, use_padchest,
                              classify_tags, classify_orientation, classify_gender, classify_chexpert, classify_questions,
                              classify_chest_imagenome, predict_bboxes_chest_imagenome,
                              n_medical_tags, n_questions_aux_task, n_chest_imagenome_labels,
                              chest_imagenome_bbox_hidden_size, chest_imagenome_train_average_bbox_coords,
                              chest_imagenome_bbox_regressor_version, num_regions=None,
                              merge_findings=False, n_findings=None):
        
        # Optional auxiliary tasks
        
        self.merge_findings = merge_findings
        
        # 1) medical tags classification
        self.classify_tags = classify_tags
        if classify_tags:
            assert n_medical_tags is not None
            self.W_tags = nn.Linear(self.global_feat_size, n_medical_tags)
        
        # 2) orientation classifiction
        self.classify_orientation = classify_orientation
        if classify_orientation:
            if use_mimiccxr:
                self.W_ori_mimiccxr = nn.Linear(self.global_feat_size, len(MIMICCXR_IMAGE_ORIENTATIONS))
            if use_iuxray:
                self.W_ori_iuxray = nn.Linear(self.global_feat_size, len(IUXRAY_IMAGE_ORIENTATIONS))
            if use_chexpert or use_cxr14: # weight sharing among chexpert & CRX14
                self.W_ori_chexpert = nn.Linear(self.global_feat_size, len(CHEXPERT_ORIENTATIONS))

        # 3) questions classification
        self.classify_questions = classify_questions
        if classify_questions:
            self.W_q = nn.Linear(self.global_feat_size, n_questions_aux_task)

        # 4) gender classification
        self.classify_gender = classify_gender
        if classify_gender:
            self.W_gender = nn.Linear(self.global_feat_size, len(CHEXPERT_GENDERS))

        if merge_findings:
            assert n_findings is not None
            self.W_findings = nn.Linear(self.global_feat_size, n_findings)
        else:        
            # 6) chexpert classifiction
            self.classify_chexpert = classify_chexpert
            if classify_chexpert:
                self.W_chx = nn.Linear(self.global_feat_size, len(CHEXPERT_LABELS))

            # 7) CXR14 specific labels
            if use_cxr14:
                self.W_cxr14 = nn.Linear(self.global_feat_size, len(CXR14_LABELS))

            # 8) VinBig specific labels
            if use_vinbig:
                self.W_vinbig = nn.Linear(self.global_feat_size, len(VINBIG_DISEASES))

        # 9) PadChest specific tasks
        if use_padchest:
            self.W_padchest_labels = nn.Linear(self.global_feat_size, PADCHEST_NUM_LABELS)
            self.W_padchest_loc = nn.Linear(self.global_feat_size, PADCHEST_NUM_LOCALIZATIONS)
            self.W_padchest_ori = nn.Linear(self.global_feat_size, len(PADCHEST_PROJECTIONS))

        # 10) Chest ImaGenome specific tasks
        self.classify_chest_imagenome = classify_chest_imagenome
        if classify_chest_imagenome:
            assert n_chest_imagenome_labels is not None
            self.W_chst_imgn = nn.Linear(self.global_feat_size, n_chest_imagenome_labels)
        self.predict_bboxes_chest_imagenome = predict_bboxes_chest_imagenome
        if predict_bboxes_chest_imagenome:
            assert chest_imagenome_bbox_hidden_size is not None
            assert chest_imagenome_bbox_regressor_version is not None
            if chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V1:
                self.bbox_regressor_chst_imgn = BoundingBoxRegressor_v1(
                    local_feat_dim=self.local_feat_size,
                    global_feat_dim=self.global_feat_size,
                    hidden_dim=chest_imagenome_bbox_hidden_size,
                    num_classes=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
                    train_average_bbox_coords=chest_imagenome_train_average_bbox_coords,
                )
            elif chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V2:
                assert num_regions is not None
                self.bbox_regressor_chst_imgn = BoundingBoxRegressor_v2(
                    local_feat_dim=self.local_feat_size,
                    global_feat_dim=self.global_feat_size,
                    hidden_dim=chest_imagenome_bbox_hidden_size,
                    num_classes=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
                    train_average_bbox_coords=chest_imagenome_train_average_bbox_coords,
                    num_regions=num_regions,
                )
            elif chest_imagenome_bbox_regressor_version == BBoxRegressorVersion.V3:
                assert num_regions is not None
                self.bbox_regressor_chst_imgn = BoundingBoxRegressor_v3(
                    local_feat_dim=self.local_feat_size,
                    global_feat_dim=self.global_feat_size,
                    hidden_dim=chest_imagenome_bbox_hidden_size,
                    num_classes=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
                    train_average_bbox_coords=chest_imagenome_train_average_bbox_coords,
                    num_regions=num_regions,                    
                )
            else:
                raise ValueError(f'Unknown bbox regressor version: {chest_imagenome_bbox_regressor_version}')

    @property
    def name(self):
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
        **unused_kwargs,
    ):
        assert (raw_images is not None) or (visual_features is not None)
        local_feat = None
        global_list = []        
        
        if raw_images is not None:

            if self.raw_image_encoding == RawImageEncoding.DENSENET_121:
                # compute local features
                local_feat = self.raw_image_encoder(raw_images)
                batch_size = raw_images.size(0)
                feat_size = local_feat.size(1)
                local_feat = local_feat.permute(0,2,3,1).view(batch_size, -1, feat_size)
                # compute global features
                global_avg_pool = local_feat.mean(1)
                global_max_pool = local_feat.max(1)[0]
                global_list.append(global_avg_pool)
                global_list.append(global_max_pool)

            elif self.raw_image_encoding == RawImageEncoding.DENSENET_121__TORCHXRAYVISION:
                # compute local features
                local_feat = self.raw_image_encoder.features(raw_images)
                batch_size = raw_images.size(0)
                feat_size = local_feat.size(1)
                local_feat = local_feat.permute(0,2,3,1).view(batch_size, -1, feat_size)
                # compute global features
                global_avg_pool = local_feat.mean(1)
                global_max_pool = local_feat.max(1)[0]
                global_list.append(global_avg_pool)
                global_list.append(global_max_pool)
            
            elif self.raw_image_encoding == RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION:
                # compute local features
                local_feat = self.raw_image_encoder.encode(raw_images)
                batch_size = raw_images.size(0)
                feat_size = local_feat.size(1)
                local_feat = local_feat.permute(0,2,3,1).view(batch_size, -1, feat_size)
                # compute global features
                global_avg_pool = local_feat.mean(1)
                global_max_pool = local_feat.max(1)[0]
                global_list.append(global_avg_pool)
                global_list.append(global_max_pool)

            elif self.raw_image_encoding == RawImageEncoding.CLIP_RESNET:
                global_feat, local_feat = self.raw_image_encoder(raw_images, return_local_features=True)
                batch_size = raw_images.size(0)
                feat_size = local_feat.size(1)
                local_feat = local_feat.permute(0,2,3,1).view(batch_size, -1, feat_size)                
                global_list.append(global_feat)
            
            elif self.raw_image_encoding == RawImageEncoding.CLIP_VIT:
                global_feat, local_feat = self.raw_image_encoder(raw_images, return_local_features=True)
                global_list.append(global_feat)
            
            elif self.raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE or \
                    self.raw_image_encoding == RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE or \
                    self.raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE or \
                    self.raw_image_encoding == RawImageEncoding.VITMODEL_LARGE__HUGGINGFACE:
                tmp = self.raw_image_encoder(raw_images)
                global_feat, local_feat = tmp.pooler_output, tmp.last_hidden_state
                global_list.append(global_feat)
            
            else: assert False, f'Unknown raw image encoding {self.raw_image_encoding}'

        if visual_features is not None:
            global_vf  = self.mlp_vf_encoder(visual_features)
            global_list.append(global_vf)

        if len(global_list) > 1:
            global_feat = torch.cat(global_list, 1)
        else:
            global_feat = global_list[0]

        output = {
            'global_feat': global_feat,
            'local_feat': local_feat,
        }

        if self.merge_findings:
            output['pred_findings'] = self.W_findings(global_feat)
            output['pred_findings_probs'] = torch.sigmoid(output['pred_findings'])

        if chexpert_forward:
            if self.classify_orientation:
                output['pred_orientation'] = self.W_ori_chexpert(global_feat)
            if self.classify_gender:
                output['pred_gender'] = self.W_gender(global_feat)
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
            if not self.merge_findings:
                output['pred_vinbig'] = self.W_vinbig(global_feat)
                output['pred_vinbig_probs'] = torch.sigmoid(output['pred_vinbig'])
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
            if not self.merge_findings and self.classify_chexpert:
                output['pred_chexpert'] = self.W_chx(global_feat)
                output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
            if self.classify_chest_imagenome:
                output['pred_chest_imagenome'] = self.W_chst_imgn(global_feat)
                output['pred_chest_imagenome_probs'] = torch.sigmoid(output['pred_chest_imagenome'])
            if self.predict_bboxes_chest_imagenome:
                pred_bbox_coords, pred_bbox_presence = self.bbox_regressor_chst_imgn(local_feat, global_feat)
                output['pred_chest_imagenome_bbox_coords'] = pred_bbox_coords
                output['pred_chest_imagenome_bbox_presence'] = pred_bbox_presence

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
    print('create_densenet121_feature_extractor()')
    print(f'   drop_rate: {drop_rate}')
    # Load pre-trained CNN weights
    if pretrained_weights_path:
        densenet = models.densenet121(pretrained=False, drop_rate=drop_rate)
        pretrained_weights = torch.load(pretrained_weights_path, map_location='cuda')
        densenet.load_state_dict(pretrained_weights, strict=False)
        print("DenseNet121's pretrained weights loaded from", pretrained_weights_path)
    elif imagenet_pretrained:
        densenet = models.densenet121(pretrained=True, drop_rate=drop_rate)
        print("DenseNet121's pretrained weights loaded from ImageNet")
    else:
        densenet = models.densenet121(pretrained=False, drop_rate=drop_rate)
    return densenet.features

def create_torchxrayvision_densenet121_feature_extractor(weights_name):
    print('create_torchxrayvision_densenet121_feature_extractor()')
    model = xrv.models.DenseNet(weights=weights_name)
    return model

def create_torchxrayvision_resnet_feature_extractor(weights_name):
    print('create_torchxrayvision_resnet_feature_extractor()')
    model = xrv.models.ResNet(weights=weights_name)
    model.forward = _torchxrayvision_resnet_modified_forward.__get__(model) # HACK: monkey patching
    return model

def create_torchxrayvision_resnet_autoencoder_feature_extractor(weights_name):
    print('create_torchxrayvision_resnet_autoencoder_feature_extractor()')
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

HUGGINGFACE_VITMODEL_UNFROZEN_PARAM_NAMES_REGEX = re.compile(r'\bpooler\b')

def _load_pretrained_model_state_dict(model, pretrained_weights_path):
    data = torch.load(pretrained_weights_path)
    if 'model' in data: data = data['model']
    if 'state_dict' in data: data = data['state_dict']
    load_model_state_dict(model, data)
    print(f'Pre-trained weights successfully loaded from {pretrained_weights_path}')

def create_clip_vit_feature_extractor(clip_vit_version, pretrained_weights_path):
    assert clip_vit_version in _CLIP_VIT_VERSIONS, f'Unknown CLIP ViT version {clip_vit_version}'
    model, _ = clip.load(clip_vit_version)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    vit = model.visual.float()
    vit.forward = _get_clip_vit_modified_forward(model.dtype).__get__(vit) # HACK
    return vit

def create_clip_resnet_feature_extractor(clip_resnet_version, pretrained_weights_path):
    assert clip_resnet_version in _CLIP_RESNET_VERSIONS, f'Unknown CLIP ResNet version {clip_resnet_version}'
    model, _ = clip.load(clip_resnet_version)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    resnet = model.visual.float()
    resnet.forward = _clip_resnet_modified_forward.__get__(resnet) # HACK
    return resnet

def create_huggingface_clip_vit_feature_extractor(clip_vit_version, pretrained_weights_path):
    assert clip_vit_version in _HUGGINGFACE_CLIP_VIT_VERSIONS, f'Unknown Hugginface CLIP ViT version {clip_vit_version}'
    model = AutoModel.from_pretrained(clip_vit_version, use_auth_token=True)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    vit = model.vision_model.float()
    return vit

def create_huggingface_vitmodel_feature_extractor(vitmodel_version, pretrained_weights_path):
    assert vitmodel_version in _HUGGINGFACE_VITMODEL_VERSIONS, f'Unknown Hugginface ViTModel version {vitmodel_version}'
    model = ViTModel.from_pretrained(vitmodel_version, use_auth_token=True)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    return model