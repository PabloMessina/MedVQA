import torch
import torch.nn as nn
import torchvision.models as models
import clip
from medvqa.datasets.mimiccxr import MIMICCXR_IMAGE_ORIENTATIONS
from medvqa.datasets.iuxray import IUXRAY_IMAGE_ORIENTATIONS
from medvqa.models.common import freeze_parameters, load_model_state_dict
from medvqa.utils.constants import CHEXPERT_LABELS, CHEXPERT_GENDERS, CHEXPERT_ORIENTATIONS
from transformers import AutoModel, ViTModel
import re

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
                n_medical_tags=None,
                n_questions_aux_task=None,
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
    model = ViTModel.from_pretrained(vitmodel_version)
    if pretrained_weights_path: _load_pretrained_model_state_dict(model, pretrained_weights_path)
    return model