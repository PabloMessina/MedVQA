import torch
import torch.nn as nn
import torchvision.models as models
from medvqa.datasets.mimiccxr import MIMICCXR_IMAGE_ORIENTATIONS
from medvqa.datasets.iuxray import IUXRAY_IMAGE_ORIENTATIONS
from medvqa.models.common import freeze_parameters
from medvqa.utils.constants import CHEXPERT_LABELS, CHEXPERT_GENDERS, CHEXPERT_ORIENTATIONS

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
                freeze_cnn=False,
                **unused_kwargs,
        ):
        super().__init__()
        self.name = 'densenet121'
        if pretrained:
            if densenet_pretrained_weights_path:
                densenet = models.densenet121(pretrained=False)
                pretrained_weights = torch.load(densenet_pretrained_weights_path, map_location='cuda')
                densenet.load_state_dict(pretrained_weights, strict=False)
                print("DenseNet121's pretrained weights loaded from", densenet_pretrained_weights_path)
            else:
                densenet = models.densenet121(pretrained=True)
                print("DenseNet121's pretrained weights loaded from ImageNet")
        else:
            densenet = models.densenet121(pretrained=False)

        self.image_encoder = densenet.features

        if freeze_cnn: freeze_parameters(self.image_encoder)

        # Optional auxiliary tasks        
        
        # 1) medical tags prediction
        if classify_tags:
            self.W_tags = nn.Linear(image_local_feat_size * 2, n_medical_tags)
            self.tags_aux_task = True
        else:
            self.tags_aux_task = False
        
        # 2) orientation classifiction
        if classify_orientation:
            self.W_ori_mimiccxr = nn.Linear(image_local_feat_size * 2, len(MIMICCXR_IMAGE_ORIENTATIONS))
            self.W_ori_iuxray = nn.Linear(image_local_feat_size * 2, len(IUXRAY_IMAGE_ORIENTATIONS))
            self.orien_aux_task = True
        else:
            self.orien_aux_task = False

        # 3) chexpert classifiction
        if classify_chexpert:
            self.W_chx = nn.Linear(image_local_feat_size * 2, len(CHEXPERT_LABELS))
            self.chx_aux_task = True
        else:
            self.chx_aux_task = False

        # 4) questions classification
        if classify_questions:
            self.W_q = nn.Linear(image_local_feat_size * 2, n_questions_aux_task)
            self.q_aux_task = True
        else:
            self.q_aux_task = False

        if use_chexpert_forward:
            self.W_gender_chexpert = nn.Linear(image_local_feat_size * 2, len(CHEXPERT_GENDERS))
            self.W_ori_chexpert = nn.Linear(image_local_feat_size * 2, len(CHEXPERT_ORIENTATIONS))

    def forward(self, images, iuxray_foward=False, mimiccxr_foward=False, chexpert_forward=False):
        
        # local features
        batch_size = images.size(0)
        local_feat = self.image_encoder(images)
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
                output['pred_chexpert'] = self.W_chx(global_feat)
                output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
            if self.q_aux_task:
                output['pred_qlabels'] = self.W_q(global_feat)
        
        return output