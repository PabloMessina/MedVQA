import torch
from torch import nn
from medvqa.models.common import freeze_parameters

from medvqa.models.vision.visual_modules import (
    CLIP_RESNET_GLOBAL_FEAT_SIZE,
    CLIP_VIT_GLOBAL_FEAT_SIZE,
    HUGGINGFACE_CLIP_VIT_GLOBAL_FEAT_SIZE,
    HUGGINGFACE_CLIP_VIT_VERSIONS_2_SHORT,
    create_clip_resnet_feature_extractor,
    create_clip_vit_feature_extractor,
    create_densenet121_feature_extractor,
    create_huggingface_clip_vit_feature_extractor,
)
from medvqa.datasets.mimiccxr import MIMICCXR_IMAGE_ORIENTATIONS
from medvqa.datasets.iuxray import IUXRAY_IMAGE_ORIENTATIONS
from medvqa.models.nlp.text_encoder import BiLSTMBasedTextEncoder
from medvqa.models.nlp.text_decoder import LSTMBasedTextDecoder
from medvqa.models.vqa.open_ended_vqa import RawImageEncoding
from medvqa.utils.constants import CHEXPERT_GENDERS, CHEXPERT_LABELS, CHEXPERT_ORIENTATIONS, CXR14_LABELS, VINBIG_DISEASES

class ImageTextEncoder(nn.Module):

    def __init__(self,
                 # Vocab args
                 vocab_size, embed_size,
                 start_idx=None,
                 # Image Encoder args
                 raw_image_encoding=RawImageEncoding.DENSENET_121,
                 image_local_feat_size=None,
                 freeze_image_encoder=False,
                 image_encoder_pretrained_weights_path=None,
                 imagenet_pretrained=True,
                 clip_version=None,
                 use_image_features_in_qclass=True,
                 # Text Encoder args
                 text_vec_size=None,
                 text_hidden_size=None,                 
                 # Auxiliary tasks args
                 use_mimiccxr=False,
                 use_iuxray=False,
                 use_chexpert=False,
                 use_cxr14=False,
                 use_vinbig=False,
                 classify_orientation=False,
                 classify_chexpert=False,
                 classify_questions=False,
                 n_questions=None,
                 use_chexpert_embeddings=False,
                 chexpert_embed_size=None,
                 # Other args
                 device=None,
                 **unused_kwargs,
                 ): 
        super().__init__()

        if use_chexpert_embeddings:
            assert classify_chexpert
        
        # Init vocab embedding table
        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=0,
        )
        
        # Init image encoder
        self._init_raw_image_encoder(raw_image_encoding, image_encoder_pretrained_weights_path,
                                    imagenet_pretrained, clip_version, freeze_image_encoder)
        self.image_global_feat_size = self._get_raw_image_encoder_global_feat_size(image_local_feat_size)
        
        # Init text encoder
        if classify_questions:
            self._init_text_encoder(text_hidden_size, embed_size, text_vec_size, vocab_size, start_idx, device)
            
        # Init auxiliary tasks
        self._init_auxiliary_tasks(use_mimiccxr, use_iuxray, use_chexpert, use_cxr14, use_vinbig,
                             classify_orientation, classify_chexpert,
                             classify_questions, n_questions,
                             use_image_features_in_qclass,
                             use_chexpert_embeddings, chexpert_embed_size)

    def _get_raw_image_encoder_global_feat_size(self, image_local_feat_size):
        if self.raw_image_encoding == RawImageEncoding.DENSENET_121:
            return 2 * image_local_feat_size
        if self.raw_image_encoding == RawImageEncoding.CLIP_VIT:
            return CLIP_VIT_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.CLIP_RESNET:
            return CLIP_RESNET_GLOBAL_FEAT_SIZE
        if self.raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE:
            return HUGGINGFACE_CLIP_VIT_GLOBAL_FEAT_SIZE
        assert False
    
    def _init_raw_image_encoder(self, raw_image_encoding, pretrained_weights_path,
                                imagenet_pretrained, clip_version, freeze_image_encoder):
        self.raw_image_encoding = raw_image_encoding
        self.clip_version = clip_version        
        if raw_image_encoding == RawImageEncoding.DENSENET_121:
            self.raw_image_encoder = create_densenet121_feature_extractor(pretrained_weights_path, imagenet_pretrained)
        elif raw_image_encoding == RawImageEncoding.CLIP_RESNET:
            self.raw_image_encoder = create_clip_resnet_feature_extractor(clip_version, pretrained_weights_path)
        elif raw_image_encoding == RawImageEncoding.CLIP_VIT:
            self.raw_image_encoder = create_clip_vit_feature_extractor(clip_version, pretrained_weights_path)
        elif raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE:
            self.raw_image_encoder = create_huggingface_clip_vit_feature_extractor(clip_version, pretrained_weights_path)
        else: assert False, f'Unknown image encoding {raw_image_encoding}'
        if freeze_image_encoder: freeze_parameters(self.raw_image_encoder)

    def _init_text_encoder(self, text_hidden_size, embed_size, text_vec_size, vocab_size, start_idx, device):
        
        self.text_vec_size = text_vec_size
        self.text_encoder = BiLSTMBasedTextEncoder(self.embedding_table,
                                                    embed_size,
                                                    text_hidden_size,
                                                    text_vec_size,
                                                    device)
        self.text_decoder = LSTMBasedTextDecoder(self.embedding_table,
                                        embed_size,
                                        text_hidden_size,
                                        text_vec_size,
                                        vocab_size,
                                        start_idx)

    def _init_auxiliary_tasks(self, use_mimiccxr, use_iuxray, use_chexpert, use_cxr14, use_vinbig,
                             classify_orientation, classify_chexpert,
                             classify_questions, n_questions,
                             use_image_features_in_qclass,
                             use_chexpert_embeddings, chexpert_embed_size):
        
        self.orien_aux_task = classify_orientation
        self.q_aux_task = classify_questions
        self.use_image_features_in_qclass = use_image_features_in_qclass
        self.use_chx_embed = use_chexpert_embeddings
        
        # 1) MIMIC-CXR specific tasks
        if use_mimiccxr:
            if classify_orientation:
                self.W_ori_mimiccxr = nn.Linear(self.image_global_feat_size, len(MIMICCXR_IMAGE_ORIENTATIONS))
        
        # 2) IU-Xray specific tasks
        if use_iuxray:
            if classify_orientation:
                self.W_ori_iuxray = nn.Linear(self.image_global_feat_size, len(IUXRAY_IMAGE_ORIENTATIONS))

        # 3) questions classification
        if classify_questions:
            self.multimodal_global_feat_size = self.text_vec_size
            if use_image_features_in_qclass:
                self.multimodal_global_feat_size += self.image_global_feat_size
            if use_chexpert_embeddings:
                self.chx_embedding_table = nn.parameter.Parameter(torch.Tensor(len(CHEXPERT_LABELS), chexpert_embed_size))
                self.chx_embedding_table.data.uniform_(-1, 1)
                self.multimodal_global_feat_size += chexpert_embed_size
            print(f'self.multimodal_global_feat_size = {self.multimodal_global_feat_size}')
            self.W_q = nn.Linear(self.multimodal_global_feat_size, n_questions)

        # 4) Chexpert & CRX14's specific tasks: gender & orientaition
        if use_chexpert or use_cxr14:
            self.W_gender_chexpert = nn.Linear(self.image_global_feat_size, len(CHEXPERT_GENDERS))
            self.W_ori_chexpert = nn.Linear(self.image_global_feat_size, len(CHEXPERT_ORIENTATIONS))
        
        # 5) chexpert classifiction
        if classify_chexpert:
            self.W_chx = nn.Linear(self.image_global_feat_size, len(CHEXPERT_LABELS))
            self.chx_aux_task = True
        else:
            self.chx_aux_task = False

        # 6) CXR14 specific tasks
        if use_cxr14:
            self.W_cxr14 = nn.Linear(self.image_global_feat_size, len(CXR14_LABELS))

        # 7) VinBig specific tasks
        if use_vinbig:
            self.W_vinbig = nn.Linear(self.image_global_feat_size, len(VINBIG_DISEASES))

    @property
    def name(self):        
        strings = []
        
        if self.raw_image_encoding == RawImageEncoding.DENSENET_121:
            img_str = 'dense121'
        elif self.raw_image_encoding in (RawImageEncoding.CLIP_VIT,
                                         RawImageEncoding.CLIP_RESNET):
            img_str = f'clip-{self.clip_version}'
        elif self.raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE:
            img_str = HUGGINGFACE_CLIP_VIT_VERSIONS_2_SHORT[self.clip_version]
        else: assert False
        if not self.use_image_features_in_qclass:
            img_str += '(niqc)'
        strings.append(img_str)

        if self.q_aux_task:
            if self.use_chx_embed:
                strings.append('chx-emb')
            txt_enc_str = 'txtenc=bilstm'
            strings.append(txt_enc_str)
            txt_dec_str = 'txtdec=lstm'
            strings.append(txt_dec_str)
        
        name = f'imgtxtenc({"+".join(strings)})'
        return name

    def forward(
        self,
        raw_images,
        texts=None,
        text_lengths=None,
        iuxray_forward=False,
        mimiccxr_forward=False,
        chexpert_forward=False,
        cxr14_forward=False,
        vinbig_forward=False,
    ):
        # Visual Component
        local_feat = None
        global_list = []

        if self.raw_image_encoding == RawImageEncoding.DENSENET_121:
            # densenet local features
            local_feat = self.raw_image_encoder(raw_images)
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
        elif self.raw_image_encoding == RawImageEncoding.CLIP_VIT__HUGGINGFACE:
            tmp = self.raw_image_encoder(raw_images)
            global_feat, local_feat = tmp.pooler_output, tmp.last_hidden_state
            global_list.append(global_feat)        
        else: assert False

        if len(global_list) > 1:
            global_feat = torch.cat(global_list, 1)
        else:
            global_feat = global_list[0]

        output = {}

        if chexpert_forward:
            output['pred_orientation'] = self.W_ori_chexpert(global_feat)
            output['pred_gender'] = self.W_gender_chexpert(global_feat)
            output['pred_chexpert'] = self.W_chx(global_feat)
            output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
        
        elif cxr14_forward:
            output['pred_orientation'] = self.W_ori_chexpert(global_feat) # weight sharing with chexpert
            output['pred_gender'] = self.W_gender_chexpert(global_feat) # weight sharing with chexpert
            output['pred_cxr14'] = self.W_cxr14(global_feat)
            output['pred_cxr14_probs'] = torch.sigmoid(output['pred_cxr14'])
        
        elif vinbig_forward:
            output['pred_vinbig'] = self.W_vinbig(global_feat)
            output['pred_vinbig_probs'] = torch.sigmoid(output['pred_vinbig'])
        
        elif iuxray_forward or mimiccxr_forward:
            
            if self.orien_aux_task:
                if iuxray_forward:
                    output['iuxray_pred_orientation'] = self.W_ori_iuxray(global_feat)
                if mimiccxr_forward:
                    output['mimiccxr_pred_orientation'] = self.W_ori_mimiccxr(global_feat)
            
            if self.chx_aux_task:
                output['pred_chexpert'] = self.W_chx(global_feat)
                output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])

            if self.q_aux_task:
                multimodal_list = []
                # global visual features
                if self.use_image_features_in_qclass:
                    multimodal_list.append(global_feat)
                # process text
                text_vectors = self.text_encoder(texts, text_lengths)
                multimodal_list.append(text_vectors)
                # recover text from vectors (autoencoder)
                output['pred_texts'] = self.text_decoder(text_vectors, texts, text_lengths)
                # chexpert embeddings
                if self.use_chx_embed:
                    chexpert_vectors = torch.matmul(output['pred_chexpert_probs'], self.chx_embedding_table)
                    multimodal_list.append(chexpert_vectors)
                # multimodal features
                if len(multimodal_list) > 1:
                    multimodal_feat = torch.cat(multimodal_list, 1)
                else:
                    multimodal_feat = multimodal_list[0]
                # classify questions
                output['pred_qlabels'] = self.W_q(multimodal_feat)
                output['pred_qlabels_probs'] = torch.sigmoid(output['pred_qlabels'])

        return output
