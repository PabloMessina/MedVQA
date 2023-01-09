import torch
from torch import nn
from medvqa.models.common import freeze_parameters
from medvqa.models.nlp.text_encoder import BiLSTMBasedTextEncoder
from medvqa.models.nlp.text_decoder import LSTMBasedTextDecoder
from medvqa.models.vision.visual_modules import (
    CLIP_RESNET_GLOBAL_FEAT_SIZE,
    CLIP_VIT_GLOBAL_FEAT_SIZE,
    HUGGINGFACE_CLIP_VIT_GLOBAL_FEAT_SIZE,
    HUGGINGFACE_CLIP_VIT_LARGE_GLOBAL_FEAT_SIZE,
    HUGGINGFACE_CLIP_VIT_NAMES_2_SHORT,
    HUGGINGFACE_VITMODEL_GLOBAL_FEAT_SIZE,
    HUGGINGFACE_VITMODEL_LARGE_GLOBAL_FEAT_SIZE,
    HUGGINGFACE_VITMODEL_NAMES_2_SHORT,
    HUGGINGFACE_VITMODEL_UNFROZEN_PARAM_NAMES_REGEX,
    create_clip_vit_feature_extractor,
    create_clip_resnet_feature_extractor,
    create_densenet121_feature_extractor,
    create_huggingface_clip_vit_feature_extractor,
    create_huggingface_vitmodel_feature_extractor,
)
from medvqa.models.vqa.lstm_answer_decoder import LSTMAnswerDecoder
from medvqa.models.mlp import MLP
from medvqa.datasets.mimiccxr import MIMICCXR_IMAGE_ORIENTATIONS
from medvqa.datasets.iuxray import IUXRAY_IMAGE_ORIENTATIONS
from medvqa.models.vqa.transformer_answer_decoder import TransformerAnswerDecoder
from medvqa.utils.constants import (
    CHEXPERT_LABELS,
    CHEXPERT_GENDERS,
    CHEXPERT_ORIENTATIONS,
    CHEXPERT_TASKS,
    CXR14_LABELS,
    VINBIG_DISEASES,
    PADCHEST_NUM_LABELS,
    PADCHEST_NUM_LOCALIZATIONS,
    PADCHEST_PROJECTIONS,
)

class QuestionEncoding:
    BILSTM = 'bilstm'
    ONE_HOT = 'one-hot'

class AnswerDecoding:
    LSTM = 'lstm'
    TRANSFORMER  = 'transformer'

class RawImageEncoding:
    DENSENET_121 = 'densenet-121'
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

class OpenEndedVQA(nn.Module):

    def __init__(self,
                 # Vocab args
                 vocab_size, embed_size,
                 start_idx=None,
                 eos_idx=None,
                 padding_idx=None,
                 # Image Encoder args
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
                 huggingface_model_name=None,
                 # Question Encoder args
                 question_encoding = QuestionEncoding.BILSTM,
                 question_vec_size=None,
                 question_hidden_size=None,
                 # Answer Decoder args
                 answer_decoding =  AnswerDecoding.LSTM,
                 answer_hidden_size=None,
                 n_lstm_layers=None,
                 transf_dec_nhead=None,
                 transf_dec_dim_forward=None,
                 transf_dec_num_layers=None,
                 # Auxiliary tasks args
                 classify_tags=False,
                 classify_orientation=False,
                 classify_chexpert=False,
                 classify_questions=False,
                 n_medical_tags=None,
                 n_questions=None,
                 n_questions_aux_task=None,
                 use_cxr14=False,
                 use_vinbig=False,
                 use_padchest=False,
                 merge_findings=False,
                 n_findings=None,
                 # Other args
                 chexpert_mode=None,
                 dropout_prob=0,
                 device=None,
                 **unused_kwargs,
                 ): 
        super().__init__()

        print('OpenEndedVQA():')

        print('  image_encoder_pretrained_weights_path', image_encoder_pretrained_weights_path)

        self.chexpert_mode = chexpert_mode
        if chexpert_mode == CHEXPERT_TASKS.VQA:
            assert question_encoding == QuestionEncoding.ONE_HOT
        
        # Init vocab embedding table
        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=0,
        )
        
        # Init Visual Module
        self.visual_input_mode = visual_input_mode
        self.clip_version = clip_version
        self.huggingface_model_name = huggingface_model_name
        if raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE:
            assert huggingface_model_name is not None
        self._init_visual_module(visual_input_mode, raw_image_encoding, image_encoder_pretrained_weights_path,
                            imagenet_pretrained, image_local_feat_size, mlp_in_dim, mlp_out_dim, mlp_hidden_dims,
                            clip_version, huggingface_model_name, freeze_image_encoder)
        
        # Init Question Encoder
        self.question_encoding = question_encoding
        self._init_question_encoder(question_encoding, question_hidden_size=question_hidden_size,
                        embed_size=embed_size, question_vec_size=question_vec_size,
                        vocab_size=vocab_size, start_idx=start_idx, n_questions=n_questions,
                        device=device)
          
        # Init Answer Decoder
        self.answer_decoding = answer_decoding
        self._init_answer_decoder(image_local_feat_size, question_vec_size, embed_size,
                            answer_hidden_size, n_lstm_layers, start_idx, eos_idx, padding_idx, vocab_size,
                            transf_dec_nhead, transf_dec_dim_forward, transf_dec_num_layers, dropout_prob,
                            use_local_features=does_include_image(visual_input_mode))
            
        # Init auxiliary tasks
        self._init_auxiliary_tasks(classify_tags, classify_orientation, classify_chexpert, classify_questions,
                              chexpert_mode, use_cxr14, use_vinbig, use_padchest, n_medical_tags, n_questions_aux_task,
                              merge_findings=merge_findings, n_findings=n_findings)

        # Logging
        print(f'  n_questions = {n_questions}\n  n_questions_aux_task = {n_questions_aux_task}\n'
              f'  question_encoding = {question_encoding}\n  answer_decoding = {answer_decoding}\n'
              f'  visual_input_mode = {visual_input_mode}\n  name = {self.name}')

    def _init_visual_module(self, visual_input_mode, raw_image_encoding, image_encoder_pretrained_weights_path,
                            imagenet_pretrained, image_local_feat_size, mlp_in_dim, mlp_out_dim, mlp_hidden_dims,
                            clip_version, huggingface_model_name, freeze_image_encoder):
        global_feat_size = 0
        
        if does_include_image(visual_input_mode):
            model_name = clip_version if clip_version is not None else huggingface_model_name
            self._init_raw_image_encoder(raw_image_encoding, image_encoder_pretrained_weights_path,
                                         imagenet_pretrained, model_name, freeze_image_encoder)
            global_feat_size += self._get_raw_image_encoder_global_feat_size(image_local_feat_size)
        
        if does_include_visual_features(visual_input_mode):
            self._init_mlp_visual_feat_encoder(mlp_in_dim, mlp_out_dim, mlp_hidden_dims, freeze_image_encoder)
            global_feat_size += mlp_out_dim
        
        assert global_feat_size > 0
        self.global_feat_size = global_feat_size
        print('  self.global_feat_size =', self.global_feat_size)

    def _get_raw_image_encoder_global_feat_size(self, image_local_feat_size):
        if self.raw_image_encoding == RawImageEncoding.DENSENET_121:
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

    def _init_question_encoder(self, question_encoding, question_hidden_size=None,
                               embed_size=None, question_vec_size=None, vocab_size=None,
                               start_idx=None, n_questions=None, device=None):
        if question_encoding == QuestionEncoding.BILSTM:
            assert question_hidden_size is not None
            self.question_encoder = BiLSTMBasedTextEncoder(self.embedding_table,
                                                        embed_size,
                                                        question_hidden_size,
                                                        question_vec_size,
                                                        device)
            self.question_decoder = LSTMBasedTextDecoder(self.embedding_table,
                                            embed_size,
                                            question_hidden_size,
                                            question_vec_size,
                                            vocab_size,
                                            start_idx)
        elif question_encoding == QuestionEncoding.ONE_HOT:
            assert n_questions is not None
            self.question_encoder = nn.Embedding(
                num_embeddings=n_questions,
                embedding_dim=question_vec_size,
            )
        else:
            assert False, f'Unknown question encoding strategy {question_encoding}'

    def _init_answer_decoder(self, image_local_feat_size, question_vec_size, embed_size,
                            answer_hidden_size, n_lstm_layers, start_idx, eos_idx, padding_idx, vocab_size,
                            transf_dec_nhead, transf_dec_dim_forward, transf_dec_num_layers, dropout_prob,
                            use_local_features=True):
        if self.answer_decoding == AnswerDecoding.LSTM:
            self.answer_decoder = LSTMAnswerDecoder(self.embedding_table,
                                                    image_local_feat_size,
                                                    self.global_feat_size,
                                                    question_vec_size,
                                                    embed_size,
                                                    answer_hidden_size,
                                                    n_lstm_layers,
                                                    start_idx,
                                                    vocab_size,
                                                    dropout_prob,
                                                    eos_idx=eos_idx,
                                                    padding_idx=padding_idx,
                                                    use_local_features=use_local_features)
        elif self.answer_decoding == AnswerDecoding.TRANSFORMER:
            self.answer_decoder = TransformerAnswerDecoder(self.embedding_table,
                                                           embed_size,
                                                           answer_hidden_size,
                                                           question_vec_size,
                                                           image_local_feat_size,
                                                           self.global_feat_size,
                                                           transf_dec_nhead,
                                                           transf_dec_dim_forward,
                                                           transf_dec_num_layers,
                                                           start_idx,
                                                           vocab_size,
                                                           dropout_prob,
                                                           use_local_features=use_local_features)
        else:
            assert False, f'Unknown answer decoding module {self.answer_decoding}'

    def _init_auxiliary_tasks(self, classify_tags, classify_orientation, classify_chexpert, classify_questions,
                              chexpert_mode, use_cxr14, use_vinbig, use_padchest, n_medical_tags, n_questions_aux_task,
                              merge_findings=False, n_findings=None):
        
        # Optional auxiliary tasks
        self.merge_findings = merge_findings
        
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

        # 4) gender classification (chexpert & CRX14 & PadChest)
        if chexpert_mode is not None or use_cxr14 or use_padchest:
            self.W_gender_chexpert = nn.Linear(self.global_feat_size, len(CHEXPERT_GENDERS))

        # 5) orientation classification (weight sharing among chexpert & CRX14)
        if chexpert_mode is not None or use_cxr14:
            self.W_ori_chexpert = nn.Linear(self.global_feat_size, len(CHEXPERT_ORIENTATIONS))

        if merge_findings:
            assert n_findings is not None
            self.W_findings = nn.Linear(self.global_feat_size, n_findings)
        else:        
            # 6) chexpert classifiction
            if classify_chexpert:
                self.W_chx = nn.Linear(self.global_feat_size, len(CHEXPERT_LABELS))
                self.chx_aux_task = True
            else:
                self.chx_aux_task = False

            # 7) CXR14 specific tasks
            if use_cxr14:
                self.W_cxr14 = nn.Linear(self.global_feat_size, len(CXR14_LABELS))

            # 8) VinBig specific tasks
            if use_vinbig:
                self.W_vinbig = nn.Linear(self.global_feat_size, len(VINBIG_DISEASES))

            # 9) PadChest specific tasks
            if use_padchest:
                self.W_padchest_labels = nn.Linear(self.global_feat_size, PADCHEST_NUM_LABELS)
                self.W_padchest_loc = nn.Linear(self.global_feat_size, PADCHEST_NUM_LOCALIZATIONS)
                self.W_padchest_ori = nn.Linear(self.global_feat_size, len(PADCHEST_PROJECTIONS))

    @property
    def name(self):        
        if self.raw_image_encoding == RawImageEncoding.DENSENET_121:
            img_str = 'dense121'
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
        else: assert False
        q_enc_str = 'bilstm' if self.question_encoding == QuestionEncoding.BILSTM else 'onehot'
        a_dec_str = 'lstm' if self.answer_decoding == AnswerDecoding.LSTM else 'transf'
        strings = [vm_str, q_enc_str, a_dec_str]
        name = f'oevqa({"+".join(strings)})'
        return name

    def forward(
        self,
        raw_images=None,
        visual_features=None,
        questions=None,
        question_lengths=None,
        answers=None,
        max_answer_length=None,
        mode='train',
        beam_search_k=None,
        iuxray_foward=False,
        mimiccxr_foward=False,
        chexpert_forward=False,
        cxr14_forward=False,
        vinbig_forward=False,
        padchest_forward=False,
        device=None,
    ):
        # Visual Component                
        assert (raw_images is not None) or (visual_features is not None)
        local_feat = None
        global_list = []        
        
        if raw_images is not None:

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

        output = {}
        question_vectors = None

        if self.merge_findings:
            output['pred_findings'] = self.W_findings(global_feat)
            output['pred_findings_probs'] = torch.sigmoid(output['pred_findings'])

        if chexpert_forward:
            if self.chexpert_mode == CHEXPERT_TASKS.VQA:                
                question_vectors = self.question_encoder(questions)            
            output['pred_orientation'] = self.W_ori_chexpert(global_feat)
            output['pred_gender'] = self.W_gender_chexpert(global_feat)
            if not self.merge_findings:
                output['pred_chexpert'] = self.W_chx(global_feat)
                output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
        elif cxr14_forward:
            question_vectors = self.question_encoder(questions)
            output['pred_orientation'] = self.W_ori_chexpert(global_feat) # weight sharing with chexpert
            output['pred_gender'] = self.W_gender_chexpert(global_feat) # weight sharing with chexpert
            if not self.merge_findings:
                output['pred_cxr14'] = self.W_cxr14(global_feat)
                output['pred_cxr14_probs'] = torch.sigmoid(output['pred_cxr14'])
        elif vinbig_forward:
            question_vectors = self.question_encoder(questions)
            if not self.merge_findings:
                output['pred_vinbig'] = self.W_vinbig(global_feat)
                output['pred_vinbig_probs'] = torch.sigmoid(output['pred_vinbig'])
        elif padchest_forward:
            question_vectors = self.question_encoder(questions)
            output['pred_orientation'] = self.W_padchest_ori(global_feat)
            output['pred_gender'] = self.W_gender_chexpert(global_feat) # weight sharing with chexpert
            output['pred_padchest_labels'] = self.W_padchest_labels(global_feat)
            output['pred_padchest_labels_probs'] = torch.sigmoid(output['pred_padchest_labels'])
            output['pred_padchest_loc'] = self.W_padchest_loc(global_feat)
            output['pred_padchest_loc_probs'] = torch.sigmoid(output['pred_padchest_loc'])
        else:
            # process questions
            if self.question_encoding == QuestionEncoding.BILSTM:
                question_vectors = self.question_encoder(questions, question_lengths)
            else: # one-hot
                question_vectors = self.question_encoder(questions)            

            # recover questions from vectors if in BILSTM mode
            if self.question_encoding == QuestionEncoding.BILSTM:
                pred_questions = self.question_decoder(question_vectors, questions, question_lengths)
                output['pred_questions'] = pred_questions

            # auxiliary tasks (optional)
            
            if self.tags_aux_task:
                output['pred_tags'] = self.W_tags(global_feat)
            
            if self.orien_aux_task:
                if iuxray_foward:
                    output['iuxray_pred_orientation'] = self.W_ori_iuxray(global_feat)
                if mimiccxr_foward:
                    output['mimiccxr_pred_orientation'] = self.W_ori_mimiccxr(global_feat)

            if self.q_aux_task:
                output['pred_qlabels'] = self.W_q(global_feat)

            if not self.merge_findings and self.chx_aux_task:
                output['pred_chexpert'] = self.W_chx(global_feat)
                output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])

        # predict answers (if required)
        if question_vectors is not None:
            if self.answer_decoding == AnswerDecoding.LSTM: # LSTM-based decoding            
                if mode == 'train':
                    pred_answers = self.answer_decoder.teacher_forcing_decoding(local_feat, global_feat, question_vectors, answers)
                elif beam_search_k:
                    pred_answers = self.answer_decoder.beam_search_decoding(local_feat, global_feat, question_vectors, max_answer_length, device, beam_search_k)
                else:
                    pred_answers = self.answer_decoder.greedy_search_decoding(local_feat, global_feat, question_vectors, max_answer_length)                
            elif self.answer_decoding == AnswerDecoding.TRANSFORMER: # Transformr-based decoding
                if mode == 'train':
                    pred_answers = self.answer_decoder.teacher_forcing_decoding(local_feat, global_feat, question_vectors, answers, device)
                else:
                    pred_answers = self.answer_decoder.greedy_search_decoding(local_feat, global_feat, question_vectors, max_answer_length, device)
            else: assert False
            output['pred_answers'] = pred_answers

        return output