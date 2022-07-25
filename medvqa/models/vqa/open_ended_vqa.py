import torch
import torch.nn as nn
import torchvision.models as models
from medvqa.models.common import freeze_parameters
from medvqa.models.nlp.question_encoder import QuestionEncoder_BiLSTM
from medvqa.models.nlp.question_decoder import QuestionDecoder
from medvqa.models.vqa.lstm_answer_decoder import LSTMAnswerDecoder
from medvqa.datasets.mimiccxr import MIMICCXR_IMAGE_ORIENTATIONS
from medvqa.datasets.iuxray import IUXRAY_IMAGE_ORIENTATIONS
from medvqa.models.vqa.transformer_answer_decoder import TransformerAnswerDecoder
from medvqa.utils.constants import (
    CHEXPERT_LABELS,
    CHEXPERT_GENDERS,
    CHEXPERT_ORIENTATIONS,
    CHEXPERT_TASKS,
)

class QuestionEncoding:
    BILSTM = 'bilstm'
    ONE_HOT = 'one-hot'

class AnswerDecoding:
    LSTM = 'lstm'
    TRANSFORMER  = 'transformer'

class OpenEndedVQA(nn.Module):

    def __init__(self, vocab_size, start_idx, embed_size, answer_hidden_size,
                 question_vec_size, image_local_feat_size, dropout_prob, device,                 
                 question_encoding = QuestionEncoding.BILSTM,
                 answer_decoding =  AnswerDecoding.LSTM,
                 n_lstm_layers=None,
                 transf_dec_nhead=None,
                 transf_dec_dim_forward=None,
                 transf_dec_num_layers=None,
                 question_hidden_size = None,
                 densenet_pretrained_weights_path=None,
                 freeze_cnn=False,
                 n_medical_tags=None,
                 n_questions=None,
                 n_questions_aux_task=None,
                 classify_orientation=False,
                 classify_chexpert=False,
                 classify_questions=False,
                 eos_idx=None,
                 padding_idx=None,
                 chexpert_mode=None,
                 **unused_kwargs,
                 ):
        print(f'OpenEndedVQA(): n_questions = {n_questions}, n_questions_aux_task = {n_questions_aux_task}'
              f' question_encoding = {question_encoding}, answer_decoding = {answer_decoding}')
        super().__init__()

        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=0,
        )

        if chexpert_mode == CHEXPERT_TASKS.VQA:
            assert question_encoding == QuestionEncoding.ONE_HOT
        
        # Load pre-trained CNN weights
        if densenet_pretrained_weights_path:
            densenet = models.densenet121(pretrained=False)
            pretrained_weights = torch.load(densenet_pretrained_weights_path, map_location='cuda')            
            densenet.load_state_dict(pretrained_weights, strict=False)
            print('Using densenet121 with pretrained weights from', densenet_pretrained_weights_path)
        else:
            densenet = models.densenet121(pretrained=True)
            print('Using densenet121 with ImageNet pretrained weights')        
        # self.image_encoder = nn.Sequential(*list(densenet.children())[:-1])
        self.image_encoder = densenet.features

        if freeze_cnn: freeze_parameters(self.image_encoder)

        self.question_encoding = question_encoding
        self.answer_decoding = answer_decoding
        self.chexpert_mode = chexpert_mode
        
        # Question encoding
        if question_encoding == QuestionEncoding.BILSTM:
            assert question_hidden_size is not None
            self.question_encoder = QuestionEncoder_BiLSTM(self.embedding_table,
                                                        embed_size,
                                                        question_hidden_size,
                                                        question_vec_size,
                                                        device)
            self.question_decoder = QuestionDecoder(self.embedding_table,
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
          
        # Answer decoding
        if answer_decoding == AnswerDecoding.LSTM:
            self.answer_decoder = LSTMAnswerDecoder(self.embedding_table,
                                                    image_local_feat_size,
                                                    question_vec_size,
                                                    embed_size,
                                                    answer_hidden_size,
                                                    n_lstm_layers,
                                                    start_idx,
                                                    vocab_size,
                                                    dropout_prob,
                                                    eos_idx=eos_idx,
                                                    padding_idx=padding_idx)
        elif answer_decoding == AnswerDecoding.TRANSFORMER:
            self.answer_decoder = TransformerAnswerDecoder(self.embedding_table,
                                                           embed_size,
                                                           answer_hidden_size,
                                                           question_vec_size,
                                                           image_local_feat_size,
                                                           transf_dec_nhead,
                                                           transf_dec_dim_forward,
                                                           transf_dec_num_layers,
                                                           start_idx,
                                                           vocab_size,
                                                           dropout_prob)
        else:
            assert False, f'Unknown answer decoding modul {answer_decoding}'
            
        # Optional auxiliary tasks        
        
        # 1) medical tags prediction
        if n_medical_tags is not None:
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

        if chexpert_mode is not None:
            self.W_gender_chexpert = nn.Linear(image_local_feat_size * 2, len(CHEXPERT_GENDERS))
            self.W_ori_chexpert = nn.Linear(image_local_feat_size * 2, len(CHEXPERT_ORIENTATIONS))

    @property
    def name(self):
        strings = [
            'dense121',
            'bilstm' if self.question_encoding == QuestionEncoding.BILSTM else 'onehot',
            'lstm' if self.answer_decoding == AnswerDecoding.LSTM else 'transf',
        ]
        name = f'oevqa({"+".join(strings)})'
        return name

    def forward(
        self,
        images,
        questions=None,
        question_lengths=None,
        answers=None,
        max_answer_length=None,
        mode='train',
        beam_search_k=None,
        iuxray_foward=False,
        mimiccxr_foward=False,
        chexpert_forward=False,
        device=None,
    ):
        # cnn local features
        batch_size = images.size(0)
        local_feat = self.image_encoder(images)
        feat_size = local_feat.size(1)
        local_feat = local_feat.permute(0,2,3,1).view(batch_size, -1, feat_size)

        # compute global features
        global_avg_pool = local_feat.mean(1)
        global_max_pool = local_feat.max(1)[0]
        global_feat = torch.cat((global_avg_pool, global_max_pool), 1)

        output = {}

        question_vectors = None

        if chexpert_forward:
            output['pred_chexpert'] = self.W_chx(global_feat)
            output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])
            output['pred_orientation'] = self.W_ori_chexpert(global_feat)
            output['pred_gender'] = self.W_gender_chexpert(global_feat)
            if self.chexpert_mode == CHEXPERT_TASKS.VQA:
                # process questions
                question_vectors = self.question_encoder(questions)
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
            
            if self.chx_aux_task:
                output['pred_chexpert'] = self.W_chx(global_feat)
                output['pred_chexpert_probs'] = torch.sigmoid(output['pred_chexpert'])

            if self.q_aux_task:
                output['pred_qlabels'] = self.W_q(global_feat)

        # predict answers (if required)
        if question_vectors is not None:
            if self.answer_decoding == AnswerDecoding.LSTM: # LSTM-based decoding            
                if mode == 'train':
                    pred_answers = self.answer_decoder.teacher_forcing_decoding(local_feat, global_feat, question_vectors, answers)
                elif beam_search_k:
                    pred_answers = self.answer_decoder.beam_search_decoding(local_feat, global_feat, question_vectors, max_answer_length, device, beam_search_k)
                else:
                    pred_answers = self.answer_decoder.greedy_search_decoding(local_feat, global_feat, question_vectors, max_answer_length)                
            else: # Transformr-based decoding
                if mode == 'train':
                    pred_answers = self.answer_decoder.teacher_forcing_decoding(local_feat, global_feat, question_vectors, answers, device)
                else:
                    pred_answers = self.answer_decoder.greedy_search_decoding(local_feat, global_feat, question_vectors, max_answer_length, device)
            output['pred_answers'] = pred_answers

        return output