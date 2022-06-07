import torch
import torch.nn as nn
import torchvision.models as models
from medvqa.models.nlp.question_encoder import QuestionEncoder_BiLSTM
from medvqa.models.nlp.question_decoder import QuestionDecoder
from medvqa.models.vqa.answer_decoder import AnswerDecoder
from medvqa.datasets.mimiccxr import MIMICCXR_IMAGE_ORIENTATIONS
from medvqa.datasets.iuxray import IUXRAY_IMAGE_ORIENTATIONS
from medvqa.utils.constants import CHEXPERT_LABELS

class QuestionEncoding:
    BILSTM = 'bilstm'
    ONE_HOT = 'one-hot'

class OpenEndedVQA(nn.Module):

    def __init__(self, vocab_size, start_idx, embed_size, answer_hidden_size,
                 n_lstm_layers,
                 question_vec_size, image_local_feat_size, dropout_prob, device,                 
                 question_encoding = QuestionEncoding.BILSTM,
                 question_hidden_size = None,
                 densenet_pretrained_weights_path=None,
                 n_medical_tags=None,
                 n_questions=None,
                 classify_orientation=False,
                 classify_chexpert=False,
                 classify_questions=False,
                 eos_idx=None,
                 padding_idx=None,
                 **unused_kwargs,
                 ):
        super().__init__()
        self.name = 'oevqa(densenet121+bilstm+lstm)'
        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=0,
        )
        
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

        self.question_encoding = question_encoding
        
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
        self.answer_decoder = AnswerDecoder(self.embedding_table,
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
            self.W_q = nn.Linear(image_local_feat_size * 2, n_questions)
            self.q_aux_task = True
        else:
            self.q_aux_task = False

    def forward(
        self,
        images,
        questions,
        device,
        question_lengths=None,
        answers=None,
        max_answer_length=None,
        mode='train',
        beam_search_k=None,
        iuxray_foward=False,
        mimiccxr_foward=False,
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

        # process questions
        if self.question_encoding == QuestionEncoding.BILSTM:
            question_vectors = self.question_encoder(questions, question_lengths)
        else: # one-hot
            question_vectors = self.question_encoder(questions)

        # predict answers
        if mode == 'train':
            pred_answers = self.answer_decoder.teacher_forcing_decoding(local_feat, global_feat, question_vectors, answers)
        elif beam_search_k:
            pred_answers = self.answer_decoder.beam_search_decoding(local_feat, global_feat, question_vectors, max_answer_length, device, beam_search_k)
        else:
            pred_answers = self.answer_decoder.greedy_search_decoding(local_feat, global_feat, question_vectors, max_answer_length)

        output = { 'pred_answers': pred_answers }

        # recover questions from vectors
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

        if self.q_aux_task:
            output['pred_qlabels'] = self.W_q(global_feat)

        return output