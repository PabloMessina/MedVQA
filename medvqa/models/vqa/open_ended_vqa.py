import torch
import torch.nn as nn
import torchvision.models as models
from medvqa.models.vqa.question_encoder import QuestionEncoder_BiLSTM
from medvqa.models.vqa.question_decoder import QuestionDecoder
from medvqa.models.vqa.answer_decoder import AnswerDecoder
from medvqa.datasets.mimiccxr import MIMICCXR_IMAGE_ORIENTATIONS
from medvqa.datasets.iuxray import IUXRAY_IMAGE_ORIENTATIONS

class OpenEndedVQA(nn.Module):

    def __init__(self, vocab_size, start_idx, embed_size, hidden_size,
                 question_vec_size, image_local_feat_size, dropout_prob, device,
                 densenet_pretrained_weights_path=None,                 
                 n_medical_tags=None,
                 classify_orientation=False):
        super().__init__()
        self.name = 'oevqa(densenet121+bilstm+lstm)'
        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=0,
        )        
        # Load pre-trained weights
        if densenet_pretrained_weights_path:
            densenet = models.densenet121(pretrained=False)
            pretrained_weights = torch.load(densenet_pretrained_weights_path, map_location='cuda')
            # begin HACK
            del pretrained_weights["prediction.weight"]
            del pretrained_weights["prediction.bias"]            
            densenet.classifier = None
            # end HACK
            densenet.load_state_dict(pretrained_weights)
            print('Using densenet121 with pretrained weights from', densenet_pretrained_weights_path)
        else:
            densenet = models.densenet121(pretrained=True)
            print('Using densenet121 with ImageNet pretrained weights')        
        # self.image_encoder = nn.Sequential(*list(densenet.children())[:-1])
        self.image_encoder = densenet.features
        self.question_encoder = QuestionEncoder_BiLSTM(self.embedding_table,
                                                       embed_size,
                                                       hidden_size,
                                                       question_vec_size,
                                                       device)
        self.question_decoder = QuestionDecoder(self.embedding_table,
                                            embed_size,
                                            hidden_size,
                                            question_vec_size,
                                            vocab_size,
                                            start_idx)        
        self.answer_decoder = AnswerDecoder(self.embedding_table,
                                            image_local_feat_size,
                                            question_vec_size,
                                            embed_size,
                                            hidden_size,
                                            start_idx,
                                            vocab_size,
                                            dropout_prob)
        
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

    def forward(
        self,
        images,
        questions,
        question_lengths,
        answers=None,
        max_answer_length=None,
        mode='train',
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
        question_vectors = self.question_encoder(questions, question_lengths)
        
        # recover questions from vectors
        pred_questions = self.question_decoder(question_vectors, questions, question_lengths)

        # predict answers
        if mode == 'train':
            pred_answers = self.answer_decoder(local_feat, global_feat, question_vectors, answers=answers, mode=mode)
        else:
            pred_answers = self.answer_decoder(local_feat, global_feat, question_vectors, max_answer_length=max_answer_length, mode=mode)

        output = {
            'pred_answers': pred_answers,
            'pred_questions': pred_questions,
        }

        # auxiliary tasks (optional)
        
        if self.tags_aux_task:
            tags_logits = self.W_tags(global_feat)
            output['pred_tags'] = tags_logits
        
        if self.orien_aux_task:            
            if iuxray_foward:
                output['iuxray_pred_orientation'] = self.W_ori_iuxray(global_feat)
            if mimiccxr_foward:
                output['mimiccxr_pred_orientation'] = self.W_ori_mimiccxr(global_feat)

        return output