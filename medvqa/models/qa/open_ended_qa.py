import torch.nn as nn
from medvqa.models.nlp.question_encoder import QuestionEncoder_BiLSTM
from medvqa.models.nlp.question_decoder import QuestionDecoder
from medvqa.models.qa.answer_decoder import AnswerDecoder

class OpenEndedQA(nn.Module):

    def __init__(self, vocab_size, start_idx, embed_size, question_hidden_size, answer_hidden_size,
                 n_lstm_layers, question_vec_size, dropout_prob, device):
        super().__init__()
        self.name = 'oeqa(bilstm+lstm)'
        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=0,
        )
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
        self.answer_decoder = AnswerDecoder(self.embedding_table,
                                            question_vec_size,
                                            embed_size,
                                            answer_hidden_size,
                                            n_lstm_layers,
                                            start_idx,
                                            vocab_size,
                                            dropout_prob)

    def forward(
        self,
        questions,
        question_lengths,
        answers=None,
        max_answer_length=None,
        mode='train',
    ):
        # process questions
        question_vectors = self.question_encoder(questions, question_lengths)
        
        # recover questions from vectors
        pred_questions = self.question_decoder(question_vectors, questions, question_lengths)

        # predict answers
        if mode == 'train':
            pred_answers = self.answer_decoder(question_vectors, answers=answers, mode=mode)
        else:
            pred_answers = self.answer_decoder(question_vectors, max_answer_length=max_answer_length, mode=mode)

        output = {
            'pred_answers': pred_answers,
            'pred_questions': pred_questions,
        }

        return output