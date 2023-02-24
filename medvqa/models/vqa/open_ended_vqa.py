from torch import nn
from medvqa.models.nlp.text_encoder import BiLSTMBasedTextEncoder
from medvqa.models.nlp.text_decoder import LSTMBasedTextDecoder
from medvqa.models.vision.visual_modules import (
    MultiPurposeVisualModule,
    RawImageEncoding,
    VisualInputMode,
    does_include_image,
)
from medvqa.models.vqa.lstm_answer_decoder import LSTMAnswerDecoder
from medvqa.models.vqa.transformer_answer_decoder import TransformerAnswerDecoder
from medvqa.utils.constants import CHEXPERT_TASKS

class QuestionEncoding:
    BILSTM = 'bilstm'
    ONE_HOT = 'one-hot'

class AnswerDecoding:
    LSTM = 'lstm'
    TRANSFORMER  = 'transformer'

class OpenEndedVQA(MultiPurposeVisualModule):

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
                num_regions=None,
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
                classify_gender=False,
                classify_chexpert=False,
                classify_questions=False,
                classify_chest_imagenome=False,
                predict_bboxes_chest_imagenome=False,
                chest_imagenome_train_average_bbox_coords=None,
                n_medical_tags=None,
                n_questions=None,
                n_questions_aux_task=None,
                n_chest_imagenome_labels=None,
                chest_imagenome_bbox_hidden_size=None,
                chest_imagenome_bbox_regressor_version=None,
                use_mimiccxr=False,
                use_iuxray=False,
                use_chexpert=False,
                use_cxr14=False,
                use_vinbig=False,
                use_padchest=False,
                merge_findings=False,
                n_findings=None,
                # Other args
                chexpert_mode=None,
                dropout_prob=0,
                device=None,
                use_visual_module_only=False,
                **unused_kwargs,
                ):

        print('OpenEndedVQA():')

        print('  image_encoder_pretrained_weights_path', image_encoder_pretrained_weights_path)
        self.use_visual_module_only = use_visual_module_only

        self.chexpert_mode = chexpert_mode
        if chexpert_mode == CHEXPERT_TASKS.VQA:
            assert question_encoding == QuestionEncoding.ONE_HOT
        
        self.question_encoding = question_encoding
        self.answer_decoding = answer_decoding

        # Init MultiPurposeVisualModule components (for image encoder and auxiliary tasks)
        super().__init__(
            # Image Encoder kwargs
            visual_input_mode=visual_input_mode,
            raw_image_encoding=raw_image_encoding,
            image_local_feat_size=image_local_feat_size,
            freeze_image_encoder=freeze_image_encoder,
            image_encoder_pretrained_weights_path=image_encoder_pretrained_weights_path,
            imagenet_pretrained=imagenet_pretrained,
            mlp_in_dim=mlp_in_dim,
            mlp_out_dim=mlp_out_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            clip_version=clip_version,
            num_regions=num_regions,
            huggingface_model_name=huggingface_model_name,
            # Auxiliary tasks kwargs
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
            chest_imagenome_train_average_bbox_coords=chest_imagenome_train_average_bbox_coords,
            n_medical_tags=n_medical_tags,
            n_questions_aux_task=n_questions_aux_task,
            n_chest_imagenome_labels=n_chest_imagenome_labels,
            chest_imagenome_bbox_hidden_size=chest_imagenome_bbox_hidden_size,
            chest_imagenome_bbox_regressor_version=chest_imagenome_bbox_regressor_version,
            merge_findings=merge_findings,
            n_findings=n_findings,
        )

        if not self.use_visual_module_only:
            # Init vocab embedding table
            self.embedding_table = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_size,
                padding_idx=0,
            )

            # Init Question Encoder            
            self._init_question_encoder(question_encoding, question_hidden_size=question_hidden_size,
                            embed_size=embed_size, question_vec_size=question_vec_size,
                            vocab_size=vocab_size, start_idx=start_idx, n_questions=n_questions,
                            device=device)
            
            # Init Answer Decoder
            self._init_answer_decoder(image_local_feat_size, question_vec_size, embed_size,
                                answer_hidden_size, n_lstm_layers, start_idx, eos_idx, padding_idx, vocab_size,
                                transf_dec_nhead, transf_dec_dim_forward, transf_dec_num_layers, dropout_prob,
                                use_local_features=does_include_image(visual_input_mode))

        # Logging
        print(f'  n_questions = {n_questions}\n  n_questions_aux_task = {n_questions_aux_task}\n'
              f'  question_encoding = {question_encoding}\n  answer_decoding = {answer_decoding}\n'
              f'  visual_input_mode = {visual_input_mode}\n  name = {self.get_name()}')

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

    def get_name(self):
        vm_str = super().get_name() # visual module name
        if not self.use_visual_module_only:
            q_enc_str = 'bilstm' if self.question_encoding == QuestionEncoding.BILSTM else 'onehot'
            a_dec_str = 'lstm' if self.answer_decoding == AnswerDecoding.LSTM else 'transf'
            strings = [vm_str, q_enc_str, a_dec_str]
        else:
            strings = [vm_str]
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
        iuxray_forward=False,
        mimiccxr_forward=False,
        chexpert_forward=False,
        cxr14_forward=False,
        vinbig_forward=False,
        padchest_forward=False,
        device=None,
    ):
        # Visual Component
        output = super().forward(
            raw_images=raw_images,
            visual_features=visual_features,
            iuxray_forward=iuxray_forward,
            mimiccxr_forward=mimiccxr_forward,
            chexpert_forward=chexpert_forward,
            cxr14_forward=cxr14_forward,
            vinbig_forward=vinbig_forward,
            padchest_forward=padchest_forward,
        )

        if not self.use_visual_module_only:
            
            includes_questions = (
                (chexpert_forward and self.chexpert_mode == CHEXPERT_TASKS.VQA) or
                cxr14_forward or vinbig_forward or padchest_forward or iuxray_forward or mimiccxr_forward
            )

            if includes_questions:
                # encode questions
                if self.question_encoding == QuestionEncoding.BILSTM: # BiLSTM (verbose)
                    question_vectors = self.question_encoder(questions, question_lengths)
                elif self.question_encoding == QuestionEncoding.ONE_HOT: # one-hot (non-verbose)
                    question_vectors = self.question_encoder(questions)
                else:
                    raise ValueError(f'Unknown question encoding strategy {self.question_encoding}')
                
                # decode questions back from encoded vectors (if in verbose mode)
                if self.question_encoding == QuestionEncoding.BILSTM:
                    pred_questions = self.question_decoder(question_vectors, questions, question_lengths)
                    output['pred_questions'] = pred_questions            

                # predict answers
                if question_vectors is not None:
                    local_feat = output['local_feat']
                    global_feat = output['global_feat']
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