import torch
from torch import nn
from medvqa.models.nlp.text_decoder import TransformerTextDecoder
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule

class Image2ReportModel(MultiPurposeVisualModule):

    def __init__(self,
                # Vocab args
                vocab_size,
                start_idx,
                # Image Encoder args
                raw_image_encoding,
                image_local_feat_size,
                freeze_image_encoder,
                image_encoder_pretrained_weights_path,
                num_regions,
                yolov8_model_name_or_path,
                yolov8_model_alias,
                # Report Decoder args
                embedding_dim,
                transf_dec_nhead,
                transf_dec_dim_forward,
                transf_dec_num_layers,
                transf_dec_hidden_dim,
                # Auxiliary tasks args
                classify_gender,
                classify_chexpert,
                classify_chest_imagenome,
                predict_bboxes_chest_imagenome,
                n_chest_imagenome_labels,
                chest_imagenome_anatomy_to_labels,
                chest_imagenome_anatomy_group_to_labels,
                n_chest_imagenome_bboxes,
                chest_imagenome_mlc_version,
                chest_imagenome_mlc_hidden_size,
                # Other args
                dropout_prob,
                **unused_kwargs,
                ):
        print('Image2ReportModel')
        print('   vocab_size:', vocab_size)
        print('   start_idx:', start_idx)
        print('   raw_image_encoding:', raw_image_encoding)
        print('   image_local_feat_size:', image_local_feat_size)
        print('   freeze_image_encoder:', freeze_image_encoder)
        print('   image_encoder_pretrained_weights_path:', image_encoder_pretrained_weights_path)
        print('   num_regions:', num_regions)
        print('   yolov8_model_name_or_path:', yolov8_model_name_or_path)
        print('   yolov8_model_alias:', yolov8_model_alias)
        print('   embedding_dim:', embedding_dim)
        print('   transf_dec_nhead:', transf_dec_nhead)
        print('   transf_dec_dim_forward:', transf_dec_dim_forward)
        print('   transf_dec_num_layers:', transf_dec_num_layers)
        print('   transf_dec_hidden_dim:', transf_dec_hidden_dim)

        # Init MultiPurposeVisualModule components (for image encoder and auxiliary tasks)
        super().__init__(
            # Image Encoder kwargs
            raw_image_encoding=raw_image_encoding,
            image_local_feat_size=image_local_feat_size,
            freeze_image_encoder=freeze_image_encoder,
            image_encoder_pretrained_weights_path=image_encoder_pretrained_weights_path,
            num_regions=num_regions,
            yolov8_model_name_or_path=yolov8_model_name_or_path,
            yolov8_model_alias=yolov8_model_alias,
            # Auxiliary tasks kwargs
            classify_gender=classify_gender,
            classify_chexpert=classify_chexpert,
            classify_chest_imagenome=classify_chest_imagenome,
            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
            chest_imagenome_anatomy_to_labels=chest_imagenome_anatomy_to_labels,
            chest_imagenome_anatomy_group_to_labels=chest_imagenome_anatomy_group_to_labels,
            n_chest_imagenome_labels=n_chest_imagenome_labels,
            n_chest_imagenome_bboxes=n_chest_imagenome_bboxes,
            chest_imagenome_mlc_version=chest_imagenome_mlc_version,
            chest_imagenome_mlc_hidden_size=chest_imagenome_mlc_hidden_size,
        )

        # Word embedding table
        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # Transformer decoder (for report generation)
        self.report_decoder = TransformerTextDecoder(
            embedding_table=self.embedding_table,
            embed_size=embedding_dim,
            hidden_size=transf_dec_hidden_dim,
            nhead=transf_dec_nhead,
            dim_feedforward=transf_dec_dim_forward,
            num_layers=transf_dec_num_layers,
            start_idx=start_idx,
            vocab_size=vocab_size,
            dropout_prob=dropout_prob,
        )

        # For projecting image features into a input memory for the report decoder
        self.W_local_feat = nn.Linear(self.local_feat_size, transf_dec_hidden_dim)
        self.W_global_feat = nn.Linear(self.global_feat_size, transf_dec_hidden_dim)
    
    def _get_image_memory(self, local_feat, global_feat):
        # merge local and global features
        batch_size = global_feat.size(0)
        image_question_memory = torch.cat((
            self.W_local_feat(local_feat),
            self.W_global_feat(global_feat).view(batch_size, 1, -1),
        ), 1)
        return image_question_memory

    def get_name(self):
        return f'Image2ReportModel({super().get_name()}->{self.report_decoder.get_name()})'

    def forward(
        self,
        raw_images,
        mimiccxr_forward=False,
        device=None,
        reports=None,
        max_report_length=None,
        mode='train',
    ):
        # Visual Component
        output = super().forward(
            raw_images=raw_images,
            mimiccxr_forward=mimiccxr_forward,
            return_local_features=True,
            return_global_features=True,
        )
        
        # Report Decoder
        decoder_input_memory = self._get_image_memory(output['local_feat'], output['global_feat'])
        if mode == 'train':
            pred_reports = self.report_decoder(input_memory=decoder_input_memory, device=device,
                                            texts=reports, mode=mode)
        else:
            pred_reports = self.report_decoder(input_memory=decoder_input_memory, device=device,
                                            max_text_length=max_report_length, mode=mode)
        output['pred_reports'] = pred_reports

        return output