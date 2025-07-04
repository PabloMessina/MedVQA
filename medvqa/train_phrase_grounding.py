import  os
import argparse
import torch
import shlex
import logging
from medvqa.datasets.padchest.padchest_dataset_management import PadChestGRPhraseTrainer
from medvqa.utils.logging_utils import log_title, setup_logging
setup_logging() # Setup logging as early as possible
from medvqa.datasets.chexlocalize.chexlocalize_dataset_management import CheXlocalizePhraseGroundingTrainer
from medvqa.datasets.chexpert.chexpert_dataset_management import CheXpertPhraseGroundingTrainer, CheXpertTrainingMode
from medvqa.datasets.image_transforms_factory import create_image_transforms
from medvqa.datasets.iuxray.iuxray_phrase_grounding_dataset_management import IUXRayPhraseGroundingTrainer
from medvqa.datasets.mimiccxr import MIMICCXR_ImageSizeModes
from medvqa.datasets.mimiccxr.mimiccxr_phrase_grounding_dataset_management import MIMICCXR_PhraseGroundingTrainer
from medvqa.datasets.ms_cxr import MS_CXR_TrainingMode
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBigPhraseTaskMode, VinBigPhraseTrainer, VinBigTrainingMode
from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.phrase_grounding.phrase_grounder import PhraseGrounder, PhraseGroundingMode
from medvqa.models.vision.visual_modules import comes_with_positional_encoding, inject_mean_std_for_image_normalization
from medvqa.models.vqa.open_ended_vqa import RawImageEncoding
from medvqa.training.utils import append_metric_name, run_common_boilerplate_code_and_start_training
from medvqa.utils.constants import DATASET_NAMES
from medvqa.utils.common import WORKSPACE_DIR, DictWithDefault
from medvqa.metrics import (
    attach_condition_aware_accuracy,
    attach_condition_aware_bbox_cnr,
    attach_condition_aware_bbox_iou_class_agnostic,
    attach_condition_aware_bbox_iou_open_class,
    attach_condition_aware_class_averaged_prc_auc,
    attach_condition_aware_loss,
    attach_condition_aware_prc_auc,
    attach_condition_aware_segmask_iou,
)
from medvqa.models.checkpoint import load_metadata
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.files_utils import get_checkpoint_folder_path
from medvqa.training.phrase_grounding import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    get_phrase_grounding_collate_batch_fn,
    multi_cyclic_dataloaders_generator,
)
from medvqa.metrics.utils import get_merge_metrics_fn
from medvqa.utils.metrics_utils import average_ignoring_nones_and_nans

logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # --- Required arguments

    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs the model will be trained')
    parser.add_argument('--batches_per_epoch', type=int, required=True, help='Number of batches per epoch')
    parser.add_argument('--max_images_per_batch', type=int, required=True, help='Max number of images per batch')
    parser.add_argument('--max_phrases_per_batch', type=int, required=True, help='Max number of phrases per batch')
    parser.add_argument('--max_phrases_per_image', type=int, required=True, help='Max number of phrases per image')
    parser.add_argument('--val_batch_size_factor', type=float, required=True, help='Factor to multiply batch size for test dataloader')

    # --- Other arguments
    
    parser.add_argument('--checkpoint_folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')

    # Model arguments
    parser.add_argument('--pretrained_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--pretrained_checkpoint_folder_paths', type=str, nargs='+', default=None)
    parser.add_argument('--freeze_image_encoder', action='store_true')
    parser.add_argument('--raw_image_encoding', type=str, default=RawImageEncoding.YOLOV8)
    parser.add_argument('--huggingface_model_name', type=str, default=None)
    parser.add_argument('--num_regions', type=int, default=None)
    parser.add_argument('--image_local_feat_size', type=int, default=None)
    parser.add_argument('--image_encoder_pretrained_weights_path', type=str, default=None)
    parser.add_argument('--image_encoder_dropout_p', type=float, default=0)
    parser.add_argument('--yolov8_model_name_or_path', type=str, default=None)
    parser.add_argument('--yolov8_model_alias', type=str, default=None)
    parser.add_argument('--phrase_embedding_size', type=int, default=None)
    parser.add_argument('--regions_width', type=int, default=None)
    parser.add_argument('--regions_height', type=int, default=None)
    parser.add_argument('--qkv_size', type=int, default=None)
    parser.add_argument('--phrase_grounding_mode', type=str, default=None, choices=PhraseGroundingMode.get_choices())
    parser.add_argument('--phrase_classifier_hidden_size', type=int, default=None)
    parser.add_argument('--transf_d_model', type=int, default=None)
    parser.add_argument('--transf_nhead', type=int, default=None)
    parser.add_argument('--transf_dim_feedforward', type=int, default=None)
    parser.add_argument('--transf_dropout', type=int, default=0)
    parser.add_argument('--transf_num_layers', type=int, default=None)
    parser.add_argument('--visual_feature_proj_size', type=int, default=None)
    parser.add_argument('--visual_grounding_hidden_size', type=int, default=None)
    parser.add_argument('--phrase_mlp_hidden_dims', nargs='+', type=int, default=None)
    parser.add_argument('--predict_global_alignment', action='store_true')
    parser.add_argument('--alignment_proj_size', type=int, default=None)
    parser.add_argument('--yolov11_model_name_or_path', type=str, default=None)
    parser.add_argument('--yolov11_model_alias', type=str, default=None)
    parser.add_argument('--bbox_format', type=str, default='cxcywh', choices=['xyxy', 'cxcywh'])
    parser.add_argument('--predict_relative_bbox_coords', action='store_true')
    
    # Optimization arguments
    parser.add_argument('--optimizer_name', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='reduce-lr-on-plateau')
    parser.add_argument('--lr_decay', type=float, default=0.76, help='Learning rate decay')
    parser.add_argument('--lr_decay_patience', type=int, default=2, help='Learning rate decay patience')
    parser.add_argument('--warmup_and_decay_args', type=str, default=None)
    parser.add_argument('--warmup_and_cosine_args', type=str, default=None)
    parser.add_argument('--warmup_decay_and_cyclic_decay_args', type=str, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='For gradient accumulation')
    parser.add_argument('--override_lr', action='store_true')
    parser.add_argument('--max_grad_norm', type=float, default=None, help='Max gradient norm')
    # Loss weights
    parser.add_argument('--attention_supervision_loss_weight', type=float, default=1.0)
    parser.add_argument('--phrase_classifier_loss_weight', type=float, default=1.0)
    parser.add_argument('--foreground_loss_weight', type=float, default=1.0)
    parser.add_argument('--background_loss_weight', type=float, default=1.0)
    parser.add_argument('--focal_loss_weight', type=float, default=1.0)
    parser.add_argument('--bce_loss_weight', type=float, default=1.0)
    parser.add_argument('--wbce_loss_weight', type=float, default=1.0)
    # Other loss arguments
    parser.add_argument('--binary_multilabel_classif_loss_name', type=str, default='bce')
    parser.add_argument('--use_weighted_phrase_classifier_loss', action='store_true')
    parser.add_argument('--cluster_and_label_weights_for_facts_filepath', type=str, default=None)
    parser.add_argument('--use_attention_regularization_loss', action='store_true')
    parser.add_argument('--use_contrastive_phrase_grounding_loss', action='store_true')
    parser.add_argument('--nt_xent_temperature', type=float, default=0.1)

    # Dataset and dataloading arguments
    parser.add_argument('--num_train_workers', type=int, default=0, help='Number of workers for train dataloader')
    parser.add_argument('--num_val_workers', type=int, default=0, help='Number of workers for test dataloader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_train_data_augmentations', action='store_true',
                        help='Whether to use image augmentations for training')
    parser.add_argument('--image_size', nargs='+', type=int, default=(416, 416))
    parser.add_argument('--dicom_id_to_pos_neg_facts_filepath', type=str, default=None)
    parser.add_argument('--iuxray_image_id_to_pos_neg_facts_filepath', type=str, default=None)
    parser.add_argument('--mscxr_phrase2embedding_filepath', type=str, default=None)
    parser.add_argument('--chest_imagenome_augmented_phrase_groundings_filepath', type=str, default=None)
    parser.add_argument('--chest_imagenome_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--chest_imagenome_bbox_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--vinbig_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--chexlocalize_class_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--chexpert_class_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--padchest_gr_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--mimiccxr_exclude_noisy_images', action='store_true')
    parser.add_argument('--mimiccxr_facts_weight', type=float, default=1.0)
    parser.add_argument('--chest_imagenome_pg_weight', type=float, default=1.0)
    parser.add_argument('--chest_imagenome_alg_weight', type=float, default=1.0)
    parser.add_argument('--mscxr_weight', type=float, default=1.0)
    parser.add_argument('--cxrlt2024_weight', type=float, default=1.0)
    parser.add_argument('--vinbig_weight', type=float, default=1.0)
    parser.add_argument('--chexlocalize_weight', type=float, default=1.0)
    parser.add_argument('--chexpert_weight', type=float, default=1.0)
    parser.add_argument('--iuxray_weight', type=float, default=1.0)
    parser.add_argument('--padchest_gr_weight', type=float, default=1.0)
    parser.add_argument('--pos_area_prior', type=float, default=0.4, help='Prior for positive area')
    parser.add_argument('--neg_area_prior', type=float, default=0.0, help='Prior for negative area')
    parser.add_argument('--use_mimiccxr_facts_for_train', action='store_true')
    parser.add_argument('--use_mimiccxr_facts_for_test', action='store_true')
    parser.add_argument('--use_mscxr_for_train', action='store_true')
    parser.add_argument('--use_mscxr_for_val', action='store_true')
    parser.add_argument('--use_chest_imagenome_for_train', action='store_true')
    parser.add_argument('--use_chest_imagenome_for_val', action='store_true')
    parser.add_argument('--use_vinbig_for_train', action='store_true')
    parser.add_argument('--use_vinbig_for_test', action='store_true')
    parser.add_argument('--use_chexlocalize_for_train', action='store_true')
    parser.add_argument('--use_chexlocalize_for_test', action='store_true')
    parser.add_argument('--use_chexpert_for_train', action='store_true')
    parser.add_argument('--use_chexpert_for_test', action='store_true')
    parser.add_argument('--use_iuxray_for_train', action='store_true')
    parser.add_argument('--use_iuxray_for_test', action='store_true')
    parser.add_argument('--use_padchest_gr_for_train', action='store_true')
    parser.add_argument('--use_padchest_gr_for_val', action='store_true')
    parser.add_argument('--use_cxrlt2024_challenge_split', action='store_true')
    parser.add_argument('--use_cxrlt2024_custom_labels', action='store_true')
    parser.add_argument('--use_cxrlt2024_official_labels', action='store_true')
    parser.add_argument('--use_all_cxrlt2024_official_labels_for_training', action='store_true')
    parser.add_argument('--vinbig_training_data_mode', type=str, default=VinBigTrainingMode.TRAIN.value,
                        choices=VinBigTrainingMode.get_choices())
    parser.add_argument('--vinbig_task_mode', type=str, default=VinBigPhraseTaskMode.GROUNDING.value,
                        choices=VinBigPhraseTaskMode.get_choices())
    parser.add_argument('--chexpert_training_data_mode', type=str, default=CheXpertTrainingMode.ALL.value,
                        choices=CheXpertTrainingMode.get_choices())
    parser.add_argument('--mimiccxr_balance_long_middle_short_tail', action='store_true')
    parser.add_argument('--mimiccxr_long_middle_short_tail_thresholds', nargs=2, type=float, default=(0.02, 0.05))
    parser.add_argument('--mimiccxr_report_fact_nli_integrated_data_filepath', type=str, default=None)
    parser.add_argument('--mimiccxr_use_interpret_cxr_challenge_split', action='store_true')
    parser.add_argument('--mimiccxr_interpret_cxr_challenge_split_filepath', type=str, default=None)
    parser.add_argument('--iuxray_use_interpret_cxr_challenge_split', action='store_true')
    parser.add_argument('--iuxray_interpret_cxr_challenge_split_filepath', type=str, default=None)
    parser.add_argument('--chexpert_use_interpret_cxr_challenge_split', action='store_true')
    parser.add_argument('--chexpert_interpret_cxr_challenge_split_filepath', type=str, default=None)
    parser.add_argument('--chexlocalize_use_interpret_cxr_challenge_split', action='store_true')
    parser.add_argument('--chexlocalize_interpret_cxr_challenge_split_filepath', type=str, default=None)
    parser.add_argument('--cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath', type=str, default=None)
    parser.add_argument('--cxrlt2024_official_training_labels_for_fact_classification_filepath', type=str, default=None)
    parser.add_argument('--cxrlt2024_do_balanced_sampling', action='store_true')
    parser.add_argument('--do_visual_grounding_with_bbox_regression', action='store_true')
    parser.add_argument('--replace_phrase_embeddings_with_random_vectors', action='store_true')
    parser.add_argument('--use_vinbig_with_modified_labels', action='store_true')
    parser.add_argument('--mscxr_training_data_mode', type=str, default=MS_CXR_TrainingMode.TRAIN.value,
                        choices=MS_CXR_TrainingMode.get_choices())
    parser.add_argument('--mscxr_do_grounding_only', action='store_true')
    parser.add_argument('--padchest_gr_training_split', type=str, default='train', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--skip_nms', action='store_true')
    
    # Checkpoint saving arguments
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no_save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    parser.add_argument('--override_metric_weights', nargs="*", help="Override metric weights (format: metric=value, e.g., vbg_prc_auc=7.5)")
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = DictWithDefault(default=1.0) # Default weight is 1.0
# Most important metrics for MICCAI CXR-LT 2024 challenge
_METRIC_WEIGHTS['cxrlt2024o_prc_auc'] = 5.0 
_METRIC_WEIGHTS['cxrlt2024c_prc_auc'] = 5.0

_METRIC_WEIGHTS['vbg_prc_auc'] = 10.0 # Assign more weight to classification metrics for VinBig

def _metric_getter(metrics_dict, key):
    metric = metrics_dict[key]
    if key.endswith('_loss'):
        return 1 / (1 + metric) # convert loss to score
    if isinstance(metric, list):
        return average_ignoring_nones_and_nans(metric)
    return metric

def train_model(
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_trainer_kwargs,
    vinbig_trainer_kwargs,
    padchestgr_trainer_kwargs,
    chexlocalize_trainer_kwargs,
    chexpert_trainer_kwargs,
    iuxray_trainer_kwargs,
    dataloading_kwargs,
    # collate_batch_fn_kwargs,
    train_image_transform_kwargs,
    val_image_transform_kwargs,
    trainer_engine_kwargs,
    validator_engine_kwargs,
    metrics_kwargs,
    epochs,
    batches_per_epoch,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    val_batch_size_factor,
    num_train_workers,
    num_val_workers,
    device='cuda',
    checkpoint_folder_path=None,
    save=True,
    override_lr=False,
    debug=False,
):  
    # Pull out some args from kwargs
    use_mimiccxr_facts_for_train = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_facts_for_train', False)
    use_mimiccxr_facts_for_test = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_facts_for_test', False)
    use_mscxr_for_train = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_mscxr_for_train', False)
    use_mscxr_for_val = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_mscxr_for_val', False)
    mscxr_do_grounding_only = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('mscxr_do_grounding_only', False)
    use_chest_imagenome_for_train = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_chest_imagenome_for_train', False)
    use_chest_imagenome_for_val = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_chest_imagenome_for_val', False)
    use_cxrlt2024_challenge_split = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_cxrlt2024_challenge_split', False)
    use_cxrlt2024_custom_labels = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_cxrlt2024_custom_labels', False)
    use_cxrlt2024_official_labels = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_cxrlt2024_official_labels', False)
    use_vinbig_for_train = vinbig_trainer_kwargs is not None and vinbig_trainer_kwargs.get('use_training_set', False)
    use_vinbig_for_test = vinbig_trainer_kwargs is not None and vinbig_trainer_kwargs.get('use_validation_set', False)
    use_padchest_gr_for_train = padchestgr_trainer_kwargs is not None and padchestgr_trainer_kwargs.get('use_training_set', False)
    use_padchest_gr_for_val = padchestgr_trainer_kwargs is not None and padchestgr_trainer_kwargs.get('use_validation_set', False)
    use_chexlocalize_for_train = chexlocalize_trainer_kwargs is not None and chexlocalize_trainer_kwargs.get('use_training_set', False)
    use_chexlocalize_for_test = chexlocalize_trainer_kwargs is not None and chexlocalize_trainer_kwargs.get('use_validation_set', False)
    use_chexpert_for_train = chexpert_trainer_kwargs is not None and chexpert_trainer_kwargs.get('use_training_set', False)
    use_chexpert_for_test = chexpert_trainer_kwargs is not None and chexpert_trainer_kwargs.get('use_validation_set', False)
    use_iuxray_for_train = iuxray_trainer_kwargs is not None and iuxray_trainer_kwargs.get('do_train', False)
    use_iuxray_for_test = iuxray_trainer_kwargs is not None and iuxray_trainer_kwargs.get('do_test', False)
    use_attention_regularization_loss = trainer_engine_kwargs['use_attention_regularization_loss']
    use_contrastive_phrase_grounding_loss = trainer_engine_kwargs['use_contrastive_phrase_grounding_loss']
    use_global_alignment_contrastive_loss = trainer_engine_kwargs['use_global_image_phrase_contrastive_loss']
    vinbig_task_mode = vinbig_trainer_kwargs.get('task_mode') if vinbig_trainer_kwargs is not None else None
    skip_nms = validator_engine_kwargs.get('skip_nms', False)
    bbox_format = model_kwargs['bbox_format']
    regions_height = model_kwargs['regions_height']
    regions_width = model_kwargs['regions_width']

    # Sanity checks
    if use_chest_imagenome_for_val:
        assert use_chest_imagenome_for_train
    if use_mscxr_for_val:
        assert use_mimiccxr_facts_for_train or use_mscxr_for_train or use_chest_imagenome_for_train
    if use_chexpert_for_test:
        assert use_chexpert_for_train
    if use_vinbig_for_test:
        assert use_vinbig_for_train
    if use_iuxray_for_test:
        assert use_iuxray_for_train

    # Update metric weights
    for metric_name, weight in metrics_kwargs.items():
        _METRIC_WEIGHTS[metric_name] = weight

    if use_mimiccxr_facts_for_train:
        if dataloading_kwargs['mimiccxr_facts_weight'] == 0:
            logger.warning('use_mimiccxr_facts_for_train is True but mimiccxr_facts_weight is 0', bold=True)
            use_mimiccxr_facts_for_train = False
    if use_chest_imagenome_for_train:
        if (dataloading_kwargs['chest_imagenome_pg_weight'] == 0 or
            dataloading_kwargs['chest_imagenome_alg_weight'] == 0):
            logger.warning('use_chest_imagenome_for_train is True but chest_imagenome_pg_weight'
                           ' or chest_imagenome_alg_weight are 0', bold=True)
            use_chest_imagenome_for_train = False
    if use_mscxr_for_train:
        if dataloading_kwargs['mscxr_weight'] == 0:
            logger.warning('use_mscxr_for_train is True but mscxr_weight is 0', bold=True)
            use_mscxr_for_train = False
    if use_cxrlt2024_challenge_split:
        if dataloading_kwargs['cxrlt2024_weight'] == 0:
            logger.warning('use_cxrlt2024_challenge_split is True but cxrlt2024_weight is 0', bold=True)
            use_cxrlt2024_challenge_split = False
    if use_vinbig_for_train:
        if dataloading_kwargs['vinbig_weight'] == 0:
            logger.warning('use_vinbig_for_train is True but vinbig_weight is 0', bold=True)
            use_vinbig_for_train = False
    if use_padchest_gr_for_train:
        if dataloading_kwargs['padchest_gr_weight'] == 0:
            logger.warning('use_padchest_gr is True but padchest_gr_weight is 0', bold=True)
            use_padchest_gr = False
    if use_chexlocalize_for_train:
        if dataloading_kwargs['chexlocalize_weight'] == 0:
            logger.warning('use_chexlocalize_for_train is True but chexlocalize_weight is 0', bold=True)
            use_chexlocalize_for_train = False
    if use_chexpert_for_train:
        if dataloading_kwargs['chexpert_weight'] == 0:
            logger.warning('use_chexpert_for_train is True but chexpert_weight is 0', bold=True)
            use_chexpert_for_train = False
    if use_iuxray_for_train:
        if dataloading_kwargs['iuxray_weight'] == 0:
            logger.warning('use_iuxray_for_train is True but iuxray_weight is 0', bold=True)
            use_iuxray_for_train = False

    use_mimiccxr = use_mimiccxr_facts_for_train or use_mscxr_for_val or use_mscxr_for_train or\
                   use_chest_imagenome_for_train or use_chest_imagenome_for_val or use_cxrlt2024_challenge_split
    
    use_chexlocalize = use_chexlocalize_for_train or use_chexlocalize_for_test

    use_vinbig = use_vinbig_for_train or use_vinbig_for_test

    use_padchest_gr = use_padchest_gr_for_train or use_padchest_gr_for_val

    use_chexpert = use_chexpert_for_train or use_chexpert_for_test

    use_iuxray = use_iuxray_for_train or use_iuxray_for_test

    assert sum([use_mimiccxr_facts_for_train, use_chest_imagenome_for_train, use_mscxr_for_train,
                use_cxrlt2024_challenge_split, use_vinbig_for_train, use_chexlocalize_for_train,
                use_chexpert_for_train, use_iuxray_for_train, use_cxrlt2024_challenge_split]) > 0

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    logger.info(f'device = {device}')

    # Create model
    log_title(logger, 'Creating instance of PhraseGrounder')
    model = PhraseGrounder(**model_kwargs, device=device)
    model = model.to(device)

    # Optimizer
    log_title(logger, 'Defining optimizer')
    optimizer = create_optimizer(params=model.parameters(), **optimizer_kwargs)

    # Learning rate scheduler
    log_title(logger, 'Defining scheduler')
    lr_scheduler, update_lr_batchwise = create_lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)

    # Create trainer and validator engines
    log_title(logger, 'Creating trainer and validator engines')
    if model_kwargs['raw_image_encoding'] == RawImageEncoding.YOLOV8:
        model_for_yolov8 = model.raw_image_encoder
    else:
        model_for_yolov8 = None
    trainer_engine = get_engine(model=model, optimizer=optimizer, device=device,
        update_lr_batchwise=update_lr_batchwise, lr_scheduler=lr_scheduler,
        model_for_yolov8=model_for_yolov8, **trainer_engine_kwargs)
    validator_engine = get_engine(model=model, device=device, **validator_engine_kwargs)

    # Create CheXLocalize trainer
    if use_chexlocalize:
        log_title(logger, 'Creating CheXLocalize Phrase Grounding Trainer')
        chexlocalize_trainer = CheXlocalizePhraseGroundingTrainer(
            train_image_transform=create_image_transforms(**train_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE]),
            val_image_transform=create_image_transforms(**val_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE]),
            collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cl']),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            test_batch_size_factor=val_batch_size_factor,
            num_train_workers=num_train_workers,
            num_val_workers=num_val_workers,
            **chexlocalize_trainer_kwargs,
        )

    # Create CheXpert trainer
    if use_chexpert:
        log_title(logger, 'Creating CheXpert Phrase Grounding Trainer')
        chexpert_trainer = CheXpertPhraseGroundingTrainer(
            train_image_transform=create_image_transforms(**train_image_transform_kwargs[DATASET_NAMES.CHEXPERT]),
            val_image_transform=create_image_transforms(**val_image_transform_kwargs[DATASET_NAMES.CHEXPERT]),
            collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['chxp']),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            test_batch_size_factor=val_batch_size_factor,
            num_train_workers=num_train_workers,
            num_val_workers=num_val_workers,
            **chexpert_trainer_kwargs,
        )

    # Create VINBIG trainer
    if use_vinbig:
        log_title(logger, 'Creating VinBig Phrase Grounding Trainer')
        vinbig_trainer = VinBigPhraseTrainer(
            train_image_transform=create_image_transforms(**train_image_transform_kwargs[DATASET_NAMES.VINBIG]),
            val_image_transform=create_image_transforms(**val_image_transform_kwargs[DATASET_NAMES.VINBIG]),            
            max_images_per_batch=max_images_per_batch,
            val_batch_size_factor=val_batch_size_factor,
            num_train_workers=num_train_workers,
            num_val_workers=num_val_workers,
            **vinbig_trainer_kwargs,
        )

    # Create PadChest-GR trainer
    if use_padchest_gr:
        log_title(logger, 'Creating PadChest-GR Phrase Grounding Trainer')
        padchestgr_trainer = PadChestGRPhraseTrainer(
            max_images_per_batch=max_images_per_batch,
            val_batch_size_factor=val_batch_size_factor,
            num_train_workers=num_train_workers,
            num_val_workers=num_val_workers,
            **padchestgr_trainer_kwargs,
        )

    # Create MIMIC-CXR trainer
    if use_mimiccxr:
        log_title(logger, 'Creating MIMIC-CXR Phrase Grounding Trainer')
        # if use_mimiccxr_facts_for_train or use_mimiccxr_facts_for_test:
        #     fact_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['mimfg'])
        # else:
        #     fact_grounding_collate_batch_fn = None
        # if use_mscxr_for_train or use_mscxr_for_val:
        #     mscxr_phrase_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['mscxr'])
        # else:
        #     mscxr_phrase_grounding_collate_batch_fn = None
        # if use_chest_imagenome_for_train or use_chest_imagenome_gold_for_test:
        #     bbox_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cibg'])
        # else:
        #     bbox_grounding_collate_batch_fn = None
        # if use_cxrlt2024_custom_labels:
        #     cxrlt2024_image_phrase_classifier_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cxrlt2024c'])
        # else:
        #     cxrlt2024_image_phrase_classifier_collate_batch_fn = None
        # if use_cxrlt2024_official_labels:
        #     cxrlt2024_multilabel_classifier_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cxrlt2024o'])
        # else:
        #     cxrlt2024_multilabel_classifier_collate_batch_fn = None
        mimiccxr_trainer = MIMICCXR_PhraseGroundingTrainer(
            train_image_transform = create_image_transforms(**train_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            test_image_transform = create_image_transforms(**val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            max_phrases_per_image=max_phrases_per_image,
            test_batch_size_factor=val_batch_size_factor,
            # fact_grounding_collate_batch_fn=fact_grounding_collate_batch_fn,
            # bbox_grounding_collate_batch_fn=bbox_grounding_collate_batch_fn,
            # cxrlt2024_image_phrase_classifier_collate_batch_fn=cxrlt2024_image_phrase_classifier_collate_batch_fn,
            # cxrlt2024_multilabel_classifier_collate_batch_fn=cxrlt2024_multilabel_classifier_collate_batch_fn,
            # mscxr_phrase_grounding_collate_batch_fn=mscxr_phrase_grounding_collate_batch_fn,
            num_train_workers=num_train_workers,
            num_test_workers=num_val_workers,
            **mimiccxr_trainer_kwargs,
        )

    # Create IU X-Ray trainer
    if use_iuxray:
        log_title(logger, 'Creating IU X-Ray Phrase Grounding Trainer')
        iuxray_trainer = IUXRayPhraseGroundingTrainer(
            train_image_transform=create_image_transforms(**train_image_transform_kwargs[DATASET_NAMES.IUXRAY]),
            test_image_transform=create_image_transforms(**val_image_transform_kwargs[DATASET_NAMES.IUXRAY]),
            collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['iufg']),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            max_phrases_per_image=max_phrases_per_image,
            test_batch_size_factor=val_batch_size_factor,
            num_train_workers=num_train_workers,
            num_test_workers=num_val_workers,
            **iuxray_trainer_kwargs,
        )

    if debug: # if debugging
        output = {}
        if use_mimiccxr:
            output['mimiccxr_trainer'] = mimiccxr_trainer
        if use_vinbig:
            output['vinbig_trainer'] = vinbig_trainer
        if use_padchest_gr:
            output['padchestgr_trainer'] = padchestgr_trainer
        if use_chexlocalize:
            output['chexlocalize_trainer'] = chexlocalize_trainer
        if use_chexpert:
            output['chexpert_trainer'] = chexpert_trainer
        if use_iuxray:
            output['iuxray_trainer'] = iuxray_trainer
        return output

    # Create complex dataloaders
    log_title(logger, 'Creating dataloaders')
    
    _train_weights = []
    _train_dataloaders = []
    _val_dataloaders = []
    _dataset_names = []

    if use_mimiccxr_facts_for_train:
        _dataset_names.append('mim-facts')
        _train_weights.append(dataloading_kwargs['mimiccxr_facts_weight'])
        _train_dataloaders.append(mimiccxr_trainer.train_fact_dataloader)
        logger.info(f'len(mimiccxr_trainer.train_fact_dataloader) = {len(mimiccxr_trainer.train_fact_dataloader)}')

    if use_mimiccxr_facts_for_test:
        _val_dataloaders.append(mimiccxr_trainer.test_fact_dataloader)
        logger.info(f'len(mimiccxr_trainer.test_fact_dataloader) = {len(mimiccxr_trainer.test_fact_dataloader)}')

    if use_mscxr_for_train:
        _dataset_names.append('mscxr')
        _train_weights.append(dataloading_kwargs['mscxr_weight'])
        _train_dataloaders.append(mimiccxr_trainer.mscxr_train_dataloader)
        logger.info(f'len(mimiccxr_trainer.mscxr_train_dataloader) = {len(mimiccxr_trainer.mscxr_train_dataloader)}')

    if use_mscxr_for_val:
        _val_dataloaders.append(mimiccxr_trainer.mscxr_val_dataloader)
        logger.info(f'len(mimiccxr_trainer.mscxr_val_dataloader) = {len(mimiccxr_trainer.mscxr_val_dataloader)}')

    if use_chest_imagenome_for_train:
        _dataset_names.append('chst-img-alg')
        _train_weights.append(dataloading_kwargs['chest_imagenome_alg_weight'])
        _train_dataloaders.append(mimiccxr_trainer.chest_imagenome_alg_train_dataloader)
        logger.info(f'len(mimiccxr_trainer.chest_imagenome_alg_train_dataloader) = {len(mimiccxr_trainer.chest_imagenome_alg_train_dataloader)}')

        _dataset_names.append('chst-img-pg')
        _train_weights.append(dataloading_kwargs['chest_imagenome_pg_weight'])
        _train_dataloaders.append(mimiccxr_trainer.chest_imagenome_pg_train_dataloader)
        logger.info(f'len(mimiccxr_trainer.chest_imagenome_pg_train_dataloader) = {len(mimiccxr_trainer.chest_imagenome_pg_train_dataloader)}')

    if use_chest_imagenome_for_val:
        _val_dataloaders.append(mimiccxr_trainer.chest_imagenome_alg_val_dataloader)
        logger.info(f'len(mimiccxr_trainer.chest_imagenome_alg_val_dataloader) = {len(mimiccxr_trainer.chest_imagenome_alg_val_dataloader)}')
        
        _val_dataloaders.append(mimiccxr_trainer.chest_imagenome_pg_val_dataloader)
        logger.info(f'len(mimiccxr_trainer.chest_imagenome_pg_val_dataloader) = {len(mimiccxr_trainer.chest_imagenome_pg_val_dataloader)}')

    if use_cxrlt2024_challenge_split or use_cxrlt2024_custom_labels:
        assert use_cxrlt2024_custom_labels or use_cxrlt2024_official_labels
        wo = 0.5 * use_cxrlt2024_official_labels # 50% for official labels
        wc = 0.5 * use_cxrlt2024_custom_labels # 50% for custom labels
        wt = wo + wc
        wo, wc = wo / wt, wc / wt # normalize weights
        assert abs(wo + wc - 1) < 1e-6 # check if normalized weights sum to 1
        wo *= dataloading_kwargs['cxrlt2024_weight'] # scale weights
        wc *= dataloading_kwargs['cxrlt2024_weight'] # scale weights
        if use_cxrlt2024_official_labels:
            _dataset_names.append('cxrlt2024(Off)')
            _train_weights.append(wo) # weight for official labels
            _train_dataloaders.append(mimiccxr_trainer.cxrlt2024_official_train_dataloader)
            logger.info(f'len(mimiccxr_trainer.cxrlt2024_official_train_dataloader) = {len(mimiccxr_trainer.cxrlt2024_official_train_dataloader)}')
            if not mimiccxr_trainer.use_all_cxrlt2024_official_labels_for_training: # if not all official labels are used for training
                _val_dataloaders.append(mimiccxr_trainer.cxrlt2024_official_val_dataloader)
                logger.info(f'len(mimiccxr_trainer.cxrlt2024_official_val_dataloader) = {len(mimiccxr_trainer.cxrlt2024_official_val_dataloader)}')
        if use_cxrlt2024_custom_labels:
            _dataset_names.append('cxrlt2024(GPT4)')
            _train_weights.append(wc) # weight for custom labels
            _train_dataloaders.append(mimiccxr_trainer.cxrlt2024_custom_train_dataloader)
            logger.info(f'len(mimiccxr_trainer.cxrlt2024_custom_train_dataloader) = {len(mimiccxr_trainer.cxrlt2024_custom_train_dataloader)}')
            if not use_cxrlt2024_official_labels or mimiccxr_trainer.use_all_cxrlt2024_official_labels_for_training:
                _val_dataloaders.append(mimiccxr_trainer.cxrlt2024_custom_dev_dataloader)
                logger.info(f'len(mimiccxr_trainer.cxrlt2024_custom_dev_dataloader) = {len(mimiccxr_trainer.cxrlt2024_custom_dev_dataloader)}')

    if use_vinbig_for_train:
        _dataset_names.append('vinbig')
        _train_weights.append(dataloading_kwargs['vinbig_weight'])
        _train_dataloaders.append(vinbig_trainer.train_dataloader)
        logger.info(f'len(vinbig_trainer.train_dataloader) = {len(vinbig_trainer.train_dataloader)}')

    if use_vinbig_for_test:
        _val_dataloaders.append(vinbig_trainer.val_dataloader)
        logger.info(f'len(vinbig_trainer.val_dataloader) = {len(vinbig_trainer.val_dataloader)}')

    if use_padchest_gr_for_train:
        _dataset_names.append('padchest-gr')
        _train_weights.append(dataloading_kwargs['padchest_gr_weight'])
        _train_dataloaders.append(padchestgr_trainer.train_dataloader)
        logger.info(f'len(padchestgr_trainer.train_dataloader) = {len(padchestgr_trainer.train_dataloader)}')

    if use_padchest_gr_for_val:
        _val_dataloaders.append(padchestgr_trainer.val_dataloader)
        logger.info(f'len(padchestgr_trainer.val_dataloader) = {len(padchestgr_trainer.val_dataloader)}')

    if use_chexlocalize_for_train:
        _dataset_names.append('chexloc')
        _train_weights.append(dataloading_kwargs['chexlocalize_weight'])
        _train_dataloaders.append(chexlocalize_trainer.train_dataloader)
        # _train_dataloaders.append(chexlocalize_trainer.val_dataloader) # for debugging
        logger.info(f'len(chexlocalize_trainer.train_dataloader) = {len(chexlocalize_trainer.train_dataloader)}')

    if use_chexlocalize_for_test:
        _val_dataloaders.append(chexlocalize_trainer.val_dataloader)
        logger.info(f'len(chexlocalize_trainer.val_dataloader) = {len(chexlocalize_trainer.val_dataloader)}')

    if use_chexpert_for_train:
        _dataset_names.append('chxp')
        _train_weights.append(dataloading_kwargs['chexpert_weight'])
        _train_dataloaders.append(chexpert_trainer.train_dataloader)
        logger.info(f'len(chexpert_trainer.train_dataloader) = {len(chexpert_trainer.train_dataloader)}')

    if use_chexpert_for_test:
        _val_dataloaders.append(chexpert_trainer.val_dataloader)
        logger.info(f'len(chexpert_trainer.val_dataloader) = {len(chexpert_trainer.val_dataloader)}')

    if use_iuxray_for_train:
        _dataset_names.append('iuxray')
        _train_weights.append(dataloading_kwargs['iuxray_weight'])
        _train_dataloaders.append(iuxray_trainer.train_dataloader)
        logger.info(f'len(iuxray_trainer.train_dataloader) = {len(iuxray_trainer.train_dataloader)}')

    if use_iuxray_for_test:
        _val_dataloaders.append(iuxray_trainer.test_dataloader)
        logger.info(f'len(iuxray_trainer.test_dataloader) = {len(iuxray_trainer.test_dataloader)}')
    
    assert len(_train_dataloaders) > 0
    assert len(_val_dataloaders) > 0
    assert len(_train_dataloaders) == len(_train_weights)
    logger.info(f'len(_train_dataloaders) = {len(_train_dataloaders)}')
    logger.info(f'len(_val_dataloaders) = {len(_val_dataloaders)}')
    logger.info(f'_train_weights = {_train_weights}')

    # final train dataloader
    if len(_train_dataloaders) > 1:
        train_dataloader = balanced_dataloaders_generator(_train_dataloaders, _train_weights)
    else:
        train_dataloader = _train_dataloaders[0]
    
    # final validation dataloader
    val_dataloader_size = sum(len(d) for d in _val_dataloaders)
    val_dataloader = multi_cyclic_dataloaders_generator(_val_dataloaders)
    
    merged_dataset_name = '+'.join(_dataset_names)
    logger.info(f'merged_dataset_name = {merged_dataset_name}')
    
    # Attach metrics, losses, timer and events to engines    
    log_title(logger, 'Attaching metrics, losses, timer and events to engines')

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []

    attach_condition_aware_loss(trainer_engine, 'loss')
    metrics_to_print.append('loss')

    if use_mimiccxr_facts_for_train or use_mimiccxr_facts_for_test:
        _cond_func = lambda x: x['dataset_name'] == 'mimfg'
        in_train = use_mimiccxr_facts_for_train
        in_val = use_mimiccxr_facts_for_test
        if in_train:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(trainer_engine, 'attention_regularization_loss', _cond_func, 'mimfg_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:                
                attach_condition_aware_loss(trainer_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'mimfg_cpg_loss')
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(trainer_engine, 'global_alignment_contrastive_loss', _cond_func, 'mimfg_gac_loss')
            attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'mimfg_phrcls_loss')
            attach_condition_aware_prc_auc(trainer_engine, 'classifier_sigmoids', 'gt_labels', 'mimfg_prc_auc', _cond_func)
        if in_val:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(validator_engine, 'attention_regularization_loss', _cond_func, 'mimfg_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:                
                attach_condition_aware_loss(validator_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'mimfg_cpg_loss')
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(validator_engine, 'global_alignment_contrastive_loss', _cond_func, 'mimfg_gac_loss')
            attach_condition_aware_loss(validator_engine, 'phrase_classifier_loss', _cond_func, 'mimfg_phrcls_loss')
            attach_condition_aware_prc_auc(validator_engine, 'classifier_sigmoids', 'gt_labels', 'mimfg_prc_auc', _cond_func)
        # for logging
        if use_attention_regularization_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mimfg_att_reg_loss', train=in_train, val=in_val)
        if use_contrastive_phrase_grounding_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mimfg_cpg_loss', train=in_train, val=in_val)
        if use_global_alignment_contrastive_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mimfg_gac_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mimfg_phrcls_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mimfg_prc_auc', train=in_train, val=in_val)

    if use_chest_imagenome_for_train or use_chest_imagenome_for_val:
        in_train = use_chest_imagenome_for_train
        in_val = use_chest_imagenome_for_val
        
        # --- Chest Imagenome Phrase Grounding ---
        _cond_func = lambda x: x['dataset_name'] == 'chest-imagenome-pg'
        if in_train:
            attach_condition_aware_loss(trainer_engine, 'visual_grounding_confidence_loss', _cond_func, 'cipg_vgconf_loss')
        if in_val:
            attach_condition_aware_loss(validator_engine, 'visual_grounding_confidence_loss', _cond_func, 'cipg_vgconf_loss')
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cipg_vgconf_loss', val=in_val, train=in_train)
        
        # --- Chest Imagenome Anatomical Location Grounding ---
        _cond_func = lambda x: x['dataset_name'] == 'chest-imagenome-alg'
        if in_train:
            attach_condition_aware_loss(trainer_engine, 'visual_grounding_confidence_loss', _cond_func, 'cialg_vgconf_loss')
            attach_condition_aware_loss(trainer_engine, 'visual_grounding_bbox_loss', _cond_func, 'cialg_vgbbox_loss')
        if in_val:
            if skip_nms:
                attach_condition_aware_bbox_cnr(validator_engine, field_names=['pred_bbox_probs', 'bbox_coords', 'bbox_classes'],
                                                metric_name='cialg_cnr', feature_map_dimensions=(regions_height, regions_width),
                                                bbox_format=bbox_format, condition_function=_cond_func)
            else:
                attach_condition_aware_bbox_iou_open_class(validator_engine,
                                                    field_names=['predicted_bboxes', 'bbox_coords', 'bbox_classes'],
                                                    metric_name='cialg_bbox_iou', condition_function=_cond_func)
        # for logging
        if in_train:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cialg_vgconf_loss', val=False)
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cialg_vgbbox_loss', val=False)
        if in_val:
            if skip_nms:
                append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cialg_cnr', train=False)
            else:
                append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cialg_bbox_iou', train=False)

    if use_mscxr_for_train or use_mscxr_for_val:
        _cond_func = lambda x: x['dataset_name'] == 'mscxr'
        in_train = use_mscxr_for_train
        in_val = use_mscxr_for_val
        if in_train:
            if not mscxr_do_grounding_only:
                attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'mscxr_phrcls_loss')
                attach_condition_aware_prc_auc(trainer_engine, 'pred_probs', 'gt_labels', 'mscxr_prc_auc', _cond_func)
            attach_condition_aware_loss(trainer_engine, 'visual_grounding_confidence_loss', _cond_func, 'mscxr_vgconf_loss')
            attach_condition_aware_loss(trainer_engine, 'visual_grounding_bbox_loss', _cond_func, 'mscxr_vgbbox_loss')
        if in_val:
            if not mscxr_do_grounding_only:
                attach_condition_aware_prc_auc(validator_engine, 'pred_probs', 'gt_labels', 'mscxr_prc_auc', _cond_func)
                attach_condition_aware_bbox_iou_open_class(validator_engine,
                                                    field_names=['predicted_bboxes', 'bbox_coords', 'bbox_classes'],
                                                    metric_name='mscxr_bbox_iou', condition_function=_cond_func)
            else:
                if skip_nms:
                    attach_condition_aware_bbox_cnr(validator_engine, field_names=['pred_bbox_probs', 'bbox_coords'],
                                               metric_name='mscxr_cnr', feature_map_dimensions=(regions_height, regions_width),
                                               bbox_format=bbox_format, condition_function=_cond_func)
                else:
                    attach_condition_aware_bbox_iou_class_agnostic(validator_engine,
                                                        field_names=['predicted_bboxes', 'bbox_coords'],
                                                        metric_name='mscxr_bbox_iou', bbox_format=bbox_format,
                                                        condition_function=_cond_func)
        # for logging
        if in_train:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mscxr_vgconf_loss', val=False)
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mscxr_vgbbox_loss', val=False)
            if not mscxr_do_grounding_only:
                metrics_to_print.append('mscxr_phrcls_loss')
        if in_val:
            if skip_nms:
                append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mscxr_cnr', train=False)
            else:
                append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mscxr_bbox_iou', train=False)
        if not mscxr_do_grounding_only:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'mscxr_prc_auc', train=in_train, val=in_val)
    
    if use_vinbig_for_train or use_vinbig_for_test:
        _cond_func = lambda x: x['dataset_name'] == 'vinbig'
        in_train = use_vinbig_for_train
        in_val = use_vinbig_for_test
        assert vinbig_task_mode is not None

        if in_train:
            if vinbig_task_mode != VinBigPhraseTaskMode.GROUNDING.value:
                attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'vbg_phrcls_loss')
                attach_condition_aware_class_averaged_prc_auc(trainer_engine, 'pred_probs', 'gt_labels', None, 'vbg_prc_auc', _cond_func)
            if vinbig_task_mode != VinBigPhraseTaskMode.CLASSIFICATION.value:
                attach_condition_aware_loss(trainer_engine, 'visual_grounding_confidence_loss', _cond_func, 'vbg_vgconf_loss')
                attach_condition_aware_loss(trainer_engine,'visual_grounding_bbox_loss', _cond_func, 'vbg_vgbbox_loss')
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(trainer_engine, 'global_alignment_contrastive_loss', _cond_func, 'vbg_gac_loss')
            if use_attention_regularization_loss:
                attach_condition_aware_loss(trainer_engine, 'attention_regularization_loss', _cond_func, 'vbg_att_reg_loss')
        if in_val:
            if vinbig_task_mode != VinBigPhraseTaskMode.GROUNDING.value:
                attach_condition_aware_class_averaged_prc_auc(validator_engine, 'pred_probs', 'gt_labels', None, 'vbg_prc_auc', _cond_func)
            if vinbig_task_mode != VinBigPhraseTaskMode.CLASSIFICATION.value:
                if skip_nms:
                    attach_condition_aware_bbox_cnr(validator_engine, field_names=['pred_bbox_probs', 'bbox_coords'],
                                               metric_name='vbg_cnr', feature_map_dimensions=(regions_height, regions_width),
                                               bbox_format=bbox_format, condition_function=_cond_func)
                else:
                    attach_condition_aware_bbox_iou_class_agnostic(validator_engine,
                                                        field_names=['predicted_bboxes', 'bbox_coords'],
                                                        metric_name='vbg_bbox_iou', bbox_format=bbox_format,
                                                        condition_function=_cond_func)
        # for logging
        if in_train:
            if vinbig_task_mode != VinBigPhraseTaskMode.GROUNDING.value:
                metrics_to_print.append('vbg_phrcls_loss')
            if vinbig_task_mode != VinBigPhraseTaskMode.CLASSIFICATION.value:
                append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vbg_vgconf_loss', train=in_train, val=False)
                append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vbg_vgbbox_loss', train=in_train, val=False)
            if use_attention_regularization_loss:
                metrics_to_print.append('vbg_att_reg_loss')
            if use_global_alignment_contrastive_loss:
                metrics_to_print.append('vbg_gac_loss')
        if in_val:
            if vinbig_task_mode != VinBigPhraseTaskMode.CLASSIFICATION.value:
                if skip_nms:
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vbg_cnr', train=False)
                else:
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vbg_bbox_iou', train=False)
        if vinbig_task_mode != VinBigPhraseTaskMode.GROUNDING.value:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vbg_prc_auc', train=in_train, val=in_val)

    if use_padchest_gr_for_train or use_padchest_gr_for_val:
        _cond_func = lambda x: x['dataset_name'] == 'padchest_gr'
        in_train = use_padchest_gr_for_train
        in_val = use_padchest_gr_for_val

        if in_train:
            attach_condition_aware_loss(trainer_engine, 'visual_grounding_confidence_loss', _cond_func, 'pchstgr_vgconf_loss')
            attach_condition_aware_loss(trainer_engine,'visual_grounding_bbox_loss', _cond_func, 'pchstgr_vgbbox_loss')
        if in_val:
            if skip_nms:
                attach_condition_aware_bbox_cnr(validator_engine, field_names=['pred_bbox_probs', 'bbox_coords'],
                                                metric_name='pchstgr_cnr', feature_map_dimensions=(regions_height, regions_width),
                                                bbox_format=bbox_format, condition_function=_cond_func)
            else:
                attach_condition_aware_bbox_iou_class_agnostic(validator_engine,
                                                    field_names=['predicted_bboxes', 'bbox_coords'],
                                                    metric_name='pchstgr_bbox_iou', bbox_format=bbox_format,
                                                    condition_function=_cond_func)
        # for logging
        if in_train:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'pchstgr_vgconf_loss', train=in_train, val=False)
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'pchstgr_vgbbox_loss', train=in_train, val=False)
        if in_val:
            if skip_nms:
                append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'pchstgr_cnr', train=False)
            else:
                append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'pchstgr_bbox_iou', train=False)

    if use_chexlocalize:
        _cond_func = lambda x: x['dataset_name'] == 'cl'
        in_train = use_chexlocalize_for_train
        in_val = use_chexlocalize_for_test
        if in_train:
            attach_condition_aware_loss(trainer_engine,'attention_supervision_loss', _cond_func, 'cl_att_sup_loss')
            attach_condition_aware_segmask_iou(trainer_engine, 'pred_mask', 'gt_mask', 'cl_segmask_iou', _cond_func)
            attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'cl_phrcls_loss')
            attach_condition_aware_accuracy(trainer_engine, 'pred_labels', 'gt_labels', 'cl_phrase_acc', _cond_func)
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(trainer_engine, 'global_alignment_contrastive_loss', _cond_func, 'cl_gac_loss')
        if in_val:
            attach_condition_aware_loss(validator_engine,'attention_supervision_loss', _cond_func, 'cl_att_sup_loss')
            attach_condition_aware_segmask_iou(validator_engine, 'pred_mask', 'gt_mask', 'cl_segmask_iou', _cond_func)
            attach_condition_aware_accuracy(validator_engine, 'pred_labels', 'gt_labels', 'cl_phrase_acc', _cond_func)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cl_att_sup_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cl_segmask_iou', train=in_train, val=in_val)
        metrics_to_print.append('cl_phrcls_loss')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cl_phrase_acc', train=in_train, val=in_val)
        if use_global_alignment_contrastive_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cl_gac_loss', train=in_train, val=False)

    if use_chexpert:
        _cond_func = lambda x: x['dataset_name'] == 'chxp'
        in_train = use_chexpert_for_train
        in_val = use_chexpert_for_test
        if in_train:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(trainer_engine, 'attention_regularization_loss', _cond_func, 'chxp_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:
                attach_condition_aware_loss(trainer_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'chxp_cpg_loss')
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(trainer_engine, 'global_alignment_contrastive_loss', _cond_func, 'chxp_gac_loss')
            attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'chxp_phrcls_loss')
            attach_condition_aware_class_averaged_prc_auc(trainer_engine, 'classifier_sigmoids', 'gt_labels', None, 'chxp_prc_auc', _cond_func)
        if in_val:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(validator_engine, 'attention_regularization_loss', _cond_func, 'chxp_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:                
                attach_condition_aware_loss(validator_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'chxp_cpg_loss')
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(validator_engine, 'global_alignment_contrastive_loss', _cond_func, 'chxp_gac_loss')
            attach_condition_aware_loss(validator_engine, 'phrase_classifier_loss', _cond_func, 'chxp_phrcls_loss')
            attach_condition_aware_class_averaged_prc_auc(validator_engine, 'classifier_sigmoids', 'gt_labels', None, 'chxp_prc_auc', _cond_func)
        # for logging
        if use_attention_regularization_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chxp_att_reg_loss', train=in_train, val=in_val)
        if use_contrastive_phrase_grounding_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chxp_cpg_loss', train=in_train, val=in_val)
        if use_global_alignment_contrastive_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chxp_gac_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chxp_phrcls_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chxp_prc_auc', train=in_train, val=in_val)

    if use_iuxray:
        _cond_func = lambda x: x['dataset_name'] == 'iufg'
        in_train = use_iuxray_for_train
        in_val = use_iuxray_for_test
        if in_train:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(trainer_engine, 'attention_regularization_loss', _cond_func, 'iufg_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:                
                attach_condition_aware_loss(trainer_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'iufg_cpg_loss')
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(trainer_engine, 'global_alignment_contrastive_loss', _cond_func, 'iufg_gac_loss')
            attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'iufg_phrcls_loss')
            attach_condition_aware_prc_auc(trainer_engine, 'classifier_sigmoids', 'gt_labels', 'iufg_prc_auc', _cond_func)
        if in_val:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(validator_engine, 'attention_regularization_loss', _cond_func, 'iufg_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:                
                attach_condition_aware_loss(validator_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'iufg_cpg_loss')
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(validator_engine, 'global_alignment_contrastive_loss', _cond_func, 'iufg_gac_loss')
            attach_condition_aware_loss(validator_engine, 'phrase_classifier_loss', _cond_func, 'iufg_phrcls_loss')
            attach_condition_aware_prc_auc(validator_engine, 'classifier_sigmoids', 'gt_labels', 'iufg_prc_auc', _cond_func)
        # for logging
        if use_attention_regularization_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'iufg_att_reg_loss', train=in_train, val=in_val)
        if use_contrastive_phrase_grounding_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'iufg_cpg_loss', train=in_train, val=in_val)
        if use_global_alignment_contrastive_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'iufg_gac_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'iufg_phrcls_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'iufg_prc_auc', train=in_train, val=in_val)

    if use_cxrlt2024_custom_labels:
        _cond_func = lambda x: x['dataset_name'] == 'cxrlt2024c'
        # Train
        if use_attention_regularization_loss:
            attach_condition_aware_loss(trainer_engine, 'attention_regularization_loss', _cond_func, 'cxrlt2024c_att_reg_loss')
        attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'cxrlt2024c_phrcls_loss')
        attach_condition_aware_class_averaged_prc_auc(trainer_engine, 'classifier_sigmoids', 'gt_labels', 'phrase_indices',
                                                      'cxrlt2024c_prc_auc', _cond_func)
        if use_global_alignment_contrastive_loss:
            attach_condition_aware_loss(trainer_engine, 'global_alignment_contrastive_loss', _cond_func, 'cxrlt2024c_gac_loss')
    
        # Validation
        in_val = not use_cxrlt2024_official_labels or mimiccxr_trainer.use_all_cxrlt2024_official_labels_for_training
        if in_val:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(validator_engine, 'attention_regularization_loss', _cond_func, 'cxrlt2024c_att_reg_loss')
            attach_condition_aware_loss(validator_engine, 'phrase_classifier_loss', _cond_func, 'cxrlt2024c_phrcls_loss')
            attach_condition_aware_class_averaged_prc_auc(validator_engine, 'classifier_sigmoids', 'gt_labels', 'phrase_indices',
                                                        'cxrlt2024c_prc_auc', _cond_func)
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(validator_engine, 'global_alignment_contrastive_loss', _cond_func, 'cxrlt2024c_gac_loss')
        # for logging
        if use_attention_regularization_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cxrlt2024c_att_reg_loss', val=in_val)
        if use_global_alignment_contrastive_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cxrlt2024c_gac_loss', val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cxrlt2024c_phrcls_loss', val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cxrlt2024c_prc_auc', val=in_val)

    if use_cxrlt2024_official_labels:
        _cond_func = lambda x: x['dataset_name'] == 'cxrlt2024o'
        # Train
        if use_attention_regularization_loss:
            attach_condition_aware_loss(trainer_engine, 'attention_regularization_loss', _cond_func, 'cxrlt2024o_att_reg_loss')
        if use_global_alignment_contrastive_loss:
            attach_condition_aware_loss(trainer_engine, 'global_alignment_contrastive_loss', _cond_func, 'cxrlt2024o_gac_loss')
        attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'cxrlt2024o_phrcls_loss')
        attach_condition_aware_class_averaged_prc_auc(trainer_engine, 'classifier_sigmoids', 'gt_labels', None,
                                                      'cxrlt2024o_prc_auc', _cond_func)
    
        # Validation
        in_val = not mimiccxr_trainer.use_all_cxrlt2024_official_labels_for_training
        if in_val:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(validator_engine, 'attention_regularization_loss', _cond_func, 'cxrlt2024o_att_reg_loss')
            if use_global_alignment_contrastive_loss:
                attach_condition_aware_loss(validator_engine, 'global_alignment_contrastive_loss', _cond_func, 'cxrlt2024o_gac_loss')
            attach_condition_aware_loss(validator_engine, 'phrase_classifier_loss', _cond_func, 'cxrlt2024o_phrcls_loss')
            attach_condition_aware_class_averaged_prc_auc(validator_engine, 'classifier_sigmoids', 'gt_labels', None,
                                                        'cxrlt2024o_prc_auc', _cond_func)
        # for logging
        if use_attention_regularization_loss:
            metrics_to_print.append('cxrlt2024o_att_reg_loss')
        if use_global_alignment_contrastive_loss:
            metrics_to_print.append('cxrlt2024o_gac_loss')
        metrics_to_print.append('cxrlt2024o_phrcls_loss')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cxrlt2024o_prc_auc', val=in_val)

    # Score function
    assert len(val_metrics_to_merge) > 0
    if len(train_metrics_to_merge) > 0:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0.05, 0.95, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(trainer_engine.state.metrics, validator_engine.state.metrics)
    else:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0, 1, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(validator_engine.state.metrics)

    # Run common boilerplate code and start training
    run_common_boilerplate_code_and_start_training(
        update_lr_batchwise=update_lr_batchwise,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        score_fn=score_fn,
        model=model,
        optimizer=optimizer,
        save=save,
        checkpoint_folder_path=checkpoint_folder_path,
        build_custom_checkpoint_folder_path=lambda: get_checkpoint_folder_path('phrase_grounding', merged_dataset_name, model.get_name()),
        metadata_kwargs=dict(
            model_kwargs=model_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
            vinbig_trainer_kwargs=vinbig_trainer_kwargs,
            padchestgr_trainer_kwargs=padchestgr_trainer_kwargs,
            chexlocalize_trainer_kwargs=chexlocalize_trainer_kwargs,
            chexpert_trainer_kwargs=chexpert_trainer_kwargs,
            iuxray_trainer_kwargs=iuxray_trainer_kwargs,
            dataloading_kwargs=dataloading_kwargs,
            # collate_batch_fn_kwargs=collate_batch_fn_kwargs,
            train_image_transform_kwargs=train_image_transform_kwargs,
            val_image_transform_kwargs=val_image_transform_kwargs,
            trainer_engine_kwargs=trainer_engine_kwargs,
            validator_engine_kwargs=validator_engine_kwargs,
            metrics_kwargs=metrics_kwargs,
        ),
        device=device,
        trainer_engine=trainer_engine,
        validator_engine=validator_engine,
        train_metrics_to_merge=train_metrics_to_merge,
        val_metrics_to_merge=val_metrics_to_merge,
        metrics_to_print=metrics_to_print,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        val_dataloader_size=val_dataloader_size,
        model_kwargs=model_kwargs,
        override_lr=override_lr,
    )

def train_from_scratch(
    # Model args
    freeze_image_encoder,
    raw_image_encoding,
    huggingface_model_name,
    num_regions,   
    image_local_feat_size,
    image_encoder_pretrained_weights_path,
    image_encoder_dropout_p,
    pretrained_checkpoint_folder_path,
    pretrained_checkpoint_folder_paths,
    yolov8_model_name_or_path,
    yolov8_model_alias,
    yolov11_model_name_or_path,
    yolov11_model_alias,
    phrase_embedding_size,
    regions_width,
    regions_height,
    qkv_size,
    phrase_classifier_hidden_size,
    phrase_grounding_mode,
    transf_d_model,
    transf_nhead,
    transf_dim_feedforward,
    transf_dropout,
    transf_num_layers,
    visual_feature_proj_size,
    visual_grounding_hidden_size,
    phrase_mlp_hidden_dims,
    predict_global_alignment,
    alignment_proj_size,
    bbox_format,
    predict_relative_bbox_coords,
    # Optimizer args
    optimizer_name,
    lr,
    max_grad_norm,
    # lr_scheduler args
    scheduler,
    lr_decay,
    lr_decay_patience,
    warmup_and_decay_args,
    warmup_and_cosine_args,
    warmup_decay_and_cyclic_decay_args ,
    # Image transform args
    image_size,
    # Dataset args
    dicom_id_to_pos_neg_facts_filepath,
    mscxr_phrase2embedding_filepath,
    chest_imagenome_augmented_phrase_groundings_filepath,
    chest_imagenome_phrase_embeddings_filepath,
    chest_imagenome_bbox_phrase_embeddings_filepath,
    vinbig_phrase_embeddings_filepath,
    vinbig_task_mode,
    chexlocalize_class_phrase_embeddings_filepath,
    chexpert_class_phrase_embeddings_filepath,
    padchest_gr_phrase_embeddings_filepath,
    mimiccxr_exclude_noisy_images,
    mimiccxr_balance_long_middle_short_tail,
    mimiccxr_long_middle_short_tail_thresholds,
    mimiccxr_report_fact_nli_integrated_data_filepath,
    iuxray_image_id_to_pos_neg_facts_filepath,
    mimiccxr_use_interpret_cxr_challenge_split,
    mimiccxr_interpret_cxr_challenge_split_filepath,
    iuxray_use_interpret_cxr_challenge_split,
    iuxray_interpret_cxr_challenge_split_filepath,
    chexpert_use_interpret_cxr_challenge_split,
    chexpert_interpret_cxr_challenge_split_filepath,
    chexlocalize_use_interpret_cxr_challenge_split,
    chexlocalize_interpret_cxr_challenge_split_filepath,
    cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath,
    cxrlt2024_official_training_labels_for_fact_classification_filepath,
    # Dataloading args
    mimiccxr_facts_weight,
    chest_imagenome_pg_weight,
    chest_imagenome_alg_weight,
    mscxr_weight,
    cxrlt2024_weight,
    vinbig_weight,
    chexlocalize_weight,
    chexpert_weight,
    iuxray_weight,
    padchest_gr_weight,
    use_train_data_augmentations,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    val_batch_size_factor,
    num_train_workers,
    num_val_workers,
    # Fixed traning args
    use_mimiccxr_facts_for_train,
    use_mimiccxr_facts_for_test,
    use_mscxr_for_train,
    use_mscxr_for_val,
    use_chest_imagenome_for_train,
    use_chest_imagenome_for_val,
    use_vinbig_for_train,
    use_vinbig_for_test,
    use_padchest_gr_for_train,
    use_padchest_gr_for_val,
    use_chexlocalize_for_train,
    use_chexlocalize_for_test,
    use_chexpert_for_train,
    use_chexpert_for_test,
    use_iuxray_for_train,
    use_iuxray_for_test,
    use_cxrlt2024_challenge_split,
    use_cxrlt2024_custom_labels,
    use_cxrlt2024_official_labels,
    use_all_cxrlt2024_official_labels_for_training,
    cxrlt2024_do_balanced_sampling,
    vinbig_training_data_mode,
    chexpert_training_data_mode,
    mscxr_training_data_mode,
    mscxr_do_grounding_only,
    padchest_gr_training_split,
    use_amp,
    gradient_accumulation_steps,
    pos_area_prior,
    neg_area_prior,
    do_visual_grounding_with_bbox_regression,
    replace_phrase_embeddings_with_random_vectors,
    use_vinbig_with_modified_labels,
    skip_nms,
    # Loss weights
    attention_supervision_loss_weight,
    phrase_classifier_loss_weight,
    foreground_loss_weight,
    background_loss_weight,
    focal_loss_weight,
    bce_loss_weight,
    wbce_loss_weight,
    # Other loss args
    binary_multilabel_classif_loss_name,
    use_weighted_phrase_classifier_loss,
    cluster_and_label_weights_for_facts_filepath,
    use_attention_regularization_loss,
    use_contrastive_phrase_grounding_loss,
    nt_xent_temperature,
    # Variable traning args
    epochs,
    batches_per_epoch,
    # GPU
    device,
    # Other args
    save,
    override_metric_weights,
    debug=False,
):
    log_title(logger, 'Training model from scratch')
    
    use_yolov8 = raw_image_encoding == RawImageEncoding.YOLOV8
    predict_bboxes_chest_imagenome = (use_chest_imagenome_for_train or use_chest_imagenome_for_val) and use_yolov8
    use_mimiccxr = use_mimiccxr_facts_for_train or use_mscxr_for_train or use_mscxr_for_val or\
                     use_chest_imagenome_for_train or use_chest_imagenome_for_val or\
                     use_cxrlt2024_challenge_split # this challenge is built on top of MIMIC-CXR
    use_vinbig = use_vinbig_for_train or use_vinbig_for_test
    use_padchest_gr = use_padchest_gr_for_train or use_padchest_gr_for_val
    use_chexlocalize = use_chexlocalize_for_train or use_chexlocalize_for_test
    use_chexpert = use_chexpert_for_train or use_chexpert_for_test
    use_iuxray = use_iuxray_for_train or use_iuxray_for_test
    predict_bboxes_vinbig = (use_vinbig_for_train or use_vinbig_for_test) and use_yolov8
    yolov8_use_multiple_detection_layers = predict_bboxes_chest_imagenome and predict_bboxes_vinbig

    assert use_mimiccxr or use_vinbig or use_padchest_gr or use_chexlocalize or use_chexpert or use_iuxray
    assert use_chest_imagenome_for_val or use_mscxr_for_val or use_vinbig_for_test or use_chexlocalize_for_test or\
           use_mimiccxr_facts_for_test or use_chexpert_for_test or use_iuxray_for_test or use_cxrlt2024_challenge_split or\
           use_padchest_gr_for_val
    if use_chest_imagenome_for_val:
        assert use_chest_imagenome_for_train
    if use_mscxr_for_val:
        assert use_mimiccxr_facts_for_train or use_mscxr_for_train or use_chest_imagenome_for_train
    if use_vinbig_for_test:
        assert use_vinbig_for_train
    # if use_chexlocalize_for_test:
    #     assert use_chexlocalize_for_train
    if use_mimiccxr_facts_for_test:
        assert use_mimiccxr_facts_for_train
    if use_chexpert_for_test:
        assert use_chexpert_for_train
    if use_iuxray_for_test:
        assert use_iuxray_for_train
    
    model_kwargs = dict(
        pretrained_checkpoint_folder_path=pretrained_checkpoint_folder_path,
        pretrained_checkpoint_folder_paths=pretrained_checkpoint_folder_paths,
        # Image encoder
        raw_image_encoding=raw_image_encoding,
        huggingface_model_name=huggingface_model_name,
        freeze_image_encoder=freeze_image_encoder,
        image_local_feat_size=image_local_feat_size,
        image_encoder_dropout_p=image_encoder_dropout_p,
        image_encoder_pretrained_weights_path=image_encoder_pretrained_weights_path,
        num_regions=num_regions,
        yolov8_model_name_or_path=yolov8_model_name_or_path,
        yolov8_model_alias=yolov8_model_alias,
        yolov8_use_one_detector_per_dataset=(predict_bboxes_chest_imagenome and predict_bboxes_vinbig),
        yolov11_model_name_or_path=yolov11_model_name_or_path,
        yolov11_model_alias=yolov11_model_alias,
        visual_feature_proj_size=visual_feature_proj_size,
        visual_grounding_hidden_size=visual_grounding_hidden_size,
        phrase_mlp_hidden_dims=phrase_mlp_hidden_dims,
        image_size=image_size if isinstance(image_size, int) else image_size[0],
        # Aux tasks
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
        predict_global_alignment=predict_global_alignment,
        predict_relative_bbox_coords=predict_relative_bbox_coords,
        alignment_proj_size=alignment_proj_size,
        # Other
        apply_positional_encoding=not comes_with_positional_encoding(raw_image_encoding), # apply PE only if not comes with PE
        phrase_embedding_size=phrase_embedding_size,
        regions_width=regions_width,
        regions_height=regions_height,
        qkv_size=qkv_size,
        phrase_classifier_hidden_size=phrase_classifier_hidden_size,
        phrase_grounding_mode=phrase_grounding_mode,
        transf_d_model=transf_d_model,
        transf_nhead=transf_nhead,
        transf_dim_feedforward=transf_dim_feedforward,
        transf_dropout=transf_dropout,
        transf_num_layers=transf_num_layers,
        bbox_format=bbox_format,
    )
    
    optimizer_kwargs = dict(
        name=optimizer_name,
        lr=lr,
    )

    lr_scheduler_kwargs = dict(
        name=scheduler,
        factor=lr_decay,
        patience=lr_decay_patience,
        warmup_and_decay_args=warmup_and_decay_args,
        warmup_and_cosine_args=warmup_and_cosine_args,
        warmup_decay_and_cyclic_decay_args=warmup_decay_and_cyclic_decay_args,
        n_batches_per_epoch=batches_per_epoch,
    )
    
    dataloading_kwargs = dict(
        mimiccxr_facts_weight=mimiccxr_facts_weight,
        chest_imagenome_pg_weight=chest_imagenome_pg_weight,
        chest_imagenome_alg_weight=chest_imagenome_alg_weight,
        mscxr_weight=mscxr_weight,
        cxrlt2024_weight=cxrlt2024_weight,
        vinbig_weight=vinbig_weight,
        chexlocalize_weight=chexlocalize_weight,
        chexpert_weight=chexpert_weight,
        iuxray_weight=iuxray_weight,
        padchest_gr_weight=padchest_gr_weight,
    )

    # Image transforms
    train_image_transform_kwargs = {}
    val_image_transform_kwargs = {}
    is_train = use_train_data_augmentations # True if augmentations are applied
    _train_kwargs = dict(
        use_model_specific_transforms=False,
        image_size=image_size,
        is_train=is_train,
        bbox_format=bbox_format if is_train else None, # In order to use bbox-aware augmentations
    )
    _val_kwargs = dict(
        use_model_specific_transforms=False,
        image_size=image_size,
    )
    inject_mean_std_for_image_normalization(_train_kwargs, raw_image_encoding)
    inject_mean_std_for_image_normalization(_val_kwargs, raw_image_encoding)
    if use_mimiccxr:
        train_image_transform_kwargs[DATASET_NAMES.MIMICCXR] = _train_kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.MIMICCXR] =_val_kwargs.copy()
    if use_vinbig:
        train_image_transform_kwargs[DATASET_NAMES.VINBIG] = _train_kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.VINBIG] = _val_kwargs.copy()
    if use_chexlocalize:
        train_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE] = _train_kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE] = _val_kwargs.copy()
    if use_chexpert:
        train_image_transform_kwargs[DATASET_NAMES.CHEXPERT] = _train_kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.CHEXPERT] = _val_kwargs.copy()
    if use_iuxray:
        train_image_transform_kwargs[DATASET_NAMES.IUXRAY] = _train_kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.IUXRAY] = _val_kwargs.copy()
    if use_padchest_gr:
        train_image_transform_kwargs[DATASET_NAMES.PADCHEST_GR] = _train_kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.PADCHEST_GR] = _val_kwargs.copy()
    
    # # Collate batch functions
    # _kwargs = dict(
    #     use_yolo=use_yolo,
    # )
    # collate_batch_fn_kwargs = {}
    # if use_mimiccxr:
    #     include_loss_weights = binary_multilabel_classif_loss_name in [
    #         BinaryMultiLabelClassificationLossNames.WBCE,
    #         BinaryMultiLabelClassificationLossNames.FOCAL_BCE_WBCE,
    #     ]
    #     logger.info(f'include_loss_weights = {include_loss_weights}')
    #     # fact grounding
    #     collate_batch_fn_kwargs['mimfg'] = { 'dataset_name': 'mimfg', 'include_loss_weights': include_loss_weights, **_kwargs }
    #     # MS-CXR phrase grounding
    #     collate_batch_fn_kwargs['mscxr'] = { 'dataset_name': 'mscxr', **_kwargs }
    #     # chest imagenome bbox grounding
    #     collate_batch_fn_kwargs['cibg'] = { 'dataset_name': 'cibg', **_kwargs }
    #     # cxrlt2024 challenge custom labels
    #     collate_batch_fn_kwargs['cxrlt2024c'] = { 'dataset_name': 'cxrlt2024c', **_kwargs }
    #     # cxrlt2024 challenge official labels
    #     collate_batch_fn_kwargs['cxrlt2024o'] = { 'dataset_name': 'cxrlt2024o', **_kwargs }
    # if use_vinbig:
    #     collate_batch_fn_kwargs['vbg'] = { 'dataset_name': 'vbg', **_kwargs }
    # if use_chexlocalize:
    #     collate_batch_fn_kwargs['cl'] = { 'dataset_name': 'cl', **_kwargs }
    # if use_chexpert:
    #     collate_batch_fn_kwargs['chxp'] = { 'dataset_name': 'chxp', **_kwargs }
    # if use_iuxray:
    #     collate_batch_fn_kwargs['iufg'] = { 'dataset_name': 'iufg', **_kwargs }
    
    if use_mimiccxr:
        x = image_size if type(image_size) is int else image_size[0]
        if x > 512:
            source_image_size_mode = MIMICCXR_ImageSizeModes.ORIGINAL
        elif x > 256:
            source_image_size_mode = MIMICCXR_ImageSizeModes.MEDIUM_512
        else:
            source_image_size_mode = MIMICCXR_ImageSizeModes.SMALL_256x256
        logger.info(f'source_image_size_mode: {source_image_size_mode}')
        mimiccxr_trainer_kwargs = dict(
            mask_width=regions_width,
            mask_height=regions_height,
            use_facts_for_train=use_mimiccxr_facts_for_train,            
            use_facts_for_test=use_mimiccxr_facts_for_test,
            dicom_id_to_pos_neg_facts_filepath=dicom_id_to_pos_neg_facts_filepath,
            use_mscxr_for_train=use_mscxr_for_train,
            use_mscxr_for_val=use_mscxr_for_val,
            mscxr_phrase2embedding_filepath=mscxr_phrase2embedding_filepath,
            use_chest_imagenome_for_train=use_chest_imagenome_for_train,
            use_chest_imagenome_for_val=use_chest_imagenome_for_val,
            chest_imagenome_augmented_phrase_groundings_filepath=chest_imagenome_augmented_phrase_groundings_filepath,
            chest_imagenome_phrase_embeddings_filepath=chest_imagenome_phrase_embeddings_filepath,
            chest_imagenome_bbox_phrase_embeddings_filepath=chest_imagenome_bbox_phrase_embeddings_filepath,
            source_image_size_mode=source_image_size_mode,
            exclude_noisy_images=mimiccxr_exclude_noisy_images,
            use_yolov8=use_yolov8,
            balance_long_middle_short_tail=mimiccxr_balance_long_middle_short_tail,
            long_middle_short_tail_thresholds=mimiccxr_long_middle_short_tail_thresholds,
            report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
            use_weighted_phrase_classifier_loss=use_weighted_phrase_classifier_loss,
            cluster_and_label_weights_for_facts_filepath=cluster_and_label_weights_for_facts_filepath,
            use_interpret_cxr_challenge_split=mimiccxr_use_interpret_cxr_challenge_split,
            interpret_cxr_challenge_split_filepath=mimiccxr_interpret_cxr_challenge_split_filepath,
            use_cxrlt2024_challenge_split=use_cxrlt2024_challenge_split,
            use_cxrlt2024_custom_labels=use_cxrlt2024_custom_labels,
            use_cxrlt2024_official_labels=use_cxrlt2024_official_labels,
            use_all_cxrlt2024_official_labels_for_training=use_all_cxrlt2024_official_labels_for_training,
            cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath=cxrlt2024_custom_dicom_id_to_pos_neg_facts_filepath,
            cxrlt2024_official_training_labels_for_fact_classification_filepath=cxrlt2024_official_training_labels_for_fact_classification_filepath,
            cxrlt2024_do_balanced_sampling=cxrlt2024_do_balanced_sampling,
            do_visual_grounding_with_bbox_regression=do_visual_grounding_with_bbox_regression,
            data_augmentation_enabled=use_train_data_augmentations,
            replace_phrase_embeddings_with_random_vectors=replace_phrase_embeddings_with_random_vectors,
            mscxr_training_data_mode=mscxr_training_data_mode,
            mscxr_do_grounding_only=mscxr_do_grounding_only,
            bbox_format=bbox_format,
        )
    else:
        mimiccxr_trainer_kwargs = None

    if use_vinbig:
        vinbig_trainer_kwargs = dict(
            task_mode=vinbig_task_mode,
            mask_height=regions_height,
            mask_width=regions_width,
            phrase_embeddings_filepath=vinbig_phrase_embeddings_filepath,
            training_data_mode=vinbig_training_data_mode,
            use_training_set=use_vinbig_for_train,
            use_validation_set=use_vinbig_for_test,
            data_augmentation_enabled=use_train_data_augmentations,
            replace_phrase_embeddings_with_random_vectors=replace_phrase_embeddings_with_random_vectors,
            use_modified_labels=use_vinbig_with_modified_labels,
            bbox_format=bbox_format,
        )
    else:
        vinbig_trainer_kwargs = None

    if use_padchest_gr:
        padchestgr_trainer_kwargs = dict(
            mask_height=regions_height,
            mask_width=regions_width,
            phrase_embeddings_filepath=padchest_gr_phrase_embeddings_filepath,
            bbox_format=bbox_format,
            use_training_set=use_padchest_gr_for_train,
            use_validation_set=use_padchest_gr_for_val,
            training_split=padchest_gr_training_split,
            train_image_transforms_kwargs=train_image_transform_kwargs[DATASET_NAMES.PADCHEST_GR],
            val_image_transforms_kwargs=val_image_transform_kwargs[DATASET_NAMES.PADCHEST_GR],
        )
    else:
        padchestgr_trainer_kwargs = None

    if use_chexlocalize:
        chexlocalize_trainer_kwargs = dict(
            use_training_set=use_chexlocalize_for_train,
            use_validation_set=use_chexlocalize_for_test,
            mask_height=regions_height,
            mask_width=regions_width,
            class_phrase_embeddings_filepath=chexlocalize_class_phrase_embeddings_filepath,
            use_interpret_cxr_challenge_split=chexlocalize_use_interpret_cxr_challenge_split,
            interpret_cxr_challenge_split_filepath=chexlocalize_interpret_cxr_challenge_split_filepath,
        )
    else:
        chexlocalize_trainer_kwargs = None

    if use_chexpert:
        chexpert_trainer_kwargs = dict(
            training_data_mode=chexpert_training_data_mode,
            use_training_set=use_chexpert_for_train,
            use_validation_set=use_chexpert_for_test,
            phrase_embeddings_filepath=chexpert_class_phrase_embeddings_filepath,
            use_interpret_cxr_challenge_split=chexpert_use_interpret_cxr_challenge_split,
            interpret_cxr_challenge_split_filepath=chexpert_interpret_cxr_challenge_split_filepath,
        )
    else:
        chexpert_trainer_kwargs = None

    if use_iuxray:
        iuxray_trainer_kwargs = dict(
            do_train=use_iuxray_for_train,
            do_test=use_iuxray_for_test,
            image_id_to_pos_neg_facts_filepath=iuxray_image_id_to_pos_neg_facts_filepath,
            use_interpret_cxr_challenge_split=iuxray_use_interpret_cxr_challenge_split,
            interpret_cxr_challenge_split_filepath=iuxray_interpret_cxr_challenge_split_filepath,
        )
    else:
        iuxray_trainer_kwargs = None

    trainer_engine_kwargs = dict(
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=use_amp, training=True, validating=False, testing=False,
        using_yolov8=use_yolov8,
        yolov8_use_multiple_detection_layers=yolov8_use_multiple_detection_layers,
        pos_area_prior=pos_area_prior,
        neg_area_prior=neg_area_prior,
        max_grad_norm=max_grad_norm,
        vinbig_task_mode=vinbig_task_mode,
        mscxr_do_grounding_only=mscxr_do_grounding_only,
        # loss weights
        attention_supervision_loss_weight=attention_supervision_loss_weight,
        phrase_classifier_loss_weight=phrase_classifier_loss_weight,
        foreground_loss_weight=foreground_loss_weight,
        background_loss_weight=background_loss_weight,
        binary_multilabel_classif_loss_name=binary_multilabel_classif_loss_name,
        focal_loss_weight=focal_loss_weight,
        bce_loss_weight=bce_loss_weight,
        wbce_loss_weight=wbce_loss_weight,
        # other loss args
        use_attention_regularization_loss=use_attention_regularization_loss,
        use_contrastive_phrase_grounding_loss=use_contrastive_phrase_grounding_loss,
        use_global_image_phrase_contrastive_loss=predict_global_alignment,
        nt_xent_temperature=nt_xent_temperature,
    )

    validator_engine_kwargs = dict(
        training=False, validating=True, testing=False,
        using_yolov8=use_yolov8,
        yolov8_use_multiple_detection_layers=yolov8_use_multiple_detection_layers,
        binary_multilabel_classif_loss_name=binary_multilabel_classif_loss_name,
        vinbig_task_mode=vinbig_task_mode,
        mscxr_do_grounding_only=mscxr_do_grounding_only,
        # loss weights
        focal_loss_weight=focal_loss_weight,
        bce_loss_weight=bce_loss_weight,
        wbce_loss_weight=wbce_loss_weight,
        # other loss args
        use_attention_regularization_loss=use_attention_regularization_loss,
        use_contrastive_phrase_grounding_loss=use_contrastive_phrase_grounding_loss,
        use_global_image_phrase_contrastive_loss=predict_global_alignment,
        nt_xent_temperature=nt_xent_temperature,
        skip_nms=skip_nms,
    )

    metrics_kwargs = dict()
    if override_metric_weights:
        for override in override_metric_weights:
            try:
                key, value = override.split("=")
                metrics_kwargs[key] = float(value)
            except ValueError:
                raise ValueError(f"Invalid metric override format: {override}. Expected format: metric=value")
        logger.info(f'Overriden metric weights: {metrics_kwargs}')

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                vinbig_trainer_kwargs=vinbig_trainer_kwargs,
                padchestgr_trainer_kwargs=padchestgr_trainer_kwargs,
                chexlocalize_trainer_kwargs=chexlocalize_trainer_kwargs,
                chexpert_trainer_kwargs=chexpert_trainer_kwargs,
                iuxray_trainer_kwargs=iuxray_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                # collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                train_image_transform_kwargs=train_image_transform_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                metrics_kwargs=metrics_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                max_images_per_batch=max_images_per_batch,
                max_phrases_per_batch=max_phrases_per_batch,
                max_phrases_per_image=max_phrases_per_image,
                num_train_workers=num_train_workers,
                num_val_workers=num_val_workers,
                val_batch_size_factor=val_batch_size_factor,
                device=device,
                save=save,
                debug=debug)

def resume_training(
    checkpoint_folder,
    scheduler,
    optimizer_name,
    lr,
    lr_decay,
    lr_decay_patience,
    warmup_and_decay_args,
    warmup_and_cosine_args,
    warmup_decay_and_cyclic_decay_args,
    epochs,
    batches_per_epoch,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    num_train_workers,
    num_val_workers,
    val_batch_size_factor,
    device='cuda',
    save=True,
    override_lr=False,
    debug=False,
    **unused_kwargs,
):
    log_title(logger, 'Resuming training')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
    vinbig_trainer_kwargs = metadata['vinbig_trainer_kwargs']
    padchestgr_trainer_kwargs = metadata['padchestgr_trainer_kwargs']
    chexlocalize_trainer_kwargs = metadata['chexlocalize_trainer_kwargs']
    chexpert_trainer_kwargs = metadata['chexpert_trainer_kwargs']
    iuxray_trainer_kwargs = metadata['iuxray_trainer_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    train_image_transform_kwargs = metadata['train_image_transform_kwargs']
    val_image_transform_kwargs = metadata['val_image_transform_kwargs']
    trainer_engine_kwargs = metadata['trainer_engine_kwargs']
    validator_engine_kwargs = metadata['validator_engine_kwargs']
    metrics_kwargs = metadata['metrics_kwargs']

    if override_lr:
        optimizer_kwargs = dict(
            name = optimizer_name,
            lr = lr,
        )
        lr_scheduler_kwargs = dict(
            name = scheduler,
            factor = lr_decay,
            patience = lr_decay_patience,
            warmup_and_decay_args = warmup_and_decay_args,
            warmup_and_cosine_args = warmup_and_cosine_args,
            warmup_decay_and_cyclic_decay_args = warmup_decay_and_cyclic_decay_args,
            n_batches_per_epoch = batches_per_epoch,
        )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                vinbig_trainer_kwargs=vinbig_trainer_kwargs,
                padchestgr_trainer_kwargs=padchestgr_trainer_kwargs,
                chexlocalize_trainer_kwargs=chexlocalize_trainer_kwargs,
                chexpert_trainer_kwargs=chexpert_trainer_kwargs,
                iuxray_trainer_kwargs=iuxray_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                train_image_transform_kwargs=train_image_transform_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                metrics_kwargs=metrics_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                max_images_per_batch=max_images_per_batch,
                max_phrases_per_batch=max_phrases_per_batch,
                max_phrases_per_image=max_phrases_per_image,
                num_train_workers=num_train_workers,
                num_val_workers=num_val_workers,
                val_batch_size_factor=val_batch_size_factor,
                device=device,
                checkpoint_folder_path=checkpoint_folder,
                save=save,
                override_lr=override_lr,
                debug=debug)

def debug_main(args_string):
    args = shlex.split(args_string)
    args = parse_args(args)
    args = parsed_args_to_dict(args)
    if args['checkpoint_folder'] is not None:
        return resume_training(**args, debug=True)
    else:
        del args['checkpoint_folder']
        del args['override_lr']
        return train_from_scratch(**args, debug=True)

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    if args['checkpoint_folder'] is not None:
        resume_training(**args)
    else:
        del args['checkpoint_folder']
        del args['override_lr']
        train_from_scratch(**args)