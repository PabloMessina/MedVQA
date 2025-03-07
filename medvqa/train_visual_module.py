import  os
import argparse
import torch

from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_NUM_BBOX_CLASSES, get_anaxnet_bbox_sorted_indices
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    get_chest_imagenome_train_average_bbox_coords,
    get_labels_per_anatomy_and_anatomy_group,
    load_chest_imagenome_label_names,
)
from medvqa.datasets.cxr14.cxr14_dataset_management import CXR14_VisualModuleTrainer
from medvqa.datasets.chexpert.chexpert_dataset_management import CheXpert_VisualModuleTrainer
from medvqa.datasets.mimiccxr import MIMICCXR_ImageSizeModes
from medvqa.datasets.utils import get_merged_findings
from medvqa.datasets.vinbig import VINBIG_NUM_BBOX_CLASSES__MODIFIED
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBig_VisualModuleTrainer, VinBigTrainingMode
from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.vision.multilabel_classification import MLCVersion
from medvqa.models.vision.visual_modules import (
    DETECTRON2_HAS_RPN,
    MultiPurposeVisualModule,
    does_include_visual_features,
    inject_mean_std_for_image_normalization,
)
from medvqa.models.vqa.open_ended_vqa import (
    RawImageEncoding,
    does_include_image,
)
from medvqa.training.utils import append_metric_name, run_common_boilerplate_code_and_start_training
from medvqa.utils.constants import (
    CXR14_DATASET_ID,
    CHEXPERT_DATASET_ID,
    DATASET_NAMES,
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID,
    MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE,
    VINBIG_DATASET_ID,
    PADCHEST_DATASET_ID,
    VINBIG_NUM_BBOX_CLASSES,
    MetricNames,
)
from medvqa.utils.common import WORKSPACE_DIR, DictWithDefault
from medvqa.metrics import (
    attach_condition_aware_bbox_iou_per_class,
    attach_condition_aware_class_averaged_prc_auc,
    attach_condition_aware_loss,
    attach_dataset_aware_chest_imagenome_bbox_mae,
    attach_dataset_aware_chest_imagenome_bbox_iou,
    attach_dataset_aware_chest_imagenome_labels_auc,
    attach_dataset_aware_chest_imagenome_bbox_meanf1,
    attach_dataset_aware_chest_imagenome_labels_prcauc,
    attach_dataset_aware_chexpert_labels_auc,
    attach_dataset_aware_chexpert_labels_prcauc,
    attach_dataset_aware_vinbig_bbox_iou,
    attach_dataset_aware_vinbig_bbox_meanf1,
    attach_dataset_aware_cxr14_labels_macroavgf1,
    attach_dataset_aware_cxr14_labels_microavgf1,
    attach_dataset_aware_padchest_labels_macroavgf1,
    attach_dataset_aware_padchest_labels_microavgf1,
    attach_dataset_aware_padchest_localization_macroavgf1,
    attach_dataset_aware_padchest_localization_microavgf1,
    attach_medical_tags_f1score,
    attach_dataset_aware_orientation_accuracy,
    attach_dataset_aware_question_labels_macroavgf1,
    attach_dataset_aware_question_labels_microavgf1,
    attach_dataset_aware_gender_accuracy,
    attach_dataset_aware_loss,
    attach_loss,
)
from medvqa.models.checkpoint import load_metadata
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.files import (
    get_checkpoint_folder_path,
)
from medvqa.training.vision import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    multi_cyclic_dataloaders_generator,
    get_vision_collate_batch_fn,
)
from medvqa.metrics.utils import get_merge_metrics_fn
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleTrainer
from medvqa.datasets.iuxray.iuxray_vision_dataset_management import IUXRAY_VisualModuleTrainer
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter, print_blue, print_magenta
from medvqa.utils.metrics import average_ignoring_nones_and_nans

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # --- Required arguments

    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs the model will be trained')
    parser.add_argument('--batches_per_epoch', type=int, required=True, help='Number of batches per epoch')

    # --- Optional arguments

    parser.add_argument('--checkpoint_folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')

    # Image encoder
    parser.add_argument('--visual_input_mode', type=str, default='raw-image')
    parser.add_argument('--raw_image_encoding', type=str, default=RawImageEncoding.DENSENET_121)
    parser.add_argument('--image_local_feat_size', type=int, default=1024,
                        help='Size of local feature vectors from the CNN. They must match the actual vectors output by the CNN')
    parser.add_argument('--image_encoder_pretrained_weights_path', type=str, default=None)
    parser.add_argument('--freeze_image_encoder', action='store_true')
    parser.add_argument('--imagenet_pretrained', action='store_true')
    parser.add_argument('--visual_features_mlp_in_dim', type=int, default=None)
    parser.add_argument('--visual_features_mlp_out_dim', type=int, default=None)
    parser.add_argument('--visual_features_mlp_hidden_dims', nargs='+', type=int, default=None)
    parser.add_argument('--classification_mlp_hidden_dims', nargs='+', type=int, default=None)
    parser.add_argument('--iuxray_precomputed_visual_features_path', type=str, default=None)
    parser.add_argument('--mimiccxr_precomputed_visual_features_path', type=str, default=None)
    parser.add_argument('--chexpert_precomputed_visual_features_path', type=str, default=None)
    parser.add_argument('--vinbig_precomputed_visual_features_path', type=str, default=None)
    parser.add_argument('--clip_version', type=str, default=None)
    parser.add_argument('--huggingface_model_name', type=str, default=None)
    parser.add_argument('--chexpert_mlc_version', type=str, default=None, choices=MLCVersion.get_versions())
    parser.add_argument('--chexpert_mlc_hidden_size', type=int, default=128)
    parser.add_argument('--chest_imagenome_bbox_hidden_size', type=int, default=128)
    parser.add_argument('--chest_imagenome_bbox_regressor_version', type=str, default=None)
    parser.add_argument('--chest_imagenome_mlc_version', type=str, default=None)
    parser.add_argument('--chest_imagenome_mlc_hidden_size', type=int, default=128)
    parser.add_argument('--vinbig_mlc_hidden_size', type=int, default=128)
    parser.add_argument('--torchxrayvision_weights_name', type=str, default=None)
    parser.add_argument('--detectron2_model_yaml', type=str, default=None)
    parser.add_argument('--num_regions', type=int, default=None)
    parser.add_argument('--roi_heads_batch_size_per_image', type=int, default=128)
    parser.add_argument('--rpn_batch_size_per_image', type=int, default=128)
    parser.add_argument('--roi_align_output_size', type=int, default=None)
    parser.add_argument('--yolov8_model_name_or_path', type=str, default=None)
    parser.add_argument('--yolov8_model_alias', type=str, default=None)
    parser.add_argument('--yolov8_use_one_detector_per_dataset', action='store_true')
    parser.add_argument('--yolov11_model_name_or_path', type=str, default=None)
    parser.add_argument('--yolov11_model_alias', type=str, default=None)
    parser.add_argument('--query_embed_size', type=int, default=None)
    parser.add_argument('--local_attention_hidden_size', type=int, default=None)
    parser.add_argument('--use_linear_head_for_classification', action='store_true')
    
    parser.add_argument('--optimizer_name', type=str, default='adam')
    
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='reduce-lr-on-plateau')
    parser.add_argument('--lr_decay', type=float, default=0.76, help='Learning rate decay')
    parser.add_argument('--lr_decay_patience', type=int, default=2, help='Learning rate decay patience')
    parser.add_argument('--warmup_and_decay_args', type=str, default=None)
    parser.add_argument('--warmup_and_cosine_args', type=str, default=None)
    parser.add_argument('--warmup_decay_and_cyclic_decay_args', type=str, default=None)
    
    parser.add_argument('--train_batch_size', type=int, default=45, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=45, help='Batch size for validation')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for parallel dataloading')    
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')    
    parser.add_argument('--img_aug_mode', type=str, default=None, help='Mode of data augmentation used for images')
    parser.add_argument('--image_size', nargs=2, type=int, default=(256,256))
    parser.add_argument('--horizontal_flip_prob', type=float, default=0)

    # Weights for the different datasets. Used for training with multiple datasets
    parser.add_argument('--mimiccxr_weight', type=float, default=1)
    parser.add_argument('--chexpert_weight', type=float, default=0.3)
    parser.add_argument('--cxr14_weight', type=float, default=0.3)
    parser.add_argument('--vinbig_weight', type=float, default=0.3)
    parser.add_argument('--iuxray_weight', type=float, default=0.05)    
    parser.add_argument('--padchest_weight', type=float, default=0.4)  

    parser.add_argument('--mimiccxr_view_mode', type=str, default='any_single')    
    parser.add_argument('--mimiccxr_balanced_sampling_mode', type=str, default=None)
    parser.add_argument('--mimiccxr_balanced_batch_size', type=int, default=None)
    
    parser.add_argument('--chest_imagenome_labels_filename', type=str, default=None)
    parser.add_argument('--chest_imagenome_label_names_filename', type=str, default=None)
    parser.add_argument('--use_chest_imagenome_decent_images_only', action='store_true')

    parser.add_argument('--use_amp', action='store_true')
    
    parser.add_argument('--pretrained_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--pretrained_checkpoint_path', type=str, default=None)

    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no_save', dest='save', action='store_false')
    parser.set_defaults(save=True)

    parser.add_argument('--override_lr', action='store_true')

    parser.add_argument('--use_mimiccxr', dest='train_mimiccxr', action='store_true')
    parser.add_argument('--use_iuxray', dest='train_iuxray', action='store_true')
    parser.add_argument('--use_chexpert', dest='train_chexpert', action='store_true')
    parser.add_argument('--use_cxr14', dest='train_cxr14', action='store_true')
    
    # VinBigData arguments
    parser.add_argument('--use_vinbig', dest='train_vinbig', action='store_true')
    parser.add_argument('--vinbig_training_data_mode', type=str, default=VinBigTrainingMode.TRAIN_ONLY, choices=VinBigTrainingMode.get_all())
    parser.add_argument('--vinbig_use_validation', action='store_true')
    parser.add_argument('--use_vinbig_with_modified_labels', action='store_true')

    # PadChest arguments
    parser.add_argument('--use_padchest', dest='train_padchest', action='store_true')
    parser.add_argument('--padchest_training_data_mode', type=str, default='train')
    parser.add_argument('--padchest_use_validation', action='store_true')
    parser.add_argument('--padchest_train_study_ids_path', type=str, default=None)
    parser.add_argument('--padchest_val_study_ids_path', type=str, default=None)
    parser.add_argument('--padchest_test_study_ids_path', type=str, default=None)

    parser.add_argument('--binary_loss_name', type=str, default='bce')
    parser.add_argument('--focal_loss_weight', type=float, default=1)
    parser.add_argument('--bce_loss_weight', type=float, default=1)
    parser.add_argument('--wbce_loss_weight', type=float, default=1)

    # Auxiliary tasks arguments
    
    # medical tags
    parser.add_argument('--classify_tags', action='store_true')
    parser.add_argument('--n_medical_tags', type=int, default=None, help='Number of medical tags (for tag prediction auxiliary task)')
    parser.add_argument('--iuxray_medical_tags_per_report_filename', type=str, default=None)
    parser.add_argument('--mimiccxr_medical_tags_per_report_filename', type=str, default=None)
    # orientation
    parser.add_argument('--classify_orientation', action='store_true')
    # gender
    parser.add_argument('--classify_gender', action='store_true')
    # chexpert labels
    parser.add_argument('--classify_chexpert', action='store_true')
    parser.add_argument('--iuxray_chexpert_labels_filename', type=str, default=None)
    parser.add_argument('--mimiccxr_chexpert_labels_filename', type=str, default=None)
    # chest imagenome labels
    parser.add_argument('--classify_chest_imagenome', action='store_true')
    parser.add_argument('--predict_bboxes_chest_imagenome', action='store_true')
    parser.add_argument('--predict_labels_and_bboxes_chest_imagenome', action='store_true')
    parser.add_argument('--clamp_bboxes_chest_imagenome', action='store_true')
    parser.add_argument('--use_anaxnet_bbox_subset', action='store_true')
    parser.add_argument('--chest_imagenome_bbox_loss_weight', type=float, default=1.0)
    parser.add_argument('--pass_pred_bbox_coords_as_input', action='store_true')
    parser.add_argument('--use_gt_bboxes_as_predictions', action='store_true')
    # vinbig labels
    parser.add_argument('--predict_bboxes_vinbig', action='store_true')
    parser.add_argument('--classify_labels_vinbig', action='store_true')
    # question classification
    parser.add_argument('--classify_questions', action='store_true')
    parser.add_argument('--n_mined_questions', type=int, default=None)
    parser.add_argument('--iuxray_question_labels_filename', type=str, default=None)
    parser.add_argument('--mimiccxr_question_labels_filename', type=str, default=None)

    parser.add_argument('--merge_findings', action='store_true')
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = DictWithDefault(default=1.0) # Default weight is 1.0

def _metric_getter(metrics_dict, key):
    if key.endswith('_loss'):
        return 1 / (1 + metrics_dict[key]) # convert loss to score
    if key == MetricNames.CHESTIMAGENOMELABELAUC or\
        key == MetricNames.CHESTIMAGENOMELABELPRCAUC or\
        key == MetricNames.CHXLABEL_AUC or\
        key == MetricNames.CHXLABEL_PRCAUC or\
        key == MetricNames.VINBIGLABELAUC or\
        key == MetricNames.VINBIGLABELPRCAUC:
        scores = metrics_dict[key]
        return 0.5 * (scores['macro_avg'] + scores['micro_avg'])
    metric = metrics_dict[key]
    if isinstance(metric, list):
        return average_ignoring_nones_and_nans(metric)
    return metric

def train_model(
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_trainer_kwargs,
    iuxray_trainer_kwargs,
    chexpert_trainer_kwargs,
    cxr14_trainer_kwargs,
    vinbig_trainer_kwargs,
    padchest_trainer_kwargs,
    dataloading_kwargs,
    collate_batch_fn_kwargs,
    train_image_transform_kwargs,
    val_image_transform_kwargs,
    training_kwargs,
    trainer_engine_kwargs,
    validator_engine_kwargs,
    auxiliary_tasks_kwargs,
    epochs,
    batches_per_epoch,
    num_workers,
    device = 'GPU',
    checkpoint_folder_path = None,
    save = True,
    override_lr = False,
    debug = False,
):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    train_batch_size = dataloading_kwargs['train_batch_size']
    val_batch_size = dataloading_kwargs['val_batch_size']
    train_iuxray = training_kwargs['train_iuxray']
    train_mimiccxr = training_kwargs['train_mimiccxr']
    train_chexpert = training_kwargs['train_chexpert']
    train_cxr14 = training_kwargs['train_cxr14']
    train_vinbig = training_kwargs['train_vinbig']
    train_padchest = training_kwargs['train_padchest']  
    use_merged_findings = trainer_engine_kwargs.get('use_merged_findings', False)
    use_detectron2 = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_detectron2', False)
    use_yolov8 = (mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_yolov8', False)) or\
                 (vinbig_trainer_kwargs is not None and vinbig_trainer_kwargs.get('use_yolov8', False))
    use_yolov11 = (mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_yolov11', False)) or\
                    (vinbig_trainer_kwargs is not None and vinbig_trainer_kwargs.get('use_yolov11', False))
    use_vinbig_with_modified_labels = vinbig_trainer_kwargs is not None and vinbig_trainer_kwargs.get('use_vinbig_with_modified_labels', False)
    
    visual_input_mode = model_kwargs['visual_input_mode']
    include_image = does_include_image(visual_input_mode)
    include_visual_features = does_include_visual_features(visual_input_mode)
    assert include_image or include_visual_features

    # auxiliary task: medical tags prediction
    classify_tags = auxiliary_tasks_kwargs['classify_tags']
    # auxiliary task: orientation classification
    classify_orientation = auxiliary_tasks_kwargs['classify_orientation']
    # auxiliary task: gender classification
    classify_gender = auxiliary_tasks_kwargs['classify_gender']
    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs['classify_chest_imagenome']
    predict_bboxes_chest_imagenome = auxiliary_tasks_kwargs['predict_bboxes_chest_imagenome']
    # auxiliary task: vinbig labels
    classify_labels_vinbig = auxiliary_tasks_kwargs['classify_labels_vinbig']
    predict_bboxes_vinbig = auxiliary_tasks_kwargs['predict_bboxes_vinbig']
    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)
    n_questions_aux_task = auxiliary_tasks_kwargs.get('n_questions_aux_task', None)
    iuxray_question_labels_filename = auxiliary_tasks_kwargs.get('iuxray_question_labels_filename', None)
    mimiccxr_question_labels_filename = auxiliary_tasks_kwargs.get('mimiccxr_question_labels_filename', None)
    
    # Sanity checks
    
    if classify_questions:
        assert n_questions_aux_task is not None
        if train_iuxray: assert iuxray_question_labels_filename is not None
        if train_mimiccxr: assert mimiccxr_question_labels_filename is not None
    
    if train_chexpert:
        assert classify_chexpert

    if train_vinbig:
        assert classify_labels_vinbig or predict_bboxes_vinbig

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Create model
    count_print('Creating instance of MultiPurposeVisualModule ...')
    model = MultiPurposeVisualModule(**model_kwargs, device=device)
    model = model.to(device)

    # Check dataset weights
    if dataloading_kwargs['iuxray_weight'] == 0:
        train_iuxray = False
    if dataloading_kwargs['mimiccxr_weight'] == 0:
        train_mimiccxr = False
    if dataloading_kwargs['chexpert_weight'] == 0:
        train_chexpert = False
    if dataloading_kwargs['cxr14_weight'] == 0:
        train_cxr14 = False
    if dataloading_kwargs['vinbig_weight'] == 0:
        train_vinbig = False
    if dataloading_kwargs['padchest_weight'] == 0:
        train_padchest = False

    # Optimizer
    count_print('Defining optimizer ...')
    optimizer = create_optimizer(params=model.parameters(), **optimizer_kwargs)

    # Learning rate scheduler
    count_print('Defining scheduler ...')
    lr_scheduler, update_lr_batchwise = create_lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)

    # Create trainer and validator engines
    count_print('Creating trainer and validator engines ...')
    if model_kwargs['raw_image_encoding'] == RawImageEncoding.YOLOV8:
        model_for_yolov8 = model.raw_image_encoder
    else:
        model_for_yolov8 = None
    trainer_engine = get_engine(model=model, optimizer=optimizer, device=device, 
        update_lr_batchwise=update_lr_batchwise, lr_scheduler=lr_scheduler,
        model_for_yolov8=model_for_yolov8, **trainer_engine_kwargs)
    validator_engine = get_engine(model=model, device=device, **validator_engine_kwargs)
    
    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    if train_mimiccxr:
        if use_detectron2: # special case for detectron2
            mimiccxr_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs['mim-cim-det2'])
        else:
            mimiccxr_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs['mimiccxr'])
    if train_iuxray:
        iuxray_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs['iuxray'])
    if train_chexpert:
        chexpert_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs['chexpert'])
    if train_cxr14:
        cxr14_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs['cxr14'])
    if train_vinbig:
        vinbig_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs['vinbig'])
    if train_padchest:
        padchest_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs['padchest'])

    # Create MIMIC-CXR trainer
    if train_mimiccxr:
        count_print('Creating MIMIC-CXR visual module trainer ...')
        mimiccxr_trainer = MIMICCXR_VisualModuleTrainer(
            train_image_transform=get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            collate_batch_fn=mimiccxr_collate_batch_fn,            
            num_workers=num_workers,
            **mimiccxr_trainer_kwargs,
        )
    
    # Create IU X-Ray trainer
    if train_iuxray:
        count_print('Creating IU X-Ray visual module trainer ...')
        iuxray_trainer = IUXRAY_VisualModuleTrainer(
            train_image_transform=get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.IUXRAY]),
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.IUXRAY]),
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            collate_batch_fn=iuxray_collate_batch_fn,            
            num_workers=num_workers,
            **iuxray_trainer_kwargs,
        )

    # Create CheXpert trainer
    if train_chexpert:
        count_print('Creating CheXpert visual module trainer ...')
        chexpert_trainer = CheXpert_VisualModuleTrainer(
            train_image_transform=get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.CHEXPERT]),
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.CHEXPERT]),
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            collate_batch_fn=chexpert_collate_batch_fn,
            num_workers=num_workers,
            **chexpert_trainer_kwargs,
        )

    # Create CXR14 trainer
    if train_cxr14:
        count_print('Creating CXR14 visual module trainer ...')
        cxr14_trainer = CXR14_VisualModuleTrainer(
            train_image_transform=get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.CXR14]),
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.CXR14]),
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            collate_batch_fn=cxr14_collate_batch_fn,
            num_workers=num_workers,
            **cxr14_trainer_kwargs,
        )

    # Create VinBig trainer
    if train_vinbig:
        count_print('Creating VinBig visual module trainer ...')
        vinbig_trainer = VinBig_VisualModuleTrainer(
            train_image_transform=get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.VINBIG]),
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.VINBIG]),
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            collate_batch_fn=vinbig_collate_batch_fn,
            num_workers=num_workers,
            **vinbig_trainer_kwargs,
        )

    # Create PadChest trainer
    if train_padchest:
        count_print('Creating PadChest visual module trainer ...')
        padchest_trainer = PadChest_VisualModuleTrainer(
            train_image_transform=get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.PADCHEST]),
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.PADCHEST]),
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            collate_batch_fn=padchest_collate_batch_fn,
            num_workers=num_workers,
            **padchest_trainer_kwargs,
        )

    if debug: # if debugging
        output = {}
        if train_mimiccxr: output['mimiccxr_vqa_trainer'] = mimiccxr_trainer
        if train_iuxray: output['iuxray_trainer'] = iuxray_trainer
        if train_chexpert: output['chexpert_trainer'] = chexpert_trainer
        if train_cxr14: output['cxr14_trainer'] = cxr14_trainer
        if train_vinbig: output['vinbig_trainer'] = vinbig_trainer
        if train_padchest: output['padchest_trainer'] = padchest_trainer
        return output

    # Create complex dataloaders
    count_print('Creating dataloaders ...')
    
    _train_weights = []
    _train_dataloaders = []
    _val_dataloaders = []
    _dataset_names = []

    if train_mimiccxr:
        _dataset_names.append('mim')
        _train_weights.append(dataloading_kwargs['mimiccxr_weight'])
        _train_dataloaders.append(mimiccxr_trainer.train_dataloader)
        _val_dataloaders.append(mimiccxr_trainer.val_dataloader)        
    if train_iuxray:
        _dataset_names.append('iu')
        _train_weights.append(dataloading_kwargs['iuxray_weight'])
        _train_dataloaders.append(iuxray_trainer.train_dataloader)
    if train_chexpert:
        _train_weights.append(dataloading_kwargs['chexpert_weight'])
        _train_dataloaders.append(chexpert_trainer.dataloader)
        _dataset_names.append('chexp')
    if train_cxr14:
        _dataset_names.append('cxr14')
        _train_weights.append(dataloading_kwargs['cxr14_weight'])
        _train_dataloaders.append(cxr14_trainer.dataloader)
    if train_vinbig:
        _dataset_names.append('vinbig')
        _train_weights.append(dataloading_kwargs['vinbig_weight'])
        _train_dataloaders.append(vinbig_trainer.train_dataloader)
        if vinbig_trainer.use_validation_set:
            _val_dataloaders.append(vinbig_trainer.val_dataloader)
    if train_padchest:
        _dataset_names.append('padchest')
        _train_weights.append(dataloading_kwargs['padchest_weight'])
        _train_dataloaders.append(padchest_trainer.train_dataloader)
        if padchest_trainer.use_validation_set:
            _val_dataloaders.append(padchest_trainer.val_dataloader)
    
    assert len(_train_dataloaders) > 0
    assert len(_val_dataloaders) > 0
    assert len(_train_dataloaders) == len(_train_weights)
    print(f'len(_train_dataloaders) = {len(_train_dataloaders)}')
    print(f'len(_val_dataloaders) = {len(_val_dataloaders)}')
    print(f'_train_weights = {_train_weights}')

    # final train dataloader
    if len(_train_dataloaders) > 1:
        train_dataloader = balanced_dataloaders_generator(_train_dataloaders, _train_weights)
    else:
        train_dataloader = _train_dataloaders[0]
    
    # final validation dataloader
    val_dataloader_size = sum(len(d) for d in _val_dataloaders)
    val_dataloader = multi_cyclic_dataloaders_generator(_val_dataloaders)
    
    merged_dataset_name = '+'.join(_dataset_names)
    print('merged_dataset_name =', merged_dataset_name)
    
    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    _mim_datasets = [MIMICCXR_DATASET_ID]
    _iu_mim_datasets = [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID]    
    _orientation_datasets = _iu_mim_datasets + [CHEXPERT_DATASET_ID, CXR14_DATASET_ID, PADCHEST_DATASET_ID]
    _chexpert_labels_datasets = _iu_mim_datasets + [CHEXPERT_DATASET_ID]
    _gender_datasets = [CHEXPERT_DATASET_ID, CXR14_DATASET_ID, PADCHEST_DATASET_ID, MIMICCXR_DATASET_ID]
    _yolov8_datasets = [MIMICCXR_DATASET_ID, VINBIG_DATASET_ID]

    if use_merged_findings:
        _findings_remapper = trainer_engine_kwargs['findings_remapper']
        _chexpert_class_indices = _findings_remapper[CHEXPERT_DATASET_ID]
        _cxr14_class_indices = _findings_remapper[CXR14_DATASET_ID]
        _vinbig_class_indices = _findings_remapper[VINBIG_DATASET_ID]
    else:
        _chexpert_class_indices = _cxr14_class_indices = _vinbig_class_indices = None

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []
    
    attach_condition_aware_loss(trainer_engine, 'loss', lambda _: True, 'loss')
    metrics_to_print.append('loss')
    
    if classify_tags:
        attach_medical_tags_f1score(trainer_engine, device)
        attach_medical_tags_f1score(validator_engine, device)
        attach_loss(MetricNames.MEDTAGS_LOSS, trainer_engine, device)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.MEDTAGF1)
        metrics_to_print.append(MetricNames.MEDTAGS_LOSS)

    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(trainer_engine, _orientation_datasets)
        attach_dataset_aware_orientation_accuracy(validator_engine, _orientation_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.ORIENTATION_LOSS, _orientation_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.ORIENACC )
        metrics_to_print.append(MetricNames.ORIENTATION_LOSS)

    if classify_questions:
        attach_dataset_aware_question_labels_macroavgf1(trainer_engine, _iu_mim_datasets)
        attach_dataset_aware_question_labels_microavgf1(trainer_engine, _iu_mim_datasets)
        attach_dataset_aware_question_labels_macroavgf1(validator_engine, _iu_mim_datasets)
        attach_dataset_aware_question_labels_microavgf1(validator_engine, _iu_mim_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.QLABELS_LOSS, _iu_mim_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.QLABELS_MICROAVGF1)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.QLABELS_MACROAVGF1)
        metrics_to_print.append(MetricNames.QLABELS_LOSS)
    
    if classify_chexpert:
        attach_dataset_aware_chexpert_labels_auc(validator_engine, _chexpert_labels_datasets, 'cpu')
        attach_dataset_aware_chexpert_labels_prcauc(trainer_engine, _chexpert_labels_datasets, 'cpu')
        attach_dataset_aware_chexpert_labels_prcauc(validator_engine, _chexpert_labels_datasets, 'cpu')        
        attach_dataset_aware_loss(trainer_engine, MetricNames.CHEXPERT_LOSS, _chexpert_labels_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHXLABEL_AUC, train=False)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHXLABEL_PRCAUC)
        metrics_to_print.append(MetricNames.CHEXPERT_LOSS)

    if classify_chest_imagenome:
        attach_dataset_aware_chest_imagenome_labels_auc(validator_engine, _mim_datasets, 'cpu')
        attach_dataset_aware_chest_imagenome_labels_prcauc(trainer_engine, _mim_datasets, 'cpu')
        attach_dataset_aware_chest_imagenome_labels_prcauc(validator_engine, _mim_datasets, 'cpu')
        attach_dataset_aware_loss(trainer_engine, MetricNames.CHEST_IMAGENOME_LABEL_LOSS, _mim_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMELABELAUC, train=False)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMELABELPRCAUC)
        metrics_to_print.append(MetricNames.CHEST_IMAGENOME_LABEL_LOSS)

    if predict_bboxes_chest_imagenome and not (use_detectron2 or use_yolov8 or use_yolov11):
        attach_dataset_aware_chest_imagenome_bbox_mae(trainer_engine, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_mae(validator_engine, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_iou(trainer_engine, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_iou(validator_engine, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_meanf1(trainer_engine, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_meanf1(validator_engine, _mim_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.CHEST_IMAGENOME_BBOX_LOSS, _mim_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMEBBOXMEANF1)
        metrics_to_print.append(MetricNames.CHEST_IMAGENOME_BBOX_LOSS)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXIOU)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXMAE)
    
    if use_yolov8:
        assert predict_bboxes_chest_imagenome or predict_bboxes_vinbig
        attach_dataset_aware_loss(trainer_engine, MetricNames.YOLOV8_LOSS, _yolov8_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.YOLOV8_BOX_LOSS, _yolov8_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.YOLOV8_CLS_LOSS, _yolov8_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.YOLOV8_DFL_LOSS, _yolov8_datasets)
        if predict_bboxes_chest_imagenome:
            attach_dataset_aware_chest_imagenome_bbox_iou(validator_engine, _mim_datasets, use_yolov8=True)
        if predict_bboxes_vinbig:
            attach_dataset_aware_vinbig_bbox_iou(validator_engine, [VINBIG_DATASET_ID])
            attach_dataset_aware_vinbig_bbox_meanf1(validator_engine, [VINBIG_DATASET_ID])
        # for logging
        if predict_bboxes_chest_imagenome:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMEBBOXIOU, train=False)
        if predict_bboxes_vinbig:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.VINBIGBBOXIOU, train=False)
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.VINBIGBBOXMEANF1, train=False)
        metrics_to_print.append(MetricNames.YOLOV8_LOSS)
        metrics_to_print.append(MetricNames.YOLOV8_BOX_LOSS)
        metrics_to_print.append(MetricNames.YOLOV8_CLS_LOSS)
        metrics_to_print.append(MetricNames.YOLOV8_DFL_LOSS)

    if use_yolov11:
        assert predict_bboxes_chest_imagenome or predict_bboxes_vinbig
        # VinBigData related metrics
        _cond_func = lambda x: x['flag'] == 'vinbig'
        if predict_bboxes_vinbig:
            if use_vinbig_with_modified_labels:
                nc = VINBIG_NUM_BBOX_CLASSES__MODIFIED
            else:
                nc = VINBIG_NUM_BBOX_CLASSES
            attach_condition_aware_loss(trainer_engine, 'vinbig_yolov11_loss', _cond_func, 'vnb_y11_loss')
            attach_condition_aware_loss(trainer_engine, 'vinbig_yolov11_box_loss', _cond_func, 'vnb_y11_box_loss')
            attach_condition_aware_loss(trainer_engine, 'vinbig_yolov11_cls_loss', _cond_func, 'vnb_y11_cls_loss')
            attach_condition_aware_loss(trainer_engine, 'vinbig_yolov11_dfl_loss', _cond_func, 'vnb_y11_dfl_loss')
            attach_condition_aware_bbox_iou_per_class(validator_engine,
                                                      field_names=('yolov11_predictions', 'vinbig_bbox_coords', 'vinbig_bbox_classes'),
                                                      metric_name='vnb_y11_bbox_iou', nc=nc, condition_function=_cond_func,
                                                      for_vinbig=True, use_yolov8=True)
        # Chest ImageNome related metrics
        _cond_func = lambda x: x['flag'] == 'mimiccxr'
        if predict_bboxes_chest_imagenome:
            attach_condition_aware_loss(trainer_engine, 'yolov11_loss', _cond_func, 'cig_y11_loss')
            attach_condition_aware_loss(trainer_engine, 'yolov11_box_loss', _cond_func, 'cig_y11_box_loss')
            attach_condition_aware_loss(trainer_engine, 'yolov11_cls_loss', _cond_func, 'cig_y11_cls_loss')
            attach_condition_aware_loss(trainer_engine, 'yolov11_dfl_loss', _cond_func, 'cig_y11_dfl_loss')
            attach_condition_aware_bbox_iou_per_class(validator_engine,
                                                      field_names=('yolov11_predictions', 'chest_imagenome_bbox_coords', 'chest_imagenome_bbox_presence'),
                                                      metric_name='cig_y11_bbox_iou', nc=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
                                                      condition_function=_cond_func, use_yolov8=True)

        # for logging
        if predict_bboxes_vinbig:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vnb_y11_bbox_iou', train=False)
            metrics_to_print.append('vnb_y11_loss')
            metrics_to_print.append('vnb_y11_box_loss')
            metrics_to_print.append('vnb_y11_cls_loss')
            metrics_to_print.append('vnb_y11_dfl_loss')
        if predict_bboxes_chest_imagenome:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cig_y11_bbox_iou', train=False)
            metrics_to_print.append('cig_y11_loss')
            metrics_to_print.append('cig_y11_box_loss')
            metrics_to_print.append('cig_y11_cls_loss')
            metrics_to_print.append('cig_y11_dfl_loss')

    if use_detectron2:
        _d2_datasets = [MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE]
        attach_dataset_aware_chest_imagenome_bbox_mae(validator_engine, _d2_datasets, use_detectron2=True)
        attach_dataset_aware_chest_imagenome_bbox_iou(validator_engine, _d2_datasets, use_detectron2=True)
        attach_dataset_aware_chest_imagenome_bbox_meanf1(validator_engine, _d2_datasets, use_detectron2=True)
        attach_dataset_aware_loss(trainer_engine, MetricNames.DETECTRON2_BOX_REG_LOSS, _d2_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.DETECTRON2_CLS_LOSS, _d2_datasets)
        if trainer_engine_kwargs['detectron2_includes_rpn']:
            attach_dataset_aware_loss(trainer_engine, MetricNames.DETECTRON2_RPN_CLS_LOSS, _d2_datasets)
            attach_dataset_aware_loss(trainer_engine, MetricNames.DETECTRON2_RPN_LOC_LOSS, _d2_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMEBBOXMEANF1)
        metrics_to_print.append(MetricNames.DETECTRON2_BOX_REG_LOSS)
        metrics_to_print.append(MetricNames.DETECTRON2_CLS_LOSS)
        if trainer_engine_kwargs['detectron2_includes_rpn']:
            metrics_to_print.append(MetricNames.DETECTRON2_RPN_CLS_LOSS)
            metrics_to_print.append(MetricNames.DETECTRON2_RPN_LOC_LOSS)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXIOU)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXMAE)

    if classify_gender:
        attach_dataset_aware_gender_accuracy(trainer_engine, _gender_datasets, ignore_index=2)
        attach_dataset_aware_loss(trainer_engine, MetricNames.GENDER_LOSS, _gender_datasets)
        in_val = (train_padchest and padchest_trainer.use_validation_set) or train_mimiccxr
        if in_val: attach_dataset_aware_gender_accuracy(validator_engine, [PADCHEST_DATASET_ID, MIMICCXR_DATASET_ID])
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.GENDER_ACC, val=in_val)
        metrics_to_print.append(MetricNames.GENDER_LOSS)

    if train_cxr14:
        attach_dataset_aware_cxr14_labels_macroavgf1(trainer_engine, [CXR14_DATASET_ID], _cxr14_class_indices)
        attach_dataset_aware_cxr14_labels_microavgf1(trainer_engine, [CXR14_DATASET_ID], _cxr14_class_indices)
        attach_dataset_aware_loss(trainer_engine, MetricNames.CXR14_LOSS, [CXR14_DATASET_ID])
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CXR14MICROAVGF1, val=False)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CXR14MACROAVGF1, val=False)
        metrics_to_print.append(MetricNames.CXR14_LOSS)

    if train_vinbig:
        _cond_func = lambda x: x['flag'] == 'vinbig'
        in_val = vinbig_trainer.use_validation_set
        if classify_labels_vinbig:
            attach_condition_aware_class_averaged_prc_auc(trainer_engine, 'pred_vinbig_probs', 'vinbig_labels', None, 'vnb_macro_prcauc', _cond_func)
            attach_condition_aware_loss(trainer_engine, 'vinbig_label_loss', _cond_func, 'vnb_label_loss')
            if in_val:
                attach_condition_aware_class_averaged_prc_auc(validator_engine, 'pred_vinbig_probs', 'vinbig_labels', None, 'vnb_macro_prcauc', _cond_func)
            # for logging
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vnb_macro_prcauc', val=in_val)
            metrics_to_print.append('vnb_label_loss')
    
    if train_padchest:        
        attach_dataset_aware_padchest_labels_macroavgf1(trainer_engine, [PADCHEST_DATASET_ID])
        attach_dataset_aware_padchest_labels_microavgf1(trainer_engine, [PADCHEST_DATASET_ID])
        attach_dataset_aware_loss(trainer_engine, MetricNames.PADCHEST_LABEL_LOSS, [PADCHEST_DATASET_ID])
        attach_dataset_aware_padchest_localization_macroavgf1(trainer_engine, [PADCHEST_DATASET_ID])
        attach_dataset_aware_padchest_localization_microavgf1(trainer_engine, [PADCHEST_DATASET_ID])
        attach_dataset_aware_loss(trainer_engine, MetricNames.PADCHEST_LOCALIZATION_LOSS, [PADCHEST_DATASET_ID])
        in_val = padchest_trainer.use_validation_set
        if in_val:
            attach_dataset_aware_padchest_labels_macroavgf1(validator_engine, [PADCHEST_DATASET_ID])
            attach_dataset_aware_padchest_labels_microavgf1(validator_engine, [PADCHEST_DATASET_ID])
            attach_dataset_aware_padchest_localization_macroavgf1(validator_engine, [PADCHEST_DATASET_ID])
            attach_dataset_aware_padchest_localization_microavgf1(validator_engine, [PADCHEST_DATASET_ID])
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.PADCHEST_LABEL_MACROAVGF1, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.PADCHEST_LABEL_MICROAVGF1, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.PADCHEST_LOC_MACROAVGF1, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.PADCHEST_LOC_MICROAVGF1, val=in_val)
        metrics_to_print.append(MetricNames.PADCHEST_LABEL_LOSS)
        metrics_to_print.append(MetricNames.PADCHEST_LOCALIZATION_LOSS)

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
        build_custom_checkpoint_folder_path=lambda: get_checkpoint_folder_path('visual_module', merged_dataset_name, model.get_name(),
                f'dws={",".join(map(str, _train_weights))}' if len(_train_weights) > 1 else None,
            ),
        metadata_kwargs=dict(
            model_kwargs=model_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
            iuxray_trainer_kwargs=iuxray_trainer_kwargs,
            chexpert_trainer_kwargs=chexpert_trainer_kwargs,
            cxr14_trainer_kwargs=cxr14_trainer_kwargs,
            vinbig_trainer_kwargs=vinbig_trainer_kwargs,
            padchest_trainer_kwargs=padchest_trainer_kwargs,
            dataloading_kwargs=dataloading_kwargs,
            collate_batch_fn_kwargs=collate_batch_fn_kwargs,
            train_image_transform_kwargs=train_image_transform_kwargs,
            val_image_transform_kwargs=val_image_transform_kwargs,
            training_kwargs=training_kwargs,
            trainer_engine_kwargs=trainer_engine_kwargs,
            validator_engine_kwargs=validator_engine_kwargs,
            auxiliary_tasks_kwargs=auxiliary_tasks_kwargs
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
        count_print=count_print,
        override_lr=override_lr,
    )


def train_from_scratch(
    # Model args
    visual_input_mode,
    freeze_image_encoder,
    raw_image_encoding,
    imagenet_pretrained,
    visual_features_mlp_in_dim,
    visual_features_mlp_out_dim,
    visual_features_mlp_hidden_dims,
    classification_mlp_hidden_dims,
    num_regions,   
    image_local_feat_size,
    image_encoder_pretrained_weights_path,
    pretrained_checkpoint_folder_path,
    pretrained_checkpoint_path,
    chexpert_mlc_version,
    chexpert_mlc_hidden_size,
    chest_imagenome_bbox_hidden_size,
    chest_imagenome_bbox_regressor_version,
    chest_imagenome_mlc_version,
    chest_imagenome_mlc_hidden_size,
    clip_version,
    huggingface_model_name,
    torchxrayvision_weights_name,
    detectron2_model_yaml,
    roi_heads_batch_size_per_image,
    rpn_batch_size_per_image,
    roi_align_output_size,
    yolov8_model_name_or_path,
    yolov8_model_alias,
    yolov8_use_one_detector_per_dataset,
    yolov11_model_name_or_path,
    yolov11_model_alias,
    vinbig_mlc_hidden_size,
    query_embed_size,
    local_attention_hidden_size,
    # Optimizer args
    optimizer_name,
    lr,
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
    mimiccxr_view_mode,
    iuxray_precomputed_visual_features_path,
    mimiccxr_precomputed_visual_features_path,
    chexpert_precomputed_visual_features_path,
    vinbig_precomputed_visual_features_path,
    padchest_train_study_ids_path,
    padchest_val_study_ids_path,
    padchest_test_study_ids_path,
    chest_imagenome_labels_filename,
    chest_imagenome_label_names_filename,
    use_chest_imagenome_decent_images_only,
    # Dataloading args
    train_batch_size,
    val_batch_size,
    num_workers,
    mimiccxr_weight,
    iuxray_weight,
    chexpert_weight,
    cxr14_weight,
    vinbig_weight,
    padchest_weight,
    img_aug_mode,
    horizontal_flip_prob,
    mimiccxr_balanced_sampling_mode,
    mimiccxr_balanced_batch_size,
    # Fixed traning args
    train_mimiccxr,
    train_iuxray,
    train_chexpert,
    train_vinbig,
    train_cxr14,
    train_padchest,
    vinbig_training_data_mode,
    vinbig_use_validation,
    use_vinbig_with_modified_labels,
    padchest_training_data_mode,
    padchest_use_validation,
    binary_loss_name,
    focal_loss_weight,
    bce_loss_weight,
    wbce_loss_weight,
    use_amp,
    gradient_accumulation_steps,
    # Variable traning args
    epochs,
    batches_per_epoch,
    # Auxiliary tasks args
    classify_tags,
    n_medical_tags,
    iuxray_medical_tags_per_report_filename,
    mimiccxr_medical_tags_per_report_filename,
    classify_orientation,
    classify_gender,
    classify_chexpert,
    iuxray_chexpert_labels_filename,
    mimiccxr_chexpert_labels_filename,
    classify_questions,
    n_mined_questions,
    iuxray_question_labels_filename,
    mimiccxr_question_labels_filename,
    classify_chest_imagenome,
    predict_bboxes_chest_imagenome,
    predict_labels_and_bboxes_chest_imagenome,
    pass_pred_bbox_coords_as_input,
    use_gt_bboxes_as_predictions,
    clamp_bboxes_chest_imagenome,
    chest_imagenome_bbox_loss_weight,
    use_anaxnet_bbox_subset,
    classify_labels_vinbig,
    predict_bboxes_vinbig,
    merge_findings,
    use_linear_head_for_classification,
    # GPU
    device,
    # Other args
    save,
    debug = False,
):
    print_blue('----- Training model from scratch ------', bold=True)

    assert train_mimiccxr or train_iuxray or train_chexpert or\
           train_cxr14 or train_vinbig or train_padchest, 'No dataset selected for training'
    
    use_clip = raw_image_encoding in (
                RawImageEncoding.CLIP_RESNET,
                RawImageEncoding.CLIP_VIT,
                RawImageEncoding.CLIP_VIT__HUGGINGFACE,
                RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE,
                RawImageEncoding.CLIP_RESNET__HUGGINGFACE)
    use_huggingface_vitmodel = raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE
    use_torchxrayvision_transform = raw_image_encoding in (
        RawImageEncoding.DENSENET_121__TORCHXRAYVISION,
        RawImageEncoding.RESNET__TORCHXRAYVISION,
        RawImageEncoding.RESNET_AUTOENCODER__TORCHXRAYVISION,
    )
    use_bbox_aware_transform = predict_bboxes_chest_imagenome or pass_pred_bbox_coords_as_input or predict_bboxes_vinbig
    use_detectron2 = raw_image_encoding == RawImageEncoding.DETECTRON2
    use_yolov8 = raw_image_encoding == RawImageEncoding.YOLOV8
    use_yolov11 = raw_image_encoding == RawImageEncoding.YOLOV11_FOR_DET_MLC

    if use_yolov8 or use_yolov11:
        assert predict_bboxes_vinbig or predict_bboxes_chest_imagenome
    
    if use_clip or use_huggingface_vitmodel:
        if use_clip: assert clip_version is not None
        if use_huggingface_vitmodel: assert huggingface_model_name is not None
        assert image_size == 224 or image_size == [224, 224]
        if type(image_size) is list: image_size = tuple(image_size)

    if classify_chest_imagenome:
        assert chest_imagenome_label_names_filename is not None
        n_chest_imagenome_labels = len(load_chest_imagenome_label_names(chest_imagenome_label_names_filename))
    else:
        n_chest_imagenome_labels = None

    if merge_findings:
        finding_labels_remapper, merged_finding_labels = get_merged_findings(train_chexpert, train_cxr14, train_vinbig)
        print(f'len(merged_finding_labels)={len(merged_finding_labels)}')
        n_findings = len(merged_finding_labels)
    else:
        n_findings = None
        
    model_kwargs = dict(
        pretrained_checkpoint_folder_path=pretrained_checkpoint_folder_path,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        # Image encoder
        visual_input_mode=visual_input_mode,
        raw_image_encoding=raw_image_encoding,
        freeze_image_encoder=freeze_image_encoder,
        image_local_feat_size=image_local_feat_size,
        image_encoder_pretrained_weights_path=image_encoder_pretrained_weights_path,
        imagenet_pretrained=imagenet_pretrained,
        visual_features_mlp_in_dim=visual_features_mlp_in_dim,
        visual_features_mlp_out_dim=visual_features_mlp_out_dim,
        visual_features_mlp_hidden_dims=visual_features_mlp_hidden_dims,
        clip_version=clip_version,
        huggingface_model_name=huggingface_model_name,
        torchxrayvision_weights_name=torchxrayvision_weights_name,
        detectron2_model_yaml=detectron2_model_yaml,
        num_regions=num_regions,
        roi_heads_batch_size_per_image=roi_heads_batch_size_per_image,
        rpn_batch_size_per_image=rpn_batch_size_per_image,
        roi_align_output_size=roi_align_output_size,
        yolov8_model_name_or_path=yolov8_model_name_or_path,
        yolov8_model_alias=yolov8_model_alias,
        yolov8_use_one_detector_per_dataset=yolov8_use_one_detector_per_dataset,
        yolov11_model_name_or_path=yolov11_model_name_or_path,
        yolov11_model_alias=yolov11_model_alias,
        query_embed_size=query_embed_size,
        classification_mlp_hidden_dims=classification_mlp_hidden_dims,
        local_attention_hidden_size=local_attention_hidden_size,
        image_size=image_size if isinstance(image_size, int) else image_size[0],
        # Aux tasks
        n_medical_tags=n_medical_tags,
        classify_orientation=classify_orientation,
        classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        classify_gender=classify_gender,
        classify_chest_imagenome=classify_chest_imagenome,
        classify_labels_vinbig=classify_labels_vinbig,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_labels_and_bboxes_chest_imagenome=predict_labels_and_bboxes_chest_imagenome,
        n_questions_aux_task=n_mined_questions,
        n_chest_imagenome_labels=n_chest_imagenome_labels,
        chexpert_mlc_version=chexpert_mlc_version,
        chexpert_mlc_hidden_size=chexpert_mlc_hidden_size,
        chest_imagenome_bbox_hidden_size=chest_imagenome_bbox_hidden_size,
        chest_imagenome_bbox_regressor_version=chest_imagenome_bbox_regressor_version,
        chest_imagenome_mlc_version=chest_imagenome_mlc_version,
        chest_imagenome_mlc_hidden_size=chest_imagenome_mlc_hidden_size,
        use_anaxnet_bbox_subset=use_anaxnet_bbox_subset,
        use_cxr14=train_cxr14,
        use_vinbig=train_vinbig,
        use_padchest=train_padchest,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
        vinbig_mlc_hidden_size=vinbig_mlc_hidden_size,
        merge_findings=merge_findings,
        n_findings=n_findings,
        use_linear_head_for_classification=use_linear_head_for_classification,
        use_vinbig_with_modified_labels=use_vinbig_with_modified_labels,
    )
    if predict_bboxes_chest_imagenome:
        avg_coords = get_chest_imagenome_train_average_bbox_coords(
            clamp_bbox_coords=clamp_bboxes_chest_imagenome,
            use_decent_images_only=use_chest_imagenome_decent_images_only,
        )
        if use_anaxnet_bbox_subset:
            anaxnet_bbox_indices = get_anaxnet_bbox_sorted_indices()
            avg_coords = avg_coords.reshape(-1, 4)[anaxnet_bbox_indices].reshape(-1)
        print('avg_coords.shape=', avg_coords.shape)
        avg_coords = avg_coords.tolist()
        model_kwargs['chest_imagenome_train_average_bbox_coords'] = avg_coords
    if predict_labels_and_bboxes_chest_imagenome or (classify_chest_imagenome and\
                                                      chest_imagenome_mlc_version in (MLCVersion.V1, MLCVersion.V2)):
        tmp = get_labels_per_anatomy_and_anatomy_group(chest_imagenome_label_names_filename, for_training=True)
        model_kwargs['chest_imagenome_anatomy_to_labels'] = tmp['anatomy_to_localized_labels']
        model_kwargs['chest_imagenome_anatomy_group_to_labels'] = tmp['anatomy_group_to_global_labels']
        model_kwargs['n_chest_imagenome_bboxes'] = len(tmp['anatomy_names'])
    
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
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        mimiccxr_weight=mimiccxr_weight,
        iuxray_weight=iuxray_weight,
        chexpert_weight=chexpert_weight,
        cxr14_weight=cxr14_weight,
        vinbig_weight=vinbig_weight,
        padchest_weight=padchest_weight,
    )

    if merge_findings:
        _merged_findings_kwargs = dict(
            use_merged_findings=True,
            findings_remapper=finding_labels_remapper,
            n_findings=n_findings,
        )

    # Image transforms
    train_image_transform_kwargs = {}
    val_image_transform_kwargs = {}
    _kwargs = dict(
        image_size=image_size,
        augmentation_mode=img_aug_mode,
        use_clip_transform=use_clip,
        clip_version=clip_version,
        use_huggingface_vitmodel_transform=use_huggingface_vitmodel,
        huggingface_vitmodel_name=huggingface_model_name,
        use_torchxrayvision_transform=use_torchxrayvision_transform,
        use_bbox_aware_transform=use_bbox_aware_transform,
        horizontal_flip_prob=horizontal_flip_prob,
        use_detectron2_transform=use_detectron2,
        for_yolov8=use_yolov8,
        for_yolov11=use_yolov11,
    )
    inject_mean_std_for_image_normalization(_kwargs, raw_image_encoding)
    if train_mimiccxr:
        train_image_transform_kwargs[DATASET_NAMES.MIMICCXR] = _kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.MIMICCXR] = train_image_transform_kwargs[DATASET_NAMES.MIMICCXR].copy()
        val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]['augmentation_mode'] = None # no augmentation for validation
    if train_vinbig:
        train_image_transform_kwargs[DATASET_NAMES.VINBIG] = _kwargs.copy()
        train_image_transform_kwargs[DATASET_NAMES.VINBIG]['for_vinbig'] = True
        val_image_transform_kwargs[DATASET_NAMES.VINBIG] = train_image_transform_kwargs[DATASET_NAMES.VINBIG].copy()
        val_image_transform_kwargs[DATASET_NAMES.VINBIG]['augmentation_mode'] = None # no augmentation for validation

    include_image = does_include_image(visual_input_mode)
    include_visual_features = does_include_visual_features(visual_input_mode)
    if include_visual_features:
        if train_mimiccxr: assert mimiccxr_precomputed_visual_features_path is not None
        if train_iuxray: assert iuxray_precomputed_visual_features_path is not None
        if train_chexpert: assert chexpert_precomputed_visual_features_path is not None
        if train_vinbig: assert vinbig_precomputed_visual_features_path is not None

    _kwargs = dict(
        include_image=include_image,
        include_visual_features=include_visual_features,
        classify_tags=classify_tags,
        n_tags=n_medical_tags,
        classify_orientation=classify_orientation,
        classify_gender=classify_gender,
        classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
        pass_pred_bbox_coords_as_input=pass_pred_bbox_coords_as_input,
        use_yolov8=use_yolov8,
        use_yolov11=use_yolov11,
    )
    collate_batch_fn_kwargs = {}
    if train_mimiccxr:
        if use_detectron2: # special case for detectron2
            collate_batch_fn_kwargs['mim-cim-det2'] = \
                { 'flag': 'mim-cim-det2', **_kwargs }
        else:
            collate_batch_fn_kwargs['mimiccxr'] = { 'flag': 'mimiccxr', **_kwargs }
    if train_iuxray:
        collate_batch_fn_kwargs['iuxray'] = { 'flag': 'iuxray', **_kwargs }
    if train_chexpert:
        collate_batch_fn_kwargs['chexpert'] = { 'flag': 'chexpert', **_kwargs }
    if train_cxr14:
        collate_batch_fn_kwargs['cxr14'] = { 'flag': 'cxr14', **_kwargs }
    if train_vinbig:
        collate_batch_fn_kwargs['vinbig'] = { 'flag': 'vinbig', **_kwargs }
    if train_padchest:
        collate_batch_fn_kwargs['padchest'] = { 'flag': 'padchest', **_kwargs }
    
    if train_mimiccxr:
        x = image_size if type(image_size) is int else image_size[0]
        if x > 256:
            source_image_size_mode = MIMICCXR_ImageSizeModes.MEDIUM_512
        else:
            source_image_size_mode = MIMICCXR_ImageSizeModes.SMALL_256x256
        print(f'source_image_size_mode: {source_image_size_mode}')
        mimiccxr_trainer_kwargs = dict(
            chest_imagenome_labels_filename=chest_imagenome_labels_filename,
            chest_imagenome_label_names_filename=chest_imagenome_label_names_filename,
            include_image=include_image,
            view_mode=mimiccxr_view_mode,
            source_image_size_mode=source_image_size_mode,
            use_precomputed_visual_features=include_visual_features,
            precomputed_visual_features_path=mimiccxr_precomputed_visual_features_path,
            classify_tags=classify_tags,
            medical_tags_per_report_filename=mimiccxr_medical_tags_per_report_filename,
            classify_orientation=classify_orientation,
            classify_chexpert=classify_chexpert,
            chexpert_labels_filename=mimiccxr_chexpert_labels_filename,
            classify_questions=classify_questions,
            classify_gender=classify_gender,
            classify_chest_imagenome=classify_chest_imagenome,
            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
            predict_labels_and_bboxes_chest_imagenome=predict_labels_and_bboxes_chest_imagenome,
            clamp_bboxes_chest_imagenome=clamp_bboxes_chest_imagenome,
            use_anaxnet_bbox_subset=use_anaxnet_bbox_subset,
            use_decent_images_only=use_chest_imagenome_decent_images_only,
            question_labels_filename=mimiccxr_question_labels_filename,
            data_augmentation_enabled=img_aug_mode is not None,
            use_detectron2=use_detectron2,
            balanced_sampling_mode=mimiccxr_balanced_sampling_mode,
            balanced_batch_size=mimiccxr_balanced_batch_size,
            pass_pred_bbox_coords_to_model=pass_pred_bbox_coords_as_input,
            use_gt_bboxes_as_pred=use_gt_bboxes_as_predictions,
            use_yolov8=use_yolov8,
            use_yolov11=use_yolov11,
        )
        if merge_findings:
            mimiccxr_trainer_kwargs.update(_merged_findings_kwargs)
        if use_detectron2:
            assert detectron2_model_yaml is not None
            from detectron2.config import get_cfg
            from detectron2.model_zoo import model_zoo
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(detectron2_model_yaml))
            mimiccxr_trainer_kwargs['detectron2_cfg'] = cfg
    else:
        mimiccxr_trainer_kwargs = None

    if train_iuxray:
        iuxray_trainer_kwargs=dict(
            include_image=include_image,
            use_precomputed_visual_features=include_visual_features,
            precomputed_visual_features_path=iuxray_precomputed_visual_features_path,
            classify_tags=classify_tags,
            medical_tags_per_report_filename=iuxray_medical_tags_per_report_filename,
            classify_orientation=classify_orientation,
            classify_chexpert=classify_chexpert,
            chexpert_labels_filename=iuxray_chexpert_labels_filename,
            classify_questions=classify_questions,
            question_labels_filename=iuxray_question_labels_filename,
        )
        if merge_findings:
            iuxray_trainer_kwargs.update(_merged_findings_kwargs)
    else:
        iuxray_trainer_kwargs = None

    if train_chexpert:
        chexpert_trainer_kwargs = dict(
            include_image = include_image,
            use_precomputed_visual_features = include_visual_features,
            precomputed_visual_features_path = chexpert_precomputed_visual_features_path,
        )
        if merge_findings:
            chexpert_trainer_kwargs.update(_merged_findings_kwargs)
    else:
        chexpert_trainer_kwargs = None

    if train_cxr14:
        cxr14_trainer_kwargs = {}
        if merge_findings:
            cxr14_trainer_kwargs.update(_merged_findings_kwargs)
    else:
        cxr14_trainer_kwargs = None

    if train_vinbig:
        vinbig_class_id_offset = 0
        if predict_bboxes_chest_imagenome and not (yolov8_use_one_detector_per_dataset or use_yolov11):
            vinbig_class_id_offset += CHEST_IMAGENOME_NUM_BBOX_CLASSES
        print_magenta(f'vinbig_class_id_offset: {vinbig_class_id_offset}', bold=True)
        vinbig_trainer_kwargs = dict(
            training_data_mode=vinbig_training_data_mode,
            use_validation_set=vinbig_use_validation,
            data_augmentation_enabled=img_aug_mode is not None,
            use_bounding_boxes=predict_bboxes_vinbig,
            use_yolov8=use_yolov8,
            use_yolov11=use_yolov11,
            class_id_offset=vinbig_class_id_offset,
            use_vinbig_with_modified_labels=use_vinbig_with_modified_labels,
        )
        if merge_findings:
            vinbig_trainer_kwargs.update(_merged_findings_kwargs)
    else:
        vinbig_trainer_kwargs = None

    if train_padchest:
        padchest_trainer_kwargs = dict(
            include_image=include_image,
            train_study_ids_path=padchest_train_study_ids_path,
            val_study_ids_path=padchest_val_study_ids_path,
            test_study_ids_path=padchest_test_study_ids_path,
            training_data_mode=padchest_training_data_mode,
            use_validation_set=padchest_use_validation,
        )
    else:
        padchest_trainer_kwargs = None

    trainer_engine_kwargs = dict(
        classify_tags=classify_tags,
        classify_orientation=classify_orientation,
        classify_gender=classify_gender,
        classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        classify_chest_imagenome=classify_chest_imagenome,
        classify_labels_vinbig=classify_labels_vinbig,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
        pass_pred_bbox_coords_as_input=pass_pred_bbox_coords_as_input,
        binary_loss_name=binary_loss_name,
        focal_loss_weight=focal_loss_weight,
        bce_loss_weight=bce_loss_weight,
        wbce_loss_weight=wbce_loss_weight,
        include_image=include_image,
        include_visual_features=include_visual_features,
        use_amp=use_amp,
        training=True,
        use_chexpert_dataset=train_chexpert,
        use_cxr14_dataset=train_cxr14,
        use_vinbig_dataset=train_vinbig,
        use_padchest_dataset=train_padchest,
        gradient_accumulation_steps=gradient_accumulation_steps,
        chest_imagenome_bbox_loss_weight=chest_imagenome_bbox_loss_weight,
        using_yolov8=use_yolov8,
        yolov8_use_multiple_detection_layers=yolov8_use_one_detector_per_dataset,
        using_yolov11=use_yolov11,
    )
    if use_detectron2:
        trainer_engine_kwargs['detectron2_includes_rpn'] = DETECTRON2_HAS_RPN[detectron2_model_yaml]
    if merge_findings:
        trainer_engine_kwargs.update(_merged_findings_kwargs)

    validator_engine_kwargs = dict(
        classify_tags=classify_tags,
        classify_orientation=classify_orientation,
        classify_gender=classify_gender,
        classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        classify_chest_imagenome=classify_chest_imagenome,
        classify_labels_vinbig=classify_labels_vinbig,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
        pass_pred_bbox_coords_as_input=pass_pred_bbox_coords_as_input,
        include_image=include_image,
        include_visual_features=include_visual_features,        
        training=False,
        use_vinbig_dataset=train_vinbig,
        use_padchest_dataset=train_padchest,
        use_merged_findings=merge_findings,
        using_yolov8=use_yolov8,
        yolov8_use_multiple_detection_layers=yolov8_use_one_detector_per_dataset,
        using_yolov11=use_yolov11,
    )
    if use_detectron2:
        validator_engine_kwargs['detectron2_includes_rpn'] = DETECTRON2_HAS_RPN[detectron2_model_yaml]
    
    training_kwargs = dict(
        use_amp=use_amp,
        train_mimiccxr=train_mimiccxr,
        train_iuxray=train_iuxray,
        train_chexpert=train_chexpert,
        train_cxr14=train_cxr14,
        train_vinbig=train_vinbig,
        train_padchest=train_padchest,
        binary_loss_name=binary_loss_name,
    )

    auxiliary_tasks_kwargs = dict(
        # medical tags
        classify_tags=classify_tags,
        n_medical_tags=n_medical_tags,
        iuxray_medical_tags_per_report_filename=iuxray_medical_tags_per_report_filename,
        mimiccxr_medical_tags_per_report_filename=mimiccxr_medical_tags_per_report_filename,
        # image orientation
        classify_orientation=classify_orientation,
        # gender
        classify_gender=classify_gender,
        # chexpert labels
        classify_chexpert=classify_chexpert,
        iuxray_chexpert_labels_filename=iuxray_chexpert_labels_filename,
        mimiccxr_chexpert_labels_filename=mimiccxr_chexpert_labels_filename,
        # question labels
        classify_questions=classify_questions,
        n_questions_aux_task=n_mined_questions,
        iuxray_question_labels_filename=iuxray_question_labels_filename,
        mimiccxr_question_labels_filename=mimiccxr_question_labels_filename,
        # chest imagenome labels
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        # vinbig labels
        classify_labels_vinbig=classify_labels_vinbig,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
    )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                iuxray_trainer_kwargs=iuxray_trainer_kwargs,
                chexpert_trainer_kwargs=chexpert_trainer_kwargs,
                cxr14_trainer_kwargs=cxr14_trainer_kwargs,
                vinbig_trainer_kwargs=vinbig_trainer_kwargs,
                padchest_trainer_kwargs=padchest_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                train_image_transform_kwargs=train_image_transform_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                training_kwargs=training_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                auxiliary_tasks_kwargs=auxiliary_tasks_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                num_workers=num_workers,
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
    num_workers,    
    epochs = 1,
    batches_per_epoch = 1000,
    device = 'GPU',
    save = True,
    override_lr = False,
    debug = False,
    **unused_kwargs,
):
    print_blue('----- Resuming training ------', bold=True)

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
    iuxray_trainer_kwargs = metadata['iuxray_trainer_kwargs']
    chexpert_trainer_kwargs = metadata['chexpert_trainer_kwargs']
    cxr14_trainer_kwargs = metadata['cxr14_trainer_kwargs']
    vinbig_trainer_kwargs = metadata['vinbig_trainer_kwargs']
    padchest_trainer_kwargs = metadata['padchest_trainer_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    train_image_transform_kwargs = metadata['train_image_transform_kwargs']
    val_image_transform_kwargs = metadata['val_image_transform_kwargs']
    training_kwargs = metadata['training_kwargs']
    trainer_engine_kwargs = metadata['trainer_engine_kwargs']
    validator_engine_kwargs = metadata['validator_engine_kwargs']                
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']

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
                iuxray_trainer_kwargs=iuxray_trainer_kwargs,
                chexpert_trainer_kwargs=chexpert_trainer_kwargs,
                cxr14_trainer_kwargs=cxr14_trainer_kwargs,
                vinbig_trainer_kwargs=vinbig_trainer_kwargs,
                padchest_trainer_kwargs=padchest_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                train_image_transform_kwargs=train_image_transform_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                training_kwargs=training_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                auxiliary_tasks_kwargs=auxiliary_tasks_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                num_workers=num_workers,
                device=device,
                checkpoint_folder_path=checkpoint_folder,
                save=save,
                override_lr=override_lr,
                debug=debug)

def debug_main(args):
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