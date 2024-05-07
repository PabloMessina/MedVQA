import  os
import argparse

import torch

from medvqa.datasets.chest_imagenome import get_chest_imagenome_gold_class_mask
from medvqa.datasets.chexlocalize.chexlocalize_dataset_management import CheXlocalizePhraseGroundingTrainer
from medvqa.datasets.chexpert.chexpert_dataset_management import CheXpertPhraseGroundingTrainer, CheXpertTrainingMode
from medvqa.datasets.mimiccxr import MIMICCXR_ImageSizeModes
from medvqa.datasets.mimiccxr.mimiccxr_phrase_grounding_dataset_management import MIMICCXR_PhraseGroundingTrainer
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBigPhraseGroundingTrainer, VinBigTrainingMode
from medvqa.losses import BinaryMultiLabelClassificationLossNames
from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler

from medvqa.models.phrase_grounding.phrase_grounder import PhraseGrounder

from medvqa.models.vqa.open_ended_vqa import RawImageEncoding
from medvqa.training.utils import append_metric_name, run_common_boilerplate_code_and_start_training
from medvqa.utils.constants import (
    DATASET_NAMES,
    MetricNames,
)
from medvqa.utils.common import WORKSPACE_DIR, DictWithDefault
from medvqa.metrics import (
    attach_condition_aware_accuracy,
    attach_condition_aware_chest_imagenome_bbox_iou,
    attach_condition_aware_loss,
    attach_condition_aware_prc_auc,
    attach_condition_aware_segmask_iou,
    attach_condition_aware_vinbig_bbox_iou,
)
from medvqa.models.checkpoint import (
    load_metadata,
)
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.files import get_checkpoint_folder_path
from medvqa.training.phrase_grounding import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    get_phrase_grounding_collate_batch_fn,
    multi_cyclic_dataloaders_generator,
)
from medvqa.metrics.utils import get_merge_metrics_fn
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter, print_blue, print_orange

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
    parser.add_argument('--freeze_image_encoder', action='store_true', default=False)
    parser.add_argument('--raw_image_encoding', type=str, default=RawImageEncoding.YOLOV8)
    parser.add_argument('--num_regions', type=int, default=None)
    parser.add_argument('--image_local_feat_size', type=int, default=None)
    parser.add_argument('--image_encoder_pretrained_weights_path', type=str, default=None)
    parser.add_argument('--yolov8_model_name_or_path', type=str, default=None)
    parser.add_argument('--yolov8_model_alias', type=str, default=None)
    parser.add_argument('--phrase_embedding_size', type=int, default=None)
    parser.add_argument('--regions_width', type=int, default=None)
    parser.add_argument('--regions_height', type=int, default=None)
    parser.add_argument('--qkv_size', type=int, default=None)
    parser.add_argument('--phrase_classifier_hidden_size', type=int, default=None)
    
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
    parser.add_argument('--override_lr', action='store_true', default=False)
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
    parser.add_argument('--use_weighted_phrase_classifier_loss', action='store_true', default=False)
    parser.add_argument('--cluster_and_label_weights_for_facts_filepath', type=str, default=None)
    parser.add_argument('--use_attention_regularization_loss', action='store_true', default=False)
    parser.add_argument('--use_contrastive_phrase_grounding_loss', action='store_true', default=False)

    # Dataset and dataloading arguments
    parser.add_argument('--num_train_workers', type=int, default=0, help='Number of workers for train dataloader')
    parser.add_argument('--num_val_workers', type=int, default=0, help='Number of workers for test dataloader')
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--image_size', nargs='+', type=int, default=(416, 416))
    parser.add_argument('--dicom_id_to_pos_neg_facts_filepath', type=str, default=None)
    parser.add_argument('--mscxr_phrase2embedding_filepath', type=str, default=None)
    parser.add_argument('--chest_imagenome_bbox_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--vinbig_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--chexlocalize_class_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--chexpert_class_phrase_embeddings_filepath', type=str, default=None)
    parser.add_argument('--mimiccxr_exclude_noisy_images', action='store_true', default=False)
    parser.add_argument('--mimiccxr_facts_weight', type=float, default=1.0)
    parser.add_argument('--chest_imagenome_anatlocs_weight', type=float, default=1.0)
    parser.add_argument('--mscxr_weight', type=float, default=1.0)
    parser.add_argument('--vinbig_weight', type=float, default=1.0)
    parser.add_argument('--chexlocalize_weight', type=float, default=1.0)
    parser.add_argument('--chexpert_weight', type=float, default=1.0)
    parser.add_argument('--img_aug_mode', type=str, default=None, help='Image augmentation mode')
    parser.add_argument('--pos_area_prior', type=float, default=0.4, help='Prior for positive area')
    parser.add_argument('--neg_area_prior', type=float, default=0.0, help='Prior for negative area')
    parser.add_argument('--use_mimiccxr_facts_for_train', action='store_true', default=False)
    parser.add_argument('--use_mimiccxr_facts_for_test', action='store_true', default=False)
    parser.add_argument('--use_mscxr_for_train', action='store_true', default=False)
    parser.add_argument('--use_mscxr_for_test', action='store_true', default=False)
    parser.add_argument('--use_chest_imagenome_for_train', action='store_true', default=False)
    parser.add_argument('--use_chest_imagenome_gold_for_test', action='store_true', default=False)
    parser.add_argument('--use_vinbig_for_train', action='store_true', default=False)
    parser.add_argument('--use_vinbig_for_test', action='store_true', default=False)
    parser.add_argument('--use_chexlocalize_for_train', action='store_true', default=False)
    parser.add_argument('--use_chexlocalize_for_test', action='store_true', default=False)
    parser.add_argument('--use_chexpert_for_train', action='store_true', default=False)
    parser.add_argument('--use_chexpert_for_test', action='store_true', default=False)
    parser.add_argument('--vinbig_training_data_mode', type=str, default=VinBigTrainingMode.TRAIN_ONLY, choices=VinBigTrainingMode.get_choices())
    parser.add_argument('--chexpert_training_data_mode', type=str, default=CheXpertTrainingMode.ALL, choices=CheXpertTrainingMode.get_choices())
    parser.add_argument('--mask_exponent', type=float, default=1.0)
    parser.add_argument('--mimiccxr_balance_long_middle_short_tail', action='store_true', default=False)
    parser.add_argument('--mimiccxr_long_middle_short_tail_thresholds', nargs=2, type=float, default=(0.02, 0.05))
    parser.add_argument('--mimiccxr_report_fact_nli_integrated_data_filepath', type=str, default=None)

    # Checkpoint saving arguments
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no_save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = DictWithDefault(default=1.0) # Default weight is 1.0

def _metric_getter(metrics_dict, key):
    if key.endswith('_loss'):
        return 1 / (1 + metrics_dict[key]) # convert loss to score
    return metrics_dict[key]

def train_model(
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_trainer_kwargs,
    vinbig_trainer_kwargs,
    chexlocalize_trainer_kwargs,
    chexpert_trainer_kwargs,
    dataloading_kwargs,
    collate_batch_fn_kwargs,
    train_image_transform_kwargs,
    val_image_transform_kwargs,
    trainer_engine_kwargs,
    validator_engine_kwargs,
    epochs,
    batches_per_epoch,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    val_batch_size_factor,
    num_train_workers,
    num_val_workers,
    device = 'GPU',
    checkpoint_folder_path = None,
    save = True,
    override_lr = False,
    debug = False,
):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    use_mimiccxr_facts_for_train = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_facts_for_train', False)
    use_mimiccxr_facts_for_test = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_facts_for_test', False)
    use_mscxr_for_test = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_mscxr_for_test', False)
    use_mscxr_for_train = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_mscxr_for_train', False)
    use_chest_imagenome_for_train = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_chest_imagenome_for_train', False)
    use_chest_imagenome_gold_for_test = mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_chest_imagenome_gold_for_test', False)
    use_yolov8 = model_kwargs['raw_image_encoding'] == RawImageEncoding.YOLOV8
    use_vinbig_for_train = vinbig_trainer_kwargs is not None and vinbig_trainer_kwargs.get('use_training_set', False)
    use_vinbig_for_test = vinbig_trainer_kwargs is not None and vinbig_trainer_kwargs.get('use_validation_set', False)
    use_chexlocalize_for_train = chexlocalize_trainer_kwargs is not None and chexlocalize_trainer_kwargs.get('use_training_set', False)
    use_chexlocalize_for_test = chexlocalize_trainer_kwargs is not None and chexlocalize_trainer_kwargs.get('use_validation_set', False)
    use_chexpert_for_train = chexpert_trainer_kwargs is not None and chexpert_trainer_kwargs.get('use_training_set', False)
    use_chexpert_for_test = chexpert_trainer_kwargs is not None and chexpert_trainer_kwargs.get('use_validation_set', False)
    use_attention_regularization_loss = trainer_engine_kwargs['use_attention_regularization_loss']
    use_contrastive_phrase_grounding_loss = trainer_engine_kwargs['use_contrastive_phrase_grounding_loss']

    # Sanity checks
    if use_chest_imagenome_gold_for_test:
        assert use_chest_imagenome_for_train
    if use_mscxr_for_test:
        assert use_mimiccxr_facts_for_train or use_mscxr_for_train
    if use_chexpert_for_test:
        assert use_chexpert_for_train

    if use_mimiccxr_facts_for_train:
        if dataloading_kwargs['mimiccxr_facts_weight'] == 0:
            print_orange('WARNING: use_mimiccxr_facts_for_train is True but mimiccxr_facts_weight is 0', bold=True)
            use_mimiccxr_facts_for_train = False
    if use_chest_imagenome_for_train:
        if dataloading_kwargs['chest_imagenome_anatlocs_weight'] == 0:
            print_orange('WARNING: use_chest_imagenome_for_train is True but chest_imagenome_anatlocs_weight is 0', bold=True)
            use_chest_imagenome_for_train = False
    if use_mscxr_for_train:
        if dataloading_kwargs['mscxr_weight'] == 0:
            print_orange('WARNING: use_mscxr_for_train is True but mscxr_weight is 0', bold=True)
            use_mscxr_for_train = False
    if use_vinbig_for_train:
        if dataloading_kwargs['vinbig_weight'] == 0:
            print_orange('WARNING: use_vinbig_for_train is True but vinbig_weight is 0', bold=True)
            use_vinbig_for_train = False
    if use_chexlocalize_for_train:
        if dataloading_kwargs['chexlocalize_weight'] == 0:
            print_orange('WARNING: use_chexlocalize_for_train is True but chexlocalize_weight is 0', bold=True)
            use_chexlocalize_for_train = False

    use_mimiccxr = use_mimiccxr_facts_for_train or use_mscxr_for_test or use_mscxr_for_train or\
                        use_chest_imagenome_for_train or use_chest_imagenome_gold_for_test
    
    use_chexlocalize = use_chexlocalize_for_train or use_chexlocalize_for_test

    use_vinbig = use_vinbig_for_train or use_vinbig_for_test

    use_chexpert = use_chexpert_for_train or use_chexpert_for_test

    assert sum([use_mimiccxr_facts_for_train, use_chest_imagenome_for_train, use_mscxr_for_train,
                use_vinbig_for_train, use_chexlocalize_for_train, use_chexpert_for_train]) > 0

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Create model
    count_print('Creating instance of PhraseGrounder ...')
    model = PhraseGrounder(**model_kwargs)
    model = model.to(device)

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

    # Create CheXLocalize trainer
    if use_chexlocalize:
        count_print('Creating CheXLocalize Phrase Grounding Trainer ...')
        chexlocalize_trainer = CheXlocalizePhraseGroundingTrainer(
            train_image_transform=get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE]),
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE]),
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
        count_print('Creating CheXpert Phrase Grounding Trainer ...')
        chexpert_trainer = CheXpertPhraseGroundingTrainer(
            train_image_transform=get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.CHEXPERT]),
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.CHEXPERT]),
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
        count_print('Creating VinBig Phrase Grounding Trainer ...')
        vinbig_trainer = VinBigPhraseGroundingTrainer(
            train_image_transform=get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.VINBIG]),
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.VINBIG]),
            collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['vbg']),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            test_batch_size_factor=val_batch_size_factor,
            num_train_workers=num_train_workers,
            num_val_workers=num_val_workers,
            **vinbig_trainer_kwargs,
        )

    # Create MIMIC-CXR trainer
    if use_mimiccxr:
        count_print('Creating MIMIC-CXR Phrase Grounding Trainer ...')
        if use_mimiccxr_facts_for_train or use_mimiccxr_facts_for_test:
            fact_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['fg'])
        else:
            fact_grounding_collate_batch_fn = None
        if use_mscxr_for_train or use_mscxr_for_test:
            phrase_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['pg'])
        else:
            phrase_grounding_collate_batch_fn = None
        if use_chest_imagenome_for_train or use_chest_imagenome_gold_for_test:
            bbox_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cibg'])
        else:
            bbox_grounding_collate_batch_fn = None
        mimiccxr_trainer = MIMICCXR_PhraseGroundingTrainer(
            train_image_transform = get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            test_image_transform = get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            max_phrases_per_image=max_phrases_per_image,
            test_batch_size_factor=val_batch_size_factor,
            fact_grounding_collate_batch_fn=fact_grounding_collate_batch_fn,
            phrase_grounding_collate_batch_fn=phrase_grounding_collate_batch_fn,
            bbox_grounding_collate_batch_fn=bbox_grounding_collate_batch_fn,
            num_train_workers=num_train_workers,
            num_test_workers=num_val_workers,
            **mimiccxr_trainer_kwargs,
        )

    if debug: # if debugging
        output = {}
        if use_mimiccxr:
            output['mimiccxr_trainer'] = mimiccxr_trainer
        if use_vinbig:
            output['vinbig_trainer'] = vinbig_trainer
        if use_chexlocalize:
            output['chexlocalize_trainer'] = chexlocalize_trainer
        if use_chexpert:
            output['chexpert_trainer'] = chexpert_trainer
        return output

    # Create complex dataloaders
    count_print('Creating dataloaders ...')
    
    _train_weights = []
    _train_dataloaders = []
    _val_dataloaders = []
    _dataset_names = []

    if use_mimiccxr_facts_for_train:
        _dataset_names.append('mim-facts')
        _train_weights.append(dataloading_kwargs['mimiccxr_facts_weight'])
        _train_dataloaders.append(mimiccxr_trainer.train_fact_dataloader)
        print(f'len(mimiccxr_trainer.train_fact_dataloader) = {len(mimiccxr_trainer.train_fact_dataloader)}')

    if use_mimiccxr_facts_for_test:
        _val_dataloaders.append(mimiccxr_trainer.test_fact_dataloader)
        print(f'len(mimiccxr_trainer.test_fact_dataloader) = {len(mimiccxr_trainer.test_fact_dataloader)}')

    if use_mscxr_for_train:
        _dataset_names.append('mscxr')
        _train_weights.append(dataloading_kwargs['mscxr_weight'])
        # _train_dataloaders.append(mimiccxr_trainer.train_mscxr_dataloader)
        _train_dataloaders.append(mimiccxr_trainer.test_mscxr_dataloader)
        print(f'len(mimiccxr_trainer.train_mscxr_dataloader) = {len(mimiccxr_trainer.train_mscxr_dataloader)}')

    if use_mscxr_for_test:
        _val_dataloaders.append(mimiccxr_trainer.test_mscxr_dataloader)
        print(f'len(mimiccxr_trainer.test_mscxr_dataloader) = {len(mimiccxr_trainer.test_mscxr_dataloader)}')

    if use_chest_imagenome_for_train:
        _dataset_names.append('chst-img-anat')
        _train_weights.append(dataloading_kwargs['chest_imagenome_anatlocs_weight'])
        _train_dataloaders.append(mimiccxr_trainer.train_chest_imagenome_dataloader)
        print(f'len(mimiccxr_trainer.train_chest_imagenome_dataloader) = {len(mimiccxr_trainer.train_chest_imagenome_dataloader)}')

    if use_chest_imagenome_gold_for_test:
        _val_dataloaders.append(mimiccxr_trainer.test_chest_imagenome_gold_dataloader)
        print(f'len(mimiccxr_trainer.test_chest_imagenome_gold_dataloader) = {len(mimiccxr_trainer.test_chest_imagenome_gold_dataloader)}')

    if use_vinbig_for_train:
        _dataset_names.append('vinbig')
        _train_weights.append(dataloading_kwargs['vinbig_weight'])
        _train_dataloaders.append(vinbig_trainer.train_dataloader)
        print(f'len(vinbig_trainer.train_dataloader) = {len(vinbig_trainer.train_dataloader)}')

    if use_vinbig_for_test:
        _val_dataloaders.append(vinbig_trainer.val_dataloader)
        print(f'len(vinbig_trainer.val_dataloader) = {len(vinbig_trainer.val_dataloader)}')

    if use_chexlocalize_for_train:
        _dataset_names.append('chexloc')
        _train_weights.append(dataloading_kwargs['chexlocalize_weight'])
        _train_dataloaders.append(chexlocalize_trainer.train_dataloader)
        # _train_dataloaders.append(chexlocalize_trainer.val_dataloader) # for debugging
        print(f'len(chexlocalize_trainer.train_dataloader) = {len(chexlocalize_trainer.train_dataloader)}')

    if use_chexlocalize_for_test:
        _val_dataloaders.append(chexlocalize_trainer.val_dataloader)
        print(f'len(chexlocalize_trainer.val_dataloader) = {len(chexlocalize_trainer.val_dataloader)}')

    if use_chexpert_for_train:
        _dataset_names.append('chxp')
        _train_weights.append(dataloading_kwargs['chexpert_weight'])
        _train_dataloaders.append(chexpert_trainer.train_dataloader)
        print(f'len(chexpert_trainer.train_dataloader) = {len(chexpert_trainer.train_dataloader)}')

    if use_chexpert_for_test:
        _val_dataloaders.append(chexpert_trainer.val_dataloader)
        print(f'len(chexpert_trainer.val_dataloader) = {len(chexpert_trainer.val_dataloader)}')
    
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

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []

    attach_condition_aware_loss(trainer_engine, 'loss')
    metrics_to_print.append('loss')

    if use_mimiccxr_facts_for_train or use_mimiccxr_facts_for_test:
        _cond_func = lambda x: x['flag'] == 'fg'
        in_train = use_mimiccxr_facts_for_train
        in_val = use_mimiccxr_facts_for_test
        if in_train:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(trainer_engine, 'attention_regularization_loss', _cond_func, 'fg_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:                
                attach_condition_aware_loss(trainer_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'fg_cpg_loss')
            attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'fg_phrcls_loss')
        if in_val:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(validator_engine, 'attention_regularization_loss', _cond_func, 'fg_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:                
                attach_condition_aware_loss(validator_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'fg_cpg_loss')
            attach_condition_aware_loss(validator_engine, 'phrase_classifier_loss', _cond_func, 'fg_phrcls_loss')
            attach_condition_aware_prc_auc(validator_engine, 'classifier_sigmoids', 'gt_labels', 'fg_prc_auc', _cond_func)
        # for logging
        if use_attention_regularization_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'fg_att_reg_loss', train=in_train, val=in_val)
        if use_contrastive_phrase_grounding_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'fg_cpg_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'fg_phrcls_loss', train=in_train, val=in_val)
        if in_val:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'fg_prc_auc', train=False, val=True)
    
    if use_chest_imagenome_for_train:
        assert use_yolov8 # TODO: eventually support other bbox predictors
        _cond_func = lambda x: x['flag'] == 'cibg'
        attach_condition_aware_loss(trainer_engine, MetricNames.YOLOV8_LOSS, _cond_func, 'cibg_y8_loss')
        attach_condition_aware_loss(trainer_engine, MetricNames.YOLOV8_BOX_LOSS, _cond_func, 'cibg_y8_box_loss')
        attach_condition_aware_loss(trainer_engine, MetricNames.YOLOV8_CLS_LOSS, _cond_func, 'cibg_y8_cls_loss')
        attach_condition_aware_loss(trainer_engine, MetricNames.YOLOV8_DFL_LOSS, _cond_func, 'cibg_y8_dfl_loss')
        if use_chest_imagenome_gold_for_test:
            _gold_class_mask = get_chest_imagenome_gold_class_mask()
            attach_condition_aware_chest_imagenome_bbox_iou(
                validator_engine, _cond_func, use_yolov8=True, class_mask=_gold_class_mask, metric_name='cibg_y8_bbox_iou')
        # for logging
        metrics_to_print.append('cibg_y8_loss')
        metrics_to_print.append('cibg_y8_box_loss')
        metrics_to_print.append('cibg_y8_cls_loss')
        metrics_to_print.append('cibg_y8_dfl_loss')
        metrics_to_print.append('cibg_phrcls_loss')
        if use_chest_imagenome_gold_for_test:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cibg_y8_bbox_iou', train=False)

    if use_chest_imagenome_for_train or use_chest_imagenome_gold_for_test:
        _cond_func = lambda x: x['flag'] == 'cibg'
        in_train = use_chest_imagenome_for_train
        in_val = use_chest_imagenome_gold_for_test
        if in_train:
            attach_condition_aware_loss(trainer_engine,'attention_supervision_loss', _cond_func, 'cibg_att_sup_loss')
            attach_condition_aware_segmask_iou(trainer_engine, 'pred_mask', 'gt_mask', 'cibg_segmask_iou', _cond_func)
        if in_val:
            attach_condition_aware_loss(validator_engine,'attention_supervision_loss', _cond_func, 'cibg_att_sup_loss')
            attach_condition_aware_segmask_iou(validator_engine, 'pred_mask', 'gt_mask', 'cibg_segmask_iou', _cond_func)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cibg_att_sup_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cibg_segmask_iou', train=in_train, val=in_val)

    if use_mscxr_for_train or use_mscxr_for_test:
        _cond_func = lambda x: x['flag'] == 'pg'
        in_train = use_mscxr_for_train
        in_val = use_mscxr_for_test
        if in_train:
            attach_condition_aware_loss(trainer_engine,'attention_supervision_loss', _cond_func, 'pg_att_sup_loss')
            attach_condition_aware_segmask_iou(trainer_engine, 'pred_mask', 'gt_mask', 'pg_segmask_iou', _cond_func)
        if in_val:
            attach_condition_aware_loss(validator_engine,'attention_supervision_loss', _cond_func, 'pg_att_sup_loss')
            attach_condition_aware_segmask_iou(validator_engine, 'pred_mask', 'gt_mask', 'pg_segmask_iou', _cond_func)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'pg_att_sup_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'pg_segmask_iou', train=in_train, val=in_val)

    if use_vinbig_for_train:
        _cond_func = lambda x: x['flag'] == 'vbg'
        if use_yolov8:
            attach_condition_aware_loss(trainer_engine, MetricNames.YOLOV8_LOSS, _cond_func, 'vbg_y8_loss')
            attach_condition_aware_loss(trainer_engine, MetricNames.YOLOV8_BOX_LOSS, _cond_func, 'vbg_y8_box_loss')
            attach_condition_aware_loss(trainer_engine, MetricNames.YOLOV8_CLS_LOSS, _cond_func, 'vbg_y8_cls_loss')
            attach_condition_aware_loss(trainer_engine, MetricNames.YOLOV8_DFL_LOSS, _cond_func, 'vbg_y8_dfl_loss')
        attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'vbg_phrcls_loss')
        attach_condition_aware_prc_auc(trainer_engine, 'classifier_sigmoids', 'gt_labels', 'vbg_prc_auc', _cond_func)
        if use_vinbig_for_test:
            if use_yolov8:
                attach_condition_aware_vinbig_bbox_iou(
                    validator_engine, _cond_func, use_yolov8=True, metric_name='vbg_y8_bbox_iou')
            attach_condition_aware_prc_auc(validator_engine, 'classifier_sigmoids', 'gt_labels', 'vbg_prc_auc', _cond_func)
        # for logging
        if use_yolov8:
            metrics_to_print.append('vbg_y8_loss')
            metrics_to_print.append('vbg_y8_box_loss')
            metrics_to_print.append('vbg_y8_cls_loss')
            metrics_to_print.append('vbg_y8_dfl_loss')
        metrics_to_print.append('vbg_phrcls_loss')
        if use_vinbig_for_test:
            if use_yolov8:
                append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vbg_y8_bbox_iou', train=False)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vbg_prc_auc', train=True, val=use_vinbig_for_test)
    
    if use_vinbig_for_train or use_vinbig_for_test:
        _cond_func = lambda x: x['flag'] == 'vbg'
        in_train = use_vinbig_for_train
        in_val = use_vinbig_for_test
        if in_train:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(trainer_engine, 'attention_regularization_loss', _cond_func, 'vbg_att_reg_loss')
            attach_condition_aware_loss(trainer_engine, 'attention_supervision_loss', _cond_func, 'vbg_att_sup_loss')
            attach_condition_aware_segmask_iou(trainer_engine, 'pred_mask', 'gt_mask', 'vbg_segmask_iou', _cond_func)
        if in_val:
            attach_condition_aware_loss(validator_engine,'attention_supervision_loss', _cond_func, 'vbg_att_sup_loss')
            attach_condition_aware_segmask_iou(validator_engine, 'pred_mask', 'gt_mask', 'vbg_segmask_iou', _cond_func)
        # for logging
        if use_attention_regularization_loss:            
            metrics_to_print.append('vbg_att_reg_loss')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vbg_att_sup_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'vbg_segmask_iou', train=in_train, val=in_val)

    if use_chexlocalize:
        _cond_func = lambda x: x['flag'] == 'cl'
        in_train = use_chexlocalize_for_train
        in_val = use_chexlocalize_for_test
        if in_train:
            attach_condition_aware_loss(trainer_engine,'attention_supervision_loss', _cond_func, 'cl_att_sup_loss')
            attach_condition_aware_segmask_iou(trainer_engine, 'pred_mask', 'gt_mask', 'cl_segmask_iou', _cond_func)
            attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'cl_phrcls_loss')
            attach_condition_aware_accuracy(trainer_engine, 'pred_phrase_labels', 'gt_phrase_labels', 'cl_phrase_acc', _cond_func)
        if in_val:
            attach_condition_aware_loss(validator_engine,'attention_supervision_loss', _cond_func, 'cl_att_sup_loss')
            attach_condition_aware_segmask_iou(validator_engine, 'pred_mask', 'gt_mask', 'cl_segmask_iou', _cond_func)
            attach_condition_aware_accuracy(validator_engine, 'pred_phrase_labels', 'gt_phrase_labels', 'cl_phrase_acc', _cond_func)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cl_att_sup_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cl_segmask_iou', train=in_train, val=in_val)
        metrics_to_print.append('cl_phrcls_loss')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cl_phrase_acc', train=in_train, val=in_val)

    if use_chexpert:
        _cond_func = lambda x: x['flag'] == 'chxp'
        in_train = use_chexpert_for_train
        in_val = use_chexpert_for_test
        if in_train:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(trainer_engine, 'attention_regularization_loss', _cond_func, 'chxp_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:
                attach_condition_aware_loss(trainer_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'chxp_cpg_loss')
            attach_condition_aware_loss(trainer_engine, 'phrase_classifier_loss', _cond_func, 'chxp_phrcls_loss')
        if in_val:
            if use_attention_regularization_loss:
                attach_condition_aware_loss(validator_engine, 'attention_regularization_loss', _cond_func, 'chxp_att_reg_loss')
            if use_contrastive_phrase_grounding_loss:                
                attach_condition_aware_loss(validator_engine, 'contrastive_phrase_grounding_loss', _cond_func, 'chxp_cpg_loss')
            attach_condition_aware_loss(validator_engine, 'phrase_classifier_loss', _cond_func, 'chxp_phrcls_loss')
            attach_condition_aware_prc_auc(validator_engine, 'classifier_sigmoids', 'gt_labels', 'chxp_prc_auc', _cond_func)
        # for logging
        if use_attention_regularization_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chxp_att_reg_loss', train=in_train, val=in_val)
        if use_contrastive_phrase_grounding_loss:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chxp_cpg_loss', train=in_train, val=in_val)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chxp_phrcls_loss', train=in_train, val=in_val)
        if in_val:
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chxp_prc_auc', train=False, val=True)

    # Score function
    assert len(val_metrics_to_merge) > 0
    if len(train_metrics_to_merge) > 0:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0.1, 0.9, _metric_getter)
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
            chexlocalize_trainer_kwargs=chexlocalize_trainer_kwargs,
            dataloading_kwargs=dataloading_kwargs,
            collate_batch_fn_kwargs=collate_batch_fn_kwargs,
            train_image_transform_kwargs=train_image_transform_kwargs,
            val_image_transform_kwargs=val_image_transform_kwargs,
            trainer_engine_kwargs=trainer_engine_kwargs,
            validator_engine_kwargs=validator_engine_kwargs,
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
    freeze_image_encoder,
    raw_image_encoding,
    num_regions,   
    image_local_feat_size,
    image_encoder_pretrained_weights_path,
    pretrained_checkpoint_folder_path,
    pretrained_checkpoint_folder_paths,
    yolov8_model_name_or_path,
    yolov8_model_alias,
    phrase_embedding_size,
    regions_width,
    regions_height,
    qkv_size,
    phrase_classifier_hidden_size,
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
    chest_imagenome_bbox_phrase_embeddings_filepath,
    vinbig_phrase_embeddings_filepath,
    chexlocalize_class_phrase_embeddings_filepath,
    chexpert_class_phrase_embeddings_filepath,
    mimiccxr_exclude_noisy_images,
    mimiccxr_balance_long_middle_short_tail,
    mimiccxr_long_middle_short_tail_thresholds,
    mimiccxr_report_fact_nli_integrated_data_filepath,
    # Dataloading args
    mimiccxr_facts_weight,
    chest_imagenome_anatlocs_weight,
    mscxr_weight,
    vinbig_weight,
    chexlocalize_weight,
    chexpert_weight,
    img_aug_mode,
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
    use_mscxr_for_test,
    use_chest_imagenome_for_train,
    use_chest_imagenome_gold_for_test,
    use_vinbig_for_train,
    use_vinbig_for_test,
    use_chexlocalize_for_train,
    use_chexlocalize_for_test,
    use_chexpert_for_train,
    use_chexpert_for_test,
    vinbig_training_data_mode,
    chexpert_training_data_mode,
    use_amp,
    gradient_accumulation_steps,
    pos_area_prior,
    neg_area_prior,
    mask_exponent,    
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
    # Variable traning args
    epochs,
    batches_per_epoch,
    # GPU
    device,
    # Other args
    save,
    debug = False,
):
    print_blue('----- Training model from scratch ------', bold=True)
    
    use_yolov8 = raw_image_encoding == RawImageEncoding.YOLOV8
    use_bbox_aware_transform = use_yolov8
    predict_bboxes_chest_imagenome = use_chest_imagenome_for_train or use_chest_imagenome_gold_for_test
    use_mimiccxr = use_mimiccxr_facts_for_train or use_mscxr_for_train or use_mscxr_for_test or\
                     use_chest_imagenome_for_train or use_chest_imagenome_gold_for_test
    use_vinbig = use_vinbig_for_train or use_vinbig_for_test
    use_chexlocalize = use_chexlocalize_for_train or use_chexlocalize_for_test
    use_chexpert = use_chexpert_for_train or use_chexpert_for_test
    predict_bboxes_vinbig = (use_vinbig_for_train or use_vinbig_for_test) and use_yolov8
    yolov8_use_multiple_detection_layers = predict_bboxes_chest_imagenome and predict_bboxes_vinbig

    assert use_mimiccxr or use_vinbig or use_chexlocalize
    assert use_chest_imagenome_gold_for_test or use_mscxr_for_test or use_vinbig_for_test or use_chexlocalize_for_test or\
           use_mimiccxr_facts_for_test
    if use_chest_imagenome_gold_for_test:
        assert use_chest_imagenome_for_train
    if use_mscxr_for_test:
        assert use_mimiccxr_facts_for_train or use_mscxr_for_train
    if use_vinbig_for_test:
        assert use_vinbig_for_train
    if use_chexlocalize_for_test:
        assert use_chexlocalize_for_train
    if use_mimiccxr_facts_for_test:
        assert use_mimiccxr_facts_for_train
    
    model_kwargs = dict(
        pretrained_checkpoint_folder_path=pretrained_checkpoint_folder_path,
        pretrained_checkpoint_folder_paths=pretrained_checkpoint_folder_paths,
        # Image encoder
        raw_image_encoding=raw_image_encoding,
        freeze_image_encoder=freeze_image_encoder,
        image_local_feat_size=image_local_feat_size,
        image_encoder_pretrained_weights_path=image_encoder_pretrained_weights_path,
        num_regions=num_regions,
        yolov8_model_name_or_path=yolov8_model_name_or_path,
        yolov8_model_alias=yolov8_model_alias,
        yolov8_use_one_detector_per_dataset=(predict_bboxes_chest_imagenome and predict_bboxes_vinbig),
        # Aux tasks
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
        # Other
        apply_positional_encoding=True,
        phrase_embedding_size=phrase_embedding_size,
        regions_width=regions_width,
        regions_height=regions_height,
        qkv_size=qkv_size,
        phrase_classifier_hidden_size=phrase_classifier_hidden_size,
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
        chest_imagenome_anatlocs_weight=chest_imagenome_anatlocs_weight,
        mscxr_weight=mscxr_weight,
        vinbig_weight=vinbig_weight,
        chexlocalize_weight=chexlocalize_weight,
        chexpert_weight=chexpert_weight,
    )

    # Image transforms
    train_image_transform_kwargs = {}
    val_image_transform_kwargs = {}
    _kwargs = dict(
        image_size=image_size,
        augmentation_mode=img_aug_mode,
        use_bbox_aware_transform=use_bbox_aware_transform,
        for_yolov8=use_yolov8,
    )
    if use_mimiccxr:
        train_image_transform_kwargs[DATASET_NAMES.MIMICCXR] = _kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.MIMICCXR] = train_image_transform_kwargs[DATASET_NAMES.MIMICCXR].copy()
        val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]['augmentation_mode'] = None # no augmentation for validation
    if use_vinbig:
        train_image_transform_kwargs[DATASET_NAMES.VINBIG] = _kwargs.copy()
        train_image_transform_kwargs[DATASET_NAMES.VINBIG]['for_vinbig'] = True
        val_image_transform_kwargs[DATASET_NAMES.VINBIG] = train_image_transform_kwargs[DATASET_NAMES.VINBIG].copy()
        val_image_transform_kwargs[DATASET_NAMES.VINBIG]['augmentation_mode'] = None # no augmentation for validation
    if use_chexlocalize:
        train_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE] = _kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE] = train_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE].copy()
        val_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE]['augmentation_mode'] = None # no augmentation for validation
    if use_chexpert:
        train_image_transform_kwargs[DATASET_NAMES.CHEXPERT] = _kwargs.copy()
        val_image_transform_kwargs[DATASET_NAMES.CHEXPERT] = train_image_transform_kwargs[DATASET_NAMES.CHEXPERT].copy()
        val_image_transform_kwargs[DATASET_NAMES.CHEXPERT]['augmentation_mode'] = None # no augmentation for validation
    
    # Collate batch functions
    _kwargs = dict(
        use_yolov8=use_yolov8,
    )
    collate_batch_fn_kwargs = {}
    if use_mimiccxr:
        include_loss_weights = binary_multilabel_classif_loss_name in [
            BinaryMultiLabelClassificationLossNames.WBCE,
            BinaryMultiLabelClassificationLossNames.FOCAL_BCE_WBCE,
        ]
        print(f'include_loss_weights = {include_loss_weights}')
        # fact grounding
        collate_batch_fn_kwargs['fg'] = { 'flag': 'fg', 'include_loss_weights': include_loss_weights, **_kwargs }
        # phrase grounding
        collate_batch_fn_kwargs['pg'] = { 'flag': 'pg', **_kwargs }
        # chest imagenome bbox grounding
        collate_batch_fn_kwargs['cibg'] = { 'flag': 'cibg', **_kwargs }
    if use_vinbig:
        collate_batch_fn_kwargs['vbg'] = { 'flag': 'vbg', **_kwargs }
    if use_chexlocalize:
        collate_batch_fn_kwargs['cl'] = { 'flag': 'cl', **_kwargs }
    if use_chexpert:
        collate_batch_fn_kwargs['chxp'] = { 'flag': 'chxp', **_kwargs }
    
    if use_mimiccxr:
        x = image_size if type(image_size) is int else image_size[0]
        if x > 256:
            source_image_size_mode = MIMICCXR_ImageSizeModes.MEDIUM_512
        else:
            source_image_size_mode = MIMICCXR_ImageSizeModes.SMALL_256x256
        print(f'source_image_size_mode: {source_image_size_mode}')
        mimiccxr_trainer_kwargs = dict(
            mask_width=regions_width,
            mask_height=regions_height,
            use_facts_for_train=use_mimiccxr_facts_for_train,            
            use_facts_for_test=use_mimiccxr_facts_for_test,
            dicom_id_to_pos_neg_facts_filepath=dicom_id_to_pos_neg_facts_filepath,
            use_mscxr_for_train=use_mscxr_for_train,
            use_mscxr_for_test=use_mscxr_for_test,
            mscxr_phrase2embedding_filepath=mscxr_phrase2embedding_filepath,
            use_chest_imagenome_for_train=use_chest_imagenome_for_train,
            use_chest_imagenome_gold_for_test=use_chest_imagenome_gold_for_test,
            chest_imagenome_bbox_phrase_embeddings_filepath=chest_imagenome_bbox_phrase_embeddings_filepath,
            source_image_size_mode=source_image_size_mode,
            exclude_noisy_images=mimiccxr_exclude_noisy_images,
            use_yolov8=use_yolov8,
            mask_exponent=mask_exponent,
            balance_long_middle_short_tail=mimiccxr_balance_long_middle_short_tail,
            long_middle_short_tail_thresholds=mimiccxr_long_middle_short_tail_thresholds,
            report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
            use_weighted_phrase_classifier_loss=use_weighted_phrase_classifier_loss,
            cluster_and_label_weights_for_facts_filepath=cluster_and_label_weights_for_facts_filepath,
        )
    else:
        mimiccxr_trainer_kwargs = None

    if use_vinbig:
        vinbig_trainer_kwargs = dict(
            training_data_mode=vinbig_training_data_mode,
            use_training_set=use_vinbig_for_train,
            use_validation_set=use_vinbig_for_test,
            data_augmentation_enabled=img_aug_mode is not None,
            use_yolov8=use_yolov8,
            mask_height=regions_height,
            mask_width=regions_width,
            phrase_embeddings_filepath=vinbig_phrase_embeddings_filepath,
        )
    else:
        vinbig_trainer_kwargs = None

    if use_chexlocalize:
        chexlocalize_trainer_kwargs = dict(
            use_training_set=use_chexlocalize_for_train,
            use_validation_set=use_chexlocalize_for_test,
            mask_height=regions_height,
            mask_width=regions_width,
            class_phrase_embeddings_filepath=chexlocalize_class_phrase_embeddings_filepath,
        )
    else:
        chexlocalize_trainer_kwargs = None

    if use_chexpert:
        chexpert_trainer_kwargs = dict(
            training_data_mode=chexpert_training_data_mode,
            use_training_set=use_chexpert_for_train,
            use_validation_set=use_chexpert_for_test,
            phrase_embeddings_filepath=chexpert_class_phrase_embeddings_filepath,
        )
    else:
        chexpert_trainer_kwargs = None

    trainer_engine_kwargs = dict(
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=use_amp, training=True, validating=False, testing=False,
        using_yolov8=use_yolov8,
        yolov8_use_multiple_detection_layers=yolov8_use_multiple_detection_layers,
        pos_area_prior=pos_area_prior,
        neg_area_prior=neg_area_prior,
        max_grad_norm=max_grad_norm,
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
    )

    validator_engine_kwargs = dict(
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_bboxes_vinbig=predict_bboxes_vinbig,
        training=False, validating=True, testing=False,
        using_yolov8=use_yolov8,
        yolov8_use_multiple_detection_layers=yolov8_use_multiple_detection_layers,
        binary_multilabel_classif_loss_name=binary_multilabel_classif_loss_name,
        focal_loss_weight=focal_loss_weight,
        bce_loss_weight=bce_loss_weight,
        wbce_loss_weight=wbce_loss_weight,
    )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                vinbig_trainer_kwargs=vinbig_trainer_kwargs,
                chexlocalize_trainer_kwargs=chexlocalize_trainer_kwargs,
                chexpert_trainer_kwargs=chexpert_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                train_image_transform_kwargs=train_image_transform_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
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
    vinbig_trainer_kwargs = metadata['vinbig_trainer_kwargs']
    chexlocalize_trainer_kwargs = metadata.get('chexlocalize_trainer_kwargs', None) # backward compatibility
    chexpert_trainer_kwargs = metadata.get('chexpert_trainer_kwargs', None) # backward compatibility
    dataloading_kwargs = metadata['dataloading_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    train_image_transform_kwargs = metadata['train_image_transform_kwargs']
    val_image_transform_kwargs = metadata['val_image_transform_kwargs']
    trainer_engine_kwargs = metadata['trainer_engine_kwargs']
    validator_engine_kwargs = metadata['validator_engine_kwargs']

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
                chexlocalize_trainer_kwargs=chexlocalize_trainer_kwargs,
                chexpert_trainer_kwargs=chexpert_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                train_image_transform_kwargs=train_image_transform_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
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