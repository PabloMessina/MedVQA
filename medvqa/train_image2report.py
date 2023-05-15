import  os
import argparse

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    get_chest_imagenome_train_average_bbox_coords,
    get_labels_per_anatomy_and_anatomy_group,
    load_chest_imagenome_label_names,
)
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR, MIMICCXR_ImageSizeModes
from medvqa.datasets.mimiccxr.mimiccxr_image2report_dataset_management import MIMICCXR_Image2ReportTrainer
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.common import load_model_state_dict
from medvqa.models.report_generation.image2report import Image2ReportModel
from medvqa.models.vision.multilabel_classification import MLCVersion

from medvqa.models.vqa.open_ended_vqa import RawImageEncoding
from medvqa.training.utils import append_metric_name
from medvqa.utils.constants import (
    DATASET_NAMES,
    MIMICCXR_DATASET_ID,
    MetricNames,
)
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_dataset_aware_chest_imagenome_bbox_mae,
    attach_dataset_aware_chest_imagenome_bbox_iou,
    attach_dataset_aware_chest_imagenome_labels_auc,
    attach_dataset_aware_chest_imagenome_bbox_meanf1,
    attach_dataset_aware_chest_imagenome_labels_prcauc,
    attach_dataset_aware_chexpert_labels_auc,
    attach_dataset_aware_chexpert_labels_prcauc,
    attach_dataset_aware_ciderd,
    attach_dataset_aware_weighted_medical_completeness,
    attach_dataset_aware_gender_accuracy,
    attach_dataset_aware_loss,
    attach_loss,
)
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    load_metadata,
    save_metadata,
)
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.handlers import (
    get_log_metrics_handler,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
    get_lr_sch_handler,
    get_checkpoint_handler,
)
from medvqa.utils.files import (
    get_checkpoint_folder_path,
)
from medvqa.training.image2report import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    get_image2report_collate_batch_fn,
    multi_cyclic_dataloaders_generator,
)
from medvqa.metrics.utils import (
    get_merge_metrics_fn,
    get_hybrid_score_name,
)
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter, print_blue, print_red

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # --- Required arguments

    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs the model will be trained')
    parser.add_argument('--batches-per-epoch', type=int, required=True, help='Number of batches per epoch')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')

    # --- Optional arguments

    parser.add_argument('--checkpoint-folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')

    # Model arguments
    parser.add_argument('--pretrained-checkpoint-folder-path', type=str, default=None)
    # Report decoder
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--transf-dec-hidden-dim', type=int, default=256)
    parser.add_argument('--transf-dec-nhead', type=int, default=2)
    parser.add_argument('--transf-dec-dim-forward', type=int, default=256)
    parser.add_argument('--transf-dec-num-layers', type=int, default=2)
    parser.add_argument('--dropout-prob', type=float, default=0)
    # Image encoder
    parser.add_argument('--raw-image-encoding', type=str, default=RawImageEncoding.DENSENET_121)
    parser.add_argument('--image-local-feat-size', type=int, default=1024)
    parser.add_argument('--image-encoder-pretrained-weights-path', type=str, default=None)
    parser.add_argument('--freeze-image-encoder', action='store_true', default=False)
    parser.add_argument('--image-encoder-only-compute-features', action='store_true', default=False)
    parser.add_argument('--chexpert-mlc-version', type=str, default=None)
    parser.add_argument('--chexpert-mlc-hidden-size', type=int, default=128)
    parser.add_argument('--chest-imagenome-mlc-version', type=str, default=None)
    parser.add_argument('--chest-imagenome-mlc-hidden-size', type=int, default=128)
    parser.add_argument('--chest-imagenome-bbox-regressor-version', type=str, default=None)
    parser.add_argument('--chest-imagenome-bbox-hidden-size', type=int, default=128)
    parser.add_argument('--num-regions', type=int, default=None)
    parser.add_argument('--yolov8-model-name-or-path', type=str, default=None)
    parser.add_argument('--yolov8-model-alias', type=str, default=None)

    # Tokenizer arguments
    parser.add_argument('--vocab-min-freq', type=int, default=10)
    parser.add_argument('--use-medical-tokenization', action='store_true', default=False)
    parser.add_argument('--medical-terms-frequency-filename', type=str, default=None)
    
    # Optimization arguments
    parser.add_argument('--optimizer-name', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='reduce-lr-on-plateau')
    parser.add_argument('--lr-decay', type=float, default=0.76, help='Learning rate decay')
    parser.add_argument('--lr-decay-patience', type=int, default=2, help='Learning rate decay patience')
    parser.add_argument('--warmup-and-decay-args', type=str, default=None)
    parser.add_argument('--warmup-and-cosine-args', type=str, default=None)
    parser.add_argument('--warmup-decay-and-cyclic-decay-args', type=str, default=None)
    parser.add_argument('--iters-to-accumulate', type=int, default=1, help='For gradient accumulation')
    parser.add_argument('--override-lr', action='store_true', default=False)
    parser.add_argument('--binary-loss-name', type=str, default='bce')
    parser.add_argument('--focal-loss-weight', type=float, default=1)
    parser.add_argument('--bce-loss-weight', type=float, default=1)
    parser.add_argument('--wbce-loss-weight', type=float, default=1)

    # Data loading arguments
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for parallel dataloading')    
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    parser.add_argument('--use-amp', action='store_true', default=False)
    parser.add_argument('--img-aug-mode', type=str, default=None, help='Mode of data augmentation used for images')
    parser.add_argument('--image-size', nargs='+', type=int, default=(256,256))
    parser.add_argument('--horizontal-flip-prob', type=float, default=0)

    # MIMIC-CXR arguments
    parser.add_argument('--use-mimiccxr', dest='train_mimiccxr', action='store_true', default=False)
    parser.add_argument('--mimiccxr-weight', type=float, default=1)
    parser.add_argument('--mimiccxr-view-mode', type=str, default='any_single')    
    parser.add_argument('--mimiccxr-balanced-sampling-mode', type=str, default=None)
    parser.add_argument('--mimiccxr-balanced-batch-size', type=int, default=None)
    parser.add_argument('--mimiccxr-qa-adapted-reports-filename', type=str, default=None)
    
    # Chest ImaGenome arguments (NOTE: Chest ImaGenome is built on top of MIMIC-CXR)
    parser.add_argument('--chest-imagenome-labels-filename', type=str, default=None)
    parser.add_argument('--chest-imagenome-label-names-filename', type=str, default=None)
    parser.add_argument('--use-chest-imagenome-decent-images-only', action='store_true', default=False)
    parser.add_argument('--clamp-bboxes-chest-imagenome', action='store_true', default=False)

    # Checkpoint saving arguments
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    
    # Auxiliary tasks arguments
    parser.add_argument('--classify-gender', action='store_true', default=False)
    # chexpert labels
    parser.add_argument('--classify-chexpert', action='store_true', default=False)
    parser.add_argument('--mimiccxr-chexpert-labels-filename', type=str, default=None)
    # chest imagenome labels
    parser.add_argument('--classify-chest-imagenome', action='store_true', default=False)
    parser.add_argument('--predict-bboxes-chest-imagenome', action='store_true', default=False)
    parser.add_argument('--chest-imagenome-bbox-loss-weight', type=float, default=1.0)
    parser.add_argument('--predict-labels-and-bboxes-chest-imagenome', action='store_true', default=False)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = {
    MetricNames.CIDER_D: 0.1,
    MetricNames.WMEDCOMP: 1,
    MetricNames.CHXLABEL_AUC: 1,
    MetricNames.CHXLABEL_PRCAUC: 1,
    MetricNames.CHESTIMAGENOMELABELAUC: 1,
    MetricNames.CHESTIMAGENOMELABELPRCAUC: 1,
    MetricNames.GENDER_ACC: 1,
    MetricNames.CHESTIMAGENOMEBBOXIOU: 1,
}

def _metric_getter(metrics_dict, key):
    if key == MetricNames.CHESTIMAGENOMELABELAUC or\
            key == MetricNames.CHESTIMAGENOMELABELPRCAUC or\
            key == MetricNames.CHXLABEL_AUC or\
            key == MetricNames.CHXLABEL_PRCAUC:
        scores = metrics_dict[key]
        return 0.5 * (scores['macro_avg'] + scores['micro_avg'])
    return metrics_dict[key]

def train_model(
    model_kwargs,
    tokenizer_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_trainer_kwargs,
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
    batch_size = dataloading_kwargs['batch_size']
    train_mimiccxr = training_kwargs['train_mimiccxr']
    use_yolov8 = mimiccxr_trainer_kwargs.get('use_yolov8', False)

    # auxiliary task: gender classification
    classify_gender = auxiliary_tasks_kwargs['classify_gender']
    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs['classify_chest_imagenome']
    predict_bboxes_chest_imagenome = auxiliary_tasks_kwargs['predict_bboxes_chest_imagenome']

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    tokenizer = Tokenizer(**tokenizer_kwargs)
    tokenizer_kwargs['vocab_filepath'] = tokenizer.vocab_filepath # Remember vocab filepath in case we need to reload tokenizer

    # Create model
    count_print('Creating instance of Image2ReportModel ...')
    model = Image2ReportModel(vocab_size=tokenizer.vocab_size,
                              start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                              device=device, **model_kwargs)
    model = model.to(device)
    print(model.get_name())

    # Check dataset weights
    if dataloading_kwargs['mimiccxr_weight'] == 0:
        train_mimiccxr = False

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
    trainer_engine = get_engine(model=model, tokenizer=tokenizer, optimizer=optimizer, device=device,
                                update_lr_batchwise=update_lr_batchwise, lr_scheduler=lr_scheduler,
                                model_for_yolov8=model_for_yolov8, **trainer_engine_kwargs)
    validator_engine = get_engine(model=model, tokenizer=tokenizer, device=device, **validator_engine_kwargs)
    
    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    if train_mimiccxr:
        mimiccxr_collate_batch_fn = get_image2report_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR])

    # Create MIMIC-CXR trainer
    if train_mimiccxr:
        count_print('Creating MIMIC-CXR_Image2ReportTrainer ...')
        mimiccxr_trainer = MIMICCXR_Image2ReportTrainer(
            tokenizer=tokenizer,
            train_image_transform = get_image_transform(**train_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            val_image_transform = get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            batch_size=batch_size,
            collate_batch_fn=mimiccxr_collate_batch_fn,            
            num_workers=num_workers,
            **mimiccxr_trainer_kwargs,
        )

    if debug: # if debugging
        output = {}
        if train_mimiccxr: output['mimiccxr_trainer'] = mimiccxr_trainer
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
    _chexpert_labels_datasets = _mim_datasets
    _gender_datasets = _mim_datasets

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []

    attach_loss('loss', trainer_engine, device)
    # for logging
    metrics_to_print.append('loss')    
    
    if train_mimiccxr:
        attach_dataset_aware_weighted_medical_completeness(trainer_engine, tokenizer, _mim_datasets, field='reports')
        attach_dataset_aware_weighted_medical_completeness(validator_engine, tokenizer, _mim_datasets, field='reports')
        attach_dataset_aware_ciderd(validator_engine, _mim_datasets, field='reports')
        attach_dataset_aware_loss(trainer_engine, 'report_loss', _mim_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.WMEDCOMP)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CIDER_D, train=False)
        metrics_to_print.append('report_loss')

    if classify_gender:
        attach_dataset_aware_gender_accuracy(trainer_engine, _gender_datasets, ignore_index=2)
        attach_dataset_aware_loss(trainer_engine, MetricNames.GENDER_LOSS, _gender_datasets)
        in_val = train_mimiccxr
        if in_val: attach_dataset_aware_gender_accuracy(validator_engine, [MIMICCXR_DATASET_ID])
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.GENDER_ACC, val=in_val)
        metrics_to_print.append(MetricNames.GENDER_LOSS)
    
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

    if predict_bboxes_chest_imagenome and not use_yolov8:
        attach_dataset_aware_chest_imagenome_bbox_iou(trainer_engine, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_iou(validator_engine, _mim_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.CHEST_IMAGENOME_BBOX_LOSS, _mim_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMEBBOXIOU)
        metrics_to_print.append(MetricNames.CHEST_IMAGENOME_BBOX_LOSS)
    
    if use_yolov8 and not model_kwargs.get('only_compute_features', False):
        assert predict_bboxes_chest_imagenome
        attach_dataset_aware_loss(trainer_engine, MetricNames.YOLOV8_LOSS, _mim_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.YOLOV8_BOX_LOSS, _mim_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.YOLOV8_CLS_LOSS, _mim_datasets)
        attach_dataset_aware_loss(trainer_engine, MetricNames.YOLOV8_DFL_LOSS, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_iou(validator_engine, _mim_datasets, use_yolov8=True)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMEBBOXIOU, train=False)
        metrics_to_print.append(MetricNames.YOLOV8_LOSS)
        metrics_to_print.append(MetricNames.YOLOV8_BOX_LOSS)
        metrics_to_print.append(MetricNames.YOLOV8_CLS_LOSS)
        metrics_to_print.append(MetricNames.YOLOV8_DFL_LOSS)

    # Timer
    timer = Timer()
    timer.attach(trainer_engine, start=Events.EPOCH_STARTED)
    timer.attach(validator_engine, start=Events.EPOCH_STARTED)

    # Score function
    assert len(val_metrics_to_merge) > 0
    if len(train_metrics_to_merge) > 0:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0.1, 0.9, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(trainer_engine.state.metrics, validator_engine.state.metrics)
    else:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0, 1, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(validator_engine.state.metrics)

    # Learning rate scheduler
    if not update_lr_batchwise:
        count_print('Defining learning rate scheduler handler ...')
        lr_sch_handler = get_lr_sch_handler(lr_scheduler, lr_scheduler_kwargs['name'], score_fn=score_fn)    

    # Checkpoint saving
    model_wrapper = ModelWrapper(model, optimizer, lr_scheduler)
    pretrained_checkpoint_folder_path = model_kwargs.get('pretrained_checkpoint_folder_path', None)    
    if checkpoint_folder_path is None: # first time
        if save: # only if we want to save checkpoints to disk
            count_print('Defining checkpoint folder path ...')
            checkpoint_folder_path = get_checkpoint_folder_path('report_gen', merged_dataset_name, model.get_name(),
                f'dws={",".join(map(str, _train_weights))}' if len(_train_weights) > 1 else None,
            )
            print_red('checkpoint_folder_path =', checkpoint_folder_path)
            save_metadata(checkpoint_folder_path,
                        model_kwargs=model_kwargs,
                        tokenizer_kwargs=tokenizer_kwargs,
                        optimizer_kwargs=optimizer_kwargs,
                        lr_scheduler_kwargs=lr_scheduler_kwargs,
                        mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                        dataloading_kwargs=dataloading_kwargs,
                        collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                        train_image_transform_kwargs=train_image_transform_kwargs,
                        val_image_transform_kwargs=val_image_transform_kwargs,
                        training_kwargs=training_kwargs,
                        trainer_engine_kwargs=trainer_engine_kwargs,
                        validator_engine_kwargs=validator_engine_kwargs,
                        auxiliary_tasks_kwargs=auxiliary_tasks_kwargs)
        if pretrained_checkpoint_folder_path is not None:
            count_print(f'Loading pretrained weights ...')
            pretrained_checkpoint_path = get_checkpoint_filepath(pretrained_checkpoint_folder_path)
            print(f'pretrained_checkpoint_path = {pretrained_checkpoint_path}')
            checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
            load_model_state_dict(model_wrapper.model, checkpoint['model'])
            print('Checkpoint successfully loaded!')    
    else: # resuming
        checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
        count_print('Loading model from checkpoint ...')
        print('checkpoint_path =', checkpoint_path)
        model_wrapper.load_checkpoint(checkpoint_path, device, model_only=override_lr)
    
    if save: # only if we want to save checkpoints to disk
        checkpoint_handler = get_checkpoint_handler(model_wrapper, checkpoint_folder_path, trainer_engine,
                                                    epoch_offset=model_wrapper.get_epoch(),
                                                    score_name=get_hybrid_score_name(train_metrics_to_merge, val_metrics_to_merge),
                                                    score_fn=score_fn)

    # Logging
    count_print('Defining log_metrics_handler ...')

    log_metrics_handler = get_log_metrics_handler(timer,
                                                   metrics_to_print=metrics_to_print,
                                                   log_to_disk=save,
                                                   checkpoint_folder=checkpoint_folder_path)
    log_iteration_handler = get_log_iteration_handler()    
    
    # Attach handlers
    trainer_engine.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    trainer_engine.add_event_handler(Events.EPOCH_STARTED, lambda : print(f'(1) Training stage (lr = {optimizer.param_groups[0]["lr"]:.6f}) ...'))
    trainer_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda : validator_engine.run(val_dataloader,
                                     max_epochs=1, epoch_length=val_dataloader_size))
    validator_engine.add_event_handler(Events.EPOCH_STARTED, lambda : print('(2) Validation stage ...'))
    validator_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    validator_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    if not update_lr_batchwise:
        validator_engine.add_event_handler(Events.EPOCH_COMPLETED, lr_sch_handler)
    if save: # only if we want to save checkpoints to disk
        validator_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # Start training
    count_print('Running trainer engine ...')
    trainer_engine.run(train_dataloader, max_epochs = epochs, epoch_length = batches_per_epoch)


def train_from_scratch(
    # Model args
    embedding_dim,
    transf_dec_nhead,
    transf_dec_dim_forward,
    transf_dec_num_layers,
    transf_dec_hidden_dim,
    dropout_prob,
    freeze_image_encoder,
    image_encoder_only_compute_features,
    raw_image_encoding,
    num_regions,   
    image_local_feat_size,
    image_encoder_pretrained_weights_path,
    pretrained_checkpoint_folder_path,
    chexpert_mlc_version,
    chexpert_mlc_hidden_size,
    chest_imagenome_mlc_version,
    chest_imagenome_mlc_hidden_size,
    chest_imagenome_bbox_regressor_version,
    chest_imagenome_bbox_hidden_size,
    yolov8_model_name_or_path,
    yolov8_model_alias,
    # Tokenizer args
    vocab_min_freq,
    use_medical_tokenization,
    medical_terms_frequency_filename,
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
    mimiccxr_qa_adapted_reports_filename,
    chest_imagenome_labels_filename,
    chest_imagenome_label_names_filename,
    use_chest_imagenome_decent_images_only,
    # Dataloading args
    batch_size,
    num_workers,
    mimiccxr_weight,
    img_aug_mode,
    horizontal_flip_prob,
    mimiccxr_balanced_sampling_mode,
    mimiccxr_balanced_batch_size,
    # Fixed traning args
    train_mimiccxr,
    binary_loss_name,
    focal_loss_weight,
    bce_loss_weight,
    wbce_loss_weight,
    use_amp,
    iters_to_accumulate,
    # Variable traning args
    epochs,
    batches_per_epoch,
    # Auxiliary tasks args
    classify_gender,
    classify_chexpert,
    mimiccxr_chexpert_labels_filename,
    classify_chest_imagenome,
    predict_bboxes_chest_imagenome,
    predict_labels_and_bboxes_chest_imagenome,
    clamp_bboxes_chest_imagenome,
    chest_imagenome_bbox_loss_weight,
    # GPU
    device,
    # Other args
    save,
    debug = False,
):
    print_blue('----- Training model from scratch ------', bold=True)

    assert train_mimiccxr, 'No dataset selected for training'

    use_yolov8 = raw_image_encoding == RawImageEncoding.YOLOV8
    use_bbox_aware_transform = predict_bboxes_chest_imagenome or use_yolov8

    if classify_chest_imagenome:
        assert chest_imagenome_label_names_filename is not None
        n_chest_imagenome_labels = len(load_chest_imagenome_label_names(chest_imagenome_label_names_filename))
    else:
        n_chest_imagenome_labels = None

    tokenizer_kwargs = dict(
        vocab_min_freq=vocab_min_freq,
        use_medical_tokenization=use_medical_tokenization,
        medical_terms_frequency_filename=medical_terms_frequency_filename,
    )
    if train_mimiccxr:
        assert mimiccxr_qa_adapted_reports_filename is not None
        mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, mimiccxr_qa_adapted_reports_filename)
        tokenizer_kwargs['qa_adapted_dataset_paths'] = [mimiccxr_qa_adapted_reports_path]
        
    model_kwargs = dict(
        pretrained_checkpoint_folder_path=pretrained_checkpoint_folder_path,
        # Image encoder
        raw_image_encoding=raw_image_encoding,
        image_local_feat_size=image_local_feat_size,
        freeze_image_encoder=freeze_image_encoder,
        only_compute_features=image_encoder_only_compute_features,
        image_encoder_pretrained_weights_path=image_encoder_pretrained_weights_path,
        num_regions=num_regions,
        yolov8_model_name_or_path=yolov8_model_name_or_path,
        yolov8_model_alias=yolov8_model_alias,
        # Report Decoder args
        embedding_dim=embedding_dim,
        transf_dec_nhead=transf_dec_nhead,
        transf_dec_dim_forward=transf_dec_dim_forward,
        transf_dec_num_layers=transf_dec_num_layers,
        transf_dec_hidden_dim=transf_dec_hidden_dim,
        dropout_prob=dropout_prob,
        # Aux tasks
        classify_gender=classify_gender,
        classify_chexpert=classify_chexpert,
        classify_chest_imagenome=classify_chest_imagenome,
        chexpert_mlc_version=chexpert_mlc_version,
        chexpert_mlc_hidden_size=chexpert_mlc_hidden_size,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        predict_labels_and_bboxes_chest_imagenome=predict_labels_and_bboxes_chest_imagenome,
        n_chest_imagenome_labels=n_chest_imagenome_labels,
        chest_imagenome_mlc_version=chest_imagenome_mlc_version,
        chest_imagenome_mlc_hidden_size=chest_imagenome_mlc_hidden_size,
        chest_imagenome_bbox_regressor_version=chest_imagenome_bbox_regressor_version,
        chest_imagenome_bbox_hidden_size=chest_imagenome_bbox_hidden_size,
    )
    if predict_bboxes_chest_imagenome:
        avg_coords = get_chest_imagenome_train_average_bbox_coords(
            clamp_bbox_coords=clamp_bboxes_chest_imagenome,
            use_decent_images_only=use_chest_imagenome_decent_images_only,
        )
        print('avg_coords.shape=', avg_coords.shape)
        avg_coords = avg_coords.tolist()
        model_kwargs['chest_imagenome_train_average_bbox_coords'] = avg_coords
    else:
        model_kwargs['chest_imagenome_train_average_bbox_coords'] = None
    if predict_labels_and_bboxes_chest_imagenome or (classify_chest_imagenome and\
                                                     chest_imagenome_mlc_version in (MLCVersion.V1, MLCVersion.V2)):
        tmp = get_labels_per_anatomy_and_anatomy_group(chest_imagenome_label_names_filename, for_training=True)
        model_kwargs['chest_imagenome_anatomy_to_labels'] = tmp['anatomy_to_localized_labels']
        model_kwargs['chest_imagenome_anatomy_group_to_labels'] = tmp['anatomy_group_to_global_labels']
        model_kwargs['n_chest_imagenome_bboxes'] = len(tmp['anatomy_names'])
    else:
        model_kwargs['chest_imagenome_anatomy_to_labels'] = None
        model_kwargs['chest_imagenome_anatomy_group_to_labels'] = None
        model_kwargs['n_chest_imagenome_bboxes'] = None
    
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
        batch_size=batch_size,
        mimiccxr_weight=mimiccxr_weight,
    )

    # Image transforms
    train_image_transform_kwargs = {}
    val_image_transform_kwargs = {}
    if train_mimiccxr:
        train_image_transform_kwargs[DATASET_NAMES.MIMICCXR] = dict(
            image_size=image_size,
            augmentation_mode=img_aug_mode,
            use_bbox_aware_transform=use_bbox_aware_transform,
            horizontal_flip_prob=horizontal_flip_prob,
            for_yolov8=use_yolov8,
        )
        val_image_transform_kwargs[DATASET_NAMES.MIMICCXR] = train_image_transform_kwargs[DATASET_NAMES.MIMICCXR].copy()
        val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]['augmentation_mode'] = None # no augmentation for validation
    
    _kwargs = dict(
        classify_gender=classify_gender,
        classify_chexpert=classify_chexpert,
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        use_yolov8=use_yolov8,
    )
    collate_batch_fn_kwargs = {}
    if train_mimiccxr:
        collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR] = { 'dataset_id': MIMICCXR_DATASET_ID, **_kwargs }
    
    if train_mimiccxr:
        x = image_size if type(image_size) is int else image_size[0]
        if x > 256:
            source_image_size_mode = MIMICCXR_ImageSizeModes.MEDIUM_512
        else:
            source_image_size_mode = MIMICCXR_ImageSizeModes.SMALL_256x256
        print(f'source_image_size_mode: {source_image_size_mode}')
        mimiccxr_trainer_kwargs = dict(
            qa_adapted_reports_filename=mimiccxr_qa_adapted_reports_filename,
            chest_imagenome_labels_filename=chest_imagenome_labels_filename,
            chest_imagenome_label_names_filename=chest_imagenome_label_names_filename,
            view_mode=mimiccxr_view_mode,
            source_image_size_mode=source_image_size_mode,
            classify_chexpert=classify_chexpert,
            chexpert_labels_filename=mimiccxr_chexpert_labels_filename,
            classify_gender=classify_gender,
            classify_chest_imagenome=classify_chest_imagenome,
            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
            clamp_bboxes_chest_imagenome=clamp_bboxes_chest_imagenome,
            use_decent_images_only=use_chest_imagenome_decent_images_only,
            data_augmentation_enabled=img_aug_mode is not None,
            balanced_sampling_mode=mimiccxr_balanced_sampling_mode,
            balanced_batch_size=mimiccxr_balanced_batch_size,
            use_yolov8=use_yolov8,
        )
    else:
        mimiccxr_trainer_kwargs = None

    trainer_engine_kwargs = dict(
        classify_gender=classify_gender,
        classify_chexpert=classify_chexpert,
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        binary_loss_name=binary_loss_name,
        focal_loss_weight=focal_loss_weight,
        bce_loss_weight=bce_loss_weight,
        wbce_loss_weight=wbce_loss_weight,
        use_amp=use_amp,
        training=True,
        iters_to_accumulate=iters_to_accumulate,
        chest_imagenome_bbox_loss_weight=chest_imagenome_bbox_loss_weight,
        using_yolov8=use_yolov8,
        include_report=True,
        shift_tokens_for_transformer=True,
    )

    validator_engine_kwargs = dict(
        classify_gender=classify_gender,
        classify_chexpert=classify_chexpert,
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        training=False,
        using_yolov8=use_yolov8,
        include_report=True,
        shift_tokens_for_transformer=True,
    )
    
    training_kwargs = dict(
        use_amp=use_amp,
        train_mimiccxr=train_mimiccxr,
        binary_loss_name=binary_loss_name,
    )

    auxiliary_tasks_kwargs = dict(
        # gender
        classify_gender=classify_gender,
        # chexpert labels
        classify_chexpert=classify_chexpert,
        mimiccxr_chexpert_labels_filename=mimiccxr_chexpert_labels_filename,
        # chest imagenome labels
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
    )

    return train_model(
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
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
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
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
                tokenizer_kwargs=tokenizer_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
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