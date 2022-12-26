import  os
import argparse

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.iuxray.iuxray_multimodal_dataset_management import IUXRAY_Multimodal_Trainer
from medvqa.datasets.mimiccxr.mimiccxr_multimodal_dataset_management import MIMICCXR_Multimodal_Trainer
from medvqa.datasets.chexpert.chexpert_dataset_management import Chexpert_VisualModuleTrainer
from medvqa.datasets.cxr14.cxr14_dataset_management import CXR14_VisualModuleTrainer
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBig_VisualModuleTrainer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.common import load_model_state_dict
from medvqa.training.utils import append_metric_name

from medvqa.utils.constants import (
    CHEXPERT_DATASET_ID,
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID,
    CXR14_DATASET_ID,
    VINBIG_DATASET_ID,
    MetricNames,
)
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_dataset_aware_chexpert_labels_accuracy,
    attach_dataset_aware_chexpert_labels_macroavgf1,
    attach_dataset_aware_chexpert_labels_microavgf1,
    attach_dataset_aware_chexpert_labels_roc_auc,
    attach_dataset_aware_cxr14_labels_macroavgf1,
    attach_dataset_aware_cxr14_labels_microavgf1,
    attach_dataset_aware_question_labels_macroavgf1,
    attach_dataset_aware_question_labels_microavgf1,
    attach_dataset_aware_vinbig_labels_macroavgf1,
    attach_dataset_aware_vinbig_labels_microavgf1,
    attach_dataset_aware_orientation_accuracy,
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
    get_log_metrics_handlers,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
    get_lr_sch_handler,
    get_checkpoint_handler,
)
from medvqa.models.vision import DensenetVisualModule
from medvqa.utils.files import get_checkpoint_folder_path
from medvqa.training.multimodal import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    get_multimodal_collate_batch_fn,
    multi_cyclic_dataloaders_generator,
    get_vision_collate_batch_fn,
)
from medvqa.metrics.utils import (
    get_merge_metrics_fn,
    get_hybrid_score_name,
)
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs the model will be trained')
    parser.add_argument('--batches-per-epoch', type=int, required=True,
                        help='Number of batches per epoch')

    # optional arguments
    parser.add_argument('--checkpoint-folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')
    parser.add_argument('--iuxray-preprocessed-data-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-preprocessed-data-filename', type=str, default=None)
    
    parser.add_argument('--image-local-feat-size', type=int, default=1024,
                        help='Size of local feature vectors from the CNN. They must match the actual vectors output by the CNN')
    
    parser.add_argument('--imagenet-pretrained', dest='imagenet_pretrained', action='store_true')
    parser.set_defaults(imagenet_pretrained=False)
    
    parser.add_argument('--freeze-cnn', dest='freeze_cnn', action='store_true')
    parser.set_defaults(freeze_cnn=False)

    parser.add_argument('--densenet-pretrained-weights-path', type=str, default=None,
                        help='Path to densenet 121 pretrained weights')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.76,
                        help='Learning rate decay')
    parser.add_argument('--lr-decay-patience', type=int, default=2,
                        help='Learning rate decay patience')
    parser.add_argument('--batch-size', type=int, default=45,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for parallel dataloading')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')    
    parser.add_argument('--img-aug-mode', type=str, default=None,
                        help='Mode of data augmentation used for images')
    
    parser.add_argument('--mimiccxr-weight', type=float, default=1.0,
                        help='Relative number of batches to sample from MIMIC-CXR dataset (for rebalancing purposes)')
    parser.add_argument('--chexpert-weight', type=float, default=0.8,
                        help='Relative number of batches to sample from CheXpert dataset (for rebalancing purposes)')
    parser.add_argument('--iuxray-weight', type=float, default=0.08,
                        help='Relative number of batches to sample from IU X-ray dataset (for rebalancing purposes)')

    parser.add_argument('--question-balanced', dest='question_balanced', action='store_true')
    parser.set_defaults(question_balanced=False)
    
    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)

    parser.add_argument('--pretrained-checkpoint-folder-path', type=str, default=None)

    parser.add_argument('--one-question-per-batch', dest='one_question_per_batch', action='store_true')
    parser.set_defaults(one_question_per_batch=False)

    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=True)

    parser.add_argument('--override-lr', dest='override_lr', action='store_true')
    parser.set_defaults(override_lr=False)

    parser.add_argument('--no-mimiccxr', dest='train_mimiccxr', action='store_false')
    parser.set_defaults(train_mimiccxr=True)
    parser.add_argument('--no-iuxray', dest='train_iuxray', action='store_false')
    parser.set_defaults(train_iuxray=True)
    parser.add_argument('--no-chexpert', dest='train_chexpert', action='store_false')
    parser.set_defaults(train_chexpert=True)

    parser.add_argument('--binary-loss-name', type=str, default='bce')

    # Auxiliary tasks arguments
    
    # medical tags
    parser.add_argument('--classify-tags', dest='classify_tags', action='store_true')
    parser.set_defaults(classify_tags=False)
    parser.add_argument('--n-medical-tags', type=int, default=None,
                        help='Number of medical tags (for tag prediction auxiliary task)')
    parser.add_argument('--iuxray-rid2tags-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-rid2tags-filename', type=str, default=None)
    # orientation
    parser.add_argument('--classify-orientation', dest='classify_orientation', action='store_true')
    parser.set_defaults(classify_orientation=False)
    # chexpert labels
    parser.add_argument('--classify-chexpert', dest='classify_chexpert', action='store_true')
    parser.set_defaults(classify_chexpert=False)
    parser.add_argument('--iuxray-chexpert-labels-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-chexpert-labels-filename', type=str, default=None)
    # question classification
    parser.add_argument('--classify-questions', dest='classify_questions', action='store_true')
    parser.set_defaults(classify_questions=False)
    parser.add_argument('--n-questions', type=int, default=None)
    parser.add_argument('--iuxray-question-labels-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-question-labels-filename', type=str, default=None)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = {
    MetricNames.MEDTAGF1: 1,
    MetricNames.ORIENACC: 1,
    MetricNames.CHXLABELMICROAVGF1: 0.5,
    MetricNames.CHXLABELMACROAVGF1: 0.5,
    MetricNames.CHXLABEL_ROCAUC: 1,
    MetricNames.QLABELS_MICROAVGF1: 1,
    MetricNames.QLABELS_MACROAVGF1: 1,
    MetricNames.GENDER_ACC: 1,
}

def train_model(
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_trainer_kwargs,
    iuxray_trainer_kwargs,
    chexpert_trainer_kwargs,
    cxr14_trainer_kwargs,
    vinbig_trainer_kwargs,
    dataloading_kwargs,
    image_transform_kwargs,
    training_kwargs,
    trainer_engine_kwargs,
    val_engine_kwargs,
    auxiliary_tasks_kwargs,
    epochs,
    batch_size,
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
    train_iuxray = training_kwargs['train_iuxray']
    train_mimiccxr = training_kwargs['train_mimiccxr']
    train_chexpert = training_kwargs['train_chexpert']
    train_cxr14 = training_kwargs['train_cxr14']
    train_vinbig = training_kwargs['train_vinbig']
    use_amp = training_kwargs['use_amp']
    assert train_iuxray or train_mimiccxr

    # auxiliary task: medical tags prediction
    classify_tags = auxiliary_tasks_kwargs['classify_tags']
    n_medical_tags = auxiliary_tasks_kwargs['n_medical_tags']
    iuxray_rid2tags_filename = auxiliary_tasks_kwargs['iuxray_rid2tags_filename']
    mimiccxr_rid2tags_filename = auxiliary_tasks_kwargs['mimiccxr_rid2tags_filename']
    if classify_tags:
        assert n_medical_tags is not None
        if train_iuxray: assert iuxray_rid2tags_filename is not None
        if train_mimiccxr: assert mimiccxr_rid2tags_filename is not None
    
    # auxiliary task: orientation classification
    classify_orientation = auxiliary_tasks_kwargs['classify_orientation']

    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']

    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)
    n_questions = auxiliary_tasks_kwargs.get('n_questions', None)
    iuxray_question_labels_filename = auxiliary_tasks_kwargs.get('iuxray_question_labels_filename', None)
    mimiccxr_question_labels_filename = auxiliary_tasks_kwargs.get('mimiccxr_question_labels_filename', None)
    if classify_questions:
        assert n_questions is not None
        if train_iuxray: assert iuxray_question_labels_filename is not None
        if train_mimiccxr: assert mimiccxr_question_labels_filename is not None

    if train_chexpert:
        assert classify_chexpert
        assert classify_orientation
        assert classify_questions

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)
    
    # Create model
    count_print('Creating instance of DensenetVisualModule model ...')
    model = DensenetVisualModule(**model_kwargs)
    model = model.to(device)

    # Optimizer
    count_print('Defining Adam optimizer ...')
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    # Learning rate scheduler
    count_print('Defining scheduler ...')
    lr_scheduler, update_lr_batchwise = create_lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)

    # Create trainer and validator engines
    count_print('Creating trainer and validator engines ...')    
    trainer = get_engine(model=model, optimizer=optimizer, device=device,
                         update_lr_batchwise=update_lr_batchwise, lr_scheduler=lr_scheduler, **trainer_engine_kwargs)
    validator = get_engine(model=model, device=device, **val_engine_kwargs)

    # Define image transform
    count_print('Defining image transform ...')
    img_transform = get_image_transform(augmentation_mode=dataloading_kwargs['img_aug_mode'])

    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    _kwargs = dict(use_text=False,
                   classify_orientation = classify_orientation,
                   classify_chexpert = classify_chexpert,
                   classify_questions = classify_questions)
    if train_mimiccxr:
        mimiccxr_collate_batch_fn = get_multimodal_collate_batch_fn(MIMICCXR_DATASET_ID, **_kwargs)
    if train_iuxray:
        iuxray_collate_batch_fn = get_multimodal_collate_batch_fn(IUXRAY_DATASET_ID, **_kwargs)
    if train_chexpert:
        chexpert_collate_batch_fn = get_vision_collate_batch_fn(CHEXPERT_DATASET_ID)
    if train_cxr14:
        cxr14_collate_batch_fn = get_vision_collate_batch_fn(CXR14_DATASET_ID)
    if train_vinbig:
        vinbig_collate_batch_fn = get_vision_collate_batch_fn(VINBIG_DATASET_ID)

    if train_mimiccxr:
        count_print('Creating MIMIC-CXR multimodal trainer ...')
        mimiccxr_trainer = MIMICCXR_Multimodal_Trainer(
            transform = img_transform,
            batch_size = batch_size,
            collate_batch_fn = mimiccxr_collate_batch_fn,            
            num_workers = num_workers,
            **mimiccxr_trainer_kwargs,
        )

    if train_iuxray:
        count_print('Creating IU X-Ray multimodal trainer ...')
        iuxray_trainer = IUXRAY_Multimodal_Trainer(
            transform = img_transform,
            batch_size = batch_size,
            collate_batch_fn = iuxray_collate_batch_fn,
            num_workers = num_workers,
            **iuxray_trainer_kwargs,
        )

    if train_chexpert:
        count_print('Creating CheXpert visual module trainer ...')
        chexpert_trainer = Chexpert_VisualModuleTrainer(
            transform=img_transform,
            batch_size=batch_size,
            collate_batch_fn=chexpert_collate_batch_fn,
            num_workers=num_workers,
            **chexpert_trainer_kwargs,
        )

    if train_cxr14:
        count_print('Creating CXR14 visual module trainer ...')
        cxr14_trainer = CXR14_VisualModuleTrainer(
            transform=img_transform,
            batch_size=batch_size,
            collate_batch_fn=cxr14_collate_batch_fn,
            num_workers=num_workers,
            **cxr14_trainer_kwargs,
        )

    if train_vinbig:
        count_print('Creating VinBig visual module trainer ...')
        vinbig_trainer = VinBig_VisualModuleTrainer(
            transform=img_transform,
            batch_size=batch_size,
            collate_batch_fn=vinbig_collate_batch_fn,
            num_workers=num_workers,
            **vinbig_trainer_kwargs,
        )

    if debug: # if debugging
        output = {}
        if train_mimiccxr: output['mimiccxr_trainer'] = mimiccxr_trainer
        if train_iuxray: output['iuxray_trainer'] = iuxray_trainer
        if train_chexpert: output['chexpert_trainer'] = chexpert_trainer
        if train_cxr14: output['cxr14_trainer'] = cxr14_trainer
        if train_vinbig: output['vinbig_trainer'] = vinbig_trainer
        return output

    # Create complex dataloaders
    count_print('Creating dataloaders ...')

    _train_weights = []
    _train_dataloaders = []
    _val_dataloaders = []
    _dataset_names = []

    if train_mimiccxr:
        _dataset_names.append('mim')
        _train_weights.append(dataloading_kwargs['mimiccxr_weight'] * 0.5)
        _train_dataloaders.append(mimiccxr_trainer.train_dataloader__question_balanced)
        _train_weights.append(dataloading_kwargs['mimiccxr_weight'] * 0.5)
        _train_dataloaders.append(mimiccxr_trainer.train_dataloader__chexpert_balanced)
        _val_dataloaders.append(mimiccxr_trainer.test_dataloader)
    if train_iuxray:
        _dataset_names.append('iu')
        _train_weights.append(dataloading_kwargs['iuxray_weight'] * 0.5)
        _train_dataloaders.append(iuxray_trainer.train_dataloader__question_balanced)
        _train_weights.append(dataloading_kwargs['iuxray_weight'] * 0.5)
        _train_dataloaders.append(iuxray_trainer.train_dataloader__chexpert_balanced)
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
    
    assert len(_train_dataloaders) > 0
    assert len(_val_dataloaders) > 0
    assert len(_train_dataloaders) == len(_train_weights)
    print('len(_train_dataloaders) =', len(_train_dataloaders))
    print('len(_val_dataloaders) =', len(_val_dataloaders))
    print('_train_weights =', _train_weights)

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

    _iu_mim_datasets = [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID]
    _orientation_datasets = _iu_mim_datasets + [CHEXPERT_DATASET_ID, CXR14_DATASET_ID]
    _chexpert_labels_datasets = _iu_mim_datasets + [CHEXPERT_DATASET_ID]
    _gender_datasets = [CHEXPERT_DATASET_ID, CXR14_DATASET_ID]
    
    attach_loss('loss', trainer, device)

    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(trainer, _orientation_datasets)
        attach_dataset_aware_orientation_accuracy(validator, _orientation_datasets)
        attach_dataset_aware_loss(trainer, MetricNames.ORIENTATION_LOSS, _orientation_datasets)

    if classify_questions:
        attach_dataset_aware_question_labels_macroavgf1(trainer, _iu_mim_datasets)
        attach_dataset_aware_question_labels_microavgf1(trainer, _iu_mim_datasets)
        attach_dataset_aware_question_labels_macroavgf1(validator, _iu_mim_datasets)
        attach_dataset_aware_question_labels_microavgf1(validator, _iu_mim_datasets)
        attach_dataset_aware_loss(trainer, MetricNames.QLABELS_LOSS, _iu_mim_datasets)
    
    if classify_chexpert:
        attach_dataset_aware_chexpert_labels_accuracy(trainer, _chexpert_labels_datasets)
        attach_dataset_aware_chexpert_labels_accuracy(validator, _chexpert_labels_datasets)
        attach_dataset_aware_chexpert_labels_macroavgf1(trainer, _chexpert_labels_datasets)
        attach_dataset_aware_chexpert_labels_macroavgf1(validator, _chexpert_labels_datasets)
        attach_dataset_aware_chexpert_labels_microavgf1(trainer, _chexpert_labels_datasets)
        attach_dataset_aware_chexpert_labels_microavgf1(validator, _chexpert_labels_datasets)
        attach_dataset_aware_chexpert_labels_roc_auc(trainer, _chexpert_labels_datasets, 'cpu')
        attach_dataset_aware_chexpert_labels_roc_auc(validator, _chexpert_labels_datasets, 'cpu')
        attach_dataset_aware_loss(trainer, MetricNames.CHEXPERT_LOSS, _chexpert_labels_datasets)

    if train_chexpert or train_cxr14:
        attach_dataset_aware_gender_accuracy(trainer, _gender_datasets)
        attach_dataset_aware_loss(trainer, MetricNames.GENDER_LOSS, _gender_datasets)

    if train_cxr14:
        attach_dataset_aware_cxr14_labels_macroavgf1(trainer, [CXR14_DATASET_ID])
        attach_dataset_aware_cxr14_labels_microavgf1(trainer, [CXR14_DATASET_ID])
        attach_dataset_aware_loss(trainer, MetricNames.CXR14_LOSS, [CXR14_DATASET_ID])

    if train_vinbig:
        attach_dataset_aware_vinbig_labels_macroavgf1(trainer, [VINBIG_DATASET_ID])
        attach_dataset_aware_vinbig_labels_microavgf1(trainer, [VINBIG_DATASET_ID])
        attach_dataset_aware_loss(trainer, MetricNames.VINBIG_LOSS, [VINBIG_DATASET_ID])

    # Timer
    timer = Timer()
    timer.attach(trainer, start=Events.EPOCH_STARTED)
    timer.attach(validator, start=Events.EPOCH_STARTED)
    
    # Learning rate scheduler
    count_print('Defining learning rate scheduler handler ...')

    # train_metrics_to_merge = []
    # val_metrics_to_merge = []
    
    # if classify_tags:
    #     append_metric_name(train_metrics_to_merge, val_metrics_to_merge, MetricNames.MEDTAGF1)
    # if classify_orientation:
    #     append_metric_name(train_metrics_to_merge, val_metrics_to_merge, MetricNames.ORIENACC)
    # if classify_chexpert:
    #     append_metric_name(train_metrics_to_merge, val_metrics_to_merge, MetricNames.CHXLABELMICROAVGF1)
    #     append_metric_name(train_metrics_to_merge, val_metrics_to_merge, MetricNames.CHXLABELMACROAVGF1)
    #     append_metric_name(train_metrics_to_merge, val_metrics_to_merge, MetricNames.CHXLABEL_ROCAUC)
    # if classify_questions:
    #     append_metric_name(train_metrics_to_merge, val_metrics_to_merge, MetricNames.QLABELS_MICROAVGF1)
    #     append_metric_name(train_metrics_to_merge, val_metrics_to_merge, MetricNames.QLABELS_MACROAVGF1)
    # if train_chexpert:
    #     append_metric_name(train_metrics_to_merge, val_metrics_to_merge, MetricNames.GENDER_ACC, val=False)

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = ['loss']

    if classify_orientation:
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.ORIENACC)
        metrics_to_print.append(MetricNames.ORIENTATION_LOSS)
    if classify_chexpert:
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHXLABELMICROAVGF1)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHXLABELMACROAVGF1)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHXLABEL_ROCAUC)
        metrics_to_print.append(MetricNames.CHEXPERT_LOSS)
        metrics_to_print.append(MetricNames.CHXLABELACC)
    if classify_questions:
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.QLABELS_MICROAVGF1)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.QLABELS_MACROAVGF1)
        metrics_to_print.append(MetricNames.QLABELS_LOSS)
    if train_chexpert:
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.GENDER_ACC, val=False)
        metrics_to_print.append(MetricNames.GENDER_LOSS)
    if train_cxr14:
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CXR14MICROAVGF1, val=False)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CXR14MACROAVGF1, val=False)
        metrics_to_print.append(MetricNames.CXR14_LOSS)
    if train_vinbig:
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.VINBIGMICROAVGF1, val=False)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.VINBIGMACROAVGF1, val=False)
        metrics_to_print.append(MetricNames.VINBIG_LOSS)

    def _metric_getter(metrics_dict, key):
        if key == MetricNames.CHXLABEL_ROCAUC:
            return 0.5 * (metrics_dict[key]['micro_avg'] + metrics_dict[key]['macro_avg'])
        return metrics_dict[key]

    merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0.05, 0.95, _metric_getter)

    score_fn = lambda _ : merge_metrics_fn(trainer.state.metrics, validator.state.metrics)

    if not update_lr_batchwise:
        lr_sch_handler = get_lr_sch_handler(lr_scheduler, lr_scheduler_kwargs['name'], score_fn=score_fn)

    # Checkpoint saving    
    model_wrapper = ModelWrapper(model, optimizer, lr_scheduler)

    pretrained_checkpoint_folder_path = model_kwargs.get('pretrained_checkpoint_folder_path', None)
    
    if checkpoint_folder_path is None: # first time
        if save: # only if we want to save checkpoints to disk
            count_print('Defining checkpoint folder path ...')
            checkpoint_folder_path = get_checkpoint_folder_path('visual_module', merged_dataset_name, model.name,
                f'visenc-pretr={int(bool(model_kwargs["image_encoder_pretrained_weights_path"]))}',
                f'dws={",".join(map(str, _train_weights))}' \
                    if len(_train_weights) > 1 else None,
                'orien' if classify_orientation else None,
                'chx' if classify_chexpert else None,
                'ql' if classify_questions else None,
                'amp' if use_amp else None,
            )
            print('checkpoint_folder_path =', checkpoint_folder_path)
            save_metadata(checkpoint_folder_path,
                        model_kwargs = model_kwargs,
                        optimizer_kwargs = optimizer_kwargs,
                        lr_scheduler_kwargs = lr_scheduler_kwargs,
                        mimiccxr_trainer_kwargs = mimiccxr_trainer_kwargs,
                        iuxray_trainer_kwargs = iuxray_trainer_kwargs,
                        chexpert_trainer_kwargs = chexpert_trainer_kwargs,
                        cxr14_trainer_kwargs = cxr14_trainer_kwargs,
                        vinbig_trainer_kwargs = vinbig_trainer_kwargs,
                        dataloading_kwargs = dataloading_kwargs,
                        image_transform_kwargs = image_transform_kwargs,
                        training_kwargs = training_kwargs,
                        trainer_engine_kwargs = trainer_engine_kwargs,
                        val_engine_kwargs = val_engine_kwargs,
                        auxiliary_tasks_kwargs = auxiliary_tasks_kwargs)

        if pretrained_checkpoint_folder_path is not None:
            pretrained_checkpoint_path = get_checkpoint_filepath(pretrained_checkpoint_folder_path)
            count_print(f'Loading pretrained weights ...')
            print(f'pretrained_checkpoint_path = {pretrained_checkpoint_path}')
            checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
            load_model_state_dict(model_wrapper.model, checkpoint['model'])
            print('Checkpoint successfully loaded!')
    
    else: # resuming
        checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
        count_print('Loading model from checkpoint ...')
        print('checkpoint_path =', checkpoint_path)
        model_wrapper.load_checkpoint(checkpoint_path, device, model_only=override_lr)
    
    score_fn = lambda _ : merge_metrics_fn(trainer.state.metrics, validator.state.metrics)

    if save: # only if we want to save checkpoints to disk
        checkpoint_handler = get_checkpoint_handler(model_wrapper, checkpoint_folder_path, trainer,
                                                    epoch_offset=model_wrapper.get_epoch(),
                                                    score_name=get_hybrid_score_name(train_metrics_to_merge, val_metrics_to_merge),
                                                    score_fn=score_fn)

    # Logging
    count_print('Defining log_metrics_handler ...')

    log_metrics_handler = get_log_metrics_handlers(timer,
                                                   metrics_to_print=metrics_to_print,
                                                   log_to_disk=save,
                                                   checkpoint_folder=checkpoint_folder_path)
    log_iteration_handler = get_log_iteration_handler()    
    
    
    # Attach handlers    
    trainer.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda : print(f'(1) Training stage (lr = {lr_scheduler.optimizer.param_groups[0]["lr"]:.6f}) ...'))
    trainer.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda : validator.run(val_dataloader,
                                     max_epochs=1, epoch_length=val_dataloader_size))
    validator.add_event_handler(Events.EPOCH_STARTED, lambda : print('(2) Validation stage ...'))
    validator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    validator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    if not update_lr_batchwise:
        validator.add_event_handler(Events.EPOCH_COMPLETED, lr_sch_handler)
    if save: # only if we want to save checkpoints to disk
        validator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # Start training
    count_print('Running trainer engine ...')
    trainer.run(train_dataloader,
                max_epochs = epochs,
                epoch_length = batches_per_epoch)


def train_from_scratch(
    # Model's args
    freeze_cnn,
    image_local_feat_size,
    densenet_pretrained_weights_path,
    pretrained_checkpoint_folder_path,
    imagenet_pretrained,
    # Optimizer's args
    lr,
    # lr_scheduler's args
    lr_decay,
    lr_decay_patience,
    # Dataset args
    iuxray_qa_adapted_reports_filename,
    mimiccxr_qa_adapted_reports_filename,
    # Dataloading args
    batch_size,
    num_workers,
    mimiccxr_weight,
    iuxray_weight,
    chexpert_weight,
    img_aug_mode,
    # Fixed traning args
    train_mimiccxr,
    train_iuxray,
    train_chexpert,
    train_vinbig,
    train_cxr14,
    vinbig_training_data,
    binary_loss_name,
    use_amp,
    iters_to_accumulate,
    # Variable traning args
    epochs,
    batches_per_epoch,
    one_question_per_batch,
    # Auxiliary tasks args
    classify_tags,
    n_medical_tags,
    iuxray_rid2tags_filename,
    mimiccxr_rid2tags_filename,
    classify_orientation,
    classify_chexpert,
    iuxray_chexpert_labels_filename,
    mimiccxr_chexpert_labels_filename,
    classify_questions,
    n_questions,
    iuxray_question_labels_filename,
    mimiccxr_question_labels_filename,
    # GPU
    device,
    # Other args
    save,
    debug = False,
    **unused_kwargs,
):
    print('----- Training model from scratch ------')

    assert train_mimiccxr or train_iuxray or train_chexpert or train_cxr14 or train_vinbig

    model_kwargs = dict(
        image_local_feat_size = image_local_feat_size,
        pretrained = imagenet_pretrained or densenet_pretrained_weights_path,
        densenet_pretrained_weights_path = densenet_pretrained_weights_path,
        pretrained_checkpoint_folder_path = os.path.join(WORKSPACE_DIR, pretrained_checkpoint_folder_path) \
            if pretrained_checkpoint_folder_path is not None else None,
        classify_tags = classify_tags,
        classify_orientation = classify_orientation,
        classify_chexpert = classify_chexpert,
        classify_questions = classify_questions,
        n_medical_tags = n_medical_tags,
        n_questions = n_questions,
        use_chexpert_forward = train_chexpert,
        freeze_cnn = freeze_cnn,
    )
    optimizer_kwargs = dict(
        lr = lr,
    )
    lr_scheduler_kwargs = dict(
        factor = lr_decay,
        patience = lr_decay_patience,
    )

    mimiccxr_trainer_kwargs = dict(        
        classify_orientation = classify_orientation,
        classify_chexpert = classify_chexpert,
        chexpert_labels_filename = mimiccxr_chexpert_labels_filename,
        classify_questions = classify_questions,
        question_labels_filename = mimiccxr_question_labels_filename,
        qa_adapted_reports_filename = mimiccxr_qa_adapted_reports_filename,
    )
    iuxray_trainer_kwargs = dict(
        classify_orientation = classify_orientation,
        classify_chexpert = classify_chexpert,
        chexpert_labels_filename = iuxray_chexpert_labels_filename,
        classify_questions = classify_questions,
        question_labels_filename = iuxray_question_labels_filename,
        qa_adapted_reports_filename = iuxray_qa_adapted_reports_filename,
    )
    chexpert_trainer_kwargs = {}
    cxr14_trainer_kwargs = {}
    vinbig_trainer_kwargs = dict(
        training_data = vinbig_training_data,
    )

    trainer_engine_kwargs = dict(
        use_text=False,
        classify_orientation=classify_orientation,
        classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        binary_loss_name=binary_loss_name,        
        use_amp=use_amp,
        training=True,
        use_chexpert=train_chexpert,        
        use_cxr14=train_cxr14,
        use_vinbig=train_vinbig,
        iters_to_accumulate=iters_to_accumulate,
    )
    val_engine_kwargs = dict(
        use_text=False,
        classify_orientation=classify_orientation,
        classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        training=False,
    )
    
    dataloading_kwargs = dict(
        img_aug_mode = img_aug_mode,
        mimiccxr_weight = mimiccxr_weight,
        iuxray_weight = iuxray_weight,
        chexpert_weight = chexpert_weight,
    )

    image_transform_kwargs = dict(
        image_size = image_size,
        augmentation_mode = img_aug_mode,
        use_clip_transform = use_clip,
    )
    
    training_kwargs = dict(
        use_amp = use_amp,
        train_mimiccxr = train_mimiccxr,
        train_iuxray = train_iuxray,
        train_chexpert = train_chexpert,
        binary_loss_name = binary_loss_name,
    )
    auxiliary_tasks_kwargs = dict(
        # medical tags
        classify_tags = classify_tags,
        n_medical_tags = n_medical_tags,
        iuxray_rid2tags_filename = iuxray_rid2tags_filename,
        mimiccxr_rid2tags_filename = mimiccxr_rid2tags_filename,
        # image orientation
        classify_orientation = classify_orientation,
        # chexpert labels
        classify_chexpert = classify_chexpert,
        iuxray_chexpert_labels_filename = iuxray_chexpert_labels_filename,
        mimiccxr_chexpert_labels_filename = mimiccxr_chexpert_labels_filename,
        # question labels
        classify_questions = classify_questions,
        n_questions = n_questions,
        iuxray_question_labels_filename = iuxray_question_labels_filename,
        mimiccxr_question_labels_filename = mimiccxr_question_labels_filename,
    )

    return train_model(model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs,
                iuxray_trainer_kwargs,
                chexpert_trainer_kwargs,
                cxr14_trainer_kwargs,
                vinbig_trainer_kwargs,
                dataloading_kwargs,
                image_transform_kwargs,
                training_kwargs,
                trainer_engine_kwargs,
                val_engine_kwargs,
                auxiliary_tasks_kwargs,
                epochs,
                batches_per_epoch,
                num_workers,
                device=device,
                save=save,
                debug=debug)

def resume_training(
    checkpoint_folder,
    lr,
    lr_decay,
    lr_decay_patience,
    batch_size,
    num_workers,
    one_question_per_batch,
    epochs,
    batches_per_epoch,
    device = 'GPU',
    save = True,
    override_lr = False,    
    debug = False,
    **unused_kwargs,
):
    print('----- Resuming training ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
    iuxray_trainer_kwargs = metadata['iuxray_trainer_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    training_kwargs = metadata['training_kwargs']
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']

    if override_lr:
        optimizer_kwargs = dict(
            lr = lr,
        )
        lr_scheduler_kwargs = dict(
            factor = lr_decay,
            patience = lr_decay_patience,
        )

    return train_model(model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs,
                iuxray_trainer_kwargs,
                dataloading_kwargs,
                training_kwargs,
                auxiliary_tasks_kwargs,
                epochs,
                batch_size,
                batches_per_epoch,
                one_question_per_batch,
                num_workers,
                device = device,
                checkpoint_folder_path = checkpoint_folder,
                save = save,
                override_lr = override_lr,
                debug = debug)

def debug_main(args):
    args = parse_args(args)
    args = parsed_args_to_dict(args)
    if args['checkpoint_folder'] is not None:
        return resume_training(**args, debug=True)
    else:
        return train_from_scratch(**args, debug=True)

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    if args['checkpoint_folder'] is not None:
        resume_training(**args)
    else:
        train_from_scratch(**args)