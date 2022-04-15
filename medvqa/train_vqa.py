import  os
import argparse

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ignite.engine import Events
from ignite.handlers.timing import Timer

from medvqa.utils.constants import (
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID,
)
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_bleu_question,
    attach_ciderd,
    attach_weighted_medical_completeness,
    attach_medical_tags_f1score,
    attach_chexpert_labels_accuracy,
    attach_dataset_aware_orientation_accuracy,
    attach_loss
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
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.models.vqa.open_ended_vqa import OpenEndedVQA
from medvqa.utils.files import (
    load_json_file,
    get_checkpoint_folder_path,
)
from medvqa.training.vqa import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    multi_cyclic_dataloader_sampler,
    get_collate_batch_fn
)
from medvqa.metrics.utils import (
    get_merge_metrics_fn,
    get_hybrid_score_name,
)
from medvqa.datasets.mimiccxr.mimiccxr_vqa_dataset_management import MIMICCXR_VQA_Trainer
from medvqa.datasets.iuxray.iuxray_vqa_dataset_management import IUXRAY_VQA_Trainer
from medvqa.datasets.images import get_image_transform
from medvqa.utils.logging import CountPrinter

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs the model will be trained')
    parser.add_argument('--batches-per-epoch', type=int, required=True,
                        help='Number of batches per epoch')

    # optional arguments
    parser.add_argument('--checkpoint-folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')    
    parser.add_argument('--iuxray-qa-adapted-reports-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-qa-adapted-reports-filename', type=str, default=None)
    parser.add_argument('--vocab-min-freq', type=int, default=5,
                        help='Min frequency of tokens in vocabulary')
    parser.add_argument('--embed-size', type=int, default=128,
                        help='Size of word embeddings')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Size of hidden state vectors')
    parser.add_argument('--question-vec-size', type=int, default=128,
                        help='Size of vector that encodes the question')
    parser.add_argument('--image-local-feat-size', type=int, default=1024,
                        help='Size of local feature vectors from the CNN. They must match the actual vectors output by the CNN')
    parser.add_argument('--dropout-prob', type=int, default=0,
                        help='Dropout probability')
    parser.add_argument('--densenet-pretrained-weights-path', type=str, default='',
                        help='Path to densenet 121 pretrained weights')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.76,
                        help='Learning rate decay')
    parser.add_argument('--lr-decay-patience', type=int, default=2,
                        help='Learning rate decay patience')
    parser.add_argument('--n-val-examples-per-question', type=int, default=10,
                        help='Number of validation examples per question')
    parser.add_argument('--min-train-examples-per-question', type=int, default=100,
                        help='Minimum number of train examples per question to include validation examples')
    parser.add_argument('--batch-size', type=int, default=45,
                        help='Batch size')
    parser.add_argument('--mimic-iuxray-freqs', nargs=2, type=int, default=[20 * 10, 10],
                        help='Relative number of batches to sample from MIMIC-CXR and IU X-Ray (for rebalancing purposes)')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')    
    parser.add_argument('--img-aug-mode', type=str, default=None,
                        help='Mode of data augmentation used for images')

    # balanced dataset arguments
    parser.add_argument('--balanced-split', dest='balanced_split', action='store_true')
    parser.set_defaults(balanced_split=False)
    parser.add_argument('--n-healthy-per-question', type=int, default=2)
    parser.add_argument('--n-unhealthy-per-question', type=int, default=3)
    parser.add_argument('--min-question-count', type=int, default=100)
    parser.add_argument('--iuxray-balanced-metadata-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-balanced-metadata-filename', type=str, default=None)

    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=True)

    parser.add_argument('--override-lr', dest='override_lr', action='store_true')
    parser.set_defaults(override_lr=False)

    parser.add_argument('--no-mimiccxr', dest='train_mimiccxr', action='store_false')
    parser.set_defaults(train_mimiccxr=True)
    parser.add_argument('--no-iuxray', dest='train_iuxray', action='store_false')
    parser.set_defaults(train_iuxray=True)

    # Auxiliary tasks arguments
    
    # medical tags
    parser.add_argument('--use-tags', dest='use_tags', action='store_true')
    parser.set_defaults(use_tags=False)
    parser.add_argument('--n-medical-tags', type=int, default=None,
                        help='Number of medical tags (for tag prediction auxiliary task)')
    parser.add_argument('--iuxray-medical-tags-per-report-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-medical-tags-per-report-filename', type=str, default=None)
    # orientation
    parser.add_argument('--use-orientation', dest='use_orientation', action='store_true')
    parser.set_defaults(use_orientation=False)
    # chexpert labels
    parser.add_argument('--use-chexpert', dest='use_chexpert', action='store_true')
    parser.set_defaults(use_chexpert=False)
    parser.add_argument('--iuxray-chexpert-labels-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-chexpert-labels-filename', type=str, default=None)
    
    return parser.parse_args()

_METRIC_WEIGHTS = {
    'bleu_question': 1,
    'ciderD': 0.1,
    'wmedcomp': 1,
    'medtagf1': 1,
    'orienacc': 1,
    'chxlabelacc': 1,
}

def train_model(
    tokenizer_kwargs,
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_vqa_trainer_kwargs,
    iuxray_vqa_trainer_kwargs,
    dataloading_kwargs,
    auxiliary_tasks_kwargs,
    epochs,
    batches_per_epoch,
    device = 'GPU',
    checkpoint_folder_path = None,
    save = True,
    override_lr = False,
    train_iuxray = True,
    train_mimiccxr = True,    
):

    assert train_iuxray or train_mimiccxr

    count_print = CountPrinter()
    
    # Pull out some args from kwargs

    # auxiliary task: medical tags prediction
    use_tags = auxiliary_tasks_kwargs['use_tags']
    n_medical_tags = auxiliary_tasks_kwargs['n_medical_tags']
    iuxray_medical_tags_per_report_filename = auxiliary_tasks_kwargs['iuxray_medical_tags_per_report_filename']
    mimiccxr_medical_tags_per_report_filename = auxiliary_tasks_kwargs['mimiccxr_medical_tags_per_report_filename']
    if use_tags:
        assert n_medical_tags is not None
        if train_iuxray:
            assert iuxray_medical_tags_per_report_filename is not None
        if train_mimiccxr:
            assert mimiccxr_medical_tags_per_report_filename is not None
    
    # auxiliary task: orientation classification
    use_orientation = auxiliary_tasks_kwargs['use_orientation']

    # auxiliary task: chexpert labels
    use_chexpert = auxiliary_tasks_kwargs['use_chexpert']
    iuxray_chexpert_labels_filename = auxiliary_tasks_kwargs['iuxray_chexpert_labels_filename']
    mimiccxr_chexpert_labels_filename = auxiliary_tasks_kwargs['mimiccxr_chexpert_labels_filename']
    
    # QA dataset filenames
    iuxray_qa_adapted_reports_filename = iuxray_vqa_trainer_kwargs['qa_adapted_reports_filename']
    mimiccxr_qa_adapted_reports_filename = mimiccxr_vqa_trainer_kwargs['qa_adapted_reports_filename']
    assert iuxray_qa_adapted_reports_filename is not None
    assert mimiccxr_qa_adapted_reports_filename is not None

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Load qa adapted reports
    count_print('Loading iuxray and mimiccxr QA adapted reports ...')
    iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, iuxray_qa_adapted_reports_filename)
    mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, mimiccxr_qa_adapted_reports_filename)
    iuxray_qa_reports = load_json_file(iuxray_qa_adapted_reports_path)
    mimiccxr_qa_reports = load_json_file(mimiccxr_qa_adapted_reports_path)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    vocab_min_freq = tokenizer_kwargs['vocab_min_freq']
    tokenizer = Tokenizer(qa_adapted_filenames=[iuxray_qa_adapted_reports_filename,
                                                mimiccxr_qa_adapted_reports_filename],
                          qa_adapted_datasets=[iuxray_qa_reports, mimiccxr_qa_reports],
                          min_freq=vocab_min_freq)
    
    # Create model
    count_print('Creating instance of OpenEndedVQA model ...')    
    model = OpenEndedVQA(vocab_size=tokenizer.vocab_size,
                         start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                         device=device, 
                         n_medical_tags=n_medical_tags,
                         classify_orientation=use_orientation,
                         classify_chexpert=use_chexpert,
                         **model_kwargs)
    model = model.to(device)

    # Optimizer
    count_print('Defining Adam optimizer ...')
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    # Learning rate scheduler
    count_print('Defining ReduceLROnPlateau scheduler ...')
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True,
                                     **lr_scheduler_kwargs)

    # Create trainer and validator engines
    count_print('Creating trainer and validator engines ...')    
    trainer = get_engine(model, tokenizer,
                         use_tags, use_orientation, use_chexpert,
                         device, training=True, optimizer=optimizer)
    validator = get_engine(model, tokenizer,
                         use_tags, use_orientation, use_chexpert,
                         device, training=False)

    # Default image transform
    count_print('Defining image transform ...')
    img_transform = get_image_transform(augmentation_mode=dataloading_kwargs['img_aug_mode'])

    # Define collate_batch_fn
    if train_mimiccxr:
        mimiccxr_collate_batch_fn = get_collate_batch_fn(MIMICCXR_DATASET_ID,
                                                        use_tags = use_tags,
                                                        n_tags = n_medical_tags,
                                                        use_orientation = use_orientation,
                                                        use_chexpert = use_chexpert)
    if train_iuxray:
        iuxray_collate_batch_fn = get_collate_batch_fn(IUXRAY_DATASET_ID,
                                                    use_tags = use_tags,
                                                    n_tags = n_medical_tags,
                                                    use_orientation = use_orientation,
                                                    use_chexpert = use_chexpert)

    # Create MIMIC-CXR vqa trainer
    if train_mimiccxr:
        count_print('Creating MIMIC-CXR vqa trainer ...')
        mimiccxr_vqa_trainer = MIMICCXR_VQA_Trainer(
            transform = img_transform,
            collate_batch_fn = mimiccxr_collate_batch_fn,
            tokenizer = tokenizer,
            mimiccxr_qa_reports = mimiccxr_qa_reports,
            use_tags = use_tags,
            medical_tags_per_report_filename = mimiccxr_medical_tags_per_report_filename,
            use_orientation = use_orientation,
            use_chexpert = use_chexpert,
            chexpert_labels_filename = mimiccxr_chexpert_labels_filename,
            **mimiccxr_vqa_trainer_kwargs,
        )
    
    # Create IU X-Ray vqa trainer
    if train_iuxray:
        count_print('Creating IU X-Ray vqa trainer ...')
        iuxray_vqa_trainer = IUXRAY_VQA_Trainer(
            transform = img_transform,
            collate_batch_fn = iuxray_collate_batch_fn,
            tokenizer = tokenizer,        
            iuxray_qa_reports = iuxray_qa_reports,
            use_tags = use_tags,
            medical_tags_per_report_filename = iuxray_medical_tags_per_report_filename,
            use_orientation = use_orientation,
            use_chexpert = use_chexpert,
            chexpert_labels_filename = iuxray_chexpert_labels_filename,
            **iuxray_vqa_trainer_kwargs,
        )

    # Create complex dataloaders
    count_print('Creating dataloaders ...')
    if train_mimiccxr:
        mimiccxr_balanced_dataloader = balanced_dataloaders_generator(
                        mimiccxr_vqa_trainer.train_dataloaders,
                        mimiccxr_vqa_trainer.train_dataset_weights)
    if train_iuxray:
        iuxray_balanced_dataloader = balanced_dataloaders_generator(
                        iuxray_vqa_trainer.train_dataloaders,
                        iuxray_vqa_trainer.train_dataset_weights)

    if train_mimiccxr and train_iuxray:
        mimic_iuxray_freqs = dataloading_kwargs['mimic_iuxray_freqs']
        train_dataloader = multi_cyclic_dataloader_sampler(
            [mimiccxr_balanced_dataloader, iuxray_balanced_dataloader],
            mimic_iuxray_freqs)
        val_dataloaders = [mimiccxr_vqa_trainer.val_dataloader, iuxray_vqa_trainer.val_dataloader]
        val_dataloader_size = sum(len(d) for d in val_dataloaders)
        val_dataloader = multi_cyclic_dataloader_sampler(val_dataloaders)    
    elif train_mimiccxr:        
        train_dataloader = mimiccxr_balanced_dataloader
        val_dataloader = mimiccxr_vqa_trainer.val_dataloader
        val_dataloader_size = len(val_dataloader)
    else:
        assert train_iuxray
        train_dataloader = iuxray_balanced_dataloader
        val_dataloader = iuxray_vqa_trainer.val_dataloader
        val_dataloader_size = len(val_dataloader)

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    # Metrics
    attach_bleu_question(trainer, device)
    attach_bleu_question(validator, device)
    
    attach_ciderd(trainer, device)
    attach_ciderd(validator, device)
    
    attach_weighted_medical_completeness(trainer, device, tokenizer)
    attach_weighted_medical_completeness(validator, device, tokenizer)
    
    if use_tags:
        attach_medical_tags_f1score(trainer, device)
        attach_medical_tags_f1score(validator, device)

    if use_orientation:
        attach_dataset_aware_orientation_accuracy(trainer)
        attach_dataset_aware_orientation_accuracy(validator)

    if use_chexpert:
        attach_chexpert_labels_accuracy(trainer, device)
        attach_chexpert_labels_accuracy(validator, device)

    # Losses
    attach_loss('loss', trainer, device)
    attach_loss('loss', validator, device)
    
    # Timer
    timer = Timer()
    timer.attach(trainer, start=Events.EPOCH_STARTED)
    timer.attach(validator, start=Events.EPOCH_STARTED)
    
    # Logging
    metrics_to_print=['loss', 'bleu_question', 'ciderD', 'wmedcomp']    
    if use_tags:
        metrics_to_print.append('medtagf1')
    if use_orientation:
        metrics_to_print.append('orienacc')
    if use_chexpert:
        metrics_to_print.append('chxlabelacc')

    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()

    # Learning rate scheduler
    metrics_to_merge = ['bleu_question', 'ciderD', 'wmedcomp']
    if use_tags:
        metrics_to_merge.append('medtagf1')
    if use_orientation:
        metrics_to_merge.append('orienacc')
    if use_chexpert:
        metrics_to_merge.append('chxlabelacc')
    
    merge_metrics_fn = get_merge_metrics_fn(metrics_to_merge, _METRIC_WEIGHTS, 0.5, 0.5)

    lr_sch_handler = get_lr_sch_handler(trainer, validator, lr_scheduler, merge_metrics_fn)

    # Checkpoint saving    
    model_wrapper = ModelWrapper(model, optimizer, lr_scheduler)
    
    if checkpoint_folder_path is None: # first time
        if save: # only if we want to save checkpoints to disk
            if train_iuxray and train_mimiccxr:
                folder = 'mimiccxr+iuxray'
            elif train_iuxray:
                folder = 'iuxray'
            else:
                folder = 'mimiccxr'
            checkpoint_folder_path = get_checkpoint_folder_path('vqa', folder, model.name,
                f'vocab-minf={vocab_min_freq}',
                f'emb-size={model_kwargs["embed_size"]}',
                f'hidden-size={model_kwargs["hidden_size"]}',
                f'q-vec-size={model_kwargs["question_vec_size"]}',
                f'img-loc-feat-size={model_kwargs["image_local_feat_size"]}',
                f'drop-prob={model_kwargs["dropout_prob"]}',
                f'cnn-pretrained={bool(model_kwargs["densenet_pretrained_weights_path"])}',
                f'mimic-iuxray-freqs={",".join(map(str, mimic_iuxray_freqs))}' \
                    if (train_iuxray and train_mimiccxr) else None,
                f'use_tags={use_tags}',
                f'use_orien={use_orientation}',
                f'use_chx={use_orientation}',
            )
            print('checkpoint_folder_path =', checkpoint_folder_path)
            save_metadata(checkpoint_folder_path,
                        tokenizer_kwargs = tokenizer_kwargs,
                        model_kwargs = model_kwargs,
                        optimizer_kwargs = optimizer_kwargs,
                        lr_scheduler_kwargs = lr_scheduler_kwargs,
                        mimiccxr_vqa_trainer_kwargs = mimiccxr_vqa_trainer_kwargs,
                        iuxray_vqa_trainer_kwargs = iuxray_vqa_trainer_kwargs,
                        dataloading_kwargs = dataloading_kwargs,
                        auxiliary_tasks_kwargs = auxiliary_tasks_kwargs)
    else: # resuming
        checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
        count_print('Loading model from checkpoint ...')
        print('checkpoint_path = ', checkpoint_path)
        model_wrapper.load_checkpoint(checkpoint_path, device, model_only=override_lr)
    
    score_fn = lambda _ : merge_metrics_fn(trainer.state.metrics, validator.state.metrics)

    if save: # only if we want to save checkpoints to disk
        checkpoint_handler = get_checkpoint_handler(model_wrapper, checkpoint_folder_path, trainer,
                                                    epoch_offset=model_wrapper.get_epoch(),
                                                    score_name=get_hybrid_score_name(metrics_to_merge),
                                                    score_fn=score_fn)

    # Attach handlers
    trainer.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda : print('(1) Training stage ...'))
    trainer.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda : validator.run(val_dataloader,
                                     max_epochs=1, epoch_length=val_dataloader_size))
    validator.add_event_handler(Events.EPOCH_STARTED, lambda : print('(2) Validation stage ...'))
    validator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    validator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    validator.add_event_handler(Events.EPOCH_COMPLETED, lr_sch_handler)
    if save: # only if we want to save checkpoints to disk
        validator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # Start training
    count_print('Running trainer engine ...')
    trainer.run(train_dataloader,
                max_epochs = epochs,
                epoch_length = batches_per_epoch)


def train_from_scratch(
    # Tokenizer's args
    vocab_min_freq,
    # Model's args
    embed_size,
    hidden_size,
    question_vec_size,
    image_local_feat_size,
    dropout_prob,
    densenet_pretrained_weights_path,
    # Optimizer's args
    lr,
    # lr_scheduler's args
    lr_decay,
    lr_decay_patience,
    # Dataset args
    n_val_examples_per_question,
    min_train_examples_per_question,
    iuxray_qa_adapted_reports_filename,
    mimiccxr_qa_adapted_reports_filename,
    balanced_split,
    n_healthy_per_question,
    n_unhealthy_per_question,
    min_question_count,
    iuxray_balanced_metadata_filename,
    mimiccxr_balanced_metadata_filename,
    # Dataloading args
    batch_size,
    mimic_iuxray_freqs,
    img_aug_mode,
    # Traning args
    epochs,
    batches_per_epoch,
    train_mimiccxr,
    train_iuxray,
    # Auxiliary tasks args
    use_tags,
    n_medical_tags,
    iuxray_medical_tags_per_report_filename,
    mimiccxr_medical_tags_per_report_filename,
    use_orientation,
    use_chexpert,
    iuxray_chexpert_labels_filename,
    mimiccxr_chexpert_labels_filename,
    # GPU
    device,
    # Other args
    save,
    **unused_kwargs,
):
    print('----- Training model from scratch ------')

    tokenizer_kwargs = dict(
        vocab_min_freq = vocab_min_freq
    )
    model_kwargs = dict(
        embed_size = embed_size,
        hidden_size = hidden_size,
        question_vec_size = question_vec_size,
        image_local_feat_size = image_local_feat_size,
        dropout_prob = dropout_prob,
        densenet_pretrained_weights_path = densenet_pretrained_weights_path,
    )
    optimizer_kwargs = dict(
        lr = lr,
    )
    lr_scheduler_kwargs = dict(
        factor = lr_decay,
        patience = lr_decay_patience,
    )
    if balanced_split:
        split_kwargs = dict(
            n_healthy_per_question = n_healthy_per_question,
            n_unhealthy_per_question = n_unhealthy_per_question,
            min_question_count = min_question_count,
        )
        assert mimiccxr_balanced_metadata_filename is not None
        assert iuxray_balanced_metadata_filename is not None
    else:
        split_kwargs = dict(
            n_val_examples_per_question = n_val_examples_per_question,
            min_train_examples_per_question = min_train_examples_per_question,
        )
    mimiccxr_vqa_trainer_kwargs = dict(
        batch_size = batch_size,
        split_kwargs = split_kwargs,
        qa_adapted_reports_filename = mimiccxr_qa_adapted_reports_filename,
        balanced_split = balanced_split,
        balanced_metadata_filename = mimiccxr_balanced_metadata_filename,
    )
    iuxray_vqa_trainer_kwargs = dict(
        batch_size = batch_size,
        split_kwargs = split_kwargs,
        qa_adapted_reports_filename = iuxray_qa_adapted_reports_filename,
        balanced_split = balanced_split,
        balanced_metadata_filename = iuxray_balanced_metadata_filename,
    )
    dataloading_kwargs = dict(
        mimic_iuxray_freqs = mimic_iuxray_freqs,
        img_aug_mode = img_aug_mode,
    )
    auxiliary_tasks_kwargs = dict(
        # medical tags
        use_tags = use_tags,
        n_medical_tags = n_medical_tags,
        iuxray_medical_tags_per_report_filename = iuxray_medical_tags_per_report_filename,
        mimiccxr_medical_tags_per_report_filename = mimiccxr_medical_tags_per_report_filename,
        # image orientation
        use_orientation = use_orientation,
        # chexpert labels
        use_chexpert = use_chexpert,
        iuxray_chexpert_labels_filename = iuxray_chexpert_labels_filename,
        mimiccxr_chexpert_labels_filename = mimiccxr_chexpert_labels_filename,
    )

    train_model(tokenizer_kwargs,
                model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_vqa_trainer_kwargs,
                iuxray_vqa_trainer_kwargs,
                dataloading_kwargs,
                auxiliary_tasks_kwargs,
                epochs,
                batches_per_epoch,
                device = device,
                save = save,
                train_mimiccxr = train_mimiccxr,
                train_iuxray = train_iuxray)

def resume_training(
    checkpoint_folder,
    lr,
    lr_decay,
    lr_decay_patience,
    epochs = 1,
    batches_per_epoch = 1000,
    device = 'GPU',
    save = True,
    override_lr = False,    
    train_mimiccxr = True,
    train_iuxray = True,
    **unused_kwargs,
):
    print('----- Resuming training ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    mimiccxr_vqa_trainer_kwargs = metadata['mimiccxr_vqa_trainer_kwargs']
    iuxray_vqa_trainer_kwargs = metadata['iuxray_vqa_trainer_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']

    if override_lr:
        optimizer_kwargs = dict(
            lr = lr,
        )
        lr_scheduler_kwargs = dict(
            factor = lr_decay,
            patience = lr_decay_patience,
        )

    train_model(tokenizer_kwargs,
                model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_vqa_trainer_kwargs,
                iuxray_vqa_trainer_kwargs,
                dataloading_kwargs,
                auxiliary_tasks_kwargs,
                epochs,
                batches_per_epoch,
                device = device,
                checkpoint_folder_path = checkpoint_folder,
                save = save,
                override_lr = override_lr,                
                train_mimiccxr = train_mimiccxr,
                train_iuxray = train_iuxray)

if __name__ == '__main__':

    args = parse_args()
    args = parsed_args_to_dict(args)
    if args['checkpoint_folder'] is not None:
        resume_training(**args)
    else:
        train_from_scratch(**args)