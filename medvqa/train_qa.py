import  os
import argparse

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ignite.engine import Events
from ignite.handlers.timing import Timer

from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_bleu_question,
    attach_ciderd,
    attach_loss,
    attach_weighted_medical_completeness
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
from medvqa.models.qa import OpenEndedQA
from medvqa.utils.files import (
    load_json_file,
    get_checkpoint_folder_path,
)
from medvqa.training.qa import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    multi_cyclic_dataloaders_generator,
    qa_collate_batch_fn,
)
from medvqa.metrics.utils import (
    get_merge_metrics_fn,
    get_hybrid_score_name,
)
from medvqa.datasets.mimiccxr.mimiccxr_qa_dataset_management import MIMICCXR_QA_Trainer
from medvqa.datasets.iuxray.iuxray_qa_dataset_management import IUXRAY_QA_Trainer
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
    parser.add_argument('--iuxray-preprocessed-data-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-qa-adapted-reports-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-preprocessed-data-filename', type=str, default=None)
    parser.add_argument('--vocab-min-freq', type=int, default=5,
                        help='Min frequency of tokens in vocabulary')
    parser.add_argument('--embed-size', type=int, default=128,
                        help='Size of word embeddings')
    parser.add_argument('--question-hidden-size', type=int, default=128,
                        help='Size of question hidden state vectors')
    parser.add_argument('--answer-hidden-size', type=int, default=128,
                        help='Size of answer hidden state vectors')
    parser.add_argument('--n-lstm-layers', type=int, default=1,
                        help='Number of LSTM layers to use in the answer decoder')
    parser.add_argument('--question-vec-size', type=int, default=128,
                        help='Size of vector that encodes the question')
    parser.add_argument('--dropout-prob', type=int, default=0,
                        help='Dropout probability')    
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
    parser.add_argument('--mimic-iuxray-freqs', nargs=2, type=int, default=[20 * 10, 10],
                        help='Relative number of batches to sample from MIMIC-CXR and IU X-Ray (for rebalancing purposes)')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')
    
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=True)

    parser.add_argument('--override-lr', dest='override_lr', action='store_true')
    parser.set_defaults(override_lr=False)

    parser.add_argument('--no-mimiccxr', dest='train_mimiccxr', action='store_false')
    parser.set_defaults(train_mimiccxr=True)
    parser.add_argument('--no-iuxray', dest='train_iuxray', action='store_false')
    parser.set_defaults(train_iuxray=True)
    
    return parser.parse_args()

_METRIC_WEIGHTS = {
    'bleu_question': 1,
    'ciderD': 0.1,
    'wmedcomp': 1,
}

def train_model(
    tokenizer_kwargs,
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_qa_trainer_kwargs,
    iuxray_qa_trainer_kwargs,
    dataloading_kwargs,
    epochs,
    batch_size,
    batches_per_epoch,
    num_workers,
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
    
    # Filenames
    iuxray_preprocessed_data_filename = iuxray_qa_trainer_kwargs['preprocessed_data_filename']
    iuxray_qa_adapted_reports_filename = iuxray_qa_trainer_kwargs['qa_adapted_reports_filename']
    mimiccxr_preprocessed_data_filename = mimiccxr_qa_trainer_kwargs['preprocessed_data_filename']
    mimiccxr_qa_adapted_reports_filename = mimiccxr_qa_trainer_kwargs['qa_adapted_reports_filename']
    assert iuxray_preprocessed_data_filename is not None
    assert iuxray_qa_adapted_reports_filename is not None
    assert mimiccxr_preprocessed_data_filename is not None
    assert mimiccxr_qa_adapted_reports_filename is not None

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Load qa adapted reports
    count_print('Loading iuxray and mimiccxr QA adapted reports ...')
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
    count_print('Creating instance of OpenEndedQA model ...')
    model = OpenEndedQA(vocab_size=tokenizer.vocab_size,
                        start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                        device=device,
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
    trainer = get_engine(model, tokenizer, device, training=True, optimizer=optimizer)
    validator = get_engine(model, tokenizer, device, training=False)

    # Create MIMIC-CXR qa trainer
    if train_mimiccxr:
        count_print('Creating MIMIC-CXR qa trainer ...')
        mimiccxr_qa_trainer = MIMICCXR_QA_Trainer(
            batch_size = batch_size,
            collate_batch_fn = qa_collate_batch_fn,
            num_workers = num_workers,
            preprocessed_data_filename = mimiccxr_preprocessed_data_filename
        )
    
    # Create IU X-Ray qa trainer
    if train_iuxray:
        count_print('Creating IU X-Ray qa trainer ...')
        iuxray_qa_trainer = IUXRAY_QA_Trainer(
            batch_size = batch_size,
            collate_batch_fn = qa_collate_batch_fn,
            num_workers = num_workers,
            preprocessed_data_filename = iuxray_preprocessed_data_filename
        )

    # Create complex dataloaders
    count_print('Creating dataloaders ...')

    if train_mimiccxr and train_iuxray:
        mimic_iuxray_freqs = dataloading_kwargs['mimic_iuxray_freqs']
        train_dataloader = balanced_dataloaders_generator(
            [mimiccxr_qa_trainer.train_dataloader, iuxray_qa_trainer.train_dataloader],
            mimic_iuxray_freqs)
        val_dataloaders = [mimiccxr_qa_trainer.val_dataloader, iuxray_qa_trainer.val_dataloader]
        val_dataloader_size = sum(len(d) for d in val_dataloaders)
        val_dataloader = multi_cyclic_dataloaders_generator(val_dataloaders)
    elif train_mimiccxr:
        train_dataloader = mimiccxr_qa_trainer.train_dataloader
        val_dataloader = mimiccxr_qa_trainer.val_dataloader
        val_dataloader_size = len(val_dataloader)
    else:
        assert train_iuxray
        train_dataloader = iuxray_qa_trainer.train_dataloader
        val_dataloader = iuxray_qa_trainer.val_dataloader
        val_dataloader_size = len(val_dataloader)

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    attach_bleu_question(trainer, device)
    attach_bleu_question(validator, device)
    
    attach_ciderd(trainer, device)
    attach_ciderd(validator, device)

    attach_weighted_medical_completeness(trainer, device, tokenizer)
    attach_weighted_medical_completeness(validator, device, tokenizer)

    attach_loss('loss', trainer, device)
    attach_loss('loss', validator, device)

    attach_loss('question_loss', trainer, device)
    attach_loss('question_loss', validator, device)

    attach_loss('answer_loss', trainer, device)
    attach_loss('answer_loss', validator, device)
    
    # Timer
    timer = Timer()
    timer.attach(trainer, start=Events.EPOCH_STARTED)
    timer.attach(validator, start=Events.EPOCH_STARTED)
    
    # Learning rate scheduler
    metrics_to_merge = ['bleu_question', 'ciderD', 'wmedcomp']    
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
            model_string = ','.join(map(str, [
                model_kwargs["embed_size"],
                model_kwargs["question_hidden_size"],
                model_kwargs["answer_hidden_size"],
                model_kwargs["n_lstm_layers"],
                model_kwargs["question_vec_size"],
                model_kwargs["dropout_prob"],
            ]))
            checkpoint_folder_path = get_checkpoint_folder_path('qa', folder, model.name,
                f'voc-minf={vocab_min_freq}',
                f'model-args=({model_string})',
                f'mim-iux-freqs={",".join(map(str, mimic_iuxray_freqs))}' \
                    if (train_iuxray and train_mimiccxr) else None,
            )
            print('checkpoint_folder_path =', checkpoint_folder_path)
            save_metadata(checkpoint_folder_path,
                        tokenizer_kwargs = tokenizer_kwargs,
                        model_kwargs = model_kwargs,
                        optimizer_kwargs = optimizer_kwargs,
                        lr_scheduler_kwargs = lr_scheduler_kwargs,
                        mimiccxr_qa_trainer_kwargs = mimiccxr_qa_trainer_kwargs,
                        iuxray_qa_trainer_kwargs = iuxray_qa_trainer_kwargs,
                        dataloading_kwargs = dataloading_kwargs)
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

    # Logging
    metrics_to_print=['loss', 'question_loss', 'answer_loss', 'bleu_question', 'ciderD', 'wmedcomp']
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
    question_hidden_size,
    answer_hidden_size,
    n_lstm_layers,
    question_vec_size,
    dropout_prob,
    # Optimizer's args
    lr,
    # lr_scheduler's args
    lr_decay,
    lr_decay_patience,
    # Dataset args    
    iuxray_qa_adapted_reports_filename,
    iuxray_preprocessed_data_filename,
    mimiccxr_qa_adapted_reports_filename,
    mimiccxr_preprocessed_data_filename,
    # Dataloading args
    batch_size,
    num_workers,
    mimic_iuxray_freqs,
    # Traning args
    epochs,
    batches_per_epoch,
    train_mimiccxr,
    train_iuxray,    
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
        question_hidden_size = question_hidden_size,
        answer_hidden_size = answer_hidden_size,
        n_lstm_layers = n_lstm_layers,
        question_vec_size = question_vec_size,
        dropout_prob = dropout_prob,
    )
    optimizer_kwargs = dict(
        lr = lr,
    )
    lr_scheduler_kwargs = dict(
        factor = lr_decay,
        patience = lr_decay_patience,
    )
    mimiccxr_qa_trainer_kwargs = dict(
        qa_adapted_reports_filename = mimiccxr_qa_adapted_reports_filename,
        preprocessed_data_filename = mimiccxr_preprocessed_data_filename,
    )
    iuxray_qa_trainer_kwargs = dict(
        qa_adapted_reports_filename = iuxray_qa_adapted_reports_filename,
        preprocessed_data_filename = iuxray_preprocessed_data_filename,
    )
    dataloading_kwargs = dict(
        mimic_iuxray_freqs = mimic_iuxray_freqs,
    )

    train_model(tokenizer_kwargs,
                model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_qa_trainer_kwargs,
                iuxray_qa_trainer_kwargs,
                dataloading_kwargs,
                epochs,
                batch_size,
                batches_per_epoch,
                num_workers,
                device = device,
                save = save,
                train_mimiccxr = train_mimiccxr,
                train_iuxray = train_iuxray)

def resume_training(
    checkpoint_folder,
    lr,
    lr_decay,
    lr_decay_patience,
    batch_size,
    num_workers,
    epochs,
    batches_per_epoch,
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
    mimiccxr_qa_trainer_kwargs = metadata['mimiccxr_qa_trainer_kwargs']
    iuxray_qa_trainer_kwargs = metadata['iuxray_qa_trainer_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']

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
                mimiccxr_qa_trainer_kwargs,
                iuxray_qa_trainer_kwargs,
                dataloading_kwargs,
                epochs,
                batch_size,
                batches_per_epoch,
                num_workers,
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