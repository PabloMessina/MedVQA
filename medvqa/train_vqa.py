import  os
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ignite.engine import Events, Engine
from ignite.engine import Events, Engine
from ignite.handlers.timing import Timer

from medvqa.metrics import (
    attach_bleu_answer,
    attach_bleu_question,
    attach_loss
)
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    load_metadata,
    save_metadata,
)
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.utils.common import WORKSPACE_DIR
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
from medvqa.training.vqa import get_step_fn
from medvqa.datasets.dataloading_utils import (
    question_balanced_train_dataloader_generator,
    multi_cyclic_dataloader_sampler,
    collate_batch_fn
)
from medvqa.datasets.mimiccxr.vqa_training_handler import MIMICCXR_VQATrainingHandler
from medvqa.datasets.mimiccxr import (
    MIMICCXR_QA_ADAPTED_REPORTS_JSON_PATH,
)
from medvqa.datasets.iuxray.vqa_training_handler import IUXRAY_VQATrainingHandler
from medvqa.datasets.iuxray import (
    IUXRAY_QA_ADAPTED_REPORTS_JSON_PATH,
)
from medvqa.utils.images import get_image_transform
from medvqa.utils.logging import CountPrinter

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs the model will be trained')
    parser.add_argument('--batches-per-epoch', type=int, required=True,
                        help='Number of batches per epoch')

    # optional arguments
    parser.add_argument('--checkpoint-folder', type=str,
                        help='Relative path to folder with checkpoint to resume training from')
    parser.add_argument('--vocab-min-freq', type=int, default=4,
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

    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    
    return parser.parse_args()


_metric_names = ['bleu_question', 'bleu_answer']

def _merge_metrics(train_metrics, val_metrics):
    train_value = 0
    val_value = 0
    for met in _metric_names:
        train_value += train_metrics[met]
        val_value += val_metrics[met]
    return train_value * 0.65 + val_value * 0.35

def train_model(
    tokenizer_kwargs,
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_vqa_train_handler_kwargs,
    iuxray_vqa_train_handler_kwargs,
    dataloading_kwargs,
    epochs,
    batches_per_epoch,
    device = 'GPU',
    checkpoint_folder_path = None,
    save = True
):
    count_print = CountPrinter()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Load qa adapted reports
    count_print('Loading iuxray and mimiccxr QA adapted reports ...')
    iuxray_qa_reports = load_json_file(IUXRAY_QA_ADAPTED_REPORTS_JSON_PATH)
    mimiccxr_qa_reports = load_json_file(MIMICCXR_QA_ADAPTED_REPORTS_JSON_PATH)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    vocab_min_freq = tokenizer_kwargs['vocab_min_freq']
    tokenizer = Tokenizer(vocab_filepath=f'vocab/id2tokens_vqa_mimiccxr+iuxray_minf={vocab_min_freq}.pickle',
                          qa_adapted_datasets=[iuxray_qa_reports, mimiccxr_qa_reports],
                          min_freq=vocab_min_freq)
    
    # Create model
    count_print('Creating instance of OpenEndedVQA model ...')    
    model = OpenEndedVQA(vocab_size=tokenizer.vocab_size,
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

    # Criterion
    count_print('Defining loss criterion ...')
    nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss

    # Create trainer and validator engines
    count_print('Creating trainer and validator engines ...')
    train_step = get_step_fn(model, optimizer, nlg_criterion, tokenizer,
                             training=True, device=device)
    trainer = Engine(train_step)

    val_step = get_step_fn(model, optimizer, nlg_criterion, tokenizer,
                           training=False, device=device)
    validator = Engine(val_step)

    # Default image transform
    count_print('Defining image transform ...')
    img_transform = get_image_transform()

    # Create MIMIC-CXR vqa training handler
    count_print('Creating MIMIC-CXR vqa training handler ...')
    mimiccxr_vqa_train_handler = MIMICCXR_VQATrainingHandler(
        transform = img_transform,
        collate_batch_fn = collate_batch_fn,
        tokenizer = tokenizer,
        mimiccxr_qa_reports = mimiccxr_qa_reports,
        **mimiccxr_vqa_train_handler_kwargs,
    )
    
    # Create IU X-Ray vqa training handler
    count_print('Creating IU X-Ray vqa training handler ...')
    iuxray_vqa_train_handler = IUXRAY_VQATrainingHandler(
        transform = img_transform,
        collate_batch_fn = collate_batch_fn,
        tokenizer = tokenizer,
        iuxray_qa_reports = iuxray_qa_reports,
        **iuxray_vqa_train_handler_kwargs,
    )

    # Create dataloaders
    count_print('Creating dataloaders ...')
    mimiccxr_balanced_dataloader = question_balanced_train_dataloader_generator(mimiccxr_vqa_train_handler)
    iuxray_balanced_dataloader = question_balanced_train_dataloader_generator(iuxray_vqa_train_handler)
    mimic_iuxray_freqs = dataloading_kwargs['mimic_iuxray_freqs']    
    train_loader = multi_cyclic_dataloader_sampler(
        [mimiccxr_balanced_dataloader, iuxray_balanced_dataloader],
        mimic_iuxray_freqs,
        shuffle=True,
    )
    val_dataloaders = [mimiccxr_vqa_train_handler.val_dataloader, iuxray_vqa_train_handler.val_dataloader]
    val_dataloader_size = sum(len(d) for d in val_dataloaders)
    val_dataloader = multi_cyclic_dataloader_sampler(val_dataloaders)

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    # metrics
    attach_bleu_question(trainer, device)
    attach_bleu_question(validator, device)

    attach_bleu_answer(trainer, device)
    attach_bleu_answer(validator, device)

    # losses
    attach_loss('loss', trainer, device)
    attach_loss('loss', validator, device)
    
    # timer
    timer = Timer()
    timer.attach(trainer, start=Events.EPOCH_STARTED)
    timer.attach(validator, start=Events.EPOCH_STARTED)
    
    # logging
    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=[
        'loss', 'bleu_question', 'bleu_answer'
    ])
    log_iteration_handler = get_log_iteration_handler()

    # learning rate scheduler
    lr_sch_handler = get_lr_sch_handler(trainer, validator, lr_scheduler, _merge_metrics)

    # checkpoint saving    
    model_wrapper = ModelWrapper(model, optimizer, lr_scheduler)
    
    if checkpoint_folder_path is None: # first time
        if save: # only if we want to save checkpoints to disk
            checkpoint_folder_path = get_checkpoint_folder_path('vqa', 'mimiccxr+iuxray', model.name,
                f'vocab-minf={vocab_min_freq}',
                f'emb-size={model_kwargs["embed_size"]}',
                f'hidden-size={model_kwargs["hidden_size"]}',
                f'q-vec-size={model_kwargs["question_vec_size"]}',
                f'img-loc-feat-size={model_kwargs["image_local_feat_size"]}',
                f'drop-prob={model_kwargs["dropout_prob"]}',
                f'mimic-iuxray-freqs={",".join(map(str, mimic_iuxray_freqs))}'
            )
            print('checkpoint_folder_path =', checkpoint_folder_path)
            save_metadata(checkpoint_folder_path,
                        tokenizer_kwargs = tokenizer_kwargs,
                        model_kwargs = model_kwargs,
                        optimizer_kwargs = optimizer_kwargs,
                        lr_scheduler_kwargs = lr_scheduler_kwargs,
                        mimiccxr_vqa_train_handler_kwargs = mimiccxr_vqa_train_handler_kwargs,
                        iuxray_vqa_train_handler_kwargs = iuxray_vqa_train_handler_kwargs,
                        dataloading_kwargs = dataloading_kwargs)        
    else: # resuming
        checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
        count_print('Loading model from checkpoint ...')
        print('checkpoint_path = ', checkpoint_path)
        model_wrapper.load_checkpoint(checkpoint_path, device)
    
    score_fn = lambda _ : _merge_metrics(trainer.state.metrics, validator.state.metrics)

    if save: # only if we want to save checkpoints to disk
        checkpoint_handler = get_checkpoint_handler(model_wrapper, checkpoint_folder_path, trainer,
                                                    epoch_offset=model_wrapper.get_epoch(),
                                                    score_name='bleu(q+a)', score_fn=score_fn)

    # attaching handlers
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
    trainer.run(train_loader,
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
    # Optimizer's args
    lr,
    # lr_scheduler's args
    lr_decay,
    lr_decay_patience,
    # Dataset split args
    n_val_examples_per_question,
    min_train_examples_per_question,
    # Dataloading args
    batch_size,
    mimic_iuxray_freqs = [20 * 10, 10],
    # Traning args
    epochs = 1,
    batches_per_epoch = 1000,
    # GPU
    device = 'GPU',
    # Other args
    save = True,
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
    )
    optimizer_kwargs = dict(
        lr = lr,
    )
    lr_scheduler_kwargs = dict(
        factor = lr_decay,
        patience = lr_decay_patience,
    )
    split_kwargs = dict(
        n_val_examples_per_question = n_val_examples_per_question,
        min_train_examples_per_question = min_train_examples_per_question,
    )
    mimiccxr_vqa_train_handler_kwargs = dict(
        batch_size = batch_size,
        split_kwargs = split_kwargs,
    )
    iuxray_vqa_train_handler_kwargs = dict(
        batch_size = batch_size,
        split_kwargs = split_kwargs,
    )
    dataloading_kwargs = dict(
        mimic_iuxray_freqs = mimic_iuxray_freqs,
    )

    train_model(tokenizer_kwargs,
                model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_vqa_train_handler_kwargs,
                iuxray_vqa_train_handler_kwargs,
                dataloading_kwargs,
                epochs,
                batches_per_epoch,
                device = device,
                save = save)

def resume_training(
    checkpoint_folder,
    epochs = 1,
    batches_per_epoch = 1000,
    device = 'GPU',
    save = True,
):
    print('----- Resuming training ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    mimiccxr_vqa_train_handler_kwargs = metadata['mimiccxr_vqa_train_handler_kwargs']
    iuxray_vqa_train_handler_kwargs = metadata['iuxray_vqa_train_handler_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']

    train_model(tokenizer_kwargs,
                model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_vqa_train_handler_kwargs,
                iuxray_vqa_train_handler_kwargs,
                dataloading_kwargs,
                epochs,
                batches_per_epoch,
                device = device,
                checkpoint_folder_path = checkpoint_folder,
                save = save)

if __name__ == '__main__':

    args = parse_args()
    args = {k : v for k, v in vars(args).items() if v is not None}
    print('script\'s arguments:')
    for k, v in args.items():
        print(f'   {k}: {v}')

    if args.get('checkpoint_folder', None):
        resume_training(checkpoint_folder=args['checkpoint_folder'],
                        epochs=args['epochs'],
                        batches_per_epoch=args['batches_per_epoch'],
                        device=args['device'],
                        save=args['save'])
    else:
        train_from_scratch(**args)