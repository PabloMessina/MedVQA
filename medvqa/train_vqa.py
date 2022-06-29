import  os
import argparse

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.chexpert.chexpert_vision_dataset_management import Chexpert_VisualModuleTrainer
from medvqa.models.vqa.answer_decoder import AnswerDecoding

from medvqa.models.vqa.open_ended_vqa import QuestionEncoding
from medvqa.utils.constants import (
    CHEXPERT_DATASET_ID,
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID,
    MetricNames,
)
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_dataset_aware_ciderd,
    attach_dataset_aware_exactmatch_question,
    attach_exactmatch_question,
    attach_ciderd,
    attach_weighted_medical_completeness,
    attach_dataset_aware_weighted_medical_completeness,
    attach_medical_tags_f1score,
    attach_chexpert_labels_accuracy,
    attach_chexpert_labels_macroavgf1,
    attach_chexpert_labels_microavgf1,
    attach_chexpert_labels_roc_auc,
    attach_question_labels_f1score,
    attach_dataset_aware_orientation_accuracy,
    attach_dataset_aware_question_labels_f1score,
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
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.models.vqa import OpenEndedVQA
from medvqa.utils.files import (
    load_json_file,
    get_checkpoint_folder_path,
)
from medvqa.training.vqa import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    multi_cyclic_dataloaders_generator,
    get_vqa_collate_batch_fn,
    get_vision_collate_batch_fn,
)
from medvqa.metrics.utils import (
    get_merge_metrics_fn,
    get_hybrid_score_name,
)
from medvqa.datasets.mimiccxr.mimiccxr_vqa_dataset_management import MIMICCXR_VQA_Trainer
from medvqa.datasets.iuxray.iuxray_vqa_dataset_management import IUXRAY_VQA_Trainer
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
    parser.add_argument('--iuxray-qa-adapted-reports-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-qa-adapted-reports-filename', type=str, default=None)
    parser.add_argument('--vocab-min-freq', type=int, default=5,
                        help='Min frequency of tokens in vocabulary')
    parser.add_argument('--embed-size', type=int, default=128,
                        help='Size of word embeddings')
    parser.add_argument('--question-encoding', type=str, default='bilstm',
                        help='Method used to encode the question (bilstm, one-hot)')
    parser.add_argument('--question-hidden-size', type=int, default=128,
                        help='Size of question hidden state vectors')
    parser.add_argument('--answer-hidden-size', type=int, default=128,
                        help='Size of answer hidden state vectors')
    parser.add_argument('--n-lstm-layers', type=int, default=1,
                        help='Number of LSTM layers to use in the answer decoder')
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
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for parallel dataloading')    
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')    
    parser.add_argument('--img-aug-mode', type=str, default=None,
                        help='Mode of data augmentation used for images')

    parser.add_argument('--freeze-cnn', dest='freeze_cnn', action='store_true')
    parser.set_defaults(freeze_cnn=False)

    parser.add_argument('--mimiccxr-weight', type=float, default=0.73,
                        help='Relative number of batches to sample from MIMIC-CXR dataset (for rebalancing purposes)')
    parser.add_argument('--chexpert-weight', type=float, default=0.2,
                        help='Relative number of batches to sample from CheXpert dataset (for rebalancing purposes)')
    parser.add_argument('--iuxray-weight', type=float, default=0.07,
                        help='Relative number of batches to sample from IU X-ray dataset (for rebalancing purposes)')


    parser.add_argument('--val-answer-decoding', type=str, default='greedy-search')
    parser.add_argument('--beam-search-k', type=int, default=None)

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)
    
    parser.add_argument('--medical-tokenization', dest='medical_tokenization', action='store_true')
    parser.set_defaults(medical_tokenization=False)

    parser.add_argument('--medical-terms-frequency-filename', type=str, default=None)

    parser.add_argument('--allowed-questions', type=str, default=None)

    parser.add_argument('--pretrained-checkpoint-folder-path', type=str, default=None)

    # balanced dataset arguments
    parser.add_argument('--balanced-split', dest='balanced_split', action='store_true')
    parser.set_defaults(balanced_split=False)
    parser.add_argument('--balanced-dataloading', dest='balanced_dataloading', action='store_true')
    parser.set_defaults(balanced_dataloading=False)
    parser.add_argument('--n-healthy-per-question', type=int, default=2)
    parser.add_argument('--n-unhealthy-per-question', type=int, default=3)
    parser.add_argument('--n-positive-per-chexpert-label', type=int, default=7)
    parser.add_argument('--min-question-count', type=int, default=100)
    parser.add_argument('--iuxray-balanced-metadata-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-balanced-metadata-filename', type=str, default=None)

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
    parser.add_argument('--iuxray-medical-tags-per-report-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-medical-tags-per-report-filename', type=str, default=None)
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
    MetricNames.EXACTMATCH_QUESTION: 1,
    MetricNames.CIDER_D: 0.1,
    MetricNames.WMEDCOMP: 1,
    MetricNames.MEDTAGF1: 1,
    MetricNames.ORIENACC: 1,
    MetricNames.CHXLABELMICROAVGF1: 0.5,
    MetricNames.CHXLABELMACROAVGF1: 0.5,
    MetricNames.QLABELSF1: 1,
    MetricNames.GENDER_ACC: 1,
}

def train_model(
    tokenizer_kwargs,
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_vqa_trainer_kwargs,
    iuxray_vqa_trainer_kwargs,
    dataloading_kwargs,
    training_kwargs,
    auxiliary_tasks_kwargs,
    val_answer_decoding,
    epochs,
    batch_size,
    batches_per_epoch,
    one_question_per_batch,
    num_workers,
    beam_search_k = None,
    device = 'GPU',
    allowed_questions = None,
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
    use_amp = training_kwargs['use_amp']
    assert train_iuxray or train_mimiccxr
    question_encoding = model_kwargs.get('question_encoding', QuestionEncoding.BILSTM)
    verbose_question = question_encoding != QuestionEncoding.ONE_HOT
    freeze_cnn = model_kwargs['freeze_cnn']

    # auxiliary task: medical tags prediction
    classify_tags = auxiliary_tasks_kwargs['classify_tags']
    n_medical_tags = auxiliary_tasks_kwargs['n_medical_tags']
    iuxray_medical_tags_per_report_filename = auxiliary_tasks_kwargs['iuxray_medical_tags_per_report_filename']
    mimiccxr_medical_tags_per_report_filename = auxiliary_tasks_kwargs['mimiccxr_medical_tags_per_report_filename']
    if classify_tags:
        assert n_medical_tags is not None
        if train_iuxray: assert iuxray_medical_tags_per_report_filename is not None
        if train_mimiccxr: assert mimiccxr_medical_tags_per_report_filename is not None
    
    # auxiliary task: orientation classification
    classify_orientation = auxiliary_tasks_kwargs['classify_orientation']

    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    iuxray_chexpert_labels_filename = auxiliary_tasks_kwargs['iuxray_chexpert_labels_filename']
    mimiccxr_chexpert_labels_filename = auxiliary_tasks_kwargs['mimiccxr_chexpert_labels_filename']

    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)
    n_questions = auxiliary_tasks_kwargs.get('n_questions', None)
    iuxray_question_labels_filename = auxiliary_tasks_kwargs.get('iuxray_question_labels_filename', None)
    mimiccxr_question_labels_filename = auxiliary_tasks_kwargs.get('mimiccxr_question_labels_filename', None)
    if classify_questions:
        assert n_questions is not None
        if train_iuxray: assert iuxray_question_labels_filename is not None
        if train_mimiccxr: assert mimiccxr_question_labels_filename is not None

    if question_encoding == QuestionEncoding.ONE_HOT:
        assert n_questions is not None

    if train_chexpert:
        assert classify_chexpert
        assert classify_orientation
        assert classify_questions
    
    if freeze_cnn:
        assert not classify_chexpert
        assert not classify_orientation
        assert not classify_questions
        assert not train_chexpert
    
    # QA dataset filenames
    iuxray_qa_adapted_reports_filename = iuxray_vqa_trainer_kwargs['qa_adapted_reports_filename']
    mimiccxr_qa_adapted_reports_filename = mimiccxr_vqa_trainer_kwargs['qa_adapted_reports_filename']
    assert iuxray_qa_adapted_reports_filename is not None
    assert mimiccxr_qa_adapted_reports_filename is not None

    # Beam search
    if val_answer_decoding == AnswerDecoding.BEAM_SEARCH:
        assert beam_search_k is not None

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Load qa adapted reports
    count_print('Loading iuxray and mimiccxr QA adapted reports ...')
    iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, iuxray_qa_adapted_reports_filename)
    mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, mimiccxr_qa_adapted_reports_filename)
    iuxray_qa_reports = load_json_file(iuxray_qa_adapted_reports_path)
    mimiccxr_qa_reports = load_json_file(mimiccxr_qa_adapted_reports_path)
    assert iuxray_qa_reports['questions'] == mimiccxr_qa_reports['questions']
    
    # Init tokenizer
    count_print('Initializing tokenizer ...')
    vocab_min_freq = tokenizer_kwargs['vocab_min_freq']
    medical_tokenization = tokenizer_kwargs['medical_tokenization']
    medical_terms_frequency_filename = tokenizer_kwargs['medical_terms_frequency_filename']
    assert medical_tokenization == (medical_terms_frequency_filename is not None)
    tokenizer = Tokenizer(qa_adapted_dataset_paths=[iuxray_qa_adapted_reports_path,
                                                    mimiccxr_qa_adapted_reports_path],                          
                          min_freq=vocab_min_freq,
                          medical_terms_frequency_filename=medical_terms_frequency_filename)
    
    # Create model
    count_print('Creating instance of OpenEndedVQA model ...')    
    model = OpenEndedVQA(vocab_size=tokenizer.vocab_size,
                         start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                         device=device,
                         n_medical_tags=n_medical_tags,
                         n_questions=n_questions,
                         classify_orientation=classify_orientation,
                         classify_chexpert=classify_chexpert,
                         classify_questions=classify_questions,
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
    trainer = get_engine(model, tokenizer, classify_tags, classify_orientation, classify_chexpert,
                         classify_questions, question_encoding, AnswerDecoding.TEACHER_FORCING,
                         device, binary_loss_name=training_kwargs['binary_loss_name'],
                         use_amp=use_amp, training=True,
                         train_with_chexpert_dataset=train_chexpert, optimizer=optimizer)
    validator = get_engine(model, tokenizer, classify_tags, classify_orientation, classify_chexpert,
                         classify_questions, question_encoding, val_answer_decoding,                         
                         device, beam_search_k=beam_search_k, training=False)

    # Define image transform
    count_print('Defining image transform ...')
    img_transform = get_image_transform(augmentation_mode=dataloading_kwargs['img_aug_mode'])

    # Define collate_batch_fn
    if train_mimiccxr:
        mimiccxr_collate_batch_fn = get_vqa_collate_batch_fn(MIMICCXR_DATASET_ID,
                                                        verbose_question = verbose_question,
                                                        classify_tags = classify_tags,
                                                        n_tags = n_medical_tags,
                                                        classify_orientation = classify_orientation,
                                                        classify_chexpert = classify_chexpert,
                                                        classify_questions = classify_questions)
    if train_iuxray:
        iuxray_collate_batch_fn = get_vqa_collate_batch_fn(IUXRAY_DATASET_ID,
                                                    verbose_question = verbose_question,
                                                    classify_tags = classify_tags,
                                                    n_tags = n_medical_tags,
                                                    classify_orientation = classify_orientation,
                                                    classify_chexpert = classify_chexpert,
                                                    classify_questions = classify_questions)

    if train_chexpert:
        chexpert_collate_batch_fn = get_vision_collate_batch_fn(CHEXPERT_DATASET_ID)

    # Create MIMIC-CXR vqa trainer
    if train_mimiccxr:
        count_print('Creating MIMIC-CXR vqa trainer ...')
        mimiccxr_vqa_trainer = MIMICCXR_VQA_Trainer(
            transform = img_transform,
            batch_size = batch_size,
            collate_batch_fn = mimiccxr_collate_batch_fn,
            num_workers = num_workers,
            tokenizer = tokenizer,
            mimiccxr_qa_reports = mimiccxr_qa_reports,
            classify_tags = classify_tags,
            medical_tags_per_report_filename = mimiccxr_medical_tags_per_report_filename,
            classify_orientation = classify_orientation,
            classify_chexpert = classify_chexpert,
            chexpert_labels_filename = mimiccxr_chexpert_labels_filename,
            classify_questions = classify_questions,
            question_labels_filename = mimiccxr_question_labels_filename,
            allowed_questions = allowed_questions,
            verbose_question = verbose_question,
            one_question_per_batch = one_question_per_batch,
            **mimiccxr_vqa_trainer_kwargs,
        )
    
    # Create IU X-Ray vqa trainer
    if train_iuxray:
        count_print('Creating IU X-Ray vqa trainer ...')
        iuxray_vqa_trainer = IUXRAY_VQA_Trainer(
            transform = img_transform,
            batch_size = batch_size,
            collate_batch_fn = iuxray_collate_batch_fn,
            num_workers = num_workers,
            tokenizer = tokenizer,        
            iuxray_qa_reports = iuxray_qa_reports,
            classify_tags = classify_tags,
            medical_tags_per_report_filename = iuxray_medical_tags_per_report_filename,
            classify_orientation = classify_orientation,
            classify_chexpert = classify_chexpert,
            chexpert_labels_filename = iuxray_chexpert_labels_filename,
            classify_questions = classify_questions,
            question_labels_filename = iuxray_question_labels_filename,
            allowed_questions = allowed_questions,
            verbose_question = verbose_question,
            one_question_per_batch = one_question_per_batch,
            **iuxray_vqa_trainer_kwargs,
        )

    if train_chexpert:
        count_print('Creating CheXpert visual module trainer ...')
        chexpert_vision_trainer = Chexpert_VisualModuleTrainer(
            transform=img_transform,
            batch_size=batch_size,
            collate_batch_fn=chexpert_collate_batch_fn,
            num_workers=num_workers,
        )

    if debug: # if debugging
        output = {}
        if train_chexpert: output['chexpert_vision_trainer'] = chexpert_vision_trainer
        if train_mimiccxr: output['mimiccxr_vqa_trainer'] = mimiccxr_vqa_trainer
        if train_iuxray: output['iuxray_vqa_trainer'] = iuxray_vqa_trainer
        return output

    # Create complex dataloaders
    count_print('Creating dataloaders ...')
    if train_mimiccxr and train_iuxray:
        mimiccxr_weight = dataloading_kwargs['mimiccxr_weight']
        iuxray_weight = dataloading_kwargs['iuxray_weight']
        _train_weights = [mimiccxr_weight, iuxray_weight]
        _train_dataloaders = [
            mimiccxr_vqa_trainer.train_dataloader,
            iuxray_vqa_trainer.train_dataloader
        ]
        if train_chexpert:
            _train_weights.append(dataloading_kwargs['chexpert_weight'])
            _train_dataloaders.append(chexpert_vision_trainer.dataloader)
        train_dataloader = balanced_dataloaders_generator(_train_dataloaders, _train_weights)

        val_dataloaders = [mimiccxr_vqa_trainer.val_dataloader, iuxray_vqa_trainer.val_dataloader]
        val_dataloader_size = sum(len(d) for d in val_dataloaders)
        val_dataloader = multi_cyclic_dataloaders_generator(val_dataloaders)
    else:
        if train_mimiccxr:
            train_dataloader = mimiccxr_vqa_trainer.train_dataloader
            train_weight = dataloading_kwargs['mimiccxr_weight']
            val_dataloader = mimiccxr_vqa_trainer.val_dataloader
            val_dataloader_size = len(val_dataloader)
        else:
            assert train_iuxray
            train_dataloader = iuxray_vqa_trainer.train_dataloader
            train_weight = dataloading_kwargs['iuxray_weight']
            val_dataloader = iuxray_vqa_trainer.val_dataloader
            val_dataloader_size = len(val_dataloader)

        if train_chexpert:
            _train_dataloaders = [train_dataloader, chexpert_vision_trainer.dataloader]
            _train_weights = [train_weight, dataloading_kwargs['chexpert_weight']]
            train_dataloader = balanced_dataloaders_generator(_train_dataloaders, _train_weights)

    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')

    if verbose_question:
        attach_dataset_aware_exactmatch_question(trainer, [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID])
        attach_exactmatch_question(validator, device)
    
    attach_dataset_aware_ciderd(trainer, [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID])
    attach_ciderd(validator, device)
    
    attach_dataset_aware_weighted_medical_completeness(trainer, tokenizer, [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID])
    attach_weighted_medical_completeness(validator, device, tokenizer)

    attach_loss('loss', trainer, device)

    if verbose_question:
        attach_dataset_aware_loss(trainer, MetricNames.QUESTION_LOSS, [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID])

    attach_dataset_aware_loss(trainer, MetricNames.ANSWER_LOSS, [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID])
    
    if classify_tags:
        attach_medical_tags_f1score(trainer, device)
        attach_medical_tags_f1score(validator, device)
        attach_loss(MetricNames.MEDTAGS_LOSS, trainer, device)        

    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(trainer)
        attach_dataset_aware_orientation_accuracy(validator)
        attach_loss(MetricNames.ORIENTATION_LOSS, trainer, device)

    if classify_chexpert:
        attach_chexpert_labels_accuracy(trainer, device)
        attach_chexpert_labels_accuracy(validator, device)
        attach_chexpert_labels_macroavgf1(trainer, device)
        attach_chexpert_labels_macroavgf1(validator, device)
        attach_chexpert_labels_microavgf1(trainer, device)
        attach_chexpert_labels_microavgf1(validator, device)
        attach_chexpert_labels_roc_auc(trainer, 'cpu')
        attach_chexpert_labels_roc_auc(validator, 'cpu')
        attach_loss(MetricNames.CHEXPERT_LOSS, trainer, device)

    if classify_questions:
        attach_dataset_aware_question_labels_f1score(trainer, [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID])
        attach_dataset_aware_loss(trainer, MetricNames.QLABELS_LOSS, [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID])
        attach_question_labels_f1score(validator, device)

    if train_chexpert:
        attach_dataset_aware_gender_accuracy(trainer, [CHEXPERT_DATASET_ID])
        attach_dataset_aware_loss(trainer, MetricNames.GENDER_LOSS, [CHEXPERT_DATASET_ID])

    # Timer
    timer = Timer()
    timer.attach(trainer, start=Events.EPOCH_STARTED)
    timer.attach(validator, start=Events.EPOCH_STARTED)
    
    # Learning rate scheduler
    metrics_to_merge = [MetricNames.CIDER_D, MetricNames.WMEDCOMP]
    if verbose_question:
        metrics_to_merge.append(MetricNames.EXACTMATCH_QUESTION)
    if classify_tags:
        metrics_to_merge.append(MetricNames.MEDTAGF1)
    if classify_orientation:
        metrics_to_merge.append(MetricNames.ORIENACC )
    if classify_chexpert:
        metrics_to_merge.append(MetricNames.CHXLABELMICROAVGF1)
        metrics_to_merge.append(MetricNames.CHXLABELMACROAVGF1)
    if classify_questions:
        metrics_to_merge.append(MetricNames.QLABELSF1)
    if train_chexpert:
        metrics_to_merge.append(MetricNames.GENDER_ACC)
    
    merge_metrics_fn = get_merge_metrics_fn(metrics_to_merge, _METRIC_WEIGHTS, 0.3, 0.7)

    lr_sch_handler = get_lr_sch_handler(trainer, validator, lr_scheduler, merge_metrics_fn)

    # Checkpoint saving    
    model_wrapper = ModelWrapper(model, optimizer, lr_scheduler)

    pretrained_checkpoint_folder_path = model_kwargs.get('pretrained_checkpoint_folder_path', None)
    
    if checkpoint_folder_path is None: # first time
        if save: # only if we want to save checkpoints to disk
            _dataset_names = []
            _dataset_weights = []
            if train_mimiccxr:
                _dataset_names.append('mimiccxr')
                _dataset_weights.append(dataloading_kwargs['mimiccxr_weight'])
            if train_iuxray:
                _dataset_names.append('iuxray')
                _dataset_weights.append(dataloading_kwargs['iuxray_weight'])
            if train_chexpert:
                _dataset_names.append('chexpert')
                _dataset_weights.append(dataloading_kwargs['chexpert_weight'])
            folder = '+'.join(_dataset_names)            
            model_args = [
                'densenet-121',
                model_kwargs["embed_size"],
                model_kwargs["question_hidden_size"] if verbose_question else None,
                model_kwargs["answer_hidden_size"],
                model_kwargs["n_lstm_layers"],
                model_kwargs["question_vec_size"],
                model_kwargs["image_local_feat_size"],
                model_kwargs["dropout_prob"],
                f'qenc={question_encoding}',
            ]
            if pretrained_checkpoint_folder_path is not None:
                model_args.append('pretrained')
            if freeze_cnn:
                model_args.append('frozen')
            model_string = ','.join(map(str, model_args))
            checkpoint_folder_path = get_checkpoint_folder_path('vqa', folder, model.name,
                f'voc-minf={vocab_min_freq}',
                f'model-args=({model_string})',
                f'cnn-pretr={int(bool(model_kwargs["densenet_pretrained_weights_path"]))}',
                f'dataset_weights={",".join(map(str, _dataset_weights))}' \
                    if len(_dataset_weights) > 1 else None,
                f'medtok={int(medical_tokenization)}',
                f'tags={int(classify_tags)}',
                f'orien={int(classify_orientation)}',
                f'chx={int(classify_chexpert)}',
                f'ql={int(classify_questions)}',
                'use_amp' if use_amp else None,
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
                        training_kwargs = training_kwargs,
                        auxiliary_tasks_kwargs = auxiliary_tasks_kwargs)

        if pretrained_checkpoint_folder_path is not None:
            pretrained_checkpoint_path = get_checkpoint_filepath(pretrained_checkpoint_folder_path)
            count_print(f'Loading pretrained weights from {pretrained_checkpoint_path} ...')
            checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
            model_wrapper.model.load_state_dict(checkpoint['model'], strict=False)
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
                                                    score_name=get_hybrid_score_name(metrics_to_merge),
                                                    score_fn=score_fn)

    # Logging
    metrics_to_print=['loss', MetricNames.ANSWER_LOSS, MetricNames.CIDER_D, MetricNames.WMEDCOMP]
    if verbose_question:
        metrics_to_print.append(MetricNames.QUESTION_LOSS)
        metrics_to_print.append(MetricNames.EXACTMATCH_QUESTION)
    if classify_tags:
        metrics_to_print.append(MetricNames.MEDTAGS_LOSS)
        metrics_to_print.append(MetricNames.MEDTAGF1)
    if classify_orientation:
        metrics_to_print.append(MetricNames.ORIENTATION_LOSS)
        metrics_to_print.append(MetricNames.ORIENACC)
    if classify_chexpert:
        metrics_to_print.append(MetricNames.CHEXPERT_LOSS)
        metrics_to_print.append(MetricNames.CHXLABELMICROAVGF1)
        metrics_to_print.append(MetricNames.CHXLABELMACROAVGF1)
        metrics_to_print.append(MetricNames.CHXLABELACC)
        metrics_to_print.append(MetricNames.CHXLABEL_ROCAUC)
    if classify_questions:
        metrics_to_print.append(MetricNames.QLABELS_LOSS)
        metrics_to_print.append(MetricNames.QLABELSF1)
    if train_chexpert:
        metrics_to_print.append(MetricNames.GENDER_LOSS)
        metrics_to_print.append(MetricNames.GENDER_ACC)

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
    medical_tokenization,
    medical_terms_frequency_filename,
    # Model's args
    freeze_cnn,
    embed_size,
    question_encoding,
    question_hidden_size,
    answer_hidden_size,
    n_lstm_layers,
    question_vec_size,
    image_local_feat_size,
    dropout_prob,
    densenet_pretrained_weights_path,
    pretrained_checkpoint_folder_path,
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
    n_positive_per_chexpert_label,
    allowed_questions,
    # Dataloading args
    batch_size,
    num_workers,
    mimiccxr_weight,
    iuxray_weight,
    chexpert_weight,
    img_aug_mode,
    balanced_dataloading,
    # Fixed traning args
    train_mimiccxr,
    train_iuxray,
    train_chexpert,
    binary_loss_name,
    use_amp,
    # Variable traning args
    epochs,
    batches_per_epoch,
    one_question_per_batch,
    val_answer_decoding,
    beam_search_k,
    # Auxiliary tasks args
    classify_tags,
    n_medical_tags,
    iuxray_medical_tags_per_report_filename,
    mimiccxr_medical_tags_per_report_filename,
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

    tokenizer_kwargs = dict(
        vocab_min_freq = vocab_min_freq,
        medical_tokenization = medical_tokenization,
        medical_terms_frequency_filename = medical_terms_frequency_filename,
    )
    model_kwargs = dict(
        freeze_cnn = freeze_cnn,
        embed_size = embed_size,
        question_encoding = question_encoding,
        question_hidden_size = question_hidden_size,
        answer_hidden_size = answer_hidden_size,
        n_lstm_layers = n_lstm_layers,
        question_vec_size = question_vec_size,
        image_local_feat_size = image_local_feat_size,
        dropout_prob = dropout_prob,
        densenet_pretrained_weights_path = densenet_pretrained_weights_path,
        pretrained_checkpoint_folder_path = os.path.join(WORKSPACE_DIR, pretrained_checkpoint_folder_path) \
            if pretrained_checkpoint_folder_path is not None else None,
        use_chexpert_forward = train_chexpert,
    )
    optimizer_kwargs = dict(
        lr = lr,
    )
    lr_scheduler_kwargs = dict(
        factor = lr_decay,
        patience = lr_decay_patience,
    )
    if balanced_dataloading:
        assert balanced_split
    if balanced_split:
        split_kwargs = dict(
            n_healthy_per_question = n_healthy_per_question,
            n_unhealthy_per_question = n_unhealthy_per_question,
            min_question_count = min_question_count,
            n_positive_per_chexpert_label = n_positive_per_chexpert_label,
        )
        assert mimiccxr_balanced_metadata_filename is not None
        assert iuxray_balanced_metadata_filename is not None
        assert mimiccxr_chexpert_labels_filename is not None
        assert iuxray_chexpert_labels_filename is not None
    else:
        split_kwargs = dict(
            n_val_examples_per_question = n_val_examples_per_question,
            min_train_examples_per_question = min_train_examples_per_question,
        )
    mimiccxr_vqa_trainer_kwargs = dict(
        split_kwargs = split_kwargs,
        qa_adapted_reports_filename = mimiccxr_qa_adapted_reports_filename,
        balanced_split = balanced_split,
        balanced_dataloading = balanced_dataloading,
        balanced_metadata_filename = mimiccxr_balanced_metadata_filename,
    )
    iuxray_vqa_trainer_kwargs = dict(
        split_kwargs = split_kwargs,
        qa_adapted_reports_filename = iuxray_qa_adapted_reports_filename,
        balanced_split = balanced_split,
        balanced_dataloading = balanced_dataloading,
        balanced_metadata_filename = iuxray_balanced_metadata_filename,
    )
    dataloading_kwargs = dict(
        img_aug_mode = img_aug_mode,
        mimiccxr_weight = mimiccxr_weight,
        iuxray_weight = iuxray_weight,
        chexpert_weight = chexpert_weight,
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
        iuxray_medical_tags_per_report_filename = iuxray_medical_tags_per_report_filename,
        mimiccxr_medical_tags_per_report_filename = mimiccxr_medical_tags_per_report_filename,
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

    return train_model(tokenizer_kwargs,
                model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_vqa_trainer_kwargs,
                iuxray_vqa_trainer_kwargs,
                dataloading_kwargs,
                training_kwargs,
                auxiliary_tasks_kwargs,
                val_answer_decoding,
                epochs,
                batch_size,
                batches_per_epoch,
                one_question_per_batch,
                num_workers,
                beam_search_k=beam_search_k,
                allowed_questions=allowed_questions,
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
    val_answer_decoding,
    epochs = 1,
    batches_per_epoch = 1000,
    allowed_questions = None,
    device = 'GPU',
    save = True,
    override_lr = False,    
    debug = False,
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

    return train_model(tokenizer_kwargs,
                model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_vqa_trainer_kwargs,
                iuxray_vqa_trainer_kwargs,
                dataloading_kwargs,
                training_kwargs,
                auxiliary_tasks_kwargs,
                val_answer_decoding,
                epochs,
                batch_size,
                batches_per_epoch,
                one_question_per_batch,
                num_workers,
                allowed_questions = allowed_questions,
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