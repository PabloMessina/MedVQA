import  os
import argparse

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_postprocessed_label_names as load_chest_imagenome_postprocessed_label_names,
)
from medvqa.datasets.cxr14.cxr14_dataset_management import CXR14_VQA_Trainer
from medvqa.datasets.chexpert.chexpert_dataset_management import (
    Chexpert_VQA_Trainer,
    Chexpert_VisualModuleTrainer,
)
from medvqa.datasets.padchest.padchest_dataset_management import PadChest_VQA_Trainer, PadChestVocabGenerator
from medvqa.datasets.utils import get_merged_findings
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBig_VQA_Trainer
from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.common import AnswerDecoding, load_model_state_dict

from medvqa.models.vqa.open_ended_vqa import (
    QuestionEncoding,
    RawImageEncoding,
    does_include_image,
    does_include_visual_features,
)
from medvqa.training.utils import append_metric_name
from medvqa.utils.constants import (
    CXR14_DATASET_ID,
    CXR14_LABELS,
    CHEXPERT_DATASET_ID,
    CHEXPERT_LABELS,
    CHEXPERT_TASKS,
    IUXRAY_DATASET_ID,
    IUXRAY_DATASET_ID__CHEXPERT_MODE,
    MIMICCXR_DATASET_ID,
    MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE,
    MIMICCXR_DATASET_ID__CHEXPERT_MODE,
    VINBIG_DATASET_ID,
    VINBIG_DISEASES,
    PADCHEST_DATASET_ID,
    PADCHEST_NUM_QUESTIONS,
    MetricNames,
)
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_dataset_aware_chest_imagenome_bbox_mae,
    attach_dataset_aware_chest_imagenome_bbox_iou,
    attach_dataset_aware_chest_imagenome_labels_accuracy,
    attach_dataset_aware_chest_imagenome_labels_macroavgf1,
    attach_dataset_aware_chest_imagenome_labels_microavgf1,
    attach_dataset_aware_chest_imagenome_labels_roc_auc,
    attach_dataset_aware_ciderd,
    attach_dataset_aware_exactmatch_question,
    attach_dataset_aware_exactmatch_answer,
    attach_dataset_aware_vinbig_labels_macroavgf1,
    attach_dataset_aware_vinbig_labels_microavgf1,
    attach_dataset_aware_cxr14_labels_macroavgf1,
    attach_dataset_aware_cxr14_labels_microavgf1,
    attach_dataset_aware_padchest_labels_macroavgf1,
    attach_dataset_aware_padchest_labels_microavgf1,
    attach_dataset_aware_padchest_localization_macroavgf1,
    attach_dataset_aware_padchest_localization_microavgf1,
    attach_exactmatch_question,
    attach_dataset_aware_weighted_medical_completeness,
    attach_medical_tags_f1score,
    attach_dataset_aware_chexpert_labels_accuracy,
    attach_dataset_aware_chexpert_labels_macroavgf1,
    attach_dataset_aware_chexpert_labels_microavgf1,
    attach_dataset_aware_chexpert_labels_roc_auc,
    attach_dataset_aware_orientation_accuracy,
    attach_dataset_aware_question_labels_macroavgf1,
    attach_dataset_aware_question_labels_microavgf1,
    attach_dataset_aware_gender_accuracy,
    attach_dataset_aware_loss,
    attach_dataset_aware_bleu,
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
from medvqa.utils.logging import CountPrinter, print_blue, print_red

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # --- Required arguments

    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs the model will be trained')
    parser.add_argument('--batches-per-epoch', type=int, required=True,
                        help='Number of batches per epoch')

    # --- Optional arguments

    parser.add_argument('--checkpoint-folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')
    parser.add_argument('--iuxray-qa-adapted-reports-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-qa-adapted-reports-filename', type=str, default=None)
    parser.add_argument('--vocab-min-freq', type=int, default=10,
                        help='Min frequency of tokens in vocabulary')
    parser.add_argument('--embed-size', type=int, default=256,
                        help='Size of word embeddings')
    parser.add_argument('--question-encoding', type=str, default='bilstm',
                        help='Method used to encode the question (bilstm, one-hot)')
    parser.add_argument('--answer-decoding', type=str, default='lstm',
                        help='Method used to decode the answer (lstm, transformer)')
    parser.add_argument('--question-hidden-size', type=int, default=128,
                        help='Size of question hidden state vectors')
    parser.add_argument('--answer-hidden-size', type=int, default=256,
                        help='Size of answer hidden state vectors')

    # Image encoder
    parser.add_argument('--visual-input-mode', type=str, default='raw-image')
    parser.add_argument('--raw-image-encoding', type=str, default=RawImageEncoding.DENSENET_121)
    parser.add_argument('--image-local-feat-size', type=int, default=1024,
                        help='Size of local feature vectors from the CNN. They must match the actual vectors output by the CNN')
    parser.add_argument('--image-encoder-pretrained-weights-path', type=str, default=None)
    parser.add_argument('--freeze-image-encoder', dest='freeze_image_encoder', action='store_true')
    parser.set_defaults(freeze_image_encoder=False)
    parser.add_argument('--imagenet-pretrained', dest='imagenet_pretrained', action='store_true')
    parser.set_defaults(imagenet_pretrained=False)
    parser.add_argument('--visual-features-mlp-in-dim', type=int, default=None)
    parser.add_argument('--visual-features-mlp-out-dim', type=int, default=None)
    parser.add_argument('--visual-features-mlp-hidden-dims', nargs='+', type=int, default=None)
    parser.add_argument('--iuxray-precomputed-visual-features-path', type=str, default=None)
    parser.add_argument('--mimiccxr-precomputed-visual-features-path', type=str, default=None)
    parser.add_argument('--chexpert-precomputed-visual-features-path', type=str, default=None)
    parser.add_argument('--vinbig-precomputed-visual-features-path', type=str, default=None)
    parser.add_argument('--clip-version', type=str, default=None)
    parser.add_argument('--huggingface-model-name', type=str, default=None)
    parser.add_argument('--chest-imagenome-bbox-hidden-size', type=int, default=128)
    
    # LSTM decoder
    parser.add_argument('--n-lstm-layers', type=int, default=1,
                        help='Number of LSTM layers to use in the answer decoder')
    # Transformer decoder
    parser.add_argument('--transf-dec-nhead', type=int, default=2)
    parser.add_argument('--transf-dec-dim-forward', type=int, default=256)
    parser.add_argument('--transf-dec-num-layers', type=int, default=2)
    
    parser.add_argument('--question-vec-size', type=int, default=128,
                        help='Size of vector that encodes the question')
    parser.add_argument('--dropout-prob', type=int, default=0,
                        help='Dropout probability')
    parser.add_argument('--optimizer-name', type=str, default='adam')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='reduce-lr-on-plateau')
    parser.add_argument('--lr-decay', type=float, default=0.76, help='Learning rate decay')
    parser.add_argument('--lr-decay-patience', type=int, default=2, help='Learning rate decay patience')
    parser.add_argument('--warmup-and-decay-args', type=str, default=None)
    parser.add_argument('--warmup-and-cosine-args', type=str, default=None)
    
    parser.add_argument('--batch-size', type=int, default=45,
                        help='Batch size')
    parser.add_argument('--iters-to-accumulate', type=int, default=1, help='For gradient accumulation')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for parallel dataloading')    
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')    
    parser.add_argument('--img-aug-mode', type=str, default=None,
                        help='Mode of data augmentation used for images')
    parser.add_argument('--image-size', nargs='+', type=int, default=(256,256))

    # Weights for the different datasets. Used for training with multiple datasets
    parser.add_argument('--mimiccxr-weight', type=float, default=1)
    parser.add_argument('--chexpert-weight', type=float, default=0.3)
    parser.add_argument('--cxr14-weight', type=float, default=0.3)
    parser.add_argument('--vinbig-weight', type=float, default=0.3)
    parser.add_argument('--iuxray-weight', type=float, default=0.05)
    parser.add_argument('--mimiccxr-weight-chexpert-mode', type=float, default=0.2)
    parser.add_argument('--iuxray-weight-chexpert-mode', type=float, default=0.05)
    parser.add_argument('--padchest-weight', type=float, default=0.4)
    parser.add_argument('--mimiccxr-weight-chest-imagenome-mode', type=float, default=1)

    parser.add_argument('--mimiccxr-include-chexpert-mode', dest='mimiccxr_include_chexpert_mode', action='store_true')
    parser.set_defaults(mimiccxr_include_chexpert_mode=False)
    parser.add_argument('--mimiccxr-include-mined-questions-mode', dest='mimiccxr_include_mined_questions_mode', action='store_true')
    parser.set_defaults(mimiccxr_include_mined_questions_mode=False)

    # mimiccxr view mode
    parser.add_argument('--mimiccxr-view-mode', type=str, default='any_single')
    
    # Special mode for MIMIC-CXR where we use additional annotations provided by the Chest-ImaGenome project
    parser.add_argument('--mimiccxr-include-chest-imagenome-mode', dest='mimiccxr_include_chest_imagenome_mode', action='store_true')
    parser.set_defaults(mimiccxr_include_chest_imagenome_mode=False)
    parser.add_argument('--chest-imagenome-labels-filename', type=str, default=None)
    parser.add_argument('--chest-imagenome-label-names-filename', type=str, default=None)

    parser.add_argument('--iuxray-include-chexpert-mode', dest='iuxray_include_chexpert_mode', action='store_true')
    parser.set_defaults(iuxray_include_chexpert_mode=False)
    parser.add_argument('--iuxray-include-mined-questions-mode', dest='iuxray_include_mined_questions_mode', action='store_true')
    parser.set_defaults(iuxray_include_mined_questions_mode=False)

    parser.add_argument('--val-answer-decoding', type=str, default='greedy-search')
    parser.add_argument('--beam-search-k', type=int, default=None)

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)
    
    parser.add_argument('--use-medical-tokenization', dest='use_medical_tokenization', action='store_true')
    parser.set_defaults(use_medical_tokenization=False)

    parser.add_argument('--medical-terms-frequency-filename', type=str, default=None)

    parser.add_argument('--allowed-questions', type=str, default=None)

    parser.add_argument('--pretrained-checkpoint-folder-path', type=str, default=None)

    # balanced dataset arguments
    parser.add_argument('--balanced-dataloading', dest='balanced_dataloading', action='store_true')
    parser.set_defaults(balanced_dataloading=False)
    parser.add_argument('--imbalance-reduction-coef', type=float, default=0.5)
    parser.add_argument('--iuxray-balanced-metadata-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-balanced-metadata-filename', type=str, default=None)

    parser.add_argument('--one-question-per-batch', dest='one_question_per_batch', action='store_true')
    parser.set_defaults(one_question_per_batch=False)

    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=True)

    parser.add_argument('--override-lr', dest='override_lr', action='store_true')
    parser.set_defaults(override_lr=False)

    parser.add_argument('--use-mimiccxr', dest='train_mimiccxr', action='store_true')
    parser.set_defaults(train_mimiccxr=False)

    parser.add_argument('--use-iuxray', dest='train_iuxray', action='store_true')
    parser.set_defaults(train_iuxray=False)

    parser.add_argument('--use-chexpert', dest='train_chexpert', action='store_true')
    parser.set_defaults(train_chexpert=False)
    parser.add_argument('--chexpert-mode', type=str, default=None)

    parser.add_argument('--use-cxr14', dest='train_cxr14', action='store_true')
    parser.set_defaults(train_cxr14=False)
    
    # VinBigData arguments
    parser.add_argument('--use-vinbig', dest='train_vinbig', action='store_true')
    parser.set_defaults(train_vinbig=False)
    parser.add_argument('--vinbig-training-data', type=str, default='all')
    parser.add_argument('--vinbig-use-validation', dest='vinbig_use_validation', action='store_true')
    parser.set_defaults(vinbig_use_validation=False)

    # PadChest arguments
    parser.add_argument('--use-padchest', dest='train_padchest', action='store_true')
    parser.set_defaults(train_padchest=False)
    parser.add_argument('--padchest-training-data-mode', type=str, default='train')
    parser.add_argument('--padchest-use-validation', dest='padchest_use_validation', action='store_true')
    parser.set_defaults(padchest_use_validation=False)
    parser.add_argument('--padchest-train-study-ids-path', type=str, default=None)
    parser.add_argument('--padchest-val-study-ids-path', type=str, default=None)
    parser.add_argument('--padchest-test-study-ids-path', type=str, default=None)

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
    # chest imagenome labels
    parser.add_argument('--classify-chest-imagenome', dest='classify_chest_imagenome', action='store_true')
    parser.set_defaults(classify_chest_imagenome=False)
    parser.add_argument('--predict-bboxes-chest-imagenome', dest='predict_bboxes_chest_imagenome', action='store_true')
    parser.set_defaults(predict_bboxes_chest_imagenome=False)
    # question classification
    parser.add_argument('--classify-questions', dest='classify_questions', action='store_true')
    parser.set_defaults(classify_questions=False)
    parser.add_argument('--n-mined-questions', type=int, default=None)
    parser.add_argument('--iuxray-question-labels-filename', type=str, default=None)
    parser.add_argument('--mimiccxr-question-labels-filename', type=str, default=None)

    parser.add_argument('--merge-findings', dest='merge_findings', action='store_true')
    parser.set_defaults(merge_findings=False)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = {
    MetricNames.EXACTMATCH_QUESTION: 1,
    MetricNames.EXACTMATCH_ANSWER: 2,
    MetricNames.CIDER_D: 0.1,
    MetricNames.WMEDCOMP: 1,
    MetricNames.BLEU: 1,
    MetricNames.MEDTAGF1: 1,
    MetricNames.ORIENACC: 1,
    MetricNames.CHXLABELMICROAVGF1: 1,
    MetricNames.CHXLABELMACROAVGF1: 1,
    MetricNames.CXR14MACROAVGF1: 0.5,
    MetricNames.CXR14MICROAVGF1: 0.5,
    MetricNames.QLABELS_MICROAVGF1: 0.5,
    MetricNames.QLABELS_MACROAVGF1: 0.5,
    MetricNames.VINBIGMICROAVGF1: 0.5,
    MetricNames.VINBIGMACROAVGF1: 0.5,
    MetricNames.GENDER_ACC: 1,
    MetricNames.PADCHEST_LABEL_MACROAVGF1: 0.5,
    MetricNames.PADCHEST_LABEL_MICROAVGF1: 0.5,
    MetricNames.PADCHEST_LOC_MACROAVGF1: 0.5,
    MetricNames.PADCHEST_LOC_MICROAVGF1: 0.5,
    MetricNames.CHESTIMAGENOMELABELMACROAVGF1: 1,
    MetricNames.CHESTIMAGENOMELABELMICROAVGF1: 1,
    MetricNames.CHESTIMAGENOMEBBOXIOU: 1,
}

def _metric_getter(metrics_dict, key):
    if key == MetricNames.BLEU:
        scores = metrics_dict[key]
        assert len(scores) == 4
        return sum(scores) / len(scores)
    return metrics_dict[key]

def train_model(
    tokenizer_kwargs,
    model_kwargs,
    optimizer_kwargs,
    lr_scheduler_kwargs,
    mimiccxr_vqa_trainer_kwargs,
    iuxray_vqa_trainer_kwargs,
    chexpert_dataset_kwargs,
    cxr14_vqa_trainer_kwargs,
    vinbig_dataset_kwargs,
    padchest_dataset_kwargs,
    dataloading_kwargs,
    train_image_transform_kwargs,
    val_image_transform_kwargs,
    training_kwargs,
    trainer_engine_kwargs,
    val_engine_kwargs,
    auxiliary_tasks_kwargs,
    val_answer_decoding,
    epochs,
    batches_per_epoch,
    one_question_per_batch,
    num_workers,
    beam_search_k = None,
    device = 'GPU',
    checkpoint_folder_path = None,
    save = True,
    override_lr = False,
    debug = False,
):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    batch_size = dataloading_kwargs['batch_size']
    train_iuxray = training_kwargs['train_iuxray']
    train_mimiccxr = training_kwargs['train_mimiccxr']
    train_chexpert = training_kwargs['train_chexpert']
    train_cxr14 = training_kwargs['train_cxr14']
    train_vinbig = training_kwargs['train_vinbig']
    train_padchest = training_kwargs['train_padchest']
    chexpert_mode = training_kwargs['chexpert_mode']    
    use_merged_findings = trainer_engine_kwargs.get('use_merged_findings', False)

    use_amp = training_kwargs['use_amp']
    visual_input_mode = model_kwargs['visual_input_mode']
    question_encoding = model_kwargs.get('question_encoding', QuestionEncoding.BILSTM)
    verbose_question = question_encoding != QuestionEncoding.ONE_HOT
    include_image = does_include_image(visual_input_mode)
    include_visual_features = does_include_visual_features(visual_input_mode)

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
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs['classify_chest_imagenome']
    predict_bboxes_chest_imagenome = auxiliary_tasks_kwargs['predict_bboxes_chest_imagenome']
    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)
    n_questions_aux_task = auxiliary_tasks_kwargs.get('n_questions_aux_task', None)
    iuxray_question_labels_filename = auxiliary_tasks_kwargs.get('iuxray_question_labels_filename', None)
    mimiccxr_question_labels_filename = auxiliary_tasks_kwargs.get('mimiccxr_question_labels_filename', None)
    if classify_questions:
        assert n_questions_aux_task is not None
        if train_iuxray: assert iuxray_question_labels_filename is not None
        if train_mimiccxr: assert mimiccxr_question_labels_filename is not None

    if question_encoding == QuestionEncoding.ONE_HOT:
        assert model_kwargs['n_questions'] is not None

    if train_chexpert:
        assert classify_chexpert
        assert chexpert_mode is not None

    # Beam search
    if val_answer_decoding == AnswerDecoding.BEAM_SEARCH:
        assert beam_search_k is not None

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)
    
    # Load QA adapted reports
    if train_iuxray:
        count_print('Loading iuxray QA adapted reports ...')
        iuxray_qa_adapted_reports_filename = iuxray_vqa_trainer_kwargs['qa_adapted_reports_filename']
        assert iuxray_qa_adapted_reports_filename is not None
        iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, iuxray_qa_adapted_reports_filename)        
        iuxray_qa_reports = load_json_file(iuxray_qa_adapted_reports_path)
    if train_mimiccxr:
        count_print('Loading mimiccxr QA adapted reports ...')
        mimiccxr_qa_adapted_reports_filename = mimiccxr_vqa_trainer_kwargs['qa_adapted_reports_filename']
        assert mimiccxr_qa_adapted_reports_filename is not None        
        mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, mimiccxr_qa_adapted_reports_filename)        
        mimiccxr_qa_reports = load_json_file(mimiccxr_qa_adapted_reports_path)
    if train_iuxray and train_mimiccxr:
        assert iuxray_qa_reports['questions'] == mimiccxr_qa_reports['questions']
    
    # Init tokenizer
    count_print('Initializing tokenizer ...')
    vocab_min_freq = tokenizer_kwargs['vocab_min_freq']
    use_medical_tokenization = tokenizer_kwargs['use_medical_tokenization']
    medical_terms_frequency_filename = tokenizer_kwargs['medical_terms_frequency_filename']
    assert use_medical_tokenization == (medical_terms_frequency_filename is not None)
    if train_iuxray or train_mimiccxr:
        qa_adapted_dataset_paths = []
        if train_iuxray: qa_adapted_dataset_paths.append(iuxray_qa_adapted_reports_path)
        if train_mimiccxr: qa_adapted_dataset_paths.append(mimiccxr_qa_adapted_reports_path)
    else:
        qa_adapted_dataset_paths = None
    other_vocab_generators = []
    other_vocab_generators_names = []
    if train_padchest:
        other_vocab_generators.append(PadChestVocabGenerator())
        other_vocab_generators_names.append(other_vocab_generators[-1].name)
    tokenizer = Tokenizer(qa_adapted_dataset_paths=qa_adapted_dataset_paths, 
                        vocab_min_freq=vocab_min_freq,
                        medical_terms_frequency_filename=medical_terms_frequency_filename,
                        other_vocab_generators=other_vocab_generators,
                        other_vocab_generators_names=other_vocab_generators_names)
    # Rememeber vocab filepath in case we need to reload tokenizer
    tokenizer_kwargs['vocab_filepath'] = tokenizer.vocab_filepath
                        
    # Create model
    count_print('Creating instance of OpenEndedVQA model ...')
    model = OpenEndedVQA(vocab_size=tokenizer.vocab_size,
                         start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                         device=device, **model_kwargs)
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
    trainer = get_engine(model=model, tokenizer=tokenizer, optimizer=optimizer, device=device,
                         update_lr_batchwise=update_lr_batchwise, lr_scheduler=lr_scheduler, **trainer_engine_kwargs)
    validator = get_engine(model=model, tokenizer=tokenizer, device=device, **val_engine_kwargs)

    # Define image transform
    count_print('Defining image transform ...')
    print('train_image_transform_kwargs:', train_image_transform_kwargs)
    train_img_transform = get_image_transform(**train_image_transform_kwargs)
    print('val_image_transform_kwargs:', val_image_transform_kwargs)
    val_img_transform = get_image_transform(**val_image_transform_kwargs)
    
    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')

    one_hot_question_offsets = dataloading_kwargs.get('one_hot_question_offsets', None)
    if not verbose_question: assert one_hot_question_offsets is not None
    
    _kwargs = dict(verbose_question=verbose_question,
                   include_image=include_image,
                   include_visual_features=include_visual_features,
                   classify_tags=classify_tags,
                   n_tags=n_medical_tags,
                   classify_orientation=classify_orientation,
                   classify_chexpert=classify_chexpert,
                   classify_questions=classify_questions,
                   classify_chest_imagenome=classify_chest_imagenome,
                   predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
                   one_hot_question_offsets=one_hot_question_offsets,
                   use_merged_findings=use_merged_findings)
    if train_mimiccxr:
        mimiccxr_collate_batch_fn = get_vqa_collate_batch_fn(MIMICCXR_DATASET_ID, **_kwargs)
        mimiccxr_chexpert_mode_collate_batch_fn = get_vqa_collate_batch_fn(MIMICCXR_DATASET_ID__CHEXPERT_MODE, **_kwargs)\
                if mimiccxr_vqa_trainer_kwargs['include_chexpert_mode'] else None
        mimiccxr_chest_imagenome_mode_collate_batch_fn = get_vqa_collate_batch_fn(MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE, **_kwargs)\
                if mimiccxr_vqa_trainer_kwargs['include_chest_imagenome_mode'] else None
    if train_iuxray:
        iuxray_collate_batch_fn = get_vqa_collate_batch_fn(IUXRAY_DATASET_ID, **_kwargs)
        iuxray_chexpert_mode_collate_batch_fn = get_vqa_collate_batch_fn(IUXRAY_DATASET_ID__CHEXPERT_MODE, **_kwargs)\
                if iuxray_vqa_trainer_kwargs['include_chexpert_mode'] else None
    if train_chexpert:
        if chexpert_mode == CHEXPERT_TASKS.CLASSIFICATION:
            chexpert_collate_batch_fn = get_vision_collate_batch_fn(CHEXPERT_DATASET_ID)
        else:
            chexpert_collate_batch_fn = get_vqa_collate_batch_fn(CHEXPERT_DATASET_ID, **_kwargs)
    if train_cxr14:
        cxr14_collate_batch_fn = get_vqa_collate_batch_fn(CXR14_DATASET_ID, **_kwargs)
    if train_vinbig:
        vinbig_collate_batch_fn = get_vqa_collate_batch_fn(VINBIG_DATASET_ID, **_kwargs)
    if train_padchest:
        padchest_collate_batch_fn = get_vqa_collate_batch_fn(PADCHEST_DATASET_ID, **_kwargs)

    # Create MIMIC-CXR vqa trainer
    if train_mimiccxr:
        count_print('Creating MIMIC-CXR vqa trainer ...')
        mimiccxr_vqa_trainer = MIMICCXR_VQA_Trainer(
            train_image_transform = train_img_transform,
            val_image_transform = val_img_transform,
            batch_size = batch_size,
            collate_batch_fn = mimiccxr_collate_batch_fn,
            collate_batch_fn_chexpert_mode = mimiccxr_chexpert_mode_collate_batch_fn,
            collate_batch_fn_chest_imagenome_mode = mimiccxr_chest_imagenome_mode_collate_batch_fn,
            num_workers = num_workers,
            tokenizer = tokenizer,
            mimiccxr_qa_reports = mimiccxr_qa_reports,            
            one_question_per_batch = one_question_per_batch,
            **mimiccxr_vqa_trainer_kwargs,
        )
    
    # Create IU X-Ray vqa trainer
    if train_iuxray:
        count_print('Creating IU X-Ray vqa trainer ...')
        iuxray_vqa_trainer = IUXRAY_VQA_Trainer(
            train_image_transform = train_img_transform,
            val_image_transform = val_img_transform,
            batch_size = batch_size,
            collate_batch_fn = iuxray_collate_batch_fn,
            collate_batch_fn_chexpert_mode = iuxray_chexpert_mode_collate_batch_fn,
            num_workers = num_workers,
            tokenizer = tokenizer,        
            iuxray_qa_reports = iuxray_qa_reports,            
            one_question_per_batch = one_question_per_batch,
            **iuxray_vqa_trainer_kwargs,
        )

    # Create CheXpert trainer
    if train_chexpert:
        if chexpert_mode == CHEXPERT_TASKS.CLASSIFICATION:
            count_print('Creating CheXpert visual module trainer ...')
            chexpert_trainer = Chexpert_VisualModuleTrainer(
                train_image_transform = train_img_transform,
                val_image_transform = val_img_transform,
                batch_size=batch_size,
                collate_batch_fn=chexpert_collate_batch_fn,
                num_workers=num_workers,
                **chexpert_dataset_kwargs,
            )
        elif chexpert_mode == CHEXPERT_TASKS.VQA:
            count_print('Creating CheXpert vqa trainer ...')
            chexpert_trainer = Chexpert_VQA_Trainer(
                train_image_transform = train_img_transform,
                val_image_transform = val_img_transform,
                batch_size=batch_size,
                collate_batch_fn=chexpert_collate_batch_fn,
                num_workers=num_workers,
                tokenizer=tokenizer,
                **chexpert_dataset_kwargs,
            )
        else: assert False, f'Unknown chexpert_mode = {chexpert_mode}'

    # Create CXR14 vqa trainer
    if train_cxr14:
        count_print('Creating CXR14 vqa trainer ...')
        cxr14_vqa_trainer = CXR14_VQA_Trainer(
            train_image_transform = train_img_transform,
            val_image_transform = val_img_transform,
            batch_size=batch_size,
            collate_batch_fn=cxr14_collate_batch_fn,
            num_workers=num_workers,
            tokenizer=tokenizer,
            **cxr14_vqa_trainer_kwargs,
        )

    # Create VinBig vqa trainer
    if train_vinbig:
        count_print('Creating VinBig vqa trainer ...')
        vinbig_vqa_trainer = VinBig_VQA_Trainer(
            train_image_transform = train_img_transform,
            val_image_transform = val_img_transform,
            batch_size=batch_size,
            collate_batch_fn=vinbig_collate_batch_fn,
            num_workers=num_workers,
            tokenizer=tokenizer,
            **vinbig_dataset_kwargs,
        )

    # Create PadChest vqa trainer
    if train_padchest:
        count_print('Creating PadChest vqa trainer ...')
        padchest_vqa_trainer = PadChest_VQA_Trainer(
            train_image_transform = train_img_transform,
            val_image_transform = val_img_transform,
            batch_size=batch_size,
            collate_batch_fn=padchest_collate_batch_fn,
            num_workers=num_workers,
            tokenizer=tokenizer,
            **padchest_dataset_kwargs,
        )

    if debug: # if debugging
        output = {}
        if train_mimiccxr: output['mimiccxr_vqa_trainer'] = mimiccxr_vqa_trainer
        if train_iuxray: output['iuxray_vqa_trainer'] = iuxray_vqa_trainer
        if train_chexpert:
            if chexpert_mode == CHEXPERT_TASKS.CLASSIFICATION:
                output['chexpert_vision_trainer'] = chexpert_trainer
            else:
                output['chexpert_vqa_trainer'] = chexpert_trainer
        if train_cxr14: output['cxr14_vqa_trainer'] = cxr14_vqa_trainer
        if train_vinbig: output['vinbig_vqa_trainer'] = vinbig_vqa_trainer
        if train_padchest: output['padchest_vqa_trainer'] = padchest_vqa_trainer
        return output

    # Create complex dataloaders
    count_print('Creating dataloaders ...')
    
    _train_weights = []
    _train_dataloaders = []
    _val_dataloaders = []
    _dataset_names = []

    if train_mimiccxr:
        if mimiccxr_vqa_trainer.include_mined_questions_mode:
            _dataset_names.append('mim')
            _train_weights.append(dataloading_kwargs['mimiccxr_weight'])
            _train_dataloaders.append(mimiccxr_vqa_trainer.train_dataloader)
            _val_dataloaders.append(mimiccxr_vqa_trainer.val_dataloader)
        if mimiccxr_vqa_trainer.include_chexpert_mode:
            _dataset_names.append('mim(chex)')
            _train_weights.append(dataloading_kwargs['mimiccxr_weight_chexpert_mode'])
            _train_dataloaders.append(mimiccxr_vqa_trainer.train_dataloader__chexpert_mode)
            _val_dataloaders.append(mimiccxr_vqa_trainer.val_dataloader__chexpert_mode)
        if mimiccxr_vqa_trainer.include_chest_imagenome_mode:
            _dataset_names.append('mim(chst-imgn)')
            _train_weights.append(dataloading_kwargs['mimiccxr_weight_chest_imagenome_mode'])
            _train_dataloaders.append(mimiccxr_vqa_trainer.train_dataloader__chest_imagenome_mode)
            _val_dataloaders.append(mimiccxr_vqa_trainer.val_dataloader__chest_imagenome_mode)
    if train_iuxray:
        if iuxray_vqa_trainer.include_mined_questions_mode:
            _dataset_names.append('iu')
            _train_weights.append(dataloading_kwargs['iuxray_weight'])
            _train_dataloaders.append(iuxray_vqa_trainer.train_dataloader)            
        if iuxray_vqa_trainer.include_chexpert_mode:
            _dataset_names.append('iu(chex)')
            _train_weights.append(dataloading_kwargs['iuxray_weight_chexpert_mode'])
            _train_dataloaders.append(iuxray_vqa_trainer.train_dataloader__chexpert_mode)            
    if train_chexpert:
        _train_weights.append(dataloading_kwargs['chexpert_weight'])
        _train_dataloaders.append(chexpert_trainer.dataloader)
        if chexpert_mode == CHEXPERT_TASKS.CLASSIFICATION:
            _dataset_names.append('chexp(cl)')
        else:
            _dataset_names.append('chexp(vqa)')
    if train_cxr14:
        _dataset_names.append('cxr14(vqa)')
        _train_weights.append(dataloading_kwargs['cxr14_weight'])
        _train_dataloaders.append(cxr14_vqa_trainer.dataloader)
    if train_vinbig:
        _dataset_names.append('vinbig(vqa)')
        _train_weights.append(dataloading_kwargs['vinbig_weight'])
        _train_dataloaders.append(vinbig_vqa_trainer.train_dataloader)
        if vinbig_vqa_trainer.use_validation:
            _val_dataloaders.append(vinbig_vqa_trainer.val_dataloader)
    if train_padchest:
        _dataset_names.append('padchest(vqa)')
        _train_weights.append(dataloading_kwargs['padchest_weight'])
        _train_dataloaders.append(padchest_vqa_trainer.train_dataloader)
        if padchest_vqa_trainer.use_validation_set:
            _val_dataloaders.append(padchest_vqa_trainer.val_dataloader)
    
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

    _use_exactmatch_answer = (train_mimiccxr and (\
                                mimiccxr_vqa_trainer.include_chexpert_mode or\
                                mimiccxr_vqa_trainer.include_chest_imagenome_mode)) or\
                             (train_iuxray and iuxray_vqa_trainer.include_chexpert_mode) or\
                             (train_chexpert and chexpert_mode == CHEXPERT_TASKS.VQA) or\
                             train_vinbig or train_cxr14 or train_padchest

    _mim_datasets = [MIMICCXR_DATASET_ID, MIMICCXR_DATASET_ID__CHEXPERT_MODE, MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE]
    _iu_mim_datasets = [IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID]
    _iu_mim_datasets_chexp_mode =  [IUXRAY_DATASET_ID__CHEXPERT_MODE, MIMICCXR_DATASET_ID__CHEXPERT_MODE]    
    _vqa_datasets = _iu_mim_datasets + _iu_mim_datasets_chexp_mode +\
        [VINBIG_DATASET_ID,  CXR14_DATASET_ID, PADCHEST_DATASET_ID, MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE]
    _exact_match_answer_datasets = _iu_mim_datasets_chexp_mode +\
        [VINBIG_DATASET_ID, CXR14_DATASET_ID, PADCHEST_DATASET_ID, MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE]
    _orientation_datasets = _iu_mim_datasets + _iu_mim_datasets_chexp_mode +\
        [CHEXPERT_DATASET_ID, CXR14_DATASET_ID, PADCHEST_DATASET_ID, MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE]
    _chexpert_labels_datasets = _iu_mim_datasets + _iu_mim_datasets_chexp_mode +\
        [CHEXPERT_DATASET_ID, MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE]
    _gender_datasets = [CHEXPERT_DATASET_ID, CXR14_DATASET_ID, PADCHEST_DATASET_ID]

    if use_merged_findings:
        _findings_remapper = trainer_engine_kwargs['findings_remapper']
        _chexpert_class_indices = _findings_remapper[CHEXPERT_DATASET_ID]
        _cxr14_class_indices = _findings_remapper[CXR14_DATASET_ID]
        _vinbig_class_indices = _findings_remapper[VINBIG_DATASET_ID]
    else:
        _chexpert_class_indices = _cxr14_class_indices = _vinbig_class_indices = None

    if chexpert_mode == CHEXPERT_TASKS.VQA:
        _vqa_datasets.append(CHEXPERT_DATASET_ID)
        _exact_match_answer_datasets.append(CHEXPERT_DATASET_ID)

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []

    attach_loss('loss', trainer, device)
    attach_dataset_aware_loss(trainer, MetricNames.ANSWER_LOSS, _vqa_datasets)
    # for logging
    metrics_to_print.append('loss')
    metrics_to_print.append(MetricNames.ANSWER_LOSS)

    if verbose_question:
        attach_dataset_aware_exactmatch_question(trainer, _vqa_datasets)
        attach_exactmatch_question(validator, device)
        attach_dataset_aware_loss(trainer, MetricNames.QUESTION_LOSS, _vqa_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.EXACTMATCH_QUESTION)
        metrics_to_print.append(MetricNames.QUESTION_LOSS)

    if (train_mimiccxr and mimiccxr_vqa_trainer.include_mined_questions_mode) or\
         (train_iuxray and iuxray_vqa_trainer.include_mined_questions_mode):
        attach_dataset_aware_ciderd(trainer, _iu_mim_datasets)
        attach_dataset_aware_ciderd(validator, _iu_mim_datasets)
        attach_dataset_aware_weighted_medical_completeness(trainer, tokenizer, _iu_mim_datasets)
        attach_dataset_aware_weighted_medical_completeness(validator, tokenizer, _iu_mim_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CIDER_D)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.WMEDCOMP)

    if _use_exactmatch_answer:
        attach_dataset_aware_exactmatch_answer(trainer, _exact_match_answer_datasets)
        attach_dataset_aware_exactmatch_answer(validator, _exact_match_answer_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.EXACTMATCH_ANSWER)
    
    if classify_tags:
        attach_medical_tags_f1score(trainer, device)
        attach_medical_tags_f1score(validator, device)
        attach_loss(MetricNames.MEDTAGS_LOSS, trainer, device)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.MEDTAGF1)
        metrics_to_print.append(MetricNames.MEDTAGS_LOSS)

    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(trainer, _orientation_datasets)
        attach_dataset_aware_orientation_accuracy(validator, _orientation_datasets)
        attach_dataset_aware_loss(trainer, MetricNames.ORIENTATION_LOSS, _orientation_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.ORIENACC )
        metrics_to_print.append(MetricNames.ORIENTATION_LOSS)

    if classify_questions:
        attach_dataset_aware_question_labels_macroavgf1(trainer, _iu_mim_datasets + _iu_mim_datasets_chexp_mode)
        attach_dataset_aware_question_labels_microavgf1(trainer, _iu_mim_datasets + _iu_mim_datasets_chexp_mode)
        attach_dataset_aware_question_labels_macroavgf1(validator, _iu_mim_datasets + _iu_mim_datasets_chexp_mode)
        attach_dataset_aware_question_labels_microavgf1(validator, _iu_mim_datasets + _iu_mim_datasets_chexp_mode)
        attach_dataset_aware_loss(trainer, MetricNames.QLABELS_LOSS, _iu_mim_datasets + _iu_mim_datasets_chexp_mode)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.QLABELS_MICROAVGF1)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.QLABELS_MACROAVGF1)
        metrics_to_print.append(MetricNames.QLABELS_LOSS)
    
    if classify_chexpert:
        attach_dataset_aware_chexpert_labels_accuracy(trainer, _chexpert_labels_datasets, _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_accuracy(validator, _chexpert_labels_datasets, _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_macroavgf1(trainer, _chexpert_labels_datasets, _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_macroavgf1(validator, _chexpert_labels_datasets, _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_microavgf1(trainer, _chexpert_labels_datasets, _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_microavgf1(validator, _chexpert_labels_datasets, _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_roc_auc(trainer, _chexpert_labels_datasets, 'cpu', _chexpert_class_indices)
        attach_dataset_aware_chexpert_labels_roc_auc(validator, _chexpert_labels_datasets, 'cpu', _chexpert_class_indices)
        attach_dataset_aware_loss(trainer, MetricNames.CHEXPERT_LOSS, _chexpert_labels_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHXLABELMICROAVGF1)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHXLABELMACROAVGF1)
        metrics_to_print.append(MetricNames.CHEXPERT_LOSS)
        metrics_to_print.append(MetricNames.CHXLABELACC)
        metrics_to_print.append(MetricNames.CHXLABEL_ROCAUC)

    if classify_chest_imagenome:
        attach_dataset_aware_chest_imagenome_labels_accuracy(trainer, _mim_datasets)
        attach_dataset_aware_chest_imagenome_labels_accuracy(validator, _mim_datasets)
        attach_dataset_aware_chest_imagenome_labels_macroavgf1(trainer, _mim_datasets)
        attach_dataset_aware_chest_imagenome_labels_macroavgf1(validator, _mim_datasets)
        attach_dataset_aware_chest_imagenome_labels_microavgf1(trainer, _mim_datasets)
        attach_dataset_aware_chest_imagenome_labels_microavgf1(validator, _mim_datasets)
        attach_dataset_aware_chest_imagenome_labels_roc_auc(trainer, _mim_datasets, 'cpu')
        attach_dataset_aware_chest_imagenome_labels_roc_auc(validator, _mim_datasets, 'cpu')
        attach_dataset_aware_loss(trainer, MetricNames.CHEST_IMAGENOME_LABEL_LOSS, _mim_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMELABELMACROAVGF1)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMELABELMICROAVGF1)
        metrics_to_print.append(MetricNames.CHEST_IMAGENOME_LABEL_LOSS)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELACC)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELROCAUC)

    if predict_bboxes_chest_imagenome:
        attach_dataset_aware_chest_imagenome_bbox_mae(trainer, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_mae(validator, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_iou(trainer, _mim_datasets)
        attach_dataset_aware_chest_imagenome_bbox_iou(validator, _mim_datasets)
        # for logging        
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMEBBOXIOU)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXMAE)
        metrics_to_print.append(MetricNames.CHEST_IMAGENOME_BBOX_LOSS)

    if train_chexpert or train_cxr14 or train_padchest:
        attach_dataset_aware_gender_accuracy(trainer, _gender_datasets)
        attach_dataset_aware_loss(trainer, MetricNames.GENDER_LOSS, _gender_datasets)
        if train_padchest and padchest_vqa_trainer.use_validation_set:
            attach_dataset_aware_gender_accuracy(validator, [PADCHEST_DATASET_ID])
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.GENDER_ACC,
                           val=(train_padchest and padchest_vqa_trainer.use_validation_set))
        metrics_to_print.append(MetricNames.GENDER_LOSS)

    if train_cxr14:
        attach_dataset_aware_cxr14_labels_macroavgf1(trainer, [CXR14_DATASET_ID], _cxr14_class_indices)
        attach_dataset_aware_cxr14_labels_microavgf1(trainer, [CXR14_DATASET_ID], _cxr14_class_indices)
        attach_dataset_aware_loss(trainer, MetricNames.CXR14_LOSS, [CXR14_DATASET_ID])
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CXR14MICROAVGF1, val=False)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CXR14MACROAVGF1, val=False)
        metrics_to_print.append(MetricNames.CXR14_LOSS)

    if train_vinbig:
        attach_dataset_aware_vinbig_labels_macroavgf1(trainer, [VINBIG_DATASET_ID], _vinbig_class_indices)
        attach_dataset_aware_vinbig_labels_microavgf1(trainer, [VINBIG_DATASET_ID], _vinbig_class_indices)
        attach_dataset_aware_loss(trainer, MetricNames.VINBIG_LOSS, [VINBIG_DATASET_ID])
        if vinbig_vqa_trainer.use_validation:
            attach_dataset_aware_vinbig_labels_macroavgf1(validator, [VINBIG_DATASET_ID], _vinbig_class_indices)
            attach_dataset_aware_vinbig_labels_microavgf1(validator, [VINBIG_DATASET_ID], _vinbig_class_indices)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print,
                           MetricNames.VINBIGMICROAVGF1, val=vinbig_vqa_trainer.use_validation)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print,
                           MetricNames.VINBIGMACROAVGF1, val=vinbig_vqa_trainer.use_validation)
        metrics_to_print.append(MetricNames.VINBIG_LOSS)
    
    if train_padchest:
        attach_dataset_aware_bleu(trainer, [PADCHEST_DATASET_ID])
        attach_dataset_aware_padchest_labels_macroavgf1(trainer, [PADCHEST_DATASET_ID])
        attach_dataset_aware_padchest_labels_microavgf1(trainer, [PADCHEST_DATASET_ID])
        attach_dataset_aware_loss(trainer, MetricNames.PADCHEST_LABEL_LOSS, [PADCHEST_DATASET_ID])
        attach_dataset_aware_padchest_localization_macroavgf1(trainer, [PADCHEST_DATASET_ID])
        attach_dataset_aware_padchest_localization_microavgf1(trainer, [PADCHEST_DATASET_ID])
        attach_dataset_aware_loss(trainer, MetricNames.PADCHEST_LOCALIZATION_LOSS, [PADCHEST_DATASET_ID])
        if padchest_vqa_trainer.use_validation_set:
            attach_dataset_aware_bleu(validator, [PADCHEST_DATASET_ID])
            attach_dataset_aware_padchest_labels_macroavgf1(validator, [PADCHEST_DATASET_ID])
            attach_dataset_aware_padchest_labels_microavgf1(validator, [PADCHEST_DATASET_ID])
            attach_dataset_aware_padchest_localization_macroavgf1(validator, [PADCHEST_DATASET_ID])
            attach_dataset_aware_padchest_localization_microavgf1(validator, [PADCHEST_DATASET_ID])
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print,
                            MetricNames.BLEU, val=padchest_vqa_trainer.use_validation_set)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print,
                           MetricNames.PADCHEST_LABEL_MACROAVGF1, val=padchest_vqa_trainer.use_validation_set)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print,
                           MetricNames.PADCHEST_LABEL_MICROAVGF1, val=padchest_vqa_trainer.use_validation_set)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print,
                           MetricNames.PADCHEST_LOC_MACROAVGF1, val=padchest_vqa_trainer.use_validation_set)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print,
                            MetricNames.PADCHEST_LOC_MICROAVGF1, val=padchest_vqa_trainer.use_validation_set)
        metrics_to_print.append(MetricNames.PADCHEST_LABEL_LOSS)
        metrics_to_print.append(MetricNames.PADCHEST_LOCALIZATION_LOSS)

    # Timer
    timer = Timer()
    timer.attach(trainer, start=Events.EPOCH_STARTED)
    timer.attach(validator, start=Events.EPOCH_STARTED)

    # Score function
    merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0.1, 0.9, _metric_getter)
    score_fn = lambda _ : merge_metrics_fn(trainer.state.metrics, validator.state.metrics)

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
            checkpoint_folder_path = get_checkpoint_folder_path('vqa', merged_dataset_name, model.name,
                # f'voc-minf={vocab_min_freq}',
                f'visenc-pretr={int(bool(model_kwargs["image_encoder_pretrained_weights_path"]))}',
                f'dws={",".join(map(str, _train_weights))}' \
                    if len(_train_weights) > 1 else None,
                'medtok' if use_medical_tokenization else None,
                'tags' if classify_tags else None,
                'orien' if classify_orientation else None,
                'chx' if classify_chexpert else None,
                'ql' if classify_questions else None,
                'amp' if use_amp else None,
            )
            print_red('checkpoint_folder_path =', checkpoint_folder_path)
            save_metadata(checkpoint_folder_path,
                        tokenizer_kwargs = tokenizer_kwargs,
                        model_kwargs = model_kwargs,
                        optimizer_kwargs = optimizer_kwargs,
                        lr_scheduler_kwargs = lr_scheduler_kwargs,
                        mimiccxr_vqa_trainer_kwargs = mimiccxr_vqa_trainer_kwargs,
                        iuxray_vqa_trainer_kwargs = iuxray_vqa_trainer_kwargs,
                        chexpert_dataset_kwargs = chexpert_dataset_kwargs,
                        cxr14_vqa_trainer_kwargs = cxr14_vqa_trainer_kwargs,
                        vinbig_dataset_kwargs = vinbig_dataset_kwargs,
                        padchest_dataset_kwargs = padchest_dataset_kwargs,
                        dataloading_kwargs = dataloading_kwargs,
                        train_image_transform_kwargs = train_image_transform_kwargs,
                        val_image_transform_kwargs = val_image_transform_kwargs,
                        training_kwargs = training_kwargs,
                        trainer_engine_kwargs = trainer_engine_kwargs,
                        val_engine_kwargs = val_engine_kwargs,
                        auxiliary_tasks_kwargs = auxiliary_tasks_kwargs)

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
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda : print(f'(1) Training stage (lr = {optimizer.param_groups[0]["lr"]:.6f}) ...'))
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
    # Tokenizer args
    vocab_min_freq,
    use_medical_tokenization,
    medical_terms_frequency_filename,
    # Model args
    visual_input_mode,
    freeze_image_encoder,
    raw_image_encoding,
    imagenet_pretrained,
    visual_features_mlp_in_dim,
    visual_features_mlp_out_dim,
    visual_features_mlp_hidden_dims,
    embed_size,
    question_encoding,
    answer_decoding,
    question_hidden_size,
    answer_hidden_size,
    n_lstm_layers,
    transf_dec_nhead,
    transf_dec_dim_forward,
    transf_dec_num_layers,
    question_vec_size,
    image_local_feat_size,
    dropout_prob,
    image_encoder_pretrained_weights_path,
    pretrained_checkpoint_folder_path,
    chest_imagenome_bbox_hidden_size,
    clip_version,
    huggingface_model_name,
    # Optimizer args
    optimizer_name,
    lr,
    # lr_scheduler args
    scheduler,
    lr_decay,
    lr_decay_patience,
    warmup_and_decay_args,
    warmup_and_cosine_args,
    # Image transform args
    image_size,
    # Dataset args
    iuxray_qa_adapted_reports_filename,
    mimiccxr_qa_adapted_reports_filename,    
    iuxray_balanced_metadata_filename,
    mimiccxr_balanced_metadata_filename,
    mimiccxr_view_mode,
    iuxray_precomputed_visual_features_path,
    mimiccxr_precomputed_visual_features_path,
    chexpert_precomputed_visual_features_path,
    vinbig_precomputed_visual_features_path,
    allowed_questions,
    padchest_train_study_ids_path,
    padchest_val_study_ids_path,
    padchest_test_study_ids_path,
    chest_imagenome_labels_filename,
    chest_imagenome_label_names_filename,
    # Dataloading args
    batch_size,
    num_workers,
    mimiccxr_weight,
    iuxray_weight,
    chexpert_weight,
    cxr14_weight,
    vinbig_weight,
    padchest_weight,
    mimiccxr_weight_chexpert_mode,
    mimiccxr_weight_chest_imagenome_mode,
    iuxray_weight_chexpert_mode,
    img_aug_mode,
    balanced_dataloading,
    imbalance_reduction_coef,
    # Fixed traning args
    train_mimiccxr,
    train_iuxray,
    train_chexpert,
    train_vinbig,
    train_cxr14,
    train_padchest,
    mimiccxr_include_mined_questions_mode,
    mimiccxr_include_chexpert_mode,
    mimiccxr_include_chest_imagenome_mode,
    iuxray_include_mined_questions_mode,
    iuxray_include_chexpert_mode,
    chexpert_mode,
    vinbig_training_data,
    vinbig_use_validation,
    padchest_training_data_mode,
    padchest_use_validation,
    binary_loss_name,
    use_amp,
    iters_to_accumulate,
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
    n_mined_questions,
    iuxray_question_labels_filename,
    mimiccxr_question_labels_filename,
    classify_chest_imagenome,
    predict_bboxes_chest_imagenome,
    merge_findings,
    # GPU
    device,
    # Other args
    save,
    debug = False,
):
    print_blue('----- Training model from scratch ------')

    assert train_mimiccxr or train_iuxray or train_chexpert or\
           train_cxr14 or train_vinbig or train_padchest, 'No dataset selected for training'

    assert (mimiccxr_include_mined_questions_mode or\
            mimiccxr_include_chexpert_mode or\
            mimiccxr_include_chest_imagenome_mode) == train_mimiccxr
    assert (iuxray_include_mined_questions_mode or\
            iuxray_include_chexpert_mode) == train_iuxray

    tokenizer_kwargs = dict(
        vocab_min_freq = vocab_min_freq,
        use_medical_tokenization = use_medical_tokenization,
        medical_terms_frequency_filename = medical_terms_frequency_filename,
    )    

    # Compute the total number of questions (that will be encoded as one-hot vectors)
    # considerning different modes of training and datasets
    if classify_chest_imagenome or mimiccxr_include_chest_imagenome_mode:
        assert chest_imagenome_label_names_filename is not None
        n_chest_imagenome_labels = len(load_chest_imagenome_postprocessed_label_names(chest_imagenome_label_names_filename))
    n_q_total = 0
    if iuxray_include_mined_questions_mode or mimiccxr_include_mined_questions_mode:
        assert n_mined_questions is not None
        n_q_total += n_mined_questions
    if merge_findings:
        finding_labels_remapper, merged_finding_labels = get_merged_findings(train_chexpert, train_cxr14, train_vinbig)
        print(f'len(merged_finding_labels)={len(merged_finding_labels)}')
        n_findings = len(merged_finding_labels)
        n_q_total += n_findings
    else:
        n_findings = None
        if mimiccxr_include_chexpert_mode or iuxray_include_chexpert_mode or \
        (train_chexpert and chexpert_mode == CHEXPERT_TASKS.VQA):
            n_q_total += len(CHEXPERT_LABELS)
        if mimiccxr_include_chest_imagenome_mode:
            n_q_total += n_chest_imagenome_labels
        if train_vinbig:
            n_q_total += len(VINBIG_DISEASES)
        if train_cxr14:
            n_q_total += len(CXR14_LABELS)
        if train_padchest:
            n_q_total += PADCHEST_NUM_QUESTIONS

    # Compute the offset of each question in the one-hot vector
    print_blue('Computing one-hot question offsets')
    one_hot_question_offsets = {}
    _offset = 0
    if mimiccxr_include_mined_questions_mode or iuxray_include_mined_questions_mode:
        one_hot_question_offsets[str(IUXRAY_DATASET_ID)] = _offset
        one_hot_question_offsets[str(MIMICCXR_DATASET_ID)] = _offset
        _offset += n_mined_questions
    if merge_findings:
        one_hot_question_offsets[str(CHEXPERT_DATASET_ID)] = _offset
        one_hot_question_offsets[str(CXR14_DATASET_ID)] = _offset
        one_hot_question_offsets[str(VINBIG_DATASET_ID)] = _offset
        _offset += n_findings
    else:
        if train_chexpert or iuxray_include_chexpert_mode or mimiccxr_include_chexpert_mode:
            one_hot_question_offsets[str(CHEXPERT_DATASET_ID)] = _offset
            one_hot_question_offsets[str(IUXRAY_DATASET_ID__CHEXPERT_MODE)] = _offset
            one_hot_question_offsets[str(MIMICCXR_DATASET_ID__CHEXPERT_MODE)] = _offset
            _offset += len(CHEXPERT_LABELS)
        if train_cxr14:
            one_hot_question_offsets[str(CXR14_DATASET_ID)] = _offset
            _offset += len(CXR14_LABELS)
        if train_vinbig:
            one_hot_question_offsets[str(VINBIG_DATASET_ID)] = _offset
            _offset += len(VINBIG_DISEASES)
        if train_padchest:
            one_hot_question_offsets[str(PADCHEST_DATASET_ID)] = _offset
            _offset += PADCHEST_NUM_QUESTIONS
        if mimiccxr_include_chest_imagenome_mode:
            one_hot_question_offsets[str(MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE)] = _offset
            _offset += n_chest_imagenome_labels

    print(f'one_hot_question_offsets = {one_hot_question_offsets}')

    use_clip = raw_image_encoding in (
                RawImageEncoding.CLIP_RESNET,
                RawImageEncoding.CLIP_VIT,
                RawImageEncoding.CLIP_VIT__HUGGINGFACE,
                RawImageEncoding.CLIP_VIT_LARGE__HUGGINGFACE,
                RawImageEncoding.CLIP_RESNET__HUGGINGFACE)
    use_huggingface_vitmodel = raw_image_encoding == RawImageEncoding.VITMODEL__HUGGINGFACE
    if use_clip or use_huggingface_vitmodel:
        if use_clip: assert clip_version is not None
        if use_huggingface_vitmodel: assert huggingface_model_name is not None
        assert image_size == 224 or image_size == [224, 224]
        if type(image_size) is list: image_size = tuple(image_size)
        
    model_kwargs = dict(
        embed_size = embed_size,
        dropout_prob = dropout_prob,
        pretrained_checkpoint_folder_path = os.path.join(WORKSPACE_DIR, pretrained_checkpoint_folder_path) \
            if pretrained_checkpoint_folder_path is not None else None,        
        chexpert_mode = chexpert_mode,
        n_questions=n_q_total,
        # Image encoder
        visual_input_mode = visual_input_mode,
        raw_image_encoding = raw_image_encoding,
        freeze_image_encoder = freeze_image_encoder,
        image_local_feat_size = image_local_feat_size,
        image_encoder_pretrained_weights_path = image_encoder_pretrained_weights_path,
        imagenet_pretrained = imagenet_pretrained,
        mlp_in_dim = visual_features_mlp_in_dim,
        mlp_out_dim = visual_features_mlp_out_dim,
        mlp_hidden_dims = visual_features_mlp_hidden_dims,
        clip_version = clip_version,
        huggingface_model_name = huggingface_model_name,
        # Question encoder
        question_encoding = question_encoding,
        question_vec_size = question_vec_size,
        question_hidden_size = question_hidden_size,
        # Answer decoder
        answer_decoding = answer_decoding,
        answer_hidden_size = answer_hidden_size,
        n_lstm_layers = n_lstm_layers,
        transf_dec_nhead = transf_dec_nhead,
        transf_dec_dim_forward = transf_dec_dim_forward,
        transf_dec_num_layers = transf_dec_num_layers,
        # Aux tasks
        n_medical_tags=n_medical_tags,
        classify_orientation=classify_orientation,
        classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        n_questions_aux_task=n_mined_questions,
        n_chest_imagenome_labels=n_chest_imagenome_labels,
        chest_imagenome_bbox_hidden_size=chest_imagenome_bbox_hidden_size,
        use_cxr14=train_cxr14,
        use_vinbig=train_vinbig,
        use_padchest=train_padchest,
        merge_findings=merge_findings,
        n_findings=n_findings,
    )
    
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
        n_batches_per_epoch = batches_per_epoch,
    )
    
    dataloading_kwargs = dict(
        batch_size = batch_size,
        mimiccxr_weight = mimiccxr_weight,
        iuxray_weight = iuxray_weight,
        chexpert_weight = chexpert_weight,
        cxr14_weight = cxr14_weight,
        vinbig_weight = vinbig_weight,
        padchest_weight = padchest_weight,
        mimiccxr_weight_chexpert_mode = mimiccxr_weight_chexpert_mode,
        mimiccxr_weight_chest_imagenome_mode = mimiccxr_weight_chest_imagenome_mode,
        iuxray_weight_chexpert_mode = iuxray_weight_chexpert_mode,
        one_hot_question_offsets = one_hot_question_offsets,
    )

    if merge_findings:
        _merged_findings_kwargs = dict(
            use_merged_findings = True,
            findings_remapper = finding_labels_remapper,
            n_findings = n_findings,
        )

    train_image_transform_kwargs = dict(
        image_size = image_size,
        augmentation_mode = img_aug_mode,
        use_clip_transform = use_clip,
        clip_version = clip_version,
        use_huggingface_vitmodel_transform = use_huggingface_vitmodel,
        huggingface_vitmodel_name = huggingface_model_name,
    )

    # clone to avoid modifying the original one
    val_image_transform_kwargs = train_image_transform_kwargs.copy()
    val_image_transform_kwargs['augmentation_mode'] = None # no augmentation for validation
    
    verbose_question = question_encoding != QuestionEncoding.ONE_HOT

    include_image = does_include_image(visual_input_mode)
    include_visual_features = does_include_visual_features(visual_input_mode)
    if include_visual_features:
        if train_mimiccxr: assert mimiccxr_precomputed_visual_features_path is not None
        if train_iuxray: assert iuxray_precomputed_visual_features_path is not None
        if train_chexpert: assert chexpert_precomputed_visual_features_path is not None
        if train_vinbig: assert vinbig_precomputed_visual_features_path is not None
    
    if train_mimiccxr:
        print(f'Debug: balanced_dataloading = {balanced_dataloading}')
        mimiccxr_vqa_trainer_kwargs = dict(
            qa_adapted_reports_filename=mimiccxr_qa_adapted_reports_filename,
            balanced_dataloading=balanced_dataloading,
            balanced_metadata_filename=mimiccxr_balanced_metadata_filename,
            imbalance_reduction_coef=imbalance_reduction_coef,
            include_mined_questions_mode=mimiccxr_include_mined_questions_mode,
            include_chexpert_mode=mimiccxr_include_chexpert_mode,            
            include_chest_imagenome_mode=mimiccxr_include_chest_imagenome_mode,
            chest_imagenome_labels_filename=chest_imagenome_labels_filename,
            chest_imagenome_label_names_filename=chest_imagenome_label_names_filename,
            include_image=include_image,
            view_mode=mimiccxr_view_mode,
            use_precomputed_visual_features=include_visual_features,
            precomputed_visual_features_path=mimiccxr_precomputed_visual_features_path,
            classify_tags=classify_tags,
            medical_tags_per_report_filename=mimiccxr_medical_tags_per_report_filename,
            classify_orientation=classify_orientation,
            classify_chexpert=classify_chexpert,
            chexpert_labels_filename=mimiccxr_chexpert_labels_filename,
            classify_questions=classify_questions,
            classify_chest_imagenome=classify_chest_imagenome,
            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
            question_labels_filename=mimiccxr_question_labels_filename,
            allowed_questions=allowed_questions,
            verbose_question=verbose_question,
        )
        if merge_findings:
            mimiccxr_vqa_trainer_kwargs.update(_merged_findings_kwargs)
    else:
        mimiccxr_vqa_trainer_kwargs = None

    if train_iuxray:
        iuxray_vqa_trainer_kwargs=dict(
            qa_adapted_reports_filename=iuxray_qa_adapted_reports_filename,
            balanced_dataloading=balanced_dataloading,
            balanced_metadata_filename=iuxray_balanced_metadata_filename,
            imbalance_reduction_coef=imbalance_reduction_coef,
            include_mined_questions_mode=iuxray_include_mined_questions_mode,
            include_chexpert_mode=iuxray_include_chexpert_mode,
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
            allowed_questions=allowed_questions,
            verbose_question=verbose_question,
        )
        if merge_findings:
            iuxray_vqa_trainer_kwargs.update(_merged_findings_kwargs)
    else:
        iuxray_vqa_trainer_kwargs = None

    chexpert_dataset_kwargs = dict(
        include_image = include_image,
        use_precomputed_visual_features = include_visual_features,
        precomputed_visual_features_path = chexpert_precomputed_visual_features_path,
    )
    if merge_findings:
        chexpert_dataset_kwargs.update(_merged_findings_kwargs)

    cxr14_vqa_trainer_kwargs = {}
    if merge_findings:
        cxr14_vqa_trainer_kwargs.update(_merged_findings_kwargs)

    vinbig_dataset_kwargs = dict(
        include_image=include_image,
        use_precomputed_visual_features=include_visual_features,
        precomputed_visual_features_path=vinbig_precomputed_visual_features_path,
        training_data=vinbig_training_data,
        use_validation=vinbig_use_validation,
    )
    if merge_findings:
        vinbig_dataset_kwargs.update(_merged_findings_kwargs)

    padchest_dataset_kwargs = dict(
        include_image=include_image,
        train_study_ids_path=padchest_train_study_ids_path,
        val_study_ids_path=padchest_val_study_ids_path,
        test_study_ids_path=padchest_test_study_ids_path,
        training_data_mode=padchest_training_data_mode,
        use_validation_set=padchest_use_validation,
    )

    trainer_engine_kwargs = dict(
        classify_tags=classify_tags,
        classify_orientation=classify_orientation,
        classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        question_encoding=question_encoding,
        answer_decoding=AnswerDecoding.TEACHER_FORCING,
        binary_loss_name=binary_loss_name,
        include_image=include_image,
        include_visual_features=include_visual_features,
        shift_answer=(answer_decoding == 'transformer'),
        use_amp=use_amp,
        training=True,
        train_with_chexpert_dataset=train_chexpert,
        chexpert_mode=chexpert_mode,
        train_with_cxr14=train_cxr14,
        use_vinbig_dataset=train_vinbig,
        use_padchest_dataset=train_padchest,
        iters_to_accumulate=iters_to_accumulate,
    )
    if merge_findings:
        trainer_engine_kwargs.update(_merged_findings_kwargs)

    val_engine_kwargs = dict(
        classify_tags=classify_tags,
        classify_orientation=classify_orientation,
        classify_chexpert=classify_chexpert,
        classify_questions=classify_questions,
        classify_chest_imagenome=classify_chest_imagenome,
        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
        question_encoding=question_encoding,
        answer_decoding=val_answer_decoding,
        include_image=include_image,
        include_visual_features=include_visual_features,
        beam_search_k=beam_search_k,
        training=False,
        chexpert_mode=chexpert_mode,
        use_vinbig_dataset=train_vinbig,
        use_padchest_dataset=train_padchest,
        use_merged_findings=merge_findings,
    )
    
    training_kwargs = dict(
        use_amp=use_amp,
        train_mimiccxr=train_mimiccxr,
        train_iuxray=train_iuxray,
        train_chexpert=train_chexpert,
        train_cxr14=train_cxr14,
        train_vinbig=train_vinbig,
        train_padchest=train_padchest,
        chexpert_mode=chexpert_mode,
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
    )

    return train_model(
                tokenizer_kwargs,
                model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_vqa_trainer_kwargs,
                iuxray_vqa_trainer_kwargs,
                chexpert_dataset_kwargs,
                cxr14_vqa_trainer_kwargs,
                vinbig_dataset_kwargs,
                padchest_dataset_kwargs,
                dataloading_kwargs,
                train_image_transform_kwargs,
                val_image_transform_kwargs,
                training_kwargs,
                trainer_engine_kwargs,
                val_engine_kwargs,
                auxiliary_tasks_kwargs,
                val_answer_decoding,
                epochs,
                batches_per_epoch,
                one_question_per_batch,
                num_workers,
                beam_search_k=beam_search_k,
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
    num_workers,
    one_question_per_batch,
    val_answer_decoding,
    epochs = 1,
    batches_per_epoch = 1000,
    device = 'GPU',
    save = True,
    override_lr = False,
    debug = False,
    **unused_kwargs,
):
    print_blue('----- Resuming training ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    mimiccxr_vqa_trainer_kwargs = metadata['mimiccxr_vqa_trainer_kwargs']
    iuxray_vqa_trainer_kwargs = metadata['iuxray_vqa_trainer_kwargs']
    chexpert_dataset_kwargs = metadata['chexpert_dataset_kwargs']
    cxr14_vqa_trainer_kwargs = metadata['cxr14_vqa_trainer_kwargs']
    vinbig_dataset_kwargs = metadata['vinbig_dataset_kwargs']
    padchest_dataset_kwargs = metadata['padchest_dataset_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    train_image_transform_kwargs = metadata['train_image_transform_kwargs']
    val_image_transform_kwargs = metadata['val_image_transform_kwargs']
    training_kwargs = metadata['training_kwargs']
    trainer_engine_kwargs = metadata['trainer_engine_kwargs']
    val_engine_kwargs = metadata['val_engine_kwargs']                
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
            n_batches_per_epoch = batches_per_epoch,
        )

    return train_model(
                tokenizer_kwargs,
                model_kwargs,
                optimizer_kwargs,
                lr_scheduler_kwargs,
                mimiccxr_vqa_trainer_kwargs,
                iuxray_vqa_trainer_kwargs,
                chexpert_dataset_kwargs,
                cxr14_vqa_trainer_kwargs,
                vinbig_dataset_kwargs,
                padchest_dataset_kwargs,
                dataloading_kwargs,
                train_image_transform_kwargs,
                val_image_transform_kwargs,
                training_kwargs,
                trainer_engine_kwargs,
                val_engine_kwargs,
                auxiliary_tasks_kwargs,
                val_answer_decoding,
                epochs,
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