import  os
import argparse
import torch
from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.mimiccxr.mimiccxr_multimodal_dataset_management import MIMICCXR_Multimodal_Trainer
from medvqa.models.multimodal_encoding.image_text_encoder import ImageTextEncoder
from medvqa.utils.constants import MIMICCXR_DATASET_ID, MetricNames
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_chexpert_labels_accuracy,
    attach_chexpert_labels_prf1,
    attach_chexpert_labels_roc_auc,
    attach_dataset_aware_bleu_background,
    attach_dataset_aware_orientation_accuracy,
    attach_question_labels_prf1,
)
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    load_metadata,
)
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.handlers_utils import (
    attach_accumulator,
    get_log_metrics_handlers,
    get_log_iteration_handler,
)
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.utils.files_utils import get_results_folder_path, load_json_file, save_to_pickle
from medvqa.training.multimodal import get_engine
from medvqa.datasets.dataloading_utils import get_multimodal_collate_batch_fn
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging_utils import CountPrinter, print_blue

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # --- Required arguments
    parser.add_argument('--checkpoint-folder', type=str, required=True,
                        help='Relative path to folder with checkpoint to evaluate')
    parser.add_argument('--batch-size', type=int, required=True)

    # --- Optional arguments
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for parallel dataloading')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')
    
    return parser.parse_args(args=args)

def _save_question_probabilities(metrics, dataset, results_folder_path):
    output = {}
    for idx, probs in zip(metrics['idxs'], metrics['pred_qlabels_probs']):
        rid = dataset.report_ids[idx]
        output[rid] = probs
    save_path = os.path.join(results_folder_path, f'mimiccxr_question_probs.pkl')
    save_to_pickle(output, save_path)
    print (f'Question probabilities saved to {save_path}')

def _save_metrics(metrics, metric_names, results_folder_path):    
    to_save = { m : metrics[m] for m in metric_names }
    save_path = os.path.join(results_folder_path, f'mimiccxr_multimodal_metrics.pkl')
    save_to_pickle(to_save, save_path)
    print (f'Metrics successfully saved to {save_path}')

def _eval_model(
    tokenizer_kwargs,
    model_kwargs,
    mimiccxr_trainer_kwargs,
    iuxray_trainer_kwargs,
    image_transform_kwargs,
    val_engine_kwargs,
    auxiliary_tasks_kwargs,
    num_workers,
    batch_size,
    device = 'GPU',
    checkpoint_folder_path = None,
):
    count_print = CountPrinter()
    
    # auxiliary task: orientation classification
    classify_orientation = auxiliary_tasks_kwargs['classify_orientation']

    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']

    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs['classify_questions']
    n_questions = auxiliary_tasks_kwargs['n_questions']
    mimiccxr_question_labels_filename = auxiliary_tasks_kwargs['mimiccxr_question_labels_filename']
    assert classify_questions
    assert n_questions is not None
    assert mimiccxr_question_labels_filename is not None
    
    # QA dataset filenames
    iuxray_qa_adapted_reports_filename = iuxray_trainer_kwargs['qa_adapted_reports_filename']
    mimiccxr_qa_adapted_reports_filename = mimiccxr_trainer_kwargs['qa_adapted_reports_filename']
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
    assert iuxray_qa_reports['questions'] == mimiccxr_qa_reports['questions']
    
    # Init tokenizer
    count_print('Initializing tokenizer ...')    
    tokenizer = Tokenizer(**tokenizer_kwargs)
    
    # Create model
    count_print('Creating instance of ImageTextEncoder model ...')
    model = ImageTextEncoder(vocab_size=tokenizer.vocab_size,
                            start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                            device=device, **model_kwargs)
    model = model.to(device)

    # Create validator engine
    count_print('Creating validator engine ...')
    validator = get_engine(model=model, tokenizer=tokenizer, device=device, **val_engine_kwargs)

    # Define image transform
    count_print('Defining image transform ...')
    img_transform = get_image_transform(**image_transform_kwargs)
    
    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    
    _kwargs = dict(classify_orientation = classify_orientation,
                   classify_chexpert = classify_chexpert,
                   classify_questions = classify_questions)
    mimiccxr_collate_batch_fn = get_multimodal_collate_batch_fn(MIMICCXR_DATASET_ID, **_kwargs)

    # Create MIMIC-CXR multimodal trainer
    count_print('Creating MIMIC-CXR multimodal trainer ...')
    mimiccxr_trainer = MIMICCXR_Multimodal_Trainer(
        transform = img_transform,
        batch_size = batch_size,
        collate_batch_fn = mimiccxr_collate_batch_fn,            
        num_workers = num_workers,
        tokenizer = tokenizer,
        mimiccxr_qa_reports = mimiccxr_qa_reports,
        include_train = False,
        **mimiccxr_trainer_kwargs,
    )
    
    # Attach metrics, losses, timer and events to engines    
    count_print('Attaching metrics, losses, timer and events to engines ...')
        
    attach_dataset_aware_bleu_background(validator, [MIMICCXR_DATASET_ID])    
    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(validator, [MIMICCXR_DATASET_ID])
    if classify_questions:
        attach_question_labels_prf1(validator, device)
    if classify_chexpert:
        attach_chexpert_labels_accuracy(validator, device)        
        attach_chexpert_labels_prf1(validator, device)
        attach_chexpert_labels_roc_auc(validator, 'cpu')
    
    # Accumulators
    attach_accumulator(validator, 'idxs')
    attach_accumulator(validator, 'pred_qlabels_probs')

    # Timer
    timer = Timer()
    timer.attach(validator, start=Events.EPOCH_STARTED)

    # Checkpoint loading
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path =', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True)

    # Logging
    count_print('Defining log_metrics_handler ...')
    metrics_to_print=[MetricNames.BLEU_BACKGROUND]
    if classify_orientation:
        metrics_to_print.append(MetricNames.ORIENACC)
    if classify_questions:
        metrics_to_print.append(MetricNames.QLABELS_PRF1)
    if classify_chexpert:
        metrics_to_print.append(MetricNames.CHXLABEL_PRF1)
        metrics_to_print.append(MetricNames.CHXLABELACC)
        metrics_to_print.append(MetricNames.CHXLABEL_ROCAUC)
    
    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()
    
    # Attach handlers    
    validator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    validator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    validator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation
    results_folder_path = get_results_folder_path(checkpoint_folder_path)
    count_print('Running validator engine ...')
    validator.run(mimiccxr_trainer.test_dataloader)
    _save_question_probabilities(validator.state.metrics,
                                 mimiccxr_trainer.test_dataset,
                                 results_folder_path)
    _save_metrics(validator.state.metrics,
                  metrics_to_print,
                  results_folder_path)

def eval_model(
    checkpoint_folder,
    num_workers,
    batch_size,
    device = 'GPU',
):
    print_blue('----- Evaluating model ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    model_kwargs = metadata['model_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
    iuxray_trainer_kwargs = metadata['iuxray_trainer_kwargs']
    image_transform_kwargs = metadata['image_transform_kwargs']
    val_engine_kwargs = metadata['val_engine_kwargs']                
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']

    return _eval_model(
                tokenizer_kwargs,
                model_kwargs,
                mimiccxr_trainer_kwargs,
                iuxray_trainer_kwargs,
                image_transform_kwargs,
                val_engine_kwargs,
                auxiliary_tasks_kwargs,
                num_workers,
                batch_size,
                device = device,
                checkpoint_folder_path = checkpoint_folder,
            )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    eval_model(**args)