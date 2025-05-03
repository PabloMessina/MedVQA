import  os
import argparse
from copy import deepcopy
from pprint import pprint

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer

from medvqa.utils.constants import (
    CHEXPERT_DATASET_ID,
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID,
    MetricNames,
)
from medvqa.metrics import (
    attach_medical_tags_f1score,
    attach_chexpert_labels_accuracy,
    attach_chexpert_labels_prf1,
    attach_chexpert_labels_roc_auc,
    attach_dataset_aware_orientation_accuracy,
    attach_question_labels_prf1,
)
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    load_metadata,
)
from medvqa.utils.common import (
    WORKSPACE_DIR,
    parsed_args_to_dict,
)    
from medvqa.utils.handlers_utils import (
    get_log_metrics_handlers,
    get_log_iteration_handler,
    attach_accumulator,
)
from medvqa.models.vision.visual_modules import DensenetVisualModule
from medvqa.utils.files_utils import (
    get_results_folder_path,
    save_to_pickle,
)
from medvqa.training.vision import get_engine
from medvqa.datasets.dataloading_utils import get_vision_collate_batch_fn
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleEvaluator
from medvqa.datasets.iuxray.iuxray_vision_dataset_management import IUXRAY_VisualModuleTrainer
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging_utils import CountPrinter

class EvalDatasets:
    MIMICCXR_TEST_SET = 'mimiccxr_test_set'
    CHEST_IMAGENOME_GOLD = 'chest_imagenome_gold'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--checkpoint-folder', type=str, required=True,
                        help='Relative path to folder with checkpoint to evaluate')
    parser.add_argument('--eval-dataset-name', type=str, required=True)

    # optional arguments
    parser.add_argument('--mimiccxr-preprocessed-test-data-filename', type=str, default=None)
    parser.add_argument('--iuxray-preprocessed-train-data-filename', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=140,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for parallel dataloading')

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)
    
    return parser.parse_args(args=args)

def _save_metrics(results_dict, dataset_name, metric_names, results_folder_path):    
    metrics_dict = results_dict[f'{dataset_name}_metrics']
    to_save = { m : metrics_dict[m] for m in metric_names }
    save_path = os.path.join(results_folder_path, f'{dataset_name}_visual_module_metrics.pkl')
    save_to_pickle(to_save, save_path)
    print (f'Metrics successfully saved to {save_path}')

def _adapt_aux_tasks_kwargs(aux_tasks_kwargs, dataset_name):
    output = {}
    for k, v in aux_tasks_kwargs.items():
        if k.startswith(dataset_name + '_'):
            output[k[len(dataset_name)+1:]] = v
        else:
            output[k] = v
    return output

def _recover_vision_dataset_manager_kwargs(dataset_name, metadata, batch_size, preprocessed_data_filename, aux_tasks_kwargs):
    keys = [
        f'{dataset_name}_vision_trainer_kwargs',
        f'{dataset_name}_vqa_trainer_kwargs',
        f'{dataset_name}_trainer_kwargs',
    ]
    # pprint(metadata)
    kwargs = None
    for key in keys:
        if key in metadata:
            kwargs = metadata[key]
            break
    assert kwargs is not None
    kwargs['batch_size'] = batch_size
    kwargs['preprocessed_data_filename'] = preprocessed_data_filename
    kwargs.update(_adapt_aux_tasks_kwargs(aux_tasks_kwargs, dataset_name))
    return kwargs

def _recover_iuxray_vision_trainer_kwargs(metadata, batch_size, preprocessed_data_filename, aux_tasks_kwargs):
    return _recover_vision_dataset_manager_kwargs('iuxray', metadata, batch_size, preprocessed_data_filename, aux_tasks_kwargs)

def _recover_mimiccxr_vision_evaluator_kwargs(metadata, batch_size, preprocessed_data_filename, aux_tasks_kwargs):
    return _recover_vision_dataset_manager_kwargs('mimiccxr', metadata, batch_size, preprocessed_data_filename, aux_tasks_kwargs)

def _adapt_checkpoint_keys(checkpoint):
    keys = [k for k in checkpoint.keys()]
    for key in keys:
        if key.startswith('image_encoder.'):
            checkpoint['raw_' + key] = checkpoint[key]
    return checkpoint

def _evaluate_model(
    eval_dataset_name,
    model_kwargs,
    mimiccxr_vision_evaluator_kwargs,
    iuxray_vision_trainer_kwargs,
    auxiliary_tasks_kwargs,
    num_workers = 0,
    device = 'GPU',
    checkpoint_folder_path = None,
    return_results = False,
    use_amp = False,    
    debug = False,
):

    # Pull out some args from kwargs

    # auxiliary task: medical tags prediction
    classify_tags = auxiliary_tasks_kwargs.get('classify_tags', False)
    n_medical_tags = auxiliary_tasks_kwargs.get('n_medical_tags', None)
    if classify_tags:
        assert n_medical_tags is not None        
    
    # auxiliary task: orientation classification
    classify_orientation = auxiliary_tasks_kwargs['classify_orientation']

    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']

    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)        
    
    count_print = CountPrinter()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)
    
    # Create model
    count_print('Creating instance of DensenetVisualModule model ...')
    model = DensenetVisualModule(**model_kwargs)
    model = model.to(device)

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    evaluator = get_engine(model, classify_tags, classify_orientation, classify_chexpert,
                         classify_questions, device, use_amp=use_amp, training=False)    

    # Attach metrics, timer and events to engines    
    count_print('Attaching metrics, timer and events to engines ...')

    # Metrics
    if classify_tags:
        attach_medical_tags_f1score(evaluator, device)
    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(evaluator, [MIMICCXR_DATASET_ID, IUXRAY_DATASET_ID])
    if classify_chexpert:
        attach_chexpert_labels_accuracy(evaluator, device)        
        attach_chexpert_labels_prf1(evaluator, device)
        attach_chexpert_labels_roc_auc(evaluator, 'cpu')
    if classify_questions:
        attach_question_labels_prf1(evaluator, device)        

    if return_results:
        attach_accumulator(evaluator, 'idxs')
        if classify_tags:
            attach_accumulator(evaluator, 'pred_tags')
        if classify_orientation:
            attach_accumulator(evaluator, 'pred_orientation')
        if classify_chexpert:
            attach_accumulator(evaluator, 'pred_chexpert')
        if classify_questions:
            attach_accumulator(evaluator, 'pred_qlabels')
    
    # Timer
    timer = Timer()
    timer.attach(evaluator, start=Events.EPOCH_STARTED)
    
    # Logging
    metrics_to_print=[]
    if classify_tags:
        metrics_to_print.append(MetricNames.MEDTAGF1)
    if classify_orientation:
        metrics_to_print.append(MetricNames.ORIENACC)
    if classify_chexpert:
        metrics_to_print.append(MetricNames.CHXLABEL_PRF1)
        metrics_to_print.append(MetricNames.CHXLABELACC)
        metrics_to_print.append(MetricNames.CHXLABEL_ROCAUC)
    if classify_questions:
        metrics_to_print.append(MetricNames.QLABELS_PRF1)

    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()

    # Load saved checkpoint
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(_adapt_checkpoint_keys(checkpoint['model']), strict=False)
    print('Checkpoint successfully loaded!')

    # Attach handlers    
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

    # Default image transform
    count_print('Defining image transform ...')
    image_transform = get_image_transform(**image_transform_kwargs)

    # Define test dataset and dataloader
    if eval_dataset_name == EvalDatasets.MIMICCXR_TEST_SET:
        mimiccxr_collate_batch_fn = get_vision_collate_batch_fn(MIMICCXR_DATASET_ID,
                                    classify_tags = classify_tags,
                                    n_tags = n_medical_tags,
                                    classify_orientation = classify_orientation,
                                    classify_chexpert = classify_chexpert,
                                    classify_questions = classify_questions)
        mimiccxr_vision_evaluator = MIMICCXR_VisualModuleEvaluator(
            transform = image_transform,
            collate_batch_fn = mimiccxr_collate_batch_fn,
            num_workers = num_workers,
            **mimiccxr_vision_evaluator_kwargs,
        )
        if debug:
            return {'mimiccxr_vision_evaluator' : mimiccxr_vision_evaluator}
        dataset = mimiccxr_vision_evaluator.test_dataset
        dataloader = mimiccxr_vision_evaluator.test_dataloader
    else:
        raise ValueError(f'Invalid eval_dataset_name: {eval_dataset_name}')

    # Run evaluation
    metrics_to_save = metrics_to_print
    results_folder_path = get_results_folder_path(checkpoint_folder_path)
    results_dict = {}

    count_print('Running evaluator engine ...')
    print('len(dataset) =', len(dataset))
    print('len(dataloader) =', len(dataloader))
    evaluator.run(dataloader)
    results_dict['metrics'] = deepcopy(evaluator.state.metrics)
    results_dict['dataset'] = dataset
    _save_metrics(results_dict, eval_dataset_name, metrics_to_save, results_folder_path)

    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder,
    mimiccxr_preprocessed_test_data_filename,
    iuxray_preprocessed_train_data_filename,
    batch_size = 100,
    num_workers = 0,
    device = 'GPU',
    return_results = False,
    use_amp = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
    debug = False,
):
    print('----- Evaluating model ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)    
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    if model_kwargs.get('merge_findings', False):
        model_kwargs['chexpert_indices'] =\
            metadata['trainer_engine_kwargs']['findings_remapper'][str(CHEXPERT_DATASET_ID)]
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']
    mimiccxr_vision_evaluator_kwargs = _recover_mimiccxr_vision_evaluator_kwargs(metadata, batch_size,
            mimiccxr_preprocessed_test_data_filename, auxiliary_tasks_kwargs)
    iuxray_vision_trainer_kwargs = _recover_iuxray_vision_trainer_kwargs(metadata, batch_size,
            iuxray_preprocessed_train_data_filename, auxiliary_tasks_kwargs)    
    if debug:
        print('metadata:')
        pprint(metadata)
        print()
        print('mimiccxr_vision_evaluator_kwargs:')
        pprint(mimiccxr_vision_evaluator_kwargs)
        print()
        print('iuxray_vision_trainer_kwargs:')
        pprint(iuxray_vision_trainer_kwargs)
        print()

    return _evaluate_model(
                model_kwargs,
                mimiccxr_vision_evaluator_kwargs,
                iuxray_vision_trainer_kwargs,
                auxiliary_tasks_kwargs = auxiliary_tasks_kwargs,
                device = device,
                num_workers = num_workers,
                checkpoint_folder_path = checkpoint_folder,
                return_results = return_results,
                use_amp = use_amp,
                eval_iuxray = eval_iuxray,
                eval_mimiccxr = eval_mimiccxr,
                debug = debug,
            )

def debug_main(args):
    args = parse_args(args)
    args = parsed_args_to_dict(args)
    return evaluate_model(**args, debug=True)

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)