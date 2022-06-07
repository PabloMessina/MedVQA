import  os
import argparse
from copy import deepcopy

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer

from medvqa.utils.constants import (
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
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.utils.common import (
    WORKSPACE_DIR,
    parsed_args_to_dict,
)    
from medvqa.utils.handlers import (
    get_log_metrics_handlers,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
    attach_accumulator,
)
from medvqa.models.vision.visual_modules import DensenetVisualModule
from medvqa.utils.files import (
    get_results_folder_path,
    save_to_pickle,
)
from medvqa.training.vision import get_engine
from medvqa.datasets.dataloading_utils import get_vision_collate_batch_fn
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleEvaluator
from medvqa.datasets.iuxray.iuxray_vision_dataset_management import IUXRAY_VisualModuleTrainer
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--checkpoint-folder', type=str, required=True,
                        help='Relative path to folder with checkpoint to evaluate')
    parser.add_argument('--mimiccxr-preprocessed-test-data-filename', type=str, required=True)

    # optional arguments    
    parser.add_argument('--batch-size', type=int, default=140,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use (GPU or CPU)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for parallel dataloading')

    parser.add_argument('--iuxray', dest='eval_iuxray', action='store_true')
    parser.add_argument('--no-iuxray', dest='eval_iuxray', action='store_false')
    parser.set_defaults(eval_iuxray=True)

    parser.add_argument('--mimiccxr', dest='eval_mimiccxr', action='store_true')
    parser.add_argument('--no-mimiccxr', dest='eval_mimiccxr', action='store_false')
    parser.set_defaults(eval_mimiccxr=True)

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)
    
    return parser.parse_args()

def _save_metrics(results_dict, dataset_name, metric_names, results_folder_path):    
    metrics_dict = results_dict[f'{dataset_name}_metrics']
    to_save = { m : metrics_dict[m] for m in metric_names }
    save_path = os.path.join(results_folder_path, f'{dataset_name}_metrics.pkl')
    save_to_pickle(to_save, save_path)
    print (f'Metrics successfully saved to {save_path}')

def _evaluate_model(
    model_kwargs,
    mimiccxr_vision_evaluator_kwargs,
    iuxray_vision_trainer_kwargs,
    auxiliary_tasks_kwargs,
    num_workers = 0,
    device = 'GPU',
    checkpoint_folder_path = None,
    return_results = False,
    use_amp = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
):

    assert eval_iuxray or eval_mimiccxr

    # Pull out some args from kwargs

    # auxiliary task: medical tags prediction
    classify_tags = auxiliary_tasks_kwargs['classify_tags']
    n_medical_tags = auxiliary_tasks_kwargs['n_medical_tags']
    iuxray_rid2tags_filename = auxiliary_tasks_kwargs['iuxray_rid2tags_filename']
    mimiccxr_rid2tags_filename = auxiliary_tasks_kwargs['mimiccxr_rid2tags_filename']
    if classify_tags:
        assert n_medical_tags is not None
        if eval_iuxray: assert iuxray_rid2tags_filename is not None
        if eval_mimiccxr: assert mimiccxr_rid2tags_filename is not None
    
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
        if eval_iuxray: assert iuxray_question_labels_filename is not None
        if eval_mimiccxr: assert mimiccxr_question_labels_filename is not None    
    
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
    
    # Default image transform
    count_print('Defining image transform ...')
    img_transform = get_image_transform()

    # Define collate_batch_fn    
    if eval_mimiccxr:
        mimiccxr_collate_batch_fn = get_vision_collate_batch_fn(MIMICCXR_DATASET_ID,
                                                        classify_tags = classify_tags,
                                                        n_tags = n_medical_tags,
                                                        classify_orientation = classify_orientation,
                                                        classify_chexpert = classify_chexpert,
                                                        classify_questions = classify_questions)
    if eval_iuxray:
        iuxray_collate_batch_fn = get_vision_collate_batch_fn(IUXRAY_DATASET_ID,
                                                    classify_tags = classify_tags,
                                                    n_tags = n_medical_tags,
                                                    classify_orientation = classify_orientation,
                                                    classify_chexpert = classify_chexpert,
                                                    classify_questions = classify_questions)

    # Create MIMIC-CXR visual module evaluator
    if eval_mimiccxr:
        count_print('Creating MIMIC-CXR visual module evaluator ...')
        mimiccxr_vision_evaluator = MIMICCXR_VisualModuleEvaluator(
            transform = img_transform,
            collate_batch_fn = mimiccxr_collate_batch_fn,
            num_workers = num_workers,
            classify_tags = classify_tags,
            rid2tags_filename = mimiccxr_rid2tags_filename,
            classify_orientation = classify_orientation,
            classify_chexpert = classify_chexpert,
            chexpert_labels_filename = mimiccxr_chexpert_labels_filename,
            classify_questions = classify_questions,
            question_labels_filename = mimiccxr_question_labels_filename,
            **mimiccxr_vision_evaluator_kwargs,
        )
    
    # Create IU X-Ray visual module trainer
    if eval_iuxray:
        count_print('Creating IU X-Ray visual module trainer ...')
        iuxray_vision_trainer = IUXRAY_VisualModuleTrainer(
            transform = img_transform,
            collate_batch_fn = iuxray_collate_batch_fn,
            num_workers = num_workers,
            classify_tags = classify_tags,
            rid2tags_filename = iuxray_rid2tags_filename,
            classify_orientation = classify_orientation,
            classify_chexpert = classify_chexpert,
            chexpert_labels_filename = iuxray_chexpert_labels_filename,
            classify_questions = classify_questions,
            question_labels_filename = iuxray_question_labels_filename,
            validation_only = True,
            **iuxray_vision_trainer_kwargs,
        )

    # Attach metrics, timer and events to engines    
    count_print('Attaching metrics, timer and events to engines ...')

    # Metrics
    if classify_tags:
        attach_medical_tags_f1score(evaluator, device)

    if classify_orientation:
        attach_dataset_aware_orientation_accuracy(evaluator)

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
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True)

    # Attach handlers
    evaluator.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation

    metrics_to_save = metrics_to_print    
    results_folder_path = get_results_folder_path(checkpoint_folder_path)    
    results_dict = {}

    if eval_iuxray:
        print('\n========================')
        count_print('Running evaluator engine on IU X-Ray validation split ...')
        print('len(dataset) =', len(iuxray_vision_trainer.val_dataset))
        print('len(dataloader) =', len(iuxray_vision_trainer.val_dataloader))
        evaluator.run(iuxray_vision_trainer.val_dataloader)    
        results_dict['iuxray_metrics'] = deepcopy(evaluator.state.metrics)            
        results_dict['iuxray_dataset'] = iuxray_vision_trainer.val_dataset
        _save_metrics(results_dict, 'iuxray', metrics_to_save, results_folder_path)

    if eval_mimiccxr:
        print('\n========================')
        count_print('Running evaluator engine on MIMIC-CXR test split ...')
        print('len(dataset) =', len(mimiccxr_vision_evaluator.test_dataset))
        print('len(dataloader) =', len(mimiccxr_vision_evaluator.test_dataloader))
        evaluator.run(mimiccxr_vision_evaluator.test_dataloader)
        results_dict['mimiccxr_metrics'] = deepcopy(evaluator.state.metrics)
        results_dict['mimiccxr_dataset'] = mimiccxr_vision_evaluator.test_dataset            
        _save_metrics(results_dict, 'mimiccxr', metrics_to_save, results_folder_path)

    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder,
    mimiccxr_preprocessed_test_data_filename,
    batch_size = 100,
    num_workers = 0,
    device = 'GPU',
    return_results = False,
    use_amp = False,
    eval_iuxray = True,
    eval_mimiccxr = True,
):
    print('----- Evaluating model ------')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)    
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    mimiccxr_vision_evaluator_kwargs = metadata['mimiccxr_vision_trainer_kwargs']
    mimiccxr_vision_evaluator_kwargs['batch_size'] = batch_size
    mimiccxr_vision_evaluator_kwargs['preprocessed_data_filename'] = mimiccxr_preprocessed_test_data_filename
    iuxray_vision_trainer_kwargs = metadata['iuxray_vision_trainer_kwargs']
    iuxray_vision_trainer_kwargs['batch_size'] = batch_size
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']

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
            )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)