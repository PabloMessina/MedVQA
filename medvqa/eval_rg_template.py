import  os
import argparse
import numpy as np
from copy import deepcopy

import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_label_names_and_templates,
)
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.models.ensemble.multilabel_ensemble_search import MultilabelOptimalEnsembleSearcher
from medvqa.models.report_generation.templates.chex_v1 import TEMPLATES_CHEXPERT_v1
from medvqa.models.vision.visual_modules import DensenetVisualModule

from medvqa.utils.constants import (
    # IUXRAY_DATASET_ID,
    CHEXPERT_LABELS,
    MIMICCXR_DATASET_ID,
    MetricNames,
)
# from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.metrics import (
    attach_chexpert_labels_prf1,
    attach_chexpert_labels_roc_auc,
    attach_medical_tags_f1score,
    attach_chexpert_labels_accuracy,
    attach_dataset_aware_orientation_accuracy,
    attach_question_labels_prf1,
    attach_chest_imagenome_labels_accuracy,
    attach_chest_imagenome_labels_prf1,
    attach_chest_imagenome_labels_roc_auc,
)
from medvqa.models.checkpoint import (
    get_checkpoint_filepath,
    load_metadata,
)
from medvqa.utils.common import (
    WORKSPACE_DIR,
    parsed_args_to_dict,
)    
from medvqa.utils.handlers import (
    get_log_metrics_handlers,
    get_log_iteration_handler,
    attach_accumulator,
)
from medvqa.utils.files import (
    get_cached_json_file,
    get_results_folder_path,
    save_to_pickle,
)
from medvqa.training.vision import get_engine
from medvqa.datasets.dataloading_utils import get_vision_collate_batch_fn
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleEvaluator
# from medvqa.datasets.iuxray.iuxray_vision_dataset_management import IUXRAY_VisualModuleTrainer
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter, print_blue
from medvqa.evaluation.report_generation import (
    TemplateBasedModes,
    compute_report_level_metrics,
    recover_reports__template_based,
)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument('--checkpoint-folder', type=str, required=True)
    parser.add_argument('--template-based-mode', type=str, required=True)

    # optional arguments
    parser.add_argument('--calibrate-thresholds', dest='calibrate_thresholds', action='store_true')
    parser.set_defaults(calibrate_thresholds=False)

    parser.add_argument('--mimiccxr-preprocessed-test-data-filename', type=str)
    parser.add_argument('--mimiccxr-preprocessed-train-data-filename', type=str)
    parser.add_argument('--iuxray-preprocessed-train-data-filename', type=str)
    parser.add_argument('--mimiccxr-qa-adapted-reports-filename', type=str)
    parser.add_argument('--iuxray-qa-adapted-reports-filename', type=str)

    parser.add_argument('--batch-size', type=int, default=140)
    parser.add_argument('--device', type=str, default='GPU')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-processes-for-chexpert-labeler', type=int, default=4)

    parser.add_argument('--eval-iuxray', dest='eval_iuxray', action='store_true')
    parser.set_defaults(eval_iuxray=False)

    parser.add_argument('--eval-mimiccxr', dest='eval_mimiccxr', action='store_true')
    parser.set_defaults(eval_mimiccxr=False)

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.set_defaults(use_amp=False)
    
    return parser.parse_args()


_BEST_CHEXPERT_ORDER = [
    'Cardiomegaly',
    'Enlarged Cardiomediastinum',
    'Consolidation',
    'Lung Opacity',
    'Atelectasis',
    'Support Devices',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Edema',
    'Lung Lesion',
    'Fracture',
]

def _compute_and_save_report_level_metrics(results_dict, dataset_name, tokenizer, results_folder_path, 
                                           max_processes, template_based_mode, calibrate_thresholds):
    metrics = compute_report_level_metrics(results_dict[f'{dataset_name}_reports']['gt_reports'],
                                           results_dict[f'{dataset_name}_reports']['gen_reports'],
                                           tokenizer, max_processes=max_processes)    
    template_based_mode = template_based_mode.replace('_','-')
    strings = ['template-based', template_based_mode]
    if calibrate_thresholds: strings.append('thrs-calib')
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_report_level_metrics(eval_mode={",".join(strings)}).pkl')
    save_to_pickle(metrics, save_path)
    print (f'Report-level metrics successfully saved to {save_path}')
    return metrics

def _recover_tokenizer_kwargs(metadata):
    return metadata['tokenizer_kwargs']

def _recover_model_kwargs(metadata):
    kwargs = metadata['model_kwargs']
    kwargs.update(metadata['auxiliary_tasks_kwargs'])
    return kwargs

def _recover_vision_dataset_manager_kwargs(
        dataset_name, metadata, batch_size, preprocessed_data_filename, qa_adapted_reports_filename):
    keys = [
        f'{dataset_name}_vision_trainer_kwargs',
        f'{dataset_name}_vqa_trainer_kwargs',
    ]
    kwargs = None
    for key in keys:
        if key in metadata:
            kwargs = metadata[key]
            break    
    assert kwargs is not None
    kwargs['batch_size'] = batch_size
    kwargs['preprocessed_data_filename'] = preprocessed_data_filename
    if qa_adapted_reports_filename is not None:
        kwargs['qa_adapted_reports_filename'] = qa_adapted_reports_filename
    return kwargs

def _recover_iuxray_vision_trainer_kwargs(
        metadata, batch_size, preprocessed_data_filename, qa_adapted_reports_filename):
    return _recover_vision_dataset_manager_kwargs('iuxray', metadata, batch_size,
         preprocessed_data_filename, qa_adapted_reports_filename)

def _recover_mimiccxr_vision_evaluator_kwargs(
        metadata, batch_size, preprocessed_data_filename, qa_adapted_reports_filename):
    return _recover_vision_dataset_manager_kwargs('mimiccxr', metadata, batch_size,
        preprocessed_data_filename, qa_adapted_reports_filename)

def _calibrate_thresholds(model, device, use_amp, mimiccxr_vision_evaluator_kwargs,
                          classify_chexpert, classify_chest_imagenome):
    assert classify_chexpert != classify_chest_imagenome # Only one of them can be True
    
    if classify_chexpert:
        labeler_name = 'chexpert'
    elif classify_chest_imagenome:
        labeler_name = 'chest_imagenome'
    else: assert False, 'This should not happen'

    # Run model on MIMICCXR validation dataset to get predictions
    assert mimiccxr_vision_evaluator_kwargs['use_validation_indices']
    mimiccxr_vision_evaluator = MIMICCXR_VisualModuleEvaluator(**mimiccxr_vision_evaluator_kwargs)
    evaluator = get_engine(model, classify_tags=False, classify_orientation=False, classify_questions=False,
                           classify_chexpert=classify_chexpert,
                           classify_chest_imagenome=classify_chest_imagenome,
                           device=device, use_amp=use_amp, training=False)
    if classify_chexpert:
        attach_chexpert_labels_accuracy(evaluator, device)
        attach_chexpert_labels_prf1(evaluator, device)
        attach_chexpert_labels_roc_auc(evaluator, 'cpu')
    elif classify_chest_imagenome:
        attach_chest_imagenome_labels_accuracy(evaluator, device)
        attach_chest_imagenome_labels_prf1(evaluator, device)
        attach_chest_imagenome_labels_roc_auc(evaluator, 'cpu')
    else: assert False
    attach_accumulator(evaluator, f'pred_{labeler_name}_probs')
    attach_accumulator(evaluator, labeler_name)
    timer = Timer()
    timer.attach(evaluator, start=Events.EPOCH_STARTED)
    metrics_to_print=[]
    if classify_chexpert:
        metrics_to_print.append(MetricNames.CHXLABEL_PRF1)
        metrics_to_print.append(MetricNames.CHXLABELACC)
        metrics_to_print.append(MetricNames.CHXLABEL_ROCAUC)
    elif classify_chest_imagenome:
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABEL_PRF1)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELACC)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELROCAUC)
    else: assert False
    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()    
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    print('Running model on MIMICCXR validation dataset ...')
    evaluator.run(mimiccxr_vision_evaluator.test_dataloader)
    # Retrieve predictions and ground truth labels
    pred_probs = evaluator.state.metrics[f'pred_{labeler_name}_probs']
    pred_probs = torch.stack(pred_probs).numpy()
    pred_probs = np.expand_dims(pred_probs, axis=0) # add extra dimension
    gt_labels = evaluator.state.metrics[labeler_name]
    gt_labels = torch.stack(gt_labels).numpy()
    print('pred_probs.shape:', pred_probs.shape)
    print('gt_labels.shape:', gt_labels.shape)
    # Search optimal thresholds
    print('Searching optimal thresholds ...')
    mloes = MultilabelOptimalEnsembleSearcher(probs=pred_probs, gt=gt_labels)
    mloes.sample_weights(n_tries=100)
    prev_score = mloes.evaluate_best_predictions()
    while True:
        mloes.sample_weights_from_previous_ones(n_tries=100, noise_coef=0.05)
        score = mloes.evaluate_best_predictions()
        if abs(score - prev_score) < 1e-3:
            break
        prev_score = score
    thresholds = mloes.compute_best_merged_probs_and_thresholds()['thresholds']
    if classify_chexpert: # Only print thresholds for Chexpert
        print('thresholds:', thresholds)
    print('Done!')
    return thresholds

def _calibrate_thresholds_using_chexpert_for_mimiccxr(model, device, use_amp, mimiccxr_vision_evaluator_kwargs):
    print_blue('Calibrating thresholds using MIMICCXR validation dataset and CheXpert labels')
    return _calibrate_thresholds(model, device, use_amp, mimiccxr_vision_evaluator_kwargs,
                                 classify_chexpert=True, classify_chest_imagenome=False)

def _calibrate_thresholds_using_chest_imagenome_for_mimiccxr(model, device, use_amp, mimiccxr_vision_evaluator_kwargs):
    print_blue('Calibrating thresholds using MIMICCXR validation dataset and Chest-ImaGenome labels')
    return _calibrate_thresholds(model, device, use_amp, mimiccxr_vision_evaluator_kwargs,
                                 classify_chexpert=False, classify_chest_imagenome=True)

def _evaluate_model(
    tokenizer_kwargs,
    model_kwargs,
    mimiccxr_vision_evaluator_kwargs,
    iuxray_vision_trainer_kwargs,
    auxiliary_tasks_kwargs,
    template_based_mode,
    num_workers=0,
    max_processes_for_chexpert_labeler=4,
    device='GPU',
    checkpoint_folder_path=None,
    use_amp=False,
    eval_iuxray=True,
    eval_mimiccxr=True,
    calibrate_thresholds=False,
    mimiccxr_preprocessed_train_data_filename=None,
    return_results=False,
):
    # Sanity checks
    assert eval_iuxray or eval_mimiccxr    
    if eval_mimiccxr and calibrate_thresholds:
        assert mimiccxr_preprocessed_train_data_filename is not None

    # Pull out some args from kwargs

    # auxiliary task: medical tags prediction
    classify_tags = auxiliary_tasks_kwargs['classify_tags']
    n_medical_tags = auxiliary_tasks_kwargs['n_medical_tags']
    iuxray_rid2tags_filename = auxiliary_tasks_kwargs.get('iuxray_rid2tags_filename', None)
    mimiccxr_rid2tags_filename = auxiliary_tasks_kwargs.get('mimiccxr_rid2tags_filename', None)
    if classify_tags:
        assert n_medical_tags is not None
        if eval_iuxray: assert iuxray_rid2tags_filename is not None
        if eval_mimiccxr: assert mimiccxr_rid2tags_filename is not None    
    # auxiliary task: orientation classification
    classify_orientation = auxiliary_tasks_kwargs['classify_orientation']
    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs.get('classify_chest_imagenome', False)
    # auxiliary task: questions classification
    classify_questions = auxiliary_tasks_kwargs.get('classify_questions', False)    
    iuxray_question_labels_filename = auxiliary_tasks_kwargs.get('iuxray_question_labels_filename', None)
    mimiccxr_question_labels_filename = auxiliary_tasks_kwargs.get('mimiccxr_question_labels_filename', None)
    if classify_questions:
        if eval_iuxray: assert iuxray_question_labels_filename is not None
        if eval_mimiccxr: assert mimiccxr_question_labels_filename is not None
    
    count_print = CountPrinter()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Load saved checkpoint    
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path = ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    # Create model
    count_print('Creating instance of DensenetVisualModule model ...')
    model = DensenetVisualModule(**model_kwargs)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Create evaluator engine
    count_print('Creating evaluator engine ...')
    evaluator = get_engine(model, classify_tags, classify_orientation, classify_chexpert,
                         classify_questions, classify_chest_imagenome, device, use_amp=use_amp, training=False)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    for key in ['use_medical_tokenization', 'medical_tokenization']: # hack to avoid passing these args to tokenizer
        if key in tokenizer_kwargs:
            del tokenizer_kwargs[key]    
    tokenizer = Tokenizer(**tokenizer_kwargs)
        
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
                                                        classify_questions = classify_questions,
                                                        classify_chest_imagenome = classify_chest_imagenome)
    # if eval_iuxray:
    #     iuxray_collate_batch_fn = get_vision_collate_batch_fn(IUXRAY_DATASET_ID,
    #                                                 classify_tags = classify_tags,
    #                                                 n_tags = n_medical_tags,
    #                                                 classify_orientation = classify_orientation,
    #                                                 classify_chexpert = classify_chexpert,
    #                                                 classify_questions = classify_questions)
    
    # Create MIMIC-CXR visual module evaluator
    if eval_mimiccxr:
        count_print('Creating MIMIC-CXR visual module evaluator ...')
        mimiccxr_vision_evaluator = MIMICCXR_VisualModuleEvaluator(
            transform = img_transform,
            collate_batch_fn = mimiccxr_collate_batch_fn,
            num_workers = num_workers,            
            **mimiccxr_vision_evaluator_kwargs,
        )
    
    # # Create IU X-Ray visual module trainer
    # if eval_iuxray:
    #     count_print('Creating IU X-Ray visual module trainer ...')
    #     iuxray_vision_trainer = IUXRAY_VisualModuleTrainer(
    #         transform = img_transform,
    #         collate_batch_fn = iuxray_collate_batch_fn,
    #         num_workers = num_workers,
    #         classify_tags = classify_tags,
    #         rid2tags_filename = iuxray_rid2tags_filename,
    #         classify_orientation = classify_orientation,
    #         classify_chexpert = classify_chexpert,
    #         chexpert_labels_filename = iuxray_chexpert_labels_filename,
    #         classify_questions = classify_questions,
    #         question_labels_filename = iuxray_question_labels_filename,
    #         validation_only = True,
    #         **iuxray_vision_trainer_kwargs,
    #     )
    
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

    if classify_chest_imagenome:
        attach_chest_imagenome_labels_accuracy(evaluator, device)
        attach_chest_imagenome_labels_prf1(evaluator, device)
        attach_chest_imagenome_labels_roc_auc(evaluator, 'cpu')

    # Accumulators
    attach_accumulator(evaluator, 'idxs')
    if classify_chexpert:
        attach_accumulator(evaluator, 'pred_chexpert_probs')
    if classify_chest_imagenome:
        attach_accumulator(evaluator, 'pred_chest_imagenome_probs')
    if return_results:
        if classify_tags:
            attach_accumulator(evaluator, 'pred_tags')
        if classify_orientation:
            attach_accumulator(evaluator, 'pred_orientation')
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
    if classify_chest_imagenome:
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABEL_PRF1)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELACC)
        metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELROCAUC)

    log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()    
    
    # Attach handlers    
    evaluator.add_event_handler(Events.EPOCH_STARTED, lambda : print('Evaluating model ...'))
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)    

    # Run evaluation
    results_dict = { 'tokenizer': tokenizer }
    results_folder_path = get_results_folder_path(checkpoint_folder_path)

    # if eval_iuxray:
    #     print('\n========================')
    #     count_print('Running evaluator engine on IU X-Ray validation split ...')
    #     print('len(dataset) =', len(iuxray_vision_trainer.val_dataset))
    #     print('len(dataloader) =', len(iuxray_vision_trainer.val_dataloader))
    #     evaluator.run(iuxray_vision_trainer.val_dataloader)

    #     iuxray_qa_reports = get_cached_json_file(iuxray_qa_adapted_reports_path)
    #     results_dict['iuxray_qa_adapted_reports_path'] = iuxray_qa_adapted_reports_path
    #     results_dict['iuxray_metrics'] = deepcopy(evaluator.state.metrics)
    #     results_dict['iuxray_dataset'] = iuxray_vision_trainer.val_dataset
    #     results_dict['iuxray_reports'] = recover_reports__template_based(
    #         results_dict['iuxray_metrics'],
    #         results_dict['iuxray_dataset'],
    #         iuxray_qa_reports,
    #         _BEST_CHEXPERT_ORDER,
    #     )
    #     results_dict['iuxray_report_metrics'] = _compute_and_save_report_level_metrics(
    #         results_dict, 'iuxray', tokenizer, results_folder_path)

    if eval_mimiccxr:
        count_print('Running evaluator engine on MIMIC-CXR test split ...')
        print('len(mimiccxr_vision_evaluator.test_dataset) =', len(mimiccxr_vision_evaluator.test_dataset))
        print('len(mimiccxr_vision_evaluator.test_dataloader) =', len(mimiccxr_vision_evaluator.test_dataloader))
        evaluator.run(mimiccxr_vision_evaluator.test_dataloader)
        mimiccxr_qa_adapted_reports_filename = mimiccxr_vision_evaluator_kwargs['qa_adapted_reports_filename']
        mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, mimiccxr_qa_adapted_reports_filename)
        mimiccxr_qa_reports = get_cached_json_file(mimiccxr_qa_adapted_reports_path)
        results_dict['mimiccxr_qa_adapted_reports_path'] = mimiccxr_qa_adapted_reports_path
        results_dict['mimiccxr_metrics'] = deepcopy(evaluator.state.metrics)
        results_dict['mimiccxr_dataset'] = mimiccxr_vision_evaluator.test_dataset

        if calibrate_thresholds:
            kwargs = mimiccxr_vision_evaluator_kwargs.copy()
            kwargs['use_validation_indices'] = True
            assert mimiccxr_preprocessed_train_data_filename is not None
            kwargs['preprocessed_data_filename'] = mimiccxr_preprocessed_train_data_filename
            kwargs['transform'] = img_transform
            kwargs['collate_batch_fn'] = mimiccxr_collate_batch_fn
            kwargs['num_workers'] = num_workers
        
        # Determine label names and order according to template-based mode
        if template_based_mode == TemplateBasedModes.CHEXPERT_LABELS:
            print('Getting label names and templates from CheXpert ...')
            label_names = CHEXPERT_LABELS
            label_order = _BEST_CHEXPERT_ORDER
            label_templates = TEMPLATES_CHEXPERT_v1
            if calibrate_thresholds:
                label_thresholds = _calibrate_thresholds_using_chexpert_for_mimiccxr(model, device, use_amp, kwargs)
            else:
                label_thresholds = np.array([0.5] * len(label_names)) # default thresholds
        elif template_based_mode == TemplateBasedModes.CHEST_IMAGENOME_LABELS:
            print('Getting label names and templates from Chest-Imagenome ...')
            label_names_filename = mimiccxr_vision_evaluator_kwargs['chest_imagenome_label_names_filename']
            label_names, label_templates = load_chest_imagenome_label_names_and_templates(label_names_filename)            
            label_order = label_names # TODO: come up with a better order
            if calibrate_thresholds:
                label_thresholds = _calibrate_thresholds_using_chest_imagenome_for_mimiccxr(model, device, use_amp, kwargs)
            else:
                label_thresholds = np.array([0.5] * len(label_names)) # default thresholds
        else:
            raise ValueError(f'Unknown template_based_mode: {template_based_mode}')

        print('Running template-based report recovery ...')
        results_dict['mimiccxr_reports'] = recover_reports__template_based(
            mode=template_based_mode,
            metrics_dict=results_dict['mimiccxr_metrics'],
            dataset=results_dict['mimiccxr_dataset'],
            qa_adapted_dataset=mimiccxr_qa_reports,
            label_names=label_names,
            label_templates=label_templates,
            label_thresholds=label_thresholds,
            label_order=label_order,
        )
        results_dict['mimiccxr_report_metrics'] = _compute_and_save_report_level_metrics(
            results_dict, 'mimiccxr', tokenizer, results_folder_path,
            max_processes_for_chexpert_labeler, template_based_mode, calibrate_thresholds)

    torch.cuda.empty_cache()
    if return_results:
        return results_dict

def evaluate_model(
    checkpoint_folder,
    template_based_mode,
    batch_size=100,
    num_workers=0,
    max_processes_for_chexpert_labeler=4,
    device='GPU',
    return_results=False,
    use_amp=False,
    eval_iuxray=True,
    eval_mimiccxr=True,
    calibrate_thresholds=False,
    mimiccxr_preprocessed_test_data_filename=None,
    mimiccxr_preprocessed_train_data_filename=None,
    iuxray_preprocessed_train_data_filename=None,
    mimiccxr_qa_adapted_reports_filename=None,
    iuxray_qa_adapted_reports_filename=None,
):
    print()
    print_blue('----- Evaluating model ------')

    # Sanity checks
    if eval_iuxray:
        assert iuxray_preprocessed_train_data_filename is not None
    if eval_mimiccxr:
        assert mimiccxr_preprocessed_test_data_filename is not None

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    # pprint(metadata)
    print()
    tokenizer_kwargs = _recover_tokenizer_kwargs(metadata)
    model_kwargs = _recover_model_kwargs(metadata)
    if eval_mimiccxr:
        mimiccxr_vision_evaluator_kwargs = _recover_mimiccxr_vision_evaluator_kwargs(
            metadata, batch_size, mimiccxr_preprocessed_test_data_filename,
            mimiccxr_qa_adapted_reports_filename)
    else:
        mimiccxr_vision_evaluator_kwargs = None
    if eval_iuxray:
        iuxray_vision_trainer_kwargs = _recover_iuxray_vision_trainer_kwargs(
            metadata, batch_size, iuxray_preprocessed_train_data_filename,
            iuxray_qa_adapted_reports_filename)
    else:
        iuxray_vision_trainer_kwargs = None
    auxiliary_tasks_kwargs = metadata['auxiliary_tasks_kwargs']

    # Temporary hack to fix a bug in the metadata (TODO: remove this)
    auxiliary_tasks_kwargs['classify_orientation'] = False
    mimiccxr_vision_evaluator_kwargs['classify_orientation'] = False

    return _evaluate_model(
                tokenizer_kwargs,
                model_kwargs,
                mimiccxr_vision_evaluator_kwargs,
                iuxray_vision_trainer_kwargs,
                auxiliary_tasks_kwargs,
                template_based_mode,
                device=device,
                num_workers=num_workers,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                checkpoint_folder_path=checkpoint_folder,
                return_results=return_results,
                use_amp=use_amp,
                eval_iuxray=eval_iuxray,
                eval_mimiccxr=eval_mimiccxr,
                calibrate_thresholds=calibrate_thresholds,
                mimiccxr_preprocessed_train_data_filename=mimiccxr_preprocessed_train_data_filename,
            )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate_model(**args)