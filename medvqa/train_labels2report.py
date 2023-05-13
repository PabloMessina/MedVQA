import  os
import argparse
import numpy as np
import gc
import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer

from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import load_chest_imagenome_label_names
from medvqa.datasets.data_inspection_utils import inspect_labels2report_trainer
from medvqa.datasets.image_processing import get_image_transform
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR, MIMICCXR_ViewModes
from medvqa.datasets.mimiccxr.mimiccxr_labels2report_dataset_management import MIMICCXR_Labels2ReportTrainer
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleTrainer
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.eval_rg_template import _compute_probs_and_gt_labels_for_mimiccxr_test_set, _find_top_k_label_indices, _recover_mimiccxr_vision_evaluator_kwargs
from medvqa.evaluation.visual_module import calibrate_thresholds_on_mimiccxr_validation_set
from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.common import load_model_state_dict
from medvqa.models.report_generation.labels2report import GenerationMode, Labels2ReportModel, NLG_Models
from medvqa.models.vision.visual_modules import MultiPurposeVisualModule

from medvqa.training.utils import append_metric_name
from medvqa.utils.constants import (
    CHEXPERT_LABELS,
    DATASET_NAMES,
    MIMICCXR_DATASET_ID,
    MetricNames,
)
from medvqa.utils.common import WORKSPACE_DIR
from medvqa.metrics import (
    attach_condition_aware_ciderd,
    attach_condition_aware_weighted_medical_completeness,
    attach_condition_aware_loss,
    attach_condition_aware_t5_report_logger,
    attach_dataset_aware_ciderd,
    attach_dataset_aware_weighted_medical_completeness,
    attach_dataset_aware_chest_imagenome_labels_auc,
    attach_dataset_aware_chest_imagenome_labels_prcauc,
    attach_dataset_aware_chexpert_labels_auc,
    attach_dataset_aware_chexpert_labels_prcauc,
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
    attach_accumulator,
    get_log_metrics_handler,
    get_log_iteration_handler,
    get_log_epoch_started_handler,
    get_lr_sch_handler,
    get_checkpoint_handler,
)
from medvqa.utils.files import (
    get_checkpoint_folder_path,
    get_file_path_with_hashing_if_too_long,
    get_results_folder_path,
    load_pickle,
    save_to_pickle,
)
from medvqa.training.labels2report import get_engine
from medvqa.training.vision import get_engine as get_vision_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    get_vision_collate_batch_fn,
    multi_cyclic_dataloaders_generator,
    get_labels2report_collate_batch_fn,
)
from medvqa.metrics.utils import (
    get_merge_metrics_fn,
    get_hybrid_score_name,
)
from medvqa.utils.logging import CountPrinter, print_blue, print_bold, print_magenta, print_normal_and_bold, print_orange, print_red

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # --- Required arguments

    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs the model will be trained')
    parser.add_argument('--batches-per-epoch', type=int, required=True, help='Number of batches per epoch')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')

    # --- Optional arguments

    parser.add_argument('--checkpoint-folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')
    parser.add_argument('--generation-mode', type=str,  default=GenerationMode.PREDICTIONS_2_REPORT,
                        choices=GenerationMode.get_all_modes(), help='Generation mode')
    parser.add_argument('--use-ensemble', action='store_true', default=False)
    parser.add_argument('--ensemble-model-checkpoint-folder-paths', type=str, nargs='+', default=None)
    parser.add_argument('--ensemble-batch-size', type=int, default=None)
    parser.add_argument('--ensemble-num-workers', type=int, default=None)
    parser.add_argument('--use-hard-predictions', action='store_true', default=False,
                        help='Models\' sigmoid outputs will be thresholded using thresholds tuned on validation set')
    parser.add_argument('--use-gt-labels', action='store_true', default=False)
    parser.add_argument('--train-on-gt-and-eval-on-predictions', action='store_true', default=False)
    parser.add_argument('--randomly-drop-labels', action='store_true', default=False)

    # Model arguments
    parser.add_argument('--nlg-model', type=str, default=NLG_Models.PYTORCH_TRANSFORMER,
                        choices=NLG_Models.get_all_models())
    parser.add_argument('--t5-model-name', type=str, default='t5-small')
    parser.add_argument('--pretrained-checkpoint-folder-path', type=str, default=None)
    parser.add_argument('--labels-hidden-dim', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--transf-dec-num-memory-vecs', type=int, default=1)
    parser.add_argument('--transf-dec-hidden-dim', type=int, default=256)
    parser.add_argument('--transf-dec-nhead', type=int, default=2)
    parser.add_argument('--transf-dec-dim-forward', type=int, default=256)
    parser.add_argument('--transf-dec-num-layers', type=int, default=2)
    parser.add_argument('--dropout-prob', type=float, default=0)

    # Tokenizer arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pre-tokenize-reports', action='store_true', default=False)
    group.add_argument('--pre-tokenize-reports-and-convert-back-to-text', action='store_true', default=False)
    parser.add_argument('--vocab-min-freq', type=int, default=10)
    parser.add_argument('--use-medical-tokenization', action='store_true', default=False)
    parser.add_argument('--medical-terms-frequency-filename', type=str, default=None)
    
    # Optimization arguments
    parser.add_argument('--optimizer-name', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='reduce-lr-on-plateau')
    parser.add_argument('--lr-decay', type=float, default=0.76, help='Learning rate decay')
    parser.add_argument('--lr-decay-patience', type=int, default=2, help='Learning rate decay patience')
    parser.add_argument('--warmup-and-decay-args', type=str, default=None)
    parser.add_argument('--warmup-and-cosine-args', type=str, default=None)
    parser.add_argument('--warmup-decay-and-cyclic-decay-args', type=str, default=None)
    parser.add_argument('--iters-to-accumulate', type=int, default=1, help='For gradient accumulation')
    parser.add_argument('--override-lr', action='store_true', default=False)
    parser.add_argument('--binary-loss-name', type=str, default='bce')
    parser.add_argument('--focal-loss-weight', type=float, default=1)
    parser.add_argument('--bce-loss-weight', type=float, default=1)
    parser.add_argument('--wbce-loss-weight', type=float, default=1)

    # Data loading arguments
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for parallel dataloading')    
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    parser.add_argument('--use-amp', action='store_true', default=False)

    # MIMIC-CXR arguments
    parser.add_argument('--use-mimiccxr', dest='train_mimiccxr', action='store_true', default=False)
    parser.add_argument('--mimiccxr-weight', type=float, default=1)
    parser.add_argument('--mimiccxr-view-mode', type=str, default='any_single', choices=MIMICCXR_ViewModes.get_all_modes())
    parser.add_argument('--mimiccxr-balanced-sampling-mode', type=str, default=None)
    parser.add_argument('--mimiccxr-balanced-batch-size', type=int, default=None)
    parser.add_argument('--mimiccxr-qa-adapted-reports-filename', type=str, default=None)
    
    # Chest ImaGenome arguments (NOTE: Chest ImaGenome is built on top of MIMIC-CXR)
    parser.add_argument('--chest-imagenome-labels-filename', type=str, default=None)
    parser.add_argument('--chest-imagenome-label-names-filename', type=str, default=None)
    parser.add_argument('--use-chest-imagenome-decent-images-only', action='store_true', default=False)

    # Checkpoint saving arguments
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    
    # Auxiliary tasks arguments
    
    parser.add_argument('--use-gender', action='store_true', default=False)
    # chexpert labels
    parser.add_argument('--use-chexpert', action='store_true', default=False)
    parser.add_argument('--mimiccxr-chexpert-labels-filename', type=str, default=None)
    # chest imagenome labels
    parser.add_argument('--use-chest-imagenome', action='store_true', default=False)
    # label filtering
    parser.add_argument('--filter-labels', action='store_true', default=False)
    parser.add_argument('--top-k-chexpert-labels', type=int, default=None, help='if None, use all labels')
    parser.add_argument('--top-k-chest-imagenome-labels', type=int, default=None, help='if None, use all labels')
    parser.add_argument('--label-score-threshold', type=float, default=None, help='if not None, keep only labels with score >= threshold')

    # debug
    parser.add_argument('--debug', action='store_true', default=False)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = {
    MetricNames.CIDER_D: 0.1,
    MetricNames.CIDER_D_GT: 0.1,
    MetricNames.WMEDCOMP: 1,
    MetricNames.WMEDCOMP_GT: 1,
    MetricNames.CHXLABEL_AUC: 1,
    MetricNames.CHXLABEL_PRCAUC: 1,
    MetricNames.CHESTIMAGENOMELABELAUC: 1,
    MetricNames.CHESTIMAGENOMELABELPRCAUC: 1,
    MetricNames.REPORT_LOSS: 1,
    MetricNames.REPORT_LOSS_GT: 1,
}

def _metric_getter(metrics_dict, key):
    if key == MetricNames.CHESTIMAGENOMELABELAUC or\
        key == MetricNames.CHESTIMAGENOMELABELPRCAUC or\
        key == MetricNames.CHXLABEL_AUC or\
        key == MetricNames.CHXLABEL_PRCAUC:
        scores = metrics_dict[key]
        return 0.5 * (scores['macro_avg'] + scores['micro_avg'])
    if '_loss' in key:
        return 1 / (1 + metrics_dict[key]) # convert loss to score
    return metrics_dict[key]

def _get_precomputed_sigmoids_filepath(model_checkpoint_path, mimiccxr_view_mode, label_name):
    strings = [model_checkpoint_path, mimiccxr_view_mode]
    filepath = get_file_path_with_hashing_if_too_long(MIMICCXR_CACHE_DIR, f'precomputed_{label_name}_sigmoids', strings)
    return filepath

def _precompute_sigmoids_for_ensemble(model_folder_paths, classify_chexpert, classify_chest_imagenome, mimiccxr_view_mode,
                                      batch_size, num_workers, device, debug=False):
    assert type(model_folder_paths) == list
    assert len(model_folder_paths) > 0
    assert classify_chexpert or classify_chest_imagenome

    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')

    # Load metadata for each checkpoint
    metadata_list = [load_metadata(checkpoint_path) for checkpoint_path in model_folder_paths]

    # Sanity check
    if classify_chexpert:
        for metadata in metadata_list:
            assert metadata['auxiliary_tasks_kwargs']['classify_chexpert']
    if classify_chest_imagenome:
        for metadata in metadata_list:
            assert metadata['auxiliary_tasks_kwargs']['classify_chest_imagenome']

    output = [{ 'model_folder_path': x } for x in model_folder_paths]

    # Load model for each checkpoint and precompute sigmoids
    for k, (metadata, model_folder_path) in enumerate(zip(metadata_list, model_folder_paths)):

        print_magenta('============================================================', bold=True)
        print_magenta(f'Model {k+1}/{len(model_folder_paths)}', bold=True)

        print_normal_and_bold('model_folder_path = ', model_folder_path)

        model_checkpoint_path = get_checkpoint_filepath(model_folder_path)
        print_normal_and_bold('model_checkpoint_path = ', model_checkpoint_path)

        skip_chexpert = True
        skip_chest_imagenome = True
        if classify_chexpert:
            precomputed_chexpert_sigmoids_path = _get_precomputed_sigmoids_filepath(model_checkpoint_path, mimiccxr_view_mode, 'chexpert')
            print_normal_and_bold('precomputed_chexpert_sigmoids_path = ', precomputed_chexpert_sigmoids_path)
            if os.path.exists(precomputed_chexpert_sigmoids_path):
                print_red('Precomputed sigmoids for CheXpert already exist.', bold=True)
                # print file size in megabytes
                print(f'File size = {os.path.getsize(precomputed_chexpert_sigmoids_path) / 1024**2:.2f} MB')
                output[k]['chexpert_sigmoids_path'] = precomputed_chexpert_sigmoids_path
            else:
                skip_chexpert = False
        if classify_chest_imagenome:
            precomputed_chest_imagenome_sigmoids_path = _get_precomputed_sigmoids_filepath(model_checkpoint_path, mimiccxr_view_mode, 'chest_imagenome')
            print_normal_and_bold('precomputed_chest_imagenome_sigmoids_path = ', precomputed_chest_imagenome_sigmoids_path)
            if os.path.exists(precomputed_chest_imagenome_sigmoids_path):
                print_red('Precomputed sigmoids for Chest ImaGenome already exist.', bold=True)
                # print file size in megabytes
                print(f'File size = {os.path.getsize(precomputed_chest_imagenome_sigmoids_path) / 1024**2:.2f} MB')
                output[k]['chest_imagenome_sigmoids_path'] = precomputed_chest_imagenome_sigmoids_path
            else:
                skip_chest_imagenome = False
        if skip_chexpert and skip_chest_imagenome:
            print_red('Skipping...', bold=True)
            continue

        # Load model
        print('Loading model...')
        model_kwargs = metadata['model_kwargs']
        model = MultiPurposeVisualModule(**model_kwargs)
        model = model.to(device)
        model_wrapper = ModelWrapper(model)
        model_wrapper.load_checkpoint(model_checkpoint_path, device, model_only=True, strict=False)

        # Create inference engine
        validator_engine_kwargs = metadata['validator_engine_kwargs']
        # Set some flags to False to make it work (TODO: make this cleaner in the future)
        validator_engine_kwargs['classify_tags'] = False
        validator_engine_kwargs['classify_orientation'] = False
        validator_engine_kwargs['classify_questions'] = False
        validator_engine_kwargs['pass_pred_bbox_coords_as_input'] = False
        inference_engine = get_vision_engine(model=model, device=device, **validator_engine_kwargs)
        
        # Define collate_batch_fn
        collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
        mimiccxr_collate_batch_fn = get_vision_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR])

        # Create MIMIC-CXR trainer
        print('Creating MIMICCXR_VisualModuleTrainer')
        val_image_transform_kwargs = metadata['val_image_transform_kwargs']
        mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
        mimiccxr_trainer_kwargs['view_mode'] = mimiccxr_view_mode
        mimiccxr_trainer_kwargs['use_all_data'] = True
        mimiccxr_trainer_kwargs['data_augmentation_enabled'] = False
        mimiccxr_trainer = MIMICCXR_VisualModuleTrainer(
            test_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
            batch_size=batch_size,
            collate_batch_fn=mimiccxr_collate_batch_fn,
            num_workers=num_workers,
            **mimiccxr_trainer_kwargs,
        )

        # dataloader
        dataloader = mimiccxr_trainer.all_dataloader

        if debug:
            # Timer
            timer = Timer()
            timer.attach(inference_engine, start=Events.EPOCH_STARTED)
            # Print metrics to check that everything is working
            _mim_datasets = [MIMICCXR_DATASET_ID]
            metrics_to_print = []
            if classify_chexpert:
                attach_dataset_aware_chexpert_labels_auc(inference_engine, _mim_datasets, 'cpu')
                attach_dataset_aware_chexpert_labels_prcauc(inference_engine, _mim_datasets, 'cpu')
                metrics_to_print.append(MetricNames.CHXLABEL_AUC)
                metrics_to_print.append(MetricNames.CHXLABEL_PRCAUC)
            if classify_chest_imagenome:
                attach_dataset_aware_chest_imagenome_labels_auc(inference_engine, _mim_datasets, 'cpu')
                attach_dataset_aware_chest_imagenome_labels_prcauc(inference_engine, _mim_datasets, 'cpu')
                metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELAUC)
                metrics_to_print.append(MetricNames.CHESTIMAGENOMELABELPRCAUC)
            # Attach handlers
            log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
            inference_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
            inference_engine.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
            log_iteration_handler = get_log_iteration_handler()
            inference_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
            # Run inference on a small sample
            print('Running inference on a small sample')
            inference_engine.run(dataloader, max_epochs=1, epoch_length=5)
        else:
            # Timer
            timer = Timer()
            timer.attach(inference_engine, start=Events.EPOCH_STARTED)            
            # Attach handlers
            log_metrics_handler = get_log_metrics_handler(timer, [])
            inference_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
            inference_engine.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
            log_iteration_handler = get_log_iteration_handler()
            inference_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)            
            # Accumulators
            attach_accumulator(inference_engine, 'idxs')
            if classify_chexpert and not skip_chexpert:
                attach_accumulator(inference_engine, 'pred_chexpert_probs')
            if classify_chest_imagenome and not skip_chest_imagenome:
                attach_accumulator(inference_engine, 'pred_chest_imagenome_probs')            
            # Run inference on the full dataset
            print('Running inference on the full dataset')
            print('len(dataloader) =', len(dataloader))
            inference_engine.run(dataloader, max_epochs=1, epoch_length=len(dataloader))

            if classify_chexpert and not skip_chexpert:
                pred_chexpert_probs = inference_engine.state.metrics['pred_chexpert_probs']
                # assert len(pred_chexpert_probs) == len(dataloader.dataset),\
                #     f'len(pred_chexpert_probs) = {len(pred_chexpert_probs)}, len(dataloader.dataset) = {len(dataloader.dataset)}'
                pred_chexpert_probs = np.stack(pred_chexpert_probs, axis=0).astype(np.float16) # cast to a small float to save space
                print('pred_chexpert_probs.shape', pred_chexpert_probs.shape)
                print('pred_chexpert_probs.dtype', pred_chexpert_probs.dtype)
                print('pred_chexpert_probs.nbytes', pred_chexpert_probs.nbytes)
                # Save
                print('Saving pred_chexpert_probs')
                data = {
                    'pred_chexpert_probs': pred_chexpert_probs,
                    'dicom_ids': [mimiccxr_trainer.dicom_ids[i] for i in inference_engine.state.metrics['idxs']],
                }
                save_to_pickle(data, precomputed_chexpert_sigmoids_path)
                print_magenta(f'Saved to {precomputed_chexpert_sigmoids_path}', bold=True)
                output[k]['chexpert_sigmoids_path'] = precomputed_chexpert_sigmoids_path

            if classify_chest_imagenome and not skip_chest_imagenome:
                pred_chest_imagenome_probs = inference_engine.state.metrics['pred_chest_imagenome_probs']
                # assert len(pred_chest_imagenome_probs) == len(dataloader.dataset),\
                #         f'len(pred_chest_imagenome_probs) = {len(pred_chest_imagenome_probs)}, len(dataloader.dataset) = {len(dataloader.dataset)}'
                pred_chest_imagenome_probs = np.stack(pred_chest_imagenome_probs, axis=0).astype(np.float16) # cast to a small float to save space
                print('pred_chest_imagenome_probs.shape', pred_chest_imagenome_probs.shape)
                print('pred_chest_imagenome_probs.dtype', pred_chest_imagenome_probs.dtype)
                print('pred_chest_imagenome_probs.nbytes', pred_chest_imagenome_probs.nbytes)
                # Save
                print('Saving pred_chest_imagenome_probs')
                data = {
                    'pred_chest_imagenome_probs': pred_chest_imagenome_probs,
                    'dicom_ids': [mimiccxr_trainer.dicom_ids[i] for i in inference_engine.state.metrics['idxs']],
                }
                save_to_pickle(data, precomputed_chest_imagenome_sigmoids_path)
                print_magenta(f'Saved to {precomputed_chest_imagenome_sigmoids_path}', bold=True)
                output[k]['chest_imagenome_sigmoids_path'] = precomputed_chest_imagenome_sigmoids_path

        # Release GPU memory
        del inference_engine
        del dataloader
        del mimiccxr_trainer
        del model_wrapper
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return output

def _get_model_and_device_getter(checkpoint_folder_path, model_kwargs, device):
    def get_model_and_device():
        # Load saved checkpoint    
        checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
        print('Loading model from checkpoint ...')
        print('checkpoint_path = ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        # Load model
        print('Loading model...')
        model = MultiPurposeVisualModule(**model_kwargs)
        model = model.to(device)
        model.load_state_dict(checkpoint['model'], strict=False)
        return model, device
    return get_model_and_device

def _precompute_thresholds_for_ensemble(model_folder_paths, classify_chexpert, classify_chest_imagenome,
                                        batch_size, num_workers, device):
    assert type(model_folder_paths) == list
    assert len(model_folder_paths) > 0
    assert classify_chexpert or classify_chest_imagenome

    # Load metadata for each checkpoint
    metadata_list = [load_metadata(checkpoint_path) for checkpoint_path in model_folder_paths]

    # Sanity check
    if classify_chexpert:
        for metadata in metadata_list:
            assert metadata['auxiliary_tasks_kwargs']['classify_chexpert']
    if classify_chest_imagenome:
        for metadata in metadata_list:
            assert metadata['auxiliary_tasks_kwargs']['classify_chest_imagenome']

    output = [None] * len(model_folder_paths) # initialize output

    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')

    # Load model for each checkpoint and precompute thresholds
    for k, (metadata, model_folder_path) in enumerate(zip(metadata_list, model_folder_paths)):

        print_magenta('============================================================', bold=True)
        print_magenta(f'Model {k+1}/{len(model_folder_paths)}', bold=True)

        print_normal_and_bold('model_folder_path = ', model_folder_path)

        model_checkpoint_path = get_checkpoint_filepath(model_folder_path)
        print_normal_and_bold('model_checkpoint_path = ', model_checkpoint_path)
        
        results_folder_path = get_results_folder_path(model_folder_path)
        print_normal_and_bold('results_folder_path = ', results_folder_path)

        _get_model_and_device = _get_model_and_device_getter(
            checkpoint_folder_path=model_folder_path,
            model_kwargs=metadata['model_kwargs'],
            device=device,
        )
        mimiccxr_vision_evaluator_kwargs = _recover_mimiccxr_vision_evaluator_kwargs(
            metadata=metadata, batch_size=batch_size, num_workers=num_workers,
        )
        thresholds_dict = calibrate_thresholds_on_mimiccxr_validation_set(
            model_and_device_getter=_get_model_and_device,
            use_amp=metadata['trainer_engine_kwargs']['use_amp'],
            mimiccxr_vision_evaluator_kwargs=mimiccxr_vision_evaluator_kwargs,
            classify_chexpert=classify_chexpert,
            classify_chest_imagenome=classify_chest_imagenome,
            cache_thresholds=True,
            results_folder_path=results_folder_path,
            return_filepaths_instead=True,
        )
        output[k] = thresholds_dict
        # Release GPU memory (just in case)
        gc.collect()
        torch.cuda.empty_cache()
    
    return output

def _get_top_k_label_indices_save_path(results_folder_path, labeler_name, num_actual_labels, num_tot_labels,
                                       label_score_threshold, score_name):
    assert num_actual_labels <= num_tot_labels
    assert 0.0 <= label_score_threshold <= 1.0
    strings = [labeler_name, f'top_{num_actual_labels}_out_of_{num_tot_labels}_labels', score_name]
    if label_score_threshold is not None:
        strings.append(f'(score_threshold_{label_score_threshold:.2f})')
    strings.append('indices.pkl')
    return os.path.join(results_folder_path, '_'.join(strings))
    
def _precompute_filtered_labels_for_ensemble(model_folder_paths, classify_chexpert, classify_chest_imagenome,
                                             batch_size, num_workers, thresholds_paths,
                                             top_k_chexpert_labels, top_k_chest_imagenome_labels,
                                             label_score_threshold, device):
    assert type(model_folder_paths) == list
    assert len(model_folder_paths) > 0
    assert classify_chexpert or classify_chest_imagenome

    # Load metadata for each checkpoint
    metadata_list = [load_metadata(checkpoint_path) for checkpoint_path in model_folder_paths]


    # Sanity check
    if classify_chexpert:
        for metadata in metadata_list:
            assert metadata['auxiliary_tasks_kwargs']['classify_chexpert']
    if classify_chest_imagenome:
        for metadata in metadata_list:
            assert metadata['auxiliary_tasks_kwargs']['classify_chest_imagenome']

    output = [{} for _ in range(len(model_folder_paths))] # initialize output

    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')

    # Load model for each checkpoint and precompute thresholds
    for k, (metadata, model_folder_path) in enumerate(zip(metadata_list, model_folder_paths)):

        print_magenta('============================================================', bold=True)
        print_magenta(f'Model {k+1}/{len(model_folder_paths)}', bold=True)

        print_normal_and_bold('model_folder_path = ', model_folder_path)
        
        results_folder_path = get_results_folder_path(model_folder_path)
        print_normal_and_bold('results_folder_path = ', results_folder_path)
        
        mimiccxr_vision_evaluator_kwargs = _recover_mimiccxr_vision_evaluator_kwargs(
            metadata=metadata, batch_size=batch_size, num_workers=num_workers,
        )
        tmp = _compute_probs_and_gt_labels_for_mimiccxr_test_set(
            auxiliary_tasks_kwargs=metadata['auxiliary_tasks_kwargs'],
            checkpoint_folder_path=model_folder_path,
            model_kwargs=metadata['model_kwargs'],
            mimiccxr_vision_evaluator_kwargs=mimiccxr_vision_evaluator_kwargs,
            use_amp=metadata['trainer_engine_kwargs']['use_amp'],
            save_probs_and_gt_labels=True,
        )
        if classify_chexpert:
            dicom_id_to_pred_chexpert_probs = tmp['dicom_id_to_pred_chexpert_probs']
            dicom_id_to_gt_chexpert_labels = tmp['dicom_id_to_gt_chexpert_labels']
            chexpert_thresholds = load_pickle(thresholds_paths[k]['chexpert'])
            top_k_label_indices = _find_top_k_label_indices(dicom_id_to_gt_chexpert_labels,
                                                            dicom_id_to_pred_chexpert_probs,
                                                            chexpert_thresholds, top_k_chexpert_labels,
                                                            CHEXPERT_LABELS, 'f1', label_score_threshold)
            save_path = _get_top_k_label_indices_save_path(results_folder_path, 'chexpert',
                                                           len(top_k_label_indices), len(CHEXPERT_LABELS),
                                                           label_score_threshold, 'f1')
            output[k]['chexpert_indices_path'] = save_path
            output[k]['chexpert_num_indices'] = len(top_k_label_indices)
            save_to_pickle(top_k_label_indices, save_path)
        if classify_chest_imagenome:
            dicom_id_to_pred_chest_imagenome_probs = tmp['dicom_id_to_pred_chest_imagenome_probs']
            dicom_id_to_gt_chest_imagenome_labels = tmp['dicom_id_to_gt_chest_imagenome_labels']
            chest_imagenome_thresholds = load_pickle(thresholds_paths[k]['chest_imagenome'])
            chest_imagenome_label_names_filename = mimiccxr_vision_evaluator_kwargs['chest_imagenome_label_names_filename']
            chest_imagenome_label_names = load_chest_imagenome_label_names(chest_imagenome_label_names_filename,
                                                                           apply_anatomy_reordering=True)
            top_k_label_indices = _find_top_k_label_indices(dicom_id_to_gt_chest_imagenome_labels,
                                                            dicom_id_to_pred_chest_imagenome_probs,
                                                            chest_imagenome_thresholds, top_k_chest_imagenome_labels,
                                                            chest_imagenome_label_names, 'f1', label_score_threshold)
            save_path = _get_top_k_label_indices_save_path(results_folder_path, 'chest_imagenome',
                                                           len(top_k_label_indices), len(chest_imagenome_label_names),
                                                           label_score_threshold, 'f1')
            output[k]['chest_imagenome_indices_path'] = save_path
            output[k]['chest_imagenome_num_indices'] = len(top_k_label_indices)
            save_to_pickle(top_k_label_indices, save_path)
    
    return output

def _compute_number_of_labels_and_ranges_for_ensemble(
        model_folder_paths, use_chexpert, use_chest_imagenome, chest_imagenome_label_names_filename):
    num_input_labels = 0
    num_output_labels = 0
    chest_imagenome_range = None
    chexpert_range = None
    if use_chest_imagenome:
        aux = len(load_chest_imagenome_label_names(chest_imagenome_label_names_filename))
        num_input_labels += aux * len(model_folder_paths) # each model has a sigmoid for each chest imagenome label
        num_output_labels += aux # the ensemble has a sigmoid for each chest imagenome label
        chest_imagenome_range = (num_output_labels - aux, num_output_labels)
    if use_chexpert:
        num_input_labels += len(CHEXPERT_LABELS) * len(model_folder_paths) # each model has a sigmoid for each chexpert label
        num_output_labels += len(CHEXPERT_LABELS) # the ensemble has a sigmoid for each chexpert label
        chexpert_range = (num_output_labels - len(CHEXPERT_LABELS), num_output_labels)
    # sanity checks
    for model_folder_path in model_folder_paths:
        metadata = load_metadata(model_folder_path)
        if use_chexpert:
            assert metadata['auxiliary_tasks_kwargs']['classify_chexpert']
        if use_chest_imagenome:
            assert metadata['auxiliary_tasks_kwargs']['classify_chest_imagenome']
            aux = metadata['mimiccxr_trainer_kwargs']['chest_imagenome_label_names_filename']
            assert aux == chest_imagenome_label_names_filename, \
                    f'found {aux} but expected {chest_imagenome_label_names_filename}'
    return {
        'num_input_labels': num_input_labels,
        'num_output_labels': num_output_labels,
        'chest_imagenome_range': chest_imagenome_range,
        'chexpert_range': chexpert_range,
    }

def train_model(
        model_kwargs,
        tokenizer_kwargs,
        optimizer_kwargs,
        lr_scheduler_kwargs,
        mimiccxr_trainer_kwargs,
        dataloading_kwargs,
        collate_batch_fn_kwargs,
        training_kwargs,
        trainer_engine_kwargs,
        validator_engine_kwargs,
        auxiliary_tasks_kwargs,
        epochs,
        batches_per_epoch,
        num_workers,
        device='GPU',
        checkpoint_folder_path=None,
        save=True,
        override_lr=False,
        debug=False,
        ):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    batch_size = dataloading_kwargs['batch_size']
    train_mimiccxr = training_kwargs['train_mimiccxr']
    support_two_label_sources = training_kwargs['support_two_label_sources']
    train_on_gt_and_eval_on_predictions = training_kwargs['train_on_gt_and_eval_on_predictions']
    randomly_drop_labels = training_kwargs['randomly_drop_labels']
    use_t5 = training_kwargs['use_t5']

    if train_on_gt_and_eval_on_predictions:
        assert not support_two_label_sources
    
    # auxiliary task: chexpert labels
    classify_chexpert = auxiliary_tasks_kwargs['classify_chexpert']
    # auxiliary task: chest imagenome labels
    classify_chest_imagenome = auxiliary_tasks_kwargs['classify_chest_imagenome']

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Init tokenizer
    count_print('Initializing tokenizer ...')
    tokenizer = Tokenizer(**tokenizer_kwargs)
    tokenizer_kwargs['vocab_filepath'] = tokenizer.vocab_filepath # Remember vocab filepath in case we need to reload tokenizer

    # Create model
    count_print('Creating instance of Labels2ReportModel ...')
    model = Labels2ReportModel(vocab_size=tokenizer.vocab_size,
                               start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                               device=device, **model_kwargs)
    model = model.to(device)

    # Check dataset weights
    if dataloading_kwargs['mimiccxr_weight'] == 0:
        train_mimiccxr = False

    # Optimizer
    count_print('Defining optimizer ...')
    optimizer = create_optimizer(params=model.parameters(), **optimizer_kwargs)

    # Learning rate scheduler
    count_print('Defining scheduler ...')
    lr_scheduler, update_lr_batchwise = create_lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)

    # Create trainer and validator engines
    count_print('Creating trainer and validator engines ...')
    if use_t5:
        from transformers import T5Tokenizer
        t5_tokenizer = T5Tokenizer.from_pretrained(model_kwargs['t5_model_name'])
        _tokenizer = t5_tokenizer
    else:
        _tokenizer = tokenizer
    trainer_engine = get_engine(model=model, tokenizer=_tokenizer, optimizer=optimizer, device=device, 
                                update_lr_batchwise=update_lr_batchwise, lr_scheduler=lr_scheduler, **trainer_engine_kwargs)
    validator_engine = get_engine(model=model, tokenizer=_tokenizer, device=device, **validator_engine_kwargs)
    
    # Define collate_batch_fn
    count_print('Defining collate_batch_fn ...')
    if train_mimiccxr:
        if train_on_gt_and_eval_on_predictions:
            mimiccxr_collate_batch_fn = get_labels2report_collate_batch_fn(
                **collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR], flag='pred') # We add the flag 'pred' to the kwargs
        else:
            mimiccxr_collate_batch_fn = get_labels2report_collate_batch_fn(**collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR])

    # Create MIMIC-CXR trainer
    if train_mimiccxr:
        count_print('Creating MIMIC-CXR_Labels2ReportTrainer ...')
        mimiccxr_trainer = MIMICCXR_Labels2ReportTrainer(
            tokenizer=tokenizer,
            batch_size=batch_size,
            collate_batch_fn=mimiccxr_collate_batch_fn,            
            num_workers=num_workers,
            use_val_set_only=train_on_gt_and_eval_on_predictions,
            **mimiccxr_trainer_kwargs,
        )
        if support_two_label_sources or train_on_gt_and_eval_on_predictions:
            collate_batch_fn_kwargs_2 = collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR].copy()
            assert collate_batch_fn_kwargs_2['use_ground_truth_as_prediction'] == False
            collate_batch_fn_kwargs_2['use_ground_truth_as_prediction'] = True
            collate_batch_fn_kwargs_2['is_second_label_source'] = support_two_label_sources
            if train_on_gt_and_eval_on_predictions:
                collate_batch_fn_kwargs_2['flag'] = 'gt'
            if randomly_drop_labels:
                mimiccxr_train_collate_batch_fn_2 = get_labels2report_collate_batch_fn(
                    **collate_batch_fn_kwargs_2, randomly_drop_labels=True)
                mimiccxr_eval_collate_batch_fn_2 = get_labels2report_collate_batch_fn(
                    **collate_batch_fn_kwargs_2, randomly_drop_labels=False)
            else:
                mimiccxr_train_collate_batch_fn_2 = get_labels2report_collate_batch_fn(**collate_batch_fn_kwargs_2)
                mimiccxr_eval_collate_batch_fn_2 = mimiccxr_train_collate_batch_fn_2
            assert mimiccxr_trainer_kwargs['use_ensemble_predictions']         
            mimiccxr_trainer_kwargs_2 = mimiccxr_trainer_kwargs.copy()
            mimiccxr_trainer_kwargs_2['use_ensemble_predictions'] = False
            mimiccxr_trainer_kwargs_2['view_mode'] = MIMICCXR_ViewModes.ANY_SINGLE
            mimiccxr_trainer_kwargs_2['use_decent_images_only'] = False
            mimiccxr_trainer_kwargs_2['use_hard_predictions'] = False
            mimiccxr_trainer_kwargs_2['filter_labels'] = False
            print_blue('Creating MIMIC-CXR_Labels2ReportTrainer for second label source ...', bold=True)
            reorder_labels = False
            if train_on_gt_and_eval_on_predictions:
                if mimiccxr_trainer_kwargs['use_chest_imagenome']:
                    assert len(mimiccxr_trainer.ensemble_chest_imagenome_label_orders) == 1
                    assert mimiccxr_trainer.ensemble_chest_imagenome_label_names_filenames[0] ==\
                        mimiccxr_trainer_kwargs_2['chest_imagenome_label_names_filename']
                    reorder_labels = True
            mimiccxr_trainer_2 = MIMICCXR_Labels2ReportTrainer(
                tokenizer=tokenizer,
                batch_size=batch_size,
                collate_batch_fn=None,
                train_collate_batch_fn=mimiccxr_train_collate_batch_fn_2,
                eval_collate_batch_fn=mimiccxr_eval_collate_batch_fn_2,
                num_workers=num_workers,
                reorder_chest_imagenome_labels=reorder_labels,
                **mimiccxr_trainer_kwargs_2,
            )

    if debug: # if debugging
        output = {}
        if train_mimiccxr:
            output['mimiccxr_trainer'] = mimiccxr_trainer
            print('mimiccxr_trainer.train_dataset[0] =', mimiccxr_trainer.train_dataset[0])
            inspect_labels2report_trainer(mimiccxr_trainer, 'train_dataset', 0)
            print()
            print('mimiccxr_trainer.val_dataset[0] =', mimiccxr_trainer.val_dataset[0])
            inspect_labels2report_trainer(mimiccxr_trainer, 'val_dataset', 0)
        return output

    # Create complex dataloaders
    count_print('Creating dataloaders ...')
    
    _train_weights = []
    _train_dataloaders = []
    _val_dataloaders = []
    _dataset_names = []

    if train_mimiccxr:
        if not train_on_gt_and_eval_on_predictions:
            _dataset_names.append('mim')
            _train_weights.append(dataloading_kwargs['mimiccxr_weight'])
            _train_dataloaders.append(mimiccxr_trainer.train_dataloader)
        _val_dataloaders.append(mimiccxr_trainer.val_dataloader)
        if support_two_label_sources or train_on_gt_and_eval_on_predictions:
            _dataset_names.append('mim(gt)')
            _train_weights.append(dataloading_kwargs['mimiccxr_weight'])
            _train_dataloaders.append(mimiccxr_trainer_2.train_dataloader)
            _val_dataloaders.append(mimiccxr_trainer_2.val_dataloader)
    
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

    _mim_datasets = [MIMICCXR_DATASET_ID]
    _chexpert_labels_datasets = _mim_datasets

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []

    attach_loss('loss', trainer_engine, device)
    # for logging
    metrics_to_print.append('loss')

    if train_mimiccxr:
        if support_two_label_sources or train_on_gt_and_eval_on_predictions:
            if support_two_label_sources:
                src_cond_fn_1 = lambda output: output['dataset_id'] in _mim_datasets and output['is_second_label_source'] == False
                src_cond_fn_2 = lambda output: output['dataset_id'] in _mim_datasets and output['is_second_label_source'] == True
            else:
                src_cond_fn_1 = lambda output: output['dataset_id'] in _mim_datasets and output['flag'] == 'pred'
                src_cond_fn_2 = lambda output: output['dataset_id'] in _mim_datasets and output['flag'] == 'gt'
            if not use_t5:
                if support_two_label_sources:
                    attach_condition_aware_weighted_medical_completeness(
                        trainer_engine, tokenizer, field='reports', condition_function=src_cond_fn_1)
                attach_condition_aware_weighted_medical_completeness(
                    trainer_engine, tokenizer, field='reports', condition_function=src_cond_fn_2, metric_name=MetricNames.WMEDCOMP_GT)
                attach_condition_aware_weighted_medical_completeness(
                    validator_engine, tokenizer, field='reports', condition_function=src_cond_fn_1)
                attach_condition_aware_weighted_medical_completeness(
                    validator_engine, tokenizer, field='reports', condition_function=src_cond_fn_2, metric_name=MetricNames.WMEDCOMP_GT)
                attach_condition_aware_ciderd(
                    validator_engine, field='reports', condition_function=src_cond_fn_1)
                attach_condition_aware_ciderd(
                    validator_engine, field='reports', condition_function=src_cond_fn_2, metric_name=MetricNames.CIDER_D_GT)
                attach_dataset_aware_loss(trainer_engine, 'report_loss', _mim_datasets)
            else:
                if support_two_label_sources:
                    attach_condition_aware_loss(trainer_engine, 'report_loss', condition_function=src_cond_fn_1)
                attach_condition_aware_loss(trainer_engine, 'report_loss', condition_function=src_cond_fn_2, metric_name='report_loss_gt')
                attach_condition_aware_loss(validator_engine, 'report_loss', condition_function=src_cond_fn_1)
                attach_condition_aware_loss(validator_engine, 'report_loss', condition_function=src_cond_fn_2, metric_name='report_loss_gt')
                attach_condition_aware_t5_report_logger(trainer_engine, t5_tokenizer, condition_function=src_cond_fn_2)
                attach_condition_aware_t5_report_logger(validator_engine, t5_tokenizer, condition_function=src_cond_fn_1)
            # for logging
            if support_two_label_sources:
                if not use_t5:
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.WMEDCOMP)
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CIDER_D, train=False)
                    metrics_to_print.append(MetricNames.WMEDCOMP_GT)
                    metrics_to_print.append(MetricNames.CIDER_D_GT)
                    metrics_to_print.append('report_loss')
                else:
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'report_loss')
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'report_loss_gt')
            else:
                if not use_t5:
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.WMEDCOMP, train=False)
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.WMEDCOMP_GT)
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CIDER_D, train=False)
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CIDER_D_GT, train=False)
                    metrics_to_print.append('report_loss')
                else:
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'report_loss', train=False)
                    append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'report_loss_gt')
            
        else:
            attach_dataset_aware_weighted_medical_completeness(trainer_engine, tokenizer, _mim_datasets, field='reports')
            attach_dataset_aware_weighted_medical_completeness(validator_engine, tokenizer, _mim_datasets, field='reports')
            attach_dataset_aware_ciderd(validator_engine, _mim_datasets, field='reports')
            attach_dataset_aware_loss(trainer_engine, 'report_loss', _mim_datasets)
            # for logging
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.WMEDCOMP)
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CIDER_D, train=False)
            metrics_to_print.append('report_loss')
    
    if classify_chexpert:
        attach_dataset_aware_chexpert_labels_auc(validator_engine, _chexpert_labels_datasets, 'cpu')
        attach_dataset_aware_chexpert_labels_prcauc(trainer_engine, _chexpert_labels_datasets, 'cpu')
        attach_dataset_aware_chexpert_labels_prcauc(validator_engine, _chexpert_labels_datasets, 'cpu')        
        attach_dataset_aware_loss(trainer_engine, MetricNames.CHEXPERT_LOSS, _chexpert_labels_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHXLABEL_AUC, train=False)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHXLABEL_PRCAUC)
        metrics_to_print.append(MetricNames.CHEXPERT_LOSS)

    if classify_chest_imagenome:
        attach_dataset_aware_chest_imagenome_labels_auc(validator_engine, _mim_datasets, 'cpu')
        attach_dataset_aware_chest_imagenome_labels_prcauc(trainer_engine, _mim_datasets, 'cpu')
        attach_dataset_aware_chest_imagenome_labels_prcauc(validator_engine, _mim_datasets, 'cpu')
        attach_dataset_aware_loss(trainer_engine, MetricNames.CHEST_IMAGENOME_LABEL_LOSS, _mim_datasets)
        # for logging
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMELABELAUC, train=False)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, MetricNames.CHESTIMAGENOMELABELPRCAUC)
        metrics_to_print.append(MetricNames.CHEST_IMAGENOME_LABEL_LOSS)
    
    # Timer
    timer = Timer()
    timer.attach(trainer_engine, start=Events.EPOCH_STARTED)
    timer.attach(validator_engine, start=Events.EPOCH_STARTED)

    # Score function
    assert len(val_metrics_to_merge) > 0
    if len(train_metrics_to_merge) > 0:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0.1, 0.9, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(trainer_engine.state.metrics, validator_engine.state.metrics)
    else:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0, 1, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(validator_engine.state.metrics)

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
            checkpoint_folder_path = get_checkpoint_folder_path('report_gen', merged_dataset_name, model.get_name(),
                f'dws={",".join(map(str, _train_weights))}' if len(_train_weights) > 1 else None,
            )
            print_red('checkpoint_folder_path =', checkpoint_folder_path)
            save_metadata(checkpoint_folder_path,
                        tokenizer_kwargs=tokenizer_kwargs,
                        model_kwargs=model_kwargs,
                        optimizer_kwargs=optimizer_kwargs,
                        lr_scheduler_kwargs=lr_scheduler_kwargs,
                        mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                        dataloading_kwargs=dataloading_kwargs,
                        collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                        training_kwargs=training_kwargs,
                        trainer_engine_kwargs=trainer_engine_kwargs,
                        validator_engine_kwargs=validator_engine_kwargs,
                        auxiliary_tasks_kwargs=auxiliary_tasks_kwargs)
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
        checkpoint_handler = get_checkpoint_handler(model_wrapper, checkpoint_folder_path, trainer_engine,
                                                    epoch_offset=model_wrapper.get_epoch(),
                                                    score_name=get_hybrid_score_name(train_metrics_to_merge, val_metrics_to_merge),
                                                    score_fn=score_fn)

    # Logging
    count_print('Defining log_metrics_handler ...')

    log_metrics_handler = get_log_metrics_handler(timer,
                                                   metrics_to_print=metrics_to_print,
                                                   log_to_disk=save,
                                                   checkpoint_folder=checkpoint_folder_path)
    log_iteration_handler = get_log_iteration_handler()    
    
    # Attach handlers
    trainer_engine.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler(model_wrapper))
    trainer_engine.add_event_handler(Events.EPOCH_STARTED, lambda : print(f'(1) Training stage (lr = {optimizer.param_groups[0]["lr"]:.6f}) ...'))
    trainer_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda : validator_engine.run(val_dataloader,
                                     max_epochs=1, epoch_length=val_dataloader_size))
    validator_engine.add_event_handler(Events.EPOCH_STARTED, lambda : print('(2) Validation stage ...'))
    validator_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    validator_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    if not update_lr_batchwise:
        validator_engine.add_event_handler(Events.EPOCH_COMPLETED, lr_sch_handler)
    if save: # only if we want to save checkpoints to disk
        validator_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    # Start training
    count_print('Running trainer engine ...')
    trainer_engine.run(train_dataloader, max_epochs=epochs, epoch_length=batches_per_epoch)


def train_from_scratch(
    # Model args
    nlg_model,
    t5_model_name,
    pretrained_checkpoint_folder_path,
    embedding_dim,
    labels_hidden_dim,
    transf_dec_num_memory_vecs,
    transf_dec_hidden_dim,
    transf_dec_nhead,
    transf_dec_dim_forward,
    transf_dec_num_layers,
    dropout_prob,
    # Tokenizer args
    pre_tokenize_reports,
    pre_tokenize_reports_and_convert_back_to_text,
    vocab_min_freq,
    use_medical_tokenization,
    medical_terms_frequency_filename,
    # Optimizer args
    optimizer_name,
    lr,
    # lr_scheduler args
    scheduler,
    lr_decay,
    lr_decay_patience,
    warmup_and_decay_args,
    warmup_and_cosine_args,
    warmup_decay_and_cyclic_decay_args,
    # Dataset args
    mimiccxr_view_mode,
    mimiccxr_qa_adapted_reports_filename,
    chest_imagenome_labels_filename,
    chest_imagenome_label_names_filename,
    use_chest_imagenome_decent_images_only,
    # Dataloading args
    batch_size,
    num_workers,
    mimiccxr_weight,
    mimiccxr_balanced_sampling_mode,
    mimiccxr_balanced_batch_size,
    # Fixed traning args
    train_mimiccxr,
    binary_loss_name,
    focal_loss_weight,
    bce_loss_weight,
    wbce_loss_weight,
    use_amp,
    iters_to_accumulate,
    generation_mode,
    use_hard_predictions,
    # Label source args (ensemble and/or ground truth)
    use_ensemble,
    ensemble_model_checkpoint_folder_paths,
    ensemble_batch_size,
    ensemble_num_workers,
    use_gt_labels,
    train_on_gt_and_eval_on_predictions,
    randomly_drop_labels,
    # Variable traning args
    epochs,
    batches_per_epoch,
    # Auxiliary tasks args
    use_gender,
    use_chexpert,
    mimiccxr_chexpert_labels_filename,
    use_chest_imagenome,
    filter_labels,
    top_k_chexpert_labels,
    top_k_chest_imagenome_labels,
    label_score_threshold,
    # GPU
    device,
    # Other args
    save,
    debug = False,
):
    print_blue('----- Training model from scratch ------', bold=True)

    assert train_mimiccxr, 'No dataset selected for training'

    assert use_ensemble or use_gt_labels, 'No label source selected for training'

    use_t5 = nlg_model == NLG_Models.T5
    
    if use_t5:
        assert t5_model_name is not None

    if use_ensemble:
        assert ensemble_model_checkpoint_folder_paths is not None
        assert ensemble_batch_size is not None
        assert ensemble_num_workers is not None
        assert not use_gender, 'Not supported yet'
    
    if filter_labels:
        assert use_ensemble
        assert use_hard_predictions
        assert use_chexpert or use_chest_imagenome
        assert top_k_chexpert_labels is not None or top_k_chest_imagenome_labels is not None or \
            label_score_threshold is not None

    if train_on_gt_and_eval_on_predictions:
        assert use_gt_labels
        assert use_ensemble
        assert use_hard_predictions
        assert use_chexpert or use_chest_imagenome

    if randomly_drop_labels:
        assert train_on_gt_and_eval_on_predictions
        assert not use_t5

    if generation_mode == GenerationMode.PREDICTIONS_2_REFINED_PREDICTIONS_2_REPORT:
        classify_chexpert = use_chexpert
        classify_chest_imagenome = use_chest_imagenome
    else:
        classify_chexpert = False
        classify_chest_imagenome = False

    tokenizer_kwargs = dict(
        vocab_min_freq=vocab_min_freq,
        use_medical_tokenization=use_medical_tokenization,
        medical_terms_frequency_filename=medical_terms_frequency_filename,
    )
    if train_mimiccxr:
        assert mimiccxr_qa_adapted_reports_filename is not None
        mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, mimiccxr_qa_adapted_reports_filename)
        tokenizer_kwargs['qa_adapted_dataset_paths'] = [mimiccxr_qa_adapted_reports_path]
    
    if use_ensemble:
        print_bold('Computing number of labels and ranges for ensemble ...')
        tmp = _compute_number_of_labels_and_ranges_for_ensemble(
            ensemble_model_checkpoint_folder_paths, use_chexpert, use_chest_imagenome, chest_imagenome_label_names_filename)
        ensemble_num_input_labels = tmp['num_input_labels']
        ensemble_num_output_labels = tmp['num_output_labels']
        ensemble_chest_imagenome_range = tmp['chest_imagenome_range']
        ensemble_chexpert_range = tmp['chexpert_range']
        if ensemble_chest_imagenome_range is not None:
            print(f'ensemble_chest_imagenome_range: {ensemble_chest_imagenome_range}')
        if ensemble_chexpert_range is not None:
            print(f'ensemble_chexpert_range: {ensemble_chexpert_range}')
    if use_gt_labels:
        print_bold('Computing number of labels and ranges assuming ground truth labels as input ...')
        gt_num_input_labels = 0
        gt_num_output_labels = 0
        if use_gender:
            gt_num_input_labels += 2 # M, F
        if use_chest_imagenome:
            assert chest_imagenome_label_names_filename is not None
            n_chest_imagenome_labels = len(load_chest_imagenome_label_names(chest_imagenome_label_names_filename))
            gt_num_input_labels += n_chest_imagenome_labels
            gt_num_output_labels += n_chest_imagenome_labels
            gt_chest_imagenome_range = (gt_num_output_labels - n_chest_imagenome_labels, gt_num_output_labels)
            print(f'gt_chest_imagenome_range: {gt_chest_imagenome_range}')
        else:
            gt_chest_imagenome_range = None
        if use_chexpert:
            gt_num_input_labels += len(CHEXPERT_LABELS)
            gt_num_output_labels += len(CHEXPERT_LABELS)
            gt_chexpert_range = (gt_num_output_labels - len(CHEXPERT_LABELS), gt_num_output_labels)
            print(f'gt_chexpert_range: {gt_chexpert_range}')
        else:
            gt_chexpert_range = None
    
    support_two_label_sources = use_ensemble and use_gt_labels and not train_on_gt_and_eval_on_predictions
    
    if support_two_label_sources:
        num_input_labels = ensemble_num_input_labels
        num_output_labels = ensemble_num_output_labels
        num_input_labels_2 = gt_num_input_labels
        num_output_labels_2 = gt_num_output_labels
        assert ensemble_chexpert_range == gt_chexpert_range
        assert ensemble_chest_imagenome_range == gt_chest_imagenome_range
        chexpert_range = ensemble_chexpert_range
        chest_imagenome_range = ensemble_chest_imagenome_range
    elif train_on_gt_and_eval_on_predictions:
        assert ensemble_num_input_labels == gt_num_input_labels
        assert ensemble_num_output_labels == gt_num_output_labels
        assert ensemble_chexpert_range == gt_chexpert_range
        assert ensemble_chest_imagenome_range == gt_chest_imagenome_range
        num_input_labels = ensemble_num_input_labels
        num_output_labels = ensemble_num_output_labels
        num_input_labels_2 = None
        num_output_labels_2 = None
        chexpert_range = ensemble_chexpert_range
        chest_imagenome_range = ensemble_chest_imagenome_range
    elif use_ensemble:
        num_input_labels = ensemble_num_input_labels
        num_output_labels = ensemble_num_output_labels
        num_input_labels_2 = None
        num_output_labels_2 = None
        chexpert_range = ensemble_chexpert_range
        chest_imagenome_range = ensemble_chest_imagenome_range
    elif use_gt_labels:
        num_input_labels = gt_num_input_labels
        num_output_labels = gt_num_output_labels
        num_input_labels_2 = None
        num_output_labels_2 = None
        chexpert_range = gt_chexpert_range
        chest_imagenome_range = gt_chest_imagenome_range
    else: assert False
        
    model_kwargs = dict(
        gen_mode=generation_mode,
        nlg_model=nlg_model,
        t5_model_name=t5_model_name,
        pretrained_checkpoint_folder_path=pretrained_checkpoint_folder_path,
        embedding_dim=embedding_dim,
        num_input_labels=num_input_labels,
        labels_hidden_dim=labels_hidden_dim,
        num_output_labels=num_output_labels,
        transf_dec_num_memory_vecs=transf_dec_num_memory_vecs,
        transf_dec_hidden_dim=transf_dec_hidden_dim,
        transf_dec_nhead=transf_dec_nhead,
        transf_dec_dim_forward=transf_dec_dim_forward,
        transf_dec_num_layers=transf_dec_num_layers,
        dropout_prob=dropout_prob,
        support_two_label_sources=support_two_label_sources,
        num_input_labels_2=num_input_labels_2,
        labels_hidden_dim_2=labels_hidden_dim,
        num_output_labels_2=num_output_labels_2,
    )

    optimizer_kwargs = dict(
        name=optimizer_name,
        lr=lr,
    )

    lr_scheduler_kwargs = dict(
        name=scheduler,
        factor=lr_decay,
        patience=lr_decay_patience,
        warmup_and_decay_args=warmup_and_decay_args,
        warmup_and_cosine_args=warmup_and_cosine_args,
        warmup_decay_and_cyclic_decay_args=warmup_decay_and_cyclic_decay_args,
        n_batches_per_epoch=batches_per_epoch,
    )
    
    dataloading_kwargs = dict(
        batch_size=batch_size,
        mimiccxr_weight=mimiccxr_weight,
    )
        
    if train_mimiccxr:
        mimiccxr_trainer_kwargs = dict(
            qa_adapted_reports_filename=mimiccxr_qa_adapted_reports_filename,
            view_mode=mimiccxr_view_mode,
            use_decent_images_only=use_chest_imagenome_decent_images_only,
            use_gender=use_gender,
            use_chexpert=use_chexpert,
            chexpert_labels_filename=mimiccxr_chexpert_labels_filename,
            use_chest_imagenome=use_chest_imagenome,
            chest_imagenome_labels_filename=chest_imagenome_labels_filename,
            chest_imagenome_label_names_filename=chest_imagenome_label_names_filename,
            balanced_sampling_mode=mimiccxr_balanced_sampling_mode,
            balanced_batch_size=mimiccxr_balanced_batch_size,
            use_ensemble_predictions=use_ensemble,
            pre_tokenize_reports=pre_tokenize_reports,
            pre_tokenize_reports_and_convert_back_to_text=pre_tokenize_reports_and_convert_back_to_text,
        )
        # Precompute sigmoid activations for different models in ensemble
        if use_ensemble:
            assert ensemble_model_checkpoint_folder_paths is not None
            print_blue('=' * 60, bold=True)
            print_blue('Precomputing sigmoid activations for ensemble ...', bold=True)
            precomputed_sigmoid_paths = _precompute_sigmoids_for_ensemble(
                model_folder_paths=ensemble_model_checkpoint_folder_paths,
                classify_chexpert=use_chexpert,
                classify_chest_imagenome=use_chest_imagenome,
                mimiccxr_view_mode=mimiccxr_view_mode,
                batch_size=ensemble_batch_size,
                num_workers=ensemble_num_workers,
                device=device,
                debug=debug)
            assert len(precomputed_sigmoid_paths) == len(ensemble_model_checkpoint_folder_paths)
            mimiccxr_trainer_kwargs['precomputed_sigmoid_paths'] = precomputed_sigmoid_paths
            print_orange('precomputed_sigmoid_paths:', precomputed_sigmoid_paths)
            if use_hard_predictions:
                print_blue('=' * 60, bold=True)
                print_blue('Precomputing thresholds (for hard predictions) for ensemble ...', bold=True)
                precomputed_thresholds_paths = _precompute_thresholds_for_ensemble(
                    model_folder_paths=ensemble_model_checkpoint_folder_paths,
                    classify_chexpert=use_chexpert,
                    classify_chest_imagenome=use_chest_imagenome,
                    batch_size=ensemble_batch_size,
                    num_workers=ensemble_num_workers,
                    device=device)
                assert len(precomputed_thresholds_paths) == len(ensemble_model_checkpoint_folder_paths)
                mimiccxr_trainer_kwargs['use_hard_predictions'] = True
                mimiccxr_trainer_kwargs['precomputed_thresholds_paths'] = precomputed_thresholds_paths
                print_orange('precomputed_thresholds_paths:', precomputed_thresholds_paths)
                if filter_labels:
                    print_blue('=' * 60, bold=True)
                    print_blue('Precomputing filtered labels for ensemble ...', bold=True)
                    precomputed_filtered_labels_paths = _precompute_filtered_labels_for_ensemble(
                        model_folder_paths=ensemble_model_checkpoint_folder_paths,
                        classify_chexpert=use_chexpert,
                        classify_chest_imagenome=use_chest_imagenome,
                        batch_size=ensemble_batch_size,
                        num_workers=ensemble_num_workers,
                        thresholds_paths=precomputed_thresholds_paths,
                        top_k_chexpert_labels=top_k_chexpert_labels,
                        top_k_chest_imagenome_labels=top_k_chest_imagenome_labels,
                        label_score_threshold=label_score_threshold,
                        device=device)
                    assert len(precomputed_filtered_labels_paths) == len(ensemble_model_checkpoint_folder_paths)
                    mimiccxr_trainer_kwargs['filter_labels'] = True
                    mimiccxr_trainer_kwargs['precomputed_filtered_labels_paths'] = precomputed_filtered_labels_paths
                    print_orange('precomputed_filtered_labels_paths:', precomputed_filtered_labels_paths)
                    # update number of input labels
                    num_input_labels = 0
                    for x in precomputed_filtered_labels_paths:
                        if use_chexpert:
                            num_input_labels += x['chexpert_num_indices']
                        if use_chest_imagenome:
                            num_input_labels += x['chest_imagenome_num_indices']
                    assert num_input_labels > 0
                    model_kwargs['num_input_labels'] = num_input_labels
                    print_orange('num_input_labels (after update):', num_input_labels)
            if debug:
                print_red('Returning prematurely because debug=True', bold=True)
                return
    else:
        mimiccxr_trainer_kwargs = None

    _kwargs = dict(
        use_gender=use_gender,
        use_chexpert=use_chexpert,
        use_chest_imagenome=use_chest_imagenome,
        use_report=True,
        use_ground_truth_as_prediction=not use_ensemble,
        use_t5=use_t5,
        t5_model_name=t5_model_name,
        chest_imagenome_label_names_filename=chest_imagenome_label_names_filename,
        apply_anatomy_reordering=True,
    )
    collate_batch_fn_kwargs = {}
    if train_mimiccxr:
        collate_batch_fn_kwargs[DATASET_NAMES.MIMICCXR] = { 'dataset_id': MIMICCXR_DATASET_ID, **_kwargs }

    trainer_engine_kwargs = dict(
        classify_chexpert=classify_chexpert,
        classify_chest_imagenome=classify_chest_imagenome,
        chexpert_range=chexpert_range,
        chest_imagenome_range=chest_imagenome_range,
        include_report=True,
        shift_tokens_for_transformer=not use_t5,
        iters_to_accumulate=iters_to_accumulate,
        binary_loss_name=binary_loss_name,
        focal_loss_weight=focal_loss_weight,
        bce_loss_weight=bce_loss_weight,
        wbce_loss_weight=wbce_loss_weight,
        use_amp=use_amp,
        training=True,
        use_t5=use_t5,
    )
    validator_engine_kwargs = dict(
        classify_chexpert=classify_chexpert,
        classify_chest_imagenome=classify_chest_imagenome,
        chexpert_range=chexpert_range,
        chest_imagenome_range=chest_imagenome_range,
        include_report=True,
        shift_tokens_for_transformer=not use_t5,
        use_amp=use_amp,
        validating=True,
        use_t5=use_t5,
    )
    
    training_kwargs = dict(
        use_amp=use_amp,
        train_mimiccxr=train_mimiccxr,
        binary_loss_name=binary_loss_name,
        generation_mode=generation_mode,
        support_two_label_sources=support_two_label_sources,
        train_on_gt_and_eval_on_predictions=train_on_gt_and_eval_on_predictions,
        randomly_drop_labels=randomly_drop_labels,
        use_t5=use_t5,
    )

    auxiliary_tasks_kwargs = dict(
        # chexpert labels
        classify_chexpert=classify_chexpert,
        mimiccxr_chexpert_labels_filename=mimiccxr_chexpert_labels_filename,
        # chest imagenome labels
        classify_chest_imagenome=classify_chest_imagenome,
    )

    return train_model(
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                training_kwargs=training_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                auxiliary_tasks_kwargs=auxiliary_tasks_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                num_workers=num_workers,
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
        warmup_decay_and_cyclic_decay_args,
        num_workers,    
        epochs = 1,
        batches_per_epoch = 1000,
        device = 'GPU',
        save = True,
        override_lr = False,
        debug = False,
        **unused_kwargs,
        ):
    print_blue('----- Resuming training ------', bold=True)

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    training_kwargs = metadata['training_kwargs']
    trainer_engine_kwargs = metadata['trainer_engine_kwargs']
    validator_engine_kwargs = metadata['validator_engine_kwargs']                
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
            warmup_decay_and_cyclic_decay_args = warmup_decay_and_cyclic_decay_args,
            n_batches_per_epoch = batches_per_epoch,
        )

    return train_model(
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                training_kwargs=training_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                auxiliary_tasks_kwargs=auxiliary_tasks_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                num_workers=num_workers,
                device=device,
                checkpoint_folder_path=checkpoint_folder,
                save=save,
                override_lr=override_lr,
                debug=debug)

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