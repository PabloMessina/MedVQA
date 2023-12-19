import argparse
import os
import torch

from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.utils.files import get_results_folder_path, save_pickle
from medvqa.utils.handlers import (
    attach_accumulator,
    get_log_metrics_handler,
    get_log_iteration_handler,
)

from medvqa.models.checkpoint import get_checkpoint_filepath
from medvqa.models.checkpoint.model_wrapper import ModelWrapper

from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED,
    CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
    get_chest_imagenome_gold_class_mask,
)
from medvqa.datasets.mimiccxr.mimiccxr_phrase_grounding_dataset_management import MIMICCXR_PhraseGroundingTrainer

from medvqa.models.phrase_grounding.phrase_grounder import PhraseGrounder

from medvqa.utils.constants import (
    DATASET_NAMES,
    MetricNames,
)
from medvqa.metrics import (
    attach_condition_aware_accuracy,
    attach_condition_aware_chest_imagenome_bbox_iou,
    attach_condition_aware_segmask_iou,
    attach_condition_aware_segmask_iou_per_class,
)
from medvqa.models.checkpoint import (
    load_metadata,
)
from medvqa.utils.common import parsed_args_to_dict
from medvqa.training.phrase_grounding import get_engine
from medvqa.datasets.dataloading_utils import (
    get_phrase_grounding_collate_batch_fn,
)
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging import CountPrinter, print_blue

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # --- Required arguments

    parser.add_argument('--checkpoint_folder_path', type=str, default=None, help='Path to the checkpoint folder of the model to be evaluated')
    parser.add_argument('--max_images_per_batch', type=int, required=True, help='Max number of images per batch')
    parser.add_argument('--max_phrases_per_batch', type=int, required=True, help='Max number of phrases per batch')
    parser.add_argument('--max_phrases_per_image', type=int, required=True, help='Max number of phrases per image')

    # --- Other arguments

    # Dataset and dataloading arguments
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')

    # Evaluation arguments
    parser.add_argument('--eval_chest_imagenome_gold', action='store_true', default=False)
    parser.add_argument('--eval_mscxr', action='store_true', default=False)
    
    return parser.parse_args(args=args)

def _evaluate_model(
    checkpoint_folder_path,
    model_kwargs,
    mimiccxr_trainer_kwargs,
    collate_batch_fn_kwargs,
    val_image_transform_kwargs,
    evaluation_engine_kwargs,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    num_workers,
    eval_chest_imagenome_gold,
    eval_mscxr,
    device,
):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    use_yolov8 = (mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_yolov8', False))

    # Sanity checks
    assert sum([eval_chest_imagenome_gold, eval_mscxr]) > 0 # at least one dataset must be evaluated

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    count_print('device =', device)

    # Create model
    count_print('Creating instance of PhraseGrounder ...')
    model = PhraseGrounder(**model_kwargs)
    model = model.to(device)

    # Load model from checkpoint
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    count_print('Loading model from checkpoint ...')
    print('checkpoint_path =', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True)

    # Create phrase grounding trainer
    count_print('Creating MIMIC-CXR Phrase Grounding Trainer ...')
    bbox_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cibg'])
    phrase_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['pg'])
    mimiccxr_trainer_kwargs['use_facts_for_train'] = False
    mimiccxr_trainer_kwargs['use_mscxr_for_train'] = False
    mimiccxr_trainer_kwargs['use_mscxr_for_test'] = eval_mscxr
    mimiccxr_trainer_kwargs['use_chest_imagenome_for_train'] = False
    mimiccxr_trainer_kwargs['use_chest_imagenome_gold_for_test'] = eval_chest_imagenome_gold

    mimiccxr_trainer = MIMICCXR_PhraseGroundingTrainer(
        test_image_transform = get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]),
        max_images_per_batch=max_images_per_batch,
        max_phrases_per_batch=max_phrases_per_batch,
        max_phrases_per_image=max_phrases_per_image,
        bbox_grounding_collate_batch_fn=bbox_grounding_collate_batch_fn,
        phrase_grounding_collate_batch_fn=phrase_grounding_collate_batch_fn,
        num_test_workers=num_workers,
        **mimiccxr_trainer_kwargs,
    )
    
    if eval_chest_imagenome_gold:

        count_print('----- Evaluating on Chest ImaGenome Gold Bbox Phrase Grounding -----')

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

        # Attach metrics
        assert use_yolov8 # TODO: eventually support other bbox predictors
        _cond_func = lambda x: x['flag'] == 'cibg'
        _gold_class_mask = get_chest_imagenome_gold_class_mask()
        attach_condition_aware_chest_imagenome_bbox_iou(evaluation_engine, _cond_func, use_yolov8=True, class_mask=_gold_class_mask)
        attach_condition_aware_accuracy(evaluation_engine, 'pred_phrase_labels', 'gt_phrase_labels', 'phrase_acc', _cond_func)
        attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
                                                      nc=CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
                                                      condition_function=_cond_func)
        # for logging
        metrics_to_print = []
        metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXIOU)
        metrics_to_print.append('phrase_acc')
        metrics_to_print.append('segmask_iou')

        # Timer
        timer = Timer()
        timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

        # Logging
        print_blue('Defining log_metrics_handler ...', bold=True)
        log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
        log_iteration_handler = get_log_iteration_handler()
        
        # Attach handlers
        evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
        evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

        # Start evaluation
        print_blue('Running engine ...', bold=True)
        evaluation_engine.run(mimiccxr_trainer.test_chest_imagenome_gold_dataloader)

        # Print final metrics
        print_blue('Final metrics:', bold=True)
        metrics = evaluation_engine.state.metrics
        # 1) chest imagenome bbox iou
        metric_name = MetricNames.CHESTIMAGENOMEBBOXIOU
        print(f'{metric_name}: {metrics[metric_name]}')
        print()
        # 2) phrase accuracy
        metric_name = 'phrase_acc'
        print(f'{metric_name}: {metrics[metric_name]}')
        print()
        # 3) segmask iou
        metric_name = 'segmask_iou'
        print(f'{metric_name}:')
        from tabulate import tabulate
        table = []
        for bbox_name, iou in zip(CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED, metrics[metric_name]):
            table.append([bbox_name, iou])
        print(tabulate(table, headers=['bbox_name', 'iou'], tablefmt='latex_raw'))

        # Save metrics to file
        print_blue('Saving metrics to file ...', bold=True)
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        save_path = os.path.join(results_folder_path, f'chest_imagenome_gold_metrics.pkl')
        save_pickle(metrics, save_path)
        print(f'Saved metrics to {save_path}')

    if eval_mscxr:

        count_print('----- Evaluating on MSCXR Phrase Grounding -----')

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

        # Attach metrics
        metrics_to_print = []
        _cond_func = lambda x: x['flag'] == 'pg'
        attach_condition_aware_segmask_iou(evaluation_engine, 'pred_mask', 'gt_mask', 'pg_segmask_iou', _cond_func)
        metrics_to_print.append('pg_segmask_iou')

        # Attach accumulators
        attach_accumulator(evaluation_engine, 'pred_mask')
        attach_accumulator(evaluation_engine, 'gt_mask')

        # Timer
        timer = Timer()
        timer.attach(evaluation_engine, start=Events.EPOCH_STARTED)

        # Logging
        print_blue('Defining log_metrics_handler ...', bold=True)
        log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
        log_iteration_handler = get_log_iteration_handler()
        
        # Attach handlers
        evaluation_engine.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
        evaluation_engine.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)

        # Start evaluation
        print_blue('Running engine ...', bold=True)
        evaluation_engine.run(mimiccxr_trainer.test_mscxr_dataloader)

        # Print final metrics
        print_blue('Final metrics:', bold=True)
        metrics = evaluation_engine.state.metrics
        # 1) segmask iou
        metric_name = 'pg_segmask_iou'
        print(f'{metric_name}: {metrics[metric_name]}')

        # Save metrics to file
        image_paths = []
        phrases = []
        for dataset in  mimiccxr_trainer.mscxr_datasets:
            for i in dataset.indices:
                image_paths.append(dataset.image_paths[i])
                phrases.append(dataset.phrases[i])
            print(f'len(image_paths) = {len(image_paths)}')
        assert len(image_paths) == len(metrics['pred_mask']) == len(metrics['gt_mask'])
        print_blue('Saving metrics to file ...', bold=True)
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        save_path = os.path.join(results_folder_path, f'mscxr_metrics.pkl')
        pred_masks = metrics['pred_mask']
        gt_masks = metrics['gt_mask']
        output = dict(
            image_paths=[],
            phrases=[],
            pred_masks=[],
            gt_masks=[],
            ious=[],
            pg_segmask_iou=metrics['pg_segmask_iou'],
        )
        for i in range(len(image_paths)):
            for j in range(len(pred_masks[i])):
                intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
                union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
                iou = intersection / union
                iou = iou.item()
                output['image_paths'].append(image_paths[i])
                output['phrases'].append(phrases[i][j])
                output['pred_masks'].append(pred_masks[i][j].cpu().numpy())
                output['gt_masks'].append(gt_masks[i][j].cpu().numpy())
                output['ious'].append(iou)
        print('mean_iou =', sum(output['ious']) / len(output['ious']))
        save_pickle(output, save_path)
        print(f'Saved metrics to {save_path}')

def evaluate(
    checkpoint_folder_path,
    num_workers,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    eval_chest_imagenome_gold,
    eval_mscxr,
    device,
):
    print_blue('----- Evaluating model -----', bold=True)

    metadata = load_metadata(checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    try:
        val_image_transform_kwargs = metadata['val_image_transform_kwargs']
    except KeyError: # HACK: when val_image_transform_kwargs is missing due to a bug
        val_image_transform_kwargs = {
            DATASET_NAMES.MIMICCXR: dict(
                image_size=(416, 416),
                augmentation_mode=None,
                use_bbox_aware_transform=True,
                for_yolov8=True,
            )
        }
    validator_engine_kwargs = metadata['validator_engine_kwargs']

    return _evaluate_model(
                checkpoint_folder_path=checkpoint_folder_path,
                model_kwargs=model_kwargs,
                mimiccxr_trainer_kwargs=mimiccxr_trainer_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                evaluation_engine_kwargs=validator_engine_kwargs,
                max_images_per_batch=max_images_per_batch,
                max_phrases_per_batch=max_phrases_per_batch,
                max_phrases_per_image=max_phrases_per_image,
                num_workers=num_workers,
                eval_chest_imagenome_gold=eval_chest_imagenome_gold,
                eval_mscxr=eval_mscxr,
                device=device,
            )


if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)