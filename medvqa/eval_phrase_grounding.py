import argparse
import math
import os
from pprint import pprint
import torch
import numpy as np
from ignite.engine import Events
from ignite.handlers.timing import Timer
from tqdm import tqdm
from medvqa.datasets.chexlocalize import CHEXLOCALIZE_CLASS_NAMES
from medvqa.datasets.chexlocalize.chexlocalize_dataset_management import CheXlocalizePhraseGroundingTrainer
from medvqa.datasets.image_transforms_factory import create_image_transforms
from medvqa.datasets.mimiccxr import MIMICCXR_ImageSizeModes
from medvqa.datasets.ms_cxr import get_ms_cxr_category_names, get_ms_cxr_phrase_to_category_name
from medvqa.datasets.vinbig import (
    VINBIG_CHEX_CLASSES,
    VINBIG_CHEX_IOU_THRESHOLDS,
    VINBIG_RAD_DINO_CLASSES,
    VINBIGDATA_CHALLENGE_CLASSES,
    VINBIGDATA_CHALLENGE_IOU_THRESHOLD,
)
# from medvqa.datasets.vinbig.vinbig_dataset_management import VinBigPhraseGroundingTrainer
from medvqa.evaluation.bootstrapping import (
    apply_stratified_bootstrapping,
    stratified_multilabel_bootstrap_metrics,
    stratified_vinbig_bootstrap_iou_map,
)
from medvqa.metrics.bbox.utils import (
    compute_mean_bbox_union_iou,
    compute_iou_with_nms,
    compute_mAP__yolov11,
    compute_mean_iou_per_class__yolov11,
    compute_probability_map_iou,
    find_optimal_conf_iou_thresholds,
    find_optimal_conf_iou_max_det_thresholds__single_class,
    find_optimal_probability_map_conf_threshold,
)
from medvqa.metrics.classification.prc_auc import prc_auc_fn, prc_auc_score
from medvqa.models.vision.visual_modules import RawImageEncoding
from medvqa.utils.files_utils import get_results_folder_path, save_pickle
from medvqa.utils.handlers_utils import (
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
    VINBIG_BBOX_NAMES,
    VINBIG_NUM_BBOX_CLASSES,
    MetricNames,
)
from medvqa.metrics import (
    attach_condition_aware_accuracy,
    attach_condition_aware_bbox_iou_per_class,
    attach_condition_aware_chest_imagenome_bbox_iou,
    attach_condition_aware_segmask_iou_per_class,
)
from medvqa.models.checkpoint import (
    load_metadata,
)
from medvqa.utils.common import activate_determinism, parsed_args_to_dict
from medvqa.training.phrase_grounding import get_engine
from medvqa.datasets.dataloading_utils import (
    get_phrase_grounding_collate_batch_fn,
)
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.logging_utils import CountPrinter, print_blue, print_bold, print_magenta, print_orange, setup_logging

setup_logging()

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
    parser.add_argument('--mscxr_phrase2embedding_filepath', type=str, default=None, help='Path to the MS-CXR phrase2embedding file')
    parser.add_argument('--mimicxr_dicom_id_to_pos_neg_facts_filepath', type=str, default=None, help='Path to the MIMIC-CXR DICOM ID to pos/neg facts file')
    parser.add_argument('--vinbig_use_training_indices_for_validation', action='store_true')
    parser.add_argument('--checkpoint_folder_path_to_borrow_metadata_from', type=str, default=None, help='Path to metadata file to borrow trainer kwargs from')
    parser.add_argument('--override_bbox_format', type=str, default=None, choices=['xyxy', 'cxcywh'], help='Override the bbox format used in the dataset')

    # Evaluation arguments
    parser.add_argument('--eval_chest_imagenome_gold', action='store_true')
    parser.add_argument('--eval_mscxr', action='store_true')
    parser.add_argument('--eval_chexlocalize', action='store_true')
    parser.add_argument('--eval_vinbig', action='store_true')
    parser.add_argument('--optimize_thresholds', action='store_true')
    parser.add_argument('--candidate_iou_thresholds', type=float, nargs='+', default=None)
    parser.add_argument('--candidate_conf_thresholds', type=float, nargs='+', default=None)
    parser.add_argument('--map_iou_thresholds', type=float, nargs='+', default=[0., 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--use_classifier_confs_for_map', action='store_true', help='Use classifier confidences for mAP computation')
    
    return parser.parse_args(args=args)

def _evaluate_model(
    checkpoint_folder_path,
    model_kwargs,
    mimiccxr_trainer_kwargs,
    chexlocalize_trainer_kwargs,
    vinbig_trainer_kwargs,
    val_image_transform_kwargs,
    evaluation_engine_kwargs,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    num_workers,
    eval_chest_imagenome_gold,
    eval_mscxr,
    eval_chexlocalize,
    eval_vinbig,
    mscxr_phrase2embedding_filepath,
    mimicxr_dicom_id_to_pos_neg_facts_filepath,
    device,
    vinbig_use_training_indices_for_validation,
    optimize_thresholds,
    candidate_iou_thresholds,
    candidate_conf_thresholds,
    map_iou_thresholds,
    use_amp,
    use_classifier_confs_for_map,
    checkpoint_folder_path_to_borrow_metadata_from,
    override_bbox_format,
):
    count_print = CountPrinter()
    
    # Pull out some args from kwargs
    use_yolov8 = (mimiccxr_trainer_kwargs is not None and mimiccxr_trainer_kwargs.get('use_yolov8', False))
    use_fact_conditioned_yolo = model_kwargs['raw_image_encoding'] == RawImageEncoding.YOLOV11_FACT_CONDITIONED
    do_visual_grounding_with_bbox_regression = evaluation_engine_kwargs.get('do_visual_grounding_with_bbox_regression', False)
    do_visual_grounding_with_segmentation = evaluation_engine_kwargs.get('do_visual_grounding_with_segmentation', False)
    print(f'do_visual_grounding_with_bbox_regression = {do_visual_grounding_with_bbox_regression}')
    print(f'do_visual_grounding_with_segmentation = {do_visual_grounding_with_segmentation}')

    # Sanity checks
    assert sum([eval_chest_imagenome_gold, eval_mscxr, eval_chexlocalize, eval_vinbig]) > 0 # at least one dataset must be evaluated

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
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True, strict=False)

    # Create phrase grounding trainers

    # if eval_chest_imagenome_gold or eval_mscxr:

    #     if checkpoint_folder_path_to_borrow_metadata_from is not None:
    #         metadata = load_metadata(checkpoint_folder_path_to_borrow_metadata_from)
    #         # collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    #         mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
        
    #     try: 
    #         image_transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]
    #     except KeyError:
    #         image_transform_kwargs = next(iter(val_image_transform_kwargs.values())) # get the first value

    #     count_print('Creating MIMIC-CXR Phrase Grounding Trainer ...')
    #     # if eval_chest_imagenome_gold:
    #     #     bbox_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cibg'])
    #     # else:
    #     #     bbox_grounding_collate_batch_fn = None
    #     # if eval_mscxr:
    #     #     mscxr_phrase_grounding_collate_batch_fn = get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['mscxr'])
    #     # else:
    #     #     mscxr_phrase_grounding_collate_batch_fn = None
    #     mimiccxr_trainer_kwargs['use_facts_for_train'] = False
    #     mimiccxr_trainer_kwargs['use_facts_for_test'] = False
    #     mimiccxr_trainer_kwargs['use_mscxr_for_train'] = False
    #     mimiccxr_trainer_kwargs['use_mscxr_for_val'] = False
    #     mimiccxr_trainer_kwargs['use_mscxr_for_test'] = eval_mscxr
    #     mimiccxr_trainer_kwargs['mscxr_test_on_all_images'] = eval_mscxr # if True, test on all MSCXR
    #     mimiccxr_trainer_kwargs['use_cxrlt2024_challenge_split'] = False 
    #     mimiccxr_trainer_kwargs['use_cxrlt2024_official_labels'] = False
    #     mimiccxr_trainer_kwargs['use_cxrlt2024_custom_labels'] = False
    #     mimiccxr_trainer_kwargs['use_chest_imagenome_for_train'] = False
    #     mimiccxr_trainer_kwargs['use_chest_imagenome_gold_for_test'] = eval_chest_imagenome_gold
    #     if mscxr_phrase2embedding_filepath is not None:
    #         mimiccxr_trainer_kwargs['mscxr_phrase2embedding_filepath'] = mscxr_phrase2embedding_filepath
    #     if override_bbox_format:
    #         print_orange('Overriding bbox format to', override_bbox_format)
    #         mimiccxr_trainer_kwargs['bbox_format'] = override_bbox_format

    #     mimiccxr_trainer = MIMICCXR_PhraseGroundingTrainer(
    #         test_image_transform = create_image_transforms(**image_transform_kwargs),
    #         max_images_per_batch=max_images_per_batch,
    #         max_phrases_per_batch=max_phrases_per_batch,
    #         max_phrases_per_image=max_phrases_per_image,
    #         # bbox_grounding_collate_batch_fn=bbox_grounding_collate_batch_fn,
    #         # mscxr_phrase_grounding_collate_batch_fn=mscxr_phrase_grounding_collate_batch_fn,
    #         num_test_workers=num_workers,
    #         **mimiccxr_trainer_kwargs,
    #     )

    if eval_chexlocalize:

        count_print('Creating CheXlocalize Phrase Grounding Trainer ...')
        chexlocalize_trainer_kwargs['use_training_set'] = False
        chexlocalize_trainer_kwargs['use_validation_set'] = True
        chexlocalize_trainer = CheXlocalizePhraseGroundingTrainer(
            val_image_transform=get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.CHEXLOCALIZE]),
            collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs['cl']),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            num_val_workers=num_workers,
            **chexlocalize_trainer_kwargs,
        )
    
    if eval_vinbig:

        count_print('Creating VinBig Phrase Grounding Trainer ...')
        if vinbig_trainer_kwargs is None:
            assert checkpoint_folder_path_to_borrow_metadata_from is not None
            metadata = load_metadata(checkpoint_folder_path_to_borrow_metadata_from)
            vinbig_trainer_kwargs = metadata['vinbig_trainer_kwargs']
            try: 
                transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.VINBIG]
            except KeyError:
                first_key = list(val_image_transform_kwargs.keys())[0]
                transform_kwargs = val_image_transform_kwargs[first_key]
                print_orange(f'Using transform_kwargs from {first_key} for VinBig')
            try:
                collate_batch_fn_kwargs = collate_batch_fn_kwargs['vbg']
            except KeyError:
                collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']['vbg']
                print_orange(f'Borrowing collate_batch_fn_kwargs from {checkpoint_folder_path_to_borrow_metadata_from} for VinBig')
        else:
            transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.VINBIG]
            collate_batch_fn_kwargs = collate_batch_fn_kwargs['vbg']
        vinbig_trainer_kwargs['use_training_set'] = False
        vinbig_trainer_kwargs['use_validation_set'] = True
        vinbig_trainer = VinBigPhraseGroundingTrainer(
            val_image_transform=get_image_transform(**transform_kwargs),
            collate_batch_fn=get_phrase_grounding_collate_batch_fn(**collate_batch_fn_kwargs),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            num_val_workers=num_workers,
            use_training_indices_for_validation=vinbig_use_training_indices_for_validation,
            **vinbig_trainer_kwargs,
        )

    # Evaluate on datasets
    
    if eval_chest_imagenome_gold:

        count_print('----- Evaluating on Chest ImaGenome Gold Bbox Phrase Grounding -----')

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

        # Attach metrics
        _cond_func = lambda x: x['flag'] == 'cibg'
        if use_yolov8:
            _gold_class_mask = get_chest_imagenome_gold_class_mask()
            attach_condition_aware_chest_imagenome_bbox_iou(evaluation_engine, _cond_func, use_yolov8=True, class_mask=_gold_class_mask)
        if do_visual_grounding_with_bbox_regression:
            attach_condition_aware_bbox_iou_per_class(evaluation_engine,
                                                      field_names=['predicted_bboxes', 'chest_imagenome_bbox_coords', 'chest_imagenome_bbox_presence'],
                                                      metric_name='bbox_iou', nc=CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
                                                      condition_function=_cond_func)
        else:
            attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
                                                        nc=CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
                                                        condition_function=_cond_func)
        # Attach accumulators
        if do_visual_grounding_with_bbox_regression:
            attach_accumulator(evaluation_engine, 'predicted_bboxes')
            attach_accumulator(evaluation_engine, 'chest_imagenome_bbox_coords')
            attach_accumulator(evaluation_engine, 'chest_imagenome_bbox_presence')
        else:
            attach_accumulator(evaluation_engine, 'pred_mask')
            attach_accumulator(evaluation_engine, 'gt_mask')

        # for logging
        metrics_to_print = []
        if use_yolov8:
            metrics_to_print.append(MetricNames.CHESTIMAGENOMEBBOXIOU)
        if do_visual_grounding_with_bbox_regression:
            metrics_to_print.append('bbox_iou')
        else:
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
        if use_yolov8:
            # 1) chest imagenome bbox iou
            metric_name = MetricNames.CHESTIMAGENOMEBBOXIOU
            print(f'{metric_name}: {metrics[metric_name]}')
            print()
        # 2) bbox iou / segmask iou
        if do_visual_grounding_with_bbox_regression:
            metric_name = 'bbox_iou'
        else:
            metric_name = 'segmask_iou'
        print(f'{metric_name}:')
        from tabulate import tabulate
        table = []
        for bbox_name, iou in zip(CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED, metrics[metric_name]):
            table.append([bbox_name, iou])
        print(tabulate(table, headers=['bbox_name', 'iou'], tablefmt='latex_raw'))

        # Save metrics to file
        dataset = mimiccxr_trainer.test_chest_imagenome_gold_dataset
        image_paths = dataset.image_paths
        phrases = mimiccxr_trainer.test_chest_imagenome_gold_bbox_phrases

        if do_visual_grounding_with_bbox_regression:
            pred_bboxes = metrics['predicted_bboxes']
            gt_bboxes = metrics['chest_imagenome_bbox_coords']
            gt_presence = metrics['chest_imagenome_bbox_presence']
            assert len(image_paths) == len(pred_bboxes) == len(gt_bboxes) == len(gt_presence)
            print_blue('Saving metrics to file ...', bold=True)
            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            save_path = os.path.join(results_folder_path, f'chest_imagenome_gold_metrics_bbox_regression.pkl')
            output = dict(
                image_paths=[],
                phrases=[],
                pred_bboxes=[],
                gt_bboxes=[],
                ious=[],
                bbox_iou=metrics['bbox_iou'],
            )
            for i in range(len(image_paths)):
                for j in range(len(pred_bboxes[i])):
                    if gt_presence[i][j] == 1:
                        if pred_bboxes[i][j] is None:
                            iou = 0
                        else:
                            iou = compute_mean_bbox_union_iou(pred_bboxes[i][j][0], gt_bboxes[i][j])
                        output['image_paths'].append(image_paths[i])
                        output['phrases'].append(phrases[j])
                        output['pred_bboxes'].append(pred_bboxes[i][j][0].cpu().numpy() if pred_bboxes[i][j] is not None else None)
                        output['gt_bboxes'].append(gt_bboxes[i][j].cpu().numpy())
                        output['ious'].append(iou)
        else:
            gt_masks = metrics['gt_mask']
            pred_masks = metrics['pred_mask']
            assert len(image_paths) == len(pred_masks) == len(gt_masks)
            print_blue('Saving metrics to file ...', bold=True)
            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            save_path = os.path.join(results_folder_path, f'chest_imagenome_gold_metrics_segmask.pkl')
            output = dict(
                image_paths=[],
                phrases=[],
                pred_masks=[],
                gt_masks=[],
                ious=[],
                segmask_iou=metrics['segmask_iou'],
            )
            for i in range(len(image_paths)):
                for j in range(len(pred_masks[i])):
                    intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
                    union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
                    iou = intersection / union
                    iou = iou.item()
                    output['image_paths'].append(image_paths[i])
                    output['phrases'].append(phrases[j])
                    output['pred_masks'].append(pred_masks[i][j].cpu().numpy())
                    output['gt_masks'].append(gt_masks[i][j].cpu().numpy())
                    output['ious'].append(iou)

        print_magenta('mean_iou =', sum(output['ious']) / len(output['ious']), bold=True)
        save_pickle(output, save_path)
        print(f'Saved metrics to {save_path}')

    if eval_mscxr:

        count_print('----- Evaluating on MSCXR Phrase Grounding -----')
        
        assert candidate_conf_thresholds is not None
        assert candidate_iou_thresholds is not None

        # Get image transform kwargs
        try: 
            image_transform_kwargs = val_image_transform_kwargs[DATASET_NAMES.MIMICCXR]
        except KeyError:
            image_transform_kwargs = next(iter(val_image_transform_kwargs.values())) # get the first value

        # Initialize trainer
        mimiccxr_trainer = MIMICCXR_PhraseGroundingTrainer(
            use_mscxr_for_test=True,
            test_image_transform = create_image_transforms(**image_transform_kwargs),
            max_images_per_batch=max_images_per_batch,
            max_phrases_per_batch=max_phrases_per_batch,
            max_phrases_per_image=max_phrases_per_image,
            num_test_workers=num_workers,
            mscxr_do_grounding_only=False,
            mscxr_test_on_all_images=True,
            bbox_format=override_bbox_format,
            source_image_size_mode=MIMICCXR_ImageSizeModes.MEDIUM_512,
            mscxr_phrase2embedding_filepath=mscxr_phrase2embedding_filepath,
            dicom_id_to_pos_neg_facts_filepath=mimicxr_dicom_id_to_pos_neg_facts_filepath,
        )

        # Get dataset and dataloader
        dataset = mimiccxr_trainer.mscxr_test_dataset
        dataloader = mimiccxr_trainer.mscxr_test_dataloader

        # Aux variables
        train_preds_and_gt = {
            'pred_bbox_prob_maps': [],
            'pred_bbox_coord_maps': [],
            'pred_classification_probs': [],
            'gt_bbox_coords': [],
            'phrases': [],
            'categories': [],
            'image_paths': [],
        }
        val_preds_and_gt = {
            'pred_bbox_prob_maps': [],
            'pred_bbox_coord_maps': [],
            'pred_classification_probs': [],
            'gt_bbox_coords': [],
            'phrases': [],
            'categories': [],
            'image_paths': [],
        }
        test_preds_and_gt = {
            'pred_bbox_prob_maps': [],
            'pred_bbox_coord_maps': [],
            'pred_classification_probs': [],
            'gt_bbox_coords': [],
            'phrases': [],
            'categories': [],
            'image_paths': [],
        }
        n_train = len(mimiccxr_trainer.mscxr_train_indices)
        n_val = len(mimiccxr_trainer.mscxr_val_indices)
        n_test = len(mimiccxr_trainer.mscxr_test_indices)
        train_prc_aucs = []
        val_prc_aucs = []
        test_prc_aucs = []

        category_names = get_ms_cxr_category_names()
        category_name_to_idx = {name: idx for idx, name in enumerate(category_names)}
        phrase_to_category_name = get_ms_cxr_phrase_to_category_name()

        # Run evaluation
        print_blue('Running evaluation ...', bold=True)
        model.eval()
        H, W = None, None
        idx = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', unit='batch', mininterval=2):
                images = batch['i'].to(device)
                phrase_embeddings = batch['pe'].to(device)
                phrase_classification_labels = batch['pcl']
                bboxes = batch['bboxes']
                classes = batch['classes']
                output = model(
                    raw_images=images,
                    phrase_embeddings=phrase_embeddings,
                    predict_bboxes=True,
                    only_compute_features=True,
                    apply_nms=False, # Skip NMS in order to get all predictions
                )
                phrase_classifier_logits = output['phrase_classifier_logits']
                visual_grounding_bbox_prob_logits = output['visual_grounding_confidence_logits'] # (B, N, H * W, 1)
                visual_grounding_bbox_prob_logits = visual_grounding_bbox_prob_logits.squeeze(-1) # (B, N, H * W)
                visual_grounding_bbox_coord_logits = output['visual_grounding_bbox_logits'] # (B, N, H * W, 4)
                visual_grounding_bbox_coord_logits = visual_grounding_bbox_coord_logits.cpu().numpy()
                phrase_classifier_probs = torch.sigmoid(phrase_classifier_logits).cpu().numpy() # (B, N)
                visual_grounding_bbox_probs = torch.sigmoid(visual_grounding_bbox_prob_logits).cpu().numpy()
                assert visual_grounding_bbox_probs.ndim == 3
                assert visual_grounding_bbox_coord_logits.ndim == 4

                if H is None:
                    # Get the integer square root of H * W, assuming H = W
                    H = W = math.isqrt(visual_grounding_bbox_prob_logits.shape[-1])
                    assert H * W == visual_grounding_bbox_prob_logits.shape[-1]

                batch_size = images.size(0)

                for b in range(batch_size):
                
                    i = dataset.indices[idx]
                    phrase_idxs = dataset.phrase_idxs[i]
                    image_path = dataset.image_paths[i]
                    prc_auc = prc_auc_score(phrase_classification_labels[b], phrase_classifier_probs[b])
                    
                    if idx < n_train:
                        preds_and_gt = train_preds_and_gt
                        train_prc_aucs.append(prc_auc)
                    elif idx < n_train + n_val:
                        preds_and_gt = val_preds_and_gt
                        val_prc_aucs.append(prc_auc)
                    else:
                        preds_and_gt = test_preds_and_gt
                        test_prc_aucs.append(prc_auc)
                    
                    for j, phrase_idx in enumerate(phrase_idxs):
                        phrase = mimiccxr_trainer.mscxr_phrases[phrase_idx]
                        phrase_bboxes = [bbox for bbox, cls in zip(bboxes[b], classes[b]) if cls == j]
                        assert len(phrase_bboxes) > 0
                        pred_bbox_prob_map = visual_grounding_bbox_probs[b, j] # (H * W,)
                        pred_bbox_coord_map = visual_grounding_bbox_coord_logits[b, j] # (H * W, 4)
                        pred_bbox_prob_map = pred_bbox_prob_map.reshape(H, W)
                        pred_bbox_coord_map = pred_bbox_coord_map.reshape(H, W, 4)
                        preds_and_gt['pred_bbox_prob_maps'].append(pred_bbox_prob_map)
                        preds_and_gt['pred_bbox_coord_maps'].append(pred_bbox_coord_map)
                        preds_and_gt['pred_classification_probs'].append(phrase_classifier_probs[b, j])
                        preds_and_gt['gt_bbox_coords'].append(phrase_bboxes)
                        preds_and_gt['phrases'].append(phrase)
                        preds_and_gt['categories'].append(phrase_to_category_name[phrase])
                        preds_and_gt['image_paths'].append(image_path)

                    idx += 1

        # Print a few stats
        print_blue('Stats:', bold=True)
        print(f'H = {H}, W = {W}')
        print(f'n_train = {n_train}')
        print(f'n_val = {n_val}')
        print(f'n_test = {n_test}')
        print(f'len(train_preds_and_gt["pred_bbox_prob_maps"]) = {len(train_preds_and_gt["pred_bbox_prob_maps"])}')
        print(f'len(val_preds_and_gt["pred_bbox_prob_maps"]) = {len(val_preds_and_gt["pred_bbox_prob_maps"])}')
        print(f'len(test_preds_and_gt["pred_bbox_prob_maps"]) = {len(test_preds_and_gt["pred_bbox_prob_maps"])}')

        # Print PRC-AUCs
        print_blue('PRC-AUCs:', bold=True)
        print(f'Train PRC-AUC: {sum(train_prc_aucs) / len(train_prc_aucs)}')
        print(f'Val PRC-AUC: {sum(val_prc_aucs) / len(val_prc_aucs)}')
        print(f'Test PRC-AUC: {sum(test_prc_aucs) / len(test_prc_aucs)}')

        # Compute IoU based on bbox probability maps on train, val, and test sets with bootstrapping

        def _compute_iou(split, preds_and_gt):
            # Train
            print_blue(f'Finding optimal probability threshold for IoU on the {split} set ...', bold=True)
            tmp = find_optimal_probability_map_conf_threshold(
                prob_maps=np.array(preds_and_gt['pred_bbox_prob_maps']), # (N, H, W)
                gt_bboxes_list=preds_and_gt['gt_bbox_coords'],
                bbox_format=dataset.bbox_format,
            )
            print(tmp)
            best_conf_th = tmp['best_conf_th']
            ious = [compute_probability_map_iou(prob_map, gt_bboxes, best_conf_th, bbox_format=dataset.bbox_format) for\
                        prob_map, gt_bboxes in zip(preds_and_gt['pred_bbox_prob_maps'],
                                                    preds_and_gt['gt_bbox_coords'])]
            classes = [category_name_to_idx[phrase_to_category_name[phrase]] for phrase in preds_and_gt['phrases']]
            class_to_indices = [[] for _ in range(len(category_names))]
            for i, class_idx in enumerate(classes):
                class_to_indices[class_idx].append(i)
            iou_with_boostrapping = apply_stratified_bootstrapping(
                metric_values=ious, class_to_indices=class_to_indices,
                class_names=category_names, metric_name='iou', num_bootstraps=500, num_processes=6)
            print_bold(f'{split.capitalize()} IoU with bootstrapping:')
            pprint(iou_with_boostrapping)
            return best_conf_th, ious, iou_with_boostrapping, class_to_indices
        
        # Train
        train_best_conf_th, train_ious, train_iou_with_boostrapping, train_class_to_indices =\
            _compute_iou('train', train_preds_and_gt)
        train_preds_and_gt['ious'] = train_ious
        # Val
        val_best_conf_th, val_ious, val_iou_with_boostrapping, val_class_to_indices =\
            _compute_iou('val', val_preds_and_gt)
        val_preds_and_gt['ious'] = val_ious
        # Test
        test_best_conf_th, test_ious, test_iou_with_boostrapping, test_class_to_indices =\
            _compute_iou('test', test_preds_and_gt)
        test_preds_and_gt['ious'] = test_ious

        # Compute IoU based on bbox coordinates on train, val, and test sets with bootstrapping

        def _compute_iou_2(split, preds_and_gt, class_to_indices):
            print_blue(f'Finding optimal IoU and conf thresholds for IoU based on bounding box coordinates on the {split} set ...', bold=True)
            tmp = find_optimal_conf_iou_max_det_thresholds__single_class(
                gt_coords_list=preds_and_gt['gt_bbox_coords'],
                pred_boxes_list=preds_and_gt['pred_bbox_coord_maps'],
                pred_confs_list=preds_and_gt['pred_bbox_prob_maps'],
                iou_thresholds=candidate_iou_thresholds,
                conf_thresholds=candidate_conf_thresholds,
                verbose=False,
                bbox_format=dataset.bbox_format,
            )
            best_iou_th = tmp['best_iou_threshold']
            best_conf_th = tmp['best_conf_threshold']
            best_pre_nms_max_det = tmp['best_pre_nms_max_det']
            best_post_nms_max_det = tmp['best_post_nms_max_det']
            print(f'{split}_best_iou_threshold = {best_iou_th}')
            print(f'{split}_best_conf_threshold = {best_conf_th}')
            print(f'{split}_best_pre_nms_max_det = {best_pre_nms_max_det}')
            print(f'{split}_best_post_nms_max_det = {best_post_nms_max_det}')            
            ious = []
            for pred_bbox_coord_map, pred_bbox_prob_map, gt_bbox_coords in zip(preds_and_gt['pred_bbox_coord_maps'],
                                                                            preds_and_gt['pred_bbox_prob_maps'],
                                                                            preds_and_gt['gt_bbox_coords']):
                iou = compute_iou_with_nms(
                    gt_bboxes=gt_bbox_coords,
                    pred_bbox_coords=pred_bbox_coord_map.reshape(-1, 4),
                    pred_bbox_probs=pred_bbox_prob_map.reshape(-1),
                    iou_th=best_iou_th,
                    conf_th=best_conf_th,
                    pre_nms_max_det=best_pre_nms_max_det,
                    post_nms_max_det=best_post_nms_max_det,
                    bbox_format=dataset.bbox_format,
                )
                ious.append(iou)
            iou_with_boostrapping = apply_stratified_bootstrapping(
                metric_values=ious, class_to_indices=class_to_indices,
                class_names=category_names, metric_name='iou', num_bootstraps=500, num_processes=6)
            print_bold(f'{split.capitalize()} IoU with bootstrapping:')
            pprint(iou_with_boostrapping)
            return best_iou_th, best_conf_th, best_pre_nms_max_det, best_post_nms_max_det, ious, iou_with_boostrapping

        # Train
        train_best_iou_th_2, train_best_conf_th_2, train_best_pre_nms_max_det_2, train_best_post_nms_max_det_2,\
            train_ious_2, train_iou_with_boostrapping_2 =\
            _compute_iou_2('train', train_preds_and_gt, train_class_to_indices)
        train_preds_and_gt['ious_2'] = train_ious_2
        # Val
        val_best_iou_th_2, val_best_conf_th_2, val_best_pre_nms_max_det_2, val_best_post_nms_max_det_2,\
            val_ious_2, val_iou_with_boostrapping_2 =\
            _compute_iou_2('val', val_preds_and_gt, val_class_to_indices)
        val_preds_and_gt['ious_2'] = val_ious_2
        # Test
        test_best_iou_th_2, test_best_conf_th_2, test_best_pre_nms_max_det_2, test_best_post_nms_max_det_2,\
            test_ious_2, test_iou_with_boostrapping_2 =\
            _compute_iou_2('test', test_preds_and_gt, test_class_to_indices)
        test_preds_and_gt['ious_2'] = test_ious_2

        # Compute average classification probability on train, val, and test sets with bootstrapping

        def _compute_avg_classification_prob(split, preds_and_gt, class_to_indices):
            print_blue(f'Computing average classification probability on the {split} set ...', bold=True)
            avg_classification_prob_with_boostrapping = apply_stratified_bootstrapping(
                metric_values=preds_and_gt['pred_classification_probs'], class_to_indices=class_to_indices,
                class_names=category_names, metric_name='avg_classification_prob', num_bootstraps=500, num_processes=6)
            print_bold(f'{split.capitalize()} Avg Classification Prob with bootstrapping:')
            pprint(avg_classification_prob_with_boostrapping)
            return avg_classification_prob_with_boostrapping
        
        # Train
        train_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob(
            'train', train_preds_and_gt, train_class_to_indices)
        # Val
        val_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob(
            'val', val_preds_and_gt, val_class_to_indices)
        # Test
        test_avg_classification_prob_with_bootstrappig = _compute_avg_classification_prob(
            'test', test_preds_and_gt, test_class_to_indices)

        # Save metrics to file
        print_blue('Saving metrics to file ...', bold=True)
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        save_path = os.path.join(results_folder_path, f'mscxr_metrics.pkl')
        output = dict(
            train_preds_and_gt=train_preds_and_gt,
            val_preds_and_gt=val_preds_and_gt,
            test_preds_and_gt=test_preds_and_gt,
            train_iou_with_boostrapping=train_iou_with_boostrapping,
            val_iou_with_boostrapping=val_iou_with_boostrapping,
            test_iou_with_boostrapping=test_iou_with_boostrapping,
            train_iou_with_boostrapping_2=train_iou_with_boostrapping_2,
            val_iou_with_boostrapping_2=val_iou_with_boostrapping_2,
            test_iou_with_boostrapping_2=test_iou_with_boostrapping_2,
            train_avg_classification_prob_with_bootstrappig=train_avg_classification_prob_with_bootstrappig,
            val_avg_classification_prob_with_bootstrappig=val_avg_classification_prob_with_bootstrappig,
            test_avg_classification_prob_with_bootstrappig=test_avg_classification_prob_with_bootstrappig,
            train_best_conf_th=train_best_conf_th,
            val_best_conf_th=val_best_conf_th,
            test_best_conf_th=test_best_conf_th,
            train_best_iou_th_2=train_best_iou_th_2,
            val_best_iou_th_2=val_best_iou_th_2,
            test_best_iou_th_2=test_best_iou_th_2,
            train_best_conf_th_2=train_best_conf_th_2,
            val_best_conf_th_2=val_best_conf_th_2,
            test_best_conf_th_2=test_best_conf_th_2,
            train_best_pre_nms_max_det_2=train_best_pre_nms_max_det_2,
            val_best_pre_nms_max_det_2=val_best_pre_nms_max_det_2,
            test_best_pre_nms_max_det_2=test_best_pre_nms_max_det_2,
            train_best_post_nms_max_det_2=train_best_post_nms_max_det_2,
            val_best_post_nms_max_det_2=val_best_post_nms_max_det_2,
            test_best_post_nms_max_det_2=test_best_post_nms_max_det_2,
        )
        save_pickle(output, save_path)
        print_bold(f'Saved metrics to {save_path}')

    if eval_chexlocalize:

        count_print('----- Evaluating on CheXlocalize Phrase Grounding -----')

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)

        # Attach metrics
        metrics_to_print = []
        _cond_func = lambda x: x['flag'] == 'cl'
        attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
                                                      nc=len(CHEXLOCALIZE_CLASS_NAMES),
                                                      condition_function=_cond_func)
        attach_condition_aware_accuracy(evaluation_engine, 'pred_labels', 'gt_labels', 'classif_acc', _cond_func)
        metrics_to_print.append('segmask_iou')
        metrics_to_print.append('classif_acc')

        # Attach accumulators
        attach_accumulator(evaluation_engine, 'pred_mask')
        attach_accumulator(evaluation_engine, 'gt_mask')
        attach_accumulator(evaluation_engine, 'pred_probs')
        attach_accumulator(evaluation_engine, 'gt_labels')

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
        evaluation_engine.run(chexlocalize_trainer.val_dataloader)

        # Print final metrics
        print_blue('Final metrics:', bold=True)
        metrics = evaluation_engine.state.metrics
        # 1) segmask iou
        metric_name = 'segmask_iou'
        print(f'{metric_name}: {metrics[metric_name]}')
        # 2) classif acc
        metric_name = 'classif_acc'
        print(f'{metric_name}: {metrics[metric_name]}')
        # 3) PRC-AUC
        pred_probs = metrics['pred_probs']
        pred_probs = torch.tensor(pred_probs).cpu().numpy()
        assert pred_probs.ndim == 1
        gt_labels = metrics['gt_labels']
        gt_labels = torch.tensor(gt_labels).cpu().numpy()
        assert gt_labels.ndim == 1
        pred_probs =  pred_probs.reshape(-1, len(CHEXLOCALIZE_CLASS_NAMES))
        gt_labels = gt_labels.reshape(-1, len(CHEXLOCALIZE_CLASS_NAMES))
        assert pred_probs.shape == gt_labels.shape
        assert pred_probs.shape[0] == len(chexlocalize_trainer.val_dataset)
        prc_auc_metrics = prc_auc_fn(pred_probs, gt_labels)
        for class_name, prc_auc in zip(CHEXLOCALIZE_CLASS_NAMES, prc_auc_metrics['per_class']):
            print(f'  PRC-AUC({class_name}): {prc_auc}')
        print(f'PRC-AUC(macro_avg): {prc_auc_metrics["macro_avg"]}')
        print(f'PRC-AUC(micro_avg): {prc_auc_metrics["micro_avg"]}')

        # Save metrics to file
        dataset = chexlocalize_trainer.val_dataset
        image_paths = dataset.image_paths
        phrases = chexlocalize_trainer.class_phrases
        gt_masks = metrics['gt_mask']
        pred_masks = metrics['pred_mask']
        assert len(image_paths) == len(pred_masks) == len(gt_masks)
        print_blue('Saving metrics to file ...', bold=True)
        results_folder_path = get_results_folder_path(checkpoint_folder_path)
        save_path = os.path.join(results_folder_path, f'chexlocalize_metrics.pkl')
        output = dict(
            image_paths=[],
            phrases=[],
            pred_masks=[],
            gt_masks=[],
            ious=[],
            segmask_iou=metrics['segmask_iou'],
            classif_acc=metrics['classif_acc'],
            prc_auc=prc_auc_metrics,
        )
        for i in range(len(image_paths)):
            for j in range(len(pred_masks[i])):
                if gt_labels[i,j] == 1:
                    intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
                    union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
                    iou = intersection / union
                    iou = iou.item()
                    output['image_paths'].append(image_paths[i])
                    output['phrases'].append(phrases[j])
                    output['pred_masks'].append(pred_masks[i][j].cpu().numpy())
                    output['gt_masks'].append(gt_masks[i][j].cpu().numpy())
                    output['ious'].append(iou)
                else:
                    assert torch.all(gt_masks[i][j] == 0)

        print_magenta('mean_iou =', sum(output['ious']) / len(output['ious']), bold=True)
        save_pickle(output, save_path)
        print(f'Saved metrics to {save_path}')

    if eval_vinbig:

        count_print('----- Evaluating on VinDr-CXR Phrase Grounding -----')

        # Create evaluation engine
        print_blue('Creating evaluation engine ...', bold=True)
        if optimize_thresholds:
            evaluation_engine_kwargs['skip_nms'] = True # We need to skip NMS to optimize thresholds
        evaluation_engine_kwargs['use_amp'] = use_amp
        evaluation_engine = get_engine(model=model, device=device, **evaluation_engine_kwargs)
        use_vinbig_with_modified_labels = vinbig_trainer_kwargs.get('use_vinbig_with_modified_labels', False)
        
        if use_vinbig_with_modified_labels:
            print_orange('NOTE: Using VinDr-CXR with modified labels', bold=True)
            from medvqa.datasets.vinbig import VINBIG_BBOX_NAMES__MODIFIED
            vinbig_bbox_names = VINBIG_BBOX_NAMES__MODIFIED
            vinbig_num_bbox_classes = len(VINBIG_BBOX_NAMES__MODIFIED)
        else:
            vinbig_bbox_names = VINBIG_BBOX_NAMES
            vinbig_num_bbox_classes = VINBIG_NUM_BBOX_CLASSES            

        # Attach metrics
        metrics_to_print = []
        _cond_func = lambda x: x['flag'] == 'vbg'
        if do_visual_grounding_with_bbox_regression:
            if not optimize_thresholds:
                if use_fact_conditioned_yolo:
                    attach_condition_aware_bbox_iou_per_class(evaluation_engine,
                                                        field_names=['yolo_predictions', 'vinbig_bbox_coords', 'vinbig_bbox_classes'],
                                                        metric_name='bbox_iou', nc=vinbig_num_bbox_classes, condition_function=_cond_func,
                                                        for_vinbig=True, use_fact_conditioned_yolo=True)
                else:
                    attach_condition_aware_bbox_iou_per_class(evaluation_engine,
                                                            field_names=['predicted_bboxes', 'vinbig_bbox_coords', 'vinbig_bbox_classes'],
                                                            metric_name='bbox_iou', nc=vinbig_num_bbox_classes, condition_function=_cond_func,
                                                            for_vinbig=True)
                metrics_to_print.append('bbox_iou')
        if do_visual_grounding_with_segmentation:
            attach_condition_aware_segmask_iou_per_class(evaluation_engine, 'pred_mask', 'gt_mask', 'segmask_iou',
                                                        nc=vinbig_num_bbox_classes, condition_function=_cond_func)
            metrics_to_print.append('segmask_iou')

        # Attach accumulators
        attach_accumulator(evaluation_engine, 'pred_probs')
        attach_accumulator(evaluation_engine, 'gt_labels')
        if do_visual_grounding_with_bbox_regression:
            if use_fact_conditioned_yolo:
                if optimize_thresholds:
                    attach_accumulator(evaluation_engine, 'yolo_predictions', append_instead_of_extend=True)
                    attach_accumulator(evaluation_engine, 'resized_shape', append_instead_of_extend=True)
                else:
                    attach_accumulator(evaluation_engine, 'yolo_predictions')
            else:
                if optimize_thresholds:
                    attach_accumulator(evaluation_engine, 'pred_bbox_probs')
                    attach_accumulator(evaluation_engine, 'pred_bbox_coords')
                else:
                    attach_accumulator(evaluation_engine, 'predicted_bboxes')
            attach_accumulator(evaluation_engine, 'vinbig_bbox_coords')
            attach_accumulator(evaluation_engine, 'vinbig_bbox_classes')
        elif do_visual_grounding_with_segmentation:
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

        # Run evaluation
        print_blue('Running engine ...', bold=True)
        evaluation_engine.run(vinbig_trainer.val_dataloader)
        metrics = evaluation_engine.state.metrics

        # Print some running metrics
        
        # 1) bbox iou
        if do_visual_grounding_with_bbox_regression:
            if not optimize_thresholds:
                metric_name = 'bbox_iou'
                print(f'{metric_name}: {metrics[metric_name]}')
        # 1) segmask iou
        if do_visual_grounding_with_segmentation:
            metric_name = 'segmask_iou'
            print(f'{metric_name}: {metrics[metric_name]}')

        # Compute metrics and prepare output to save to file

        output_to_save = dict()
        
        # --- Classification metrics

        dataset = vinbig_trainer.val_dataset
        phrases = vinbig_trainer.phrases
        classification_label_names = vinbig_trainer.actual_label_names[:] # copy
        pred_probs = metrics['pred_probs']
        pred_probs = torch.stack(pred_probs).cpu().numpy()
        assert pred_probs.ndim == 2
        gt_labels = metrics['gt_labels']
        gt_labels = torch.stack(gt_labels).cpu().numpy()
        assert gt_labels.ndim == 2
        assert pred_probs.shape == gt_labels.shape
        assert pred_probs.shape[0] == len(dataset)

        classif_pred_probs = pred_probs.copy() # (num_samples, num_classes)
        classif_gt_labels = gt_labels.copy() # (num_samples, num_classes)

        # Convert "Abnormal finding" to "No finding" if applicable
        if use_vinbig_with_modified_labels:
            print_orange('NOTE: Converting "Abnormal finding" to "No finding" for VinDr-CXR with modified labels', bold=True)
            assert "Abnormal finding" in classification_label_names
            assert "No finding" not in classification_label_names
            abnormal_finding_idx = classification_label_names.index("Abnormal finding")
            classification_label_names[abnormal_finding_idx] = "No finding"
            classif_gt_labels[:, abnormal_finding_idx] = np.logical_not(classif_gt_labels[:, abnormal_finding_idx])
            classif_pred_probs[:, abnormal_finding_idx] = 1 - classif_pred_probs[:, abnormal_finding_idx]

        # Remove classes without any ground truth
        classif_gt_labels_sum = classif_gt_labels.sum(axis=0)
        no_gt_classes = np.where(classif_gt_labels_sum == 0)[0]
        if len(no_gt_classes) > 0:
            print_orange('NOTE: Removing the following classes without any positive classification labels:', bold=True)
            for i in no_gt_classes:
                print_orange(f'  {classification_label_names[i]}')
            classif_pred_probs = np.delete(classif_pred_probs, no_gt_classes, axis=1)
            classif_gt_labels = np.delete(classif_gt_labels, no_gt_classes, axis=1)
            classification_label_names = [x for i, x in enumerate(classification_label_names) if i not in no_gt_classes]
            print(f'classif_pred_probs.shape = {classif_pred_probs.shape}')
            print(f'classif_gt_labels.shape = {classif_gt_labels.shape}')
            print(f'len(classification_label_names) = {len(classification_label_names)}')

        # Compute PRC-AUC without bootstrapping
        prc_auc_metrics = prc_auc_fn(classif_gt_labels, classif_pred_probs)

        # Compute PRC-AUC with bootstrapping
        prc_auc_metrics_with_boot = stratified_multilabel_bootstrap_metrics(
            gt_labels=classif_gt_labels, pred_probs=classif_pred_probs, metric_fn=prc_auc_score, num_bootstraps=500)
        
        # Save classification metrics to output
        output_to_save['classification'] = dict(
            classification_label_names=classification_label_names,
            pred_probs=classif_pred_probs,
            gt_labels=classif_gt_labels,
            prc_auc=prc_auc_metrics,
            prc_auc_with_bootstrapping=prc_auc_metrics_with_boot,
        )

        # Print some classification metrics
        for class_name, mean, std in zip(classification_label_names,
                                         prc_auc_metrics_with_boot['mean_per_class'],
                                         prc_auc_metrics_with_boot['std_per_class']):
            if class_name in VINBIG_RAD_DINO_CLASSES:
                print_bold(f'PRC-AUC({class_name}): {mean} ± {std}')
            else:
                print(f'PRC-AUC({class_name}): {mean} ± {std}')
        print_magenta(f'PRC-AUC(macro_avg) with bootstrapping: {prc_auc_metrics_with_boot["mean_macro_avg"]} ± {prc_auc_metrics_with_boot["std_macro_avg"]}', bold=True)
        print_magenta(f'PRC-AUC(macro_avg): {prc_auc_metrics["macro_avg"]}', bold=True)

        #  --- Visual grounding (object detection / segmentation) metrics

        image_paths = [dataset.image_paths[i] for i in dataset.indices]
        output_to_save['image_paths'] = image_paths # for saving to file

        if do_visual_grounding_with_bbox_regression:

            for iou_thr in VINBIG_CHEX_IOU_THRESHOLDS: assert iou_thr in map_iou_thresholds
            assert VINBIGDATA_CHALLENGE_IOU_THRESHOLD in map_iou_thresholds

            gt_bboxes = metrics['vinbig_bbox_coords']
            gt_classes = metrics['vinbig_bbox_classes']
            assert len(image_paths) == len(gt_bboxes) == len(gt_classes)
            gt_coords_list = [[[] for _ in range(vinbig_num_bbox_classes)] for _ in range(len(gt_bboxes))]
            for i in range(len(gt_bboxes)):
                for bbox, cls in zip(gt_bboxes[i], gt_classes[i]):
                    gt_coords_list[i][cls].append(bbox)
            
            # Convert to numpy
            for i in range(len(gt_coords_list)):
                for j in range(len(gt_coords_list[i])):
                    gt_coords_list[i][j] = np.stack(gt_coords_list[i][j]) if len(gt_coords_list[i][j]) > 0 else np.empty((0, 4))

            if use_classifier_confs_for_map:
                classifier_confs = pred_probs[:, :vinbig_num_bbox_classes] # (num_samples, num_classes)
                print_bold('Using classifier confidences for mAP computation')
                print(f'classifier_confs.shape = {classifier_confs.shape}')
            else:
                classifier_confs = None
            
            if optimize_thresholds: # Optimize thresholds
                assert candidate_iou_thresholds is not None
                assert candidate_conf_thresholds is not None
                num_classes = vinbig_num_bbox_classes
                if use_fact_conditioned_yolo:
                    out = find_optimal_conf_iou_thresholds(
                        gt_coords_list=gt_coords_list,
                        yolo_predictions_list=metrics['yolo_predictions'],
                        resized_shape_list=metrics['resized_shape'],
                        is_fact_conditioned_yolo=True,
                        iou_thresholds=candidate_iou_thresholds,
                        conf_thresholds=candidate_conf_thresholds,
                        classifier_confs=classifier_confs,
                        num_classes=num_classes,
                        verbose=True,
                    )
                else:
                    pred_bbox_probs = metrics['pred_bbox_probs']
                    pred_bbox_coords = metrics['pred_bbox_coords']
                    num_regions = pred_bbox_probs[0].shape[1]
                    assert pred_bbox_probs[0].ndim == 3 # (num_classes, num_regions, 1)
                    assert pred_bbox_coords[0].ndim == 3 # (num_classes, num_regions, 4)
                    assert pred_bbox_probs[0].shape == (num_classes, num_regions, 1)
                    assert pred_bbox_coords[0].shape == (num_classes, num_regions, 4)
                    out = find_optimal_conf_iou_thresholds(
                        gt_coords_list=gt_coords_list,
                        pred_boxes_list=pred_bbox_coords,
                        pred_confs_list=pred_bbox_probs,
                        iou_thresholds=candidate_iou_thresholds,
                        conf_thresholds=candidate_conf_thresholds,
                        classifier_confs=classifier_confs,
                        verbose=True,
                    )
                best_iou_threshold = out['best_iou_threshold']
                best_conf_threshold = out['best_conf_threshold']
                pred_boxes_list = out['pred_boxes_list']
                pred_classes_list = out['pred_classes_list']
                pred_confs_list = out['pred_confs_list']

            else:
                if use_fact_conditioned_yolo:
                    pred_bboxes = metrics['yolo_predictions']
                    assert len(image_paths) == len(pred_bboxes)
                    pred_boxes_list = []
                    pred_confs_list = []
                    pred_classes_list = []
                    for preds in pred_bboxes:
                        assert len(preds) == vinbig_num_bbox_classes
                        boxes = []
                        confs = []
                        classes = []
                        for i, pred in enumerate(preds):
                            pred = pred.cpu().numpy()
                            boxes.append(pred[:, :4])
                            confs.append(pred[:, 4])
                            classes.append(np.full((len(pred),), i))
                        pred_boxes_list.append(np.concatenate(boxes, axis=0))
                        pred_confs_list.append(np.concatenate(confs, axis=0))
                        pred_classes_list.append(np.concatenate(classes, axis=0))
                else:
                    pred_bboxes = metrics['predicted_bboxes']
                    assert len(image_paths) == len(pred_bboxes)
                    pred_boxes_list = []
                    pred_confs_list = []
                    pred_classes_list = []
                    for preds in pred_bboxes:
                        assert len(preds) == 3 # (boxes, confs, classes)
                        pred_boxes_list.append(preds[0].cpu().numpy())
                        pred_confs_list.append(preds[1].cpu().numpy())
                        pred_classes_list.append(preds[2].cpu().numpy())

            # Remove classes without any ground truth
            gt_counts_per_class = np.zeros(vinbig_num_bbox_classes, dtype=int)
            for i in range(len(gt_coords_list)):
                for j in range(len(gt_coords_list[i])):
                    gt_counts_per_class[j] += len(gt_coords_list[i][j])
            no_gt_classes = np.where(gt_counts_per_class == 0)[0]
            with_gt_classes = np.where(gt_counts_per_class > 0)[0]
            if len(no_gt_classes) > 0:
                print_orange('NOTE: Removing the following classes without any bounding box annotations:', bold=True)
                for i in no_gt_classes:
                    print_orange(f'  {vinbig_bbox_names[i]}')
                print(f'gt_counts_per_class = {gt_counts_per_class}')
                # Clean gt_coords_list
                for i in range(len(gt_coords_list)):
                    gt_coords_list[i] = [gt_coords_list[i][j] for j in with_gt_classes]
                print(f'len(gt_coords_list) = {len(gt_coords_list)}')
                print(f'len(gt_coords_list[0]) = {len(gt_coords_list[0])}')
                # Clean classifier_confs
                if use_classifier_confs_for_map:
                    classifier_confs = classifier_confs[:, with_gt_classes]
                    print(f'classifier_confs.shape = {classifier_confs.shape}')
                # Clean vinbig_bbox_names
                vinbig_bbox_names = [x for i, x in enumerate(vinbig_bbox_names) if i in with_gt_classes]
                vinbig_num_bbox_classes = len(vinbig_bbox_names)
                print(f'len(vinbig_bbox_names) = {len(vinbig_bbox_names)}')
                # Clean pred_boxes_list, pred_classes_list, pred_confs_list
                old_class_idx_to_new_class_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(with_gt_classes)}
                for i in range(len(pred_boxes_list)):
                    if len(pred_classes_list[i]) > 0:
                        valid_idxs = np.where(np.isin(pred_classes_list[i], with_gt_classes))[0]
                        pred_boxes_list[i] = pred_boxes_list[i][valid_idxs]
                        pred_confs_list[i] = pred_confs_list[i][valid_idxs]
                        pred_classes_list[i] = np.array([old_class_idx_to_new_class_idx[x] for x in pred_classes_list[i][valid_idxs]])
                print(f'len(pred_boxes_list) = {len(pred_boxes_list)}')
                print(f'len(pred_classes_list) = {len(pred_classes_list)}')
                print(f'len(pred_confs_list) = {len(pred_confs_list)}')


            if optimize_thresholds: # Print optimal thresholds
                print_magenta(f'best_iou_threshold: {best_iou_threshold}', bold=True)
                print_magenta(f'best_conf_threshold: {best_conf_threshold}', bold=True)

            # Compute metrics without bootstrapping
            
            # 1. IoU
            tmp = compute_mean_iou_per_class__yolov11(
                pred_boxes=pred_boxes_list,
                pred_classes=pred_classes_list,
                gt_coords=gt_coords_list,
                compute_iou_per_sample=True,
                compute_micro_average_iou=True,
                return_counts=True,
            )
            class_ious = tmp['class_ious']
            sample_ious = tmp['sample_ious']
            class_counts = tmp['class_counts']
            sample_counts = tmp['sample_counts']
            micro_iou = tmp['micro_iou']
            class_idxs = np.where(class_counts > 0)[0]
            macro_iou = class_ious[class_idxs].mean()
            sample_idxs = np.where(sample_counts > 0)[0]
            sample_iou = sample_ious[sample_idxs].mean()

            # 2. mAP
            tmp = compute_mAP__yolov11(
                pred_boxes=pred_boxes_list,
                pred_classes=pred_classes_list,
                pred_confs=pred_confs_list,
                classifier_confs=classifier_confs,
                gt_coords=gt_coords_list,
                iou_thresholds=map_iou_thresholds,
                compute_micro_average=True,
            )
            class_aps = tmp['class_aps']
            micro_aps = tmp['micro_aps']

            # 2.1 vinbigdata mAP
            class_idxs = [vinbig_bbox_names.index(x) for x in VINBIGDATA_CHALLENGE_CLASSES]
            iou_idx = map_iou_thresholds.index(VINBIGDATA_CHALLENGE_IOU_THRESHOLD)
            vbdc_mAP = class_aps[iou_idx, class_idxs].mean() # vbdc = vinbigdata challenge

            # 2.2 ChEX mAP
            class_idxs = [vinbig_bbox_names.index(x) for x in VINBIG_CHEX_CLASSES]
            iou_idxs = [map_iou_thresholds.index(x) for x in VINBIG_CHEX_IOU_THRESHOLDS]
            chex_mAP = class_aps[iou_idxs][:, class_idxs].mean()

            # Update output
            output_to_save['detection'] = dict(
                pred_boxes_list=pred_boxes_list,
                pred_classes_list=pred_classes_list,
                pred_confs_list=pred_confs_list,
                classifier_confs=classifier_confs,
                gt_bboxes=gt_coords_list,
                bbox_class_names=vinbig_bbox_names,
                map_iou_thresholds=map_iou_thresholds, # (num_iou_thresholds,)
                sample_ious=sample_ious, # (num_samples,)
                metrics_without_bootstrapping=dict(
                    class_ious=class_ious, # (num_classes,)
                    micro_iou=micro_iou, # scalar
                    macro_iou=macro_iou, # scalar
                    class_aps=class_aps, # (num_iou_thresholds, num_classes)
                    micro_aps=micro_aps, # (num_iou_thresholds,)
                    vbdc_mAP=vbdc_mAP, # scalar
                    chex_mAP=chex_mAP, # scalar
                ),
            )

            # Compute metrics with bootstrapping
            iou_map_metrics_with_boot = stratified_vinbig_bootstrap_iou_map(
                pred_boxes_list=pred_boxes_list,
                pred_classes_list=pred_classes_list,
                pred_confs_list=pred_confs_list,
                classifier_confs=classifier_confs,
                gt_coords_list=gt_coords_list,
                vinbig_bbox_names=vinbig_bbox_names,
                map_iou_thresholds=map_iou_thresholds,
                compute_mean_iou_per_class_fn=compute_mean_iou_per_class__yolov11,
                compute_mAP_fn=compute_mAP__yolov11,
                num_bootstraps=60,
                num_processes=12,
            )

            # Update output
            output_to_save['detection']['metrics_with_bootstrapping'] = iou_map_metrics_with_boot

            if optimize_thresholds:
                output_to_save['best_iou_threshold'] = best_iou_threshold
                output_to_save['best_conf_threshold'] = best_conf_threshold

            # Print some metrics
            for class_name, iou, count in zip(vinbig_bbox_names, class_ious, class_counts):
                print(f'mean_iou({class_name}): {iou} ({count} samples)')
            print_magenta(f'macro_iou: {macro_iou}', bold=True)
            print_magenta(f'mean_sample_iou: {sample_iou}', bold=True)
            print(f'\t{sample_idxs.shape[0]} / {len(gt_coords_list)} samples have at least one ground truth bbox')
            print_magenta(f'micro_iou: {micro_iou} (count={sample_counts.sum()})', bold=True)
            
            for iou_thresh, map_ in zip(map_iou_thresholds, class_aps.mean(axis=1)):
                print_magenta(f'mAP@{iou_thresh}: {map_}', bold=True)

            for iou_thresh, ap in zip(map_iou_thresholds, micro_aps):
                print_magenta(f'micro_AP@{iou_thresh}: {ap}', bold=True)

            print_magenta(f'vbdc_mAP: {vbdc_mAP}', bold=True)
            print_magenta(f'chex_mAP: {chex_mAP}', bold=True)

            print_magenta(f'micro_iou (with bootstrap): {iou_map_metrics_with_boot["micro_iou"]["mean"]} ± {iou_map_metrics_with_boot["micro_iou"]["std"]}', bold=True)
            print_magenta(f'macro_iou (with bootstrap): {iou_map_metrics_with_boot["macro_iou"]["mean"]} ± {iou_map_metrics_with_boot["macro_iou"]["std"]}', bold=True)
            print_magenta(f'vbdc_mAP (with bootstrap): {iou_map_metrics_with_boot["vbdc_mAP"]["mean"]} ± {iou_map_metrics_with_boot["vbdc_mAP"]["std"]}', bold=True)
            print_magenta(f'chex_mAP (with bootstrap): {iou_map_metrics_with_boot["chex_mAP"]["mean"]} ± {iou_map_metrics_with_boot["chex_mAP"]["std"]}', bold=True)
            
            # Save metrics to file
            print_blue('Saving metrics to file ...', bold=True)
            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            strings = [
                'detection',
                f'{len(vinbig_trainer.val_dataset)}',
            ]
            if optimize_thresholds:
                strings.append(f'opt_thr({best_iou_threshold:.2f},{best_conf_threshold:.2f})')
            if use_classifier_confs_for_map:
                strings.append('use_classifier_confs')
            save_path = os.path.join(results_folder_path, f'vindrcxr_metrics({",".join(strings)}).pkl')

        elif do_visual_grounding_with_segmentation:

            gt_masks = metrics['gt_mask']
            pred_masks = metrics['pred_mask']
            assert len(image_paths) == len(pred_masks) == len(gt_masks),\
                (f'len(image_paths) = {len(image_paths)}, len(pred_masks) = {len(pred_masks)}, len(gt_masks) = {len(gt_masks)}')
            print_blue('Saving metrics to file ...', bold=True)
            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            save_path = os.path.join(results_folder_path, f'vindrcxr_metrics(segmask,{len(vinbig_trainer.val_dataset)}).pkl')
            output_to_save = dict(
                image_paths=[],
                phrases=[],
                pred_masks=[],
                gt_masks=[],
                ious=[],
                segmask_iou=metrics['segmask_iou'],
                prc_auc=prc_auc_metrics,
            )
            for i in range(len(image_paths)):
                for j in range(len(pred_masks[i])):
                    if gt_labels[i, j] == 1:
                        intersection = torch.min(pred_masks[i][j], gt_masks[i][j]).sum()
                        union = torch.max(pred_masks[i][j], gt_masks[i][j]).sum()
                        iou = intersection / union
                        iou = iou.item()
                        output_to_save['image_paths'].append(image_paths[i])
                        output_to_save['phrases'].append(phrases[j])
                        output_to_save['pred_masks'].append(pred_masks[i][j].cpu().numpy())
                        output_to_save['gt_masks'].append(gt_masks[i][j].cpu().numpy())
                        output_to_save['ious'].append(iou)
                    else:
                        assert torch.all(gt_masks[i][j] == 0)

            print_magenta('mean_iou =', sum(output_to_save['ious']) / len(output_to_save['ious']), bold=True)

        else:

            results_folder_path = get_results_folder_path(checkpoint_folder_path)
            save_path = os.path.join(results_folder_path, f'vindrcxr_metrics(classification,{len(vinbig_trainer.val_dataset)}).pkl')
        
        save_pickle(output_to_save, save_path)
        print(f'Saved metrics to {save_path}')

def evaluate(
    checkpoint_folder_path,
    num_workers,
    max_images_per_batch,
    max_phrases_per_batch,
    max_phrases_per_image,
    eval_chest_imagenome_gold,
    eval_mscxr,
    eval_chexlocalize,
    eval_vinbig,
    mscxr_phrase2embedding_filepath,
    mimicxr_dicom_id_to_pos_neg_facts_filepath,
    device,
    vinbig_use_training_indices_for_validation,
    optimize_thresholds,
    candidate_iou_thresholds,
    candidate_conf_thresholds,
    map_iou_thresholds,
    use_amp,
    use_classifier_confs_for_map,
    checkpoint_folder_path_to_borrow_metadata_from,
    override_bbox_format,
):  
    # Force deterministic behavior
    activate_determinism()
    
    print_blue('----- Evaluating model -----', bold=True)

    metadata = load_metadata(checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    mimiccxr_trainer_kwargs = metadata['mimiccxr_trainer_kwargs']
    chexlocalize_trainer_kwargs = metadata['chexlocalize_trainer_kwargs']
    vinbig_trainer_kwargs = metadata['vinbig_trainer_kwargs']
    # collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
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
                chexlocalize_trainer_kwargs=chexlocalize_trainer_kwargs,
                vinbig_trainer_kwargs=vinbig_trainer_kwargs,
                # collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                evaluation_engine_kwargs=validator_engine_kwargs,
                max_images_per_batch=max_images_per_batch,
                max_phrases_per_batch=max_phrases_per_batch,
                max_phrases_per_image=max_phrases_per_image,
                num_workers=num_workers,
                eval_chest_imagenome_gold=eval_chest_imagenome_gold,
                eval_mscxr=eval_mscxr,
                eval_chexlocalize=eval_chexlocalize,
                eval_vinbig=eval_vinbig,
                mscxr_phrase2embedding_filepath=mscxr_phrase2embedding_filepath,
                mimicxr_dicom_id_to_pos_neg_facts_filepath=mimicxr_dicom_id_to_pos_neg_facts_filepath,
                device=device,
                vinbig_use_training_indices_for_validation=vinbig_use_training_indices_for_validation,
                optimize_thresholds=optimize_thresholds,
                candidate_iou_thresholds=candidate_iou_thresholds,
                candidate_conf_thresholds=candidate_conf_thresholds,
                map_iou_thresholds=map_iou_thresholds,
                use_amp=use_amp,
                use_classifier_confs_for_map=use_classifier_confs_for_map,
                checkpoint_folder_path_to_borrow_metadata_from=checkpoint_folder_path_to_borrow_metadata_from,
                override_bbox_format=override_bbox_format,
            )


if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)