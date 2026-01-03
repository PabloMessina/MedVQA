import math
import os
import random
import textwrap

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm

from medvqa.datasets.chestxdet.chestxdet_phrase_grounding_dataset_management import (
    polygons_to_mask,
)
from medvqa.evaluation.bootstrapping import (
    apply_bootstrapping,
)
from medvqa.metrics.bbox.utils import (
    compute_bbox_union_iou,
    compute_probability_map_iou,
    find_optimal_probability_map_conf_threshold,
)
from medvqa.utils.bbox_utils import convert_bboxes_into_presence_map
from medvqa.utils.constants import VINBIG_LABEL2PHRASE
from medvqa.utils.files_utils import load_jsonl, load_pickle, save_pickle
from medvqa.utils.metrics_utils import (
    calculate_cnr,
    calculate_dice,
    calculate_rand_index,
    calculate_segmentation_iou,
    calculate_segmentation_precision,
    calculate_segmentation_recall,
    calculate_soft_dice,
    find_optimal_probability_map_conf_threshold_for_masks,
)


class PhraseGroundingResultsVisualizer:
    def __init__(self, experiment_results):
        """
        Args:
            experiment_results (list): List of dicts with keys: 
                                       "experiment_name", "pred_and_gt_path", "metrics_path"
        """
        self.experiments = experiment_results
        self.metrics_data = {}
        self.preds_data = {}  # Cache for heavy prediction files
        self._load_all_metrics()
        
    def _load_all_metrics(self):
        """Loads metrics. For internal models, loads pkl. For external, computes them."""
        for experiment in self.experiments:
            alias = experiment["experiment_name"]
            model_type = experiment.get("model_type", "internal") # Default to internal
            
            if model_type == "internal":
                metrics_path = experiment["metrics_path"]
                if not os.path.exists(metrics_path):
                    print(f"Warning: File not found {metrics_path}")
                    continue
                self.metrics_data[alias] = load_pickle(metrics_path)

            elif model_type == "maira-2":
                print(f"Computing metrics for {alias}...")
                pred_path = experiment["pred_and_gt_path"]
                data = load_jsonl(pred_path)
                metrics_path = pred_path + ".metrics.pkl"
                if os.path.exists(metrics_path):
                    print(f"Loading precomputed metrics for {alias} at {metrics_path}...")
                    self.metrics_data[alias] = load_pickle(metrics_path)
                else:
                    print(f"No precomputed metrics found for {alias} at {metrics_path}. Computing metrics and saving to {metrics_path}...")
                    self.metrics_data[alias] = self._compute_maira2_metrics(data)
                    save_pickle(self.metrics_data[alias], metrics_path)

            elif model_type == "biovil-t":
                print(f"Computing metrics for {alias}...")
                pred_path = experiment["pred_and_gt_path"]
                data = load_pickle(pred_path)
                metrics_path = pred_path + ".metrics.pkl"
                if os.path.exists(metrics_path):
                    print(f"Loading precomputed metrics for {alias} at {metrics_path}...")
                    self.metrics_data[alias] = load_pickle(metrics_path)
                else:
                    print(f"No precomputed metrics found for {alias} at {metrics_path}. Computing metrics and saving to {metrics_path}...")
                    self.metrics_data[alias] = self._compute_biovilt_metrics(data)
                    save_pickle(self.metrics_data[alias], metrics_path)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
    
    def _xyxy_to_cxcywh(self, boxes):
        """Converts boxes from (x1, y1, x2, y2) to (cx, cy, w, h)."""
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        if boxes.size == 0:
            return boxes
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return np.stack([cx, cy, w, h], axis=-1)
    
    def _get_sample_key(self, image_path, phrase):
        """Generates a unique key for alignment: (image_filename_no_ext, phrase)."""
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return (stem, phrase)

    def _get_preds_data(self, alias):
        """
        Lazily loads predictions and converts them into a lookup dictionary 
        keyed by (image_stem, phrase).
        """
        if alias in self.preds_data:
            return self.preds_data[alias]
        
        exp_info = next((e for e in self.experiments if e["experiment_name"] == alias), None)
        if not exp_info:
            raise ValueError(f"Alias {alias} not found.")
            
        path = exp_info["pred_and_gt_path"]
        model_type = exp_info.get("model_type", "internal")
        
        lookup_data = {} # Structure: { split_key: { (stem, phrase): {data..., '_original_idx': i} } }

        if model_type == "internal":
            print(f"Loading predictions for {alias}...")
            raw_data = load_pickle(path)
            
            # Internal format is usually { 'test_preds_and_gt': {'image_paths': [...], ...} }
            for split_key, split_data in raw_data.items():
                if not isinstance(split_data, dict) or 'image_paths' not in split_data:
                    continue
                
                lookup_data[split_key] = {}
                n_samples = len(split_data['image_paths'])
                
                # Assume all list-columns have the same length
                keys = list(split_data.keys())
                
                for i in range(n_samples):
                    img_path = split_data['image_paths'][i]
                    phrase = split_data['phrases'][i]
                    unique_key = self._get_sample_key(img_path, phrase)
                    
                    # Create a row object for this sample
                    row_obj = {k: split_data[k][i] for k in keys}
                    row_obj['_original_index'] = i
                    lookup_data[split_key][unique_key] = row_obj

        elif model_type == "maira-2":
            print(f"Loading/Adapting MAIRA-2 predictions for {alias}...")
            raw_list = load_jsonl(path)
            split_key = 'test_preds_and_gt' # MAIRA-2 files usually implied test/val
            lookup_data[split_key] = {}

            if os.path.basename(path).startswith('vindrcxr'):
                def phrase_getter(row):
                    return VINBIG_LABEL2PHRASE.get(row['phrase'], row['phrase'])
            else:
                def phrase_getter(row):
                    return row['phrase']
            
            for i, row in enumerate(raw_list):
                phrase = phrase_getter(row)
                unique_key = self._get_sample_key(row['image_path'], phrase)
                
                adapted_row = {
                    'image_path': row['image_path'],
                    'phrase': phrase,
                    'gt_bbox_coords': row['gt_bboxes'],
                    'maira2_pred_bboxes': row['predicted_bboxes'] if 'predicted_bboxes' in row else row['predicted_bboxes_maira2'],
                    'pred_bbox_prob_maps': None,
                    'pred_bbox_coord_maps': None,
                    'maira_decoded_text': row['maira_decoded_text'] if 'maira_decoded_text' in row else row['raw_maira2_decoded_text'],
                    'maira_raw_output': row['maira_raw_output'] if 'maira_raw_output' in row else row['raw_maira2_structured_output'],
                    '_original_index': i
                }
                # Handle GT Mask/Polygons if they exist in source
                for key in ['polygons', 'gt_polygons', 'gt_mask']:
                    if key in row:
                        adapted_row[key] = row[key]
                
                lookup_data[split_key][unique_key] = adapted_row

        elif model_type == "biovil-t":
            print(f"Loading/Adapting BioVil-T predictions for {alias}...")
            raw_list = load_pickle(path)

            if os.path.basename(path).startswith('vindrcxr'):
                def phrase_getter(row):
                    return VINBIG_LABEL2PHRASE.get(row['phrase'], row['phrase'])
            else:
                def phrase_getter(row):
                    return row['phrase']
            
            # We must group by split here as BioVil-T files might have mixed splits
            for row in raw_list:
                s_key = f"{row.get('split', 'test')}_preds_and_gt"
                if s_key not in lookup_data:
                    lookup_data[s_key] = {}

                phrase = phrase_getter(row)
                
                unique_key = self._get_sample_key(row['image_path'], phrase)
                i = len(lookup_data[s_key])
                adapted_row = {
                    'image_path': row['image_path'],
                    'phrase': phrase,
                    'gt_bbox_coords': row['gt_bboxes'],
                    'pred_bbox_prob_maps': row['similarity_map'],
                    'pred_bbox_coord_maps': None,
                    '_original_index': i
                }
                # Handle GT Mask/Polygons if they exist in source
                for key in ['polygons', 'gt_polygons', 'gt_mask']:
                    if key in row:
                        adapted_row[key] = row[key]
                lookup_data[s_key][unique_key] = adapted_row

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.preds_data[alias] = lookup_data
        return self.preds_data[alias]

    def _compute_maira2_metrics(self, predictions, split='test'):
        # 1. Prepare data container
        metrics_dict = {f'{split}_metrics': {}}

        # 2. Calculate raw metrics per sample
        values = {'cnrs': [], 'bbox_ious': [], 'prob_ious': [], 'dices': [], 'soft_dices': [], 'precisions': [], 'recalls': [], 'rand_indices': []}
        
        for x in tqdm(predictions, desc="Computing metrics for MAIRA-2", total=len(predictions)):
            gt_bboxes = x['gt_bboxes']
            predicted_bboxes = x['predicted_bboxes'] if 'predicted_bboxes' in x else x['predicted_bboxes_maira2']
            pred_mask = convert_bboxes_into_presence_map(predicted_bboxes, (130, 130))
            if 'polygons' in x:
                gt_mask = polygons_to_mask(x['polygons'], 130, 130)
                values['prob_ious'].append(calculate_segmentation_iou(gt_mask=gt_mask, prob_map=pred_mask))
            else:
                gt_mask = convert_bboxes_into_presence_map(gt_bboxes, (130, 130))
                values['prob_ious'].append(compute_probability_map_iou(gt_bboxes=gt_bboxes, prob_map=pred_mask))
            values['cnrs'].append(calculate_cnr(gt_mask, pred_mask))
            values['bbox_ious'].append(compute_bbox_union_iou(gt_bboxes, predicted_bboxes))
            values['dices'].append(calculate_dice(gt_mask, pred_mask))
            values['soft_dices'].append(calculate_soft_dice(gt_mask, pred_mask))
            values['precisions'].append(calculate_segmentation_precision(gt_mask, pred_mask))
            values['recalls'].append(calculate_segmentation_recall(gt_mask, pred_mask))
            values['rand_indices'].append(calculate_rand_index(gt_mask, pred_mask))

        # 3. Bootstrap and format for get_summary_table
        # Mapping standard metric names to the structure expected by get_summary_table
        key_map = {
            'cnrs': ('cnr', 'cnr_with_bootstrapping'),
            'bbox_ious': ('bbox_iou', 'bbox_iou_with_bootstrapping'),
            'prob_ious': ('prob_iou', 'prob_iou_with_bootstrapping'),
            'dices': ('dice', 'dice_with_bootstrapping'),
            'soft_dices': ('soft_dice', 'soft_dice_with_bootstrapping'),
            'precisions': ('precision', 'precision_with_bootstrapping'),
            'recalls': ('recall', 'recall_with_bootstrapping'),
            'rand_indices': ('rand_index', 'rand_index_with_bootstrapping'),
        }

        for metric_list_name, val_list in values.items():
            metric_name, metric_bootstrapping_name = key_map[metric_list_name]
            stats = apply_bootstrapping(metric_values=val_list,
                                        metric_name=metric_name,
                                        num_bootstraps=200,
                                        num_processes=8,
                                        seed_base=0,
                                        use_tqdm=False,
                                        )
            metrics_dict[f'{split}_metrics'][metric_bootstrapping_name] = stats
            metrics_dict[f'{split}_metrics'][metric_list_name] = val_list

        return metrics_dict

    def _compute_biovilt_metrics(self, predictions, split='test'):
        # Filter by split
        predictions = [x for x in predictions if x.get('split', 'test') == split]

        # Find threshold
        sampled_idxs = random.sample(range(len(predictions)), min(50, len(predictions))) # Sample 50 samples for threshold optimization
        sampled_pred_masks = np.array([predictions[i]['similarity_map'] for i in sampled_idxs])
        if 'polygons' in predictions[0]:
            print('Using polygons for threshold optimization')
            sampled_gt_masks = [polygons_to_mask(predictions[i]['polygons'], 130, 130) for i in sampled_idxs]
            out = find_optimal_probability_map_conf_threshold_for_masks(sampled_pred_masks, sampled_gt_masks)
        else:
            print('Using bboxes for threshold optimization')
            sampled_gt_bboxes = [predictions[i]['gt_bboxes'] for i in sampled_idxs]
            out = find_optimal_probability_map_conf_threshold(sampled_pred_masks, sampled_gt_bboxes)
        best_conf_th = out['best_conf_th']
        
        values = {'cnrs': [], 'prob_ious': [], 'dices': [], 'soft_dices': [], 'precisions': [], 'recalls': [], 'rand_indices': []}
        
        for x in tqdm(predictions, desc="Computing metrics for BioViL-T", total=len(predictions)):
            gt_bboxes = x['gt_bboxes']
            pred_mask = x['similarity_map']
            if 'polygons' in x:
                gt_mask = polygons_to_mask(x['polygons'], 130, 130)
                values['prob_ious'].append(calculate_segmentation_iou(gt_mask=gt_mask, prob_map=pred_mask, threshold=best_conf_th))
            else:
                gt_mask = convert_bboxes_into_presence_map(gt_bboxes, (130, 130))
                values['prob_ious'].append(compute_probability_map_iou(prob_map=pred_mask, gt_bboxes=gt_bboxes, conf_th=best_conf_th))
            values['cnrs'].append(calculate_cnr(gt_mask, pred_mask))
            values['dices'].append(calculate_dice(gt_mask, pred_mask, threshold=best_conf_th))
            values['soft_dices'].append(calculate_soft_dice(gt_mask, pred_mask))
            values['precisions'].append(calculate_segmentation_precision(gt_mask, pred_mask, threshold=best_conf_th))
            values['recalls'].append(calculate_segmentation_recall(gt_mask, pred_mask, threshold=best_conf_th))
            values['rand_indices'].append(calculate_rand_index(gt_mask, pred_mask, threshold=best_conf_th))

        metrics_dict = {f'{split}_metrics': {'best_prob_conf_th': best_conf_th}}
        
        key_map = {
            'cnrs': ('cnr', 'cnr_with_bootstrapping'),
            'prob_ious': ('prob_iou', 'prob_iou_with_bootstrapping'),
            'dices': ('dice', 'dice_with_bootstrapping'),
            'soft_dices': ('soft_dice', 'soft_dice_with_bootstrapping'),
            'precisions': ('precision', 'precision_with_bootstrapping'),
            'recalls': ('recall', 'recall_with_bootstrapping'),
            'rand_indices': ('rand_index', 'rand_index_with_bootstrapping'),
        }

        for metric_list_name, val_list in values.items():
            metric_name, metric_bootstrapping_name = key_map[metric_list_name]
            stats = apply_bootstrapping(metric_values=val_list,
                                        metric_name=metric_name,
                                        num_bootstraps=200,
                                        num_processes=8,
                                        seed_base=0,
                                        use_tqdm=False,
                                        )
            metrics_dict[f'{split}_metrics'][metric_bootstrapping_name] = stats
            metrics_dict[f'{split}_metrics'][metric_list_name] = val_list
            
        return metrics_dict

    def _cxcywh_to_xyxy(self, boxes):
        """Converts boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
        if isinstance(boxes, np.ndarray):
            boxes = torch.tensor(boxes)
        cx, cy, w, h = boxes.unbind(-1)
        b = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
        return torch.stack(b, dim=-1)

    def _filter_predictions(self, bbox_coords, bbox_probs, metrics_dict, bbox_format='cxcywh'):
        """
        Replicates the evaluation logic: Pre-NMS TopK -> Conf Thresh -> NMS -> Post-NMS TopK
        """
        # 1. Retrieve optimal hyperparameters from metrics
        # If keys are missing, default to loose thresholds
        conf_th = metrics_dict.get('best_bbox_conf_th', 0.5)
        iou_th = metrics_dict.get('best_bbox_iou_th', 0.5)
        pre_nms_max = metrics_dict.get('best_bbox_pre_nms_max_det', 100)
        post_nms_max = metrics_dict.get('best_bbox_post_nms_max_det', 10) # Default to reasonable number

        # Convert to torch for NMS ops
        if isinstance(bbox_coords, np.ndarray):
            bbox_coords = torch.tensor(bbox_coords, dtype=torch.float32)
        if isinstance(bbox_probs, np.ndarray):
            bbox_probs = torch.tensor(bbox_probs, dtype=torch.float32)

        if bbox_coords.ndim == 3:
            bbox_coords = bbox_coords.view(-1, 4)
        if bbox_probs.ndim == 2:
            bbox_probs = bbox_probs.view(-1)
        assert bbox_coords.shape[0] == bbox_probs.shape[0], "Number of boxes and probabilities must match"

        # 2. Pre-NMS Top-K
        if bbox_coords.shape[0] > pre_nms_max:
            bbox_probs, idxs = torch.topk(bbox_probs, pre_nms_max)
            bbox_coords = bbox_coords[idxs]

        # 3. Confidence Threshold
        mask = bbox_probs > conf_th
        bbox_coords = bbox_coords[mask]
        bbox_probs = bbox_probs[mask]
        if bbox_coords.shape[0] == 0:
            return np.array([]), np.array([])

        # 4. NMS
        # NMS requires xyxy format
        if bbox_format == 'cxcywh':
            boxes_xyxy = self._cxcywh_to_xyxy(bbox_coords)
        else:
            boxes_xyxy = bbox_coords

        keep_idxs = nms(boxes_xyxy, bbox_probs, iou_th)
        bbox_coords = bbox_coords[keep_idxs]
        bbox_probs = bbox_probs[keep_idxs]

        # 5. Post-NMS Top-K
        if post_nms_max is not None and bbox_coords.shape[0] > post_nms_max:
            bbox_probs, idxs = torch.topk(bbox_probs, post_nms_max)
            bbox_coords = bbox_coords[idxs]

        return bbox_coords.numpy(), bbox_probs.numpy()

    def get_summary_table(self, split='test', metrics_to_show=None):
        """
        Generates a Pandas DataFrame comparing average metrics across methods.

        Args:
            split (str): 'train', 'val', or 'test'. Defaults to 'test'.
            metrics_to_show (list): Optional list of internal metric keys to display.
                                    Defaults to standard grounding metrics.
        
        Returns:
            pd.DataFrame: Summary table of results.
        """
        rows = []
        
        if metrics_to_show is None:
            metrics_map = {
                'CNR': ('cnr_with_bootstrapping', 'cnr'),
                'IoU (Prob)': ('prob_iou_with_bootstrapping', ['iou', 'prob_iou']),
                'IoU (BBox)': ('bbox_iou_with_bootstrapping', ['iou', 'bbox_iou']),
                'Soft Dice': ('soft_dice_with_bootstrapping', 'soft_dice'),
                'Dice': ('dice_with_bootstrapping', 'dice'),
                'Precision': ('precision_with_bootstrapping', 'precision'),
                'Recall': ('recall_with_bootstrapping', 'recall'),
                'Rand Index': ('rand_index_with_bootstrapping', 'rand_index'),
            }
        else:
            metrics_map = metrics_to_show

        for experiment in self.experiments:
            alias = experiment["experiment_name"]
            if alias not in self.metrics_data:
                continue
                
            data = self.metrics_data[alias]
            split_key = f"{split}_metrics"
            
            if split_key not in data:
                print(f"Warning: {split_key} not found for {alias}")
                continue
            
            split_data = data[split_key]
            row = {'Method': alias}
            
            # Extract standard metrics
            for col_name, keys in metrics_map.items():
                if keys is None:
                    continue # Skip derived fields for now
                
                group_key, metric_keys = keys
                if isinstance(metric_keys, str):
                    metric_keys = [metric_keys]
                for metric_key in metric_keys:
                    try:
                        # Navigation: split -> group (e.g. bbox_iou...) -> metric (e.g. iou) -> mean
                        val = split_data[group_key][metric_key]['mean']
                        row[col_name] = val
                        break
                    except KeyError:
                        row[col_name] = None
                # if row[col_name] is None:
                #     print(f"Warning: {alias} and {col_name} (group_key = {group_key}, metric_keys = {metric_keys}) not found in {split_data}")

            rows.append(row)

        df = pd.DataFrame(rows)
        
        # Set Method as index for cleaner plotting/display
        if not df.empty:
            df.set_index('Method', inplace=True)
            
        return df

    def plot_gradient_table(self, split='test', cmap='RdYlGn', float_precision=3):
        """
        Generates a styled DataFrame with gradient coloring and specific alignment.
        """
        df = self.get_summary_table(split)
        
        if df.empty:
            print("No data available to plot.")
            return None

        # Drop columns where all values are NaN or None, print which columns are dropped
        columns_to_drop = []
        for column in df.columns:
            if df[column].isna().all() or df[column].isnull().all():
                columns_to_drop.append(column)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"Dropped columns due to all NaN or None values: {columns_to_drop}")

        # Define CSS styles
        # 1. th.row_heading: Targets the Index column (Method names) -> Left Align
        # 2. td: Targets the data cells -> Center Align
        # 3. th.col_heading: Targets the column headers -> Center Align
        styles = [
            dict(selector="th.row_heading", props=[("text-align", "left"), ("padding-left", "10px")]),
            dict(selector="td", props=[("text-align", "center")]),
            dict(selector="th.col_heading", props=[("text-align", "center")])
        ]

        return df.style.background_gradient(cmap=cmap, axis=0)\
                       .format(f"{{:.{float_precision}f}}")\
                       .set_caption(f"Results Summary ({split} set)")\
                       .set_table_styles(styles)

    def visualize_example(self, example_idx, method_aliases, split='test', 
                          figsize=(4, 4), max_title_width=30, show_heatmap=True,
                          max_cols=3):
        """
        Visualizes predictions with updated features:
        - Word wrapped titles.
        - Live IoU calculation vs Precomputed IoU.
        - Grid/Heatmap visualization of raw probabilities.
        - Grid layout control to prevent wide aspect ratios.
        """
        if not method_aliases:
            print("No method aliases provided.")
            return

        # 1. Determine the Target Key using the Reference Method (first alias)
        ref_alias = method_aliases[0]
        ref_data_full = self._get_preds_data(ref_alias)
        split_key = f"{split}_preds_and_gt"
        
        if split_key not in ref_data_full:
            print(f"Split {split_key} not found in {ref_alias}.")
            return
            
        ref_lookup = ref_data_full[split_key]
        sorted_keys = sorted(list(ref_lookup.keys()))

        try:
            target_key = sorted_keys[example_idx]
        except IndexError:
            print(f"Index {example_idx} out of bounds (Total samples: {len(sorted_keys)}).")
            return
        
        # 2. Setup Plot Grid
        n_methods = len(method_aliases)
        n_cols = min(n_methods, max_cols)
        n_rows = math.ceil(n_methods / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        # 3. Iterate over methods
        for i, (ax, alias) in enumerate(zip(axes_flat, method_aliases)):
            
            # --- Retrieve Method Specific Data ---
            method_data_full = self._get_preds_data(alias)
            
            if split_key not in method_data_full or target_key not in method_data_full[split_key]:
                print(f"Warning: Key {target_key} not found in {alias}. Skipping.")
                ax.set_title(f"{alias}\n(Sample not found)", fontsize=10, color='red')
                continue

            # Get predictions
            preds = method_data_full[split_key][target_key]

            # Get phrase
            phrase = preds['phrases'] if 'phrases' in preds else preds['phrase']
            
            # Get image path
            image_path = preds['image_paths'] if 'image_paths' in preds else preds['image_path']
            if not os.path.exists(image_path): # Fix image path issue in local environment
                candidate_fix_pairs = [
                    ('/mnt/workspace/physionet.org/', '/mnt/researchers/denis-parra/datasets/physionet.org/'),
                    ('/mnt/workspace/ChestX-Det-Dataset/', '/mnt/researchers/denis-parra/datasets/ChestX-Det-Dataset/'),
                    ('/mnt/workspace/vinbig-cxr/dataset-jpg/images_hq-512x512(keep_aspect_ratio)/', '/mnt/researchers/denis-parra/datasets/vindr/physionet.org/files/vindr-cxr/1.0.0/images_jpg_hq/test/'),
                    ('/mnt/workspace/BIMCV-Padchest-GR/PadChest_GR_JPG_600/', '/mnt/researchers/denis-parra/datasets/BIMCV-Padchest-GR/PadChest_GR_JPG_600/'),
                ]
                for candidate_fix_pair in candidate_fix_pairs:
                    if image_path.startswith(candidate_fix_pair[0]):
                        image_path = image_path.replace(candidate_fix_pair[0], candidate_fix_pair[1])
                        break
                if not os.path.exists(image_path):
                    print(f"Error: Image path {image_path} does not exist. Skipping.")
                    continue

            # Get GT bounding box coordinates
            gt_bbox_coords = preds['gt_bbox_coords']
            if isinstance(gt_bbox_coords, list):
                gt_bbox_coords = np.array(gt_bbox_coords)
            if gt_bbox_coords.ndim == 1:
                gt_bbox_coords = gt_bbox_coords.reshape(-1, 4)
            if alias in ['MAIRA-2', 'BioViL-T']:
                gt_bboxes_xyxy_normalized = gt_bbox_coords.copy() # Assume xyxy normalized coordinates by default
            else:
                gt_bboxes_xyxy_normalized = self._cxcywh_to_xyxy(gt_bbox_coords) # Convert to xyxy normalized coordinates

            # Get ground truth polygons
            gt_polygons = preds.get('gt_polygons') or preds.get('polygons')

            # Get ground truth mask
            gt_mask = preds.get('gt_mask')
            if gt_mask is None:
                if alias in ['MAIRA-2', 'BioViL-T']:
                    if gt_polygons is not None:
                        gt_mask = polygons_to_mask(gt_polygons, 130, 130)
                    else:
                        gt_mask = convert_bboxes_into_presence_map(gt_bboxes_xyxy_normalized, feature_map_size=(130, 130))
                else:
                    if gt_polygons is not None:
                        gt_mask = polygons_to_mask(gt_polygons, 100, 100)
                    else:
                        gt_mask = convert_bboxes_into_presence_map(gt_bboxes_xyxy_normalized, feature_map_size=(100, 100))
            assert gt_mask.ndim == 2, "Ground truth mask must be 2D"
            
            # Get predicted probability map
            raw_probs = preds['pred_bbox_prob_maps']
            if alias == 'MAIRA-2':
                assert raw_probs is None, "MAIRA-2 should not have a probability map"
                maira2_pred_bboxes_xyxy = preds['maira2_pred_bboxes']
                raw_probs = convert_bboxes_into_presence_map(maira2_pred_bboxes_xyxy, (130, 130))
                print("MAIRA-2's specific outputs:")
                print(f"\tpreds['maira_decoded_text'] = {preds['maira_decoded_text']}")
                print(f"\tpreds['maira_raw_output'] = {preds['maira_raw_output']}")
            
            # Retrieve Original Index to look up precomputed metrics
            original_idx = preds['_original_index']
            metrics = self.metrics_data[alias][f'{split}_metrics']

            if i == 0: # Print debug info for the first method
                print(f'Key: {target_key}')
                print(f'image_path = {image_path}')
                print(f'phrase = {phrase}')
                print(f'gt_bbox_coords = {gt_bbox_coords}')

            # --- Load Image ---
            try:
                pil_img = Image.open(image_path).convert('RGB')
                img_w, img_h = pil_img.size
            except Exception as e:
                print(f"Error loading image: {e}")
                continue
            
            # --- Display Image ---
            ax.imshow(pil_img)
            ax.axis('off')
            
            # --- Handle standard/BioVil-T Heatmaps ---
            if show_heatmap and raw_probs is not None:
                if isinstance(raw_probs, np.ndarray):
                    heatmap_data = raw_probs
                    grid_side = raw_probs.shape[0]
                    assert raw_probs.ndim == 2 and grid_side == raw_probs.shape[1], \
                        f"Raw probabilities must be 2D and have the same shape: {raw_probs.shape}"
                    heatmap_img = Image.fromarray((heatmap_data * 255).astype(np.uint8), mode='L')
                    heatmap_resized = heatmap_img.resize((img_w, img_h), Image.BILINEAR)
                    heatmap_resized = np.array(heatmap_resized) / 255.0
                    ax.imshow(heatmap_resized, cmap='jet', alpha=0.4)
                    cell_w = img_w / grid_side
                    cell_h = img_h / grid_side
                    for k in range(1, grid_side):
                        ax.axvline(k * cell_w, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
                        ax.axhline(k * cell_h, color="white", linestyle="--", linewidth=0.5, alpha=0.5)

            # --- Draw Ground Truth (Green) ---
            for box in gt_bboxes_xyxy_normalized:
                x1, y1, x2, y2 = box
                x1_px, y1_px = x1 * img_w, y1 * img_h
                x2_px, y2_px = x2 * img_w, y2 * img_h
                rect = patches.Rectangle((x1_px, y1_px), x2_px - x1_px, y2_px - y1_px, linewidth=2, 
                                edgecolor='lime', facecolor='none', linestyle='--')
                ax.add_patch(rect)
            # Draw polygons if they are provided
            if gt_polygons is not None:
                for polygon in gt_polygons:
                    # Method: Use patches.Polygon (Cleaner, handles closure automatically)
                    polygon_px = [[p[0] * img_w, p[1] * img_h] for p in polygon] # Convert to pixel coordinates
                    poly_patch = patches.Polygon(
                        polygon_px, 
                        linewidth=3, 
                        edgecolor='lime',
                        facecolor='none',
                    )
                    ax.add_patch(poly_patch)

            # --- Process Predictions ---
            
            pred_bboxes_xyxy_normalized = []

            # CASE 1: MAIRA-2 (Direct Boxes, No Heatmap/Coord Map)
            if 'maira2_pred_bboxes' in preds and preds['maira2_pred_bboxes'] is not None:
                maira_boxes_xyxy = preds['maira2_pred_bboxes']
                if len(maira_boxes_xyxy) > 0:
                    for i, box in enumerate(maira_boxes_xyxy):
                        x1, y1, x2, y2 = box
                        pred_bboxes_xyxy_normalized.append([x1, y1, x2, y2])
                        
                        x1_px, y1_px = x1 * img_w, y1 * img_h
                        w_px = (x2 - x1) * img_w
                        h_px = (y2 - y1) * img_h

                        rect = patches.Rectangle((x1_px, y1_px), w_px, h_px, linewidth=2, 
                                                edgecolor='yellow', facecolor='none')
                        ax.add_patch(rect)
                        # MAIRA-2 usually doesn't give a confidence score per box in the simple output, 
                        # or it's 1.0 if thresholded.
            
            # CASE 2: Internal Model (Coord Maps + Prob Maps)
            elif preds['pred_bbox_coord_maps'] is not None:
                raw_coords = preds['pred_bbox_coord_maps']
                final_boxes, final_probs = self._filter_predictions(raw_coords, raw_probs, metrics)
                
                if len(final_boxes) > 0:
                    for box, prob in zip(final_boxes, final_probs):
                        cx, cy, w, h = box
                        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                        pred_bboxes_xyxy_normalized.append([x1, y1, x2, y2])
                        x_px, y_px = x1 * img_w, y1 * img_h
                        w_px = (x2 - x1) * img_w
                        h_px = (y2 - y1) * img_h
                        rect = patches.Rectangle((x_px, y_px), w_px, h_px, linewidth=2, 
                                                edgecolor='yellow', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x_px, y_px - 5, f'{prob:.2f}', color='yellow', fontsize=8, weight='bold',
                                bbox=dict(facecolor='black', alpha=0.5, pad=1))
            
            else:
                pass
            
            # IoU Stats
            def get_precomputed_metric(metric_name):
                # Use original_idx (from the specific method's file order) 
                # instead of example_idx (which is just the loop counter for the reference)
                idx_to_use = original_idx 
                
                if 'sampled_indices' in metrics:
                    indices = metrics['sampled_indices']
                    try:
                        # Find where the original index is in the sampled list
                        metric_value = metrics[metric_name][indices.index(idx_to_use)]
                    except (KeyError, IndexError, ValueError):
                        metric_value = None
                else:
                    try:
                        metric_value = metrics[metric_name][idx_to_use]
                    except IndexError:
                        metric_value = None
                return f"{metric_value:.3f}" if metric_value is not None else "?"

            metric_tuples = []
            best_prob_conf_th = metrics.get('best_prob_conf_th', 0.5)
            
            # Bbox IoU
            if 'bbox_ious' in metrics:
                calc_bbox_iou = compute_bbox_union_iou(pred_bboxes_xyxy_normalized, gt_bboxes_xyxy_normalized)
                calc_bbox_iou_str = f"{calc_bbox_iou:.3f}"
                precomputed_bbox_iou_str = get_precomputed_metric('bbox_ious')
                metric_tuples.append(('Bbox IoU', calc_bbox_iou_str, precomputed_bbox_iou_str))

            # Probability Map IoU
            if 'prob_ious' in metrics:
                if gt_polygons is not None:
                    calc_prob_iou = calculate_segmentation_iou(gt_mask=gt_mask,
                                                               prob_map=raw_probs,
                                                               threshold=best_prob_conf_th)
                else:
                    calc_prob_iou = compute_probability_map_iou(gt_bboxes=gt_bboxes_xyxy_normalized,
                                                                prob_map=raw_probs,
                                                                conf_th=best_prob_conf_th)

                calc_prob_iou_str = f"{calc_prob_iou:.3f}"
                precomputed_prob_iou_str = get_precomputed_metric('prob_ious')
                metric_tuples.append(('Prob Map IoU', calc_prob_iou_str, precomputed_prob_iou_str))

            # CNR
            if 'cnrs' in metrics:
                calc_cnr = calculate_cnr(gt_mask, raw_probs)
                calc_cnr_str = f"{calc_cnr:.3f}"
                precomputed_cnr_str = get_precomputed_metric('cnrs')
                metric_tuples.append(('CNR', calc_cnr_str, precomputed_cnr_str))

            # Soft Dice
            if 'soft_dices' in metrics:
                calc_soft_dice = calculate_soft_dice(gt_mask, raw_probs)
                calc_soft_dice_str = f"{calc_soft_dice:.3f}"
                precomputed_soft_dice_str = get_precomputed_metric('soft_dices')
                metric_tuples.append(('Soft Dice', calc_soft_dice_str, precomputed_soft_dice_str))

            # Dice
            if 'dices' in metrics:
                calc_dice = calculate_dice(gt_mask, raw_probs, threshold=best_prob_conf_th)
                calc_dice_str = f"{calc_dice:.3f}"
                precomputed_dice_str = get_precomputed_metric('dices')
                metric_tuples.append(('Dice', calc_dice_str, precomputed_dice_str))

            # Precision
            if 'precisions' in metrics:
                calc_precision = calculate_segmentation_precision(gt_mask, raw_probs, threshold=best_prob_conf_th)
                calc_precision_str = f"{calc_precision:.3f}"
                precomputed_precision_str = get_precomputed_metric('precisions')
                metric_tuples.append(('Precision', calc_precision_str, precomputed_precision_str))

            # Recall
            if 'recalls' in metrics:
                calc_recall = calculate_segmentation_recall(gt_mask, raw_probs, threshold=best_prob_conf_th)
                calc_recall_str = f"{calc_recall:.3f}"
                precomputed_recall_str = get_precomputed_metric('recalls')
                metric_tuples.append(('Recall', calc_recall_str, precomputed_recall_str))

            # Rand Index
            if 'rand_indices' in metrics:
                calc_rand_index = calculate_rand_index(gt_mask, raw_probs, threshold=best_prob_conf_th)
                calc_rand_index_str = f"{calc_rand_index:.3f}"
                precomputed_rand_index_str = get_precomputed_metric('rand_indices')
                metric_tuples.append(('Rand Index', calc_rand_index_str, precomputed_rand_index_str))

            # Title
            wrapped_alias = "\n".join(textwrap.wrap(alias, width=max_title_width))

            # Metrics string
            metrics_str = []
            for metric_name, calc_metric_str, precomputed_metric_str in metric_tuples:
                metrics_str.append(f"\n - {metric_name}: {calc_metric_str} (precomputed: {precomputed_metric_str})")
            metrics_str = "".join(metrics_str)
            final_title = f"{wrapped_alias}{metrics_str}"
            ax.set_title(final_title, fontsize=10, color='black')

        # Hide unused subplots
        # If n_rows * n_cols > n_methods, we need to hide the extra axes
        total_plots = len(axes_flat)
        for j in range(n_methods, total_plots):
            axes_flat[j].axis('off')

        # Global Title
        wrapped_phrase = "\n".join(textwrap.wrap(f"Phrase: {phrase}", width=60))
        plt.suptitle(wrapped_phrase, fontsize=12, y=1.02) # Adjusted Y slightly
        plt.tight_layout()
        plt.show()