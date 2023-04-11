import os
import random
from sklearn.metrics import (
    auc, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)
import seaborn as sns
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ignite.engine import Events
from ignite.handlers.timing import Timer
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_BBOX_NAME_TO_SHORT,
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    get_train_val_test_stats_per_label,
    load_postprocessed_label_names,
    load_postprocessed_labels,
    load_scene_graph,
    visualize_ground_truth_bounding_boxes,
    visualize_predicted_bounding_boxes,
    visualize_scene_graph,
)
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import MIMICCXR_VisualModuleTrainer
from medvqa.metrics import (
    attach_chest_imagenome_labels_accuracy,
    attach_chest_imagenome_labels_prf1,
    attach_chest_imagenome_labels_roc_auc,
    attach_chexpert_labels_accuracy,
    attach_chexpert_labels_prf1,
    attach_chexpert_labels_roc_auc,
)
from medvqa.metrics.bbox.utils import compute_iou
from medvqa.models.ensemble.multilabel_ensemble_search import MultilabelOptimalEnsembleSearcher
from medvqa.training.vision import get_engine
from medvqa.utils.constants import MetricNames
from medvqa.utils.constants import (
    CHEXPERT_LABEL2SHORT,
    CHEXPERT_LABELS,
    METRIC2SHORT,
)
from medvqa.utils.files import get_cached_pickle_file, save_to_pickle
from medvqa.utils.handlers import (
    attach_accumulator,
    get_log_iteration_handler,
    get_log_metrics_handler,
)

_VISUAL_MODULE_METRIC_NAMES = [
    MetricNames.ORIENACC,
    MetricNames.CHXLABELACC,
    MetricNames.CHXLABEL_PRF1,
    MetricNames.CHXLABEL_ROCAUC,
    MetricNames.QLABELS_PRF1,
]

_empty_dict = {} # to avoid creating a new dict every time

def get_visual_module_metrics_dataframe(metrics_paths, metric_names=_VISUAL_MODULE_METRIC_NAMES):
    assert type(metrics_paths) == list or type(metrics_paths) == str
    if type(metrics_paths) is str:
        metrics_paths  = [metrics_paths]
    columns = ['metrics_path']
    data = [[] for _ in range(len(metrics_paths))]
    for row_i, metrics_path in tqdm(enumerate(metrics_paths)):
        data[row_i].append(metrics_path)
        metrics_dict = get_cached_pickle_file(metrics_path)
        
        for mn in metric_names:            
            met = metrics_dict.get(mn, _empty_dict)
            
            if mn == MetricNames.CHXLABEL_PRF1:
                data[row_i].append(met.get('f1_macro_avg', None))
                if row_i == 0: columns.append('f1(macro)')
                data[row_i].append(met.get('p_macro_avg', None))
                if row_i == 0: columns.append('p(macro)')
                data[row_i].append(met.get('r_macro_avg', None))
                if row_i == 0: columns.append('r(macro)')

                data[row_i].append(met.get('f1_micro_avg', None))
                if row_i == 0: columns.append('f1(micro)')
                data[row_i].append(met.get('p_micro_avg', None))
                if row_i == 0: columns.append('p(micro)')
                data[row_i].append(met.get('r_micro_avg', None))
                if row_i == 0: columns.append('r(micro)')

                _f1 = met.get('f1', None)
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(_f1[i] if _f1 else None)
                    if row_i == 0: columns.append(f'f1({CHEXPERT_LABEL2SHORT[label]})')
                _p = met.get('p', None)
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(_p[i] if _p else None)
                    if row_i == 0: columns.append(f'p({CHEXPERT_LABEL2SHORT[label]})')
                _r = met.get('r', None)
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(_r[i] if _r else None)
                    if row_i == 0: columns.append(f'r({CHEXPERT_LABEL2SHORT[label]})')
                
            elif mn == MetricNames.CHXLABEL_ROCAUC:                
                data[row_i].append(met.get('macro_avg', None))
                if row_i == 0: columns.append('rocauc(macro)')
                data[row_i].append(met.get('micro_avg', None))
                if row_i == 0: columns.append('rocauc(micro)')
                _pc = met.get('per_class', None)
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(_pc[i] if _pc else None)
                    if row_i == 0: columns.append(f'rocauc({CHEXPERT_LABEL2SHORT[label]})')            
            
            elif mn == MetricNames.QLABELS_PRF1:
                data[row_i].append(met.get('f1_macro_avg', None))
                if row_i == 0: columns.append('ql_f1(macro)')
                data[row_i].append(met.get('p_macro_avg', None))
                if row_i == 0: columns.append('ql_p(macro)')
                data[row_i].append(met.get('r_macro_avg', None))
                if row_i == 0: columns.append('ql_r(macro)')

                data[row_i].append(met.get('f1_micro_avg', None))
                if row_i == 0: columns.append('ql_f1(micro)')
                data[row_i].append(met.get('p_micro_avg', None))
                if row_i == 0: columns.append('ql_p(micro)')
                data[row_i].append(met.get('r_micro_avg', None))
                if row_i == 0: columns.append('ql_r(micro)')
                
            else:
                data[row_i].append(met if met is not _empty_dict else None)
                if row_i == 0: columns.append(METRIC2SHORT.get(mn, mn))
    
    return pd.DataFrame(data=data, columns=columns)

_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
_chest_imagenome_metric_names = []
# IoU and MAE (per class and macro)
for _bbox_name in CHEST_IMAGENOME_BBOX_NAMES:
    _bbox_name = CHEST_IMAGENOME_BBOX_NAME_TO_SHORT[_bbox_name]
    _chest_imagenome_metric_names.append(f'iou_{_bbox_name}')
    _chest_imagenome_metric_names.append(f'mae_{_bbox_name}')
_chest_imagenome_metric_names.append('mean_iou')
_chest_imagenome_metric_names.append('mean_mae')
# Precision, Recall, F1 (per class and macro, at different IoU thresholds)
for _t in _thresholds:
    for _bbox_name in CHEST_IMAGENOME_BBOX_NAMES:
        _bbox_name = CHEST_IMAGENOME_BBOX_NAME_TO_SHORT[_bbox_name]
        _chest_imagenome_metric_names.append(f'p@{_t}_{_bbox_name}')
        _chest_imagenome_metric_names.append(f'r@{_t}_{_bbox_name}')
        _chest_imagenome_metric_names.append(f'f1@{_t}_{_bbox_name}')
    _chest_imagenome_metric_names.append(f'mean_p@{_t}')
    _chest_imagenome_metric_names.append(f'mean_r@{_t}')
    _chest_imagenome_metric_names.append(f'mean_f1@{_t}')
_chest_imagenome_metric_names.append('mean_p')
_chest_imagenome_metric_names.append('mean_r')
_chest_imagenome_metric_names.append('mean_f1')

def get_chest_imagenome_bbox_metrics_dataframe(metrics_paths):
    assert type(metrics_paths) == list or type(metrics_paths) == str
    if type(metrics_paths) is str:
        metrics_paths  = [metrics_paths]    
    print(f'Loading {len(metrics_paths)} metrics files...')
    metrics_dict_list = [get_cached_pickle_file(metrics_path) for metrics_path in tqdm(metrics_paths)]    
    # Create dataframe
    columns = ['metrics_path']
    columns.extend(_chest_imagenome_metric_names)
    data = [[] for _ in range(len(metrics_paths))]
    for row_i, metrics_dict in tqdm(enumerate(metrics_dict_list)):
        row = [None] * len(columns)
        row[0] = metrics_paths[row_i]
        for key, value in metrics_dict.items():
            if key == 'bbox_names':
                assert type(value) == list
                continue
            if key.startswith('mean') and not key.startswith('mean_'):
                key = 'mean_' + key[4:] # e.g. meanp -> mean_p (for backward compatibility)
                idx = columns.index(key)
                assert type(value) in [float, np.float32, np.float64]
                row[idx] = value
                continue
            # if value is a numpy array or list, it means it's a per-class metric
            if type(value) in [list, np.ndarray]:
                assert len(value) == CHEST_IMAGENOME_NUM_BBOX_CLASSES or\
                    len(value) == CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES or\
                    'bbox_names' in metrics_dict
                if len(value) == CHEST_IMAGENOME_NUM_BBOX_CLASSES:
                    bbox_names = CHEST_IMAGENOME_BBOX_NAMES
                elif len(value) == CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES:
                    bbox_names = CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED
                else:
                    bbox_names = metrics_dict['bbox_names']
                for i, v in enumerate(value):
                    key_ = f'{key}_{CHEST_IMAGENOME_BBOX_NAME_TO_SHORT[bbox_names[i]]}'
                    idx = columns.index(key_)
                    row[idx] = v
            else: # otherwise it's a single value
                assert type(value) in [float, np.float32, np.float64], f'key={key}, value={value}, type(value)={type(value)}'
                if not key.startswith('mean_'):
                    key = 'mean_' + key # e.g. mae -> mean_mae (for backward compatibility)
                idx = columns.index(key)
                row[idx] = value
        data[row_i] = row
    return pd.DataFrame(data=data, columns=columns)

_CHEST_IMAGENOME_MULTILABEL_CLASSIFICATION_METRIC_NAMES = [
    MetricNames.CHESTIMAGENOMELABELACC,
    MetricNames.CHESTIMAGENOMELABEL_PRF1,
    MetricNames.CHESTIMAGENOMELABELROCAUC,
    MetricNames.CHESTIMAGENOMELABELAUC,
    MetricNames.CHESTIMAGENOMELABELPRCAUC,
]

def get_chest_imagenome_multilabel_classification_metrics_dataframe(
        metrics_paths, metric_names=_CHEST_IMAGENOME_MULTILABEL_CLASSIFICATION_METRIC_NAMES):

    assert type(metrics_paths) == list or type(metrics_paths) == str
    if type(metrics_paths) is str:
        metrics_paths  = [metrics_paths]
    columns = ['metrics_path', 'num_labels']
    data = [[] for _ in range(len(metrics_paths))]
    metrics_dict_list = [get_cached_pickle_file(metrics_path) for metrics_path in tqdm(metrics_paths)]
    label_names_list = []
    all_label_names = set()
    for metrics_dict in metrics_dict_list:
        if 'chest_imagenome_label_names' in metrics_dict:
            label_names = metrics_dict['chest_imagenome_label_names']
            # print(f'len(label_names)={len(label_names)}')
        else:
            # TODO: remove this hack
            if len(metrics_dict[MetricNames.CHESTIMAGENOMELABEL_PRF1]['p']) == 627:
                label_names = load_postprocessed_label_names('labels(min_freq=100).pkl')
            else:
                assert False
        label_names_list.append(label_names)
        all_label_names.update(label_names)
    all_label_names = sorted(list(all_label_names))
    label_name_2_idx = {label_name: idx for idx, label_name in enumerate(all_label_names)}

    def _label2str(label_name):
        if len(label_name) == 2:
            label_name = label_name[1]
        else:
            assert len(label_name) == 3, f'len(label_name)={len(label_name)}'
            label_name = label_name[0] + ' ' + label_name[2]
        return label_name

    offset = 2
    metric2offset = {}
    for metric_name in metric_names:
        if metric_name == MetricNames.CHESTIMAGENOMELABEL_PRF1:
            metric2offset[metric_name] = offset
            columns.append('f1(macro)')
            columns.append('p(macro)')
            columns.append('r(macro)')
            columns.append('f1(micro)')
            columns.append('p(micro)')
            columns.append('r(micro)')
            for label_name in all_label_names:
                label_name = _label2str(label_name)
                columns.append(f'f1({label_name})')
                columns.append(f'p({label_name})')
                columns.append(f'r({label_name})')
            offset += 6 + 3 * len(all_label_names)
        elif metric_name == MetricNames.CHESTIMAGENOMELABELACC:
            metric2offset[metric_name] = offset
            columns.append('acc')
            offset += 1
        elif metric_name == MetricNames.CHESTIMAGENOMELABELROCAUC:
            metric2offset[metric_name] = offset
            columns.append('rocauc(macro)')
            columns.append('rocauc(micro)')
            for label_name in all_label_names:
                label_name = _label2str(label_name)
                columns.append(f'rocauc({label_name})')
            offset += 2 + len(all_label_names)
        elif metric_name == MetricNames.CHESTIMAGENOMELABELAUC:
            metric2offset[metric_name] = offset
            columns.append('auc(macro)')
            columns.append('auc(micro)')
            for label_name in all_label_names:
                label_name = _label2str(label_name)
                columns.append(f'auc({label_name})')
            offset += 2 + len(all_label_names)
        elif metric_name == MetricNames.CHESTIMAGENOMELABELPRCAUC:
            metric2offset[metric_name] = offset
            columns.append('prcauc(macro)')
            columns.append('prcauc(micro)')
            for label_name in all_label_names:
                label_name = _label2str(label_name)
                columns.append(f'prcauc({label_name})')
            offset += 2 + len(all_label_names)
        else:
            assert False, f'unknown metric_name={metric_name}'

    for row_i, metrics_path in enumerate(tqdm(metrics_paths)):
        data[row_i] = [None] * len(columns)
        data[row_i][0] = metrics_path
        data[row_i][1] = len(label_names_list[row_i])
        metrics_dict = metrics_dict_list[row_i]
        
        for mn in metric_names:
            met = metrics_dict.get(mn, None)

            if mn == MetricNames.CHESTIMAGENOMELABEL_PRF1:
                offset = metric2offset[mn]
                data[row_i][offset + 0] = met['f1_macro_avg'] if met is not None else None
                data[row_i][offset + 1] = met['p_macro_avg'] if met is not None else None
                data[row_i][offset + 2] = met['r_macro_avg'] if met is not None else None
                data[row_i][offset + 3] = met['f1_micro_avg'] if met is not None else None
                data[row_i][offset + 4] = met['p_micro_avg'] if met is not None else None
                data[row_i][offset + 5] = met['r_micro_avg'] if met is not None else None
                for i, label_name in enumerate(label_names_list[row_i]):
                    label_idx = label_name_2_idx[label_name]
                    data[row_i][offset + 6 + 3 * label_idx + 0] = met['f1'][i] if met is not None else None
                    data[row_i][offset + 6 + 3 * label_idx + 1] = met['p'][i] if met is not None else None
                    data[row_i][offset + 6 + 3 * label_idx + 2] = met['r'][i] if met is not None else None

            elif mn == MetricNames.CHESTIMAGENOMELABELACC:
                offset = metric2offset[mn]
                data[row_i][offset + 0] = met if met is not None else None

            elif mn == MetricNames.CHESTIMAGENOMELABELROCAUC or\
                    mn == MetricNames.CHESTIMAGENOMELABELAUC or\
                    mn == MetricNames.CHESTIMAGENOMELABELPRCAUC:
                offset = metric2offset[mn]
                data[row_i][offset + 0] = met['macro_avg'] if met is not None else None
                data[row_i][offset + 1] = met['micro_avg'] if met is not None else None
                for i, label_name in enumerate(label_names_list[row_i]):
                    label_idx = label_name_2_idx[label_name]
                    data[row_i][offset + 2 + label_idx] = met['per_class'][i] if met is not None else None
            
    return pd.DataFrame(data=data, columns=columns)

class ChestImagenomeBboxPredictionsVisualizer:

    def __init__(self, predictions_path=None, data=None):
        if predictions_path is not None:
            data = get_cached_pickle_file(predictions_path)
        elif data is None:
            raise ValueError('Either predictions_path or data must be provided')
        self.dicom_ids = data['dicom_ids']
        self.pred_bbox_coords = data['pred_bbox_coords']
        self.pred_bbox_presences = data['pred_bbox_presences']
        self.test_bbox_coords = data['test_bbox_coords']
        self.test_bbox_presences = data['test_bbox_presences']
        n_bbox_classes = len(self.test_bbox_presences[0])
        if n_bbox_classes == CHEST_IMAGENOME_NUM_BBOX_CLASSES:
            self.bbox_names = CHEST_IMAGENOME_BBOX_NAMES
        elif n_bbox_classes == CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES:
            self.bbox_names = CHEST_IMAGENOME_GOLD_BBOX_NAMES__SORTED
        else:
            raise ValueError(f'Unknown number of bbox classes: {n_bbox_classes}')
        iou_scores = [None] * len(self.dicom_ids)
        scores = np.empty((n_bbox_classes,), dtype=float)
        for i in tqdm(range(len(self.dicom_ids))):
            pred_coords = self.pred_bbox_coords[i]
            pred_presences = self.pred_bbox_presences[i]
            test_coords = self.test_bbox_coords[i]
            test_presences = self.test_bbox_presences[i]
            for j in range(n_bbox_classes):
                if test_presences[j] == 1:
                    if pred_presences[j] > 0:
                        scores[j] = compute_iou(pred_coords[j*4:(j+1)*4], test_coords[j*4:(j+1)*4])
                    else:
                        scores[j] = 0
                else:
                    scores[j] = 1 if pred_presences[j] <= 0 else 0
            iou_scores[i] = scores.mean()
        self.iou_scores = np.array(iou_scores)
        self.iou_scores_mean = np.mean(self.iou_scores)
        self.iou_scores_std = np.std(self.iou_scores)
        self.iou_ranked_indices = np.argsort(self.iou_scores)
        if predictions_path is not None:
            print(f'Loaded and processed {len(self.dicom_ids)} predictions from {predictions_path}')
        else:
            print(f'Processed {len(self.dicom_ids)} predictions')
        print(f'IoU mean: {self.iou_scores_mean:.4f}, std: {self.iou_scores_std:.4f}')

    def plot_iou_scores(self, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.hist(self.iou_scores, bins=100)
        plt.title(f'IoU scores (mean: {self.iou_scores_mean:.4f}, std: {self.iou_scores_std:.4f})')
        plt.xlabel('IoU score')
        plt.ylabel('Count')
        plt.show()

    def print_iou_scores_statistics(self):
        print(f'IoU mean: {self.iou_scores_mean:.4f}')
        print(f'IoU std: {self.iou_scores_std:.4f}')
        print(f'IoU min: {np.min(self.iou_scores):.4f}')
        print(f'IoU max: {np.max(self.iou_scores):.4f}')
        # Number of predictions with IoU between consecutive thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        acc_count = 0
        for i in range(len(thresholds)):
            if i == 0:
                count = np.sum(self.iou_scores <= thresholds[i])
                acc_count += count
                print(f'IoU <= {thresholds[i]:.1f}: {count}, cumulative: {acc_count}')
            else:
                count = np.sum((self.iou_scores > thresholds[i-1]) & (self.iou_scores <= thresholds[i]))
                acc_count += count
                print(f'{thresholds[i-1]:.1f} < IoU <= {thresholds[i]:.1f}: {count}, cumulative: {acc_count}')            

    def visualize_bbox_predictions_example(self, i=None):
        if i is None:
            i = np.random.randint(len(self.dicom_ids))
        i = self.iou_ranked_indices[i]
        dicom_id = self.dicom_ids[i]
        pred_bbox_coords = self.pred_bbox_coords[i]
        pred_bbox_presences = self.pred_bbox_presences[i]
        test_bbox_coords = self.test_bbox_coords[i]
        test_bbox_presences = self.test_bbox_presences[i]
        iou_score = self.iou_scores[i]
        print(f'IoU score: {iou_score:.4f}')
        print(f'DICOM ID: {dicom_id}')
        print(f'Predicted bounding box coordinates: {pred_bbox_coords}')
        print(f'Predicted bounding box presences: {pred_bbox_presences}')
        print(f'Test bounding box coordinates: {test_bbox_coords}')
        print(f'Test bounding box presences: {test_bbox_presences}')
        visualize_ground_truth_bounding_boxes(dicom_id)
        visualize_predicted_bounding_boxes(dicom_id, pred_bbox_coords, pred_bbox_presences,
                                            test_bbox_coords, test_bbox_presences,
                                            bbox_names=self.bbox_names)

class ChestImaGenomeMLCVisualizer:

    def __init__(self, mlc_metrics_path, dicom_id_to_probs_path, 
                 test_label_names_filename='labels(min_freq=1000).pkl',
                 test_labels_filename='imageId2labels(min_freq=1000).pkl',
                 use_gold_in_test=False):

        self.metrics = get_cached_pickle_file(mlc_metrics_path)
        self.dicom_id_to_probs = get_cached_pickle_file(dicom_id_to_probs_path)
        self.dicom_ids = list(self.dicom_id_to_probs.keys())
        self.label_names = self.metrics['chest_imagenome_label_names']
        self.stats_per_label = get_train_val_test_stats_per_label('labels(min_freq=1000).pkl',
                                                                   'imageId2labels(min_freq=1000).pkl', 
                                                                   use_gold_in_test=use_gold_in_test)
        self.gt_labels = load_postprocessed_labels(test_labels_filename)
        self.gt_label_names = load_postprocessed_label_names(test_label_names_filename)

    def print_auc_per_class(self):
        set_completer_delims = self.metrics[MetricNames.CHESTIMAGENOMELABELAUC]['per_class']
        idxs = np.argsort(set_completer_delims)
        for i in idxs:
            print(f'{i}: {self.label_names[i]}: {set_completer_delims[i]:.4f}')
    
    def print_f1_per_class(self):
        scores = self.metrics[MetricNames.CHESTIMAGENOMELABEL_PRF1]['f1']
        idxs = np.argsort(scores)
        for i in idxs:
            print(f'{i}: {self.label_names[i]}: {scores[i]:.4f}')

    def plot_label_frequency_vs_metric(self, metric):
        if metric in ['p', 'r', 'f1']:
            scores = self.metrics[MetricNames.CHESTIMAGENOMELABEL_PRF1][metric]
        elif metric == 'auc':
            scores = self.metrics[MetricNames.CHESTIMAGENOMELABELAUC]['per_class']
        elif metric == 'roc_auc':
            scores = self.metrics[MetricNames.CHESTIMAGENOMELABELROCAUC]['per_class']
        elif metric == 'prc_auc':
            scores = self.metrics[MetricNames.CHESTIMAGENOMELABELPRCAUC]['per_class']
        else:
            raise ValueError(f'Unknown metric: {metric}')
        label_freqs = [self.stats_per_label[k]['train_fraction'] for k in self.label_names]
        plt.figure(figsize=(10, 10))
        plt.scatter(label_freqs, scores)
        plt.xlabel('Label frequency')
        plt.ylabel(metric)
        plt.show()

    def print_labels_in_range(self, metric, min_val=0, max_val=1, min_freq=0, max_freq=1):
        if metric in ['p', 'r', 'f1']:
            scores = self.metrics[MetricNames.CHESTIMAGENOMELABEL_PRF1][metric]
        elif metric == 'auc':
            scores = self.metrics[MetricNames.CHESTIMAGENOMELABELAUC]['per_class']
        elif metric == 'roc_auc':
            scores = self.metrics[MetricNames.CHESTIMAGENOMELABELROCAUC]['per_class']
        elif metric == 'prc_auc':
            scores = self.metrics[MetricNames.CHESTIMAGENOMELABELPRCAUC]['per_class']
        else:
            raise ValueError(f'Unknown metric: {metric}')
        label_freqs = [self.stats_per_label[k]['train_fraction'] for k in self.label_names]
        # sort indices by score, ignoring None values
        idxs = [i for i in range(len(scores)) if scores[i] is not None]
        idxs = sorted(idxs, key=lambda i: scores[i])
        for i in idxs:
            if min_val <= scores[i] <= max_val and min_freq <= label_freqs[i] <= max_freq:
                print(f'score: {scores[i]:.4f}, freq: {label_freqs[i]:.4f}, label: {self.label_names[i]}')

    def compute_metrics_from_probs(self, label_name):
        label_idx = self.label_names.index(label_name)
        gt_label_idx = self.gt_label_names.index(label_name)
        probs = []
        gt = []
        for dicom_id in self.dicom_ids:
            probs.append(self.dicom_id_to_probs[dicom_id][label_idx])
            gt.append(self.gt_labels[dicom_id][gt_label_idx])
        probs = np.array(probs)
        gt = np.array(gt)
        
        # plot precision recall curve
        precision, recall, _ = precision_recall_curve(gt, probs)
        no_skill = len(gt[gt==1]) / len(gt)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='Logistic')
        plt.title(f'Precision-Recall Curve for {label_name}')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
        
        # plot confusion matrix
        cm = confusion_matrix(gt, probs > 0.5)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Confusion matrix for {label_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        # p, r, f1
        p = precision_score(gt, probs > 0.5)
        r = recall_score(gt, probs > 0.5)
        f1 = f1_score(gt, probs > 0.5)
        # roc-auc
        roc_auc = roc_auc_score(gt, probs)
        # pr auc
        pr_auc = auc(recall, precision)
        print(f'p: {p:.4f}, r: {r:.4f}, f1: {f1:.4f}, roc_auc: {roc_auc:.4f}, pr_auc: {pr_auc:.4f}')

    def _plot_example_for_label(self, label_name, pos=True):
        label_idx = self.label_names.index(label_name)
        gt_label_idx = self.gt_label_names.index(label_name)
        idxs = []
        for i, dicom_id in enumerate(self.dicom_ids):
             if (self.gt_labels[dicom_id][gt_label_idx] == 1) == pos:
                idxs.append(i)
        idx = random.choice(idxs)
        dicom_id = self.dicom_ids[idx]
        label_prob = self.dicom_id_to_probs[dicom_id][label_idx]
        if pos:
            print('Positive example:')
        else:
            print('Negative example:')
        print(f'dicom_id: {dicom_id}')
        print(f'Label: {label_name}, prob: {label_prob:.4f}')
        print('-' * 50)
        visualize_scene_graph(load_scene_graph(dicom_id))

    def plot_positive_example_for_label(self, label_name):
        self._plot_example_for_label(label_name, pos=True)

    def plot_negative_example_for_label(self, label_name):
        self._plot_example_for_label(label_name, pos=False)
    
def calibrate_thresholds_on_mimiccxr_validation_set(
    model, device, use_amp, mimiccxr_vision_evaluator_kwargs,
    classify_chexpert, classify_chest_imagenome, save_probs=False, results_folder_path=None):
    
    assert classify_chexpert != classify_chest_imagenome # Only one of them can be True
    
    if classify_chexpert:
        labeler_name = 'chexpert'
    elif classify_chest_imagenome:
        labeler_name = 'chest_imagenome'
    else: assert False, 'This should not happen'

    # Run model on MIMICCXR validation dataset to get predictions
    assert mimiccxr_vision_evaluator_kwargs['use_val_set_only']
    mimiccxr_vision_evaluator = MIMICCXR_VisualModuleTrainer(**mimiccxr_vision_evaluator_kwargs)
    evaluator = get_engine(model=model, classify_tags=False, classify_orientation=False, classify_questions=False,
                            classify_gender=False, predict_bboxes_chest_imagenome=False, 
                            classify_chexpert=classify_chexpert, classify_chest_imagenome=classify_chest_imagenome,
                            pass_pred_bbox_coords_as_input=mimiccxr_vision_evaluator_kwargs.get('pass_pred_bbox_coords_to_model', False),
                            device=device, use_amp=use_amp, training=False,
                            using_yolov8=mimiccxr_vision_evaluator_kwargs.get('use_yolov8', False))
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
    log_metrics_handler = get_log_metrics_handler(timer, metrics_to_print=metrics_to_print)
    log_iteration_handler = get_log_iteration_handler()    
    evaluator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
    print('Running model on MIMICCXR validation dataset ...')
    evaluator.run(mimiccxr_vision_evaluator.val_dataloader)
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
    mloes.sample_weights(n_tries=1)
    thresholds = mloes.compute_best_merged_probs_and_thresholds()['thresholds']
    print('thresholds.shape:', thresholds.shape)
    if classify_chexpert: # Only print thresholds for CheXpert
        print('thresholds:', thresholds)
    print('Done!')
    # Save probabilities
    if save_probs:
        assert results_folder_path is not None
        print('Saving probabilities on MIMICCXR validation set ...')
        dicom_id_to_pred_probs = {}
        pred_probs = pred_probs[0] # remove extra dimension
        assert len(mimiccxr_vision_evaluator.val_indices) == len(pred_probs)
        for i, idx in enumerate(mimiccxr_vision_evaluator.val_indices):
            dicom_id = mimiccxr_vision_evaluator.dicom_ids[idx]
            dicom_id_to_pred_probs[dicom_id] = pred_probs[i]
        save_path = os.path.join(results_folder_path, f'dicom_id_to_pred_probs__mimiccxr_val__{labeler_name}.pkl')
        save_to_pickle(dicom_id_to_pred_probs, save_path)
        print('Probabilities saved to:', save_path)
    return thresholds

def _prepare_data_for_ensemble(dicom_id_2_probs_list, label_names_list, dicom_id_2_gt_labels, gt_label_names):
    # Determine the intersection of dicom_ids
    dicom_ids = None
    for dicom_id_2_probs in dicom_id_2_probs_list:
        print(f'   num_dicom_ids: {len(dicom_id_2_probs)}')
        if dicom_ids is None:
            dicom_ids = set(dicom_id_2_probs.keys())
        else:
            dicom_ids = dicom_ids.intersection(set(dicom_id_2_probs.keys()))
    dicom_ids = list(dicom_ids)
    dicom_ids.sort()
    print('Number of dicom_ids after intersection:', len(dicom_ids))
    # Determine the intersection of label_names
    label_names = set(gt_label_names)
    for x in label_names_list:
        print(f'   num_labels: {len(x)}')
        label_names = label_names.intersection(set(x))
    label_names = list(label_names)
    label_names.sort()
    print('Number of label_names after intersection:', len(label_names))
    # Create numpy arrays
    n_methods = len(dicom_id_2_probs_list)
    probs_matrix = np.zeros((n_methods, len(dicom_ids), len(label_names)))
    gt = np.zeros((len(dicom_ids), len(label_names)))
    for i, dicom_id_2_probs in enumerate(dicom_id_2_probs_list):
        label_idxs = [label_names_list[i].index(x) for x in label_names]
        for j, dicom_id in enumerate(dicom_ids):
            probs = dicom_id_2_probs[dicom_id]
            probs = probs[label_idxs]
            probs_matrix[i, j, :] = probs
    gt_label_idxs = [gt_label_names.index(x) for x in label_names]
    for i, dicom_id in enumerate(dicom_ids):
        gt_labels = dicom_id_2_gt_labels[dicom_id]
        gt_labels = gt_labels[gt_label_idxs]
        gt[i, :] = gt_labels
    # Return
    print(f'probs_matrix.shape: {probs_matrix.shape}')
    print(f'gt.shape: {gt.shape}')
    return probs_matrix, gt, label_names
    
def calibrate_weights_and_thresholds_for_ensemble(
        dicom_id_2_probs_list, label_names_list, dicom_id_2_gt_labels, gt_label_names, max_num_runs=5):
    # Prepare data
    probs_matrix, gt, _ = _prepare_data_for_ensemble(
        dicom_id_2_probs_list, label_names_list, dicom_id_2_gt_labels, gt_label_names)
    # Search optimal weights and thresholds
    print('Searching optimal thresholds ...')
    print(f'probs_matrix.shape: {probs_matrix.shape}')
    print(f'gt.shape: {gt.shape}')
    mloes = MultilabelOptimalEnsembleSearcher(probs=probs_matrix, gt=gt)
    mloes.try_basic_weight_heuristics()
    score = mloes.evaluate_best_predictions()
    mloes.sample_weights(n_tries=50)
    score = mloes.evaluate_best_predictions()
    for _ in range(max_num_runs):
        mloes.sample_weights_from_previous_ones(n_tries=50, noise_coef=0.1)
        new_score = mloes.evaluate_best_predictions()
        if abs(new_score - score) < 0.001:
            break
        score = new_score
    output = mloes.compute_best_merged_probs_and_thresholds()
    weights = output['weights']
    thresholds = output['thresholds']
    print(f'weights.shape: {weights.shape}')
    print(f'thresholds.shape: {thresholds.shape}')
    return weights, thresholds

def merge_probabilities(dicom_id_2_probs_list, label_names_list, dicom_id_2_gt_labels, gt_label_names, weights, thresholds):
    # Prepare data
    probs_matrix, gt, label_names = _prepare_data_for_ensemble(
        dicom_id_2_probs_list, label_names_list, dicom_id_2_gt_labels, gt_label_names)
    # Merge probabilities
    print('Merging probabilities ...')
    merged_probs = np.zeros_like(probs_matrix[0])
    for i in range(probs_matrix.shape[1]):
        for j in range(probs_matrix.shape[2]):
            merged_probs[i, j] = np.sum(weights[j] * probs_matrix[:, i, j])
    pred_labels = (merged_probs > thresholds).astype(np.int)
    # Return
    return merged_probs, pred_labels, gt, label_names

