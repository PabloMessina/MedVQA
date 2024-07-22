import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from medvqa.datasets.chest_imagenome import (
    ANAXNET_BBOX_NAMES,
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_GOLD_BBOX_NAMES,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
)

from medvqa.utils.files import get_cached_pickle_file
from medvqa.utils.metrics import average_ignoring_nones_and_nans
from medvqa.utils.logging import print_blue, print_orange, print_bold, rgba_to_ansi

# List of 30 different colors
_COLORS = np.concatenate((plt.cm.tab20(np.linspace(0, 1, 15)), plt.cm.tab20b(np.linspace(0, 1, 15))), axis=0)

def _replace_nans_with_local_avgs(scores):
    scores_ = []
    for i, x in enumerate(scores):
        if np.isnan(x):
            # find first non-nan value to the left
            x_left = None
            for j in range(i-1, -1, -1):
                if not np.isnan(scores[j]):
                    x_left = scores[j]
                    break
            # find first non-nan value to the right
            x_right = None
            for j in range(i+1, len(scores)):
                if not np.isnan(scores[j]):
                    x_right = scores[j]
                    break
            # if both left and right are non-nan, take average
            if x_left is not None and x_right is not None:
                x = (x_left + x_right) / 2
            # if only left is non-nan, take left
            elif x_left is not None:
                x = x_left
            # if only right is non-nan, take right
            elif x_right is not None:
                x = x_right
            assert x is not None
        scores_.append(x)
    return scores_

def plot_train_val_curves(logs_path, metrics, metric_names, agg_fn=max, single_plot_figsize=(8, 6),
                          use_min_with_these_metrics=None, use_max_with_these_metrics=None):

    assert len(metrics) == len(metric_names)
    assert len(metrics) > 0
    n = len(metrics)    
    ncols = 2 if n > 1 else 1
    nrows = n // ncols + bool(n % ncols)

    # load csv without index column    
    logs = pd.read_csv(logs_path, index_col=False)

    figsize = (single_plot_figsize[0] * ncols, single_plot_figsize[1] * nrows)
    plt.figure(figsize=figsize)

    for j in range(n):

        metric = metrics[j]
        metric_name = metric_names[j]

        if use_min_with_these_metrics is not None and metric in use_min_with_these_metrics:
            _agg_fn = min
        elif use_max_with_these_metrics is not None and metric in use_max_with_these_metrics:
            _agg_fn = max
        else:
            _agg_fn = agg_fn

        metric_scores = logs[metric]
        train_scores = []
        val_scores = []
        for i in range(len(metric_scores)):
            if i % 2 == 0:
                train_scores.append(metric_scores[i])
            else:
                val_scores.append(metric_scores[i])
            
        train_has_only_nans = all(np.isnan(x) for x in train_scores)
        train_has_some_nans = any(np.isnan(x) for x in train_scores)
        val_has_only_nans = all(np.isnan(x) for x in val_scores)
        val_has_some_nans = any(np.isnan(x) for x in val_scores)

        if train_has_some_nans and not train_has_only_nans:
            train_scores = _replace_nans_with_local_avgs(train_scores)
            print(f'WARNING: {metric_name} train_scores has some nans, but not all. Replacing nans with nearest non-nan values')

        if val_has_some_nans and not val_has_only_nans:
            val_scores = _replace_nans_with_local_avgs(val_scores)
            print(f'WARNING: {metric_name} val_scores has some nans, but not all. Replacing nans with nearest non-nan values')
        
        if len(train_scores) != len(val_scores):
            print(f'WARNING: {metric_name} train_scores and val_scores have different lengths ({len(train_scores)} vs {len(val_scores)}). Truncating the longer one.')
            min_len = min(len(train_scores), len(val_scores))
            train_scores = train_scores[:min_len]
            val_scores = val_scores[:min_len]
        
        epochs = list(range(1, len(train_scores)+1))        
        
        eps = 0.9
        
        ax = plt.subplot(nrows, ncols, j+1)
        ax.set_xlim(epochs[0]-eps, epochs[-1]+eps)
        ax.set_title(f'{metric_name} per epoch')
        if not train_has_only_nans:
            ax.plot(epochs, train_scores, label=f'{metric_name} (Training)')
        if not val_has_only_nans:
            ax.plot(epochs, val_scores, label=f'{metric_name} (Validation)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.legend()
        if not train_has_only_nans:
            best_train_score, best_train_i = _agg_fn((a,i) for i,a in enumerate(train_scores))
            ax.hlines(best_train_score, epochs[0], epochs[-1], colors=('green',), linestyles='dashed',
                    label=f'best train {metric_name}={best_train_score:.3f}, epoch={best_train_i}')
        if not val_has_only_nans:
            best_val_score, best_val_i = _agg_fn((a,i) for i,a in enumerate(val_scores))
            ax.hlines(best_val_score, epochs[0], epochs[-1], colors=('red',), linestyles='dashed',
                    label=f'best val {metric_name}={best_val_score:.3f}, epoch={best_val_i}')
        ax.legend()
    
    plt.show()

def plot_chest_imagenome_bbox_metrics_at_thresholds(metrics_paths, method_aliases, metric_name, metric_alias,
                                                  dataset_name, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9], figsize=(8, 6)):
    assert type(metrics_paths) == list
    assert type(method_aliases) == list
    assert len(metrics_paths) == len(method_aliases)
    assert len(metrics_paths) > 0    
    metrics_list = [get_cached_pickle_file(path) for path in metrics_paths]
    n = len(metrics_list)
    # Get the scores for each method at each threshold
    scores_per_method = []
    for i in range(n):
        scores_per_method.append([])
        for t in thresholds:
            scores_per_method[i].append(metrics_list[i][f'{metric_name}@{t}'])
    # Sort methods by the mean score
    method_idxs = list(range(n))
    mean_scores = [sum(scores_per_method[i]) / len(scores_per_method[i]) for i in range(n)]
    method_idxs.sort(key=lambda i: mean_scores[i], reverse=True)
    # Create a vertical barchart plot, where each method has a bar for each threshold
    # The height of the bar is the score
    # Each method is a different color
    plt.figure(figsize=figsize)
    width = 0.8 / n
    for i in range(n):
        label = f'{method_aliases[method_idxs[i]]} ({mean_scores[method_idxs[i]]:.3f})'
        plt.bar([j + i*width for j in range(1, len(thresholds)+1)], scores_per_method[method_idxs[i]],
                 width=width, label=label, color=_COLORS[method_idxs[i] % len(_COLORS)])
    plt.xticks([width * (n/2-0.5)+ i for i in range(1, len(thresholds)+1)], thresholds)
    plt.xlabel('Threshold')
    plt.ylabel(metric_alias)
    plt.title(f'{metric_alias} per IoU threshold on {dataset_name}')
    # Plot legend outside the plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    # Plot horizontal lines of grid
    plt.grid(axis='y')
    plt.show()

def plot_chest_imagenome_bbox_metrics_per_bbox_class(metrics_paths, method_aliases, metric_name, metric_alias,
        dataset_name, figsize=(8,6), horizontal=True, bbox_class_names=None, make_anaxnet_ticks_red=False):
    assert type(metrics_paths) == list
    assert type(method_aliases) == list
    assert len(metrics_paths) == len(method_aliases)
    assert len(metrics_paths) > 0
    metrics_list = [get_cached_pickle_file(path) for path in metrics_paths]
    n = len(metrics_list)

    if bbox_class_names is None:
        # Collect all bounding box names from all methods
        all_bbox_names = set()
        for i in range(n):
            metrics_dict = metrics_list[i]
            if 'bbox_names' in metrics_dict:
                all_bbox_names.update(metrics_dict['bbox_names'])
            else:
                metrics_per_class = metrics_dict[metric_name]
                assert type(metrics_per_class) == list or type(metrics_per_class) == np.ndarray
                assert len(metrics_per_class) == CHEST_IMAGENOME_NUM_BBOX_CLASSES or\
                        len(metrics_per_class) == CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES
                if len(metrics_per_class) == CHEST_IMAGENOME_NUM_BBOX_CLASSES:
                    all_bbox_names.update(CHEST_IMAGENOME_BBOX_NAMES)
                elif len(metrics_per_class) == CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES:
                    all_bbox_names.update(CHEST_IMAGENOME_GOLD_BBOX_NAMES)
                else: assert False
        all_bbox_names = list(all_bbox_names)
        n_bboxes = len(all_bbox_names)
    else:
        all_bbox_names = bbox_class_names
        n_bboxes = len(all_bbox_names)
    
    # Collect the scores for each method and bounding box class
    scores_per_method = []
    for i in range(n):
        metrics_dict = metrics_list[i]
        metrics_per_class = metrics_dict[metric_name]
        if 'bbox_names' in metrics_dict:
            bbox_names = metrics_dict['bbox_names']
            assert len(metrics_per_class) == len(bbox_names)
        elif len(metrics_per_class) == CHEST_IMAGENOME_NUM_BBOX_CLASSES:
            bbox_names = CHEST_IMAGENOME_BBOX_NAMES
        elif len(metrics_per_class) == CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES:
            bbox_names = CHEST_IMAGENOME_GOLD_BBOX_NAMES
        else: assert False
        scores_per_method.append([None] * n_bboxes)
        for j in range(len(bbox_names)):
            try:
                idx = all_bbox_names.index(bbox_names[j])
                scores_per_method[i][idx] = metrics_per_class[j]
            except ValueError:
                pass

    # Sort methods by the mean score (ignoring None values)
    method_idxs = list(range(n))
    mean_scores = [average_ignoring_nones_and_nans(scores_per_method[i]) for i in range(n)]
    method_idxs.sort(key=lambda i: mean_scores[i], reverse=True)

    # Sort bbox classes by the mean score
    bbox_idxs = list(range(n_bboxes))
    mean_scores_per_bbox = [average_ignoring_nones_and_nans(scores_per_method[i][j] for i in range(n)) for j in range(n_bboxes)]
    bbox_idxs.sort(key=lambda i: mean_scores_per_bbox[i], reverse=horizontal)
    
    # Create a horizontal scatter plot, where each method has one point for each bounding box class
    # Each point is a pair of (score, bounding box class)
    # Each method is a different color
    plt.figure(figsize=figsize)
    
    for i in range(n):
        label = f'{method_aliases[method_idxs[i]]} ({mean_scores[method_idxs[i]]:.3f})'
        sorted_scores = [scores_per_method[method_idxs[i]][bbox_idxs[j]] for j in range(n_bboxes)]
        sorted_indices = [j + 1 for j in range(n_bboxes) if sorted_scores[j] is not None]
        sorted_scores = [x for x in sorted_scores if x is not None]
        assert len(sorted_scores) == len(sorted_indices)
        if horizontal:
            plt.scatter(sorted_indices, sorted_scores, label=label, color=_COLORS[method_idxs[i] % len(_COLORS)])
        else:
            plt.scatter(sorted_scores, sorted_indices, label=label, color=_COLORS[method_idxs[i] % len(_COLORS)])
    if horizontal:
        # Rotate the xticks by 45 degrees and move them to the right so they don't overlap
        plt.xticks(range(1, n_bboxes+1), [all_bbox_names[i] for i in bbox_idxs], rotation=45, ha='right')
        # Change xtick color to red if the bounding box class is in ANAXNET_BBOX_NAMES
        if make_anaxnet_ticks_red:
            for i in range(n_bboxes):
                if all_bbox_names[bbox_idxs[i]] in ANAXNET_BBOX_NAMES:
                    plt.gca().get_xticklabels()[i].set_color('red')
        plt.ylabel(metric_alias)
        plt.xlabel('Bounding box class')
        plt.grid(axis='y')
    else:
        plt.yticks(range(1, n_bboxes+1), [all_bbox_names[i] for i in bbox_idxs])
        # Change ytick color to red if the bounding box class is in ANAXNET_BBOX_NAMES
        if make_anaxnet_ticks_red:
            for i in range(n_bboxes):
                if all_bbox_names[bbox_idxs[i]] in ANAXNET_BBOX_NAMES:
                    plt.gca().get_yticklabels()[i].set_color('red')
        plt.xlabel(metric_alias)
        plt.ylabel('Bounding box class')
        plt.grid(axis='x')
    plt.title(f'{metric_alias} per bounding box class on {dataset_name}')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    
def plot_multilabel_classification_metrics(metrics_paths, method_aliases, metric_getters, metric_aliases, dataset_name,
                                            figsize=(10, 8), rotate_xticks=False):
    n = len(metrics_paths)
    m = len(metric_getters)
    assert n == len(method_aliases)
    assert len(metric_getters) == len(metric_aliases)
    assert len(metric_getters) > 0
    
    # Load the metrics
    metrics_list = [get_cached_pickle_file(metrics_path) for metrics_path in metrics_paths]
    mean_score_per_method = [np.mean([mg(metrics) for mg in metric_getters]) for metrics in metrics_list]
    method_idxs = list(range(n))
    method_idxs.sort(key=lambda i: mean_score_per_method[i], reverse=True)

    # Create a single plot with multiple vertical bar charts, one bar for each method and metric    
    # The height of the bar is the metric score
    # Each method is a different color
    plt.figure(figsize=figsize)
    width = 0.8 / n
    for i in range(n):
        label = method_aliases[method_idxs[i]]
        plt.bar([j + (n-1-i)*width for j in range(1, m+1)], [mg(metrics_list[method_idxs[i]]) for mg in metric_getters],
                 width=width, label=label, color=_COLORS[method_idxs[i] % len(_COLORS)])
    if rotate_xticks:
        # Rotate the xticks by 45 degrees and move them to the right so they don't overlap
        plt.xticks([width * (n/2-0.5) + i for i in range(1, m+1)], metric_aliases, rotation=45, ha='right')
    else:
        plt.xticks([width * (n/2-0.5) + i for i in range(1, m+1)], metric_aliases)
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title(f'Metrics on {dataset_name}')
    # Plot legend outside the plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    # Plot horizontal lines of grid
    plt.grid(axis='y')
    plt.show()

def plot_metric_bars_per_method(dataframe_rows, method_aliases, metric_names, metric_aliases, title,
                                figsize=(10, 8), scores_fontsize=7, metrics_tick_fontsize=10, metrics_axis_size=1.0,
                                sort_metrics=True, row_idx_to_sort_by=None, sort_methods=True, vertical=False):
    n = len(dataframe_rows)
    assert n == len(method_aliases)
    assert n > 0
    m = len(metric_names)
    assert m == len(metric_aliases)
    assert m > 0

    scores_per_method = [[dataframe_rows[i][k] for k in metric_names] for i in range(n)]
    mean_score_per_metric = [np.mean([scores_per_method[i][j] for i in range(n)]) for j in range(m)]
    metric_idxs = list(range(m))
    if sort_metrics:
        if row_idx_to_sort_by is not None: # sort by a specific row
            metric_idxs.sort(key=lambda i: scores_per_method[row_idx_to_sort_by][i])
        else: # sort by the mean score
            metric_idxs.sort(key=lambda i: mean_score_per_metric[i])
    mean_score_per_method = [np.mean(scores_per_method[i]) for i in range(n)]
    method_idxs = list(range(n))
    if sort_methods:
        method_idxs.sort(key=lambda i: mean_score_per_method[i])
    min_score = min(min(scores_per_method[i]) for i in range(n))
    max_score = max(max(scores_per_method[i]) for i in range(n))

    # Create a single plot with multiple horizontal bar charts, one bar for each method and metric    
    # The height of the bar is the metric score
    # Each method is a different color    
    plt.figure(figsize=figsize)
    if vertical:
        bar_width = 0.9 * metrics_axis_size / (n * m)
        for i in range(n):
            label = method_aliases[method_idxs[i]]
            label = f'({mean_score_per_method[method_idxs[i]]:.3f}) {label}'
            positions = [j * (metrics_axis_size / m) + i*bar_width for j in range(1, m+1)]
            scores = [scores_per_method[method_idxs[i]][metric_idxs[j]] for j in range(m)]
            plt.bar(positions, scores, width=bar_width, label=label, color=_COLORS[method_idxs[i] % len(_COLORS)])
            # plot the scores as text on top of the bars
            for j in range(m):
                plt.text(positions[j], scores[j] + (max_score - min_score) * 0.01 , f'{scores[j]:.3f}', ha='center',
                         va='bottom', fontsize=scores_fontsize, rotation=90)
        plt.xticks([j * (metrics_axis_size / m) + bar_width * 0.5 * (n - 1) for j in range(1, m+1)], [metric_aliases[i] for i in metric_idxs],
                   fontsize=metrics_tick_fontsize, rotation=90, ha='right')
        print([(j+0.5) * (metrics_axis_size / m) for j in range(0, m)])
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.grid(axis='y')
        # plot legend above the plot
        plt.legend(bbox_to_anchor=(0.5, 1.1), loc='lower center', borderaxespad=0.)
    else:
        bar_height = 0.9 * metrics_axis_size / (n * m)
        for i in range(n):
            label = method_aliases[method_idxs[n-1-i]]
            label = f'({mean_score_per_method[method_idxs[n-1-i]]:.3f}) {label}'
            positions = [j * (metrics_axis_size / m) + (n-1-i)*bar_height for j in range(1, m+1)]
            scores = [scores_per_method[method_idxs[n-1-i]][metric_idxs[j]] for j in range(m)]
            plt.barh(positions, scores, height=bar_height, label=label, color=_COLORS[method_idxs[n-1-i] % len(_COLORS)])
            # plot the scores as text on top of the bars
            for j in range(m):
                plt.text(scores[j] + (max_score - min_score) * 0.01 , positions[j], f'{scores[j]:.3f}', ha='left', va='center', fontsize=scores_fontsize)
        plt.yticks([j * (metrics_axis_size / m) + bar_height * 0.5 * (n - 1) for j in range(1, m+1)], [metric_aliases[i] for i in metric_idxs], fontsize=metrics_tick_fontsize)
        plt.ylabel('Metric')
        plt.xlabel('Score')
        plt.grid(axis='x')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.title(title)
    # Plot legend outside the plot
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.show()

def plot_class_frequency_vs_metric_scores_per_method(dataframe_rows, method_aliases, metric_names, label_frequencies,
                                                     title, ylabel, figsize=(10, 8)):
    n = len(dataframe_rows)
    assert n == len(method_aliases)
    assert n > 0
    m = len(metric_names)
    assert m == len(label_frequencies)
    assert m > 0

    scores_per_method = [[dataframe_rows[i][k] for k in metric_names] for i in range(n)]
    mean_score_per_method = [average_ignoring_nones_and_nans(scores_per_method[i]) for i in range(n)]
    method_idxs = list(range(n))
    method_idxs.sort(key=lambda i: mean_score_per_method[i], reverse=True)

    # Create a scatter plot per method, where each point is a pair (class frequency, metric score)
    plt.figure(figsize=figsize)
    for i in range(n-1, -1, -1):
        label = f'{method_aliases[method_idxs[i]]} ({mean_score_per_method[method_idxs[i]]:.3f})'
        plt.scatter(label_frequencies, scores_per_method[method_idxs[i]], label=label, color=_COLORS[method_idxs[i] % len(_COLORS)])
    plt.xlabel('Class frequency')
    plt.ylabel(ylabel)
    plt.title(title)
    # Plot legend outside the plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.show()

def plot_metrics(metric_names, metric_values, title, xlabel, ylabel, figsize=(10, 8), color='blue',
                 horizontal=False, sort_metrics=False, show_metrics_above_bars=False, eps=0.005, draw_grid=False,
                 append_average_to_title=False, xticks_rotation=0, yticks_rotation=0, text_rotation=0):
    n = len(metric_values)
    assert n == len(metric_values)
    assert n == len(metric_names)
    assert n > 0
    if sort_metrics:
        metric_idxs = list(range(n))
        metric_idxs.sort(key=lambda i: metric_values[i], reverse=not horizontal)
        metric_values = [metric_values[i] for i in metric_idxs]
        metric_names = [metric_names[i] for i in metric_idxs]
    plt.figure(figsize=figsize)
    if horizontal:
        plt.barh(range(1, n+1), metric_values, tick_label=metric_names, color=color)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if show_metrics_above_bars:
            for i in range(n):
                plt.text(metric_values[i] + eps, i+1, f'{metric_values[i]:.3f}', ha='left', va='center',
                         rotation=text_rotation)
        if draw_grid:
            plt.grid(axis='x')        
    else:
        plt.bar(range(1, n+1), metric_values, tick_label=metric_names, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if show_metrics_above_bars:
            for i in range(n):
                plt.text(i+1, metric_values[i] + eps, f'{metric_values[i]:.3f}', ha='center', va='bottom',
                         rotation=text_rotation)
        if draw_grid:
            plt.grid(axis='y')
    if append_average_to_title:
        # remove nan values from metric_values
        metric_values_ = []
        for i in range(n):
            if np.isnan(metric_values[i]):
                print_orange(f'WARNING: metric_values[{i}] ({metric_names[i]}) is nan. Skipping it.')
            else:
                metric_values_.append(metric_values[i])
        title = f'{title} (average={np.mean(metric_values_):.3f})'
    if xticks_rotation != 0:
        plt.xticks(rotation=xticks_rotation)
    if yticks_rotation != 0:
        plt.yticks(rotation=yticks_rotation)
    plt.title(title)
    plt.show()

def visualize_predicted_bounding_boxes__yolo(image_path, pred_coords, pred_classes, class_names, figsize, format='xywh'):
    from PIL import Image
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, figsize=figsize)

    # Image
    image = Image.open(image_path)
    image = image.convert('RGB')
    width = image.size[0]
    height = image.size[1]
    ax.imshow(image)

    # Predicted bounding boxes
    for i in range(len(pred_classes)):
        if format == 'xywh':
            x_mid = pred_coords[i, 0] * width
            y_mid = pred_coords[i, 1] * height
            w = pred_coords[i, 2] * width
            h = pred_coords[i, 3] * height
            x1, y1, x2, y2 = x_mid - w / 2, y_mid - h / 2, x_mid + w / 2, y_mid + h / 2
        elif format == 'xyxy':
            x1 = pred_coords[i, 0] * width
            y1 = pred_coords[i, 1] * height
            x2 = pred_coords[i, 2] * width
            y2 = pred_coords[i, 3] * height
        else:
            raise ValueError(f'Unknown format {format}')
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=plt.cm.tab20(pred_classes[i] % 20), facecolor='none', linestyle='dashed')
        ax.add_patch(rect)
        ax.text(x1, y1-3, class_names[pred_classes[i]], fontsize=10, bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.1))
    
    plt.show()

def visualize_attention_map(image_path, attention_map, figsize, title=None, attention_factor=1.0, bbox=None, draw_grid=False):
    from PIL import Image
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, figsize=figsize)

    # Image
    image = Image.open(image_path)
    image = image.convert('RGB')
    width = image.size[0]
    height = image.size[1]
    ax.imshow(image)

    # Attention map (heatmap): each pixel is a yellow rectangle with opacity proportional to the attention value
    for i in range(attention_map.shape[0]):
        for j in range(attention_map.shape[1]):
            x1 = j * width / attention_map.shape[1]
            y1 = i * height / attention_map.shape[0]
            x2 = (j+1) * width / attention_map.shape[1]
            y2 = (i+1) * height / attention_map.shape[0]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0, edgecolor='none', facecolor='yellow', alpha=attention_map[i, j] * attention_factor)
            ax.add_patch(rect)
    
    # Bounding box
    if bbox is not None:
        x1 = bbox[0] * width
        y1 = bbox[1] * height
        x2 = bbox[2] * width
        y2 = bbox[3] * height
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Grid
    if draw_grid:
        for i in range(attention_map.shape[0]):
            for j in range(attention_map.shape[1]):
                x1 = j * width / attention_map.shape[1]
                y1 = i * height / attention_map.shape[0]
                x2 = (j+1) * width / attention_map.shape[1]
                y2 = (i+1) * height / attention_map.shape[0]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)

    if title is not None:
        plt.title(title)

    plt.show()

def visualize_attention_maps(image_path, attention_maps, figsize, titles=None, max_cols=3, attention_factor=1.0):
    from PIL import Image
    import matplotlib.patches as patches
    import textwrap

    # Create a grid of subplots
    n_cols = min(len(attention_maps), max_cols)
    n_rows = math.ceil(len(attention_maps) / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    assert ax.shape == (n_rows, n_cols)
    
    # Define wrap length for titles based on number of columns and the figure size
    wrap_length = (figsize[0] / n_cols) * (60 / 7.5)

    # Image
    image = Image.open(image_path)
    image = image.convert('RGB')
    width = image.size[0]
    height = image.size[1]

    # Attention maps (heatmaps): each pixel is a yellow rectangle with opacity proportional to the attention value
    for k in range(len(attention_maps)):
        row = k // n_cols
        col = k % n_cols
        # show image in subplot
        ax[row, col].imshow(image) # show image in subplot
        for i in range(attention_maps[k].shape[0]):
            for j in range(attention_maps[k].shape[1]):
                x1 = j * width / attention_maps[k].shape[1]
                y1 = i * height / attention_maps[k].shape[0]
                x2 = (j+1) * width / attention_maps[k].shape[1]
                y2 = (i+1) * height / attention_maps[k].shape[0]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0, edgecolor='none', facecolor='yellow', alpha=attention_maps[k][i, j] * attention_factor)
                ax[row, col].add_patch(rect)

        if titles is not None:
            wrapped_title = "\n".join(textwrap.wrap(titles[k], wrap_length))
            ax[row, col].set_title(wrapped_title)

    plt.show()

def visualize_image_and_polygons(image_path, polygons_list, polygon_names, figsize, title=None, as_segmentation=False, mask_resolution=None):
    from PIL import Image, ImageDraw

    fig, ax = plt.subplots(1, figsize=figsize)

    # Image
    image = Image.open(image_path)
    image = image.convert('RGBA')
    image_width = image.size[0]
    image_height = image.size[1]
    
    if as_segmentation:
        # Draw polygons as segmentation masks
        if mask_resolution is None:
            overlays = []
            for i, polygons in enumerate(polygons_list):
                for polygon in polygons:
                    coords = [(point[0], point[1]) for point in polygon]
                    c = _COLORS[i % len(_COLORS)]
                    c_fill = (int(c[0]*255), int(c[1]*255), int(c[2]*255), 60) # 24% opacity
                    c_outline = (int(c[0]*255), int(c[1]*255), int(c[2]*255), 200) # 80% opacity
                    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))  # Transparent overlay
                    ImageDraw.Draw(overlay).polygon(coords, fill=c_fill, outline=c_outline)
                    overlays.append(overlay)
            # Combine the image with the overlays
            combined = image
            for overlay in overlays:
                combined = Image.alpha_composite(combined, overlay)
            ax.imshow(combined)
        else:
            mask_height, mask_width = mask_resolution
            masks = []
            for i, polygons in enumerate(polygons_list):
                for polygon in polygons:
                    coords = [(point[0] * mask_width / image_width, point[1] * mask_height / image_height) for point in polygon]
                    c = _COLORS[i % len(_COLORS)]
                    c_fill = (int(c[0]*255), int(c[1]*255), int(c[2]*255), 60) # 24% opacity
                    c_outline = (int(c[0]*255), int(c[1]*255), int(c[2]*255), 200) # 80% opacity
                    mask = Image.new('RGBA', (mask_width, mask_height))
                    ImageDraw.Draw(mask).polygon(coords, fill=c_fill, outline=c_outline)
                    masks.append(mask)
            # Combine the image with the masks
            combined = image
            for mask in masks:
                mask = mask.resize((image_width, image_height), Image.NEAREST)
                combined = Image.alpha_composite(combined, mask)
            ax.imshow(combined)
    else:
        ax.imshow(image)
        for i, polygons in enumerate(polygons_list):
            # Draw polygons
            for polygon in polygons:
                x = [p[0] for p in polygon]
                y = [p[1] for p in polygon]
                ax.plot(x, y, linewidth=3, color=_COLORS[i % len(_COLORS)])
    
    # Draw polygon names
    for polygons, name in zip(polygons_list, polygon_names):
        for polygon in polygons:
            # find upper left corner of polygon
            y, x = min((y, x) for x, y in polygon)
            ax.text(x, y, name, fontsize=10, bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.1))

    if title is not None:
        plt.title(title)

    plt.show()

def plot_segmentation_mask_area_vs_iou(masks_array, iou_array, figsize=(10, 8), title=None, xlabel=None, ylabel=None,
                                       normalize_area=False, sentences=None, num_annotations=10,
                                       highlight_points_with_sentences_containing=None):
    """
    Plots the IoU vs. area for a set of segmentation masks.

    :param masks_array: An array of shape (n_masks, height * width) containing the segmentation masks.
    :param iou_array: An array of shape (n_masks,) containing the IoU values.
    :param figsize: The size of the plot.
    :param title: The title of the plot.
    :param xlabel: The label of the x-axis.
    :param ylabel: The label of the y-axis.
    :param normalize_area: Whether to normalize the area by the number of pixels in the mask.
    :param sentences: A list of sentences corresponding to each mask.
    :param num_annotations: The number of points to annotate with sentences.
    :param highlight_points_with_sentences_containing: A string to highlight points with sentences containing this string.
    """
    masks_array =  np.array(masks_array)
    iou_array = np.array(iou_array)
    n_masks = len(masks_array)
    assert n_masks == len(iou_array)
    assert masks_array.ndim == 2
    if highlight_points_with_sentences_containing is not None:
        assert sentences is not None

    # Calculate the area of each mask
    areas = masks_array.sum(-1)
    if normalize_area:
        # Normalize the area by the number of pixels in the mask
        areas /= masks_array[0].size

    # Create the plot
    if title is None:
        title = 'IoU vs. Area'
    if xlabel is None:
        xlabel = 'Area' if not normalize_area else 'Area (normalized)'
    if ylabel is None:
        ylabel = 'IoU'
    plt.figure(figsize=figsize)
    if highlight_points_with_sentences_containing is not None:
        idxs_with = [i for i, sentence in enumerate(sentences) if highlight_points_with_sentences_containing in sentence]
        idxs_without = [i for i, sentence in enumerate(sentences) if highlight_points_with_sentences_containing not in sentence]
        plt.scatter(areas[idxs_without], iou_array[idxs_without], c='blue', alpha=0.5)
        plt.scatter(areas[idxs_with], iou_array[idxs_with], c='red', alpha=0.5)
    else:
        plt.scatter(areas, iou_array, c='blue', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Annotate some of the points with sentences
    if sentences is not None:
        # Sort by IoU
        sorted_indices = np.argsort(iou_array)
        # Split the indices into 20 consecutive groups
        groups = np.array_split(sorted_indices, 20)
        # Choose the same number of points randomly from each group
        num_annotations_per_group = math.ceil(num_annotations / 20)
        indices = []
        for group in groups:
            indices.extend(random.sample(list(group), min(num_annotations_per_group, len(group))))
        # Annotate the points
        tuples = []
        for idx in indices:
            plt.annotate(sentences[idx], (areas[idx], iou_array[idx]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=10)
            tuples.append((sentences[idx], iou_array[idx], areas[idx], idx))
    
    # Show plot
    plt.show()

    # Print annotated sentences
    if sentences is not None:
        tuples.sort(key=lambda x: x[1], reverse=True)
        for sentence, iou, area, idx in tuples:
            print_bold(f"(IoU={iou:.3f}, area={area:.3f}, idx={idx})", end=' ')
            print(sentence)

def plot_wordclouds_per_bin(sentences, scores, num_bins, figsize=(10, 5), num_words=30, width=800, height=400, background_color='white'):
    """
    Plots word clouds for sentences in each bin of scores.

    :param sentences: A list of sentences.
    :param scores: A list of scores.
    :param num_bins: The number of bins to divide the scores into.
    :param figsize: The size of the plot.
    :param num_words: The number of words to display in each word cloud.
    :param width: The width of the word cloud.
    :param height: The height of the word cloud.
    :param background_color: The background color of the word cloud.
    """
    assert len(sentences) == len(scores)
    assert num_bins > 0

    # Create bins
    bins = np.linspace(min(scores), max(scores), num_bins+1)
    bin_indices = np.digitize(scores, bins)

    # Create word clouds for each bin
    from wordcloud import WordCloud
    for i in range(1, num_bins+1):
        bin_sentences = [sentences[j] for j in range(len(sentences)) if bin_indices[j] == i]
        bin_scores = [scores[j] for j in range(len(scores)) if bin_indices[j] == i]
        if len(bin_sentences) > 0:
            print_bold(f'==================== Bin {i} ({bins[i-1]:.3f}-{bins[i]:.3f})')
            # plot_wordcloud(bin_sentences, bin_scores, title=f'{title} (bin {i})', score_name=score_name, figsize=figsize, num_words=num_words)
            text = ' '.join(bin_sentences)
            wordcloud = WordCloud(width=width, height=height, max_words=num_words, background_color=background_color).generate(text)
            # Display the word cloud
            plt.figure(figsize=figsize)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()

def plot_embeddings_and_clusters(X_dataset, X_clusters):
    # Apply PCA to reduce dimensionality to 2
    import sklearn.decomposition
    pca = sklearn.decomposition.PCA(n_components=2)
    X_dataset_2d = pca.fit_transform(X_dataset)
    X_clusters_2d = pca.transform(X_clusters)

    # Plot dataset and clusters
    plt.figure(figsize=(10, 10))
    plt.title('Dataset and clusters')
    plt.scatter(X_dataset_2d[:, 0], X_dataset_2d[:, 1], s=1, c='black', alpha=0.5)
    plt.scatter(X_clusters_2d[:, 0], X_clusters_2d[:, 1], s=10, c='red')
    plt.show()

def plot_embeddings(X):
    # Apply PCA to reduce dimensionality to 2
    import sklearn.decomposition
    pca = sklearn.decomposition.PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Plot embeddings
    plt.figure(figsize=(10, 10))
    plt.title('Dataset')
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=1, c='black', alpha=0.5)
    
    # Show plot
    plt.show()

_dim_reducers_cache = {}

def plot_embeddings_sentences_and_scores(embeddings, sentences, scores, title, score_name, num_annotations=10, figsize=(12, 10),
                                         sentence_max_length=50, sentence_fontsize=10, sample_more_from_low_scores=False,
                                         highlight_points_with_sentences_containing=None, use_tsne=False, use_umap=False):
    """
    Plots 2D embeddings of sentences colored by their scores and randomly annotates some of them with sentences.

    :param embeddings: An array-like of shape (n_samples, dim) containing the embeddings.
    :param sentences: A list of sentences corresponding to each embedding.
    :param scores: A list or array-like of scores for each embedding.
    :param title: Title of the plot.
    :param score_name: Name of the score to display in the colorbar.
    :param num_annotations: Number of sentences to annotate on the plot.
    :param figsize: Size of the plot.
    :param sentence_max_length: Maximum length of the sentence to display.
    :param sentence_fontsize: Font size of the sentences.
    :param sample_more_from_low_scores: Whether to sample more annotations from the lowest scores.
    :param highlight_points_with_sentences_containing: A string to highlight points with sentences containing this string.
    :param use_tsne: Whether to use t-SNE instead of PCA for dimensionality reduction.
    :param use_umap: Whether to use UMAP instead of PCA for dimensionality reduction.
    """
    assert len(embeddings) == len(sentences) == len(scores), "The number of embeddings, sentences, and scores must match."
    
    embeddings = np.array(embeddings)
    scores = np.array(scores)

    # Apply PCA to reduce dimensionality to 2
    if embeddings.shape[1] > 2:
        cache_key = f'2d+{embeddings.shape[1]}+{len(sentences)}+{sum(len(s) for s in sentences)}+{embeddings.sum():.3f}'
        if use_tsne:
            cache_key += '+tsne'
            xlabel = 't-SNE1'
            ylabel = 't-SNE2'
        elif use_umap:
            cache_key += '+umap'
            xlabel = 'UMAP1'
            ylabel = 'UMAP2'
        else:
            cache_key += '+pca'
            xlabel = 'PCA1'
            ylabel = 'PCA2'
        if cache_key in _dim_reducers_cache:
            embeddings = _dim_reducers_cache[cache_key]
        else:
            if use_tsne:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=0)
                embeddings = tsne.fit_transform(embeddings)
            elif use_umap:
                import umap
                reducer = umap.UMAP(n_components=2)
                embeddings = reducer.fit_transform(embeddings)
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                embeddings = pca.fit_transform(embeddings)
            _dim_reducers_cache[cache_key] = embeddings
        
    # Plot embeddings colored by scores
    plt.figure(figsize=figsize)
    if highlight_points_with_sentences_containing is not None:
        idxs_with = [i for i, sentence in enumerate(sentences) if highlight_points_with_sentences_containing in sentence]
        idxs_without = [i for i, sentence in enumerate(sentences) if highlight_points_with_sentences_containing not in sentence]
        scatter = plt.scatter(embeddings[idxs_without, 0], embeddings[idxs_without, 1], c=scores[idxs_without], cmap='coolwarm', edgecolor='k', alpha=0.7)
        # Use purple color, a thicker edge, and larger radius for highlighted points
        plt.scatter(embeddings[idxs_with, 0], embeddings[idxs_with, 1], c=scores[idxs_with], cmap='coolwarm', edgecolor='purple', linewidth=4, alpha=0.7, s=100)
    else:
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=scores, cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.colorbar(scatter, label=score_name)
    plt.title(f'{title} (mean {score_name}={np.mean(scores):.3f}, n={len(scores)})')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # Get the colors from the scatter plot
    colors = scatter.to_rgba(scores)
    
    # Randomly choose indices for annotations
    if sample_more_from_low_scores:
        sorted_indices = np.argsort(scores)
        # Attempt to sample half of the annotations from the lowest 20% of scores
        num_low_scores = len(scores) // 5
        num_low_score_annotations = min(num_annotations // 2, num_low_scores)
        num_high_score_annotations = min(num_annotations - num_low_score_annotations, len(scores) - num_low_scores)
        low_score_indices = sorted_indices[:num_low_scores]
        other_indices = sorted_indices[num_low_scores:]
        indices = random.sample(list(low_score_indices), num_low_score_annotations) +\
                    random.sample(list(other_indices), num_high_score_annotations)
    else:
        indices = random.sample(range(len(sentences)), min(num_annotations, len(sentences)))
    
    tuples = []
    for idx in indices:
        x, y = embeddings[idx]
        sentence = sentences[idx]
        if len(sentence) > sentence_max_length: # truncate long sentences
            sentence = sentence[:sentence_max_length] + '...'
        plt.annotate(sentence, (x, y), textcoords="offset points", xytext=(5, 5), ha='center',
                     fontsize=sentence_fontsize, color=colors[idx])
        
        # Convert RGBA to ANSI and print to console
        ansi_color = rgba_to_ansi(colors[idx])
        tuples.append((f"{ansi_color}{sentences[idx]}\033[0m", scores[idx], idx))
    
    # Show plot
    plt.show()

    # Print annotated sentences
    tuples.sort(key=lambda x: x[1], reverse=True) # sort by score
    for sentence, score, idx in tuples:
        print(f"{sentence} ({score:.3f}, idx={idx})")

def plot_metric_lists(metric_lists, method_names, title, metric_name, xlabel='Epoch', ylabel=None, figsize=(10, 10), first_k=None):
    assert type(metric_lists) == list
    assert len(metric_lists) > 0
    assert all(type(metric_list) == list or type(metric_list) == np.ndarray for metric_list in metric_lists)
    assert all(len(metric_list) > 0 for metric_list in metric_lists)
    assert all(len(metric_list) == len(metric_lists[0]) for metric_list in metric_lists)
    assert len(method_names) == len(metric_lists)

    if first_k is not None:
        metric_lists = [metric_list[:first_k] for metric_list in metric_lists]

    averages = [sum(metric_list) / len(metric_list) for metric_list in metric_lists]
    method_idxs = list(range(len(method_names)))
    method_idxs.sort(key=lambda i: averages[i], reverse=True)

    plt.figure(figsize=figsize)
    plt.title(title)
    m = len(metric_lists)
    n = len(metric_lists[0])
    for i in range(m):
        i = method_idxs[i]
        label = f'{method_names[i]} ({averages[i]:.3f})'
        plt.plot(range(1, n+1), metric_lists[i], label=label, color=_COLORS[i % len(_COLORS)])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel is not None else metric_name)
    plt.grid(axis='y')
    plt.legend()
    plt.show()

def plot_correlation_matrix(correlation_matrix, method_names, title, figsize=(10, 10)):
    assert type(correlation_matrix) == np.ndarray
    assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
    assert correlation_matrix.shape[0] == len(method_names)

    # plt.figure(figsize=figsize)
    # plt.title(title)
    # plt.imshow(correlation_matrix, vmin=vmin, vmax=vmax)
    # plt.xticks(range(len(method_names)), method_names, rotation=45, ha='right')
    # plt.yticks(range(len(method_names)), method_names)
    # plt.colorbar()
    # plt.show()

    # Create a DataFrame from the correlation matrix and names
    correlation_df = pd.DataFrame(data=correlation_matrix, columns=method_names, index=method_names)
    
    # Create a heatmap using seaborn
    plt.figure(figsize=figsize)
    import seaborn as sns
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.show()

class SentenceClusteringVisualizer:
    def __init__(self,
                 sentence_embeddings_filepath,
                 closest_cluster_centers_filepath,
                 kmedoids_refinement_filepath,
                 sample_size=None,
                 ):
        self.sentence_embeddings_filepath = sentence_embeddings_filepath
        self.closest_cluster_centers_filepath = closest_cluster_centers_filepath
        self.kmedoids_refinement_filepath = kmedoids_refinement_filepath
        self.sentence_embeddings = get_cached_pickle_file(sentence_embeddings_filepath)
        self.ccc = get_cached_pickle_file(closest_cluster_centers_filepath)
        self.kmedoids_refinement = get_cached_pickle_file(kmedoids_refinement_filepath)
        
        if sample_size is not None:
            print(f'Randomly selecting {sample_size} samples from dataset')
            cluster2indices = dict()
            for i, c in zip(self.ccc['sentence_idxs'], self.ccc['closest_cluster_centers']):
                if c not in cluster2indices:
                    cluster2indices[c] = []
                cluster2indices[c].append(i)
            idxs = []
            for c in cluster2indices:
                cluster_sample_size = math.ceil(sample_size * len(cluster2indices[c]) / len(self.ccc['sentence_idxs']))
                try:
                    idxs.extend(random.sample(cluster2indices[c], cluster_sample_size))
                except ValueError:
                    print(f'Error: cluster {c} has only {len(cluster2indices[c])} samples, but {cluster_sample_size} were requested')
                    raise
            idxs = set(idxs)
            idxs.update(self.kmedoids_refinement['refined_cluster_center_sentence_idxs'])
            idxs = list(idxs)
            idxs.sort()
            self.sample_idxs = idxs
        else:
            self.sample_idxs = None

        # Fit PCA to reduce dimensionality to 50
        print('Fitting PCA to reduce dimensionality to 50')
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=50)
        self.pca_embeddings = self.pca.fit_transform(self.sentence_embeddings['embeddings'])
        print('self.pca_embeddings.shape =', self.pca_embeddings.shape)

        # Fit UMAP to reduce dimensionality to 2
        print('Fitting UMAP to reduce dimensionality to 2') 
        if sample_size is None:
            embeddings = self.pca_embeddings[self.ccc['sentence_idxs']]
        else:
            embeddings = self.pca_embeddings[self.sample_idxs]
        print('embeddings.shape =', embeddings.shape)
        import umap
        self.umap = umap.UMAP(n_components=2, metric='cosine', n_neighbors=15, verbose=1)
        self.umap_embeddings = self.umap.fit_transform(embeddings)
        print('self.umap_embeddings.shape =', self.umap_embeddings.shape)

        # Apply PCA and UMAP to the cluster centers
        cluster_centers = self.kmedoids_refinement['refined_cluster_centers']
        print('cluster_centers.shape =', cluster_centers.shape)
        self.pca_cluster_centers = self.pca.transform(cluster_centers)
        print('self.pca_cluster_centers.shape =', self.pca_cluster_centers.shape)
        self.umap_cluster_centers = self.umap.transform(self.pca_cluster_centers)
        print('self.umap_cluster_centers.shape =', self.umap_cluster_centers.shape)

    def plot_embeddings(self, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        plt.title('UMAP Sentence embeddings')
        plt.scatter(self.umap_embeddings[:, 0], self.umap_embeddings[:, 1], s=1, c='black', alpha=0.5)
        plt.show()

    def plot_embeddings_and_clusters(self, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        plt.title('UMAP Sentence embeddings and clusters')
        plt.scatter(self.umap_embeddings[:, 0], self.umap_embeddings[:, 1], s=1, c='black', alpha=0.5)
        plt.scatter(self.umap_cluster_centers[:, 0], self.umap_cluster_centers[:, 1], s=2, c='red')
        plt.show()

    def plot_embeddings_and_clusters_around_point(self, x, y, r, figsize=(10, 10), plot_sentences=True,
                                                      other_sentences=None, other_umap_embeddings=None):
        assert (other_sentences is None) == (other_umap_embeddings is None)
        plt.figure(figsize=figsize)
        plt.title(f'UMAP Sentence embeddings around point (x={x:.3f}, y={y:.3f}, r={r:.3f})')
        plt.scatter(self.umap_embeddings[:, 0], self.umap_embeddings[:, 1], s=1, c='black', alpha=0.5)
        plt.scatter(self.umap_cluster_centers[:, 0], self.umap_cluster_centers[:, 1], s=2, c='red')
        if other_sentences is not None:
            assert len(other_sentences) == len(other_umap_embeddings)
            plt.scatter(other_umap_embeddings[:, 0], other_umap_embeddings[:, 1], s=1, c='blue')
        x1, x2, y1, y2 = x - r, x + r, y - r, y + r
        plt.xlim(x1, x2)
        plt.ylim(y1, y2)
        if plot_sentences:
            # Find sentences within rectangle
            sentences_in_rectangle = []
            for i in range(len(self.umap_embeddings)):
                if x1 <= self.umap_embeddings[i, 0] <= x2 and y1 <= self.umap_embeddings[i, 1] <= y2:
                    if self.sample_idxs is None:
                        s_idx = self.ccc['sentence_idxs'][i]
                    else:
                        s_idx = self.sample_idxs[i]
                    sentences_in_rectangle.append((s_idx, self.umap_embeddings[i, 0], self.umap_embeddings[i, 1]))
            # Choose a random subset of sentences to show
            sample = random.sample(sentences_in_rectangle, min(20, len(sentences_in_rectangle)))
            # Plot sentences
            for i, x, y in sample:
                plt.text(x, y, self.sentence_embeddings['sentences'][i], fontsize=10, bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.1))
            if other_sentences is not None:
                for i in range(len(other_sentences)):
                    plt.text(other_umap_embeddings[i, 0], other_umap_embeddings[i, 1], other_sentences[i], fontsize=10, fontweight='bold', color='blue',
                              bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.1))

        plt.show()

        if plot_sentences:
            # Print sentences
            if other_sentences is not None:
                for i in range(len(other_sentences)):
                    print_blue(other_sentences[i], bold=True)
            sentences = [self.sentence_embeddings['sentences'][i] for i, _, _ in sample]
            sentences.sort(key=lambda s: (len(s), s))
            for s in sentences:
                print(s)

    def plot_rectangle_around_sentence(self, sentence, r, figsize=(10, 10)):
        # Find sentence index
        sentence_idx = self.sentence_embeddings['sentences'].index(sentence)        
        # Obtain embedding
        sentence_embedding = self.sentence_embeddings['embeddings'][sentence_idx]
        pca_sentence_embedding = self.pca.transform([sentence_embedding])[0]
        umap_sentence_embedding = self.umap.transform([pca_sentence_embedding])[0]
        x = umap_sentence_embedding[0]
        y = umap_sentence_embedding[1]
        # Plot rectangle
        self.plot_embeddings_and_clusters_around_point(x, y, r, figsize=figsize, plot_sentences=True,
                                                       other_sentences=[sentence], other_umap_embeddings=umap_sentence_embedding.reshape((1, 2)))


def plot_nli_distribution(report_nli_input_output_jsonl_filepaths, figsize1=(10, 10), figsize2=(10, 10)):

    from collections import Counter
    from medvqa.datasets.seq2seq.seq2seq_dataset_management import load_report_nli_examples_filepaths
    from medvqa.utils.constants import LABEL_BASED_FACTS
    
    report_nli_input_texts, report_nli_output_texts, _ = load_report_nli_examples_filepaths(
        report_nli_input_output_jsonl_filepaths, nli1_only=True)
    
    print(Counter(report_nli_output_texts))

    # Plot metrics for each fact
    fact2stats = { fact: {'e': 0, 'n': 0, 'c': 0} for fact in LABEL_BASED_FACTS }
    other2stats = {'e': 0, 'n': 0, 'c': 0}
    for input_text, output_text in zip(report_nli_input_texts, report_nli_output_texts):
        fact = input_text.split(' #H: ')[1]
        if fact in LABEL_BASED_FACTS:
            fact2stats[fact][output_text[0]] += 1
        else:
            other2stats[output_text[0]] += 1

    # Plot:
    # For each fact, generate 3 horizontal bars: one for each label
    # Add some space between each fact
    # Add a title and a legend
    plt.figure(figsize=figsize1)
    plt.title('NLI distribution for each fact')
    yticks = []
    yticklabels = []
    facts = list(fact2stats.keys())
    facts.sort(key=lambda f: sum(fact2stats[f].values()))
    for i, fact in enumerate(facts):
        yticks.append(i * 4)
        yticklabels.append(f'{fact} (e={fact2stats[fact]["e"]}, n={fact2stats[fact]["n"]}, c={fact2stats[fact]["c"]})')
        for j, label in enumerate(['e', 'n', 'c']):
            if i == 0:
                long_label = {'e': 'entailment', 'n': 'neutral', 'c': 'contradiction'}[label]
                plt.barh(i * 4 + (2-j) - 1.0, fact2stats[fact][label], color=_COLORS[j], label=long_label)
            else:
                plt.barh(i * 4 + (2-j) - 1.0, fact2stats[fact][label], color=_COLORS[j])
    plt.yticks(yticks, yticklabels)
    plt.xlabel('Count')
    plt.ylabel('Fact')
    plt.legend()
    plt.show()

    # Plot:
    # Generate 3 horizontal bars: one for each label for the other facts
    plt.figure(figsize=figsize2)
    plt.title('NLI distribution for other facts')
    for j, label in enumerate(['e', 'n', 'c']):
        long_label = {'e': 'entailment', 'n': 'neutral', 'c': 'contradiction'}[label]
        plt.barh(2-j, other2stats[label], color=_COLORS[2-j], label=long_label)
    plt.yticks(range(3), [f'entailment (e={other2stats["e"]})', f'neutral (n={other2stats["n"]})', f'contradiction (c={other2stats["c"]})'])
    plt.xlabel('Count')
    plt.ylabel('Label')
    plt.legend()
    plt.show()


def plot_barchart(values, title, xlabel, ylabel, figsize=(10, 10), color='blue', horizontal=False, bar_names=None, sort_values=False,
                    write_values_on_bars=False, values_fontsize=10, values_color='black', values_rotation=0):
    n = len(values)
    if sort_values:
        indices = list(range(n))
        indices.sort(key=lambda i: values[i], reverse=not horizontal)
        values = [values[i] for i in indices]
        if bar_names is not None:
            bar_names = [bar_names[i] for i in indices]
    if bar_names is None:
        bar_names = range(1, n+1)
    plt.figure(figsize=figsize)
    if horizontal:
        plt.barh(range(1, n+1), values, color=color)
        plt.yticks(range(1, n+1), bar_names)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if write_values_on_bars:
            for i in range(n):
                plt.text(values[i], i+1, f'{values[i]:.1f}', ha='left', va='center', fontsize=values_fontsize, color=values_color, rotation=values_rotation)
    else:
        plt.bar(range(1, n+1), values, color=color)
        plt.xticks(range(1, n+1), bar_names)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if write_values_on_bars:
            for i in range(n):
                plt.text(i+1, values[i], f'{values[i]:.1f}', ha='center', va='bottom', fontsize=values_fontsize, color=values_color, rotation=values_rotation)
    plt.title(title)
    plt.show()

def plot_images(image_paths, titles=None, image_figsize=(5, 5), max_cols=3):
    import matplotlib.pyplot as plt
    import os
    import cv2

    # Create a grid of subplots
    n_cols = min(len(image_paths), max_cols)
    n_rows = math.ceil(len(image_paths) / n_cols)
    whole_figsize = (image_figsize[0] * n_cols, image_figsize[1] * n_rows)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=whole_figsize, squeeze=False)
    assert ax.shape == (n_rows, n_cols)

    # Load and show images
    for k, image_path in enumerate(image_paths):
        row = k // n_cols
        col = k % n_cols
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
        ax[row, col].imshow(image)
        ax[row, col].axis('off')
        if titles is not None:
            ax[row, col].set_title(titles[k])
        else:
            ax[row, col].set_title(os.path.basename(image_path))

    plt.show()