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
from medvqa.utils.metrics import average_ignoring_nones

# Consider 20 different colors
_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

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

        assert len(train_scores) == len(val_scores)
        
        epochs = list(range(1, len(train_scores)+1))        
        
        eps = 0.9
        
        ax = plt.subplot(nrows, ncols, j+1)
        ax.set_xlim(epochs[0]-eps, epochs[-1]+eps)
        ax.set_title(f'{metric_name} per epoch')
        ax.plot(epochs, train_scores, label=f'{metric_name} (Training)')
        ax.plot(epochs, val_scores, label=f'{metric_name} (Validation)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)    
        ax.legend()
        best_train_score, best_train_i = _agg_fn((a,i) for i,a in enumerate(train_scores))
        ax.hlines(best_train_score, epochs[0], epochs[-1], colors=('green',), linestyles='dashed',
                label=f'best train {metric_name}={best_train_score:.3f}, epoch={best_train_i}')
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
    mean_scores = [average_ignoring_nones(scores_per_method[i]) for i in range(n)]
    method_idxs.sort(key=lambda i: mean_scores[i], reverse=True)

    # Sort bbox classes by the mean score
    bbox_idxs = list(range(n_bboxes))
    mean_scores_per_bbox = [average_ignoring_nones(scores_per_method[i][j] for i in range(n)) for j in range(n_bboxes)]
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

def plot_per_class_classification_metrics(dataframe_rows, method_aliases, metric_names, metric_aliases, dataset_name,
                                          figsize=(10, 8), fontsize=7):
    n = len(dataframe_rows)
    assert n == len(method_aliases)
    assert n > 0
    m = len(metric_names)
    assert m == len(metric_aliases)
    assert m > 0

    scores_per_method = [[dataframe_rows[i][k] for k in metric_names] for i in range(n)]
    mean_score_per_metric = [np.mean([scores_per_method[i][j] for i in range(n)]) for j in range(m)]
    metric_idxs = list(range(m))
    metric_idxs.sort(key=lambda i: mean_score_per_metric[i])
    mean_score_per_method = [np.mean(scores_per_method[i]) for i in range(n)]
    method_idxs = list(range(n))
    method_idxs.sort(key=lambda i: mean_score_per_method[i])
    min_score = min(min(scores_per_method[i]) for i in range(n))
    max_score = max(max(scores_per_method[i]) for i in range(n))

    # Create a single plot with multiple horizontal bar charts, one bar for each method and metric    
    # The height of the bar is the metric score
    # Each method is a different color    
    plt.figure(figsize=figsize)
    height = 0.9 / n
    for i in range(n):
        label = method_aliases[method_idxs[n-1-i]]
        positions = [j + (n-1-i)*height for j in range(1, m+1)]
        scores = [scores_per_method[method_idxs[n-1-i]][metric_idxs[j]] for j in range(m)]
        plt.barh(positions, scores, height=height, label=label, color=_COLORS[method_idxs[n-1-i] % len(_COLORS)])
        # plot the scores as text on top of the bars
        for j in range(m):
            plt.text(scores[j] + (max_score - min_score) * 0.01 , positions[j], f'{scores[j]:.2f}', ha='left', va='center', fontsize=fontsize)
    plt.yticks([height * (n/2-0.5)+ i for i in range(1, m+1)], [metric_aliases[i] for i in metric_idxs])
    plt.ylabel('Metric')
    plt.xlabel('Score')
    plt.title(f'Metrics on {dataset_name}')
    # Plot legend outside the plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    # Plot horizontal lines of grid
    plt.grid(axis='x')
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
    mean_score_per_method = [average_ignoring_nones(scores_per_method[i]) for i in range(n)]
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