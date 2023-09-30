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
from medvqa.utils.logging import print_blue

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
        
        assert len(train_scores) == len(val_scores)
        
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

def plot_per_class_classification_metrics(dataframe_rows, method_aliases, metric_names, metric_aliases, dataset_name,
                                          figsize=(10, 8), scores_fontsize=7, ytick_fontsize=10,
                                          plot_vertical_size=1.0, sort_metrics=True, sort_methods=True):
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
    bar_height = 0.9 * plot_vertical_size / (n * m)
    for i in range(n):
        label = method_aliases[method_idxs[n-1-i]]
        label = f'({mean_score_per_method[method_idxs[n-1-i]]:.3f}) {label}'
        positions = [j * (plot_vertical_size / m) + (n-1-i)*bar_height for j in range(1, m+1)]
        scores = [scores_per_method[method_idxs[n-1-i]][metric_idxs[j]] for j in range(m)]
        plt.barh(positions, scores, height=bar_height, label=label, color=_COLORS[method_idxs[n-1-i] % len(_COLORS)])
        # plot the scores as text on top of the bars
        for j in range(m):
            plt.text(scores[j] + (max_score - min_score) * 0.01 , positions[j], f'{scores[j]:.3f}', ha='left', va='center', fontsize=scores_fontsize)
    plt.yticks([(j+0.5) * (plot_vertical_size / m) for j in range(1, m+1)], [metric_aliases[i] for i in metric_idxs], fontsize=ytick_fontsize)
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

def plot_metric_lists(metric_lists, method_names, title, metric_name, xlabel='Epoch', ylabel=None, figsize=(10, 10), first_k=None):
    assert type(metric_lists) == list
    assert len(metric_lists) > 0
    assert all(type(metric_list) == list for metric_list in metric_lists)
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