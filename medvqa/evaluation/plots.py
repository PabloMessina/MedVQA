import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Tuple
from PIL.Image import Image as PILImage
from medvqa.datasets.chest_imagenome import (
    ANAXNET_BBOX_NAMES,
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_GOLD_BBOX_NAMES,
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES,
)
from medvqa.utils.files_utils import get_cached_pickle_file
from medvqa.utils.metrics_utils import average_ignoring_nones_and_nans
from medvqa.utils.logging_utils import print_blue, print_orange, print_bold, rgba_to_ansi

# List of 40 different colors
_COLORS = []
for k in range(2):
    for i in range(10):
        _COLORS.append(plt.cm.tab20.colors[i * 2 + k])
for k in range(4):
    for i in range(5):
        _COLORS.append(plt.cm.tab20b.colors[i * 4 + k])

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


def plot_train_val_curves(
    logs_path: str,
    metrics: Optional[List[str]] = None,
    metric_names: Optional[List[str]] = None,
    agg_fn: Callable = max,
    single_plot_figsize: Tuple[int, int] = (8, 6),
    use_min_with_these_metrics: Optional[List[str]] = None,
    use_max_with_these_metrics: Optional[List[str]] = None,
    train_color: Optional[str] = None,
    val_color: Optional[str] = None,
):
    """
    Plots training and validation curves for specified metrics from a CSV log
    file. Training and validation lines will have consistent, distinct colors
    across all subplots.

    The CSV file is expected to have metrics logged in an alternating fashion
    for training and validation (e.g., train_metric_epoch1,
    val_metric_epoch1, train_metric_epoch2, val_metric_epoch2, ...).

    Args:
        logs_path (str): Path to the CSV file containing the training logs.
        metrics (list of str, optional): A list of column names from the CSV
            file to be plotted. If None, all columns in the CSV will be
            treated as metrics and plotted. Defaults to None.
        metric_names (list of str, optional): A list of display names for the
            metrics specified in the `metrics` argument. Must be of the same
            length as `metrics`. If None (and `metrics` is None), the original
            column names are used. If `metrics` is provided and `metric_names`
            is None, `metrics` will be used as display names. Defaults to None.
        agg_fn (function, optional): The aggregation function (e.g., `min`,
            `max`) used to determine the "best" score for a metric. This
            function is applied to an iterable of (score, index) tuples.
            Defaults to `max`.
        single_plot_figsize (tuple, optional): A tuple `(width, height)`
            specifying the size of each individual subplot.
            Defaults to (8, 6).
        use_min_with_these_metrics (list of str, optional): A list of metric
            column names for which the `min` function should be used as the
            aggregation function, overriding `agg_fn`. If None, it defaults
            to metrics whose names contain 'loss'. Defaults to None.
        use_max_with_these_metrics (list of str, optional): A list of metric
            column names for which the `max` function should be used as the
            aggregation function, overriding `agg_fn`. Defaults to None.
        train_color (str, optional): Color for the training curves. If None,
            defaults to the first color in Matplotlib's default cycle.
        val_color (str, optional): Color for the validation curves. If None,
            defaults to the second color in Matplotlib's default cycle.
    """
    # Load CSV log file, assuming no index column is stored in the CSV
    logs = pd.read_csv(logs_path, index_col=False)

    if metrics is None:
        assert metric_names is None, (
            "metric_names should be None if metrics is None"
        )
        metrics = logs.columns.tolist()
        metric_names = metrics
    elif metric_names is None:
        metric_names = metrics

    assert len(metrics) == len(metric_names), (
        "metrics and metric_names must have the same length"
    )
    assert len(metrics) > 0, "At least one metric must be specified"

    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = n // ncols + bool(n % ncols)

    figsize = (single_plot_figsize[0] * ncols, single_plot_figsize[1] * nrows)
    plt.figure(figsize=figsize)

    # Define colors for train and validation
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]

    _train_color = train_color if train_color else default_colors[0]
    _val_color = (
        val_color if val_color else default_colors[1 % len(default_colors)]
    ) # Ensure val_color is different if only 1 default color

    if use_min_with_these_metrics is None:
        use_min_with_these_metrics = [
            x for x in metrics if "loss" in x.lower()
        ]

    for j in range(n):
        metric = metrics[j]
        metric_name = metric_names[j]

        _agg_fn = agg_fn
        if (
            use_min_with_these_metrics is not None
            and metric in use_min_with_these_metrics
        ):
            _agg_fn = min
        elif (
            use_max_with_these_metrics is not None
            and metric in use_max_with_these_metrics
        ):
            _agg_fn = max

        metric_scores_all = logs[metric]
        train_scores = []
        val_scores = []

        for i_score in range(len(metric_scores_all)):
            if i_score % 2 == 0:
                train_scores.append(metric_scores_all[i_score])
            else:
                val_scores.append(metric_scores_all[i_score])

        train_has_only_nans = all(np.isnan(x) for x in train_scores)
        train_has_some_nans = any(np.isnan(x) for x in train_scores)
        val_has_only_nans = all(np.isnan(x) for x in val_scores)
        val_has_some_nans = any(np.isnan(x) for x in val_scores)

        if train_has_some_nans and not train_has_only_nans:
            train_scores = _replace_nans_with_local_avgs(train_scores)
            print_orange(
                f"WARNING: {metric_name} train_scores had NaNs. "
                "Attempted replacement.",
                bold=True,
            )
        if val_has_some_nans and not val_has_only_nans:
            val_scores = _replace_nans_with_local_avgs(val_scores)
            print_orange(
                f"WARNING: {metric_name} val_scores had NaNs. "
                "Attempted replacement.",
                bold=True,
            )

        if len(train_scores) != len(val_scores):
            print_orange(
                f"WARNING: {metric_name} train_scores and val_scores have "
                f"different lengths ({len(train_scores)} vs "
                f"{len(val_scores)}). Truncating the longer one.",
                bold=True,
            )
            min_len = min(len(train_scores), len(val_scores))
            train_scores = train_scores[:min_len]
            val_scores = val_scores[:min_len]

        num_epochs = len(train_scores)
        epochs = list(range(1, num_epochs + 1))

        ax = plt.subplot(nrows, ncols, j + 1)
        if epochs:
            eps = 0.9
            ax.set_xlim(epochs[0] - eps, epochs[-1] + eps)
        ax.set_title(f"{metric_name} per epoch")

        if not train_has_only_nans and train_scores:
            ax.plot(
                epochs,
                train_scores,
                label=f"{metric_name} (Training)",
                color=_train_color,
                linestyle="-",
            )
            valid_train_scores = [
                (s, idx)
                for idx, s in enumerate(train_scores)
                if not np.isnan(s)
            ]
            if valid_train_scores:
                best_train_score, best_train_i = _agg_fn(valid_train_scores)
                ax.hlines(
                    best_train_score,
                    epochs[0],
                    epochs[-1],
                    colors=_train_color, # Use train_color
                    linestyles="dashed",
                    label=(
                        f"Best Train {metric_name}={best_train_score:.3f}, "
                        f"Epoch={best_train_i + 1}"
                    ),
                )

        if not val_has_only_nans and val_scores:
            ax.plot(
                epochs,
                val_scores,
                label=f"{metric_name} (Validation)",
                color=_val_color,
                linestyle="--",
            )
            valid_val_scores = [
                (s, idx)
                for idx, s in enumerate(val_scores)
                if not np.isnan(s)
            ]
            if valid_val_scores:
                best_val_score, best_val_i = _agg_fn(valid_val_scores)
                ax.hlines(
                    best_val_score,
                    epochs[0],
                    epochs[-1],
                    colors=_val_color, # Use val_color
                    linestyles="dotted",
                    label=(
                        f"Best Val {metric_name}={best_val_score:.3f}, "
                        f"Epoch={best_val_i + 1}"
                    ),
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_multiple_experiment_curves(
    logs_paths: List[str],
    experiment_names: List[str],
    target_metrics: Optional[List[str]] = None,
    metric_display_names: Optional[List[str]] = None,
    agg_fn: Callable = max,
    single_plot_figsize: Tuple[float, float] = (6.5, 5.0),
    use_min_with_these_metrics: Optional[List[str]] = None,
    use_max_with_these_metrics: Optional[List[str]] = None,
    sort_experiments_by_best_score: bool = False,
    metric_name_to_sort_by: Optional[str] = None,
    draw_point_on_best_score: bool = False,
    point_markersize: int = 5,
):
    """
    Plots training and validation curves for multiple experiments on shared
    subplots, one subplot per metric. Each experiment maintains a consistent
    color across all subplots. Experiments can optionally be sorted by their
    best validation score on a specified metric.

    Each CSV log file is expected to have metrics logged in an alternating
    fashion for training and validation (e.g., train_metric_epoch1,
    val_metric_epoch1, ...).

    Args:
        logs_paths (List[str]): Paths to CSV log files.
        experiment_names (List[str]): Names for the experiments.
        target_metrics (Optional[List[str]]): Metrics to plot. If None,
            all unique metrics are plotted.
        metric_display_names (Optional[List[str]]): Display names for
            target_metrics.
        agg_fn (Callable): Aggregation function (e.g., `min`, `max`) for
            finding the best score on plots. Applied to (score, index) tuples.
            Defaults to `max`.
        single_plot_figsize (Tuple[float, float]): Size `(width, height)` for
            each subplot. Defaults to (6.5, 5.0).
        use_min_with_these_metrics (Optional[List[str]]): Metrics for which
            `min` is better (e.g., losses).
        use_max_with_these_metrics (Optional[List[str]]): Metrics for which
            `max` is better (overrides `use_min` if conflicting).
        sort_experiments_by_best_score (bool): If True, experiments are
            sorted based on their best validation score for the
            `metric_name_to_sort_by`. Defaults to False.
        metric_name_to_sort_by (Optional[str]): The metric column name to use
            for sorting if `sort_experiments_by_best_score` is True.
            Performance is judged on validation scores.
        draw_point_on_best_score (bool): If True, a point is drawn on the
            best score for each experiment. Defaults to False.
        point_markersize (int): Size of the point drawn on the best score.
            Defaults to 5.
    """
    if not logs_paths:
        print("No log paths provided. Nothing to plot.")
        return
    assert len(logs_paths) == len(experiment_names), (
        "logs_paths and experiment_names must have the same length."
    )

    # Load experiment data first, without colors
    experiment_data_temp = []
    all_metric_columns = set()

    for path, name in zip(logs_paths, experiment_names):
        try:
            logs_df = pd.read_csv(path, index_col=False)
            if logs_df.empty:
                print(
                    f"WARNING: Log file for experiment '{name}' at {path} "
                    "is empty. Skipping."
                )
                continue
            experiment_data_temp.append(
                {"name": name, "logs": logs_df, "path": path}
            )
            all_metric_columns.update(logs_df.columns)
        except FileNotFoundError:
            print(
                f"WARNING: Log file not found for experiment '{name}' at "
                f"{path}. Skipping."
            )
        except pd.errors.EmptyDataError:
            print(
                f"WARNING: Log file for experiment '{name}' at {path} "
                "is empty. Skipping."
            )
        except Exception as e:
            print(
                f"WARNING: Could not load log file for experiment '{name}' "
                f"at {path} due to: {e}. Skipping."
            )

    if not experiment_data_temp:
        print("No valid log data could be loaded. Nothing to plot.")
        return

    # Sort experiments if requested
    if sort_experiments_by_best_score:
        if not metric_name_to_sort_by:
            print(
                "WARNING: sort_experiments_by_best_score is True, but "
                "metric_name_to_sort_by is not specified. Skipping sorting."
            )
        elif metric_name_to_sort_by not in all_metric_columns:
            print(
                f"WARNING: metric_name_to_sort_by '{metric_name_to_sort_by}' "
                "not found in any loaded experiment logs. Skipping sorting."
            )
        else:
            # Determine aggregation logic for the sorting metric
            sort_metric_is_minimized = False
            if use_min_with_these_metrics and (
                metric_name_to_sort_by in use_min_with_these_metrics
            ):
                sort_metric_is_minimized = True
            elif "loss" in metric_name_to_sort_by.lower() and not (
                use_max_with_these_metrics
                and metric_name_to_sort_by in use_max_with_these_metrics
            ):
                # Default for 'loss' unless explicitly in use_max
                sort_metric_is_minimized = True

            if use_max_with_these_metrics and (
                metric_name_to_sort_by in use_max_with_these_metrics
            ): # Explicit max overrides
                sort_metric_is_minimized = False

            default_worst_score = (
                np.inf if sort_metric_is_minimized else -np.inf
            )

            for exp in experiment_data_temp:
                exp["sort_score"] = default_worst_score  # Initialize
                if metric_name_to_sort_by in exp["logs"].columns:
                    metric_values = exp["logs"][metric_name_to_sort_by]
                    val_scores_for_sort = [
                        metric_values[i]
                        for i in range(len(metric_values))
                        if i % 2 != 0 # Odd indices for validation
                    ]
                    # Handle NaNs for sorting scores
                    val_scores_for_sort = _replace_nans_with_local_avgs(
                        val_scores_for_sort
                    )
                    valid_scores = [
                        s for s in val_scores_for_sort if not np.isnan(s)
                    ]

                    if valid_scores:
                        if sort_metric_is_minimized:
                            exp["sort_score"] = min(valid_scores)
                        else:
                            exp["sort_score"] = max(valid_scores)
            
            # Sort: if minimizing, ascending (reverse=False). If maximizing, descending (reverse=True).
            experiment_data_temp.sort(
                key=lambda x: x.get("sort_score", default_worst_score),
                reverse=not sort_metric_is_minimized,
            )
            print(
                f"INFO: Experiments sorted by best validation "
                f"'{metric_name_to_sort_by}'. "
                f"Order: {[e['name'] for e in experiment_data_temp]}"
            )

    # Assign colors after potential sorting
    experiment_data = []
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]
    for i, exp_dict_temp in enumerate(experiment_data_temp):
        exp_dict_temp["color"] = default_colors[i % len(default_colors)]
        experiment_data.append(exp_dict_temp)


    if target_metrics is None:
        metrics_to_plot = sorted(list(all_metric_columns))
        _metric_display_names = metrics_to_plot
        if not metrics_to_plot:
            print("No metrics found in any of the provided log files.")
            return
    else:
        metrics_to_plot = target_metrics
        if metric_display_names is None:
            _metric_display_names = target_metrics
        else:
            assert len(target_metrics) == len(metric_display_names), (
                "target_metrics and metric_display_names length mismatch."
            )
            _metric_display_names = metric_display_names

    if not metrics_to_plot:
        print("No metrics selected or available to plot.")
        return

    n_metrics = len(metrics_to_plot)
    ncols = 2 if n_metrics > 1 else 1
    nrows = n_metrics // ncols + bool(n_metrics % ncols)
    fig_width = single_plot_figsize[0] * ncols
    fig_height = single_plot_figsize[1] * nrows
    plt.figure(figsize=(fig_width, fig_height))

    # Determine default for use_min_with_these_metrics if not provided
    _use_min_with_these_metrics = use_min_with_these_metrics
    if _use_min_with_these_metrics is None:
        _use_min_with_these_metrics = [
            m for m in metrics_to_plot if "loss" in m.lower()
        ]

    for i_metric, metric_col_name in enumerate(metrics_to_plot):
        metric_disp_name = _metric_display_names[i_metric]
        ax = plt.subplot(nrows, ncols, i_metric + 1)
        ax.set_title(f"{metric_disp_name} per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_disp_name)

        max_epochs_across_exps = 0

        # Determine aggregation function for the current PLOTTED metric
        current_plot_agg_fn = agg_fn # Default from function args

        if _use_min_with_these_metrics and (
            metric_col_name in _use_min_with_these_metrics
        ):
            current_plot_agg_fn = min
        
        if use_max_with_these_metrics and (
            metric_col_name in use_max_with_these_metrics
        ):
            current_plot_agg_fn = max

        for exp in experiment_data: # Iterate through (potentially sorted) experiments
            exp_name = exp["name"]
            logs_df = exp["logs"]
            exp_color = exp["color"]

            if metric_col_name not in logs_df.columns:
                continue

            metric_scores_all = logs_df[metric_col_name]
            train_scores, val_scores = [], []
            for k_score, score_val in enumerate(metric_scores_all):
                if k_score % 2 == 0:
                    train_scores.append(score_val)
                else:
                    val_scores.append(score_val)

            # NaN Handling for plotting
            train_has_only_nans = all(np.isnan(s) for s in train_scores)
            train_has_some_nans = any(np.isnan(s) for s in train_scores)
            val_has_only_nans = all(np.isnan(s) for s in val_scores)
            val_has_some_nans = any(np.isnan(s) for s in val_scores)

            if train_has_some_nans and not train_has_only_nans:
                train_scores = _replace_nans_with_local_avgs(train_scores)
                print(
                    f"WARNING: {exp_name} - {metric_disp_name} train_scores "
                    "had NaNs. Attempted replacement."
                )
            if val_has_some_nans and not val_has_only_nans:
                val_scores = _replace_nans_with_local_avgs(val_scores)
                print(
                    f"WARNING: {exp_name} - {metric_disp_name} val_scores "
                    "had NaNs. Attempted replacement."
                )

            min_len = min(len(train_scores), len(val_scores))
            train_scores = train_scores[:min_len]
            val_scores = val_scores[:min_len]

            num_epochs = len(train_scores)
            if num_epochs == 0:
                continue

            epochs = list(range(1, num_epochs + 1))
            max_epochs_across_exps = max(max_epochs_across_exps, num_epochs)

            # Plotting train scores
            if not all(np.isnan(s) for s in train_scores):
                # agg_fn expects (value, index) tuples
                valid_train_s_tuples = [
                    (s, idx)
                    for idx, s in enumerate(train_scores)
                    if not np.isnan(s)
                ]
                if valid_train_s_tuples:
                    best_s_val_tuple = current_plot_agg_fn(valid_train_s_tuples)
                    best_s_val, best_s_idx = best_s_val_tuple[0], best_s_val_tuple[1]

                    ax.plot(
                        epochs,
                        train_scores,
                        color=exp_color,
                        linestyle="-",
                        label=(
                            f"{exp_name} - Train. Best Score: "
                            r"$\mathbf{" + f"{best_s_val:.3f}" + r"}$ "
                            f"(Ep. {best_s_idx + 1})"
                        ),
                    )

                    ax.hlines(
                        best_s_val,
                        epochs[0],
                        epochs[-1],
                        colors=exp_color,
                        linestyles="dashed",
                        alpha=0.9,
                    )

                    if draw_point_on_best_score:
                        # Draw point on best score with a black border
                        ax.plot(
                            epochs[best_s_idx],
                            best_s_val,
                            "o",
                            color=exp_color,
                            markersize=point_markersize,
                            alpha=0.9,
                            markeredgecolor="black",
                        )

            # Plotting validation scores
            if not all(np.isnan(s) for s in val_scores):
                valid_val_s_tuples = [
                    (s, idx)
                    for idx, s in enumerate(val_scores)
                    if not np.isnan(s)
                ]
                if valid_val_s_tuples:
                    best_s_val_tuple = current_plot_agg_fn(valid_val_s_tuples)
                    best_s_val, best_s_idx = best_s_val_tuple[0], best_s_val_tuple[1]

                    ax.plot(
                        epochs,
                        val_scores,
                        color=exp_color,
                        linestyle="--",
                        label=(
                            f"{exp_name} - Val. Best Score: "
                            r"$\mathbf{" + f"{best_s_val:.3f}" + r"}$ "
                            f"(Ep. {best_s_idx + 1})"
                        ),
                    )

                    ax.hlines(
                        best_s_val,
                        epochs[0],
                        epochs[-1],
                        colors=exp_color,
                        linestyles="dotted",
                        alpha=0.9,
                    )
                    
                    if draw_point_on_best_score:
                        ax.plot(
                            epochs[best_s_idx],
                            best_s_val,
                            "o",
                            color=exp_color,
                            markersize=point_markersize,
                            alpha=0.9,
                            markeredgecolor="black",
                        )

        if max_epochs_across_exps > 0:
            ax.set_xlim(1 - 0.9, max_epochs_across_exps + 0.9)
        ax.legend(fontsize="small")

    plt.tight_layout()
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

def generate_color_variants(color: str, n: int):
    import matplotlib.colors as mcolors
    base_rgb = np.array(mcolors.to_rgb(color))  # Convert to RGB (0-1 scale)
    white_rgb = np.array([1, 1, 1])  # White color in RGB
    return [
        mcolors.to_hex(base_rgb + (white_rgb - base_rgb) * (i / (n - 1)))
        for i in range(n)
    ]

def plot_metric_bars_per_method(method_dicts, method_aliases, metric_names, metric_aliases, title,
                                figsize=(10, 8), scores_fontsize=7, metrics_tick_fontsize=10, metrics_axis_size=1.0,
                                sort_metrics=True, method_idx_to_sort_by=None, sort_methods=True, vertical=False,
                                bbox_to_anchor=None, show_std=False, xtick_rotation=90, xtick_ha='right',
                                xlabel=None, ylabel=None, prepend_mean_score_to_legend=True, xlim=None, ylim=None,
                                save_as_pdf=False, save_path=None, vertical_score_text_rotation=90, colors=_COLORS,
                                hide_xticks=False, hide_yticks=False):
    n = len(method_dicts)
    assert n == len(method_aliases)
    assert n > 0
    m = len(metric_names)
    assert m == len(metric_aliases)
    assert m > 0
    
    scores_per_method = [[method_dicts[i].get(k, 0) for k in metric_names] for i in range(n)]
    std_per_method = [[method_dicts[i].get(f'{k}_std', 0) for k in metric_names] for i in range(n)]
    upper_err_per_method = []
    lower_err_per_method = []
    for i in range(n):
        upper_err_list = []
        lower_err_list = []
        for j in range(m):
            upper_bound = scores_per_method[i][j] + std_per_method[i][j]
            lower_bound = scores_per_method[i][j] - std_per_method[i][j]
            if upper_bound > 1.0:
                delta = upper_bound - 1.0
                upper_bound -= delta
                lower_bound -= delta
            if lower_bound < 0.0:
                delta = -lower_bound
                upper_bound += delta
                lower_bound += delta
            upper_err_list.append(upper_bound - scores_per_method[i][j])
            lower_err_list.append(scores_per_method[i][j] - lower_bound)
        upper_err_per_method.append(upper_err_list)
        lower_err_per_method.append(lower_err_list)
    metric_idxs = list(range(m))
    if sort_metrics:
        scores_minus_lower_err_per_method = [[scores_per_method[i][j] - lower_err_per_method[i][j] for j in range(m)] for i in range(n)]
        if method_idx_to_sort_by is not None: # sort by a specific row
            metric_idxs.sort(key=lambda i: scores_minus_lower_err_per_method[method_idx_to_sort_by][i], reverse=vertical)
        else: # sort by the mean score
            mean_score_per_metric = [np.mean([scores_minus_lower_err_per_method[i][j] for i in range(n)]) for j in range(m)]
            metric_idxs.sort(key=lambda i: mean_score_per_metric[i], reverse=vertical)
    mean_score_per_method = [np.mean(scores_per_method[i]) for i in range(n)]
    method_idxs = list(range(n))
    if sort_methods:
        method_idxs.sort(key=lambda i: mean_score_per_method[i], reverse=vertical)
    min_score = min(min(scores_per_method[i]) for i in range(n))
    max_score = max(max(scores_per_method[i]) for i in range(n))

    plt.figure(figsize=figsize)
    if vertical:
        bar_width = 0.9 * metrics_axis_size / (n * m)
        for i in range(n):
            label = method_aliases[method_idxs[i]]
            if prepend_mean_score_to_legend:
                label = f'({mean_score_per_method[method_idxs[i]]:.3f}) {label}'
            positions = [j * (metrics_axis_size / m) + i * bar_width for j in range(1, m+1)]
            scores = [scores_per_method[method_idxs[i]][metric_idxs[j]] for j in range(m)]
            # stds = [std_per_method[method_idxs[i]][metric_idxs[j]] for j in range(m)] if show_std else None
            if show_std:
                upper_err = [upper_err_per_method[method_idxs[i]][metric_idxs[j]] for j in range(m)]
                lower_err = [lower_err_per_method[method_idxs[i]][metric_idxs[j]] for j in range(m)]
                yerr = [lower_err, upper_err]
            else:
                yerr = None
            plt.bar(positions, scores, width=bar_width, label=label, color=colors[method_idxs[i] % len(colors)], yerr=yerr, capsize=3)
            for j in range(m):
                plt.text(positions[j], scores[j] + (max_score - min_score) * 0.01, f'{scores[j]:.3f}', ha='center',
                         va='bottom', fontsize=scores_fontsize, rotation=vertical_score_text_rotation)
        if not hide_xticks:
            plt.xticks([j * (metrics_axis_size / m) + bar_width * 0.5 * (n - 1) for j in range(1, m+1)], [metric_aliases[i] for i in metric_idxs],
                    fontsize=metrics_tick_fontsize, rotation=xtick_rotation, ha=xtick_ha)
        else:
            plt.xticks([])
        if xlabel is None: xlabel = 'Metric'
        if ylabel is None: ylabel = 'Score'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='y')
        if bbox_to_anchor is None:
            plt.legend(loc='upper left')
        else:
            plt.legend(bbox_to_anchor=bbox_to_anchor, loc='lower center', borderaxespad=0.)
    else:
        bar_height = 0.9 * metrics_axis_size / (n * m)
        for i in range(n):
            label = method_aliases[method_idxs[n-1-i]]
            if prepend_mean_score_to_legend:
                label = f'({mean_score_per_method[method_idxs[n-1-i]]:.3f}) {label}'
            positions = [j * (metrics_axis_size / m) + (n-1-i) * bar_height for j in range(1, m+1)]
            scores = [scores_per_method[method_idxs[n-1-i]][metric_idxs[j]] for j in range(m)]
            # stds = [std_per_method[method_idxs[n-1-i]][metric_idxs[j]] for j in range(m)] if show_std else None
            if show_std:
                upper_err = [upper_err_per_method[method_idxs[n-1-i]][metric_idxs[j]] for j in range(m)]
                lower_err = [lower_err_per_method[method_idxs[n-1-i]][metric_idxs[j]] for j in range(m)]
                xerr = [lower_err, upper_err]
            else:
                xerr = None
            plt.barh(positions, scores, height=bar_height, label=label, color=colors[method_idxs[n-1-i] % len(colors)], xerr=xerr, capsize=3)
            for j in range(m):
                plt.text(scores[j] + (max_score - min_score) * 0.01, positions[j], f'{scores[j]:.3f}', ha='left', va='center', fontsize=scores_fontsize)
        if not hide_yticks:
            plt.yticks([j * (metrics_axis_size / m) + bar_height * 0.5 * (n - 1) for j in range(1, m+1)], [metric_aliases[i] for i in metric_idxs],
                    fontsize=metrics_tick_fontsize)
        else:
            plt.yticks([])
        if xlabel is None: xlabel = 'Score'
        if ylabel is None: ylabel = 'Metric'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='x')
        if bbox_to_anchor is None:
            plt.legend(loc='upper left')
        else:
            plt.legend(bbox_to_anchor=bbox_to_anchor, loc='upper left', borderaxespad=0.)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)

    # Save the plot as a PDF file
    if save_as_pdf:
        assert save_path is not None
        assert save_path.endswith('.pdf')
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # create the directory if it doesn't exist
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, format='pdf')
        print_blue(f'Saved the plot as a PDF file: {save_path}')

    # Show the plot
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

def visualize_predicted_bounding_boxes__yolo(image_path, pred_coords, pred_classes, pred_confs,
                                             class_names, figsize, format='xywh', gt_bbox_coords=None,
                                             classes_to_highlight=None, hide_other_classes=False,
                                             minimum_confidence=0.0):
    from PIL import Image
    import matplotlib.patches as patches

    print(f'image_path={image_path}')

    show_gt = gt_bbox_coords is not None

    if classes_to_highlight is not None:
        assert isinstance(classes_to_highlight, list)
    
    if show_gt:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize) # 1 row, 2 columns (left: GT, right: Pred)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize) # 1 row, 1 column

    # Image
    image = Image.open(image_path)
    image = image.convert('RGB')
    width = image.size[0]
    height = image.size[1]
    if show_gt:
        ax[0].imshow(image)
        ax[1].imshow(image)
    else:
        ax.imshow(image)

    def _adapt_bbox_format(coords, format):
        if format == 'xywh':
            x_mid = coords[0] * width
            y_mid = coords[1] * height
            w = coords[2] * width
            h = coords[3] * height
            x1, y1, x2, y2 = x_mid - w / 2, y_mid - h / 2, x_mid + w / 2, y_mid + h / 2
        elif format == 'xyxy':
            x1 = coords[0] * width
            y1 = coords[1] * height
            x2 = coords[2] * width
            y2 = coords[3] * height
        else:
            raise ValueError(f'Unknown format {format}')
        return x1, y1, x2, y2
    
    if show_gt:
        # Ground truth bounding boxes
        for i in range(len(gt_bbox_coords)):
            if classes_to_highlight is not None and class_names[i] not in classes_to_highlight:
                if hide_other_classes:
                    continue
            if gt_bbox_coords[i] is not None:
                coords_list = gt_bbox_coords[i]
                color = _COLORS[i % len(_COLORS)]
                for coords in coords_list:
                    x1, y1, x2, y2 = _adapt_bbox_format(coords, format)
                    alpha = 1.0 if (classes_to_highlight is None or class_names[i] in classes_to_highlight) else 0.4
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=color, facecolor='none', alpha=alpha)
                    ax[0].add_patch(rect)
                    ax[0].text(x1, y1-10, class_names[i], color=color, fontsize=12, backgroundcolor=(0, 0, 0, alpha), alpha=alpha)

    # Predicted bounding boxes
    if show_gt:
        pred_ax = ax[1]
    else:
        pred_ax = ax
    if len(pred_coords) > 0:
        idxs = np.argsort(pred_confs)
        for i in idxs:
            if pred_confs[i] < minimum_confidence:
                continue
            if classes_to_highlight is not None and class_names[pred_classes[i]] not in classes_to_highlight:
                if hide_other_classes:
                    continue
            x1, y1, x2, y2 = _adapt_bbox_format(pred_coords[i], format)
            color = _COLORS[pred_classes[i] % len(_COLORS)]
            alpha = 1.0 if (classes_to_highlight is None or class_names[pred_classes[i]] in classes_to_highlight) else 0.1
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=color, facecolor='none', alpha=alpha)
            pred_ax.add_patch(rect)
            pred_ax.text(x1, y1-10, f'{class_names[pred_classes[i]]} ({pred_confs[i]:.2f})', color=color, fontsize=12, backgroundcolor=(0, 0, 0, alpha), alpha=alpha)

    # Set titles
    if show_gt:
        ax[0].set_title('Ground Truth')
        ax[1].set_title('Predictions')
    else:
        pred_ax.set_title('Predictions')

    # remove ticks
    if show_gt:
        ax[0].axis('off')
        ax[1].axis('off')
    else:
        pred_ax.axis('off')

    # Show the plot
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


# def visualize_visual_grounding_as_bboxes(phrases, phrase_classifier_probs, bbox_coords, bbox_probs, phrase_ids,
#                                          figsize, max_cols=3, gt_phrases_to_highlight=None, image=None, image_path=None,
#                                          show_heatmaps=False, heatmaps=None, bbox_format='xyxy'):
#     """
#     Visualizes visual grounding by displaying bounding boxes over an image with associated phrases.
    
#     Args:
#         image (PIL.Image.Image, optional): Image object. Default is None.
#         image_path (str, optional): Path to the image file. Default is None.
#         phrases (list of str): List of phrases corresponding to detected objects.
#         phrase_classifier_probs (numpy.ndarray): Array of shape (n,) containing confidence scores for each phrase.
#         bbox_coords (numpy.ndarray): Array of shape (n, 4) containing bounding box coordinates in normalized format.
#         bbox_probs (numpy.ndarray): Array of shape (n,) containing confidence scores for each bounding box.
#         phrase_ids (numpy.ndarray): Array of shape (n,) containing indices mapping bounding boxes to phrases.
#         figsize (tuple): Figure size for visualization.
#         max_cols (int, optional): Maximum number of columns in the grid. Default is 3.
#         gt_phrases_to_highlight (list of str, optional): List of phrases to highlight in the visualization. Default is None.
#         show_heatmaps (bool, optional): Whether to display heatmaps. Default is False.
#         heatmaps (numpy.ndarray, optional): Array of shape (num_phrases, num_regions), where num_regions = H * W.
#         bbox_format (str, optional): Format of bounding box coordinates. Default is 'xyxy'.
#     """
#     assert image is not None or image_path is not None, "Either image or image_path must be provided."
#     assert len(phrases) == len(phrase_classifier_probs), "Mismatched dimensions between phrases and phrase classifier probabilities."
#     assert bbox_coords.shape[0] == bbox_probs.shape[0] == phrase_ids.shape[0], "Mismatched dimensions between bounding box arrays."
#     assert bbox_coords.shape[1] == 4, "Bounding box coordinates must have shape (n, 4)."
#     if show_heatmaps:
#         assert heatmaps is not None, "Heatmaps must be provided when show_heatmaps is True."
#         assert heatmaps.shape[0] == len(phrases), "Heatmaps should have the same first dimension as phrases."
    
#     from PIL import Image
#     import matplotlib.patches as patches
#     import textwrap

#     # Determine grid size
#     n_phrases = len(phrases)
#     n_cols = min(n_phrases, max_cols)
#     n_rows = math.ceil(n_phrases / n_cols)
#     fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

#     # Define wrap length for titles
#     wrap_length = (figsize[0] / n_cols) * (60 / 7.5)

#     # Load and process image if path is provided
#     if image is None:
#         assert image_path is not None, "Image path must be provided."
#         image = Image.open(image_path).convert('RGB')
#     width, height = image.size

#     # Infer heatmap dimensions
#     if show_heatmaps:
#         H = W = math.isqrt(heatmaps.shape[1])  # Assuming square heatmap regions
#         assert H * W == heatmaps.shape[1], "Heatmap size mismatch: Expected H * W regions."
#         # Compute grid lines positions based on the image size and feature map size.
#         cell_width = width / W
#         cell_height = height / H
#         # Grid lines (excluding image borders):
#         vlines = [cell_width * i for i in range(1, W)]
#         hlines = [cell_height * i for i in range(1, H)]

#     # Iterate through unique phrases
#     for phrase_id in range(n_phrases):
#         row, col = divmod(phrase_id, n_cols)
#         ax[row, col].imshow(image)

#         phrase_prob = phrase_classifier_probs[phrase_id]

#         # Overlay heatmap if enabled
#         if show_heatmaps:
#             heatmap = heatmaps[phrase_id].reshape(H, W)
#             heatmap_resized = np.array(Image.fromarray(heatmap).resize((width, height), Image.BILINEAR))
#             ax[row, col].imshow(heatmap_resized, cmap='jet', alpha=0.5)
#             for x in vlines:
#                 ax[row, col].axvline(x, color="white", linestyle="--", linewidth=1)
#             for y in hlines:
#                 ax[row, col].axhline(y, color="white", linestyle="--", linewidth=1)

#         # Filter bounding boxes for the current phrase
#         mask = phrase_ids == phrase_id
#         for bbox, prob in zip(bbox_coords[mask], bbox_probs[mask]):
#             if bbox_format == 'xyxy':
#                 x1, y1, x2, y2 = bbox * np.array([width, height, width, height])
#             elif bbox_format == 'cxcywh':
#                 cx, cy, w, h = bbox
#                 x1 = (cx - w / 2) * width
#                 y1 = (cy - h / 2) * height
#                 x2 = (cx + w / 2) * width
#                 y2 = (cy + h / 2) * height
#             else: raise ValueError(f"Unknown bbox_format: {bbox_format}")
#             rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='yellow', facecolor='none')
#             ax[row, col].add_patch(rect)
#             ax[row, col].text(x1, y1, f'{prob:.2f}', color='red', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

#         # Set subplot title with wrapped text
#         title = f'{phrases[phrase_id]}\n({phrase_prob:.2f})'
#         wrapped_title = "\n".join(textwrap.wrap(title, wrap_length))
#         if gt_phrases_to_highlight is not None and phrases[phrase_id] in gt_phrases_to_highlight:
#             ax[row, col].set_title(wrapped_title, color='red')
#         else:
#             ax[row, col].set_title(wrapped_title)

#     plt.show()
    
def visualize_visual_grounding_as_bboxes(
    phrases: List[str],
    phrase_classifier_probs: np.ndarray,
    bbox_coords: np.ndarray,
    bbox_probs: np.ndarray,
    phrase_ids: np.ndarray,
    figsize: Tuple[float, float],
    max_cols: int = 3,
    gt_phrases_to_highlight: Optional[List[str]] = None,
    image: Optional[PILImage] = None,
    image_path: Optional[str] = None,
    show_heatmaps: bool = False,
    heatmaps: Optional[np.ndarray] = None,
    bbox_format: str = 'xyxy',
    gt_bbox_coords: Optional[np.ndarray] = None  # New argument
):
    """
    Visualizes visual grounding by displaying bounding boxes over an image with associated phrases.
    
    Args:
        phrases (List[str]): List of phrases corresponding to detected objects.
        phrase_classifier_probs (np.ndarray): Array of shape (N,) containing confidence scores for each phrase.
        bbox_coords (np.ndarray): Array of shape (M, 4) containing bounding box coordinates in normalized format.
        bbox_probs (np.ndarray): Array of shape (M,) containing confidence scores for each bounding box.
        phrase_ids (np.ndarray): Array of shape (M,) containing indices mapping bounding boxes to phrases (0 to N-1).
        figsize (Tuple[float, float]): Figure size for visualization.
        max_cols (int, optional): Maximum number of columns in the grid. Default is 3.
        gt_phrases_to_highlight (Optional[List[str]], optional): List of phrases to highlight in the visualization. Default is None.
        image (Optional[PILImage], optional): PIL Image object. Default is None.
        image_path (Optional[str], optional): Path to the image file. Default is None.
        show_heatmaps (bool, optional): Whether to display heatmaps. Default is False.
        heatmaps (Optional[np.ndarray], optional): Array of shape (N, num_regions), where num_regions = H * W.
        bbox_format (str, optional): Format of bounding box coordinates ('xyxy' or 'cxcywh'). Default is 'xyxy'.
        gt_bbox_coords (Optional[np.ndarray], optional): Array of shape (K, 4) containing ground-truth
                                                      bounding box coordinates in normalized format. Default is None.
    """
    from PIL import Image as PILImageModule # For loading image from path and operations
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import textwrap
    import math

    assert image is not None or image_path is not None, "Either image or image_path must be provided."
    assert len(phrases) == len(phrase_classifier_probs), \
        "Mismatched dimensions between phrases and phrase classifier probabilities."
    if bbox_coords.size > 0: # Allow empty bbox_coords if no detections
        assert bbox_coords.shape[0] == bbox_probs.shape[0] == phrase_ids.shape[0], \
            "Mismatched dimensions between bounding box arrays."
        assert bbox_coords.ndim == 2 and bbox_coords.shape[1] == 4, \
            "Bounding box coordinates must have shape (M, 4)."
    elif not (bbox_probs.size == 0 and phrase_ids.size == 0):
        raise ValueError("If bbox_coords is empty, bbox_probs and phrase_ids must also be empty.")


    if show_heatmaps:
        assert heatmaps is not None, "Heatmaps must be provided when show_heatmaps is True."
        assert heatmaps.shape[0] == len(phrases), \
            "Heatmaps should have the same first dimension as phrases."
        if len(phrases) > 0 : # Only check heatmap content if there are phrases to map to
             assert heatmaps.shape[1] > 0, "Heatmap has zero regions (heatmaps.shape[1] is 0)."


    if gt_bbox_coords is not None:
        assert gt_bbox_coords.ndim == 2 and gt_bbox_coords.shape[1] == 4, \
            "gt_bbox_coords must be a 2D array with shape (K, 4)."

    n_phrases = len(phrases)
    
    plot_single_gt_only = False
    if n_phrases == 0:
        if gt_bbox_coords is not None and (image is not None or image_path is not None):
            plot_single_gt_only = True
            n_rows_fig, n_cols_fig = 1, 1
        else:
            print("No phrases to visualize and no ground truth boxes to display on an image.")
            return
    else: # n_phrases > 0
        n_cols_fig = min(n_phrases, max_cols)
        n_rows_fig = math.ceil(n_phrases / n_cols_fig)
    
    fig, ax_array = plt.subplots(n_rows_fig, n_cols_fig, figsize=figsize, squeeze=False)

    wrap_length = (figsize[0] / n_cols_fig) * (60 / 7.5) # Approx chars per line

    if image is None:
        assert image_path is not None, "Image path must be provided if image is None."
        current_image = PILImageModule.open(image_path).convert('RGB')
    else:
        current_image = image.convert('RGB') if image.mode != 'RGB' else image
    width, height = current_image.size

    H, W = 0, 0 # Initialize H, W
    vlines, hlines = [], []
    if show_heatmaps and n_phrases > 0 and heatmaps is not None and heatmaps.shape[1] > 0:
        # Ensure heatmaps is not None and has regions before trying to calculate H, W
        H = W = math.isqrt(heatmaps.shape[1])
        if not (H * W == heatmaps.shape[1]): # Check if perfect square
            print(f"Warning: Heatmap regions {heatmaps.shape[1]} is not a perfect square. Heatmap display might be incorrect.")
            # Fallback or error, for now, proceed with caution or disable heatmaps for this case
            show_heatmaps = False # Disable heatmap if dimensions are problematic
        else:
            cell_width = width / W
            cell_height = height / H
            vlines = [cell_width * i for i in range(1, W)]
            hlines = [cell_height * i for i in range(1, H)]


    if plot_single_gt_only:
        ax_current = ax_array[0, 0]
        ax_current.imshow(current_image)
        ax_current.set_title("Ground Truth Bounding Boxes")
        ax_current.axis('off')

        if gt_bbox_coords is not None:
            for gt_bbox in gt_bbox_coords:
                if bbox_format == 'xyxy':
                    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox * np.array([width, height, width, height])
                elif bbox_format == 'cxcywh':
                    cx_gt, cy_gt, w_gt, h_gt = gt_bbox
                    x1_gt = (cx_gt - w_gt / 2) * width
                    y1_gt = (cy_gt - h_gt / 2) * height
                    x2_gt = (cx_gt + w_gt / 2) * width
                    y2_gt = (cy_gt + h_gt / 2) * height
                else:
                    raise ValueError(f"Unknown bbox_format: {bbox_format}")
                gt_rect = patches.Rectangle(
                    (x1_gt, y1_gt), x2_gt - x1_gt, y2_gt - y1_gt, 
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                ax_current.add_patch(gt_rect)
    else: # n_phrases > 0
        for phrase_idx in range(n_phrases):
            row, col = divmod(phrase_idx, n_cols_fig)
            ax_current = ax_array[row, col]
            ax_current.imshow(current_image)
            ax_current.axis('off')

            if gt_bbox_coords is not None:
                for gt_bbox in gt_bbox_coords:
                    if bbox_format == 'xyxy':
                        x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox * np.array([width, height, width, height])
                    elif bbox_format == 'cxcywh':
                        cx_gt, cy_gt, w_gt, h_gt = gt_bbox
                        x1_gt = (cx_gt - w_gt / 2) * width
                        y1_gt = (cy_gt - h_gt / 2) * height
                        x2_gt = (cx_gt + w_gt / 2) * width
                        y2_gt = (cy_gt + h_gt / 2) * height
                    else:
                        raise ValueError(f"Unknown bbox_format: {bbox_format}")
                    gt_rect = patches.Rectangle(
                        (x1_gt, y1_gt), x2_gt - x1_gt, y2_gt - y1_gt,
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    ax_current.add_patch(gt_rect)
            
            phrase_prob = phrase_classifier_probs[phrase_idx]

            if show_heatmaps and heatmaps is not None and phrase_idx < heatmaps.shape[0] and H > 0 and W > 0 :
                heatmap_data = heatmaps[phrase_idx].reshape(H, W)
                heatmap_contiguous = np.ascontiguousarray(heatmap_data)
                heatmap_pil = PILImageModule.fromarray(heatmap_contiguous)
                heatmap_resized = np.array(heatmap_pil.resize((width, height), PILImageModule.BILINEAR))
                ax_current.imshow(heatmap_resized, cmap='jet', alpha=0.5)
                for x_val in vlines:
                    ax_current.axvline(x_val, color="white", linestyle="--", linewidth=0.8)
                for y_val in hlines:
                    ax_current.axhline(y_val, color="white", linestyle="--", linewidth=0.8)

            mask = phrase_ids == phrase_idx
            if bbox_coords.size > 0: # Process predicted boxes only if they exist
                for bbox, prob in zip(bbox_coords[mask], bbox_probs[mask]):
                    if bbox_format == 'xyxy':
                        x1, y1, x2, y2 = bbox * np.array([width, height, width, height])
                    elif bbox_format == 'cxcywh':
                        cx, cy, w_box, h_box = bbox # Renamed w, h to avoid conflict
                        x1 = (cx - w_box / 2) * width
                        y1 = (cy - h_box / 2) * height
                        x2 = (cx + w_box / 2) * width
                        y2 = (cy + h_box / 2) * height
                    else: 
                        raise ValueError(f"Unknown bbox_format: {bbox_format}")
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, 
                        linewidth=2, edgecolor='yellow', facecolor='none'
                    )
                    ax_current.add_patch(rect)
                    ax_current.text(x1, y1 - 2, f'{prob:.2f}', color='red', fontsize=8, 
                                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

            title = f'{phrases[phrase_idx]}\n(prob: {phrase_prob:.2f})'
            wrapped_title = "\n".join(textwrap.wrap(title, int(wrap_length)))
            title_color = 'red' if gt_phrases_to_highlight and phrases[phrase_idx] in gt_phrases_to_highlight else 'black'
            ax_current.set_title(wrapped_title, color=title_color, fontsize=10)

        for i in range(n_phrases, n_rows_fig * n_cols_fig):
            row, col = divmod(i, n_cols_fig)
            if row < ax_array.shape[0] and col < ax_array.shape[1]:
                 ax_array[row, col].axis('off')
                 ax_array[row, col].set_frame_on(False)


    plt.tight_layout()
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


def plot_barchart(
    values,
    title,
    x_label,
    y_label,
    figsize=(10, 10),
    bar_color='blue',
    horizontal=False,
    bar_labels=None,
    sort_values=False,
    show_values_on_bars=False,
    values_fontsize=10,
    values_color='black',
    values_rotation=0,
    values_decimal_points=3,
    show_stddev=False,
    stddevs=None,
    x_limits=None,
    y_limits=None,
    title_alignment='center',
    save_as_pdf=False,
    save_path=None,
    x_label_alignment='center',
    tick_rotation=0
):
    """
    Plots a bar chart with various customization options.

    Parameters:
    - values (list): List of numerical values for the bars.
    - title (str): Title of the chart.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    - figsize (tuple): Size of the figure (default: (10, 10)).
    - bar_color (str): Color of the bars (default: 'blue').
    - horizontal (bool): Whether to plot a horizontal bar chart (default: False).
    - bar_labels (list): Labels for the bars (default: None).
    - sort_values (bool): Whether to sort the values (default: False).
    - show_values_on_bars (bool): Whether to display values on the bars (default: False).
    - values_fontsize (int): Font size for the values displayed on the bars (default: 10).
    - values_color (str): Color of the values displayed on the bars (default: 'black').
    - values_rotation (int): Rotation angle for the values displayed on the bars (default: 0).
    - values_decimal_points (int): Number of decimal points to display for values (default: 3).
    - show_stddev (bool): Whether to show standard deviation as error bars (default: False).
    - stddevs (list): List of standard deviation values (required if show_stddev=True).
    - x_limits (tuple): Limits for the x-axis (default: None).
    - y_limits (tuple): Limits for the y-axis (default: None).
    - title_alignment (str): Alignment of the title (default: 'center').
    - save_as_pdf (bool): Whether to save the plot as a PDF file (default: False).
    - save_path (str): Path to save the PDF file (required if save_as_pdf=True).
    - x_label_alignment (str): Horizontal alignment of the x-axis label (default: 'center').
    - tick_rotation (int): Rotation angle for x-axis or y-axis ticks (default: 0).

    Returns:
    - None: Displays the plot and optionally saves it as a PDF.
    """
    # Validate inputs for standard deviation
    if show_stddev:
        assert stddevs is not None, "Standard deviations must be provided when show_stddev=True."
        assert len(values) == len(stddevs), "Length of values and stddevs must match."

    # Sort values and labels if required
    num_bars = len(values)
    if sort_values:
        indices = list(range(num_bars))
        indices.sort(key=lambda i: values[i], reverse=not horizontal)
        values = [values[i] for i in indices]
        if bar_labels is not None:
            bar_labels = [bar_labels[i] for i in indices]

    # Default bar labels if none are provided
    if bar_labels is None:
        bar_labels = range(1, num_bars + 1)

    # Create the plot
    plt.figure(figsize=figsize)
    if horizontal:
        # Horizontal bar chart
        if show_stddev:
            plt.barh(
                range(1, num_bars + 1),
                values,
                xerr=stddevs,
                color=bar_color,
                capsize=3
            )
        else:
            plt.barh(range(1, num_bars + 1), values, color=bar_color)
        plt.yticks(range(1, num_bars + 1), bar_labels, rotation=tick_rotation, ha='right' if tick_rotation != 0 else 'center')
        plt.ylabel(y_label)
        plt.xlabel(x_label, ha=x_label_alignment)

        # Display values on bars
        if show_values_on_bars:
            for i in range(num_bars):
                plt.text(
                    values[i],
                    i + 1,
                    f'{values[i]:.{values_decimal_points}f}',
                    ha='left',
                    va='center',
                    fontsize=values_fontsize,
                    color=values_color,
                    rotation=values_rotation
                )
    else:
        # Vertical bar chart
        if show_stddev:
            plt.bar(
                range(1, num_bars + 1),
                values,
                yerr=stddevs,
                color=bar_color,
                capsize=3
            )
        else:
            plt.bar(range(1, num_bars + 1), values, color=bar_color)
        plt.xticks(range(1, num_bars + 1), bar_labels, rotation=tick_rotation, ha='right' if tick_rotation != 0 else 'center')
        plt.xlabel(x_label, ha=x_label_alignment)
        plt.ylabel(y_label)

        # Display values on bars
        if show_values_on_bars:
            for i in range(num_bars):
                plt.text(
                    i + 1,
                    values[i],
                    f'{values[i]:.{values_decimal_points}f}',
                    ha='center',
                    va='bottom',
                    fontsize=values_fontsize,
                    color=values_color,
                    rotation=values_rotation
                )

    # Set axis limits if provided
    if x_limits is not None:
        plt.xlim(x_limits)
    if y_limits is not None:
        plt.ylim(y_limits)

    # Set the title
    plt.title(title, loc=title_alignment)

    # Save the plot as a PDF file if required
    if save_as_pdf:
        import os
        assert save_path is not None, "Save path must be provided when save_as_pdf=True."
        assert save_path.endswith('.pdf'), "Save path must end with '.pdf'."
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, format='pdf')
        print(f'Saved the plot as a PDF file: {save_path}')

    # Display the plot
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