import matplotlib.pyplot as plt
import pandas as pd

def plot_train_val_curves(logs_path, metrics, metric_names, agg_fn=max, single_plot_figsize=(8, 6),
                          use_min_with_these_metrics=None, use_max_with_these_metrics=None):

    assert len(metrics) == len(metric_names)
    assert len(metrics) > 0
    n = len(metrics)    
    ncols = 2 if n > 1 else 1
    nrows = n // ncols + bool(n % ncols)

    logs = pd.read_csv(logs_path)

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