import matplotlib.pyplot as plt
import pandas as pd

def plot_train_val_curves(logs_path, metric, metric_name, agg_fn=max, figsize=(12, 6)):
    
    logs = pd.read_csv(logs_path)
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
    
    plt.figure(figsize=figsize)
    eps = 0.9
    
    ax = plt.subplot(111)
    ax.set_xlim(epochs[0]-eps, epochs[-1]+eps)
    ax.set_title(f'{metric_name} per epoch')
    ax.plot(epochs, train_scores, label=f'{metric_name} (Training)')
    ax.plot(epochs, val_scores, label=f'{metric_name} (Validation)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)    
    ax.legend()
    best_train_score, best_train_i = agg_fn((a,i) for i,a in enumerate(train_scores))
    ax.hlines(best_train_score, epochs[0], epochs[-1], colors=('green',), linestyles='dashed',
               label=f'best train {metric_name}={best_train_score:.3f}, epoch={best_train_i}')
    best_val_score, best_val_i = agg_fn((a,i) for i,a in enumerate(val_scores))
    ax.hlines(best_val_score, epochs[0], epochs[-1], colors=('red',), linestyles='dashed',
               label=f'best val {metric_name}={best_val_score:.3f}, epoch={best_val_i}')
    ax.legend()
    
    plt.show()