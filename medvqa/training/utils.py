def append_metric_name(train_list, val_list, log_list, metric_name, train=True, val=True, log=True):
    if train: train_list.append(metric_name)
    if val: val_list.append(metric_name)
    if log: log_list.append(metric_name)