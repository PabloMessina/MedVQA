def append_metric_name(train_list, val_list, metric_name, train=True, val=True):
    if train: train_list.append(metric_name)
    if val: val_list.append(metric_name)