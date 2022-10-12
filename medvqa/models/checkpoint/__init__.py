from collections import namedtuple
import os
import re

from medvqa.utils.files import load_json_file, save_to_json

_CHECKPOINT_REGEX = re.compile(r'^[A-Za-z]+_(\d+)(?:_((.+)=)?([\d\.]+)\.pt)?$')
CheckpointInfo = namedtuple('CheckpointInfo', ('name', 'epoch', 'metric', 'value'))

def split_checkpoint_name(name):
    matched = _CHECKPOINT_REGEX.match(name)
    epoch, _, metric_name, metric_value = matched.groups()
    return CheckpointInfo(name, int(epoch), metric_name, float(metric_value))

def get_checkpoint_filepath(folder, verbose=True):
    checkpoint_names = [f for f in os.listdir(folder) if f.endswith('.pt')]
    if len(checkpoint_names) == 0:
        raise Exception('No checkpoints found in', folder)
    # TODO: this assumes a single metric, so consider generalizing this
    # to more than one metric in the same folder
    best_value = -9999
    best_epoch = -1
    best_name = None
    if verbose:
        print('checkpoint_names =', checkpoint_names)
    for name in checkpoint_names:
        info = split_checkpoint_name(name)
        if info.value > best_value or (info.value == best_value and info.epoch > best_epoch):
            best_value = info.value
            best_epoch = info.epoch
            best_name = info.name
    return os.path.join(folder, best_name)

def get_matching_checkpoint_epoch(results_file_path):
    str_a = f'{os.path.sep}results{os.path.sep}'
    str_b = f'{os.path.sep}models{os.path.sep}'
    assert str_a in results_file_path
    ref_time = os.path.getmtime(results_file_path)
    model_folder_path = os.path.dirname(results_file_path).replace(str_a, str_b)
    checkpoint_names = [f for f in os.listdir(model_folder_path) if f.endswith('.pt')]
    best_value = -9999
    best_epoch = -1
    for name in checkpoint_names:
        checkpoint_path = os.path.join(model_folder_path, name)
        if os.path.getmtime(checkpoint_path) <= ref_time:
            info = split_checkpoint_name(name)
            if info.value > best_value or (info.value == best_value and info.epoch > best_epoch):
                best_epoch = info.epoch
                best_value = info.value
    assert best_epoch != -1
    return best_epoch

def load_metadata(folder):
    fpath = os.path.join(folder, 'metadata.json')    
    data = load_json_file(fpath)
    print ('metadata loaded from', fpath)
    return data

def save_metadata(folder, **kwargs):
    data = dict(kwargs)
    fpath = os.path.join(folder, 'metadata.json')
    save_to_json(data, fpath)
    print ('metadata saved to', fpath)


