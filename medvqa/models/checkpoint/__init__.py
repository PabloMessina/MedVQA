from collections import namedtuple
import os
import re

from medvqa.utils.files import load_json_file, save_to_json

_CHECKPOINT_REGEX = re.compile(r'^[A-Za-z]+_(\d+)(?:_((.+)=)?([\d\.]+)\.pt)?$')
CheckpointInfo = namedtuple('CheckpointInfo', ('name', 'epoch', 'metric', 'value'))

def _split_checkpoint_name(name):
    matched = _CHECKPOINT_REGEX.match(name)
    epoch, _, metric_name, metric_value = matched.groups()
    return CheckpointInfo(name, int(epoch), metric_name, float(metric_value))

def get_checkpoint_filepath(folder):
    checkpoint_names = [f for f in os.listdir(folder) if f.endswith('.pt')]
    if len(checkpoint_names) == 0:
        raise Exception('No checkpoints found in', folder)
    # TODO: this assumes a single metric, so consider generalizing this
    # to more than one metric in the same folder
    best_value = -9999
    best_epoch = -1
    best_name = None
    print('checkpoint_names =', checkpoint_names)
    for name in checkpoint_names:
        info = _split_checkpoint_name(name)
        if info.value > best_value or (info.value == best_value and info.epoch > best_epoch):
            best_value = info.value
            best_epoch = info.epoch
            best_name = info.name
    return os.path.join(folder, best_name)

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


