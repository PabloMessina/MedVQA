from collections import namedtuple
import os
import random
import re

from medvqa.utils.files import get_cached_json_file, save_json
from medvqa.utils.logging import print_orange

_CHECKPOINT_REGEX = re.compile(r'^[A-Za-z]+_(\d+)(?:_((.+)=)?([\d\.]+)\.pt)?$')
CheckpointInfo = namedtuple('CheckpointInfo', ('name', 'epoch', 'metric', 'value'))

def split_checkpoint_name(name):
    matched = _CHECKPOINT_REGEX.match(name)
    epoch, _, metric_name, metric_value = matched.groups()
    return CheckpointInfo(name, int(epoch), metric_name, float(metric_value))

def get_checkpoint_filepath(folder, verbose=True):
    checkpoint_names = [f for f in os.listdir(folder) if f.endswith('.pt')]
    if len(checkpoint_names) == 0:
        raise FileNotFoundError(f'No checkpoint found in {folder}')
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

def get_model_name_from_checkpoint_path(checkpoint_path):
    x = f'{os.path.sep}models{os.path.sep}'
    assert x in checkpoint_path
    return checkpoint_path.split(x)[1].split(os.path.sep)[0]

def load_metadata(folder, verbose=True):
    fpath = os.path.join(folder, 'metadata.json')    
    data = get_cached_json_file(fpath)
    if verbose:
        print('metadata loaded from', fpath)
    return data

def save_metadata(folder, verbose=True, **kwargs):
    data = dict(kwargs)
    fpath = os.path.join(folder, 'metadata.json')
    save_json(data, fpath)
    if verbose:
        print('metadata saved to', fpath)

def load_model_state_dict(model, state_dict, ignore_size_mismatch=True, strict=False):
    if ignore_size_mismatch:
        model_state_dict = model.state_dict()
        to_delete = []
        for k in state_dict.keys():
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}")
                    to_delete.append(k)
        for k in to_delete:
            del state_dict[k]
    # count intersection over union of keys
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    intersection = model_keys & state_dict_keys
    union = model_keys | state_dict_keys
    if len(intersection) != len(union):
        print_orange(f"Warning: model state dict has {len(model_keys)} keys, "
            f"loaded state dict has {len(state_dict_keys)} keys, "
            f"intersection has {len(intersection)} keys, "
            f"union has {len(union)} keys.")
        missing_keys = list(model_keys - state_dict_keys)
        if len(missing_keys) > 0:
            print_orange("Examples of keys in model but not in loaded state dict:")
            missing_keys = random.sample(missing_keys, min(10, len(missing_keys)))
            for k in missing_keys:
                print_orange(f"  {k}")
        missing_keys = list(state_dict_keys - model_keys)
        if len(missing_keys) > 0:
            print_orange("Examples of keys in loaded state dict but not in model:")
            missing_keys = random.sample(missing_keys, min(10, len(missing_keys)))
            for k in missing_keys:
                print_orange(f"  {k}")
    model.load_state_dict(state_dict, strict=strict)

# define named tuple for model training history
ModelTrainingInstance = namedtuple('ModelTrainingInstance', ('timestamp', 'model_dir', 'datasets',
                                                             'best_epoch', 'batches_per_epoch', 'batch_size'))
_training_history_cache = {}

def get_model_training_history(model_checkpoint_folder_path):
    if model_checkpoint_folder_path[-1] == os.path.sep:
        model_checkpoint_folder_path = model_checkpoint_folder_path[:-1]
    if model_checkpoint_folder_path in _training_history_cache:
        return _training_history_cache[model_checkpoint_folder_path]
    
    checkpoint_path = get_checkpoint_filepath(model_checkpoint_folder_path, verbose=False)
    metadata = load_metadata(model_checkpoint_folder_path, verbose=False)
    best_epoch = split_checkpoint_name(os.path.basename(checkpoint_path)).epoch
    batch_size = metadata['dataloading_kwargs']['batch_size']
    folder = os.path.basename(model_checkpoint_folder_path)
    model_dir = os.path.basename(os.path.dirname(model_checkpoint_folder_path))
    timestamp = folder[:15]
    try:
        datasets = folder[16:folder.index('_',16)]
    except ValueError:
        print(f'WARNING: could not parse datasets from {folder} (model_dir={model_dir})')
        raise
    datasets = f'{len(datasets.split("+"))}:{datasets}'
    training_instance = ModelTrainingInstance(
        timestamp=timestamp,
        model_dir=model_dir,
        datasets=datasets,
        best_epoch=best_epoch,
        batches_per_epoch=metadata['lr_scheduler_kwargs']['n_batches_per_epoch'],
        batch_size=batch_size,
    )
    pretrained_checkpoint_folder_path = metadata['model_kwargs']['pretrained_checkpoint_folder_path']
    if pretrained_checkpoint_folder_path is None:
        training_history = [training_instance]
    else:
        training_history = get_model_training_history(pretrained_checkpoint_folder_path) + [training_instance]
    _training_history_cache[model_checkpoint_folder_path] = training_history
    return training_history
    
def print_model_training_history_summary(model_checkpoint_folder_path, show_details=False):
    training_history = get_model_training_history(model_checkpoint_folder_path)
    num_training_runs = len(training_history)
    print(f'Number of training runs: {num_training_runs}')
    num_training_examples = 0
    for instance in training_history:
        num_training_examples += instance.batches_per_epoch * instance.batch_size * instance.best_epoch
    print(f'Number of training examples: {num_training_examples}')
    if show_details:
        print('--' * 30)
        for instance in training_history[::-1]:
            for k,v in instance._asdict().items():
                print(f'{k}: {v}')
            print('--' * 15)