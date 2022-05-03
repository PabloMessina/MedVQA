import json
import pickle
import os

from medvqa.utils.common import (
    WORKSPACE_DIR,
    get_timestamp,
)

MAX_FILENAME_LENGTH = os.statvfs('/').f_namemax

_json_cache = dict()
_pickle_cache = dict()

def get_cached_json_file(path):
    try:
        file = _json_cache[path]
    except KeyError:
        file = _json_cache[path] = load_json_file(path)
    return file

def get_cached_pickle_file(path):
    try:
        file = _pickle_cache[path]        
    except KeyError:
        file = None
    if file is None:
        file = _pickle_cache[path] = load_pickle(path)
    return file

def load_json_file(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    
def read_lines_from_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    clean_lines = [None] * len(lines)
    i = 0
    for line in lines:
        line = line.strip()
        if line:
            clean_lines[i] = line
            i += 1
    clean_lines = clean_lines[:i]
    return clean_lines

def make_dirs_in_filepath(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def save_to_pickle(obj, path):
    make_dirs_in_filepath(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def save_to_json(obj, path):
    make_dirs_in_filepath(path)
    with open(path, 'w') as f:
        json.dump(obj, f)

def get_checkpoint_folder_path(task, dataset_name, model_name, *args):
    timestamp = get_timestamp()
    folder_name = f'{timestamp}_{dataset_name}_{model_name}'
    if args: folder_name = f'{folder_name}_{"_".join(arg for arg in args if arg is not None)}'
    full_path = os.path.join(WORKSPACE_DIR, 'models', task, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def get_results_folder_path(checkpoint_folder_path):
    results_folder_path = checkpoint_folder_path.replace(f'{os.path.sep}models{os.path.sep}',
                                                         f'{os.path.sep}results{os.path.sep}')
    return results_folder_path
