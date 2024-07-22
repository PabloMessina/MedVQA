import json
import pickle
import os
import datetime
from tqdm import tqdm

from medvqa.utils.common import (
    WORKSPACE_DIR,
    get_timestamp,
)
from medvqa.utils.hashing import hash_string

MAX_FILENAME_LENGTH = os.statvfs('/').f_namemax

_json_cache = dict()
_jsonl_cache = dict()
_pickle_cache = dict()
_csv_dataframe_cache = dict()

def get_cached_json_file(path):
    try:
        file = _json_cache[path]
    except KeyError:
        file = _json_cache[path] = load_json(path)
    return file

def get_cached_jsonl_file(path):
    try:
        file = _jsonl_cache[path]
    except KeyError:
        file = _jsonl_cache[path] = load_jsonl(path)
    return file

def get_cached_pickle_file(path):
    try:
        file = _pickle_cache[path]        
    except KeyError:
        file = None
    if file is None:
        file = _pickle_cache[path] = load_pickle(path)
    return file

def get_cached_dataframe_from_csv(path):
    import pandas as pd
    try:
        file = _csv_dataframe_cache[path]
    except KeyError:
        file = None
    if file is None:
        file = _csv_dataframe_cache[path] = pd.read_csv(path)
    return file

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_jsonl(path):
    assert os.path.exists(path)
    assert os.path.isfile(path)
    assert path.endswith('.jsonl')
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

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

def read_txt(path):
    with open(path, 'r') as f:
        return f.read()

def make_dirs_in_filepath(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def save_pickle(obj, path, add_to_cache=False):
    make_dirs_in_filepath(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    if add_to_cache:
        _pickle_cache[path] = obj

def save_json(obj, path):
    make_dirs_in_filepath(path)
    with open(path, 'w') as f:
        json.dump(obj, f)

def save_jsonl(obj_list, path, append=False):
    assert isinstance(obj_list, list)
    make_dirs_in_filepath(path)
    mode = 'a' if append else 'w'
    with open(path, mode) as f:
        for obj in obj_list:
            json_string = json.dumps(obj)
            f.write(json_string + "\n")

def save_txt(strings_list, path):
    make_dirs_in_filepath(path)
    with open(path, 'w') as f:
        for s in strings_list:
            f.write(f'{s}\n')

def get_checkpoint_folder_path(task, dataset_name, model_name, *args):
    timestamp = get_timestamp()
    folder_name = f'{timestamp}_{dataset_name}_{model_name}'
    if args: 
        args = [arg for arg in args if arg is not None]
        if len(args) > 0:
            folder_name = f'{folder_name}_{"_".join(args)}'
    folder_name = folder_name.replace(os.path.sep, '-') # prevent path-related bugs
    full_path = os.path.join(WORKSPACE_DIR, 'models', task, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def get_results_folder_path(checkpoint_folder_path, create_if_not_exists=True):
    results_folder_path = checkpoint_folder_path.replace(f'{os.path.sep}models{os.path.sep}',
                                                         f'{os.path.sep}results{os.path.sep}')
    if create_if_not_exists:
        os.makedirs(results_folder_path, exist_ok=True)
    return results_folder_path

def get_file_path_with_hashing_if_too_long(folder_path, prefix, strings=[], ext='pkl', force_hashing=False):
    assert os.path.sep not in prefix # prevent path-related bugs
    clean_strings = []
    for s in strings:
        if s is None:
            continue
        if os.path.sep in s:
            s = s.replace(os.path.sep, '_') # prevent path-related bugs
        clean_strings.append(s)
    strings = clean_strings
    if strings:
        file_path = os.path.join(folder_path, f'{prefix}({";".join(strings)}).{ext}')
    else:
        file_path = os.path.join(folder_path, f'{prefix}.{ext}')
    if len(file_path) > MAX_FILENAME_LENGTH or force_hashing:
        h = hash_string(file_path)
        file_path = os.path.join(folder_path, f'{prefix}(hash={h[0]},{h[1]}).{ext}')
    return file_path

def find_inconsistencies_between_directories(dir_path_1, dir_path_2):
    # 1) Check that all files and subdirectories in dir_path_1 are also in dir_path_2
    print(f'Checking that all files and subdirectories in {dir_path_1} are also in {dir_path_2}...')
    in_dir1_not_in_dir2 = []
    for root, dirs, files in tqdm(os.walk(dir_path_1)):
        for f in files:
            path_1 = os.path.join(root, f)
            path_2 = path_1.replace(dir_path_1, dir_path_2)
            if not os.path.exists(path_2):
                in_dir1_not_in_dir2.append(path_1)
        for d in dirs:
            path_1 = os.path.join(root, d)
            path_2 = path_1.replace(dir_path_1, dir_path_2)
            if not os.path.exists(path_2):
                in_dir1_not_in_dir2.append(path_1)
    # 2) Check that all files and subdirectories in dir_path_2 are also in dir_path_1
    print(f'Checking that all files and subdirectories in {dir_path_2} are also in {dir_path_1}...')
    in_dir2_not_in_dir1 = []
    for root, dirs, files in tqdm(os.walk(dir_path_2)):
        for f in files:
            path_2 = os.path.join(root, f)
            path_1 = path_2.replace(dir_path_2, dir_path_1)
            if not os.path.exists(path_1):
                in_dir2_not_in_dir1.append(path_2)
        for d in dirs:
            path_2 = os.path.join(root, d)
            path_1 = path_2.replace(dir_path_2, dir_path_1)
            if not os.path.exists(path_1):
                in_dir2_not_in_dir1.append(path_2)
    # 3) Print a summary
    print(f'In {dir_path_1} but not in {dir_path_2}: {len(in_dir1_not_in_dir2)}')
    print(f'In {dir_path_2} but not in {dir_path_1}: {len(in_dir2_not_in_dir1)}')
    if len(in_dir1_not_in_dir2) == 0 and len(in_dir2_not_in_dir1) == 0:
        print('No inconsistencies found!')
    # 4) Return the results
    return {
        'in_dir1_not_in_dir2': in_dir1_not_in_dir2,
        'in_dir2_not_in_dir1': in_dir2_not_in_dir1,
    }

def zip_files_and_folders_in_dir(dir_path, file_or_folder_names, output_path):
    import zipfile
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for name in file_or_folder_names:
            path = os.path.join(dir_path, name)
            if os.path.isfile(path):
                zipf.write(path, arcname=name)
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for f in files:
                        file_path = os.path.join(root, f)
                        arcname = os.path.join(name, file_path.replace(path, ''))
                        zipf.write(file_path, arcname=arcname)

def list_filepaths_with_prefix_and_timestamps(path_prefix, must_contain=None):
    if type(must_contain) == str:
        must_contain = [must_contain]
    matching_files = []
    directory = os.path.dirname(path_prefix)
    for root, _, files in os.walk(directory):
        for filename in files:
            full_path = os.path.join(root, filename)
            if full_path.startswith(path_prefix):
                if must_contain is not None:
                    if not all(s in full_path for s in must_contain):
                        continue
                creation_timestamp = os.path.getctime(full_path)
                timestamp_human_readable = datetime.datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                matching_files.append((full_path, timestamp_human_readable))
    matching_files.sort(key=lambda x:x[1], reverse=True)
    return matching_files