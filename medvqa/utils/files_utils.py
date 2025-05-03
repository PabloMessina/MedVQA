import json
import pickle
import os
import datetime
import logging
from tqdm import tqdm
from typing import Union, List, Optional
from pathlib import Path

from medvqa.utils.common import (
    WORKSPACE_DIR,
    get_timestamp,
)
from medvqa.utils.hashing_utils import hash_string

logger = logging.getLogger(__name__)

MAX_FILENAME_LENGTH = os.statvfs('/').f_namemax

_json_cache = dict()
_jsonl_cache = dict()
_pickle_cache = dict()
_csv_dataframe_cache = dict()

def get_cached_json_file(path, force_reload=False):
    if force_reload:
        file = _json_cache[path] = load_json(path)
    else:
        try:
            file = _json_cache[path]
        except KeyError:
            file = _json_cache[path] = load_json(path)
    return file

def get_cached_jsonl_file(path, force_reload=False):
    if force_reload:
        file = _jsonl_cache[path] = load_jsonl(path)
    else:
        try:
            file = _jsonl_cache[path]
        except KeyError:
            file = _jsonl_cache[path] = load_jsonl(path)
    return file

def get_cached_pickle_file(path, force_reload=False):
    if force_reload:
        file = _pickle_cache[path] = load_pickle(path)
    else:
        try:
            file = _pickle_cache[path]        
        except KeyError:
            file = None
        if file is None:
            file = _pickle_cache[path] = load_pickle(path)
    return file

def get_cached_dataframe_from_csv(path, force_reload=False):
    import pandas as pd
    if force_reload:
        file = _csv_dataframe_cache[path] = pd.read_csv(path)
    else:
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
    with open(path, 'rb') as f:
        return pickle.load(f)
    
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
    logger.info(f'Checking that all files and subdirectories in {dir_path_1} are also in {dir_path_2}...')
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
    logger.info(f'Checking that all files and subdirectories in {dir_path_2} are also in {dir_path_1}...')
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
    logger.info(f'In {dir_path_1} but not in {dir_path_2}: {len(in_dir1_not_in_dir2)}')
    logger.info(f'In {dir_path_2} but not in {dir_path_1}: {len(in_dir2_not_in_dir1)}')
    if len(in_dir1_not_in_dir2) == 0 and len(in_dir2_not_in_dir1) == 0:
        logger.info('No inconsistencies found!')
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

def print_file_size(filepath):
    size = os.path.getsize(filepath)
    size_kb = size / 1024
    size_mb = size_kb / 1024
    size_gb = size_mb / 1024
    logger.info(f'File size: {size} bytes ({size_kb:.2f} KB, {size_mb:.2f} MB, {size_gb:.2f} GB)')

def load_regex_from_files(txt_filepaths):
    if type(txt_filepaths) == str:
        txt_filepaths = [txt_filepaths]
    import re
    pattern = ''
    for txt_filepath in txt_filepaths:
        with open(txt_filepath, 'r') as f:
            for line in f.readlines():
                if len(pattern) > 0:
                    pattern += '|'
                pattern += f'({line.strip()})'
    return re.compile(pattern, re.IGNORECASE)

def load_class_specific_regex(dataset_name, class_name=None):
    from medvqa.utils.common import REGULAR_EXPRESSIONS_FOLDER
    class_name_to_regex_files = load_json(os.path.join(REGULAR_EXPRESSIONS_FOLDER, "unified_classes", f'class_name_to_regex_files.json'))
    if class_name is None:
        # Load all regexes for the dataset
        class_name_to_regex = dict()
        for class_name, filenames in class_name_to_regex_files[dataset_name].items():
            filepaths = [os.path.join(REGULAR_EXPRESSIONS_FOLDER, "unified_classes", filename) for filename in filenames]
            class_name_to_regex[class_name] = load_regex_from_files(filepaths)
        return class_name_to_regex
    else:
        filenames = class_name_to_regex_files[dataset_name][class_name]
        filepaths = [os.path.join(REGULAR_EXPRESSIONS_FOLDER, "unified_classes", filename) for filename in filenames]
        return load_regex_from_files(filepaths)


DEFAULT_IGNORE_LIST = [
    '.DS_Store',  # macOS
    'Thumbs.db',  # Windows
    '__pycache__',
    '.git',
    '.idea',
    '.vscode',
    '.pytest_cache',
    '.ipynb_checkpoints',
    'venv',
    'env',
    'node_modules',
    '__tmp',
]

def print_directory_tree(
    start_path: Union[str, Path],
    level: int = -1,
    prefix: str = '',
    ignore_list: Optional[List[str]] = None,
    ignore_extensions: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    include_files: bool = True,
    print_root: bool = True,
):
    """
    Prints a directory tree structure similar to the 'tree' command.

    Args:
        start_path (Union[str, Path]): The directory path to start from.
        level (int): Current recursion depth (internal use).
        prefix (str): String prefix for lines (internal use).
        ignore_list (Optional[List[str]]): List of directory/file names to ignore.
                                            Defaults to DEFAULT_IGNORE_LIST.
        ignore_extensions (Optional[List[str]]): List of file extensions to ignore.
        max_depth (Optional[int]): Maximum depth to traverse. None means infinite.
        include_files (bool): Whether to include files in the output. Defaults to True.
        print_root (bool): Whether to print the root directory name. Defaults to True.
    """
    if ignore_list is None:
        ignore_list = DEFAULT_IGNORE_LIST

    start_path = Path(start_path)
    if not start_path.is_dir():
        print(f"Error: '{start_path}' is not a valid directory.")
        return

    # --- Initial Call Setup ---
    if level == -1: # First call
        if print_root:
            print(f"{start_path.name}/")
        level = 0 # Start recursion depth count
        # Don't increment level here, do it in recursive call

    # --- Stop Condition for Depth ---
    if max_depth is not None and level >= max_depth:
        return

    # --- Get and Filter Contents ---
    try:
        # Use list comprehension for potentially large directories
        contents = [
            item for item in start_path.iterdir()
            if item.name not in ignore_list
            and (include_files or item.is_dir())
            and (
                item.is_dir() or
                item.suffix not in ignore_extensions
            )
        ]
        # Sort alphabetically, directories might naturally come first depending on OS/locale
        contents.sort(key=lambda x: x.name)
    except PermissionError:
        print(f"{prefix}├── [Permission Denied]")
        return
    except OSError as e:
        print(f"{prefix}├── [Error Reading Directory: {e}]")
        return

    # --- Iterate and Print ---
    pointers = ['├── '] * (len(contents) - 1) + ['└── ']

    for pointer, item in zip(pointers, contents):
        # Print current item
        if item.is_dir():
            print(f"{prefix}{pointer}{item.name}/")
            # Prepare prefix for the next level
            extension = '│   ' if pointer == '├── ' else '    '
            # Recursive call
            print_directory_tree(
                item,
                level=level + 1, # Increment level for sub-directory
                prefix=prefix + extension,
                ignore_list=ignore_list,
                ignore_extensions=ignore_extensions,
                max_depth=max_depth,
                include_files=include_files,
                print_root=False # Don't print root name in recursive calls
            )
        elif include_files: # Only print files if requested
            print(f"{prefix}{pointer}{item.name}")