from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
import time
import os

SOURCE_DIR = os.environ['MEDVQA_SOURCE_DIR']
WORKSPACE_DIR = os.environ['MEDVQA_WORKSPACE_DIR']
FAST_WORKSPACE_DIR = os.environ['MEDVQA_FAST_WORKSPACE_DIR']
LARGE_FAST_WORKSPACE_DIR = os.environ['MEDVQA_LARGE_FAST_WORKSPACE_DIR']
CACHE_DIR = os.path.join(WORKSPACE_DIR, 'cache')
FAST_CACHE_DIR = os.path.join(FAST_WORKSPACE_DIR, 'cache')
LARGE_FAST_CACHE_DIR = os.path.join(LARGE_FAST_WORKSPACE_DIR, 'cache')
TMP_DIR = os.path.join(WORKSPACE_DIR, 'tmp')
FAST_TMP_DIR = os.path.join(FAST_WORKSPACE_DIR, 'tmp')
RESULTS_DIR = os.path.join(WORKSPACE_DIR, 'results')
REGULAR_EXPRESSIONS_FOLDER = os.path.join(SOURCE_DIR, 'medvqa', 'datasets', 'regular_expressions')

# NOTE: The following assumes that you have git cloned the YOLOv5 repo somewhere in your filesystem
# and have set the YOLOv5_PYTHON_PATH environment variable to an appropriate python executable.
# yolov5 is available at https://github.com/ultralytics/yolov5
YOLOv5_PYTHON_PATH = os.environ['YOLOv5_PYTHON_PATH']
YOLOv5_TRAIN_SCRIPT_PATH = os.environ['YOLOv5_TRAIN_SCRIPT_PATH']
YOLOv5_DETECT_SCRIPT_PATH = os.environ['YOLOv5_DETECT_SCRIPT_PATH']
YOLOv5_RUNS_DETECT_DIR = os.environ['YOLOv5_RUNS_DETECT_DIR']

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

def parsed_args_to_dict(args, verbose=True):
    # args = {k : v for k, v in vars(args).items() if v is not None}
    args = {k : v for k, v in vars(args).items()}
    if verbose:
        print('script\'s arguments:')
        for k, v in args.items():
            print(f'   {k}: {v}')
    return args

class DictWithDefault:
    def __init__(self, default, initial_values={}):
        self.values = initial_values
        self.default = default
    def __getitem__(self, key):
        return self.values.get(key, self.default)
    def __setitem__(self, key, value):
        self.values[key] = value
    def items(self):
        return self.values.items()
    
def activate_determinism(seed=42, verbose=True):
    if verbose:
        from medvqa.utils.logging import print_red
        print_red(f'Activating determinism(seed={seed})...', bold=True)
    import torch
    import random
    import numpy as np
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def deactivate_determinism():
    import torch
    import random
    import numpy as np
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True  # Enables faster training for some models
    torch.backends.cudnn.deterministic = False
    # Reset seeds using time-based randomness
    new_seed = int(time.time()) % (2**32 - 1)
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed_all(new_seed)
    random.seed(new_seed)
    np.random.seed(new_seed)

def print_nested_dict(d, indent=0):
    """
    Recursively prints a dictionary where:
    - Keys are printed as strings.
    - Values that are dictionaries are expanded.
    - Leaf values are replaced with their type.
    - Lists are printed as "list of {type} (len={len})" if they contain at least one item.
    
    Parameters:
    d (dict): The dictionary to print.
    indent (int): The current indentation level for nested structures.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + str(key) + ":")
            print_nested_dict(value, indent + 4)
        elif isinstance(value, list) and len(value) > 0:
            print(" " * indent + f"{key}: list of {type(value[0]).__name__} (len={len(value)})")
        else:
            print(" " * indent + f"{key}: {type(value).__name__}")