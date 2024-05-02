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

INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH = os.environ['INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH']
INTERPRET_CXR_TEST_PUBLIC_CSV_PATH = os.environ['INTERPRET_CXR_TEST_PUBLIC_CSV_PATH']

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
