from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
import time
import os

SOURCE_DIR = os.environ['MEDVQA_SOURCE_DIR']
WORKSPACE_DIR = os.environ['MEDVQA_WORKSPACE_DIR']
CACHE_DIR = os.path.join(WORKSPACE_DIR, 'cache')
TMP_DIR = os.path.join(WORKSPACE_DIR, 'tmp')

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

