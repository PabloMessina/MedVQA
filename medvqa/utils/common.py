from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
import time
import os

SOURCE_DIR = os.environ['MEDVQA_SOURCE_DIR']
WORKSPACE_DIR = os.environ['MEDVQA_WORKSPACE_DIR']
CACHE_DIR = os.path.join(WORKSPACE_DIR, 'cache')

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')