from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
import time
import os
import logging
from enum import Enum
from typing import List, Any

logger = logging.getLogger(__name__)


def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

def parsed_args_to_dict(args, verbose=True):
    # args = {k : v for k, v in vars(args).items() if v is not None}
    args = {k : v for k, v in vars(args).items()}
    if verbose:
        string_to_log = '\nscript\'s arguments:'
        for k, v in args.items():
            string_to_log += f'\n   {k}: {v}'
        logger.info(string_to_log)
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
    
class ChoiceEnum(Enum):
    """
    Base Enum class that provides a method to get member values as a list.
    Useful for populating choices in argument parsers or similar scenarios.
    """
    @classmethod
    def get_choices(cls) -> List[Any]:
        """Returns a list of the values of the enum members."""
        return [member.value for member in cls]
    
def activate_determinism(seed=42, verbose=True):
    if verbose:
        from medvqa.utils.logging_utils import print_red
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


def inspect_available_ram(throw_warning_if_low=True):
    import psutil
    ram = psutil.virtual_memory()
    total_ram = ram.total / (1024 ** 3)  # Convert bytes to GB
    available_ram = ram.available / (1024 ** 3)  # Convert bytes to GB
    used_ram = ram.used / (1024 ** 3)  # Convert bytes to GB
    percent_used = ram.percent
    logger.info(f"Total RAM: {total_ram:.2f} GB")
    logger.info(f"Available RAM: {available_ram:.2f} GB")
    logger.info(f"Used RAM: {used_ram:.2f} GB ({percent_used}%)")
    if throw_warning_if_low and percent_used > 60:
        logger.warning("Warning: More than 60% of RAM is used. Consider closing some applications or increasing system memory.")
    return total_ram, available_ram, used_ram, percent_used