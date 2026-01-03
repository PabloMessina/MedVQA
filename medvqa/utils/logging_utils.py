import os
import logging
import sys
import json
import colorlog
import glob
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from termcolor import colored
from medvqa.utils.files_utils import make_dirs_in_filepath
from medvqa.utils.constants import CHEXPERT_LABELS

logger = logging.getLogger(__name__)


# Define the color log format with alignment
# %-8s ensures the level name (INFO, DEBUG) takes up 8 spaces for alignment
COLOR_LOG_FORMAT = (
    "%(log_color)s%(asctime)s%(reset)s | "
    "%(log_color)s%(levelname)-8s%(reset)s | "
    "%(log_color)s%(name)s%(reset)s: "
    "%(message_log_color)s%(message)s"
)

# Standard format for files (clean, aligned, no color codes)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"

# --- ANSI Escape Codes ---
ANSI_BLUE_BOLD = "\033[1;34m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA_BOLD = "\033[1;35m"
ANSI_BLACK_BOLD = "\033[1;30m"
ANSI_RED_BOLD = "\033[1;31m"
ANSI_BOLD = "\033[1m"
ANSI_DARK_GREEN_BOLD = "\033[1;32m"
ANSI_ORANGE_BOLD = "\033[1;33m"
ANSI_RESET = "\033[0m"
# --- End ANSI Codes ---

LOG_LEVEL = logging.INFO # Default level


def setup_logging(
    log_level=LOG_LEVEL,
    log_format=LOG_FORMAT,
    color_log_format=COLOR_LOG_FORMAT,
    use_console=True,
    use_color=True,
    date_format="%Y-%m-%d %H:%M:%S" # Added cleaner date format
):
    """
    Configures the root logger with aligned, colored output.

    Args:
        log_level: The minimum logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Path to the file for logging. If None, only console logging is used (if enabled).
        log_format: The format string for the file log handler.
        color_log_format: The format string for the colored console log handler.
        use_console: Whether to log to the console (stderr).
        use_color: Whether to use color for console output.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handlers = []

    # Console Handler
    if use_console:
        if use_color:
            console_formatter = colorlog.ColoredFormatter(
                color_log_format,
                datefmt=date_format,
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                },
                secondary_log_colors={
                    'message': {
                        'ERROR':    'red',
                        'CRITICAL': 'red'
                    }
                },
                style='%'
            )
        else:
            console_formatter = logging.Formatter(log_format, datefmt=date_format)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    if not handlers:
        logger.addHandler(logging.NullHandler())
    else:
        for handler in handlers:
            logger.addHandler(handler)

    # Don't log the configuration message if no handlers were added
    if handlers:
        logging.info(f"Logging configured (Color: {use_color if use_console else 'N/A'}).")


def _print_with_color(color, *args, bold=False, end='\n'):
    if bold:
        print(*[colored(x, color, attrs=['bold']) for x in args], end=end)
    else:
        print(*[colored(x, color) for x in args], end=end)


def print_blue(*args, bold=False, end='\n'):
    _print_with_color('blue', *args, bold=bold, end=end)


def print_red(*args, bold=False, end='\n'):
    _print_with_color('red', *args, bold=bold, end=end)


def print_magenta(*args, bold=False, end='\n'):
    _print_with_color('magenta', *args, bold=bold, end=end)


def print_green(*args, bold=False, end='\n'):
    _print_with_color('green', *args, bold=bold, end=end)


def print_orange(*args, bold=False, end='\n'):
    if bold:
        print(*[f'\033[93m\033[1m{x}\033[0m' for x in args], end=end)
    else:
        print(*[f'\033[93m{x}\033[0m' for x in args], end=end)


def print_bold(*args, end='\n'):
    print(*[colored(x, attrs=['bold']) for x in args], end=end)


def print_normal_and_bold(normal, bold):
    print(normal, end='')
    print_bold(bold)


def rgba_to_ansi(rgba):
    """
    Convert RGBA to ANSI color code.
    Args:
        rgba (tuple): RGBA color value (0-1 range).
    Returns:
        str: ANSI color code.
    """
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    return f'\033[38;2;{r};{g};{b}m'


def log_title(logger, title, color='blue', bold=True):
    """
    Log a title with a specific color and bold formatting.
    Args:
        logger (logging.Logger): Logger instance.
        title (str): Title to log.
        color (str): Color for the title.
        bold (bool): Whether to make the title bold.
    """
    wrapped_title = f'{"=" * 10} {title} {"=" * 10}'
    if bold:
        logger.info(colored(wrapped_title, color, attrs=['bold']))
    else:
        logger.info(colored(wrapped_title, color))


class CountPrinter:
    def __init__(self, color='blue', bold=True):
        self.count = 1
        self.color = color
        self.bold = bold
    def __call__(self, *args):
        if self.bold:
            print(colored('-' * 50, self.color, attrs=['bold']))
            print(colored(f'{self.count}) ', self.color, attrs=['bold']), end='')
            print(*[colored(x, self.color, attrs=['bold']) for x in args])
        else:
            print(colored('-' * 50, self.color))
            print(colored(f'{self.count}) ', self.color), end='')
            print(*[colored(x, self.color) for x in args])
        self.count += 1


class MetricsLogger:
    def __init__(self, checkpoint_folder):
        self.metrics_logs_path = os.path.join(checkpoint_folder, 'metrics_logs.jsonl')
        logger.info(f"MetricsLogger: we'll be logging to {self.metrics_logs_path}")
        
        if not os.path.exists(self.metrics_logs_path):
            make_dirs_in_filepath(self.metrics_logs_path)
            
    def _json_serializer(self, obj):
        """Helper to convert objects to JSON-serializable types."""
        if hasattr(obj, 'item'):
            return obj.item() # Handles numpy scalars and torch scalars
        if isinstance(obj, (list, tuple)):
            return [self._json_serializer(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._json_serializer(v) for k, v in obj.items()}
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def log_metrics(self, metric_names, scores, split, epoch):
        record = {
            "epoch": epoch,
            "split": split,
        }
        
        metrics_dict = dict(zip(metric_names, scores))
        record.update(metrics_dict)

        with open(self.metrics_logs_path, 'a') as f:
            # Pass the static method reference here
            f.write(json.dumps(record, default=self._json_serializer) + '\n')


def chexpert_label_array_to_string(label_array):
    return ', '.join(CHEXPERT_LABELS[i] for i, label in enumerate(label_array) if label == 1)


def chest_imagenome_label_array_to_string(label_array, label_names):
    return '\n'.join(f'({", ".join(label_names[i])})' for i, label in enumerate(label_array) if label == 1)


def question_label_array_to_string(questions, label_array):
    assert len(questions) == len(label_array)
    return ', '.join(questions[i] for i, label in enumerate(label_array) if label == 1)

def print_first_list_item(lst: list, indent: int = 0):
    first_item = lst[0]
    if isinstance(first_item, np.ndarray):
        print(' ' * indent + f'first item: numpy array (shape={first_item.shape})')
    elif isinstance(first_item, list):
        print(' ' * indent + f'first item: list of {type(first_item[0]).__name__} (len={len(first_item)})')
    elif isinstance(first_item, tuple):
        print(' ' * indent + f'first item: tuple of {type(first_item[0]).__name__} (len={len(first_item)})')
    print(' ' * indent + f'first item: {first_item}')

def print_nested_dict(d: dict, indent: int = 0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(' ' * indent + f'{k}:')
            print_nested_dict(v, indent + 2)
        elif isinstance(v, list) and len(v) > 0:
            print(' ' * indent + f'{k}: list of {type(v[0]).__name__} (len={len(v)})')
            print_first_list_item(v, indent + 2)
        elif isinstance(v, tuple) and len(v) > 0:
            print(' ' * indent + f'{k}: tuple of {type(v[0]).__name__} (len={len(v)})')
            print_first_list_item(v, indent + 2)
        elif isinstance(v, np.ndarray):
            print(' ' * indent + f'{k}: numpy array (shape={v.shape})')
        else:
            print(' ' * indent + f'{k}: {v}')


# =============================================================================
# Wandb Initialization/Finalization
# =============================================================================

# Try importing wandb, but allow the code to run without it if not used
try:
    import wandb
    # Check if a run is active (wandb.run is None if not initialized)
    WANDB_ACTIVE = wandb.run is not None
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None # Define wandb as None if import fails
    WANDB_ACTIVE = False
    WANDB_AVAILABLE = False

def find_latest_wandb_run_id(wandb_dir):
    """
    Finds the most recent wandb run ID in the given directory.
    Returns None if not found.
    """
    run_dirs = glob.glob(os.path.join(wandb_dir, "run-*"))
    if not run_dirs:
        return None
    # Sort by modification time, descending
    run_dirs = sorted(run_dirs, key=os.path.getmtime, reverse=True)
    latest_run_dir = run_dirs[0]
    # Extract run_id from directory name: run-YYYYMMDD_HHMMSS-<run_id>
    run_id = latest_run_dir.split("-")[-1]
    return run_id
    
def initialize_wandb(
    experiment_dir: Union[str, os.PathLike],
    wandb_config: Dict[str, Any],
    full_experiment_config: Dict[str, Any],
    resume_if_possible: bool = True,
    **kwargs # Allow passing extra args to wandb.init
) -> Optional[Any]: # Returns wandb.Run object or None
    """
    Initializes a Weights & Biases run.
    Args:
        experiment_dir: Directory where the wandb run will be stored.
        wandb_config: Wandb configuration dictionary.
        full_experiment_config: Full experiment configuration dictionary.
        resume_if_possible: If True, attempts to resume the latest run if available.
        **kwargs: Additional arguments to pass to wandb.init.
    Returns:
        wandb_run: The initialized wandb run object, or None if wandb is not available.
    """
    experiment_dir = Path(experiment_dir) # Ensure experiment_dir is a Path object
    use_wandb = wandb_config.get("enabled", False)
    if not use_wandb:
        logger.info("Weights & Biases tracking is disabled in the configuration.")
        return None

    if not WANDB_AVAILABLE:
        logger.warning("Wandb tracking is enabled, but the 'wandb' library is not installed. Skipping.")
        return None

    try:
        project = wandb_config.get("project")
        if project is None:
            raise ValueError("Wandb project name is required in the configuration.")
        entity = wandb_config.get("entity")
        run_name = wandb_config.get("run_name")
        if run_name is None:
            raise ValueError("Wandb run name is required in the configuration.")
        notes = wandb_config.get("notes")
        tags = wandb_config.get("tags")
        epochwise_metrics = wandb_config.get("epochwise_metrics")
        stepwise_metrics = wandb_config.get("stepwise_metrics")

        # --- Resume logic ---
        run_id = None
        resume = None
        if resume_if_possible:
            run_id = find_latest_wandb_run_id(experiment_dir / "wandb")
            if run_id:
                logger.info(f"Found previous wandb run with id: {run_id}. Will attempt to resume.")
                resume = "allow"

        logger.info(f"Initializing Wandb run: project='{project}', entity='{entity}', name='{run_name}'")
        wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=full_experiment_config,
            dir=experiment_dir,
            notes=notes,
            tags=tags,
            id=run_id,
            resume=resume,
            **kwargs
        )

        # --- Set up metrics with prefixes 'train/' and 'val/' ---
        splits = ['train', 'val']
        if epochwise_metrics is not None:
            logger.info(f"Defining epoch-wise metrics: {epochwise_metrics}")
            wandb_run.define_metric("epoch")
            for metric in epochwise_metrics:
                # 1. Define the base metric (just in case)
                wandb_run.define_metric(metric, step_metric="epoch")
                # 2. Define the prefixed metrics
                for split in splits:
                    wandb_run.define_metric(f"{split}/{metric}", step_metric="epoch")
        if stepwise_metrics is not None:
            logger.info(f"Defining step-wise metrics: {stepwise_metrics}")
            wandb_run.define_metric("step")
            for metric in stepwise_metrics:
                # 1. Define the base metric (just in case)
                wandb_run.define_metric(metric, step_metric="step")
                # 2. Define the prefixed metrics
                for split in splits:
                    wandb_run.define_metric(f"{split}/{metric}", step_metric="step")

        global WANDB_ACTIVE
        WANDB_ACTIVE = True
        logger.info(f"Wandb run initialized. Run page: {wandb_run.url}")
        return wandb_run

    except Exception as e:
        logger.error(f"Failed to initialize Wandb: {e}", exc_info=True)
        WANDB_ACTIVE = False
        return None

def finalize_wandb(wandb_run: Optional[Any]) -> None:
    """Finalizes the current Weights & Biases run, if active."""
    global WANDB_ACTIVE
    if wandb_run is not None and WANDB_ACTIVE and WANDB_AVAILABLE:
        try:
            wandb.finish()
            logger.info("Wandb run finished.")
            WANDB_ACTIVE = False
        except Exception as e:
            logger.error(f"Error finishing Wandb run: {e}", exc_info=True)
    elif WANDB_ACTIVE:
        logger.warning("Attempted to finalize wandb, but no valid run object or library was found.")
        WANDB_ACTIVE = False # Ensure flag is reset


# =============================================================================
# Weights & Biases Metric Logging Function
# =============================================================================

def log_metrics_to_wandb(
    metrics_dict: Dict[str, Any],
    step: int,
    step_metric: str = "step",
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Logs metrics to Weights & Biases.
    Args:
        metrics_dict: Dictionary where keys are metric names (e.g., 'train/loss')
                      and values are the corresponding scores or data. Values
                      should ideally be numbers or simple types wandb can handle.
        step: The current step (e.g., epoch number, global batch step) to associate
              with the metrics in wandb.
        step_metric: The name of the step metric (default is "step").
        wandb_run: The active wandb run object. If None, wandb is not used.
    """
    if wandb_run is None:
        return
    if WANDB_AVAILABLE and hasattr(wandb_run, "log"):
        try:
            wandb_run.log({step_metric: step, **metrics_dict})
        except Exception as e:
            logger.error(
                f"Failed to log metrics to wandb at step {step}: {e}", exc_info=True
            )
    else:
        if not WANDB_AVAILABLE:
            logger.warning(
                f"Wandb run object provided for step {step}, but wandb library not installed. Skipping wandb log."
            )
        elif not hasattr(wandb_run, "log"):
            logger.warning(
                f"Object provided as wandb_run for step {step} lacks .log method. Skipping wandb log."
            )