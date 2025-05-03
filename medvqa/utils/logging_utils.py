import os
import logging
import sys
import os
import colorlog
from termcolor import colored
from medvqa.utils.files_utils import make_dirs_in_filepath
from medvqa.utils.constants import CHEXPERT_LABELS

logger = logging.getLogger(__name__)

# Keep the standard format for files or as a base
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# Define the color log format
COLOR_LOG_FORMAT = (
    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s"
    " - %(message_log_color)s%(message)s" # Color the message too
)
LOG_LEVEL = logging.INFO # Default level

# --- ANSI Escape Codes ---
ANSI_BLUE_BOLD = "\033[1;34m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA_BOLD = "\033[1;35m"
ANSI_BLACK_BOLD = "\033[1;30m"
ANSI_RED_BOLD = "\033[1;31m"
ANSI_BOLD = "\033[1m"
ANSI_DARK_GREEN_BOLD = "\033[1;32m"
ANSI_RESET = "\033[0m"
# --- End ANSI Codes ---

def setup_logging(
    log_level=LOG_LEVEL,
    log_format=LOG_FORMAT,
    color_log_format=COLOR_LOG_FORMAT, # Add color format
    use_console=True,
    use_color=True, # Flag to enable/disable color
):
    """
    Configures the root logger with optional color for console output.

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
            # Use colorlog's ColoredFormatter for the console
            console_formatter = colorlog.ColoredFormatter(
                color_log_format,
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                },
                secondary_log_colors={
                    'message': { # Optional: color parts of the message itself
                        'ERROR':    'red',
                        'CRITICAL': 'red'
                    }
                },
                style='%' # Use %-style formatting
            )
        else:
            # Use standard formatter if color is disabled
            console_formatter = logging.Formatter(log_format)

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
        self.metrics_logs_path = os.path.join(checkpoint_folder, 'metrics_logs.csv')
        logger.info(f'MetricsLogger: we\'ll be logging to {self.metrics_logs_path}')
        if not os.path.exists(self.metrics_logs_path):
            make_dirs_in_filepath(self.metrics_logs_path)            
    
    def log_metrics(self, metric_names, scores):
        if not os.path.exists(self.metrics_logs_path):
            with open(self.metrics_logs_path, 'w') as f:
                f.write(','.join(metric_names) + '\n')        
        with open(self.metrics_logs_path, 'a') as f:
            f.write(','.join(f'{s:.5f}' if s is not None else '' for s in scores) + '\n')


def chexpert_label_array_to_string(label_array):
    return ', '.join(CHEXPERT_LABELS[i] for i, label in enumerate(label_array) if label == 1)

def chest_imagenome_label_array_to_string(label_array, label_names):
    return '\n'.join(f'({", ".join(label_names[i])})' for i, label in enumerate(label_array) if label == 1)

def question_label_array_to_string(questions, label_array):
    assert len(questions) == len(label_array)
    return ', '.join(questions[i] for i, label in enumerate(label_array) if label == 1)