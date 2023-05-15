import os
from termcolor import colored
from medvqa.utils.files import make_dirs_in_filepath
from medvqa.utils.constants import CHEXPERT_LABELS

def _print_with_color(color, *args, bold=False):
    if bold:
        print(*[colored(x, color, attrs=['bold']) for x in args])
    else:
        print(*[colored(x, color) for x in args])

def print_blue(*args, bold=False):
    _print_with_color('blue', *args, bold=bold)

def print_red(*args, bold=False):
    _print_with_color('red', *args, bold=bold)

def print_magenta(*args, bold=False):
    _print_with_color('magenta', *args, bold=bold)

def print_orange(*args, bold=False):
    if bold:
        print(*[f'\033[93m\033[1m{x}\033[0m' for x in args])
    else:
        print(*[f'\033[93m{x}\033[0m' for x in args])

def print_bold(*args):
    print(*[colored(x, attrs=['bold']) for x in args])

def print_normal_and_bold(normal, bold):
    print(normal, end='')
    print_bold(bold)

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
        print(f'MetricsLogger :: we\'ll be logging to {self.metrics_logs_path}')
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