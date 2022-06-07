import os
from medvqa.utils.files import make_dirs_in_filepath

class CountPrinter:
    def __init__(self):
        self.count = 1
    def __call__(self, *args):
        print(f'{self.count}) ', end='')
        print(*args)
        self.count += 1

class MetricsLogger:
    def __init__(self, checkpoint_folder, metrics_names):
        self.metrics_logs_path = os.path.join(checkpoint_folder, 'metrics_logs.csv')
        self.metrics_names = metrics_names
        print(f'MetricsLogger :: we\'ll be logging to {self.metrics_logs_path}')
        if not os.path.exists(self.metrics_logs_path):
            make_dirs_in_filepath(self.metrics_logs_path)
            with open(self.metrics_logs_path, 'w') as f:
                f.write(','.join(metrics_names) + '\n')
    
    def log_metrics(self, scores):
        with open(self.metrics_logs_path, 'a+') as f:
            f.write(','.join(f'{s:.5f}' if s is not None else '' for s in scores) + '\n')
