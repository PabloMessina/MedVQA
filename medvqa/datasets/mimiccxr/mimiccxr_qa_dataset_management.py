import os
from medvqa.datasets.qa import QA_Evaluator, QA_Trainer
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR

class MIMICCXR_QA_Trainer(QA_Trainer):

    def __init__(self, batch_size, collate_batch_fn, num_workers, preprocessed_data_filename):
        preprocessed_data_path = os.path.join(MIMICCXR_CACHE_DIR, preprocessed_data_filename)
        super().__init__(batch_size, collate_batch_fn, preprocessed_data_path, num_workers)

class MIMICCXR_QA_Evaluator(QA_Evaluator):

    def __init__(self, batch_size, collate_batch_fn, num_workers, preprocessed_data_filename):
        preprocessed_data_path = os.path.join(MIMICCXR_CACHE_DIR, preprocessed_data_filename)
        super().__init__(batch_size, collate_batch_fn, preprocessed_data_path, num_workers)