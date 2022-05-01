import os
from medvqa.datasets.qa import QA_Trainer
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR

class IUXRAY_QA_Trainer(QA_Trainer):

    def __init__(self, batch_size, collate_batch_fn, num_workers,
                preprocessed_data_filename, validation_only=False):
        preprocessing_save_path = os.path.join(IUXRAY_CACHE_DIR, preprocessed_data_filename)
        super().__init__(batch_size, collate_batch_fn, preprocessing_save_path, num_workers,
                         validation_only=validation_only)
