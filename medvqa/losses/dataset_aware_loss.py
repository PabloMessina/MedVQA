from medvqa.losses.condition_aware_loss import ConditionAwareLoss

class DatasetAwareLoss(ConditionAwareLoss):

    def __init__(self, output_transform, allowed_dataset_ids):
        condition_function = lambda output: output['dataset_id'] in allowed_dataset_ids
        super().__init__(output_transform, condition_function)