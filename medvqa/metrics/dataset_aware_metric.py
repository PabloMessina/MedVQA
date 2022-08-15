import warnings
import torch
from ignite.engine import Events
from ignite.exceptions import NotComputableError

class DatasetAwareMetric:

    def __init__(self, output_transform, allowed_dataset_ids):
        self.allowed_dataset_ids = allowed_dataset_ids
        self.output_transform = output_transform
    
    def reset(self):
        raise NotImplementedError('Make sure your specialized class implements this function')

    def update(self, *args):
        raise NotImplementedError('Make sure your specialized class implements this function')
    
    def compute(self):
        raise NotImplementedError('Make sure your specialized class implements this function')
    
    def attach(self, engine, metric_alias):
        
        def epoch_started_handler(unused_engine):
            self.reset()

        def iteration_completed_handler(engine):
            output = engine.state.output
            dataset_id = output['dataset_id'] # make sure your step_fn returns this
            if dataset_id in self.allowed_dataset_ids:
                self.update(self.output_transform(output))

        def epoch_completed_handler(engine):
            engine.state.metrics[metric_alias] = self.compute()

        engine.add_event_handler(Events.EPOCH_STARTED, epoch_started_handler)
        engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_completed_handler)
        engine.add_event_handler(Events.EPOCH_COMPLETED, epoch_completed_handler)


class DatasetAwareEpochMetric(DatasetAwareMetric):

    def __init__(
        self,
        compute_fn,
        output_transform,
        allowed_dataset_ids,
        check_compute_fn = True,
        device = torch.device("cpu"),
    ):
        if not callable(compute_fn):
            raise TypeError("Argument compute_fn should be callable.")

        self.compute_fn = compute_fn
        self._check_compute_fn = check_compute_fn
        if type(device) is str: device = torch.device(device)
        self._device = device
        super().__init__(output_transform, allowed_dataset_ids)
    
    def reset(self):
        self._predictions = []
        self._targets = []

    def _check_shape(self, output):
        y_pred, y = output
        if y_pred.ndimension() not in (1, 2):
            raise ValueError("Predictions should be of shape (batch_size, n_targets) or (batch_size, ).")

        if y.ndimension() not in (1, 2):
            raise ValueError("Targets should be of shape (batch_size, n_targets) or (batch_size, ).")

    def _check_type(self, output):
        y_pred, y = output
        if len(self._predictions) < 1:
            return
        dtype_preds = self._predictions[-1].dtype
        if dtype_preds != y_pred.dtype:
            raise ValueError(
                f"Incoherent types between input y_pred and stored predictions: {dtype_preds} vs {y_pred.dtype}"
            )

        dtype_targets = self._targets[-1].dtype
        if dtype_targets != y.dtype:
            raise ValueError(f"Incoherent types between input y and stored targets: {dtype_targets} vs {y.dtype}")

    def update(self, output):
        self._check_shape(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        y_pred = y_pred.clone().to(self._device)
        y = y.clone().to(self._device)

        self._check_type((y_pred, y))
        self._predictions.append(y_pred)
        self._targets.append(y)

        # Check once the signature and execution of compute_fn
        if len(self._predictions) == 1 and self._check_compute_fn:
            try:
                self.compute_fn(self._predictions[0], self._targets[0])
            except Exception as e:
                warnings.warn(f"Probably, there can be a problem with `compute_fn`:\n {e}.", EpochMetricWarning)

    def compute(self):
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError("EpochMetric must have at least one example before it can be computed.")
        
        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)
        result = self.compute_fn(_prediction_tensor, _target_tensor)
        
        return result

class EpochMetricWarning(UserWarning):
    pass