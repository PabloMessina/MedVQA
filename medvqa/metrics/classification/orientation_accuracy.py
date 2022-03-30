from ignite.exceptions import NotComputableError
from ignite.engine import Events

from medvqa.utils.constants import MIMICCXR_DATASET_ID
# import numbers

class DatasetAwareOrientationAccuracy:

    def __init__(self, record_scores=False):
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        if record_scores:
            self._scores = []
    
    def reset(self):
        self._acc_score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()

    def update(self, pred_orientations, gt_orientations, dataset_id):
        if dataset_id == MIMICCXR_DATASET_ID:
            self._update_score_ignoring_index(pred_orientations, gt_orientations, 0)
        else:
            self._update_score(pred_orientations, gt_orientations)
    
    def _update_score(self, pred_orientations, gt_orientations):
        n = pred_orientations.size(0)
        for i in range(n):
            gt = gt_orientations[i]
            pred = pred_orientations[i]
            score = (gt == pred).item() # 0 or 1
            # assert isinstance(score, numbers.Number)
            self._acc_score += score
            if self.record_scores:
                self._scores.append(score)
        self._count += n

    def _update_score_ignoring_index(self, pred_orientations, gt_orientations, index=0):
        n = pred_orientations.size(0)
        for i in range(n):
            gt = gt_orientations[i]
            pred = pred_orientations[i]
            score = (gt == pred).item() # 0 or 1
            if gt != index:
                self._acc_score += score
                self._count += 1
            else:
                score = None
            if self.record_scores:
                self._scores.append(score)
    
    def attach(self, engine, metric_alias):
        
        def epoch_started_handler(unused_engine):
            self.reset()

        def iteration_completed_handler(engine):
            output = engine.state.output
            pred_orientation = output['pred_orientation']
            orientation = output['orientation']
            dataset_id = output['dataset_id'] # make sure your step_fn returns this
            self.update(pred_orientation, orientation, dataset_id)

        def epoch_completed_handler(engine):
            engine.state.metrics[metric_alias] = self.compute()

        engine.add_event_handler(Events.EPOCH_STARTED, epoch_started_handler)
        engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_completed_handler)
        engine.add_event_handler(Events.EPOCH_COMPLETED, epoch_completed_handler)

    def compute(self):
        # print('************** DEBUG: orientation_accuracy ************')
        if self._count == 0:
            raise NotComputableError('DatasetAwareOrientationAccuracy needs at least one example before it can be computed.')
        if self.record_scores:
            # assert isinstance(self._scores[0], numbers.Number)
            # print('self._scores[0]=', self._scores[0])
            return self._scores
        return self._acc_score / self._count