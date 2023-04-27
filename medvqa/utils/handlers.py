from ignite.handlers import Checkpoint, DiskSaver
from ignite.engine import Events
from torch import Tensor

from termcolor import colored
import operator

from medvqa.losses.schedulers import LRSchedulerNames
from medvqa.utils.constants import METRIC2SHORT, MetricNames
from medvqa.utils.metrics import average_ignoring_nones_and_nans
from medvqa.utils.logging import MetricsLogger

def get_log_epoch_started_handler(model_wrapper):
    epoch_offset = model_wrapper.get_epoch()
    def handler(trainer):
        epoch = epoch_offset + trainer.state.epoch
        max_epochs = epoch_offset + trainer.state.max_epochs
        print(colored(f'---- Epoch {epoch}/{max_epochs}', attrs=['bold']))
        model_wrapper.set_epoch(epoch)
    return handler

def get_log_iteration_handler(log_every=25):

    def handler(engine):
        i = engine.state.iteration
        if i % log_every == 0:
            print(f'   iteration {i}', end='\r')

    return handler

def get_log_metrics_handler(timer, metrics_to_print, log_to_disk=False, checkpoint_folder=None):

    if log_to_disk:
        assert checkpoint_folder is not None
        metrics_logger = MetricsLogger(checkpoint_folder)

    def handler(engine):
        metrics = engine.state.metrics        
        scores = []
        metric_names = []
        for m in metrics_to_print:
            score = metrics.get(m, None)
            if m == MetricNames.BLEU:
                for k in range(0, 4):
                    name_k = f'bleu-{k+1}'
                    metric_names.append(name_k)
                    if score is None:
                        scores.append(None)
                if score is not None:
                    assert len(score) == 4 or (len(score) == 2 and len(score[0]) == 4)
                    if len(score) == 2:
                        score = score[0]
                    for k in range(0, 4):
                        score_k = score[k]
                        scores.append(score_k)
            elif m == MetricNames.CIDER_D:
                metric_names.append(m)
                if score is not None:
                    if type(score) is tuple:
                        assert len(score) == 2
                        score = score[0]
                scores.append(score)
            elif m == MetricNames.CHXLABEL_PRF1:
                metric_names.append(MetricNames.CHXLABELMACROAVGF1)
                metric_names.append(MetricNames.CHXLABELMICROAVGF1)
                if score is not None:
                    scores.append(score['f1_macro_avg'])
                    scores.append(score['f1_micro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            elif m == MetricNames.CHESTIMAGENOMELABEL_PRF1:
                metric_names.append(MetricNames.CHESTIMAGENOMELABELMACROAVGF1)
                metric_names.append(MetricNames.CHESTIMAGENOMELABELMICROAVGF1)
                if score is not None:
                    scores.append(score['f1_macro_avg'])
                    scores.append(score['f1_micro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            elif m == MetricNames.QLABELS_PRF1:
                metric_names.append(MetricNames.QLABELS_MACROAVGF1)
                metric_names.append(MetricNames.QLABELS_MICROAVGF1)
                if score is not None:
                    scores.append(score['f1_macro_avg'])
                    scores.append(score['f1_micro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            elif m == MetricNames.CHXLABEL_ROCAUC:
                metric_names.append(MetricNames.CHXLABEL_ROCAUC_MICRO)
                metric_names.append(MetricNames.CHXLABEL_ROCAUC_MACRO)
                if score is not None:
                    scores.append(score['micro_avg'])
                    scores.append(score['macro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            elif m == MetricNames.CHXLABEL_AUC:
                metric_names.append(MetricNames.CHXLABEL_AUC_MICRO)
                metric_names.append(MetricNames.CHXLABEL_AUC_MACRO)
                if score is not None:
                    scores.append(score['micro_avg'])
                    scores.append(score['macro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            elif m == MetricNames.CHESTIMAGENOMELABELROCAUC:
                metric_names.append(MetricNames.CHESTIMAGENOMELABELROCAUC_MICRO)
                metric_names.append(MetricNames.CHESTIMAGENOMELABELROCAUC_MACRO)
                if score is not None:
                    scores.append(score['micro_avg'])
                    scores.append(score['macro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            elif m == MetricNames.CHESTIMAGENOMELABELAUC:
                metric_names.append(MetricNames.CHESTIMAGENOMELABELAUC_MICRO)
                metric_names.append(MetricNames.CHESTIMAGENOMELABELAUC_MACRO)
                if score is not None:
                    scores.append(score['micro_avg'])
                    scores.append(score['macro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            elif m == MetricNames.CHESTIMAGENOMELABELPRCAUC:
                metric_names.append(MetricNames.CHESTIMAGENOMELABELPRCAUC_MICRO)
                metric_names.append(MetricNames.CHESTIMAGENOMELABELPRCAUC_MACRO)
                if score is not None:
                    scores.append(score['micro_avg'])
                    scores.append(score['macro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            elif m == MetricNames.CHXLABEL_AUC:
                metric_names.append(MetricNames.CHXLABEL_AUC_MICRO)
                metric_names.append(MetricNames.CHXLABEL_AUC_MACRO)
                if score is not None:
                    scores.append(score['micro_avg'])
                    scores.append(score['macro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            elif m == MetricNames.CHXLABEL_PRCAUC:
                metric_names.append(MetricNames.CHXLABEL_PRCAUC_MICRO)
                metric_names.append(MetricNames.CHXLABEL_PRCAUC_MACRO)
                if score is not None:
                    scores.append(score['micro_avg'])
                    scores.append(score['macro_avg'])
                else:
                    scores.append(None)
                    scores.append(None)
            else:
                metric_names.append(m)
                if score is not None:
                    if hasattr(score, '__len__') and not (type(score) is Tensor and score.dim() == 0):
                        try:
                            score = average_ignoring_nones_and_nans(score)
                        except TypeError:
                            print(f'm = {m}, score = {score}, type(score) = {type(score)}')
                            raise
                    scores.append(score)
                else:
                    scores.append(None)
        
        # print(metric_names, scores)

        assert len(metric_names) == len(scores)
        nonnull_scores = [s for s in scores if s is not None]
        nonnull_metric_names = [m for m, s in zip(metric_names, scores) if s is not None]
        metrics_str = ', '.join(f'{METRIC2SHORT.get(m, m)} {s:.5f}' for m, s in zip(nonnull_metric_names, nonnull_scores))
        duration = timer._elapsed()
        print(f'{metrics_str}, {duration:.2f} secs')

        if log_to_disk:
            metrics_logger.log_metrics(metric_names, scores)
    
    return handler

def get_lr_sch_handler(lr_scheduler, lr_scheduler_name, score_fn=None):

    if lr_scheduler_name == LRSchedulerNames.ReduceLROnPlateau:
        assert score_fn is not None
        def handler():
            lr_scheduler.step(score_fn())
    else:
        def handler():
            lr_scheduler.step()

    return handler

def get_checkpoint_handler(model_wrapper, folder_path, trainer, epoch_offset, score_name, score_fn):    
    checkpoint = Checkpoint(
        to_save=model_wrapper.to_save(),
        save_handler=DiskSaver(folder_path, require_empty=False, atomic=False),
        global_step_transform=lambda *_: trainer.state.epoch + epoch_offset,
        score_function=score_fn,
        score_name=score_name,
        greater_or_equal=True,
    )
    return checkpoint

class Accumulator():

    def __init__(self, output_transform):
        self._list = []
        self._output_transform = output_transform
    
    def reset(self):
        self._list.clear()

    def update(self, output):
        self._list.extend(self._output_transform(output))

    def get_list(self):
        return self._list

def attach_accumulator(engine, output_name):
    accumulator = Accumulator(output_transform=operator.itemgetter(output_name))
    
    def epoch_started_handler(unused_engine):
        accumulator.reset()

    def iteration_completed_handler(engine):
        # print('Debugging::accumulator')
        # print('   questions[0] = ', engine.state.output['questions'][0])
        # print('   pred_questions[0] = ', engine.state.output['pred_questions'][0])
        accumulator.update(engine.state.output)

    def epoch_completed_handler(engine):
        engine.state.metrics[output_name] = accumulator.get_list()

    engine.add_event_handler(Events.EPOCH_STARTED, epoch_started_handler)
    engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_completed_handler)
    engine.add_event_handler(Events.EPOCH_COMPLETED, epoch_completed_handler)
    
