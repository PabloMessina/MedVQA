from medvqa.metrics.nlp.bleu import Bleu
from medvqa.metrics.nlp.rouge import RougeL
from medvqa.metrics.nlp.cider import CiderD

from ignite.metrics import RunningAverage
import operator

def _get_output_transform(pred_key, gt_key):
    def output_transform(output):
        return output[pred_key], output[gt_key]
    return output_transform

def attach_bleu_question(engine, device, record_scores=False):
    blue = Bleu(output_transform = _get_output_transform('pred_questions', 'questions'),
                device = device, record_scores=record_scores)
    blue.attach(engine, 'bleu_question')

def attach_bleu(engine, device, record_scores=False, ks=None):
    if ks is None:
        blue = Bleu(output_transform = _get_output_transform('pred_answers', 'answers'),
                    device = device, record_scores=record_scores)
        blue.attach(engine, 'bleu')
    else:
        for k in ks:
            blue = Bleu(k = k, output_transform = _get_output_transform('pred_answers', 'answers'),
                    device = device, record_scores=record_scores)
            blue.attach(engine, f'bleu-{k}')

def attach_rougel(engine, device, record_scores=False):
    rougel = RougeL(output_transform = _get_output_transform('pred_answers', 'answers'),
                    device = device, record_scores=record_scores)
    rougel.attach(engine, 'rougeL')

def attach_ciderd(engine, device, record_scores=False):
    ciderd = CiderD(output_transform = _get_output_transform('pred_answers', 'answers'),
                    device = device, record_scores=record_scores)
    ciderd.attach(engine, 'ciderD')

def attach_loss(loss_name, engine, device):
    metric = RunningAverage(
        output_transform=operator.itemgetter(loss_name),
        alpha = 1, device = device)
    metric.attach(engine, loss_name)
