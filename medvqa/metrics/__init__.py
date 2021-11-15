from medvqa.metrics.nlp.bleu import Bleu
from ignite.metrics import RunningAverage
import operator

def _get_output_transform(pred_key, gt_key):
    def output_transform(output):
        return output[pred_key], output[gt_key]
    return output_transform

def attach_bleu_question(engine, device):
    blue = Bleu(output_transform = _get_output_transform('pred_questions', 'questions'),
                device = device)
    blue.attach(engine, 'bleu_question')

def attach_bleu_answer(engine, device):
    blue = Bleu(output_transform = _get_output_transform('pred_answers', 'answers'),
                device = device)
    blue.attach(engine, 'bleu_answer')

def attach_loss(loss_name, engine, device):
    metric = RunningAverage(
        output_transform=operator.itemgetter(loss_name),
        alpha = 1, device = device
    )
    metric.attach(engine, loss_name)
