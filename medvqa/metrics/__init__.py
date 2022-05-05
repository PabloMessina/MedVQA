from medvqa.metrics.nlp import Bleu, RougeL, Meteor, CiderD, ExactMatch
from medvqa.metrics.medical import (
    MedicalCompleteness,
    WeightedMedicalCompleteness,
    ChexpertLabelsF1score,
)
from medvqa.metrics.classification import (
    MultiLabelF1score,
    MultiLabelAccuracy,
    DatasetAwareOrientationAccuracy,
)

from ignite.metrics import RunningAverage
import operator

def _get_output_transform(pred_key, gt_key):
    def output_transform(output):
        return output[pred_key], output[gt_key]
    return output_transform

# def attach_bleu_question(engine, device, record_scores=False):
#     blue = Bleu(output_transform = _get_output_transform('pred_questions', 'questions'),
#                 device = device, record_scores=record_scores)
#     blue.attach(engine, 'bleu_question')

def attach_exactmatch_question(engine, device, record_scores=False):
    em = ExactMatch(output_transform = _get_output_transform('pred_questions', 'questions'),
                device = device, record_scores=record_scores)
    em.attach(engine, 'exactmatch_question')


def attach_bleu(engine, device, record_scores=False, ks=None):
    # if ks is None:
    #     blue = Bleu(output_transform = _get_output_transform('pred_answers', 'answers'),
    #                 device = device, record_scores=record_scores)
    #     blue.attach(engine, 'bleu')
    # else:
    #     for k in ks:
    #         blue = Bleu(k = k, output_transform = _get_output_transform('pred_answers', 'answers'),
    #                 device = device, record_scores=record_scores)
    #         blue.attach(engine, f'bleu-{k}')    
    blue = Bleu(output_transform = _get_output_transform('pred_answers', 'answers'),
                device = device, record_scores=record_scores)
    blue.attach(engine, 'bleu')

def attach_rougel(engine, device, record_scores=False):
    rougel = RougeL(output_transform = _get_output_transform('pred_answers', 'answers'),
                    device = device, record_scores=record_scores)
    rougel.attach(engine, 'rougeL')

def attach_meteor(engine, device, record_scores=False):
    meteor = Meteor(output_transform = _get_output_transform('pred_answers', 'answers'),
                    device = device, record_scores=record_scores)
    meteor.attach(engine, 'meteor')

def attach_ciderd(engine, device, record_scores=False):
    ciderd = CiderD(output_transform = _get_output_transform('pred_answers', 'answers'),
                    device=device, record_scores=record_scores)
    ciderd.attach(engine, 'ciderD')

def attach_medical_completeness(engine, device, tokenizer, record_scores=False):
    medcomp = MedicalCompleteness(tokenizer,
                output_transform = _get_output_transform('pred_answers', 'answers'),
                device = device, record_scores=record_scores)
    medcomp.attach(engine, 'medcomp')

def attach_weighted_medical_completeness(engine, device, tokenizer, record_scores=False):
    medcomp = WeightedMedicalCompleteness(tokenizer,
                output_transform = _get_output_transform('pred_answers', 'answers'),
                device = device, record_scores=record_scores)
    medcomp.attach(engine, 'wmedcomp')

def attach_medical_tags_f1score(engine, device, record_scores=False):
    medtagf1 = MultiLabelF1score(output_transform = _get_output_transform('pred_tags', 'tags'),
                                device=device, record_scores=record_scores)
    medtagf1.attach(engine, 'medtagf1')

def attach_chexpert_labels_accuracy(engine, device, record_scores=False):
    chxlabelacc = MultiLabelAccuracy(output_transform = _get_output_transform('pred_chexpert', 'chexpert'),
                                device=device, record_scores=record_scores)
    chxlabelacc.attach(engine, 'chxlabelacc')

def attach_chexpert_labels_f1score(engine, device, record_scores=False):
    chxlabelacc = ChexpertLabelsF1score(output_transform = _get_output_transform('pred_chexpert', 'chexpert'),
                                device=device, record_scores=record_scores)
    chxlabelacc.attach(engine, 'chxlabelf1')

def attach_dataset_aware_orientation_accuracy(engine, record_scores=False):
    orienacc = DatasetAwareOrientationAccuracy(record_scores=record_scores)
    orienacc.attach(engine, 'orienacc')

def attach_loss(loss_name, engine, device):
    metric = RunningAverage(
        output_transform=operator.itemgetter(loss_name),
        alpha = 1, device = device)
    metric.attach(engine, loss_name)