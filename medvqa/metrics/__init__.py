from medvqa.metrics.nlp import Bleu, RougeL, Meteor, CiderD, ExactMatch
from medvqa.metrics.medical import (
    MedicalCompleteness,
    WeightedMedicalCompleteness,
    ChexpertLabelsF1score,
)
from medvqa.metrics.classification import (
    MultiLabelF1score,
    MultiLabelMacroAvgF1,
    MultiLabelMicroAvgF1,
    MultiLabelAccuracy,
    MultiLabelPRF1,
    DatasetAwareOrientationAccuracy,
    roc_auc_fn,
)

from ignite.metrics import RunningAverage, EpochMetric
import operator

from medvqa.utils.constants import MetricNames

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
    em.attach(engine, MetricNames.EXACTMATCH_QUESTION)


def attach_bleu(engine, device, record_scores=False):
    # if ks is None:
    #     blue = Bleu(output_transform = _get_output_transform('pred_answers', 'answers'),
    #                 device = device, record_scores=record_scores)
    #     blue.attach(engine, 'bleu')
    # else:
    #     for k in ks:
    #         blue = Bleu(k = k, output_transform = _get_output_transform('pred_answers', 'answers'),
    #                 device = device, record_scores=record_scores)
    #         blue.attach(engine, f'bleu-{k}')    
    met = Bleu(output_transform = _get_output_transform('pred_answers', 'answers'),
                device = device, record_scores=record_scores)
    met.attach(engine, MetricNames.BLEU)

def attach_rougel(engine, device, record_scores=False):
    met = RougeL(output_transform = _get_output_transform('pred_answers', 'answers'),
                    device = device, record_scores=record_scores)
    met.attach(engine, MetricNames.ROUGE_L)

def attach_meteor(engine, device, record_scores=False):
    met = Meteor(output_transform = _get_output_transform('pred_answers', 'answers'),
                    device = device, record_scores=record_scores)
    met.attach(engine, MetricNames.METEOR)

def attach_ciderd(engine, device, record_scores=False):
    met = CiderD(output_transform = _get_output_transform('pred_answers', 'answers'),
                    device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.CIDER_D)

def attach_medical_completeness(engine, device, tokenizer, record_scores=False):
    met = MedicalCompleteness(tokenizer,
                output_transform = _get_output_transform('pred_answers', 'answers'),
                device = device, record_scores=record_scores)
    met.attach(engine, MetricNames.MEDCOMP)

def attach_weighted_medical_completeness(engine, device, tokenizer, record_scores=False):
    met = WeightedMedicalCompleteness(tokenizer,
                output_transform = _get_output_transform('pred_answers', 'answers'),
                device = device, record_scores=record_scores)
    met.attach(engine, MetricNames.WMEDCOMP)

def attach_medical_tags_f1score(engine, device, record_scores=False):
    met = MultiLabelF1score(output_transform = _get_output_transform('pred_tags', 'tags'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.MEDTAGF1)

def attach_chexpert_labels_accuracy(engine, device, record_scores=False):
    met = MultiLabelAccuracy(output_transform = _get_output_transform('pred_chexpert', 'chexpert'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.CHXLABELACC)

def attach_chexpert_labels_f1score(engine, device, record_scores=False):
    met = ChexpertLabelsF1score(output_transform = _get_output_transform('pred_chexpert', 'chexpert'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.CHXLABELF1)

def attach_chexpert_labels_macroavgf1(engine, device):
    met = MultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_chexpert', 'chexpert'), device=device)
    met.attach(engine, MetricNames.CHXLABELMACROAVGF1)

def attach_chexpert_labels_microavgf1(engine, device):
    met = MultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_chexpert', 'chexpert'), device=device)
    met.attach(engine, MetricNames.CHXLABELMICROAVGF1)

def attach_chexpert_labels_prf1(engine, device):
    met = MultiLabelPRF1(output_transform = _get_output_transform('pred_chexpert', 'chexpert'), device=device)
    met.attach(engine, MetricNames.CHXLABEL_PRF1)

def attach_chexpert_labels_roc_auc(engine, device):
    met = EpochMetric(compute_fn=roc_auc_fn, output_transform=_get_output_transform('pred_chexpert_probs', 'chexpert'), device=device)
    met.attach(engine, MetricNames.CHXLABEL_ROCAUC)

def attach_question_labels_f1score(engine, device, record_scores=False):
    met = MultiLabelF1score(output_transform = _get_output_transform('pred_qlabels', 'qlabels'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.QLABELSF1)

def attach_question_labels_prf1(engine, device):
    met = MultiLabelPRF1(output_transform = _get_output_transform('pred_qlabels', 'qlabels'), device=device)
    met.attach(engine, MetricNames.QLABELS_PRF1)

def attach_dataset_aware_orientation_accuracy(engine, record_scores=False):
    met = DatasetAwareOrientationAccuracy(record_scores=record_scores)
    met.attach(engine, MetricNames.ORIENACC)

def attach_loss(loss_name, engine, device):
    met = RunningAverage(
        output_transform=operator.itemgetter(loss_name),
        alpha = 1, device = device)
    met.attach(engine, loss_name)