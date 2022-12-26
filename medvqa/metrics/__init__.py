from medvqa.metrics.classification.multilabel_accuracy import DatasetAwareMultiLabelAccuracy
from medvqa.metrics.classification.multilabel_prf1 import (
    DatasetAwareMultiLabelMacroAvgF1,
    DatasetAwareMultiLabelMicroAvgF1,
)
from medvqa.metrics.dataset_aware_metric import DatasetAwareEpochMetric
from medvqa.metrics.nlp import Bleu, RougeL, Meteor, CiderD, ExactMatch
from medvqa.metrics.medical import (
    MedicalCompleteness,
    WeightedMedicalCompleteness,
    DatasetAwareWeightedMedicalCompleteness,
    ChexpertLabelsF1score,
)
from medvqa.metrics.classification import (
    MultiLabelF1score,
    MultiLabelMacroAvgF1,
    MultiLabelMicroAvgF1,
    MultiLabelAccuracy,
    MultiLabelPRF1,
    DatasetAwareMultilabelF1score,
    DatasetAwareSinglelabelAccuracy,
    DatasetAwareOrientationAccuracy,
    roc_auc_fn,
)
from medvqa.losses import DatasetAwareLoss

from ignite.metrics import RunningAverage, EpochMetric
import operator
from medvqa.metrics.nlp.bleu import DatasetAwareBleu
from medvqa.metrics.nlp.cider import DatasetAwareCiderD
from medvqa.metrics.nlp.exact_match import DatasetAwareExactMatch

from medvqa.utils.constants import MetricNames

def _get_output_transform(pred_key, gt_key, valid_class_indices=None):
    if valid_class_indices is None:
        def output_transform(output):
            return output[pred_key], output[gt_key]
    else:
        print('_get_output_transform(): valid_class_indices =', valid_class_indices)
        def output_transform(output):
            return output[pred_key][:, valid_class_indices],\
                   output[gt_key][:, valid_class_indices]
    return output_transform

# def attach_bleu_question(engine, device, record_scores=False):
#     blue = Bleu(output_transform = _get_output_transform('pred_questions', 'questions'),
#                 device = device, record_scores=record_scores)
#     blue.attach(engine, 'bleu_question')

def attach_exactmatch_question(engine, device, record_scores=False):
    em = ExactMatch(output_transform = _get_output_transform('pred_questions', 'questions'),
                device = device, record_scores=record_scores)
    em.attach(engine, MetricNames.EXACTMATCH_QUESTION)

def attach_dataset_aware_exactmatch_question(engine, allowed_dataset_ids, record_scores=False):
    em = DatasetAwareExactMatch(output_transform = _get_output_transform('pred_questions', 'questions'),
                allowed_dataset_ids=allowed_dataset_ids,
                record_scores=record_scores)
    em.attach(engine, MetricNames.EXACTMATCH_QUESTION)

def attach_dataset_aware_exactmatch_answer(engine, allowed_dataset_ids, record_scores=False):
    em = DatasetAwareExactMatch(output_transform = _get_output_transform('pred_answers', 'answers'),
                allowed_dataset_ids=allowed_dataset_ids,
                record_scores=record_scores)
    em.attach(engine, MetricNames.EXACTMATCH_ANSWER)

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

def attach_dataset_aware_bleu_background(engine, allowed_dataset_ids, record_scores=False):
    met = DatasetAwareBleu(output_transform=_get_output_transform('pred_backgrounds', 'backgrounds'),
                           allowed_dataset_ids=allowed_dataset_ids, record_scores=record_scores)
    met.attach(engine, MetricNames.BLEU_BACKGROUND)

def attach_dataset_aware_bleu(engine, allowed_dataset_ids, record_scores=False):
    met = DatasetAwareBleu(output_transform=_get_output_transform('pred_answers', 'answers'),
                           allowed_dataset_ids=allowed_dataset_ids, record_scores=record_scores)
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

def attach_dataset_aware_ciderd(engine, allowed_dataset_ids, record_scores=False):
    met = DatasetAwareCiderD(output_transform = _get_output_transform('pred_answers', 'answers'),
                allowed_dataset_ids=allowed_dataset_ids,
                record_scores=record_scores)
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

def attach_dataset_aware_weighted_medical_completeness(engine, tokenizer, allowed_dataset_ids, record_scores=False):
    met = DatasetAwareWeightedMedicalCompleteness(tokenizer,
                output_transform = _get_output_transform('pred_answers', 'answers'),
                allowed_dataset_ids = allowed_dataset_ids,
                record_scores=record_scores)
    met.attach(engine, MetricNames.WMEDCOMP)

def attach_medical_tags_f1score(engine, device, record_scores=False):
    met = MultiLabelF1score(output_transform = _get_output_transform('pred_tags', 'tags'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.MEDTAGF1)

def attach_chexpert_labels_accuracy(engine, device, record_scores=False):
    met = MultiLabelAccuracy(output_transform = _get_output_transform('pred_chexpert', 'chexpert'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.CHXLABELACC)

def attach_dataset_aware_chexpert_labels_accuracy(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelAccuracy(output_transform=_get_output_transform('pred_chexpert', 'chexpert', class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CHXLABELACC)

def attach_chexpert_labels_f1score(engine, device, record_scores=False):
    met = ChexpertLabelsF1score(output_transform = _get_output_transform('pred_chexpert', 'chexpert'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.CHXLABELF1)

def attach_chexpert_labels_macroavgf1(engine, device):
    met = MultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_chexpert', 'chexpert'), device=device)
    met.attach(engine, MetricNames.CHXLABELMACROAVGF1)

def attach_dataset_aware_chexpert_labels_macroavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform=_get_output_transform('pred_chexpert', 'chexpert', class_indices),
                                           allowed_dataset_ids=allowed_dataset_ids)
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

def attach_dataset_aware_chexpert_labels_roc_auc(engine, allowed_dataset_ids, device, class_indices=None):
    met = DatasetAwareEpochMetric(compute_fn=roc_auc_fn,
                                  output_transform=_get_output_transform('pred_chexpert_probs', 'chexpert', class_indices),
                                  allowed_dataset_ids=allowed_dataset_ids,
                                  device=device)
    met.attach(engine, MetricNames.CHXLABEL_ROCAUC)

def attach_question_labels_f1score(engine, device, record_scores=False):
    met = MultiLabelF1score(output_transform = _get_output_transform('pred_qlabels', 'qlabels'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.QLABELSF1)

def attach_dataset_aware_question_labels_f1score(engine, allowed_dataset_ids, record_scores=False):
    met = DatasetAwareMultilabelF1score(output_transform = _get_output_transform('pred_qlabels', 'qlabels'),
                                allowed_dataset_ids=allowed_dataset_ids,
                                record_scores=record_scores)
    met.attach(engine, MetricNames.QLABELSF1)

def attach_dataset_aware_question_labels_macroavgf1(engine, allowed_dataset_ids):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_qlabels', 'qlabels'),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.QLABELS_MACROAVGF1)

def attach_dataset_aware_question_labels_microavgf1(engine, allowed_dataset_ids):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_qlabels', 'qlabels'),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.QLABELS_MICROAVGF1)

def attach_chexpert_labels_microavgf1(engine, device):
    met = MultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_chexpert', 'chexpert'), device=device)
    met.attach(engine, MetricNames.CHXLABELMICROAVGF1)

def attach_dataset_aware_chexpert_labels_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform=_get_output_transform('pred_chexpert', 'chexpert', class_indices),
                                           allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CHXLABELMICROAVGF1)

def attach_dataset_aware_vinbig_labels_macroavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_vinbig_labels', 'vinbig_labels', class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.VINBIGMACROAVGF1)

def attach_dataset_aware_vinbig_labels_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_vinbig_labels', 'vinbig_labels', class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.VINBIGMICROAVGF1)

def attach_dataset_aware_cxr14_labels_macroavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_cxr14', 'cxr14', class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CXR14MACROAVGF1)

def attach_dataset_aware_cxr14_labels_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_cxr14', 'cxr14', class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CXR14MICROAVGF1)

def attach_dataset_aware_padchest_labels_macroavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_padchest_labels', 'padchest_labels', class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.PADCHEST_LABEL_MACROAVGF1)

def attach_dataset_aware_padchest_labels_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_padchest_labels', 'padchest_labels', class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.PADCHEST_LABEL_MICROAVGF1)

def attach_dataset_aware_padchest_localization_macroavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_padchest_loc', 'padchest_loc', class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.PADCHEST_LOC_MACROAVGF1)

def attach_dataset_aware_padchest_localization_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_padchest_loc', 'padchest_loc', class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.PADCHEST_LOC_MICROAVGF1)

def attach_dataset_aware_gender_accuracy(engine, allowed_dataset_ids, record_scores=False):
    met = DatasetAwareSinglelabelAccuracy(output_transform = _get_output_transform('pred_gender', 'gender'),
                                allowed_dataset_ids=allowed_dataset_ids,
                                record_scores=record_scores)
    met.attach(engine, MetricNames.GENDER_ACC)

def attach_question_labels_prf1(engine, device):
    met = MultiLabelPRF1(output_transform = _get_output_transform('pred_qlabels', 'qlabels'), device=device)
    met.attach(engine, MetricNames.QLABELS_PRF1)

def attach_dataset_aware_orientation_accuracy(engine, allowed_dataset_ids, record_scores=False):
    met = DatasetAwareOrientationAccuracy(allowed_dataset_ids, record_scores=record_scores)
    met.attach(engine, MetricNames.ORIENACC)

def attach_dataset_aware_loss(engine, loss_name, allowed_dataset_ids):
    met = DatasetAwareLoss(
        output_transform=operator.itemgetter(loss_name),
        allowed_dataset_ids=allowed_dataset_ids
    )
    met.attach(engine, loss_name)

def attach_loss(loss_name, engine, device):
    met = RunningAverage(
        output_transform=operator.itemgetter(loss_name),
        alpha = 1, device = device)
    met.attach(engine, loss_name)