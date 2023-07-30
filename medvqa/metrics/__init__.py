from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_NUM_BBOX_CLASSES
from medvqa.metrics.bbox import DatasetAwareBboxIOU, DatasetAwareBboxMAE, DatasetAwareBboxMeanF1
from medvqa.metrics.classification.auc import auc_fn
from medvqa.metrics.classification.multilabel_accuracy import DatasetAwareMultiLabelAccuracy
from medvqa.metrics.classification.multilabel_prf1 import (
    ConditionAwareMultiLabelMacroAvgF1,
    ConditionAwareMultiLabelMultiClassMacroAvgF1,
    DatasetAwareMultiLabelMacroAvgF1,
    DatasetAwareMultiLabelMicroAvgF1,
)
from medvqa.metrics.classification.prc_auc import prc_auc_fn
from medvqa.metrics.classification.singlelabel_accuracy import InstanceConditionedTripletAccuracy
from medvqa.metrics.dataset_aware_metric import DatasetAwareEpochMetric
from medvqa.metrics.logging.condition_aware_t5_report_logger import ConditionAwareSeq2SeqOutputLogger
from medvqa.metrics.medical.med_completeness import ConditionAwareWeightedMedicalCompleteness
from medvqa.metrics.nlp import Bleu, RougeL, Meteor, CiderD, ExactMatch
from medvqa.metrics.medical import (
    MedicalCompleteness,
    WeightedMedicalCompleteness,
    DatasetAwareWeightedMedicalCompleteness,
    ChexpertLabelsF1score,
)
from medvqa.metrics.classification import (
    ConditionAwareSingleLabelAccuracy,
    MultiLabelF1score,
    MultiLabelMacroAvgF1,
    MultiLabelMicroAvgF1,
    MultiLabelAccuracy,
    MultiLabelPRF1,
    DatasetAwareMultilabelF1score,
    DatasetAwareSingleLabelAccuracy,
    DatasetAwareOrientationAccuracy,
    roc_auc_fn,
)
from medvqa.losses import DatasetAwareLoss, ConditionAwareLoss

from ignite.metrics import RunningAverage, EpochMetric
import operator
from medvqa.metrics.nlp.bleu import DatasetAwareBleu
from medvqa.metrics.nlp.cider import ConditionAwareCiderD, DatasetAwareCiderD
from medvqa.metrics.nlp.exact_match import DatasetAwareExactMatch

from medvqa.utils.constants import VINBIG_BBOX_NAMES, MetricNames

def _get_output_transform(*keys, class_indices=None):
    if class_indices is None:
        def output_transform(output):
            # try:
            return tuple(output[key] for key in keys)
            # except KeyError:
            #     print('output =', output)
            #     print('keys =', keys)
            #     raise
    else:
        # print('_get_output_transform(): class_indices =', class_indices)
        def output_transform(output):
            return tuple(output[key][:, class_indices] for key in keys)
    return output_transform


# ------------------------------------------------------------
# NLP metrics (BLEU, ROUGE-L, METEOR, CIDEr-D, ExactMatch)
# ------------------------------------------------------------

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

def attach_ciderd(engine, device, record_scores=False, field='answers'):
    met = CiderD(output_transform = _get_output_transform(f'pred_{field}', field),
                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.CIDER_D)

def attach_dataset_aware_ciderd(engine, allowed_dataset_ids, record_scores=False, field='answers'):
    met = DatasetAwareCiderD(output_transform = _get_output_transform(f'pred_{field}', field),
                allowed_dataset_ids=allowed_dataset_ids,
                record_scores=record_scores)
    met.attach(engine, MetricNames.CIDER_D)

def attach_condition_aware_ciderd(engine, condition_function, record_scores=False, field='answers',
                                  metric_name=MetricNames.CIDER_D):
    met = ConditionAwareCiderD(output_transform = _get_output_transform(f'pred_{field}', field),
                condition_function=condition_function,
                record_scores=record_scores)
    met.attach(engine, metric_name)

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

# ------------------------------------------------------------------------------
# Medical Completeness related metrics (these evaluate natural language answers)
# ------------------------------------------------------------------------------

def attach_medical_completeness(engine, device, tokenizer, record_scores=False, field='answers'):
    met = MedicalCompleteness(tokenizer,
                output_transform = _get_output_transform(f'pred_{field}', field),
                device = device, record_scores=record_scores)
    met.attach(engine, MetricNames.MEDCOMP)

def attach_weighted_medical_completeness(engine, device, tokenizer, record_scores=False, field='answers'):
    met = WeightedMedicalCompleteness(tokenizer,
                output_transform = _get_output_transform(f'pred_{field}', field),
                device = device, record_scores=record_scores)
    met.attach(engine, MetricNames.WMEDCOMP)

def attach_dataset_aware_weighted_medical_completeness(engine, tokenizer, allowed_dataset_ids, record_scores=False, field='answers'):
    met = DatasetAwareWeightedMedicalCompleteness(tokenizer,
                output_transform = _get_output_transform(f'pred_{field}', field),
                allowed_dataset_ids = allowed_dataset_ids,
                record_scores=record_scores)
    met.attach(engine, MetricNames.WMEDCOMP)

def attach_condition_aware_weighted_medical_completeness(engine, tokenizer, condition_function, record_scores=False, field='answers',
                                                         metric_name=MetricNames.WMEDCOMP):
    met = ConditionAwareWeightedMedicalCompleteness(tokenizer,
                output_transform = _get_output_transform(f'pred_{field}', field),
                condition_function = condition_function,
                record_scores=record_scores)
    met.attach(engine, metric_name)

# ---------------------------------------------
# Medical tags prediction metrics
# ---------------------------------------------

def attach_medical_tags_f1score(engine, device, record_scores=False):
    met = MultiLabelF1score(output_transform = _get_output_transform('pred_tags', 'tags'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.MEDTAGF1)

# ---------------------------------------------
# CheXpert related metrics
# ---------------------------------------------

def attach_chexpert_labels_accuracy(engine, device, record_scores=False):
    met = MultiLabelAccuracy(output_transform = _get_output_transform('pred_chexpert', 'chexpert'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.CHXLABELACC)

def attach_dataset_aware_chexpert_labels_accuracy(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelAccuracy(output_transform=_get_output_transform('pred_chexpert', 'chexpert', class_indices=class_indices),
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
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform=_get_output_transform('pred_chexpert', 'chexpert', class_indices=class_indices),
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
                                  output_transform=_get_output_transform('pred_chexpert_probs', 'chexpert', class_indices=class_indices),
                                  allowed_dataset_ids=allowed_dataset_ids,
                                  device=device)
    met.attach(engine, MetricNames.CHXLABEL_ROCAUC)

def attach_chexpert_labels_microavgf1(engine, device):
    met = MultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_chexpert', 'chexpert'), device=device)
    met.attach(engine, MetricNames.CHXLABELMICROAVGF1)

def attach_dataset_aware_chexpert_labels_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform=_get_output_transform('pred_chexpert', 'chexpert', class_indices=class_indices),
                                           allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CHXLABELMICROAVGF1)

def attach_dataset_aware_chexpert_labels_auc(engine, allowed_dataset_ids, device):
    met = DatasetAwareEpochMetric(compute_fn=auc_fn,
                                  output_transform=_get_output_transform('pred_chexpert_probs', 'chexpert'),
                                  allowed_dataset_ids=allowed_dataset_ids,
                                  device=device)
    met.attach(engine, MetricNames.CHXLABEL_AUC)

def attach_dataset_aware_chexpert_labels_prcauc(engine, allowed_dataset_ids, device):
    met = DatasetAwareEpochMetric(compute_fn=prc_auc_fn,
                                  output_transform=_get_output_transform('pred_chexpert_probs', 'chexpert'),
                                  allowed_dataset_ids=allowed_dataset_ids,
                                  device=device)
    met.attach(engine, MetricNames.CHXLABEL_PRCAUC)

# ---------------------------------------------
# Chest-ImaGenome related metrics
# ---------------------------------------------

def attach_chest_imagenome_labels_accuracy(engine, device, record_scores=False):
    met = MultiLabelAccuracy(output_transform = _get_output_transform('pred_chest_imagenome', 'chest_imagenome'),
                                device=device, record_scores=record_scores)
    met.attach(engine, MetricNames.CHESTIMAGENOMELABELACC)

def attach_chest_imagenome_labels_prf1(engine, device):
    met = MultiLabelPRF1(output_transform = _get_output_transform('pred_chest_imagenome', 'chest_imagenome'), device=device)
    met.attach(engine, MetricNames.CHESTIMAGENOMELABEL_PRF1)

def attach_chest_imagenome_labels_roc_auc(engine, device):
    met = EpochMetric(compute_fn=roc_auc_fn, output_transform=_get_output_transform('pred_chest_imagenome_probs', 'chest_imagenome'), device=device)
    met.attach(engine, MetricNames.CHESTIMAGENOMELABELROCAUC)

def attach_dataset_aware_chest_imagenome_labels_accuracy(engine, allowed_dataset_ids):
    met = DatasetAwareMultiLabelAccuracy(output_transform=_get_output_transform('pred_chest_imagenome', 'chest_imagenome'),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CHESTIMAGENOMELABELACC)

def attach_dataset_aware_chest_imagenome_labels_macroavgf1(engine, allowed_dataset_ids):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform=_get_output_transform('pred_chest_imagenome', 'chest_imagenome'),
                                           allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CHESTIMAGENOMELABELMACROAVGF1)

def attach_dataset_aware_chest_imagenome_labels_microavgf1(engine, allowed_dataset_ids):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform=_get_output_transform('pred_chest_imagenome', 'chest_imagenome'),
                                           allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CHESTIMAGENOMELABELMICROAVGF1)

def attach_dataset_aware_chest_imagenome_labels_roc_auc(engine, allowed_dataset_ids, device):
    met = DatasetAwareEpochMetric(compute_fn=roc_auc_fn,
                                  output_transform=_get_output_transform('pred_chest_imagenome_probs', 'chest_imagenome'),
                                  allowed_dataset_ids=allowed_dataset_ids,
                                  device=device)
    met.attach(engine, MetricNames.CHESTIMAGENOMELABELROCAUC)

def attach_dataset_aware_chest_imagenome_labels_auc(engine, allowed_dataset_ids, device):
    met = DatasetAwareEpochMetric(compute_fn=auc_fn,
                                  output_transform=_get_output_transform('pred_chest_imagenome_probs', 'chest_imagenome'),
                                  allowed_dataset_ids=allowed_dataset_ids,
                                  device=device)
    met.attach(engine, MetricNames.CHESTIMAGENOMELABELAUC)

def attach_dataset_aware_chest_imagenome_labels_prcauc(engine, allowed_dataset_ids, device):
    met = DatasetAwareEpochMetric(compute_fn=prc_auc_fn,
                                  output_transform=_get_output_transform('pred_chest_imagenome_probs', 'chest_imagenome'),
                                  allowed_dataset_ids=allowed_dataset_ids,
                                  device=device)
    met.attach(engine, MetricNames.CHESTIMAGENOMELABELPRCAUC)

def attach_dataset_aware_chest_imagenome_bbox_mae(engine, allowed_dataset_ids, use_detectron2=False, use_yolov8=False):
    if use_detectron2:
        met = DatasetAwareBboxMAE(output_transform=_get_output_transform('pred_boxes', 'pred_classes', 'scores',
                                                                        'bbox_coords', 'bbox_presence'),
                                allowed_dataset_ids=allowed_dataset_ids, use_detectron2=True)
    elif use_yolov8:
        met = DatasetAwareBboxMAE(output_transform=_get_output_transform('yolov8_predictions',
                                                                        'chest_imagenome_bbox_coords',
                                                                        'chest_imagenome_bbox_presence'),
                                allowed_dataset_ids=allowed_dataset_ids, use_yolov8=True)
    else:
        met = DatasetAwareBboxMAE(output_transform=_get_output_transform('pred_chest_imagenome_bbox_coords',
                                                                            'chest_imagenome_bbox_coords',
                                                                            'chest_imagenome_bbox_presence'),
                                    allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CHESTIMAGENOMEBBOXMAE)

def attach_dataset_aware_chest_imagenome_bbox_iou(engine, allowed_dataset_ids, use_detectron2=False, use_yolov8=False):
    if use_detectron2:
        met = DatasetAwareBboxIOU(output_transform=_get_output_transform('pred_boxes', 'pred_classes', 'scores',
                                                                        'bbox_coords', 'bbox_presence'),
                                allowed_dataset_ids=allowed_dataset_ids, use_detectron2=True)
    elif use_yolov8:
        met = DatasetAwareBboxIOU(output_transform=_get_output_transform('yolov8_predictions',
                                                                        'chest_imagenome_bbox_coords',
                                                                        'chest_imagenome_bbox_presence'),
                                allowed_dataset_ids=allowed_dataset_ids, use_yolov8=True)
    else:
        met = DatasetAwareBboxIOU(output_transform=_get_output_transform('pred_chest_imagenome_bbox_coords',
                                                                            'chest_imagenome_bbox_coords',
                                                                            'chest_imagenome_bbox_presence'),
                                    allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CHESTIMAGENOMEBBOXIOU)

def attach_dataset_aware_chest_imagenome_bbox_meanf1(engine, allowed_dataset_ids, use_detectron2=False, use_yolov8=False):
    if use_detectron2:
        met = DatasetAwareBboxMeanF1(output_transform=_get_output_transform('pred_boxes', 'pred_classes', 'scores',
                                                                            'bbox_coords', 'bbox_presence'),
                                    allowed_dataset_ids=allowed_dataset_ids, use_detectron2=True,
                                    n_classes=CHEST_IMAGENOME_NUM_BBOX_CLASSES)
    elif use_yolov8:
        met = DatasetAwareBboxMeanF1(output_transform=_get_output_transform('yolov8_predictions',
                                                                        'chest_imagenome_bbox_coords',
                                                                        'chest_imagenome_bbox_presence'),
                                    allowed_dataset_ids=allowed_dataset_ids, use_yolov8=True,
                                    n_classes=CHEST_IMAGENOME_NUM_BBOX_CLASSES)
    else:
        met = DatasetAwareBboxMeanF1(output_transform=_get_output_transform(
                                    'pred_chest_imagenome_bbox_coords', 'chest_imagenome_bbox_coords',
                                    'pred_chest_imagenome_bbox_presence', 'chest_imagenome_bbox_presence',
                                    ), allowed_dataset_ids=allowed_dataset_ids, n_classes=CHEST_IMAGENOME_NUM_BBOX_CLASSES)
    met.attach(engine, MetricNames.CHESTIMAGENOMEBBOXMEANF1)

# ---------------------------------------------
# Question labels related metrics
# ---------------------------------------------

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

def attach_question_labels_prf1(engine, device):
    met = MultiLabelPRF1(output_transform = _get_output_transform('pred_qlabels', 'qlabels'), device=device)
    met.attach(engine, MetricNames.QLABELS_PRF1)

# ---------------------------------------------
# VinBigData related metrics
# ---------------------------------------------

def attach_dataset_aware_vinbig_labels_macroavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_vinbig_labels', 'vinbig_labels', class_indices=class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.VINBIGMACROAVGF1)

def attach_dataset_aware_vinbig_labels_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_vinbig_labels', 'vinbig_labels', class_indices=class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.VINBIGMICROAVGF1)

def attach_dataset_aware_vinbig_labels_auc(engine, allowed_dataset_ids, device):
    met = DatasetAwareEpochMetric(compute_fn=auc_fn,
                                  output_transform=_get_output_transform('pred_vinbig_probs', 'vinbig_labels'),
                                  allowed_dataset_ids=allowed_dataset_ids,
                                  device=device)
    met.attach(engine, MetricNames.VINBIGLABELAUC)

def attach_dataset_aware_vinbig_labels_prcauc(engine, allowed_dataset_ids, device):
    met = DatasetAwareEpochMetric(compute_fn=prc_auc_fn,
                                  output_transform=_get_output_transform('pred_vinbig_probs', 'vinbig_labels'),
                                  allowed_dataset_ids=allowed_dataset_ids,
                                  device=device)
    met.attach(engine, MetricNames.VINBIGLABELPRCAUC)

def attach_dataset_aware_vinbig_bbox_iou(engine, allowed_dataset_ids):
    met = DatasetAwareBboxIOU(output_transform=_get_output_transform('yolov8_predictions',
                                                                     'vinbig_bbox_coords',
                                                                     'vinbig_bbox_classes'),
                            allowed_dataset_ids=allowed_dataset_ids, use_yolov8=True, for_vinbig=True)
    met.attach(engine, MetricNames.VINBIGBBOXIOU)

def attach_dataset_aware_vinbig_bbox_meanf1(engine, allowed_dataset_ids):
    met = DatasetAwareBboxMeanF1(output_transform=_get_output_transform('yolov8_predictions',
                                                                        'vinbig_bbox_coords',
                                                                        'vinbig_bbox_classes'),
                                allowed_dataset_ids=allowed_dataset_ids, use_yolov8=True, for_vinbig=True,
                                n_classes=len(VINBIG_BBOX_NAMES),
                                iou_thresholds=[0.5])
    met.attach(engine, MetricNames.VINBIGBBOXMEANF1)

# ---------------------------------------------
# CXR14 related metrics
# ---------------------------------------------

def attach_dataset_aware_cxr14_labels_macroavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_cxr14', 'cxr14', class_indices=class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CXR14MACROAVGF1)

def attach_dataset_aware_cxr14_labels_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_cxr14', 'cxr14', class_indices=class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.CXR14MICROAVGF1)

# ---------------------------------------------
# PadChest related metrics
# ---------------------------------------------

def attach_dataset_aware_padchest_labels_macroavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_padchest_labels', 'padchest_labels', class_indices=class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.PADCHEST_LABEL_MACROAVGF1)

def attach_dataset_aware_padchest_labels_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_padchest_labels', 'padchest_labels', class_indices=class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.PADCHEST_LABEL_MICROAVGF1)

def attach_dataset_aware_padchest_localization_macroavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMacroAvgF1(output_transform = _get_output_transform('pred_padchest_loc', 'padchest_loc', class_indices=class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.PADCHEST_LOC_MACROAVGF1)

def attach_dataset_aware_padchest_localization_microavgf1(engine, allowed_dataset_ids, class_indices=None):
    met = DatasetAwareMultiLabelMicroAvgF1(output_transform = _get_output_transform('pred_padchest_loc', 'padchest_loc', class_indices=class_indices),
                                        allowed_dataset_ids=allowed_dataset_ids)
    met.attach(engine, MetricNames.PADCHEST_LOC_MICROAVGF1)

# ---------------------------------------------
# Other metrics
# ---------------------------------------------

# Gender
def attach_dataset_aware_gender_accuracy(engine, allowed_dataset_ids, record_scores=False, ignore_index=-1):
    met = DatasetAwareSingleLabelAccuracy(output_transform = _get_output_transform('pred_gender', 'gender'),
                                allowed_dataset_ids=allowed_dataset_ids,
                                record_scores=record_scores, ignore_index=ignore_index)
    met.attach(engine, MetricNames.GENDER_ACC)

# Orientation/View/Projection
def attach_dataset_aware_orientation_accuracy(engine, allowed_dataset_ids, record_scores=False):
    met = DatasetAwareOrientationAccuracy(allowed_dataset_ids, record_scores=record_scores)
    met.attach(engine, MetricNames.ORIENACC)

# Triplet accuracy
def attach_condition_aware_triplet_accuracy(engine, accepted_id, metric_name, condition_function=lambda _: True):
    met = InstanceConditionedTripletAccuracy(
        output_transform=_get_output_transform('triplet_logits', 'rule_id'),
        accepted_id=accepted_id,
        condition_function=condition_function,
    )
    met.attach(engine, metric_name)

# General purpose accuracy
def attach_condition_aware_accuracy(engine, pred_field_name, gt_field_name, metric_name, condition_function=lambda _: True):
    met = ConditionAwareSingleLabelAccuracy(
        output_transform=_get_output_transform(pred_field_name, gt_field_name),
        condition_function=condition_function,
    )
    met.attach(engine, metric_name)

# General purpose multi-label multi-class f1-score
def attach_condition_aware_multilabel_multiclass_f1score(engine, pred_field_name, gt_field_name, metric_name, condition_function=lambda _: True):
    met = ConditionAwareMultiLabelMultiClassMacroAvgF1(
        output_transform=_get_output_transform(pred_field_name, gt_field_name),
        condition_function=condition_function,
    )
    met.attach(engine, metric_name)

# General purpose multi-label f1-score
def attach_condition_aware_multilabel_f1score(engine, pred_field_name, gt_field_name, metric_name, condition_function=lambda _: True):
    met = ConditionAwareMultiLabelMacroAvgF1(
        output_transform=_get_output_transform(pred_field_name, gt_field_name),
        condition_function=condition_function,
    )
    met.attach(engine, metric_name)

# ---------------------------------------------
# Losses
# ---------------------------------------------

def attach_dataset_aware_loss(engine, loss_name, allowed_dataset_ids):
    met = DatasetAwareLoss(
        output_transform=operator.itemgetter(loss_name),
        allowed_dataset_ids=allowed_dataset_ids,
    )
    met.attach(engine, loss_name)

def attach_condition_aware_loss(engine, loss_name, condition_function=lambda _: True, metric_name=None):
    if metric_name is None:
        metric_name = loss_name
    met = ConditionAwareLoss(
        output_transform=operator.itemgetter(loss_name),
        condition_function=condition_function,
    )
    met.attach(engine, metric_name)

def attach_loss(loss_name, engine, device):
    met = RunningAverage(
        output_transform=operator.itemgetter(loss_name),
        alpha = 1, device = device)
    met.attach(engine, loss_name)

# ---------------------------------------------
# Logging
# ---------------------------------------------
def attach_condition_aware_seq2seq_output_logger(engine, tokenizer=None, condition_function=lambda _: True,
                                         field_name='pred_reports', metric_name=None):
    # Tested with T5 and BART
    met = ConditionAwareSeq2SeqOutputLogger(
        output_transform=operator.itemgetter(field_name),
        condition_function=condition_function,
        tokenizer=tokenizer,
    )
    if metric_name is None:
        metric_name = field_name
    met.attach(engine, metric_name)