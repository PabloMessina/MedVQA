from enum import Enum

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

CHEXPERT_LABEL2SHORT = {
    'No Finding': 'NF',
    'Cardiomegaly': 'Card',
    'Enlarged Cardiomediastinum': 'EC',
    'Consolidation': 'Cons',
    'Lung Opacity': 'LO',
    'Atelectasis': 'A',
    'Support Devices': 'SD',
    'Pleural Effusion': 'PE',
    'Pleural Other': 'PO',
    'Pneumonia': 'Pn',
    'Pneumothorax': 'Pt',
    'Edema': 'E',
    'Lung Lesion': 'LL',
    'Fracture': 'F',
}

CHEXPERT_METRICS = [
    'chexpert_accuracy',
    'chexpert_prf1s',
]

NLP_METRICS = [
    'bleu_question',
    'bleu',
    'bleu-1',
    'bleu-2',
    'bleu-3',
    'bleu-4',
    'rougeL',
    'ciderD',
]

METRIC2SHORT = {
    'chexpert_accuracy': 'chx_acc',
    'chexpert_prf1s': 'chx_prf1s',
    'bleu_question': 'bq',
    'exactmatch_question': 'emq',
    'bleu': 'b',
    'bleu-1': 'b1',
    'bleu-2': 'b2',
    'bleu-3': 'b3',
    'bleu-4': 'b4',
    'rougeL': 'rg-L',
    'meteor': 'met',
    'ciderD': 'cD',
    'medcomp': 'mdcmp',
    'wmedcomp': 'wmdcmp',
    'medtagf1': 'mtf1',
    'orienacc': 'oracc',
    'chxlabelacc': 'chxlacc',
    'chxlabelf1': 'chxlf1',
    'qlabelsf1': 'qlf1',
    'question_loss': 'q_loss',
    'answer_loss': 'a_loss',
    'orientation_loss': 'orien_loss',
    'chexpert_loss': 'chx_loss',
    'qlabels_loss': 'ql_loss',
}

IUXRAY_DATASET_ID = 0
MIMICCXR_DATASET_ID = 1

class ReportEvalMode(Enum):
    GROUND_TRUTH = 'ground-truth'
    MOST_POPULAR = 'most-popular'
    QUESTION_CLASSIFICATION = 'question-classification'
    NEAREST_NEIGHBOR = 'nearest-neighbor'