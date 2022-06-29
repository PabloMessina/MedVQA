from enum import Enum

CHEXPERT_LABELS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',    
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',    
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
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

CHEXPERT_GENDERS = ['Female', 'Male']

CHEXPERT_ORIENTATIONS = ['FrontalAP', 'Lateral', 'FrontalPA']

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

class MetricNames:
    CHEXPERT_ACCURACY = 'chexpert_accuracy'
    CHEXPERT_PRF1S = 'chexpert_prf1s'
    BLEU_QUESTION = 'bleu_question'
    EXACTMATCH_QUESTION = 'exactmatch_question'
    BLEU = 'bleu'
    BLEU_1 = 'bleu-1'
    BLEU_2 = 'bleu-2'
    BLEU_3 = 'bleu-3'
    BLEU_4 = 'bleu-4'
    ROUGE_L = 'rougeL'
    METEOR = 'meteor'
    CIDER_D = 'ciderD'
    MEDCOMP = 'medcomp'
    WMEDCOMP = 'wmedcomp'
    MEDTAGF1 = 'medtagf1'
    ORIENACC = 'orienacc'
    CHXLABELACC = 'chxlabelacc'
    CHXLABELF1 = 'chxlabelf1'
    CHXLABELMACROAVGF1 = 'chxlabelmacroavgf1'
    CHXLABELMICROAVGF1 = 'chxlabelmicroavgf1'
    CHXLABEL_ROCAUC = 'chxlabel_rocauc'
    CHXLABEL_ROCAUC_MICRO = 'chxlabel_rocauc_micro'
    CHXLABEL_ROCAUC_MACRO = 'chxlabel_rocauc_macro'
    CHXLABEL_PRF1 = 'chxlabel_prf1'
    QLABELSF1 = 'qlabelsf1'
    QLABELS_PRF1 = 'qlabels_prf1'
    QLABELS_MACROAVGF1 = 'qlabels_macroavgf1'
    QLABELS_MICROAVGF1 = 'qlabels_microavgf1'
    QUESTION_LOSS = 'question_loss'
    ANSWER_LOSS = 'answer_loss'
    ORIENTATION_LOSS = 'orientation_loss'
    CHEXPERT_LOSS = 'chexpert_loss'
    QLABELS_LOSS = 'qlabels_loss'
    MEDTAGS_LOSS = 'medtags_loss'
    GENDER_LOSS = 'gender_loss'
    GENDER_ACC = 'gender_acc'

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
    'chxlabelmacroavgf1': 'chxlmacf1',
    'chxlabelmicroavgf1': 'chxlmicf1',
    'chxlabel_rocauc': 'chxlrocauc',
    'chxlabel_rocauc_micro': 'chxlrocaucmic',
    'chxlabel_rocauc_macro': 'chxlrocaucmac',
    'chxlabel_prf1': 'chxlprf1',
    'qlabelsf1': 'qlf1',
    'qlabels_prf1': 'qlprf1',
    'qlabels_macroavgf1': 'qlmacf1',
    'qlabels_microavgf1': 'qlmicf1',
    'question_loss': 'q_loss',
    'answer_loss': 'a_loss',
    'orientation_loss': 'orien_loss',
    'chexpert_loss': 'chx_loss',
    'qlabels_loss': 'ql_loss',
    'gender_loss': 'gloss',
    'gender_acc': 'gacc',
}

IUXRAY_DATASET_ID = 0
MIMICCXR_DATASET_ID = 1
CHEXPERT_DATASET_ID = 2

class ReportEvalMode(Enum):
    GROUND_TRUTH = 'ground-truth'
    MOST_POPULAR = 'most-popular'
    QUESTION_CLASSIFICATION = 'question-classification'
    NEAREST_NEIGHBOR = 'nearest-neighbor'