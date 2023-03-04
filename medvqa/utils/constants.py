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

CXR14_LABELS = [
    'No Finding',
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
]

CXR14_LABEL2SHORT = {
    'No Finding': 'NF',
    'Atelectasis' : 'A',
    'Cardiomegaly' : 'Card',
    'Consolidation' : 'Cons',
    'Edema' : 'Ed',
    'Effusion': 'Ef',
    'Emphysema': 'Emp',
    'Fibrosis': 'Fib',
    'Hernia': 'Her',
    'Infiltration': 'Inf',
    'Mass': 'Mass',
    'Nodule': 'Nod',
    'Pleural_Thickening' : 'Pthi',
    'Pneumonia': 'Pn',
    'Pneumothorax': 'Ptho',
}

VINBIG_DISEASES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification',
    'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema',
    'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity',
    'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
    'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD',
    'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other disease',
    'No finding',
]

CHEXPERT_CXR14_SYNONYMS = [
    ('No Finding', 'No Finding'),
    ('Cardiomegaly', 'Cardiomegaly'),
    ('Edema', 'Edema'),
    ('Consolidation', 'Consolidation'),
    ('Pneumonia', 'Pneumonia'),
    ('Atelectasis', 'Atelectasis'),
    ('Pneumothorax', 'Pneumothorax'),
    ('Pleural Effusion', 'Effusion'),
]

CHEXPERT_VINBIG_SYNONYMS = [
    ('No Finding', 'No finding'),
    ('Cardiomegaly', 'Cardiomegaly'),
    ('Lung Opacity', 'Lung Opacity'),
    ('Edema', 'Edema'),
    ('Consolidation', 'Consolidation'),
    ('Pneumonia', 'Pneumonia'),
    ('Atelectasis', 'Atelectasis'),
    ('Pneumothorax', 'Pneumothorax'),
    ('Pleural Effusion', 'Pleural effusion'),
]

CXR14_VINBIG_SYNONYMS = [
    ('No Finding', 'No finding'),
    ('Atelectasis', 'Atelectasis'),
    ('Cardiomegaly', 'Cardiomegaly'),
    ('Consolidation', 'Consolidation'),
    ('Edema', 'Edema'),
    ('Effusion', 'Pleural effusion'),
    ('Emphysema', 'Emphysema'),
    ('Fibrosis', 'Pulmonary fibrosis'),
    ('Infiltration', 'Infiltration'),
    ('Pleural_Thickening', 'Pleural thickening'),
    ('Pneumonia', 'Pneumonia'),
    ('Pneumothorax', 'Pneumothorax'),
]

CHEXPERT_GENDERS = ['Female', 'Male']
CHEXPERT_ORIENTATIONS = ['FrontalAP', 'Lateral', 'FrontalPA']
CHEXPERT_ORIENTATION2ID = {
    'FrontalAP': 0,
    'Lateral': 1,
    'FrontalPA': 2
}
CHEXPERT_GENDER2ID = {
    'Female': 0,
    'Male': 1,
}

CXR14_GENDERS = ['F', 'M']
CXR14_ORIENTATIONS = ['AP', 'PA']
CXR14_ORIENTATION2ID = {
    'AP': 0,
    'PA': 2 # make consistent with chexpert
}
CXR14_GENDER2ID = {
    'F': 0,
    'M': 1,
}

PADCHEST_GENDERS = ['F', 'M']
PADCHEST_GENDER2ID = {
    'F': 0,
    'M': 1,
}
PADCHEST_PROJECTIONS = ['PA', 'AP', 'AP_horizontal', 'L', 'COSTAL']
PADCHEST_NUM_LABELS = 193
PADCHEST_NUM_QUESTIONS = 193 + 3 # 193 for each label, 1 for all labels, 1 for localizations, 1 for all labels and localizations
PADCHEST_NUM_LOCALIZATIONS = 104

class CHEXPERT_TASKS:
    CLASSIFICATION = 'classification'
    VQA = 'vqa'

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
    LOSS = 'loss'
    CHEXPERT_ACCURACY = 'chexpert_accuracy'
    CHEXPERT_PRF1S = 'chexpert_prf1s'
    EXACTMATCH_QUESTION = 'exactmatch_question'
    EXACTMATCH_ANSWER = 'exactmatch_answer'
    BLEU_QUESTION = 'bleu_question'
    BLEU_BACKGROUND = 'bleu_background'
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
    VINBIGMACROAVGF1 = 'vinbigmacroavgf1'
    VINBIGMICROAVGF1 = 'vinbigmicroavgf1'
    CXR14MACROAVGF1 = 'cxr14_macroavgf1'
    CXR14MICROAVGF1 = 'cxr14_microavgf1'
    PADCHEST_LABEL_MACROAVGF1 = 'padchest_label_macroavgf1'
    PADCHEST_LABEL_MICROAVGF1 = 'padchest_label_microavgf1'
    PADCHEST_LOC_MACROAVGF1 = 'padchest_loc_macroavgf1'
    PADCHEST_LOC_MICROAVGF1 = 'padchest_loc_microavgf1'
    CHESTIMAGENOMELABELACC = 'chestimagenome_label_acc'
    CHESTIMAGENOMELABELMACROAVGF1 = 'chestimagenome_label_macroavgf1'
    CHESTIMAGENOMELABELMICROAVGF1 = 'chestimagenome_label_microavgf1'
    CHESTIMAGENOMELABELROCAUC = 'chestimagenome_label_rocauc'
    CHESTIMAGENOMELABELROCAUC_MICRO = 'chestimagenome_label_rocauc_micro'
    CHESTIMAGENOMELABELROCAUC_MACRO = 'chestimagenome_label_rocauc_macro'
    CHESTIMAGENOMELABEL_PRF1 = 'chestimagenome_label_prf1'
    CHESTIMAGENOMEBBOXIOU = 'chestimagenome_bbox_iou'
    CHESTIMAGENOMEBBOXMAE = 'chestimagenome_bbox_mae'
    CHESTIMAGENOMEBBOXMEANF1 = 'chestimagenome_bbox_meanf1'
    QUESTION_LOSS = 'question_loss'
    ANSWER_LOSS = 'answer_loss'
    BACKGROUND_LOSS = 'background_loss'
    ORIENTATION_LOSS = 'orientation_loss'
    CHEXPERT_LOSS = 'chexpert_loss'
    VINBIG_LOSS = 'vinbig_loss'
    CXR14_LOSS = 'cxr14_loss'
    PADCHEST_LABEL_LOSS = 'padchest_label_loss'
    PADCHEST_LOCALIZATION_LOSS = 'padchest_loc_loss'
    CHEST_IMAGENOME_LABEL_LOSS = 'chestimagenome_label_loss'
    CHEST_IMAGENOME_BBOX_LOSS = 'chestimagenome_bbox_loss'
    DETECTRON2_CLS_LOSS = 'detectron2_cls_loss'
    DETECTRON2_BOX_REG_LOSS = 'detectron2_box_reg_loss'
    DETECTRON2_RPN_CLS_LOSS = 'detectron2_rpn_cls_loss'
    DETECTRON2_RPN_LOC_LOSS = 'detectron2_rpn_loc_loss'
    QLABELS_LOSS = 'qlabels_loss'
    MEDTAGS_LOSS = 'medtags_loss'
    GENDER_LOSS = 'gender_loss'
    GENDER_ACC = 'gender_acc'    

METRIC2SHORT = {
    'loss': 'loss',
    'chexpert_accuracy': 'chx_acc',
    'chexpert_prf1s': 'chx_prf1s',
    'exactmatch_question': 'emq',
    'exactmatch_answer': 'ema',
    'bleu_question': 'bq',
    'bleu_background': 'b_bg',
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
    'vinbigmacroavgf1': 'vnbgmacf1',
    'vinbigmicroavgf1': 'vnbgmicf1',
    'cxr14_macroavgf1': 'cxr14macf1',
    'cxr14_microavgf1': 'cxr14micf1',
    'padchest_label_macroavgf1': 'padchxlmacf1',
    'padchest_label_microavgf1': 'padchxlmicf1',
    'padchest_loc_macroavgf1': 'padchxlzmacf1',
    'padchest_loc_microavgf1': 'padchxlzmicf1',
    'padchest_label_loss': 'padchxl_loss',
    'padchest_loc_loss': 'padchxlz_loss',
    'chestimagenome_label_loss': 'chestimgl_loss',
    'chestimagenome_label_acc': 'chestimgl_acc',
    'chestimagenome_label_macroavgf1': 'chestimglmacf1',
    'chestimagenome_label_microavgf1': 'chestimglmicf1',
    'chestimagenome_label_rocauc': 'chestimglrocauc',
    'chestimagenome_label_rocauc_micro': 'chestimglrocaucmic',
    'chestimagenome_label_rocauc_macro': 'chestimglrocaucmac',
    'chestimagenome_label_prf1': 'chestimglprf1',
    'chestimagenome_bbox_iou': 'chestimgbbiou',
    'chestimagenome_bbox_mae': 'chestimgbbmae',
    'chestimagenome_bbox_meanf1': 'chestimgbbmf1',
    'chestimagenome_bbox_loss': 'chestimgbb_loss',
    'detectron2_cls_loss': 'd2cls_loss',
    'detectron2_box_reg_loss': 'd2box_loss',
    'detectron2_rpn_cls_loss': 'd2rpncls_loss',
    'detectron2_rpn_loc_loss': 'd2rpnloc_loss',
    'question_loss': 'q_loss',
    'answer_loss': 'a_loss',
    'background_loss': 'bg_loss',
    'orientation_loss': 'orien_loss',
    'chexpert_loss': 'chx_loss',
    'vinbig_loss': 'vnbg_loss',
    'cxr14_loss': 'cxr14_loss',
    'qlabels_loss': 'ql_loss',
    'gender_loss': 'gloss',
    'gender_acc': 'gacc',
}

IUXRAY_DATASET_ID = 0
MIMICCXR_DATASET_ID = 1
CHEXPERT_DATASET_ID = 2
IUXRAY_DATASET_ID__CHEXPERT_MODE = 3
MIMICCXR_DATASET_ID__CHEXPERT_MODE = 4
VINBIG_DATASET_ID = 5
CXR14_DATASET_ID = 6
PADCHEST_DATASET_ID = 7
MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE = 8
MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE = 9

class DATASET_NAMES:
    IUXRAY = 'iuxray'
    MIMICCXR = 'mimic-cxr'
    CHEXPERT = 'chexpert'
    IUXRAY_CHEXPERT_MODE = 'iuxray-chexpert-mode'
    MIMICCXR_CHEXPERT_MODE = 'mimic-cxr-chexpert-mode'
    VINBIG = 'vinbig'
    CXR14 = 'cxr14'
    PADCHEST = 'padchest'
    MIMICCXR_CHEST_IMAGENOME_MODE = 'mimic-cxr-chest-imagenome-mode'
    MIMICCXR_CHEST_IMAGENOME__DETECTRON2_MODE = 'mimic-cxr-chest-imagenome-detectron2-mode'

DATASET_NAME_TO_SHORT = {
    DATASET_NAMES.IUXRAY: 'iux',
    DATASET_NAMES.MIMICCXR: 'mim',
    DATASET_NAMES.CHEXPERT: 'chx',
    DATASET_NAMES.IUXRAY_CHEXPERT_MODE: 'iux-chx',
    DATASET_NAMES.MIMICCXR_CHEXPERT_MODE: 'mim-chx',
    DATASET_NAMES.VINBIG: 'vnbg',
    DATASET_NAMES.CXR14: 'cxr14',
    DATASET_NAMES.PADCHEST: 'padch',
    DATASET_NAMES.MIMICCXR_CHEST_IMAGENOME_MODE: 'mim-chestimg',
    DATASET_NAMES.MIMICCXR_CHEST_IMAGENOME__DETECTRON2_MODE: 'mim-chestimg-d2',
}

class ReportEvalMode:
    GROUND_TRUTH = 'ground-truth'
    MOST_POPULAR = 'most-popular'
    QUESTION_CLASSIFICATION = 'question-classification'
    NEAREST_NEIGHBOR = 'nearest-neighbor'
    CHEXPERT_LABELS = 'chexpert-labels'
    CHEXPERT_AND_QUESTION_CLASSIFICATION = 'chexpert+qclass'
    VINBIG_DISEASES = 'vinbig-diseases'

def get_enum_class_attribute_values(enum_class):
    for name, value in vars(enum_class).items():
        if not name.startswith('__'):
            yield value