import numpy as np

CHEXPERT_LABELS = [ # Note: I'm annotating roughly equivalent labels in the Chest ImaGenome dataset
    'No Finding', # NOT abnormal
    'Enlarged Cardiomediastinum', # mediastinal widening
    'Cardiomegaly', # enlarged cardiac silhouette
    'Lung Lesion', # lung lesion
    'Lung Opacity', # lung opacity
    'Edema', # pulmonary edema/hazy opacity
    'Consolidation', # consolidation
    'Pneumonia', # pneumonia
    'Atelectasis', # atelectasis
    'Pneumothorax', # pneumothorax
    'Pleural Effusion', # pleural effusion
    'Pleural Other', # pleural/parenchymal scarring
    'Fracture', # spinal fracture, rib fracture, bone lesion
    'Support Devices', # cardiac pacer and wires, cabg grafts, prosthetic valve,
    # chest tube, swan-ganz catheter, enteric tube, endotracheal tube, tracheostomy tube,
    # subclavian line, picc, chest port, ij line
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

CHEXPERT_LABEL2PHRASE = {
    'No Finding': 'no findings',
    'Cardiomegaly': 'cardiomegaly seen',
    'Enlarged Cardiomediastinum': 'enlarged cardiomediastinum seen',
    'Consolidation': 'consolidation seen',
    'Lung Opacity': 'lung opacity seen',
    'Atelectasis': 'atelectasis seen',
    'Support Devices': 'support devices seen',
    'Pleural Effusion': 'pleural effusion seen',
    'Pleural Other': 'other pleural abnormality seen',
    'Pneumonia': 'pneumonia seen',
    'Pneumothorax': 'pneumothorax seen',
    'Edema': 'edema seen',
    'Lung Lesion': 'lung lesion seen',
    'Fracture': 'fracture seen',
}

CHEXBERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices", "No Finding",
]

CHEXBERT_LABELS_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
CHEXPERT_LABELS_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

CHEXBERT_LABELS_5_INDICES = np.where(np.isin(CHEXBERT_LABELS, CHEXBERT_LABELS_5))[0]
CHEXPERT_LABELS_5_INDICES = np.where(np.isin(CHEXPERT_LABELS, CHEXPERT_LABELS_5))[0]

# See https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/1.1.0/
CXRLT2023_CLASSES = [
    'Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture',
    'Hernia', 'Infiltration', 'Lung Lesion', 'Lung Opacity', 'Mass',
    'No Finding', 'Nodule', 'Pleural Effusion', 'Pleural Other',
    'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia',
    'Pneumoperitoneum', 'Pneumothorax', 'Subcutaneous Emphysema',
    'Support Devices', 'Tortuous Aorta',
]
assert len(CXRLT2023_CLASSES) == 26

# See https://bionlplab.github.io/2024_MICCAI_CXRLT/
CXRLT2024_CLASSES = [
    # With labels
    'Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta', 'Cardiomegaly', 'Clavicle Fracture',
    'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fissure', 'Fracture', 'Granuloma',
    'Hernia', 'Hydropneumothorax', 'Infarction', 'Infiltration', 'Kyphosis', 'Lobar Atelectasis', 'Lung Lesion',
    'Lung Opacity', 'Mass', 'Nodule', 'Normal', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum',
    'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture',
    'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta', 'Tuberculosis',
    # Without labels
    'Bulla', 'Cardiomyopathy', 'Hilum', 'Osteopenia', 'Scoliosis',
]
assert len(CXRLT2024_CLASSES) == 45
for x in CXRLT2023_CLASSES:
    if x == 'No Finding': continue
    assert x in CXRLT2024_CLASSES, f'{x} not in CXRLT2024_CLASSES'

CXRLT2024_TASK1_CLASSES = [
    'Adenopathy','Atelectasis','Azygos Lobe','Calcification of the Aorta','Cardiomegaly','Clavicle Fracture',
    'Consolidation','Edema','Emphysema','Enlarged Cardiomediastinum','Fibrosis','Fissure','Fracture','Granuloma',
    'Hernia','Hydropneumothorax','Infarction','Infiltration','Kyphosis','Lobar Atelectasis','Lung Lesion',
    'Lung Opacity','Mass','Nodule','Normal','Pleural Effusion','Pleural Other','Pleural Thickening',
    'Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax','Pulmonary Embolism','Pulmonary Hypertension',
    'Rib Fracture','Round(ed) Atelectasis','Subcutaneous Emphysema','Support Devices','Tortuous Aorta','Tuberculosis',
]
assert len(CXRLT2024_TASK1_CLASSES) == 40
assert CXRLT2024_CLASSES[:40] == CXRLT2024_TASK1_CLASSES # make sure the first 40 classes are the same

CXRLT2024_TASK2_CLASSES = [
    'Atelectasis','Calcification of the Aorta','Cardiomegaly','Consolidation','Edema','Emphysema', 'Enlarged Cardiomediastinum',
    'Fibrosis','Fracture','Hernia','Infiltration','Lung Lesion','Lung Opacity','Mass','Normal','Nodule','Pleural Effusion',
    'Pleural Other','Pleural Thickening','Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax',
    'Subcutaneous Emphysema','Support Devices','Tortuous Aorta',
]
assert len(CXRLT2024_TASK2_CLASSES) == 26
assert all(x in CXRLT2024_CLASSES for x in CXRLT2024_TASK2_CLASSES)

CXRLT2024_TASK3_CLASSES = [
    'Bulla', 'Cardiomyopathy', 'Hilum', 'Osteopenia', 'Scoliosis',
]
assert len(CXRLT2024_TASK3_CLASSES) == 5
assert all(x in CXRLT2024_CLASSES for x in CXRLT2024_TASK3_CLASSES)

CXRLT2024_CLASS_2_SENTENCE = {
    'Adenopathy': 'adenopathy seen',
    'Atelectasis': 'atelectasis seen',
    'Azygos Lobe': 'azygos lobe seen',
    'Calcification of the Aorta': 'calcification of the aorta seen',
    'Cardiomegaly': 'cardiomegaly seen',
    'Clavicle Fracture': 'clavicle fracture seen',
    'Consolidation': 'consolidation seen',
    'Edema': 'edema seen',
    'Emphysema': 'emphysema seen',
    'Enlarged Cardiomediastinum': 'enlarged cardiomediastinum seen',
    'Fibrosis': 'fibrosis seen',
    'Fissure': 'fissure seen',
    'Fracture': 'fracture seen',
    'Granuloma': 'granuloma seen',
    'Hernia': 'hernia seen',
    'Hydropneumothorax': 'hydropneumothorax seen',
    'Infarction': 'infarction seen',
    'Infiltration': 'infiltration seen',
    'Kyphosis': 'kyphosis seen',
    'Lobar Atelectasis': 'lobar atelectasis seen',
    'Lung Lesion': 'lung lesion seen',
    'Lung Opacity': 'lung opacity seen',
    'Mass': 'mass seen',
    'Nodule': 'nodule seen',
    'Normal': 'no abnormalities seen',
    'Pleural Effusion': 'pleural effusion seen',
    'Pleural Other': 'pleural abnormalities seen',
    'Pleural Thickening': 'pleural thickening seen',
    'Pneumomediastinum': 'pneumomediastinum seen',
    'Pneumonia': 'pneumonia seen',
    'Pneumoperitoneum': 'pneumoperitoneum seen',
    'Pneumothorax': 'pneumothorax seen',
    'Pulmonary Embolism': 'pulmonary embolism seen',
    'Pulmonary Hypertension': 'pulmonary hypertension seen',
    'Rib Fracture': 'rib fracture seen',
    'Round(ed) Atelectasis': 'round atelectasis seen',
    'Subcutaneous Emphysema': 'subcutaneous emphysema seen',
    'Support Devices': 'support devices seen',
    'Tortuous Aorta': 'tortuous aorta seen',
    'Tuberculosis': 'tuberculosis seen',
    'Bulla': 'bulla seen',
    'Cardiomyopathy': 'cardiomyopathy seen',
    'Hilum': 'hilar abnormalities seen',
    'Osteopenia': 'osteopenia seen',
    'Scoliosis': 'scoliosis seen',
}
CXRLT2024_SENTENCE_2_CLASS = { v: k for k, v in CXRLT2024_CLASS_2_SENTENCE.items() }
assert len(CXRLT2024_CLASS_2_SENTENCE) == len(CXRLT2024_SENTENCE_2_CLASS)

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

VINBIG_LABELS = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Clavicle fracture',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Enlarged PA',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Lung cavity',
    'Lung cyst',
    'Mediastinal shift',
    'Nodule/Mass',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'Rib fracture',
    'Other lesion',
    'COPD',
    'Lung tumor',
    'Pneumonia',
    'Tuberculosis',
    'Other disease',
    'No finding',
]

VINBIG_BBOX_NAMES = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Clavicle fracture',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Enlarged PA',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Lung cavity',
    'Lung cyst',
    'Mediastinal shift',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'Rib fracture',
]
assert all([l in VINBIG_LABELS for l in VINBIG_BBOX_NAMES])

VINBIG_LABEL2PHRASE = {
    'Aortic enlargement': 'aortic enlargement seen',
    'Atelectasis': 'atelectasis seen',
    'Calcification': 'calcification seen',
    'Cardiomegaly': 'cardiomegaly seen',
    'Clavicle fracture': 'clavicle fracture seen',
    'Consolidation': 'consolidation seen',
    'Edema': 'edema seen',
    'Emphysema': 'emphysema seen',
    'Enlarged PA': 'enlarged pulmonary artery seen',
    'ILD': 'interstitial lung disease seen',
    'Infiltration': 'infiltration seen',
    'Lung Opacity': 'lung opacity seen',
    'Lung cavity': 'lung cavity seen',
    'Lung cyst': 'lung cyst seen',
    'Mediastinal shift': 'mediastinal shift seen',
    'Nodule/Mass': 'nodule/mass seen',
    'Other lesion': 'other lesion seen',
    'Pleural effusion': 'pleural effusion seen',
    'Pleural thickening': 'pleural thickening seen',
    'Pneumothorax': 'pneumothorax seen',
    'Pulmonary fibrosis': 'pulmonary fibrosis seen',
    'Rib fracture': 'rib fracture seen',
    'Other disease': 'other disease seen',
    'COPD': 'copd seen',
    'Lung tumor': 'lung tumor seen',
    'Pneumonia': 'pneumonia seen',
    'Tuberculosis': 'tuberculosis seen',
    'Other disease': 'other disease seen',
    'No finding': 'no abnormalities seen',
}
assert all([l in VINBIG_LABEL2PHRASE for l in VINBIG_LABELS])
assert all([l in VINBIG_LABELS for l in VINBIG_LABEL2PHRASE])



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

CHEST_IMAGENOME_GENDERS = ['F', 'M', 'UNK']
CHEST_IMAGENOME_GENDER2ID = {'F': 0, 'M': 1, 'UNK': 2}

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
    CIDER_D_GT = 'ciderD_gt'
    MEDCOMP = 'medcomp'
    WMEDCOMP = 'wmedcomp'
    WMEDCOMP_GT = 'wmedcomp_gt'
    MEDTAGF1 = 'medtagf1'
    ORIENACC = 'orienacc'
    CHXLABELACC = 'chxlabelacc'
    CHXLABELF1 = 'chxlabelf1'
    CHXLABELMACROAVGF1 = 'chxlabelmacroavgf1'
    CHXLABELMICROAVGF1 = 'chxlabelmicroavgf1'
    CHXLABEL_ROCAUC = 'chxlabel_rocauc'
    CHXLABEL_ROCAUC_MICRO = 'chxlabel_rocauc_micro'
    CHXLABEL_ROCAUC_MACRO = 'chxlabel_rocauc_macro'
    CHXLABEL_AUC = 'chxlabel_auc'
    CHXLABEL_AUC_MICRO = 'chxlabel_auc_micro'
    CHXLABEL_AUC_MACRO = 'chxlabel_auc_macro'
    CHXLABEL_PRCAUC = 'chxlabel_prcauc'
    CHXLABEL_PRCAUC_MICRO = 'chxlabel_prcauc_micro'
    CHXLABEL_PRCAUC_MACRO = 'chxlabel_prcauc_macro'
    CHXLABEL_PRF1 = 'chxlabel_prf1'
    QLABELSF1 = 'qlabelsf1'
    QLABELS_PRF1 = 'qlabels_prf1'
    QLABELS_MACROAVGF1 = 'qlabels_macroavgf1'
    QLABELS_MICROAVGF1 = 'qlabels_microavgf1'
    VINBIGMACROAVGF1 = 'vinbigmacroavgf1'
    VINBIGMICROAVGF1 = 'vinbigmicroavgf1'
    VINBIGBBOXIOU = 'vinbig_bbox_iou'
    VINBIGBBOXMEANF1 = 'vinbig_bbox_meanf1'
    VINBIGLABELAUC = 'vinbig_label_auc'
    VINBIGLABELAUC_MICRO = 'vinbig_label_auc_micro'
    VINBIGLABELAUC_MACRO = 'vinbig_label_auc_macro'
    VINBIGLABELPRCAUC = 'vinbig_label_prcauc'
    VINBIGLABELPRCAUC_MICRO = 'vinbig_label_prcauc_micro'
    VINBIGLABELPRCAUC_MACRO = 'vinbig_label_prcauc_macro'
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
    CHESTIMAGENOMELABELAUC = 'chestimagenome_label_auc'
    CHESTIMAGENOMELABELAUC_MICRO = 'chestimagenome_label_auc_micro'
    CHESTIMAGENOMELABELAUC_MACRO = 'chestimagenome_label_auc_macro'
    CHESTIMAGENOMELABELPRCAUC = 'chestimagenome_label_prcauc'
    CHESTIMAGENOMELABELPRCAUC_MICRO = 'chestimagenome_label_prcauc_micro'
    CHESTIMAGENOMELABELPRCAUC_MACRO = 'chestimagenome_label_prcauc_macro'
    CHESTIMAGENOMELABEL_PRF1 = 'chestimagenome_label_prf1'
    CHESTIMAGENOMEBBOXIOU = 'chestimagenome_bbox_iou'
    CHESTIMAGENOMEBBOXMAE = 'chestimagenome_bbox_mae'
    CHESTIMAGENOMEBBOXMEANF1 = 'chestimagenome_bbox_meanf1'
    QUESTION_LOSS = 'question_loss'
    ANSWER_LOSS = 'answer_loss'
    BACKGROUND_LOSS = 'background_loss'
    ORIENTATION_LOSS = 'orientation_loss'
    CHEXPERT_LOSS = 'chexpert_loss'
    VINBIG_LABEL_LOSS = 'vinbig_label_loss'
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
    YOLOV8_LOSS = 'yolov8_loss'
    YOLOV8_BOX_LOSS = 'yolov8_box_loss'
    YOLOV8_CLS_LOSS = 'yolov8_cls_loss'
    YOLOV8_DFL_LOSS = 'yolov8_dfl_loss'
    REPORT_LOSS = 'report_loss'
    REPORT_LOSS_GT = 'report_loss_gt'
    LOCAL_FEATURE_COORDS_LOSS = 'local_feature_coords_loss'
    SEQ2SEQ_LOSS = 'seq2seq_loss'

METRIC2SHORT = {
    'loss': 'loss',
    'report_loss': 'rpt_loss',
    'report_loss_gt': 'rpt_loss_gt',
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
    'ciderD_gt': 'cD_gt',
    'medcomp': 'mdcmp',
    'wmedcomp': 'wmdcmp',
    'wmedcomp_gt': 'wmdcmp_gt',
    'medtagf1': 'mtf1',
    'orienacc': 'oracc',
    'chxlabelacc': 'chxlacc',
    'chxlabelf1': 'chxlf1',
    'chxlabelmacroavgf1': 'chxlmacf1',
    'chxlabelmicroavgf1': 'chxlmicf1',
    'chxlabel_rocauc': 'chxlrocauc',
    'chxlabel_rocauc_micro': 'chxlrocaucmic',
    'chxlabel_rocauc_macro': 'chxlrocaucmac',
    'chxlabel_auc': 'chxlauc',
    'chxlabel_auc_micro': 'chxlaucmic',
    'chxlabel_auc_macro': 'chxlaucmac',
    'chxlabel_prcauc': 'chxlprcauc',
    'chxlabel_prcauc_micro': 'chxlprcaucmic',
    'chxlabel_prcauc_macro': 'chxlprcaucmac',
    'chxlabel_prf1': 'chxlprf1',
    'qlabelsf1': 'qlf1',
    'qlabels_prf1': 'qlprf1',
    'qlabels_macroavgf1': 'qlmacf1',
    'qlabels_microavgf1': 'qlmicf1',
    'vinbigmacroavgf1': 'vnbgmacf1',
    'vinbigmicroavgf1': 'vnbgmicf1',
    'vinbig_bbox_iou': 'vnbgbbiou',
    'vinbig_label_auc': 'vnbglauc',
    'vinbig_label_auc_micro': 'vnbglaucmic',
    'vinbig_label_auc_macro': 'vnbglaucmac',
    'vinbig_label_prcauc': 'vnbgprcauc',
    'vinbig_label_prcauc_micro': 'vnbgprcaucmic',
    'vinbig_label_prcauc_macro': 'vnbgprcaucmac',
    'vinbig_bbox_meanf1': 'vnbgbbmf1',
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
    'chestimagenome_label_auc': 'chestimglauc',
    'chestimagenome_label_auc_micro': 'chestimglaucmic',
    'chestimagenome_label_auc_macro': 'chestimglaucmac',
    'chestimagenome_label_prcauc': 'chestimglprcauc',
    'chestimagenome_label_prcauc_micro': 'chestimglprcaucmic',
    'chestimagenome_label_prcauc_macro': 'chestimglprcaucmac',
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
    'vinbig_label_loss': 'vnbgl_loss',
    'cxr14_loss': 'cxr14_loss',
    'qlabels_loss': 'ql_loss',
    'gender_loss': 'gloss',
    'gender_acc': 'gacc',
    'yolov8_loss': 'y8_loss',
    'yolov8_box_loss': 'y8box_loss',
    'yolov8_cls_loss': 'y8cls_loss',
    'yolov8_dfl_loss': 'y8dfl_loss',
    'local_feature_coords_loss': 'lfcoords_loss',
    'seq2seq_loss': 's2s_loss',
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
    CHEXLOCALIZE = 'chexlocalize'

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

LABEL_BASED_FACTS = [
    'atelectasis seen', 'calcification of the aorta seen', 'cardiomegaly seen', 'consolidation seen',
    'edema seen', 'emphysema seen', 'enlarged cardiomediastinum seen', 'fibrosis seen', 'fracture seen',
    'hernia seen', 'infiltration seen', 'lung lesion seen', 'lung opacity seen', 'mass seen', 'no abnormalities seen',
    'nodule seen', 'pleural effusion seen', 'pleural abnormalities seen', 'pleural thickening seen',
    'pneumomediastinum seen', 'pneumonia seen', 'pneumoperitoneum seen', 'pneumothorax seen',
    'subcutaneous emphysema seen', 'support devices seen', 'tortuous aorta seen', 'airspace opacity seen',
    'atelectasis seen', 'bone lesion seen', 'bronchiectasis seen', 'calcified nodule seen', 'clavicle fracture seen',
    'consolidation seen', 'costophrenic angle blunting seen', 'cyst/bullae seen',
    'diaphragmatic eventration (benign) seen', 'elevated hemidiaphragm seen', 'enlarged cardiac silhouette seen',
    'enlarged hilum seen', 'hernia seen', 'hydropneumothorax seen', 'hyperaeration seen',
    'increased reticular markings/ild pattern seen', 'infiltration seen', 'linear/patchy atelectasis seen',
    'lobar/segmental collapse seen', 'lung lesion seen', 'lung opacity seen', 'mass/nodule (not otherwise specified) seen',
    'mediastinal displacement seen', 'mediastinal widening seen', 'multiple masses/nodules seen', 'pleural effusion seen',
    'pleural/parenchymal scarring seen', 'pneumomediastinum seen', 'pneumothorax seen',
    'pulmonary edema/hazy opacity seen', 'rib fracture seen', 'scoliosis seen', 'shoulder osteoarthritis seen',
    'spinal degenerative changes seen', 'spinal fracture seen', 'sub-diaphragmatic air seen', 'subcutaneous air seen',
    'superior mediastinal mass/enlargement seen', 'tortuous aorta seen', 'vascular calcification seen',
    'vascular congestion seen', 'vascular redistribution seen', 'aortic graft/repair seen', 'cabg grafts seen',
    'cardiac pacer and wires seen', 'prosthetic valve seen', 'alveolar hemorrhage seen', 'aspiration seen',
    'copd/emphysema seen', 'fluid overload/heart failure seen', 'goiter seen', 'granulomatous disease seen',
    'interstitial lung disease seen', 'lung cancer seen', 'pericardial effusion seen', 'pneumonia seen', 'artifact seen',
    'breast/nipple shadows seen', 'low lung volumes seen', 'rotated seen', 'skin fold seen', 'alveolar texture seen',
    'calcified texture seen', 'interstitial texture seen', 'opacity texture seen', 'chest port seen', 'chest tube seen',
    'endotracheal tube seen', 'enteric tube seen', 'ij line seen', 'intra-aortic balloon pump seen',
    'mediastinal drain seen', 'picc seen', 'pigtail catheter seen', 'subclavian line seen', 'swan-ganz catheter seen',
    'tracheostomy tube seen'
]
# deduplication
LABEL_BASED_FACTS = list(set(LABEL_BASED_FACTS))
LABEL_BASED_FACTS.sort()

LABEL_BASED_FACTS__CHEXPERT = [
    'no abnormalities seen', # No Finding
    'enlarged cardiomediastinum seen', # Enlarged Cardiomediastinum
    'cardiomegaly seen', # Cardiomegaly
    'lung lesion seen', # Lung Lesion
    'lung opacity seen', # Lung Opacity
    'edema seen', # Edema
    'consolidation seen', # Consolidation
    'pneumonia seen', # Pneumonia
    'atelectasis seen', # Atelectasis
    'pneumothorax seen', # Pneumothorax
    'pleural effusion seen', # Pleural Effusion
    'pleural abnormalities seen', # Pleural Other
    'fracture seen', # Fracture
    'support devices seen', # Support Devices
]
assert len(LABEL_BASED_FACTS__CHEXPERT) == len(CHEXPERT_LABELS)
assert all(l in LABEL_BASED_FACTS for l in LABEL_BASED_FACTS__CHEXPERT)

LABEL_BASED_FACTS__CHEXPERT_2_SHORT = {
    'no abnormalities seen': 'NoAb',
    'enlarged cardiomediastinum seen': 'ECM',
    'cardiomegaly seen': 'Card',
    'lung lesion seen': 'LL',
    'lung opacity seen': 'LO',
    'edema seen': 'Ed',
    'consolidation seen': 'Cons',
    'pneumonia seen': 'Pn',
    'atelectasis seen': 'A',
    'pneumothorax seen': 'Ptho',
    'pleural effusion seen': 'Ef',
    'pleural abnormalities seen': 'PleAb',
    'fracture seen': 'Frac',
    'support devices seen': 'SD',
}
assert all(l in LABEL_BASED_FACTS__CHEXPERT_2_SHORT for l in LABEL_BASED_FACTS__CHEXPERT)
assert all(l in LABEL_BASED_FACTS__CHEXPERT for l in LABEL_BASED_FACTS__CHEXPERT_2_SHORT)
    

LABEL_BASED_FACTS__MIMIC_CXR_LT = [
    'atelectasis seen', # Atelectasis
    'calcification of the aorta seen', # Calcification of the Aorta
    'cardiomegaly seen', # Cardiomegaly
    'consolidation seen', # Consolidation
    'edema seen', # Edema
    'emphysema seen', # Emphysema
    'enlarged cardiomediastinum seen', # Enlarged Cardiomediastinum
    'fibrosis seen', # Fibrosis
    'fracture seen', # Fracture
    'hernia seen', # Hernia
    'infiltration seen', # Infiltration
    'lung lesion seen', # Lung Lesion
    'lung opacity seen', # Lung Opacity
    'mass seen', # Mass
    'no abnormalities seen', # No Finding
    'nodule seen', # Nodule
    'pleural effusion seen', # Pleural Effusion
    'pleural abnormalities seen', # Pleural Other
    'pleural thickening seen', # Pleural Thickening
    'pneumomediastinum seen', # Pneumomediastinum
    'pneumonia seen', # Pneumonia
    'pneumoperitoneum seen', # Pneumoperitoneum
    'pneumothorax seen', # Pneumothorax
    'subcutaneous emphysema seen', # Subcutaneous Emphysema
    'support devices seen', # Support Devices
    'tortuous aorta seen', # Tortuous Aorta
]
assert len(LABEL_BASED_FACTS__MIMIC_CXR_LT) == len(CXRLT2023_CLASSES)
assert all(l in LABEL_BASED_FACTS for l in LABEL_BASED_FACTS__MIMIC_CXR_LT)

LABEL_BASED_FACTS__MIMIC_CXR_LT_2_SHORT = {
    'atelectasis seen': 'A',
    'calcification of the aorta seen': 'CoA',
    'cardiomegaly seen': 'Card',
    'consolidation seen': 'Cons',
    'edema seen': 'Ed',
    'emphysema seen': 'Emp',
    'enlarged cardiomediastinum seen': 'ECM',
    'fibrosis seen': 'Fib',
    'fracture seen': 'Frac',
    'hernia seen': 'Hern',
    'infiltration seen': 'Inf',
    'lung lesion seen': 'LL',
    'lung opacity seen': 'LO',
    'mass seen': 'Mass',
    'no abnormalities seen': 'NoAb',
    'nodule seen': 'Nod',
    'pleural effusion seen': 'Ef',
    'pleural abnormalities seen': 'PleAb',
    'pleural thickening seen': 'PleTh',
    'pneumomediastinum seen': 'PnMd',
    'pneumonia seen': 'Pn',
    'pneumoperitoneum seen': 'PnPt',
    'pneumothorax seen': 'Ptho',
    'subcutaneous emphysema seen': 'SEmp',
    'support devices seen': 'SD',
    'tortuous aorta seen': 'TA',
}
assert all(l in LABEL_BASED_FACTS__MIMIC_CXR_LT_2_SHORT for l in LABEL_BASED_FACTS__MIMIC_CXR_LT)
assert all(l in LABEL_BASED_FACTS__MIMIC_CXR_LT for l in LABEL_BASED_FACTS__MIMIC_CXR_LT_2_SHORT)

LABEL_BASED_FACTS__CHEST_IMAGENOME = [
    'airspace opacity seen', # Airspace Opacity
    'atelectasis seen', # Atelectasis
    'bone lesion seen', # Bone Lesion
    'bronchiectasis seen', # Bronchiectasis
    'calcified nodule seen', # Calcified Nodule
    'clavicle fracture seen', # Clavicle Fracture
    'consolidation seen', # Consolidation
    'costophrenic angle blunting seen', # Costophrenic Angle Blunting
    'cyst/bullae seen', # Cyst/Bullae
    'diaphragmatic eventration (benign) seen', # Diaphragmatic Eventration (Benign)
    'elevated hemidiaphragm seen', # Elevated Hemidiaphragm
    'enlarged cardiac silhouette seen', # Enlarged Cardiac Silhouette
    'enlarged hilum seen', # Enlarged Hilum
    'hernia seen', # Hernia
    'hydropneumothorax seen', # Hydropneumothorax
    'hyperaeration seen', # Hyperaeration
    'increased reticular markings/ild pattern seen', # Increased Reticular Markings/ILD Pattern 
    'infiltration seen', # Infiltration
    'linear/patchy atelectasis seen', # Linear/Patchy Atelectasis
    'lobar/segmental collapse seen', # Lobar/Segmental Collapse
    'lung lesion seen', # Lung Lesion
    'lung opacity seen', # Lung Opacity
    'mass/nodule (not otherwise specified) seen', # Mass/Nodule (Not Otherwise Specified)
    'mediastinal displacement seen', # Mediastinal Displacement
    'mediastinal widening seen', # Mediastinal Widening
    'multiple masses/nodules seen', # Multiple Masses/Nodules
    'pleural effusion seen', # Pleural Effusion
    'pleural/parenchymal scarring seen', # Pleural/Parenchymal Scarring
    'pneumomediastinum seen', # Pneumomediastinum
    'pneumothorax seen', # Pneumothorax
    'pulmonary edema/hazy opacity seen', # Pulmonary Edema/Hazy Opacity
    'rib fracture seen', # Rib Fracture
    'scoliosis seen', # Scoliosis
    'shoulder osteoarthritis seen', # Shoulder Osteoarthritis
    'spinal degenerative changes seen', # Spinal Degenerative Changes
    'spinal fracture seen', # Spinal Fracture
    'sub-diaphragmatic air seen', # Sub-Diaphragmatic Air
    'subcutaneous air seen', # Subcutaneous Air
    'superior mediastinal mass/enlargement seen', # Superior Mediastinal Mass/Enlargement
    'tortuous aorta seen', # Tortuous Aorta
    'vascular calcification seen', # Vascular Calcification
    'vascular congestion seen', # Vascular Congestion
    'vascular redistribution seen', # Vascular Redistribution
    'aortic graft/repair seen', # Aortic Graft/Repair
    'cabg grafts seen', # CABG Grafts
    'cardiac pacer and wires seen', # Cardiac Pacer and Wires
    'prosthetic valve seen', # Prosthetic Valve
    'alveolar hemorrhage seen', # Alveolar Hemorrhage
    'aspiration seen', # Aspiration
    'copd/emphysema seen', # COPD/Emphysema
    'fluid overload/heart failure seen', # Fluid Overload/Heart Failure
    'goiter seen', # Goiter
    'granulomatous disease seen', # Granulomatous Disease
    'interstitial lung disease seen', # Interstitial Lung Disease
    'lung cancer seen', # Lung Cancer
    'pericardial effusion seen', # Pericardial Effusion
    'pneumonia seen', # Pneumonia
    'artifact seen', # Artifact
    'breast/nipple shadows seen', # Breast/Nipple Shadows
    'low lung volumes seen', # Low Lung Volumes
    'rotated seen', # Rotated
    'skin fold seen', # Skin Fold
    'alveolar texture seen', # Alveolar Texture
    'calcified texture seen', # Calcified Texture
    'interstitial texture seen', # Interstitial Texture
    'opacity texture seen', # Opacity Texture
    'chest port seen', # Chest Port
    'chest tube seen', # Chest Tube
    'endotracheal tube seen', # Endotracheal Tube
    'enteric tube seen', # Enteric Tube
    'ij line seen', # IJ Line
    'intra-aortic balloon pump seen', # Intra-Aortic Balloon Pump
    'mediastinal drain seen', # Mediastinal Drain
    'picc seen', # PICC
    'pigtail catheter seen', # Pigtail Catheter
    'subclavian line seen', # Subclavian Line
    'swan-ganz catheter seen', # Swan-Ganz Catheter
    'tracheostomy tube seen', # Tracheostomy Tube
]
assert all(l in LABEL_BASED_FACTS for l in LABEL_BASED_FACTS__CHEST_IMAGENOME)

LABEL_BASED_FACTS__CHEST_IMAGENOME_2_SHORT = {
    'airspace opacity seen': 'AO',
    'atelectasis seen': 'A',
    'bone lesion seen': 'BL',
    'bronchiectasis seen': 'B',
    'calcified nodule seen': 'CN',
    'clavicle fracture seen': 'CF',
    'consolidation seen': 'Cons',
    'costophrenic angle blunting seen': 'CAB',
    'cyst/bullae seen': 'CB',
    'diaphragmatic eventration (benign) seen': 'DE',
    'elevated hemidiaphragm seen': 'EHemi',
    'enlarged cardiac silhouette seen': 'ECS',
    'enlarged hilum seen': 'Ehil',
    'hernia seen': 'Her',
    'hydropneumothorax seen': 'HPtho',
    'hyperaeration seen': 'HA',
    'increased reticular markings/ild pattern seen': 'IRM',
    'infiltration seen': 'Inf',
    'linear/patchy atelectasis seen': 'LPA',
    'lobar/segmental collapse seen': 'LSC',
    'lung lesion seen': 'LL',
    'lung opacity seen': 'LO',
    'mass/nodule (not otherwise specified) seen': 'MN',
    'mediastinal displacement seen': 'MDis',
    'mediastinal widening seen': 'MW',
    'multiple masses/nodules seen': 'MM',
    'pleural effusion seen': 'Ef',
    'pleural/parenchymal scarring seen': 'PPS',
    'pneumomediastinum seen': 'PM',
    'pneumothorax seen': 'Ptho',
    'pulmonary edema/hazy opacity seen': 'PEHO',
    'rib fracture seen': 'RF',
    'scoliosis seen': 'S',
    'shoulder osteoarthritis seen': 'SO',
    'spinal degenerative changes seen': 'SDC',
    'spinal fracture seen': 'SFrac',
    'sub-diaphragmatic air seen': 'SDA',
    'subcutaneous air seen': 'SA',
    'superior mediastinal mass/enlargement seen': 'SMM',
    'tortuous aorta seen': 'TA',
    'vascular calcification seen': 'VCal',
    'vascular congestion seen': 'VCon',
    'vascular redistribution seen': 'VR',
    'aortic graft/repair seen': 'AGR',
    'cabg grafts seen': 'CG',
    'cardiac pacer and wires seen': 'CPW',
    'prosthetic valve seen': 'PV',
    'alveolar hemorrhage seen': 'AH',
    'aspiration seen': 'Asp',
    'copd/emphysema seen': 'CE',
    'fluid overload/heart failure seen': 'FO',
    'goiter seen': 'G',
    'granulomatous disease seen': 'GD',
    'interstitial lung disease seen': 'ILD',
    'lung cancer seen': 'LC',
    'pericardial effusion seen': 'PE',
    'pneumonia seen': 'Pn',
    'artifact seen': 'Art',
    'breast/nipple shadows seen': 'BNS',
    'low lung volumes seen': 'LLV',
    'rotated seen': 'Rot',
    'skin fold seen': 'SFo',
    'alveolar texture seen': 'AT',
    'calcified texture seen': 'CaT',
    'interstitial texture seen': 'IntT',
    'opacity texture seen': 'OT',
    'chest port seen': 'CP',
    'chest tube seen': 'CT',
    'endotracheal tube seen': 'EndT',
    'enteric tube seen': 'EntT',
    'ij line seen': 'IJ',
    'intra-aortic balloon pump seen': 'IABP',
    'mediastinal drain seen': 'MDr',
    'picc seen': 'P',
    'pigtail catheter seen': 'PC',
    'subclavian line seen': 'SL',
    'swan-ganz catheter seen': 'SG',
    'tracheostomy tube seen': 'TT',
}
for x in LABEL_BASED_FACTS__CHEST_IMAGENOME:
    assert x in LABEL_BASED_FACTS__CHEST_IMAGENOME_2_SHORT, x
for x in LABEL_BASED_FACTS__CHEST_IMAGENOME_2_SHORT:
    assert x in LABEL_BASED_FACTS__CHEST_IMAGENOME, x
_unique = set()
for x in LABEL_BASED_FACTS__CHEST_IMAGENOME_2_SHORT.values():
    assert x not in _unique, x
    _unique.add(x)