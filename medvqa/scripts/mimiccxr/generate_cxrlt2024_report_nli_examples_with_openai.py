import os
import re
import sys
import math
import argparse
import random
import numpy as np
import pandas as pd
from collections import Counter
from medvqa.datasets.mimiccxr.report_utils import concatenate_report_parts
from medvqa.evaluation.plots import plot_metrics
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.constants import CXRLT2024_CLASS_2_SENTENCE, CXRLT2024_CLASSES, CXRLT2024_SENTENCE_2_CLASS, CXRLT2024_TASK1_CLASSES
from medvqa.utils.hashing import compute_hashes_in_parallel, hash_string
from medvqa.utils.logging import get_console_logger, print_orange
from medvqa.datasets.mimiccxr import (
    MIMIC_CXR_LT_2024_TASK1_DEV_CSV_PATH,
    MIMIC_CXR_LT_2024_TASK1_TRAIN_CSV_PATH,
    MIMICCXR_LARGE_FAST_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
    get_imageId2PartPatientStudy,
    get_imageId2reportId,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.openai_api import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files import load_jsonl, load_pickle, read_txt, save_pickle

INSTRUCTIONS = """Given a report (#R) and a hypothesis (#H), output "Reason: {reason}. Label: {label}" where {reason} is one or two short explanation sentences and {label} is one of {definitely true, likely true, unknown, likely false, definitely false}. Be careful with tricky sentences that mention multiple findings. Remember that unknown applies when both #R and #H might be true but there is no conclusive way to know with the information provided."""

POSSIBLE_LABELS = [
    "label: definitely true",
    "label: likely true",
    "label: unknown",
    "label: likely false",
    "label: definitely false",
]

LABEL_TO_BINARY = {
    "definitely true": 1,
    "likely true": 1,
    "unknown": 0,
    "likely false": 0,
    "definitely false": 0,
}

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    original_text = text
    text = text.lower()
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    assert text.startswith("reason: "), f"No reason found in output: {text}"
    for label in POSSIBLE_LABELS:
        try:
            idx = text.index(label)
            assert idx > 8, f"idx: {idx}, label: {label}, text: {text}"
            reason = original_text[8:idx].strip()
            label = label[7:] # Remove "label: "
            return {
                "reason": reason,
                "label": label,
            }
        except ValueError:
            continue
    raise RuntimeError(f"Could not parse output: {text}")

# Mapping from CXR-LT labels to similar labels in the Chest ImaGenome dataset
_CXRLT2024_LABEL_2_CHEST_IMAGENOME_LABELS = {
    'Adenopathy': [], # No similar label
    'Atelectasis': ['atelectasis', 'linear/patchy atelectasis'],
    'Azygos Lobe': [], # No similar label
    'Calcification of the Aorta': ['vascular calcification', 'calcified'],
    'Cardiomegaly': ['enlarged cardiac silhouette'],
    'Clavicle Fracture': ['clavicle fracture'],
    'Consolidation': ['consolidation'],
    'Edema': ['pulmonary edema/hazy opacity'],
    'Emphysema': ['copd/emphysema'],
    'Enlarged Cardiomediastinum': ['enlarged cardiac silhouette', 'superior mediastinal mass/enlargement'],
    'Fibrosis': [], # No similar label
    'Fissure': [], # No similar label
    'Fracture': ['clavicle fracture', 'rib fracture', 'spinal fracture'],
    'Granuloma': ['granulomatous disease'],
    'Hernia': ['hernia'],
    'Hydropneumothorax': ['hydropneumothorax', 'pneumothorax'],
    'Infarction': [], # No similar label
    'Infiltration': ['infiltration'],
    'Kyphosis': [], # No similar label
    'Lobar Atelectasis': ['atelectasis', 'linear/patchy atelectasis'],
    'Lung Lesion': ['lung lesion'],
    'Lung Opacity': ['lung opacity', 'pulmonary edema/hazy opacity', 'opacity'],
    'Mass': ['mass/nodule (not otherwise specified)', 'multiple masses/nodules'],
    'Nodule': ['calcified nodule', 'mass/nodule (not otherwise specified)', 'multiple masses/nodules'],
    'Normal': ['abnormal'], # NOTE: REMEMBER TO NEGATE
    'Pleural Effusion': ['pleural effusion'],
    'Pleural Other': ['pleural effusion', 'pleural/parenchymal scarring'],
    'Pleural Thickening': [], # No similar label
    'Pneumomediastinum': ['pneumomediastinum'],
    'Pneumonia': ['pneumonia'],
    'Pneumoperitoneum': [], # No similar label
    'Pneumothorax': ['pneumothorax', 'hydropneumothorax'],
    'Pulmonary Embolism': [], # No similar label
    'Pulmonary Hypertension': [], # No similar label
    'Rib Fracture': ['rib fracture'],
    'Round(ed) Atelectasis': [], # No similar label
    'Subcutaneous Emphysema': ['copd/emphysema'],
    'Support Devices': ['chest port', 'chest tube', 'endotracheal tube', 'enteric tube', 'ij line',
                        'intra-aortic balloon pump', 'mediastinal drain', 'picc', 'pigtail catheter',
                        'subclavian line', 'swan-ganz catheter', 'tracheostomy tube',
                        'aortic graft/repair', 'cabg grafts', 'cardiac pacer and wires', 'prosthetic valve'],
    'Tortuous Aorta': ['tortuous aorta'],
    'Tuberculosis': [], # No similar label
    'Bulla': ['cyst/bullae'],
    'Cardiomyopathy': [], # No similar label
    'Hilum': ['enlarged hilum'],
    'Osteopenia': [], # No similar label
    'Scoliosis': ['scoliosis', 'spinal degenerative changes'],
}

# Mapping from CXR-LT labels to regular expressions
_SUPPORT_DEVICE_SUBSTRINGS = [
    r'\bcath',
    r'\btube(s)?\b',
    r'\btubing\b',
    r'\btubular structure\b'
    r'\bsideport',
    r'\bport\b',
    r'mediport\b',
    r'\bport-?a-?cath',
    r'\bPICC\b',
    r'\b(?:left|right|central|PIC|jugular|EJ|sided|subclavian|tip|course|position|support)\b.*\blines?\b',
    r'\b(?:line|the) tips?\b',
    r'\btip\b.*\b(?:lies|terminates|projects|is)\b',
    r'pacemaker',
    r'\blead\b',
    r'\b(?:dual|pace|maker|devic|correc|positi|stand|place|aicd|termin|cours|overl).*\bleads\b',
    r'\bleads\b.*\b(?:dual|pace|maker|devic|correc|positi|stand|place|aicd|termin|cours|overl)',
    r'\bstent(?:s|ing)?\b',
    r'\bCVL\b',
    r'\bETT\b',
    r'\bNGT\b',
    r'\bIJ\b',
    r'\bintroducer\b',
    r'\bcuffs?\b',
    r'internal jugular vein sheath',
    r'Dobbhoff',
    r'endotracheal',
    r'\bdual\b.*\blumen\b',
    r'\bkink(?:ing|ed)?\b',
    r'\bintubat',
    r'\bogt?\b',
    r'\bptbd\b',
    r'\bstaple\b',
    r'\bprobes?\b',
    r'\bvp\b',
    r'\bhardware',
    r'embolization',
    r'\bclips?\b',
    r'\bcoils?\b',
    r'\bdevices?\b',
    r'\bequipment',
    r'\bpens?\b',
    r'\bwires?\b',
    r'\bmetal(lic)?\b',
    r'\bdrains?\b',
    r'\bdrainage',
    r'\banchor\b',
    r'\bpacers?\b',
    r'\bpacing\b',
    r'\boral contrast',
    r'\bscattered contrast',
    r'contrast agent',
    r'\bballoon\b',
    r'\bpump\b',
    r'nipple (?:marker|shadow|ring)',
    r'\bornament',
    r'valve ring',
    r'\bshadow.*\bnipple',
    r'\bgenerator',
    r'\bpiercing',
    r'\bTIPS\b',
    r'\bshunt(?:s|ing)?\b',
    r'\breplace.*\bvalv',
    r'\bvalv.*\breplace',
    r'endobronchial valve',
    r'\b(?:lap|gastric)\b.*\bband',
    r'\bAICD\b',
    r'\bICD\b',
    r'\bLVAD\b',
    r'prosthe(?:sis|ses|tic)',
    r'\bfilters?\b',
    r'\bstructure.*\b(?:courses|project)',
    r'projecting.*\bstructure',
    r'projectile',
    r'\bstimulator',
    r'\bimplant',
    r'defibrillator',
    r'\bjewelry\b',
    r'\bbarium\b',
    r'\bfecal\b',
    r'\bstool\b',
    r'\bbullets?\b',
    r'foreign bod(?:y|ies)',
    r'\bscrews?\b',
    r'\bgallstone',
    r'\bgown\b',
    r'\bcables?\b',
    r'appliances',
    r'\bavr\b',
    r'battery',
    r'\bbbs?\b',
    r'\bcbd stent',
    r'\bcement\b',
    r'\bcerclage\b',
    r'\bchains?\b',
    r'\bexpander',
    r'\bmask\b',
    r'\bgraft',
    r'grafts?\b',
    r'\bhair(?:pin)?',
    r'\bpins?\b',
    r'neurostimulator',
    r'\bpack',
    r'\bpellet',
    r'\bpills?\b',
    r'\bporcine',
    r'\brecorder\b',
    r'\brods?\b',
    r'\bsaline\b',
    r'\bscratch',
    r'shrapnel',
    r'\bspacer',
    r'\bsponge',
    r'\bspoon',
    r'\bspring',
    r'stylet',
    r'\bsu(?:tur|rg).*\bmaterial',
    r'\btacks\b',
    r'\btooth\b',
    r'\bunit\b',
    r'\bvns\b',
    r'\bmonitor\b',
    r'\bfiducial',
]

_CXRLT2024_LABEL_2_REGEX = {
    'Adenopathy': re.compile(r'adenopathy', re.IGNORECASE),
    'Atelectasis': re.compile(r'atelecta(?:sis|ses|tic|is)|atalectasis|atelctasis|atelecasis|atelecatsis|ateletasis|atlectasis', re.IGNORECASE),
    'Azygos Lobe': re.compile(r'azygou?s lobe', re.IGNORECASE),
    'Calcification of the Aorta': re.compile(r'(\bcalcifi.*\s+.*\baort|\baort.*\s+.*\bcalcifi)', re.IGNORECASE),
    'Cardiomegaly': re.compile(r'cardiomegaly', re.IGNORECASE),
    'Clavicle Fracture': re.compile(r'(\b(?:clavic|collarbone).*\s+\bfractur|\bfractur.*\s+\b(?:clavic|collarbone))', re.IGNORECASE),
    'Consolidation': re.compile(r'\bconsolidat', re.IGNORECASE),
    'Edema': re.compile(r'(\bedema\b|anasarca)', re.IGNORECASE),
    'Emphysema': re.compile(r'emphysema', re.IGNORECASE),
    'Enlarged Cardiomediastinum': re.compile(r'\b(?:cardio|para)?mediasti.*\b(?:silhouette|contour|margin|border|(ab)?normal|wide(ned)?|enlarge?ment|shift|structure|surface|shadow|configuration|(post)?radiation|venous|vascul|vein)|\b(shift|convex).*mediasti|mediastinum', re.IGNORECASE),
    'Fibrosis': re.compile(r'\bfibro[cnst]', re.IGNORECASE),
    'Fissure': re.compile(r'\bfissure', re.IGNORECASE),
    'Fracture': re.compile(r'\bfracture|compression deformit|vertebroplast', re.IGNORECASE),
    'Granuloma': re.compile(r'\bgranuloma', re.IGNORECASE),
    'Hernia': re.compile(r'\bhernia', re.IGNORECASE),
    'Hydropneumothorax': re.compile(r'hydropneumothor', re.IGNORECASE),
    'Infarction': re.compile(r'\binfarct', re.IGNORECASE),
    'Infiltration': re.compile(r'\binfiltrat', re.IGNORECASE),
    'Kyphosis': re.compile(r'\bkypho', re.IGNORECASE),
    'Lobar Atelectasis': re.compile(r'\b(?:lobar|lobe|segmental)\s+atelectasis', re.IGNORECASE),
    'Lung Lesion': re.compile(r'\b(?:lung|pulmonary).*(?:\blesion|hematoma|\bcontusion|\binjur[yi]|\bwound)|(?:\blesion|hematoma|\bcontusion|\binjur[yi]|\bwound).*(?:\blung|pulmonary)', re.IGNORECASE),
    'Lung Opacity': re.compile(r'(?:\b(?:radi)?opaci[tf]|\b(?:radio?)?opaque).*\b(?:lung|pulmon|lobe|lobular)|\b(?:lung|pulmon|lobe|lobular).*(?:\b(?:radi)?opaci[tf]|\b(?:radio?)?opaque)', re.IGNORECASE),
    'Mass': re.compile(r'\b(?:mass|masses|tumor|tumors|lesion|lesions|neoplasm|neoplasms|growth|growths|cyst|cysts)\b', re.IGNORECASE),
    'Nodule': re.compile(r'\b(?:nodule|nodules|tumor|tumors|lesion|lesions|neoplasm|neoplasms|growth|growths|cyst|cysts)\b', re.IGNORECASE),
    'Pleural Effusion': re.compile(r'\beffusion', re.IGNORECASE),
    'Pleural Thickening': re.compile(r'\bpleur(?:a|al)?.*\bthicken(?:ing|ed)?\b|\bthicken(?:ing|ed)?.*\bpleur(?:a|al)', re.IGNORECASE),
    'Pneumomediastinum': re.compile(r'pneumomediastinum|mediastinal.*(?:emphysema|\bair\b)', re.IGNORECASE),
    'Pneumonia': re.compile(r'pn(?:eu|ue)mon|\bpna\b|penumonia', re.IGNORECASE),
    'Pneumoperitoneum': re.compile(r'pneumoperitoneum', re.IGNORECASE),
    'Pneumothorax': re.compile(r'deep sulcus sign|pnemothorax|pneumothorax|pneumothroax|pneumothoraces|pnuemothorax|\bptx\b', re.IGNORECASE),
    'Pulmonary Embolism': re.compile(r'\b(?:pulmonary\s+embol|PE)\b', re.IGNORECASE),
    'Pulmonary Hypertension': re.compile(r'\b(?:Pulmonary\s+Hypertension|PH|PHTN|PAH)\b', re.IGNORECASE),
    'Rib Fracture': re.compile(r'\brib\.*\s+fractur|\bfractur\.*\s+rib', re.IGNORECASE),
    'Round(ed) Atelectasis': re.compile(r'\bround(?:ed)?\s+atelectasis\b', re.IGNORECASE),
    'Subcutaneous Emphysema': re.compile(r'\bsubcutaneous\s+emphysema\b', re.IGNORECASE),
    'Support Devices': re.compile(r'|'.join(_SUPPORT_DEVICE_SUBSTRINGS), re.IGNORECASE),
    'Tortuous Aorta': re.compile(r'\btortuos', re.IGNORECASE),
    'Tuberculosis': re.compile(r'(\btb\b|tubercul)', re.IGNORECASE),
    'Bulla': re.compile(r'\b(?:bulla|bullae|bullectomy|bullous|cyst|cystic|cysts)\b', re.IGNORECASE),
    'Cardiomyopathy': re.compile(r'cardiomyopathy', re.IGNORECASE),
    'Hilum': re.compile(r'\b(?:peri)?(?:hila|hilar|hill|hilum|hilus)', re.IGNORECASE),
    'Osteopenia': re.compile(r'osteopen', re.IGNORECASE),
    'Scoliosis': re.compile(r'scolio(?:tic|sis)', re.IGNORECASE),
}

def _build_report(metadata, idx):
    background = metadata['backgrounds'][idx]
    findings = metadata['findings'][idx]
    impression = metadata['impressions'][idx]
    return concatenate_report_parts(background, findings, impression)

def _collect_candidate_study_ids_per_class(
        chest_imagenome_image_id_to_labels_filepath,
        chest_imagenome_label_names_filepath,
        mimiccxr_background_findings_and_impression_per_report_filepath,
    ):
    """
    Collect candidate study IDs for each class in the CXR-LT 2024 dataset.
    """

    save_path = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "cxrlt2024_candidate_study_ids_per_class.pkl")
    if os.path.exists(save_path):
        logger.info(f"Loading candidate study IDs per class from {save_path}")
        return load_pickle(save_path)

    output = {}

    assert os.path.exists(chest_imagenome_image_id_to_labels_filepath)
    assert os.path.exists(chest_imagenome_label_names_filepath)
    cimgn_imageId2labels = load_pickle(chest_imagenome_image_id_to_labels_filepath)
    cimgn_label_names = load_pickle(chest_imagenome_label_names_filepath)
    cimgn_class2idx = { x[1]: i for i, x in enumerate(cimgn_label_names) if len(x) == 2 }
    logger.info(f"Loaded {len(cimgn_imageId2labels)} image IDs from the Chest ImaGenome dataset")
    logger.info(f"Loaded {len(cimgn_label_names)} label names from the Chest ImaGenome dataset")
    logger.info(f"Loaded {len(cimgn_class2idx)} classes from the Chest ImaGenome dataset")
    metadata = load_mimiccxr_reports_detailed_metadata(
        background_findings_and_impression_per_report_filepath=mimiccxr_background_findings_and_impression_per_report_filepath)
    n_reports = len(metadata['backgrounds'])
    reports = [_build_report(metadata, i) for i in range(n_reports)]
    logger.info(f"Loaded {n_reports} reports from the MIMIC-CXR dataset")

    imageId2PartPatientStudy = get_imageId2PartPatientStudy()

    train_df = pd.read_csv(MIMIC_CXR_LT_2024_TASK1_TRAIN_CSV_PATH)
    
    for class_name in CXRLT2024_CLASSES:

        output[class_name] = {}

        logger.info(f"Processing class: {class_name}")

        # Find reports already pre-labeled with the class in the official training set
        if class_name in train_df.columns:
            study_ids = train_df[train_df[class_name] == 1]['study_id'].unique()
            assert len(study_ids) > 0
            assert study_ids[0][0] == 's'
            study_ids = [int(study_id[1:]) for study_id in study_ids] # Remove 's' prefix
            logger.info(f"Found {len(study_ids)} reports already labeled with {class_name} in the official training set")
            output[class_name]['CXRLT2024'] = study_ids
        else:
            logger.warning(f"No reports already labeled with {class_name} in the official training set")

        # Find reports already pre-labeled with the class in the Chest ImaGenome dataset
        cimgn_label_names = _CXRLT2024_LABEL_2_CHEST_IMAGENOME_LABELS[class_name]
        if len(cimgn_label_names) > 0:
            image_ids = set()
            for cimgn_label_name in cimgn_label_names:
                assert cimgn_label_name in cimgn_class2idx
                class_idx = cimgn_class2idx[cimgn_label_name]
                if cimgn_label_name == "abnormal":
                    check_func = lambda labels: labels[class_idx] != 1 # Remember to negate
                else:
                    check_func = lambda labels: labels[class_idx] == 1
                for image_id, labels in cimgn_imageId2labels.items():
                    if check_func(labels):
                        image_ids.add(image_id) # Add image ID
            assert len(image_ids) > 0
            study_ids = set()
            for image_id in image_ids:
                study_id = imageId2PartPatientStudy[image_id][-1] # Get study ID
                study_id = int(study_id)
                study_ids.add(study_id)
            logger.info(f"Found {len(study_ids)} reports already labeled with {class_name} in the Chest ImaGenome dataset")
            study_ids = list(study_ids)
            output[class_name]['ChestImageNome'] = study_ids
        else:
            logger.warning(f"No reports already labeled with {class_name} in the Chest ImaGenome dataset")

        # Find reports matching regular expressions for the class
        if class_name in _CXRLT2024_LABEL_2_REGEX:
            regex = _CXRLT2024_LABEL_2_REGEX[class_name]
            study_ids = []
            for study_id, report in zip(metadata['study_ids'], reports):
                if regex.search(report):
                    study_ids.append(study_id)
            assert len(study_ids) > 0
            logger.info(f"Found {len(study_ids)} reports matching the regular expression for {class_name}")
            output[class_name]['Regex'] = study_ids
        else:
            logger.warning(f"No regular expression found for {class_name}")

    # Save the output
    save_pickle(output, save_path)
    logger.info(f"Saved the output to {save_path}")

    return output

def _build_query(report, hypothesis):
    return f"#R: {report} | #H: {hypothesis}"

def sample_queries_label_based(
        num_samples, split,
        chest_imagenome_image_id_to_labels_filepath,
        chest_imagenome_label_names_filepath,
        mimiccxr_background_findings_and_impression_per_report_filepath,
        already_processed_queries,
        specific_classes=None,
        specific_dicom_ids_filepath=None,
        pos_weight=0.85, # 85% positive samples by default
    ):
    """
    Sample queries based on label-based facts.
    """

    assert 0 < pos_weight < 1

    metadata = load_mimiccxr_reports_detailed_metadata(
        background_findings_and_impression_per_report_filepath=mimiccxr_background_findings_and_impression_per_report_filepath)
    n_reports = len(metadata['backgrounds'])
    reports = [_build_report(metadata, i) for i in range(n_reports)]
    study_id_to_report = { study_id: report for study_id, report in zip(metadata['study_ids'], reports) }
    assert len(study_id_to_report) == n_reports

    assert split in ['train', 'dev']

    if specific_dicom_ids_filepath is not None:
        assert os.path.exists(specific_dicom_ids_filepath)
        assert specific_dicom_ids_filepath.endswith('.csv')
        specific_dicom_ids = pd.read_csv(specific_dicom_ids_filepath)['dicom_id'].values
        logger.info(f"Loaded {len(specific_dicom_ids)} specific DICOM IDs")
        specific_dicom_ids = set(specific_dicom_ids)
        specific_study_ids = set()
        for dicom_id_view_pos_pairs, study_id in zip(metadata['dicom_id_view_pos_pairs'], metadata['study_ids']):
            for dicom_id, _ in dicom_id_view_pos_pairs:
                if dicom_id in specific_dicom_ids:
                    specific_study_ids.add(study_id)
                
        logger.info(f"Found {len(specific_study_ids)} specific study IDs")
        allowed_study_ids = specific_study_ids # Only use specific study IDs
    else:
        if split == 'train':
            train_df = pd.read_csv(MIMIC_CXR_LT_2024_TASK1_TRAIN_CSV_PATH)
            allowed_study_ids = train_df['study_id'].unique()
            assert allowed_study_ids[0][0] == 's'
            allowed_study_ids = set(int(s[1:]) for s in allowed_study_ids) # Remove 's' prefix
        elif split == 'dev':
            allowed_study_ids = set(metadata['study_ids'])
            train_df = pd.read_csv(MIMIC_CXR_LT_2024_TASK1_TRAIN_CSV_PATH)
            for study_id in train_df['study_id'].unique():
                study_id = int(study_id[1:]) # Remove 's' prefix
                if study_id in allowed_study_ids:
                    allowed_study_ids.remove(study_id) # Remove study ID in the train set
        else: assert False
        logger.info(f"Loaded {len(allowed_study_ids)} allowed study IDs from the {split} set")
 
    sampled_queries = []

    candidate_study_ids_per_class = _collect_candidate_study_ids_per_class(
        chest_imagenome_image_id_to_labels_filepath=chest_imagenome_image_id_to_labels_filepath,
        chest_imagenome_label_names_filepath=chest_imagenome_label_names_filepath,
        mimiccxr_background_findings_and_impression_per_report_filepath=mimiccxr_background_findings_and_impression_per_report_filepath,
    )

    if specific_classes is not None:
        assert len(specific_classes) > 0
        assert len(specific_classes) == len(set(specific_classes)) # No duplicates
        assert all([class_name in CXRLT2024_CLASSES for class_name in specific_classes]) # All classes are valid
        classes_to_process = specific_classes
    else:
        classes_to_process = CXRLT2024_CLASSES

    num_samples_per_class = math.ceil(num_samples / len(classes_to_process))
    num_pos_samples_per_class = math.ceil(num_samples_per_class * pos_weight) # Positive samples
    num_neg_samples_per_class = max(num_samples_per_class - num_pos_samples_per_class, 1) # At least 1 negative sample
    logger.info(f"Sampling {num_samples} queries ({num_pos_samples_per_class} positive, {num_neg_samples_per_class} negative) per class")

    for class_name in classes_to_process:

        class_sentence = CXRLT2024_CLASS_2_SENTENCE[class_name]
            
        logger.info(f"Processing class: {class_name}")

        len_before = len(sampled_queries)

        # Sample positive examples
        candidate_study_ids = candidate_study_ids_per_class[class_name]
        candidate_study_ids_list = []
        unique_candidate_study_ids = set()
        for key in ('CXRLT2024', 'ChestImageNome', 'Regex'):
            if key in candidate_study_ids:
                candidate_study_ids_ = set(candidate_study_ids[key])
                candidate_study_ids_ = candidate_study_ids_ & allowed_study_ids # Filter out study IDs not in the allowed set
                if len(candidate_study_ids_) == 0:
                    continue
                candidate_study_ids_ = list(candidate_study_ids_)
                candidate_study_ids_list.append(candidate_study_ids_)
                unique_candidate_study_ids.update(candidate_study_ids_)
        assert len(candidate_study_ids_list) > 0, f'No candidate study ids found for {class_name} ({candidate_study_ids.keys()})'
        num_pos_samples_per_source = math.ceil(num_pos_samples_per_class / len(candidate_study_ids_list))

        deficit = 0
        
        for candidate_study_ids_ in candidate_study_ids_list:
            num_samples_ = min(num_pos_samples_per_source, len(candidate_study_ids_))
            random.shuffle(candidate_study_ids_)
            count = 0
            for study_id in candidate_study_ids_:
                query = _build_query(study_id_to_report[study_id], class_sentence)
                query_hash = hash_string(query)
                if query_hash in already_processed_queries:
                    continue
                sampled_queries.append(query)
                already_processed_queries.add(query_hash)
                count += 1
                if count >= num_samples_: break
            deficit += num_pos_samples_per_source - count
        if deficit > 0: # Try to find missing positive examples from the union of all candidate study IDs
            unique_candidate_study_ids_list = list(unique_candidate_study_ids)
            random.shuffle(unique_candidate_study_ids_list)
            for study_id in unique_candidate_study_ids_list:
                query = _build_query(study_id_to_report[study_id], class_sentence)
                query_hash = hash_string(query)
                if query_hash in already_processed_queries:
                    continue
                sampled_queries.append(query)
                already_processed_queries.add(query_hash)
                deficit -= 1
                if deficit == 0: break

        if deficit > 0: # Warn if there is still a deficit
            logger.warning(f"A deficit of {deficit} positive examples for class {class_name}")

        # Sample negative examples
        count = 0
        max_tries = 10 * num_samples_
        tries = 0
        allowed_neg_study_ids = allowed_study_ids - unique_candidate_study_ids
        allowed_neg_study_ids_list = list(allowed_neg_study_ids)
        num_samples_ = min(num_neg_samples_per_class + deficit, len(allowed_neg_study_ids_list)) # Use the deficit to sample more negative examples
        while count < num_samples_:
            study_id = random.choice(allowed_neg_study_ids_list)
            query = _build_query(study_id_to_report[study_id], class_sentence)
            query_hash = hash_string(query)
            if query_hash in already_processed_queries:
                tries += 1
                if tries >= max_tries:
                    break
                continue
            sampled_queries.append(query)
            already_processed_queries.add(query_hash)
            count += 1

        len_after = len(sampled_queries)

        logger.info(f"Added {len_after - len_before} queries for class {class_name}")

    return sampled_queries

def integrate_nli_queries_for_fact_classification(
        mimiccxr_background_findings_and_impression_per_report_filepath,
        queries_to_integrate_filepaths,
        fact_encoder_model_name,
        fact_encoder_checkpoint_folder_path,
        batch_size,
        num_workers,
    ):
    """
    Integrate NLI queries for fact classification.
    """

    # Load queries to integrate
    queries_to_integrate = []
    for queries_to_integrate_filepath in queries_to_integrate_filepaths:
        rows = load_jsonl(queries_to_integrate_filepath)
        queries_to_integrate.extend(rows)
    logger.info(f"Loaded {len(queries_to_integrate)} queries to integrate")
    query_reports = []
    query_hypotheses = []
    query_labels = []
    for query_ in queries_to_integrate:
        query = query_['metadata']['query']
        label = query_['parsed_response']['label']
        parts = query.split(" | #H: ")
        assert len(parts) == 2
        assert parts[0].startswith("#R: ")
        query_reports.append(parts[0][4:])
        query_hypotheses.append(parts[1])
        query_labels.append(label)

    # Print label distribution
    logger.info(f"Label distribution: {Counter(query_labels)}")

    # Create mapping from reports to binary labels
    logger.info(f"Creating mapping from reports to binary labels")
    label2binary = {
        'definitely true': 1,
        'likely true': 1,
        'unknown': 0,
        'likely false': 0,
        'definitely false': 0,
    }
    report2labels = {}
    for report, query, label in zip(query_reports, query_hypotheses, query_labels):
        if report not in report2labels:
            report2labels[report] = ([], []) # Positive and negative labels
        label = label2binary[label]
        query = CXRLT2024_SENTENCE_2_CLASS[query]
        query = CXRLT2024_CLASSES.index(query)
        report2labels[report][label].append(query) # Add label

    # Load CXR-LT 2024 train and dev study IDs
    logger.info(f"Loading CXR-LT 2024 train and dev study IDs")
    train_df = pd.read_csv(MIMIC_CXR_LT_2024_TASK1_TRAIN_CSV_PATH)
    dev_df = pd.read_csv(MIMIC_CXR_LT_2024_TASK1_DEV_CSV_PATH)
    train_study_ids = train_df['study_id'].unique()
    train_study_ids = set(int(x[1:]) for x in train_study_ids) # Remove 's' prefix
    dev_study_ids = dev_df['study_id'].unique()
    dev_study_ids = set(int(x[1:]) for x in dev_study_ids) # Remove 's' prefix
    logger.info(f"Loaded {len(train_study_ids)} train study IDs and {len(dev_study_ids)} dev study IDs")

    # Map dicom ids to labels
    logger.info(f"Mapping dicom ids to labels")
    mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata(
        background_findings_and_impression_per_report_filepath=mimiccxr_background_findings_and_impression_per_report_filepath)
    output = {
        'train': {},
        'dev': {},
    }
    reports_that_matched = set()
    anomalous_study_ids = []
    for i, (dicom_id_view_pos_pairs, study_id) in enumerate(zip(
        mimiccxr_metadata['dicom_id_view_pos_pairs'],
        mimiccxr_metadata['study_ids'],
    )):
        report = _build_report(mimiccxr_metadata, i)
        try:
            labels = report2labels[report]
            reports_that_matched.add(report)
        except KeyError:
            continue
            
        if study_id in train_study_ids:
            aux = output['train']
        else:
            aux = output['dev']
            if study_id not in dev_study_ids:
                anomalous_study_ids.append(study_id) # It matches a report that was not found in the train or dev set

        for dicom_id, _ in dicom_id_view_pos_pairs:
            assert dicom_id not in aux
            aux[dicom_id] = labels # Add labels

    if len(anomalous_study_ids) > 0:
        logger.warning(f"Found {len(anomalous_study_ids)} anomalous study IDs")
        logger.warning(f"Example anomalous study IDs: {anomalous_study_ids[:5]}")
    
    logger.info(f"len(output['train']) = {len(output['train'])}")
    logger.info(f"len(output['dev']) = {len(output['dev'])}")
    logger.info(f"len(reports_that_matched) = {len(reports_that_matched)}")

    assert len(reports_that_matched) == len(report2labels)

    # Add class sentences to the output
    output['class_sentences'] = [CXRLT2024_CLASS_2_SENTENCE[x] for x in CXRLT2024_CLASSES]

    # Add class sentence embeddings to the output
    emb_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_encoder_model_name,
        model_checkpoint_folder_path=fact_encoder_checkpoint_folder_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    output['class_sentence_embeddings'] = emb_extractor.compute_text_embeddings(output['class_sentences'])
    
    logger.info(f"Computed class sentence embeddings")

    # Save the output
    save_path = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "cxrlt2024_integrated_nli_queries_for_fact_classification.pkl")
    save_pickle(output, save_path)
    logger.info(f"Saved integrated NLI queries for fact classification to {save_path}")

def eval_nli_queries_for_fact_classification(
        cxrlt2024_integrated_nli_queries_for_fact_classification_filepath,
    ):
    """
    Evaluate NLI queries for fact classification.
    """

    # Load integrated NLI queries for fact classification
    output = load_pickle(cxrlt2024_integrated_nli_queries_for_fact_classification_filepath)

    # Load CXR-LT 2024 train data
    train_df = pd.read_csv(MIMIC_CXR_LT_2024_TASK1_TRAIN_CSV_PATH)
    train_labels = train_df[CXRLT2024_TASK1_CLASSES].values
    train_dicom_ids = train_df['dicom_id'].values
    dicom_id_to_idx = { dicom_id: i for i, dicom_id in enumerate(train_dicom_ids) }

    not_found = 0
    tps = [ 0 for _ in CXRLT2024_TASK1_CLASSES ]
    fps = [ 0 for _ in CXRLT2024_TASK1_CLASSES ]
    fns = [ 0 for _ in CXRLT2024_TASK1_CLASSES ]
    tns = [ 0 for _ in CXRLT2024_TASK1_CLASSES ]

    mistakes_per_class = { x: [] for x in CXRLT2024_TASK1_CLASSES }
    for split in ['train', 'dev']:
        dicom_id_to_neg_pos_labels = output[split]
        for dicom_id, (neg_idxs, pos_idxs) in dicom_id_to_neg_pos_labels.items():
            if dicom_id not in dicom_id_to_idx:
                not_found += 1
                continue
            idx = dicom_id_to_idx[dicom_id]
            if len(neg_idxs) > 0:
                for neg_idx in neg_idxs:
                    if neg_idx >= len(CXRLT2024_TASK1_CLASSES): # Skip zero-shot classes
                        continue
                    if train_labels[idx, neg_idx] == 1: # False negative
                        mistakes_per_class[CXRLT2024_TASK1_CLASSES[neg_idx]].append((dicom_id, 'fn'))
                        fns[neg_idx] += 1
                    else: # True negative
                        tns[neg_idx] += 1
            if len(pos_idxs) > 0:
                for pos_idx in pos_idxs:
                    if pos_idx >= len(CXRLT2024_TASK1_CLASSES): # Skip zero-shot classes
                        continue
                    if train_labels[idx, pos_idx] == 1: # True positive
                        tps[pos_idx] += 1
                    else: # False positive
                        mistakes_per_class[CXRLT2024_TASK1_CLASSES[pos_idx]].append((dicom_id, 'fp'))
                        fps[pos_idx] += 1
    
    if not_found > 0:
        print_orange(f"{not_found} dicom ids not found in the train set", bold=True)

    total_mistakes = sum(len(x) for x in mistakes_per_class.values())
    if total_mistakes > 0:
        print_orange(f"Total mistakes: {total_mistakes}", bold=True)
    
    # Compute f1 per class
    f1s = []
    for tp, fp, fn in zip(tps, fps, fns):
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        f1s.append(f1)

    # Compute accuracy per class
    accs = []
    for tp, fp, fn, tn in zip(tps, fps, fns, tns):
        acc = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0
        accs.append(acc)

    names = [f"{x} (tp={tp}, tn={tn}, fp={fp}, fn={fn})" for x, tp, tn, fp, fn in zip(CXRLT2024_TASK1_CLASSES, tps, tns, fps, fns)]
    plot_metrics(
        metric_names=names,
        metric_values=f1s,
        title='F1 per class',
        ylabel='Class',
        xlabel='F1',
        append_average_to_title=True,
        horizontal=True,
        sort_metrics=True,
        show_metrics_above_bars=True,
        figsize=(5,10),
    )
    plot_metrics(
        metric_names=names,
        metric_values=accs,
        title='Accuracy per class',
        ylabel='Class',
        xlabel='Accuracy',
        append_average_to_title=True,
        horizontal=True,
        sort_metrics=True,
        show_metrics_above_bars=True,
        figsize=(5,10),
    )

    return mistakes_per_class

def generate_dataframe_for_manual_inspection(
        cxrlt2024_integrated_nli_queries_for_fact_classification_filepath,
        background_findings_and_impression_per_report_filepath,
        processed_queries_filepath,
    ):
    """
    Generate a dataframe for manual inspection.
    """

    # Load integrated NLI queries for fact classification
    output = load_pickle(cxrlt2024_integrated_nli_queries_for_fact_classification_filepath)

    # Load CXR-LT 2024 train data
    train_df = pd.read_csv(MIMIC_CXR_LT_2024_TASK1_TRAIN_CSV_PATH)
    train_labels = train_df[CXRLT2024_TASK1_CLASSES].values
    train_dicom_ids = train_df['dicom_id'].values
    dicom_id_to_idx = { dicom_id: i for i, dicom_id in enumerate(train_dicom_ids) }

    not_found = 0
    cases_per_class = { x: {'tp': [], 'fp': [], 'tn': [], 'fn': [] } for x in CXRLT2024_TASK1_CLASSES }

    for split in ['train', 'dev']:
        dicom_id_to_neg_pos_labels = output[split]
        for dicom_id, (neg_idxs, pos_idxs) in dicom_id_to_neg_pos_labels.items():
            if dicom_id not in dicom_id_to_idx:
                not_found += 1
                continue
            idx = dicom_id_to_idx[dicom_id]
            if len(neg_idxs) > 0:
                for neg_idx in neg_idxs:
                    if neg_idx >= len(CXRLT2024_TASK1_CLASSES): # Skip zero-shot classes
                        continue
                    c = 'tn' if train_labels[idx, neg_idx] == 0 else 'fn'
                    cases_per_class[CXRLT2024_TASK1_CLASSES[neg_idx]][c].append(dicom_id)
            if len(pos_idxs) > 0:
                for pos_idx in pos_idxs:
                    if pos_idx >= len(CXRLT2024_TASK1_CLASSES): # Skip zero-shot classes
                        continue
                    c = 'tp' if train_labels[idx, pos_idx] == 1 else 'fp'
                    cases_per_class[CXRLT2024_TASK1_CLASSES[pos_idx]][c].append(dicom_id)
    
    if not_found > 0:
        print_orange(f"{not_found} dicom ids not found in the train set", bold=True)

    # Map dicom ids to reports
    imageId2reportId = get_imageId2reportId()
    for cases in cases_per_class.values():
        for c in ['tp', 'fp', 'tn', 'fn']:
            dicom_ids = cases[c]
            report_ids = [imageId2reportId[dicom_id] for dicom_id in dicom_ids]
            report_ids = list(set(report_ids)) # Remove duplicates
            cases[c] = report_ids

    # Generate dataframe
    metadata = load_mimiccxr_reports_detailed_metadata(
        background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath)
    processed_queries = load_jsonl(processed_queries_filepath)
    columns = ['case', 'class', 'CXRLT2024_label', 'GPT-4_label', 'filepath', 'part_id', 'subject_id', 'study_id',
               'original_report', 'parsed_report', 'GPT-4_query', 'GPT-4_reason', 'GPT-4_answer']
    rows = []
    for class_name, cases in cases_per_class.items():
        for c in ['tp', 'fp', 'tn', 'fn']:
            report_ids = cases[c]
            sampled_report_ids = random.sample(report_ids, min(2, len(report_ids))) # Sample at most 2 reports
            for report_id in sampled_report_ids:
                filepath = metadata['filepaths'][report_id]
                part_id = metadata['part_ids'][report_id]
                subject_id = metadata['subject_ids'][report_id]
                study_id = metadata['study_ids'][report_id]
                original_report = read_txt(filepath)
                parsed_report = _build_report(metadata, report_id)
                query = _build_query(parsed_report, CXRLT2024_CLASS_2_SENTENCE[class_name])
                found = False
                for idx, pq in enumerate(processed_queries):
                    if pq['metadata']['query'] == query:
                        found = True
                        break
                assert found, f"Query not found: {query}"
                rows.append({
                    'case': c,
                    'class': class_name,
                    'CXRLT2024_label': 1 if c in ['tp', 'fn'] else 0,
                    'GPT-4_label': 1 if c in ['tp', 'fp'] else 0,
                    'filepath': filepath,
                    'part_id': part_id,
                    'subject_id': subject_id,
                    'study_id': study_id,
                    'original_report': original_report,
                    'parsed_report': parsed_report,
                    'GPT-4_query': query,
                    'GPT-4_reason': processed_queries[idx]['parsed_response']['reason'],
                    'GPT-4_answer': processed_queries[idx]['parsed_response']['label'],
                })

    return pd.DataFrame(rows, columns=columns) # Return dataframe


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--split", type=str, default=None, choices=["train", "dev"])
    parser.add_argument("--chest_imagenome_image_id_to_labels_filepath", type=str, default=None)
    parser.add_argument("--chest_imagenome_label_names_filepath", type=str, default=None)
    parser.add_argument("--mimiccxr_background_findings_and_impression_per_report_filepath", type=str, default=None)
    parser.add_argument("--queries_to_skip_filepaths", type=str, nargs="+", default=None)
    parser.add_argument("--specific_classes", type=str, nargs="+", default=None)
    parser.add_argument("--specific_dicom_ids_filepath", type=str, default=None)
    parser.add_argument("--pos_weight", type=float, default=0.85)
    
    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_request", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    parser.add_argument("--not_delete_api_requests_and_responses", action="store_true", default=False)
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--use_batch_api", action="store_true", default=False)
    parser.add_argument("--batch_description", type=str, default=None)
    parser.add_argument("--batch_input_file_id", type=str, default=None)

    # Additional arguments
    parser.add_argument("--integrate_nli_queries_for_fact_classification", action="store_true", default=False)
    parser.add_argument("--queries_to_integrate_filepaths", type=str, nargs="+", default=None)
    parser.add_argument("--fact_encoder_model_name", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--fact_encoder_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.integrate_nli_queries_for_fact_classification: # Integrate NLI queries for fact classification
        assert args.mimiccxr_background_findings_and_impression_per_report_filepath is not None
        assert args.queries_to_integrate_filepaths is not None
        assert args.fact_encoder_checkpoint_folder_path is not None
        integrate_nli_queries_for_fact_classification(
            mimiccxr_background_findings_and_impression_per_report_filepath=args.mimiccxr_background_findings_and_impression_per_report_filepath,
            queries_to_integrate_filepaths=args.queries_to_integrate_filepaths,
            fact_encoder_model_name=args.fact_encoder_model_name,
            fact_encoder_checkpoint_folder_path=args.fact_encoder_checkpoint_folder_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        sys.exit(0) # Exit

    assert args.split is not None

    processed_queries_save_filepath = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_cxrlt2024_report_nli_queries_{args.split}{args.alias}.jsonl")

    if args.api_responses_filepath is None and args.batch_input_file_id is None:

        # Load already processed queries if they exist
        already_processed = set()
        if os.path.exists(processed_queries_save_filepath):
            rows = load_jsonl(processed_queries_save_filepath)
            # for row in rows:
            #     already_processed.add(hash_string(row['metadata']['query']))
            already_processed.update(compute_hashes_in_parallel([x['metadata']['query'] for x in rows]))
            logger.info(f"Loaded {len(rows)} already processed queries from {processed_queries_save_filepath}")

        # Load queries to skip
        if args.queries_to_skip_filepaths is not None:
            for queries_to_skip_filepath in args.queries_to_skip_filepaths:
                rows = load_jsonl(queries_to_skip_filepath)
                # for row in rows:
                #     already_processed.add(hash_string(row['metadata']['query']))
                already_processed.update(compute_hashes_in_parallel([x['metadata']['query'] for x in rows]))
                logger.info(f"Loaded {len(rows)} queries to skip from {queries_to_skip_filepath}")

        # Sample queries
        queries_to_process = sample_queries_label_based(
            num_samples=args.num_samples,
            split=args.split,
            chest_imagenome_image_id_to_labels_filepath=args.chest_imagenome_image_id_to_labels_filepath,
            chest_imagenome_label_names_filepath=args.chest_imagenome_label_names_filepath,
            mimiccxr_background_findings_and_impression_per_report_filepath=args.mimiccxr_background_findings_and_impression_per_report_filepath,
            already_processed_queries=already_processed,
            specific_classes=args.specific_classes,
            specific_dicom_ids_filepath=args.specific_dicom_ids_filepath,
            pos_weight=args.pos_weight,
        )

        logger.info(f"Total number of queries to process: {len(queries_to_process)}")

        # Print example queries
        logger.info(f"Example queries to process:")
        for i in np.linspace(0, len(queries_to_process)-1, min(5, len(queries_to_process)), dtype=int):
            logger.info(f"{i+1}. {queries_to_process[i]}")

    else:
        if args.api_responses_filepath is not None:
            assert os.path.exists(args.api_responses_filepath)
        queries_to_process = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=queries_to_process,
        system_instructions=INSTRUCTIONS,
        api_key_name=args.api_key_name,
        openai_model_name=args.openai_model_name,
        openai_request_url=args.openai_request_url,
        max_tokens_per_request=args.max_tokens_per_request,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        temperature=args.temperature,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logger=logger,
        logging_level=args.logging_level,
        parse_openai_output=parse_openai_model_output,
        tmp_dir=MIMICCXR_FAST_TMP_DIR,
        save_filepath=processed_queries_save_filepath,
        use_batch_api=args.use_batch_api,
        batch_description=args.batch_description,
        batch_input_file_id=args.batch_input_file_id,
    )