# From https://github.com/pdpino/medical-ai/blob/master/medai/models/report_generation/templates/chex_v1.py

"""Templates chex-v1.

Chosen sentences from the dataset to represent each disease.

- Some sentences do not appear explicitly in the dataset as GT,
    but have been tested with chexpert.
"""

TEMPLATES_CHEXPERT_v1 = {
    'No Finding': {
        0: 'findings are present',
        1: 'no findings',
    },
    'Cardiomegaly': {
        0: 'heart size is normal',
        1: 'the heart is enlarged',
    },
    'Enlarged Cardiomediastinum': {
        0: 'the mediastinal contour is normal',
        1: 'the cardiomediastinal silhouette is enlarged',
    },
    'Lung Lesion': {
        0: 'no pulmonary nodules or mass lesions identified',
        1: 'there are pulmonary nodules or mass identified', # Not present in GT sentences
    },
    'Lung Opacity': {
        0: 'the lungs are free of focal airspace disease',
        1: 'one or more airspace opacities are seen', # Not present in GT sentences
    },
    'Edema': {
        0: 'no pulmonary edema',
        1: 'pulmonary edema is seen', # Not present in GT sentences
    },
    'Consolidation': {
        0: 'no focal consolidation',
        1: 'there is focal consolidation', # Not present in GT sentences
    },
    'Pneumonia': {
        0: 'no pneumonia',
        1: 'there is evidence of pneumonia', # Not present in GT sentences
    },
    'Atelectasis': {
        # most negative sentences are paired with other diseases
        0: 'no atelectasis', # Not present in GT sentences
        1: 'appearance suggest atelectasis',
    },
    'Pneumothorax': {
        0: 'no pneumothorax is seen',
        1: 'there is pneumothorax',
    },
    'Pleural Effusion': {
        0: 'no pleural effusion',
        1: 'pleural effusion is seen', # Not present in GT sentences
    },
    'Pleural Other': {
        0: 'no fibrosis',
        1: 'pleural thickening is present', # Not present in GT sentences
    },
    'Fracture': {
        0: 'no fracture is seen',
        1: 'a fracture is identified', # Not present in GT sentences
    },
    'Support Devices': {
        0: '', # Empty on purpose
        1: 'a device is seen', # Not present in GT sentences
    },
}

# Same as v1, but all negatives are empty
import copy
TEMPLATES_CHEXPERT_v2 = copy.deepcopy(TEMPLATES_CHEXPERT_v1)
for key in TEMPLATES_CHEXPERT_v2.keys():
    TEMPLATES_CHEXPERT_v2[key][0] = ''

# Minimal version: empty for all negatives, and just the observation name for positives
TEMPLATES_CHEXPERT_v3 = {
    'No Finding': {
        0: '',
        1: 'no findings',
    },
    'Cardiomegaly': {
        0: '',
        1: 'cardiomegaly',
    },
    'Enlarged Cardiomediastinum': {
        0: '',
        1: 'enlarged cardiomediastinum',
    },
    'Lung Lesion': {
        0: '',
        1: 'lung lesion',
    },
    'Lung Opacity': {
        0: '',
        1: 'lung opacity',
    },
    'Edema': {
        0: '',
        1: 'edema',
    },
    'Consolidation': {
        0: '',
        1: 'consolidation',
    },
    'Pneumonia': {
        0: '',
        1: 'pneumonia',
    },
    'Atelectasis': {
        0: '',
        1: 'atelectasis',
    },
    'Pneumothorax': {
        0: '',
        1: 'pneumothorax',
    },
    'Pleural Effusion': {
        0: '',
        1: 'pleural effusion',
    },
    'Pleural Other': {
        0: '',
        1: 'pleural other',
    },
    'Fracture': {
        0: '',
        1: 'fracture',
    },
    'Support Devices': {
        0: '',
        1: 'support devices',
    },
}

TEMPLATES_CHEXPERT_v4 = {
    'No Finding': {
        0: '', # Empty on purpose
        1: '', # Empty on purpose
    },
    'Cardiomegaly': {
        0: 'heart size is normal', 
        1: 'the heart is stable, mild, moderate, severe or enlarged in size'
    }, 
    'Enlarged Cardiomediastinum': {
        0: 'the cardiomediastinal silhouette is normal', 
        1: 'the cardiomediastinal silhouette is unchanged or enlarged or widened'
    },
    'Lung Lesion': {
        0: 'no lung nodules or masses',
        1: 'there are left or right pulmonary lung nodules observed'
    },
    'Lung Opacity': {
        0: 'no parenchymal opacities',
        1: 'there are left or right present lung airspace opacities'
    },
    'Edema': {
        0: 'there is no pulmonary edema',
        1: 'there is noted mild or moderate or severe pulmonary edema'
    },
    'Consolidation': {
        0: 'the lungs are clear without focal consolidation',
        1: 'there is observed left or right lung consolidation'
    },
    'Pneumonia': {
        0: 'no pneumonia',
        1: 'observed process left or right lung pneumonia'
    },
    'Atelectasis': {
        0: 'no atelectasis',
        1: 'there is observed left or right lung present atelectasis'
    },
    'Pneumothorax': {
        0: 'there is no pneumothorax',
        1: 'there is noted left sided or right sided, small, moderate or large pneumothorax in the lung'
    },
    'Pleural Effusion': 
    {
        0: 'no pleural effusions',
        1: 'there is an observed left or right or bilateral, small, moderate or large pleural effusion'
    },
    'Pleural Other': {
        0: 'there is no evidence of fibrosis',
        1: 'there is present left or right, minimal, mild or severe pleural thickening'
    },
    'Fracture': {
        0: 'no displaced fracture is seen',
        1: 'there is a rib or clavicular left or right sided fracture'
    },
    'Support Devices': {
        0: '',
        1: 'there is a noted right sided or left sided picc, central venous catheter or endotracheal tube'
    }
}

_LUNG_RELATED_DISEASES = (
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
)

GROUPS_v1 = [
    (
        _LUNG_RELATED_DISEASES, 0, 'the lungs are clear',
    ),
    (
        ('Consolidation', 'Pleural Effusion', 'Pneumothorax'), 0,
        'there is no focal consolidation, pleural effusion, or pneumothorax',
        # { 'redundant': False },
    ),
    (
        ('Pneumothorax', 'Pleural Effusion'), 0,
        'there is no pleural effusion or pneumothorax',
        #{ 'redundant': False },
    ),
    (
        ('Pneumothorax', 'Consolidation'), 0,
        'there is no focal consolidation or pneumothorax',
        #{ 'redundant': False },
    ),
]

def CreateSimpleReport(labels_dict, Template_Single):
    sentences = []
    for disease, label in labels_dict.items():
        if disease in Template_Single:
            sentences.append(Template_Single[disease][label])
    return '. '.join(sentences)

def CreateReportGivenLabels(labels_dict, Template_Single, Template_Groups, redundant_reports=False):
    report = ''
    covered_diseases = set()  # To keep track of diseases covered by group reports
    
    # Function to check if a disease is already covered
    def disease_was_already_covered(disease):
        nonlocal covered_diseases
        return disease in covered_diseases

    # Function to check if a disease is already covered in the given input tuple returns true or false
    def some_disease_in_group_already_covered(group):
        return any(disease_was_already_covered(disease) for disease in group)

    # Function to add a template sentence to the report
    def add_to_report(template):
        nonlocal report
        if template: # If the template is not empty
            if report: # If the report is not empty, add a period and a space
                if report[-1] != '.':
                    report += '. '
                else:
                    report += ' '
            report += template
    
    # Iterate through each group in the template groups
    for group, group_label, group_template in Template_Groups:
        # If the group_label is just an integer, convert it to a tuple
        if isinstance(group_label, int):
            group_label = (group_label,) * len(group)
        # Create a tuple of labels containing the label for each disease in the group
        actual_group_label = tuple(labels_dict[disease] for disease in group)
        # Create a tuple of name of labels containing the name of label for each disease in the group (keys of labels_dict)
        actual_name_group_label = tuple(disease for disease in group)
        # Check if the actual_group_label matches the group_label
        if actual_group_label == group_label and actual_name_group_label == group and (redundant_reports or not some_disease_in_group_already_covered(group)):
            # Add the group template to the report if it meets the conditions
            add_to_report(group_template)
            # Add all diseases in the group to the covered diseases
            for disease in group:
                covered_diseases.add(disease)
        
    # Fill the report with individual diseases that were not covered by the groups
    for disease, label in labels_dict.items():
        if disease not in covered_diseases:
            # Check if the disease has a single report template
            if disease in Template_Single:
                if label in Template_Single[disease]:
                    add_to_report(Template_Single[disease][label])

    return report.strip()







# NOTE: Wrong chexpert labeler:
# enlarged-cardiom == 1
#   'heart and mediastinum within normal limits',
#   'contour irregularity of the left clavicle appears chronic and suggests old injury',
#   'chronic appearing contour deformity of the distal right clavicle suggests old injury .',
#   'elevated right hemidiaphragm , with a nodular soft tissue contour , containing liver .',

# lung-lesion == 1
#   'ct scan is more sensitive in detecting small nodules',
#   'no suspicious appearing lung nodules .',
#   'there is no evidence for mass lung apices .'

# lung-opacity == 1
#   'no focal air space opacities .',
#   'this opacity cannot be well identified on the lateral view .',

# consolidation == 1
#   'no focal airspace consolidations .',
#   'no focal air is prominent consolidation .',

# pleural-effusion == 0
#   'no knee joint effusion',

# pleural-effusion == 1
#   'no findings to suggest pleural effusion',

# fracture==1
#   no visible fractures
#   no displaced rib fractures
#  'no acute , displaced rib fractures .',
#   'no displaced rib fracture visualized .',
#   'no definite visualized rib fractures .',
#   'no displaced rib fractures identified .',
#   'no displaced rib fractures visualized .'
#    'limited exam , for evaluation of fractures .',

# support-devices == 0
#  'no evidence of tuberculosis .',
#   'no evidence of active tuberculosis .',
# 'there is no evidence of tuberculous disease .',
# 'specifically , there is no evidence of tuberculous disease .'
