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
TEMPLATES_CHEXPERT_v2 = TEMPLATES_CHEXPERT_v1.copy()
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
