import re
import os
from pprint import pprint
from collections import OrderedDict
from nltk import wordpunct_tokenize

from medvqa.datasets.qa_pairs_extractor import REGULAR_EXPRESSIONS_FOLDER
from medvqa.metrics.medical.med_completeness import MEDICAL_TERMS_PATH
from medvqa.utils.files import read_lines_from_txt

_re_header = re.compile(r'(^|\n)\s*([A-Z][a-zA-Z]*(( |-|&)+[a-zA-Z]+)*?:)')
_re_paragraph_breaks = re.compile(r'(\s*\n(\s*\n\s*)+)|(\n\s*_+\s*\n)')

def extract_report_and_patient_background(report_path, debug=False):
    sections = _split_report_into_sections(report_path, debug=debug)
    
    if debug: pprint(sections)
    
    report_chunks = []
    background_chunks = []
    
    if debug: print('===================================================')
    
    for k_, v in sections.items():
        k = k_.lower()
        
        if k in _SECTION_HEADERS_FOR_PATIENT_BACKGROUND:
            
            if type(v) is list:
                for i in range(len(v)):
                    v[i] = v[i].replace('_', ' ')
                    v[i] = v[i].strip()
                    if len(v[i]) > 0 and v[i][-1] != '.': v[i] += '.'
                if k == '(after) wet read:':
                    v = ' '.join(x for x in v if len(x) > 0 and _contains_medical_terms(x, 1))
                else:
                    v = ' '.join(x for x in v if len(x) > 0)
            else:
                v = v.replace('_', ' ').strip()
                if k == '(after) wet read:' and not _contains_medical_terms(v, 1):
                    v = ''                    
            v = ' '.join(v.split())
            if len(v) == 0 or v == '.': continue
            if v[-1] != '.': v += '.'
            background_chunks.append((k, v))
            
        elif k in _SECTION_HEADERS_FOR_REPORT:
            
            if type(v) is list:
                for i in range(len(v)):
                    v[i] = v[i].replace('_', ' ')
                    v[i] = v[i].strip()
                    if len(v[i]) > 0 and v[i][-1] != '.': v[i] += '.'
                v = ' '.join(x for x in v if len(x) > 0)
            else:
                v = v.replace('_', ' ').strip()
            v = ' '.join(v.split())
            if len(v) == 0 or v == '.': continue
            if v[-1] != '.': v += '.'            
            
            report_chunks.append((k, v))
            
        elif k not in _IGNORABLE_HEADERS:
            
            if type(v) is list:
                for i in range(len(v)):
                    v[i] = v[i].replace('_', ' ')
                    v[i] = v[i].strip()
                v = ' '.join(x for x in v if len(x) > 0 and\
                             _contains_medical_terms(x, 2) and\
                             _contains_no_invalid_patterns(x))
            elif _contains_medical_terms(v, 2) and _contains_no_invalid_patterns(v):
                v = v.replace('_', ' ').strip()
            else:
                v = ''
            v = ' '.join(v.split())
            if len(v) == 0 or v == '.': continue
            if v[-1] != '.': v += '.'
                
            if (not k.startswith('(after)') and _contains_medical_terms(k, 1)):
                v = f'{k_} {v}'
            
            report_chunks.append((k, v))
            
    if len(report_chunks) == 0:
        for i, p in enumerate(background_chunks):
            if _contains_medical_terms(p[1], 3) and _contains_no_invalid_patterns(p[1]):
                report_chunks.append(p)
                background_chunks[i] = None
        background_chunks = [p for p in background_chunks if p is not None]

    # Final report
    report = ' '.join(x[1] for x in report_chunks)
    # Final background
    background = ' '.join(v if k.startswith('(after)') else f'{k} {v}' for k, v in background_chunks)

    return dict(
        report = report,
        background = background,
    )

def extract_background_findings_and_impression(report_path, debug=False):
    sections = _split_report_into_sections(report_path, debug=debug)
    
    if debug: pprint(sections)
    
    background_chunks = []
    findings_chunks = []
    impression_chunks = []
    
    if debug: print('===================================================')
    
    for k_, v in sections.items():
        k = k_.lower()
        
        if k in _SECTION_HEADERS_FOR_PATIENT_BACKGROUND:
            
            if type(v) is list:
                for i in range(len(v)):
                    v[i] = v[i].replace('_', ' ')
                    v[i] = v[i].strip()
                    if len(v[i]) > 0 and v[i][-1] != '.': v[i] += '.'
                v = ' '.join(x for x in v if len(x) > 0)
            else:
                v = v.replace('_', ' ').strip()
            v = ' '.join(v.split())
            if len(v) == 0 or v == '.': continue
            if v[-1] != '.': v += '.'
            background_chunks.append((k, v))
            
        elif k in _SECTION_HEADERS_FOR_FINDINGS:
            
            if type(v) is list:
                for i in range(len(v)):
                    v[i] = v[i].replace('_', ' ')
                    v[i] = v[i].strip()
                    if len(v[i]) > 0 and v[i][-1] != '.': v[i] += '.'
                if k == '(after) wet read:':
                    v = ' '.join(x for x in v if len(x) > 0 and _contains_medical_terms(x, 1))
                else:
                    v = ' '.join(x for x in v if len(x) > 0)
            else:
                v = v.replace('_', ' ').strip()
                if k == '(after) wet read:' and not _contains_medical_terms(v, 1):
                    v = ''
            v = ' '.join(v.split())
            if len(v) == 0 or v == '.': continue
            if v[-1] != '.': v += '.'
            findings_chunks.append((k, v))

        elif k in _SECTION_HEADERS_FOR_IMPRESSION:

            if type(v) is list:
                for i in range(len(v)):
                    v[i] = v[i].replace('_', ' ')
                    v[i] = v[i].strip()
                    if len(v[i]) > 0 and v[i][-1] != '.': v[i] += '.'
                v = ' '.join(x for x in v if len(x) > 0)
            else:
                v = v.replace('_', ' ').strip()
            v = ' '.join(v.split())
            if len(v) == 0 or v == '.': continue
            if v[-1] != '.': v += '.'
            impression_chunks.append((k, v))
            
        elif k not in _IGNORABLE_HEADERS:
            
            if type(v) is list:
                for i in range(len(v)):
                    v[i] = v[i].replace('_', ' ')
                    v[i] = v[i].strip()
                v = ' '.join(x for x in v if len(x) > 0 and\
                             _contains_medical_terms(x, 2))
            elif _contains_medical_terms(v, 2):
                v = v.replace('_', ' ').strip()
            else:
                v = ''
            v = ' '.join(v.split())
            if len(v) == 0 or v == '.': continue
            if v[-1] != '.': v += '.'
                
            if (not k.startswith('(after)') and _contains_medical_terms(k, 1)):
                v = f'{k_} {v}'
            
            findings_chunks.append((k, v))
            
    if len(findings_chunks) == 0 and len(impression_chunks) == 0:
        for i, p in enumerate(background_chunks):
            if _contains_medical_terms(p[1], 4):
                findings_chunks.append(p)
                background_chunks[i] = None
        background_chunks = [p for p in background_chunks if p is not None]

    len_findings = sum(len(x[1]) for x in findings_chunks)
    len_impression = sum(len(x[1]) for x in impression_chunks)
    if len_findings < 0.5 * len_impression:
        # Move impression to findings
        findings_chunks += impression_chunks
        impression_chunks = []

    # Final background
    background = ' '.join(v if k.startswith('(after)') else f'{k} {v}' for k, v in background_chunks)
    # Final findings
    findings = ' '.join(x[1] for x in findings_chunks)
    # Final impression
    impression = ' '.join(x[1] for x in impression_chunks)

    return dict(
        background = background,
        findings = findings,
        impression = impression,
    )

def _split_report_into_sections(report_path, debug=False):
    with open(report_path) as f:
        text = f.read()
    if debug:
        print(text)
    paragraphs = _re_paragraph_breaks.split(text)
    sections = OrderedDict()
    last_k = None
    for p in paragraphs:
        if p is None: continue
        spans = [i.span() for i in _re_header.finditer(p)]
        if len(spans) > 0:            
            if spans[0][0] > 0:
                if last_k is None:
                    k = '(HEADERLESS PARAGRAPH)'
                else:
                    k = f'(AFTER) {last_k}'
                try:
                    tmp = sections[k]
                except KeyError:
                    tmp = sections[k] = []
                tmp.append(p[:spans[0][0]])
            for i, span in enumerate(spans):
                k = p[span[0]:span[1]]
                k = ' '.join(k.split()).strip()
                v = p[span[1]:] if i+1 == len(spans) else p[span[1]:spans[i+1][0]]
                try:
                    tmp = sections[k]
                except KeyError:
                    tmp = sections[k] = []
                tmp.append(v)
                last_k = k
        else:
            if last_k is None:
                k = '(HEADERLESS PARAGRAPH)'
            else:
                k = f'(AFTER) {last_k}'
            try:
                tmp = sections[k]
            except KeyError:
                tmp = sections[k] = []
            tmp.append(p)
    return sections

def _load_invalid_patterns_regex():
    pattern = ''
    with open(os.path.join(REGULAR_EXPRESSIONS_FOLDER, 'invalid_sentence_patterns.txt')) as f:
        for line in f.readlines():
            if len(pattern) > 0:
                pattern += '|'
            pattern += f'({line.strip()})'
    return re.compile(pattern, re.IGNORECASE)

_re_invalid = _load_invalid_patterns_regex()

_medical_terms = set(read_lines_from_txt(MEDICAL_TERMS_PATH))

def _contains_medical_terms(text, k):
    count = 0
    for x in wordpunct_tokenize(text.lower()):
        if x in _medical_terms:
            count += 1
            if count >= k: return True
    return False

def _contains_no_invalid_patterns(text):
    return not _re_invalid.search(text)

_SECTION_HEADERS_FOR_IMPRESSION = set([
    'MPRESSION:',
    'IMPRESSION:',
    'IMPRESSON:',
    'IMPRESSIONS:',
    'IMPRESION:',
    'IMPESSION:',
    'IMPRSSION:',
    'IMPRESSOIN:',
    'Impression:',
    'CONCLUSION:',
    'Conclusion:',
])
_SECTION_HEADERS_FOR_IMPRESSION.update([
    f'(AFTER) {key}' for key in _SECTION_HEADERS_FOR_IMPRESSION
])

_SECTION_HEADERS_FOR_FINDINGS = set([
    'FINDINGS:',
    'FINDNINGS:',
    'FINGDINGS:',
    'FINIDNGS:',
    'Findings:',
    'FINDINS:',
    'FINDING:',
    'FINDINDGS:',
    'FIMPRESSION:',
    'FINSINGS:',
    'FINDIGNS:',
    'FINDINGS and IMPRESSION:',
    'Findings and Impression:',
    'FINDINGS AND IMPRESSION:',
    'REPORT:',
    'THEY REPORT TEXT FOLLOWS:',
    'PORTABLE SUPINE FRONTAL VIEW OF THE CHEST:',
    'FRONTAL AND LATERAL CHEST RADIOGRAPHS:',
    'FRONTAL AND LATERAL CHEST RADIOGRAPH:',
    'FRONTAL AND LATERAL VIEWS OF THE CHEST:',
    'PA AND LATERAL CHEST RADIOGRAPH:',
    'PA AND LATERAL VIEWS OF THE CHEST:',
    'PA AND LATERAL:',
    'AP AND LATERAL:',
    'PA AND LATERAL CHEST:',
    'FRONTAL AND LATERAL CHEST:',
    'TWO VIEWS OF THE CHEST:',
    'PORTABLE AP CHEST:',
    'PORTABLE AP CHEST RADIOGRAPH:',
    'PA AND LATERAL CHEST RADIOGRAPHS:',
    'UPRIGHT PORTABLE RADIOGRAPH OF THE CHEST:',
    'PORTABLE RADIOGRAPH OF THE CHEST:',
    'FRONTAL CHEST RADIOGRAPH:',
    'FRONTAL UPRIGHT PORTABLE CHEST:',
    'PORTABLE FRONTAL CHEST RADIOGRAPH:',
    'PORTABLE UPRIGHT FRONTAL VIEW OF THE CHEST:',
    'UPRIGHT AP VIEW OF THE CHEST:',
    'PORTABLE CHEST RADIOGRAPH:',
    'SINGLE AP PORTABLE VIEW:',
    'SINGLE PORTABLE CHEST RADIOGRAPH:',
    'SINGLE PORTABLE VIEW OF THE CHEST:',
    'UPRIGHT FRONTAL CHEST RADIOGRAPHS:',
    'AP:',
    'AP UPRIGHT:',
    'AP CHEST:',
    'CHEST RADIOGRAPH:',
    'TWO VIEWS OF THE THORACIC SPINE:',
    'TWO VIEWS:',
    'AP AND LATERAL VIEWS OF THE CHEST:',
    'ERECT FRONTAL CHEST RADIOGRAPH:',
    'PORTABLE FRONTAL VIEW OF THE CHEST:',
    'PORTABLE RADIOGRAPH:',
    'CHEST AP:',
    'ONE VIEW OF THE CHEST:',
    'AP FILM:',
    'SUPINE AP VIEW OF THE CHEST:',
    'AP VIEW OF THE CHEST:',
    'AP AND LATERAL CHEST RADIOGRAPHS:',
    'AP PORTABLE CHEST:',
    'PORTABLE AP UPRIGHT CHEST RADIOGRAPH:',
    'SINGLE FRONTAL VIEW OF THE CHEST:',
    'DOUBLE CHEST RADIOGRAPH:',
    'CHEST TWO VIEWS:',
    'SUPINE PORTABLE CHEST RADIOGRAPH:',
    'SINGLE PORTABLE AP CHEST RADIOGRAPH:',
    'PORTABLE AP VIEW OF THE CHEST:',
    'PORTABLE ERECT RADIOGRAPH:',
    'PFI:',
    'PORTABLE SUPINE RADIOGRAPH OF THE CHEST:',
    'SINGLE AP VIEW:',
    'PORTABLE UPRIGHT AP VIEW OF THE CHEST:',
    'FRONTAL VIEW OF THE CHEST:',
    'SUPINE PORTABLE CHEST:',
    'PORTABLE UPRIGHT RADIOGRAPH OF THE CHEST:',
    'SINGLE AP VIEW OF THE CHEST:',
    'SINGLE AP PORTABLE CHEST RADIOGRAPH:',
    'PORTABLE AP AND LATERAL CHEST RADIOGRAPH:',
    'CHEST PORTABLE:',
    'PA AND LATERAL VIEWS:',
    'AP SUPINE:',
    'CHEST SINGLE VIEW:',
    'FOUR IMAGES:',
    'PORTABLE SUPINE CHEST RADIOGRAPH:',
    'SEMIUPRIGHT PORTABLE RADIOGRAPH OF THE CHEST:',
    'SINGLE FRONTAL CHEST RADIOGRAPHS:',
    'FRONTAL SUPINE PORTABLE CHEST:',
    'CHEST PA AND LAT RADIOGRAPH:',
    'AP UPRIGHT AND LATERAL:',
    'AP PORTABLE FRONTAL CHEST RADIOGRAPH:',
    'SINGLE PORTABLE FRONTAL VIEW OF THE CHEST:',
    'VIEWS:',
    'FRONTAL AND LATERAL VIEWS THE CHEST:',
    'AP FRONTAL CHEST RADIOGRAPH:',
    'KYPHOTIC POSITIONING:',
    'PORTABLE UPRIGHT CHEST RADIOGRAPH:',
    'UPRIGHT FRONTAL AND LATERAL CHEST RADIOGRAPHS:',
    'SUPINE CHEST RADIOGRAPH:',
    'AP PORTABLE:',
    'OSSEOUS STRUCTURES:',
    'SINGLE VIEW:',
    'SINGLE PORTABLE UPRIGHT VIEW OF THE CHEST:',    
    'SINGLE FRONTAL PORTABLE VIEW OF THE CHEST:',
    'FRONTAL AND LATERAL RADIOGRAPHS:',
    'RADIOGRAPH:',
    'SINGLE AP SUPINE PORTABLE VIEW:',
    'LORDOTIC POSITIONING:',
    'UPRIGHT AP AND LATERAL VIEWS OF THE CHEST:',
    'SUPINE PORTABLE FRONTAL CHEST RADIOGRAPH:',
    'AP CHEST RADIOGRAPH:',
    'FRONTAL AND LATERAL VIEWS CHEST:',
    'FRONTAL PORTABLE CHEST:',
    'FRONTAL PORTABLE SUPINE CHEST:',
    'UPRIGHT FRONTAL VIEW OF THE CHEST:',
    'PORTABLE SUPINE AP CHEST RADIOGRAPH:',
    'PA AND LATERAL VIEWS OF CHEST:',
    'TECHNIQUE PA AND LATERAL VIEWS OF THE CHEST:',
    'AP FRONTAL AND LATERAL CHEST RADIOGRAPHS:',
    'CHEST PA:',
    'AP PORTABLE CHEST RADIOGRAPH:',
    'PORTABLE AP FRONTAL CHEST RADIOGRAPH:',
    'SINGLE AP PORTABLE VIEW WHICH INCLUDES THE UPPER ABDOMEN:',
    'LATERAL VIEWS OF THE CHEST:',
    'SUPINE FRONTAL VIEW OF THE CHEST:',
    'FRONTAL CHEST RADIOGRAPH WITH THE PATIENT IN SUPINE AND UPRIGHT POSITIONS:',
    'AP UPRIGHT VIEW OF THE CHEST:',
    'AP AND LATERAL CHEST RADIOGRAPH:',
    'PORTABLE SEMI-ERECT FRONTAL CHEST RADIOGRAPH:',
    'PORTABLE SUPINE FRONTAL CHEST RADIOGRAPH:',
    'FRONTAL SEMI-UPRIGHT PORTABLE CHEST:',
    'SEMI-ERECT PORTABLE AP CHEST RADIOGRAPH:',
    'PORTABLE SEMI-UPRIGHT AP CHEST RADIOGRAPH:',
    'PA AND LATERAL RADIOGRAPHS OF THE CHEST:',
    'FRONTAL PORTABLE UPRIGHT RADIOGRAPH:',
    'UPRIGHT PORTABLE CHEST RADIOGRAPH:',
    'AP PORTABLE UPRIGHT CHEST RADIOGRAPH:',
    'PORTABLE UPRIGHT RADIOGRAPH CHEST:',
    'SINGLE AP PORTABLE VIEW OF THE CHEST:',
    'SEMI-ERECT FRONTAL CHEST RADIOGRAPH:',
    'SEMI-UPRIGHT AP VIEW OF THE CHEST:',
    'FRONTAL SEMI-SUPINE PORTABLE CHEST:',
    'FRONTAL PORTABLE UPRIGHT CHEST:',
    'CHEST AP SUPINE:',
    'AP UPRIGHT AND LATERAL CHEST RADIOGRAPHS:',
    'PORTABLE UPRIGHT AP CHEST RADIOGRAPH:',
    'PORTABLE UPRIGHT AP VIEW OF THE ABDOMEN:',
    'PORTABLE AP SEMI-UPRIGHT CHEST RADIOGRAPH:',
    'PA AND LAT CHEST RADIOGRAPH:',
    'PORTABLE AP SEMI-ERECT RADIOGRAPH:',
    'Chest:',
    'PORTABLE SEMI-UPRIGHT FRONTAL CHEST RADIOGRAPH:',
    'PA AND LAT:',
    'FRONTAL AP AND LATERAL CHEST:',
    'SEMI-ERECT PORTABLE CHEST RADIOGRAPH:',
    'AP UPRIGHT CHEST RADIOGRAPH:',
    'PA AND LATERAL UPRIGHT CHEST RADIOGRAPHS:',
    'TWO PORTABLE ERECT VIEWS OF THE CHEST:',
    'CHEST AND UPPER ABDOMEN:',
    'Portable chest:',
    'PA and lateral chest reviewed in the absence of prior chest imaging:',
    'CHEST AND PELVIS FILMS:',
    'Chest radiographs:',
    'FRONTAL and LATERAL VIEWS OF THE CHEST:',
    'PORTABLES SEMI-ERECT CHEST RADIOGRAPH:',
    'BEDSIDE FRONTAL CHEST RADIOGRAPH:',
    'SEMI-ERECT PORTABLE CHEST:',
    'SEMI-UPRIGHT PORTABLE RADIOGRAPH OF THE CHEST:',
    'AP UPRIGHT VIEWS OF THE CHEST DURING INSPIRATION AND EXPIRATION:',
    'SEMI-ERECT PORTABLE FRONTAL CHEST RADIOGRAPH:',
    'PORTABLE SEMI-UPRIGHT VIEW OF THE CHEST:',
    'FRONTAL SUPINE PORTABLE VIEW OF THE CHEST:',
    'PA & LATERAL VIEW OF THE CHEST:',
    'PA and lateral views of the chest:',
    'In comparison to previous radiographs:',
    'SUPINE CHEST:',
    'TWO PORTABLE AP VIEWS OF THE CHEST:',
    'PORTABLE SUPINE CHEST:',
    'SUPINE PORTABLE FRONTAL VIEW OF THE CHEST:',
    'BEDSIDE AP UPRIGHT RADIOGRAPH OF THE CHEST:',
    'Again demonstrated are:',
    'RIGHT AND LEFT FRONTAL OBLIQUE VIEWS OF THE CHEST:',
    'PORTABLE SEMI-UPRIGHT RADIOGRAPH OF THE CHEST:',
    'AP UPRIGHT AND LATERAL VIEWS OF THE CHEST:',
    'AP SUPINE CHEST RADIOGRAPH:',
    'FRONTAL VIEWS OF THE CHEST:',
    'PA AND LATERAL CHEST FILMS:',
    'BEDSIDE AP RADIOGRAPH OF THE CHEST:',
    'PORTABLE AP UPRIGHT RADIOGRAPH OF THE CHEST:',
    'CHEST RADIOGRAPHS:',
    'PA AND LATERAL VIEWS CHEST:',
    'PORTABLE SEMI-ERECT AP CHEST RADIOGRAPH:',
    'SINGLE FRONTAL CHEST RADIOGRAPH:',
    'PORTABLE AP FRONTAL VIEW OF THE CHEST:',
    'SUPINE PORTABLE CHEST RADIOGRAPHS:',
    'PORTABLE AP UPRIGHT VIEW OF THE CHEST:',
    'PORTABLE PA CHEST RADIOGRAPH:',
    'AP view of chest:',
    'UPRIGHT AP AND LATERAL VIEWS OF CHEST:',
    'PORTABLE SEMI-ERECT CHEST RADIOGRAPH:',
    'SEMI-UPRIGHT PORTABLE CHEST RADIOGRAPH:',
    'SINGLE AP ERECT PORTABLE VIEW OF THE CHEST:',
    'SINGLE UPRIGHT PORTABLE CHEST RADIOGRAPH:',
    'AP and lateral views of the chest:',
    'UPRIGHT AP AND LATERAL CHEST RADIOGRAPH:',
    'UPRIGHT FRONTAL AND LATERAL VIEWS OF THE CHEST:',
    'PORTABLE UPRIGHT FRONTAL CHEST RADIOGRAPH:',
    'FRONTAL AND LATERAL VIEWS OF CHEST:',
    'FRONTAL CHEST RADIOGRAPHS:',
    'PA and lateral views of chest:',
    'SINGLE PORTABLE UPRIGHT CHEST RADIOGRAPH:',
    'SUPINE AP PORTABLE CHEST RADIOGRAPH:',
    'Additional findings:',
    'SUPINE PORTABLE AP CHEST RADIOGRAPH:',
    'SINGLE SEMI-ERECT PORTABLE VIEW OF THE CHEST:',
    'PORTABLE SEMI-UPRIGHT CHEST RADIOGRAPH:',
    'FRONTAL LATERAL CHEST RADIOGRAPH:',
    'SEMI-ERECT PORTABLE VIEW OF THE CHEST:',
    'PA AND LATERAL FILMS OF THE CHEST:',
    'SEMI-UPRIGHT AP AND LATERAL VIEWS OF THE CHEST:',
    'CHEST PA AND LATERAL RADIOGRAPH:',
    'FRONTAL AND LATERAL UPRIGHT CHEST RADIOGRAPH:',
    'PORTABLE SEMI-ERECT FRONTAL CHEST RADIOGRAPHS:',
    'OPINION:',
    'FRONTAL PORTABLE CHEST RADIOGRAPH:',
    'SEMIERECT PORTABLE RADIOGRAPH OF THE CHEST:',
    'PORTABLE SEMI-ERECT AP AND PA CHEST RADIOGRAPH:',
    'FRONTAL SEMI UPRIGHT PORTABLE CHEST:',
    'SEMIERECT AP VIEW OF THE CHEST:',
    'AP VIEW AND LATERAL VIEW OF THE CHEST:',
    'PORTABLE SEMI-ERECT CHEST:',
    'PORTABLE SEMI-UPRIGHT FRONTAL VIEW OF THE CHEST:',
    'AP SEMI-ERECT CHEST RADIOGRAPH:',
    'SINGLE AP UPRIGHT PORTABLE CHEST RADIOGRAPH:',
    'SINGLE SUPINE PORTABLE VIEW OF THE CHEST:',
    'PA AND AP CHEST RADIOGRAPH:',
    'SINGLE PORTABLE AP VIEW OF THE CHEST:',
    'PORTABLE AP CHEST RADIOGRAPHS:',
    'PORTABLE SEMI-UPRIGHT FRONTAL CHEST RADIOGRAPHS:',
    'SEMI-UPRIGHT AP AND LATERAL CHEST RADIOGRAPHS:',
    'SINGLE PORTABLE CHEST X-RAY:',
    'PORTABLE SEMI-UPRIGHT RADIOGRAPH:',
    'AP portable view of the chest:',
    'AP chest reviewed in the absence of prior chest radiographs:',
    'PA and lateral chest reviewed in the absence of prior chest radiographs:',
])
_SECTION_HEADERS_FOR_FINDINGS.update([
    f'(AFTER) {key}' for key in _SECTION_HEADERS_FOR_FINDINGS
])
_SECTION_HEADERS_FOR_FINDINGS.update([
    'WET READ:',
    '(AFTER) WET READ:',
    'RESIDENT WET READ:',
    'PRELIMINARY RESIDENT WET READ:',
    'PRELIMINARY REPORT:',
])

# make the union of the above two sets
_SECTION_HEADERS_FOR_REPORT = set()
_SECTION_HEADERS_FOR_REPORT.update(_SECTION_HEADERS_FOR_IMPRESSION)
_SECTION_HEADERS_FOR_REPORT.update(_SECTION_HEADERS_FOR_FINDINGS)

_SECTION_HEADERS_FOR_PATIENT_BACKGROUND = set([
    'REASON FOR EXAMINATION:',
    'REASON FOR EXAM:',
    'REASON FORE EXAM:',
    'REASON FOR THE EXAM:',
    'Reason for exam:',
    'REASON FOR INDICATION:',
    'INDICATION:',
    'INDCATION:',
    'IDICATION:',
    'NDICATION:',
    'INDICATIONS:',
    'Indication:',
    'ADDENDUM Indication:',
    'CLINICAL HISTORY:',
    'CLINICAL HISTORY History:',
    'CLINICAL INDICATION:',
    'Clincal indication:',
    'CLINICAL INFORMATION:',
    'CLINIC INDICATION:',
    'ADDITIONAL CLINICAL HISTORY PROVIDED:',
    'HISTORY:',
    'History:',
    'ADDENDUM INDICATION:',
    'PATIENT HISTORY:',
    'COR:',
    'CHOCTAW:',
    'ICHCt abd:',
    'NCHCT:',
    'ILLNESS:',
    'Surg:',
    'Pre-op:',
    'Pre-op CXR Surg:',
    'Pre-op Surg:',
    'Pre-op chest xray Surg:',
    'Pre-operative planning Surg:',
    'Please obtain pre-op CXR Surg:',
    'CXR Surg:',
    'BKA Surg:',
    'X-ray Surg:',
    'Story:',
    'Head CT:',
    'CT Head:',
    'Please eval for acute process Surg:',
    'Please assess for any abnormalities Surg:',
    'Please evaluate Surg:',
])

_IGNORABLE_HEADERS = set([
    'EXAM:', 'STUDY:', 'TYPE OF EXAMINATION:',
    'DATE:', 'RECOMMENDATIONS:', 'CXR:', 'TYPE OF THE EXAMINATION:',
    'RECOMMENDATION:', 'CT:', 'CTA:', 'RUQ:', 'QUESTIONS TO BE ANSWERED:',
    'CLINICAL INFORMATION & QUESTIONS TO BE ANSWERED:',
    'CC:', 'Compressing:', 'Contact name:', 'Common:',
])

_SECTION_HEADERS_FOR_IMPRESSION.update([x.lower() for x in _SECTION_HEADERS_FOR_IMPRESSION])
_SECTION_HEADERS_FOR_FINDINGS.update([x.lower() for x in _SECTION_HEADERS_FOR_FINDINGS])
_SECTION_HEADERS_FOR_REPORT.update([x.lower() for x in _SECTION_HEADERS_FOR_REPORT])
_SECTION_HEADERS_FOR_PATIENT_BACKGROUND.update([x.lower() for x in _SECTION_HEADERS_FOR_PATIENT_BACKGROUND])
_IGNORABLE_HEADERS.update([x.lower() for x in _IGNORABLE_HEADERS])