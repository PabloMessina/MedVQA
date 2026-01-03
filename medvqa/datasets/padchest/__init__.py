from medvqa.utils.files_utils import load_json
from medvqa.settings import (
    PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH,
)

def _clean_sentence(sentence: str):
    """
    Cleans the sentence by removing leading and trailing spaces and dots.
    """
    sentence = sentence.strip()
    if sentence.endswith('.'):
        sentence = sentence[:-1]
    return sentence

def get_padchest_gr_sentences_from_reports(language: str = 'en'):
    lang_key = f'sentence_{language}'
    reports_json_list = load_json(PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH)
    unique_sentences = set()
    for report_info in reports_json_list:
        findings = report_info['findings']
        for finding in findings:
            sentence = finding.get(lang_key, "")
            sentence = _clean_sentence(sentence)
            # Skip empty sentences
            if not sentence:
                continue
            # Add to set
            unique_sentences.add(sentence)
    # Sort the sentences
    unique_sentences = sorted(list(unique_sentences))
    return unique_sentences

            
def get_padchest_gr_labels():
    reports_json_list = load_json(PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH)
    unique_labels = set()
    for report_info in reports_json_list:
        findings = report_info['findings']
        for finding in findings:
            labels = finding.get('labels', [])
            unique_labels.update(labels)
    # Sort the labels
    unique_labels = sorted(list(unique_labels))
    return unique_labels

def get_padchest_gr_phrase_groundings(language: str = 'en'):
    reports_json_list = load_json(PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH)
    lang_key = f'sentence_{language}'
    output = []
    for report_info in reports_json_list:
        image_id = report_info['ImageID']
        findings = report_info['findings']
        for finding in findings:
            sentence = finding[lang_key]
            sentence = _clean_sentence(sentence)
            boxes = finding.get('boxes')
            if boxes:
                output.append({
                    'image_id': image_id,
                    'phrase': sentence,
                    'boxes': boxes
                })
    return output