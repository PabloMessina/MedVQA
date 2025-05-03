from medvqa.utils.files_utils import (
    load_json_file,
    save_to_json,
)
from medvqa.datasets.qa_pairs_extractor import QuestionAnswerExtractor
from medvqa.datasets.iuxray import (
    IUXRAY_REPORTS_MIN_JSON_PATH,
    IUXRAY_CACHE_DIR,
)
from medvqa.utils.common import get_timestamp
from tqdm import tqdm
import os

if __name__ == '__main__':

    qa_extractor = QuestionAnswerExtractor()
    
    print(f'Loading original reports from {IUXRAY_REPORTS_MIN_JSON_PATH}')
    original_reports = load_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)

    print('Processing each report ...')    
    qa_adapted_reports = []
    for x in tqdm(original_reports.values()):
        findings = x['findings']
        impression = x['impression']
        if findings or impression:
            if findings and not impression:
                text = findings
            elif not findings and impression:
                text = impression
            else:
                if findings[-1] == '.':
                    text = findings + ' ' + impression
                else:
                    text = findings + '. ' + impression
            qa_info = qa_extractor.generate_qa_pairs_compact_version(text)
            qa_info['filename'] = x['filename']
            qa_info['background'] = x['indication']
            qa_adapted_reports.append(qa_info)
    
    qa_dataset = {
        'questions': qa_extractor.questions,
        'reports': qa_adapted_reports,
    }

    output_path = os.path.join(IUXRAY_CACHE_DIR, f'qa_adapted_reports__{get_timestamp()}.json')
    print(f'Saving qa adapted reports to {output_path}')
    save_to_json(qa_dataset, output_path)
    print('Done!')

