import os
from tqdm import tqdm

from medvqa.utils.files import save_to_json
from medvqa.datasets.qa_pairs_extractor import QuestionAnswerExtractor
from medvqa.utils.common import CACHE_DIR
from medvqa.datasets.mimiccxr.preprocessing import (
    extract_findings_and_impression,
    report_paths_generator
)

if __name__ == '__main__':

    qa_extractor = QuestionAnswerExtractor()

    qa_adapted_reports = [None] * 300000

    print('Loading MIMIC-CXR\'s original reports')
    
    for i, filepath in tqdm(enumerate(report_paths_generator())):
        text = extract_findings_and_impression(filepath.as_posix())
        qa_info = qa_extractor.generate_qa_pairs_compact_version(text)
        qa_info['filepath'] = str(filepath)
        qa_adapted_reports[i] = qa_info
    
    qa_adapted_reports = qa_adapted_reports[:i+1]    
    
    qa_dataset = {
        'questions': qa_extractor.questions,
        'reports': qa_adapted_reports,
    }

    save_path = os.path.join(CACHE_DIR, 'mimiccxr', 'qa_adapted_reports.json')
    print(f'Saving qa adapted reports to {save_path}')
    save_to_json(qa_dataset, save_path)
    print('Done!')

