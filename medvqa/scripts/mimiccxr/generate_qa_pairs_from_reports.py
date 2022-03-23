from medvqa.utils.files import save_to_json
from medvqa.datasets.qa_pairs_extractor import QuestionAnswerExtractor
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.datasets.mimiccxr.preprocessing import (
    extract_findings_and_impression,
    report_paths_generator,
)
from medvqa.utils.common import get_timestamp
from tqdm import tqdm
import os

if __name__ == '__main__':
    
    qa_extractor = QuestionAnswerExtractor()

    qa_adapted_reports = [None] * 300000

    print('Processing MIMIC-CXR\'s original reports ...')

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

    output_path = os.path.join(MIMICCXR_CACHE_DIR, f'qa_adapted_reports__{get_timestamp()}.json')
    print(f'Saving qa adapted reports to {output_path}')
    save_to_json(qa_dataset, output_path)
    print('Done!')

