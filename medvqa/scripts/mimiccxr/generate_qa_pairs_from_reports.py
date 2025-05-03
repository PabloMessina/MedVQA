from medvqa.utils.files_utils import save_to_json
from medvqa.datasets.qa_pairs_extractor import QuestionAnswerExtractor
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.datasets.mimiccxr.preprocessing import (
    extract_report_and_patient_background,
    get_reports_txt_paths,
)
from medvqa.utils.common import get_timestamp
from tqdm import tqdm
import os

if __name__ == '__main__':
    
    qa_extractor = QuestionAnswerExtractor()

    qa_adapted_reports = [None] * 300000

    print('Processing MIMIC-CXR\'s original reports ...')

    for i, filepath in tqdm(enumerate(get_reports_txt_paths())):
        out = extract_report_and_patient_background(filepath)
        qa_info = qa_extractor.generate_qa_pairs_compact_version(out['report'])
        qa_info['filepath'] = str(filepath)
        qa_info['background'] = out['background']
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

