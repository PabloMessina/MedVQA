from medvqa.utils.files_utils import load_json_file, save_to_pickle
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import get_timestamp
from tqdm import tqdm
import argparse
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iuxray-qa-dataset-filename', type=str, required=True)
    parser.add_argument('--mimiccxr-qa-dataset-filename', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, args.iuxray_qa_dataset_filename)
    iuxray_qa_adapted_reports = load_json_file(iuxray_qa_adapted_reports_path)

    mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, args.mimiccxr_qa_dataset_filename)
    mimiccxr_qa_adapted_reports = load_json_file(mimiccxr_qa_adapted_reports_path)

    assert iuxray_qa_adapted_reports['questions'] == mimiccxr_qa_adapted_reports['questions']
    questions = iuxray_qa_adapted_reports['questions']
    n_questions = len(questions)

    for qa_adapted_reports, cache_dir in zip(
                        [iuxray_qa_adapted_reports, mimiccxr_qa_adapted_reports],
                        [IUXRAY_CACHE_DIR, MIMICCXR_CACHE_DIR],
        ):
    
        output = np.zeros((len(qa_adapted_reports['reports']), n_questions), np.int8)        
        for ri, report in tqdm(enumerate(qa_adapted_reports['reports'])):
            for qid in report['question_ids']:
                output[ri][qid] = 1
    
        output_path = os.path.join(cache_dir, f'question_labels_per_report__{get_timestamp()}.pkl')
        save_to_pickle(output, output_path)
        print('Question labels saved to', output_path)