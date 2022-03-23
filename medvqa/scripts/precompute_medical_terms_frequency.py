from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.metrics.medical.med_completeness import MEDICAL_TERMS_PATH
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.files import (
    load_json_file,
    read_lines_from_txt,
    save_to_pickle,
)
from medvqa.datasets.preprocessing import get_sentences
from medvqa.utils.common import get_timestamp

from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iuxray-qa-dataset-filename', type=str, required=True)
    parser.add_argument('--mimiccxr-qa-dataset-filename', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    print('Computing medical terms frequency ...')
    
    iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, args.iuxray_qa_dataset_filename)
    mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, args.mimiccxr_qa_dataset_filename)
    iuxray_qa_adapted_reports = load_json_file(iuxray_qa_adapted_reports_path)
    mimiccxr_qa_adapted_reports = load_json_file(mimiccxr_qa_adapted_reports_path)
    medical_terms = set(read_lines_from_txt(MEDICAL_TERMS_PATH))
    
    term2freq = {}
    for sentence in tqdm(get_sentences(
        [iuxray_qa_adapted_reports, mimiccxr_qa_adapted_reports],
        include_unmatched=False)):
        for token in wordpunct_tokenize(sentence):
            if token in medical_terms:
                term2freq[token] = term2freq.get(token, 0) + 1
    
    output_path = os.path.join(CACHE_DIR, f'medical_terms_frequency__{get_timestamp()}.pkl')
    save_to_pickle(term2freq, output_path)
    print('Medical terms frequency computed and saved to', output_path)