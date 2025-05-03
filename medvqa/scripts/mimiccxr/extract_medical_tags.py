from medvqa.utils.files_utils import load_json_file, save_to_pickle
from medvqa.datasets.medical_tags_extractor import MedicalTagsExtractor
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import get_timestamp
from tqdm import tqdm
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimiccxr-qa-dataset-filename', type=str, required=True)
    parser.add_argument('--medical-terms-frequency-filename', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, args.mimiccxr_qa_dataset_filename)
    
    med_tags_extractor = MedicalTagsExtractor(args.medical_terms_frequency_filename)
    qa_adapted_reports = load_json_file(qa_adapted_reports_path)
    
    output = [None] * len(qa_adapted_reports['reports'])
    
    for ri, report in tqdm(enumerate(qa_adapted_reports['reports'])):
        sentences = report['sentences']
        text = '. '.join(sentences[i] for i in report['matched'])
        output[ri] = med_tags_extractor.extract_tag_ids(text)
    
    output_path = os.path.join(MIMICCXR_CACHE_DIR,
        f'medical_tags_per_report__from({args.mimiccxr_qa_dataset_filename})__{get_timestamp()}.pkl')
    save_to_pickle(output, output_path)
    print('MIMICCXR medical tags per report computed and saved to', output_path)
        

