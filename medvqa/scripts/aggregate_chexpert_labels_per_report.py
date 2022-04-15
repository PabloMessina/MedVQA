from medvqa.utils.files import load_pickle, load_json_file, save_to_pickle
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import CACHE_DIR, get_timestamp
from medvqa.utils.hashing import hash_string
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.metrics.medical.chexpert import merge_raw_labels
from tqdm import tqdm
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iuxray-qa-dataset-filename', type=str, required=True)
    parser.add_argument('--mimiccxr-qa-dataset-filename', type=str, required=True)
    parser.add_argument('--chexpert-labels-cache-filename', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, args.iuxray_qa_dataset_filename)
    iuxray_qa_adapted_reports = load_json_file(iuxray_qa_adapted_reports_path)

    mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, args.mimiccxr_qa_dataset_filename)
    mimiccxr_qa_adapted_reports = load_json_file(mimiccxr_qa_adapted_reports_path)

    chexpert_labels_cache_path = os.path.join(CACHE_DIR, args.chexpert_labels_cache_filename)
    chexpert_labels_cache = load_pickle(chexpert_labels_cache_path)

    tokenizer = Tokenizer(qa_adapted_filenames=[args.iuxray_qa_dataset_filename, args.mimiccxr_qa_dataset_filename])

    for qa_adapted_reports, cache_dir in zip(
                        [iuxray_qa_adapted_reports, mimiccxr_qa_adapted_reports],
                        [IUXRAY_CACHE_DIR, MIMICCXR_CACHE_DIR],
        ):
    
        output = [None] * len(qa_adapted_reports['reports'])
        
        for ri, report in tqdm(enumerate(qa_adapted_reports['reports'])):
            sentences = report['sentences']
            labels_list = []
            for i in report['matched']:
                clean_s = tokenizer.ids2string(tokenizer.clean_sentence(tokenizer.string2ids(sentences[i].lower())))
                hash = hash_string(clean_s)
                labels_list.append(chexpert_labels_cache[hash])
            output[ri] = merge_raw_labels(labels_list)
            
            # # DEBUGGING
            # if 30 <= ri < 60:
            #     print('----------------------')
            #     print('\n'.join(report['sentences'][i] for i in report['matched']))
            #     print(output[ri])
            #     print(', '.join(CHEXPERT_LABELS[i] for i, x in enumerate(output[ri]) if x))
            # if ri == 60:
            #     assert False
    
        output_path = os.path.join(cache_dir, f'chexpert_labels_per_report__{get_timestamp()}.pkl')
        save_to_pickle(output, output_path)
        print('Chexpert labels aggregated and saved to', output_path)