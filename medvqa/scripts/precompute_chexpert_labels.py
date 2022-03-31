from tokenize import Token
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.files import (
    load_json_file,
    save_to_pickle,
)
from medvqa.datasets.preprocessing import get_sentences
from medvqa.utils.common import get_timestamp
from medvqa.utils.hashing import hash_string
from medvqa.metrics.medical.chexpert import invoke_chexpert_labeler_process
from medvqa.datasets.tokenizer import Tokenizer

from tqdm import tqdm
import argparse
import os
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iuxray-qa-dataset-filename', type=str, required=True)
    parser.add_argument('--mimiccxr-qa-dataset-filename', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('--max-processes', type=int, default=16)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, args.iuxray_qa_dataset_filename)
    mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, args.mimiccxr_qa_dataset_filename)
    iuxray_qa_adapted_reports = load_json_file(iuxray_qa_adapted_reports_path)
    mimiccxr_qa_adapted_reports = load_json_file(mimiccxr_qa_adapted_reports_path)

    hash2text = dict()

    tokenizer = Tokenizer(qa_adapted_filenames=[args.iuxray_qa_dataset_filename, args.mimiccxr_qa_dataset_filename],
                         qa_adapted_datasets=[iuxray_qa_adapted_reports, mimiccxr_qa_adapted_reports])

    print('Collecting unique sentences ...')

    for s in tqdm(get_sentences(
        [iuxray_qa_adapted_reports, mimiccxr_qa_adapted_reports],
        include_unmatched=False)):

        clean_s =  tokenizer.ids2string(tokenizer.clean_sentence(tokenizer.string2ids(s)))

        h = hash_string(clean_s)
        prev_s = hash2text.get(h, None)
        if prev_s is not None:
            assert prev_s == clean_s, (h, prev_s, clean_s)
        else:
            hash2text[h] = clean_s

    text2hash = {t:h for h,t in hash2text.items()}
    texts = list(text2hash.keys())
    random.shuffle(texts)

    print('len(texts):', len(texts))
    print('average length:', sum(len(x) for x in texts) / len(texts))

    timestamp = get_timestamp()

    n = len(texts)
    n_chunks = n // args.chunk_size + (n % args.chunk_size > 0)
    labels = invoke_chexpert_labeler_process(texts, f'_{timestamp}',
                                    n_chunks=n_chunks, max_processes=args.max_processes)
    hash2label = { text2hash[texts[i]] : labels[i] for i in range(len(texts)) }
    
    output_path = os.path.join(CACHE_DIR, f'precomputed_chexpert_labels_{timestamp}.pkl')
    save_to_pickle(hash2label, output_path)
    print(f'Precomputed chexpert labels saved to {output_path}')



