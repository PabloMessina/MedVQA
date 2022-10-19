from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.files import (
    load_json_file,
    load_pickle,
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
    parser.add_argument('--chexpert-labels-cache-filename', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, args.iuxray_qa_dataset_filename)
    mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, args.mimiccxr_qa_dataset_filename)
    iuxray_qa_adapted_reports = load_json_file(iuxray_qa_adapted_reports_path)
    mimiccxr_qa_adapted_reports = load_json_file(mimiccxr_qa_adapted_reports_path)

    hash2text = dict()

    tokenizer = Tokenizer(qa_adapted_dataset_paths=[iuxray_qa_adapted_reports_path, mimiccxr_qa_adapted_reports_path])

    print('Collecting unique sentences ...')

    for s in tqdm(get_sentences(
        [iuxray_qa_adapted_reports, mimiccxr_qa_adapted_reports],
        include_unmatched=False)):

        clean_s =  tokenizer.clean_text(s)
        h = hash_string(clean_s)
        prev_s = hash2text.get(h, None)
        
        if prev_s is not None:
            assert prev_s == clean_s, (h, prev_s, clean_s)
        else:
            hash2text[h] = clean_s

    text2hash = {t:h for h,t in hash2text.items()}
    
    hash2label = dict()

    if args.chexpert_labels_cache_filename is not None:
        labels_cache_path = os.path.join(CACHE_DIR, args.chexpert_labels_cache_filename)
        cached_labels = load_pickle(labels_cache_path)
        unlabeled_texts = []
        for text, hash in text2hash.items():
            try:
                hash2label[hash] = cached_labels[hash]
            except KeyError:
                unlabeled_texts.append(text)
    else:
        unlabeled_texts = list(text2hash.keys())


    timestamp = get_timestamp()

    n = len(unlabeled_texts)
    print('len(unlabeled_texts):', n)
    
    if n > 0:
        random.shuffle(unlabeled_texts)
        print('average length:', sum(len(x.split()) for x in unlabeled_texts) / len(unlabeled_texts))

        n_chunks = n // args.chunk_size + (n % args.chunk_size > 0)
        if n_chunks < args.max_processes:
            n_chunks = args.max_processes
        labels = invoke_chexpert_labeler_process(unlabeled_texts, f'_{timestamp}',
                                        n_chunks=n_chunks, max_processes=args.max_processes)
        for i in range(n):
            hash2label[text2hash[unlabeled_texts[i]]] = labels[i]
    
    output_path = os.path.join(CACHE_DIR, f'precomputed_chexpert_labels_{timestamp}.pkl')
    save_to_pickle(hash2label, output_path)
    print(f'Precomputed chexpert labels saved to {output_path}')
