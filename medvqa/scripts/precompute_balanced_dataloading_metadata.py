from medvqa.utils.files_utils import load_pickle, load_json_file, save_to_pickle
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.common import CACHE_DIR, get_timestamp
from medvqa.utils.hashing_utils import hash_string
from medvqa.datasets.tokenizer import Tokenizer
from medvqa.metrics.medical.chexpert import merge_raw_labels
from medvqa.datasets.medical_tags_extractor import MedicalTagsExtractor
from tqdm import tqdm
import argparse
import os
# from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iuxray-qa-dataset-filename', type=str, required=True)
    parser.add_argument('--mimiccxr-qa-dataset-filename', type=str, required=True)
    parser.add_argument('--chexpert-labels-cache-filename', type=str, required=True)
    parser.add_argument('--medical-terms-frequency-filename', type=str, required=True)
    return parser.parse_args()

def _get_top_k_ngrams(sentences, med_tags_extractor, used_vocab, k, n):
    freq = dict()
    for s in sentences:
        tags = med_tags_extractor.extract_tags_sequence(s)
        for j in range(len(tags)-n+1):
            key = tuple(tags[j:j+n])
            freq[key] = freq.get(key, 0) + 1
    pairs = [(f,k) for k,f in freq.items()]
    pairs.sort(reverse=True)
    output = []
    for p in pairs:
        valid = True
        for w in p[1]:
            if w in used_vocab:
                valid = False
                break
        if valid:
            output.append(p)
            used_vocab.update(p[1])
            if len(output) == k:
                break
    return output

def get_top_k_ngrams(sentences, med_tags_extractor, ks=[50, 50], ns=[2, 1]):
    used_vocab = set()
    output = []
    for k, n in zip(ks, ns):
        pairs = _get_top_k_ngrams(sentences, med_tags_extractor, used_vocab, k, n)
        output.extend(pairs)
    output.sort()
    return output

def sentence_generator(reports, q_id, idxs, health_info, healthy=True):
    for ri in idxs:
        if healthy != health_info[ri][q_id]:
            continue
        report = reports[ri]
        sentences = report['sentences']
        qa = report['qa']
        for i in qa[q_id]:
            yield sentences[i]

def classify_sentence(s, med_tags_extractor, top_ngrams, max_n):
    tags = med_tags_extractor.extract_tags_sequence(s)
    ngram_sets = [set() for _ in range(max_n)]
    for n in range(1, max_n+1):
        for i in range(len(tags)-n+1):
            ngram = tuple(tags[i:i+n])
            ngram_sets[n-1].add(ngram)
    for i, ngram in enumerate(top_ngrams):
        if ngram[1] in ngram_sets[len(ngram[1])-1]:
            return i
    return len(top_ngrams)

if __name__ == '__main__':

    args = parse_args()

    print('Loading files ...')

    iuxray_qa_adapted_reports_path = os.path.join(IUXRAY_CACHE_DIR, args.iuxray_qa_dataset_filename)
    iuxray_qa_adapted_reports = load_json_file(iuxray_qa_adapted_reports_path)

    mimiccxr_qa_adapted_reports_path = os.path.join(MIMICCXR_CACHE_DIR, args.mimiccxr_qa_dataset_filename)
    mimiccxr_qa_adapted_reports = load_json_file(mimiccxr_qa_adapted_reports_path)

    chexpert_labels_cache_path = os.path.join(CACHE_DIR, args.chexpert_labels_cache_filename)
    chexpert_labels_cache = load_pickle(chexpert_labels_cache_path)

    med_tags_extractor = MedicalTagsExtractor(args.medical_terms_frequency_filename)

    tokenizer = Tokenizer(qa_adapted_dataset_paths=[iuxray_qa_adapted_reports_path, mimiccxr_qa_adapted_reports_path])

    print('Precomputing metadata ...')

    for qa_adapted_reports, cache_dir in zip(
            [iuxray_qa_adapted_reports, mimiccxr_qa_adapted_reports],
            [IUXRAY_CACHE_DIR, MIMICCXR_CACHE_DIR],
        ):
    
        reports = qa_adapted_reports['reports']

        output = {
            'healthy': [dict() for _ in range(len(reports))],
            'tags_based_class': [dict() for _ in range(len(reports))],
            'top_ngrams': dict()
        }

        q2idxs = { str(k):[] for k in range(len(qa_adapted_reports['questions'])) }


        for ri, report in tqdm(enumerate(reports)):
            sentences = report['sentences']
            tmp = output['healthy'][ri]
            for k, v in report['qa'].items():
                q2idxs[k].append(ri)
                labels_list = []
                for i in v:
                    clean_s = tokenizer.ids2string(tokenizer.clean_sentence(tokenizer.string2ids(sentences[i].lower())))
                    hash = hash_string(clean_s)
                    labels_list.append(chexpert_labels_cache[hash])
                merged_labels = merge_raw_labels(labels_list)
                tmp[k] = merged_labels[0]

        q2hngrams = dict()
        q2unhngrams = dict()

        # DEBUG = True

        for q_id, idxs in tqdm(q2idxs.items()):

            healthy_sentences = [s for s in sentence_generator(reports, q_id, idxs, output['healthy'], True)]
            unhealthy_sentences = [s for s in sentence_generator(reports, q_id, idxs, output['healthy'], False)]
            assert len(healthy_sentences) + len(unhealthy_sentences) >= len(idxs), (len(healthy_sentences), len(unhealthy_sentences), len(idxs))
            healthy_top_ngrams = get_top_k_ngrams(healthy_sentences, med_tags_extractor)
            unhealthy_top_ngrams = get_top_k_ngrams(unhealthy_sentences, med_tags_extractor)
            q2hngrams[q_id] = healthy_top_ngrams
            q2unhngrams[q_id] = unhealthy_top_ngrams
            output['top_ngrams'][(int(q_id), 1)] = healthy_top_ngrams
            output['top_ngrams'][(int(q_id), 0)] = unhealthy_top_ngrams

            # if DEBUG and len(idxs) > 0:
            #     print(qa_adapted_reports['questions'][int(q_id)])
            #     pprint(q2hngrams[q_id])
            #     pprint(q2unhngrams[q_id])
            #     DEBUG = False
        
        for ri, report in tqdm(enumerate(reports)):

            sentences = report['sentences']
            tmp = output['healthy'][ri]
            tmp2 = output['tags_based_class'][ri]

            # print("=================")
            # pprint(sentences)
            # pprint(tmp)
            for k, v in report['qa'].items():
                healthy = tmp[k]
                ngrams = q2hngrams[k] if healthy else q2unhngrams[k]
                assert len(v) > 0
                class_label = min(classify_sentence(sentences[i], med_tags_extractor, ngrams, 2) for i in v)
                tmp2[k] = class_label            
            # pprint(tmp2)
            # assert (False)
    
        output_path = os.path.join(cache_dir, f'balanced_dataloading_metadata__{get_timestamp()}.pkl')
        save_to_pickle(output, output_path)
        print('Balanced dataloading metadata saved to', output_path)