import argparse
import math
import os
import numpy as np
import torch
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from medvqa.utils.data_structures import UnionFind
from medvqa.utils.files import (
    get_file_path_with_hashing_if_too_long,
    load_pickle,
    save_pickle,
)
from medvqa.datasets.mimiccxr import MIMICCXR_LARGE_FAST_CACHE_DIR
from medvqa.utils.logging import print_blue, print_bold

class FactEmbeddingsDataset(Dataset):
    def __init__(self, fact_embeddings):
        self.fact_embeddings = torch.tensor(fact_embeddings)

    def __len__(self):
        return len(self.fact_embeddings)
    
    def __getitem__(self, i):
        return self.fact_embeddings[i]
    
    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)
    
_shared_nearest_idxs = None
_shared_similarities = None
def find_most_similar_facts(i, threshold, max_len):
    most_similar = []
    for j in range(len(_shared_nearest_idxs[i])-1, -1, -1):
        nearest_idx = _shared_nearest_idxs[i, j].item()
        if _shared_similarities[i, nearest_idx] < threshold:
            break
        most_similar.append(nearest_idx)
        if len(most_similar) == max_len:
            break
    return most_similar
def find_least_similar_facts(i, threshold, max_len):
    least_similar = []
    for j in range(len(_shared_nearest_idxs[i])):
        nearest_idx = _shared_nearest_idxs[i, j].item()
        if _shared_similarities[i, nearest_idx] > threshold:
            break
        least_similar.append(nearest_idx)
        if len(least_similar) == max_len:
            break
    return least_similar


def deduplicate_facts_with_union_find(facts, fact_embeddings, threshold):
    # run kmeans
    n_clusters = math.ceil(len(facts) / 500) # roughly 500 facts per cluster
    n_clusters = max(n_clusters, 2) # at least 2 clusters
    print(f'Running kmeans with n_clusters={n_clusters}...')
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', max_iter=300, random_state=0).fit(fact_embeddings)
    labels = kmeans.labels_
    c2idxs = {}
    for i, label in enumerate(labels):
        if label not in c2idxs:
            c2idxs[label] = []
        c2idxs[label].append(i)
    # deduplicate facts
    print('Deduplicating facts...')
    uf = UnionFind(len(facts))
    for idxs in c2idxs.values():
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                if np.dot(fact_embeddings[idxs[i]], fact_embeddings[idxs[j]]) >= threshold:
                    uf.unionSet(idxs[i], idxs[j])
    # for each set, choose the longest fact
    print('Choosing longest fact for each set...')
    set2idx = {}
    for i in range(len(facts)):
        s = uf.findSet(i)
        if s not in set2idx:
            set2idx[s] = i
        else:
            if len(facts[i]) > len(facts[set2idx[s]]):
                set2idx[s] = i
    # return deduplicated facts and embeddings
    dedup_idxs = list(set2idx.values())
    dedup_idxs.sort()
    print('Number of facts removed:', len(facts) - len(dedup_idxs))
    dedup_facts = [facts[i] for i in dedup_idxs]
    dedup_fact_embeddings = fact_embeddings[dedup_idxs]
    return dedup_facts, dedup_fact_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrated_embeddings_and_report_annotations_filepath', type=str, required=True)
    parser.add_argument('--finding_only', action='store_true')
    # clustering parameters
    parser.add_argument('--num_kmeans_clusters', type=int, default=500)
    parser.add_argument('--num_kmeans_iterations', type=int, default=300)
    parser.add_argument('--num_kmedoids_clusters', type=int, default=25000)
    parser.add_argument("--num_kmedoids_iterations", type=int, default=300)
    parser.add_argument("--kmedoids_method", type=str, default="alternate", choices=["pam", "alternate"])
    # nearest neighbors parameters
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pos_sim_threshold", type=float, default=0.9)
    parser.add_argument("--neg_sim_threshold", type=float, default=0.3)
    parser.add_argument("--union_find_threshold", type=float, default=0.97)
    parser.add_argument("--max_num_pos", type=int, default=100)
    parser.add_argument("--max_num_neg", type=int, default=100)
    # misc
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_processes", type=int, default=8)
    args = parser.parse_args()

    # Load integrated embeddings and report annotations
    print_bold('Loading integrated embeddings and report annotations from:', args.integrated_embeddings_and_report_annotations_filepath)
    integrated_embeddings_and_report_annotations = load_pickle(args.integrated_embeddings_and_report_annotations_filepath)
    
    facts = integrated_embeddings_and_report_annotations['facts']
    fact_embeddings = integrated_embeddings_and_report_annotations['fact_embeddings']
    print('len(facts):', len(facts))
    print('fact_embeddings.shape:', fact_embeddings.shape)
    if args.finding_only:
        print('Selecting only findings facts...')
        findings_fact_idxs = set()
        for r in integrated_embeddings_and_report_annotations['reports']:
            findings_fact_idxs.update(r['findings_fact_idxs'])
        findings_fact_idxs = list(findings_fact_idxs)
        findings_fact_idxs.sort()
        facts = [facts[i] for i in findings_fact_idxs]
        fact_embeddings = fact_embeddings[findings_fact_idxs]
        print('  len(facts):', len(facts))
        print('  fact_embeddings.shape:', fact_embeddings.shape)
    fact2idx = {f: i for i, f in enumerate(facts)}
    
    # Find representative facts using kmeans and kmedoids

    representative_facts_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
        prefix='representative_facts',
        strings=[
            args.integrated_embeddings_and_report_annotations_filepath,
            f'num_kmeans_clusters={args.num_kmeans_clusters}',
            f'num_kmeans_iterations={args.num_kmeans_iterations}',
            f'num_kmedoids_clusters={args.num_kmedoids_clusters}',
            f'num_kmedoids_iterations={args.num_kmedoids_iterations}',
            f'kmedoids_method={args.kmedoids_method}',
            f'k={args.k}',
            f'finding_only={args.finding_only}',
            f'union_find_threshold={args.union_find_threshold}',
        ],
        force_hashing=True,
    )
    if os.path.exists(representative_facts_filepath):
        # Load representative facts
        print_bold('Loading representative facts from:', representative_facts_filepath)
        tmp = load_pickle(representative_facts_filepath)
        representative_facts = tmp['dedup_facts'] # use deduplicated facts
        representative_fact_embeddings = tmp['dedup_fact_embeddings'] # use deduplicated fact embeddings
        print('len(representative_facts):', len(representative_facts))
        print('representative_fact_embeddings.shape:', representative_fact_embeddings.shape)
    else:
        # 1. Run kmeans
        print_blue('Running kmeans...')
        kmeans = KMeans(n_clusters=args.num_kmeans_clusters, n_init='auto', max_iter=args.num_kmeans_iterations, random_state=0).fit(fact_embeddings)
        kmeans_labels = kmeans.labels_
        kmeans_c2idxs = {}
        for i, label in enumerate(kmeans_labels):
            if label not in kmeans_c2idxs:
                kmeans_c2idxs[label] = []
            kmeans_c2idxs[label].append(i)

        representative_fact_idxs = set()
        
        for idxs in kmeans_c2idxs.values():
            cluster_fact_embeddings = fact_embeddings[idxs]
            # 2. Run kmedoids
            print_blue('Running kmedoids...')
            n_clusters = math.ceil(len(idxs) * args.num_kmedoids_clusters / len(fact_embeddings)) # scale number of clusters based on number of facts
            print(f'  n_clusters: {n_clusters}, len(idxs): {len(idxs)}, cluster_fact_embeddings.shape: {cluster_fact_embeddings.shape}')
            kmedoids = KMedoids(n_clusters=n_clusters, metric='cosine', method=args.kmedoids_method,
                                max_iter=args.num_kmedoids_iterations, random_state=0).fit(cluster_fact_embeddings)
            kmedoids_labels = kmedoids.labels_
            kmedoids_centroids = kmedoids.cluster_centers_
            print(f'  kmedoids_labels.shape: {kmedoids_labels.shape}, kmedoids_centroids.shape: {kmedoids_centroids.shape}')
            c2idxs = {}
            for i, label in enumerate(kmedoids_labels):
                if label not in c2idxs:
                    c2idxs[label] = []
                c2idxs[label].append(i)
            for label, sub_idxs in c2idxs.items():
                centroid = kmedoids_centroids[label]
                sub_cluster_fact_embeddings = cluster_fact_embeddings[sub_idxs]
                # Find k nearest facts to centroid
                similarities = np.dot(sub_cluster_fact_embeddings, centroid)
                nearest_sentence_idxs = np.argsort(similarities)[::-1][:args.k]
                # Choose the longest sentence as the representative sentence
                max_len = 0
                max_len_idx = None
                for idx in nearest_sentence_idxs:
                    f = facts[idxs[sub_idxs[idx]]]
                    if len(f) > max_len:
                        max_len = len(f)
                        max_len_idx = idx
                assert max_len_idx is not None
                max_len_idx = idxs[sub_idxs[max_len_idx]]
                assert max_len_idx not in representative_fact_idxs
                representative_fact_idxs.add(max_len_idx)

        print('len(representative_fact_idxs):', len(representative_fact_idxs))
        representative_fact_idxs = list(representative_fact_idxs)
        representative_fact_idxs.sort()
        representative_fact_embeddings = fact_embeddings[representative_fact_idxs]
        representative_facts = [facts[i] for i in representative_fact_idxs]
        
        # De-duplicate representative facts with union-find
        print_blue('De-duplicating representative facts with union-find...')
        dedup_facts, dedup_fact_embeddings = deduplicate_facts_with_union_find(
            representative_facts, representative_fact_embeddings, args.union_find_threshold)
        print('len(dedup_facts):', len(dedup_facts))
        print('dedup_fact_embeddings.shape:', dedup_fact_embeddings.shape)

        print('Saving representative facts to:', representative_facts_filepath)
        save_pickle({
            'representative_facts': representative_facts,
            'representative_fact_embeddings': representative_fact_embeddings,
            'dedup_facts': dedup_facts,
            'dedup_fact_embeddings': dedup_fact_embeddings,
        }, representative_facts_filepath)

        # Use deduplicated facts and embeddings
        representative_fact_embeddings = dedup_fact_embeddings
        representative_facts = dedup_facts

    # Find positive and negative representative facts for each fact
    pos_and_neg_representative_facts_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
        prefix='pos_and_neg_representative_facts',
        strings=[
            args.integrated_embeddings_and_report_annotations_filepath,
            f'num_kmeans_clusters={args.num_kmeans_clusters}',
            f'num_kmeans_iterations={args.num_kmeans_iterations}',
            f'num_kmedoids_clusters={args.num_kmedoids_clusters}',
            f'num_kmedoids_iterations={args.num_kmedoids_iterations}',
            f'kmedoids_method={args.kmedoids_method}',
            f'k={args.k}',
            f'pos_sim_threshold={args.pos_sim_threshold}',
            f'neg_sim_threshold={args.neg_sim_threshold}',
            f'max_num_pos={args.max_num_pos}',
            f'max_num_neg={args.max_num_neg}',
            f'finding_only={args.finding_only}',
            f'union_find_threshold={args.union_find_threshold}',
        ],
        force_hashing=True,
    )
    if os.path.exists(pos_and_neg_representative_facts_filepath):
        # Load positive and negative representative facts
        print_bold('Loading positive and negative representative facts from:', pos_and_neg_representative_facts_filepath)
        pos_and_neg_representative_facts = load_pickle(pos_and_neg_representative_facts_filepath)
        most_similar = pos_and_neg_representative_facts['most_similar']
        least_similar = pos_and_neg_representative_facts['least_similar']
        print('len(most_similar):', len(most_similar))
        print('len(least_similar):', len(least_similar))
    else:
        print_blue('Finding positive and negative representative facts for each fact...')
        most_similar = [[] for _ in range(len(fact_embeddings))]
        least_similar = [[] for _ in range(len(fact_embeddings))]
        # Create dataset and dataloader
        dataset = FactEmbeddingsDataset(fact_embeddings)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=dataset.collate_fn)
        r_f_embs = torch.from_numpy(representative_fact_embeddings).cuda()
        offset = 0
        for batch in tqdm(dataloader, mininterval=2):
            batch_embs = batch.cuda()
            similarities = torch.matmul(batch_embs, r_f_embs.T)
            nearest_idxs = torch.argsort(similarities, dim=1)
            similarities = similarities.cpu().numpy()
            nearest_idxs = nearest_idxs.cpu().numpy()
            assert similarities[0, nearest_idxs[0, 0]] < similarities[0, nearest_idxs[0, -1]]
            global _shared_nearest_idxs
            global _shared_similarities
            _shared_nearest_idxs = nearest_idxs
            _shared_similarities = similarities
            # Find negative representative facts
            tasks = [(i, args.neg_sim_threshold, args.max_num_neg) for i in range(len(nearest_idxs))]
            with mp.Pool(processes=args.num_processes) as pool:
                results = pool.starmap(find_least_similar_facts, tasks)
            for i, result in enumerate(results):
                idx = offset + i
                least_similar[idx] = result
            # Find positive representative facts
            tasks = [(i, args.pos_sim_threshold, args.max_num_pos) for i in range(len(nearest_idxs))]
            with mp.Pool(processes=args.num_processes) as pool:
                results = pool.starmap(find_most_similar_facts, tasks)
            for i, result in enumerate(results):
                idx = offset + i
                most_similar[idx] = result
            
            if offset == 0: # sanity check
                for i in range(len(batch)):
                    if len(most_similar[i]) == 0:
                        continue
                    print(f'facts[{i}]: {facts[i]}')
                    print()
                    print(f'most_similar[{i}]:')
                    for j in range(min(len(most_similar[i]), 5)):
                        print(f'  {representative_facts[most_similar[i][j]]}')
                    print()
                    print(f'least_similar[{i}]:')
                    for j in range(min(len(least_similar[i]), 5)):
                        print(f'  {representative_facts[least_similar[i][j]]}')
                    break # only print one

            offset += len(batch)
        print('len(most_similar):', len(most_similar))
        print('len(least_similar):', len(least_similar))
        print('Saving positive and negative representative facts to:', pos_and_neg_representative_facts_filepath)
        save_pickle({
            'most_similar': most_similar,
            'least_similar': least_similar,
        }, pos_and_neg_representative_facts_filepath)

    # Find positive and negative representative facts for each report
    output = {
        'reports': [],
        'representative_facts': representative_facts,
    }
    orig_facts = integrated_embeddings_and_report_annotations['facts']
    for r in tqdm(integrated_embeddings_and_report_annotations['reports'], mininterval=2):
        if args.finding_only:
            r_fact_idxs = r['findings_fact_idxs']
        else:
            r_fact_idxs = r['findings_fact_idxs'] + r['impression_fact_idxs']
        r_fact_idxs = list(set(r_fact_idxs)) # remove duplicates
        r_fact_idxs = [fact2idx[orig_facts[i]] for i in r_fact_idxs] # use updated indices
        
        # Find positive representative facts
        pos_representative_fact_idxs = set()
        if len(r_fact_idxs) > 0:
            r_fact_idxs.sort(key=lambda x: len(most_similar[x]), reverse=True)
            for i in range(len(most_similar[r_fact_idxs[0]])):
                for j in range(len(r_fact_idxs)):
                    if len(most_similar[r_fact_idxs[j]]) <= i:
                        break
                    pos_representative_fact_idxs.add(most_similar[r_fact_idxs[j]][i])
                    if len(pos_representative_fact_idxs) == args.max_num_pos:
                        break
                if len(pos_representative_fact_idxs) == args.max_num_pos:
                    break
        pos_representative_fact_idxs = list(pos_representative_fact_idxs)
        
        # Find negative representative facts
        if len(r_fact_idxs) > 0:
            r_least_similar = set()
            for idx in r_fact_idxs:
                r_least_similar.update(least_similar[idx])
            r_least_similar = list(r_least_similar)
            if len(r_least_similar) > 0:
                r_least_similar_embeddings = representative_fact_embeddings[r_least_similar]
                r_fact_embeddings = fact_embeddings[r_fact_idxs]
                # Compute similarities between least similar facts and report facts
                similarities = np.dot(r_fact_embeddings, r_least_similar_embeddings.T)
                max_similarities = np.max(similarities, axis=0)
                assert len(max_similarities) == len(r_least_similar)
                max_similarities_idxs = np.argsort(max_similarities) # ascending order
                neg_representative_fact_idxs = []
                for idx in max_similarities_idxs:
                    if max_similarities[idx] > args.neg_sim_threshold:
                        break
                    neg_representative_fact_idxs.append(r_least_similar[idx])
                    if len(neg_representative_fact_idxs) == args.max_num_neg:
                        break
            else:
                neg_representative_fact_idxs = []
        else:
            neg_representative_fact_idxs = []

        # Add to output
        output['reports'].append({
            'path': r['path'],
            'part_id': r['part_id'],
            'subject_id': r['subject_id'],
            'study_id': r['study_id'],
            'positive_facts': pos_representative_fact_idxs,
            'negative_facts': neg_representative_fact_idxs,
        })

    # Save output
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
        prefix='pos_and_neg_representative_facts_per_report',
        strings=[
            args.integrated_embeddings_and_report_annotations_filepath,
            f'num_kmeans_clusters={args.num_kmeans_clusters}',
            f'num_kmeans_iterations={args.num_kmeans_iterations}',
            f'num_kmedoids_clusters={args.num_kmedoids_clusters}',
            f'num_kmedoids_iterations={args.num_kmedoids_iterations}',
            f'kmedoids_method={args.kmedoids_method}',
            f'k={args.k}',
            f'pos_sim_threshold={args.pos_sim_threshold}',
            f'neg_sim_threshold={args.neg_sim_threshold}',
            f'max_num_pos={args.max_num_pos}',
            f'max_num_neg={args.max_num_neg}',
            f'finding_only={args.finding_only}'
        ],
        force_hashing=True,
    )
    print_bold('Saving output to:', output_filepath)
    save_pickle(output, output_filepath)
    print('Done!')

if __name__ == '__main__':
    main()