import argparse
import math
import os
import time
import numpy as np
import torch
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from medvqa.datasets.text_data_utils import word_tokenize_texts_in_parallel
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata, load_model_state_dict
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.models.nlp.nli import BertBasedNLI
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
    
class NLIEmbeddingsDataset(Dataset):
    def __init__(self, P_embeddings, H_embeddings, P_index, H_index):
        self.P_embeddings = P_embeddings
        self.H_embeddings = H_embeddings
        self.P_index = P_index
        self.H_index = H_index

    def __len__(self):
        return len(self.P_index)
    
    def __getitem__(self, i):
        return (self.P_index[i], self.H_index[i])
    
    def collate_fn(self, batch):
        P_indexes = [x[0] for x in batch]
        H_indexes = [x[1] for x in batch]
        P = torch.tensor(self.P_embeddings[P_indexes])
        H = torch.tensor(self.H_embeddings[H_indexes])
        return {
            'P': P,
            'H': H,
        }
    
_shared_nearest_idxs = None
_shared_similarities = None

def _find_most_similar_facts(i, threshold, max_len):
    most_similar = []
    for j in range(len(_shared_nearest_idxs[i])-1, -1, -1):
        nearest_idx = _shared_nearest_idxs[i, j].item()
        if _shared_similarities[i, nearest_idx] < threshold:
            break
        most_similar.append(nearest_idx)
        if len(most_similar) == max_len:
            break
    most_similar_similarities = _shared_similarities[i, most_similar]
    return most_similar, most_similar_similarities

def _find_most_similar_facts_no_threshold(i, max_len):
    most_similar = []
    for j in range(len(_shared_nearest_idxs[i])-1, -1, -1):
        nearest_idx = _shared_nearest_idxs[i, j].item()
        most_similar.append(nearest_idx)
        if len(most_similar) == max_len:
            break
    most_similar_similarities = _shared_similarities[i, most_similar]
    return most_similar, most_similar_similarities

def _find_least_similar_facts(i, threshold, max_len):
    least_similar = []
    for j in range(len(_shared_nearest_idxs[i])):
        nearest_idx = _shared_nearest_idxs[i, j].item()
        if _shared_similarities[i, nearest_idx] > threshold:
            break
        least_similar.append(nearest_idx)
        if len(least_similar) == max_len:
            break
    least_similar_similarities = _shared_similarities[i, least_similar]
    return least_similar, least_similar_similarities

def _find_least_similar_facts_no_threshold(i, max_len):
    least_similar = []
    for j in range(len(_shared_nearest_idxs[i])):
        nearest_idx = _shared_nearest_idxs[i, j].item()
        least_similar.append(nearest_idx)
        if len(least_similar) == max_len:
            break
    least_similar_similarities = _shared_similarities[i, least_similar]
    return least_similar, least_similar_similarities

def _find_middle_similar_facts(i, num_pos, num_middle, num_neg):
    n_total_middle = len(_shared_nearest_idxs[i]) - num_pos - num_neg
    assert n_total_middle > num_middle
    idxs = np.random.choice(n_total_middle, num_middle, replace=False)
    middle_similar = [_shared_nearest_idxs[i, j+num_pos].item() for j in idxs]
    middle_similar_similarities = _shared_similarities[i, middle_similar]
    return middle_similar, middle_similar_similarities

def _deduplicate_facts_with_union_find(facts, fact_embeddings, threshold, fact_selection_method, tokenized_facts=None, token2count=None):
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
    # for each set, choose a single representative fact
    print('Choosing representative fact per set...')
    set2idx = {}
    if fact_selection_method == 'shortest':
        for i in range(len(facts)):
            s = uf.findSet(i)
            if s not in set2idx:
                set2idx[s] = i
            else:
                if len(facts[i]) < len(facts[set2idx[s]]):
                    set2idx[s] = i
    elif fact_selection_method == 'longest':
        for i in range(len(facts)):
            s = uf.findSet(i)
            if s not in set2idx:
                set2idx[s] = i
            else:
                if len(facts[i]) > len(facts[set2idx[s]]):
                    set2idx[s] = i
    elif fact_selection_method == 'max_avg_word_count':
        assert tokenized_facts is not None
        assert token2count is not None
        fact_scores = [None] * len(facts)
        for i in range(len(facts)):
            s = uf.findSet(i)
            fact_scores[i] = sum(token2count[token] for token in tokenized_facts[i]) / len(tokenized_facts[i])
            if s not in set2idx:
                set2idx[s] = i
            else:
                if fact_scores[i] > fact_scores[set2idx[s]]:
                    set2idx[s] = i
    else:
        raise ValueError(f'Invalid fact_selection_method: {fact_selection_method}')
    # return deduplicated facts and embeddings
    dedup_idxs = list(set2idx.values())
    dedup_idxs.sort()
    print('Number of facts removed:', len(facts) - len(dedup_idxs))
    dedup_facts = [facts[i] for i in dedup_idxs]
    dedup_fact_embeddings = fact_embeddings[dedup_idxs]
    return dedup_facts, dedup_fact_embeddings

def find_representative_facts(args, facts, fact_embeddings):

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
            f'nearest_k={args.nearest_k}',
            f'finding_only={args.finding_only}',
            f'union_find_threshold={args.union_find_threshold}',
            f'fact_selection_method={args.fact_selection_method}',
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

        # 0. If fact_selection_method is max_avg_word_count, precompute tokenized facts and token2count
        if args.fact_selection_method == 'max_avg_word_count':
            print_blue('Precomputing tokenized facts and token2count...')
            tokenized_facts = word_tokenize_texts_in_parallel(facts)
            token2count = {}
            for f in tokenized_facts:
                for token in f:
                    token2count[token] = token2count.get(token, 0) + 1

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
                nearest_sentence_idxs = np.argsort(similarities)[::-1][:args.nearest_k]
                # Choose representative fact
                if args.fact_selection_method == 'shortest':
                    min_len = 999999999
                    min_len_idx = None
                    for idx in nearest_sentence_idxs:
                        f = facts[idxs[sub_idxs[idx]]]
                        if len(f) < min_len:
                            min_len = len(f)
                            min_len_idx = idx
                    assert min_len_idx is not None
                    min_len_idx = idxs[sub_idxs[min_len_idx]]
                    assert min_len_idx not in representative_fact_idxs
                    representative_fact_idxs.add(min_len_idx)
                elif args.fact_selection_method == 'longest':
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
                elif args.fact_selection_method == 'max_avg_word_count':
                    max_avg_word_count = 0
                    max_avg_word_count_idx = None
                    for idx in nearest_sentence_idxs:
                        f = tokenized_facts[idxs[sub_idxs[idx]]]
                        avg_word_count = sum(token2count[token] for token in f) / len(f)
                        if avg_word_count > max_avg_word_count:
                            max_avg_word_count = avg_word_count
                            max_avg_word_count_idx = idx
                    assert max_avg_word_count_idx is not None
                    max_avg_word_count_idx = idxs[sub_idxs[max_avg_word_count_idx]]
                    assert max_avg_word_count_idx not in representative_fact_idxs
                    representative_fact_idxs.add(max_avg_word_count_idx)
                else:
                    raise ValueError(f'Invalid fact_selection_method: {args.fact_selection_method}')

        print('len(representative_fact_idxs):', len(representative_fact_idxs))
        representative_fact_idxs = list(representative_fact_idxs)
        representative_fact_idxs.sort()
        representative_fact_embeddings = fact_embeddings[representative_fact_idxs]
        representative_facts = [facts[i] for i in representative_fact_idxs]
        
        # De-duplicate representative facts with union-find
        print_blue('De-duplicating representative facts with union-find...')
        dedup_facts, dedup_fact_embeddings = _deduplicate_facts_with_union_find(
            representative_facts, representative_fact_embeddings, args.union_find_threshold, args.fact_selection_method,
            tokenized_facts=tokenized_facts, token2count=token2count)
        print('len(dedup_facts):', len(dedup_facts))
        print('dedup_fact_embeddings.shape:', dedup_fact_embeddings.shape)

        print_bold('Saving representative facts to:', representative_facts_filepath)
        save_pickle({
            'representative_facts': representative_facts,
            'representative_fact_embeddings': representative_fact_embeddings,
            'dedup_facts': dedup_facts,
            'dedup_fact_embeddings': dedup_fact_embeddings,
        }, representative_facts_filepath)

        # Use deduplicated facts and embeddings
        representative_fact_embeddings = dedup_fact_embeddings
        representative_facts = dedup_facts

    return representative_facts, representative_fact_embeddings

def algorithm1__pos_and_neg_representative_facts_per_report_based_on_similarity(
        args, integrated_embeddings_and_report_annotations, facts, fact_embeddings, fact2idx,
        representative_facts, representative_fact_embeddings):
    
    global _shared_nearest_idxs
    global _shared_similarities

    # Find positive and negative representative facts for each fact
    pos_and_neg_representative_facts_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
        prefix='pos_and_neg_representative_facts',
        strings=[
            'algorithm1',
            args.integrated_embeddings_and_report_annotations_filepath,
            f'num_kmeans_clusters={args.num_kmeans_clusters}',
            f'num_kmeans_iterations={args.num_kmeans_iterations}',
            f'num_kmedoids_clusters={args.num_kmedoids_clusters}',
            f'num_kmedoids_iterations={args.num_kmedoids_iterations}',
            f'kmedoids_method={args.kmedoids_method}',
            f'nearest_k={args.nearest_k}',
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
            _shared_nearest_idxs = nearest_idxs
            _shared_similarities = similarities
            # Find negative representative facts
            tasks = [(i, args.neg_sim_threshold, args.max_num_neg) for i in range(len(nearest_idxs))]
            with mp.Pool(processes=args.num_processes) as pool:
                results = pool.starmap(_find_least_similar_facts, tasks)
            for i, result in enumerate(results):
                idx = offset + i
                least_similar[idx] = result
            # Find positive representative facts
            tasks = [(i, args.pos_sim_threshold, args.max_num_pos) for i in range(len(nearest_idxs))]
            with mp.Pool(processes=args.num_processes) as pool:
                results = pool.starmap(_find_most_similar_facts, tasks)
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
        'representative_fact_embeddings': representative_fact_embeddings,
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

    return output

def _compute_NLI_softmaxes_and_classes(args, P_embeddings, H_embeddings, P_index, H_index):

    # Device
    device = torch.device(args.device)

    # Load model
    metadata = load_metadata(args.nli_checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    if 'hidden_size' not in model_kwargs:
        assert 'nli_hidden_layer_size' in model_kwargs
        model_kwargs['hidden_size'] = model_kwargs['nli_hidden_layer_size'] # for compatibility with other models
    model = BertBasedNLI(**model_kwargs)
    model = model.to(device)

    # Load model weights
    print_bold('Load model weights')
    model_checkpoint_path = get_checkpoint_filepath(args.nli_checkpoint_folder_path)
    print('model_checkpoint_path = ', model_checkpoint_path)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_model_state_dict(model, checkpoint['model'])

    # Create dataset and dataloader
    dataset = NLIEmbeddingsDataset(P_embeddings, H_embeddings, P_index, H_index)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.nli_batch_size,
        shuffle=False,
        num_workers=args.nli_num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    # Run inference
    offset = 0
    NLI_softmaxes = np.zeros((len(H_index), 3), dtype=np.float16)
    NLI_classes = np.zeros((len(H_index),), dtype=np.int8) # 0: entailment, 1: neutral, 2: contradiction
    with torch.set_grad_enabled(False):
        model.train(False)
        for batch in tqdm(dataloader, total=len(dataloader), mininterval=2):
            P = batch['P'].to(device)
            H = batch['H'].to(device)
            logits = model.forward_with_precomputed_embeddings(p_vectors=P, h_vectors=H)
            softmax = torch.softmax(logits, dim=1)
            NLI_softmaxes[offset:offset+len(P)] = softmax.cpu().numpy()
            NLI_classes[offset:offset+len(P)] = torch.argmax(softmax, dim=1).cpu().numpy().astype(np.int8)
            offset += len(P)
    assert offset == len(H_index)
    return NLI_softmaxes, NLI_classes

def _compute_NLI_softmaxes_and_classes__all_pairs(args, embeddings):

    n = len(embeddings)
    P_index = np.empty((n ** 2,), dtype=np.int32)
    H_index = np.empty((n ** 2,), dtype=np.int32)
    offset = 0
    for i in range(n):
        for j in range(n):
            P_index[offset] = i
            H_index[offset] = j
            offset += 1
    assert offset == len(H_index)

    return _compute_NLI_softmaxes_and_classes(args, embeddings, embeddings, P_index, H_index)

_shared_original_facts = None
_shared_reports = None
_shared_fact2idx = None
_shared_index_range_size = None
_shared_NLI_classes = None
_shared_similarities = None
_shared_H_index = None

def _find_positive_negative_neutral_representative_facts_for_report(r_idx, finding_only, min_pos_sim, max_neg_sim):
    r = _shared_reports[r_idx]
    if finding_only:
        r_fact_idxs = r['findings_fact_idxs']
    else:
        r_fact_idxs = r['findings_fact_idxs'] + r['impression_fact_idxs']
    r_fact_idxs = list(set(r_fact_idxs)) # remove duplicates
    r_fact_idxs = [_shared_fact2idx[_shared_original_facts[i]] for i in r_fact_idxs] # use updated indices
    
    # Find positive and negative representative facts
    pos_representative_fact_idxs = set()
    neg_representative_fact_idxs = set()
    neu_representative_fact_idxs = set()
    for f_idx in r_fact_idxs:
        offset = f_idx * _shared_index_range_size
        for i in range(_shared_index_range_size):
            c = _shared_NLI_classes[offset + i]
            sim = _shared_similarities[offset + i]
            if c == 0 and sim >= min_pos_sim: # entailment
                pos_representative_fact_idxs.add(_shared_H_index[offset + i])
            elif c == 2 and sim <= max_neg_sim: # contradiction
                neg_representative_fact_idxs.add(_shared_H_index[offset + i])
            elif c == 1 and max_neg_sim < sim < min_pos_sim: # neutral
                neu_representative_fact_idxs.add(_shared_H_index[offset + i])

    # remove positives and negatives from neutrals
    neu_representative_fact_idxs -= pos_representative_fact_idxs
    neu_representative_fact_idxs -= neg_representative_fact_idxs

    # remove intersection between positives and negatives from both, because the relation of these facts to the report is ambiguous
    intersection = pos_representative_fact_idxs & neg_representative_fact_idxs
    pos_representative_fact_idxs -= intersection
    neg_representative_fact_idxs -= intersection

    # convert to lists
    pos_representative_fact_idxs = list(pos_representative_fact_idxs)
    neg_representative_fact_idxs = list(neg_representative_fact_idxs)
    neu_representative_fact_idxs = list(neu_representative_fact_idxs)
    
    return pos_representative_fact_idxs, neg_representative_fact_idxs, neu_representative_fact_idxs

def algorithm2__pos_and_neg_representative_facts_per_report_based_on_NLI(
        args, integrated_embeddings_and_report_annotations, facts, fact_embeddings, fact2idx,
        representative_facts, representative_fact_embeddings):
    
    global _shared_nearest_idxs
    global _shared_original_facts
    global _shared_reports
    global _shared_fact2idx
    global _shared_index_range_size
    global _shared_NLI_classes
    global _shared_similarities
    global _shared_H_index
    
    if args.prefilter_based_on_similarity:
        # 1) Find positive, middle, and negative representative facts for each fact
        pos_mid_neg_representative_facts_filepath = get_file_path_with_hashing_if_too_long(
            folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
            prefix='pos_mid_neg_representative_facts',
            strings=[
                'algorithm2',
                args.integrated_embeddings_and_report_annotations_filepath,
                f'num_kmeans_clusters={args.num_kmeans_clusters}',
                f'num_kmeans_iterations={args.num_kmeans_iterations}',
                f'num_kmedoids_clusters={args.num_kmedoids_clusters}',
                f'num_kmedoids_iterations={args.num_kmedoids_iterations}',
                f'kmedoids_method={args.kmedoids_method}',
                f'nearest_k={args.nearest_k}',
                f'fact_selection_method={args.fact_selection_method}',
                f'num_pos={args.num_pos}',
                f'num_neg={args.num_neg}',
                f'num_middle={args.num_middle}',
                f'finding_only={args.finding_only}',
                f'union_find_threshold={args.union_find_threshold}',
            ],
            force_hashing=True,
        )
        if os.path.exists(pos_mid_neg_representative_facts_filepath):
            # Load positive, middle, and negative representative facts
            print_bold('Loading positive, middle, and negative representative facts from:', pos_mid_neg_representative_facts_filepath)
            pos_mid_neg_representative_facts = load_pickle(pos_mid_neg_representative_facts_filepath)
            most_similar = pos_mid_neg_representative_facts['most_similar']
            least_similar = pos_mid_neg_representative_facts['least_similar']
            middle_similar = pos_mid_neg_representative_facts['middle_similar']
            most_similar_similarities = pos_mid_neg_representative_facts['most_similar_similarities']
            least_similar_similarities = pos_mid_neg_representative_facts['least_similar_similarities']
            middle_similar_similarities = pos_mid_neg_representative_facts['middle_similar_similarities']
            print('len(most_similar):', len(most_similar))
            print('len(least_similar):', len(least_similar))
            print('len(middle_similar):', len(middle_similar))
            # Print a few examples
            for i in range(len(most_similar)):
                if len(most_similar[i]) == 0:
                    continue
                print(f'facts[{i}]: {facts[i]}')
                print()
                print(f'most_similar[{i}]:')
                for j in range(min(len(most_similar[i]), 5)):
                    print(f'  {representative_facts[most_similar[i][j]]}')
                    print(f'  {most_similar_similarities[i][j]}')
                print()
                print(f'least_similar[{i}]:')
                for j in range(min(len(least_similar[i]), 5)):
                    print(f'  {representative_facts[least_similar[i][j]]}')
                    print(f'  {least_similar_similarities[i][j]}')
                print()
                print(f'middle_similar[{i}]:')
                for j in range(min(len(middle_similar[i]), 5)):
                    print(f'  {representative_facts[middle_similar[i][j]]}')
                    print(f'  {middle_similar_similarities[i][j]}')
                break
        else:
            print_blue('Finding positive, middle, and negative representative facts for each fact...')
            assert args.num_pos + args.num_neg + args.num_middle < len(representative_facts)
            most_similar = [[] for _ in range(len(fact_embeddings))]
            most_similar_similarities = [[] for _ in range(len(fact_embeddings))]
            least_similar = [[] for _ in range(len(fact_embeddings))]
            least_similar_similarities = [[] for _ in range(len(fact_embeddings))]
            middle_similar = [[] for _ in range(len(fact_embeddings))]
            middle_similar_similarities = [[] for _ in range(len(fact_embeddings))]
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
                _shared_nearest_idxs = nearest_idxs
                _shared_similarities = similarities
                # Find negative representative facts
                tasks = [(i, args.num_neg) for i in range(len(nearest_idxs))]
                with mp.Pool(processes=args.num_processes) as pool:
                    results = pool.starmap(_find_least_similar_facts_no_threshold, tasks)
                for i, result in enumerate(results):
                    idx = offset + i
                    least_similar[idx] = result[0]
                    least_similar_similarities[idx] = result[1]
                # Find positive representative facts
                tasks = [(i, args.num_pos) for i in range(len(nearest_idxs))]
                with mp.Pool(processes=args.num_processes) as pool:
                    results = pool.starmap(_find_most_similar_facts_no_threshold, tasks)
                for i, result in enumerate(results):
                    idx = offset + i
                    most_similar[idx] = result[0]
                    most_similar_similarities[idx] = result[1]
                # Find middle representative facts
                tasks = [(i, args.num_pos, args.num_middle, args.num_neg) for i in range(len(nearest_idxs))]
                with mp.Pool(processes=args.num_processes) as pool:
                    results = pool.starmap(_find_middle_similar_facts, tasks)
                for i, result in enumerate(results):
                    idx = offset + i
                    middle_similar[idx] = result[0]
                    middle_similar_similarities[idx] = result[1]
                
                if offset == 0: # sanity check
                    for i in range(len(batch)):
                        if len(most_similar[i]) == 0:
                            continue
                        print(f'facts[{i}]: {facts[i]}')
                        print()
                        print(f'most_similar[{i}]:')
                        for j in range(min(len(most_similar[i]), 5)):
                            print(f'  {representative_facts[most_similar[i][j]]}')
                            print(f'  {most_similar_similarities[i][j]}')
                        print()
                        print(f'least_similar[{i}]:')
                        for j in range(min(len(least_similar[i]), 5)):
                            print(f'  {representative_facts[least_similar[i][j]]}')
                            print(f'  {least_similar_similarities[i][j]}')
                        print()
                        print(f'middle_similar[{i}]:')
                        for j in range(min(len(middle_similar[i]), 5)):
                            print(f'  {representative_facts[middle_similar[i][j]]}')
                            print(f'  {middle_similar_similarities[i][j]}')
                        break # only print one

                offset += len(batch)
            print('len(most_similar):', len(most_similar))
            print('len(least_similar):', len(least_similar))
            print('len(middle_similar):', len(middle_similar))
            print('Saving positive, middle, and negative representative facts to:', pos_mid_neg_representative_facts_filepath)
            save_pickle({
                'most_similar': most_similar,
                'least_similar': least_similar,
                'middle_similar': middle_similar,
                'most_similar_similarities': most_similar_similarities,
                'least_similar_similarities': least_similar_similarities,
                'middle_similar_similarities': middle_similar_similarities,
            }, pos_mid_neg_representative_facts_filepath)
        
    else:
        # 1) Precompute similarities between facts and representative facts

        similarities_filepath = get_file_path_with_hashing_if_too_long(
            folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
            prefix='fact_repr_fact_similarities',
            strings=[
                'algorithm2',
                args.integrated_embeddings_and_report_annotations_filepath,
                f'num_kmeans_clusters={args.num_kmeans_clusters}',
                f'num_kmeans_iterations={args.num_kmeans_iterations}',
                f'num_kmedoids_clusters={args.num_kmedoids_clusters}',
                f'num_kmedoids_iterations={args.num_kmedoids_iterations}',
                f'kmedoids_method={args.kmedoids_method}',
                f'nearest_k={args.nearest_k}',
                f'fact_selection_method={args.fact_selection_method}',
                f'finding_only={args.finding_only}',
                f'union_find_threshold={args.union_find_threshold}',
            ],
            force_hashing=True,
        )
        if os.path.exists(similarities_filepath):
            # Load similarities
            print_bold('Loading similarities from:', similarities_filepath)
            similarity_matrix = load_pickle(similarities_filepath)
            print('similarity_matrix.shape:', similarity_matrix.shape)
        else:
            print_blue('Precomputing similarities between facts and representative facts...')
            similarity_matrix = np.zeros((len(fact_embeddings), len(representative_fact_embeddings)), dtype=np.float32)
            # Create dataset and dataloader
            dataset = FactEmbeddingsDataset(fact_embeddings)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=dataset.collate_fn)
            r_f_embs = torch.from_numpy(representative_fact_embeddings).cuda()
            offset = 0
            for batch in tqdm(dataloader, mininterval=2):
                batch_embs = batch.cuda()
                similarities = torch.matmul(batch_embs, r_f_embs.T)
                similarities = similarities.cpu().numpy()
                similarity_matrix[offset:offset+len(batch)] = similarities
                offset += len(batch)
            print('Saving similarities to:', similarities_filepath)
            save_pickle(similarity_matrix, similarities_filepath)
    
    # assert False

    # 2) For each pair of facts and representative facts, compute NLI classes
    nli_classes_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
        prefix='nli_classes',
        strings=[
            'algorithm2',
            args.integrated_embeddings_and_report_annotations_filepath,
            f'num_kmeans_clusters={args.num_kmeans_clusters}',
            f'num_kmeans_iterations={args.num_kmeans_iterations}',
            f'num_kmedoids_clusters={args.num_kmedoids_clusters}',
            f'num_kmedoids_iterations={args.num_kmedoids_iterations}',
            f'kmedoids_method={args.kmedoids_method}',
            f'nearest_k={args.nearest_k}',
            f'fact_selection_method={args.fact_selection_method}',
            f'num_pos={args.num_pos}',
            f'num_neg={args.num_neg}',
            f'num_middle={args.num_middle}',
            f'finding_only={args.finding_only}',
            f'union_find_threshold={args.union_find_threshold}',
            f'nli_huggingface_model_name={args.nli_huggingface_model_name}',
            f'nli_model_checkpoint_folder_path={args.nli_checkpoint_folder_path}',
            f'prefilter_based_on_similarity={args.prefilter_based_on_similarity}',
        ],
        force_hashing=True,
    )
    if os.path.exists(nli_classes_filepath):
        # Load NLI classes
        print_bold('Loading NLI classes from:', nli_classes_filepath)
        tmp = load_pickle(nli_classes_filepath)
        P_index = tmp['P_index']
        H_index = tmp['H_index']
        NLI_softmaxes = tmp['NLI_softmaxes']
        NLI_classes = tmp['NLI_classes']
        sim_array = tmp['similarities']
        print(f'NLI_softmaxes.shape: {NLI_softmaxes.shape}')
        print(f'NLI_classes.shape: {NLI_classes.shape}')
        
        ctee = CachedTextEmbeddingExtractor(
            model_name=args.nli_huggingface_model_name,
            device=args.device,
            model_checkpoint_folder_path=args.nli_checkpoint_folder_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        nli_fact_embeddings = ctee.compute_text_embeddings(facts)
        nli_representative_fact_embeddings = ctee.compute_text_embeddings(representative_facts)
    else:
        # Precompute NLI embeddings
        print_blue('Precomputing NLI embeddings...', bold=True)
        ctee = CachedTextEmbeddingExtractor(
            model_name=args.nli_huggingface_model_name,
            device=args.device,
            model_checkpoint_folder_path=args.nli_checkpoint_folder_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        nli_fact_embeddings = ctee.compute_text_embeddings(facts)
        nli_representative_fact_embeddings = ctee.compute_text_embeddings(representative_facts)
        print('nli_fact_embeddings.shape:', nli_fact_embeddings.shape)
        print('nli_representative_fact_embeddings.shape:', nli_representative_fact_embeddings.shape)
        print('Examples:')
        print('  facts[0]:', facts[0])
        print('  nli_fact_embeddings[0][0:10]:', nli_fact_embeddings[0][0:10])
        print('  representative_facts[0]:', representative_facts[0])
        print('  nli_representative_fact_embeddings[0][0:10]:', nli_representative_fact_embeddings[0][0:10])
        
        # Compute NLI classes
        print_blue('Computing NLI classes...', bold=True)
        if args.prefilter_based_on_similarity:
            n = (args.num_pos + args.num_neg + args.num_middle) * len(fact_embeddings)
        else:
            n = len(representative_facts) * len(fact_embeddings)
        P_index = np.zeros((n,), dtype=np.int32) # index of premise
        H_index = np.zeros((n,), dtype=np.int32) # index of hypothesis
        sim_array = np.zeros((n,), dtype=np.float32) # similarity between premise and hypothesis
        offset = 0
        if args.prefilter_based_on_similarity:
            for i in tqdm(range(len(fact_embeddings)), mininterval=2):
                for j in range(args.num_pos):
                    P_index[offset] = i
                    H_index[offset] = most_similar[i][j]
                    sim_array[offset] = most_similar_similarities[i][j]
                    offset += 1
                for j in range(args.num_middle):
                    P_index[offset] = i
                    H_index[offset] = middle_similar[i][j]
                    sim_array[offset] = middle_similar_similarities[i][j]
                    offset += 1
                for j in range(args.num_neg):
                    P_index[offset] = i
                    H_index[offset] = least_similar[i][j]
                    sim_array[offset] = least_similar_similarities[i][j]
                    offset += 1
                # if i == 30: break # DEBUG
        else:
            for i in tqdm(range(len(fact_embeddings)), mininterval=2):
                for j in range(len(representative_fact_embeddings)):
                    P_index[offset] = i
                    H_index[offset] = j
                    sim_array[offset] = similarity_matrix[i, j]
                    offset += 1
        assert offset == n # sanity check
        NLI_softmaxes, NLI_classes = _compute_NLI_softmaxes_and_classes(args,
                                                                        P_embeddings=nli_fact_embeddings,
                                                                        H_embeddings=nli_representative_fact_embeddings,
                                                                        P_index=P_index, H_index=H_index)
        print_bold('Saving NLI classes to:', nli_classes_filepath)
        save_pickle({
            'P_index': P_index,
            'H_index': H_index,
            'NLI_softmaxes': NLI_softmaxes,
            'NLI_classes': NLI_classes,
            'similarities': sim_array,
        }, nli_classes_filepath)

    print('NLI examples:')
    # for i in random.sample(range(len(NLI_classes)), 10):
    for i in range(2):
        print(f'  ------------------ {i} ------------------')
        print(f'  P_index[{i}]: {P_index[i]}')
        print(f'  H_index[{i}]: {H_index[i]}')
        print(f'  facts[{P_index[i]}]: {facts[P_index[i]]}')
        print(f'  representative_facts[{H_index[i]}]: {representative_facts[H_index[i]]}')
        print(f'  nli_fact_embeddings[{P_index[i]}][0]: {nli_fact_embeddings[P_index[i]][0]}')
        print(f'  nli_representative_fact_embeddings[{H_index[i]}][0]: {nli_representative_fact_embeddings[H_index[i]][0]}')
        print(f'  NLI_classes[{i}]: {NLI_classes[i]}')
        for j in range(3):
            print(f'    NLI_softmaxes[{i}][{j}]: {NLI_softmaxes[i][j]:.4f}')
        print(f'  sim_array[{i}]: {sim_array[i]}')

    for i in range(2):
        i = len(NLI_classes) - 1 - i
        print(f'  ------------------ {i} ------------------')
        print(f'  P_index[{i}]: {P_index[i]}')
        print(f'  H_index[{i}]: {H_index[i]}')
        print(f'  facts[{P_index[i]}]: {facts[P_index[i]]}')
        print(f'  representative_facts[{H_index[i]}]: {representative_facts[H_index[i]]}')
        print(f'  nli_fact_embeddings[{P_index[i]}][0]: {nli_fact_embeddings[P_index[i]][0]}')
        print(f'  nli_representative_fact_embeddings[{H_index[i]}][0]: {nli_representative_fact_embeddings[H_index[i]][0]}')
        print(f'  NLI_classes[{i}]: {NLI_classes[i]}')
        for j in range(3):
            print(f'    NLI_softmaxes[{i}][{j}]: {NLI_softmaxes[i][j]:.4f}')
        print(f'  sim_array[{i}]: {sim_array[i]}')

    if NLI_classes.dtype != np.int8:
        print('Converting NLI_classes to np.int8')
        NLI_classes = NLI_classes.astype(np.int8)
    if NLI_softmaxes.dtype != np.float16:
        print('Converting NLI_softmaxes to np.float16')
        NLI_softmaxes = NLI_softmaxes.astype(np.float16)
    if sim_array.dtype != np.float16:
        print('Converting sim_array to np.float16')
        sim_array = sim_array.astype(np.float16)
    
    # 3) Compute similarity matrix between all pairs of representative facts
    print_blue('Computing similarity matrix between all pairs of representative facts...')
    similarity_matrix_repr = np.dot(representative_fact_embeddings, representative_fact_embeddings.T).astype(np.float16)
    print('similarity_matrix_repr.shape:', similarity_matrix_repr.shape)

    # 4) Compute NLI softmaxes and classes for all pairs of facts and representative facts
    print_blue('Computing NLI softmaxes and classes for all pairs of representative facts...')
    NLI_softmaxes_repr, NLI_classes_repr = _compute_NLI_softmaxes_and_classes__all_pairs(args, nli_representative_fact_embeddings)
    NLI_softmaxes_repr = NLI_softmaxes_repr.reshape((len(representative_facts), len(representative_facts), 3))
    NLI_classes_repr = NLI_classes_repr.reshape((len(representative_facts), len(representative_facts)))
    print('NLI_softmaxes_repr.shape:', NLI_softmaxes_repr.shape)
    print('NLI_classes_repr.shape:', NLI_classes_repr.shape)

    index_range_size = (args.num_pos + args.num_neg + args.num_middle)\
         if args.prefilter_based_on_similarity else len(representative_facts)
    print('index_range_size:', index_range_size)

    # 5) Find positive, negative, and neutral representative facts for each report
    print_blue('Finding positive, negative, and neutral representative facts for each report...')

    output = {
        'reports': [],
        'facts': facts,
        'fact_embeddings': fact_embeddings.astype(np.float16), # convert to np.float16 to save space
        'representative_facts': representative_facts,
        'representative_fact_embeddings': representative_fact_embeddings.astype(np.float16), # convert to np.float16 to save space
        'NLI_all_facts': {
            'fact_index': P_index, # index of premise (fact index)
            'rep_fact_index': H_index, # index of hypothesis (representative fact index)
            'nli_softmax': NLI_softmaxes, # softmax of NLI classes
            'nli_class': NLI_classes, # 0: entailment, 1: neutral, 2: contradiction
            'embed_sim': sim_array, # similarity between premise and hypothesis
        },
        'NLI_representative_facts': {
            'NLI_softmaxes': NLI_softmaxes_repr,
            'NLI_classes': NLI_classes_repr,
            'embed_sim': similarity_matrix_repr,
        }
    }

    # Parallelize
    _shared_original_facts = integrated_embeddings_and_report_annotations['facts']
    _shared_reports = integrated_embeddings_and_report_annotations['reports']
    _shared_fact2idx = fact2idx
    _shared_index_range_size = index_range_size
    _shared_NLI_classes = NLI_classes
    _shared_similarities = sim_array
    _shared_H_index = H_index

    tasks = [(i, args.finding_only, args.pos_sim_threshold, args.neg_sim_threshold) for i in range(len(_shared_reports))]
    print('len(tasks):', len(tasks))
    print('args.num_processes:', args.num_processes)
    start_time = time.time()
    with mp.Pool(processes=args.num_processes) as pool:
        results = pool.starmap(_find_positive_negative_neutral_representative_facts_for_report, tasks)
    print('Time taken:', time.time() - start_time)
    
    # Add to output
    for i, (pos_repr_fact_idxs, neg_repr_fact_idxs, neu_repr_fact_idxs) in tqdm(enumerate(results), mininterval=2):
        output['reports'].append({
            'path': _shared_reports[i]['path'],
            'part_id': _shared_reports[i]['part_id'],
            'subject_id': _shared_reports[i]['subject_id'],
            'study_id': _shared_reports[i]['study_id'],
            'pos_repr_facts': pos_repr_fact_idxs,
            'neg_repr_facts': neg_repr_fact_idxs,
            'neu_repr_facts': neu_repr_fact_idxs,
        })

    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrated_embeddings_and_report_annotations_filepath', type=str, required=True)
    parser.add_argument('--finding_only', action='store_true')
    parser.add_argument('--algorithm', type=str, default='1', choices=['1', '2'])
    # clustering parameters
    parser.add_argument('--num_kmeans_clusters', type=int, default=500)
    parser.add_argument('--num_kmeans_iterations', type=int, default=300)
    parser.add_argument('--num_kmedoids_clusters', type=int, default=25000)
    parser.add_argument("--num_kmedoids_iterations", type=int, default=300)
    parser.add_argument("--kmedoids_method", type=str, default="alternate", choices=["pam", "alternate"])
    # nearest neighbors parameters
    parser.add_argument("--nearest_k", type=int, default=10)
    parser.add_argument("--fact_selection_method", type=str, default="max_avg_word_count", choices=["shortest", "longest", "max_avg_word_count"])
    parser.add_argument("--pos_sim_threshold", type=float, default=0.9)
    parser.add_argument("--neg_sim_threshold", type=float, default=0.3)
    parser.add_argument("--union_find_threshold", type=float, default=0.97)
    parser.add_argument("--max_num_pos", type=int, default=100)
    parser.add_argument("--max_num_neg", type=int, default=100)
    parser.add_argument("--prefilter_based_on_similarity", action="store_true")
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--num_pos", type=int, default=100)
    parser.add_argument("--num_middle", type=int, default=100)
    # NLI parameters
    parser.add_argument("--nli_huggingface_model_name", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--nli_batch_size", type=int, default=32)
    parser.add_argument("--nli_num_workers", type=int, default=4)
    parser.add_argument('--nli_checkpoint_folder_path', type=str, default=None)
    # misc
    parser.add_argument("--device", type=str, default="cuda")
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

    # Find representative facts
    representative_facts, representative_fact_embeddings = find_representative_facts(args, facts, fact_embeddings)

    # Find positive and negative representative facts for each fact
    if args.algorithm == '1':
        output = algorithm1__pos_and_neg_representative_facts_per_report_based_on_similarity(
            args, integrated_embeddings_and_report_annotations, facts, fact_embeddings, fact2idx,
            representative_facts, representative_fact_embeddings)
    elif args.algorithm == '2':
        output = algorithm2__pos_and_neg_representative_facts_per_report_based_on_NLI(
            args, integrated_embeddings_and_report_annotations, facts, fact_embeddings, fact2idx,
            representative_facts, representative_fact_embeddings)
    else:
        raise Exception(f'Unknown algorithm: {args.algorithm}')
    
    # Add command line args to output
    output['command_line_args'] = vars(args)

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
            f'nearest_k={args.nearest_k}',
            f'pos_sim_threshold={args.pos_sim_threshold}',
            f'neg_sim_threshold={args.neg_sim_threshold}',
            f'max_num_pos={args.max_num_pos}',
            f'max_num_neg={args.max_num_neg}',
            f'num_neg={args.num_neg}',
            f'num_pos={args.num_pos}',
            f'finding_only={args.finding_only}'
            f'union_find_threshold={args.union_find_threshold}',
            f'algorithm={args.algorithm}',
            f'prefilter_based_on_similarity={args.prefilter_based_on_similarity}',
            f'nli_huggingface_model_name={args.nli_huggingface_model_name}',
            f'nli_model_checkpoint_folder_path={args.nli_checkpoint_folder_path}',
        ],
        force_hashing=True,
    )
    print_bold('Saving output to:', output_filepath)
    save_pickle(output, output_filepath)
    print('Done!')

if __name__ == '__main__':
    main()