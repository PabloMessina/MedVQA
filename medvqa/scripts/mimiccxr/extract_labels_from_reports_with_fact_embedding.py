import os
import math
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from nltk.tokenize import sent_tokenize, word_tokenize
from medvqa.models.huggingface_utils import (
    compute_text_embeddings_with_BiomedVLP_BioVilT,
    compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized,
)
from medvqa.utils.files import (
    get_cached_jsonl_file,
    get_cached_json_file,
    get_file_path_with_hashing_if_too_long,
    load_jsonl,
    save_pickle,
    get_cached_pickle_file,
)
from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.utils.logging import print_blue, print_bold, print_magenta, print_orange

class SentenceRarenessScorer:
    
    def __init__(self, background_findings_and_impression_per_report_filepath):
        self._background_findings_and_impression_per_report_filepath = background_findings_and_impression_per_report_filepath
        self._w2c = None

    def _compute_word2count(self):
        print('Computing word2count...')
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=MIMICCXR_FAST_CACHE_DIR,
            prefix='word2count',
            strings=[
                self._background_findings_and_impression_per_report_filepath,
            ],
            force_hashing=True,
        )
        if os.path.exists(save_path):
            print('word2count found at:', save_path)
            return get_cached_pickle_file(save_path)
        word2count = {}
        reports = get_cached_json_file(self._background_findings_and_impression_per_report_filepath)
        for row in tqdm(reports, mininterval=2):
            findings = row['findings']
            impression = row['impression']
            for text in (findings, impression):
                if not text:
                    continue
                for s in sent_tokenize(text):
                    for w in word_tokenize(s):
                        word2count[w] = word2count.get(w, 0) + 1
        print('len(word2count):', len(word2count))
        print('Saving word2count to:', save_path)
        save_pickle(word2count, save_path, add_to_cache=True)
        return word2count

    def __call__(self, sentence):
        if self._w2c is None:
            self._w2c = self._compute_word2count()
        score = 0
        for word in word_tokenize(sentence):
            score += 1 / self._w2c.get(word, 1e-6) # Add 1e-6 to avoid division by zero
        return score

def _compute_kmeans_and_kmedoids_clustering(alias, sentence_embeddings_filepath, sentence_idxs,
                                            num_kmeans_clusters, num_kmeans_iterations,
                                            num_kmedoids_clusters, num_kmedoids_iterations, kmedoids_method,
                                            sentence_rareness_scorer, k_nearest_sentences):
    
    assert num_kmedoids_clusters > num_kmeans_clusters

    # Step 1: compute kmeans clustering
    kmeans_clustering_save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_FAST_CACHE_DIR,
        prefix=f'kmeans_clustering_{alias}',
        strings=[
            sentence_embeddings_filepath,
            f'num_kmeans_clusters={num_kmeans_clusters}',
            f'num_kmeans_iterations={num_kmeans_iterations}',
            f'len(sentence_idxs)={len(sentence_idxs)}',
        ],
    )
    if os.path.exists(kmeans_clustering_save_path):
        print('Kmeans clustering found at:', kmeans_clustering_save_path)
    else:
        sentence_embeddings = get_cached_pickle_file(sentence_embeddings_filepath)
        embeddings = sentence_embeddings['embeddings'][sentence_idxs] # (num_sentences, embedding_dim)

        print(f"Clustering sentences with KMeans (n_clusters={num_kmeans_clusters}, n_init='auto', verbose=2, max_iter={num_kmeans_iterations})")        
        kmeans = KMeans(n_clusters=num_kmeans_clusters, random_state=0, n_init='auto', verbose=2,
                        max_iter=num_kmeans_iterations).fit(embeddings)

        # Save clustering
        kmeans_clustering = {
            'sentence_idxs': sentence_idxs,
            'labels': kmeans.labels_,
            'cluster_centers': kmeans.cluster_centers_,
        }
        print(f"Saving KMeans clustering to {kmeans_clustering_save_path}")
        save_pickle(kmeans_clustering, kmeans_clustering_save_path, add_to_cache=True)

    # Step 2: compute kmedoids clustering for each kmeans cluster
    kmedoids_clustering_save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_FAST_CACHE_DIR,
        prefix=f'kmedoids_clustering_{alias}',
        strings=[
            sentence_embeddings_filepath,
            f'num_kmeans_clusters={num_kmeans_clusters}',
            f'num_kmeans_iterations={num_kmeans_iterations}',
            f'num_kmedoids_clusters={num_kmedoids_clusters}',
            f'num_kmedoids_iterations={num_kmedoids_iterations}',
            f'kmedoids_method={kmedoids_method}',
            f'len(sentence_idxs)={len(sentence_idxs)}',
        ],
    )
    if os.path.exists(kmedoids_clustering_save_path):
        print('Kmedoids clustering found at:', kmedoids_clustering_save_path)
    else:
        sentence_embeddings = get_cached_pickle_file(sentence_embeddings_filepath)
        # Load kmeans clustering
        print(f"Loading KMeans clustering from {kmeans_clustering_save_path}")
        kmeans_clustering = get_cached_pickle_file(kmeans_clustering_save_path)
        cluster_ids = kmeans_clustering['labels']
        assert len(sentence_idxs) == len(cluster_ids)
        cid2sidxs = {cid:[] for cid in range(num_kmeans_clusters)}
        for sidx, cid in zip(sentence_idxs, cluster_ids):
            cid2sidxs[cid].append(sidx)
        kmedoids_clustering = { 'subclusters': [] }
        for sidxs in cid2sidxs.values():
            cluster_embeddings = sentence_embeddings['embeddings'][sidxs] # (num_sentences, embedding_dim)
            n_clusters = math.ceil(num_kmedoids_clusters * len(sidxs) / len(sentence_idxs)) # Scale num_kmedoids_clusters based on the number of sentences in the sub-cluster
            print(f"Clustering {len(sidxs)} sentences with KMedoids (n_clusters={n_clusters},"
                  f" metric='cosine', method={kmedoids_method}, max_iter={num_kmedoids_iterations})")
            kmedoids = KMedoids(n_clusters=n_clusters, metric='cosine', method=kmedoids_method,
                                max_iter=num_kmedoids_iterations).fit(cluster_embeddings)
            kmedoids_clustering['subclusters'].append({
                'sentence_idxs': sidxs,
                'labels': kmedoids.labels_,
                'cluster_centers': kmedoids.cluster_centers_,
            })
        print(f"Saving KMedoids clustering to {kmedoids_clustering_save_path}")
        save_pickle(kmedoids_clustering, kmedoids_clustering_save_path, add_to_cache=True)

    # Step 3: for each kmedoids cluster center, find the K nearest sentences and replace the cluster center
    # with the embedding of the sentence with the minimum "rareness" score
    kmedoids_refinement_save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_FAST_CACHE_DIR,
        prefix=f'kmedoids_refinement_{alias}',
        strings=[
            sentence_embeddings_filepath,
            sentence_rareness_scorer._background_findings_and_impression_per_report_filepath,
            f'num_kmeans_clusters={num_kmeans_clusters}',
            f'num_kmeans_iterations={num_kmeans_iterations}',
            f'num_kmedoids_clusters={num_kmedoids_clusters}',
            f'num_kmedoids_iterations={num_kmedoids_iterations}',
            f'kmedoids_method={kmedoids_method}',
            f'k_nearest_sentences={k_nearest_sentences}', # Number of nearest sentences to consider for each cluster center
            f'len(sentence_idxs)={len(sentence_idxs)}',
        ],
    )
    if os.path.exists(kmedoids_refinement_save_path):
        print('Kmedoids refinement found at:', kmedoids_refinement_save_path)
    else:
        # Load sentence embeddings
        sentence_embeddings = get_cached_pickle_file(sentence_embeddings_filepath)
        # Load kmedoids clustering
        print(f"Loading KMedoids clustering from {kmedoids_clustering_save_path}")
        kmedoids_clustering = get_cached_pickle_file(kmedoids_clustering_save_path)
        # Find the K nearest sentences for each cluster center
        print(f'Finding the least rare sentence among the {k_nearest_sentences} nearest sentences for each cluster center...')
        refined_cluster_centers = []
        refined_cluster_center_sentence_idxs = []
        refined_cluster_center_sentences = []
        for subcluster in kmedoids_clustering['subclusters']:
            subcluster_sidxs = subcluster['sentence_idxs']
            subcluster_embeddings = sentence_embeddings['embeddings'][subcluster_sidxs]
            for cluster_center in subcluster['cluster_centers']:
                # Find the K nearest sentences
                similarities = np.dot(subcluster_embeddings, cluster_center)
                nearest_sentence_idxs = np.argsort(similarities)[::-1][:k_nearest_sentences]
                nearest_sentence_idxs = [subcluster_sidxs[i] for i in nearest_sentence_idxs] # Convert to original sentence idxs
                nearest_sentences = [sentence_embeddings['sentences'][i] for i in nearest_sentence_idxs]
                s2i = {s:i for i, s in enumerate(nearest_sentences)}
                # Find the sentence with the minimum rareness score
                nearest_sentences.sort(key=sentence_rareness_scorer)
                least_rare_sentence = nearest_sentences[0]
                least_rare_sentence_idx = nearest_sentence_idxs[s2i[least_rare_sentence]]
                least_rare_sentence_embedding = sentence_embeddings['embeddings'][least_rare_sentence_idx]
                refined_cluster_centers.append(least_rare_sentence_embedding)
                refined_cluster_center_sentence_idxs.append(least_rare_sentence_idx)
                refined_cluster_center_sentences.append(least_rare_sentence)
        refined_cluster_centers = np.stack(refined_cluster_centers, axis=0)
        print('refined_cluster_centers.shape:', refined_cluster_centers.shape)
        refined_cluster_center_sentence_idxs = np.array(refined_cluster_center_sentence_idxs)
        # Remove duplicate sentences
        idxs = np.unique(refined_cluster_center_sentence_idxs, return_index=True)[1]
        if len(idxs) < len(refined_cluster_center_sentence_idxs):
            print_orange(f'WARNING: Removing {len(refined_cluster_center_sentence_idxs) - len(idxs)} duplicate sentences...', bold=True)
            refined_cluster_centers = refined_cluster_centers[idxs]
            refined_cluster_center_sentence_idxs = refined_cluster_center_sentence_idxs[idxs]
            refined_cluster_center_sentences = [refined_cluster_center_sentences[i] for i in idxs]
        # Save refined cluster centers
        print(f'Saving refined cluster centers to: {kmedoids_refinement_save_path}')
        save_pickle({
            'refined_cluster_centers': refined_cluster_centers,
            'refined_cluster_center_sentence_idxs': refined_cluster_center_sentence_idxs,
            'refined_cluster_center_sentences': refined_cluster_center_sentences,
        }, kmedoids_refinement_save_path, add_to_cache=True)

    return {
        'kmeans_clustering_filepath': kmeans_clustering_save_path,
        'kmedoids_clustering_filepath': kmedoids_clustering_save_path,
        'kmedoids_refinement_filepath': kmedoids_refinement_save_path,
    }

_shared_embeddings = None
_shared_cluster_centers = None
def _closest_cluster_center_to_sentence(i):
    return np.argmax(np.dot(_shared_embeddings[i], _shared_cluster_centers.T))
def _closest_sentence_to_cluster_center(i):
        return np.argmax(np.dot(_shared_embeddings, _shared_cluster_centers[i]))

def _assign_closest_cluster_center_to_each_sentence(alias, sentence_embeddings_filepath, sentence_idxs,
                                             refined_kmedoids_clustering_filepath, num_processes):
    
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_FAST_CACHE_DIR,
        prefix=f'closest_cluster_centers_{alias}',
        strings=[
            sentence_embeddings_filepath,
            refined_kmedoids_clustering_filepath,
            f'len(sentence_idxs)={len(sentence_idxs)}',
        ],
    )
    if os.path.exists(save_path): # Check if already computed
        print(f'Closest cluster centers found at: {save_path}')
        return save_path
    
    # Load sentence embeddings
    sentence_embeddings = get_cached_pickle_file(sentence_embeddings_filepath)
    embeddings = sentence_embeddings['embeddings'][sentence_idxs] # (num_sentences, embedding_dim)
    
    # Load refined kmedoids clustering
    print(f"Loading refined KMedoids clustering from {refined_kmedoids_clustering_filepath}")
    refined_kmedoids_clustering = get_cached_pickle_file(refined_kmedoids_clustering_filepath)
    cluster_centers = refined_kmedoids_clustering['refined_cluster_centers']
    print('cluster_centers.shape:', cluster_centers.shape)
    
    # Assign closest cluster to each sentence
    print(f'Assigning closest cluster to each sentence with {num_processes} processes...')
    global _shared_embeddings, _shared_cluster_centers
    _shared_embeddings = embeddings
    _shared_cluster_centers = cluster_centers    
    with mp.Pool(processes=num_processes) as pool:
        closest_cluster_centers = pool.map(_closest_cluster_center_to_sentence, range(len(embeddings)))
    closest_cluster_centers = np.array(closest_cluster_centers)
    print('closest_cluster_centers.shape:', closest_cluster_centers.shape)
    assert len(closest_cluster_centers) == len(embeddings)
    assert len(closest_cluster_centers) == len(sentence_idxs)

    # Save closest cluster centers
    print(f'Saving closest cluster centers to: {save_path}')
    save_pickle({
        'sentence_idxs': sentence_idxs,
        'closest_cluster_centers': closest_cluster_centers,
    }, save_path, add_to_cache=True)

    return save_path

def _assign_closest_sentence_to_each_cluster_center(alias, sentence_embeddings_filepath, sentence_idxs,
                                             kmedoids_clustering_filepath, num_processes):
    
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_FAST_CACHE_DIR,
        prefix=f'closest_sentence_to_cluster_center_{alias}',
        strings=[
            sentence_embeddings_filepath,
            kmedoids_clustering_filepath,
            f'len(sentence_idxs)={len(sentence_idxs)}',
        ],
    )
    if os.path.exists(save_path): # Check if already computed
        print(f'Closest sentence to cluster center found at: {save_path}')
        return save_path
    
    # Load sentence embeddings
    sentence_embeddings = get_cached_pickle_file(sentence_embeddings_filepath)
    embeddings = sentence_embeddings['embeddings'][sentence_idxs] # (num_sentences, embedding_dim)
    
    # Load kmedoids clustering
    print(f"Loading KMedoids clustering from {kmedoids_clustering_filepath}")
    kmedoids_clustering = get_cached_pickle_file(kmedoids_clustering_filepath)
    cluster_centers = []
    for subcluster in kmedoids_clustering['subclusters']:
        cluster_centers.extend(subcluster['cluster_centers'])
    cluster_centers = np.stack(cluster_centers, axis=0) # (num_clusters, embedding_dim)
    print('cluster_centers.shape:', cluster_centers.shape)
    
    # Assign closest sentence to each cluster center
    print(f'Assigning closest sentence to each cluster center with {num_processes} processes...')
    global _shared_embeddings, _shared_cluster_centers
    _shared_embeddings = embeddings
    _shared_cluster_centers = cluster_centers
    with mp.Pool(processes=num_processes) as pool:
        closest_sentence_idxs = pool.map(_closest_sentence_to_cluster_center, range(len(cluster_centers)))
    closest_sentence_idxs = [sentence_idxs[i] for i in closest_sentence_idxs] # Convert to original sentence idxs
    closest_sentence_idxs = np.array(closest_sentence_idxs)
    print('closest_sentence_idxs.shape:', closest_sentence_idxs.shape)
    assert len(closest_sentence_idxs) == len(cluster_centers)
    assert len(closest_sentence_idxs) < len(sentence_idxs)

    # Save closest sentence to cluster center
    print(f'Saving closest sentence to cluster center to: {save_path}')
    save_pickle({
        'closest_sentence_idxs': closest_sentence_idxs,
        'closest_sentences': [sentence_embeddings['sentences'][i] for i in closest_sentence_idxs],
    }, save_path, add_to_cache=True)

    return save_path

def _assign_cluster_based_labels_to_each_fact(
    sentence2index,
    integrated_fact_metadata_filepath,
    closest_cluster_center_to_sentence_obs_filepath,
    closest_cluster_center_to_sentence_anatloc_filepath,
    sentence_rareness_scorer,
):
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_FAST_CACHE_DIR,
        prefix='cluster_based_labels_per_fact',
        strings=[
            integrated_fact_metadata_filepath,
            closest_cluster_center_to_sentence_obs_filepath,
            closest_cluster_center_to_sentence_anatloc_filepath,
            sentence_rareness_scorer._background_findings_and_impression_per_report_filepath,
        ],
    )
    if os.path.exists(save_path):
        print('Shortest fact per (obs cluster, anatloc cluster) pair found at:', save_path)
    else:
        # Load integrated fact metadata
        print('Loading integrated fact metadata from:', integrated_fact_metadata_filepath)
        integrated_fact_metadata = get_cached_jsonl_file(integrated_fact_metadata_filepath)
        
        # Load closest cluster center to sentence fact
        print('Loading closest cluster centers to observation/fact sentences from:', closest_cluster_center_to_sentence_obs_filepath)
        closest_cluster_center_to_sentence_obs_fact = get_cached_pickle_file(closest_cluster_center_to_sentence_obs_filepath)
        idx2ccid_obs_fact = {sidx:ccid for sidx, ccid in zip(closest_cluster_center_to_sentence_obs_fact['sentence_idxs'],
                                                             closest_cluster_center_to_sentence_obs_fact['closest_cluster_centers'])}

        
        # Load closest cluster center to sentence anatloc
        print('Loading closest cluster center to sentence anatloc from:', closest_cluster_center_to_sentence_anatloc_filepath)
        closest_cluster_center_to_sentence_anatloc = get_cached_pickle_file(closest_cluster_center_to_sentence_anatloc_filepath)
        idx2ccid_anatloc = {sidx:ccid for sidx, ccid in zip(closest_cluster_center_to_sentence_anatloc['sentence_idxs'],
                                                            closest_cluster_center_to_sentence_anatloc['closest_cluster_centers'])}        
        
        fact2labels = {}
        obs_al2fact = {}        
        for row in integrated_fact_metadata:
            metadata = row['metadata']
            labels = {}
            # fact
            fact = row['fact']
            assert len(fact) > 0
            fact_idx = sentence2index[fact]
            fact_ccid = idx2ccid_obs_fact[fact_idx]
            labels['fact_ccid'] = fact_ccid
            fact2labels[fact] = labels
            # short observation
            short_obs = metadata['short observation']
            if short_obs:
                short_obs_idx = sentence2index[short_obs]
                short_obs_ccid = idx2ccid_obs_fact[short_obs_idx]
                labels['short_obs_ccid'] = short_obs_ccid
            # detailed observation
            detailed_obs = metadata['detailed observation']
            if detailed_obs:
                detailed_obs_idx = sentence2index[detailed_obs]
                detailed_obs_ccid = idx2ccid_obs_fact[detailed_obs_idx]
                labels['detailed_obs_ccid'] = detailed_obs_ccid
            # anatomical location
            anatloc = metadata['anatomical location']
            if anatloc:
                anatloc_idx = sentence2index[anatloc]
                anatloc_ccid = idx2ccid_anatloc[anatloc_idx]
                labels['anatloc_ccid'] = anatloc_ccid
            # fact + anatomical location
            if fact and anatloc:
                obs_al = (fact_ccid, anatloc_ccid)
                if obs_al not in obs_al2fact:
                    obs_al2fact[obs_al] = fact
                else:
                    other_fact = obs_al2fact[obs_al]
                    if sentence_rareness_scorer(fact) < sentence_rareness_scorer(other_fact):
                        obs_al2fact[obs_al] = fact # Keep the least rare fact
        print('len(fact2labels):', len(fact2labels))
        print('len(obs_al2fact):', len(obs_al2fact))
        print('Saving fact2labels and obs_al2fact to:', save_path)
        save_pickle({
            'fact2labels': fact2labels,
            'obs_al2fact': obs_al2fact,
        }, save_path, add_to_cache=True)

    return save_path

def _assign_cluster_based_labels_to_each_report(
    facts_and_cluster_based_labels_filepath,
    background_findings_and_impression_per_report_filepath,
    integrated_sentence_facts_filepath,
    refined_kmedoids_clustering_obs_filepath,
    max_num_labels=500, # Keep only the top "max_num_labels" most frequent labels
):
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_FAST_CACHE_DIR,
        prefix='cluster_based_labels_per_report',
        strings=[
            facts_and_cluster_based_labels_filepath,
            background_findings_and_impression_per_report_filepath,
            integrated_sentence_facts_filepath,
            refined_kmedoids_clustering_obs_filepath,
            f'max_num_labels={max_num_labels}',
        ],
    )
    if os.path.exists(save_path):
        print('Cluster-based labels per report found at:', save_path)
    else:
        # Load facts and cluster-based labels
        print('Loading facts and cluster-based labels from:', facts_and_cluster_based_labels_filepath)
        facts_and_cluster_based_labels = get_cached_pickle_file(facts_and_cluster_based_labels_filepath)
        fact2labels = facts_and_cluster_based_labels['fact2labels']
        obs_al2fact = facts_and_cluster_based_labels['obs_al2fact']
        # Count labels
        label2count = {}
        for labels in fact2labels.values():
            found = 0
            assert len(labels) > 0
            for key in ('fact_ccid', 'short_obs_ccid', 'detailed_obs_ccid'):
                if key not in labels:
                    continue
                found += 1
                label = labels[key]
                label2count[label] = label2count.get(label, 0) + 1
            if 'anatloc_ccid' in labels:
                found += 1
            assert found == len(labels)
            if 'fact_ccid' in labels and 'anatloc_ccid' in labels:
                obs_al = (labels['fact_ccid'], labels['anatloc_ccid'])
                label2count[obs_al] = label2count.get(obs_al, 0) + 1
        # Keep only the top "max_num_labels" most frequent labels
        label_count_pairs = [(label, count) for label, count in label2count.items()]
        label_count_pairs.sort(key=lambda x: x[1], reverse=True)
        label_count_pairs = label_count_pairs[:max_num_labels]
        top_labels = [label for label, _ in label_count_pairs]
        top_label_counts = [count for _, count in label_count_pairs]
        top_labels_set = set(top_labels)
        print('len(top_labels):', len(top_labels))
        # Label names
        top_label_names = []
        rkmc = get_cached_pickle_file(refined_kmedoids_clustering_obs_filepath)
        for label in top_labels:
            if isinstance(label, tuple):
                top_label_names.append(obs_al2fact[label])
            else:
                top_label_names.append(rkmc['refined_cluster_center_sentences'][label])
        # Load reports
        print('Loading reports from:', background_findings_and_impression_per_report_filepath)
        reports = get_cached_json_file(background_findings_and_impression_per_report_filepath)
        print('len(reports):', len(reports))
        # Load integrated sentence facts
        print('Loading integrated sentence facts from:', integrated_sentence_facts_filepath)
        integrated_sentence_facts = get_cached_jsonl_file(integrated_sentence_facts_filepath)
        sentence2facts = { row['sentence']: row['facts'] for row in integrated_sentence_facts }
        print('len(sentence2facts):', len(sentence2facts))
        # Assign cluster-based labels to each report
        labeled_reports = []
        for row in tqdm(reports, mininterval=2):
            report_path = row['path']
            findings = row['findings']
            impression = row['impression']
            row_labels = []
            for text in (findings, impression):
                if not text:
                    continue
                for s in sent_tokenize(text):
                    facts = sentence2facts[s]
                    for fact in facts:
                        labels = fact2labels[fact]
                        assert 'fact_ccid' in labels
                        label = labels['fact_ccid']
                        if label in top_labels_set:
                            row_labels.append(label)
                        if 'anatloc_ccid' in labels:
                            obs_al = (labels['fact_ccid'], labels['anatloc_ccid'])
                            if obs_al in top_labels_set:
                                row_labels.append(obs_al)
            # remove duplicates while preserving order
            seen = set()
            row_labels = [x for x in row_labels if not (x in seen or seen.add(x))]
            labeled_reports.append({
                'report_path': report_path,
                'labels': row_labels,
            })
        print('Saving cluster-based labels per report to:', save_path)
        output =  {
            'top_label_names': top_label_names,
            'top_label_ids': top_labels,
            'top_label_counts': top_label_counts,
            'labeled_reports': labeled_reports,
        }
        save_pickle(output, save_path, add_to_cache=True)
    
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrated_fact_metadata_filepath', type=str, required=True)
    parser.add_argument('--integrated_sentence_facts_filepath', type=str, required=True)
    parser.add_argument('--background_findings_and_impression_per_report_filepath', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, choices=['cxr-bert-specialized', 'biovil-t'])
    parser.add_argument('--device', type=str, default='GPU', choices=['CPU', 'GPU'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)
    parser.add_argument('--max_num_labels', type=int, default=500) # Keep only the top "max_num_labels" most frequent labels
    parser.add_argument('--paraphrases_jsonl_filepaths', type=str, nargs='+', required=True)
    # clustering parameters
    parser.add_argument('--num_kmeans_clusters__obs_sentences', type=int, default=100)
    parser.add_argument('--num_kmedoids_clusters__obs_sentences', type=int, default=500)
    parser.add_argument('--num_kmeans_clusters__anatloc_sentences', type=int, default=50)
    parser.add_argument('--num_kmedoids_clusters__anatloc_sentences', type=int, default=300)
    parser.add_argument('--num_kmeans_iterations', type=int, default=300)
    parser.add_argument('--num_kmedoids_iterations', type=int, default=300)
    parser.add_argument("--kmedoids_method", type=str, default="alternate", choices=["pam", "alternate"])
    parser.add_argument('--k_nearest_sentences', type=int, default=30) # Number of nearest sentences to consider for each cluster center

    args = parser.parse_args()

    # Load paraphrases
    print_bold('Loading paraphrases...')
    sentence2paraphrases = {}
    for filepath in args.paraphrases_jsonl_filepaths:
        print('Loading paraphrases from:', filepath)
        paraphrases = load_jsonl(filepath)
        print('len(paraphrases):', len(paraphrases))
        for row in paraphrases:
            s = next(iter(row['metadata'].values()))
            p = row['parsed_response']
            if s not in sentence2paraphrases:
                sentence2paraphrases[s] = []
            sentence2paraphrases[s].extend(p)
    print('len(sentence2paraphrases):', len(sentence2paraphrases))

    # Load integrated fact metadata
    print_bold('Loading integrated fact metadata from:', args.integrated_fact_metadata_filepath)
    integrated_fact_metadata = get_cached_jsonl_file(args.integrated_fact_metadata_filepath)
    sentences = set()
    obs_sentences = set()
    anatloc_sentences = set()
    for row in integrated_fact_metadata:
        fact = row['fact']
        metadata = row['metadata']
        short_obs = metadata['short observation']
        detailed_obs = metadata['detailed observation']
        anatloc = metadata['anatomical location']
        for x in (fact, short_obs, detailed_obs):
            if x:
                sentences.add(x)
                obs_sentences.add(x)
                if x in sentence2paraphrases:
                    p = sentence2paraphrases[x]
                    sentences.update(p)
                    obs_sentences.update(p)
        if anatloc:
            sentences.add(anatloc)
            anatloc_sentences.add(anatloc)
            if anatloc in sentence2paraphrases:
                p = sentence2paraphrases[anatloc]
                sentences.update(p)
                anatloc_sentences.update(p)
    sentences = list(sentences)
    sentences.sort(key=lambda x: (len(x), x)) # Sort by length and then alphabetically
    print('len(sentences):', len(sentences))
    print('len(obs_sentences):', len(obs_sentences))
    print('len(anatloc_sentences):', len(anatloc_sentences))
    sentence2index = {s:i for i, s in enumerate(sentences)}
    obs_idxs = [sentence2index[s] for s in obs_sentences]
    obs_idxs.sort()
    anatloc_idxs = [sentence2index[s] for s in anatloc_sentences]
    anatloc_idxs.sort()

    # Obtain embeddings for each sentence
    len_sum = sum(len(x) for x in sentences)
    num_sentences = len(sentences)
    sentence_embeddings_save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_FAST_CACHE_DIR,
        prefix='sentence_embeddings',
        strings=[
            args.integrated_fact_metadata_filepath,
            args.model_checkpoint_folder_path,
            f'num_sentences={num_sentences}',
            f'len_sum={len_sum}',
        ],
    )
    if os.path.exists(sentence_embeddings_save_path):
        print_bold('Sentence embeddings found at:', sentence_embeddings_save_path)
    else:
        print_bold('Computing embeddings for each sentence...')
        if args.model_name == 'cxr-bert-specialized':
            embeddings = compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized(
                texts=sentences, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
                model_checkpoint_folder_path=args.model_checkpoint_folder_path)
        elif args.model_name == 'biovil-t':
            embeddings = compute_text_embeddings_with_BiomedVLP_BioVilT(
                texts=sentences, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
                model_checkpoint_folder_path=args.model_checkpoint_folder_path)
        else: assert False
        print('Embeddings shape:', embeddings.shape)
        print('Saving sentence embeddings to:', sentence_embeddings_save_path)
        save_pickle({
            'sentences': sentences,
            'embeddings': embeddings,
        }, sentence_embeddings_save_path, add_to_cache=True)

    sentence_rareness_scorer = SentenceRarenessScorer(
        background_findings_and_impression_per_report_filepath=args.background_findings_and_impression_per_report_filepath,
    )

    # Compute kmeans and kmedoids clustering for observation sentences
    print_bold('Computing kmeans and kmedoids clustering for observation sentences...')    
    clustering_results_obs = _compute_kmeans_and_kmedoids_clustering(
        alias='obs_sentences',
        sentence_embeddings_filepath=sentence_embeddings_save_path,
        sentence_idxs=obs_idxs,
        num_kmeans_clusters=args.num_kmeans_clusters__obs_sentences,
        num_kmeans_iterations=args.num_kmeans_iterations,
        num_kmedoids_clusters=args.num_kmedoids_clusters__obs_sentences,
        num_kmedoids_iterations=args.num_kmedoids_iterations,
        kmedoids_method=args.kmedoids_method,
        sentence_rareness_scorer=sentence_rareness_scorer,
        k_nearest_sentences=args.k_nearest_sentences,
    )

    # Compute kmeans and kmedoids clustering for anatomical location sentences
    print_bold('Computing kmeans and kmedoids clustering for anatomical location sentences...')
    clustering_results_anatloc = _compute_kmeans_and_kmedoids_clustering(
        alias='anatloc_sentences',
        sentence_embeddings_filepath=sentence_embeddings_save_path,
        sentence_idxs=anatloc_idxs,
        num_kmeans_clusters=args.num_kmeans_clusters__anatloc_sentences,
        num_kmeans_iterations=args.num_kmeans_iterations,
        num_kmedoids_clusters=args.num_kmedoids_clusters__anatloc_sentences,
        num_kmedoids_iterations=args.num_kmedoids_iterations,
        kmedoids_method=args.kmedoids_method,
        sentence_rareness_scorer=sentence_rareness_scorer,
        k_nearest_sentences=args.k_nearest_sentences,
    )

    # Assign closest obs cluster center to each obs sentence
    print_bold('Assigning closest obs cluster center to each observation sentence...')
    closest_cluster_center_to_sentence_obs_filepath = _assign_closest_cluster_center_to_each_sentence(
        alias='(obs_sentences,obs_clusters)',
        sentence_embeddings_filepath=sentence_embeddings_save_path,
        sentence_idxs=obs_idxs,
        refined_kmedoids_clustering_filepath=clustering_results_obs['kmedoids_refinement_filepath'],
        num_processes=args.num_processes,
    )

    # Assign closest anatloc cluster center to each anatloc sentence
    print_bold('Assigning closest anatloc cluster center to each anatloc sentence...')
    closest_cluster_center_to_sentence_anatloc_filepath = _assign_closest_cluster_center_to_each_sentence(
        alias='(anatloc_sentences,anatloc_clusters)',
        sentence_embeddings_filepath=sentence_embeddings_save_path,
        sentence_idxs=anatloc_idxs,
        refined_kmedoids_clustering_filepath=clustering_results_anatloc['kmedoids_refinement_filepath'],
        num_processes=args.num_processes,
    )

    # Assign closest obs sentence to each obs cluster center
    print_bold('Assigning closest obs sentence to each obs cluster center...')
    _assign_closest_sentence_to_each_cluster_center(
        alias='(obs_sentences,obs_clusters)',
        sentence_embeddings_filepath=sentence_embeddings_save_path,
        sentence_idxs=obs_idxs,
        kmedoids_clustering_filepath=clustering_results_obs['kmedoids_clustering_filepath'],
        num_processes=args.num_processes,
    )

    # Assign closest anatloc sentence to each anatloc cluster center
    print_bold('Assigning closest anatloc sentence to each anatloc cluster center...')
    _assign_closest_sentence_to_each_cluster_center(
        alias='(anatloc_sentences,anatloc_clusters)',
        sentence_embeddings_filepath=sentence_embeddings_save_path,
        sentence_idxs=anatloc_idxs,
        kmedoids_clustering_filepath=clustering_results_anatloc['kmedoids_clustering_filepath'],
        num_processes=args.num_processes,
    )

    # Assign cluster-based labels to each fact
    print_bold('Assigning cluster-based labels to each fact...')
    facts_and_cluster_based_labels_filepath = _assign_cluster_based_labels_to_each_fact(
        sentence2index=sentence2index,
        integrated_fact_metadata_filepath=args.integrated_fact_metadata_filepath,
        closest_cluster_center_to_sentence_obs_filepath=closest_cluster_center_to_sentence_obs_filepath,
        closest_cluster_center_to_sentence_anatloc_filepath=closest_cluster_center_to_sentence_anatloc_filepath,
        sentence_rareness_scorer=sentence_rareness_scorer,
    )

    # Assign cluster-based labels to each report
    print_bold('Assigning cluster-based labels to each report...')
    _assign_cluster_based_labels_to_each_report(
        facts_and_cluster_based_labels_filepath=facts_and_cluster_based_labels_filepath,
        background_findings_and_impression_per_report_filepath=args.background_findings_and_impression_per_report_filepath,
        integrated_sentence_facts_filepath=args.integrated_sentence_facts_filepath,
        refined_kmedoids_clustering_obs_filepath=clustering_results_obs['kmedoids_refinement_filepath'],
        max_num_labels=args.max_num_labels,
    )

    print('Done!')

# Utility class for data inspection

class ClusteringBasedLabelVisualizer:

    def __init__(
            self,
            background_findings_and_impression_per_report_filepath,
            integrated_sentence_facts_filepath,
            integrated_fact_metadata_filepath,
            kmedoids_refinement_obs_sentences_filepath,
            kmedoids_refinement_anatloc_sentences_filepath,
            cluster_based_labels_per_fact_filepath,
            cluster_based_labels_per_report_filepath,
        ):
        print(f'Loading {background_findings_and_impression_per_report_filepath}...')
        self.report_data = get_cached_json_file(background_findings_and_impression_per_report_filepath)
        print(f'Loading {integrated_sentence_facts_filepath}...')
        self.integrated_sentence_facts = get_cached_jsonl_file(integrated_sentence_facts_filepath)
        self.sentence2facts = { row['sentence']: row['facts'] for row in self.integrated_sentence_facts }
        print(f'Loading {integrated_fact_metadata_filepath}...')
        self.integrated_fact_metadata = get_cached_jsonl_file(integrated_fact_metadata_filepath)
        self.fact2metadata = { row['fact']: row['metadata'] for row in self.integrated_fact_metadata }
        print(f'Loading {cluster_based_labels_per_fact_filepath}...')
        self.cblpf = get_cached_pickle_file(cluster_based_labels_per_fact_filepath)
        self.fact2labels = self.cblpf['fact2labels']
        self.obs_al2fact = self.cblpf['obs_al2fact']
        print(f'Loading {kmedoids_refinement_obs_sentences_filepath}...')
        self.kmros = get_cached_pickle_file(kmedoids_refinement_obs_sentences_filepath)
        self.obs_cc_sentences = self.kmros['refined_cluster_center_sentences']
        print(f'Loading {kmedoids_refinement_anatloc_sentences_filepath}...')
        self.kmrals = get_cached_pickle_file(kmedoids_refinement_anatloc_sentences_filepath)
        self.anatloc_cc_sentences = self.kmrals['refined_cluster_center_sentences']
        print(f'Loading {cluster_based_labels_per_report_filepath}...')
        self.cblpr = get_cached_pickle_file(cluster_based_labels_per_report_filepath)
        self.report_label_id_2_name = {x:y for x, y in zip(self.cblpr['top_label_ids'], self.cblpr['top_label_names'])}

    def _print_fact_metadata_and_labels(self, fact):
        metadata = self.fact2metadata[fact]
        labels = self.fact2labels[fact]
        
        print_blue('fact:', bold=True, end=' '); print_magenta(fact, bold=True)
        assert 'fact_ccid' in labels
        print_blue('fact_ccid:', bold=True, end=' '); print(f'{labels["fact_ccid"]} -> {self.obs_cc_sentences[labels["fact_ccid"]]}')
        
        detailed_obs = metadata['detailed observation']
        if detailed_obs:
            assert 'detailed_obs_ccid' in labels
            print_blue('detailed obs:', bold=True, end=' '); print(detailed_obs)
            print_blue('detailed_obs_ccid:', bold=True, end=' '); print(f'{labels["detailed_obs_ccid"]} -> {self.obs_cc_sentences[labels["detailed_obs_ccid"]]}')

        short_obs = metadata['short observation']
        if short_obs:
            assert 'short_obs_ccid' in labels
            print_blue('short obs:', bold=True, end=' '); print(short_obs)
            print_blue('short_obs_ccid:', bold=True, end=' '); print(f'{labels["short_obs_ccid"]} -> {self.obs_cc_sentences[labels["short_obs_ccid"]]}')

        anatloc = metadata['anatomical location']
        if anatloc:
            assert 'anatloc_ccid' in labels
            print_blue('anatloc:', bold=True, end=' '); print(anatloc)
            print_blue('anatloc_ccid:', bold=True, end=' '); print(f'{labels["anatloc_ccid"]} -> {self.anatloc_cc_sentences[labels["anatloc_ccid"]]}')

        if 'fact_ccid' in labels and 'anatloc_ccid' in labels:
            obs_al = (labels['fact_ccid'], labels['anatloc_ccid'])
            print_blue('obs_al:', bold=True, end=' '); print(f'{obs_al} -> {self.obs_al2fact[obs_al]}')

    def display_report(self, i=None):
        if i is None:
            i = np.random.randint(len(self.report_data))
        row = self.report_data[i]
        print(f'i = {i}')
        print_bold('Report path:')
        print(row['path'])
        print()
        
        print_bold('Original report:')
        with open(row['path'], 'r') as f:
            print(f.read())
        print()
        
        print_bold('Background:')
        print(row['background'])
        print()
        
        print_bold('Findings:')
        print(row['findings'])        
        print()

        print_bold('Impression:')
        print(row['impression'])
        print()

        print('-' * 80)
        print_bold('Facts per sentence:')
        facts = []
        for s in sent_tokenize(row['findings']):
            print_blue(s, bold=True)
            for f in self.sentence2facts[s]:
                print(f'\t- {f}')
                facts.append(f)
            print()
        for s in sent_tokenize(row['impression']):
            print_blue(s, bold=True)
            for f in self.sentence2facts[s]:
                print(f'\t- {f}')
                facts.append(f)
            print()
        
        print()
        print('-' * 80)
        print_bold('Labels per fact:')
        for f in facts:
            self._print_fact_metadata_and_labels(f)
            print()

        print()
        print('-' * 80)
        print_bold('Labels in report:')
        for label in self.cblpr['labeled_reports'][i]['labels']:
            print(label, '->', self.report_label_id_2_name[label])

if __name__ == '__main__':
    main()