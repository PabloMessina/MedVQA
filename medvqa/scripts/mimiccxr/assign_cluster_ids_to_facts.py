import argparse
import os
import random
import numpy as np
import multiprocessing as mp

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_similarity

from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.models.huggingface_utils import compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized
from medvqa.utils.logging_utils import get_console_logger
from medvqa.utils.files_utils import load_jsonl, load_pickle, save_pickle

def _sample_from_each_cluster(cluster_ids, embeddings, cluster_centers, n_samples_per_cluster):
    """
    Sample n_samples_per_cluster from each cluster.
    """
    num_clusters = cluster_centers.shape[0]
    sample_idxs = []
    cid2idxs = {}
    for i, cid in enumerate(cluster_ids):
        try:
            cid2idxs[cid].append(i)
        except KeyError:
            cid2idxs[cid] = [i]
    for i in range(num_clusters):
        idxs = cid2idxs[i]
        if len(idxs) <= n_samples_per_cluster:
            sample_idxs.extend(idxs)
            continue
        cluster_center = cluster_centers[i]
        cluster_center = cluster_center.reshape(1, -1)
        cluster_embeddings = embeddings[idxs]
        cluster_center_similarities = cosine_similarity(cluster_center, cluster_embeddings) # (1, n_samples)
        cluster_center_similarities = cluster_center_similarities.reshape(-1) # (n_samples)
        sorted_indices = np.argsort(cluster_center_similarities)[::-1] # (n_samples)
        sorted_indices = sorted_indices[np.linspace(0, len(sorted_indices)-1, n_samples_per_cluster, dtype=int)]
        sample_idxs.extend(idxs[j] for j in sorted_indices)
    return sample_idxs

def _compute_anatomical_location_embeddings(facts_metadata, logger):
    # Load facts
    anat_loc_freq = {}
    for row in facts_metadata:
        anat_loc = row['metadata']['anatomical location']
        anat_loc_freq[anat_loc] = anat_loc_freq.get(anat_loc, 0) + 1
    anat_locs = list(anat_loc_freq.keys())
    logger.info(f"Loaded {len(anat_locs)} unique anatomical locations")
    logger.info(f"Num of anatomical locations with freq >= 10: {len([x for x in anat_loc_freq.values() if x >= 10])}")
    logger.info(f"Num of anatomical locations with freq >= 100: {len([x for x in anat_loc_freq.values() if x >= 100])}")
    # Print some examples
    logger.info(f"Examples of anatomical locations:")
    for i in random.sample(range(len(anat_locs)), 5):
        logger.info(f"    {anat_locs[i]} ({anat_loc_freq[anat_locs[i]]})")
    
    # Compute embeddings
    anat_loc_total_length = sum(len(x) for x in anat_locs)
    anat_loc_embeddings_save_path = os.path.join(MIMICCXR_CACHE_DIR, f"anatomical_location_embeddings(BiomedVLP-CXR-BERT-specialized,{len(anat_locs)},{anat_loc_total_length}).pkl")
    if os.path.exists(anat_loc_embeddings_save_path):
        logger.info(f"Anatomical location embeddings already exist at {anat_loc_embeddings_save_path}")
    else:
        embeddings = compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized(anat_locs,
                                                                                device=args.device,
                                                                                batch_size=args.batch_size,
                                                                                num_workers=args.num_workers,
                                                                                )
        logger.info(f"embeddings.shape: {embeddings.shape}")
        logger.info(f"Saving embeddings to {anat_loc_embeddings_save_path}")
        anat_locs_and_embeddings = {
            'anatomical locations': anat_locs,
            'embeddings': embeddings,
        }
        # Save embeddings
        save_pickle(anat_locs_and_embeddings, anat_loc_embeddings_save_path)
        logger.info(f"anat_locs_and_embeddings['embeddings'].shape: {anat_locs_and_embeddings['embeddings'].shape}")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated_sentence_facts_filepath", type=str, required=True)
    parser.add_argument("--integrated_facts_metadata_filepath", type=str, required=True)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--num_kmeans_clusters", type=int, default=1000)
    parser.add_argument("--num_kmeans_iterations", type=int, default=300)
    parser.add_argument("--num_samples_per_kmeans_cluster", type=int, default=50)
    parser.add_argument("--num_kmedoids_clusters", type=int, default=1000)
    parser.add_argument("--num_kmedoids_iterations", type=int, default=300)
    parser.add_argument("--kmedoids_method", type=str, default="alternate", choices=["pam", "alternate"])

    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    # Compute anatomical location embeddings
    logger.info(f"Loading integrated facts metadata from {args.integrated_facts_metadata_filepath}")
    facts_metadata = load_jsonl(args.integrated_facts_metadata_filepath)
    _compute_anatomical_location_embeddings(facts_metadata, logger)

    # Load facts
    fact_freq = {}
    assert os.path.exists(args.integrated_sentence_facts_filepath), f"File not found: {args.integrated_sentence_facts_filepath}"
    logger.info(f"Loading facts from {args.integrated_sentence_facts_filepath}")
    sentence_facts = load_jsonl(args.integrated_sentence_facts_filepath)
    logger.info(f"Loaded {len(sentence_facts)} sentences with facts")
    for row in sentence_facts:
        for f in row['facts']:
            fact_freq[f] = fact_freq.get(f, 0) + 1
    facts = list(fact_freq.keys())
    logger.info(f"Loaded {len(facts)} unique facts")
    logger.info(f"Num of facts with freq >= 10: {len([x for x in fact_freq.values() if x >= 10])}")
    logger.info(f"Num of facts with freq >= 100: {len([x for x in fact_freq.values() if x >= 100])}")
    # Print some examples
    logger.info(f"Examples of facts:")
    for i in random.sample(range(len(facts)), 5):
        logger.info(f"    {facts[i]} ({fact_freq[facts[i]]})")
    
    # Compute fact embeddings
    facts_total_length = sum(len(f) for f in facts)
    facts_embeddings_save_path = os.path.join(MIMICCXR_CACHE_DIR, f"fact_embeddings(BiomedVLP-CXR-BERT-specialized,{len(facts)},{facts_total_length}).pkl")
    if os.path.exists(facts_embeddings_save_path):
        logger.info(f"Facts embeddings already exist at {facts_embeddings_save_path}")
    else:
        embeddings = compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized(facts,
                                                                                device=args.device,
                                                                                batch_size=args.batch_size,
                                                                                num_workers=args.num_workers,
                                                                                )
        logger.info(f"embeddings.shape: {embeddings.shape}")
        logger.info(f"Saving embeddings to {facts_embeddings_save_path}")
        facts_and_embeddings = {
            'facts': facts,
            'embeddings': embeddings,
        }
        # Save embeddings
        save_pickle(facts_and_embeddings, facts_embeddings_save_path)
        logger.info(f"facts_and_embeddings['embeddings'].shape: {facts_and_embeddings['embeddings'].shape}")

    # Cluster facts with KMeans
    kmeans_clustering_save_path = os.path.join(MIMICCXR_CACHE_DIR, f"fact_kmeans_clustering(BiomedVLP-CXR-BERT-specialized,{len(facts)},{facts_total_length},"
                                               f"{args.num_kmeans_clusters},{args.num_kmeans_iterations}).pkl")
    if os.path.exists(kmeans_clustering_save_path):
        logger.info(f"KMeans clustering already exists at {kmeans_clustering_save_path}")
    else:
        logger.info(f"Loading facts and embeddings from {facts_embeddings_save_path}")
        facts_and_embeddings = load_pickle(facts_embeddings_save_path)
        facts = facts_and_embeddings['facts']
        embeddings = facts_and_embeddings['embeddings']
        logger.info(f"Clustering facts with KMeans (n_clusters={args.num_kmeans_clusters}, n_init='auto', verbose=2, max_iter={args.num_kmeans_iterations})")
        kmeans = KMeans(n_clusters=args.num_kmeans_clusters, random_state=0, n_init='auto', verbose=2,
                        max_iter=args.num_kmeans_iterations).fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        cluster_ids = kmeans.labels_
        assert cluster_ids.shape[0] == len(facts)

        # Find the closest fact to each cluster center
        def _closest_fact_to_cluster_center(i):
            return np.argmax(cosine_similarity(cluster_centers[i].reshape(1, -1), embeddings))
        logger.info(f"Finding the closest fact to each cluster center (n_clusters={args.num_kmeans_clusters})")
        with mp.Pool(processes=args.num_processes) as pool:
            closest_facts_to_cluster_centers = pool.map(_closest_fact_to_cluster_center, range(args.num_kmeans_clusters))

        # Save clustering
        kmeans_clustering = {
            'cluster_ids': cluster_ids,
            'cluster_centers': cluster_centers,
            'closest_facts_to_cluster_centers': closest_facts_to_cluster_centers,
        }
        logger.info(f"Saving KMeans clustering to {kmeans_clustering_save_path}")
        save_pickle(kmeans_clustering, kmeans_clustering_save_path)

    # Cluster facts with KMedoids based on KMeans clustering
    kmedoids_clustering_save_path = os.path.join(MIMICCXR_CACHE_DIR, f"fact_kmedoids_clustering(BiomedVLP-CXR-BERT-specialized,{len(facts)},{facts_total_length},"
                                                    f"{args.num_kmedoids_clusters},{args.num_kmedoids_iterations},{args.num_kmeans_clusters},{args.num_kmeans_iterations},"
                                                    f"{args.num_samples_per_kmeans_cluster}).pkl")
    if os.path.exists(kmedoids_clustering_save_path):
        logger.info(f"KMedoids clustering already exists at {kmedoids_clustering_save_path}")
    else:
        logger.info(f"Loading KMeans clustering from {kmeans_clustering_save_path}")
        kmeans_clustering = load_pickle(kmeans_clustering_save_path)
        cluster_ids = kmeans_clustering['cluster_ids']
        cluster_centers = kmeans_clustering['cluster_centers']
        
        if 'facts_and_embeddings' not in locals(): # check if facts_and_embeddings has been defined
            logger.info(f"Loading facts and embeddings from {facts_embeddings_save_path}")
            facts_and_embeddings = load_pickle(facts_embeddings_save_path)
        fact_embeddings = facts_and_embeddings['embeddings']
        logger.info(f"fact_embeddings.shape: {fact_embeddings.shape}")
        
        logger.info(f"Sampling {args.num_samples_per_kmeans_cluster} facts from each KMeans cluster")
        sampled_idxs = _sample_from_each_cluster(cluster_ids, fact_embeddings, cluster_centers, args.num_samples_per_kmeans_cluster)
        logger.info(f"len(sampled_idxs): {len(sampled_idxs)}")

        logger.info(f"Clustering facts with KMedoids (n_clusters={args.num_kmedoids_clusters}, metric='cosine', method='{args.kmedoids_method}', "
                    f"max_iter={args.num_kmedoids_iterations})")
        sampled_embeddings = fact_embeddings[sampled_idxs]
        kmedoids = KMedoids(n_clusters=args.num_kmedoids_clusters, metric='cosine', method=args.kmedoids_method,
                            max_iter=args.num_kmedoids_iterations).fit(sampled_embeddings)
        cluster_ids = kmedoids.labels_
        cluster_centers = kmedoids.cluster_centers_
        assert len(cluster_ids) == len(sampled_embeddings)

        # Find the closest fact to each cluster center
        def _closest_fact_to_cluster_center(i):
            return np.argmax(cosine_similarity(cluster_centers[i].reshape(1, -1), fact_embeddings))
        logger.info(f"Finding the closest fact to each cluster center (n_clusters={args.num_kmedoids_clusters})")
        with mp.Pool(processes=args.num_processes) as pool:
            closest_facts_to_cluster_centers = pool.map(_closest_fact_to_cluster_center, range(args.num_kmedoids_clusters))

        # Find the closest cluster center to each fact
        def _closest_cluster_center_to_fact(i):
            return np.argmax(cosine_similarity(fact_embeddings[i].reshape(1, -1), cluster_centers))
        logger.info(f"Finding the closest cluster center to each fact (n_facts={len(fact_embeddings)})")
        with mp.Pool(processes=args.num_processes) as pool:
            closest_cluster_centers_to_facts = pool.map(_closest_cluster_center_to_fact, range(len(fact_embeddings)))

        # Save clustering
        kmedoids_clustering = {
            'cluster_centers': cluster_centers,
            'closest_facts_to_cluster_centers': closest_facts_to_cluster_centers,
            'closest_cluster_centers_to_facts': closest_cluster_centers_to_facts,
        }
        logger.info(f"Saving KMedoids clustering to {kmedoids_clustering_save_path}")
        save_pickle(kmedoids_clustering, kmedoids_clustering_save_path)
    