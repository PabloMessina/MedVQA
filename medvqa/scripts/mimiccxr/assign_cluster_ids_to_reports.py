import argparse
import os
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
import numpy as np

from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.models.huggingface_utils import compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized
from medvqa.utils.logging import get_console_logger
from medvqa.utils.files import load_json, load_pickle, read_lines_from_txt, save_pickle
from medvqa.metrics.medical.med_completeness import MEDICAL_TERMS_PATH

import multiprocessing as mp

from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_similarity

_medical_terms = set(read_lines_from_txt(MEDICAL_TERMS_PATH))
def _contains_medical_terms(text, k):
    count = 0
    for x in wordpunct_tokenize(text.lower()):
        if x in _medical_terms:
            count += 1
            if count >= k: return True
    return False

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_reports_filename", type=str, required=True)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--num_clusters", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=50000)

    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    # Load reports
    preprocessed_reports_filepath = os.path.join(MIMICCXR_CACHE_DIR, args.preprocessed_reports_filename)
    assert os.path.exists(preprocessed_reports_filepath), f"Preprocessed reports file {preprocessed_reports_filepath} does not exist"
    logger.info(f"Loading preprocessed reports from {preprocessed_reports_filepath}")
    reports = load_json(preprocessed_reports_filepath)

    # Load report embeddings
    report_embeddings_save_path = os.path.join(MIMICCXR_CACHE_DIR, f"report_embeddings(BiomedVLP-CXR-BERT-specialized,{len(reports)}).pkl")
    if os.path.exists(report_embeddings_save_path):
        logger.info(f"Loading report embeddings from {report_embeddings_save_path}")
        report_embeddings = load_pickle(report_embeddings_save_path)
    else:

        # Extract sentences from reports
        logger.info(f"Extracting sentences from reports")
        sentences_per_report = [[] for _ in range(len(reports))]
        unique_sentences = set()
        empty_reports_idxs = []
        for i, r in tqdm(enumerate(reports), total=len(reports), mininterval=2):
            impression = r['impression']
            findings = r['findings']
            if len(impression) > 0:
                for s in sent_tokenize(impression):
                    if _contains_medical_terms(s, 1):
                        sentences_per_report[i].append(s)
                        unique_sentences.add(s)
            if len(findings) > 0:
                for s in sent_tokenize(findings):
                    if _contains_medical_terms(s, 1):
                        sentences_per_report[i].append(s)
                        unique_sentences.add(s)
            if len(sentences_per_report[i]) == 0:
                empty_reports_idxs.append(i)
        logger.info(f"Found {len(unique_sentences)} unique sentences")
        if len(empty_reports_idxs) > 0:
            logger.warning(f"Found {len(empty_reports_idxs)}/{len(reports)} empty reports")
            logger.warning(f"Examples of empty reports:")
            for i in empty_reports_idxs[:5]:
                logger.warning(f"    {reports[i]}")
        unique_sentences = list(unique_sentences)

        # Compute sentence-level embeddings
        _n_s = len(unique_sentences)
        _n_chars = sum(len(s) for s in unique_sentences)
        sentences_and_embeddings_save_path = os.path.join(MIMICCXR_CACHE_DIR, f"sentence_embeddings(BiomedVLP-CXR-BERT-specialized,{_n_s},{_n_chars}).pkl")
        if os.path.exists(sentences_and_embeddings_save_path):
            logger.info(f"Loading embeddings from {sentences_and_embeddings_save_path}")
            sentences_and_embeddings = load_pickle(sentences_and_embeddings_save_path)
        else:
            logger.info(f"Computing embeddings for {len(unique_sentences)} unique sentences")
            embeddings = compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized(unique_sentences,
                                                                                     device=args.device,
                                                                                     logger=logger,
                                                                                     batch_size=args.batch_size,
                                                                                     num_workers=args.num_workers,
                                                                                     )
            logger.info(f"embeddings.shape: {embeddings.shape}")
            logger.info(f"Saving embeddings to {sentences_and_embeddings_save_path}")
            sentences_and_embeddings = {
                'sentences': unique_sentences,
                'embeddings': embeddings,
            }
            # Save embeddings
            save_pickle(sentences_and_embeddings, sentences_and_embeddings_save_path)
        logger.info(f"sentences_and_embeddings['embeddings'].shape: {sentences_and_embeddings['embeddings'].shape}")

        # Map senteces to indices
        sentence_to_idx = {s: i for i, s in enumerate(sentences_and_embeddings['sentences'])}

        # Compute report-level embeddings by averaging sentence-level embeddings
        logger.info(f"Computing report embeddings")
        report_embeddings = np.zeros((len(reports), sentences_and_embeddings['embeddings'].shape[1]), dtype=np.float32)
        for i, sentences in tqdm(enumerate(sentences_per_report), total=len(sentences_per_report), mininterval=2):
            for s in sentences:
                report_embeddings[i] += sentences_and_embeddings['embeddings'][sentence_to_idx[s]]
            report_embeddings[i] /= max(len(sentences), 1) # Avoid division by zero
        logger.info(f"Saving report embeddings to {report_embeddings_save_path}")
        save_pickle(report_embeddings, report_embeddings_save_path)

    # Cluster report embeddings
    report_clusters_save_path = os.path.join(MIMICCXR_CACHE_DIR, f"report_clusters(BiomedVLP-CXR-BERT-specialized,{len(reports)},{args.num_clusters},{args.num_samples}).pkl")
    if os.path.exists(report_clusters_save_path):
        logger.info(f"Loading report clusters from {report_clusters_save_path}")
        report_clusters = load_pickle(report_clusters_save_path)
        X_cluster_centers = report_clusters['X_cluster_centers']
        closest_cluster_centers_to_reports = report_clusters['closest_cluster_centers_to_reports']
        closest_reports_to_cluster_centers = report_clusters['closest_reports_to_cluster_centers']
        logger.info(f"X_cluster_centers.shape: {X_cluster_centers.shape}")
        logger.info(f"closest_cluster_centers_to_reports.shape: {closest_cluster_centers_to_reports.shape}")
        logger.info(f"closest_reports_to_cluster_centers.shape: {closest_reports_to_cluster_centers.shape}")
    else:
        logger.info(f"Clustering {len(reports)} report embeddings into {args.num_clusters} clusters")
        if args.num_samples > len(reports):
            logger.warning(f"Number of samples ({args.num_samples}) is greater than the number of reports ({len(reports)}). Using all reports.")
            args.num_samples = len(reports)
            X = report_embeddings
        else:
            logger.info(f"Sampling {args.num_samples} reports")            
            sampled_indices = np.random.choice(len(reports), size=args.num_samples, replace=False)
            X = report_embeddings[sampled_indices]
        
        # Run K-Medoids
        logger.info(f"Running K-Medoids")
        cobj = KMedoids(n_clusters=args.num_clusters, metric='cosine').fit(X)
        labels = cobj.labels_
        X_cluster_centers = cobj.cluster_centers_
        logger.info(f"labels.shape: {labels.shape}")
        logger.info(f"X_cluster_centers.shape: {X_cluster_centers.shape}")

        # Find the closet cluster center for each report
        logger.info(f"Finding the closest cluster center for each report")

        def _closest_cluster_center_to_report(i):
            return np.argmax(cosine_similarity(report_embeddings[i].reshape(1, -1), X_cluster_centers))

        with mp.Pool(processes=args.num_processes) as pool:
            closest_cluster_centers_to_reports = pool.map(_closest_cluster_center_to_report, range(len(report_embeddings)))

        # Find the closest report to each cluster center
        logger.info(f"Finding the closest report to each cluster center")
        
        def _closest_report_to_cluster_center(i):
            return np.argmax(cosine_similarity(X_cluster_centers[i].reshape(1, -1), report_embeddings))
        
        with mp.Pool(processes=args.num_processes) as pool:
            closest_reports_to_cluster_centers = pool.map(_closest_report_to_cluster_center, range(args.num_clusters))

        # Save report clusters
        report_clusters = {
            'X_cluster_centers': X_cluster_centers,
            'closest_cluster_centers_to_reports': closest_cluster_centers_to_reports,
            'closest_reports_to_cluster_centers': closest_reports_to_cluster_centers,
        }
        logger.info(f"Saving report clusters to {report_clusters_save_path}")
        save_pickle(report_clusters, report_clusters_save_path)