import os
import math
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from medvqa.evaluation.plots import plot_metrics
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.constants import LABEL_BASED_FACTS, MIMIC_CXR_LT_LABELS
from medvqa.utils.hashing import hash_string
from medvqa.utils.logging import get_console_logger
from medvqa.datasets.mimiccxr import (
    MIMIC_CXR_LT_LABELS_TRAIN_CSV_PATH,
    MIMIC_CXR_LT_LABELS_DEV_CSV_PATH,
    MIMIC_CXR_LT_LABELS_TEST_CSV_PATH,
    MIMICCXR_FAST_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
    get_imageId2reportId,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.openai_api import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files import load_jsonl, load_pickle

INSTRUCTIONS = """Given a list of radiological factual statements (#F) and a hypothesis (#H), output "Reason: {reason}. Label: {label}" where {reason} is a short explanatory sentence and {label} is one of {definitely true, likely true, unknown, likely false, definitely false}."""

print(INSTRUCTIONS)

POSSIBLE_LABELS = [
    "label: definitely true",
    "label: likely true",
    "label: unknown",
    "label: likely false",
    "label: definitely false",
]

LABEL_TO_BINARY = {
    "definitely true": 1,
    "likely true": 1,
    "unknown": 0,
    "likely false": 0,
    "definitely false": 0,
}

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    original_text = text
    text = text.lower()
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    assert text.startswith("reason: "), f"No reason found in output: {text}"
    for label in POSSIBLE_LABELS:
        try:
            idx = text.index(label)
            assert idx > 8, f"idx: {idx}, label: {label}, text: {text}"
            reason = original_text[8:idx].strip()
            label = label[7:] # Remove "label: "
            return {
                "reason": reason,
                "label": label,
            }
        except ValueError:
            continue
    raise RuntimeError(f"Could not parse output: {text}")

def sample_queries_label_based(num_samples, integrated_report_facts_metadata_jsonl_filepath,
                               chest_imagenome_image_id_to_labels_filepath, chest_imagenome_label_names_filepath,
                               already_processed_queries, allowed_label_based_facts=None,
                               use_mimiccxr_dev_test_sets_only=False):
    """
    Sample queries for the label-based data mode.
    """

    logger.info(f"Sampling {num_samples} queries for the label-based data mode")

    if use_mimiccxr_dev_test_sets_only:
        logger.info("Using MIMIC-CXR dev and test sets only")
        metadata = load_mimiccxr_reports_detailed_metadata()
        splits = metadata['splits']
        logger.info(f"len(splits): {len(splits)}")

    # Load integrated report facts metadata
    reports = load_jsonl(integrated_report_facts_metadata_jsonl_filepath)
    logger.info(f"Loaded {len(reports)} reports from {integrated_report_facts_metadata_jsonl_filepath}")
    study_id_to_report_idx = dict()
    for i, row in enumerate(reports):
        study_id = row['path'].split('/')[-1][1:-4] # Remove s and .txt extension
        study_id = int(study_id) # Convert to int
        assert study_id not in study_id_to_report_idx
        study_id_to_report_idx[study_id] = i
    n_reports = len(reports)

    # Load MIMIC-CXR-LT labels
    df_train = pd.read_csv(MIMIC_CXR_LT_LABELS_TRAIN_CSV_PATH)
    df_dev = pd.read_csv(MIMIC_CXR_LT_LABELS_DEV_CSV_PATH)
    df_test = pd.read_csv(MIMIC_CXR_LT_LABELS_TEST_CSV_PATH)
    logger.info(f"len(df_train): {len(df_train)}")
    logger.info(f"len(df_dev): {len(df_dev)}")
    logger.info(f"len(df_test): {len(df_test)}")

    mcxrlt_label_id_to_report_idxs = [set() for _ in range(len(MIMIC_CXR_LT_LABELS))]
    
    for df in [df_train, df_dev, df_test]:
        labels = df[MIMIC_CXR_LT_LABELS].values
        for i, study_id in enumerate(df['study_id']):
            for j, x in enumerate(labels[i]):
                if x == 1:
                    ridx = study_id_to_report_idx[study_id]
                    if use_mimiccxr_dev_test_sets_only:
                        if splits[ridx] == 'train':
                            continue
                    mcxrlt_label_id_to_report_idxs[j].add(ridx)

    # print the number of reports for each label
    for i, label in enumerate(MIMIC_CXR_LT_LABELS):
        logger.info(f"{label}: {len(mcxrlt_label_id_to_report_idxs[i])}")

    # Load chest imagenome labels
    ci_label_names = load_pickle(chest_imagenome_label_names_filepath)
    ci_image_id_to_labels = load_pickle(chest_imagenome_image_id_to_labels_filepath)
    logger.info(f"Loaded {len(ci_label_names)} label names from {chest_imagenome_label_names_filepath}")
    logger.info(f"Loaded {len(ci_image_id_to_labels)} image labels from {chest_imagenome_image_id_to_labels_filepath}")

    image_id_to_report_idx = get_imageId2reportId()
    valid_label_idxs = [i for i, label in enumerate(ci_label_names) if len(label) == 2\
                         and label[0] not in ["laterality", "severity", "nlp", "temporal", ]]
    ci_label_names = [ci_label_names[i] for i in valid_label_idxs]
    ci_label_matrix = np.zeros((n_reports, len(ci_label_names)), dtype=int)
    for i, (image_id, labels) in enumerate(ci_image_id_to_labels.items()):
        if image_id in image_id_to_report_idx:
            report_idx = image_id_to_report_idx[image_id]
            ci_label_matrix[report_idx] = labels[valid_label_idxs]

    logger.info(f"ci_label_matrix.shape: {ci_label_matrix.shape}")

    ci_label_id_to_report_idxs = [set() for _ in range(len(ci_label_names))]
    for i in range(n_reports):
        if use_mimiccxr_dev_test_sets_only:
            if splits[i] == 'train':
                continue
        for j in range(len(ci_label_names)):
            if ci_label_matrix[i, j] == 1:
                ci_label_id_to_report_idxs[j].add(i)

    # print the number of reports for each label
    for i, label in enumerate(ci_label_names):
        logger.info(f"{label}: {len(ci_label_id_to_report_idxs[i])}")
       
    # Define label aliases
    mcxrlt_label2alias = { label: f'{label.lower()} seen' for label in MIMIC_CXR_LT_LABELS }
    mcxrlt_label2alias['Pleural Other'] = 'pleural abnormalities seen'
    mcxrlt_label2alias['No Finding'] = 'no abnormalities seen'
    assert len(mcxrlt_label2alias) == len(MIMIC_CXR_LT_LABELS)
    mcxrlt_label_names = MIMIC_CXR_LT_LABELS

    ci_label2alias = {}
    for label in ci_label_names:
        if label[0] == 'texture':
            alias = f'{label[1]} texture'
        else:
            alias = label[1]
        alias = f'{alias} seen'
        ci_label2alias[label] = alias

    # Merge two sets of labels
    logger.info("Merging two sets of labels...")

    label_based_facts = set(mcxrlt_label2alias.values()).union(set(ci_label2alias.values()))
    label_based_facts = list(label_based_facts)
    label_based_facts.sort()
    logger.info(f"len(label_based_facts): {len(label_based_facts)}")
    logger.info(f'Label based facts: {label_based_facts}')
    assert label_based_facts == LABEL_BASED_FACTS
    assert len(ci_label2alias) == len(ci_label_names)

    if allowed_label_based_facts is not None:
        logger.info(f'Allowed label based facts: {allowed_label_based_facts}')
        assert set(allowed_label_based_facts).issubset(set(label_based_facts))
        label_based_facts = allowed_label_based_facts
        logger.info(f'len(label_based_facts): {len(label_based_facts)}')
    
    fact2ridxs = { fact: set() for fact in label_based_facts }
    for i, label in enumerate(mcxrlt_label_names):
        fact = mcxrlt_label2alias[label]
        ridxs = mcxrlt_label_id_to_report_idxs[i]
        if fact in fact2ridxs:
            fact2ridxs[fact].update(ridxs)
    for i, label in enumerate(ci_label_names):
        fact = ci_label2alias[label]
        ridxs = ci_label_id_to_report_idxs[i]
        if fact in fact2ridxs:
            fact2ridxs[fact].update(ridxs)

    # print the number of reports for each fact
    for fact in label_based_facts:
        logger.info(f"{fact}: {len(fact2ridxs[fact])}")

    # Sample queries
    queries = []
    n_queries_per_label = math.ceil(num_samples / len(label_based_facts))
    n_pos_queries_per_label = math.ceil(n_queries_per_label * 0.7)
    n_neg_queries_per_label = n_queries_per_label - n_pos_queries_per_label
    logger.info(f"n_queries_per_label: {n_queries_per_label}")
    logger.info(f"n_pos_queries_per_label: {n_pos_queries_per_label}")
    logger.info(f"n_neg_queries_per_label: {n_neg_queries_per_label}")

    if use_mimiccxr_dev_test_sets_only:
        all_report_idxs = [i for i in range(n_reports) if splits[i] != 'train']
    else:
        all_report_idxs = list(range(n_reports))
    logger.info(f"len(all_report_idxs): {len(all_report_idxs)}")

    for fact, ridxs in fact2ridxs.items():
        ridxs_list = list(ridxs)
        random.shuffle(ridxs_list)
        cnt = 0
        for j, ridx in enumerate(ridxs_list):
            if use_mimiccxr_dev_test_sets_only:
                assert splits[ridx] in ['validate', 'test']
            fact_based_report = reports[ridx]['fact_based_report']
            query = f"#F {fact_based_report} | #H {fact}"
            query_hash = hash_string(query)
            if query_hash not in already_processed_queries:
                queries.append(query)
                already_processed_queries.add(query_hash)
                cnt += 1
            if cnt == n_pos_queries_per_label:
                break
        if cnt < n_pos_queries_per_label:
            logger.info(f"Only {cnt}/{n_pos_queries_per_label} positive queries for label {fact}")
        cnt -= n_pos_queries_per_label # transfer deficit to negative queries
        while cnt < n_neg_queries_per_label:
            ridx = random.choice(all_report_idxs)
            if ridx not in ridxs:
                fact_based_report = reports[ridx]['fact_based_report']
                query = f"#F {fact_based_report} | #H {fact}"
                query_hash = hash_string(query)
                if query_hash not in already_processed_queries:
                    queries.append(query)
                    already_processed_queries.add(query_hash)
                    cnt += 1

    return queries

def _max_sim(mat, vec):
    sim = np.dot(mat, vec)
    return np.max(sim)

def sample_queries_fact_based(num_samples, integrated_report_facts_metadata_jsonl_filepath,
                              cxr_bert_model_name, cxr_bert_checkpoint_folder_path,
                              batch_size, num_workers, num_clusters, num_iterations,
                              already_processed_queries, use_mimiccxr_dev_test_sets_only=False):

    """
    Sample queries for the fact-based data mode.
    """

    logger.info(f"Sampling {num_samples} queries for the fact-based data mode")

    if use_mimiccxr_dev_test_sets_only:
        logger.info("Using MIMIC-CXR dev and test sets only")
        metadata = load_mimiccxr_reports_detailed_metadata()
        splits = metadata['splits']
        logger.info(f"len(splits): {len(splits)}")
        
    # Load integrated report facts metadata
    reports = load_jsonl(integrated_report_facts_metadata_jsonl_filepath)
    unique_facts = set()
    logger.info(f"Loaded {len(reports)} reports from {integrated_report_facts_metadata_jsonl_filepath}")
    fact2reportidxs = {}
    for i, row in enumerate(reports):
        facts = row['facts']
        unique_facts.update(facts)
        for fact in facts:
            if fact not in fact2reportidxs:
                fact2reportidxs[fact] = []
            fact2reportidxs[fact].append(i)

    unique_facts = list(unique_facts)
    unique_facts.sort()
    logger.info(f"len(unique_facts): {len(unique_facts)}")

    fact2idx = {fact: i for i, fact in enumerate(unique_facts)}
    factidx2reportidxs = [fact2reportidxs[fact] for fact in unique_facts]
    reportidx2factidxs = [[fact2idx[fact] for fact in row['facts']] for row in reports]

    if use_mimiccxr_dev_test_sets_only:
        factidx2reportidxs_ = [None] * len(factidx2reportidxs)
        for fidx, ridxs in enumerate(factidx2reportidxs):
            factidx2reportidxs_[fidx] = [ridx for ridx in ridxs if splits[ridx] != 'train']
    else:
        factidx2reportidxs_ = factidx2reportidxs

    # Obtain kmeans cluster labels for facts
    emb_extractor = CachedTextEmbeddingExtractor(
        model_name=cxr_bert_model_name,
        model_checkpoint_folder_path=cxr_bert_checkpoint_folder_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    embeddings = emb_extractor.compute_text_embeddings(unique_facts)
    kmeans_labels = emb_extractor.compute_kmeans_labels(unique_facts, num_clusters=num_clusters,
                                                        num_iterations=num_iterations, embeddings=embeddings)
    assert len(kmeans_labels) == len(unique_facts)
    logger.info(f"len(kmeans_labels): {len(kmeans_labels)}")

    label2idxs = [[] for _ in range(num_clusters)]
    for i, label in enumerate(kmeans_labels):
        label2idxs[label].append(i)

    # Sample queries
    num_samples_per_cluster = math.ceil(num_samples / num_clusters)
    num_pos_samples_per_cluster = math.ceil(num_samples_per_cluster * 0.5)
    num_neg_samples_per_cluster = num_samples_per_cluster - num_pos_samples_per_cluster
    queries = []

    logger.info(f"num_samples_per_cluster: {num_samples_per_cluster}")
    logger.info(f"num_pos_samples_per_cluster: {num_pos_samples_per_cluster}")
    logger.info(f"num_neg_samples_per_cluster: {num_neg_samples_per_cluster}")
    
    for cluster_fact_idxs in tqdm(label2idxs, mininterval=2):
        if use_mimiccxr_dev_test_sets_only:
            cluster_fact_idxs_ = [fidx for fidx in cluster_fact_idxs if factidx2reportidxs_[fidx]]
        else:
            cluster_fact_idxs_ = cluster_fact_idxs
        if len(cluster_fact_idxs_) == 0:
            continue
        
        random.shuffle(cluster_fact_idxs_)
        n_pos = min(num_pos_samples_per_cluster, len(cluster_fact_idxs_))
        n_pos_per_report = math.ceil(n_pos / len(cluster_fact_idxs_))
        for i in range(n_pos):
            fact_idx = cluster_fact_idxs_[i]
            report_idxs = factidx2reportidxs_[fact_idx]
            for _ in range(n_pos_per_report):
                tries = 0
                while tries < 10:
                    tries += 1
                    report_idx = random.choice(report_idxs)
                    fact_based_report = reports[report_idx]['fact_based_report']
                    assert fact_idx in reportidx2factidxs[report_idx]
                    sampled_fact_idxs = random.sample(cluster_fact_idxs, min(10, len(cluster_fact_idxs)))
                    max_sim = -1
                    best_fact_idx = -1
                    report_mat = embeddings[reportidx2factidxs[report_idx]]
                    for sampled_fact_idx in sampled_fact_idxs:
                        if sampled_fact_idx in reportidx2factidxs[report_idx]:
                            continue
                        sim = _max_sim(report_mat, embeddings[sampled_fact_idx])
                        if sim > max_sim:
                            max_sim = sim
                            best_fact_idx = sampled_fact_idx
                    query = f"#F {fact_based_report} | #H {unique_facts[best_fact_idx]}"
                    query_hash = hash_string(query)
                    if query_hash in already_processed_queries:
                        continue
                    if use_mimiccxr_dev_test_sets_only:
                        assert splits[report_idx] in ['validate', 'test']
                    queries.append(query)
                    already_processed_queries.add(query_hash)
                    break
        n_neg = min(num_neg_samples_per_cluster, len(cluster_fact_idxs_))
        n_neg_per_report = math.ceil(n_neg / len(cluster_fact_idxs_))
        for i in range(n_neg):
            fact_idx = cluster_fact_idxs_[i]
            report_idxs = factidx2reportidxs_[fact_idx]
            for _ in range(n_neg_per_report):
                tries = 0
                while tries < 10:
                    tries += 1
                    report_idx = random.choice(report_idxs)
                    fact_based_report = reports[report_idx]['fact_based_report']
                    if random.random() < 0.5: # Randomly sample from the same cluster
                        other_fact_idx = random.choice(cluster_fact_idxs)
                    else: # Randomly sample from a different cluster
                        random_cluster = random.choice(label2idxs)
                        other_fact_idx = random.choice(random_cluster)
                    assert fact_idx in reportidx2factidxs[report_idx]
                    if other_fact_idx in reportidx2factidxs[report_idx]:
                        continue
                    query = f"#F {fact_based_report} | #H {unique_facts[other_fact_idx]}"
                    query_hash = hash_string(query)
                    if query_hash in already_processed_queries:
                        continue
                    if use_mimiccxr_dev_test_sets_only:
                        assert splits[report_idx] in ['validate', 'test']
                    queries.append(query)
                    already_processed_queries.add(query_hash)
                    break

    return queries
   
def _evaluate_queries(query_jsonl_filepaths, report2labels, label_names, alias2labelidx, figsize):

    # Load queries
    label2tp = { label: 0 for label in label_names }
    label2fp = { label: 0 for label in label_names }
    label2tn = { label: 0 for label in label_names }
    label2fn = { label: 0 for label in label_names }
    label2examples = { label: {'tp': [], 'fp': [], 'tn': [], 'fn': []} for label in label_names }
    key_error_count = 0
    good_queries = 0
    for query_jsonl_filepath in query_jsonl_filepaths:
        queries = load_jsonl(query_jsonl_filepath)
        for query in queries:
            query_text = query['metadata']['query']
            report_start_idx = query_text.index("#F") + 3
            report_end_idx = query_text.index(" | #H")
            fact_based_report = query_text[report_start_idx:report_end_idx]
            hypothesis = query_text[report_end_idx+6:]
            try:
                labels = report2labels[fact_based_report]
                label_idx = alias2labelidx[hypothesis]
                label = label_names[label_idx]
                pred_value = LABEL_TO_BINARY[query['parsed_response']]
                true_value = labels[label_idx]
                example = {
                    'fact_based_report': fact_based_report,
                    'hypothesis': hypothesis,
                    'label': label,
                    'true_value': true_value,
                    'pred_value': pred_value,
                }
                if true_value == 1:
                    if pred_value == 1:
                        label2tp[label] += 1
                        label2examples[label]['tp'].append(example)
                    else:
                        label2fn[label] += 1
                        label2examples[label]['fn'].append(example)
                else:
                    if pred_value == 1:
                        label2fp[label] += 1
                        label2examples[label]['fp'].append(example)
                    else:
                        label2tn[label] += 1
                        label2examples[label]['tn'].append(example)
                good_queries += 1
            except KeyError:
                key_error_count += 1
                continue
    print(f"key_error_count: {key_error_count}")
    print(f"good_queries: {good_queries}")
    # Precision
    metric_names = [f'{label} (tp: {label2tp[label]}, fp: {label2fp[label]}, tn: {label2tn[label]}, fn: {label2fn[label]})' for label in label_names]
    precs = [label2tp[label] / max(label2tp[label] + label2fp[label], 1) for label in label_names]
    plot_metrics(metric_names=metric_names, metric_values=precs, title="Precision",
                 ylabel="Label", xlabel="Precision", append_average_to_title=True, horizontal=True, sort_metrics=True,
                 show_metrics_above_bars=True, draw_grid=True, figsize=figsize)
    # Recall
    recs = [label2tp[label] / max(label2tp[label] + label2fn[label], 1) for label in label_names]
    plot_metrics(metric_names=metric_names, metric_values=recs, title="Recall",
                    ylabel="Label", xlabel="Recall", append_average_to_title=True, horizontal=True, sort_metrics=True,
                 show_metrics_above_bars=True, draw_grid=True, figsize=figsize)
    # F1
    f1s = [2 * precs[i] * recs[i] / max(precs[i] + recs[i], 1) for i in range(len(label_names))]
    plot_metrics(metric_names=metric_names, metric_values=f1s, title="F1",
                    ylabel="Label", xlabel="F1", append_average_to_title=True, horizontal=True, sort_metrics=True,
                 show_metrics_above_bars=True, draw_grid=True, figsize=figsize)
    # Accuracy
    accs = [(label2tp[label] + label2tn[label]) / max(label2tp[label] + label2tn[label] + label2fp[label] + label2fn[label], 1)\
             for label in label_names]
    plot_metrics(metric_names=metric_names, metric_values=accs, title="Accuracy",
                    ylabel="Label", xlabel="Accuracy", append_average_to_title=True, horizontal=True, sort_metrics=True,
                 show_metrics_above_bars=True, draw_grid=True, figsize=figsize)
    
    # Return examples
    return label2examples
    
def evaluate_queries_mimic_cxr_lt(integrated_report_facts_metadata_jsonl_filepath, query_jsonl_filepaths, figsize=(8, 7)):

    # Load integrated report facts metadata
    reports = load_jsonl(integrated_report_facts_metadata_jsonl_filepath)
    print(f"Loaded {len(reports)} reports from {integrated_report_facts_metadata_jsonl_filepath}")
    study_id_to_report_idx = dict()
    for i, row in enumerate(reports):
        study_id = row['path'].split('/')[-1][1:-4] # Remove s and .txt extension
        study_id = int(study_id) # Convert to int
        assert study_id not in study_id_to_report_idx
        study_id_to_report_idx[study_id] = i

    # Load MIMIC-CXR-LT labels
    df_train = pd.read_csv(MIMIC_CXR_LT_LABELS_TRAIN_CSV_PATH)
    df_dev = pd.read_csv(MIMIC_CXR_LT_LABELS_DEV_CSV_PATH)
    df_test = pd.read_csv(MIMIC_CXR_LT_LABELS_TEST_CSV_PATH)
    print(f"len(df_train): {len(df_train)}")
    print(f"len(df_dev): {len(df_dev)}")
    print(f"len(df_test): {len(df_test)}")

    report2labels = {}    
    for df in [df_train, df_dev, df_test]:
        labels = df[MIMIC_CXR_LT_LABELS].values
        for i, study_id in enumerate(df['study_id']):
            report_idx = study_id_to_report_idx[study_id]
            report = reports[report_idx]['fact_based_report']
            report2labels[report] = labels[i]

    print(f"len(report2labels): {len(report2labels)}")

    mcxrlt_label2alias = { label: f'{label.lower()} seen' for label in MIMIC_CXR_LT_LABELS }
    mcxrlt_label2alias['Pleural Other'] = 'pleural abnormalities seen'
    mcxrlt_label2alias['No Finding'] = 'no abnormalities seen'
    mcxrlt_alias2labelidx = { v: MIMIC_CXR_LT_LABELS.index(k) for k, v in mcxrlt_label2alias.items() }
    print(mcxrlt_alias2labelidx)
    assert len(mcxrlt_label2alias) == len(MIMIC_CXR_LT_LABELS)
    assert len(mcxrlt_alias2labelidx) == len(MIMIC_CXR_LT_LABELS)

    return _evaluate_queries(query_jsonl_filepaths, report2labels, MIMIC_CXR_LT_LABELS, mcxrlt_alias2labelidx, figsize)

def evaluate_queries_chest_imagenome(integrated_report_facts_metadata_jsonl_filepath, query_jsonl_filepaths,
                                     chest_imagenome_label_names_filepath, chest_imagenome_image_id_to_labels_filepath, figsize=(8, 14)):

    # Load integrated report facts metadata
    reports = load_jsonl(integrated_report_facts_metadata_jsonl_filepath)
    print(f"Loaded {len(reports)} reports from {integrated_report_facts_metadata_jsonl_filepath}")
    study_id_to_report_idx = dict()
    for i, row in enumerate(reports):
        study_id = row['path'].split('/')[-1][1:-4] # Remove s and .txt extension
        study_id = int(study_id) # Convert to int
        assert study_id not in study_id_to_report_idx
        study_id_to_report_idx[study_id] = i
    
    # Load chest imagenome labels
    ci_label_names = load_pickle(chest_imagenome_label_names_filepath)
    ci_image_id_to_labels = load_pickle(chest_imagenome_image_id_to_labels_filepath)
    print(f"Loaded {len(ci_label_names)} label names from {chest_imagenome_label_names_filepath}")
    print(f"Loaded {len(ci_image_id_to_labels)} image labels from {chest_imagenome_image_id_to_labels_filepath}")

    report2labels = {}
    image_id_to_report_idx = get_imageId2reportId()
    valid_label_idxs = [i for i, label in enumerate(ci_label_names) if len(label) == 2\
                         and label[0] not in ["laterality", "severity", "nlp", "temporal"]]
    ci_label_names = [ci_label_names[i] for i in valid_label_idxs]
    for i, (image_id, labels) in enumerate(ci_image_id_to_labels.items()):
        if image_id in image_id_to_report_idx:
            report_idx = image_id_to_report_idx[image_id]
            report = reports[report_idx]['fact_based_report']
            report2labels[report] = labels[valid_label_idxs]

    print(f"len(report2labels): {len(report2labels)}")

    ci_label2alias = {}
    for label in ci_label_names:
        if label[0] == 'texture':
            alias = f'{label[1]} texture'
        else:
            alias = label[1]
        alias = f'{alias} seen'
        ci_label2alias[label] = alias
    ci_alias2labelidx = { v: ci_label_names.index(k) for k, v in ci_label2alias.items() }
    print(ci_alias2labelidx)
    assert len(ci_label2alias) == len(ci_label_names)
    assert len(ci_alias2labelidx) == len(ci_label_names)

    return _evaluate_queries(query_jsonl_filepaths, report2labels, ci_label_names, ci_alias2labelidx, figsize)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)

    parser.add_argument("--data_mode", type=str, required=True, choices=["label_based", "fact_based"])
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--integrated_report_facts_metadata_jsonl_filepath", type=str, required=True)
    parser.add_argument("--chest_imagenome_image_id_to_labels_filepath", type=str, default=None)
    parser.add_argument("--chest_imagenome_label_names_filepath", type=str, default=None)
    parser.add_argument("--allowed_label_based_facts", type=str, nargs="+", default=None)

    parser.add_argument("--queries_to_skip_filepaths", type=str, nargs="+", default=None)
    parser.add_argument("--use_mimiccxr_dev_test_sets_only", action="store_true", default=False)

    parser.add_argument("--cxr_bert_model_name", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--cxr_bert_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_clusters", type=int, default=400)
    parser.add_argument("--num_iterations", type=int, default=300)

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    parser.add_argument("--not_delete_api_requests_and_responses", action="store_true", default=False)
    args = parser.parse_args()

    processed_queries_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_fact_based_report_nli_queries{args.alias}.jsonl")
    
    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None:

        # Load already processed queries if they exist
        already_processed = set()
        if os.path.exists(processed_queries_save_filepath):
            rows = load_jsonl(processed_queries_save_filepath)
            for row in rows:
                already_processed.add(hash_string(row['metadata']['query']))
            logger.info(f"Loaded {len(rows)} already processed queries from {processed_queries_save_filepath}")

        # Load queries to skip
        if args.queries_to_skip_filepaths is not None:
            for queries_to_skip_filepath in args.queries_to_skip_filepaths:
                rows = load_jsonl(queries_to_skip_filepath)
                for row in rows:
                    already_processed.add(hash_string(row['metadata']['query']))
                logger.info(f"Loaded {len(rows)} queries to skip from {queries_to_skip_filepath}")

        # Sample queries
        if args.data_mode == "label_based":
            assert args.chest_imagenome_image_id_to_labels_filepath is not None
            assert args.chest_imagenome_label_names_filepath is not None
            queries = sample_queries_label_based(
                num_samples=args.num_samples,
                integrated_report_facts_metadata_jsonl_filepath=args.integrated_report_facts_metadata_jsonl_filepath,
                chest_imagenome_image_id_to_labels_filepath=args.chest_imagenome_image_id_to_labels_filepath,
                chest_imagenome_label_names_filepath=args.chest_imagenome_label_names_filepath,
                already_processed_queries=already_processed,
                allowed_label_based_facts=args.allowed_label_based_facts,
                use_mimiccxr_dev_test_sets_only=args.use_mimiccxr_dev_test_sets_only,
            )
        elif args.data_mode == "fact_based":
            assert args.cxr_bert_checkpoint_folder_path is not None
            queries = sample_queries_fact_based(
                num_samples=args.num_samples,
                integrated_report_facts_metadata_jsonl_filepath=args.integrated_report_facts_metadata_jsonl_filepath,
                cxr_bert_model_name=args.cxr_bert_model_name,
                cxr_bert_checkpoint_folder_path=args.cxr_bert_checkpoint_folder_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_clusters=args.num_clusters,
                num_iterations=args.num_iterations,
                already_processed_queries=already_processed,
                use_mimiccxr_dev_test_sets_only=args.use_mimiccxr_dev_test_sets_only,
            )
        
        queries_to_process = queries

        logger.info(f"Total number of queries to process: {len(queries_to_process)}")

        # Print example queries
        logger.info(f"Example queries to process:")
        for i in np.linspace(0, len(queries_to_process)-1, min(20, len(queries_to_process)), dtype=int):
            logger.info(f"{i+1}. {queries_to_process[i]}")

    else:
        assert os.path.exists(args.api_responses_filepath)
        queries_to_process = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=queries_to_process,
        system_instructions=INSTRUCTIONS,
        api_key_name=args.api_key_name,
        openai_model_name=args.openai_model_name,
        openai_request_url=args.openai_request_url,
        max_tokens_per_request=args.max_tokens_per_request,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        temperature=args.temperature,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logger=logger,
        logging_level=args.logging_level,
        parse_openai_output=parse_openai_model_output,
        tmp_dir=MIMICCXR_FAST_TMP_DIR,
        save_filepath=processed_queries_save_filepath,
        delete_api_requests_and_responses=not args.not_delete_api_requests_and_responses,
    )