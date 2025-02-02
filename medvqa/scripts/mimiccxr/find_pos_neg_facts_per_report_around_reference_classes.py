import argparse
import math
import random
import numpy as np
import multiprocessing as mp
import time
from multiprocessing import Pool
from tqdm import tqdm
from medvqa.datasets.mimiccxr import load_mimiccxr_reports_detailed_metadata
from medvqa.datasets.text_data_utils import sentence_tokenize_texts_in_parallel
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.scripts.mimiccxr.find_pos_neg_neutral_facts_per_report_v2 import _compute_mlp_fact_based_nli_softmaxes_per_report
from medvqa.utils.constants import VINBIG_LABELS
from medvqa.utils.files import get_file_path_with_hashing_if_too_long, load_class_specific_regex, load_json, load_jsonl, load_pickle, save_pickle
from medvqa.utils.common import LARGE_FAST_CACHE_DIR
from medvqa.utils.logging import print_blue, print_red

class _Task:
    FIND_FACTS_RELEVANT_TO_ANCHOR_FACTS = 'find_facts_relevant_to_anchor_facts'
    FIND_POSITIVE_NEGATIVE_FACTS_PER_REPORT_WITH_FACT_EMBEDDINGS_AND_MLP_NLI = 'find_positive_negative_facts_per_report_with_fact_embeddings_and_mlp_nli'
    
    @staticmethod
    def choices():
        return [
            _Task.FIND_FACTS_RELEVANT_TO_ANCHOR_FACTS,
            _Task.FIND_POSITIVE_NEGATIVE_FACTS_PER_REPORT_WITH_FACT_EMBEDDINGS_AND_MLP_NLI,
        ]
    
def _filter_facts(integrated_fact_metadata):
    facts = []
    skipped = 0
    f2gf = { x['fact'] : x['metadata']['general_observation'] for x in integrated_fact_metadata }
    for x in integrated_fact_metadata:
        fact = x['fact']
        metadata = x['metadata']
        too_noisy_or_irrelevant = metadata['too_noisy_or_irrelevant']
        if too_noisy_or_irrelevant == 'yes':
            skipped += 1
            continue
        general_observation = metadata['general_observation']
        if general_observation == 'does not apply':
            skipped += 1
            continue
        visually_observable = metadata['visually_observable']
        abnormality_status = metadata['abnormality_status']
        if visually_observable == 'no':
            if abnormality_status not in ('major abnormality', 'partial or minor abnormality'):
                skipped += 1
                continue
        category = metadata['category']
        if category == 'does not apply':
            skipped += 1
            continue
        if category in ('anatomical finding', 'disease', 'foreign body'):
            if abnormality_status == 'completely normal or healthy':
                skipped += 1
                continue
        if category == 'symptom':
            if abnormality_status not in ('major abnormality', 'partial or minor abnormality'):
                skipped += 1
                continue
        anatomical_location = metadata['anatomical_location']
        if category == 'procedure':
            if anatomical_location == 'does not apply':
                skipped += 1
                continue
        if category == 'technical assessment':
            skipped += 1
            continue
        facts.append(fact)
    unique_specific_facts = set(facts)
    banned_facts = set(x['fact'] for x in integrated_fact_metadata if x['fact'] not in unique_specific_facts)
    print(f'len(banned_facts): {len(banned_facts)}')
    unique_general_facts = set()
    for f in unique_specific_facts:
        gf = f2gf[f]
        if gf != 'does not apply' and gf not in banned_facts:
            unique_general_facts.add(gf)

    unique_specific_facts = list(unique_specific_facts)
    unique_specific_facts.sort()
    unique_general_facts = list(unique_general_facts)
    unique_general_facts.sort()
    print(f'Loaded {len(unique_specific_facts)} unique facts (skipped {skipped} facts)')
    print(f'Loaded {len(unique_general_facts)} unique general facts')

    return unique_specific_facts, unique_general_facts, f2gf, banned_facts

    
def _find_relevant_facts_with_respect_to_anchor_facts(
    integrated_fact_metadata_jsonl_filepath: str,
    integrated_sentence_to_negative_facts_jsonl_filepath: str,
    anchor_facts: list,
    anchor_fact_to_regex: dict,
):
    """
    Find facts relevant to the anchor facts.

    Args:
        integrated_fact_metadata_jsonl_filepath (str): Path to the integrated fact metadata JSONL file.
        integrated_sentence_to_negative_facts_jsonl_filepath (str): Path to the integrated sentence to negative facts JSONL file.
        anchor_facts (list): List of anchor facts.
        anchor_fact_to_regex (dict): Regular expression per anchor fact.
    """

    print_blue(f'Running _find_relevant_facts_with_respect_to_anchor_facts()...', bold=True)
    print(f'len(anchor_facts): {len(anchor_facts)}')

    assert len(anchor_facts) == len(anchor_fact_to_regex)

    print(f'Loading {integrated_fact_metadata_jsonl_filepath}...')
    integrated_fact_metadata = load_jsonl(integrated_fact_metadata_jsonl_filepath)
    
    # Pre-filter facts heuristically based on the associated metadata
    unique_specific_facts, unique_general_facts, f2gf, banned_facts = _filter_facts(integrated_fact_metadata)

    # Load integrated sentence to negative facts
    print(f'Loading {integrated_sentence_to_negative_facts_jsonl_filepath}...')
    integrated_sentence_to_negative_facts = load_jsonl(integrated_sentence_to_negative_facts_jsonl_filepath)
    unique_negative_facts = set()
    for x in integrated_sentence_to_negative_facts:
        for facts in x['ruled_out_abnormalities'].values():
            unique_negative_facts.update(facts)
    print(f'len(unique_negative_facts): {len(unique_negative_facts)}')
    unique_negative_facts -= banned_facts # remove banned facts
    print(f'len(unique_negative_facts): {len(unique_negative_facts)} (after removing banned facts)')

    # Collect all facts
    all_facts = set()
    all_facts.update(unique_specific_facts)
    all_facts.update(unique_general_facts)
    all_facts.update(unique_negative_facts)
    all_facts -= banned_facts # remove banned facts again (just in case)
    all_facts = list(all_facts)
    all_facts.sort()

    # Match anchor facts to all facts
    anchors_per_fact = [[] for _ in range(len(all_facts))]
    for i, anchor in enumerate(anchor_facts):
        print(f'Matching anchor fact {i+1}/{len(anchor_facts)}: {anchor}...')
        regex = anchor_fact_to_regex[anchor]
        count = 0
        for j, fact in tqdm(enumerate(all_facts), total=len(all_facts), mininterval=2):
            if regex.search(fact):
                anchors_per_fact[j].append(i)
                count += 1
        print(f'\tMatched {count} facts')

    # Keep only facts with at least one anchor
    idxs = [i for i, anchors in enumerate(anchors_per_fact) if anchors]
    relevant_facts = [all_facts[i] for i in idxs]
    anchors_per_fact = [anchors_per_fact[i] for i in idxs]
    print(f'len(relevant_facts): {len(relevant_facts)}/{len(all_facts)}')

    # Save output
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix=(f'mimiccxr_find_facts_relevant_to_anchor_facts(af={len(anchor_facts)},rf={len(relevant_facts)})'),
        strings=[
            f'len(anchor_facts): {len(anchor_facts)}',
            f'len(relevant_facts): {len(relevant_facts)}',
            integrated_fact_metadata_jsonl_filepath,
            integrated_sentence_to_negative_facts_jsonl_filepath,            
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle({
        'anchor_facts': anchor_facts,
        'relevant_facts': relevant_facts,
        'anchors_per_fact': anchors_per_fact,
    }, output_filepath)


_shared_report_facts = None
_shared_fact2gfact = None
_shared_sentence2idx = None
_shared_relevant_fact_idxs_set = None
_shared_cluster_labels = None
_shared_fidx2anchors = None
_shared_anchor_to_clusters = None
_shared_strong_negative_facts_per_report = None
_shared_max_negative_facts_per_report = None

def _assign_positive_and_negative_facts_to_report(ridx):
    # Get report facts
    report_fidxs = set()
    for f in _shared_report_facts[ridx]['facts']:
        gf = _shared_fact2gfact[f]
        report_fidxs.add(_shared_sentence2idx[f])
        if gf != 'does not apply':
            report_fidxs.add(_shared_sentence2idx[gf])
    
    # Get positive facts for the report
    pos_fidxs = [fidx for fidx in report_fidxs if fidx in _shared_relevant_fact_idxs_set]
    # positive_fact_idxs_per_report[ridx] = pos_fidxs
    
    # Find used clusters and used anchor facts
    used_cluster_ids = set()
    for fidx in report_fidxs:
        used_cluster_ids.add(_shared_cluster_labels[fidx])
    used_anchor_ids = set()
    for fidx in pos_fidxs:
        used_anchor_ids.update(_shared_fidx2anchors[fidx])

    # Collect clusters to sample from
    unused_clusters = []
    for i, (akey, sub_clusters) in enumerate(_shared_anchor_to_clusters):
        if any(a in used_anchor_ids for a in akey): continue # skip used anchor facts
        for j, (c, _) in enumerate(sub_clusters):
            if c not in used_cluster_ids:
                unused_clusters.append((i, j)) # anchor index, cluster index

    # Add strong negative facts
    snfidxs = _shared_strong_negative_facts_per_report[ridx]
    clean_snfidxs = []
    contradiction = False
    for fidx in snfidxs:
        if any(a in used_anchor_ids for a in _shared_fidx2anchors[fidx]): # strong negative fact is from a used anchor
            contradiction = True
            continue
        assert fidx not in report_fidxs # sanity check
        clean_snfidxs.append(fidx)
    # strong_negative_fact_idxs_per_report[ridx] = clean_snfidxs

    # if contradiction:
    #     report_idxs_with_contradictions.append(ridx)
    
    # Sample negative facts from non-used clusters
    num_samples_per_cluster = math.ceil(_shared_max_negative_facts_per_report / len(unused_clusters))
    sampled_idxs = []
    for i, j in unused_clusters:
        cluster = _shared_anchor_to_clusters[i][1][j]
        fidxs = cluster[1]
        if num_samples_per_cluster < len(fidxs):
            sampled_idxs.extend(random.sample(fidxs, num_samples_per_cluster))
        else:
            sampled_idxs.extend(fidxs)

    if len(sampled_idxs) > _shared_max_negative_facts_per_report: # truncate
        random.shuffle(sampled_idxs) # shuffle to avoid bias
        sampled_idxs = sampled_idxs[:_shared_max_negative_facts_per_report]
    elif len(sampled_idxs) < _shared_max_negative_facts_per_report: # pad with random facts
        idxs_to_skip = set(sampled_idxs)
        idxs_to_skip.update(report_fidxs)
        assert len(idxs_to_skip) == len(sampled_idxs) + len(report_fidxs) # sanity check that there are no duplicates
        while len(sampled_idxs) < _shared_max_negative_facts_per_report:
            i, j = random.choice(unused_clusters)
            fidxs = _shared_anchor_to_clusters[i][1][j][1]
            fidx = random.choice(fidxs)
            if fidx in idxs_to_skip: continue
            sampled_idxs.append(fidx)

    assert len(sampled_idxs) == _shared_max_negative_facts_per_report
    # s = ridx * max_negative_facts_per_report
    # e = s + max_negative_facts_per_report
    # cand_neg_ridx_fidx_pairs[s:e, 0] = ridx # report index
    # cand_neg_ridx_fidx_pairs[s:e, 1] = sampled_idxs # fact index

    return (pos_fidxs, clean_snfidxs, sampled_idxs, contradiction) # positive, strong negative, weak negative, contradiction


def _find_positive_negative_facts_per_report_with_fact_embeddings_and_mlp_nli(
    integrated_report_facts_jsonl_filepath: str,
    integrated_fact_metadata_jsonl_filepath: str,
    background_findings_and_impression_json_filepath: str,
    facts_relevant_to_anchor_facts_pickle_filepath: str,
    integrated_sentence_to_negative_facts_jsonl_filepath: str,
    device: str,
    fact_embedding_model_name: str,
    fact_embedding_model_checkpoint_folder_path: str,
    fact_embedding_batch_size: int,
    fact_embedding_num_workers: int,
    skip_nli: bool,
    mlp_batch_size: int,
    mlp_num_workers: int,
    mlp_nli_checkpoint_folder_path: str,
    mlp_nli_entailment_threshold: float,
    num_clusters: int,
    max_negative_facts_per_report: int,
):
    """
    Sample negative facts per report using fact embeddings and a pre-trained MLP NLI model.

    Args:
        integrated_report_facts_jsonl_filepath (str): Path to the integrated report facts JSONL file.
        integrated_fact_metadata_jsonl_filepath (str): Path to the integrated fact metadata JSONL file.
        background_findings_and_impression_json_filepath (str): Path to the background findings and impression JSON file.
        facts_relevant_to_anchor_facts_pickle_filepath (str): Path to the facts relevant to anchor facts pickle file.
        integrated_sentence_to_negative_facts_jsonl_filepath (str): Path to the integrated sentence to negative facts JSONL file.
        device (str): Device to use for computing text embeddings.
        fact_embedding_model_name (str): Name of the fact embedding model.
        fact_embedding_model_checkpoint_folder_path (str): Path to the fact embedding model checkpoint folder.
        fact_embedding_batch_size (int): Batch size for computing text embeddings.
        fact_embedding_num_workers (int): Number of workers for computing text embeddings.
        skip_nli (bool): Whether to skip the NLI step.
        mlp_batch_size (int): Batch size for computing NLI softmaxes.
        mlp_num_workers (int): Number of workers for computing NLI softmaxes.
        mlp_nli_checkpoint_folder_path (str): Path to the pre-trained MLP NLI model checkpoint folder.
        mlp_nli_entailment_threshold (float): Entailment threshold for filtering out negative facts.
        num_clusters (int): Number of clusters for KMeans clustering.
        max_negative_facts_per_report (int): Maximum number of negative facts to sample per report.
    """

    print_blue(f'Running _find_positive_negative_facts_per_report_with_fact_embeddings_and_mlp_nli()...', bold=True)

    # --- Load integrated report facts
    print(f'Reading {integrated_report_facts_jsonl_filepath}...')
    report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    n_reports = len(report_facts)
    print(f'n_reports: {n_reports}')
    all_facts = set()
    for rf in report_facts:
        all_facts.update(rf['facts'])
    print(f'len(all_facts): {len(all_facts)}')

    # --- Load integrated fact metadata
    print(f'Loading {integrated_fact_metadata_jsonl_filepath}...')
    integrated_fact_metadata = load_jsonl(integrated_fact_metadata_jsonl_filepath)
    fact2gfact = { x['fact'] : x['metadata']['general_observation'] for x in integrated_fact_metadata }
    for f, gf in fact2gfact.items():
        all_facts.add(f)
        if gf != 'does not apply':
            all_facts.add(gf)
    print(f'len(all_facts): {len(all_facts)}')

    # --- Load facts relevant to anchor facts
    print(f'Loading {facts_relevant_to_anchor_facts_pickle_filepath}...')
    tmp = load_pickle(facts_relevant_to_anchor_facts_pickle_filepath)
    relevant_facts = tmp['relevant_facts']
    anchor_facts = tmp['anchor_facts']
    anchors_per_fact = tmp['anchors_per_fact']
    print(f'len(relevant_facts): {len(relevant_facts)}')
    print(f'len(anchor_facts): {len(anchor_facts)}')
    print(f'len(anchors_per_fact): {len(anchors_per_fact)}')
    
    all_facts.update(relevant_facts)
    all_facts = list(all_facts)
    all_facts.sort(key=lambda x: (len(x), x)) # sort by length and then alphabetically
    sentence2idx = {s: i for i, s in enumerate(all_facts)}
    print(f'len(all_facts): {len(all_facts)}')

    relevant_fact_idxs = [sentence2idx[f] for f in relevant_facts]
    relevant_fact_idxs_set = set(relevant_fact_idxs)

    # --- Assign relevant facts to anchor keys
    anchor_to_relevant_facts = dict()
    fidx2anchors = [None] * len(all_facts)
    for fidx, anchors in zip(relevant_fact_idxs, anchors_per_fact):
        key = sorted(anchors)
        key = tuple(key)
        if key not in anchor_to_relevant_facts:
            anchor_to_relevant_facts[key] = []
        anchor_to_relevant_facts[key].append(fidx)
        fidx2anchors[fidx] = key
    print(f'len(anchor_to_relevant_facts): {len(anchor_to_relevant_facts)}')
    min_size = min(len(fidxs) for fidxs in anchor_to_relevant_facts.values())
    max_size = max(len(fidxs) for fidxs in anchor_to_relevant_facts.values())
    avg_size = np.mean([len(fidxs) for fidxs in anchor_to_relevant_facts.values()])
    print(f'Min size: {min_size}, Max size: {max_size}, Avg size: {avg_size:.2f}')

    # --- Pre-assign strong negative facts to reports
    print(f'Loading {background_findings_and_impression_json_filepath}...')
    preprocessed_reports = load_json(background_findings_and_impression_json_filepath)
    assert len(preprocessed_reports) == n_reports
    # Load integrated sentence to negative facts
    print(f'Loading {integrated_sentence_to_negative_facts_jsonl_filepath}...')
    integrated_sentence_to_negative_facts = load_jsonl(integrated_sentence_to_negative_facts_jsonl_filepath)
    s2nfs = { x['sentence'] : x['ruled_out_abnormalities'] for x in integrated_sentence_to_negative_facts }
    
    strong_negative_facts_per_report = [set() for _ in range(n_reports)]

    texts = []
    ranges = []
    for row in preprocessed_reports:
        offset = len(texts)
        if row['findings']:
            texts.append(row['findings'])
        if row['impression']:
            texts.append(row['impression'])
        ranges.append((offset, len(texts)))
    sentences_per_text = sentence_tokenize_texts_in_parallel(texts)    
    
    for ridx, r in enumerate(ranges):
        for i in range(r[0], r[1]):
            for s in sentences_per_text[i]:
                nfs = s2nfs[s]
                for v in nfs.values():
                    assert isinstance(v, list)
                    for f in v:
                        if f not in sentence2idx: continue
                        fidx = sentence2idx[f]
                        if fidx in relevant_fact_idxs_set:
                            strong_negative_facts_per_report[ridx].add(fidx)

    strong_negative_facts_per_report = [list(x) for x in strong_negative_facts_per_report] # convert to list

    # Print statistics
    avg_num_negative_facts_per_report = np.mean([len(fidxs) for fidxs in strong_negative_facts_per_report])
    print(f'Average number of strong negative facts per report: {avg_num_negative_facts_per_report:.2f}')
    total_num_negative_facts = sum(len(fidxs) for fidxs in strong_negative_facts_per_report)
    print(f'Total number of strong negative facts: {total_num_negative_facts}')
    
    # --- Extract embeddings
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=device,
    )
    all_fact_embeddings = embedding_extractor.compute_text_embeddings(all_facts)
    relevant_fact_embeddings = embedding_extractor.compute_text_embeddings(relevant_facts)
    print(f'all_fact_embeddings.shape: {all_fact_embeddings.shape}')
    print(f'relevant_fact_embeddings.shape: {relevant_fact_embeddings.shape}')

    # --- Cluster embeddings
    print("Clustering embeddings...")
    cluster_labels = embedding_extractor.compute_kmeans_labels(
        texts=all_facts, num_clusters=num_clusters, embeddings=all_fact_embeddings)

    # Partition anchor groups into clusters
    anchor_to_clusters = []
    for a, fidxs in anchor_to_relevant_facts.items():
        sub_clusters = [[c, []] for c in range(num_clusters)]
        for fidx in fidxs:
            c = cluster_labels[fidx]
            sub_clusters[c][1].append(fidx)
        sub_clusters = [x for x in sub_clusters if x[1]] # remove empty clusters
        anchor_to_clusters.append((a, sub_clusters)) # anchor, sub_clusters
        min_cluster_size = min(len(x[1]) for x in sub_clusters)
        max_cluster_size = max(len(x[1]) for x in sub_clusters)
        avg_cluster_size = np.mean([len(x[1]) for x in sub_clusters])
        
        print(f'Anchor {a}: {len(fidxs)} facts, {len(sub_clusters)} non-empty clusters, '
                f'min={min_cluster_size}, max={max_cluster_size}, avg={avg_cluster_size:.2f}')

    # --- Assign positive and negative facts to reports
    print("Assigning positive and negative facts to reports with multiprocessing...")
    
    positive_fact_idxs_per_report = [None] * n_reports
    strong_negative_fact_idxs_per_report = [None] * n_reports
    cand_neg_ridx_fidx_pairs = np.empty((n_reports * max_negative_facts_per_report, 2), dtype=np.int32)

    report_idxs_with_contradictions = []

    global _shared_report_facts
    global _shared_fact2gfact
    global _shared_sentence2idx
    global _shared_relevant_fact_idxs_set
    global _shared_cluster_labels
    global _shared_fidx2anchors
    global _shared_anchor_to_clusters
    global _shared_strong_negative_facts_per_report
    global _shared_max_negative_facts_per_report
    _shared_report_facts = report_facts
    _shared_fact2gfact = fact2gfact
    _shared_sentence2idx = sentence2idx
    _shared_relevant_fact_idxs_set = relevant_fact_idxs_set
    _shared_cluster_labels = cluster_labels
    _shared_fidx2anchors = fidx2anchors
    _shared_anchor_to_clusters = anchor_to_clusters
    _shared_strong_negative_facts_per_report = strong_negative_facts_per_report
    _shared_max_negative_facts_per_report = max_negative_facts_per_report
    
    start_time = time.time()
    with Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(_assign_positive_and_negative_facts_to_report, range(n_reports))
    print(f'Elapsed time: {time.time() - start_time:.2f}s')
    
    for ridx, (pos_fidxs, snfidxs, wnfidxs, contradiction) in tqdm(enumerate(results), total=n_reports, mininterval=2):
        positive_fact_idxs_per_report[ridx] = pos_fidxs
        strong_negative_fact_idxs_per_report[ridx] = snfidxs
        if contradiction:
            report_idxs_with_contradictions.append(ridx)
        s = ridx * max_negative_facts_per_report
        e = s + max_negative_facts_per_report
        cand_neg_ridx_fidx_pairs[s:e, 0] = ridx # report index
        cand_neg_ridx_fidx_pairs[s:e, 1] = wnfidxs # fact index

    if report_idxs_with_contradictions:
        print_red(f'WARNING: {len(report_idxs_with_contradictions)}/{n_reports} reports have contradictions!', bold=True)

    if not skip_nli:

        # Compute softmaxes over candidate weak negative facts
        print("Computing softmaxes over candidate weak negative facts...")
        tmp = _compute_mlp_fact_based_nli_softmaxes_per_report(
            embeddings=all_fact_embeddings,
            sentence2idx=sentence2idx,
            report_facts=report_facts,
            ridx_fidx_pairs=cand_neg_ridx_fidx_pairs,
            representative_facts=all_facts,
            mlp_batch_size=mlp_batch_size,
            mlp_num_workers=mlp_num_workers,
            mlp_nli_checkpoint_folder_path=mlp_nli_checkpoint_folder_path,
            device=device,
        )
        softmaxes = tmp['softmaxes']
        assert softmaxes.shape[0] == cand_neg_ridx_fidx_pairs.shape[0]

        # Filter out negative facts with high entailment softmaxes
        print("Filtering out negative facts with high entailment softmaxes...")
        valid_idxs = np.where(softmaxes[:, 0] < mlp_nli_entailment_threshold)[0]
        weak_negative_fact_idxs_per_report = [[] for _ in range(n_reports)]
        for i in valid_idxs:
            ridx, fidx = cand_neg_ridx_fidx_pairs[i]
            weak_negative_fact_idxs_per_report[ridx].append(fidx.item()) # convert to int

        # Print statistics
        print(f'Number of valid weak negative facts: {len(valid_idxs)}/{len(softmaxes)}')
        print(f'Percentage of valid weak negative facts: {len(valid_idxs) / len(softmaxes) * 100:.2f}%')
        # average number of negative facts per report
        avg_num_negative_facts_per_report = np.mean([len(fidxs) for fidxs in weak_negative_fact_idxs_per_report])
        print(f'Average number of weak negative facts per report: {avg_num_negative_facts_per_report:.2f}')
    
    else:
        # Skip NLI
        weak_negative_fact_idxs_per_report = cand_neg_ridx_fidx_pairs[:, 1].reshape((n_reports, max_negative_facts_per_report)).tolist()

    # Build dicom_id to pos/neg facts
    detailed_metadata = load_mimiccxr_reports_detailed_metadata()
    dicom_id_view_pos_pairs = detailed_metadata['dicom_id_view_pos_pairs']
    assert len(dicom_id_view_pos_pairs) == n_reports
    dicom_id_to_pos_facts = {}
    dicom_id_to_strong_neg_facts = {}
    dicom_id_to_weak_neg_facts = {}
    for ridx, pairs in enumerate(dicom_id_view_pos_pairs):
        for dicom_id, _ in pairs:
            dicom_id_to_pos_facts[dicom_id] = positive_fact_idxs_per_report[ridx]
            dicom_id_to_strong_neg_facts[dicom_id] = strong_negative_fact_idxs_per_report[ridx]
            dicom_id_to_weak_neg_facts[dicom_id] = weak_negative_fact_idxs_per_report[ridx]

    # Save output
    output = {
        'facts': all_facts,
        'embeddings': all_fact_embeddings,
        'dicom_id_to_pos_facts': dicom_id_to_pos_facts,
        'dicom_id_to_strong_neg_facts': dicom_id_to_strong_neg_facts,
        'dicom_id_to_weak_neg_facts': dicom_id_to_weak_neg_facts,
    }
    if report_idxs_with_contradictions:
        output['report_idxs_with_contradictions'] = report_idxs_with_contradictions
    strings=[
        f'len(cand_neg_ridx_fidx_pairs): {len(cand_neg_ridx_fidx_pairs)}',
        f'max_negative_facts_per_report={max_negative_facts_per_report}',
        integrated_report_facts_jsonl_filepath,
        integrated_fact_metadata_jsonl_filepath,
        background_findings_and_impression_json_filepath,
        facts_relevant_to_anchor_facts_pickle_filepath,
        fact_embedding_model_name,
        fact_embedding_model_checkpoint_folder_path,
    ]
    if not skip_nli:
        strings.extend([
            f'len(valid_idxs): {len(valid_idxs)}',
            f'mlp_nli_checkpoint_folder_path={mlp_nli_checkpoint_folder_path}',
            f'mlp_nli_entailment_threshold={mlp_nli_entailment_threshold}',
        ])
        prefix = (f'mimiccxr_dicom_id_to_pos_neg_facts(num_rel_facts={len(relevant_facts)},num_clusters={num_clusters},'
                f'max_neg={max_negative_facts_per_report},ent_th={mlp_nli_entailment_threshold:.3f})')
    else:
        prefix = (f'mimiccxr_dicom_id_to_pos_neg_facts(num_rel_facts={len(relevant_facts)},num_clusters={num_clusters},'
                f'max_neg={max_negative_facts_per_report},skip_nli)')
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix=prefix,
        strings=strings,
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=_Task.choices())
    parser.add_argument('--integrated_fact_metadata_jsonl_filepath', type=str)
    parser.add_argument('--integrated_report_facts_jsonl_filepath', type=str)
    parser.add_argument('--integrated_sentence_to_negative_facts_jsonl_filepath', type=str)
    parser.add_argument('--background_findings_and_impression_json_filepath', type=str)
    parser.add_argument('--fact_embedding_model_name', type=str)
    parser.add_argument('--fact_embedding_model_checkpoint_folder_path', type=str)
    parser.add_argument('--fact_embedding_batch_size', type=int, default=32)
    parser.add_argument('--fact_embedding_num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--facts_relevant_to_anchor_facts_pickle_filepath', type=str)
    parser.add_argument('--skip_nli', action='store_true')
    parser.add_argument('--mlp_batch_size', type=int, default=32)
    parser.add_argument('--mlp_num_workers', type=int, default=4)
    parser.add_argument('--mlp_nli_checkpoint_folder_path', type=str)
    parser.add_argument('--mlp_nli_entailment_threshold', type=float, default=0.5)
    parser.add_argument('--num_clusters', type=int)
    parser.add_argument('--max_negative_facts_per_report', type=int)
    args = parser.parse_args()

    if args.task == _Task.FIND_FACTS_RELEVANT_TO_ANCHOR_FACTS:
        assert args.integrated_fact_metadata_jsonl_filepath is not None
        assert args.integrated_sentence_to_negative_facts_jsonl_filepath is not None
        dataset_name = 'VinDr-CXR' # TODO: support other datasets
        class_name_to_regex = load_class_specific_regex(dataset_name)
        anchor_facts = list(class_name_to_regex.keys())
        anchor_facts.sort()
        assert all(af in VINBIG_LABELS for af in anchor_facts)
        _find_relevant_facts_with_respect_to_anchor_facts(
            integrated_fact_metadata_jsonl_filepath=args.integrated_fact_metadata_jsonl_filepath,
            integrated_sentence_to_negative_facts_jsonl_filepath=args.integrated_sentence_to_negative_facts_jsonl_filepath,
            anchor_facts=anchor_facts,
            anchor_fact_to_regex=class_name_to_regex,
        )
    elif args.task == _Task.FIND_POSITIVE_NEGATIVE_FACTS_PER_REPORT_WITH_FACT_EMBEDDINGS_AND_MLP_NLI:
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.integrated_fact_metadata_jsonl_filepath is not None
        assert args.background_findings_and_impression_json_filepath is not None
        assert args.facts_relevant_to_anchor_facts_pickle_filepath is not None
        assert args.integrated_sentence_to_negative_facts_jsonl_filepath is not None
        assert args.fact_embedding_model_name is not None
        assert args.fact_embedding_model_checkpoint_folder_path is not None
        assert args.fact_embedding_batch_size is not None
        assert args.fact_embedding_num_workers is not None
        assert args.num_clusters is not None
        assert args.max_negative_facts_per_report is not None
        if not args.skip_nli:
            assert args.mlp_nli_checkpoint_folder_path is not None
            assert args.mlp_batch_size is not None
            assert args.mlp_num_workers is not None
            assert args.mlp_nli_entailment_threshold is not None
        _find_positive_negative_facts_per_report_with_fact_embeddings_and_mlp_nli(
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            integrated_fact_metadata_jsonl_filepath=args.integrated_fact_metadata_jsonl_filepath,
            background_findings_and_impression_json_filepath=args.background_findings_and_impression_json_filepath,
            facts_relevant_to_anchor_facts_pickle_filepath=args.facts_relevant_to_anchor_facts_pickle_filepath,
            integrated_sentence_to_negative_facts_jsonl_filepath=args.integrated_sentence_to_negative_facts_jsonl_filepath,
            device=args.device,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
            skip_nli=args.skip_nli,
            mlp_batch_size=args.mlp_batch_size,
            mlp_num_workers=args.mlp_num_workers,
            mlp_nli_checkpoint_folder_path=args.mlp_nli_checkpoint_folder_path,
            mlp_nli_entailment_threshold=args.mlp_nli_entailment_threshold,
            num_clusters=args.num_clusters,
            max_negative_facts_per_report=args.max_negative_facts_per_report,
        )
    else:
        raise ValueError(f'Invalid task: {args.task}')
    
    print_blue(f'Done!', bold=True)

if __name__ == '__main__':
    main()