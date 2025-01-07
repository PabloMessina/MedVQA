import argparse
import math
import random
import numpy as np
from tqdm import tqdm
from medvqa.datasets.mimiccxr import load_mimiccxr_reports_detailed_metadata
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.scripts.mimiccxr.find_pos_neg_neutral_facts_per_report_v2 import _compute_mlp_fact_based_nli_softmaxes_per_report
from medvqa.utils.constants import MULTIDATASET_UNIFIED_CLASSES
from medvqa.utils.files import get_file_path_with_hashing_if_too_long, load_jsonl, load_pickle, save_pickle
from medvqa.utils.common import LARGE_FAST_CACHE_DIR
from medvqa.utils.logging import print_blue

class _Task:
    FIND_FACTS_SIMILAR_TO_ANCHOR_FACTS = 'find_facts_similar_to_anchor_facts'
    FIND_POSITIVE_NEGATIVE_FACTS_PER_REPORT_WITH_FACT_EMBEDDINGS_AND_MLP_NLI = 'find_positive_negative_facts_per_report_with_fact_embeddings_and_mlp_nli'
    
    @staticmethod
    def choices():
        return [
            _Task.FIND_FACTS_SIMILAR_TO_ANCHOR_FACTS,
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
        if category in ('anatomical finding', 'disease'):
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
    unique_facts = set(facts)
    banned_facts = set(x['fact'] for x in integrated_fact_metadata if x['fact'] not in unique_facts)
    print(f'len(banned_facts): {len(banned_facts)}')
    unique_general_facts = set()
    for f in unique_facts:
        gf = f2gf[f]
        if gf != 'does not apply' and gf not in banned_facts:
            unique_general_facts.add(gf)

    unique_facts = list(unique_facts)
    unique_facts.sort()
    unique_general_facts = list(unique_general_facts)
    unique_general_facts.sort()
    print(f'Loaded {len(unique_facts)} unique facts (skipped {skipped} facts)')
    print(f'Loaded {len(unique_general_facts)} unique general facts')

    return unique_facts, unique_general_facts, f2gf, banned_facts

    
def _find_facts_similar_to_anchor_facts(
    integrated_fact_metadata_jsonl_filepath: str,
    fact_embedding_model_name: str,
    fact_embedding_model_checkpoint_folder_path: str,
    fact_embedding_batch_size: int,
    fact_embedding_num_workers: int,
    device: str,
    anchor_facts: list,
    min_similarity_threshold: float,
):
    """
    Find facts similar to anchor facts using the given fact embedding model.
    The goal is to start from anchor facts curated with domain knowledge and find lots of similar relevant facts
    "in the wild" extracted from reports that can be used to augment the anchor facts.

    Args:
        integrated_fact_metadata_jsonl_filepath (str): Path to the integrated fact metadata JSONL file.
        fact_embedding_model_name (str): Name of the fact embedding model.
        fact_embedding_model_checkpoint_folder_path (str): Path to the fact embedding model checkpoint folder.
        fact_embedding_batch_size (int): Batch size for computing text embeddings.
        fact_embedding_num_workers (int): Number of workers for computing text embeddings.
        device (str): Device to use for computing text embeddings.
        anchor_facts (list): List of anchor facts.
        min_similarity_threshold (float): Minimum similarity threshold for finding similar facts.
    """

    print_blue(f'Running _find_facts_similar_to_anchor_facts()...', bold=True)
    print(f'len(anchor_facts): {len(anchor_facts)}')
    print(f'min_similarity_threshold: {min_similarity_threshold}')
    print(f'Loading {integrated_fact_metadata_jsonl_filepath}...')
    integrated_fact_metadata = load_jsonl(integrated_fact_metadata_jsonl_filepath)
    
    # Pre-filter facts heuristically based on the associated metadata
    unique_facts, unique_general_facts, f2gf, _ = _filter_facts(integrated_fact_metadata)
    uf2idx = {f: i for i, f in enumerate(unique_facts)}
    ugf2idx = {f: i for i, f in enumerate(unique_general_facts)}

    # # Remove noisy facts
    # invalid_regex_path = os.path.join(SOURCE_DIR, 'medvqa', 'datasets', 'regular_expressions', 'hard_to_identify_irrelevant_normal_facts_patterns.txt')
    # invalid_facts_regex = load_regex_from_files(invalid_regex_path)
    # print(f'len(facts) before removing invalid facts: {len(facts)}')
    # facts = [f for f in tqdm(facts, mininterval=2) if not invalid_facts_regex.search(f)]
    # print(f'len(facts) after removing invalid facts: {len(facts)}')

    # Compute fact embeddings
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=device,
    )
    unique_fact_embeddings = embedding_extractor.compute_text_embeddings(unique_facts)
    unique_general_fact_embeddings = embedding_extractor.compute_text_embeddings(unique_general_facts)
    anchor_fact_embeddings = embedding_extractor.compute_text_embeddings(anchor_facts)
    print(f'unique_fact_embeddings.shape: {unique_fact_embeddings.shape}')
    print(f'unique_general_fact_embeddings.shape: {unique_general_fact_embeddings.shape}')
    print(f'anchor_fact_embeddings.shape: {anchor_fact_embeddings.shape}')

    # Find similar facts
    print(f'Finding similar facts...')
    most_similar_to_unique_facts = [None] * len(unique_facts)
    most_similar_to_unique_general_facts = [None] * len(unique_general_facts)

    for i, anchor_fact_embedding in tqdm(enumerate(anchor_fact_embeddings), total=len(anchor_fact_embeddings), mininterval=2):
        similarities = np.dot(unique_fact_embeddings, anchor_fact_embedding)
        idxs = np.where(similarities >= min_similarity_threshold)[0]
        for j in idxs:
            sim = similarities[j]
            if most_similar_to_unique_facts[j] is None or sim > most_similar_to_unique_facts[j][0]:
                most_similar_to_unique_facts[j] = (sim, i)
        similarities = np.dot(unique_general_fact_embeddings, anchor_fact_embedding)
        idxs = np.where(similarities >= min_similarity_threshold)[0]
        for j in idxs:
            sim = similarities[j]
            if most_similar_to_unique_general_facts[j] is None or sim > most_similar_to_unique_general_facts[j][0]:
                most_similar_to_unique_general_facts[j] = (sim, i)

    preserved_facts = []
    most_similar_to_fact = []
    seen_facts = set()
    preserved_general_facts = []
    most_similar_to_general_fact = []
    seen_general_facts = set()
    for f, gf in tqdm(f2gf.items(), total=len(f2gf), mininterval=2):
        try:
            fidx = uf2idx[f]
        except KeyError:
            continue
        gfidx = ugf2idx[gf] if gf in ugf2idx else None
        if (most_similar_to_unique_facts[fidx] is not None or
            (gfidx is not None and most_similar_to_unique_general_facts[gfidx] is not None)):
            if f not in seen_facts:
                seen_facts.add(f)
                preserved_facts.append(f)
                most_similar_to_fact.append(most_similar_to_unique_facts[fidx])
            if gfidx is not None and gf not in seen_general_facts:
                seen_general_facts.add(gf)
                preserved_general_facts.append(gf)
                most_similar_to_general_fact.append(most_similar_to_unique_general_facts[gfidx])

    # Save output
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix=(f'mimiccxr_find_facts_similar_to_anchor_facts(af={len(anchor_facts)},sf={len(preserved_facts)},'
               f'sgf={len(preserved_general_facts)},sim_th={min_similarity_threshold:.3f})'),
        strings=[
            f'len(anchor_facts): {len(anchor_facts)}',
            f'len(preserved_facts): {len(preserved_facts)}',
            f'len(preserved_general_facts): {len(preserved_general_facts)}',
            fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path,
            integrated_fact_metadata_jsonl_filepath,
            f'min_similarity_threshold={min_similarity_threshold}',
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle({
        'anchor_facts': anchor_facts,
        'preserved_general_facts': preserved_general_facts,
        'preserved_facts': preserved_facts,
        'most_similar_to_fact': most_similar_to_fact,
        'most_similar_to_general_fact': most_similar_to_general_fact,
    }, output_filepath)


def _find_positive_negative_facts_per_report_with_fact_embeddings_and_mlp_nli(
    integrated_report_facts_jsonl_filepath: str,
    integrated_fact_metadata_jsonl_filepath: str,
    facts_similar_to_anchor_facts_pickle_filepath: str,
    device: str,
    fact_embedding_model_name: str,
    fact_embedding_model_checkpoint_folder_path: str,
    fact_embedding_batch_size: int,
    fact_embedding_num_workers: int,
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
        facts_similar_to_anchor_facts_pickle_filepath (str): Path to the facts similar to anchor facts pickle file.
        device (str): Device to use for computing text embeddings.
        fact_embedding_model_name (str): Name of the fact embedding model.
        fact_embedding_model_checkpoint_folder_path (str): Path to the fact embedding model checkpoint folder.
        fact_embedding_batch_size (int): Batch size for computing text embeddings.
        fact_embedding_num_workers (int): Number of workers for computing text embeddings.
        mlp_batch_size (int): Batch size for computing NLI softmaxes.
        mlp_num_workers (int): Number of workers for computing NLI softmaxes.
        mlp_nli_checkpoint_folder_path (str): Path to the pre-trained MLP NLI model checkpoint folder.
        mlp_nli_entailment_threshold (float): Entailment threshold for filtering out negative facts.
        num_clusters (int): Number of clusters for KMeans clustering.
        max_negative_facts_per_report (int): Maximum number of negative facts to sample per report.
    """

    print_blue(f'Running _find_positive_negative_facts_per_report_with_fact_embeddings_and_mlp_nli()...', bold=True)

    # Load integrated report facts
    print(f'Reading {integrated_report_facts_jsonl_filepath}...')
    report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    n_reports = len(report_facts)
    print(f'n_reports: {n_reports}')
    all_facts = set()
    for rf in report_facts:
        all_facts.update(rf['facts'])
    print(f'len(all_facts): {len(all_facts)}')

    # Load integrated fact metadata
    print(f'Loading {integrated_fact_metadata_jsonl_filepath}...')
    integrated_fact_metadata = load_jsonl(integrated_fact_metadata_jsonl_filepath)
    fact2gfact = { x['fact'] : x['metadata']['general_observation'] for x in integrated_fact_metadata }
    for f, gf in fact2gfact.items():
        all_facts.add(f)
        if gf != 'does not apply':
            all_facts.add(gf)
    print(f'len(all_facts): {len(all_facts)}')

    # Load facts similar to anchor facts
    print(f'Loading {facts_similar_to_anchor_facts_pickle_filepath}...')
    tmp = load_pickle(facts_similar_to_anchor_facts_pickle_filepath)
    preserved_facts = tmp['preserved_facts']
    preserved_general_facts = tmp['preserved_general_facts']
    print(f'len(preserved_facts): {len(preserved_facts)}')
    print(f'len(preserved_general_facts): {len(preserved_general_facts)}')
    
    all_facts.update(preserved_facts)
    all_facts.update(preserved_general_facts)
    all_facts = list(all_facts)
    all_facts.sort()
    sentence2idx = {s: i for i, s in enumerate(all_facts)}
    print(f'len(all_facts): {len(all_facts)}')

    similar_facts = set()
    similar_facts.update(preserved_facts)
    similar_facts.update(preserved_general_facts)
    similar_facts = list(similar_facts)
    similar_facts.sort()
    similar_fact_idxs = [sentence2idx[f] for f in similar_facts]
    similar_fact_idxs_set = set(similar_fact_idxs)
    
    # Extract embeddings
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=device,
    )
    all_fact_embeddings = embedding_extractor.compute_text_embeddings(all_facts)
    similar_fact_embeddings = embedding_extractor.compute_text_embeddings(similar_facts)
    print(f'all_fact_embeddings.shape: {all_fact_embeddings.shape}')
    print(f'similar_fact_embeddings.shape: {similar_fact_embeddings.shape}')

    # Cluster embeddings
    print("Clustering embeddings...")
    cluster_labels = embedding_extractor.compute_kmeans_labels(
        texts=all_facts, num_clusters=num_clusters, embeddings=all_fact_embeddings)
    cluster2simfidxs = [[] for _ in range(num_clusters)] # cluster to similar fact indices
    for idx in similar_fact_idxs:
        c = cluster_labels[idx]
        cluster2simfidxs[c].append(idx)
    nonempty_clusters = [c for c in range(num_clusters) if cluster2simfidxs[c]]
    print(f'len(cluster2simfidxs): {len(cluster2simfidxs)}')
    print(f'len(nonempty_clusters): {len(nonempty_clusters)}')

    # Assign candidate negative facts to reports randomly
    print("Assigning candidate negative facts to reports...")
    
    positive_fact_idxs_per_report = [None] * n_reports
    cand_neg_ridx_fidx_pairs = np.empty((n_reports * max_negative_facts_per_report, 2), dtype=np.int32)
    
    for ridx in tqdm(range(n_reports), total=n_reports, mininterval=3):        
        
        # Get report facts
        report_fidxs = set()
        for f in report_facts[ridx]['facts']:
            gf = fact2gfact[f]
            report_fidxs.add(sentence2idx[f])
            if gf != 'does not apply':
                report_fidxs.add(sentence2idx[gf])
        
        # Get positive facts for the report
        pos_fidxs = [fidx for fidx in report_fidxs if fidx in similar_fact_idxs_set]
        positive_fact_idxs_per_report[ridx] = pos_fidxs
        
        # Find used clusters
        used_cluster_ids = set()
        for fidx in report_fidxs:
            used_cluster_ids.add(cluster_labels[fidx])
        clusters_to_sample_from = [c for c in nonempty_clusters if c not in used_cluster_ids]
        assert len(clusters_to_sample_from) > 0
        num_samples_per_cluster = math.ceil(max_negative_facts_per_report / len(clusters_to_sample_from))

        # Sample negative facts uniformly from non-used clusters
        sampled_idxs = []
        for c in clusters_to_sample_from:
            cidxs = cluster2simfidxs[c]
            sampled_idxs.extend(random.sample(cidxs, num_samples_per_cluster) if len(cidxs) > num_samples_per_cluster else cidxs)
        sampled_idxs = [i for i in sampled_idxs if i not in report_fidxs] # remove positive facts
        
        if len(sampled_idxs) > max_negative_facts_per_report: # truncate
            random.shuffle(sampled_idxs) # shuffle to avoid bias
            sampled_idxs = sampled_idxs[:max_negative_facts_per_report]
        elif len(sampled_idxs) < max_negative_facts_per_report: # pad with random facts
            idxs_to_skip = set(sampled_idxs)
            idxs_to_skip.update(report_fidxs)
            assert len(idxs_to_skip) == len(sampled_idxs) + len(report_fidxs) # sanity check that there are no duplicates
            while len(sampled_idxs) < max_negative_facts_per_report:
                i = random.choice(similar_fact_idxs) # sample from all similar facts
                if i in idxs_to_skip: continue
                sampled_idxs.append(i)
                idxs_to_skip.add(i) # avoid duplicates

        assert len(sampled_idxs) == max_negative_facts_per_report
        s = ridx * max_negative_facts_per_report
        e = s + max_negative_facts_per_report
        cand_neg_ridx_fidx_pairs[s:e, 0] = ridx # report index
        cand_neg_ridx_fidx_pairs[s:e, 1] = sampled_idxs # fact index

    # Compute softmaxes over candidate negative facts
    print("Computing softmaxes over candidate negative facts...")
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
    negative_fact_idxs_per_report = [[] for _ in range(n_reports)]
    for i in valid_idxs:
        ridx, fidx = cand_neg_ridx_fidx_pairs[i]
        negative_fact_idxs_per_report[ridx].append(fidx.item()) # convert to int
    
    # Print statistics
    print(f'Number of valid negative facts: {len(valid_idxs)}/{len(softmaxes)}')
    print(f'Percentage of valid negative facts: {len(valid_idxs) / len(softmaxes) * 100:.2f}%')
    # average number of negative facts per report
    avg_num_negative_facts_per_report = np.mean([len(fidxs) for fidxs in negative_fact_idxs_per_report])
    print(f'Average number of negative facts per report: {avg_num_negative_facts_per_report:.2f}')

    # Build dicom_id to pos/neg facts
    detailed_metadata = load_mimiccxr_reports_detailed_metadata()
    dicom_id_view_pos_pairs = detailed_metadata['dicom_id_view_pos_pairs']
    assert len(dicom_id_view_pos_pairs) == n_reports
    dicom_id_to_pos_neg_facts = {}
    for ridx, pairs in enumerate(dicom_id_view_pos_pairs):
        for dicom_id, _ in pairs:
            dicom_id_to_pos_neg_facts[dicom_id] = (positive_fact_idxs_per_report[ridx],
                                                   negative_fact_idxs_per_report[ridx])

    # Save output
    output = {
        'facts': all_facts,
        'embeddings': all_fact_embeddings,
        'dicom_id_to_pos_neg_facts': dicom_id_to_pos_neg_facts,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix=(f'mimiccxr_dicom_id_to_pos_neg_facts(num_sim_facts={len(similar_facts)},num_clusters={num_clusters},'
                f'max_neg={max_negative_facts_per_report},ent_th={mlp_nli_entailment_threshold:.3f})'),
        strings=[
            f'len(cand_neg_ridx_fidx_pairs): {len(cand_neg_ridx_fidx_pairs)}',
            f'max_negative_facts_per_report={max_negative_facts_per_report}',
            f'mlp_nli_entailment_threshold={mlp_nli_entailment_threshold}',
            f'len(valid_idxs): {len(valid_idxs)}',
            integrated_report_facts_jsonl_filepath,
            facts_similar_to_anchor_facts_pickle_filepath,
            fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path,
            mlp_nli_checkpoint_folder_path,
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=_Task.choices())
    parser.add_argument('--integrated_fact_metadata_jsonl_filepath', type=str)
    parser.add_argument('--integrated_report_facts_jsonl_filepath', type=str)
    parser.add_argument('--fact_embedding_model_name', type=str)
    parser.add_argument('--fact_embedding_model_checkpoint_folder_path', type=str)
    parser.add_argument('--fact_embedding_batch_size', type=int, default=32)
    parser.add_argument('--fact_embedding_num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--min_similarity_threshold', type=float, default=0.6)
    parser.add_argument('--facts_similar_to_anchor_facts_pickle_filepath', type=str)
    parser.add_argument('--mlp_batch_size', type=int, default=32)
    parser.add_argument('--mlp_num_workers', type=int, default=4)
    parser.add_argument('--mlp_nli_checkpoint_folder_path', type=str)
    parser.add_argument('--mlp_nli_entailment_threshold', type=float, default=0.5)
    parser.add_argument('--num_clusters', type=int)
    parser.add_argument('--max_negative_facts_per_report', type=int)
    args = parser.parse_args()

    if args.task == _Task.FIND_FACTS_SIMILAR_TO_ANCHOR_FACTS:
        assert args.integrated_fact_metadata_jsonl_filepath is not None
        assert args.fact_embedding_model_name is not None
        assert args.fact_embedding_model_checkpoint_folder_path is not None
        _find_facts_similar_to_anchor_facts(
            integrated_fact_metadata_jsonl_filepath=args.integrated_fact_metadata_jsonl_filepath,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
            device=args.device,
            anchor_facts=MULTIDATASET_UNIFIED_CLASSES,
            min_similarity_threshold=args.min_similarity_threshold,
        )
    elif args.task == _Task.FIND_POSITIVE_NEGATIVE_FACTS_PER_REPORT_WITH_FACT_EMBEDDINGS_AND_MLP_NLI:
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.integrated_fact_metadata_jsonl_filepath is not None
        assert args.facts_similar_to_anchor_facts_pickle_filepath is not None
        assert args.fact_embedding_model_name is not None
        assert args.fact_embedding_model_checkpoint_folder_path is not None
        assert args.mlp_nli_checkpoint_folder_path is not None
        assert args.fact_embedding_batch_size is not None
        assert args.fact_embedding_num_workers is not None
        assert args.mlp_batch_size is not None
        assert args.mlp_num_workers is not None
        assert args.mlp_nli_entailment_threshold is not None
        assert args.num_clusters is not None
        assert args.max_negative_facts_per_report is not None
        _find_positive_negative_facts_per_report_with_fact_embeddings_and_mlp_nli(
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            integrated_fact_metadata_jsonl_filepath=args.integrated_fact_metadata_jsonl_filepath,
            facts_similar_to_anchor_facts_pickle_filepath=args.facts_similar_to_anchor_facts_pickle_filepath,
            device=args.device,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
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