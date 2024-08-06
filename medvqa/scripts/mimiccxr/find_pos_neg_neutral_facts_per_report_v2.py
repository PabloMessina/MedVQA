import argparse
import numpy as np
import torch
import math
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from medvqa.datasets.dataloading_utils import embedding_based_nli_collate_batch_fn
from medvqa.datasets.mimiccxr import load_mimiccxr_reports_detailed_metadata
from medvqa.datasets.mimiccxr.report_utils import concatenate_report_parts
from medvqa.datasets.text_data_utils import word_tokenize_texts_in_parallel
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.models.nlp.nli import EmbeddingBasedNLI
from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.scripts.mimiccxr.find_pos_and_neg_facts_per_report_with_fact_embeddings import FactEmbeddingsDataset
from medvqa.utils.data_structures import UnionFind
from medvqa.utils.files import get_cached_jsonl_file, get_file_path_with_hashing_if_too_long, load_jsonl, load_pickle, save_pickle
from medvqa.utils.constants import LABEL_BASED_FACTS
from medvqa.utils.common import LARGE_FAST_CACHE_DIR
from medvqa.utils.logging import print_blue, print_orange, print_red
from medvqa.utils.math import rank_vectors_by_dot_product

class _Task:
    ASSIGN_GPT4_LABEL_BASED_FACTS_TO_REPORTS = 'assign_gpt4_label_based_facts_to_reports'
    ASSIGN_GPT4_REPRESENTATIVE_FACTS_TO_REPORTS = 'assign_gpt4_representative_facts_to_reports'
    COMPUTE_MLP_LABEL_BASED_NLI_SOFTMAXES = 'compute_mlp_label_based_nli_softmaxes'
    COMPUTE_MLP_FACT_BASED_NLI_SOFTMAXES = 'compute_mlp_fact_based_nli_softmaxes'
    COMPUTE_BART_LABEL_BASED_NLI_PREDICTIONS = 'compute_bart_label_based_nli_predictions'
    COMPUTE_BART_FACT_BASED_NLI_PREDICTIONS = 'compute_bart_fact_based_nli_predictions'
    COMPUTE_HYBRID_LABEL_BASED_NLI_PREDICTIONS = 'compute_hybrid_label_based_nli_predictions'
    COMPUTE_HYBRID_FACT_BASED_NLI_PREDICTIONS = 'compute_hybrid_fact_based_nli_predictions'
    FIND_REPRESENTATIVE_FACTS = 'find_representative_facts'
    FIND_K_MOST_SIMILAR_REPRESENTATIVE_FACTS_FOR_EACH_FACT = 'find_k_most_similar_representative_facts_for_each_fact'
    ASSIGN_REPRESENTATIVE_FACTS_TO_REPORTS = 'assign_representative_facts_to_reports'
    INTEGRATE_AND_EXPORT_ALL_DATA = 'integrate_and_export_all_data'
    SAMPLE_NEGATIVE_FACTS_PER_REPORT_WITH_FACT_EMBEDDINGS_AND_MLP_NLI = 'sample_negative_facts_per_report_with_fact_embeddings_and_mlp_nli'
    EXPORT_DICOM_ID_TO_POSITIVE_NEGATIVE_FACTS = 'export_dicom_id_to_positive_negative_facts'
    EXPORT_DICOM_ID_TO_POSITIVE_NEGATIVE_FACTS__REPLACE_EMBEDDINGS = 'export_dicom_id_to_positive_negative_facts__replace_embeddings'
    EXPORT_DICOM_ID_TO_POSITIVE_NEGATIVE_FACTS__IMPROVED_MLP_NLI_BASED_NEGATIVE_SAMPLING = 'export_dicom_id_to_positive_negative_facts__improved_mlp_nli_based_negative_sampling'
    COMPUTE_CLUSTERS_AND_CLUSTER_WEIGHTS_FOR_FACTS = 'compute_clusters_and_cluster_weights_for_facts'
    
    @staticmethod
    def choices():
        return [
            _Task.ASSIGN_GPT4_LABEL_BASED_FACTS_TO_REPORTS,
            _Task.ASSIGN_GPT4_REPRESENTATIVE_FACTS_TO_REPORTS,
            _Task.COMPUTE_MLP_LABEL_BASED_NLI_SOFTMAXES,
            _Task.COMPUTE_MLP_FACT_BASED_NLI_SOFTMAXES,
            _Task.COMPUTE_BART_LABEL_BASED_NLI_PREDICTIONS,
            _Task.COMPUTE_BART_FACT_BASED_NLI_PREDICTIONS,
            _Task.COMPUTE_HYBRID_LABEL_BASED_NLI_PREDICTIONS,
            _Task.COMPUTE_HYBRID_FACT_BASED_NLI_PREDICTIONS,
            _Task.FIND_REPRESENTATIVE_FACTS,
            _Task.FIND_K_MOST_SIMILAR_REPRESENTATIVE_FACTS_FOR_EACH_FACT,
            _Task.ASSIGN_REPRESENTATIVE_FACTS_TO_REPORTS,
            _Task.INTEGRATE_AND_EXPORT_ALL_DATA,
            _Task.SAMPLE_NEGATIVE_FACTS_PER_REPORT_WITH_FACT_EMBEDDINGS_AND_MLP_NLI,
            _Task.EXPORT_DICOM_ID_TO_POSITIVE_NEGATIVE_FACTS,
            _Task.EXPORT_DICOM_ID_TO_POSITIVE_NEGATIVE_FACTS__REPLACE_EMBEDDINGS,
            _Task.EXPORT_DICOM_ID_TO_POSITIVE_NEGATIVE_FACTS__IMPROVED_MLP_NLI_BASED_NEGATIVE_SAMPLING,
            _Task.COMPUTE_CLUSTERS_AND_CLUSTER_WEIGHTS_FOR_FACTS,
        ]
    
GPT4_OUTPUT_TO_NLI = {
    "definitely true": 0, # entailment
    "likely true": 0, # entailment
    "unknown": 1, # neutral
    "likely false": 2, # contradiction
    "definitely false": 2, # contradiction
}

def _assign_gpt4_facts_to_reports(
    gpt4_report_nli_input_output_jsonl_filepaths,
    integrated_report_facts_jsonl_filepaths,
    allowed_facts,
    save_path_prefix,
    return_without_saving=False,
):
    report2labels = {}
    skipped = 0
    not_skipped = 0
    for filepath in gpt4_report_nli_input_output_jsonl_filepaths:
        print(f'Loading {filepath}...')
        input_output_jsonl = get_cached_jsonl_file(filepath)
        for input_output in input_output_jsonl:
            query = input_output['metadata']['query']
            try:
                report_start_idx = query.index("#F ") + 3
                report_end_idx = query.index(" | #H ")
                h = query[report_end_idx+6:]
            except ValueError: # handle alternative format
                report_start_idx = query.index("#R: ") + 4
                report_end_idx = query.index(" | #H: ")
                h = query[report_end_idx+7:]
            if allowed_facts is not None and h not in allowed_facts:
                skipped += 1
                continue
            not_skipped += 1
            p = query[report_start_idx:report_end_idx]
            r = input_output['parsed_response']
            if type(r) == str:
                l = GPT4_OUTPUT_TO_NLI[r]
            else:
                assert type(r) == dict
                assert 'reason' in r
                assert 'label' in r
                l = GPT4_OUTPUT_TO_NLI[r['label']]
            if p not in report2labels:
                report2labels[p] = {h: l}
            else:
                report2labels[p][h] = l

    print(f'len(report2labels): {len(report2labels)}')
    print(f'Number of skipped pairs: {skipped}')
    print(f'Number of not skipped pairs: {not_skipped}')
    output = None
    for filepath in integrated_report_facts_jsonl_filepaths:
        print(f'Loading {filepath}...')
        count = 0
        rows = get_cached_jsonl_file(filepath)
        n = len(rows)
        if output is None:
            output = [{} for _ in range(n)]
        else:
            assert len(output) == n
        for i, row in enumerate(rows):
            fbr = row['fact_based_report']
            fullr = concatenate_report_parts(row['background'], row['findings'], row['impression'])
            found = False
            for x in [fbr, fullr]:
                if x in report2labels:
                    output[i].update(report2labels[x])
                    found = True
            if found:
                count += 1
        print(f'Number of reports with labels: {count}/{n}')

    if return_without_saving:
        return output

    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix=save_path_prefix,
        strings=[
            *integrated_report_facts_jsonl_filepaths,
            *gpt4_report_nli_input_output_jsonl_filepaths,
            f'skipped={skipped}',
            f'not_skipped={not_skipped}',
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def assign_gpt4_label_based_facts_to_reports(
    integrated_report_facts_jsonl_filepaths,
    gpt4_report_nli_input_output_jsonl_filepaths
):
    print_blue(f'Running assign_gpt4_label_based_facts_to_reports()...', bold=True)
    _assign_gpt4_facts_to_reports(
        gpt4_report_nli_input_output_jsonl_filepaths=gpt4_report_nli_input_output_jsonl_filepaths,
        integrated_report_facts_jsonl_filepaths=integrated_report_facts_jsonl_filepaths,
        allowed_facts=set(LABEL_BASED_FACTS),
        save_path_prefix='mimiccxr_gpt4_label_based_facts_assigned_to_reports',
    )

def assign_gpt4_representative_facts_to_reports(
    integrated_report_facts_jsonl_filepaths,
    gpt4_report_nli_input_output_jsonl_filepaths,
    representative_facts_filepath,
):
    print_blue(f'Running assign_gpt4_label_based_facts_to_reports()...', bold=True)
    data = load_pickle(representative_facts_filepath)
    representative_facts = [data['facts'][i] for i in data['dedup_representative_fact_idxs']]
    print(f'len(representative_facts): {len(representative_facts)}')
    _assign_gpt4_facts_to_reports(
        gpt4_report_nli_input_output_jsonl_filepaths=gpt4_report_nli_input_output_jsonl_filepaths,
        integrated_report_facts_jsonl_filepaths=integrated_report_facts_jsonl_filepaths,
        allowed_facts=set(representative_facts),
        save_path_prefix='mimiccxr_gpt4_representative_facts_assigned_to_reports',
    )

def _get_nli_embedding_inputs_for_mlp(p_embs, h_emb):
    if len(p_embs) == 0:
        zero_vector = np.zeros((h_emb.shape[0],), dtype=np.float32)
        p_most_sim_emb = zero_vector
        p_least_sim_emb = zero_vector
        p_max_emb = zero_vector
        p_avg_emb = zero_vector
    else:
        sorted_idxs = rank_vectors_by_dot_product(p_embs, h_emb)
        p_most_sim_emb = p_embs[sorted_idxs[0]]
        p_least_sim_emb = p_embs[sorted_idxs[-1]]
        p_max_emb = np.max(p_embs, axis=0)
        p_avg_emb = np.mean(p_embs, axis=0)
    return h_emb, p_most_sim_emb, p_least_sim_emb, p_max_emb, p_avg_emb

class LabelNLIDataset(Dataset):
    def __init__(self, embeddings, sentence2idx, report_facts, ridx_lidx_pairs):
        self.embeddings = embeddings
        self.lidx2sidx = [sentence2idx[s] for s in LABEL_BASED_FACTS]
        self.report_sidxs = [[sentence2idx[s] for s in rf['facts']] for rf in report_facts]
        self.ridx_lidx_pairs = ridx_lidx_pairs
        self._zero_vector = np.zeros((self.embeddings.shape[1]), dtype=np.float32)
    
    def __len__(self):
        return len(self.ridx_lidx_pairs)

    def __getitem__(self, i):
        ridx, lidx = self.ridx_lidx_pairs[i]
        l_sidx = self.lidx2sidx[lidx]
        r_sidxs = self.report_sidxs[ridx]
        h_emb = self.embeddings[l_sidx]
        p_embs = self.embeddings[r_sidxs]
        h_emb, p_most_sim_emb, p_least_sim_emb, p_max_emb, p_avg_emb = _get_nli_embedding_inputs_for_mlp(p_embs, h_emb)
        return {
            'h_emb': h_emb,
            'p_most_sim_emb': p_most_sim_emb,
            'p_least_sim_emb': p_least_sim_emb,
            'p_max_emb': p_max_emb,
            'p_avg_emb': p_avg_emb,
        }
    
class FactNLIDataset(Dataset):
    def __init__(self, embeddings, sentence2idx, report_facts, ridx_fidx_pairs, representative_facts):
        self.embeddings = embeddings
        self.fidx2sidx = [sentence2idx[s] for s in representative_facts]
        self.report_sidxs = [[sentence2idx[s] for s in rf['facts']] for rf in report_facts]
        self.ridx_fidx_pairs = ridx_fidx_pairs
        self._zero_vector = np.zeros((self.embeddings.shape[1]), dtype=np.float32)
    
    def __len__(self):
        return len(self.ridx_fidx_pairs)

    def __getitem__(self, i):
        ridx, fidx = self.ridx_fidx_pairs[i]
        f_sidx = self.fidx2sidx[fidx]
        r_sidxs = self.report_sidxs[ridx]
        h_emb = self.embeddings[f_sidx]
        p_embs = self.embeddings[r_sidxs]
        h_emb, p_most_sim_emb, p_least_sim_emb, p_max_emb, p_avg_emb = _get_nli_embedding_inputs_for_mlp(p_embs, h_emb)
        return {
            'h_emb': h_emb,
            'p_most_sim_emb': p_most_sim_emb,
            'p_least_sim_emb': p_least_sim_emb,
            'p_max_emb': p_max_emb,
            'p_avg_emb': p_avg_emb,
        }

def compute_mlp_label_based_nli_softmaxes(
    report_to_gpt4_qa_filepath,
    integrated_report_facts_jsonl_filepath,
    device,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    mlp_batch_size,
    mlp_num_workers,
    mlp_nli_checkpoint_folder_path,
):  
    print(f'Reading {integrated_report_facts_jsonl_filepath}...')
    report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    n_reports = len(report_facts)
    print(f'n_reports: {n_reports}')
    unique_sentences = set()
    unique_sentences.update(LABEL_BASED_FACTS)
    for rf in report_facts:
        unique_sentences.update(rf['facts'])
    unique_sentences = list(unique_sentences)
    unique_sentences.sort()
    sentence2idx = {s: i for i, s in enumerate(unique_sentences)}
    print(f'len(unique_sentences): {len(unique_sentences)}')
    
    label2idx = {l: i for i, l in enumerate(LABEL_BASED_FACTS)}
    ridx_lidx_pairs = np.empty((n_reports * len(LABEL_BASED_FACTS), 2), dtype=np.int32)
    print(f'Reading {report_to_gpt4_qa_filepath}...')
    r2gpt4labels = load_pickle(report_to_gpt4_qa_filepath)
    print(f'len(r2gpt4labels): {len(r2gpt4labels)}')
    i = 0
    mask = np.zeros(len(LABEL_BASED_FACTS), dtype=bool)
    for ridx, d in enumerate(r2gpt4labels):
        mask.fill(False)
        for k in d.keys():
            lidx = label2idx[k]
            mask[lidx] = True
        for j in range(len(LABEL_BASED_FACTS)):
            if mask[j]: continue
            ridx_lidx_pairs[i] = ridx, j
            i += 1
    ridx_lidx_pairs = ridx_lidx_pairs[:i]
    print(f'ridx_lidx_pairs.shape: {ridx_lidx_pairs.shape}')
    print(f'Number of skipped pairs: {n_reports * len(LABEL_BASED_FACTS) - i}')
    
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=device,
    )
    embeddings = embedding_extractor.compute_text_embeddings(unique_sentences)
    print(f'embeddings.shape: {embeddings.shape}')

    nli_dataset = LabelNLIDataset(
        embeddings=embeddings,
        sentence2idx=sentence2idx,
        report_facts=report_facts,
        ridx_lidx_pairs=ridx_lidx_pairs,
    )
    dataloader = DataLoader(
        nli_dataset,
        batch_size=mlp_batch_size,
        shuffle=False,
        num_workers=mlp_num_workers,
        collate_fn=lambda *args: embedding_based_nli_collate_batch_fn(*args, include_labels=False),
        pin_memory=True,
    )

    # Load model metadata
    metadata = load_metadata(mlp_nli_checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and device in ['cuda', 'gpu', 'GPU'] else 'cpu')
    
    # Create model
    print("Creating model")
    model = EmbeddingBasedNLI(**model_kwargs)
    model = model.to(device)

    # Load model weights
    print(f"Loading model weights from {mlp_nli_checkpoint_folder_path}")
    checkpoint_path = get_checkpoint_filepath(mlp_nli_checkpoint_folder_path)
    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Set model to evaluation mode
    model.eval()

    # Estimate output memory size
    all_softmaxes = np.empty((len(ridx_lidx_pairs), 3), dtype=np.float32)
    output = {
        'ridx_lidx_pairs': ridx_lidx_pairs,
        'softmaxes': all_softmaxes,
    }
    memory_size = sum([output[k].nbytes for k in output])
    print(f'Estimated output memory size: {memory_size / 1024**2:.2f} MB')

    # Compute softmaxes
    print("Computing softmaxes...")
    with torch.no_grad():
        i = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            h_embs = batch['h_embs'].to(device)
            p_most_sim_embs = batch['p_most_sim_embs'].to(device)
            p_least_sim_embs = batch['p_least_sim_embs'].to(device)
            p_max_embs = batch['p_max_embs'].to(device)
            p_avg_embs = batch['p_avg_embs'].to(device)
            logits = model(h_embs, p_most_sim_embs, p_least_sim_embs, p_max_embs, p_avg_embs)
            softmaxes = torch.softmax(logits, dim=1)
            bsize = len(softmaxes)
            all_softmaxes[i:i+bsize] = softmaxes.cpu().numpy()
            i += bsize
        assert i == len(ridx_lidx_pairs)

    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_label_based_facts_mlp_nli_softmaxes',
        strings=[
            report_to_gpt4_qa_filepath,
            f'len(ridx_lidx_pairs): {len(ridx_lidx_pairs)}',
            integrated_report_facts_jsonl_filepath,
            fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path,
            mlp_nli_checkpoint_folder_path,
            *LABEL_BASED_FACTS,
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle({
        'ridx_lidx_pairs': ridx_lidx_pairs,
        'softmaxes': all_softmaxes,
    }, output_filepath)

def _compute_mlp_fact_based_nli_softmaxes_per_report(
    embeddings,
    sentence2idx,
    report_facts,
    ridx_fidx_pairs,
    representative_facts,
    mlp_batch_size,
    mlp_num_workers,
    mlp_nli_checkpoint_folder_path,
    device,        
):
    nli_dataset = FactNLIDataset(
        embeddings=embeddings,
        sentence2idx=sentence2idx,
        report_facts=report_facts,
        ridx_fidx_pairs=ridx_fidx_pairs,
        representative_facts=representative_facts,
    )
    dataloader = DataLoader(
        nli_dataset,
        batch_size=mlp_batch_size,
        shuffle=False,
        num_workers=mlp_num_workers,
        collate_fn=lambda *args: embedding_based_nli_collate_batch_fn(*args, include_labels=False),
        pin_memory=True,
    )

    # Load model metadata
    metadata = load_metadata(mlp_nli_checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and device in ['cuda', 'gpu', 'GPU'] else 'cpu')
    
    # Create model
    print("Creating model")
    model = EmbeddingBasedNLI(**model_kwargs)
    model = model.to(device)

    # Load model weights
    print(f"Loading model weights from {mlp_nli_checkpoint_folder_path}")
    checkpoint_path = get_checkpoint_filepath(mlp_nli_checkpoint_folder_path)
    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Set model to evaluation mode
    model.eval()

    # Estimate output memory size
    all_softmaxes = np.empty((len(ridx_fidx_pairs), 3), dtype=np.float32)
    output = {
        'ridx_fidx_pairs': ridx_fidx_pairs,
        'softmaxes': all_softmaxes,
    }
    memory_size = sum([output[k].nbytes for k in output])
    print(f'Estimated output memory size: {memory_size / 1024**2:.2f} MB')

    # Compute softmaxes
    print("Computing softmaxes...")
    with torch.no_grad():
        i = 0
        for batch in tqdm(dataloader, total=len(dataloader), mininterval=3):
            h_embs = batch['h_embs'].to(device)
            p_most_sim_embs = batch['p_most_sim_embs'].to(device)
            p_least_sim_embs = batch['p_least_sim_embs'].to(device)
            p_max_embs = batch['p_max_embs'].to(device)
            p_avg_embs = batch['p_avg_embs'].to(device)
            logits = model(h_embs, p_most_sim_embs, p_least_sim_embs, p_max_embs, p_avg_embs)
            softmaxes = torch.softmax(logits, dim=1)
            bsize = len(softmaxes)
            all_softmaxes[i:i+bsize] = softmaxes.cpu().numpy()
            i += bsize
        assert i == len(ridx_fidx_pairs)

    # Return output
    return output

def compute_mlp_fact_based_nli_softmaxes(
    deduplicated_representative_facts_filepath,
    assigned_representative_facts_to_reports_filepath,
    integrated_report_facts_jsonl_filepath,
    report_to_gpt4_qa_filepath,
    device,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    mlp_batch_size,
    mlp_num_workers,
    mlp_nli_checkpoint_folder_path,    
):
    # Load representative facts
    print(f'Reading {deduplicated_representative_facts_filepath}...')
    data = load_pickle(deduplicated_representative_facts_filepath)
    representative_fact_idxs = data['dedup_representative_fact_idxs']
    representative_facts = [data['facts'][i] for i in representative_fact_idxs]
    rfact2idx = {rf: i for i, rf in enumerate(representative_facts)}
    print(f'len(representative_facts): {len(representative_facts)}')

    # Load assigned representative facts to reports
    print(f'Reading {assigned_representative_facts_to_reports_filepath}...')
    data = load_pickle(assigned_representative_facts_to_reports_filepath)
    assigned_representative_fact_idxs = data['assigned_representative_fact_idxs']
    print(f'assigned_representative_fact_idxs.shape: {assigned_representative_fact_idxs.shape}')

    # Load integrated report facts
    print(f'Reading {integrated_report_facts_jsonl_filepath}...')
    report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    n_reports = len(report_facts)
    print(f'n_reports: {n_reports}')
    unique_sentences = set()
    unique_sentences.update(representative_facts)
    for rf in report_facts:
        unique_sentences.update(rf['facts'])
    unique_sentences = list(unique_sentences)
    unique_sentences.sort()
    sentence2idx = {s: i for i, s in enumerate(unique_sentences)}
    print(f'len(unique_sentences): {len(unique_sentences)}')

    assert n_reports == assigned_representative_fact_idxs.shape[0]
    n_repr_facts_per_report = assigned_representative_fact_idxs.shape[1]
    
    ridx_fidx_pairs = np.empty((n_reports * n_repr_facts_per_report, 2), dtype=np.int32)
    print(f'Reading {report_to_gpt4_qa_filepath}...')
    r2gpt4qa = load_pickle(report_to_gpt4_qa_filepath)
    print(f'len(r2gpt4qa): {len(r2gpt4qa)}')
    i = 0
    mask = np.zeros(len(representative_facts), dtype=int)
    mask_idx = 1
    for ridx, d in enumerate(r2gpt4qa):
        for k in d.keys():
            fidx = rfact2idx[k]
            mask[fidx] = mask_idx
        for fidx in assigned_representative_fact_idxs[ridx]:
            if mask[fidx] == mask_idx: continue
            ridx_fidx_pairs[i] = ridx, fidx
            i += 1
        mask_idx += 1 # increment mask index
    ridx_fidx_pairs = ridx_fidx_pairs[:i]
    
    print(f'ridx_fidx_pairs.shape: {ridx_fidx_pairs.shape}')
    print(f'Number of skipped pairs: {n_reports * n_repr_facts_per_report - i}')
    
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=device,
    )
    embeddings = embedding_extractor.compute_text_embeddings(unique_sentences)
    print(f'embeddings.shape: {embeddings.shape}')

    output = _compute_mlp_fact_based_nli_softmaxes_per_report(
        embeddings=embeddings,
        sentence2idx=sentence2idx,
        report_facts=report_facts,
        ridx_fidx_pairs=ridx_fidx_pairs,
        representative_facts=representative_facts,
        mlp_batch_size=mlp_batch_size,
        mlp_num_workers=mlp_num_workers,
        mlp_nli_checkpoint_folder_path=mlp_nli_checkpoint_folder_path,
        device=device,
    )

    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_representative_facts_mlp_nli_softmaxes',
        strings=[
            report_to_gpt4_qa_filepath,
            f'len(ridx_fidx_pairs): {len(ridx_fidx_pairs)}',
            integrated_report_facts_jsonl_filepath,
            fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path,
            mlp_nli_checkpoint_folder_path,
            deduplicated_representative_facts_filepath,
            assigned_representative_facts_to_reports_filepath,
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def sample_negative_facts_per_report_with_fact_embeddings_and_mlp_nli(
    integrated_report_facts_jsonl_filepath,
    device,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    mlp_batch_size,
    mlp_num_workers,
    mlp_nli_checkpoint_folder_path,
    mlp_nli_entailment_threshold,
    num_clusters,
    max_negative_facts_per_report,
):
    # Load integrated report facts
    print(f'Reading {integrated_report_facts_jsonl_filepath}...')
    report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    n_reports = len(report_facts)
    print(f'n_reports: {n_reports}')
    unique_sentences = set()
    for rf in report_facts:
        unique_sentences.update(rf['facts'])
    unique_sentences = list(unique_sentences)
    unique_sentences.sort()
    sentence2idx = {s: i for i, s in enumerate(unique_sentences)}
    print(f'len(unique_sentences): {len(unique_sentences)}')
    
    # Extract embeddings
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=device,
    )
    embeddings = embedding_extractor.compute_text_embeddings(unique_sentences)
    print(f'embeddings.shape: {embeddings.shape}')

    # Cluster embeddings
    print("Clustering embeddings...")
    cluster_labels = embedding_extractor.compute_kmeans_labels(
        texts=unique_sentences, num_clusters=num_clusters, embeddings=embeddings)
    cluster2idxs = [[] for _ in range(num_clusters)]
    for i, c in enumerate(cluster_labels):
        cluster2idxs[c].append(i)

    # Assign candidate negative facts to reports randomly
    print("Assigning candidate negative facts to reports...")
    ridx_fidx_pairs = np.empty((n_reports * max_negative_facts_per_report, 2), dtype=np.int32)
    num_samples_per_cluster = math.ceil(max_negative_facts_per_report / num_clusters)
    for ridx in tqdm(range(n_reports), total=n_reports, mininterval=3):
        report_fidxs = set([sentence2idx[f] for f in report_facts[ridx]['facts']])
        sampled_idxs = []
        for cidxs in cluster2idxs:
            sampled_idxs.extend(random.sample(cidxs, num_samples_per_cluster))
        sampled_idxs = [i for i in sampled_idxs if i not in report_fidxs] # remove positive facts
        
        if len(sampled_idxs) > max_negative_facts_per_report: # truncate
            random.shuffle(sampled_idxs) # shuffle to avoid bias
            sampled_idxs = sampled_idxs[:max_negative_facts_per_report]
        elif len(sampled_idxs) < max_negative_facts_per_report: # pad with random facts
            idxs_to_skip = set(sampled_idxs)
            idxs_to_skip.update(report_fidxs)
            assert len(idxs_to_skip) == len(sampled_idxs) + len(report_fidxs) # sanity check that there are no duplicates
            while len(sampled_idxs) < max_negative_facts_per_report:
                i = random.randint(0, len(unique_sentences) - 1)
                if i in idxs_to_skip: continue
                sampled_idxs.append(i)
                idxs_to_skip.add(i) # avoid duplicates

        assert len(sampled_idxs) == max_negative_facts_per_report
        s = ridx * max_negative_facts_per_report
        e = s + max_negative_facts_per_report
        ridx_fidx_pairs[s:e, 0] = ridx # report index
        ridx_fidx_pairs[s:e, 1] = sampled_idxs # fact index

    # Compute softmaxes over candidate negative facts
    print("Computing softmaxes over candidate negative facts...")
    tmp = _compute_mlp_fact_based_nli_softmaxes_per_report(
        embeddings=embeddings,
        sentence2idx=sentence2idx,
        report_facts=report_facts,
        ridx_fidx_pairs=ridx_fidx_pairs,
        representative_facts=unique_sentences,
        mlp_batch_size=mlp_batch_size,
        mlp_num_workers=mlp_num_workers,
        mlp_nli_checkpoint_folder_path=mlp_nli_checkpoint_folder_path,
        device=device,
    )
    softmaxes = tmp['softmaxes']
    assert softmaxes.shape[0] == ridx_fidx_pairs.shape[0]

    # Filter out negative facts with high entailment softmaxes
    print("Filtering out negative facts with high entailment softmaxes...")
    valid_idxs = np.where(softmaxes[:, 0] < mlp_nli_entailment_threshold)[0]
    negative_fact_idxs_per_report = [[] for _ in range(n_reports)]
    for i in valid_idxs:
        ridx, fidx = ridx_fidx_pairs[i]
        negative_fact_idxs_per_report[ridx].append(fidx.item()) # convert to int
    
    # Print statistics
    print(f'Number of valid negative facts: {len(valid_idxs)}/{len(softmaxes)}')
    print(f'Percentage of valid negative facts: {len(valid_idxs) / len(softmaxes) * 100:.2f}%')
    # average number of negative facts per report
    avg_num_negative_facts_per_report = np.mean([len(fidxs) for fidxs in negative_fact_idxs_per_report])
    print(f'Average number of negative facts per report: {avg_num_negative_facts_per_report:.2f}')
    
    # Save output
    output = {
        'facts': unique_sentences,
        'integrated_report_facts_jsonl_filepath': integrated_report_facts_jsonl_filepath,
        'negative_fact_idxs_per_report': negative_fact_idxs_per_report,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_negative_facts_assigned_to_reports',
        strings=[
            f'len(ridx_fidx_pairs): {len(ridx_fidx_pairs)}',
            f'max_negative_facts_per_report={max_negative_facts_per_report}',
            f'mlp_nli_entailment_threshold={mlp_nli_entailment_threshold}',
            f'len(valid_idxs): {len(valid_idxs)}',
            integrated_report_facts_jsonl_filepath,
            fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path,
            mlp_nli_checkpoint_folder_path,
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def compute_BART_label_based_nli_predictions(
    integrated_report_facts_jsonl_filepath,
    report_to_gpt4_qa_filepath,
    bart_checkpoint_folder_path,
    device,
    bart_batch_size,
    bart_num_workers,
    bart_max_length,
    bart_num_beams,
    part_x_of_y=None,
):  
    
    print(f'Reading {integrated_report_facts_jsonl_filepath}...')
    report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    n_reports = len(report_facts)
    print(f'n_reports: {n_reports}')

    print(f'Reading {report_to_gpt4_qa_filepath}...')
    r2gpt4labels = load_pickle(report_to_gpt4_qa_filepath)
    print(f'len(r2gpt4labels): {len(r2gpt4labels)}')
    assert len(r2gpt4labels) == n_reports
    
    label2idx = {l: i for i, l in enumerate(LABEL_BASED_FACTS)}

    ridx_lidx_pairs = np.empty((n_reports * len(LABEL_BASED_FACTS), 2), dtype=np.int32)
    
    i = 0
    mask = np.zeros(len(LABEL_BASED_FACTS), dtype=bool)
    for ridx, d in tqdm(enumerate(r2gpt4labels), total=n_reports, mininterval=2):
        mask.fill(False)
        for k in d.keys():
            lidx = label2idx[k]
            mask[lidx] = True
        for j in range(len(LABEL_BASED_FACTS)):
            if mask[j]: continue
            # p = report_facts[ridx]['fact_based_report']
            # h = LABEL_BASED_FACTS[j]
            ridx_lidx_pairs[i] = ridx, j
            i += 1

    ridx_lidx_pairs = ridx_lidx_pairs[:i]
    
    if part_x_of_y is not None:
        x, y = part_x_of_y
        assert x < y
        assert y > 0
        assert x >= 0
        n_reports = len(ridx_lidx_pairs)
        n_reports_per_part = [n_reports // y] * y
        for i in range(n_reports % y):
            n_reports_per_part[i] += 1
        assert sum(n_reports_per_part) == n_reports
        start_idx = sum(n_reports_per_part[:x])
        end_idx = start_idx + n_reports_per_part[x]
        ridx_lidx_pairs = ridx_lidx_pairs[start_idx:end_idx]
        print(f'Processing part {x+1}/{y} with {len(ridx_lidx_pairs)} pairs...')

    input_texts = [f'NLI1: {report_facts[ridx]["fact_based_report"]} #H: {LABEL_BASED_FACTS[lidx]}' for ridx, lidx in ridx_lidx_pairs]
    
    print(f'len(input_texts): {len(input_texts)}')
    # print memory consumption of input_texts
    memory_size = sum([len(s) for s in input_texts])
    print(f'Estimated input_texts memory size: {memory_size / 1024**2:.2f} MB')
    print(f'Number of skipped pairs: {n_reports * len(LABEL_BASED_FACTS) - i}')

    output2label = {'e': 0, 'n': 1, 'c': 2}
    nli_predictions, unprocessed_sentences = apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=bart_checkpoint_folder_path,
        sentences=input_texts,
        logger=None,
        device=device,
        batch_size=bart_batch_size,
        num_workers=bart_num_workers,
        max_length=bart_max_length,
        num_beams=bart_num_beams,
        postprocess_input_output_func=lambda _, output: output2label[output[0]],
        save_outputs=False,
    )
    assert len(nli_predictions) == len(input_texts)
    assert len(unprocessed_sentences) == 0

    output = {
        'ridx_lidx_pairs': ridx_lidx_pairs,
        'predictions': nli_predictions,
    }
    if part_x_of_y is not None:
        prefix = f'mimiccxr_label_based_facts_bart_nli_predictions({part_x_of_y[0]+1}_{part_x_of_y[1]})'
    else:
        prefix = 'mimiccxr_label_based_facts_bart_nli_predictions'
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix=prefix,
        strings=[
            integrated_report_facts_jsonl_filepath,
            report_to_gpt4_qa_filepath,
            bart_checkpoint_folder_path,
            *LABEL_BASED_FACTS,
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def compute_BART_fact_based_nli_predictions(
    deduplicated_representative_facts_filepath,
    assigned_representative_facts_to_reports_filepath,
    integrated_report_facts_jsonl_filepath,
    report_to_gpt4_qa_filepath,
    bart_checkpoint_folder_path,
    device,
    bart_batch_size,
    bart_num_workers,
    bart_max_length,
    bart_num_beams,
    part_x_of_y=None,
):  
    # Load representative facts
    print(f'Reading {deduplicated_representative_facts_filepath}...')
    data = load_pickle(deduplicated_representative_facts_filepath)
    representative_fact_idxs = data['dedup_representative_fact_idxs']
    representative_facts = [data['facts'][i] for i in representative_fact_idxs]
    rfact2idx = {rf: i for i, rf in enumerate(representative_facts)}
    print(f'len(representative_facts): {len(representative_facts)}')

    # Load assigned representative facts to reports
    print(f'Reading {assigned_representative_facts_to_reports_filepath}...')
    data = load_pickle(assigned_representative_facts_to_reports_filepath)
    assigned_representative_fact_idxs = data['assigned_representative_fact_idxs']
    print(f'assigned_representative_fact_idxs.shape: {assigned_representative_fact_idxs.shape}')

    # Load integrated report facts
    print(f'Reading {integrated_report_facts_jsonl_filepath}...')
    report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    n_reports = len(report_facts)
    print(f'n_reports: {n_reports}')

    assert n_reports == assigned_representative_fact_idxs.shape[0]
    n_repr_facts_per_report = assigned_representative_fact_idxs.shape[1]

    ridx_fidx_pairs = np.empty((n_reports * n_repr_facts_per_report, 2), dtype=np.int32)
    print(f'Reading {report_to_gpt4_qa_filepath}...')
    r2gpt4qa = load_pickle(report_to_gpt4_qa_filepath)
    print(f'len(r2gpt4qa): {len(r2gpt4qa)}')
    assert len(r2gpt4qa) == n_reports
    i = 0
    mask = np.zeros(len(representative_facts), dtype=int)
    mask_idx = 1
    for ridx, d in enumerate(r2gpt4qa):
        for k in d.keys():
            fidx = rfact2idx[k]
            mask[fidx] = mask_idx
        for fidx in assigned_representative_fact_idxs[ridx]:
            if mask[fidx] == mask_idx: continue
            ridx_fidx_pairs[i] = ridx, fidx
            i += 1
        mask_idx += 1 # increment mask index
    ridx_fidx_pairs = ridx_fidx_pairs[:i]

    print(f'Number of skipped pairs: {n_reports * n_repr_facts_per_report - i}')
    
    if part_x_of_y is not None:
        x, y = part_x_of_y
        assert x < y
        assert y > 0
        assert x >= 0
        n_reports = len(ridx_fidx_pairs)
        n_reports_per_part = [n_reports // y] * y
        for i in range(n_reports % y):
            n_reports_per_part[i] += 1
        assert sum(n_reports_per_part) == n_reports
        start_idx = sum(n_reports_per_part[:x])
        end_idx = start_idx + n_reports_per_part[x]
        ridx_fidx_pairs = ridx_fidx_pairs[start_idx:end_idx]
        print(f'Processing part {x+1}/{y} with {len(ridx_fidx_pairs)} pairs...')

    input_texts = [f'NLI1: {report_facts[ridx]["fact_based_report"]} #H: {representative_facts[fidx]}' for ridx, fidx in ridx_fidx_pairs]
    
    print(f'len(input_texts): {len(input_texts)}')
    # print memory consumption of input_texts
    memory_size = sum([len(s) for s in input_texts])
    print(f'Estimated input_texts memory size: {memory_size / 1024**2:.2f} MB')

    output2label = {'e': 0, 'n': 1, 'c': 2}
    nli_predictions, unprocessed_sentences = apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=bart_checkpoint_folder_path,
        sentences=input_texts,
        logger=None,
        device=device,
        batch_size=bart_batch_size,
        num_workers=bart_num_workers,
        max_length=bart_max_length,
        num_beams=bart_num_beams,
        postprocess_input_output_func=lambda _, output: output2label[output[0]],
        save_outputs=False,
    )
    assert len(nli_predictions) == len(input_texts)
    assert len(unprocessed_sentences) == 0

    output = {
        'ridx_fidx_pairs': ridx_fidx_pairs,
        'predictions': nli_predictions,
    }
    if part_x_of_y is not None:
        prefix = f'mimiccxr_representative_facts_bart_nli_predictions({part_x_of_y[0]+1}_{part_x_of_y[1]})'
    else:
        prefix = 'mimiccxr_representative_facts_bart_nli_predictions'
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix=prefix,
        strings=[
            integrated_report_facts_jsonl_filepath,
            report_to_gpt4_qa_filepath,
            bart_checkpoint_folder_path,
            deduplicated_representative_facts_filepath,
            assigned_representative_facts_to_reports_filepath,
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

_HYBRID_ID = 0
_GPT4_ID = 1

class _HybridLabelBasedNLIPredictor:
    def __init__(self, report_nli_hybrid_metadata_filepath):
        print(f'Reading {report_nli_hybrid_metadata_filepath}...')
        report_nli_hybrid_metadata = load_pickle(report_nli_hybrid_metadata_filepath)
        self.report_nli_hybrid_metadata = report_nli_hybrid_metadata
        self.mlp_et = [report_nli_hybrid_metadata[x]['mlp_et'] for x in LABEL_BASED_FACTS]
        self.mlp_ct = [report_nli_hybrid_metadata[x]['mlp_ct'] for x in LABEL_BASED_FACTS]
        self.mlp_f1 = [report_nli_hybrid_metadata[x]['mlp_f1'] for x in LABEL_BASED_FACTS]
        self.bart_f1 = [report_nli_hybrid_metadata[x]['s2s_f1'] for x in LABEL_BASED_FACTS]
        self.ht1 = report_nli_hybrid_metadata['hybrid_threshold_1']
        self.ht2 = report_nli_hybrid_metadata['hybrid_threshold_2']
    def __call__(self, lidx, mlp_softmax, bart_pred):
        # Possible outputs: 0 (entailment), 1 (neutral), 2 (contradiction), 3 (undecided)
        mlp_et = self.mlp_et[lidx]
        mlp_ct = self.mlp_ct[lidx]
        mlp_f1 = self.mlp_f1[lidx]
        bart_f1 = self.bart_f1[lidx]
        if mlp_softmax[0] > mlp_et:
            mlp_pred = 0 # entailment
        elif mlp_softmax[2] > mlp_ct:
            mlp_pred = 2 # contradiction
        else:
            mlp_pred = 1 # neutral
        if mlp_pred == bart_pred:
            return mlp_pred
        if abs(mlp_pred - bart_pred) == 1: # one is neutral -> mild disagreement
            if bart_f1 > mlp_f1 + self.ht1:
                return bart_pred
            elif mlp_f1 > bart_f1 + self.ht1:
                return mlp_pred
            else:
                return 3 # undecided
        assert abs(mlp_pred - bart_pred) == 2 # one is entailment and the other is contradiction -> strong disagreement
        if bart_f1 > mlp_f1 + self.ht2:
            return bart_pred
        elif mlp_f1 > bart_f1 + self.ht2:
            return mlp_pred
        else:
            return 3

class _HybridFactBasedNLIPredictor:
    def __init__(self, report_nli_hybrid_metadata_filepath):
        print(f'Reading {report_nli_hybrid_metadata_filepath}...')
        report_nli_hybrid_metadata = load_pickle(report_nli_hybrid_metadata_filepath)
        self.report_nli_hybrid_metadata = report_nli_hybrid_metadata
        self.mlp_et = report_nli_hybrid_metadata['other']['mlp_et']
        self.mlp_ct = report_nli_hybrid_metadata['other']['mlp_ct']
        self.mlp_f1 = report_nli_hybrid_metadata['other']['mlp_f1']
        self.bart_f1 = report_nli_hybrid_metadata['other']['s2s_f1']
        self.ht1 = report_nli_hybrid_metadata['hybrid_threshold_1']
        self.ht2 = report_nli_hybrid_metadata['hybrid_threshold_2']
    def __call__(self, mlp_softmax, bart_pred):
        # Possible outputs: 0 (entailment), 1 (neutral), 2 (contradiction), 3 (undecided)
        if mlp_softmax[0] > self.mlp_et:
            mlp_pred = 0 # entailment
        elif mlp_softmax[2] > self.mlp_ct:
            mlp_pred = 2 # contradiction
        else:
            mlp_pred = 1 # neutral
        if mlp_pred == bart_pred:
            return mlp_pred
        if abs(mlp_pred - bart_pred) == 1: # one is neutral -> mild disagreement
            if self.bart_f1 > self.mlp_f1 + self.ht1:
                return bart_pred
            elif self.mlp_f1 > self.bart_f1 + self.ht1:
                return mlp_pred
            else:
                return 3 # undecided
        assert abs(mlp_pred - bart_pred) == 2 # one is entailment and the other is contradiction -> strong disagreement
        if self.bart_f1 > self.mlp_f1 + self.ht2:
            return bart_pred
        elif self.mlp_f1 > self.bart_f1 + self.ht2:
            return mlp_pred
        else:
            return 3

def integrate_label_based_nli_predictions(
    gpt4_label_based_facts_assigned_to_reports_filepath,
    label_based_facts_bart_nli_predictions_filepaths,
    label_based_facts_mlp_nli_softmaxes_filepath,
    report_nli_hybrid_metadata_filepath,
    n_reports,
):
    print(f'Reading {gpt4_label_based_facts_assigned_to_reports_filepath}...')
    gpt4_label_based_facts_assigned_to_reports = load_pickle(gpt4_label_based_facts_assigned_to_reports_filepath)
    
    print(f'Reading {label_based_facts_mlp_nli_softmaxes_filepath}...')
    label_based_facts_mlp_nli_softmaxes = load_pickle(label_based_facts_mlp_nli_softmaxes_filepath)
    mlp_ridx_lidx_pairs = label_based_facts_mlp_nli_softmaxes['ridx_lidx_pairs']
    mlp_softmaxes = label_based_facts_mlp_nli_softmaxes['softmaxes']

    assert type(label_based_facts_bart_nli_predictions_filepaths) == list
    assert len(label_based_facts_bart_nli_predictions_filepaths) > 0
    bart_ridx_lidx_pairs = []
    bart_predictions = []
    for filepath in label_based_facts_bart_nli_predictions_filepaths:
        print(f'Reading {filepath}...')
        tmp = load_pickle(filepath)
        bart_ridx_lidx_pairs.append(tmp['ridx_lidx_pairs'])
        bart_predictions.extend(tmp['predictions'])
    bart_ridx_lidx_pairs = np.concatenate(bart_ridx_lidx_pairs, axis=0)

    assert np.array_equal(mlp_ridx_lidx_pairs, bart_ridx_lidx_pairs)
    assert len(mlp_softmaxes) == len(bart_predictions)

    # First compute the hybrid predictions
    hlbnlip = _HybridLabelBasedNLIPredictor(report_nli_hybrid_metadata_filepath)
    final_predictions = np.empty((n_reports, len(LABEL_BASED_FACTS)), dtype=np.int8)
    final_predictions.fill(3) # undecided
    final_method_ids = np.empty((n_reports, len(LABEL_BASED_FACTS)), dtype=np.int8)
    final_method_ids.fill(_HYBRID_ID)
    visited = np.zeros((n_reports, len(LABEL_BASED_FACTS)), dtype=bool)
    for i, (ridx, lidx) in tqdm(enumerate(mlp_ridx_lidx_pairs), total=len(mlp_ridx_lidx_pairs), mininterval=2):
        final_predictions[ridx, lidx] = hlbnlip(lidx, mlp_softmaxes[i], bart_predictions[i])
        visited[ridx, lidx] = True

    # Next, add the GPT-4 predictions
    label2lidx = {l: i for i, l in enumerate(LABEL_BASED_FACTS)}
    for ridx, d in enumerate(gpt4_label_based_facts_assigned_to_reports):
        for label, gt in d.items():
            lidx = label2lidx[label]
            final_predictions[ridx, lidx] = gt
            final_method_ids[ridx, lidx] = _GPT4_ID
            visited[ridx, lidx] = True

    # Check if there are any unvisited pairs
    n_unvisited = np.sum(~visited)
    if n_unvisited > 0:
        print_red(f'Warning: {n_unvisited} pairs were not visited!', bold=True)

    # Save the final predictions
    output = {
        'nli_predictions': final_predictions,
        'nli_method_ids': final_method_ids,
        'nli_method_id_to_name': {
            _HYBRID_ID: 'hybrid',
            _GPT4_ID: 'gpt4',
        },
        'nli_prediction_to_name': {
            0: 'entailment',
            1: 'neutral',
            2: 'contradiction',
            3: 'undecided',
        },
        'hybrid_metadata': hlbnlip.report_nli_hybrid_metadata,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_label_based_facts_integrated_nli_predictions',
        strings=[
            gpt4_label_based_facts_assigned_to_reports_filepath,
            *label_based_facts_bart_nli_predictions_filepaths,
            label_based_facts_mlp_nli_softmaxes_filepath,
            report_nli_hybrid_metadata_filepath,
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def integrate_fact_based_nli_predictions(
    assigned_representative_facts_to_reports_filepath,
    deduplicated_representative_facts_filepath,
    gpt4_representative_facts_assigned_to_reports_filepath,
    representative_facts_bart_nli_predictions_filepaths,
    representative_facts_mlp_nli_softmaxes_filepath,
    report_nli_hybrid_metadata_filepath,
):
    # Load GPT-4 predictions
    print(f'Reading {gpt4_representative_facts_assigned_to_reports_filepath}...')
    gpt4_representative_facts_assigned_to_reports = load_pickle(gpt4_representative_facts_assigned_to_reports_filepath)
    
    # Load MLP predictions
    print(f'Reading {representative_facts_mlp_nli_softmaxes_filepath}...')
    representative_facts_mlp_nli_softmaxes = load_pickle(representative_facts_mlp_nli_softmaxes_filepath)
    mlp_ridx_fidx_pairs = representative_facts_mlp_nli_softmaxes['ridx_fidx_pairs']
    mlp_softmaxes = representative_facts_mlp_nli_softmaxes['softmaxes']

    # Load BART predictions
    assert type(representative_facts_bart_nli_predictions_filepaths) == list
    assert len(representative_facts_bart_nli_predictions_filepaths) > 0
    bart_ridx_fidx_pairs = []
    bart_predictions = []
    for filepath in representative_facts_bart_nli_predictions_filepaths:
        print(f'Reading {filepath}...')
        tmp = load_pickle(filepath)
        bart_ridx_fidx_pairs.append(tmp['ridx_fidx_pairs'])
        bart_predictions.extend(tmp['predictions'])
    bart_ridx_fidx_pairs = np.concatenate(bart_ridx_fidx_pairs, axis=0)

    assert np.array_equal(mlp_ridx_fidx_pairs, bart_ridx_fidx_pairs)
    assert len(mlp_softmaxes) == len(bart_predictions)

    # Load assigned representative facts to reports
    print(f'Reading {assigned_representative_facts_to_reports_filepath}...')
    data = load_pickle(assigned_representative_facts_to_reports_filepath)
    assigned_representative_fact_idxs = data['assigned_representative_fact_idxs']
    print(f'assigned_representative_fact_idxs.shape: {assigned_representative_fact_idxs.shape}')
    n_reports, n_repr_facts_per_report = assigned_representative_fact_idxs.shape

    # Load representative facts
    print(f'Reading {deduplicated_representative_facts_filepath}...')
    data = load_pickle(deduplicated_representative_facts_filepath)
    representative_fact_idxs = data['dedup_representative_fact_idxs']
    representative_facts = [data['facts'][i] for i in representative_fact_idxs]
    rf2fidx = {rf: i for i, rf in enumerate(representative_facts)}

    # First compute the hybrid predictions
    hfbnlip = _HybridFactBasedNLIPredictor(report_nli_hybrid_metadata_filepath)
    final_predictions = np.empty((n_reports, n_repr_facts_per_report), dtype=np.int8)
    final_predictions.fill(3) # undecided
    final_method_ids = np.empty((n_reports, n_repr_facts_per_report), dtype=np.int8)
    final_method_ids.fill(_HYBRID_ID)
    visited = np.zeros((n_reports, n_repr_facts_per_report), dtype=bool)
    fidx2idx = np.full((n_reports, len(representative_facts)), -1, dtype=np.int32)
    for ridx in range(n_reports):
        for i, fidx in enumerate(assigned_representative_fact_idxs[ridx]):
            fidx2idx[ridx, fidx] = i
    for i, (ridx, fidx) in tqdm(enumerate(mlp_ridx_fidx_pairs), total=len(mlp_ridx_fidx_pairs), mininterval=2):
        idx = fidx2idx[ridx, fidx]
        assert idx != -1
        final_predictions[ridx, idx] = hfbnlip(mlp_softmaxes[i], bart_predictions[i])
        visited[ridx, idx] = True

    # Next, add the GPT-4 predictions
    skipped = 0
    not_skipped = 0
    for ridx, d in enumerate(gpt4_representative_facts_assigned_to_reports):
        for f, gt in d.items():
            fidx = rf2fidx[f]
            idx = fidx2idx[ridx, fidx]
            if idx == -1:
                skipped += 1
                continue
            not_skipped += 1
            final_predictions[ridx, idx] = gt
            final_method_ids[ridx, idx] = _GPT4_ID
            visited[ridx, idx] = True
    print(f'Number of skipped pairs: {skipped} (not skipped: {not_skipped})')

    # Check if there are any unvisited pairs
    n_unvisited = np.sum(~visited)
    if n_unvisited > 0:
        print_red(f'Warning: {n_unvisited} pairs were not visited!', bold=True)

    # Save the final predictions
    output = {
        'nli_predictions': final_predictions,
        'nli_method_ids': final_method_ids,
        'nli_method_id_to_name': {
            _HYBRID_ID: 'hybrid',
            _GPT4_ID: 'gpt4',
        },
        'nli_prediction_to_name': {
            0: 'entailment',
            1: 'neutral',
            2: 'contradiction',
            3: 'undecided',
        },
        'hybrid_metadata': hfbnlip.report_nli_hybrid_metadata,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_fact_based_integrated_nli_predictions',
        strings=[
            assigned_representative_facts_to_reports_filepath,
            deduplicated_representative_facts_filepath,
            gpt4_representative_facts_assigned_to_reports_filepath,
            *representative_facts_bart_nli_predictions_filepaths,
            representative_facts_mlp_nli_softmaxes_filepath,
            report_nli_hybrid_metadata_filepath,
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def _deduplicate_facts_with_union_find(facts, fact_embeddings, threshold, tokenized_facts, token2count):
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
    fact_scores = [None] * len(facts)
    for i in range(len(facts)):
        s = uf.findSet(i)
        fact_scores[i] = sum(token2count[token] for token in tokenized_facts[i]) / len(tokenized_facts[i])
        if s not in set2idx:
            set2idx[s] = i
        else:
            if fact_scores[i] > fact_scores[set2idx[s]]:
                set2idx[s] = i
    # return deduplicated facts idxs
    dedup_idxs = list(set2idx.values())
    dedup_idxs.sort()
    dedup_facts = [facts[i] for i in dedup_idxs]
    print('Number of facts removed:', len(facts) - len(dedup_facts))
    return dedup_facts

def find_representative_facts(
    integrated_report_facts_jsonl_filepath,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    device,
    num_kmeans_clusters,
    num_kmeans_iterations,
    num_kmedoids_clusters,
    num_kmedoids_iterations,
    kmedoids_method,
    union_find_threshold,
    nearest_k,
):
    # Load report facts
    print(f'Reading {integrated_report_facts_jsonl_filepath}...')
    report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    facts = set()
    for rf in report_facts:
        facts.update(rf['facts'])
    facts = list(facts)
    facts.sort()
    fact2idx = {f: i for i, f in enumerate(facts)}
    print(f'len(facts): {len(facts)}')

    # Precompute tokenized facts and token2count
    print_blue('Precomputing tokenized facts and token2count...')
    tokenized_facts = word_tokenize_texts_in_parallel(facts)
    token2count = {}
    for rf in tqdm(report_facts, total=len(report_facts), mininterval=2):
        for f in rf['facts']:
            for token in tokenized_facts[fact2idx[f]]:
                token2count[token] = token2count.get(token, 0) + 1

    # Compute fact embeddings
    print_blue('Computing fact embeddings...')
    fact_embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=device,
    )
    fact_embeddings = fact_embedding_extractor.compute_text_embeddings(facts)           

    # Run kmeans
    print_blue('Running kmeans...')
    kmeans = KMeans(n_clusters=num_kmeans_clusters, n_init='auto', max_iter=num_kmeans_iterations, random_state=0).fit(fact_embeddings)
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
        n_clusters = math.ceil(len(idxs) * num_kmedoids_clusters / len(fact_embeddings)) # scale number of clusters based on number of facts
        print(f'  n_clusters: {n_clusters}, len(idxs): {len(idxs)}, cluster_fact_embeddings.shape: {cluster_fact_embeddings.shape}')
        kmedoids = KMedoids(n_clusters=n_clusters, metric='cosine', method=kmedoids_method,
                            max_iter=num_kmedoids_iterations, random_state=0).fit(cluster_fact_embeddings)
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
            nearest_sentence_idxs = np.argsort(similarities)[::-1][:nearest_k]
            # Choose representative fact
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

    print('len(representative_fact_idxs):', len(representative_fact_idxs))
    representative_fact_idxs = list(representative_fact_idxs)
    representative_fact_idxs.sort()
    representative_fact_embeddings = fact_embeddings[representative_fact_idxs]
    representative_facts = [facts[i] for i in representative_fact_idxs]
    
    # De-duplicate representative facts with union-find
    print_blue('De-duplicating representative facts with union-find...')
    dedup_representative_facts = _deduplicate_facts_with_union_find(
        representative_facts, representative_fact_embeddings, union_find_threshold, tokenized_facts, token2count)
    dedup_representative_fact_idxs = [fact2idx[f] for f in dedup_representative_facts]

    # Save output
    output = {
        'facts': facts,
        'fact_embeddings': fact_embeddings,
        'representative_fact_idxs': representative_fact_idxs,
        'dedup_representative_fact_idxs': dedup_representative_fact_idxs,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_deduplicated_representative_facts',
        strings=[
            integrated_report_facts_jsonl_filepath,
            fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path,
            str(num_kmeans_clusters),
            str(num_kmeans_iterations),
            str(num_kmedoids_clusters),
            str(num_kmedoids_iterations),
            kmedoids_method,
            str(union_find_threshold),
            str(nearest_k),
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def find_k_most_similar_representative_facts_for_each_fact(
    deduplicated_representative_facts_filepath,
    nearest_k,
    batch_size,
    num_workers,
):
    # Load deduplicated representative facts
    print(f'Reading {deduplicated_representative_facts_filepath}...')
    data = load_pickle(deduplicated_representative_facts_filepath)
    facts = data['facts']
    fact_embeddings = data['fact_embeddings']
    representative_fact_idxs = data['dedup_representative_fact_idxs']
    representative_fact_embeddings = fact_embeddings[representative_fact_idxs]
    
    assert nearest_k <= len(representative_fact_embeddings)

    # Find k most similar representative facts for each fact
    print_blue('Finding positive and negative representative facts for each fact...')
    most_similar = np.empty((len(facts), nearest_k), dtype=np.int32)
    dataset = FactEmbeddingsDataset(fact_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=dataset.collate_fn)
    r_f_embs = torch.from_numpy(representative_fact_embeddings).cuda()
    offset = 0
    for batch in tqdm(dataloader, mininterval=2):
        batch_embs = batch.cuda()
        similarities = torch.matmul(batch_embs, r_f_embs.T)
        nearest_idxs = torch.argsort(similarities, dim=1, descending=True)
        nearest_idxs = nearest_idxs.cpu().numpy()
        nearest_idxs = nearest_idxs[:, :nearest_k]
        most_similar[offset:offset+len(batch)] = nearest_idxs
        offset += len(batch)
    assert offset == len(facts)

    # Save output
    output = {
        'most_similar': most_similar,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_k_most_similar_representative_facts',
        strings=[
            deduplicated_representative_facts_filepath,
            str(nearest_k),
            f'most_similar.shape: {most_similar.shape}',
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

_shared_most_similar = None
_shared_report_facts = None
_shared_fact2idx = None
_shared_repr_fact_idxs = None
def _assign_representative_facts_to_reports(ridx, n_pos, n_rand):
    assert n_pos + n_rand <= len(_shared_repr_fact_idxs) # ensure there are enough representative facts
    facts = _shared_report_facts[ridx]['facts']
    fact_idxs = [_shared_fact2idx[f] for f in facts]
    # Find positive representative facts
    pos_idxs = set()
    for i in range(_shared_most_similar.shape[1]):        
        for fidx in fact_idxs:
            idx = _shared_most_similar[fidx, i]
            pos_idxs.add(idx)
            if len(pos_idxs) == n_pos:
                break
        if len(pos_idxs) == n_pos:
            break
    assert len(pos_idxs) <= n_pos
    n_rand += n_pos - len(pos_idxs) # add to n_rand in case n_pos is not satisfied
    # Find random representative facts
    candidate_idxs = [idx for idx in range(len(_shared_repr_fact_idxs)) if idx not in pos_idxs]
    rand_idxs = np.random.choice(candidate_idxs, n_rand, replace=False)
    # Merge into a single list
    final_list = list(pos_idxs) + list(rand_idxs)
    return final_list

def assign_representative_facts_to_reports(
    integrated_report_facts_jsonl_filepath,
    deduplicated_representative_facts_filepath,
    k_most_similar_representative_facts_per_fact_filepath,
    n_pos, n_rand, num_processes,
):
    # Load report facts
    print(f'Reading {integrated_report_facts_jsonl_filepath}...')
    report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    n_reports = len(report_facts)
    print(f'n_reports: {n_reports}')

    # Load deduplicated representative facts
    print(f'Reading {deduplicated_representative_facts_filepath}...')
    data = load_pickle(deduplicated_representative_facts_filepath)
    facts = data['facts']
    representative_fact_idxs = data['dedup_representative_fact_idxs']

    # Load k most similar representative facts for each fact
    print(f'Reading {k_most_similar_representative_facts_per_fact_filepath}...')
    data = load_pickle(k_most_similar_representative_facts_per_fact_filepath)
    most_similar = data['most_similar']

    # Prepare shared variables
    global _shared_most_similar
    global _shared_report_facts
    global _shared_fact2idx
    global _shared_repr_fact_idxs
    _shared_most_similar = most_similar
    _shared_report_facts = report_facts
    _shared_fact2idx = {f: i for i, f in enumerate(facts)}
    _shared_repr_fact_idxs = representative_fact_idxs

    # Assign representative facts to reports
    print_blue(f'Assigning representative facts to reports with num_processes={num_processes}...')
    with Pool(num_processes) as p:
        results = p.starmap(_assign_representative_facts_to_reports, [(ridx, n_pos, n_rand) for ridx in range(n_reports)])
    results = np.array(results)
    assert results.shape == (n_reports, n_pos + n_rand)
    print(f'results.shape: {results.shape}')

    # Save output
    output = {
        'assigned_representative_fact_idxs': results,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_assigned_representative_facts_to_reports',
        strings=[
            integrated_report_facts_jsonl_filepath,
            deduplicated_representative_facts_filepath,
            k_most_similar_representative_facts_per_fact_filepath,
            str(n_pos),
            str(n_rand),
            f'results.shape: {results.shape}',
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def integrate_and_export_all_data(
    fact_based_integrated_nli_predictions_filepath,
    label_based_integrated_nli_predictions_filepath,
    assigned_representative_facts_to_reports_filepath,
    deduplicated_representative_facts_filepath,
    gpt4_report_nli_input_output_jsonl_filepaths,
    integrated_report_facts_jsonl_filepath,
    integrated_sentence_facts_jsonl_filepath,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
):
    """
    Expected output:
    {
        'extracted_facts': list,
        'extracted_fact_embeddings': np.ndarray,
        'representative_fact_idxs': list,
        'label_based_facts': list,
        'label_based_fact_embeddings': np.ndarray,
        'reports': list,

        'fact_based_nli_predictions': {
            'representative_fact_idxs_per_report': np.ndarray,
            'nli_predictions': np.ndarray,
            'nli_method_ids': np.ndarray,
            'nli_method_id_to_name': dict,
            'nli_prediction_to_name': dict,
            'hybrid_metadata': dict,
        },
        'label_based_nli_predictions': {
            'nli_predictions': np.ndarray,
            'nli_method_ids': np.ndarray,
            'nli_method_id_to_name': dict,
            'nli_prediction_to_name': dict,
            'hybrid_metadata': dict,
        },

        'gpt4_nli_input_output': list,
    }
    """

    output = {}

    # Load deduplicated representative facts
    print(f'Reading {deduplicated_representative_facts_filepath}...')
    data = load_pickle(deduplicated_representative_facts_filepath)
    output['extracted_facts'] = data['facts']
    output['extracted_fact_embeddings'] = data['fact_embeddings']
    output['representative_fact_idxs'] = data['dedup_representative_fact_idxs']
    f2idx = {f: i for i, f in enumerate(output['extracted_facts'])}
    
    # Compute label-based facts and their embeddings
    output['label_based_facts'] = LABEL_BASED_FACTS

    # Compute label-based fact embeddings
    print_blue('Computing label-based fact embeddings...')
    text_encoder = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device='cuda',
    )
    output['label_based_fact_embeddings'] = text_encoder.compute_text_embeddings(LABEL_BASED_FACTS)
    print(f'label_based_fact_embeddings.shape: {output["label_based_fact_embeddings"].shape}')

    # Load integrated report facts
    from nltk.tokenize import sent_tokenize
    integrated_report_facts = load_jsonl(integrated_report_facts_jsonl_filepath)
    integrated_sentence_facts = load_jsonl(integrated_sentence_facts_jsonl_filepath)
    sentence2facts = {}
    for x in integrated_sentence_facts:
        sentence2facts[x['sentence']] = x['facts']
    for rf in tqdm(integrated_report_facts, total=len(integrated_report_facts), mininterval=2):
        findings = rf['findings']
        findings_fact_idxs = []
        impression = rf['impression']
        impression_fact_idxs = []
        if findings:
            seen = set()
            findings_sentences = sent_tokenize(findings)
            for s in findings_sentences:
                for f in sentence2facts[s]:
                    fidx = f2idx[f]
                    if fidx not in seen:
                        seen.add(fidx)
                        findings_fact_idxs.append(fidx)
        if impression:
            seen = set()
            impression_sentences = sent_tokenize(impression)
            for s in impression_sentences:
                for f in sentence2facts[s]:
                    fidx = f2idx[f]
                    if fidx not in seen:
                        seen.add(fidx)
                        impression_fact_idxs.append(fidx)
        rf['findings_fact_idxs'] = findings_fact_idxs
        rf['impression_fact_idxs'] = impression_fact_idxs

    output['reports'] = integrated_report_facts

    # Load integrated NLI predictions
    print(f'Reading {fact_based_integrated_nli_predictions_filepath}...')
    fact_based_integrated_nli_predictions = load_pickle(fact_based_integrated_nli_predictions_filepath)
    print(f'Reading {label_based_integrated_nli_predictions_filepath}...')
    label_based_integrated_nli_predictions = load_pickle(label_based_integrated_nli_predictions_filepath)
    print(f'Reading {assigned_representative_facts_to_reports_filepath}...')
    assigned_representative_facts_to_reports = load_pickle(assigned_representative_facts_to_reports_filepath)

    output['fact_based_nli_predictions'] = fact_based_integrated_nli_predictions
    fact_based_integrated_nli_predictions['representative_fact_idxs_per_report'] = \
        assigned_representative_facts_to_reports['assigned_representative_fact_idxs']

    output['label_based_nli_predictions'] = label_based_integrated_nli_predictions

    output['gpt4_report_nli_input_output'] = []
    for filepath in gpt4_report_nli_input_output_jsonl_filepaths:
        print(f'Reading {filepath}...')
        rows = load_jsonl(filepath)
        for row in rows:
            query = row['metadata']['query']
            report_start_idx = query.index("#F ") + 3
            report_end_idx = query.index(" | #H ")
            h = query[report_end_idx+6:]
            p = query[report_start_idx:report_end_idx]
            r = row['parsed_response']
            if type(r) == str:
                label = r
                reason = None
            else:
                assert type(r) == dict
                label = r['label']
                reason = r['reason']
            item = {
                'premise': p,
                'hypothesis': h,
                'label': label,
            }
            if reason: item['reason'] = reason
            output['gpt4_report_nli_input_output'].append(item)

    # Save output
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_report_fact_nli_integrated_data',
        strings=[
            fact_based_integrated_nli_predictions_filepath,
            label_based_integrated_nli_predictions_filepath,
            assigned_representative_facts_to_reports_filepath,
            deduplicated_representative_facts_filepath,
            *gpt4_report_nli_input_output_jsonl_filepaths,
            integrated_report_facts_jsonl_filepath,
            integrated_sentence_facts_jsonl_filepath,
            fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path,
        ],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)


_shared_nli_predictions = None
_shared_reports = None
_shared_representative_fact_idxs_per_report = None
_shared_representative_fact_idxs = None
_shared_gpt4_nli_preds_per_report = None
_shared_negative_fact_idxs_per_report = None

def _get_pos_neg_facts_per_report_label_based__task(ridx):
    pos_facts = []
    neg_facts = []
    for lidx in range(len(LABEL_BASED_FACTS)):
        pred = _shared_nli_predictions[ridx, lidx]
        if pred == 0: # entailment
            pos_facts.append(lidx)
        elif pred == 1 or pred == 2: # neutral or contradiction
            neg_facts.append(lidx)
        elif pred == 3: # undecided
            continue
        else:
            raise ValueError(f'Invalid prediction: {pred}')
    return pos_facts, neg_facts

def _get_pos_neg_facts_per_report_fact_based__task(ridx):
    pos_facts = set()
    neg_facts = set()
    # Find all facts in the report
    fact_idxs = _shared_reports[ridx]['findings_fact_idxs'] + _shared_reports[ridx]['impression_fact_idxs']
    pos_facts.update(fact_idxs) # all facts in the report are true -> positive
    # Find representative facts
    for i, idx in enumerate(_shared_representative_fact_idxs_per_report[ridx]):
        fidx = _shared_representative_fact_idxs[idx]
        pred = _shared_nli_predictions[ridx, i]
        if pred == 0: # entailment
            pos_facts.add(fidx) # positive
        elif pred == 1 or pred == 2: # neutral or contradiction
            neg_facts.add(fidx) # negative
        elif pred == 3: # undecided
            continue
        else:
            raise ValueError(f'Invalid prediction: {pred}')
    # Facts annotated by GPT-4
    for f, l in _shared_gpt4_nli_preds_per_report[ridx].items():
        fidx = _shared_fact2idx[f]
        if l == 0:
            pos_facts.add(fidx)
        elif l == 1 or l == 2:
            neg_facts.add(fidx)
        else:
            raise ValueError(f'Invalid label: {l}')
        
    return list(pos_facts), list(neg_facts)

def _get_pos_neg_facts_per_report_fact_based_v2__task(ridx):
    pos_facts = set()
    neg_facts = set()
    # Find all facts in the report
    fact_idxs = [_shared_fact2idx[f] for f in _shared_integrated_report_facts[ridx]['facts']]
    pos_facts.update(fact_idxs) # all facts in the report are true -> positive
    # Facts annotated by GPT-4
    for f, l in _shared_gpt4_nli_preds_per_report[ridx].items():
        fidx = _shared_fact2idx[f]
        if l == 0:
            pos_facts.add(fidx)
        elif l == 1 or l == 2:
            neg_facts.add(fidx)
        else:
            raise ValueError(f'Invalid label: {l}')
    for fidx in neg_facts:
        assert fidx not in pos_facts # sanity check
    # Negative facts
    for fidx in _shared_negative_fact_idxs_per_report[ridx]:
        if fidx not in pos_facts: # do not add negative facts that are already positive
            neg_facts.add(fidx)
        
    return list(pos_facts), list(neg_facts)

def _get_pos_neg_facts_per_report_label_based(data, num_processes=10):
    label_based_nli_predictions = data['label_based_nli_predictions']
    nli_predictions = label_based_nli_predictions['nli_predictions']
    n_reports = len(data['reports'])
    global _shared_nli_predictions
    _shared_nli_predictions = nli_predictions
    print_blue(f'Getting positive and negative facts per report with num_processes={num_processes}...')
    with Pool(num_processes) as p:
        pos_neg_facts_per_report = p.map(_get_pos_neg_facts_per_report_label_based__task, range(n_reports))
    return pos_neg_facts_per_report

def _get_pos_neg_facts_per_report_fact_based(data, gpt4_report_nli_input_output_jsonl_filepaths,
                                                integrated_report_facts_jsonl_filepaths,
                                                fact_embedding_model_name, fact_embedding_model_checkpoint_folder_path,
                                                fact_embedding_batch_size, fact_embedding_num_workers,
                                                num_processes=10):
    assert gpt4_report_nli_input_output_jsonl_filepaths is not None
    assert integrated_report_facts_jsonl_filepaths is not None
    fact_based_nli_predictions = data['fact_based_nli_predictions']
    reports = data['reports']
    representative_fact_idxs = data['representative_fact_idxs']
    nli_predictions = fact_based_nli_predictions['nli_predictions']
    representative_fact_idxs_per_report = fact_based_nli_predictions['representative_fact_idxs_per_report']
    facts = data['extracted_facts']
    embeddings = data['extracted_fact_embeddings']
    fact2idx = {f: i for i, f in enumerate(facts)}
    n_reports = len(data['reports'])

    gpt4_nli_preds_per_report = _assign_gpt4_facts_to_reports(
        gpt4_report_nli_input_output_jsonl_filepaths=gpt4_report_nli_input_output_jsonl_filepaths,
        integrated_report_facts_jsonl_filepaths=integrated_report_facts_jsonl_filepaths,
        allowed_facts=None,
        save_path_prefix=None,
        return_without_saving=True,
    )
    assert len(gpt4_nli_preds_per_report) == n_reports
    
    gpt4_facts = set()
    for d in gpt4_nli_preds_per_report:
        gpt4_facts.update(d.keys())
    facts_set = set(facts)
    gpt4_extract_facts = list(gpt4_facts - facts_set)
    
    if len(gpt4_extract_facts):
        print_orange(f'NOTE: {len(gpt4_extract_facts)} facts extracted by GPT-4 are not in the extracted facts.', bold=True)
        print_orange('Adding them to the extracted facts...', bold=True)
        assert fact_embedding_model_name is not None
        assert fact_embedding_model_checkpoint_folder_path is not None
        assert fact_embedding_batch_size is not None
        assert fact_embedding_num_workers is not None
        all_facts = facts + gpt4_extract_facts
        fact_encoder = CachedTextEmbeddingExtractor(
            model_name=fact_embedding_model_name,
            model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            batch_size=fact_embedding_batch_size,
            num_workers=fact_embedding_num_workers,
            device='cuda',
        )
        all_fact_embeddings = fact_encoder.compute_text_embeddings(all_facts)
        embeddings = all_fact_embeddings
        facts = all_facts
        fact2idx = {f: i for i, f in enumerate(facts)}
        print(f'len(facts): {len(facts)}')
        print(f'embeddings.shape: {embeddings.shape}')

    global _shared_nli_predictions
    global _shared_reports
    global _shared_representative_fact_idxs_per_report
    global _shared_representative_fact_idxs
    global _shared_gpt4_nli_preds_per_report
    global _shared_fact2idx
    _shared_nli_predictions = nli_predictions
    _shared_reports = reports
    _shared_representative_fact_idxs_per_report = representative_fact_idxs_per_report
    _shared_representative_fact_idxs = representative_fact_idxs
    _shared_gpt4_nli_preds_per_report = gpt4_nli_preds_per_report
    _shared_fact2idx = fact2idx

    print_blue(f'Getting positive and negative facts per report with num_processes={num_processes}...')
    with Pool(num_processes) as p:
        pos_neg_facts_per_report = p.map(_get_pos_neg_facts_per_report_fact_based__task, range(n_reports))

    return pos_neg_facts_per_report, facts, embeddings

def _get_pos_neg_facts_per_report_fact_based_v2(gpt4_report_nli_input_output_jsonl_filepaths,
                                                mimiccxr_negative_facts_assigned_to_reports_filepath,
                                                integrated_report_facts_jsonl_filepaths,
                                                fact_embedding_model_name, fact_embedding_model_checkpoint_folder_path,
                                                fact_embedding_batch_size, fact_embedding_num_workers,
                                                num_processes=10):
    assert gpt4_report_nli_input_output_jsonl_filepaths is not None
    assert mimiccxr_negative_facts_assigned_to_reports_filepath is not None
    assert integrated_report_facts_jsonl_filepaths is not None
    assert fact_embedding_model_name is not None
    assert fact_embedding_model_checkpoint_folder_path is not None
    assert fact_embedding_batch_size is not None
    assert fact_embedding_num_workers is not None

    # Load the negative facts assigned to reports
    print(f'Reading {mimiccxr_negative_facts_assigned_to_reports_filepath}...')
    negative_facts_assigned_to_reports = load_pickle(mimiccxr_negative_facts_assigned_to_reports_filepath)
    facts = negative_facts_assigned_to_reports['facts']
    negative_fact_idxs_per_report = negative_facts_assigned_to_reports['negative_fact_idxs_per_report']

    # Assign GPT-4 facts to reports
    gpt4_nli_preds_per_report = _assign_gpt4_facts_to_reports(
        gpt4_report_nli_input_output_jsonl_filepaths=gpt4_report_nli_input_output_jsonl_filepaths,
        integrated_report_facts_jsonl_filepaths=integrated_report_facts_jsonl_filepaths,
        allowed_facts=None,
        save_path_prefix=None,
        return_without_saving=True,
    )
    
    # Collect GPT-4 facts
    gpt4_facts = set()
    for d in gpt4_nli_preds_per_report:
        gpt4_facts.update(d.keys())
    facts_set = set(facts)
    gpt4_extract_facts = list(gpt4_facts - facts_set)
    all_facts = facts + gpt4_extract_facts
    fact2idx = {f: i for i, f in enumerate(all_facts)}
    print(f'len(facts): {len(facts)}')
    print(f'len(gpt4_extract_facts): {len(gpt4_extract_facts)}')
    print(f'len(all_facts): {len(all_facts)}')

    # Obtain embeddings for all facts
    fact_encoder = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device='cuda',
    )
    embeddings = fact_encoder.compute_text_embeddings(all_facts)
    print(f'embeddings.shape: {embeddings.shape}')

    # Load integrated report facts
    integrated_report_facts = get_cached_jsonl_file(integrated_report_facts_jsonl_filepaths[-1]) # only the last file is needed
    
    global _shared_integrated_report_facts
    global _shared_gpt4_nli_preds_per_report
    global _shared_fact2idx
    global _shared_negative_fact_idxs_per_report
    _shared_integrated_report_facts = integrated_report_facts
    _shared_gpt4_nli_preds_per_report = gpt4_nli_preds_per_report
    _shared_fact2idx = fact2idx
    _shared_negative_fact_idxs_per_report = negative_fact_idxs_per_report

    print_blue(f'Getting positive and negative facts per report with num_processes={num_processes}...')
    n_reports = len(integrated_report_facts)
    with Pool(num_processes) as p:
        pos_neg_facts_per_report = p.map(_get_pos_neg_facts_per_report_fact_based_v2__task, range(n_reports))

    return pos_neg_facts_per_report, all_facts, embeddings

def export_dicom_id_to_positive_negative_facts(
    mode,
    mimiccxr_report_fact_nli_integrated_data_filepath,
    gpt4_report_nli_input_output_jsonl_filepaths,
    integrated_report_facts_jsonl_filepaths,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
):
    assert mode in ['label_based', 'fact_based', 'all']
    assert mimiccxr_report_fact_nli_integrated_data_filepath is not None
    print(f'Reading {mimiccxr_report_fact_nli_integrated_data_filepath}...')
    data = load_pickle(mimiccxr_report_fact_nli_integrated_data_filepath)
    n_reports = len(data['reports'])

    if mode == 'label_based':
        facts = data['label_based_facts']
        embeddings = data['label_based_fact_embeddings']
        pos_neg_facts_per_report = _get_pos_neg_facts_per_report_label_based(data)
        prefix = 'mimiccxr_dicom_id_to_label_based_pos_neg_facts'
        filepath_strings = [mimiccxr_report_fact_nli_integrated_data_filepath]
    elif mode == 'fact_based':
        pos_neg_facts_per_report, facts, embeddings = _get_pos_neg_facts_per_report_fact_based(
            data=data, gpt4_report_nli_input_output_jsonl_filepaths=gpt4_report_nli_input_output_jsonl_filepaths,
            integrated_report_facts_jsonl_filepaths=integrated_report_facts_jsonl_filepaths,
            fact_embedding_model_name=fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=fact_embedding_batch_size,
            fact_embedding_num_workers=fact_embedding_num_workers,
        )
        prefix = 'mimiccxr_dicom_id_to_fact_based_pos_neg_facts'
        filepath_strings = [
            mimiccxr_report_fact_nli_integrated_data_filepath,
            *gpt4_report_nli_input_output_jsonl_filepaths,
            *integrated_report_facts_jsonl_filepaths,
        ]
    elif mode == 'all':
        facts1 = data['label_based_facts']
        embeddings1 = data['label_based_fact_embeddings']
        pos_neg_facts_per_report1 = _get_pos_neg_facts_per_report_label_based(data)
        assert len(facts1) == len(embeddings1)
        assert n_reports == len(pos_neg_facts_per_report1)

        pos_neg_facts_per_report2, facts2, embeddings2 = _get_pos_neg_facts_per_report_fact_based(
            data=data, gpt4_report_nli_input_output_jsonl_filepaths=gpt4_report_nli_input_output_jsonl_filepaths,
            integrated_report_facts_jsonl_filepaths=integrated_report_facts_jsonl_filepaths,
            fact_embedding_model_name=fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=fact_embedding_batch_size,
            fact_embedding_num_workers=fact_embedding_num_workers,
        )
        assert len(facts2) == len(embeddings2)
        assert n_reports == len(pos_neg_facts_per_report1)
        
        print_blue('Merging label-based and fact-based data...')

        f2idx2 = {f: i for i, f in enumerate(facts2)}
        facts1_set = set(facts1)
        facts2_filtered = [f for f in facts2 if f not in facts1_set]
        facts = facts1 + facts2_filtered
        f2idx = {f: i for i, f in enumerate(facts)}
        idx2_to_idx = [f2idx[f] for f in facts2]
        embeddings = np.concatenate([embeddings1, embeddings2[[f2idx2[f] for f in facts2_filtered]]], axis=0)
        pos_neg_facts_per_report = [None] * n_reports
        for ridx in tqdm(range(n_reports), total=n_reports, mininterval=2):
            pos_fact_idxs1 = pos_neg_facts_per_report1[ridx][0]
            neg_fact_idxs1 = pos_neg_facts_per_report1[ridx][1]
            pos_fact_idxs2 = pos_neg_facts_per_report2[ridx][0]
            neg_fact_idxs2 = pos_neg_facts_per_report2[ridx][1]
            pos_fact_idxs2 = [idx2_to_idx[i] for i in pos_fact_idxs2]
            neg_fact_idxs2 = [idx2_to_idx[i] for i in neg_fact_idxs2]
            pos_fact_idxs = set()
            neg_fact_idxs = set()
            pos_fact_idxs.update(pos_fact_idxs1)
            pos_fact_idxs.update(pos_fact_idxs2)
            neg_fact_idxs.update(neg_fact_idxs1)
            neg_fact_idxs.update(neg_fact_idxs2)
            pos_neg_facts_per_report[ridx] = (list(pos_fact_idxs), list(neg_fact_idxs))
        prefix = 'mimiccxr_dicom_id_to_all_pos_neg_facts'
        filepath_strings = [
            mimiccxr_report_fact_nli_integrated_data_filepath,
            *gpt4_report_nli_input_output_jsonl_filepaths,
            *integrated_report_facts_jsonl_filepaths,
        ]

    # Detect conflicts
    print_blue('Detecting conflicts...')
    ridxs_with_conflict = []
    for ridx in tqdm(range(n_reports), total=n_reports, mininterval=2):
        pos_fidxs = set(pos_neg_facts_per_report[ridx][0])
        neg_fidxs = set(pos_neg_facts_per_report[ridx][1])
        conflicting_fidxs = pos_fidxs & neg_fidxs
        if len(conflicting_fidxs) > 0:
            # there are conflicting facts
            conflicting_fidxs_list = list(conflicting_fidxs)
            conflicting_fidxs_list.sort()
            ridxs_with_conflict.append({
                'ridx': ridx,
                'conflicting_fidxs': conflicting_fidxs_list,
            })
            # remove negative facts that are also positive
            neg_fidxs -= conflicting_fidxs
            pos_neg_facts_per_report[ridx] = (list(pos_fidxs), list(neg_fidxs))

    if len(ridxs_with_conflict) > 0:
        print_red(f'WARNING: {len(ridxs_with_conflict)} reports have conflicting positive and negative facts.', bold=True)

    # Build dicom_id to pos/neg facts
    detailed_metadata = load_mimiccxr_reports_detailed_metadata()
    dicom_id_view_pos_pairs = detailed_metadata['dicom_id_view_pos_pairs']
    assert len(dicom_id_view_pos_pairs) == n_reports
    dicom_id_to_pos_neg_facts = {}
    for ridx, pairs in enumerate(dicom_id_view_pos_pairs):
        for dicom_id, _ in pairs:
            dicom_id_to_pos_neg_facts[dicom_id] = pos_neg_facts_per_report[ridx]

    # Save output
    output = {
        'facts': facts,
        'embeddings': embeddings,
        'dicom_id_to_pos_neg_facts': dicom_id_to_pos_neg_facts,
        'ridxs_with_conflict': ridxs_with_conflict,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR, prefix=prefix, strings=filepath_strings, force_hashing=True)
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def export_dicom_id_to_positive_negative_facts__improved_mlp_nli_based_negative_sampling(
    mode,
    mimiccxr_report_fact_nli_integrated_data_filepath,
    gpt4_report_nli_input_output_jsonl_filepaths,
    integrated_report_facts_jsonl_filepaths,
    mimiccxr_negative_facts_assigned_to_reports_filepath,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
):
    assert mode in ['label_based', 'fact_based', 'all']

    if mode == 'label_based':
        assert mimiccxr_report_fact_nli_integrated_data_filepath is not None
        print(f'Reading {mimiccxr_report_fact_nli_integrated_data_filepath}...')
        data = load_pickle(mimiccxr_report_fact_nli_integrated_data_filepath)
        n_reports = len(data['reports'])
        facts = data['label_based_facts']
        embeddings = None
        pos_neg_facts_per_report = _get_pos_neg_facts_per_report_label_based(data)
        prefix = 'mimiccxr_dicom_id_to_label_based_pos_neg_facts'
        filepath_strings = [mimiccxr_report_fact_nli_integrated_data_filepath]
    elif mode == 'fact_based':
        pos_neg_facts_per_report, facts, embeddings = _get_pos_neg_facts_per_report_fact_based_v2(
            gpt4_report_nli_input_output_jsonl_filepaths=gpt4_report_nli_input_output_jsonl_filepaths,
            mimiccxr_negative_facts_assigned_to_reports_filepath=mimiccxr_negative_facts_assigned_to_reports_filepath,
            integrated_report_facts_jsonl_filepaths=integrated_report_facts_jsonl_filepaths,
            fact_embedding_model_name=fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=fact_embedding_batch_size,
            fact_embedding_num_workers=fact_embedding_num_workers,
        )
        prefix = 'mimiccxr_dicom_id_to_fact_based_pos_neg_facts'
        filepath_strings = [
            *gpt4_report_nli_input_output_jsonl_filepaths,
            mimiccxr_negative_facts_assigned_to_reports_filepath,
            *integrated_report_facts_jsonl_filepaths,
            fact_embedding_model_checkpoint_folder_path,
        ]
        n_reports = len(pos_neg_facts_per_report)
    elif mode == 'all':
        assert mimiccxr_report_fact_nli_integrated_data_filepath is not None
        print(f'Reading {mimiccxr_report_fact_nli_integrated_data_filepath}...')
        data = load_pickle(mimiccxr_report_fact_nli_integrated_data_filepath)
        
        n_reports = len(data['reports'])
        facts1 = data['label_based_facts']
        pos_neg_facts_per_report1 = _get_pos_neg_facts_per_report_label_based(data)
        assert n_reports == len(pos_neg_facts_per_report1)

        pos_neg_facts_per_report2, facts2, _ = _get_pos_neg_facts_per_report_fact_based_v2(
            gpt4_report_nli_input_output_jsonl_filepaths=gpt4_report_nli_input_output_jsonl_filepaths,
            mimiccxr_negative_facts_assigned_to_reports_filepath=mimiccxr_negative_facts_assigned_to_reports_filepath,
            integrated_report_facts_jsonl_filepaths=integrated_report_facts_jsonl_filepaths,
            fact_embedding_model_name=fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=fact_embedding_batch_size,
            fact_embedding_num_workers=fact_embedding_num_workers,
        )
        assert n_reports == len(pos_neg_facts_per_report1)
        
        print_blue('Merging label-based and fact-based data...')
        all_facts = list(set(facts1) | set(facts2))
        all_facts.sort()
        f2idx = {f: i for i, f in enumerate(all_facts)}
        idx1_to_idx = [f2idx[f] for f in facts1]
        idx2_to_idx = [f2idx[f] for f in facts2]
        pos_neg_facts_per_report = [None] * n_reports
        for ridx in tqdm(range(n_reports), total=n_reports, mininterval=2):
            pos_fact_idxs1 = pos_neg_facts_per_report1[ridx][0]
            neg_fact_idxs1 = pos_neg_facts_per_report1[ridx][1]
            pos_fact_idxs1 = [idx1_to_idx[i] for i in pos_fact_idxs1]
            neg_fact_idxs1 = [idx1_to_idx[i] for i in neg_fact_idxs1]
            
            pos_fact_idxs2 = pos_neg_facts_per_report2[ridx][0]
            neg_fact_idxs2 = pos_neg_facts_per_report2[ridx][1]            
            pos_fact_idxs2 = [idx2_to_idx[i] for i in pos_fact_idxs2]
            neg_fact_idxs2 = [idx2_to_idx[i] for i in neg_fact_idxs2]
            
            pos_fact_idxs = set()
            neg_fact_idxs = set()
            pos_fact_idxs.update(pos_fact_idxs1)
            pos_fact_idxs.update(pos_fact_idxs2)
            neg_fact_idxs.update(neg_fact_idxs1)
            neg_fact_idxs.update(neg_fact_idxs2)
            pos_neg_facts_per_report[ridx] = (list(pos_fact_idxs), list(neg_fact_idxs))
        prefix = 'mimiccxr_dicom_id_to_all_pos_neg_facts'
        filepath_strings = [
            mimiccxr_report_fact_nli_integrated_data_filepath,
            *gpt4_report_nli_input_output_jsonl_filepaths,
            *integrated_report_facts_jsonl_filepaths,
            mimiccxr_negative_facts_assigned_to_reports_filepath,
            fact_embedding_model_checkpoint_folder_path,
        ]
        facts = all_facts
        embeddings = None

    if embeddings is None:
        fact_encoder = CachedTextEmbeddingExtractor(
            model_name=fact_embedding_model_name,
            model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            batch_size=fact_embedding_batch_size,
            num_workers=fact_embedding_num_workers,
            device='cuda',
        )
        embeddings = fact_encoder.compute_text_embeddings(facts)

    # Detect conflicts
    print_blue('Detecting conflicts...')
    ridxs_with_conflict = []
    for ridx in tqdm(range(n_reports), total=n_reports, mininterval=2):
        pos_fidxs = set(pos_neg_facts_per_report[ridx][0])
        neg_fidxs = set(pos_neg_facts_per_report[ridx][1])
        conflicting_fidxs = pos_fidxs & neg_fidxs
        if len(conflicting_fidxs) > 0:
            # there are conflicting facts
            conflicting_fidxs_list = list(conflicting_fidxs)
            conflicting_fidxs_list.sort()
            ridxs_with_conflict.append({
                'ridx': ridx,
                'conflicting_fidxs': conflicting_fidxs_list,
            })
            # remove negative facts that are also positive
            neg_fidxs -= conflicting_fidxs
            pos_neg_facts_per_report[ridx] = (list(pos_fidxs), list(neg_fidxs))

    if len(ridxs_with_conflict) > 0:
        total_conflicts = sum(len(x['conflicting_fidxs']) for x in ridxs_with_conflict)
        print_red(f'WARNING: {len(ridxs_with_conflict)} reports have conflicting positive and negative facts ({total_conflicts} conflicts).', bold=True)
                  

    # Build dicom_id to pos/neg facts
    detailed_metadata = load_mimiccxr_reports_detailed_metadata()
    dicom_id_view_pos_pairs = detailed_metadata['dicom_id_view_pos_pairs']
    assert len(dicom_id_view_pos_pairs) == n_reports
    dicom_id_to_pos_neg_facts = {}
    for ridx, pairs in enumerate(dicom_id_view_pos_pairs):
        for dicom_id, _ in pairs:
            dicom_id_to_pos_neg_facts[dicom_id] = pos_neg_facts_per_report[ridx]

    # Save output
    output = {
        'facts': facts,
        'embeddings': embeddings,
        'dicom_id_to_pos_neg_facts': dicom_id_to_pos_neg_facts,
        'ridxs_with_conflict': ridxs_with_conflict,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR, prefix=prefix, strings=filepath_strings, force_hashing=True)
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def export_dicom_id_to_positive_negative_facts__replace_embeddings(
    dicom_id_to_pos_neg_facts_filepath,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
):
    print(f'Reading {dicom_id_to_pos_neg_facts_filepath}...')
    data = load_pickle(dicom_id_to_pos_neg_facts_filepath)
    facts = data['facts']
    embeddings = data['embeddings']
    dicom_id_to_pos_neg_facts = data['dicom_id_to_pos_neg_facts']

    # Compute new embeddings
    print_blue('Computing new embeddings...')
    text_encoder = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device='cuda',
    )
    new_embeddings = text_encoder.compute_text_embeddings(facts)
    assert new_embeddings.shape == embeddings.shape

    # Save output
    output = {
        'facts': facts,
        'embeddings': new_embeddings,
        'dicom_id_to_pos_neg_facts': dicom_id_to_pos_neg_facts,
    }
    strings = [
        dicom_id_to_pos_neg_facts_filepath,
        fact_embedding_model_name,
        fact_embedding_model_checkpoint_folder_path,
    ]
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR, prefix=f'mimiccxr_dicom_id_to_pos_neg_facts(num_facts={len(facts)})',
        strings=strings, force_hashing=True)
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def compute_clusters_and_cluster_weights_for_facts(
    dicom_id_to_pos_neg_facts_filepath,
    num_clusters,
):
    print(f'Reading {dicom_id_to_pos_neg_facts_filepath}...')
    data = load_pickle(dicom_id_to_pos_neg_facts_filepath)
    facts = data['facts']
    embeddings = data['embeddings']
    dicom_id_to_pos_neg_facts = data['dicom_id_to_pos_neg_facts']

    num_label_based_facts = len(LABEL_BASED_FACTS)
    assert facts[:num_label_based_facts] == LABEL_BASED_FACTS # ensure label-based facts are at the beginning

    # Compute cluster assignments
    print_blue(f'Computing {num_clusters} clusters...')
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto', max_iter=300)
    cluster_assignments = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    
    # Compute cluster and label weights
    cluster_counts = np.zeros((num_clusters, 2), dtype=np.int32) # pos, neg
    label_counts = np.zeros((num_label_based_facts, 2), dtype=np.int32) # pos, neg
    for (pos_fidxs, neg_fidxs) in tqdm(dicom_id_to_pos_neg_facts.values(), total=len(dicom_id_to_pos_neg_facts), mininterval=2):
        for fidx in pos_fidxs:
            cluster_counts[cluster_assignments[fidx], 0] += 1
            if fidx < num_label_based_facts:
                label_counts[fidx, 0] += 1
        for fidx in neg_fidxs:
            cluster_counts[cluster_assignments[fidx], 1] += 1
            if fidx < num_label_based_facts:
                label_counts[fidx, 1] += 1
    # Make weights inversely proportional to counts
    cluster_weights = np.zeros((num_clusters, 2), dtype=np.float32)
    label_weights = np.zeros((num_label_based_facts, 2), dtype=np.float32)
    for c in range(num_clusters):
        cluster_weights[c, 0] = 1 / (cluster_counts[c, 0] + 1)
        cluster_weights[c, 1] = 1 / (cluster_counts[c, 1] + 1)
    for f in range(num_label_based_facts):
        label_weights[f, 0] = 1 / (label_counts[f, 0] + 1)
        label_weights[f, 1] = 1 / (label_counts[f, 1] + 1)

    # Save output
    output = {
        'dicom_id_to_pos_neg_facts_filepath': dicom_id_to_pos_neg_facts_filepath, # for reference
        'cluster_assignments': cluster_assignments,
        'cluster_centers': cluster_centers,
        'cluster_counts': cluster_counts,
        'cluster_weights': cluster_weights,
        'label_counts': label_counts,
        'label_weights': label_weights,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=LARGE_FAST_CACHE_DIR,
        prefix='mimiccxr_cluster_and_label_weights_for_facts',
        strings=[dicom_id_to_pos_neg_facts_filepath, str(num_clusters)],
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=_Task.choices())
    parser.add_argument('--integrated_report_facts_jsonl_filepaths', type=str, nargs='+')
    parser.add_argument('--integrated_report_facts_jsonl_filepath', type=str)
    parser.add_argument('--gpt4_report_nli_input_output_jsonl_filepaths', type=str, nargs='+')
    parser.add_argument('--report_to_gpt4_qa_filepath', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fact_embedding_model_name', type=str)
    parser.add_argument('--fact_embedding_model_checkpoint_folder_path', type=str)
    parser.add_argument('--fact_embedding_batch_size', type=int, default=32)
    parser.add_argument('--fact_embedding_num_workers', type=int, default=4)
    parser.add_argument('--mlp_batch_size', type=int, default=32)
    parser.add_argument('--mlp_num_workers', type=int, default=4)
    parser.add_argument('--mlp_nli_checkpoint_folder_path', type=str)
    parser.add_argument('--bart_checkpoint_folder_path', type=str)
    parser.add_argument('--bart_batch_size', type=int, default=100)
    parser.add_argument('--bart_num_workers', type=int, default=4)
    parser.add_argument('--bart_max_length', type=int, default=4)
    parser.add_argument('--bart_num_beams', type=int, default=1)
    parser.add_argument('--part_x_of_y', type=int, nargs=2)
    parser.add_argument('--label_based_facts_mlp_nli_softmaxes_filepath', type=str)
    parser.add_argument('--label_based_facts_bart_nli_predictions_filepaths', type=str, nargs='+')
    parser.add_argument('--representative_facts_mlp_nli_softmaxes_filepath', type=str)
    parser.add_argument('--representative_facts_bart_nli_predictions_filepaths', type=str, nargs='+')
    parser.add_argument('--gpt4_label_based_facts_assigned_to_reports_filepath', type=str)
    parser.add_argument('--gpt4_representative_facts_assigned_to_reports_filepath', type=str)
    parser.add_argument('--report_nli_hybrid_metadata_filepath', type=str)
    parser.add_argument('--n_reports', type=int)
    parser.add_argument('--num_kmeans_clusters', type=int)
    parser.add_argument('--num_kmeans_iterations', type=int)
    parser.add_argument('--num_kmedoids_clusters', type=int)
    parser.add_argument('--num_kmedoids_iterations', type=int)
    parser.add_argument('--kmedoids_method', type=str, default='alternate', choices=['pam', 'alternate'])
    parser.add_argument('--union_find_threshold', type=float)
    parser.add_argument('--nearest_k', type=int)
    parser.add_argument('--deduplicated_representative_facts_filepath', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_pos', type=int)
    parser.add_argument('--n_rand', type=int)
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--k_most_similar_representative_facts_per_fact_filepath', type=str)
    parser.add_argument('--assigned_representative_facts_to_reports_filepath', type=str)
    parser.add_argument('--fact_based_integrated_nli_predictions_filepath', type=str)
    parser.add_argument('--label_based_integrated_nli_predictions_filepath', type=str)
    parser.add_argument('--integrated_sentence_facts_jsonl_filepath', type=str)
    parser.add_argument('--mimiccxr_report_fact_nli_integrated_data_filepath', type=str)
    parser.add_argument('--pos_neg_facts_mode', type=str, choices=['label_based', 'fact_based', 'all'])
    parser.add_argument('--dicom_id_to_pos_neg_facts_filepath', type=str)
    parser.add_argument('--mlp_nli_entailment_threshold', type=float)
    parser.add_argument('--max_negative_facts_per_report', type=int)
    parser.add_argument('--mimiccxr_negative_facts_assigned_to_reports_filepath', type=str)
    args = parser.parse_args()

    if args.task == _Task.ASSIGN_GPT4_LABEL_BASED_FACTS_TO_REPORTS:
        assert len(args.integrated_report_facts_jsonl_filepaths) > 0
        assert len(args.gpt4_report_nli_input_output_jsonl_filepaths) > 0
        assign_gpt4_label_based_facts_to_reports(
            integrated_report_facts_jsonl_filepaths=args.integrated_report_facts_jsonl_filepaths,
            gpt4_report_nli_input_output_jsonl_filepaths=args.gpt4_report_nli_input_output_jsonl_filepaths,
        )
    elif args.task == _Task.ASSIGN_GPT4_REPRESENTATIVE_FACTS_TO_REPORTS:
        assert len(args.integrated_report_facts_jsonl_filepaths) > 0
        assert len(args.gpt4_report_nli_input_output_jsonl_filepaths) > 0
        assert args.deduplicated_representative_facts_filepath is not None
        assign_gpt4_representative_facts_to_reports(
            integrated_report_facts_jsonl_filepaths=args.integrated_report_facts_jsonl_filepaths,
            gpt4_report_nli_input_output_jsonl_filepaths=args.gpt4_report_nli_input_output_jsonl_filepaths,
            representative_facts_filepath=args.deduplicated_representative_facts_filepath,
        )
    elif args.task == _Task.COMPUTE_MLP_LABEL_BASED_NLI_SOFTMAXES:
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.report_to_gpt4_qa_filepath is not None
        assert args.fact_embedding_model_name is not None
        assert args.fact_embedding_model_checkpoint_folder_path is not None
        assert args.mlp_nli_checkpoint_folder_path is not None
        compute_mlp_label_based_nli_softmaxes(
            report_to_gpt4_qa_filepath=args.report_to_gpt4_qa_filepath,
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            device=args.device,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
            mlp_batch_size=args.mlp_batch_size,
            mlp_num_workers=args.mlp_num_workers,
            mlp_nli_checkpoint_folder_path=args.mlp_nli_checkpoint_folder_path,
        )
    elif args.task == _Task.COMPUTE_MLP_FACT_BASED_NLI_SOFTMAXES:
        assert args.deduplicated_representative_facts_filepath is not None
        assert args.assigned_representative_facts_to_reports_filepath is not None
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.report_to_gpt4_qa_filepath is not None
        assert args.fact_embedding_model_name is not None
        assert args.fact_embedding_model_checkpoint_folder_path is not None
        assert args.mlp_nli_checkpoint_folder_path is not None
        compute_mlp_fact_based_nli_softmaxes(
            deduplicated_representative_facts_filepath=args.deduplicated_representative_facts_filepath,
            assigned_representative_facts_to_reports_filepath=args.assigned_representative_facts_to_reports_filepath,
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            report_to_gpt4_qa_filepath=args.report_to_gpt4_qa_filepath,
            device=args.device,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
            mlp_batch_size=args.mlp_batch_size,
            mlp_num_workers=args.mlp_num_workers,
            mlp_nli_checkpoint_folder_path=args.mlp_nli_checkpoint_folder_path,
        )
    elif args.task == _Task.COMPUTE_BART_LABEL_BASED_NLI_PREDICTIONS:
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.report_to_gpt4_qa_filepath is not None
        assert args.bart_checkpoint_folder_path is not None
        compute_BART_label_based_nli_predictions(
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            report_to_gpt4_qa_filepath=args.report_to_gpt4_qa_filepath,
            bart_checkpoint_folder_path=args.bart_checkpoint_folder_path,
            device=args.device,
            bart_batch_size=args.bart_batch_size,
            bart_num_workers=args.bart_num_workers,
            bart_max_length=args.bart_max_length,
            bart_num_beams=args.bart_num_beams,
            part_x_of_y=args.part_x_of_y,
        )
    elif args.task == _Task.COMPUTE_BART_FACT_BASED_NLI_PREDICTIONS:
        assert args.deduplicated_representative_facts_filepath is not None
        assert args.assigned_representative_facts_to_reports_filepath is not None
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.report_to_gpt4_qa_filepath is not None
        assert args.bart_checkpoint_folder_path is not None
        compute_BART_fact_based_nli_predictions(
            deduplicated_representative_facts_filepath=args.deduplicated_representative_facts_filepath,
            assigned_representative_facts_to_reports_filepath=args.assigned_representative_facts_to_reports_filepath,
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            report_to_gpt4_qa_filepath=args.report_to_gpt4_qa_filepath,
            bart_checkpoint_folder_path=args.bart_checkpoint_folder_path,
            device=args.device,
            bart_batch_size=args.bart_batch_size,
            bart_num_workers=args.bart_num_workers,
            bart_max_length=args.bart_max_length,
            bart_num_beams=args.bart_num_beams,
            part_x_of_y=args.part_x_of_y,
        )
    elif args.task == _Task.COMPUTE_HYBRID_LABEL_BASED_NLI_PREDICTIONS:
        assert args.gpt4_label_based_facts_assigned_to_reports_filepath is not None
        assert args.label_based_facts_bart_nli_predictions_filepaths is not None
        assert args.label_based_facts_mlp_nli_softmaxes_filepath is not None
        assert args.report_nli_hybrid_metadata_filepath is not None
        assert args.n_reports is not None
        integrate_label_based_nli_predictions(
            gpt4_label_based_facts_assigned_to_reports_filepath=args.gpt4_label_based_facts_assigned_to_reports_filepath,
            label_based_facts_bart_nli_predictions_filepaths=args.label_based_facts_bart_nli_predictions_filepaths,
            label_based_facts_mlp_nli_softmaxes_filepath=args.label_based_facts_mlp_nli_softmaxes_filepath,
            report_nli_hybrid_metadata_filepath=args.report_nli_hybrid_metadata_filepath,
            n_reports=args.n_reports,
        )
    elif args.task == _Task.COMPUTE_HYBRID_FACT_BASED_NLI_PREDICTIONS:
        assert args.deduplicated_representative_facts_filepath is not None
        assert args.assigned_representative_facts_to_reports_filepath is not None
        assert args.gpt4_representative_facts_assigned_to_reports_filepath is not None
        assert args.representative_facts_bart_nli_predictions_filepaths is not None
        assert args.representative_facts_mlp_nli_softmaxes_filepath is not None
        assert args.report_nli_hybrid_metadata_filepath is not None
        integrate_fact_based_nli_predictions(
            assigned_representative_facts_to_reports_filepath=args.assigned_representative_facts_to_reports_filepath,
            deduplicated_representative_facts_filepath=args.deduplicated_representative_facts_filepath,
            gpt4_representative_facts_assigned_to_reports_filepath=args.gpt4_representative_facts_assigned_to_reports_filepath,
            representative_facts_bart_nli_predictions_filepaths=args.representative_facts_bart_nli_predictions_filepaths,
            representative_facts_mlp_nli_softmaxes_filepath=args.representative_facts_mlp_nli_softmaxes_filepath,
            report_nli_hybrid_metadata_filepath=args.report_nli_hybrid_metadata_filepath,
        )
    elif args.task == _Task.FIND_REPRESENTATIVE_FACTS:
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.fact_embedding_model_name is not None
        assert args.fact_embedding_model_checkpoint_folder_path is not None
        assert args.num_kmeans_clusters is not None
        assert args.num_kmeans_iterations is not None
        assert args.num_kmedoids_clusters is not None
        assert args.num_kmedoids_iterations is not None
        assert args.kmedoids_method is not None
        assert args.union_find_threshold is not None
        assert args.nearest_k is not None
        find_representative_facts(
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
            device=args.device,
            num_kmeans_clusters=args.num_kmeans_clusters,
            num_kmeans_iterations=args.num_kmeans_iterations,
            num_kmedoids_clusters=args.num_kmedoids_clusters,
            num_kmedoids_iterations=args.num_kmedoids_iterations,
            kmedoids_method=args.kmedoids_method,
            union_find_threshold=args.union_find_threshold,
            nearest_k=args.nearest_k,
        )
    elif args.task == _Task.FIND_K_MOST_SIMILAR_REPRESENTATIVE_FACTS_FOR_EACH_FACT:
        assert args.deduplicated_representative_facts_filepath is not None
        assert args.nearest_k is not None
        find_k_most_similar_representative_facts_for_each_fact(
            deduplicated_representative_facts_filepath=args.deduplicated_representative_facts_filepath,
            nearest_k=args.nearest_k,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.task == _Task.ASSIGN_REPRESENTATIVE_FACTS_TO_REPORTS:
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.deduplicated_representative_facts_filepath is not None
        assert args.k_most_similar_representative_facts_per_fact_filepath is not None
        assert args.n_pos is not None
        assert args.n_rand is not None
        assert args.num_processes is not None
        assign_representative_facts_to_reports(
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            deduplicated_representative_facts_filepath=args.deduplicated_representative_facts_filepath,
            k_most_similar_representative_facts_per_fact_filepath=args.k_most_similar_representative_facts_per_fact_filepath,
            n_pos=args.n_pos,
            n_rand=args.n_rand,
            num_processes=args.num_processes,
        )
    elif args.task == _Task.INTEGRATE_AND_EXPORT_ALL_DATA:
        assert args.fact_based_integrated_nli_predictions_filepath is not None
        assert args.label_based_integrated_nli_predictions_filepath is not None
        assert args.assigned_representative_facts_to_reports_filepath is not None
        assert args.deduplicated_representative_facts_filepath is not None
        assert args.gpt4_report_nli_input_output_jsonl_filepaths is not None
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.integrated_sentence_facts_jsonl_filepath is not None
        assert args.fact_embedding_model_name is not None
        assert args.fact_embedding_model_checkpoint_folder_path is not None
        assert args.fact_embedding_batch_size is not None
        assert args.fact_embedding_num_workers is not None
        integrate_and_export_all_data(
            fact_based_integrated_nli_predictions_filepath=args.fact_based_integrated_nli_predictions_filepath,
            label_based_integrated_nli_predictions_filepath=args.label_based_integrated_nli_predictions_filepath,
            assigned_representative_facts_to_reports_filepath=args.assigned_representative_facts_to_reports_filepath,
            deduplicated_representative_facts_filepath=args.deduplicated_representative_facts_filepath,
            gpt4_report_nli_input_output_jsonl_filepaths=args.gpt4_report_nli_input_output_jsonl_filepaths,
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            integrated_sentence_facts_jsonl_filepath=args.integrated_sentence_facts_jsonl_filepath,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
        )
    elif args.task == _Task.EXPORT_DICOM_ID_TO_POSITIVE_NEGATIVE_FACTS:
        assert args.pos_neg_facts_mode is not None
        assert args.mimiccxr_report_fact_nli_integrated_data_filepath is not None
        export_dicom_id_to_positive_negative_facts(
            mode=args.pos_neg_facts_mode,
            mimiccxr_report_fact_nli_integrated_data_filepath=args.mimiccxr_report_fact_nli_integrated_data_filepath,
            gpt4_report_nli_input_output_jsonl_filepaths=args.gpt4_report_nli_input_output_jsonl_filepaths,
            integrated_report_facts_jsonl_filepaths=args.integrated_report_facts_jsonl_filepaths,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
        )
    elif args.task == _Task.EXPORT_DICOM_ID_TO_POSITIVE_NEGATIVE_FACTS__REPLACE_EMBEDDINGS:
        export_dicom_id_to_positive_negative_facts__replace_embeddings(
            dicom_id_to_pos_neg_facts_filepath=args.dicom_id_to_pos_neg_facts_filepath,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
        )
    elif args.task == _Task.SAMPLE_NEGATIVE_FACTS_PER_REPORT_WITH_FACT_EMBEDDINGS_AND_MLP_NLI:
        assert args.integrated_report_facts_jsonl_filepath is not None
        assert args.fact_embedding_model_name is not None
        assert args.fact_embedding_model_checkpoint_folder_path is not None
        assert args.fact_embedding_batch_size is not None
        assert args.fact_embedding_num_workers is not None
        assert args.mlp_batch_size is not None
        assert args.mlp_num_workers is not None
        assert args.mlp_nli_checkpoint_folder_path is not None
        assert args.mlp_nli_entailment_threshold is not None
        assert args.num_kmeans_clusters is not None
        assert args.max_negative_facts_per_report is not None
        sample_negative_facts_per_report_with_fact_embeddings_and_mlp_nli(
            integrated_report_facts_jsonl_filepath=args.integrated_report_facts_jsonl_filepath,
            device=args.device,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
            mlp_batch_size=args.mlp_batch_size,
            mlp_num_workers=args.mlp_num_workers,
            mlp_nli_checkpoint_folder_path=args.mlp_nli_checkpoint_folder_path,
            mlp_nli_entailment_threshold=args.mlp_nli_entailment_threshold,
            num_clusters=args.num_kmeans_clusters,
            max_negative_facts_per_report=args.max_negative_facts_per_report,
        )
    elif args.task == _Task.EXPORT_DICOM_ID_TO_POSITIVE_NEGATIVE_FACTS__IMPROVED_MLP_NLI_BASED_NEGATIVE_SAMPLING:
        assert args.pos_neg_facts_mode is not None
        assert args.mimiccxr_report_fact_nli_integrated_data_filepath is not None
        assert args.gpt4_report_nli_input_output_jsonl_filepaths is not None
        assert args.integrated_report_facts_jsonl_filepaths is not None
        assert args.mimiccxr_negative_facts_assigned_to_reports_filepath is not None
        assert args.fact_embedding_model_name is not None
        export_dicom_id_to_positive_negative_facts__improved_mlp_nli_based_negative_sampling(
            mode=args.pos_neg_facts_mode,
            mimiccxr_report_fact_nli_integrated_data_filepath=args.mimiccxr_report_fact_nli_integrated_data_filepath,
            gpt4_report_nli_input_output_jsonl_filepaths=args.gpt4_report_nli_input_output_jsonl_filepaths,
            integrated_report_facts_jsonl_filepaths=args.integrated_report_facts_jsonl_filepaths,
            mimiccxr_negative_facts_assigned_to_reports_filepath=args.mimiccxr_negative_facts_assigned_to_reports_filepath,
            fact_embedding_model_name=args.fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=args.fact_embedding_batch_size,
            fact_embedding_num_workers=args.fact_embedding_num_workers,
        )
    elif args.task == _Task.COMPUTE_CLUSTERS_AND_CLUSTER_WEIGHTS_FOR_FACTS:
        assert args.num_kmeans_clusters is not None
        assert args.dicom_id_to_pos_neg_facts_filepath is not None
        compute_clusters_and_cluster_weights_for_facts(
            dicom_id_to_pos_neg_facts_filepath=args.dicom_id_to_pos_neg_facts_filepath,
            num_clusters=args.num_kmeans_clusters,
        )
    else:
        raise ValueError(f'Invalid task: {args.task}')
    
    print_blue(f'Done!', bold=True)

if __name__ == '__main__':
    main()