import os
import argparse
import random
import numpy as np
from nltk import sent_tokenize
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from medvqa.datasets.iuxray import IUXRAY_REPORTS_MIN_JSON_PATH
from medvqa.eval_report_gen_metrics import (
    compute_AUC,
    compute_chest_imagenome_gold_contradiction_matrix,
    compute_chest_imagenome_gold_relevance_matrix,
    compute_gold_accuracy_matrix,
    compute_mean_average_accuracy_at_k,
    compute_mean_contradictions_at_k,
)
from medvqa.evaluation.ranking_evaluation_utils import load_mimiccxr_custom_radiologist_annotations
from medvqa.utils.logging_utils import ANSI_MAGENTA_BOLD, ANSI_ORANGE_BOLD, ANSI_RESET, setup_logging
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor, SupportedHuggingfaceMedicalBERTModels
from medvqa.utils.common import CACHE_DIR, ChoiceEnum, get_timestamp, parsed_args_to_dict
from medvqa.utils.files_utils import load_json, load_pickle, save_pickle
from medvqa.utils.math_utils import (
    rank_vectors_by_dot_product,
)
from medvqa.utils.metrics_utils import jaccard_between_dicts
import logging

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable parallelism for tokenizers to avoid warnings

setup_logging()
logger = logging.getLogger(__name__)

class EvaluationModes(ChoiceEnum):
    MIMICCXR_RADIOLOGIST_ANNOTATIONS = 'mimiccxr_radiologist_annotations'
    IUXRAY_WITH_AUTOMATIC_LABELERS = 'iuxray_with_automatic_labelers' # chexpert + chexbert + radgraph
    CHEST_IMAGENOME_GOLD = 'chest_imagenome_gold'
    PADCHEST_GR_MSCXR_CUSTOM = 'padchest_gr_mscxr_custom'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_mode', type=str, required=True, choices=EvaluationModes.get_choices())
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all() + ['CheXbert'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['cosine', 'dot_product'])
    parser.add_argument('--average_token_embeddings', action='store_true', default=False)
    parser.add_argument('--chest_imagenome_sentence2labels_gold_filepath', type=str, default=None)
    parser.add_argument('--revised_groundings_filepath', type=str, default=None)
    parser.add_argument('--num_query_samples', type=int, default=None)
    parser.add_argument('--save_worst_queries_for_inspection', action='store_true', default=False,
                        help='If True, saves the worst queries for inspection in a pickle file.')
    return parser.parse_args(args=args)

def _load_iuxray_sentences():
    logger.info(f'Loading iuxray reports from {IUXRAY_REPORTS_MIN_JSON_PATH}')
    reports = load_json(IUXRAY_REPORTS_MIN_JSON_PATH)
    sentences = set()
    for r in reports.values():
        findings = r['findings']
        impression = r['impression']
        for x in (findings, impression):
            if x:
                for s in sent_tokenize(x):
                    s = ' '.join(s.split()) # Remove extra spaces
                    sentences.add(s)
    sentences = list(sentences)
    sentences.sort(key=lambda x: (len(x), x)) # Sort by length and then alphabetically
    sentences = [s for s in sentences if any(c.isalpha() for c in s)] # Remove sentences without any alphabetic character
    logger.info(f'Number of sentences: {len(sentences)}')
    logger.info(f'Shortest sentence: {sentences[0]}')
    logger.info(f'Longest sentence: {sentences[-1]}')
    return sentences

def _load_mimiccxr_sentences_and_radiologist_based_relevant_sentences():
    save_path = os.path.join(CACHE_DIR, 'mimiccxr_sentences_and_relevant.pkl')
    if os.path.exists(save_path):
        logger.info(f'Loading mimiccxr sentences and relevant sentences from {save_path}')
        return load_pickle(save_path)
    
    sentences, labels = load_mimiccxr_custom_radiologist_annotations()
    n = len(sentences)
    relevant_sentences = [set() for _ in range(n)]

    def _a_entails_b(a, b):
        for i in range(len(b)):
            if b[i] == -2: # nan
                continue
            if a[i] != b[i]:
                return False
        return True

    for i in tqdm(range(n), mininterval=2):
        for j in range(i+1, n):
            if _a_entails_b(labels[i], labels[j]) or _a_entails_b(labels[j], labels[i]):
                relevant_sentences[i].add(j)
                relevant_sentences[j].add(i)

    logger.info(f'Saving mimiccxr sentences and relevant sentences to {save_path}')
    output = {
        'sentences': sentences,
        'relevant_sentences': relevant_sentences,
    }
    save_pickle(output, save_path)
    return output

def _load_iuxray_sentences_and_automatic_labeler_based_relevant_sentences(thr1=0.4, thr2=0.2):
    save_path = os.path.join(CACHE_DIR, f'iuxray_sentences_and_relevant(thr1={thr1},thr2={thr2}).pkl')
    if os.path.exists(save_path):
        logger.info(f'Loading iuxray sentences and relevant sentences from {save_path}')
        return load_pickle(save_path)

    sentences = _load_iuxray_sentences()

    logger.info('Loading CheXpert labeler')
    from medvqa.metrics.medical.chexpert import ChexpertLabeler
    chexpert_labeler = ChexpertLabeler()
    chexpert_labels = chexpert_labeler.get_labels(sentences, update_cache_on_disk=True)

    logger.info('Loading CheXbert labeler')
    from medvqa.metrics.medical.chexbert import CheXbertLabeler
    chexbert_labeler = CheXbertLabeler()
    chexbert_labels = chexbert_labeler.get_labels(sentences, update_cache_on_disk=True)

    logger.info('Loading RadGraph labeler')
    from medvqa.metrics.medical.radgraph import RadGraphLabeler
    radgraph_labeler = RadGraphLabeler()
    radgraph_labels = radgraph_labeler.get_labels(sentences, update_cache_on_disk=True)

    # Obtain relevant sentences for each sentence
    relevant_sentences = [set() for _ in range(len(sentences))]
    for i in tqdm(range(len(sentences)), mininterval=2):
        if len(radgraph_labels[i]) == 0: # Skip sentences without any radgraph labels
            continue
        for j in range(i+1, len(sentences)):
            js = jaccard_between_dicts(radgraph_labels[i], radgraph_labels[j])
            if js >= thr1 or (js >= thr2 and (np.all(chexpert_labels[i] == chexpert_labels[j]) or \
                np.all(chexbert_labels[i] == chexbert_labels[j]))):
                relevant_sentences[i].add(j)
                relevant_sentences[j].add(i)
    logger.info(f'Saving iuxray sentences and relevant sentences to {save_path}')
    output = {
        'sentences': sentences,
        'relevant_sentences': relevant_sentences,
    }
    save_pickle(output, save_path)
    return output

def _compute_AUC(sorted_idxs, relevant_idxs, query_idx):
    # Compute AUC
    assert query_idx not in sorted_idxs
    assert query_idx not in relevant_idxs
    assert len(relevant_idxs) > 0
    assert len(sorted_idxs) > len(relevant_idxs)
    n_relevant = len(relevant_idxs)
    n_irrelevant = len(sorted_idxs) - n_relevant
    n_pairs = n_relevant * n_irrelevant
    n_wrong_pairs = 0
    n_irrelevant_below = 0
    relevant_found = 0
    for idx in sorted_idxs:
        if idx in relevant_idxs:
            n_wrong_pairs += n_irrelevant_below
            relevant_found += 1
        else:
            n_irrelevant_below += 1
    assert relevant_found == n_relevant
    AUC = 1 - (n_wrong_pairs / n_pairs)
    return AUC


def compute_padchest_gr_mscxr_custom_evaluation_data(
    revised_groundings_file: str,
    num_query_samples: int = 300,
    cxr_bert_model_name: str = 'microsoft/BiomedVLP-CXR-BERT-specialized',
    embedding_batch_size: int = 32,
    device: str = 'cuda',
):
    """
    Generates evaluation data for phrase ranking by comparing query phrases against candidate phrases.

    This function first gathers a set of unique query phrases (sampled) and a comprehensive list
    of candidate phrases from PadChest GR and MS-CXR datasets. It then precomputes
    embeddings (CXR-BERT, CheXbert) and RadGraph labels for all relevant phrases (union of
    queries and candidates) to optimize metric calculation. Finally, it computes several
    similarity/relevance scores (BLEU-1, Rouge-L, Levenshtein, BERTScore F1,
    CXR-BERT cosine, CheXbert cosine, RadGraph F1) for each query-candidate pair.

    The output is structured to facilitate the evaluation of sentence embedding models or
    other ranking algorithms on this medical phrase ranking task.

    Args:
        revised_groundings_file (str): Path to a pickle file containing revised groundings.
            Each entry in the loaded list is expected to have a 'phrase2locations' dictionary,
            from which query phrases are extracted.
        num_query_samples (int): Number of query phrases to sample for evaluation. If the
            total number of unique query phrases from revised groundings is less than this,
            all unique queries are used.
        cxr_bert_model_name (str): Name of the CXR-BERT model to use for embeddings.
        embedding_batch_size (int): Batch size for computing text embeddings.
        device (str): Device to use for embedding computations (e.g., 'cuda', 'cpu').

    Returns:
        dict: A dictionary with the following structure:
            {
                'query_phrases': [list of sampled query phrases (str)],
                'candidate_phrases': [list of all unique candidate phrases (str)],
                'metrics': {
                    'bleu1': np.ndarray of shape (num_query_samples, num_candidates),
                    'rouge_l': np.ndarray of shape (num_query_samples, num_candidates),
                    'levenshtein': np.ndarray of shape (num_query_samples, num_candidates),
                    'bertscore_f1': np.ndarray of shape (num_query_samples, num_candidates),
                    'cxr_bert_cosine': np.ndarray of shape (num_query_samples, num_candidates),
                    'chexbert_cosine': np.ndarray of shape (num_query_samples, num_candidates),
                    'radgraph_f1': np.ndarray of shape (num_query_samples, num_candidates),
                }
            }
    """

    # Imports
    from medvqa.datasets.padchest import get_padchest_gr_phrase_groundings
    from medvqa.datasets.ms_cxr import get_ms_cxr_dicom_id_2_phrases_and_bboxes
    from medvqa.metrics.medical.chexbert import CheXbertLabeler
    from medvqa.metrics.medical.radgraph import RadGraphLabeler
    from medvqa.metrics.nlp.bertscore import BertScore
    from medvqa.utils.metrics_utils import f1_between_dicts
    from rouge_score import rouge_scorer
    from sklearn.metrics.pairwise import cosine_similarity # For optimized cosine similarity
    import Levenshtein # For Levenshtein distance
    import nltk

    # STEP 1: Load phrases from PadChest GR and MS-CXR datasets to form candidate phrases
    logger.info("Loading phrases from PadChest GR and MS-CXR datasets...")
    padchest_gr_phrase_groundings = get_padchest_gr_phrase_groundings()
    ms_cxr_dicom_id_2_phrases_and_bboxes = get_ms_cxr_dicom_id_2_phrases_and_bboxes()
    logger.info(f'len(padchest_gr_phrase_groundings): {len(padchest_gr_phrase_groundings)}')
    logger.info(f'len(ms_cxr_dicom_id_2_phrases_and_bboxes): {len(ms_cxr_dicom_id_2_phrases_and_bboxes)}')

    candidate_phrases_set = set()
    for item in padchest_gr_phrase_groundings:
        candidate_phrases_set.add(item['phrase'])
    for item in ms_cxr_dicom_id_2_phrases_and_bboxes.values():
        phrases = item[0] # Assuming item[0] is the list of phrases
        candidate_phrases_set.update(phrases)
    
    candidate_phrases = sorted(list(candidate_phrases_set))
    logger.info(f'Number of unique candidate phrases: {len(candidate_phrases)}')

    # STEP 2: Load revised groundings and sample query phrases
    logger.info(f"Loading revised groundings from {revised_groundings_file}...")
    revised_data = load_pickle(revised_groundings_file)
    logger.info(f'Loaded {len(revised_data)} revised groundings.')

    unique_query_phrases_from_revised_set = set()
    for item in revised_data:
        unique_query_phrases_from_revised_set.update(item['phrase2locations'].keys())
    
    unique_query_phrases_from_revised = sorted(list(unique_query_phrases_from_revised_set))
    logger.info(f'Number of unique query phrases from revised groundings: {len(unique_query_phrases_from_revised)}')

    actual_num_query_samples = min(num_query_samples, len(unique_query_phrases_from_revised))
    if len(unique_query_phrases_from_revised) < num_query_samples:
        logger.warning(
            f"Requested {num_query_samples} query samples, but only "
            f"{len(unique_query_phrases_from_revised)} unique query phrases available. "
            f"Using {actual_num_query_samples} samples."
        )
    
    sampled_query_phrases = sorted(random.sample(unique_query_phrases_from_revised, actual_num_query_samples))
    logger.info(f'Number of sampled query phrases for evaluation: {len(sampled_query_phrases)}')

    # STEP 3: Consolidate all unique phrases (queries and candidates) for precomputation
    all_distinct_phrases_set = set(sampled_query_phrases).union(set(candidate_phrases))
    all_distinct_phrases = sorted(list(all_distinct_phrases_set))
    phrase_to_overall_idx = {phrase: i for i, phrase in enumerate(all_distinct_phrases)}
    logger.info(f"Total number of distinct phrases for precomputation: {len(all_distinct_phrases)}")

    # STEP 4: Precompute embeddings and RadGraph labels for ALL distinct phrases
    # This is done once to be efficient.
    logger.info(f"Precomputing CXR-BERT embeddings for {len(all_distinct_phrases)} distinct phrases...")
    cxr_bert_extractor = CachedTextEmbeddingExtractor(
        model_name=cxr_bert_model_name,
        device=device,
        batch_size=embedding_batch_size,
    )
    all_cxr_bert_embs = cxr_bert_extractor.compute_text_embeddings(all_distinct_phrases)
    logger.info(f"Shape of all_cxr_bert_embs: {all_cxr_bert_embs.shape if all_cxr_bert_embs is not None else 'None'}")

    logger.info(f"Precomputing CheXbert embeddings for {len(all_distinct_phrases)} distinct phrases...")
    chexbert_labeler = CheXbertLabeler(verbose=False)
    all_chexbert_embs = chexbert_labeler.get_embeddings(all_distinct_phrases)
    logger.info(f"Shape of all_chexbert_embs: {all_chexbert_embs.shape if all_chexbert_embs is not None else 'None'}")

    logger.info(f"Precomputing RadGraph labels for {len(all_distinct_phrases)} distinct phrases...")
    radgraph_labeler = RadGraphLabeler(verbose=False)
    all_radgraph_labels = radgraph_labeler.get_labels(all_distinct_phrases, update_cache_on_disk=True)
    logger.info(f"Computed {len(all_radgraph_labels)} RadGraph labels.")

    # STEP 5: Prepare metric arrays and BERTScore instance
    num_queries = len(sampled_query_phrases)
    num_candidates = len(candidate_phrases)
    
    metrics_results = {
        'bleu1': np.zeros((num_queries, num_candidates), dtype=np.float32),
        'rouge_l': np.zeros((num_queries, num_candidates), dtype=np.float32),
        'levenshtein': np.zeros((num_queries, num_candidates), dtype=np.float32),
        'bertscore_f1': np.zeros((num_queries, num_candidates), dtype=np.float32),
        'cxr_bert_cosine': np.zeros((num_queries, num_candidates), dtype=np.float32),
        'chexbert_cosine': np.zeros((num_queries, num_candidates), dtype=np.float32),
        'radgraph_f1': np.zeros((num_queries, num_candidates), dtype=np.float32),
    }

    bertscore_metric = BertScore(device=device) # Initialize BERTScore scorer
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # Initialize ROUGE-L scorer

    # STEP 6: Compute metrics
    # Optimized Cosine Similarities (CXR-BERT and CheXbert) using precomputed embeddings
    # Get the subset of embeddings corresponding to query phrases and candidate phrases
    query_indices_in_overall = [phrase_to_overall_idx[p] for p in sampled_query_phrases]
    candidate_indices_in_overall = [phrase_to_overall_idx[p] for p in candidate_phrases]

    if all_cxr_bert_embs is not None and len(all_cxr_bert_embs) > 0:
        query_cxr_bert_embs_matrix = all_cxr_bert_embs[query_indices_in_overall]
        candidate_cxr_bert_embs_matrix = all_cxr_bert_embs[candidate_indices_in_overall]
        metrics_results['cxr_bert_cosine'] = cosine_similarity(
            query_cxr_bert_embs_matrix, candidate_cxr_bert_embs_matrix
        )
    logger.info("Computed CXR-BERT cosine similarities.")

    if all_chexbert_embs is not None and len(all_chexbert_embs) > 0:
        query_chexbert_embs_matrix = all_chexbert_embs[query_indices_in_overall]
        candidate_chexbert_embs_matrix = all_chexbert_embs[candidate_indices_in_overall]
        metrics_results['chexbert_cosine'] = cosine_similarity(
            query_chexbert_embs_matrix, candidate_chexbert_embs_matrix
        )
    logger.info("Computed CheXbert cosine similarities.")

    # Pairwise metrics (BLEU, ROUGE-L, Levenshtein, BERTScore, RadGraph F1)
    # These are computed per query against all candidates.
    for i, query_phrase in tqdm(
        enumerate(sampled_query_phrases), total=num_queries, mininterval=2,
        desc="Computing metrics for each query phrase"
    ):  
        # BERTScore F1: query vs. all candidates
        # For BERTScore, predictions are candidates, references is the query repeated.
        if candidate_phrases: # Ensure candidate_phrases is not empty
            _, _, f1_scores_bert = bertscore_metric([query_phrase] * num_candidates, candidate_phrases)
            metrics_results['bertscore_f1'][i, :] = f1_scores_bert.cpu().numpy()
        
        # RadGraph F1: query vs. all candidates
        query_radgraph_label_idx = phrase_to_overall_idx[query_phrase]
        query_label_radgraph = all_radgraph_labels[query_radgraph_label_idx]

        for j, cand_phrase in enumerate(candidate_phrases):
            # String-based metrics
            # BLEU-1: query vs. candidate
            ref_tokens = [nltk.word_tokenize(query_phrase.lower())]
            hyp_tokens = nltk.word_tokenize(cand_phrase.lower())
            bleu1_score = nltk.translate.bleu_score.sentence_bleu(
                ref_tokens, hyp_tokens, weights=(1.0,) # BLEU-1
            )
            metrics_results['bleu1'][i, j] = bleu1_score

            # ROUGE-L: query vs. candidate
            rouge_scores = rouge_l_scorer.score(query_phrase, cand_phrase)
            metrics_results['rouge_l'][i, j] = rouge_scores['rougeL'].fmeasure

            # Levenshtein similarity: query vs. candidate
            lev_dist = Levenshtein.distance(query_phrase, cand_phrase)
            max_len = max(len(query_phrase), len(cand_phrase))
            lev_sim = 1.0 if max_len == 0 else (1.0 - lev_dist / max_len)
            metrics_results['levenshtein'][i, j] = lev_sim

            # RadGraph F1 (continued): query vs. candidate
            cand_radgraph_label_idx = phrase_to_overall_idx[cand_phrase]
            cand_label_radgraph = all_radgraph_labels[cand_radgraph_label_idx]
            metrics_results['radgraph_f1'][i, j] = f1_between_dicts(
                query_label_radgraph, cand_label_radgraph
            )
            
    logger.info("Finished computing all metrics.")

    return {
        'query_phrases': sampled_query_phrases,
        'candidate_phrases': candidate_phrases,
        'metrics': metrics_results,
    }



def evaluate(
    evaluation_mode,
    model_name,
    device,
    batch_size,
    num_workers,
    model_checkpoint_folder_path,
    distance_metric,
    average_token_embeddings,
    chest_imagenome_sentence2labels_gold_filepath,
    revised_groundings_filepath,
    num_query_samples,
    save_worst_queries_for_inspection,
):
    if evaluation_mode == EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS.value:
        tmp = _load_mimiccxr_sentences_and_radiologist_based_relevant_sentences()
        sentences, relevant_sentences = tmp['sentences'], tmp['relevant_sentences']
        logger.info('len(sentences):', len(sentences))
        logger.info('len(relevant_sentences):', len(relevant_sentences))
    
    elif evaluation_mode == EvaluationModes.IUXRAY_WITH_AUTOMATIC_LABELERS.value:
        tmp = _load_iuxray_sentences_and_automatic_labeler_based_relevant_sentences()
        sentences, relevant_sentences = tmp['sentences'], tmp['relevant_sentences']
        logger.info('len(sentences):', len(sentences))
        logger.info('len(relevant_sentences):', len(relevant_sentences))
    
    elif evaluation_mode == EvaluationModes.CHEST_IMAGENOME_GOLD.value:
        assert chest_imagenome_sentence2labels_gold_filepath is not None
        logger.info(f'Loading sentence2labels_gold from {chest_imagenome_sentence2labels_gold_filepath}')
        sentence2labels_gold = load_pickle(chest_imagenome_sentence2labels_gold_filepath)
        sentences = sentence2labels_gold['phrases']
        observation_labels = sentence2labels_gold['observation_labels']
        anatomy_labels = sentence2labels_gold['anatomy_labels']
        labels =  np.concatenate([observation_labels, anatomy_labels], axis=1)
        logger.info(f'Number of sentences: {len(sentences)}')
        logger.info(f'label.shape: {labels.shape}')
        gold_relevance_matrix = compute_chest_imagenome_gold_relevance_matrix(observation_labels, anatomy_labels)
        gold_accuracy_matrix = compute_gold_accuracy_matrix(labels)
        gold_contradiction_matrix = compute_chest_imagenome_gold_contradiction_matrix(observation_labels)
    
    elif evaluation_mode == EvaluationModes.PADCHEST_GR_MSCXR_CUSTOM.value:
        assert revised_groundings_filepath is not None
        assert num_query_samples is not None
        eval_data_save_path = os.path.join(CACHE_DIR, f'padchest_gr_mscxr_custom_eval_data(num_query_samples={num_query_samples}).pkl')
        if os.path.exists(eval_data_save_path):
            logger.info(f'Loading evaluation data from {eval_data_save_path}')
            eval_data = load_pickle(eval_data_save_path)
        else:
            logger.info(f'Computing evaluation data for PadChest GR and MS-CXR custom phrases')
            # Compute evaluation data
            eval_data = compute_padchest_gr_mscxr_custom_evaluation_data(
                revised_groundings_file=revised_groundings_filepath,
                num_query_samples= num_query_samples,            
            )
            save_pickle(eval_data, eval_data_save_path)
            logger.info(f'Saved evaluation data to {eval_data_save_path}')
        query_phrases = eval_data['query_phrases']
        candidate_phrases = eval_data['candidate_phrases']
        metrics = eval_data['metrics']
        logger.info(f'len(query_phrases): {len(query_phrases)}')
        logger.info(f'len(candidate_phrases): {len(candidate_phrases)}')
        logger.info(f'metrics.keys(): {list(metrics.keys())}')
        sentences = query_phrases + candidate_phrases # Combine queries and candidates for embedding extraction

    else:
        raise ValueError(f'Invalid evaluation_mode: {evaluation_mode}')
    
    n = len(sentences)
    
    # Obtain embeddings for each sentence
    if model_name in SupportedHuggingfaceMedicalBERTModels.get_all():
        embedding_extractor = CachedTextEmbeddingExtractor(
            model_name=model_name,
            device=device,
            model_checkpoint_folder_path=model_checkpoint_folder_path,
            batch_size=batch_size,
            num_workers=num_workers,
            average_token_embeddings=average_token_embeddings,
        )
        embeddings = embedding_extractor.compute_text_embeddings(sentences)
    elif model_name == 'CheXbert':
        from medvqa.metrics.medical.chexbert import CheXbertLabeler
        chexbert_labeler = CheXbertLabeler(device=device)
        embeddings = chexbert_labeler.get_embeddings(sentences)
    else:
        raise ValueError(f'Invalid model_name: {model_name}')
    logger.info(f'Embeddings shape: {embeddings.shape}')
    assert embeddings.shape[0] == n

    if evaluation_mode in (
        EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS.value,
        EvaluationModes.IUXRAY_WITH_AUTOMATIC_LABELERS.value,
    ):
        rank_vectors_func = None
        if distance_metric == 'cosine':
            logger.info('Normalizing embeddings (for cosine similarity)')
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # Normalize embeddings
            rank_vectors_func = rank_vectors_by_dot_product # Cosine similarity is equivalent to dot product when embeddings are normalized
        elif distance_metric == 'dot_product':
            rank_vectors_func = rank_vectors_by_dot_product
        else:
            raise ValueError(f'Invalid distance_metric: {distance_metric}')
        
        # Evaluate embeddings on ranking task with each sentence as query
        logger.info(f'{ANSI_ORANGE_BOLD}Evaluating embeddings on ranking task with each sentence as query{ANSI_RESET}')
        mean_AUC = 0
        mean_relevant = 0
        count = 0
        for i in tqdm(range(n), mininterval=2):
            relevant_idxs = relevant_sentences[i]
            if len(relevant_idxs) == 0:
                continue
            sorted_idxs = rank_vectors_func(embeddings, embeddings[i])
            sorted_idxs = [idx for idx in sorted_idxs if idx != i] # Remove query_idx
            AUC = _compute_AUC(sorted_idxs, relevant_idxs, query_idx=i)
            mean_AUC += AUC
            mean_relevant += len(relevant_idxs)
            count += 1
        mean_AUC /= count
        mean_relevant /= count
        logger.info(f'{ANSI_MAGENTA_BOLD}mean_AUC: {mean_AUC:.4f}{ANSI_RESET}')
        logger.info(f'mean_relevant: {mean_relevant:.4f}')
        logger.info(f'count: {count} / {n} ({100 * count / n:.2f}%)')
    
    elif evaluation_mode == EvaluationModes.CHEST_IMAGENOME_GOLD.value:

        if distance_metric == 'cosine':
            logger.info('Normalizing embeddings (for cosine similarity)')
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # Normalize embeddings
        logger.info('Computing score matrix')
        score_matrix = np.dot(embeddings, embeddings.T)
        logger.info('score_matrix.shape:', score_matrix.shape)
        
        # Compute AUC
        logger.info(f'{ANSI_MAGENTA_BOLD}mean_AUC: {compute_AUC(gold_relevance_matrix, score_matrix):.4f}{ANSI_RESET}')

        # Compute mean average accuracy @k for each metric
        max_k = 200
        mean_average_accuracy_at_k = compute_mean_average_accuracy_at_k(gold_accuracy_matrix, score_matrix, max_k)
        for k in (1, 5, 10, 20, 50, 100, 200):
            logger.info(f'mean_average_accuracy_at_{k}: {mean_average_accuracy_at_k[k-1]:.4f}')

        # Compute mean average contradiction @k
        mean_num_contradictions_at_k = compute_mean_contradictions_at_k(gold_contradiction_matrix, score_matrix, max_k)
        for k in (1, 5, 10, 20, 50, 100, 200):
            logger.info(f'mean_num_contradictions_at_{k}: {mean_num_contradictions_at_k[k-1]:.4f}')

    elif evaluation_mode == EvaluationModes.PADCHEST_GR_MSCXR_CUSTOM.value:
        
        # --- STEP 1: Create hybrid ground-truth scores by averaging all precomputed metrics ---
        # This creates a single "ground truth" ranking by taking the uniform average of all
        # available metrics (BLEU-1, ROUGE-L, Levenshtein, BERTScore, etc.)
        logger.info('Computing hybrid ground-truth scores by averaging all metrics...')
        metrics_matrix = np.zeros((len(query_phrases), len(candidate_phrases)), dtype=np.float32)
        for metric_name, metric_values in metrics.items():
            logger.info(f'Adding {metric_name} to hybrid score (shape: {metric_values.shape})')
            metrics_matrix += metric_values
        metrics_matrix /= len(metrics)  # Uniform average across all metrics
        logger.info(f'Hybrid metrics_matrix shape: {metrics_matrix.shape}')
        logger.info(f'Hybrid score range: [{metrics_matrix.min():.4f}, {metrics_matrix.max():.4f}]')

        # --- STEP 2: Prepare embedding-based similarity computation ---
        if distance_metric == 'cosine':
            logger.info('Using cosine similarity: normalizing embeddings for dot product equivalence')
            # Normalize embeddings so that cosine similarity = dot product
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        elif distance_metric == 'dot_product':
            pass # No normalization needed for dot product
        else:
            raise ValueError(f'Invalid distance_metric: {distance_metric}. Expected "cosine" or "dot_product".')

        # --- STEP 3: Split embeddings into query and candidate sets ---
        # Assumes embeddings are ordered as: [query_0, query_1, ..., query_N, candidate_0, candidate_1, ..., candidate_M]
        query_embeddings = embeddings[:len(query_phrases)]
        candidate_embeddings = embeddings[len(query_phrases):]
        logger.info(f'Query embeddings shape: {query_embeddings.shape}')
        logger.info(f'Candidate embeddings shape: {candidate_embeddings.shape}')

        # --- STEP 4: Compute embedding-based similarity scores ---
        # For each query, compute similarity scores against all candidates using embeddings
        logger.info('Computing embedding-based similarity scores...')
        embedding_similarity_matrix = query_embeddings @ candidate_embeddings.T  # Matrix multiplication
        logger.info(f'Embedding similarity matrix shape: {embedding_similarity_matrix.shape}')
        logger.info(f'Embedding similarity range: [{embedding_similarity_matrix.min():.4f}, {embedding_similarity_matrix.max():.4f}]')

        # --- STEP 5: Compute ranking evaluation metrics ---
        # For each query, compare the ranking induced by embedding similarity 
        # against the ranking induced by the hybrid ground-truth scores
        logger.info('Computing ranking evaluation metrics...')
        
        from scipy.stats import spearmanr, kendalltau
        from sklearn.metrics import ndcg_score
        
        # Initialize metric storage
        spearman_correlations = []
        kendall_tau_correlations = []
        ndcg_scores = []

        # AUC@K metrics - define K values for top-K evaluation
        k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        auc_at_k_results = {k: [] for k in k_values}  # Store AUC@K for each K
        
        for i in range(len(query_phrases)):
            # Get scores for this query
            gt_scores = metrics_matrix[i, :]  # Ground truth hybrid scores
            pred_scores = embedding_similarity_matrix[i, :]  # Embedding-based scores
            
            # --- Spearman Rank Correlation ---
            # Measures monotonic relationship between rankings
            spearman_corr, _ = spearmanr(pred_scores, gt_scores)
            spearman_correlations.append(spearman_corr)
            
            # --- Kendall's Tau ---
            # Alternative rank correlation metric, more robust to outliers
            kendall_corr, _ = kendalltau(pred_scores, gt_scores)
            kendall_tau_correlations.append(kendall_corr)

            # --- NDCG (Normalized Discounted Cumulative Gain) ---
            # Measures ranking quality with position-based discounting
            # NDCG requires non-negative relevance scores
            # Normalize gt_scores to [0, 1] range for this query
            gt_min = gt_scores.min()
            gt_max = gt_scores.max()
            assert gt_max > gt_min
            # Min-max normalization to [0, 1]
            gt_scores_normalized = (gt_scores - gt_min) / (gt_max - gt_min)
            # Ensure we have non-negative scores
            assert gt_scores_normalized.min() >= 0, f"Normalized scores still negative: {gt_scores_normalized.min()}"
            ndcg = ndcg_score([gt_scores_normalized], [pred_scores])
            ndcg_scores.append(ndcg)

            # --- AUC@K (Area Under Curve for Top-K) ---
            # Binarize ground truth: top-K candidates according to gt_scores are "relevant" (1), rest are "irrelevant" (0)
            # Then compute AUC using embedding similarity scores as predictions
            gt_ranking_indices = np.argsort(gt_scores)[::-1]  # Indices sorted by gt_scores (descending)
            for k in k_values:
                # Create binary relevance labels: 1 for top-K according to ground truth, 0 for rest
                binary_relevance = np.zeros(len(candidate_phrases), dtype=int)
                binary_relevance[gt_ranking_indices[:k]] = 1  # Top-K are relevant
                # Compute AUC using embedding similarity scores as predictions
                # Higher embedding similarity should predict higher relevance
                auc_k = roc_auc_score(binary_relevance, pred_scores)
                auc_at_k_results[k].append(auc_k)
        
        # Convert to numpy arrays for easier analysis
        spearman_correlations = np.array(spearman_correlations)
        kendall_tau_correlations = np.array(kendall_tau_correlations)
        ndcg_scores = np.array(ndcg_scores)

        # Convert AUC@K results to numpy arrays and compute means
        auc_at_k_arrays = {}
        mean_auc_at_k = {}
        for k in k_values:
            auc_at_k_arrays[k] = np.array(auc_at_k_results[k])
            mean_auc_at_k[k] = auc_at_k_arrays[k].mean()
        
        # --- STEP 6: Compute and log summary statistics ---
        logger.info('=== RANKING EVALUATION RESULTS ===')
        logger.info(f'{ANSI_MAGENTA_BOLD}Spearman Rank Correlation:{ANSI_RESET}')
        logger.info(f'  {ANSI_MAGENTA_BOLD}Mean: {spearman_correlations.mean():.4f} ± {spearman_correlations.std():.4f}{ANSI_RESET}')
        logger.info(f'  Median: {np.median(spearman_correlations):.4f}')
        logger.info(f'  Range: [{spearman_correlations.min():.4f}, {spearman_correlations.max():.4f}]')
        
        logger.info(f'{ANSI_MAGENTA_BOLD}Kendall Tau Correlation:{ANSI_RESET}')
        logger.info(f'  {ANSI_MAGENTA_BOLD}Mean: {kendall_tau_correlations.mean():.4f} ± {kendall_tau_correlations.std():.4f}{ANSI_RESET}')
        logger.info(f'  Median: {np.median(kendall_tau_correlations):.4f}')
        logger.info(f'  Range: [{kendall_tau_correlations.min():.4f}, {kendall_tau_correlations.max():.4f}]')
        
        logger.info(f'{ANSI_MAGENTA_BOLD}NDCG Score:{ANSI_RESET}')
        logger.info(f'  {ANSI_MAGENTA_BOLD}Mean: {ndcg_scores.mean():.4f} ± {ndcg_scores.std():.4f}{ANSI_RESET}')
        logger.info(f'  Median: {np.median(ndcg_scores):.4f}')
        logger.info(f'  Range: [{ndcg_scores.min():.4f}, {ndcg_scores.max():.4f}]')

        logger.info(f'{ANSI_MAGENTA_BOLD}AUC@K (Area Under Curve for Top-K):{ANSI_RESET}')
        for k in k_values:
            auc_array = auc_at_k_arrays[k]
            logger.info(f'  {ANSI_MAGENTA_BOLD}AUC@{k}: Mean: {auc_array.mean():.4f} ± {auc_array.std():.4f}{ANSI_RESET}, '
                    f'Median: {np.median(auc_array):.4f}, '
                    f'Range: [{auc_array.min():.4f}, {auc_array.max():.4f}]')
        
        # Compute overall mean AUC across all K values
        overall_mean_auc = np.mean(list(mean_auc_at_k.values()))
        logger.info(f'  {ANSI_MAGENTA_BOLD}Overall Mean AUC@K (averaged across K={k_values}): {overall_mean_auc:.4f}{ANSI_RESET}')
        
        logger.info('Ranking evaluation completed successfully.')

        if save_worst_queries_for_inspection:
            # Sort queries based on average AUC across all K values
            query_auc_scores = np.mean(list(auc_at_k_arrays.values()), axis=0)
            sorted_query_indices = np.argsort(query_auc_scores)
            sorted_queries = [query_phrases[i] for i in sorted_query_indices]
            sorted_auc_scores = query_auc_scores[sorted_query_indices]
            to_save = {
                'queries': sorted_queries,
                'auc_scores': sorted_auc_scores,
            }
            timestamp = get_timestamp()
            save_path = os.path.join(CACHE_DIR, f'padchest_gr_mscxr_worst_queries_for_inspection_{timestamp}.pkl')
            logger.info(f'Saving worst queries for inspection to {save_path}')
            save_pickle(to_save, save_path)

        

class SentenceRanker:
    def __init__(self, dataset_name):
        assert dataset_name in EvaluationModes.get_choices()
        if dataset_name == EvaluationModes.MIMICCXR_RADIOLOGIST_ANNOTATIONS.value:
            tmp = _load_mimiccxr_sentences_and_radiologist_based_relevant_sentences()
            self.sentences, self.relevant_sentences = tmp['sentences'], tmp['relevant_sentences']
        elif dataset_name == EvaluationModes.IUXRAY_WITH_AUTOMATIC_LABELERS.value:
            tmp = _load_iuxray_sentences_and_automatic_labeler_based_relevant_sentences()
            self.sentences, self.relevant_sentences = tmp['sentences'], tmp['relevant_sentences']
        else:
            raise ValueError(f'Invalid dataset_name: {dataset_name}')
        self.cache = {}
    
    def rank_sentences(self, query_idx, model_name, checkpoint_folder_path=None, average_token_embeddings=False,
                       top_k=10, batch_size=32, num_workers=4):
        assert isinstance(average_token_embeddings, bool)
        assert top_k > 0
        assert top_k <= len(self.sentences)
        assert 0 <= query_idx < len(self.sentences)
        if checkpoint_folder_path is None:
            key = (model_name, average_token_embeddings)
        else:
            key = (model_name, checkpoint_folder_path, average_token_embeddings)
        try:
            embedding_extractor = self.cache[key]
        except KeyError:
            embedding_extractor = self.cache[key] = CachedTextEmbeddingExtractor(
                model_name=model_name,
                device='GPU',
                model_checkpoint_folder_path=checkpoint_folder_path,
                batch_size=batch_size,
                num_workers=num_workers,
                average_token_embeddings=average_token_embeddings,
            )
        embeddings = embedding_extractor.compute_text_embeddings(self.sentences)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sorted_idxs = rank_vectors_by_dot_product(embeddings, embeddings[query_idx])
        sorted_sentences = [self.sentences[i] for i in sorted_idxs[:top_k]]
        is_relevant = [1 if i in self.relevant_sentences[query_idx] else 0 for i in sorted_idxs[:top_k]]
        return sorted_sentences, is_relevant

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)