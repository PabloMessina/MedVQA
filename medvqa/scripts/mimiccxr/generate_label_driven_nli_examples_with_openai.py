import os
import argparse
import math
import random
import pandas as pd
import numpy as np
import Levenshtein
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.logging_utils import get_console_logger
from medvqa.datasets.mimiccxr import (
    MIMICCXR_LARGE_FAST_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
)
from medvqa.utils.math_utils import rank_vectors_by_dot_product
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import load_jsonl, load_pickle, save_pickle

INSTRUCTIONS = """Given a premise (#P) and a hypothesis (#H), output "Reason: {reason}. Label: {label}" where {reason} is one or two short explanation sentences and {label} is one of {definitely true, likely true, unknown, likely false, definitely false}. Be careful with tricky sentences that mention multiple findings. Remember that unknown applies when both statements might be true but there is no clear way to know with the information provided."""

_POSSIBLE_LABELS = [
    "label: definitely true",
    "label: likely true",
    "label: unknown",
    "label: likely false",
    "label: definitely false",
]

_CHEST_IMAGENOME_OBSERVATION_TO_PHRASE = {
    'anatomicalfinding|airspace opacity': 'airspace opacity is seen',
    'anatomicalfinding|atelectasis': 'atelectasis is seen',
    'anatomicalfinding|bone lesion': 'bone lesion is seen',
    'anatomicalfinding|bronchiectasis': 'bronchiectasis is seen',
    'anatomicalfinding|calcified nodule': 'calcified nodule is seen',
    'anatomicalfinding|clavicle fracture': 'clavicle fracture is seen',
    'anatomicalfinding|consolidation': 'consolidation is seen',
    'anatomicalfinding|costophrenic angle blunting': 'costophrenic angle blunting is seen',
    'anatomicalfinding|cyst/bullae': 'cyst or bullae is seen',
    'anatomicalfinding|diaphragmatic eventration (benign)': ' benign diaphragmatic eventration is seen',
    'anatomicalfinding|elevated hemidiaphragm': 'elevated hemidiaphragm is seen',
    'anatomicalfinding|enlarged cardiac silhouette': 'enlarged cardiac silhouette is seen',
    'anatomicalfinding|enlarged hilum': 'enlarged hilum is seen',
    'anatomicalfinding|hernia': 'hernia is seen',
    'anatomicalfinding|hydropneumothorax': 'hydropneumothorax is seen',
    'anatomicalfinding|hyperaeration': 'hyperaeration is seen',
    'anatomicalfinding|increased reticular markings/ild pattern': 'increased reticular markings or ild pattern is seen',
    'anatomicalfinding|infiltration': 'infiltration is seen',
    'anatomicalfinding|linear/patchy atelectasis': 'linear or patchy atelectasis is seen',
    'anatomicalfinding|lobar/segmental collapse': 'lobar or segmental collapse is seen',
    'anatomicalfinding|lung lesion': 'lung lesion is seen',
    'anatomicalfinding|lung opacity': 'lung opacity is seen',
    'anatomicalfinding|mass/nodule (not otherwise specified)': 'mass or nodule is seen',
    'anatomicalfinding|mediastinal displacement': 'mediastinal displacement is seen',
    'anatomicalfinding|mediastinal widening': 'mediastinal widening is seen',
    'anatomicalfinding|multiple masses/nodules': 'multiple masses or nodules are seen',
    'anatomicalfinding|pleural effusion': 'pleural effusion is seen',
    'anatomicalfinding|pleural/parenchymal scarring': 'pleural or parenchymal scarring is seen',
    'anatomicalfinding|pneumomediastinum': 'pneumomediastinum is seen',
    'anatomicalfinding|pneumothorax': 'pneumothorax is seen',
    'anatomicalfinding|pulmonary edema/hazy opacity': 'pulmonary edema or hazy opacity is seen',
    'anatomicalfinding|rib fracture': 'rib fracture is seen',
    'anatomicalfinding|scoliosis': 'scoliosis is seen',
    'anatomicalfinding|shoulder osteoarthritis': 'shoulder osteoarthritis is seen',
    'anatomicalfinding|spinal degenerative changes': 'spinal degenerative changes are seen',
    'anatomicalfinding|spinal fracture': 'spinal fracture is seen',
    'anatomicalfinding|sub-diaphragmatic air': 'sub-diaphragmatic air is seen',
    'anatomicalfinding|subcutaneous air': 'subcutaneous air is seen',
    'anatomicalfinding|superior mediastinal mass/enlargement': 'superior mediastinal mass or enlargement is seen',
    'anatomicalfinding|tortuous aorta': 'tortuous aorta is seen',
    'anatomicalfinding|vascular calcification': 'vascular calcification is seen',
    'anatomicalfinding|vascular congestion': 'vascular congestion is seen',
    'anatomicalfinding|vascular redistribution': 'vascular redistribution is seen',
    'device|aortic graft/repair': 'aortic graft or repair is seen',
    'device|cabg grafts': 'cabg grafts are seen',
    'device|cardiac pacer and wires': 'cardiac pacer and wires are seen',
    'device|prosthetic valve': 'prosthetic valve is seen',
    'disease|alveolar hemorrhage': 'alveolar hemorrhage is seen',
    'disease|aspiration': 'aspiration is seen',
    'disease|copd/emphysema': 'copd or emphysema is seen',
    'disease|fluid overload/heart failure': 'fluid overload or heart failure is seen',
    'disease|goiter': 'goiter is seen',
    'disease|granulomatous disease': 'granulomatous disease is seen',
    'disease|interstitial lung disease': 'interstitial lung disease is seen',
    'disease|lung cancer': 'lung cancer is seen',
    'disease|pericardial effusion': 'pericardial effusion is seen',
    'disease|pneumonia': 'pneumonia is seen',
    'nlp|abnormal': 'abnormalities are seen',
    'nlp|normal': 'no abnormalities are seen',
    'technicalassessment|artifact': 'an artifact is seen',
    'technicalassessment|breast/nipple shadows': 'breast or nipple shadows are seen',
    'technicalassessment|low lung volumes': 'low lung volumes are seen',
    'technicalassessment|rotated': 'there is evidence of rotation',
    'technicalassessment|skin fold': 'a skin fold is seen',
    'tubesandlines|chest port': 'a chest port is seen',
    'tubesandlines|chest tube': 'a chest tube is seen',
    'tubesandlines|endotracheal tube': 'an endotracheal tube is seen',
    'tubesandlines|enteric tube': 'an enteric tube is seen',
    'tubesandlines|ij line': 'an ij line is seen',
    'tubesandlines|intra-aortic balloon pump': 'an intra-aortic balloon pump is seen',
    'tubesandlines|mediastinal drain': 'a mediastinal drain is seen',
    'tubesandlines|picc': 'a picc is seen',
    'tubesandlines|pigtail catheter': 'a pigtail catheter is seen',
    'tubesandlines|subclavian line': 'a subclavian line is seen',
    'tubesandlines|swan-ganz catheter': 'a Swan-Ganz catheter is seen',
    'tubesandlines|tracheostomy tube': 'a tracheostomy tube is seen',
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
    for label in _POSSIBLE_LABELS:
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

def get_query(p, h):
    return f"#P: {p} | #H: {h}"

def generate_chest_imagenome_NLI_examples(chest_imagenome_silver_phrase_to_labels_filepath, logger):
    filename = os.path.basename(chest_imagenome_silver_phrase_to_labels_filepath)
    save_path = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, f"{filename}_chest_imagenome_NLI_examples.pkl")
    if os.path.exists(save_path):
        logger.info(f"Loading {save_path}")
        data = load_pickle(save_path)
        return data

    assert os.path.exists(chest_imagenome_silver_phrase_to_labels_filepath)
    logger.info(f"Loading phrases and labels from {chest_imagenome_silver_phrase_to_labels_filepath}")
    data = load_pickle(chest_imagenome_silver_phrase_to_labels_filepath)
    phrases = data['phrases']
    labels = data['observation_labels']
    label_names = data['observation_names']
    label_phrases = [_CHEST_IMAGENOME_OBSERVATION_TO_PHRASE[label_name] for label_name in label_names]
    phrases.extend(label_phrases)
    phrase2idx = {phrase: i for i, phrase in enumerate(phrases)}
    
    logger.info(f"Loaded {len(phrases)} phrases")
    logger.info(f"label.shape: {labels.shape}")
    filter_2 = (labels != 2).sum(axis=1) <= 2 # At most 2 non-unknown labels
    filter_3 = (labels != 2).sum(axis=1) <= 3 # At most 3 non-unknown labels
    n_classes = labels.shape[1]

    groups = []
    for i in range(n_classes):
        class_name = data['observation_names'][i]
        class_phrase = _CHEST_IMAGENOME_OBSERVATION_TO_PHRASE[class_name]
        class_phrase_idx = phrase2idx[class_phrase]

        pos_idxs = np.where((labels.T[i] == 1) & filter_2)[0] # At most 2 non-unknown labels
        if len(pos_idxs) < 400:
            pos_idxs = np.where((labels.T[i] == 1) & filter_3)[0] # At most 3 non-unknown labels
        if len(pos_idxs) < 400:
            pos_idxs = np.where(labels.T[i] == 1)[0] # All
        pos_idxs = list(pos_idxs)
        
        neg_idxs = np.where((labels.T[i] == 0) & filter_2)[0] # At most 2 non-unknown labels
        if len(neg_idxs) < 400:
            neg_idxs = np.where((labels.T[i] == 0) & filter_3)[0] # At most 3 non-unknown labels
        if len(neg_idxs) < 400:
            neg_idxs = np.where(labels.T[i] == 0)[0] # All
        neg_idxs = list(neg_idxs)

        logger.info(f"----------------------------------")
        logger.info(f"Class: {class_name}, class_phrase: {class_phrase}, pos_idxs: {len(pos_idxs)}, neg_idxs: {len(neg_idxs)}")

        if len(pos_idxs) > 10:
            pos_ent_pairs = set()
            for idx in pos_idxs:
                # Choose 10 random positive phrases
                if len(neg_idxs) > 10:
                    sample_idxs_ = random.sample(pos_idxs, 10)
                else:
                    sample_idxs_ = pos_idxs.copy()
                sample_idxs_.sort(key=lambda idx_: Levenshtein.distance(phrases[idx], phrases[idx_]))
                count = 0
                for idx_ in sample_idxs_:
                    if count >= 3:
                        break
                    if idx == idx_:
                        continue
                    pos_ent_pairs.add((idx, idx_))
                    count += 1
            pos_ent_pairs = list(pos_ent_pairs)
            groups.append(pos_ent_pairs) # Positive Entailment pairs
            logger.info(f"Added {len(pos_ent_pairs)} positive entailment pairs")
            logger.info(f"pos_ent_pairs[0]: {(phrases[pos_ent_pairs[0][0]], phrases[pos_ent_pairs[0][1]])}")
            logger.info(f"pos_ent_pairs[-1]: {(phrases[pos_ent_pairs[-1][0]], phrases[pos_ent_pairs[-1][1]])}")

            pos_ent_pairs_2 = [(idx, class_phrase_idx) for idx in pos_idxs if idx != class_phrase_idx]
            groups.append(pos_ent_pairs_2) # Positive Entailment pairs with class_phrase
            logger.info(f"Added {len(pos_ent_pairs_2)} positive entailment pairs with class_phrase")
            logger.info(f"pos_ent_pairs_2[0]: {(phrases[pos_ent_pairs_2[0][0]], phrases[pos_ent_pairs_2[0][1]])}")

        if len(neg_idxs) > 10:
            neg_ent_pairs = set()
            for idx in neg_idxs:
                # Choose 10 random negative phrases
                if len(neg_idxs) > 10:
                    sample_idxs_ = random.sample(neg_idxs, 10)
                else:
                    sample_idxs_ = neg_idxs.copy()
                sample_idxs_.sort(key=lambda idx_: Levenshtein.distance(phrases[idx], phrases[idx_]))
                count = 0
                for idx_ in sample_idxs_:
                    if count >= 3:
                        break
                    if idx == idx_:
                        continue
                    neg_ent_pairs.add((idx, idx_))
                    count += 1
            neg_ent_pairs = list(neg_ent_pairs)
            groups.append(neg_ent_pairs) # Negative Entailment pairs
            logger.info(f"Added {len(neg_ent_pairs)} negative entailment pairs")
            logger.info(f"neg_ent_pairs[0]: {(phrases[neg_ent_pairs[0][0]], phrases[neg_ent_pairs[0][1]])}")
            logger.info(f"neg_ent_pairs[-1]: {(phrases[neg_ent_pairs[-1][0]], phrases[neg_ent_pairs[-1][1]])}")

            neg_cont_pairs_2 = [(idx, class_phrase_idx) for idx in neg_idxs if idx != class_phrase_idx]
            groups.append(neg_cont_pairs_2) # Negative Contradiction pairs with class_phrase
            logger.info(f"Added {len(neg_cont_pairs_2)} negative contradiction pairs with class_phrase")
            logger.info(f"neg_cont_pairs_2[0]: {(phrases[neg_cont_pairs_2[0][0]], phrases[neg_cont_pairs_2[0][1]])}")            

        if len(pos_idxs) > 10 and len(neg_idxs) > 10: # Include contradiction pairs
            cont_pairs = []
            for pos_idx in pos_idxs:
                p = phrases[pos_idx]
                neg_idxs_ = random.sample(list(neg_idxs), 10)  if len(neg_idxs) > 10 else neg_idxs
                neg_idxs_.sort(key=lambda idx: Levenshtein.distance(p, phrases[idx]))
                cont_pairs.extend((pos_idx, neg_idx) for neg_idx in neg_idxs_[:3])
            for neg_idx in neg_idxs:
                p = phrases[neg_idx]
                pos_idxs_ = random.sample(list(pos_idxs), 10) if len(pos_idxs) > 10 else pos_idxs
                pos_idxs_.sort(key=lambda idx: Levenshtein.distance(p, phrases[idx]))
                cont_pairs.extend((neg_idx, pos_idx) for pos_idx in pos_idxs_[:3])
            groups.append(cont_pairs) # Contradiction pairs
            logger.info(f"Added {len(cont_pairs)} contradiction pairs")
            logger.info(f"cont_pairs[0]: {(phrases[cont_pairs[0][0]], phrases[cont_pairs[0][1]])}")
            logger.info(f"cont_pairs[-1]: {(phrases[cont_pairs[-1][0]], phrases[cont_pairs[-1][1]])}")

    assert len(groups) > 0
    logger.info(f"Found {len(groups)} groups")

    data = {
        "phrases": phrases,
        "groups": groups,
    }
    logger.info(f"Saving {save_path}")
    save_pickle(data, save_path)
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_queries_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--integrated_sentence_facts_jsonl_filepath", type=str, default=None)
    
    parser.add_argument("--cxr_bert_model_name", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--cxr_bert_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_pairs_to_generate", type=int, default=20000)

    parser.add_argument("--custom_nli_csv_paths", nargs="+", default=None)
    parser.add_argument("--chest_imagenome_silver_phrase_to_labels_filepath", type=str, default=None)

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_request", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--use_batch_api", action="store_true", default=False)
    parser.add_argument("--batch_description", type=str, default=None)
    parser.add_argument("--batch_input_file_id", type=str, default=None)
    args = parser.parse_args()

    processed_texts_save_filepath = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_label_driven_nli_queries{args.alias}.jsonl")
    
    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None and args.batch_input_file_id is None:

        # Load already processed queries if they exist
        already_processed = set()
        if os.path.exists(processed_texts_save_filepath):
            rows = load_jsonl(processed_texts_save_filepath)
            for row in rows:
                already_processed.add(row['metadata']['query'])
            logger.info(f"Loaded {len(rows)} already processed texts from {processed_texts_save_filepath}")
        if args.preprocessed_queries_to_skip_filepaths is not None:
            for filepath in args.preprocessed_queries_to_skip_filepaths:
                assert os.path.exists(filepath)
                rows = load_jsonl(filepath)
                assert len(rows) > 0
                if rows[0].get('metadata', None) is not None:
                    already_processed.update(row['metadata']['query'] for row in rows)
                else:
                    assert 'premise' in rows[0] and 'hypothesis' in rows[0]
                    already_processed.update(get_query(row['premise'], row['hypothesis']) for row in rows)
                logger.info(f"Loaded {len(rows)} queries to skip from {filepath}")

        # Define queries to skip
        queries_to_skip = set()
        queries_to_skip.update(already_processed)        

        # Collect sentences and facts
        integrated_sentence_facts = load_jsonl(args.integrated_sentence_facts_jsonl_filepath)
        logger.info(f"Loaded {len(integrated_sentence_facts)} integrated_sentence_facts")
        report_sentences = set()
        report_facts = set()
        for row in integrated_sentence_facts:
            report_sentences.add(row['sentence'])
            report_facts.update(row['facts'])
        report_facts.update(x for x in _CHEST_IMAGENOME_OBSERVATION_TO_PHRASE.values()) # Add chest imagenome phrases
        logger.info(f"Found {len(report_sentences)} unique sentences and {len(report_facts)} unique facts")
        report_sentences = list(report_sentences)
        report_facts = list(report_facts)

        unique_sentences = set()
        unique_sentences.update(report_sentences)
        unique_sentences.update(report_facts)

        groups = []

        # Collect sentences from custom NLI datasets
        if args.custom_nli_csv_paths is not None:
            logger.info(f"Loading custom NLI datasets")
            for csv_path in args.custom_nli_csv_paths:
                assert os.path.exists(csv_path)
                df = pd.read_csv(csv_path)
                entailment_pairs = []
                contradiction_pairs = []
                for p, h, l in zip(df['reference'], df['candidate'], df['value']):
                    if l == 'entailment':
                        entailment_pairs.append((p, h))
                    elif l == 'contradiction':
                        contradiction_pairs.append((p, h))
                    else:
                        raise ValueError(f"Unexpected label: {l}")
                    queries_to_skip.add(get_query(p, h)) # Skip these queries
                    unique_sentences.add(p)
                    unique_sentences.add(h)
                assert len(entailment_pairs) > 0
                assert len(contradiction_pairs) > 0
                groups.append(entailment_pairs)
                groups.append(contradiction_pairs)
            assert len(groups) > 0
            logger.info(f"Found {len(groups)} groups")
        
        if args.chest_imagenome_silver_phrase_to_labels_filepath is not None:
            data = generate_chest_imagenome_NLI_examples(args.chest_imagenome_silver_phrase_to_labels_filepath, logger)
            phrases = data['phrases']
            unique_sentences.update(phrases)
            for group in data['groups']:
                group_ = [(phrases[i], phrases[j]) for i, j in group] # Convert indices to phrases
                groups.append(group_)

        assert len(groups) > 0

        logger.info(f"Unique sentences: {len(unique_sentences)}")

        # exit()

        # Sort sentences
        unique_sentences = list(unique_sentences)
        unique_sentences.sort()
        unique_sentence_to_idx = {s: i for i, s in enumerate(unique_sentences)}
        report_sentence_idxs = [unique_sentence_to_idx[s] for s in report_sentences]
        report_fact_idxs = [unique_sentence_to_idx[s] for s in report_facts]

        # Obtain sentence embeddings
        emb_extractor = CachedTextEmbeddingExtractor(
            model_name=args.cxr_bert_model_name,
            model_checkpoint_folder_path=args.cxr_bert_checkpoint_folder_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        all_embeddings = emb_extractor.compute_text_embeddings(unique_sentences)
        logger.info(f"all_embeddings.shape: {all_embeddings.shape}")

        report_sentence_embeddings = all_embeddings[report_sentence_idxs]
        report_fact_embeddings = all_embeddings[report_fact_idxs]
        logger.info(f"report_sentence_embeddings.shape: {report_sentence_embeddings.shape}")
        logger.info(f"report_fact_embeddings.shape: {report_fact_embeddings.shape}")

        # Obtain kmeans labels
        kmeans_labels = emb_extractor.compute_kmeans_labels(unique_sentences, 50, embeddings=all_embeddings)

        # Group sentences by kmeans labels
        cluster2fact_idxs = {}
        cluster2sentence_idxs = {}
        for idx in report_fact_idxs:
            cluster = kmeans_labels[idx]
            if cluster not in cluster2fact_idxs:
                cluster2fact_idxs[cluster] = []
            cluster2fact_idxs[cluster].append(idx)
        for idx in report_sentence_idxs:
            cluster = kmeans_labels[idx]
            if cluster not in cluster2sentence_idxs:
                cluster2sentence_idxs[cluster] = []
            cluster2sentence_idxs[cluster].append(idx)
        cluster2fact_embeddings = {cluster: all_embeddings[idxs] for cluster, idxs in cluster2fact_idxs.items()}
        cluster2sentence_embeddings = {cluster: all_embeddings[idxs] for cluster, idxs in cluster2sentence_idxs.items()}

        # Generate queries
        queries_to_make = []
        n_pairs_per_group = math.ceil(args.num_pairs_to_generate / len(groups))
        for group in groups:
            assert len(group) > 0
            if len(group) > n_pairs_per_group:
                group = random.sample(group, n_pairs_per_group)
            n_pairs_per_gt_pair = math.ceil(n_pairs_per_group / len(group))
            n_pairs_per_embedding = math.ceil(n_pairs_per_gt_pair / 2)

            # debug_count = 0

            for gt_pair in group:

                gt_query = get_query(*gt_pair)
                if gt_query not in queries_to_skip:
                    queries_to_make.append(gt_query)
                    queries_to_skip.add(gt_query)

                for cluster2embeddings_, cluster2idxs_ in zip(
                    [cluster2sentence_embeddings, cluster2fact_embeddings],
                    [cluster2sentence_idxs, cluster2fact_idxs],
                ):
                    premise, hypothesis = gt_pair
                    premise_idx = unique_sentence_to_idx[premise]
                    hypothesis_idx = unique_sentence_to_idx[hypothesis]
                    premise_emb = all_embeddings[premise_idx]
                    hypothesis_emb = all_embeddings[hypothesis_idx]
                    premise_cluster = kmeans_labels[premise_idx]
                    hypothesis_cluster = kmeans_labels[hypothesis_idx]
                    p_embeddings = cluster2embeddings_[premise_cluster]
                    h_embeddings = cluster2embeddings_[hypothesis_cluster]
                    p_idxs = cluster2idxs_[premise_cluster]
                    h_idxs = cluster2idxs_[hypothesis_cluster]
                    p_sorted_indices = rank_vectors_by_dot_product(p_embeddings, premise_emb)
                    h_sorted_indices = rank_vectors_by_dot_product(h_embeddings, hypothesis_emb)

                    # queries closely similar to gt_pair
                    i = 0
                    j = 0
                    count = 0
                    while count < n_pairs_per_embedding and i < len(p_sorted_indices) and j < len(h_sorted_indices):
                        p_idx = p_idxs[p_sorted_indices[i]]
                        h_idx = h_idxs[h_sorted_indices[j]]
                        if i < j:
                            i += 1
                        else:
                            j += 1
                        if p_idx == h_idx:
                            continue
                        p = unique_sentences[p_idx]
                        h = unique_sentences[h_idx]
                        query = get_query(p, h)
                        if query in queries_to_skip:
                            continue
                        queries_to_make.append(query)
                        queries_to_skip.add(query)
                        count += 1

                    assert count > 0

                    # # Sanity checking
                    # if debug_count < 2:
                    #     logger.info("---------------------")
                    #     logger.info(f"gt_pair: {gt_pair}")
                    #     logger.info(f"queries_to_make[{-count}]: {queries_to_make[-count]}")

                    # debug_count += 1

        logger.info("==========================================================")
        logger.info(f"Total number of queries to make: {len(queries_to_make)}")

        logger.info("==========================================================")
        logger.info(f"Example queries:")
        for query in (random.sample(queries_to_make, 10) if len(queries_to_make) > 10 else queries_to_make):
            logger.info(query)

        # queries_to_make = queries_to_make[:10] # For testing

    else:
        if args.api_responses_filepath is not None:
            assert os.path.exists(args.api_responses_filepath)
        queries_to_make = None

    # Run OpenAI API requests
    run_common_boilerplate_for_api_requests(
        api_responses_filepath=args.api_responses_filepath,
        texts=queries_to_make,
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
        save_filepath=processed_texts_save_filepath,
        use_batch_api=args.use_batch_api,
        batch_description=args.batch_description,
        batch_input_file_id=args.batch_input_file_id,
    )