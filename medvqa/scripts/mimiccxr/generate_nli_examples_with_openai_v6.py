import os
import argparse
import math
import random
import pandas as pd
from tqdm import tqdm
from medvqa.utils.text_data_utils import sentence_tokenize_texts_in_parallel
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.logging_utils import get_console_logger
from medvqa.datasets.mimiccxr import (
    MIMICCXR_FAST_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
)
from medvqa.utils.math_utils import rank_vectors_by_dot_product
from medvqa.utils.nlp import sort_sentences
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import load_json, load_jsonl
from medvqa.datasets.nli import MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH, RADNLI_DEV_JSONL_PATH, RADNLI_TEST_JSONL_PATH

INSTRUCTIONS = """Context: natural language inference.

Given a premise (#P) and a hypothesis (#H), output "Reason: {reason}. Label: {label}" where {reason} is a short sentence and {label} is one of "entailment," "contradiction," or "neutral."

Use "entailment" when the premise necessarily entails the truth of the hypothesis.

Use "contradiction" when premise and hypothesis are mutually exclusive/contradictory. Pay attention to subtle contradictions such as contradictory degrees of certainty, expressions suggesting presence vs. absence, etc.

Use "neutral" when there's no contradiction, but the premise doesn't necessarily entail the hypothesis.

Examples:

1. #P: increased pulmonary edema. | #H: worsened pulmonary edema.
Label: entailment

2. #P: No pulmonary edema, consolidation, or pneumothorax. | #H: No focal consolidation, pleural effusion, or pneumothorax is present.
Label: neutral"""

def parse_openai_model_output(text):
    """
    Parse the output of the OpenAI API call.
    """
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    text = text.lower()
    assert isinstance(text, str), f'Unexpected type: {type(text)} (text = {text})'
    if 'label: entailment' in text:
        return 'entailment'
    if 'label: contradiction' in text:
        return 'contradiction'
    if 'label: neutral' in text:
        return 'neutral'
    raise RuntimeError(f"Could not parse output: {text}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--preprocessed_queries_to_skip_filepaths", nargs="+", default=None)
    parser.add_argument("--preprocessed_reports_filepath", type=str, required=True)
    
    parser.add_argument("--cxr_bert_model_name", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--cxr_bert_checkpoint_folder_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_pairs_to_generate", type=int, default=20000)

    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_minute", type=int, required=True)
    parser.add_argument("--max_tokens_per_request", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    args = parser.parse_args()

    processed_texts_save_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_nli_queries_around_radnli_mscxrt{args.alias}.jsonl")
    
    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.api_responses_filepath is None:

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
                already_processed.update(row['metadata']['query'] for row in rows)
                logger.info(f"Loaded {len(rows)} queries to skip from {filepath}")

        # Collect sentences from reports        
        texts = []
        assert os.path.exists(args.preprocessed_reports_filepath)
        logger.info(f"Loading preprocessed reports from {args.preprocessed_reports_filepath}")
        reports = load_json(args.preprocessed_reports_filepath)
        for r in tqdm(reports, total=len(reports), mininterval=2):
            impression = r['impression']
            findings = r['findings']
            if len(impression) > 0:
                texts.append(impression)
            if len(findings) > 0:
                texts.append(findings)
        logger.info(f"Loaded {len(reports)} reports from {args.preprocessed_reports_filepath}")
        logger.info(f"Loaded {len(texts)} texts from reports")
        unique_sentences = set()
        sent_tokenized_texts = sentence_tokenize_texts_in_parallel(texts)
        for sentences in sent_tokenized_texts:
            unique_sentences.update(sentences)
        logger.info(f"Found {len(unique_sentences)} unique sentences in reports")
        assert len(unique_sentences) > 0

        # Collect sentences from RadNLI and MS-CXR-T
        gt_pairs = set()
        forbidden_sentences = set()
        radnli_dev_rows = load_jsonl(RADNLI_DEV_JSONL_PATH)
        radnli_test_rows = load_jsonl(RADNLI_TEST_JSONL_PATH)
        for rows in [radnli_dev_rows, radnli_test_rows]:
            for row in rows:
                premise = row["sentence1"]
                hypothesis = row["sentence2"]
                gt_pairs.add((premise, hypothesis))
                unique_sentences.add(premise)
                unique_sentences.add(hypothesis)
                forbidden_sentences.add(premise.lower())
                forbidden_sentences.add(hypothesis.lower())
        df = pd.read_csv(MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH)
        n = len(df)
        for premise, hypothesis in zip(df.sentence_1, df.sentence_2):
            gt_pairs.add((premise, hypothesis))
            unique_sentences.add(premise)
            unique_sentences.add(hypothesis)
            forbidden_sentences.add(premise.lower())
            forbidden_sentences.add(hypothesis.lower())

        logger.info(f"Found {len(gt_pairs)} ground truth pairs of sentences")
        logger.info(f"Found {len(unique_sentences)} unique sentences in reports, RadNLI, and MS-CXR-T")

        # Sort sentences
        unique_sentences = list(unique_sentences)
        unique_sentences = sort_sentences(unique_sentences, logger, by_difficulty=True, cache_ranking=True)

        # Obtain sentence embeddings
        emb_extractor = CachedTextEmbeddingExtractor(
            model_name=args.cxr_bert_model_name,
            model_checkpoint_folder_path=args.cxr_bert_checkpoint_folder_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        embeddings = emb_extractor.compute_text_embeddings(unique_sentences)
        logger.info(f"embeddings.shape: {embeddings.shape}")

        # Generate pairs around ground-truth pairs
        n_pairs_per_gt_pair = math.ceil(args.num_pairs_to_generate / len(gt_pairs))
        n_first_half = n_pairs_per_gt_pair // 2
        n_second_half = n_pairs_per_gt_pair - n_first_half
        assert n_first_half > 0
        assert n_second_half > 0
        logger.info(f'n_pairs_per_gt_pair = {n_pairs_per_gt_pair}')
        logger.info(f'n_first_half = {n_first_half}')
        logger.info(f'n_second_half = {n_second_half}')

        queries_to_skip = set()
        queries_to_skip.update(already_processed)
        queries_to_skip.update(f'#P: {p} | #H: {h}' for p, h in gt_pairs)
        logger.info(f'len(queries_to_skip) = {len(queries_to_skip)}')

        queries_to_make = []

        for gt_idx, gt_pair in tqdm(enumerate(gt_pairs), total=len(gt_pairs), mininterval=2):
            premise, hypothesis = gt_pair
            premise_idx = unique_sentences.index(premise)
            hypothesis_idx = unique_sentences.index(hypothesis)
            premise_emb = embeddings[premise_idx]
            hypothesis_emb = embeddings[hypothesis_idx]
            p_sorted_indices = rank_vectors_by_dot_product(embeddings, premise_emb)
            h_sorted_indices = rank_vectors_by_dot_product(embeddings, hypothesis_emb)
            assert len(p_sorted_indices) == len(h_sorted_indices)

            # queries closely similar to gt_pair
            i = 0
            count = 0
            while count < n_first_half:
                p_idx = p_sorted_indices[i]
                h_idx = h_sorted_indices[i]
                i += 1
                p = unique_sentences[p_idx]
                h = unique_sentences[h_idx]
                if p.lower() in forbidden_sentences or h.lower() in forbidden_sentences:
                    continue
                query = f'#P: {p} | #H: {h}'
                if query in queries_to_skip:
                    continue
                queries_to_make.append(query)
                queries_to_skip.add(query)
                count += 1

            # queries farther away
            idxs = list(range(i, i + max(n_second_half * 10, 300)))
            random.shuffle(idxs)
            for i in idxs:
                p_idx = p_sorted_indices[i]
                h_idx = h_sorted_indices[i]
                p = unique_sentences[p_idx]
                h = unique_sentences[h_idx]
                if p.lower() in forbidden_sentences or h.lower() in forbidden_sentences:
                    continue
                query = f'#P: {p} | #H: {h}'
                if query in queries_to_skip:
                    continue
                queries_to_make.append(query)
                queries_to_skip.add(query)
                count += 1
                if count == n_pairs_per_gt_pair:
                    break

            assert count > 0

            # Sanity checking
            if gt_idx < 10:
                logger.info("---------------------")
                logger.info(f"gt_pair: {gt_pair}")
                logger.info(f"queries_to_make[{-count}]: {queries_to_make[-count]}")
                logger.info(f"queries_to_make[{-1}]: {queries_to_make[-1]}")

        logger.info("===========================")
        logger.info(f"Total number of queries to make: {len(queries_to_make)}")

    else:
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
    )