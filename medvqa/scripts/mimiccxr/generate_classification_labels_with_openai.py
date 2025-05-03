import os
import argparse
import random
import numpy as np
from medvqa.utils.logging_utils import get_console_logger
from medvqa.datasets.mimiccxr import (
    MIMICCXR_LARGE_FAST_CACHE_DIR,
    MIMICCXR_FAST_TMP_DIR,
    get_path_to_report_text_dict,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, run_common_boilerplate_for_api_requests
from medvqa.utils.files_utils import get_file_path_with_hashing_if_too_long, load_jsonl, load_pickle, save_pickle

INSTRUCTIONS = """Given a report (#R) and a hypothesis (#H), output "Reason: {reason}. Label: {label}" where {reason} must be a short explanation (no more than 5 sentences) and {label} must be one of {certainly true, probably true, unknown, probably false, certainly false}. Remember that unknown applies when it's not possible to know whether #H is true or false with the information provided."""

POSSIBLE_LABELS = [
    "label: certainly true",
    "label: probably true",
    "label: unknown",
    "label: probably false",
    "label: certainly false",
]

LABEL_TO_BINARY = {
    "certainly true": 1,
    "probably true": 1,
    "unknown": 0,
    "probably false": 0,
    "certainly false": 0,
}

VINBIG_LABEL_TO_HYPOTHESIS = {
    'Aortic enlargement': 'according to the report, the current chest X-ray(s) provide visual evidence of aortic enlargement',
    'Atelectasis': 'according to the report, the current chest X-ray(s) provide visual evidence of atelectasis',
    'Calcification': 'according to the report, the current chest X-ray(s) provide visual evidence of calcification',
    'Cardiomegaly': 'according to the report, the current chest X-ray(s) provide visual evidence of cardiomegaly',
    'Clavicle fracture': 'according to the report, the current chest X-ray(s) provide visual evidence of a clavicle fracture',
    'Consolidation': 'according to the report, the current chest X-ray(s) provide visual evidence of consolidation',
    'Edema': 'according to the report, the current chest X-ray(s) provide visual evidence of edema',
    'Emphysema': 'according to the report, the current chest X-ray(s) provide visual evidence of emphysema',
    'Enlarged PA': 'according to the report, the current chest X-ray(s) provide visual evidence of an enlarged pulmonary artery',
    'ILD': 'according to the report, the current chest X-ray(s) provide visual evidence of interstitial lung disease',
    'Infiltration': 'according to the report, the current chest X-ray(s) provide visual evidence of infiltration',
    'Lung Opacity': 'according to the report, the current chest X-ray(s) provide visual evidence of lung opacity',
    'Lung cavity': 'according to the report, the current chest X-ray(s) provide visual evidence of a lung cavity',
    'Lung cyst': 'according to the report, the current chest X-ray(s) provide visual evidence of a lung cyst',
    'Mediastinal shift': 'according to the report, the current chest X-ray(s) provide visual evidence of a mediastinal shift',
    'Nodule/Mass': 'according to the report, the current chest X-ray(s) provide visual evidence of a nodule/mass',
    'Pleural effusion': 'according to the report, the current chest X-ray(s) provide visual evidence of pleural effusion',
    'Pleural thickening': 'according to the report, the current chest X-ray(s) provide visual evidence of pleural thickening',
    'Pneumothorax': 'according to the report, the current chest X-ray(s) provide visual evidence of a pneumothorax',
    'Pulmonary fibrosis': 'according to the report, the current chest X-ray(s) provide visual evidence of pulmonary fibrosis',
    'Rib fracture': 'according to the report, the current chest X-ray(s) provide visual evidence of a rib fracture',
    # 'Other lesion' # Not used
    'COPD': 'according to the report, the current chest X-ray(s) provide visual evidence of COPD',
    'Lung tumor': 'according to the report, the current chest X-ray(s) provide visual evidence of a lung tumor',
    'Pneumonia': 'according to the report, the current chest X-ray(s) provide visual evidence of pneumonia',
    'Tuberculosis': 'according to the report, the current chest X-ray(s) provide visual evidence of tuberculosis',
    # 'Other disease', # Not used
    # 'No finding', # Not used
    'Abnormal finding': 'according to the report, the current chest X-ray(s) provide visual evidence of abnormal findings',
}

_ANCHOR_TO_ADDITIONAL_ENTAILED_ANCHORS = {
    "Atelectasis": ["Lung Opacity"],
    "Consolidation": ["Lung Opacity"],
    "Edema": ["Lung Opacity"],
    "ILD": ["Lung Opacity"],
    "Infiltration": ["Lung Opacity"],
    "Nodule/Mass": ["Lung Opacity"],
    "Pneumonia": ["Lung Opacity"],
    "Pulmonary fibrosis": ["Lung Opacity"],
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
            assert len(reason) > 0, f"Empty reason: {reason}"
            label = label[7:] # Remove "label: "
            return {
                "reason": reason,
                "label": label,
            }
        except ValueError:
            continue
    raise RuntimeError(f"Could not parse output: {text}")

_report_cache = {}
def _get_report(filepath):
    path_to_report_text_dict =  get_path_to_report_text_dict()
    report = path_to_report_text_dict[filepath]
    report = ' '.join(report.split())
    _report_cache[filepath] = report
    return report

def _build_query(report, hypothesis):
    return f"#R: {report} | #H: {hypothesis}"

def _generate_all_queries_for_reports(report_paths):
    queries = []
    for report_path in report_paths:
        report = _get_report(report_path)
        for hypothesis in VINBIG_LABEL_TO_HYPOTHESIS.values():
            query = _build_query(report, hypothesis)
            queries.append(query)
    return queries

def _find_mimiccxr_reports_samples_from_split(
        mimiccxr_reports_detailed_metadata, min_num_samples_per_class, min_num_samples_without_class, total_samples,
        dicom_id_to_pos_facts, anchors_per_fact, anchor_facts, all_facts, relevant_facts, logger, split):
    assert split in ['train', 'validate', 'test']

    logger.info(f"Finding reports for the {split} subset")
    
    # Load split set    
    candidate_ridxs = [i for i, x in enumerate(mimiccxr_reports_detailed_metadata['splits']) if x == split]
    assert len(candidate_ridxs) > 0
    logger.info(f"Found {len(candidate_ridxs)} reports in the {split} set")

    if total_samples > len(candidate_ridxs):
        logger.warning(f"Total number of samples ({total_samples}) is greater than the number of candidate reports ({len(candidate_ridxs)})")
        total_samples = len(candidate_ridxs)
    
    # For each anchor fact, find all reports that mention it
    anchor_to_ridxs = [[] for _ in range(len(anchor_facts))]
    noanchor_to_ridxs = []
    rel_fact_to_idx = {x: i for i, x in enumerate(relevant_facts)}
    for ridx in candidate_ridxs:
        dicom_id = mimiccxr_reports_detailed_metadata['dicom_id_view_pos_pairs'][ridx][0][0]
        pos_fact_idxs = dicom_id_to_pos_facts[dicom_id]
        pos_rel_fact_idxs = [rel_fact_to_idx[all_facts[x]] for x in pos_fact_idxs]
        anchor_idxs = set()
        for fidx in pos_rel_fact_idxs:
            anchor_idxs.update(anchors_per_fact[fidx])
        for anchor_idx in anchor_idxs:
            anchor_to_ridxs[anchor_idx].append(ridx)
        if len(anchor_idxs) == 0:
            noanchor_to_ridxs.append(ridx)
        
    messages = []
    for anchor_fact, ridxs in zip(anchor_facts, anchor_to_ridxs):
        messages.append((len(ridxs), f"Found {len(ridxs)} reports mentioning the anchor fact: {anchor_fact}"))
    messages.sort(key=lambda x: x[0], reverse=True)
    for message in messages:
        logger.info(message[1])
    logger.info(f"Found {len(noanchor_to_ridxs)} reports with no anchor facts")

    # Sample reports to fit the number of samples
    sampled_ridxs = set()
    for ridxs in anchor_to_ridxs:
        unused_ridxs = [x for x in ridxs if x not in sampled_ridxs]
        used_count = len(ridxs) - len(unused_ridxs)
        if used_count >= min_num_samples_per_class:
            continue
        num_samples = min(min_num_samples_per_class - used_count, len(unused_ridxs))
        sampled_ridxs.update(random.sample(unused_ridxs, num_samples))
    sampled_ridxs.update(random.sample(noanchor_to_ridxs, min(min_num_samples_without_class, len(noanchor_to_ridxs)))
                         if len(noanchor_to_ridxs) > 0 else [])

    logger.info(f"Sampled {len(sampled_ridxs)} reports for the {split} subset (after sampling anchor facts)")
    
    if len(sampled_ridxs) < total_samples:
        unused_ridxs = [x for x in candidate_ridxs if x not in sampled_ridxs]
        num_samples = min(total_samples - len(sampled_ridxs), len(unused_ridxs))
        sampled_ridxs.update(random.sample(unused_ridxs, num_samples))

    assert len(sampled_ridxs) == total_samples
    logger.info(f"Sampled {len(sampled_ridxs)} reports for the {split} subset (after sampling the rest)")

    return sampled_ridxs

def _get_queries_from_missing_report_class_combinations(filepath, logger):
    rows = load_jsonl(filepath)
    seen_queries = set()
    reports = set()
    for row in rows:
        query = row['metadata']['query']
        seen_queries.add(query)
        report = query[4:query.index(" | #H: ")]
        reports.add(report)
    logger.info(f"Loaded {len(rows)} queries from {filepath}")
    logger.info(f"Found {len(reports)} unique reports")
    logger.info(f"Number of missing report-class combinations: {len(VINBIG_LABEL_TO_HYPOTHESIS) * len(reports) - len(rows)}")
    queries = []
    for report in reports:
        for hypothesis in VINBIG_LABEL_TO_HYPOTHESIS.values():
            query = _build_query(report, hypothesis)
            if query not in seen_queries:
                queries.append(query)
    logger.info(f"Generated {len(queries)} queries for missing report-class combinations")
    return queries

def _generate_class_specific_queries_from_train(
        classes_to_sample, mimiccxr_reports_detailed_metadata, num_pos_samples, num_neg_samples, 
        dicom_id_to_pos_facts, anchors_per_fact, anchor_facts, all_facts, relevant_facts, logger):

    logger.info(f"Generating queries for the classes: {classes_to_sample}")
    
    # Load training set    
    train_ridxs = [i for i, x in enumerate(mimiccxr_reports_detailed_metadata['splits']) if x == 'train']
    assert len(train_ridxs) > 0
    logger.info(f"Found {len(train_ridxs)} reports in the training set")
    
    # Find reports that mention the anchor facts
    aidx_to_additional_aidxs = {}
    for anchor, additional_anchors in _ANCHOR_TO_ADDITIONAL_ENTAILED_ANCHORS.items():
        aidx = anchor_facts.index(anchor)
        aidx_to_additional_aidxs[aidx] = [anchor_facts.index(x) for x in additional_anchors]

    target_anchor_idxs = [anchor_facts.index(x) if x != 'Abnormal finding' else -1 for x in classes_to_sample]
    pos_ridxs_per_anchor = [[] for _ in range(len(anchor_facts) + 1)]
    neg_ridxs_per_anchor = [[] for _ in range(len(anchor_facts) + 1)]
    rel_fact_to_idx = {x: i for i, x in enumerate(relevant_facts)}
    for ridx in train_ridxs:
        dicom_id = mimiccxr_reports_detailed_metadata['dicom_id_view_pos_pairs'][ridx][0][0]
        pos_fact_idxs = dicom_id_to_pos_facts[dicom_id]
        pos_rel_fact_idxs = [rel_fact_to_idx[all_facts[x]] for x in pos_fact_idxs]
        anchor_idxs = set()
        for fidx in pos_rel_fact_idxs: # Include anchor facts
            anchor_idxs.update(anchors_per_fact[fidx])
        for aidx in list(anchor_idxs): # Include additional anchors
            if aidx in aidx_to_additional_aidxs:
                for additional_aidx in aidx_to_additional_aidxs[aidx]:
                    anchor_idxs.add(additional_aidx)
        for target_anchor_id in target_anchor_idxs:
            if target_anchor_id in anchor_idxs:
                pos_ridxs_per_anchor[target_anchor_id].append(ridx)
            elif target_anchor_id == -1 and len(anchor_idxs) > 0:
                pos_ridxs_per_anchor[target_anchor_id].append(ridx)                
            else:
                neg_ridxs_per_anchor[target_anchor_id].append(ridx)
    for target_anchor_id, class_to_sample in zip(target_anchor_idxs, classes_to_sample):
        pos_ridxs = pos_ridxs_per_anchor[target_anchor_id]
        neg_ridxs = neg_ridxs_per_anchor[target_anchor_id]
        assert len(pos_ridxs) > 0
        assert len(neg_ridxs) > 0
        logger.info(f"Found {len(pos_ridxs)} with and {len(neg_ridxs)} without the anchor fact: {class_to_sample}")

    # Sample reports to fit the number of samples and generate queries
    queries = []

    for target_anchor_id, class_to_sample in zip(target_anchor_idxs, classes_to_sample):
        pos_ridxs = pos_ridxs_per_anchor[target_anchor_id]
        neg_ridxs = neg_ridxs_per_anchor[target_anchor_id]
    
        sampled_ridxs = []
        num_samples = min(num_pos_samples, len(pos_ridxs))
        sampled_ridxs.extend(random.sample(pos_ridxs, num_samples)) # Positive samples
        num_samples = min(num_neg_samples, len(neg_ridxs))
        sampled_ridxs.extend(random.sample(neg_ridxs, num_samples)) # Negative samples
        
        hypothesis = VINBIG_LABEL_TO_HYPOTHESIS[class_to_sample]
        for ridx in sampled_ridxs:
            report_path = mimiccxr_reports_detailed_metadata['filepaths'][ridx]
            report = _get_report(report_path)
            query = _build_query(report, hypothesis)
            queries.append(query)

        logger.info(f"Sampled {len(sampled_ridxs)} reports for the class: {class_to_sample}")

    return queries

def _export_val_test_labels(logger, gpt4_mined_labels_jsonl_filepath, annotation_alias):
    
    # Map report text to DICOM ID
    path_to_report_text_dict = get_path_to_report_text_dict()
    mimiccxr_reports_detailed_metadata = load_mimiccxr_reports_detailed_metadata()
    report_to_data = dict()
    val_dicom_ids = set()
    test_dicom_ids = set()
    for filepath, dicom_id_view_pos_pairs, split in zip(
        mimiccxr_reports_detailed_metadata['filepaths'],
        mimiccxr_reports_detailed_metadata['dicom_id_view_pos_pairs'],
        mimiccxr_reports_detailed_metadata['splits']
    ):
        if split == 'validate':
            for dicom_id, _ in dicom_id_view_pos_pairs:
                val_dicom_ids.add(dicom_id)
        elif split == 'test':
            for dicom_id, _ in dicom_id_view_pos_pairs:
                test_dicom_ids.add(dicom_id)
        else:
            assert split == 'train'
            continue
        report_text = path_to_report_text_dict[filepath]
        report_text = ' '.join(report_text.split())
        try:
            data = report_to_data[report_text]
        except KeyError:
            data = report_to_data[report_text] = dict(
                dicom_ids=[],
                pos_labels=[],
                neg_labels=[],
            )
        for dicom_id, _ in dicom_id_view_pos_pairs:
            data['dicom_ids'].append(dicom_id)

    assert len(val_dicom_ids) > 0
    assert len(test_dicom_ids) > 0

    logger.info(f"Found {len(val_dicom_ids)} DICOM IDs in the validation set")
    logger.info(f"Found {len(test_dicom_ids)} DICOM IDs in the test set")

    # TODO: support other classes apart from VINBIG_LABEL_TO_HYPOTHESIS
    hypothesis_to_class_name = {v: k for k, v in VINBIG_LABEL_TO_HYPOTHESIS.items()}
    class_names = list(VINBIG_LABEL_TO_HYPOTHESIS.keys())

    # Load GPT-4 mined labels
    gpt4_mined_labels = load_jsonl(gpt4_mined_labels_jsonl_filepath)
    for item in gpt4_mined_labels:
        query = item['metadata']['query']
        i = query.index(' | #H: ')
        report = query[4:i]
        hypothesis = query[i+7:]
        class_name = hypothesis_to_class_name[hypothesis]
        class_name_idx = class_names.index(class_name)
        label = item['parsed_response']['label']
        label = LABEL_TO_BINARY[label]
        try:
            data = report_to_data[report]
        except KeyError:
            continue
        data['pos_labels' if label == 1 else 'neg_labels'].append(class_name_idx) # 1: positive, 0: negative

    # Export labels
    val_dicom_id_to_labels = dict()
    test_dicom_ids_to_labels = dict()
    for data in report_to_data.values():
        dicom_ids = data['dicom_ids']
        pos_labels = data['pos_labels']
        neg_labels = data['neg_labels']
        if len(pos_labels) + len(neg_labels) == 0:
            continue
        try:
            assert len(pos_labels) + len(neg_labels) == len(VINBIG_LABEL_TO_HYPOTHESIS) # All classes should be labeled
        except AssertionError:
            logger.warning(f"Missing labels for DICOM IDs: {dicom_ids}")
            logger.warning(f"len(pos_labels): {len(pos_labels)}")
            logger.warning(f"len(neg_labels): {len(neg_labels)}")
            logger.warning(f"len(pos_labels) + len(neg_labels): {len(pos_labels) + len(neg_labels)}")
            logger.warning(f"pos_labels: {pos_labels}")
            logger.warning(f"neg_labels: {neg_labels}")
            continue
        label_array = np.full(len(VINBIG_LABEL_TO_HYPOTHESIS), -1, dtype=np.int32) # -1: unknown
        label_array[pos_labels] = 1
        label_array[neg_labels] = 0
        assert np.all(label_array != -1)
        for dicom_id in dicom_ids:
            if dicom_id in val_dicom_ids:
                val_dicom_id_to_labels[dicom_id] = label_array
            elif dicom_id in test_dicom_ids:
                test_dicom_ids_to_labels[dicom_id] = label_array
            else:
                assert False, f"Unexpected DICOM ID: {dicom_id}"

    assert len(val_dicom_id_to_labels) > 0
    assert len(test_dicom_ids_to_labels) > 0
    
    val_output = dict(
        dicom_id_to_labels=val_dicom_id_to_labels,
        label_names=class_names,
    )

    test_output = dict(
        dicom_id_to_labels=test_dicom_ids_to_labels,
        label_names=class_names,
    )
    
    # Save labels
                                            
    val_labels_save_filepath = get_file_path_with_hashing_if_too_long(
        prefix=f'mimiccxr_gpt4_val_labels({annotation_alias},nc={len(class_names)},nd={len(val_dicom_id_to_labels)})',
        folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
        strings=[
            gpt4_mined_labels_jsonl_filepath,
        ],
        force_hashing=True,
    )
    
    test_labels_save_filepath = get_file_path_with_hashing_if_too_long(
        prefix=f'mimiccxr_gpt4_test_labels({annotation_alias},nc={len(class_names)},nd={len(test_dicom_ids_to_labels)})',
        folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
        strings=[
            gpt4_mined_labels_jsonl_filepath,
        ],
        force_hashing=True,
    )

    logger.info(f"Exporting {len(val_dicom_id_to_labels)} validation DICOM IDs to {val_labels_save_filepath}")
    logger.info(f"Exporting {len(test_dicom_ids_to_labels)} test DICOM IDs to {test_labels_save_filepath}")

    save_pickle(val_output, val_labels_save_filepath)
    save_pickle(test_output, test_labels_save_filepath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mimiccxr_dicom_id_to_pos_neg_facts_filepath", type=str, default=None)
    parser.add_argument("--mimiccxr_facts_relevant_to_anchor_facts_filepath", type=str, default=None)
    parser.add_argument("--min_num_samples_per_class", type=int, default=100)
    parser.add_argument("--min_num_samples_without_class", type=int, default=100)
    parser.add_argument("--total_samples", type=int, default=1000)
    parser.add_argument("--split", type=str, default=None, choices=["train", "validate", "test"])
    parser.add_argument("--queries_to_skip_filepaths", type=str, nargs="+", default=None)
    parser.add_argument("--process_missing_report_class_combinations", action="store_true", default=False)
    parser.add_argument("--queries_with_missing_report_class_combinations_filepath", type=str, default=None)
    parser.add_argument("--sample_class_specific_queries_from_train", action="store_true", default=False)
    parser.add_argument("--classes_to_sample", type=str, nargs="+", default=None)
    parser.add_argument("--num_pos_samples", type=int, default=100)
    parser.add_argument("--num_neg_samples", type=int, default=100)
    parser.add_argument("--export_val_test_labels", action="store_true", default=False)
    parser.add_argument("--gpt4_mined_labels_jsonl_filepath", type=str, default=None)
    
    parser.add_argument("--openai_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--openai_request_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_requests_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_request", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logging_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--alias", type=str, default="")
    parser.add_argument("--not_delete_api_requests_and_responses", action="store_true", default=False)
    parser.add_argument("--api_responses_filepath", type=str, default=None)
    parser.add_argument("--use_batch_api", action="store_true", default=False)
    parser.add_argument("--batch_description", type=str, default=None)
    parser.add_argument("--batch_input_file_id", type=str, default=None)

    args = parser.parse_args()

    # Set up logging
    logger = get_console_logger(args.logging_level)

    if args.export_val_test_labels: # Export validation and test labels
        assert args.gpt4_mined_labels_jsonl_filepath is not None
        assert len(args.alias) > 0
        _export_val_test_labels(
            logger=logger,
            gpt4_mined_labels_jsonl_filepath=args.gpt4_mined_labels_jsonl_filepath,
            annotation_alias=args.alias,
        )
        exit()

    processed_queries_save_filepath = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "openai", f"{args.openai_model_name}_mimiccxr_report_classification_labels{args.alias}.jsonl")

    if args.api_responses_filepath is None and args.batch_input_file_id is None:

        # Load already processed queries if they exist
        already_processed = set()
        if os.path.exists(processed_queries_save_filepath):
            rows = load_jsonl(processed_queries_save_filepath)
            already_processed.update([x['metadata']['query'] for x in rows])
            logger.info(f"Loaded {len(rows)} already processed queries from {processed_queries_save_filepath}")

        # Load queries to skip
        if args.queries_to_skip_filepaths is not None:
            for queries_to_skip_filepath in args.queries_to_skip_filepaths:
                rows = load_jsonl(queries_to_skip_filepath)
                already_processed.update([x['metadata']['query'] for x in rows])
                logger.info(f"Loaded {len(rows)} queries to skip from {queries_to_skip_filepath}")

        # Sample queries

        if args.process_missing_report_class_combinations:
            assert args.queries_with_missing_report_class_combinations_filepath is not None
            queries_to_process = _get_queries_from_missing_report_class_combinations(
                args.queries_with_missing_report_class_combinations_filepath, logger)
        
        elif args.sample_class_specific_queries_from_train:
            assert args.classes_to_sample is not None
            assert args.num_pos_samples is not None
            assert args.num_neg_samples is not None
            assert args.mimiccxr_dicom_id_to_pos_neg_facts_filepath is not None
            assert args.mimiccxr_facts_relevant_to_anchor_facts_filepath is not None

            mimiccxr_reports_detailed_metadata = load_mimiccxr_reports_detailed_metadata()

            logger.info(f"Loading positive and negative facts for each DICOM ID from {args.mimiccxr_dicom_id_to_pos_neg_facts_filepath}")
            mimiccxr_dicom_id_to_pos_neg_facts = load_pickle(args.mimiccxr_dicom_id_to_pos_neg_facts_filepath)
            dicom_id_to_pos_facts = mimiccxr_dicom_id_to_pos_neg_facts['dicom_id_to_pos_facts']
            all_facts = mimiccxr_dicom_id_to_pos_neg_facts['facts']
            
            logger.info(f"Loading anchor facts and relevant facts from {args.mimiccxr_facts_relevant_to_anchor_facts_filepath}")
            mimiccxr_facts_relevant_to_anchor_facts = load_pickle(args.mimiccxr_facts_relevant_to_anchor_facts_filepath)
            anchors_per_fact = mimiccxr_facts_relevant_to_anchor_facts['anchors_per_fact']
            anchor_facts = mimiccxr_facts_relevant_to_anchor_facts['anchor_facts']
            relevant_facts = mimiccxr_facts_relevant_to_anchor_facts['relevant_facts']

            queries_to_process = _generate_class_specific_queries_from_train(
                classes_to_sample=args.classes_to_sample,
                mimiccxr_reports_detailed_metadata=mimiccxr_reports_detailed_metadata,
                num_pos_samples=args.num_pos_samples,
                num_neg_samples=args.num_neg_samples,
                dicom_id_to_pos_facts=dicom_id_to_pos_facts,
                anchors_per_fact=anchors_per_fact,
                anchor_facts=anchor_facts,
                all_facts=all_facts,
                relevant_facts=relevant_facts,
                logger=logger,
            )

        else:
            mimiccxr_reports_detailed_metadata = load_mimiccxr_reports_detailed_metadata()

            logger.info(f"Loading positive and negative facts for each DICOM ID from {args.mimiccxr_dicom_id_to_pos_neg_facts_filepath}")
            mimiccxr_dicom_id_to_pos_neg_facts = load_pickle(args.mimiccxr_dicom_id_to_pos_neg_facts_filepath)
            dicom_id_to_pos_facts = mimiccxr_dicom_id_to_pos_neg_facts['dicom_id_to_pos_facts']
            all_facts = mimiccxr_dicom_id_to_pos_neg_facts['facts']
            
            logger.info(f"Loading anchor facts and relevant facts from {args.mimiccxr_facts_relevant_to_anchor_facts_filepath}")
            mimiccxr_facts_relevant_to_anchor_facts = load_pickle(args.mimiccxr_facts_relevant_to_anchor_facts_filepath)
            anchors_per_fact = mimiccxr_facts_relevant_to_anchor_facts['anchors_per_fact']
            anchor_facts = mimiccxr_facts_relevant_to_anchor_facts['anchor_facts']
            relevant_facts = mimiccxr_facts_relevant_to_anchor_facts['relevant_facts']
            
            report_idxs = _find_mimiccxr_reports_samples_from_split(
                mimiccxr_reports_detailed_metadata=mimiccxr_reports_detailed_metadata,
                min_num_samples_per_class=args.min_num_samples_per_class,
                min_num_samples_without_class=args.min_num_samples_without_class,
                total_samples=args.total_samples,
                dicom_id_to_pos_facts=dicom_id_to_pos_facts,
                anchors_per_fact=anchors_per_fact,
                anchor_facts=anchor_facts,
                all_facts=all_facts,
                relevant_facts=relevant_facts,
                logger=logger,
                split=args.split,
            )
            report_paths = [mimiccxr_reports_detailed_metadata['filepaths'][i] for i in report_idxs]
            queries_to_process = _generate_all_queries_for_reports(report_paths)
        
        logger.info(f"Total number of queries to process: {len(queries_to_process)}")
        
        queries_to_process = [x for x in queries_to_process if x not in already_processed]
        logger.info(f"Number of queries to process after filtering out already processed queries: {len(queries_to_process)}")

        if len(queries_to_process) == 0:
            logger.info("No queries to process. Exiting...")
            exit()

        # Print example queries
        logger.info(f"Example queries to process:")
        for i in np.linspace(0, len(queries_to_process)-1, min(10, len(queries_to_process)), dtype=int):
            logger.info(f"{i+1}. {queries_to_process[i]}")

    else:
        if args.api_responses_filepath is not None:
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
        use_batch_api=args.use_batch_api,
        batch_description=args.batch_description,
        batch_input_file_id=args.batch_input_file_id,
        delete_api_requests_and_responses=not args.not_delete_api_requests_and_responses,
    )