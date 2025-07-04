import os
import argparse
import sys
import logging
import numpy as np
from nltk.tokenize import sent_tokenize
from medvqa.utils.common import get_timestamp
from medvqa.utils.logging_utils import setup_logging
from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.datasets.iuxray import IUXRAY_LARGE_FAST_CACHE_DIR, IUXRAY_REPORTS_MIN_JSON_PATH
from medvqa.utils.text_data_utils import parse_facts, remove_consecutive_repeated_words_from_text
from medvqa.utils.files_utils import load_json, load_jsonl, save_json

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def extract_facts(args):
    sentences_to_parse = set()

    # Collect unparsed sentences from reports
    iuxray_reports = load_json(IUXRAY_REPORTS_MIN_JSON_PATH)
    logger.info(f"Loaded {len(iuxray_reports)} reports from {IUXRAY_REPORTS_MIN_JSON_PATH}")
    for value in iuxray_reports.values():
        findings = value['findings']
        impression = value['impression']
        if impression:
            impression = impression.strip()
            for s in sent_tokenize(impression):
                sentences_to_parse.add(s)
        if findings:
            findings = findings.strip()
            for s in sent_tokenize(findings):
                sentences_to_parse.add(s)
    
    logger.info(f"Found {len(sentences_to_parse)} unique sentences to parse")
    if len(sentences_to_parse) == 0:
        logger.info("Nothing to do. Exiting.")
        sys.exit(0)

    sentences_to_parse = list(sentences_to_parse) # Convert to list for indexing
    
    # Print example sentences
    logger.info(f"Example sentences to parse:")
    indices = np.random.choice(len(sentences_to_parse), min(10, len(sentences_to_parse)), replace=False)
    for i in indices:
        logger.info(f"{i}: {sentences_to_parse[i]}")

    # Extract facts from sentences
    save_dir = os.path.join(IUXRAY_LARGE_FAST_CACHE_DIR, "huggingface")
    save_filename_prefix = "extracted_facts"
    def _postprocess_input_output_func(sentence, output_text):
        return {
            'sentence': sentence,
            'extracted_facts': parse_facts(output_text),
        }
    apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=args.checkpoint_folder_path,
        sentences=sentences_to_parse,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        num_beams=args.num_beams,
        save_dir=save_dir,
        save_filename_prefix=save_filename_prefix,
        postprocess_input_output_func=_postprocess_input_output_func,
    )


def integrate_facts(args):
    # Load the JSONL file with sentences and their extracted facts
    sentence_to_facts = load_jsonl(args.sentence_to_facts_jsonl_path)
    logger.info(f"Loaded {len(sentence_to_facts)} sentences with extracted facts from {args.sentence_to_facts_jsonl_path}")

    # Load the manual tags with slashes to sentences
    manual_tags_with_slashes_to_sentences = load_jsonl(args.manual_tags_with_slashes_to_sentences_jsonl_filepath)
    logger.info(f"Loaded {len(manual_tags_with_slashes_to_sentences)} manual tags with"
                f" slashes to sentences from {args.manual_tags_with_slashes_to_sentences_jsonl_filepath}")
    
    # Create a mapping from sentences to their facts
    sentence_to_facts = {item['sentence']: item['extracted_facts'] for item in sentence_to_facts}

    # Create a mapping from manual tags to facts
    manual_tags_to_fact = {item['metadata']['query']: item['parsed_response'] for item
                            in manual_tags_with_slashes_to_sentences}
    
    # Integrate facts into the IUXRAY reports
    final_reports = []
    iuxray_reports = load_json(IUXRAY_REPORTS_MIN_JSON_PATH)
    for key, value in iuxray_reports.items():
        report = {
            'key': key,
            'facts_from_findings': [],
            'facts_from_impression': [],
            'facts_from_manual_tags': [],
            **value,  # Copy other fields from the original report
        }
        
        # Add facts from sentences
        if value['findings']:
            findings = value['findings'].strip()
            for sentence in sent_tokenize(findings):
                if sentence in sentence_to_facts:
                    report['facts_from_findings'].extend(sentence_to_facts[sentence])
        
        if value['impression']:
            impression = value['impression'].strip()
            for sentence in sent_tokenize(impression):
                if sentence in sentence_to_facts:
                    report['facts_from_impression'].extend(sentence_to_facts[sentence])
        
        # Add facts from tags
        for tag in value['tags_manual']:
            report['facts_from_manual_tags'].append(manual_tags_to_fact.get(tag, tag))

        # Remove consecutive repeated words
        report['facts_from_findings'] = list(map(remove_consecutive_repeated_words_from_text, report['facts_from_findings']))
        report['facts_from_impression'] = list(map(remove_consecutive_repeated_words_from_text, report['facts_from_impression']))
        report['facts_from_manual_tags'] = list(map(remove_consecutive_repeated_words_from_text, report['facts_from_manual_tags']))

        # Remove duplicates while preserving order
        report['facts_from_findings'] = list(dict.fromkeys(report['facts_from_findings']))
        report['facts_from_impression'] = list(dict.fromkeys(report['facts_from_impression']))
        report['facts_from_manual_tags'] = list(dict.fromkeys(report['facts_from_manual_tags']))
        
        final_reports.append(report)

    # Save the final reports with integrated facts
    timestamp = get_timestamp()
    output_path = os.path.join(IUXRAY_LARGE_FAST_CACHE_DIR, f"iuxray_reports_with_facts_{timestamp}.json")
    logger.info(f"Saving {len(final_reports)} reports with integrated facts to {output_path}")
    save_json(final_reports, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract facts from sentences in IUXRAY reports using a seq2seq model.",
    )
    suparsers = parser.add_subparsers(dest='subcommand', required=True)

    # 1) subparser for extracting facts
    extract_facts_parser = suparsers.add_parser(
        'extract_facts',
        help="Extract facts from sentences in IUXRAY reports using a seq2seq model.",
    )
    extract_facts_parser.set_defaults(func=extract_facts)
    extract_facts_parser.add_argument("--checkpoint_folder_path", type=str, required=True)
    extract_facts_parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                                      help="Device to run the model on (default: cuda).")
    extract_facts_parser.add_argument("--batch_size", type=int, default=32)
    extract_facts_parser.add_argument("--num_workers", type=int, default=4)
    extract_facts_parser.add_argument("--max_length", type=int, default=512)
    extract_facts_parser.add_argument("--num_beams", type=int, default=1)

    # 2) subparser for report fact integration
    integrate_facts_parser = suparsers.add_parser(
        'integrate_facts',
        help="Integrate extracted facts into IUXRAY reports.",
    )
    integrate_facts_parser.set_defaults(func=integrate_facts)
    integrate_facts_parser.add_argument("--sentence_to_facts_jsonl_path", type=str, required=True,
                                        help="Path to the JSONL file containing sentences and their extracted facts.")
    integrate_facts_parser.add_argument("--manual_tags_with_slashes_to_sentences_jsonl_filepath", type=str, required=True,
                                        help="Path to the JSON file containing manual tags with slashes to sentences.")
    
    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()