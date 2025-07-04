import os
import argparse
import sys
import numpy as np
import logging
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR,MIMICCXR_FAST_CACHE_DIR, MIMICCXR_LARGE_FAST_CACHE_DIR
from medvqa.utils.common import get_timestamp
from medvqa.utils.logging_utils import setup_logging
from medvqa.utils.text_data_utils import parse_facts, remove_consecutive_repeated_words_from_text
from medvqa.utils.files_utils import load_json, load_jsonl, save_json

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def extract_facts(args):
    sentences_to_parse = set()
    already_parsed_sentences = set()

    # Load preprocessed sentences to skip if they exist
    if args.preprocessed_sentences_to_skip_filepaths is not None:
        for filepath in args.preprocessed_sentences_to_skip_filepaths:
            assert os.path.exists(filepath), f"File {filepath} does not exist."
            sentences_to_skip = load_jsonl(filepath)
            logger.info(f"Loaded {len(sentences_to_skip)} sentences to skip from {filepath}")
            for row in sentences_to_skip:
                try:
                    already_parsed_sentences.add(row['metadata']['query'])
                except KeyError:
                    already_parsed_sentences.add(row['metadata']['sentence']) # backward compatibility
    logger.info(f"Total number of sentences to skip: {len(already_parsed_sentences)}")

    # Collect unparsed sentences from reports
    preprocessed_reports_filepath = os.path.join(MIMICCXR_CACHE_DIR, args.preprocessed_reports_filename)
    logger.info(f"Loading preprocessed reports from {preprocessed_reports_filepath}")
    reports = load_json(preprocessed_reports_filepath)
    logger.info(f"Loaded {len(reports)} reports from {preprocessed_reports_filepath}")
    for r in tqdm(reports, desc="Collecting sentences from reports", total=len(reports), mininterval=2):
        impression = r['impression']
        findings = r['findings']
        if impression:
            for s in sent_tokenize(impression):
                if s not in already_parsed_sentences:
                    sentences_to_parse.add(s)
        if findings:
            for s in sent_tokenize(findings):
                if s not in already_parsed_sentences:
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
    save_dir = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, "huggingface")
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
    # Load the JSONL files with sentences and their extracted facts
    sentence_to_facts = {}
    for filepath in args.sentence_to_facts_jsonl_paths:
        items = load_jsonl(filepath)
        logger.info(f"Loaded {len(items)} sentences with extracted facts from {filepath}")
        for item in items:
            if 'metadata' in item:
                try:
                    sentence = item['metadata']['sentence']
                except KeyError:
                    sentence = item['metadata']['query']
                parsed_response = item['parsed_response']
                if 'facts' in parsed_response:
                    facts = parsed_response['facts']
                else:
                    facts = parsed_response
                assert isinstance(facts, list), f"Expected facts to be a list, got {type(facts)}"
            else:
                sentence = item['sentence']
                facts = item['extracted_facts']            
            sentence_to_facts[sentence] = facts
    logger.info(f"Total unique sentences with extracted facts: {len(sentence_to_facts)}")
    
    # Integrate facts into the MIMIC-CXR reports
    final_reports = []
    preprocessed_reports_filepath = os.path.join(MIMICCXR_CACHE_DIR, args.preprocessed_reports_filename)
    logger.info(f"Loading preprocessed reports from {preprocessed_reports_filepath}")
    reports = load_json(preprocessed_reports_filepath)
    logger.info(f"Loaded {len(reports)} reports from {preprocessed_reports_filepath}")
    for r in tqdm(reports, desc="Integrating facts into reports", total=len(reports), mininterval=2):
        report = {
            'facts_from_findings': [],
            'facts_from_impression': [],
            **r,  # Copy other fields from the original report
        }
        
        # Add facts from sentences
        findings = r['findings']
        if findings:
            for sentence in sent_tokenize(findings):
                report['facts_from_findings'].extend(sentence_to_facts[sentence])
        
        impression = r['impression']
        if impression:
            for sentence in sent_tokenize(impression):
                report['facts_from_impression'].extend(sentence_to_facts[sentence])

        # Remove consecutive repeated words
        report['facts_from_findings'] = list(map(remove_consecutive_repeated_words_from_text, report['facts_from_findings']))
        report['facts_from_impression'] = list(map(remove_consecutive_repeated_words_from_text, report['facts_from_impression']))

        # Remove duplicates while preserving order
        report['facts_from_findings'] = list(dict.fromkeys(report['facts_from_findings']))
        report['facts_from_impression'] = list(dict.fromkeys(report['facts_from_impression']))
        
        final_reports.append(report)

    # Save the final reports with integrated facts
    timestamp = get_timestamp()
    output_path = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, f"mimiccxr_reports_with_facts_{timestamp}.json")
    logger.info(f"Saving {len(final_reports)} reports with integrated facts to {output_path}")
    save_json(final_reports, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract facts from sentences in MIMIC-CXR reports using a seq2seq model.",
    )
    suparsers = parser.add_subparsers(dest='subcommand', required=True)

    # 1) subparser for extracting facts
    extract_facts_parser = suparsers.add_parser(
        'extract_facts',
        help="Extract facts from sentences in MIMIC-CXR reports using a seq2seq model.",
    )
    extract_facts_parser.set_defaults(func=extract_facts)
    extract_facts_parser.add_argument("--checkpoint_folder_path", type=str, required=True)
    extract_facts_parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                                      help="Device to run the model on (default: cuda).")
    extract_facts_parser.add_argument("--batch_size", type=int, default=32)
    extract_facts_parser.add_argument("--num_workers", type=int, default=4)
    extract_facts_parser.add_argument("--max_length", type=int, default=512)
    extract_facts_parser.add_argument("--num_beams", type=int, default=1)
    extract_facts_parser.add_argument("--preprocessed_reports_filename", type=str, required=True,
                                       help="Filename of the preprocessed reports JSON file.")
    extract_facts_parser.add_argument("--preprocessed_sentences_to_skip_filepaths", type=str, nargs='*',
                                       help="List of JSONL files containing sentences to skip (optional).")

    # 2) subparser for report fact integration
    integrate_facts_parser = suparsers.add_parser(
        'integrate_facts',
        help="Integrate extracted facts into MIMIC-CXR reports.",
    )
    integrate_facts_parser.set_defaults(func=integrate_facts)
    integrate_facts_parser.add_argument("--preprocessed_reports_filename", type=str, required=True,
                                         help="Filename of the preprocessed reports JSON file.")
    integrate_facts_parser.add_argument("--sentence_to_facts_jsonl_paths", type=str, nargs='+', required=True,
                                        help="List of JSONL files containing sentences and their extracted facts.")
    
    # Parse arguments and call the appropriate function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()