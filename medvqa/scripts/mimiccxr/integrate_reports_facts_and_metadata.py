import argparse
import os

from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.datasets.mimiccxr.report_utils import integrate_reports_facts_and_metadata, _FACT_METADATA_FIELDS
from medvqa.utils.files import save_jsonl
from medvqa.utils.logging import print_red

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_reports_filepath', type=str, required=True)
    parser.add_argument('--extracted_facts_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--fact_extraction_methods', type=str, nargs='+', required=True)
    parser.add_argument('--extracted_metadata_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--metadata_extraction_methods', type=str, nargs='+', required=True)
    args = parser.parse_args()
    assert len(args.extracted_facts_filepaths) == len(args.fact_extraction_methods)
    assert len(args.extracted_metadata_filepaths) == len(args.metadata_extraction_methods)

    # Integrate reports, facts and metadata
    sentence_facts_rows, fact_metadata_rows, report_facts_metadata_rows, facts_without_metadata = integrate_reports_facts_and_metadata(
        preprocessed_reports_filepath=args.preprocessed_reports_filepath,
        extracted_facts_filepaths=args.extracted_facts_filepaths,
        fact_extraction_methods=args.fact_extraction_methods,
        extracted_metadata_filepaths=args.extracted_metadata_filepaths,
        metadata_extraction_methods=args.metadata_extraction_methods,
        remove_consecutive_repeated_words=True)

    # Save integrated reports, facts and metadata
    sentence_total_length = 0
    facts_total_length = 0
    for row in sentence_facts_rows:
        sentence_total_length += len(row['sentence'])
        for fact in row['facts']:
            facts_total_length += len(fact)
    integrated_sentence_facts_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, f'integrated_sentence_facts({sentence_total_length},{facts_total_length}).jsonl')
    print(f'Saving integrated sentence facts to {integrated_sentence_facts_filepath}')
    save_jsonl(sentence_facts_rows, integrated_sentence_facts_filepath)

    metadata_total_length = 0
    for row in fact_metadata_rows:
        for key in _FACT_METADATA_FIELDS:
            metadata_total_length += len(row['metadata'][key])
    integrated_fact_metadata_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, f'integrated_fact_metadata({len(fact_metadata_rows)},{metadata_total_length}).jsonl')
    print(f'Saving integrated fact metadata to {integrated_fact_metadata_filepath}')
    save_jsonl(fact_metadata_rows, integrated_fact_metadata_filepath)

    if len(facts_without_metadata) > 0:
        print_red(f'WARNING: {len(facts_without_metadata)} facts without metadata.')
        facts_without_metadata_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, f'facts_without_metadata({len(facts_without_metadata)}).jsonl')
        print(f'Saving facts without metadata to {facts_without_metadata_filepath}')
        save_jsonl(facts_without_metadata, facts_without_metadata_filepath)
    
    report_total_length = 0
    for row in report_facts_metadata_rows:
        report_total_length += len(row['fact_based_report'])
    integrated_report_facts_metadata_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, f'integrated_report_facts_metadata({len(report_facts_metadata_rows)},{report_total_length}).jsonl')
    print(f'Saving integrated report facts metadata to {integrated_report_facts_metadata_filepath}')
    save_jsonl(report_facts_metadata_rows, integrated_report_facts_metadata_filepath)

    print('Done!')