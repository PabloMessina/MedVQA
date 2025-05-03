import argparse
import os

from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.datasets.mimiccxr.report_utils import integrate_reports_and_facts
from medvqa.utils.files_utils import save_jsonl

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_reports_filepath', type=str, required=True)
    parser.add_argument('--extracted_facts_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--extraction_methods', type=str, nargs='+', required=True)
    args = parser.parse_args()
    assert len(args.extracted_facts_filepaths) == len(args.extraction_methods)

    # Integrate reports and facts
    sentence_facts_rows, report_facts_rows = integrate_reports_and_facts(
        preprocessed_reports_filepath=args.preprocessed_reports_filepath,
        extracted_facts_filepaths=args.extracted_facts_filepaths,
        extraction_methods=args.extraction_methods,
        remove_consecutive_repeated_words=True)

    # Save integrated reports and facts
    sentence_total_length = 0
    facts_total_length = 0
    for row in sentence_facts_rows:
        sentence_total_length += len(row['sentence'])
        for fact in row['facts']:
            facts_total_length += len(fact)
    integrated_sentence_facts_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR,
                                                      f'integrated_sentence_facts({sentence_total_length},{facts_total_length}).jsonl')
    print(f'Saving integrated sentence facts to {integrated_sentence_facts_filepath}')
    save_jsonl(sentence_facts_rows, integrated_sentence_facts_filepath)

    report_total_length = 0
    for row in report_facts_rows:
        report_total_length += len(row['fact_based_report'])
    integrated_report_facts_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR, f'integrated_report_facts({report_total_length}).jsonl')
    print(f'Saving integrated report facts to {integrated_report_facts_filepath}')
    save_jsonl(report_facts_rows, integrated_report_facts_filepath)

    print('Done!')