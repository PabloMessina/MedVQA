def get_sentences(qa_adapted_datasets):
    for dataset in qa_adapted_datasets:
        for report in dataset['reports']:
            for idx in report['matched']:
                yield report['sentences'][idx].lower()
            for idx in report['unmatched']:
                yield report['sentences'][idx].lower()          
