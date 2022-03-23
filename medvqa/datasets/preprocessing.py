def get_sentences(qa_adapted_datasets, include_unmatched = True):
    for dataset in qa_adapted_datasets:
        for report in dataset['reports']:
            for idx in report['matched']:
                yield report['sentences'][idx].lower()
            if include_unmatched:
                for idx in report['unmatched']:
                    yield report['sentences'][idx].lower()          
