import os
from medvqa.utils.files import get_cached_json_file, load_pickle, save_to_pickle
from medvqa.utils.hashing import update_hash

def get_sentences(qa_adapted_datasets, mode='report', include_unmatched=True):
    if mode == 'report':
        for dataset in qa_adapted_datasets:
            for report in dataset['reports']:
                for idx in report['matched']:
                    yield report['sentences'][idx].lower()
                if include_unmatched:
                    for idx in report['unmatched']:
                        yield report['sentences'][idx].lower()
    elif mode == 'background':
        for dataset in qa_adapted_datasets:
            for report in dataset['reports']:
                text = report['background']
                if text: yield text
    else: assert False, f'Unknown mode {mode}'

def get_average_question_positions(cache_dir, qa_adapted_dataset_filename, report_ids):

    hash = update_hash((0,0), qa_adapted_dataset_filename)
    for rid in sorted(report_ids):
        hash = update_hash(hash, rid)

    file_path = os.path.join(cache_dir, f'average_question_positions(hash={hash[0]},{hash[1]}).pkl')
    question_scores = load_pickle(file_path)
    if question_scores is not None:
        print('question_scores loaded from', file_path)
        return question_scores
    
    qa_adapted_dataset = get_cached_json_file(os.path.join(cache_dir, qa_adapted_dataset_filename))
    reports = qa_adapted_dataset['reports']
    n_questions = len(qa_adapted_dataset['questions'])
    question_counts = [0] * n_questions
    question_scores = [0] * n_questions
    for rid in report_ids:
        report = reports[rid]
        for i, qid in enumerate(report['question_ids']):
            question_counts[qid] += 1
            question_scores[qid] += i
    for i in range(n_questions):
        question_scores[i] /= max(1, question_counts[i])

    save_to_pickle(question_scores, file_path)
    print('question_scores saved to', file_path)
    return question_scores

def get_question_frequencies(cache_dir, qa_adapted_dataset_filename, report_ids):

    hash = update_hash((0,0), qa_adapted_dataset_filename)
    for rid in sorted(report_ids):
        hash = update_hash(hash, rid)

    file_path = os.path.join(cache_dir, f'question_frequencies(hash={hash[0]},{hash[1]}).pkl')
    question_frequencies = load_pickle(file_path)
    if question_frequencies is not None:
        print('question_frequencies loaded from', file_path)
        return question_frequencies
    
    qa_adapted_dataset = get_cached_json_file(os.path.join(cache_dir, qa_adapted_dataset_filename))
    reports = qa_adapted_dataset['reports']
    n_questions = len(qa_adapted_dataset['questions'])
    question_frequencies = [0] * n_questions
    for rid in report_ids:
        report = reports[rid]
        for qid in report['question_ids']:
            question_frequencies[qid] += 1

    save_to_pickle(question_frequencies, file_path)
    print('question_frequencies saved to', file_path)
    return question_frequencies



