from tabulate import tabulate
import random
from PIL import Image

def aggregate_results_per_question(questions,
                                   metrics_list,
                                   metrics_names):

    q2mets = dict()
    for i, q in enumerate(questions):
        
        try:
            met2list = q2mets[q]
        except KeyError:
            met2list = q2mets[q] = { m:[] for m in metrics_names }
        
        for metric_name, metrics in zip(metrics_names, metrics_list):
            met2list[metric_name].append(metrics[i])

    for q, met2list in q2mets.items():
        met2list['#'] = len(met2list[metrics_names[0]])
        for m in metrics_names:
            met2list[m] = sum(met2list[m]) / len(met2list[m])
    
    unique_questions = list(q2mets.keys())
    sorted_questions = sorted(unique_questions, key=lambda q : sum(q2mets[q][m] for m in metrics_names), reverse=True)

    headers = ['Question', '#'] + metrics_names
    data = []
    for q in sorted_questions:
        row = [q, q2mets[q]['#']]
        row.extend(q2mets[q][m] for m in metrics_names)
        data.append(row)

    print(tabulate(data, headers=headers))
    
def plot_vqa_example(question,
                     images,
                     questions,
                     answers,
                     pred_questions,
                     pred_answers,
                     metrics_list,
                     metrics_names,
                     mode='random'):

    indices = [i for i, q in enumerate(questions) if q == question]
    assert len(indices) > 0, f'no match for question {question}'
    
    if mode == 'random':
        idx = random.choice(indices)
    else:
        indices.sort(key=lambda i : sum(metrics[i] for metrics in metrics_list))
        if mode == 'best':
            idx = indices[-1]
        else:
            idx = indices[0]
    
    print('idx:', idx)
    print('question:', questions[idx])
    print('answer:', answers[idx])
    print('pred_question:', pred_questions[idx])
    print('pred_answer:', pred_answers[idx])
    for metric_name, metrics in zip(metrics_names, metrics_list):
        print(f'{metric_name}:', metrics[idx])
    print('image:', images[idx])
    img = Image.open(images[idx])
    return img