import random
import numpy as np
import pandas as pd
from tabulate import tabulate
from PIL import Image
from sklearn.metrics import (
    precision_recall_fscore_support as prf1s,
    accuracy_score,
)
from medvqa.utils.constants import (
    CHEXPERT_LABELS,
    CHEXPERT_METRICS,
    NLP_METRICS,
    METRIC2SHORT,
)

def compute_aggregated_metrics(metrics_dict, dataset, tokenizer, nlp_metric_names):
    
    idxs = metrics_dict['idxs']
    questions = [tokenizer.ids2string(tokenizer.clean_sentence(dataset.questions[i])) for i in idxs]
    unique_questions = list(set(questions))
    q2i = { q:i for i,q in enumerate(unique_questions) }
    idxs_per_q = [[] for _ in range(len(unique_questions))]
    for i, q in enumerate(questions):
        idxs_per_q[q2i[q]].append(i)

    output = dict(
        overall = dict(),
        per_question = { q:dict() for q in unique_questions },
    )
    
    # NLP metrics overall
    for name in nlp_metric_names:
        output['overall'][name] = sum(metrics_dict[name]) / len(metrics_dict[name])
    
    # NLP metrics per question
    for i, q in enumerate(unique_questions):
        tmp = output['per_question'][q]
        for name in nlp_metric_names:
            tmp[name] = sum(metrics_dict[name][j] for j in idxs_per_q[i]) / len(idxs_per_q[i])

    # Chexpert metrics overall
    gt_labels = metrics_dict['chexpert_labels_gt']
    gen_labels = metrics_dict['chexpert_labels_gen']

    output['overall']['chexpert_accuracy'] = np.array([
        accuracy_score(gt_labels[:, i], gen_labels[:, i])
        for i in range(len(CHEXPERT_LABELS))
    ])
    output['overall']['chexpert_prf1s'] = np.array([
        prf1s(gt_labels[:, i], gen_labels[:, i], zero_division=0, labels=[0, 1])
        for i in range(len(CHEXPERT_LABELS))
    ])

    # Chexpert metrics per question
    for i, q in enumerate(unique_questions):
        tmp = output['per_question'][q]
        q_gt_labels = gt_labels[idxs_per_q[i]]
        q_gen_labels = gen_labels[idxs_per_q[i]]
        tmp['chexpert_accuracy'] = np.array([
            accuracy_score(q_gt_labels[:, i], q_gen_labels[:, i])
            for i in range(len(CHEXPERT_LABELS))
        ])
        tmp['chexpert_prf1s'] = np.array([
            prf1s(q_gt_labels[:, i], q_gen_labels[:, i], zero_division=0, labels=[0, 1])
            for i in range(len(CHEXPERT_LABELS))
        ])
    
    # count per question
    for i, q in enumerate(unique_questions):
        output['per_question'][q]['count'] = len(idxs_per_q[i])

    return output

def _rank_metric_name(metric):
    if metric in NLP_METRICS:
        return 0
    if metric in CHEXPERT_METRICS:
        return 2
    return 1

class _RowRanker:
    def __init__(self):
        self.indices = []
        self.weights = []
        self.offset = 0
    def __call__(self, row):
        return sum(row[i] * self.weights[j] for j, i in enumerate(self.indices))

def extend_columns(columns, metric_name):
    if metric_name == 'chexpert_prf1s':
        columns.append('chx_p_0')
        columns.append('chx_p_1')
        columns.append('chx_r_0')
        columns.append('chx_r_1')
        columns.append('chx_f1_0')
        columns.append('chx_f1_1')
        columns.append('chx_s_0')
        columns.append('chx_s_1')
    else:
        try:
            columns.append(METRIC2SHORT[metric_name])
        except KeyError:
            columns.append(metric_name)

def update_row_ranker(row_ranker, metric_name):
    if metric_name == 'chexpert_prf1s':
        for _ in range(4):
            row_ranker.offset += 1
            row_ranker.indices.append(row_ranker.offset)
            row_ranker.weights.append(1)
            row_ranker.offset += 1
    else:
        if metric_name in NLP_METRICS or 'chexpert' in metric_name:
            row_ranker.indices.append(row_ranker.offset)
            row_ranker.weights.append(1)
        row_ranker.offset += 1

def extend_row(row, metrics_dict, metric_name):
    met = metrics_dict[metric_name]
    if metric_name == 'chexpert_prf1s':
        for i in range(3):
            row.append(sum(met[j][i][0] for j in range(14)) / 14)
            row.append(sum(met[j][i][1] for j in range(14)) / 14)
        row.append(int(sum(met[j][3][0] for j in range(14))))
        row.append(int(sum(met[j][3][1] for j in range(14))))
    elif metric_name == 'chexpert_accuracy':
        row.append(sum(met) / len(met))
    else:
        row.append(met)

def get_overall_metrics_dataframe(aggregated_metrics):
    metrics_dict = aggregated_metrics['overall']
    metric_names = list(metrics_dict.keys())
    metric_names.sort(key=_rank_metric_name)    
    columns = []
    data = [[]]
    for mn in metric_names:
        extend_columns(columns, mn)
        extend_row(data[0], metrics_dict, mn)    
    return pd.DataFrame(data=data, columns=columns)

def get_per_question_metrics_dataframe(aggregated_metrics):
    q2metrics = aggregated_metrics['per_question']
    questions = list(q2metrics.keys())
    metric_names = list(q2metrics[questions[0]].keys())
    metric_names.sort(key=_rank_metric_name)
    columns = []
    data = [[] for _ in range(len(questions))]
    row_ranker = _RowRanker()
    row_ranker.offset += 1
    columns.append('Q')
    for mn in metric_names:
        extend_columns(columns, mn)
        update_row_ranker(row_ranker, mn)
    for i, q in enumerate(questions):
        data[i].append(q)
        for mn in metric_names:
            extend_row(data[i], q2metrics[q], mn)
    data.sort(key=row_ranker, reverse=True)
    return pd.DataFrame(data=data, columns=columns)
   
class VQAExamplePlotter:
    def __init__(self, dataset_name, results):
        dataset = results[f'{dataset_name}_dataset']
        tokenizer = results['tokenizer']
        metrics_dict = results[f'{dataset_name}_metrics'] 
        idxs = metrics_dict['idxs']

        self.idxs = idxs
        self.metrics_dict = metrics_dict
        self.images = [dataset.images[i] for i in idxs]
        self.questions = [tokenizer.ids2string(tokenizer.clean_sentence(dataset.questions[i])) for i in idxs]
        self.answers = [tokenizer.ids2string(tokenizer.clean_sentence(dataset.answers[i])) for i in idxs]
        self.pred_questions = [tokenizer.ids2string(x) for x in metrics_dict['pred_questions']]
        self.pred_answers = [tokenizer.ids2string(x) for x in metrics_dict['pred_answers']]

    def inspect_example(self, metrics_to_inspect, idx=None, question=None, mode='random'):

        if idx is None:
            indices = [i for i, q in enumerate(self.questions) if q == question]
            assert len(indices) > 0, f'no match for question {question}'
            
            if mode == 'random':
                idx = random.choice(indices)
            else:
                indices.sort(key=lambda i : sum(self.metrics_dict[m][i] for m in metrics_to_inspect))
                if mode == 'best':
                    idx = indices[-1]
                else:
                    idx = indices[0]
        
        print('idx:', idx)
        print('--')
        print('question:', self.questions[idx])
        print('answer:', self.answers[idx])
        print('--')
        print('pred_question:', self.pred_questions[idx])
        print('pred_answer:', self.pred_answers[idx])
        print('--')
        print('chexpert_labels_gt:', self.metrics_dict['chexpert_labels_gt'][idx])
        print('chexpert_labels_gen:', self.metrics_dict['chexpert_labels_gen'][idx])
        print('--')
        for m in metrics_to_inspect:
            print(f'{m}:', self.metrics_dict[m][idx])
        print('image:', self.images[idx])
        img = Image.open(self.images[idx])
        return img