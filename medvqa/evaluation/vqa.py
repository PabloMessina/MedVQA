import random
import numpy as np
import pandas as pd
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
from medvqa.utils.files import get_cached_json_file
from medvqa.utils.metrics import (
    average_ignoring_nones,
    chexpert_label_array_to_string,
)

def compute_aggregated_metrics(metrics_dict, dataset, tokenizer, metric_names):
    
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

    for name in metric_names:

        if name == 'chexpert_accuracy':
            gt_labels = metrics_dict['chexpert_labels_gt']
            gen_labels = metrics_dict['chexpert_labels_gen']            
            # overall
            output['overall'][name] = np.array([
                accuracy_score(gt_labels[:, i], gen_labels[:, i])
                for i in range(len(CHEXPERT_LABELS))
            ])
            # per question
            for i, q in enumerate(unique_questions):
                tmp = output['per_question'][q]
                q_gt_labels = gt_labels[idxs_per_q[i]]
                q_gen_labels = gen_labels[idxs_per_q[i]]
                tmp[name] = np.array([
                    accuracy_score(q_gt_labels[:, i], q_gen_labels[:, i])
                    for i in range(len(CHEXPERT_LABELS))
                ])

        elif name == 'chexpert_prf1s':
            gt_labels = metrics_dict['chexpert_labels_gt']
            gen_labels = metrics_dict['chexpert_labels_gen']
            # overall
            output['overall']['chexpert_prf1s'] = np.array([
                prf1s(gt_labels[:, i], gen_labels[:, i], zero_division=0, labels=[0, 1])
                for i in range(len(CHEXPERT_LABELS))
            ])
            # per question
            for i, q in enumerate(unique_questions):
                tmp = output['per_question'][q]
                q_gt_labels = gt_labels[idxs_per_q[i]]
                q_gen_labels = gen_labels[idxs_per_q[i]]                
                tmp['chexpert_prf1s'] = np.array([
                    prf1s(q_gt_labels[:, i], q_gen_labels[:, i], zero_division=0, labels=[0, 1])
                    for i in range(len(CHEXPERT_LABELS))
                ])
        
        elif name == 'bleu':
            # overall
            bleus = metrics_dict[name]
            for k in range(0, 4):
                bleu_k = f'bleu-{k+1}'
                output['overall'][bleu_k] = bleus[0][k]
            # per question
            for i, q in enumerate(unique_questions):
                tmp = output['per_question'][q]
                for k in range(0, 4):
                    bleu_k = f'bleu-{k+1}'
                    tmp[bleu_k] = average_ignoring_nones(bleus[1][k][j] for j in idxs_per_q[i])
        
        elif name == 'ciderD':
            # overall
            output['overall'][name] = metrics_dict[name][0]
            # per question
            for i, q in enumerate(unique_questions):
                tmp = output['per_question'][q]
                tmp[name] = average_ignoring_nones(metrics_dict[name][1][j] for j in idxs_per_q[i])
        else:
            # overall
            output['overall'][name] =  average_ignoring_nones(metrics_dict[name])
            # per question
            for i, q in enumerate(unique_questions):
                tmp = output['per_question'][q]
                tmp[name] = average_ignoring_nones(metrics_dict[name][j] for j in idxs_per_q[i])
    
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
        columns.append(METRIC2SHORT.get(metric_name, metric_name))

def update_row_ranker(row_ranker, metric_name, metrics_to_rank = None):
    if metric_name == 'chexpert_prf1s':
        for _ in range(4):
            row_ranker.offset += 1
            if metrics_to_rank is None or metric_name in metrics_to_rank:
                row_ranker.indices.append(row_ranker.offset)
                row_ranker.weights.append(1)
            row_ranker.offset += 1
    else:
        if (metrics_to_rank is None and metric_name in METRIC2SHORT) or\
            (metrics_to_rank is not None and metric_name in metrics_to_rank):
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

def get_overall_metrics_dataframe(aggregated_metrics, metric_names = None, metrics_to_ignore=None):
    metrics_dict = aggregated_metrics['overall']
    if metric_names is None:
        metric_names = list(metrics_dict.keys())
    if metrics_to_ignore is not None:
        metric_names = [x for x  in metric_names if x not in metrics_to_ignore]
    metric_names.sort(key=_rank_metric_name)
    columns = []
    data = [[]]
    for mn in metric_names:
        extend_columns(columns, mn)
        extend_row(data[0], metrics_dict, mn)    
    return pd.DataFrame(data=data, columns=columns)

def get_per_question_metrics_dataframe(aggregated_metrics, metrics_to_ignore=None, metrics_to_rank=None):
    q2metrics = aggregated_metrics['per_question']
    questions = list(q2metrics.keys())
    metric_names = list(q2metrics[questions[0]].keys())
    if metrics_to_ignore is not None:
        metric_names = [x for x  in metric_names if x not in metrics_to_ignore]
    metric_names.sort(key=_rank_metric_name)
    columns = []
    data = [[] for _ in range(len(questions))]
    row_ranker = _RowRanker()
    row_ranker.offset += 1
    columns.append('Q')
    for mn in metric_names:
        extend_columns(columns, mn)
        update_row_ranker(row_ranker, mn, metrics_to_rank)
    for i, q in enumerate(questions):
        data[i].append(q)
        for mn in metric_names:
            extend_row(data[i], q2metrics[q], mn)
    data.sort(key=row_ranker, reverse=True)
    return pd.DataFrame(data=data, columns=columns)
   
class VQAExamplePlotter:

    def __init__(self, dataset_name, results,
                medical_tags_extractor=None,
                orientation_names=None,
                use_chexpert=False,
                qa_adapted_reports_file_path=None,
        ):
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
        self.dataset = dataset

        # optional

        if medical_tags_extractor is not None:
            assert 'pred_tags' in metrics_dict
            self.tags = medical_tags_extractor.tags
            self.pred_tags = metrics_dict['pred_tags']
        else:
            self.tags = None
        
        if orientation_names is not None:
            assert 'pred_orientation' in metrics_dict
            self.orientation_names = orientation_names
            self.orientations = [dataset.orientations[i] for i in idxs]
            self.pred_orientations = metrics_dict['pred_orientation']
        else:
            self.orientations = None

        self.use_chexpert = use_chexpert
        if use_chexpert:
            assert 'pred_chexpert' in metrics_dict
            self.pred_chexpert_labels = metrics_dict['pred_chexpert']        

        if qa_adapted_reports_file_path is not None:
            self.reports = get_cached_json_file(qa_adapted_reports_file_path)
        else:
            self.reports = None


    def inspect_example(self, metrics_to_inspect, metrics_to_rank=None, idx=None, question=None, mode='random'):

        if idx is None:
            indices = [i for i, q in enumerate(self.questions) if q == question]
            assert len(indices) > 0, f'no match for question {question}'
            
            if mode == 'random':
                idx = random.choice(indices)
            else:
                if metrics_to_rank is None:
                    metrics_to_rank = metrics_to_inspect
                indices.sort(key=lambda i : sum(self.metrics_dict[m][i] for m in metrics_to_rank))
                if mode == 'best':
                    idx = indices[-1]
                else:
                    idx = indices[0]
        
        if self.reports:
            rid = self.dataset.report_ids[self.idxs[idx]]
            report = self.reports['reports'][rid]
            report = '. '.join(report['sentences'][i] for i in report['matched'])
            print('Report:\n')
            print(report)
            print("\n===================")
        
        print('idx:', idx)
        print('--')
        print('question:', self.questions[idx])
        print('pred_question:', self.pred_questions[idx])
        print('--')
        print('answer:', self.answers[idx])
        print('pred_answer:', self.pred_answers[idx])
        print('--')
        if self.orientations:
            orien = self.orientation_names[self.orientations[idx]]
            pred_orien = self.orientation_names[self.pred_orientations[idx]]
            print('orientation:', orien)
            print('pred orientation:', pred_orien)
            print('--')
        if self.tags:
            tags = [self.tags[i] for i in self.dataset.rid2tags[self.dataset.report_ids[self.idxs[idx]]]]
            pred_tags = [self.tags[i] for i,b in enumerate(self.pred_tags[idx]) if b]
            print('tags:', tags)
            print('pred tags:', pred_tags)
            print('--')
        if self.use_chexpert:
            chexpert_labels = chexpert_label_array_to_string(self.dataset.chexpert_labels[self.dataset.report_ids[self.idxs[idx]]])
            pred_chexpert_labels = chexpert_label_array_to_string(self.pred_chexpert_labels[idx])
            print('chexpert_labels:', chexpert_labels)
            print('pred_chexpert_labels:', pred_chexpert_labels)
            print('--')
        print('chexpert_labels_gt:', self.metrics_dict['chexpert_labels_gt'][idx])
        print('chexpert_labels_gen:', self.metrics_dict['chexpert_labels_gen'][idx])
        print('chexpert_labels_gt (verbose):', chexpert_label_array_to_string(self.metrics_dict['chexpert_labels_gt'][idx]))
        print('chexpert_labels_gen (verbose):', chexpert_label_array_to_string(self.metrics_dict['chexpert_labels_gen'][idx]))
        print('--')
        for m in metrics_to_inspect:
            print(f'{m}:', self.metrics_dict[m][idx])
        print('image:', self.images[idx])
        img = Image.open(self.images[idx])
        return img