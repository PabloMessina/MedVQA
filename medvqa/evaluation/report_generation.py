from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from medvqa.metrics.nlp import Bleu, RougeL, CiderD, Meteor
from medvqa.metrics.medical import (
    ChexpertLabelsF1score,
    MedicalCompleteness,
    WeightedMedicalCompleteness,
)
from medvqa.metrics.medical.chexpert import ChexpertLabeler
from medvqa.utils.constants import (
    CHEXPERT_LABEL2SHORT,
    CHEXPERT_LABELS,
    METRIC2SHORT,
)
from medvqa.utils.files import get_cached_json_file
from medvqa.utils.metrics import chexpert_label_array_to_string

_REPORT_LEVEL_METRIC_NAMES = ['bleu', 'ciderD', 'rougeL', 'meteor', 'medcomp', 'wmedcomp', 'chexpert_labels']

def recover_reports(metrics_dict, dataset, tokenizer, qa_adapted_dataset):
    idxs = metrics_dict['idxs']
    report_ids = [dataset.report_ids[i] for i in idxs]
    rid2indices = dict()
    for i, rid in enumerate(report_ids):
        try:
            rid2indices[rid].append(i)
        except KeyError:
            rid2indices[rid] = [i]
    for indices in rid2indices.values():
        indices.sort()

    gen_reports = []
    gt_reports = []
    for rid, indices in rid2indices.items():
        report = qa_adapted_dataset['reports'][rid]
        gt_report = '.\n '.join(report['sentences'][i].lower() for i in report['matched'])
        gt_report = tokenizer.clean_text(gt_report)
        gt_reports.append({'rid': rid, 'text': gt_report})

        gen_report = {'q':[], 'a': []}
        for i in indices:
            q = tokenizer.ids2string(tokenizer.clean_sentence(dataset.questions[idxs[i]]))
            a = tokenizer.ids2string(metrics_dict['pred_answers'][i])
            gen_report['q'].append(q)
            gen_report['a'].append(a)
        gen_reports.append(gen_report)

    return {
        'gt_reports': gt_reports,
        'gen_reports': gen_reports,
    }

def compute_report_level_metrics(gt_reports, gen_reports, tokenizer, metric_names = _REPORT_LEVEL_METRIC_NAMES):

    n = len(gt_reports)
    gt_texts = []
    gen_texts = []
    
    for i in range(n):        
        # gt text
        gt_report = gt_reports[i]
        gt_text = gt_report['text']
        gt_texts.append(tokenizer.clean_sentence(tokenizer.string2ids(gt_text)))
        # gen text
        gen_report = gen_reports[i]
        gen_text = ' . '.join(gen_report['a'])
        gen_texts.append(tokenizer.clean_sentence(tokenizer.string2ids(gen_text)))

    metrics = {}
    
    metric_name = 'bleu'
    if metric_name in metric_names:
        metric = Bleu(device='cpu', record_scores=True)
        metric.update((gen_texts, gt_texts))
        scores = metric.compute()    
        for k in range(0, 4):
            blue_k = f'bleu-{k+1}'
            metrics[blue_k] = (scores[0][k], scores[1][k])
    
    metric_name = 'rougeL'
    if metric_name in metric_names:
        metric = RougeL(device='cpu', record_scores=True)
        metric.update((gen_texts, gt_texts))
        metrics[metric_name] = metric.compute()

    metric_name = 'meteor'
    if metric_name in metric_names:
        metric = Meteor(device='cpu', record_scores=True)
        metric.update((gen_texts, gt_texts))
        metrics[metric_name] = metric.compute()
    
    metric_name = 'ciderD'
    if metric_name in metric_names:
        metric = CiderD(device='cpu', record_scores=True)
        metric.update((gen_texts, gt_texts))
        metrics[metric_name] = metric.compute()

    metric_name = 'medcomp'
    if metric_name in metric_names:
        metric = MedicalCompleteness(tokenizer, device='cpu', record_scores=True)
        metric.update((gen_texts, gt_texts))
        metrics[metric_name] = metric.compute()
    
    metric_name = 'wmedcomp'
    if metric_name in metric_names:
        metric = WeightedMedicalCompleteness(tokenizer, device='cpu', record_scores=True)
        metric.update((gen_texts, gt_texts))
        metrics[metric_name] = metric.compute()

    metric_name = 'chexpert_labels'
    if metric_name in metric_names:        
        gt_texts = [tokenizer.ids2string(x) for x in gt_texts]
        gen_texts = [tokenizer.ids2string(x) for x in gen_texts]
        labeler = ChexpertLabeler()
        metrics['chexpert_labels_gt'] = labeler.get_labels(gt_texts, tmp_suffix='_gt', update_cache_on_disk=True)
        metrics['chexpert_labels_gen'] = labeler.get_labels(gen_texts, tmp_suffix='_gen', update_cache_on_disk=True)
    
    return metrics

def get_report_level_metrics_dataframe(metrics, method_names, metric_names=_REPORT_LEVEL_METRIC_NAMES):
    assert type(metrics) is list or type(metrics) is dict
    assert type(method_names) is list or method_names is str
    if type(metrics) is dict:
        metrics  = [metrics]
        method_names = [method_names]
    columns = ['method_name']
    data = [[] for _ in range(len(metrics))]
    for row_i, (metrics_dict, method_name) in tqdm(enumerate(zip(metrics, method_names))):
        data[row_i].append(method_name)
        for mn in metric_names:
            if mn == 'chexpert_labels':
                gt_labels = metrics_dict['chexpert_labels_gt']
                gen_labels = metrics_dict['chexpert_labels_gen']
                
                chxlabf1 = ChexpertLabelsF1score(device='cpu')
                chxlabf1.update((gen_labels, gt_labels))
                data[row_i].append(chxlabf1.compute())
                if row_i == 0: columns.append('chxlabf1(hard)')

                # chxlabacc = MultiLabelAccuracy(device='cpu')
                # chxlabacc.update((gen_labels, gt_labels))
                # data[row_i].append(chxlabacc.compute())
                # if row_i == 0: columns.append(METRIC2SHORT['chxlabelacc'])

                ps, rs, f1s, accs = [], [], [], []
                for i in range(len(CHEXPERT_LABELS)):
                    ps.append(precision_score(gt_labels.T[i], gen_labels.T[i]))
                    rs.append(recall_score(gt_labels.T[i], gen_labels.T[i]))
                    f1s.append(f1_score(gt_labels.T[i], gen_labels.T[i]))
                    accs.append(accuracy_score(gt_labels.T[i], gen_labels.T[i]))
                macro_avg_p = sum(ps) / len(CHEXPERT_LABELS)
                macro_avg_r = sum(rs) / len(CHEXPERT_LABELS)
                macro_avg_f1 = sum(f1s) / len(CHEXPERT_LABELS)
                gt_flat = gt_labels.reshape(-1)
                gen_flat = gen_labels.reshape(-1)
                micro_avg_p = precision_score(gt_flat, gen_flat)
                micro_avg_r = recall_score(gt_flat, gen_flat)
                micro_avg_f1 = f1_score(gt_flat, gen_flat)
                accuracy = accuracy_score(gt_flat, gen_flat)

                data[row_i].append(micro_avg_p)
                if row_i == 0: columns.append('p(micro)')
                data[row_i].append(micro_avg_r)
                if row_i == 0: columns.append('r(micro)')
                data[row_i].append(micro_avg_f1)
                if row_i == 0: columns.append('f1(micro)')

                data[row_i].append(macro_avg_p)
                if row_i == 0: columns.append('p(macro)')
                data[row_i].append(macro_avg_r)
                if row_i == 0: columns.append('r(macro)')
                data[row_i].append(macro_avg_f1)
                if row_i == 0: columns.append('f1(macro)')

                data[row_i].append(accuracy)
                if row_i == 0: columns.append('acc')

                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(ps[i])
                    if row_i == 0: columns.append(f'p({CHEXPERT_LABEL2SHORT[label]})')
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(rs[i])
                    if row_i == 0: columns.append(f'r({CHEXPERT_LABEL2SHORT[label]})')
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(f1s[i])
                    if row_i == 0: columns.append(f'f1({CHEXPERT_LABEL2SHORT[label]})')
                
            elif mn == 'bleu':            
                for k in range(4):
                    bleu_k = f'bleu-{k+1}'
                    bleu_score = metrics_dict[bleu_k]                
                    data[row_i].append(bleu_score[0])
                    if row_i == 0: columns.append(METRIC2SHORT.get(bleu_k, bleu_k))
            elif mn == 'ciderD':
                scores = metrics_dict[mn]
                data[row_i].append(scores[0])
                if row_i == 0: columns.append(METRIC2SHORT.get(mn, mn))
            else:
                try:
                    scores = metrics_dict[mn]
                    score = sum(scores) / len(scores)
                    data[row_i].append(score)
                except KeyError:
                    data[row_i].append(None)
                if row_i == 0: columns.append(METRIC2SHORT.get(mn, mn))
    
    return pd.DataFrame(data=data, columns=columns)

class ReportGenExamplePlotter:

    def __init__(self, reports, report_metrics, tokenizer, qa_adapted_dataset_path, images_getter):
        self.reports = reports
        self.report_metrics = report_metrics
        self.tokenizer = tokenizer
        self.qa_adapted_dataset = get_cached_json_file(qa_adapted_dataset_path)
        self.n = len(reports['gt_reports'])
        self.metric_names = set(report_metrics.keys())
        self.images_getter = images_getter
        
        if 'chexpert_labels_gt' in report_metrics:
            if 'chxlabf1' not in report_metrics:
                gen_labels = report_metrics['chexpert_labels_gen']
                gt_labels = report_metrics['chexpert_labels_gt']
                chxlabf1 = ChexpertLabelsF1score(device='cpu', record_scores=True)
                chxlabf1.update((gen_labels, gt_labels))
                report_metrics['chxlabf1'] = chxlabf1.compute()
            self.metric_names.remove('chexpert_labels_gt')
            self.metric_names.remove('chexpert_labels_gen')

        self.metric_names = sorted(list(self.metric_names))

    def _get_metric(self, name, idx):
        m = self.report_metrics[name]
        if type(m) is tuple: m = m[1]
        return m[idx]        

    def inspect_example(self, metrics_to_rank=None, idx=None, mode='random'):

        if idx is None:
            
            if mode == 'random':
                idx = random.choice(range(self.n))
            else:
                if metrics_to_rank is None:
                    metrics_to_rank = self.metric_names
                if mode == 'best':
                    _, idx = max((sum(self._get_metric(name,i) for name in metrics_to_rank), i) for i in range(self.n))
                else:
                    _, idx = min((sum(self._get_metric(name,i) for name in metrics_to_rank), i) for i in range(self.n))       
        
        print('idx:', idx)        
        print('\n--')
        print('gt_report:\n')
        gt_report = self.reports['gt_reports'][idx]
        rid = gt_report['rid']
        print(gt_report['text'])
        
        print('\n--')
        print('gen_report:\n')
        gen_report = self.reports['gen_reports'][idx]
        gen_answers = []
        questions = []
        for qid in self.qa_adapted_dataset['reports'][rid]['question_ids']:
            q = self.tokenizer.clean_text(self.qa_adapted_dataset['questions'][qid])
            questions.append(q)
            gen_answers.append(gen_report[q])
        gen_text = ' . '.join(gen_answers)
        print(gen_text)

        print('\n--')
        print('answered questions:\n')
        for q in questions: print(q)

        if 'chexpert_labels_gt' in self.report_metrics:
            print('\n--')
            print('chexpert_labels_gt:', self.report_metrics['chexpert_labels_gt'][idx])
            print('chexpert_labels_gen:', self.report_metrics['chexpert_labels_gen'][idx])
            print('chexpert_labels_gt (verbose):', chexpert_label_array_to_string(self.report_metrics['chexpert_labels_gt'][idx]))
            print('chexpert_labels_gen (verbose):', chexpert_label_array_to_string(self.report_metrics['chexpert_labels_gen'][idx]))
        
        for m in self.metric_names:
            print(f'{m}:', self._get_metric(m, idx))

        print('\n--')
        print('\nimages:')
        image_paths = self.images_getter(self.qa_adapted_dataset['reports'][rid])
        for imgpath in image_paths:
            print(imgpath)
            img = Image.open(imgpath).convert('RGB')
            plt.imshow(img)
            plt.show()