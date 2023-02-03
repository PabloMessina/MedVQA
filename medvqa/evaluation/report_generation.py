import os
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
from medvqa.models.report_generation.templates.models import SimpleTemplateRGModel
from medvqa.utils.constants import (
    CHEXPERT_LABEL2SHORT,
    CHEXPERT_LABELS,
    METRIC2SHORT,
    VINBIG_DISEASES,
    ReportEvalMode,
)
from medvqa.utils.files import get_cached_json_file, load_pickle, save_to_pickle
from medvqa.utils.metrics import chexpert_label_array_to_string
from medvqa.utils.common import CACHE_DIR, get_timestamp

_REPORT_LEVEL_METRICS_CACHE_PATH = os.path.join(CACHE_DIR, 'report_level_metrics_cache.pkl')
_REPORT_LEVEL_METRIC_NAMES = ['bleu', 'ciderD', 'rougeL', 'meteor', 'medcomp', 'wmedcomp', 'chexpert_labels']

def _concatenate_report(qa_adapted_dataset, rid):
    report = qa_adapted_dataset['reports'][rid]
    n_sentences = len(report['sentences'])
    is_valid = [True] * n_sentences
    for i in report['invalid']: is_valid[i] = False
    gt_report = '.\n '.join(report['sentences'][i].lower() for i in range(n_sentences) if is_valid[i])
    return gt_report

def recover_reports(metrics_dict, dataset, tokenizer, report_eval_mode,
                    qa_adapted_dataset=None, verbose_question=True):
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

    if verbose_question:
        def _get_q(i):
            return tokenizer.ids2string(tokenizer.clean_sentence(dataset.questions[idxs[i]]))
    elif report_eval_mode in [ReportEvalMode.GROUND_TRUTH, ReportEvalMode.MOST_POPULAR,
                            ReportEvalMode.NEAREST_NEIGHBOR, ReportEvalMode.QUESTION_CLASSIFICATION]:
        def _get_q(i):
            q_id = dataset.questions[idxs[i]]
            return qa_adapted_dataset['questions'][q_id]
    elif report_eval_mode == ReportEvalMode.CHEXPERT_LABELS:
        def _get_q(i):
            q_id = dataset.questions[idxs[i]]
            return  CHEXPERT_LABELS[q_id]
    elif report_eval_mode == ReportEvalMode.VINBIG_DISEASES:
        def _get_q(i):
            q_id = dataset.questions[idxs[i]]
            return VINBIG_DISEASES[q_id]
    elif report_eval_mode == ReportEvalMode.CHEXPERT_AND_QUESTION_CLASSIFICATION:
        def _get_q(i):
            q_id = dataset.questions[idxs[i]]
            if q_id < len(qa_adapted_dataset['questions']):
                return qa_adapted_dataset['questions'][q_id]
            q_id -= len(qa_adapted_dataset['questions'])
            return CHEXPERT_LABELS[q_id]
    else: assert False, f'Unknown report_eval_mode: {report_eval_mode}'
        
    gen_reports = []
    gt_reports = []
    for rid, indices in rid2indices.items():
        # Ground truth report
        gt_report = _concatenate_report(qa_adapted_dataset, rid)
        gt_reports.append({'rid': rid, 'text': gt_report})
        # Generated report
        gen_report = {'q':[], 'a': []}
        for i in indices:
            q = _get_q(i)
            a = tokenizer.ids2string(metrics_dict['pred_answers'][i])
            gen_report['q'].append(q)
            gen_report['a'].append(a)
        gen_reports.append(gen_report)

    return {
        'gt_reports': gt_reports,
        'gen_reports': gen_reports,
    }

class TemplateBasedModes:
    CHEXPERT_LABELS = 'chexpert_labels'
    CHEST_IMAGENOME_LABELS = 'chest_imagenome_labels'
    CHEST_IMAGENOME_LABELS__ORACLE = 'chest_imagenome_labels__oracle'

def recover_reports__template_based(
        mode, metrics_dict, qa_adapted_dataset, label_names, label_templates,
        label_thresholds, label_order=None, report_ids=None, dataset=None):
    '''
    mode: TemplateBasedModes
    label_names: list of str
    label_templates: dict of str -> dict of int -> str
    label_thresholds: list of float
    label_order: list of str
    '''
    if mode == TemplateBasedModes.CHEXPERT_LABELS:
        assert dataset is not None
        report_ids = [dataset.report_ids[i] for i in metrics_dict['idxs']]
        pred_probs = metrics_dict['pred_chexpert_probs']
    elif mode == TemplateBasedModes.CHEST_IMAGENOME_LABELS:
        assert dataset is not None
        report_ids = [dataset.report_ids[i] for i in metrics_dict['idxs']]
        pred_probs = metrics_dict['pred_chest_imagenome_probs']
    elif mode == TemplateBasedModes.CHEST_IMAGENOME_LABELS__ORACLE:
        assert report_ids is not None
        pred_probs = metrics_dict['oracle_probs']
    else: assert False, f'Unknown mode: {mode}'
    
    n = len(report_ids)
    assert len(pred_probs) == n

    if label_order is None:
        label_order = label_names

    template_rg_model = SimpleTemplateRGModel(label_names, label_templates, label_thresholds, label_order)
    template_based_reports = template_rg_model(pred_probs)
    
    gen_reports = []
    gt_reports = []
    for i in range(n):        
        # Ground truth report
        rid = report_ids[i]
        gt_report = _concatenate_report(qa_adapted_dataset, rid)
        gt_reports.append({'rid': rid, 'text': gt_report})        
        # Generated report
        gen_report = {'q':[], 'a': []}
        for j in range(len(label_order)):
            gen_report['q'].append(label_order[j])
            gen_report['a'].append(template_based_reports[i][j])
        gen_reports.append(gen_report)

    return {
        'gt_reports': gt_reports,
        'gen_reports': gen_reports,
    }

def compute_report_level_metrics(gt_reports, gen_reports, tokenizer,
                                metric_names=_REPORT_LEVEL_METRIC_NAMES,
                                max_processes=10):

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
        gen_text = ' . '.join(x for x in gen_report['a'] if len(x) > 0)
        gen_texts.append(tokenizer.clean_sentence(tokenizer.string2ids(gen_text)))

    print('Computing report-level metrics...')
    print('Example gt text: ', tokenizer.ids2string(gt_texts[0]))
    print('Example gen text: ', tokenizer.ids2string(gen_texts[0]))

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
        tmp_anticolission_code = f'_{get_timestamp()}_{random.random()}'
        metrics['chexpert_labels_gt'] = labeler.get_labels(gt_texts, tmp_suffix=tmp_anticolission_code,
                                            update_cache_on_disk=True, remove_tmp_files=True,
                                            n_chunks=max_processes, max_processes=max_processes)
        metrics['chexpert_labels_gen'] = labeler.get_labels(gen_texts, tmp_suffix=tmp_anticolission_code,
                                            update_cache_on_disk=False, remove_tmp_files=True,
                                            n_chunks=max_processes, max_processes=max_processes)
    
    print('Done computing report-level metrics.')
    return metrics

def get_report_level_metrics_dataframe(metrics_paths, metric_names=_REPORT_LEVEL_METRIC_NAMES):
    # Sanity checks
    assert type(metrics_paths) is list or type(metrics_paths) is str
    if type(metrics_paths) is str:
        metrics_paths  = [metrics_paths]

    # Load metrics cache
    metrics_cache = load_pickle(_REPORT_LEVEL_METRICS_CACHE_PATH)
    needs_update = False
    if metrics_cache is None:
        metrics_cache = {}
        needs_update = True

    # Build dataframe
    columns = ['metrics_path']
    data = [[] for _ in range(len(metrics_paths))]
    for row_i, metrics_path in enumerate(metrics_paths):
        data[row_i].append(metrics_path)

        # Retrieve cached results (if any)
        cache_key = (metrics_path, os.path.getmtime(metrics_path))
        try:
            cached_results = metrics_cache[cache_key]
        except KeyError:            
            cached_results = metrics_cache[cache_key] = {}
            print("   ** Not cached key:", cache_key)
            needs_update = True
        
        metrics_dict = None

        for mn in metric_names:
            try:
                cached_metric = cached_results[mn]
                data[row_i].extend(cached_metric['values'])
                if row_i == 0: columns.extend(cached_metric['names'])
                continue
            except KeyError:
                cached_metric = cached_results[mn] = {'values':[], 'names':[]}
                if metrics_dict is None:
                    metrics_dict = load_pickle(metrics_path)
                needs_update = True

            if mn == 'chexpert_labels':
                gt_labels = metrics_dict['chexpert_labels_gt']
                gen_labels = metrics_dict['chexpert_labels_gen']
                
                chxlabf1 = ChexpertLabelsF1score(device='cpu')
                chxlabf1.update((gen_labels, gt_labels))
                cached_metric['values'].append(chxlabf1.compute())
                cached_metric['names'].append('chxlabf1(hard)')

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

                cached_metric['values'].append(micro_avg_p)
                cached_metric['names'].append('p(micro)')                
                cached_metric['values'].append(micro_avg_r)
                cached_metric['names'].append('r(micro)')
                cached_metric['values'].append(micro_avg_f1)
                cached_metric['names'].append('f1(micro)')

                cached_metric['values'].append(macro_avg_p)
                cached_metric['names'].append('p(macro)')
                cached_metric['values'].append(macro_avg_r)
                cached_metric['names'].append('r(macro)')
                cached_metric['values'].append(macro_avg_f1)
                cached_metric['names'].append('f1(macro)')

                cached_metric['values'].append(accuracy)
                cached_metric['names'].append('acc')

                for i, label in enumerate(CHEXPERT_LABELS):
                    cached_metric['values'].append(ps[i])
                    cached_metric['names'].append(f'p({CHEXPERT_LABEL2SHORT[label]})')
                for i, label in enumerate(CHEXPERT_LABELS):
                    cached_metric['values'].append(rs[i])
                    cached_metric['names'].append(f'r({CHEXPERT_LABEL2SHORT[label]})')
                for i, label in enumerate(CHEXPERT_LABELS):
                    cached_metric['values'].append(f1s[i])
                    cached_metric['names'].append(f'f1({CHEXPERT_LABEL2SHORT[label]})')
                
            elif mn == 'bleu':            
                for k in range(4):
                    bleu_k = f'bleu-{k+1}'
                    bleu_score = metrics_dict[bleu_k]                
                    cached_metric['values'].append(bleu_score[0])
                    cached_metric['names'].append(METRIC2SHORT.get(bleu_k, bleu_k))
            elif mn == 'ciderD':
                scores = metrics_dict[mn]
                cached_metric['values'].append(scores[0])
                cached_metric['names'].append(METRIC2SHORT.get(mn, mn))
            else:
                try:
                    scores = metrics_dict[mn]
                    score = sum(scores) / len(scores)
                    cached_metric['values'].append(score)
                except KeyError:
                    cached_metric['values'].append(None)
                cached_metric['names'].append(METRIC2SHORT.get(mn, mn))

            data[row_i].extend(cached_metric['values'])
            if row_i == 0: columns.extend(cached_metric['names'])
    
    # Save updated cache to disk if required
    if needs_update:
        save_to_pickle(metrics_cache, _REPORT_LEVEL_METRICS_CACHE_PATH)
        print(f'Report level metrics updated and saved to {_REPORT_LEVEL_METRICS_CACHE_PATH}')

    return pd.DataFrame(data=data, columns=columns)

def get_chexpert_based_outputs_dataframe(metrics_paths):

    columns = ['metrics_path']
    _cols = ('f1(macro)', 'p(macro)', 'r(macro)', 'f1(micro)', 'p(micro)', 'r(micro)', 'cohenkappa')
    for key in _cols: columns.append(f'{key}-vqa')
    for key in _cols: columns.append(f'{key}-vm')    
    for key in _cols: columns.append(f'{key}-agreement')
    data = [[] for _ in range(len(metrics_paths))]

    _keys = ('f1_macro_avg', 'p_macro_avg', 'r_macro_avg',
            'f1_micro_avg', 'p_micro_avg', 'r_micro_avg',
             'cohen_kappa_score')
    
    for row_i, metrics_path in enumerate(metrics_paths):
        data[row_i].append(metrics_path)
        results = load_pickle(metrics_path)
        metrics_dict = results['metrics']
        for pair in [
            ('pred_chexpert_vqa', 'chexpert'),
            ('pred_chexpert', 'chexpert'),
            ('pred_chexpert_vqa', 'pred_chexpert'),
        ]:
            try:
                mets = metrics_dict[pair]
            except KeyError:
                print(metrics_dict.keys())
                raise
            try:
                for key in _keys:
                    data[row_i].append(mets[key])
            except KeyError:
                print(metrics_path)
                raise

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

        if idx is None: idx = 0
        if mode == 'random':
            idx = random.choice(range(self.n))                
        elif mode == 'default':
            assert idx is not None
        else:            
            if metrics_to_rank is None:
                metrics_to_rank = self.metric_names
            if mode == 'best':
                indices = sorted(list(range(self.n)), key=lambda i : sum(self._get_metric(name,i) for name in metrics_to_rank), reverse=True)
            else:
                indices = sorted(list(range(self.n)), key=lambda i : sum(self._get_metric(name,i) for name in metrics_to_rank))
            idx = indices[idx]                
        
        print('idx:', idx)        
        print('\n--')
        print('gt_report:\n')
        gt_report = self.reports['gt_reports'][idx]
        rid = gt_report['rid']
        print(gt_report['text'])
        
        print('\n--')
        print('gen_report:\n')
        gen_report = self.reports['gen_reports'][idx]        
        gen_answers = gen_report['a']
        gen_text = ' . '.join(gen_answers)
        print(gen_text)

        print('\n--')
        print('answered questions:\n')
        for q in gen_report['q']: print(q)

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