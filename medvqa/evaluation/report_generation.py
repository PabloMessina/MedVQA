import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from PIL import Image
import random
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import load_chest_imagenome_label_names
from medvqa.datasets.mimiccxr import get_mimiccxr_image_paths
from medvqa.metrics.medical.chexbert import CheXbertLabeler
from medvqa.metrics.medical.fact_embedding import FactEmbeddingScorer
from medvqa.metrics.medical.radgraph import RadGraphLabeler
from medvqa.metrics.nlp import Bleu, RougeL, CiderD, Meteor
from medvqa.metrics.medical import (
    ChexpertLabelsF1score,
    MedicalCompleteness,
    WeightedMedicalCompleteness,
)
from medvqa.metrics.medical.chexpert import ChexpertLabeler
from medvqa.models.report_generation.templates.models import SimpleTemplateRGModel
from medvqa.utils.constants import (
    CHEST_IMAGENOME_GENDERS,
    CHEXBERT_LABELS,
    CHEXBERT_LABELS_5_INDICES,
    CHEXPERT_LABEL2SHORT,
    CHEXPERT_LABELS,
    CHEXPERT_LABELS_5_INDICES,
    METRIC2SHORT,
    VINBIG_LABELS,
    ReportEvalMode,
)
from medvqa.utils.files import load_pickle, save_pickle
from medvqa.utils.logging import (
    chest_imagenome_label_array_to_string,
    chexpert_label_array_to_string,
    print_blue, print_bold, print_magenta,
)
from medvqa.utils.common import CACHE_DIR, get_timestamp
from medvqa.utils.metrics import f1_between_dicts, jaccard_between_dicts, precision_between_dicts, recall_between_dicts

_REPORT_LEVEL_METRICS_CACHE_PATH = os.path.join(CACHE_DIR, 'report_level_metrics_cache.pkl')
_REPORT_LEVEL_METRIC_NAMES = ['bleu', 'ciderD', 'rougeL', 'meteor', 'medcomp', 'wmedcomp', 'chexpert_labels',
                                'chexbert_labels', 'radgraph_labels', 'fact_embedding_score']

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
    elif report_eval_mode == ReportEvalMode.VINBIG_LABELS:
        def _get_q(i):
            q_id = dataset.questions[idxs[i]]
            return VINBIG_LABELS[q_id]
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
    CHEXPERT_LABELS__ORACLE = 'chexpert_labels__oracle'
    CHEXBERT_LABELS__ORACLE = 'chexbert_labels__oracle'
    CHEST_IMAGENOME_LABELS = 'chest_imagenome_labels'
    CHEST_IMAGENOME_LABELS__ORACLE = 'chest_imagenome_labels__oracle'
    CHEXPERT_AND_CHEST_IMAGENOME_LABELS = 'chexpert_and_chest_imagenome_labels'
    FACT_EMBEDDING_LABELS__ORACLE = 'fact_embedding_labels__oracle'
    @staticmethod
    def get_choices():
        return [
            TemplateBasedModes.CHEXPERT_LABELS,
            TemplateBasedModes.CHEXPERT_LABELS__ORACLE,
            TemplateBasedModes.CHEXBERT_LABELS__ORACLE,
            TemplateBasedModes.CHEST_IMAGENOME_LABELS,
            TemplateBasedModes.CHEST_IMAGENOME_LABELS__ORACLE,
            TemplateBasedModes.CHEXPERT_AND_CHEST_IMAGENOME_LABELS,
            TemplateBasedModes.FACT_EMBEDDING_LABELS__ORACLE,
        ]

def recover_reports__template_based(
        report_ids, pred_probs, qa_adapted_dataset, label_names, label_templates, label_thresholds,
        label_order=None, top_k_label_indices=None):
    '''
    mode: TemplateBasedModes
    label_names: list of str
    label_templates: dict of str -> dict of int -> str
    label_thresholds: list of float
    label_order: list of str
    '''
    n = len(report_ids)
    assert len(pred_probs) == n

    if label_order is None:
        label_order = label_names

    template_rg_model = SimpleTemplateRGModel(label_names, label_templates, label_thresholds,
                                              label_order, top_k_label_indices)
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

def compute_report_level_metrics(gt_reports, gen_reports, metric_names=_REPORT_LEVEL_METRIC_NAMES, max_processes=10):

    n = len(gt_reports)
    gt_texts = []
    gt_texts_tokenized = []
    gen_texts = []
    gen_texts_tokenized = []
    
    for i in range(n):
        # gt text
        if type(gt_reports[i]) == str:
            gt_text = gt_reports[i]
        else:
            gt_text = gt_reports[i]['text']
        gt_texts_tokenized.append(word_tokenize(gt_text))
        gt_texts.append(gt_text)
        # gen text
        if type(gen_reports[i]) == str:
            gen_text = gen_reports[i]
        else:            
            gen_text = '. '.join(x for x in gen_reports[i]['a'] if len(x) > 0)
        gen_texts_tokenized.append(word_tokenize(gen_text))
        gen_texts.append(gen_text)

    print_blue('Computing report-level metrics...', bold=True)
    rand_idx = random.randint(0, n-1)
    print_bold('Example gt text:')
    print_magenta(gt_texts[rand_idx], bold=True)
    print_bold('Example gen text:')
    print_magenta(gen_texts[rand_idx], bold=True)

    metrics = {}
    
    metric_name = 'bleu'
    if metric_name in metric_names:
        metric = Bleu(device='cpu', record_scores=True, using_ids=False)
        metric.update((gen_texts, gt_texts))
        scores = metric.compute()    
        for k in range(0, 4):
            blue_k = f'bleu-{k+1}'
            metrics[blue_k] = (scores[0][k], scores[1][k])
    
    metric_name = 'rougeL'
    if metric_name in metric_names:
        metric = RougeL(device='cpu', record_scores=True, using_ids=False)
        metric.update((gen_texts, gt_texts))
        metrics[metric_name] = metric.compute()

    metric_name = 'meteor'
    if metric_name in metric_names:
        metric = Meteor(device='cpu', record_scores=True)
        metric.update((gen_texts_tokenized, gt_texts_tokenized))
        metrics[metric_name] = metric.compute()
    
    metric_name = 'ciderD'
    if metric_name in metric_names:
        metric = CiderD(device='cpu', record_scores=True, using_ids=False)
        metric.update((gen_texts, gt_texts))
        metrics[metric_name] = metric.compute()

    # metric_name = 'medcomp'
    # if metric_name in metric_names:
    #     metric = MedicalCompleteness(tokenizer, device='cpu', record_scores=True)
    #     metric.update((gen_texts_tokenized, gt_texts_tokenized))
    #     metrics[metric_name] = metric.compute()
    
    # metric_name = 'wmedcomp'
    # if metric_name in metric_names:
    #     metric = WeightedMedicalCompleteness(tokenizer, device='cpu', record_scores=True)
    #     metric.update((gen_texts_tokenized, gt_texts_tokenized))
    #     metrics[metric_name] = metric.compute()

    metric_name = 'chexpert_labels'
    if metric_name in metric_names:
        labeler = ChexpertLabeler(verbose=True)
        tmp_anticolission_code = f'_{get_timestamp()}_{random.random()}'
        metrics['chexpert_labels_gt'] = labeler.get_labels(gt_texts, tmp_suffix=tmp_anticolission_code,
                                            update_cache_on_disk=True, remove_tmp_files=True,
                                            n_chunks=max_processes, max_processes=max_processes)
        metrics['chexpert_labels_gen'] = labeler.get_labels(gen_texts, tmp_suffix=tmp_anticolission_code,
                                            update_cache_on_disk=True, remove_tmp_files=True,
                                            n_chunks=max_processes, max_processes=max_processes)
        
    metric_name = 'chexbert_labels'
    if metric_name in metric_names:
        labeler = CheXbertLabeler(verbose=True)
        metrics['chexbert_labels_gt'] = labeler.get_labels(gt_texts, update_cache_on_disk=True)
        metrics['chexbert_labels_gen'] = labeler.get_labels(gen_texts, update_cache_on_disk=True)

    metric_name = 'radgraph_labels'
    if metric_name in metric_names:
        labeler = RadGraphLabeler(verbose=True)
        metrics['radgraph_labels_gt'] = labeler.get_labels(gt_texts, update_cache_on_disk=True)
        metrics['radgraph_labels_gen'] = labeler.get_labels(gen_texts, update_cache_on_disk=True)

    metric_name = 'fact_embedding_score'
    if metric_name in metric_names:
        scorer = FactEmbeddingScorer(verbose=True)
        metrics[metric_name] = scorer(gen_texts, gt_texts, update_cache_on_disk=True)
    
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
    columns_set = set()
    columns_set.add('metrics_path')
    data_dicts = [dict() for _ in range(len(metrics_paths))]
    for row_i, metrics_path in enumerate(metrics_paths):
        data_dicts[row_i]['metrics_path'] = metrics_path

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
                if mn == 'chexpert_labels':
                    prefix = 'chxp_'
                elif mn == 'chexbert_labels':
                    prefix = 'chxb_'
                else:
                    prefix = ''
                for name, value in zip(cached_metric['names'], cached_metric['values']):
                    name = f'{prefix}{name}'
                    data_dicts[row_i][name] = value
                    columns_set.add(name)
                continue
            except KeyError:
                cached_metric = cached_results[mn] = {'values':[], 'names':[]}
                if metrics_dict is None:
                    metrics_dict = load_pickle(metrics_path)
                needs_update = True

            try:
                if mn == 'chexpert_labels':
                    gt_labels = metrics_dict['chexpert_labels_gt']
                    gen_labels = metrics_dict['chexpert_labels_gen']

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
                    n = len(gt_labels)
                    sample_avg_p = sum(precision_score(gt_labels[i], gen_labels[i], zero_division=1) for i in range(n)) / n
                    sample_avg_r = sum(recall_score(gt_labels[i], gen_labels[i], zero_division=1) for i in range(n)) / n
                    sample_avg_f1 = sum(f1_score(gt_labels[i], gen_labels[i], zero_division=1) for i in range(n)) / n
                    gt_5 = gt_labels.T[CHEXPERT_LABELS_5_INDICES]
                    gen_5 = gen_labels.T[CHEXPERT_LABELS_5_INDICES]
                    gt_5_flat = gt_5.reshape(-1)
                    gen_5_flat = gen_5.reshape(-1)
                    micro_avg_p_5= precision_score(gt_5_flat, gen_5_flat)
                    micro_avg_r_5 = recall_score(gt_5_flat, gen_5_flat)
                    micro_avg_f1_5 = f1_score(gt_5_flat, gen_5_flat)
                    macro_avg_p_5 = sum(ps[i] for i in CHEXPERT_LABELS_5_INDICES) / len(CHEXPERT_LABELS_5_INDICES)
                    macro_avg_r_5 = sum(rs[i] for i in CHEXPERT_LABELS_5_INDICES) / len(CHEXPERT_LABELS_5_INDICES)
                    macro_avg_f1_5 = sum(f1s[i] for i in CHEXPERT_LABELS_5_INDICES) / len(CHEXPERT_LABELS_5_INDICES)

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

                    cached_metric['values'].append(sample_avg_p)
                    cached_metric['names'].append('p(sample)')
                    cached_metric['values'].append(sample_avg_r)
                    cached_metric['names'].append('r(sample)')
                    cached_metric['values'].append(sample_avg_f1)
                    cached_metric['names'].append('f1(sample)')

                    cached_metric['values'].append(micro_avg_p_5)
                    cached_metric['names'].append('p(micro-5)')
                    cached_metric['values'].append(micro_avg_r_5)
                    cached_metric['names'].append('r(micro-5)')
                    cached_metric['values'].append(micro_avg_f1_5)
                    cached_metric['names'].append('f1(micro-5)')

                    cached_metric['values'].append(macro_avg_p_5)
                    cached_metric['names'].append('p(macro-5)')
                    cached_metric['values'].append(macro_avg_r_5)
                    cached_metric['names'].append('r(macro-5)')
                    cached_metric['values'].append(macro_avg_f1_5)
                    cached_metric['names'].append('f1(macro-5)')

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

                elif mn == 'chexbert_labels':
                    gt_labels = metrics_dict['chexbert_labels_gt']
                    gen_labels = metrics_dict['chexbert_labels_gen']

                    ps, rs, f1s, accs = [], [], [], []
                    for i in range(len(CHEXBERT_LABELS)):
                        ps.append(precision_score(gt_labels.T[i], gen_labels.T[i]))
                        rs.append(recall_score(gt_labels.T[i], gen_labels.T[i]))
                        f1s.append(f1_score(gt_labels.T[i], gen_labels.T[i]))
                        accs.append(accuracy_score(gt_labels.T[i], gen_labels.T[i]))
                    macro_avg_p = sum(ps) / len(CHEXBERT_LABELS)
                    macro_avg_r = sum(rs) / len(CHEXBERT_LABELS)
                    macro_avg_f1 = sum(f1s) / len(CHEXBERT_LABELS)
                    gt_flat = gt_labels.reshape(-1)
                    gen_flat = gen_labels.reshape(-1)
                    micro_avg_p = precision_score(gt_flat, gen_flat)
                    micro_avg_r = recall_score(gt_flat, gen_flat)
                    micro_avg_f1 = f1_score(gt_flat, gen_flat)
                    accuracy = accuracy_score(gt_flat, gen_flat)
                    n = len(gt_labels)
                    sample_avg_p = sum(precision_score(gt_labels[i], gen_labels[i], zero_division=1) for i in range(n)) / n
                    sample_avg_r = sum(recall_score(gt_labels[i], gen_labels[i], zero_division=1) for i in range(n)) / n
                    sample_avg_f1 = sum(f1_score(gt_labels[i], gen_labels[i], zero_division=1) for i in range(n)) / n
                    gt_5 = gt_labels.T[CHEXBERT_LABELS_5_INDICES]
                    gen_5 = gen_labels.T[CHEXBERT_LABELS_5_INDICES]
                    gt_5_flat = gt_5.reshape(-1)
                    gen_5_flat = gen_5.reshape(-1)
                    micro_avg_p_5= precision_score(gt_5_flat, gen_5_flat)
                    micro_avg_r_5 = recall_score(gt_5_flat, gen_5_flat)
                    micro_avg_f1_5 = f1_score(gt_5_flat, gen_5_flat)
                    macro_avg_p_5 = sum(ps[i] for i in CHEXBERT_LABELS_5_INDICES) / len(CHEXBERT_LABELS_5_INDICES)
                    macro_avg_r_5 = sum(rs[i] for i in CHEXBERT_LABELS_5_INDICES) / len(CHEXBERT_LABELS_5_INDICES)
                    macro_avg_f1_5 = sum(f1s[i] for i in CHEXBERT_LABELS_5_INDICES) / len(CHEXBERT_LABELS_5_INDICES)

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

                    cached_metric['values'].append(sample_avg_p)
                    cached_metric['names'].append('p(sample)')
                    cached_metric['values'].append(sample_avg_r)
                    cached_metric['names'].append('r(sample)')
                    cached_metric['values'].append(sample_avg_f1)
                    cached_metric['names'].append('f1(sample)')

                    cached_metric['values'].append(micro_avg_p_5)
                    cached_metric['names'].append('p(micro-5)')
                    cached_metric['values'].append(micro_avg_r_5)
                    cached_metric['names'].append('r(micro-5)')
                    cached_metric['values'].append(micro_avg_f1_5)
                    cached_metric['names'].append('f1(micro-5)')

                    cached_metric['values'].append(macro_avg_p_5)
                    cached_metric['names'].append('p(macro-5)')
                    cached_metric['values'].append(macro_avg_r_5)
                    cached_metric['names'].append('r(macro-5)')
                    cached_metric['values'].append(macro_avg_f1_5)
                    cached_metric['names'].append('f1(macro-5)')

                    cached_metric['values'].append(accuracy)
                    cached_metric['names'].append('acc')

                    for i, label in enumerate(CHEXBERT_LABELS):
                        cached_metric['values'].append(ps[i])
                        cached_metric['names'].append(f'p({CHEXPERT_LABEL2SHORT[label]})')
                    for i, label in enumerate(CHEXBERT_LABELS):
                        cached_metric['values'].append(rs[i])
                        cached_metric['names'].append(f'r({CHEXPERT_LABEL2SHORT[label]})')
                    for i, label in enumerate(CHEXBERT_LABELS):
                        cached_metric['values'].append(f1s[i])
                        cached_metric['names'].append(f'f1({CHEXPERT_LABEL2SHORT[label]})')

                elif mn == 'radgraph_labels':
                    gt_labels = metrics_dict['radgraph_labels_gt']
                    gen_labels = metrics_dict['radgraph_labels_gen']
                    n = len(gt_labels)
                    sample_avg_p = sum(precision_between_dicts(gt_labels[i], gen_labels[i]) for i in range(n)) / n
                    sample_avg_r = sum(recall_between_dicts(gt_labels[i], gen_labels[i]) for i in range(n)) / n
                    sample_avg_f1 = sum(f1_between_dicts(gt_labels[i], gen_labels[i]) for i in range(n)) / n
                    sample_avg_jaccard = sum(jaccard_between_dicts(gt_labels[i], gen_labels[i]) for i in range(n)) / n
                    cached_metric['values'].append(sample_avg_p)
                    cached_metric['names'].append('radgraph_p(sample)')
                    cached_metric['values'].append(sample_avg_r)
                    cached_metric['names'].append('radgraph_r(sample)')
                    cached_metric['values'].append(sample_avg_f1)
                    cached_metric['names'].append('radgraph_f1(sample)')
                    cached_metric['values'].append(sample_avg_jaccard)
                    cached_metric['names'].append('radgraph_jaccard(sample)')

                elif mn == 'fact_embedding_score':
                    score = metrics_dict['fact_embedding_score']
                    cached_metric['values'].append(score)
                    cached_metric['names'].append('fact_embedding_score')
                    
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

                if mn == 'chexpert_labels':
                    prefix = 'chxp_'
                elif mn == 'chexbert_labels':
                    prefix = 'chxb_'
                else:
                    prefix = ''
                for name, value in zip(cached_metric['names'], cached_metric['values']):
                    name = f'{prefix}{name}'
                    data_dicts[row_i][name] = value
                    columns_set.add(name)
            except KeyError:
                pass
    
    # Save updated cache to disk if required
    if needs_update:
        save_pickle(metrics_cache, _REPORT_LEVEL_METRICS_CACHE_PATH)
        print(f'Report level metrics updated and saved to {_REPORT_LEVEL_METRICS_CACHE_PATH}')

    # Build dataframe
    columns = sorted(list(columns_set))
    data = [[] for _ in range(len(metrics_paths))]
    for row_i in range(len(metrics_paths)):
        for col in columns:
            data[row_i].append(data_dicts[row_i].get(col, None))
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

# class ReportGenExamplePlotter:

#     def __init__(self, reports, report_metrics, tokenizer, qa_adapted_dataset_path, images_getter):
#         self.reports = reports
#         self.report_metrics = report_metrics
#         self.tokenizer = tokenizer
#         self.qa_adapted_dataset = get_cached_json_file(qa_adapted_dataset_path)
#         self.n = len(reports['gt_reports'])
#         self.metric_names = set(report_metrics.keys())
#         self.images_getter = images_getter
        
#         if 'chexpert_labels_gt' in report_metrics:
#             if 'chxlabf1' not in report_metrics:
#                 gen_labels = report_metrics['chexpert_labels_gen']
#                 gt_labels = report_metrics['chexpert_labels_gt']
#                 chxlabf1 = ChexpertLabelsF1score(device='cpu', record_scores=True)
#                 chxlabf1.update((gen_labels, gt_labels))
#                 report_metrics['chxlabf1'] = chxlabf1.compute()
#             self.metric_names.remove('chexpert_labels_gt')
#             self.metric_names.remove('chexpert_labels_gen')

#         self.metric_names = sorted(list(self.metric_names))

#     def _get_metric(self, name, idx):
#         m = self.report_metrics[name]
#         if type(m) is tuple: m = m[1]
#         return m[idx]        

#     def inspect_example(self, metrics_to_rank=None, idx=None, mode='random'):

#         if idx is None: idx = 0
#         if mode == 'random':
#             idx = random.choice(range(self.n))                
#         elif mode == 'default':
#             assert idx is not None
#         else:            
#             if metrics_to_rank is None:
#                 metrics_to_rank = self.metric_names
#             if mode == 'best':
#                 indices = sorted(list(range(self.n)), key=lambda i : sum(self._get_metric(name,i) for name in metrics_to_rank), reverse=True)
#             else:
#                 indices = sorted(list(range(self.n)), key=lambda i : sum(self._get_metric(name,i) for name in metrics_to_rank))
#             idx = indices[idx]                
        
#         print('idx:', idx)        
#         print('\n--')
#         print('gt_report:\n')
#         gt_report = self.reports['gt_reports'][idx]
#         rid = gt_report['rid']
#         print(gt_report['text'])
        
#         print('\n--')
#         print('gen_report:\n')
#         gen_report = self.reports['gen_reports'][idx]        
#         gen_answers = gen_report['a']
#         gen_text = ' . '.join(gen_answers)
#         print(gen_text)

#         print('\n--')
#         print('answered questions:\n')
#         for q in gen_report['q']: print(q)

#         if 'chexpert_labels_gt' in self.report_metrics:
#             print('\n--')
#             print('chexpert_labels_gt:', self.report_metrics['chexpert_labels_gt'][idx])
#             print('chexpert_labels_gen:', self.report_metrics['chexpert_labels_gen'][idx])
#             print('chexpert_labels_gt (verbose):', chexpert_label_array_to_string(self.report_metrics['chexpert_labels_gt'][idx]))
#             print('chexpert_labels_gen (verbose):', chexpert_label_array_to_string(self.report_metrics['chexpert_labels_gen'][idx]))
        
#         for m in self.metric_names:
#             print(f'{m}:', self._get_metric(m, idx))

#         print('\n--')
#         print('\nimages:')
#         image_paths = self.images_getter(self.qa_adapted_dataset['reports'][rid])
#         for imgpath in image_paths:
#             print(imgpath)
#             img = Image.open(imgpath).convert('RGB')
#             plt.imshow(img)
#             plt.show()

class ReportGenExamplePlotter:

    def __init__(self, reports_path, report_metrics_path, input_labels_path=None, chest_imagenome_label_names_filename=None,
                 apply_anatomy_reordering=False):
        reports_data = load_pickle(reports_path)
        self.gen_reports = reports_data['gen_reports']
        self.gt_reports = reports_data['gt_reports']
        self.gt_report_paths = reports_data['gt_report_paths']
        self.report_metrics = load_pickle(report_metrics_path)
        self.metric_names = set(self.report_metrics.keys())
        if input_labels_path is not None:
            self.input_labels = load_pickle(input_labels_path)
            if 'chest_imagenome' in self.input_labels:
                assert chest_imagenome_label_names_filename is not None
                self.chest_imagenome_label_names = load_chest_imagenome_label_names(chest_imagenome_label_names_filename,
                                                                                    apply_anatomy_reordering)
                assert len(self.chest_imagenome_label_names) == len(self.input_labels['chest_imagenome'][0])
        else:
            self.input_labels = None
        self.n = len(self.gt_reports)
        
        if 'chexpert_labels_gt' in self.report_metrics:
            if 'chxlabf1' not in self.report_metrics:
                gen_labels = self.report_metrics['chexpert_labels_gen']
                gt_labels = self.report_metrics['chexpert_labels_gt']
                chxlabf1 = ChexpertLabelsF1score(device='cpu', record_scores=True)
                chxlabf1.update((gen_labels, gt_labels))
                self.report_metrics['chxlabf1'] = chxlabf1.compute()
                self.metric_names.add('chxlabf1')
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
        if self.input_labels is not None:
            print('Input labels:\n')
            if self.input_labels['is_ensemble']:
                n_models = len(self.input_labels['model_folder_paths'])
                print(f'Ensemble of {n_models} models')
                label_names = []
                if 'chexpert' in self.input_labels:
                    label_names.append('chexpert')
                if 'chest_imagenome' in self.input_labels:
                    label_names.append('chest_imagenome')
                assert len(label_names) > 0
                for i in range(n_models):
                    print(f'Model {i}:')
                    print('model_folder_path:', self.input_labels['model_folder_paths'][i])
                    for label_name in label_names:
                        print()
                        print_bold(label_name)
                        label_array = self.input_labels[label_name][i]
                        print('array:', label_array[idx])
                        if label_name == 'chest_imagenome':
                            print('verbose:', chest_imagenome_label_array_to_string(label_array[idx], self.chest_imagenome_label_names))
                        elif label_name == 'chexpert':
                            print('verbose:', chexpert_label_array_to_string(label_array[idx]))
                        else:
                            raise ValueError(f'Unknown label name: {label_name}')
            else:
                if 'gender' in self.input_labels:
                    print()
                    print_bold('gender')
                    label_array = self.input_labels['gender']
                    print('array:', label_array[idx])
                    print('verbose:', CHEST_IMAGENOME_GENDERS[label_array[idx]])
                if 'chexpert' in self.input_labels:
                    print()
                    print_bold('chexpert')
                    label_array = self.input_labels['chexpert']
                    print('array:', label_array[idx])
                    print('verbose:', chexpert_label_array_to_string(label_array[idx]))
                if 'chest_imagenome' in self.input_labels:
                    print()
                    print_bold('chest_imagenome')
                    label_array = self.input_labels['chest_imagenome']
                    print('array:', label_array[idx])
                    print('verbose:', chest_imagenome_label_array_to_string(label_array[idx], self.chest_imagenome_label_names))
            print('\n--')
        print('gen_report:\n')
        print(self.gen_reports[idx])
        print('\n--')
        print('gt_report:\n')
        print(self.gt_reports[idx])
        print('\n--')
        print('gt_report_path:\n')
        print(self.gt_report_paths[idx])
        with open(self.gt_report_paths[idx], 'r') as f:
            print(f.read())

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
        image_paths = get_mimiccxr_image_paths(filepath=self.gt_report_paths[idx])
        for imgpath in image_paths:
            print(imgpath)
            img = Image.open(imgpath).convert('RGB')
            plt.imshow(img)
            plt.show()