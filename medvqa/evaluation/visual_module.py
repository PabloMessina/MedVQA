import pandas as pd
from tqdm import tqdm
from medvqa.utils.constants import MetricNames
from medvqa.utils.constants import (
    CHEXPERT_LABEL2SHORT,
    CHEXPERT_LABELS,
    METRIC2SHORT,
)
from medvqa.utils.files import load_pickle

_METRIC_NAMES = [
    MetricNames.ORIENACC,
    MetricNames.CHXLABELACC,
    MetricNames.CHXLABEL_PRF1,
    MetricNames.CHXLABEL_ROCAUC,
    MetricNames.QLABELS_PRF1,
]

_empty_dict = {}

def get_visual_module_metrics_dataframe(metrics_paths, metric_names=_METRIC_NAMES):
    assert type(metrics_paths) is list or type(metrics_paths) is str
    if type(metrics_paths) is str:
        metrics_paths  = [metrics_paths]
    columns = ['metrics_path']
    data = [[] for _ in range(len(metrics_paths))]
    for row_i, metrics_path in tqdm(enumerate(metrics_paths)):
        data[row_i].append(metrics_path)
        metrics_dict = load_pickle(metrics_path)
        
        for mn in metric_names:            
            met = metrics_dict.get(mn, _empty_dict)
            
            if mn == MetricNames.CHXLABEL_PRF1:                
                data[row_i].append(met.get('f1_macro_avg', None))
                if row_i == 0: columns.append('f1(macro)')
                data[row_i].append(met.get('p_macro_avg', None))
                if row_i == 0: columns.append('p(macro)')
                data[row_i].append(met.get('r_macro_avg', None))
                if row_i == 0: columns.append('r(macro)')

                data[row_i].append(met.get('f1_micro_avg', None))
                if row_i == 0: columns.append('f1(micro)')
                data[row_i].append(met.get('p_micro_avg', None))
                if row_i == 0: columns.append('p(micro)')
                data[row_i].append(met.get('r_micro_avg', None))
                if row_i == 0: columns.append('r(micro)')

                _f1 = met.get('f1', None)
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(_f1[i] if _f1 else None)
                    if row_i == 0: columns.append(f'f1({CHEXPERT_LABEL2SHORT[label]})')
                _p = met.get('p', None)
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(_p[i] if _p else None)
                    if row_i == 0: columns.append(f'p({CHEXPERT_LABEL2SHORT[label]})')
                _r = met.get('r', None)
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(_r[i] if _r else None)
                    if row_i == 0: columns.append(f'r({CHEXPERT_LABEL2SHORT[label]})')
                
            elif mn == MetricNames.CHXLABEL_ROCAUC:                
                data[row_i].append(met.get('macro_avg', None))
                if row_i == 0: columns.append('rocauc(macro)')
                data[row_i].append(met.get('micro_avg', None))
                if row_i == 0: columns.append('rocauc(micro)')
                _pc = met.get('per_class', None)
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(_pc[i] if _pc else None)
                    if row_i == 0: columns.append(f'rocauc({CHEXPERT_LABEL2SHORT[label]})')            
            
            elif mn == MetricNames.QLABELS_PRF1:
                data[row_i].append(met.get('f1_macro_avg', None))
                if row_i == 0: columns.append('ql_f1(macro)')
                data[row_i].append(met.get('p_macro_avg', None))
                if row_i == 0: columns.append('ql_p(macro)')
                data[row_i].append(met.get('r_macro_avg', None))
                if row_i == 0: columns.append('ql_r(macro)')

                data[row_i].append(met.get('f1_micro_avg', None))
                if row_i == 0: columns.append('ql_f1(micro)')
                data[row_i].append(met.get('p_micro_avg', None))
                if row_i == 0: columns.append('ql_p(micro)')
                data[row_i].append(met.get('r_micro_avg', None))
                if row_i == 0: columns.append('ql_r(micro)')
                
            else:
                data[row_i].append(met if met is not _empty_dict else None)
                if row_i == 0: columns.append(METRIC2SHORT.get(mn, mn))
    
    return pd.DataFrame(data=data, columns=columns)