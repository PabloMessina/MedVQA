import pandas as pd
from tqdm import tqdm
from medvqa.utils.constants import MetricNames
from medvqa.utils.constants import (
    CHEXPERT_LABEL2SHORT,
    CHEXPERT_LABELS,
    METRIC2SHORT,
)

_METRIC_NAMES = [
    MetricNames.ORIENACC,
    MetricNames.CHXLABELACC,
    MetricNames.CHXLABEL_PRF1,
    MetricNames.CHXLABEL_ROCAUC,
    MetricNames.QLABELS_PRF1,
]

def get_visual_module_metrics_dataframe(metrics, method_names, metric_names=_METRIC_NAMES):
    assert type(metrics) is list or type(metrics) is dict
    assert type(method_names) is list or type(method_names) is str
    if type(metrics) is dict:
        metrics  = [metrics]
        method_names = [method_names]
    columns = ['method_name']
    data = [[] for _ in range(len(metrics))]
    for row_i, (metrics_dict, method_name) in tqdm(enumerate(zip(metrics, method_names))):
        data[row_i].append(method_name)
        for mn in metric_names:
            
            met = metrics_dict[mn]
            
            if mn == MetricNames.CHXLABEL_PRF1:                
                data[row_i].append(met['f1_macro_avg'])
                if row_i == 0: columns.append('f1(macro)')
                data[row_i].append(met['p_macro_avg'])
                if row_i == 0: columns.append('p(macro)')
                data[row_i].append(met['r_macro_avg'])
                if row_i == 0: columns.append('r(macro)')

                data[row_i].append(met['f1_micro_avg'])
                if row_i == 0: columns.append('f1(micro)')
                data[row_i].append(met['p_micro_avg'])
                if row_i == 0: columns.append('p(micro)')
                data[row_i].append(met['r_micro_avg'])
                if row_i == 0: columns.append('r(micro)')

                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(met['f1'][i])
                    if row_i == 0: columns.append(f'f1({CHEXPERT_LABEL2SHORT[label]})')
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(met['p'][i])
                    if row_i == 0: columns.append(f'p({CHEXPERT_LABEL2SHORT[label]})')
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(met['r'][i])
                    if row_i == 0: columns.append(f'r({CHEXPERT_LABEL2SHORT[label]})')
                
            elif mn == MetricNames.CHXLABEL_ROCAUC:                
                data[row_i].append(met['macro_avg'])
                if row_i == 0: columns.append('rocauc(macro)')
                data[row_i].append(met['micro_avg'])
                if row_i == 0: columns.append('rocauc(micro)')
                for i, label in enumerate(CHEXPERT_LABELS):
                    data[row_i].append(met['per_class'][i])
                    if row_i == 0: columns.append(f'rocauc({CHEXPERT_LABEL2SHORT[label]})')            
            
            elif mn == MetricNames.QLABELS_PRF1:
                data[row_i].append(met['f1_macro_avg'])
                if row_i == 0: columns.append('ql_f1(macro)')
                data[row_i].append(met['p_macro_avg'])
                if row_i == 0: columns.append('ql_p(macro)')
                data[row_i].append(met['r_macro_avg'])
                if row_i == 0: columns.append('ql_r(macro)')

                data[row_i].append(met['f1_micro_avg'])
                if row_i == 0: columns.append('ql_f1(micro)')
                data[row_i].append(met['p_micro_avg'])
                if row_i == 0: columns.append('ql_p(micro)')
                data[row_i].append(met['r_micro_avg'])
                if row_i == 0: columns.append('ql_r(micro)')
                
            else:
                data[row_i].append(met)                
                if row_i == 0: columns.append(METRIC2SHORT.get(mn, mn))
    
    return pd.DataFrame(data=data, columns=columns)