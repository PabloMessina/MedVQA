from tqdm import tqdm
import pandas as pd
from medvqa.utils.files_utils import get_cached_pickle_file

_empty_dict = {}

def get_phrase_grounding_classification_metrics_dataframe(metrics_paths, metric_prefix, class_names, class_name2short):
    assert type(metrics_paths) == list or type(metrics_paths) == str
    if type(metrics_paths) is str:
        metrics_paths  = [metrics_paths]

    metric_names = [
        f'{metric_prefix}_acc', # accuracy
        f'{metric_prefix}_prf1', # precision, recall, f1
        f'{metric_prefix}_rocauc', # receiver operating characteristic area under curve
        f'{metric_prefix}_prcauc', # precision-recall curve area under curve
    ]
    columns = ['metrics_path']

    data = [[] for _ in range(len(metrics_paths))]
    for row_i, metrics_path in tqdm(enumerate(metrics_paths)):
        data[row_i].append(metrics_path)
        metrics_dict = get_cached_pickle_file(metrics_path)
        
        for mn in metric_names:            
            met = metrics_dict.get(mn, _empty_dict)

            if mn.endswith('_acc'):
                data[row_i].append(met)
                if row_i == 0: columns.append('acc')
            
            elif mn.endswith('_prf1'):
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
                for i, label in enumerate(class_names):
                    data[row_i].append(_f1[i] if _f1 else None)
                    if row_i == 0: columns.append(f'f1({class_name2short[label]})')
                _p = met.get('p', None)
                for i, label in enumerate(class_names):
                    data[row_i].append(_p[i] if _p else None)
                    if row_i == 0: columns.append(f'p({class_name2short[label]})')
                _r = met.get('r', None)
                for i, label in enumerate(class_names):
                    data[row_i].append(_r[i] if _r else None)
                    if row_i == 0: columns.append(f'r({class_name2short[label]})')
                
            elif mn.endswith('rocauc'):
                data[row_i].append(met.get('macro_avg', None))
                if row_i == 0: columns.append('rocauc(macro)')
                data[row_i].append(met.get('micro_avg', None))
                if row_i == 0: columns.append('rocauc(micro)')
                _pc = met.get('per_class', None)
                for i, label in enumerate(class_names):
                    data[row_i].append(_pc[i] if _pc else None)
                    if row_i == 0: columns.append(f'rocauc({class_name2short[label]})')            

            elif mn.endswith('prcauc'):
                data[row_i].append(met.get('macro_avg', None))
                if row_i == 0: columns.append('prcauc(macro)')
                data[row_i].append(met.get('micro_avg', None))
                if row_i == 0: columns.append('prcauc(micro)')
                _pc = met.get('per_class', None)
                for i, label in enumerate(class_names):
                    data[row_i].append(_pc[i] if _pc else None)
                    if row_i == 0: columns.append(f'prcauc({class_name2short[label]})')
            
            elif mn.endswith('prf1'):
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
                raise ValueError(f'Unknown metric name: {mn}')
    
    return pd.DataFrame(data=data, columns=columns)