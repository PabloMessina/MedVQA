import os
from medvqa.utils.common import RESULTS_DIR, WORKSPACE_DIR
from medvqa.utils.files import load_json_file
from medvqa.evaluation.report_generation import get_report_level_metrics_dataframe
from medvqa.evaluation.visual_module import get_visual_module_metrics_dataframe

def collect_report_level_results(dataset_name):
    vqa_dirs = os.listdir(os.path.join(RESULTS_DIR,'vqa'))
    qa_dirs = os.listdir(os.path.join(RESULTS_DIR,'qa'))
    vm_dirs = os.listdir(os.path.join(RESULTS_DIR,'visual_module'))
    results = []
    for dirs, kind in zip([vqa_dirs, qa_dirs, vm_dirs], ['vqa', 'qa', 'visual_module']):
        for exp_name in dirs:
            exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, kind, exp_name))\
                                    if 'report_level' in x and dataset_name in x]
            for filename in exp_result_filenames:
                results.append((kind, exp_name, filename))
    return results

def collect_visual_module_results(dataset_name):
    vqa_dirs = os.listdir(os.path.join(RESULTS_DIR,'vqa'))
    vm_dirs = os.listdir(os.path.join(RESULTS_DIR,'visual_module'))
    results = []
    for dirs, kind in zip([vqa_dirs, vm_dirs], ['vqa', 'visual_module']):
        for exp_name in dirs:
            exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, kind, exp_name))\
                                    if 'visual_module' in x and dataset_name in x]
            for filename in exp_result_filenames:
                results.append((kind, exp_name, filename))
    return results


_REPLACEMENT_PAIRS = [
    ('question-classification', 'qclass'),
    ('n_questions_per_report', 'nqpr'),
    ('n_q_per_report', 'nqpr'),
    ('most-popular', 'mostpop'),
    ('qclass_threshold', 'qclassthr'),
]

def _append_cnn_frozen_column(df, results):
    column = []
    for r in results:
        metadata = load_json_file(os.path.join(WORKSPACE_DIR, 'models', r[0], r[1], 'metadata.json'))
        column.append(metadata['model_kwargs'].get('freeze_cnn', False))
    df['cnn-frozen'] = column    

def _append_method_columns__report_level(df, results):
    # General approach
    df['folder'] = ['vm' if x[0] == 'visual_module' else x[0] for x in results]
    # Timestamp
    df['timestamp'] = [x[1][:15] for x in results]
    # Datasets used
    df['datasets'] = [x[1][16:x[1].index('_',16)] for x in results]
    # Eval mode
    eval_modes = []
    for x in results:
        try:
            filename = x[2]
            em = filename[filename.index('eval_mode=')+10:-5]
            for x in _REPLACEMENT_PAIRS:
                em = em.replace(*x)
        except ValueError:
            em = '****************'
        eval_modes.append(em)
    df['eval_mode'] = eval_modes
    # CNN frozen
    _append_cnn_frozen_column(df, results)

def _append_method_columns__visual_module(df, results):
    # General approach
    df['folder'] = ['vm' if x[0] == 'visual_module' else x[0] for x in results]
    # Timestamp
    df['timestamp'] = [x[1][:15] for x in results]
    # Datasets used
    df['datasets'] = [x[1][16:x[1].index('_',16)] for x in results]
    # CNN frozen
    _append_cnn_frozen_column(df, results)

def plot_report_level_metrics(dataset_name):
    results = collect_report_level_results(dataset_name)
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]    
    df = get_report_level_metrics_dataframe(metrics_paths)
    _append_method_columns__report_level(df, results)
    return df

def plot_visual_module_metrics(dataset_name):
    results = collect_visual_module_results(dataset_name)
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]    
    df = get_visual_module_metrics_dataframe(metrics_paths)
    _append_method_columns__visual_module(df, results)
    return df