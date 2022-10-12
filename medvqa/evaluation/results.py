import os
import re
from medvqa.models.checkpoint import get_checkpoint_filepath
from medvqa.utils.common import RESULTS_DIR, WORKSPACE_DIR
from medvqa.utils.files import get_cached_json_file
from medvqa.evaluation.report_generation import get_chexpert_based_outputs_dataframe, get_report_level_metrics_dataframe
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
    mm_dirs = os.listdir(os.path.join(RESULTS_DIR,'multimodal'))
    results = []
    for dirs, kind in zip([vqa_dirs, vm_dirs, mm_dirs], ['vqa', 'visual_module', 'multimodal']):
        for exp_name in dirs:
            try:
                exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, kind, exp_name))\
                                        if ('visual_module' in x or 'multimodal_metrics' in x) and dataset_name in x]
                for filename in exp_result_filenames:
                    results.append((kind, exp_name, filename))
            except NotADirectoryError:
                pass
    return results

def collect_chexpert_based_output_results():
    vqa_dirs = os.listdir(os.path.join(RESULTS_DIR,'vqa'))
    results = []
    for dirs, kind in zip([vqa_dirs], ['vqa']):
        for exp_name in dirs:
            exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, kind, exp_name))\
                                    if 'mimiccxr_chexpert_based_output.pkl' in x]
            for filename in exp_result_filenames:
                results.append((kind, exp_name, filename))
    return results

def collect_multimodal_question_probs(dataset_name):
    dirs = os.listdir(os.path.join(RESULTS_DIR,'multimodal'))
    results = []
    for exp_name in dirs:
        try:
            exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, 'multimodal', exp_name))\
                                    if 'question_probs' in x and dataset_name in x]
            for filename in exp_result_filenames:
                results.append(os.path.join(RESULTS_DIR, 'multimodal', exp_name, filename))
        except NotADirectoryError:
            continue
    return results

_REPLACEMENT_PAIRS = [
    ('question-classification', 'qclass'),
    ('n_questions_per_report', 'nqpr'),
    ('n_q_per_report', 'nqpr'),
    ('most-popular', 'mostpop'),
    ('qclass_threshold', 'qclassthr'),
]

def _get_metadata_generator(results):
    for r in results:
        yield get_cached_json_file(os.path.join(WORKSPACE_DIR, 'models', r[0], r[1], 'metadata.json'))

def _append_frozen_image_encoder_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        frozen = metadata['model_kwargs'].get('freeze_cnn', False) or\
                 metadata['model_kwargs'].get('freeze_image_encoder', False)
        column.append(frozen)
    df['vm-frozen'] = column

def _append_amp_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            column.append(metadata['training_kwargs'].get('use_amp', False))
        except KeyError:
            column.append(False)
    df['amp'] = column

def _append_model_column(df, results):
    models = []
    for x in results:
        s = x[1].index('_',16)+1
        e = x[1].index('_',s)
        models.append(x[1][s:e])
    df['model'] = models

def _append_datasets_column(df, results):
    column = []
    for x in results:
        d = x[1][16:x[1].index('_',16)]
        d = f'{len(d.split("+"))}:{d}'
        column.append(d)
    df['datasets'] = column

def _append_eval_mode_column(df, results):
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

def _append_batch_size_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            batch_size = metadata['dataloading_kwargs'].get('batch_size', None)
        except KeyError:
            batch_size = None
        column.append(batch_size)
    df['bs'] = column

def _append_medical_tokenization_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            medtok = metadata['tokenizer_kwargs'].get('medical_tokenization', False)
        except KeyError:
            medtok = False
        column.append(medtok)
    df['medtok'] = column

def _append_vinbig_mode(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            x = metadata['vinbig_dataset_kwargs']['training_data']
        except KeyError:
            x = None
        column.append(x)
    df['vinbig_mode'] = column

def _append_merge_findings_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        mf = metadata['model_kwargs'].get('merge_findings', False)
        column.append(mf)
    df['mergef'] = column

def _append_checkpoint_epoch_column(df, results):
    column = []
    for r in results:
        checkpoint_folder_path = os.path.join(WORKSPACE_DIR, 'models', r[0], r[1])
        checkpoint_filepath = get_checkpoint_filepath(checkpoint_folder_path, verbose=False)
        epoch = re.findall(r'checkpoint_(\d+)_', checkpoint_filepath)[0]
        column.append(epoch)
    df['epoch'] = column

def _append_pretrained_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        p = metadata['model_kwargs'].get('pretrained_checkpoint_folder_path', None) is not None
        column.append(p)
    df['pretrained'] = column

def _append_data_augmentation_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            aug1 = metadata['image_transform_kwargs']['augmentation_mode']
        except KeyError:
            aug1 = None
        try:
            aug2 = metadata['dataloading_kwargs']['img_aug_mode']
        except KeyError:
            aug2 = None
        aug = aug1 or aug2
        column.append(aug)
    df['aug'] = column

def _append_gradient_accumulation_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            iters = metadata['trainer_engine_kwargs']['iters_to_accumulate']
        except KeyError:
            iters = 1
        column.append(iters)
    df['gradacc_iters'] = column

def _append_method_columns__report_level(df, results):
    df['folder'] = ['vm' if x[0] == 'visual_module' else x[0] for x in results]
    df['timestamp'] = [x[1][:15] for x in results]
    _append_datasets_column(df, results)
    _append_vinbig_mode(df, results)
    _append_model_column(df, results)
    _append_eval_mode_column(df, results)
    _append_batch_size_column(df, results)
    _append_checkpoint_epoch_column(df, results)
    _append_pretrained_column(df, results)
    _append_frozen_image_encoder_column(df, results)
    _append_merge_findings_column(df, results)
    _append_medical_tokenization_column(df, results)
    _append_amp_column(df, results)
    _append_data_augmentation_column(df, results)
    _append_gradient_accumulation_column(df, results)

def _append_method_columns__visual_module(df, results):
    df['folder'] = ['vm' if x[0] == 'visual_module' else x[0] for x in results]
    df['timestamp'] = [x[1][:15] for x in results]
    _append_datasets_column(df, results)
    _append_model_column(df, results)
    _append_frozen_image_encoder_column(df, results)
    _append_amp_column(df, results)
    _append_merge_findings_column(df, results)
    _append_medical_tokenization_column(df, results)
    _append_vinbig_mode(df, results)
    _append_pretrained_column(df, results)
    _append_batch_size_column(df, results)
    _append_gradient_accumulation_column(df, results)
    _append_checkpoint_epoch_column(df, results)

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

def plot_chexpert_based_output_metrics():
    results = collect_chexpert_based_output_results()
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]
    df = get_chexpert_based_outputs_dataframe(metrics_paths)
    _append_method_columns__report_level(df, results)
    return df