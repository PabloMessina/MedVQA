import os
import re
import pandas as pd
from medvqa.models.checkpoint import get_checkpoint_filepath
from medvqa.utils.common import RESULTS_DIR, WORKSPACE_DIR
from medvqa.utils.constants import DATASET_NAME_TO_SHORT
from medvqa.utils.files import get_cached_json_file
from medvqa.evaluation import report_generation, visual_module

def collect_report_level_results(dataset_name):
    vqa_dirs = os.listdir(os.path.join(RESULTS_DIR,'vqa'))
    qa_dirs = os.listdir(os.path.join(RESULTS_DIR,'qa'))
    vm_dirs = os.listdir(os.path.join(RESULTS_DIR,'visual_module'))
    rg_dirs = os.listdir(os.path.join(RESULTS_DIR,'report_gen'))
    results = []
    for dirs, kind in zip([vqa_dirs, qa_dirs, vm_dirs, rg_dirs], ['vqa', 'qa', 'visual_module', 'report_gen']):
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

def collect_chest_imagenome_bbox_results(dataset_name):
    vqa_dirs = os.listdir(os.path.join(RESULTS_DIR,'vqa'))
    bbox_dirs = os.listdir(os.path.join(RESULTS_DIR,'bbox'))
    vm_dirs = os.listdir(os.path.join(RESULTS_DIR,'visual_module'))
    results = []
    for dirs, kind in zip([vqa_dirs, bbox_dirs, vm_dirs], ['vqa', 'bbox', 'visual_module']):
        for exp_name in dirs:
            exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, kind, exp_name))\
                                    if dataset_name in x and 'bbox_metrics' in x and 'chest_imagenome' in x]
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

def _get_metadata_generator(results):
    for r in results:
        try:
            yield get_cached_json_file(os.path.join(WORKSPACE_DIR, 'models', r[0], r[1], 'metadata.json'))
        except FileNotFoundError:
            yield {} # TODO: this is a hack

def _extract_image_transform_argument(metadata, arg_name):
    try:
        return metadata['image_transform_kwargs'][arg_name]
    except KeyError:
        pass
    try:
        return metadata['train_image_transform_kwargs'][arg_name]
    except KeyError:
        pass
    try:
        aug = None
        transform_kwargs = metadata['train_image_transform_kwargs']
        for long_name, short_name in DATASET_NAME_TO_SHORT.items():
            if long_name in transform_kwargs:
                if aug is not None:
                    aug += '\n'
                else:
                    aug = ''
                aug += f'{short_name}: {transform_kwargs[long_name][arg_name]}'
        return aug
    except KeyError:
        pass
    return None

def _append_frozen_image_encoder_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            frozen = metadata['model_kwargs'].get('freeze_cnn', False) or\
                    metadata['model_kwargs'].get('freeze_image_encoder', False)
        except KeyError:
            frozen = False
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


_MODEL_REPLACEMENT_PAIRS = [
    ('dense121', 'dn121'),
]
def _append_model_column(df, results):
    models = []
    for x in results:
        s = x[1].index('_',16)+1
        try:
            e = x[1].index('_',s)
        except ValueError:
            e = len(x[1])
        model_name = x[1][s:e]
        for x in _MODEL_REPLACEMENT_PAIRS:
            model_name = model_name.replace(*x)
        models.append(model_name)
    df['model'] = models

def _append_datasets_column(df, results):
    column = []
    for x in results:
        d = x[1][16:x[1].index('_',16)]
        d = f'{len(d.split("+"))}:{d}'
        column.append(d)
    df['datasets'] = column

_EVAL_MODE_REPLACEMENT_PAIRS = [
    ('question-classification', 'qclass'),
    ('n_questions_per_report', 'nqpr'),
    ('n_q_per_report', 'nqpr'),
    ('most-popular', 'mostpop'),
    ('qclass_threshold', 'qclassthr'),
]

def _append_eval_mode_column(df, results):
    eval_modes = []
    for x in results:
        try:
            filename = x[2]
            em = filename[filename.index('eval_mode=')+10:-5]
            for x in _EVAL_MODE_REPLACEMENT_PAIRS:
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

def _append_vinbig_mode_column(df, results):
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
        try:
            mf = metadata['model_kwargs'].get('merge_findings', False)
        except KeyError:
            mf = False
        column.append(mf)
    df['mergef'] = column

def _append_checkpoint_epoch_column(df, results):
    column = []
    for r in results:
        try:
            checkpoint_folder_path = os.path.join(WORKSPACE_DIR, 'models', r[0], r[1])
            checkpoint_filepath = get_checkpoint_filepath(checkpoint_folder_path, verbose=False)
            epoch = re.findall(r'checkpoint_(\d+)_', checkpoint_filepath)[0]
        except FileNotFoundError:
            epoch = None
        column.append(epoch)
    df['epoch'] = column

def _append_pretrained_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            p = metadata['model_kwargs'].get('pretrained_checkpoint_folder_path', None) is not None
        except KeyError:
            p = None
        column.append(p)
    df['pretrained'] = column

def _append_pretrained_image_encoder_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            p = metadata['model_kwargs'].get('image_encoder_pretrained_weights_path', None) is not None
        except KeyError:
            p = None
        column.append(p)
    df['pretr_imgenc'] = column

def _data_augmentation_extraction_method_1(metadata):
    try:
        return metadata['image_transform_kwargs']['augmentation_mode']
    except KeyError:
        return None
def _data_augmentation_extraction_method_2(metadata):
    try:
        return metadata['dataloading_kwargs']['img_aug_mode']
    except KeyError:
        return None
def _data_augmentation_extraction_method_3(metadata):
    try:
        return metadata['train_image_transform_kwargs']['augmentation_mode']
    except KeyError:
        return None
def _data_augmentation_extraction_method_4(metadata):
    try:
        aug = None
        transform_kwargs = metadata['train_image_transform_kwargs']
        for long_name, short_name in DATASET_NAME_TO_SHORT.items():
            if long_name in transform_kwargs:
                if aug is not None:
                    aug += '\n'
                else:
                    aug = ''
                aug += f'{short_name}: {transform_kwargs[long_name]["augmentation_mode"]}'
        return aug
    except KeyError:
        return None
def _append_data_augmentation_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        aug = _data_augmentation_extraction_method_1(metadata) \
            or _data_augmentation_extraction_method_2(metadata) \
            or _data_augmentation_extraction_method_3(metadata) \
            or _data_augmentation_extraction_method_4(metadata)
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

def _append_chest_imagenome_bbox_regressor_version_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            version = metadata['model_kwargs']['chest_imagenome_bbox_regressor_version']
        except KeyError:
            version = None
        column.append(version)
    df['chstimgn_bbox_model'] = column

def _append_image_size_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        size = _extract_image_transform_argument(metadata, 'image_size')
        column.append(size)
    df['img_size'] = column

def _append_decent_images_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            use_decent_images = metadata['mimiccxr_trainer_kwargs']['use_decent_images_only']
        except KeyError:
            use_decent_images = None
        column.append(use_decent_images)
    df['decent_images'] = column

def _append_method_columns__report_level(df, results):
    df['folder'] = ['vm' if x[0] == 'visual_module' else x[0] for x in results]
    df['timestamp'] = [x[1][:15] for x in results]
    _append_datasets_column(df, results)
    _append_vinbig_mode_column(df, results)
    _append_model_column(df, results)
    _append_eval_mode_column(df, results)
    _append_batch_size_column(df, results)
    _append_checkpoint_epoch_column(df, results)
    _append_pretrained_column(df, results)
    _append_pretrained_image_encoder_column(df, results)
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
    _append_vinbig_mode_column(df, results)
    _append_pretrained_column(df, results)
    _append_batch_size_column(df, results)
    _append_gradient_accumulation_column(df, results)
    _append_checkpoint_epoch_column(df, results)

def _append_method_columns__chest_imagenome_bbox(df, results):
    df['folder'] = [x[0] for x in results]
    df['timestamp'] = [x[1][:15] for x in results]
    _append_datasets_column(df, results)
    _append_model_column(df, results)
    _append_chest_imagenome_bbox_regressor_version_column(df, results)
    _append_data_augmentation_column(df, results)
    _append_eval_mode_column(df, results)
    _append_image_size_column(df, results)
    _append_decent_images_column(df, results)

def get_report_level_metrics_dataframe(dataset_name):
    results = collect_report_level_results(dataset_name)
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]    
    df = report_generation.get_report_level_metrics_dataframe(metrics_paths)
    _append_method_columns__report_level(df, results)
    return df

def get_visual_module_metrics_dataframe(dataset_name):
    results = collect_visual_module_results(dataset_name)
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]    
    df = visual_module.get_visual_module_metrics_dataframe(metrics_paths)
    _append_method_columns__visual_module(df, results)
    return df

def get_chexpert_based_output_metrics_dataframe():
    results = collect_chexpert_based_output_results()
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]
    df = report_generation.get_chexpert_based_outputs_dataframe(metrics_paths)
    _append_method_columns__report_level(df, results)
    return df

def get_chest_imagenome_bbox_metrics_dataframe(dataset_name):
    results = collect_chest_imagenome_bbox_results(dataset_name)
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]
    df = visual_module.get_chest_imagenome_bbox_metrics_dataframe(metrics_paths)
    _append_method_columns__chest_imagenome_bbox(df, results)
    return df

def get_validation_metrics_dataframe(metrics_logs_paths,
             columns2maximize=None, columns2minimize=None):
    assert type(metrics_logs_paths) == list
    assert len(metrics_logs_paths) > 0
    columns = None
    rows_list = []
    for path in metrics_logs_paths:
        df = pd.read_csv(path)
        if columns is None:
            columns = df.columns
        else:
            assert (columns == df.columns).all()
        # keep only even rows in df starting from 1
        df = df.iloc[1::2, :]
        # obtain best result for each column in df
        row = [path]
        for column in df.columns:
            if columns2minimize is not None and column in columns2minimize:
                row.append(df[column].min())
            elif columns2maximize is not None and column in columns2maximize:
                row.append(df[column].max())
            else:
                row.append(df[column].max())
        rows_list.append(row)
    columns = ['path'] + columns.tolist()
    main_df = pd.DataFrame(rows_list, columns=columns)
    return main_df