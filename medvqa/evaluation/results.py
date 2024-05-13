import os
import re
import pandas as pd
from medvqa.models.checkpoint import get_checkpoint_filepath, get_model_training_history
from medvqa.utils.common import RESULTS_DIR, WORKSPACE_DIR
from medvqa.utils.constants import DATASET_NAME_TO_SHORT
from medvqa.utils.files import get_cached_json_file, get_cached_pickle_file
from medvqa.evaluation import report_generation, visual_module, phrase_grounding

def collect_report_level_results(dataset_name):
    vqa_dirs = os.listdir(os.path.join(RESULTS_DIR,'vqa'))
    qa_dirs = os.listdir(os.path.join(RESULTS_DIR,'qa'))
    vm_dirs = os.listdir(os.path.join(RESULTS_DIR,'visual_module'))
    rg_dirs = os.listdir(os.path.join(RESULTS_DIR,'report_gen'))
    pg_dirs = os.listdir(os.path.join(RESULTS_DIR,'phrase_grounding'))
    results = []
    for dirs, kind in zip([vqa_dirs, qa_dirs, vm_dirs, rg_dirs, pg_dirs],
                          ['vqa', 'qa', 'visual_module', 'report_gen', 'phrase_grounding']):
        for exp_name in dirs:
            exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, kind, exp_name))\
                                    if ('report_level' in x or 'report_gen' in x) and dataset_name in x]
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

def collect_chest_imagenome_multilabel_classification_results(dataset_name):
    vqa_dirs = os.listdir(os.path.join(RESULTS_DIR,'vqa'))
    vm_dirs = os.listdir(os.path.join(RESULTS_DIR,'visual_module'))
    results = []
    for dirs, kind in zip([vqa_dirs, vm_dirs], ['vqa', 'visual_module']):
        for exp_name in dirs:
            exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, kind, exp_name))\
                                    if dataset_name in x and 'multilabel_classification_metrics' in x and\
                                        ('chest-imagenome' in x or 'chest_imagenome' in x)]
            for filename in exp_result_filenames:
                results.append((kind, exp_name, filename))
    return results

def collect_chexpert_multilabel_classification_results(dataset_name):
    vqa_dirs = os.listdir(os.path.join(RESULTS_DIR,'vqa'))
    vm_dirs = os.listdir(os.path.join(RESULTS_DIR,'visual_module'))
    results = []
    for dirs, kind in zip([vqa_dirs, vm_dirs], ['vqa', 'visual_module']):
        for exp_name in dirs:
            exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, kind, exp_name))\
                                    if dataset_name in x and 'multilabel_classification_metrics' in x and\
                                        ('chexp' in x)]
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

def collect_phrase_grounding_classification_results(dataset_name, filename_fn=None):
    dirs = os.listdir(os.path.join(RESULTS_DIR,'phrase_grounding'))
    results = []
    for exp_name in dirs:
        try:
            exp_result_filenames = [x for x in os.listdir(os.path.join(RESULTS_DIR, 'phrase_grounding', exp_name))\
                                    if 'classification_metrics' in x and dataset_name in x and\
                                        (filename_fn is None or filename_fn(x))]
            for filename in exp_result_filenames:
                results.append(('phrase_grounding', exp_name, filename))
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

def _append_experiment_timestamp_column(df, results):
    df['exp_timestamp'] = [x[1][:15] for x in results]

def _append_filesystem_modified_time_column(df, results):
    from datetime import datetime
    df['modif_time'] = [os.path.getmtime(mp) for mp in df['metrics_path']]
    # make human readable
    df['modif_time'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in df['modif_time']]

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
            medtok = metadata['tokenizer_kwargs'].get('medical_tokenization', False) or\
                    metadata['tokenizer_kwargs'].get('use_medical_tokenization', False)
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

def _append_training_history_column(df, results):
    column = []
    for i, r in enumerate(results):
        try:
            checkpoint_folder_path = os.path.join(WORKSPACE_DIR, 'models', r[0], r[1])
            history = get_model_training_history(checkpoint_folder_path)
            assert len(history) > 0
            assert history[-1].timestamp == df['exp_timestamp'][i], f'{history[-1].timestamp} != {df["exp_timestamp"][i]}' # sanity check
            num_training_examples = 0
            for instance in history:
                num_training_examples += instance.batches_per_epoch * instance.batch_size * instance.best_epoch
            history_text = f'hlen={len(history)} nte={num_training_examples}'
            if len(history) > 1:
                history_text += f' ft_from={history[-2].timestamp}'
        except FileNotFoundError:
            history_text = None
        column.append(history_text)
    df['history'] = column

def _append_ensemble_cheating_column(df, results):
    column = ['cheat' in r[2] for r in results]
    df['escheat'] = column

def _append_string_inside_parentheses_from_filename_column(df, results):
    column = []
    for _, _, filename in results:
        try:
            s = filename.index('(')
            e = filename.index(')')
            assert s < e
            column.append(filename[s+1:e])
        except (ValueError, AssertionError):
            column.append(None)
    df['parenthesis'] = column

def _append_num_ensembled_models_column(df, metrics_paths):
    column = []
    for mp in metrics_paths:
        metrics = get_cached_pickle_file(mp)
        try:
            num_models = len(metrics['ensemble_model_names'])
        except KeyError:
            num_models = None
        column.append(num_models)
    df['num_ens'] = column

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
            if metadata['model_kwargs'].get('predict_bboxes_chest_imagenome', False):
                if metadata['model_kwargs'].get('predict_labels_and_bboxes_chest_imagenome', False):
                    version = metadata['model_kwargs'].get('chest_imagenome_bbox_regressor_version', None) or 'v4'
                    if version == 'v1': 
                        version = 'v4' # due to a bug when saving the model's metadata
                else:
                    version = metadata['model_kwargs']['chest_imagenome_bbox_regressor_version']
            else:
                version = None
        except KeyError:
            version = None
        column.append(version)
    df['chstimgn_bbox_model'] = column

def _append_chest_imagenome_mlc_version_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            if metadata['model_kwargs'].get('predict_labels_and_bboxes_chest_imagenome', False):
                version = metadata['model_kwargs'].get('chest_imagenome_bbox_regressor_version', None) or 'v4'
                if version == 'v1':
                    version = 'v4' # due to a bug when saving the model's metadata
            elif metadata['model_kwargs'].get('classify_chest_imagenome', False):
                # self.chest_imagenome_mlc_version
                mlc_version =  metadata['model_kwargs'].get('chest_imagenome_mlc_version', None)
                if mlc_version is not None:
                    version = f'mlc:{mlc_version}'
                else:
                    version = 'gf->labels'
            else:
                version = None
        except KeyError:
            version = None
        column.append(version)
    df['chstimgn_mlc_model'] = column

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

def _append_binary_loss_name_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            loss_name = metadata['trainer_engine_kwargs']['binary_loss_name']
        except KeyError:
            loss_name = None
        column.append(loss_name)
    df['binary_loss'] = column

def _append_mimiccxr_balanced_sampling_mode_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            mode = metadata['mimiccxr_trainer_kwargs']['balanced_sampling_mode']
        except KeyError:
            mode = None
        column.append(mode)
    df['mimiccxr_balanced_sampling'] = column

def _append_ensemble_column(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            num_ensembled_models = len(metadata['mimiccxr_trainer_kwargs']['precomputed_sigmoid_paths'])
        except KeyError:
            num_ensembled_models = None
        column.append(num_ensembled_models)
    df['ensemble'] = column

def _append_input_labels_colum(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            strings = []
            if metadata['mimiccxr_trainer_kwargs']['use_chexpert']:
                strings.append('chexpert')
            if metadata['mimiccxr_trainer_kwargs']['use_chest_imagenome']:
                strings.append('chest_imagenome')
            input_labels = '+'.join(strings)
        except KeyError:
            input_labels = None
        column.append(input_labels)
    df['input_labels'] = column

def _append_phrase_grounding_trainer_engine_losses(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            kwargs = metadata['trainer_engine_kwargs']
            strings = []
            aslw = kwargs.get('attention_supervision_loss_weight', 0)
            pclw = kwargs.get('phrase_classifier_loss_weight', 0)
            forlw = kwargs.get('foreground_loss_weight', 0)
            blw = kwargs.get('background_loss_weight', 0)
            bmcln = kwargs.get('binary_multilabel_classif_loss_name', None)
            foclw = kwargs.get('focal_loss_weight', 0)
            bcelw = kwargs.get('bce_loss_weight', 0)
            wbcelw = kwargs.get('wbce_loss_weight', 0)
            attregl = kwargs.get('use_attention_regularization_loss', True)
            acpgl = kwargs.get('use_contrastive_phrase_grounding_loss', True)            

            if aslw > 0: strings.append(f'aslw={aslw}')
            if pclw > 0: strings.append(f'pclw={pclw}')
            if forlw > 0: strings.append(f'forlw={forlw}')
            if blw > 0: strings.append(f'blw={blw}')
            if bmcln is not None: strings.append(f'bmcln={bmcln}')
            if foclw > 0: strings.append(f'foclw={foclw}')
            if bcelw > 0: strings.append(f'bcelw={bcelw}')
            if wbcelw > 0: strings.append(f'wbcelw={wbcelw}')
            if attregl: strings.append('attregl')
            if acpgl: strings.append('acpgl')
            losses = ','.join(strings)
            
        except KeyError:
            losses = None
        column.append(losses)
    df['losses'] = column

def _append_phrase_grounding_mimiccxr_trainer_kwargs(df, results):
    column = []
    for metadata in _get_metadata_generator(results):
        try:
            kwargs = metadata['mimiccxr_trainer_kwargs']
            strings = []
            blmst = kwargs.get('balance_long_middle_short_tail', False)
            uwpcl = kwargs.get('use_weighted_phrase_classifier_loss', False)
            ditpnffp = kwargs.get('dicom_id_to_pos_neg_facts_filepath', None)
            if blmst: strings.append('blmst')
            if uwpcl: strings.append('uwpcl')
            if ditpnffp is not None:
                filename = os.path.basename(ditpnffp)
                if '_label_based_' in filename:
                    strings.append('ditpnffp=label_based')
                elif '_fact_based_' in filename:
                    strings.append('ditpnffp=fact_based')
                elif '_all_' in filename:
                    strings.append('ditpnffp=all')
                else:
                    raise ValueError(f'Unknown ditpnffp: {filename}')
            text = ','.join(strings)
        except KeyError:
            text = None
        column.append(text)
    df['mimiccxr_trainer_kwargs'] = column

def _append_method_columns__report_level(df, results):
    df['folder'] = ['vm' if x[0] == 'visual_module' else x[0] for x in results]
    _append_experiment_timestamp_column(df, results)
    _append_filesystem_modified_time_column(df, results)
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
    _append_ensemble_column(df, results)
    _append_input_labels_colum(df, results)

def _append_method_columns__visual_module(df, results):
    df['folder'] = ['vm' if x[0] == 'visual_module' else x[0] for x in results]
    _append_experiment_timestamp_column(df, results)
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

def _append_method_columns__chest_imagenome_multilabel_classification(df, results, metrics_paths):
    df['folder'] = [x[0] for x in results]
    _append_experiment_timestamp_column(df, results)
    _append_datasets_column(df, results)
    _append_model_column(df, results)
    _append_chest_imagenome_bbox_regressor_version_column(df, results)
    _append_chest_imagenome_mlc_version_column(df, results)
    _append_data_augmentation_column(df, results)
    _append_decent_images_column(df, results)
    _append_binary_loss_name_column(df, results)
    _append_mimiccxr_balanced_sampling_mode_column(df, results)
    _append_training_history_column(df, results)
    _append_ensemble_cheating_column(df, results)
    _append_num_ensembled_models_column(df, metrics_paths)

def _append_method_columns__chexpert_multilabel_classification(df, results, metrics_paths):
    df['folder'] = [x[0] for x in results]
    _append_experiment_timestamp_column(df, results)
    _append_datasets_column(df, results)
    _append_model_column(df, results)
    _append_chest_imagenome_bbox_regressor_version_column(df, results)
    _append_chest_imagenome_mlc_version_column(df, results)
    _append_data_augmentation_column(df, results)
    _append_decent_images_column(df, results)
    _append_binary_loss_name_column(df, results)
    _append_mimiccxr_balanced_sampling_mode_column(df, results)
    _append_training_history_column(df, results)
    _append_ensemble_cheating_column(df, results)
    _append_num_ensembled_models_column(df, metrics_paths)

def _append_method_columns__chest_imagenome_bbox(df, results):
    df['folder'] = [x[0] for x in results]
    _append_experiment_timestamp_column(df, results)
    _append_datasets_column(df, results)
    _append_model_column(df, results)
    _append_chest_imagenome_bbox_regressor_version_column(df, results)
    _append_chest_imagenome_mlc_version_column(df, results)
    _append_data_augmentation_column(df, results)
    _append_eval_mode_column(df, results)
    _append_image_size_column(df, results)
    _append_decent_images_column(df, results)

def _append_method_columns__phrase_grounding_classification(df, results):
    df['folder'] = [x[0] for x in results]
    _append_experiment_timestamp_column(df, results)
    _append_filesystem_modified_time_column(df, results)
    _append_datasets_column(df, results)
    _append_model_column(df, results)
    _append_data_augmentation_column(df, results)
    _append_image_size_column(df, results)
    _append_string_inside_parentheses_from_filename_column(df, results)
    _append_phrase_grounding_trainer_engine_losses(df, results)
    _append_phrase_grounding_mimiccxr_trainer_kwargs(df, results)

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

def get_chest_imagenome_multilabel_classification_metrics_dataframe(dataset_name):
    results = collect_chest_imagenome_multilabel_classification_results(dataset_name)
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]
    df = visual_module.get_chest_imagenome_multilabel_classification_metrics_dataframe(metrics_paths)
    _append_method_columns__chest_imagenome_multilabel_classification(df, results, metrics_paths)
    return df

def get_chexpert_multilabel_classification_metrics_dataframe(dataset_name):
    results = collect_chexpert_multilabel_classification_results(dataset_name)
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]
    df = visual_module.get_chexpert_multilabel_classification_metrics_dataframe(metrics_paths)
    _append_method_columns__chexpert_multilabel_classification(df, results, metrics_paths)
    return df

def get_chest_imagenome_bbox_metrics_dataframe(dataset_name):
    results = collect_chest_imagenome_bbox_results(dataset_name)
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]
    df = visual_module.get_chest_imagenome_bbox_metrics_dataframe(metrics_paths)
    _append_method_columns__chest_imagenome_bbox(df, results)
    return df

def get_phrase_grounding_classification_metrics_dataframe(dataset_name, metric_prefix, class_names, class_name2short, verbose=False):
    filename_regex = re.compile(r'\b{}\b'.format(metric_prefix))
    filename_fn = lambda x: filename_regex.search(x) is not None
    results = collect_phrase_grounding_classification_results(dataset_name, filename_fn)
    if verbose:
        print(f'Found {len(results)} results')
        print(results)
    metrics_paths = [os.path.join(RESULTS_DIR, *result) for result in results]
    for mp in metrics_paths:
        assert os.path.exists(mp), f'{mp} does not exist'
    df = phrase_grounding.get_phrase_grounding_classification_metrics_dataframe(
        metrics_paths, metric_prefix, class_names, class_name2short)
    _append_method_columns__phrase_grounding_classification(df, results)
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