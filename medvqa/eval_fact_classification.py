import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.chexpert import CHEXPERT_V1_0_SMALL_DATASET_DIR
from medvqa.datasets.iuxray import get_iuxray_image_path
from medvqa.datasets.mimiccxr import (
    get_imageId2PartPatientStudy,
    get_imageId2reportId,
    get_mimic_cxr_lt_ridx2labels,
    get_mimiccxr_medium_image_path,
    get_split2imageIds,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.evaluation.plots import plot_metrics
from medvqa.evaluation.report_generation import compute_report_level_metrics
from medvqa.metrics.classification.multilabel_accuracy import MultiLabelAccuracy
from medvqa.metrics.classification.multilabel_prf1 import MultiLabelPRF1
from medvqa.metrics.classification.prc_auc import prc_auc_fn
from medvqa.metrics.classification.roc_auc import roc_auc_fn
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.models.checkpoint import get_checkpoint_filepath
from medvqa.models.checkpoint.model_wrapper import ModelWrapper
from medvqa.models.phrase_grounding.phrase_grounder import PhraseGrounder
from medvqa.models.report_generation.templates.chexpert import TEMPLATES_CHEXPERT_v1
from medvqa.utils.constants import (
    CHEXPERT_LABELS,
    DATASET_NAMES,
    LABEL_BASED_FACTS,
    LABEL_BASED_FACTS__CHEST_IMAGENOME,
    LABEL_BASED_FACTS__CHEXPERT,
    LABEL_BASED_FACTS__MIMIC_CXR_LT,
)
from medvqa.models.checkpoint import (
    load_metadata,
)
from medvqa.utils.common import (
    INTERPRET_CXR_TEST_HIDDEN_CSV_PATH,
    INTERPRET_CXR_TEST_HIDDEN_IMAGES_FOLDER_PATH,
    INTERPRET_CXR_TEST_PUBLIC_CSV_PATH,
    INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH,
    parsed_args_to_dict,
)
from medvqa.datasets.dataloading_utils import (
    SequentialDataLoader,
)
from medvqa.datasets.image_processing import get_image_transform
from medvqa.utils.files import (
    get_cached_json_file,
    get_cached_pickle_file,
    get_file_path_with_hashing_if_too_long,
    get_results_folder_path,
    load_jsonl, load_pickle, save_pickle,
)
from medvqa.utils.logging import CountPrinter, print_blue, print_bold, print_magenta
from medvqa.utils.metrics import best_threshold_and_f1_score

class _EvalModes:
    MIMICCXR_TEST_SET_LABEL_BASED = 'mimiccxr_test_set_label_based'
    MIMICCXR_TEST_SET_LABEL_BASED__TEMPLATE_BASED_REPORT_GEN = 'mimiccxr_test_set_label_based__template_based_report_gen'
    INTERPRET_CXR_TEST_PUBLIC_LABEL_BASED__TEMPLATE_BASED_REPORT_GEN = 'interpret_cxr_test_public_label_based__template_based_report_gen'
    INTERPRET_CXR_TEST_PUBLIC_LABEL_BASED__GENERATE_JSONS_FOR_REPORT_GEN = 'interpret_cxr_test_public_label_based__generate_jsons_for_report_gen'
    INTERPRET_CXR_TEST_PUBLIC_LABEL_BASED__JSON_TO_GPT_REPORT_GEN = 'interpret_cxr_test_public_label_based__json_to_gpt_report_gen'
    INTERPRET_CXR_COMPUTE_AND_SAVE_LABEL_BASED_PREDICTIONS = 'interpret_cxr_compute_and_save_label_based_predictions'

    @staticmethod
    def choices():
        return [
            _EvalModes.MIMICCXR_TEST_SET_LABEL_BASED,
            _EvalModes.MIMICCXR_TEST_SET_LABEL_BASED__TEMPLATE_BASED_REPORT_GEN,
            _EvalModes.INTERPRET_CXR_TEST_PUBLIC_LABEL_BASED__TEMPLATE_BASED_REPORT_GEN,
            _EvalModes.INTERPRET_CXR_TEST_PUBLIC_LABEL_BASED__GENERATE_JSONS_FOR_REPORT_GEN,
            _EvalModes.INTERPRET_CXR_TEST_PUBLIC_LABEL_BASED__JSON_TO_GPT_REPORT_GEN,
            _EvalModes.INTERPRET_CXR_COMPUTE_AND_SAVE_LABEL_BASED_PREDICTIONS,
        ]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_images_per_batch', type=int, default=30)
    parser.add_argument('--max_facts_per_image', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_mode', type=str, required=True, choices=_EvalModes.choices())
    parser.add_argument('--fact_embedding_model_name', type=str, default=None)
    parser.add_argument('--fact_embedding_model_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--fact_embedding_batch_size', type=int, default=32)
    parser.add_argument('--fact_embedding_num_workers', type=int, default=4)
    parser.add_argument('--fact_embedding_device', type=str, default='cuda')
    parser.add_argument('--chexpert_labels_filepath', type=str)
    parser.add_argument('--chest_imagenome_label_names_filepath', type=str)
    parser.add_argument('--chest_imagenome_image_id_to_labels_filepath', type=str)
    parser.add_argument('--mimiccxr_report_fact_nli_integrated_data_filepath', type=str)
    parser.add_argument('--tune_thresholds', action='store_true')
    parser.add_argument('--max_processes_for_chexpert_labeler', type=int, default=10)
    parser.add_argument('--background_findings_and_impression_per_report_filepath', type=str)
    parser.add_argument('--use_alternative_chexpert_template', action='store_true')
    parser.add_argument('--eval_chexpert_only', action='store_true')
    parser.add_argument('--section_mode', type=str, default=None, choices=['findings', 'impression', 'both'])
    parser.add_argument('--dicom_id_to_pos_neg_facts_filepath', type=str, default=None)
    parser.add_argument('--f1_threshold', type=float, default=0.2)
    parser.add_argument('--label_based_json_reports_filepath', type=str, default=None)
    parser.add_argument('--json_to_gpt_reports_jsonl_filepath', type=str, default=None)
    parser.add_argument('--mimiccxr_interpret_cxr_challenge_split_filepath', type=str, default=None)
    parser.add_argument('--iuxray_interpret_cxr_challenge_split_filepath', type=str, default=None)
    parser.add_argument('--chexpert_interpret_cxr_challenge_split_filepath', type=str, default=None)

    return parser.parse_args(args=args)

class FactClassificationDataset(Dataset):

    def __init__(self, image_paths, image_transform, fact_embeddings, fact_idxs, indices):
        self.indices = indices
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.fact_embeddings = fact_embeddings
        self.fact_idxs = fact_idxs

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        image_path = self.image_paths[idx]
        image = self.image_transform(image_path)
        fact_embeddings = self.fact_embeddings[self.fact_idxs[idx]]
        return {
            'idx': idx,
            'i': image,
            'f': fact_embeddings,
        }
    
def _fact_classification_collate_batch_fn(batch):
    batch_dict = dict()
    batch_dict['idx'] = [x['idx'] for x in batch]
    batch_dict['i'] = torch.stack([x['i'] for x in batch])
    batch_dict['f'] = torch.tensor(np.array([x['f'] for x in batch]))
    return batch_dict
    
def _create_dataloader(max_images_per_batch, max_facts_per_image, num_workers,
                   image_paths, image_transform, fact_embeddings, fact_idxs):
    n = len(image_paths)
    assert len(fact_idxs) == n
    image_paths_ = []
    fact_idxs_ = []
    for i in range(n):
        if len(fact_idxs[i]) > max_facts_per_image:
            k = len(fact_idxs[i]) // max_facts_per_image
            if len(fact_idxs[i]) % max_facts_per_image != 0:
                k += 1
            for j in range(k):
                image_paths_.append(image_paths[i])
                fact_idxs_.append(fact_idxs[i][j * max_facts_per_image:(j + 1) * max_facts_per_image])
        else:
            image_paths_.append(image_paths[i])
            fact_idxs_.append(fact_idxs[i])
    
    len2idxs = {}
    for i in range(len(image_paths_)):
        len_ = len(fact_idxs_[i])
        if len_ not in len2idxs:
            len2idxs[len_] = []
        len2idxs[len_].append(i)

    dataloaders = []
    for len_, idxs in len2idxs.items():
        dataset = FactClassificationDataset(
            image_paths=image_paths_,
            image_transform=image_transform,
            fact_embeddings=fact_embeddings,
            fact_idxs=fact_idxs_,
            indices=idxs,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=max_images_per_batch,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_fact_classification_collate_batch_fn,
        )
        dataloaders.append(dataloader)
    dataloader = SequentialDataLoader(dataloaders)
    return dataloader, fact_idxs_, image_paths_
        
def _get_mimiccxr_image_paths(splits):
    assert type(splits) == str or type(splits) == list
    if type(splits) == str:
        splits = [splits]
    for split in splits:
        assert split in ['train', 'validate', 'test']
    image_paths = []
    metadata = load_mimiccxr_reports_detailed_metadata()
    for part_id, subject_id, study_id, dicom_id_view_pos_pairs, split_ in zip(
        metadata['part_ids'], metadata['subject_ids'], metadata['study_ids'], metadata['dicom_id_view_pos_pairs'],
        metadata['splits'],
    ):
        if split_ in splits:
            for dicom_id, _ in dicom_id_view_pos_pairs:
                image_path = get_mimiccxr_medium_image_path(part_id, subject_id, study_id, dicom_id)
                assert os.path.exists(image_path)
                image_paths.append(image_path)
    assert len(image_paths) > 0
    return image_paths

def _compute_and_save_classification_metrics(pred_probs, pred_labels, gt_labels, results_folder_path, dataset_name, metric_prefix, strings):
    assert np.all(np.logical_or(gt_labels == 0, gt_labels == 1)) # only 0's and 1's in gt_labels

    metrics = {}

    # compute tp, fp, tn, fn per class
    print('Computing tp, fp, tn, fn per class ...')
    metrics[f'{metric_prefix}_tp'] = []
    metrics[f'{metric_prefix}_fp'] = []
    metrics[f'{metric_prefix}_tn'] = []
    metrics[f'{metric_prefix}_fn'] = []
    for i in range(pred_labels.shape[1]):
        tp = np.sum((pred_labels[:, i] == 1) & (gt_labels[:, i] == 1))
        fp = np.sum((pred_labels[:, i] == 1) & (gt_labels[:, i] == 0))
        tn = np.sum((pred_labels[:, i] == 0) & (gt_labels[:, i] == 0))
        fn = np.sum((pred_labels[:, i] == 0) & (gt_labels[:, i] == 1))
        metrics[f'{metric_prefix}_tp'].append(tp)
        metrics[f'{metric_prefix}_fp'].append(fp)
        metrics[f'{metric_prefix}_tn'].append(tn)
        metrics[f'{metric_prefix}_fn'].append(fn)

    # compute accuracy
    print('Computing accuracy ...')
    met = MultiLabelAccuracy(device='cpu')
    met.update((pred_labels, gt_labels))
    metrics[f'{metric_prefix}_acc'] = met.compute()
    
    # compute prf1
    print('Computing prf1 ...')
    met = MultiLabelPRF1(device='cpu')
    met.update((pred_labels, gt_labels))
    metrics[f'{metric_prefix}_prf1'] = met.compute()
    
    # compute roc auc
    print('Computing roc auc ...')
    metrics[f'{metric_prefix}_rocauc'] = roc_auc_fn(pred_probs, gt_labels)
    
    # compute prc auc
    print('Computing prc auc ...')
    metrics[f'{metric_prefix}_prcauc'] = prc_auc_fn(pred_probs, gt_labels)

    strings_ = []
    strings_.append(metric_prefix)
    strings_.extend(strings)    

    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_fact_classification_metrics{"(" + ",".join(strings_) + ")"}.pkl')
    save_pickle(metrics, save_path)
    print_bold(f'Fact classification metrics saved to {save_path}')
    return metrics

def _evaluate_mimiccxr_test_set_chexpert_fact_classification(
        probs, pred_labels, chexpert_labels_filepath, dicom_ids, results_folder_path, strings):
    label_idxs = [LABEL_BASED_FACTS.index(fact) for fact in LABEL_BASED_FACTS__CHEXPERT]
    probs_ = probs[:, label_idxs]
    pred_labels_ = pred_labels[:, label_idxs]
    dicom_id2ridx = get_imageId2reportId()
    gt_labels = load_pickle(chexpert_labels_filepath)
    gt_labels_ = np.array([gt_labels[dicom_id2ridx[dicom_id]] for dicom_id in dicom_ids])
    assert gt_labels_.shape == probs_.shape
    print('probs_.shape =', probs_.shape)
    print('gt_labels_.shape =', gt_labels_.shape)
    _compute_and_save_classification_metrics(
        pred_probs=probs_,
        pred_labels=pred_labels_,
        gt_labels=gt_labels_,
        results_folder_path=results_folder_path,
        dataset_name='mimiccxr_test_set',
        metric_prefix='chexpert',
        strings=strings,
    )

def _evaluate_mimiccxr_test_set_mimic_cxr_lt_fact_classification(
        probs, pred_labels, dicom_ids, results_folder_path, strings):
    label_idxs = [LABEL_BASED_FACTS.index(fact) for fact in LABEL_BASED_FACTS__MIMIC_CXR_LT]
    probs_ = probs[:, label_idxs]
    pred_labels_ = pred_labels[:, label_idxs]
    
    dicom_id2ridx = get_imageId2reportId()
    ridx2labels = get_mimic_cxr_lt_ridx2labels()
    gt_labels_ = np.array([ridx2labels[dicom_id2ridx[dicom_id]] for dicom_id in dicom_ids])

    assert gt_labels_.shape == probs_.shape
    print('probs_.shape =', probs_.shape)
    print('gt_labels_.shape =', gt_labels_.shape)
    _compute_and_save_classification_metrics(
        pred_probs=probs_,
        pred_labels=pred_labels_,
        gt_labels=gt_labels_,
        results_folder_path=results_folder_path,
        dataset_name='mimiccxr_test_set',
        metric_prefix='mimic_cxr_lt',
        strings=strings,
    )

def _evaluate_mimiccxr_test_set_chest_imagenome_fact_classification(
        probs, pred_labels, dicom_ids, results_folder_path,
        chest_imagenome_label_names_filepath,
        chest_imagenome_image_id_to_labels_filepath, strings):
    label_idxs = [LABEL_BASED_FACTS.index(fact) for fact in LABEL_BASED_FACTS__CHEST_IMAGENOME]
    probs_ = probs[:, label_idxs]
    pred_labels_ = pred_labels[:, label_idxs]

    label_names = load_pickle(chest_imagenome_label_names_filepath)
    image_id_to_labels = load_pickle(chest_imagenome_image_id_to_labels_filepath)
    valid_label_idxs = [i for i, label in enumerate(label_names) if len(label) == 2\
                         and label[0] not in ["laterality", "severity", "nlp", "temporal"]]
    
    valid_dicom_idxs = []
    valid_dicom_ids = []
    for i, dicom_id in enumerate(dicom_ids):
        if dicom_id in image_id_to_labels:
            valid_dicom_idxs.append(i)
            valid_dicom_ids.append(dicom_id)
    assert len(valid_dicom_idxs) > 0
    if len(valid_dicom_idxs) < len(dicom_ids):
        probs_ = probs_[valid_dicom_idxs]
        pred_labels_ = pred_labels_[valid_dicom_idxs]
        print('len(dicom_ids) =', len(dicom_ids))
        print('len(valid_dicom_idxs) =', len(valid_dicom_idxs))

    gt_labels_ = np.array([image_id_to_labels[dicom_id][valid_label_idxs] for dicom_id in valid_dicom_ids])
    gt_labels_ = gt_labels_ == 1 # convert to 0's and 1's
    gt_labels_ = gt_labels_.astype(int)

    assert gt_labels_.shape == probs_.shape
    print('probs_.shape =', probs_.shape)
    print('gt_labels_.shape =', gt_labels_.shape)

    _compute_and_save_classification_metrics(
        pred_probs=probs_,
        pred_labels=pred_labels_,
        gt_labels=gt_labels_,
        results_folder_path=results_folder_path,
        dataset_name='mimiccxr_test_set',
        metric_prefix='chest_imagenome',
        strings=strings,
    )

def _evaluate_mimiccxr_test_set_nli_based_labels_fact_classification(
        probs, pred_labels, dicom_ids, results_folder_path, label_based_facts, label_group_name,
        mimiccxr_report_fact_nli_integrated_data_filepath, strings):
    data = get_cached_pickle_file(mimiccxr_report_fact_nli_integrated_data_filepath)
    label_based_nli_predictions = data['label_based_nli_predictions']
    label_based_facts_ = data['label_based_facts']
    nli_predictions = label_based_nli_predictions['nli_predictions']
    label_idxs = [label_based_facts_.index(fact) for fact in label_based_facts]
    probs_ = probs[:, label_idxs]
    pred_labels_ = pred_labels[:, label_idxs]
    
    dicom_id2ridx = get_imageId2reportId()
    gt_labels_ = np.array([nli_predictions[dicom_id2ridx[dicom_id]] for dicom_id in dicom_ids])
    n_undecided = np.sum(gt_labels_ == 3)
    print(f'n_undecided = {n_undecided}/{len(gt_labels_.flatten())}')
    gt_labels_ = gt_labels_ == 0 # convert to 0's and 1's
    gt_labels_ = gt_labels_.astype(int)
    gt_labels_ = gt_labels_[:, label_idxs]

    assert gt_labels_.shape == probs_.shape
    print('probs_.shape =', probs_.shape)
    print('gt_labels_.shape =', gt_labels_.shape)
    _compute_and_save_classification_metrics(
        pred_probs=probs_,
        pred_labels=pred_labels_,
        gt_labels=gt_labels_,
        results_folder_path=results_folder_path,
        dataset_name='mimiccxr_test_set',
        metric_prefix=f'{label_group_name}_nli_based_labels',
        strings=strings,
    )

def _compute_and_save_report_gen_metrics(
        gt_reports, gen_reports, dataset_name, results_folder_path, max_processes, strings):
    metrics = compute_report_level_metrics(gt_reports, gen_reports, max_processes=max_processes)
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_report_gen_metrics({",".join(strings)}).pkl')
    save_pickle(metrics, save_path)
    print_bold(f'Report generation metrics successfully saved to {save_path}')
    return metrics

def _save_gen_reports(gen_reports, dicom_ids, report_idxs, dataset_name, results_folder_path, strings):
    assert len(gen_reports) == len(dicom_ids)
    assert len(dicom_ids) == len(report_idxs)
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_gen_reports({",".join(strings)}).pkl')
    save_pickle({
        'gen_reports': gen_reports,
        'dicom_ids': dicom_ids,
        'report_idxs': report_idxs,
    }, save_path)
    print_bold(f'Generated reports successfully saved to {save_path}')

def _save_gen_reports_v2(gen_reports, gt_reports, image_paths_list, dataset_name, results_folder_path, strings):
    assert len(gen_reports) == len(gt_reports)
    assert len(gen_reports) == len(image_paths_list)
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_gen_reports({",".join(strings)}).pkl')
    save_pickle({
        'gen_reports': gen_reports,
        'gt_reports': gt_reports,
        'image_paths_list': image_paths_list,
    }, save_path)
    print_bold(f'Generated reports successfully saved to {save_path}')

def _evaluate_mimiccxr_test_set__nli_based_labels__template_based_report_gen(
        pred_labels, dicom_ids, results_folder_path, label_based_facts,
        background_findings_and_impression_per_report_filepath, strings,
        max_processes_for_chexpert_labeler, template_fn=None):
    
    bfipr = get_cached_json_file(background_findings_and_impression_per_report_filepath)
    dicom_id2ridx = get_imageId2reportId()

    label_idxs = [LABEL_BASED_FACTS.index(fact) for fact in label_based_facts]
    report_idxs = [dicom_id2ridx[dicom_id] for dicom_id in dicom_ids]
    pred_labels_ = pred_labels[:, label_idxs]
    gen_reports = []
    gt_reports = []
    for i in range(len(pred_labels_)):
        
        # generate report
        report = []
        for j in range(len(pred_labels_[i])):
            if template_fn is not None:
                report.append(template_fn(j, pred_labels_[i, j]))
            else:
                if pred_labels_[i, j] == 1:
                    report.append(label_based_facts[j])
                else:
                    report.append(f'no {label_based_facts[j]}')
                    if report[-1].startswith('no no '):
                        report[-1] = report[-1][6:] # remove double negation
        gen_reports.append('. '.join(report))

        # gt report
        ridx = report_idxs[i]
        findings = bfipr[ridx]['findings']
        impression = bfipr[ridx]['impression']
        if findings and impression:
            if findings[-1] == '.':
                report = findings + ' ' + impression
            else:
                report = findings + '. ' + impression
        elif findings:
            report = findings
        elif impression:
            report = impression
        else:
            report = ''
        gt_reports.append(report)

    print('len(gen_reports) =', len(gen_reports))
    print('len(gt_reports) =', len(gt_reports))

    _compute_and_save_report_gen_metrics(
        gt_reports=gt_reports,
        gen_reports=gen_reports,
        dataset_name='mimiccxr_test_set',
        results_folder_path=results_folder_path,
        max_processes=max_processes_for_chexpert_labeler,
        strings=strings,
    )
    _save_gen_reports(
        gen_reports=gen_reports,
        dicom_ids=dicom_ids,
        report_idxs=report_idxs,
        dataset_name='mimiccxr_test_set',
        results_folder_path=results_folder_path,
        strings=strings,
    )

def _evaluate_interpret_cxr_test_public__nli_based_labels__template_based_report_gen(
        pred_labels, gt_reports, image_paths_list, results_folder_path, label_based_facts, strings,
        max_processes_for_chexpert_labeler, template_fn=None):
    
    assert len(pred_labels) == len(gt_reports)
    assert len(pred_labels) == len(image_paths_list)
    assert pred_labels.shape == (len(gt_reports), len(LABEL_BASED_FACTS))

    label_idxs = [LABEL_BASED_FACTS.index(fact) for fact in label_based_facts]
    pred_labels_ = pred_labels[:, label_idxs]
    
    gen_reports = []
    for i in range(len(pred_labels_)):
        report = []
        for j in range(len(pred_labels_[i])):
            if template_fn is not None:
                report.append(template_fn(j, pred_labels_[i, j]))
            else:
                if pred_labels_[i, j] == 1:
                    report.append(label_based_facts[j])
                else:
                    report.append(f'no {label_based_facts[j]}')
                    if report[-1].startswith('no no '):
                        report[-1] = report[-1][6:] # remove double negation
        gen_reports.append('. '.join(report))
    
    print('len(gen_reports) =', len(gen_reports))
    print('len(gt_reports) =', len(gt_reports))
    assert len(gen_reports) == len(gt_reports)

    _compute_and_save_report_gen_metrics(
        gt_reports=gt_reports,
        gen_reports=gen_reports,
        dataset_name='interpret_cxr_test_public',
        results_folder_path=results_folder_path,
        max_processes=max_processes_for_chexpert_labeler,
        strings=strings,
    )
    _save_gen_reports_v2(
        gen_reports=gen_reports,
        gt_reports=gt_reports,
        image_paths_list=image_paths_list,
        dataset_name='interpret_cxr_test_public',
        results_folder_path=results_folder_path,
        strings=strings,
    )

def _evaluate_interpret_cxr_test_public__generic_report_gen(
        gen_reports, gt_reports, image_paths_list, results_folder_path, strings,
        max_processes_for_chexpert_labeler):
    
    assert len(gen_reports) == len(gt_reports)
    assert len(gen_reports) == len(image_paths_list)
    print('len(gen_reports) =', len(gen_reports))

    _compute_and_save_report_gen_metrics(
        gt_reports=gt_reports,
        gen_reports=gen_reports,
        dataset_name='interpret_cxr_test_public',
        results_folder_path=results_folder_path,
        max_processes=max_processes_for_chexpert_labeler,
        strings=strings,
    )
    _save_gen_reports_v2(
        gen_reports=gen_reports,
        gt_reports=gt_reports,
        image_paths_list=image_paths_list,
        dataset_name='interpret_cxr_test_public',
        results_folder_path=results_folder_path,
        strings=strings,
    )

def _run_inference_for_label_based_fact_classification_on_images(
        image_paths, checkpoint_folder_path, model_kwargs, val_image_transform_kwargs,
        max_images_per_batch, max_facts_per_image, num_workers, device,
        fact_embedding_model_name, fact_embedding_model_checkpoint_folder_path,
        fact_embedding_batch_size, fact_embedding_num_workers, fact_embedding_device):
    # device
    device = torch.device(device)

    # Create model
    print('Creating instance of PhraseGrounder ...')
    model = PhraseGrounder(**model_kwargs)
    model = model.to(device)

    # Load model from checkpoint
    model_wrapper = ModelWrapper(model)
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    print('Loading model from checkpoint ...')
    print('checkpoint_path =', checkpoint_path)
    model_wrapper.load_checkpoint(checkpoint_path, device, model_only=True)

    # Create eval dataloader
    print('Creating dataloader ...')

    test_image_transform = get_image_transform(**val_image_transform_kwargs[DATASET_NAMES.MIMICCXR])

    facts = LABEL_BASED_FACTS

    fact_encoder = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device=fact_embedding_device,
    )

    fact_embeddings = fact_encoder.compute_text_embeddings(facts)
    
    fact_idxs = [list(range(len(facts))) for _ in range(len(image_paths))]    

    eval_dataloader, fact_idxs, image_paths = _create_dataloader(
        max_images_per_batch=max_images_per_batch,
        max_facts_per_image=max_facts_per_image,
        num_workers=num_workers,
        image_paths=image_paths,
        image_transform=test_image_transform,
        fact_embeddings=fact_embeddings,
        fact_idxs=fact_idxs,
    )

    # Run inference
    model.eval()
    image_path_list = [None] * len(image_paths) * len(facts) # make sure to have enough space
    fact_idx_list = [None] * len(image_paths) * len(facts)
    probs_list = [None] * len(image_paths) * len(facts)
    k = 0
    print_blue('Running inference ...', bold=True)
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), mininterval=2):
            images = batch['i'].to(device)
            fact_embeddings = batch['f'].to(device)
            idxs = batch['idx']
            model_output = model(
                raw_images=images,
                phrase_embeddings=fact_embeddings,
                mimiccxr_forward=True,
            )
            logits = model_output['phrase_classifier_logits']
            probs = torch.sigmoid(logits)
            assert probs.shape == (len(images), len(fact_idxs[idxs[0]]))
            assert len(fact_idxs[idxs[0]]) == len(fact_idxs[idxs[-1]])
            for i, idx in enumerate(idxs):
                image_path = image_paths[idx]
                assert len(fact_idxs[idx]) == probs.shape[1]
                for j, fact_idx in enumerate(fact_idxs[idx]):
                    # image_path_list.append(image_path)
                    # fact_idx_list.append(fact_idx)
                    # probs_list.append(probs[i, j].item())
                    image_path_list[k] = image_path
                    fact_idx_list[k] = fact_idx
                    probs_list[k] = probs[i, j].item()
                    k += 1
    image_path_list = image_path_list[:k] # trim
    fact_idx_list = fact_idx_list[:k]
    probs_list = probs_list[:k]

    print('len(image_path_list) =', len(image_path_list))
    print('len(fact_idx_list) =', len(fact_idx_list))
    print('len(probs_list) =', len(probs_list))
    
    unique_image_paths = set(image_path_list)
    unique_image_paths = list(unique_image_paths)
    image_path_to_idx = {image_path: i for i, image_path in enumerate(unique_image_paths)}
    n_images = len(unique_image_paths)
    probs = np.zeros((n_images, len(facts)))
    seen = np.zeros((n_images, len(facts)), dtype=bool)
    for image_path, fact_idx, prob in zip(image_path_list, fact_idx_list, probs_list):
        i = image_path_to_idx[image_path]
        probs[i, fact_idx] = prob
        seen[i, fact_idx] = True
    assert np.all(seen)

    # Release memory
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Return
    return probs, unique_image_paths

def _run_inference_for_label_based_fact_classification_on_mimiccxr_split(
        split, checkpoint_folder_path, model_kwargs, val_image_transform_kwargs,
        max_images_per_batch, max_facts_per_image, num_workers, device,
        fact_embedding_model_name, fact_embedding_model_checkpoint_folder_path,
        fact_embedding_batch_size, fact_embedding_num_workers, fact_embedding_device):

    image_paths = _get_mimiccxr_image_paths(split)
    assert len(set(image_paths)) == len(image_paths)

    probs, image_paths = _run_inference_for_label_based_fact_classification_on_images(
        image_paths=image_paths,
        checkpoint_folder_path=checkpoint_folder_path,
        model_kwargs=model_kwargs,
        val_image_transform_kwargs=val_image_transform_kwargs,
        max_images_per_batch=max_images_per_batch,
        max_facts_per_image=max_facts_per_image,
        num_workers=num_workers,
        device=device,
        fact_embedding_model_name=fact_embedding_model_name,
        fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        fact_embedding_batch_size=fact_embedding_batch_size,
        fact_embedding_num_workers=fact_embedding_num_workers,
        fact_embedding_device=fact_embedding_device,
    )
    dicom_ids = [os.path.basename(image_path).split('.')[0] for image_path in image_paths]
    assert len(set(dicom_ids)) == len(dicom_ids), f'len(set(dicom_ids)) = {len(set(dicom_ids))}, len(dicom_ids) = {len(dicom_ids)}'
    return probs, dicom_ids

def _tune_thresholds_for_label_based_fact_classification(
    probs, dicom_ids, mimiccxr_report_fact_nli_integrated_data_filepath,
):
    data = get_cached_pickle_file(mimiccxr_report_fact_nli_integrated_data_filepath)
    label_based_nli_predictions = data['label_based_nli_predictions']
    label_based_facts = data['label_based_facts']
    assert label_based_facts == LABEL_BASED_FACTS
    nli_predictions = label_based_nli_predictions['nli_predictions']
    
    dicom_id2ridx = get_imageId2reportId()
    gt_labels = np.array([nli_predictions[dicom_id2ridx[dicom_id]] for dicom_id in dicom_ids])
    n_undecided = np.sum(gt_labels == 3)
    print(f'n_undecided = {n_undecided}/{len(gt_labels.flatten())}')
    gt_labels = gt_labels == 0 # convert to 0's and 1's
    gt_labels = gt_labels.astype(int)

    assert gt_labels.shape == probs.shape
    print('probs.shape =', probs.shape)
    print('gt_labels.shape =', gt_labels.shape)

    best_thresholds = np.empty(probs.shape[1])
    best_f1s = np.empty(probs.shape[1])
    best_accs = np.empty(probs.shape[1])
    for i in range(probs.shape[1]):
        best_t, best_f1 = best_threshold_and_f1_score(probs[:, i], gt_labels[:, i])
        best_thresholds[i] = best_t
        best_f1s[i] = best_f1
        best_accs[i] = accuracy_score(gt_labels[:, i], probs[:, i] > best_t)
    print(f'best_f1s.mean() = {best_f1s.mean()}')
    print(f'best_accs.mean() = {best_accs.mean()}')
    return best_thresholds, best_f1s, best_accs

def _load_interpret_cxr_test_public_data(section):
    df = pd.read_csv(INTERPRET_CXR_TEST_PUBLIC_CSV_PATH)
    df = df.replace(np.nan, '', regex=True) # replace nan with empty string
    image_paths_list = df['images_path'].tolist()
    findings_list = df['findings'].tolist()
    impression_list = df['impression'].tolist()
    image_paths_list = [eval(x) for x in image_paths_list] # convert string to list
    image_paths_list_ = []
    gt_reports_ = []

    for i, image_paths in enumerate(image_paths_list):
        
        if section == 'findings':
            gt_report = findings_list[i]
        elif section == 'impression':
            gt_report = impression_list[i]
        elif section == 'both':
            gt_report = ''
            if findings_list[i]:
                gt_report += findings_list[i]
            if impression_list[i]:
                if gt_report:
                    if gt_report[-1] == '.':
                        gt_report += ' '
                    else:
                        gt_report += '. '
                gt_report += impression_list[i]
        else:
            raise ValueError(f'Invalid section: {section}')
        
        if gt_report:
            # report
            gt_reports_.append(gt_report)
            # image paths
            image_paths_ = []
            for image_path in image_paths:
                image_path = os.path.join(INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH, os.path.basename(image_path))
                assert os.path.exists(image_path)
                image_paths_.append(image_path)
            image_paths_list_.append(image_paths_)

    return image_paths_list_, gt_reports_

def _load_interpret_cxr_test_public_image_paths():
    df = pd.read_csv(INTERPRET_CXR_TEST_PUBLIC_CSV_PATH)
    df = df.replace(np.nan, '', regex=True) # replace nan with empty string
    image_paths_list = df['images_path'].tolist()
    image_paths_list = [eval(x) for x in image_paths_list] # convert string to list
    image_path_list = []
    for image_paths in image_paths_list:
        for image_path in image_paths:
            image_path = os.path.join(INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH, os.path.basename(image_path))
            assert os.path.exists(image_path)
            image_path_list.append(image_path)
    return image_path_list

def _load_interpret_cxr_test_hidden_image_paths():
    df = pd.read_csv(INTERPRET_CXR_TEST_HIDDEN_CSV_PATH)
    df = df.replace(np.nan, '', regex=True) # replace nan with empty string
    image_paths_list = df['images_path'].tolist()
    image_paths_list = [eval(x) for x in image_paths_list] # convert string to list
    image_path_list = []
    for image_paths in image_paths_list:
        for image_path in image_paths:
            image_path = os.path.join(INTERPRET_CXR_TEST_HIDDEN_IMAGES_FOLDER_PATH, os.path.basename(image_path))
            assert os.path.exists(image_path)
            image_path_list.append(image_path)
    return image_path_list

def _compute_thresholds(mimiccxr_report_fact_nli_integrated_data_filepath, results_folder_path,
    checkpoint_folder_path, model_kwargs, val_image_transform_kwargs, max_images_per_batch,
    max_facts_per_image, num_workers, device, fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path, fact_embedding_batch_size, fact_embedding_num_workers,
    fact_embedding_device, count_print,
):
    assert mimiccxr_report_fact_nli_integrated_data_filepath is not None
    thresholds_save_path = get_file_path_with_hashing_if_too_long(
        folder_path=results_folder_path,
        prefix='thresholds',
        strings=[
            'label_based_facts',
            mimiccxr_report_fact_nli_integrated_data_filepath,
            'mimiccxr:validate+test',
        ],
        force_hashing=True,
    )
    if os.path.exists(thresholds_save_path):
        print('Loading thresholds from', thresholds_save_path)
        output = load_pickle(thresholds_save_path)
    else:
        count_print('Tuning thresholds on MIMIC-CXR validation and test sets with label-based facts')
        probs, dicom_ids = _run_inference_for_label_based_fact_classification_on_mimiccxr_split(
            split=['validate', 'test'], # use both validation and test sets
            checkpoint_folder_path=checkpoint_folder_path,
            model_kwargs=model_kwargs,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=max_images_per_batch,
            max_facts_per_image=max_facts_per_image,
            num_workers=num_workers,
            device=device,
            fact_embedding_model_name=fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=fact_embedding_batch_size,
            fact_embedding_num_workers=fact_embedding_num_workers,
            fact_embedding_device=fact_embedding_device,
        )
        assert len(probs) == len(dicom_ids)
        thresholds, f1s, accs = _tune_thresholds_for_label_based_fact_classification(
            probs=probs,
            dicom_ids=dicom_ids,
            mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
        )
        print('Saving thresholds to', thresholds_save_path)
        output = { 'thresholds': thresholds, 'f1s': f1s , 'accs': accs }
        save_pickle(output, thresholds_save_path)
    assert len(output['thresholds']) == len(LABEL_BASED_FACTS)
    return output

def _evaluate_model(
    checkpoint_folder_path,
    model_kwargs,
    val_image_transform_kwargs,
    max_images_per_batch,
    max_facts_per_image,
    num_workers,
    device,
    eval_mode,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    fact_embedding_device,
    chexpert_labels_filepath,
    chest_imagenome_label_names_filepath,
    chest_imagenome_image_id_to_labels_filepath,
    mimiccxr_report_fact_nli_integrated_data_filepath,
    tune_thresholds,
    max_processes_for_chexpert_labeler,
    background_findings_and_impression_per_report_filepath,
    use_alternative_chexpert_template,
    eval_chexpert_only,
    section_mode,
    dicom_id_to_pos_neg_facts_filepath,
    f1_threshold,
    label_based_json_reports_filepath,
    json_to_gpt_reports_jsonl_filepath,
    mimiccxr_interpret_cxr_challenge_split_filepath,
    iuxray_interpret_cxr_challenge_split_filepath,
    chexpert_interpret_cxr_challenge_split_filepath,
):
    count_print = CountPrinter()

    if eval_mode == _EvalModes.MIMICCXR_TEST_SET_LABEL_BASED:

        assert chexpert_labels_filepath is not None
        assert chest_imagenome_label_names_filepath is not None
        assert chest_imagenome_image_id_to_labels_filepath is not None
        assert mimiccxr_report_fact_nli_integrated_data_filepath is not None

        strings = []

        if tune_thresholds:
            count_print('Tuning thresholds on MIMIC-CXR validation set with label-based facts')
            val_probs, val_dicom_ids = _run_inference_for_label_based_fact_classification_on_mimiccxr_split(
                split='validate',
                checkpoint_folder_path=checkpoint_folder_path,
                model_kwargs=model_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                max_images_per_batch=max_images_per_batch,
                max_facts_per_image=max_facts_per_image,
                num_workers=num_workers,
                device=device,
                fact_embedding_model_name=fact_embedding_model_name,
                fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
                fact_embedding_batch_size=fact_embedding_batch_size,
                fact_embedding_num_workers=fact_embedding_num_workers,
                fact_embedding_device=fact_embedding_device,
            )
            assert len(val_probs) == len(val_dicom_ids)
            thresholds, f1s, accs = _tune_thresholds_for_label_based_fact_classification(
                probs=val_probs,
                dicom_ids=val_dicom_ids,
                mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
            )
            strings.append('tuned_thresholds')
        else:
            thresholds = None

        count_print('Evaluating model on MIMIC-CXR test set with label-based facts')

        test_probs, test_dicom_ids = _run_inference_for_label_based_fact_classification_on_mimiccxr_split(
            split='test',
            checkpoint_folder_path=checkpoint_folder_path,
            model_kwargs=model_kwargs,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=max_images_per_batch,
            max_facts_per_image=max_facts_per_image,
            num_workers=num_workers,
            device=device,
            fact_embedding_model_name=fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=fact_embedding_batch_size,
            fact_embedding_num_workers=fact_embedding_num_workers,
            fact_embedding_device=fact_embedding_device,
        )
        assert len(test_probs) == len(test_dicom_ids)
        if thresholds is not None:
            test_pred_labels = (test_probs > thresholds).astype(int)
        else:
            test_pred_labels = (test_probs > 0.5).astype(int) # simple thresholding

        results_folder_path = get_results_folder_path(checkpoint_folder_path)

        # Compute metrics with CheXpert labels
        count_print('Computing metrics with CheXpert labels')
        _evaluate_mimiccxr_test_set_chexpert_fact_classification(
            probs=test_probs,
            pred_labels=test_pred_labels,
            chexpert_labels_filepath=chexpert_labels_filepath,
            dicom_ids=test_dicom_ids,
            results_folder_path=results_folder_path,
            strings=strings,
        )

        # Compute metrics with MIMIC-CXR LT labels
        count_print('Computing metrics with MIMIC-CXR LT labels')
        _evaluate_mimiccxr_test_set_mimic_cxr_lt_fact_classification(
            probs=test_probs,
            pred_labels=test_pred_labels,
            dicom_ids=test_dicom_ids,
            results_folder_path=results_folder_path,
            strings=strings,
        )

        # Compute metrics with Chest ImaGenome labels
        count_print('Computing metrics with Chest ImaGenome labels')
        _evaluate_mimiccxr_test_set_chest_imagenome_fact_classification(
            probs=test_probs,
            pred_labels=test_pred_labels,
            dicom_ids=test_dicom_ids,
            results_folder_path=results_folder_path,
            chest_imagenome_label_names_filepath=chest_imagenome_label_names_filepath,
            chest_imagenome_image_id_to_labels_filepath=chest_imagenome_image_id_to_labels_filepath,
            strings=strings,
        )

        # Compute metrics with NLI-based labels and CheXpert facts
        count_print('Computing metrics with NLI-based labels and CheXpert facts')
        _evaluate_mimiccxr_test_set_nli_based_labels_fact_classification(
            probs=test_probs,
            pred_labels=test_pred_labels,
            dicom_ids=test_dicom_ids,
            results_folder_path=results_folder_path,
            label_based_facts=LABEL_BASED_FACTS__CHEXPERT,
            label_group_name='chexpert',
            mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
            strings=strings,
        )

        # Compute metrics with NLI-based labels and MIMIC-CXR LT facts
        count_print('Computing metrics with NLI-based labels and MIMIC-CXR LT facts')
        _evaluate_mimiccxr_test_set_nli_based_labels_fact_classification(
            probs=test_probs,
            pred_labels=test_pred_labels,
            dicom_ids=test_dicom_ids,
            results_folder_path=results_folder_path,
            label_based_facts=LABEL_BASED_FACTS__MIMIC_CXR_LT,
            label_group_name='mimic_cxr_lt',
            mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
            strings=strings,
        )

        # Compute metrics with NLI-based labels and Chest ImaGenome facts
        count_print('Computing metrics with NLI-based labels and Chest ImaGenome facts')
        _evaluate_mimiccxr_test_set_nli_based_labels_fact_classification(
            probs=test_probs,
            pred_labels=test_pred_labels,
            dicom_ids=test_dicom_ids,
            results_folder_path=results_folder_path,
            label_based_facts=LABEL_BASED_FACTS__CHEST_IMAGENOME,
            label_group_name='chest_imagenome',
            mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
            strings=strings,
        )

    elif eval_mode == _EvalModes.MIMICCXR_TEST_SET_LABEL_BASED__TEMPLATE_BASED_REPORT_GEN:

        assert background_findings_and_impression_per_report_filepath is not None
        assert mimiccxr_report_fact_nli_integrated_data_filepath is not None

        strings = []

        if tune_thresholds:
            tmp = _compute_thresholds(
                tune_thresholds=tune_thresholds,
                mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
                results_folder_path=results_folder_path,
                checkpoint_folder_path=checkpoint_folder_path,
                model_kwargs=model_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                max_images_per_batch=max_images_per_batch,
                max_facts_per_image=max_facts_per_image,
                num_workers=num_workers,
                device=device,
                fact_embedding_model_name=fact_embedding_model_name,
                fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
                fact_embedding_batch_size=fact_embedding_batch_size,
                fact_embedding_num_workers=fact_embedding_num_workers,
                fact_embedding_device=fact_embedding_device,
                count_print=count_print,
            )
            strings.append('tuned_thresholds')
            thresholds = tmp['thresholds']
        else:
            thresholds = 0.5 # simple thresholding

        count_print('Evaluating model on MIMIC-CXR test set with label-based facts')

        test_probs, test_dicom_ids = _run_inference_for_label_based_fact_classification_on_mimiccxr_split(
            split='test',
            checkpoint_folder_path=checkpoint_folder_path,
            model_kwargs=model_kwargs,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=max_images_per_batch,
            max_facts_per_image=max_facts_per_image,
            num_workers=num_workers,
            device=device,
            fact_embedding_model_name=fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=fact_embedding_batch_size,
            fact_embedding_num_workers=fact_embedding_num_workers,
            fact_embedding_device=fact_embedding_device,
        )
        assert len(test_probs) == len(test_dicom_ids)
        test_pred_labels = (test_probs > thresholds).astype(int)

        eval_chexpert = True
        eval_mimic_cxr_lt = True
        eval_chest_imagenome = True
        eval_all = True

        if eval_chexpert_only:
            eval_mimic_cxr_lt = False
            eval_chest_imagenome = False
            eval_all = False

        # Compute report gen metrics with NLI-based labels and CheXpert facts
        if eval_chexpert:
            count_print('Computing report gen metrics with NLI-based labels and CheXpert facts')
            strings_ = ['chexpert']
            strings_.extend(strings)
            if use_alternative_chexpert_template:
                strings_.append('v1')
                def _template_fn(i, j):
                    return TEMPLATES_CHEXPERT_v1[CHEXPERT_LABELS[i]][j]
            else:
                _template_fn = None
            _evaluate_mimiccxr_test_set__nli_based_labels__template_based_report_gen(
                pred_labels=test_pred_labels,
                dicom_ids=test_dicom_ids,
                results_folder_path=results_folder_path,
                label_based_facts=LABEL_BASED_FACTS__CHEXPERT,
                background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                strings=strings_,
                template_fn=_template_fn,
            )

        # Compute report gen metrics with NLI-based labels and MIMIC-CXR LT facts
        if eval_mimic_cxr_lt:
            count_print('Computing report gen metrics with NLI-based labels and MIMIC-CXR LT facts')
            strings_ = ['mimic_cxr_lt']
            strings_.extend(strings)
            _evaluate_mimiccxr_test_set__nli_based_labels__template_based_report_gen(
                pred_labels=test_pred_labels,
                dicom_ids=test_dicom_ids,
                results_folder_path=results_folder_path,
                label_based_facts=LABEL_BASED_FACTS__MIMIC_CXR_LT,
                background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                strings=strings_,
            )

        # Compute report gen metrics with NLI-based labels and Chest ImaGenome facts
        if eval_chest_imagenome:
            count_print('Computing report gen metrics with NLI-based labels and Chest ImaGenome facts')
            strings_ = ['chest_imagenome']
            strings_.extend(strings)
            _evaluate_mimiccxr_test_set__nli_based_labels__template_based_report_gen(
                pred_labels=test_pred_labels,
                dicom_ids=test_dicom_ids,
                results_folder_path=results_folder_path,
                label_based_facts=LABEL_BASED_FACTS__CHEST_IMAGENOME,
                background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                strings=strings_,
            )

        # Compute report gen metrics with NLI-based labels and all label-based facts
        if eval_all:
            count_print('Computing report gen metrics with NLI-based labels and all label-based facts')
            strings_ = ['chexpert+mimic_cxr_lt+chest_imagenome']
            strings_.extend(strings)
            _evaluate_mimiccxr_test_set__nli_based_labels__template_based_report_gen(
                pred_labels=test_pred_labels,
                dicom_ids=test_dicom_ids,
                results_folder_path=results_folder_path,
                label_based_facts=LABEL_BASED_FACTS,
                background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                strings=strings_,
            )

    elif eval_mode == _EvalModes.INTERPRET_CXR_TEST_PUBLIC_LABEL_BASED__TEMPLATE_BASED_REPORT_GEN:

        assert section_mode is not None

        strings = []
        strings.append('template')
        strings.append(section_mode)

        results_folder_path = get_results_folder_path(checkpoint_folder_path)

        if tune_thresholds:
            tmp = _compute_thresholds(
                tune_thresholds=tune_thresholds,
                mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
                results_folder_path=results_folder_path,
                checkpoint_folder_path=checkpoint_folder_path,
                model_kwargs=model_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                max_images_per_batch=max_images_per_batch,
                max_facts_per_image=max_facts_per_image,
                num_workers=num_workers,
                device=device,
                fact_embedding_model_name=fact_embedding_model_name,
                fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
                fact_embedding_batch_size=fact_embedding_batch_size,
                fact_embedding_num_workers=fact_embedding_num_workers,
                fact_embedding_device=fact_embedding_device,
                count_print=count_print,
            )
            thresholds = tmp['thresholds']
            strings.append('thresholds')
        else:
            thresholds = 0.5 # simple thresholding

        images_path_list, gt_reports_list = _load_interpret_cxr_test_public_data(section=section_mode)
        print('len(images_path_list) =', len(images_path_list))
        print('len(gt_reports_list) =', len(gt_reports_list))

        image_paths = []
        for image_paths_ in images_path_list:
            image_paths.extend(image_paths_)
        print('len(image_paths) =', len(image_paths))
        assert len(set(image_paths)) == len(image_paths) # no duplicates

        probs, image_paths = _run_inference_for_label_based_fact_classification_on_images(
            image_paths=image_paths,
            checkpoint_folder_path=checkpoint_folder_path,
            model_kwargs=model_kwargs,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=max_images_per_batch,
            max_facts_per_image=max_facts_per_image,
            num_workers=num_workers,
            device=device,
            fact_embedding_model_name=fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=fact_embedding_batch_size,
            fact_embedding_num_workers=fact_embedding_num_workers,
            fact_embedding_device=fact_embedding_device,
        )
        image_path_to_idx = {image_path: i for i, image_path in enumerate(image_paths)}
        assert len(image_path_to_idx) == len(image_paths)

        pred_labels = np.empty((len(images_path_list), len(LABEL_BASED_FACTS)), dtype=int)
        for i, image_paths_ in enumerate(images_path_list):
            assert len(image_paths_) > 0
            image_idxs = [image_path_to_idx[image_path] for image_path in image_paths_]
            probs_ = probs[image_idxs]
            probs_ = np.mean(probs_, axis=0) # average probs
            assert probs_.shape == (len(LABEL_BASED_FACTS),)
            pred_labels[i] = (probs_ > thresholds).astype(int) 

        print(f'pred_labels.shape = {pred_labels.shape}')

        eval_chexpert = True
        eval_mimic_cxr_lt = True
        eval_chest_imagenome = True
        eval_all = True

        if eval_chexpert_only:
            eval_mimic_cxr_lt = False
            eval_chest_imagenome = False
            eval_all = False

        # Compute report gen metrics with NLI-based labels and CheXpert facts
        if eval_chexpert:
            count_print('Computing report gen metrics with NLI-based labels and CheXpert facts')
            strings_ = ['chexpert']
            strings_.extend(strings)
            if use_alternative_chexpert_template:
                strings_.append('v1')
                def _template_fn(i, j):
                    return TEMPLATES_CHEXPERT_v1[CHEXPERT_LABELS[i]][j]
            else:
                _template_fn = None
            _evaluate_interpret_cxr_test_public__nli_based_labels__template_based_report_gen(
                pred_labels=pred_labels,
                gt_reports=gt_reports_list,
                image_paths_list=images_path_list,
                results_folder_path=results_folder_path,
                label_based_facts=LABEL_BASED_FACTS__CHEXPERT,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                strings=strings_,
                template_fn=_template_fn,
            )

        # Compute report gen metrics with NLI-based labels and MIMIC-CXR LT facts
        if eval_mimic_cxr_lt:
            count_print('Computing report gen metrics with NLI-based labels and MIMIC-CXR LT facts')
            strings_ = ['mimic_cxr_lt']
            strings_.extend(strings)
            _evaluate_interpret_cxr_test_public__nli_based_labels__template_based_report_gen(
                pred_labels=pred_labels,
                gt_reports=gt_reports_list,
                image_paths_list=images_path_list,
                results_folder_path=results_folder_path,
                label_based_facts=LABEL_BASED_FACTS__MIMIC_CXR_LT,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                strings=strings_,
            )

        # Compute report gen metrics with NLI-based labels and Chest ImaGenome facts
        if eval_chest_imagenome:
            count_print('Computing report gen metrics with NLI-based labels and Chest ImaGenome facts')
            strings_ = ['chest_imagenome']
            strings_.extend(strings)
            _evaluate_interpret_cxr_test_public__nli_based_labels__template_based_report_gen(
                pred_labels=pred_labels,
                gt_reports=gt_reports_list,
                image_paths_list=images_path_list,
                results_folder_path=results_folder_path,
                label_based_facts=LABEL_BASED_FACTS__CHEST_IMAGENOME,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                strings=strings_,
            )

        # Compute report gen metrics with NLI-based labels and all label-based facts
        if eval_all:
            count_print('Computing report gen metrics with NLI-based labels and all label-based facts')
            strings_ = ['chexpert+mimic_cxr_lt+chest_imagenome']
            strings_.extend(strings)
            _evaluate_interpret_cxr_test_public__nli_based_labels__template_based_report_gen(
                pred_labels=pred_labels,
                gt_reports=gt_reports_list,
                image_paths_list=images_path_list,
                results_folder_path=results_folder_path,
                label_based_facts=LABEL_BASED_FACTS,
                max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
                strings=strings_,
            )
    
    elif eval_mode == _EvalModes.INTERPRET_CXR_TEST_PUBLIC_LABEL_BASED__GENERATE_JSONS_FOR_REPORT_GEN:

        assert section_mode is not None
        assert dicom_id_to_pos_neg_facts_filepath is not None

        strings = []

        results_folder_path = get_results_folder_path(checkpoint_folder_path)

        if tune_thresholds:
            tmp = _compute_thresholds(
                mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
                results_folder_path=results_folder_path,
                checkpoint_folder_path=checkpoint_folder_path,
                model_kwargs=model_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                max_images_per_batch=max_images_per_batch,
                max_facts_per_image=max_facts_per_image,
                num_workers=num_workers,
                device=device,
                fact_embedding_model_name=fact_embedding_model_name,
                fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
                fact_embedding_batch_size=fact_embedding_batch_size,
                fact_embedding_num_workers=fact_embedding_num_workers,
                fact_embedding_device=fact_embedding_device,
                count_print=count_print,
            )
            strings.append('tuned_thresholds')
            thresholds = tmp['thresholds']
            f1s = tmp['f1s']
        else:
            raise ValueError('tune_thresholds must be True')

        # Count the number of positive labels for each fact
        tmp = load_pickle(dicom_id_to_pos_neg_facts_filepath)
        facts = tmp['facts']
        counts = [0] * len(facts)
        validate_counts = [0] * len(facts)
        test_counts = [0] * len(facts)
        
        dicom_id_to_pos_neg_facts = tmp['dicom_id_to_pos_neg_facts']
        split2imageIds = get_split2imageIds()
        for split, counts in zip(['validate', 'test'], [validate_counts, test_counts]):
            dicom_ids = split2imageIds[split]
            for dicom_id in dicom_ids:
                pos_fact_idxs = dicom_id_to_pos_neg_facts[dicom_id][0]
                for fact_idx in pos_fact_idxs:
                    counts[fact_idx] += 1
        fact2idx = {fact: i for i, fact in enumerate(facts)}
        total_count = len(split2imageIds['validate']) + len(split2imageIds['test'])
        label_based_fact_fractions = [counts[fact2idx[fact]] / total_count for fact in LABEL_BASED_FACTS]
        
        images_path_list, gt_reports_list = _load_interpret_cxr_test_public_data(section=section_mode)
        print('len(images_path_list) =', len(images_path_list))
        print('len(gt_reports_list) =', len(gt_reports_list))

        image_paths = []
        for image_paths_ in images_path_list:
            image_paths.extend(image_paths_)
        print('len(image_paths) =', len(image_paths))
        assert len(set(image_paths)) == len(image_paths) # no duplicates

        probs, image_paths = _run_inference_for_label_based_fact_classification_on_images(
            image_paths=image_paths,
            checkpoint_folder_path=checkpoint_folder_path,
            model_kwargs=model_kwargs,
            val_image_transform_kwargs=val_image_transform_kwargs,
            max_images_per_batch=max_images_per_batch,
            max_facts_per_image=max_facts_per_image,
            num_workers=num_workers,
            device=device,
            fact_embedding_model_name=fact_embedding_model_name,
            fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            fact_embedding_batch_size=fact_embedding_batch_size,
            fact_embedding_num_workers=fact_embedding_num_workers,
            fact_embedding_device=fact_embedding_device,
        )
        image_path_to_idx = {image_path: i for i, image_path in enumerate(image_paths)}
        assert len(image_path_to_idx) == len(image_paths)
        
        output = []
        for i, image_paths_ in enumerate(images_path_list):
            assert len(image_paths_) > 0
            image_idxs = [image_path_to_idx[image_path] for image_path in image_paths_]
            probs_ = probs[image_idxs]
            mean_prob_ = np.mean(probs_, axis=0)
            pred_labels = (probs_ > thresholds).astype(int)
            mean_pred_labels = (mean_prob_ > thresholds).astype(int)
            lines = []
            for j in range(len(LABEL_BASED_FACTS)):
                if f1s[j] < f1_threshold:
                    continue # skip if f1 is too low
                assert LABEL_BASED_FACTS[j].endswith(' seen')
                label_str = LABEL_BASED_FACTS[j][:-5] # remove ' seen'
                pred_prob_str = ','.join(f'{"yes" if x == 1 else "no"}({p:.2f})' for x, p in zip(pred_labels[:, j].tolist(), probs_[:, j].tolist()))
                if "yes(" not in pred_prob_str:
                    if label_based_fact_fractions[j] < 0.1:
                        continue
                if "yes(" in pred_prob_str and "no(" in pred_prob_str: # conflicting predictions
                    if mean_pred_labels[j] == 0:
                        continue
                prior_f1_str = f'{f1s[j]:.3f}'
                prior_prob_str = f'{label_based_fact_fractions[j]:.4f}'
                line = f'{label_str}: {pred_prob_str}; {prior_f1_str}; {prior_prob_str}'
                lines.append((label_based_fact_fractions[j], line))
            lines = [x[1] for x in sorted(lines, key=lambda x: x[0], reverse=True)]
            lines_with_yes = [x for x in lines if 'yes(' in x]
            lines_with_only_no = [x for x in lines if 'no(' in x and 'yes(' not in x]
            assert len(lines_with_yes) + len(lines_with_only_no) == len(lines)
        
            obj = {
                'num_images': len(image_paths_),
            }
            if lines_with_yes:
                obj['yes'] = lines_with_yes
            if lines_with_only_no:
                obj['no'] = lines_with_only_no
            output.append({
                'image_paths': image_paths_,
                'json_string': json.dumps(obj, indent=1),
            })

        strings.append('jsons')
        strings.append(section_mode)
        strings.append(dicom_id_to_pos_neg_facts_filepath)
        strings.append(f'f1_threshold={f1_threshold}')

        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=results_folder_path,
            prefix=f'interpret_cxr_test_public__label_based_json(section={section_mode})',
            strings=strings,
            force_hashing=True,
        )
        save_pickle(output, save_path)
        print_blue(f'Saved jsons to {save_path}', bold=True)

    elif eval_mode == _EvalModes.INTERPRET_CXR_TEST_PUBLIC_LABEL_BASED__JSON_TO_GPT_REPORT_GEN:
                
        assert section_mode is not None
        assert label_based_json_reports_filepath is not None
        assert json_to_gpt_reports_jsonl_filepath is not None

        strings = []
        strings.append('json_to_gpt')
        strings.append(section_mode)

        results_folder_path = os.path.dirname(label_based_json_reports_filepath)

        label_based_json_reports = load_pickle(label_based_json_reports_filepath)
        
        images_path_list, gt_reports_list = _load_interpret_cxr_test_public_data(section=section_mode)
        print('len(images_path_list) =', len(images_path_list))
        print('len(gt_reports_list) =', len(gt_reports_list))

        assert len(images_path_list) == len(label_based_json_reports)
        for i in range(len(images_path_list)):
            assert images_path_list[i] == label_based_json_reports[i]['image_paths']

        json_to_gpt_reports = load_jsonl(json_to_gpt_reports_jsonl_filepath)
        query2report = { x['metadata']['query'] : x['parsed_response'] for x in json_to_gpt_reports }
        assert len(query2report) >= len(images_path_list)

        gen_reports = []
        if section_mode == 'findings':
            for x in label_based_json_reports:
                gen_reports.append(query2report[x['json_string']]['findings'])
        elif section_mode == 'impression':
            for x in label_based_json_reports:
                gen_reports.append(query2report[x['json_string']]['impression'])
        elif section_mode == 'both':
            for x in label_based_json_reports:
                report = query2report[x['json_string']]
                gen_reports.append(report['findings'] + ' ' + report['impression'])
        else:
            raise ValueError(f'Invalid section_mode: {section_mode}')

        # Compute and save report gen metrics
        _evaluate_interpret_cxr_test_public__generic_report_gen(
            gen_reports=gen_reports,
            gt_reports=gt_reports_list,
            image_paths_list=images_path_list,
            results_folder_path=results_folder_path,
            strings=strings,
            max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
        )

    elif eval_mode == _EvalModes.INTERPRET_CXR_COMPUTE_AND_SAVE_LABEL_BASED_PREDICTIONS:
        # What we will do (using interpret cxr challenge's official splits):
        # 1. Collect image paths for MIMIC-CXR (train, validate)
        # 2. Collect image paths for CheXpert (train, validate)
        # 3. Collect image paths for IU X-Ray (train, validate)
        # 4. Collect image paths for the public test set
        # 5. Collect image paths for the hidden test set
        # 6. Run inference on all images
        # 7. Tune thresholds on MIMIC-CXR (validate)
        # 8. Save predictions for all images and thresholds in a pickle file

        assert checkpoint_folder_path is not None
        assert mimiccxr_interpret_cxr_challenge_split_filepath is not None
        assert chexpert_interpret_cxr_challenge_split_filepath is not None
        assert iuxray_interpret_cxr_challenge_split_filepath is not None
        assert mimiccxr_report_fact_nli_integrated_data_filepath is not None
        assert fact_embedding_model_name is not None
        assert fact_embedding_model_checkpoint_folder_path is not None

        results_folder_path = get_results_folder_path(checkpoint_folder_path)

        probs_savepath = get_file_path_with_hashing_if_too_long(
            folder_path=results_folder_path,
            prefix='interpret_cxr__label_based_probs',
            strings=['mimiccxr', 'chexpert', 'iuxray', 'public_test', 'hidden_test'],
            force_hashing=True,
        )

        if os.path.exists(probs_savepath):
            print_magenta(f'Found existing label-based probs at {probs_savepath}', bold=True)
            print_magenta('Skipping inference', bold=True)
            tmp = load_pickle(probs_savepath)
            image_paths = tmp['image_paths']
            probs = tmp['probs']
            
            # Load only MIMIC-CXR validation set
            assert mimiccxr_interpret_cxr_challenge_split_filepath is not None
            mimiccxr_splits = load_pickle(mimiccxr_interpret_cxr_challenge_split_filepath)
            imageId2PartPatientStudy = get_imageId2PartPatientStudy()
            mimiccxr_val_dicom_ids = mimiccxr_splits['val']
            mimiccxr_val_image_paths = []
            for dicom_id in mimiccxr_val_dicom_ids:
                part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
                mimiccxr_val_image_paths.append(get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id))
                assert os.path.exists(mimiccxr_val_image_paths[-1])
            print(f'len(mimiccxr_val_image_paths) = {len(mimiccxr_val_image_paths)}')
        else:
            # 1. Collect image paths for MIMIC-CXR (train, validate)
            mimiccxr_splits = load_pickle(mimiccxr_interpret_cxr_challenge_split_filepath)
            imageId2PartPatientStudy = get_imageId2PartPatientStudy()
            mimiccxr_train_dicom_ids = mimiccxr_splits['train']
            mimiccxr_val_dicom_ids = mimiccxr_splits['val']
            mimiccxr_train_image_paths = []
            mimiccxr_val_image_paths = []
            for dicom_id in mimiccxr_train_dicom_ids:
                part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
                mimiccxr_train_image_paths.append(get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id))
                assert os.path.exists(mimiccxr_train_image_paths[-1])
            for dicom_id in mimiccxr_val_dicom_ids:
                part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
                mimiccxr_val_image_paths.append(get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id))
                assert os.path.exists(mimiccxr_val_image_paths[-1])
            print(f'len(mimiccxr_train_image_paths) = {len(mimiccxr_train_image_paths)}')
            print(f'len(mimiccxr_val_image_paths) = {len(mimiccxr_val_image_paths)}')

            # 2. Collect image paths for CheXpert (train, validate)
            chexpert_splits = load_pickle(chexpert_interpret_cxr_challenge_split_filepath)
            chexpert_train_image_paths = []
            chexpert_val_image_paths = []
            for image_path in chexpert_splits['train']:
                image_path = os.path.join(CHEXPERT_V1_0_SMALL_DATASET_DIR, image_path)
                assert os.path.exists(image_path)
                chexpert_train_image_paths.append(image_path)
            for image_path in chexpert_splits['val']:
                image_path = os.path.join(CHEXPERT_V1_0_SMALL_DATASET_DIR, image_path)
                assert os.path.exists(image_path)
                chexpert_val_image_paths.append(image_path)
            print(f'len(chexpert_train_image_paths) = {len(chexpert_train_image_paths)}')
            print(f'len(chexpert_val_image_paths) = {len(chexpert_val_image_paths)}')

            # 3. Collect image paths for IU X-Ray (train, validate)
            iuxray_splits = load_pickle(iuxray_interpret_cxr_challenge_split_filepath)
            iuxray_train_image_paths = []
            iuxray_val_image_paths = []
            for image_id in iuxray_splits['train']:
                image_path = get_iuxray_image_path(image_id)
                assert os.path.exists(image_path)
                iuxray_train_image_paths.append(image_path)
            for image_id in iuxray_splits['val']:
                image_path = get_iuxray_image_path(image_id)
                assert os.path.exists(image_path)
                iuxray_val_image_paths.append(image_path)
            print(f'len(iuxray_train_image_paths) = {len(iuxray_train_image_paths)}')
            print(f'len(iuxray_val_image_paths) = {len(iuxray_val_image_paths)}')

            # 4. Collect image paths for the public test set
            interpret_cxr_public_test_image_paths = _load_interpret_cxr_test_public_image_paths()
            print(f'len(interpret_cxr_public_test_image_paths) = {len(interpret_cxr_public_test_image_paths)}')

            # 5. Collect image paths for the hidden test set
            interpret_cxr_hidden_test_image_paths = _load_interpret_cxr_test_hidden_image_paths()
            print(f'len(interpret_cxr_hidden_test_image_paths) = {len(interpret_cxr_hidden_test_image_paths)}')

            # 6. Run inference on all images
            image_paths = (
                mimiccxr_train_image_paths + mimiccxr_val_image_paths +
                chexpert_train_image_paths + chexpert_val_image_paths +
                iuxray_train_image_paths + iuxray_val_image_paths +
                interpret_cxr_public_test_image_paths + interpret_cxr_hidden_test_image_paths
            )
            print(f'len(image_paths) = {len(image_paths)}')
            probs, image_paths = _run_inference_for_label_based_fact_classification_on_images(
                image_paths=image_paths,
                checkpoint_folder_path=checkpoint_folder_path,
                model_kwargs=model_kwargs,
                val_image_transform_kwargs=val_image_transform_kwargs,
                max_images_per_batch=max_images_per_batch,
                max_facts_per_image=max_facts_per_image,
                num_workers=num_workers,
                device=device,
                fact_embedding_model_name=fact_embedding_model_name,
                fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
                fact_embedding_batch_size=fact_embedding_batch_size,
                fact_embedding_num_workers=fact_embedding_num_workers,
                fact_embedding_device=fact_embedding_device,
            )
            save_pickle({
                'image_paths': image_paths,
                'probs': probs,
            }, probs_savepath)
            print_blue(f'Saved label-based probs to {probs_savepath}', bold=True)

        # 7. Tune thresholds on MIMIC-CXR (validate)
        image_path_to_idx = {image_path: i for i, image_path in enumerate(image_paths)}
        mimiccxr_val_image_idxs = [image_path_to_idx[image_path] for image_path in mimiccxr_val_image_paths]
        mimiccxr_val_probs = probs[mimiccxr_val_image_idxs]
        thresholds, f1s, accs = _tune_thresholds_for_label_based_fact_classification(
            probs=mimiccxr_val_probs,
            dicom_ids=mimiccxr_val_dicom_ids,
            mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
        )

        # 8. Save predictions for all images and thresholds in a pickle file
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=results_folder_path,
            prefix='interpret_cxr__label_based_predictions',
            strings=[
                'mimiccxr', 'chexpert', 'iuxray', 'public_test', 'hidden_test',
                'thresholds', 'f1s', 'accs',
                mimiccxr_report_fact_nli_integrated_data_filepath,
            ],
            force_hashing=True,
        )
        save_pickle({
            'probs_filepath': probs_savepath, # path to probs instead of probs itself
            'thresholds': thresholds,
            'f1s': f1s,
            'accs': accs,
            'class_names': [x[:-5] for x in LABEL_BASED_FACTS], # remove ' seen'
        }, save_path)
        print_blue(f'Saved label-based predictions to {save_path}', bold=True)

    else:
        raise ValueError(f'Invalid eval_mode: {eval_mode}')

def plot_label_based_metrics(metrics_path, label_based_facts, dicom_id_to_pos_neg_facts_filepath, metric_prefix, figsize):
    metrics = load_pickle(metrics_path)
    assert all(fact in LABEL_BASED_FACTS for fact in label_based_facts)
    metrics_to_plot = [
        metrics[f'{metric_prefix}_prf1']['p'],
        metrics[f'{metric_prefix}_prf1']['r'],
        metrics[f'{metric_prefix}_prf1']['f1'],
        metrics[f'{metric_prefix}_rocauc']['per_class'],
        metrics[f'{metric_prefix}_prcauc']['per_class'],
    ]
    for x in metrics_to_plot:
        assert len(x) == len(label_based_facts)
    
    tp = metrics[f'{metric_prefix}_tp']
    fp = metrics[f'{metric_prefix}_fp']
    tn = metrics[f'{metric_prefix}_tn']
    fn = metrics[f'{metric_prefix}_fn']
    for x in [tp, fp, tn, fn]:
        assert len(x) == len(label_based_facts)
    
    tmp = load_pickle(dicom_id_to_pos_neg_facts_filepath)
    facts = tmp['facts']
    train_counts = [0] * len(facts)
    validate_counts = [0] * len(facts)
    test_counts = [0] * len(facts)
    
    dicom_id_to_pos_neg_facts = tmp['dicom_id_to_pos_neg_facts']
    split2imageIds = get_split2imageIds()
    for split, counts in zip(['train', 'validate', 'test'], [train_counts, validate_counts, test_counts]):
        dicom_ids = split2imageIds[split]
        for dicom_id in dicom_ids:
            pos_fact_idxs = dicom_id_to_pos_neg_facts[dicom_id][0]
            for fact_idx in pos_fact_idxs:
                counts[fact_idx] += 1
    
    fact2idx = {fact: i for i, fact in enumerate(facts)}
    label_strings = []
    for i, fact in enumerate(label_based_facts):
        tr = train_counts[fact2idx[fact]]
        va = validate_counts[fact2idx[fact]]
        te = test_counts[fact2idx[fact]]
        tr_percent = tr / len(split2imageIds['train']) * 100
        va_percent = va / len(split2imageIds['validate']) * 100
        te_percent = te / len(split2imageIds['test']) * 100
        label_strings.append(
            f'{fact} (tr: {tr} ({tr_percent:.2f}%), va: {va} ({va_percent:.2f}%), te: {te} ({te_percent:.2f}%))'
            f'; tp: {tp[i]}, fp: {fp[i]}, tn: {tn[i]}, fn: {fn[i]}'
        )

    for metric_name, metric_values in zip(
        ['Precision', 'Recall', 'F1', 'ROC AUC', 'PRC AUC'],
        metrics_to_plot,
    ):
        if metric_name == 'ROC AUC':
            metric_values = [0.5 if x is None else x for x in metric_values]
        if metric_name == 'PRC AUC':
            metric_values = [0.0 if x is None else x for x in metric_values]
        plot_metrics(metric_names=label_strings, metric_values=metric_values, title=metric_name,
                     ylabel="Label", xlabel=metric_name, append_average_to_title=True, horizontal=True,
                     show_metrics_above_bars=True, draw_grid=True, figsize=figsize, sort_metrics=True)
        
def export_generated_reports_to_txt(gen_reports_filepath):
    data = load_pickle(gen_reports_filepath)
    gen_reports = data['gen_reports']
    print(f'len(gen_reports) = {len(gen_reports)}')
    save_path = f'{gen_reports_filepath}.txt' # add .txt extension
    with open(save_path, 'w') as f:
        for gen_report in gen_reports:
            gen_report = ' '.join(gen_report.split()) # remove extra spaces
            f.write(gen_report + '\n')
    print(f'Saved generated reports to {save_path}')

def evaluate(
    checkpoint_folder_path,
    num_workers,
    max_images_per_batch,
    max_facts_per_image,
    device,
    eval_mode,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
    fact_embedding_device,
    chexpert_labels_filepath,
    chest_imagenome_label_names_filepath,
    chest_imagenome_image_id_to_labels_filepath,
    mimiccxr_report_fact_nli_integrated_data_filepath,
    tune_thresholds,
    max_processes_for_chexpert_labeler,
    background_findings_and_impression_per_report_filepath,
    use_alternative_chexpert_template,
    eval_chexpert_only,
    section_mode,
    dicom_id_to_pos_neg_facts_filepath,
    f1_threshold,
    label_based_json_reports_filepath,
    json_to_gpt_reports_jsonl_filepath,
    mimiccxr_interpret_cxr_challenge_split_filepath,
    chexpert_interpret_cxr_challenge_split_filepath,
    iuxray_interpret_cxr_challenge_split_filepath,
):
    print_blue('----- Evaluating model -----', bold=True)

    if checkpoint_folder_path is None:
        model_kwargs = None
        val_image_transform_kwargs = None
    else:
        metadata = load_metadata(checkpoint_folder_path)
        model_kwargs = metadata['model_kwargs']
        val_image_transform_kwargs = metadata['val_image_transform_kwargs']

    _evaluate_model(
        checkpoint_folder_path=checkpoint_folder_path,
        model_kwargs=model_kwargs,
        val_image_transform_kwargs=val_image_transform_kwargs,
        max_images_per_batch=max_images_per_batch,
        max_facts_per_image=max_facts_per_image,
        num_workers=num_workers,
        device=device,
        eval_mode=eval_mode,
        fact_embedding_model_name=fact_embedding_model_name,
        fact_embedding_model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        fact_embedding_batch_size=fact_embedding_batch_size,
        fact_embedding_num_workers=fact_embedding_num_workers,
        fact_embedding_device=fact_embedding_device,
        chexpert_labels_filepath=chexpert_labels_filepath,
        chest_imagenome_label_names_filepath=chest_imagenome_label_names_filepath,
        chest_imagenome_image_id_to_labels_filepath=chest_imagenome_image_id_to_labels_filepath,
        mimiccxr_report_fact_nli_integrated_data_filepath=mimiccxr_report_fact_nli_integrated_data_filepath,
        tune_thresholds=tune_thresholds,
        max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
        background_findings_and_impression_per_report_filepath=background_findings_and_impression_per_report_filepath,
        use_alternative_chexpert_template=use_alternative_chexpert_template,
        eval_chexpert_only=eval_chexpert_only,
        section_mode=section_mode,
        dicom_id_to_pos_neg_facts_filepath=dicom_id_to_pos_neg_facts_filepath,
        f1_threshold=f1_threshold,
        label_based_json_reports_filepath=label_based_json_reports_filepath,
        json_to_gpt_reports_jsonl_filepath=json_to_gpt_reports_jsonl_filepath,
        mimiccxr_interpret_cxr_challenge_split_filepath=mimiccxr_interpret_cxr_challenge_split_filepath,
        chexpert_interpret_cxr_challenge_split_filepath=chexpert_interpret_cxr_challenge_split_filepath,
        iuxray_interpret_cxr_challenge_split_filepath=iuxray_interpret_cxr_challenge_split_filepath,
    )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)