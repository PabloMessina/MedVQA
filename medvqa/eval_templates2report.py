import argparse
import os
import random
import numpy as np
from datetime import datetime

import pandas as pd

# from medvqa.datasets.mimiccxr import (
#     get_imageId2PartPatientStudy,
#     get_imageId2reportId,
#     get_mimiccxr_medium_image_path,
#     get_split2imageIds,
# )
from medvqa.datasets.seq2seq.seq2seq_dataset_management import _probs_and_preds_to_input_text
from medvqa.evaluation.report_generation import compute_report_level_metrics
from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.utils.common import (
    INTERPRET_CXR_TEST_HIDDEN_CSV_PATH,
    INTERPRET_CXR_TEST_HIDDEN_IMAGES_FOLDER_PATH,
    INTERPRET_CXR_TEST_PUBLIC_CSV_PATH,
    INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH,
    parsed_args_to_dict,
)
from medvqa.utils.constants import CHEXBERT_LABELS, CHEXPERT_LABELS
from medvqa.utils.files import  get_results_folder_path, load_pickle, save_pickle, save_txt
from medvqa.utils.logging import print_blue, print_bold, print_magenta, print_orange
from medvqa.utils.metrics import f1_between_dicts

class _EvalModes:
    INTERPRET_CXR_TEST_PUBLIC__FINDINGS = 'interpret_cxr_test_public__findings'
    INTERPRET_CXR_TEST_PUBLIC__IMPRESSION = 'interpret_cxr_test_public__impression'
    INTERPRET_CXR_TEST_HIDDEN__FINDINGS = 'interpret_cxr_test_hidden__findings'
    INTERPRET_CXR_TEST_HIDDEN__IMPRESSION = 'interpret_cxr_test_hidden__impression'

    @staticmethod
    def choices():
        return [
            _EvalModes.INTERPRET_CXR_TEST_PUBLIC__FINDINGS,
            _EvalModes.INTERPRET_CXR_TEST_PUBLIC__IMPRESSION,
            _EvalModes.INTERPRET_CXR_TEST_HIDDEN__FINDINGS,
            _EvalModes.INTERPRET_CXR_TEST_HIDDEN__IMPRESSION,
        ]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_folder_path_findings', type=str)
    parser.add_argument('--checkpoint_folder_path_impression', type=str)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_processes_for_chexpert_labeler', type=int, default=14)
    parser.add_argument('--eval_mode', type=str, required=True, choices=_EvalModes.choices())
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search')
    parser.add_argument('--findings_max_length', type=int, default=200)
    parser.add_argument('--impression_max_length', type=int, default=200)
    parser.add_argument('--background_findings_and_impression_per_report_filepath', type=str)
    parser.add_argument('--interpret_cxr__label_based_predictions_filepath', type=str)
    parser.add_argument('--first_k_classes', type=int, default=None)
    return parser.parse_args(args=args)

def _load_label_based_predictions(filepath, first_k_classes):
    print(f'Loading {filepath}...')
    tmp = load_pickle(filepath)
    probs_filepath = tmp['probs_filepath']
    thresholds = tmp['thresholds']
    f1s = tmp['f1s']
    accs = tmp['accs']
    class_names = tmp['class_names']
    print(f'Loading {probs_filepath}...')
    tmp2 = load_pickle(probs_filepath)
    probs = tmp2['probs']
    image_paths = tmp2['image_paths']
    image_path_2_idx = { image_path: i for i, image_path in enumerate(image_paths) }
    assert len(image_paths) == len(image_path_2_idx) # unique

    # Sort by hybrid score
    hybrid_score = f1s + accs
    sorted_class_idxs = np.argsort(hybrid_score)[::-1]
    class_names = [class_names[i] for i in sorted_class_idxs]
    f1s = f1s[sorted_class_idxs]
    accs = accs[sorted_class_idxs]
    probs = probs[:, sorted_class_idxs]
    binary = (probs > thresholds).astype(int)

    # Keep only first_k_classes
    if first_k_classes is not None:
        probs = probs[:, :first_k_classes]
        binary = binary[:, :first_k_classes]
        thresholds = thresholds[:first_k_classes]
        f1s = f1s[:first_k_classes]
        accs = accs[:first_k_classes]
        class_names = class_names[:first_k_classes]

    print('Class names:')
    for i, class_name in enumerate(class_names):
        print(f'{i + 1}: {class_name}, f1: {f1s[i]:.3f}, acc: {accs[i]:.3f}')

    return probs, binary, class_names, image_path_2_idx

# def _get_ground_truth_reports(background_findings_and_impression_per_report_filepath, report_idxs):
#     bfipr = get_cached_json_file(background_findings_and_impression_per_report_filepath)
#     gt_reports = []
#     for ridx in report_idxs:
#         findings = bfipr[ridx]['findings']
#         impression = bfipr[ridx]['impression']
#         if findings and impression:
#             if findings[-1] == '.':
#                 report = findings + ' ' + impression
#             else:
#                 report = findings + '. ' + impression
#         elif findings:
#             report = findings
#         elif impression:
#             report = impression
#         else:
#             report = ''
#         gt_reports.append(report)
#     return gt_reports

def _compute_and_save_report_gen_metrics(
        gt_reports, gen_reports, dataset_name, results_folder_path, max_processes, strings):
    metrics = compute_report_level_metrics(gt_reports, gen_reports, max_processes=max_processes)
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_report_gen_metrics({",".join(strings)}).pkl')
    save_pickle(metrics, save_path)
    print_bold(f'Report generation metrics successfully saved to {save_path}')
    return metrics

def _save_gen_reports(input_texts, gen_reports, image_paths_per_report, dataset_name, results_folder_path,
        strings, gt_reports=None, report_idxs=None, findings_checkpoint_path=None, impression_checkpoint_path=None,
        save_txt_for_challenge_submission=False, save_input_texts_for_challenge_submission=False):
    assert len(gen_reports) == len(input_texts)
    assert len(gen_reports) == len(image_paths_per_report)
    if gt_reports is not None:
        assert len(gen_reports) == len(gt_reports)
    if report_idxs is not None:
        assert len(gen_reports) == len(report_idxs)
    save_path = os.path.join(results_folder_path,
        f'{dataset_name}_gen_reports({",".join(strings)}).pkl')
    output = {
        'input_texts': input_texts,
        'gen_reports': gen_reports,
        'image_paths_per_report': image_paths_per_report,
    }
    if gt_reports is not None:
        output['gt_reports'] = gt_reports
    if report_idxs is not None:
        output['report_idxs'] = report_idxs
    if findings_checkpoint_path is not None:
        output['findings_checkpoint_path'] = findings_checkpoint_path
    if impression_checkpoint_path is not None:
        output['impression_checkpoint_path'] = impression_checkpoint_path
    save_pickle(output, save_path)
    print_bold(f'Generated reports successfully saved to {save_path}')

    if save_txt_for_challenge_submission:
        txt_path = os.path.join(results_folder_path,
            f'{dataset_name}_gen_reports({",".join(strings)}).txt')
        save_txt(gen_reports, txt_path)
        print_blue(f'Generated reports in txt format for challenge submission successfully saved to {txt_path}', bold=True)
    if save_input_texts_for_challenge_submission:
        txt_path = os.path.join(results_folder_path,
            f'{dataset_name}_input_texts({",".join(strings)}).txt')
        save_txt(input_texts, txt_path)
        print_orange(f'Input texts in txt format for challenge submission successfully saved to {txt_path}', bold=True)


def _run_test_public_evaluation(checkpoint_folder_path, interpret_cxr__label_based_predictions_filepath, first_k_classes,
                                report_section, device, batch_size, num_workers, max_length, num_beams,
                                max_processes_for_chexpert_labeler):
    assert report_section in ['findings', 'impression']
    assert checkpoint_folder_path is not None
    assert interpret_cxr__label_based_predictions_filepath is not None

    probs, binary, class_names, image_path_2_idx =\
        _load_label_based_predictions(interpret_cxr__label_based_predictions_filepath, first_k_classes)
    
    df = pd.read_csv(INTERPRET_CXR_TEST_PUBLIC_CSV_PATH)
    df = df.replace(np.nan, '', regex=True) # replace nan with empty string
    public_test_input_texts = []
    public_test_gt_texts = []
    public_test_image_paths = []

    for images_path, section_text in df[['images_path', report_section]].values:
        images_path = eval(images_path) # convert string to list
        image_idxs = []
        actual_image_paths = []
        for image_path in images_path:
            actual_image_path = os.path.join(INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH, os.path.basename(image_path))
            image_idx = image_path_2_idx[actual_image_path]
            image_idxs.append(image_idx)
            actual_image_paths.append(actual_image_path)
        section_text = ' '.join([x for x in section_text.split() if x]) # remove whitespace
        section_text = section_text.lower() # convert to lowercase
        if len(section_text) == 0:
            continue
        public_test_gt_texts.append(section_text)
        # Build input text
        image_probs = probs[image_idxs]
        image_preds = binary[image_idxs]
        input_text = _probs_and_preds_to_input_text(class_names, image_probs, image_preds)
        public_test_input_texts.append(input_text)
        # Save image paths
        public_test_image_paths.append(actual_image_paths)

    print(f'Number of reports: {len(public_test_input_texts)}')
    print(f'Number of images: {len(public_test_image_paths)}')

    # Generate section using model
    print_bold(f'Generating {report_section}...')
    public_test_gen_texts, unprocessed_sentences, checkpoint_path = apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=checkpoint_folder_path,
        sentences=public_test_input_texts,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        num_beams=num_beams,
        postprocess_input_output_func=lambda _, output: output,
        save_outputs=False,
        return_checkpoint_path=True,
    )
    assert len(public_test_gen_texts) == len(public_test_input_texts)
    assert len(unprocessed_sentences) == 0
    
    # Print some examples
    print_bold('Examples:')
    for i in random.sample(range(len(public_test_input_texts)), 3):
        print('-' * 50)
        print_bold('Image paths:')
        print(public_test_image_paths[i])
        print_bold('Input:')
        print_magenta(public_test_input_texts[i])
        print_bold('Gen:')
        print_magenta(public_test_gen_texts[i], bold=True)
        print_bold('GT:')
        print_orange(public_test_gt_texts[i], bold=True)

    # Get modification time in human-readable format for each checkpoint
    modification_time = os.path.getmtime(checkpoint_path)
    modification_time = datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %H:%M:%S')
    
    # Compute and save metrics
    results_folder_path = get_results_folder_path(checkpoint_folder_path) # Use checkpoint folder
    strings=[
        report_section,
        modification_time,
    ]
    if first_k_classes is not None:
        strings.append(f'first{first_k_classes}')
    _compute_and_save_report_gen_metrics(
        gt_reports=public_test_gt_texts,
        gen_reports=public_test_gen_texts,
        dataset_name='interpret_cxr_test_public',
        results_folder_path=results_folder_path,
        max_processes=max_processes_for_chexpert_labeler,
        strings=strings,
    )

    # Save generated reports
    _save_gen_reports(
        input_texts=public_test_input_texts,
        gen_reports=public_test_gen_texts,
        gt_reports=public_test_gt_texts,
        image_paths_per_report=public_test_image_paths,
        dataset_name='interpret_cxr_test_public',
        results_folder_path=results_folder_path,
        findings_checkpoint_path=checkpoint_path if report_section == 'findings' else None,
        impression_checkpoint_path=checkpoint_path if report_section == 'impression' else None,
        strings=strings,
        save_txt_for_challenge_submission=True,
    )

def _run_test_hidden_evaluation(checkpoint_folder_path, interpret_cxr__label_based_predictions_filepath, first_k_classes,
                                report_section, device, batch_size, num_workers, max_length, num_beams):
    assert checkpoint_folder_path is not None
    assert interpret_cxr__label_based_predictions_filepath is not None
    assert report_section in ['findings', 'impression']

    probs, binary, class_names, image_path_2_idx =\
        _load_label_based_predictions(interpret_cxr__label_based_predictions_filepath, first_k_classes)
    
    df = pd.read_csv(INTERPRET_CXR_TEST_HIDDEN_CSV_PATH)
    df = df.replace(np.nan, '', regex=True) # replace nan with empty string
    test_hidden_input_texts = []
    test_hidden_image_paths = []

    for images_path, section_text in df[['images_path', report_section]].values:
        images_path = eval(images_path) # convert string to list
        image_idxs = []
        actual_image_paths = []
        for image_path in images_path:
            actual_image_path = os.path.join(INTERPRET_CXR_TEST_HIDDEN_IMAGES_FOLDER_PATH, os.path.basename(image_path))
            image_idx = image_path_2_idx[actual_image_path]
            image_idxs.append(image_idx)
            actual_image_paths.append(actual_image_path)
        if len(section_text) == 0:
            continue
        # Build input text
        image_probs = probs[image_idxs]
        image_preds = binary[image_idxs]
        input_text = _probs_and_preds_to_input_text(class_names, image_probs, image_preds)
        test_hidden_input_texts.append(input_text)
        # Save image paths
        test_hidden_image_paths.append(actual_image_paths)

    print(f'Number of reports: {len(test_hidden_input_texts)}')
    print(f'Number of images: {len(test_hidden_image_paths)}')

    # Generate report section using model
    print_bold(f'Generating {report_section}...')
    test_hidden_gen_texts, unprocessed_sentences, checkpoint_path = apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=checkpoint_folder_path,
        sentences=test_hidden_input_texts,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        num_beams=num_beams,
        postprocess_input_output_func=lambda _, output: output,
        save_outputs=False,
        return_checkpoint_path=True,
    )
    assert len(test_hidden_gen_texts) == len(test_hidden_input_texts)
    assert len(unprocessed_sentences) == 0
    
    # Print some examples
    print_bold('Examples:')
    for i in random.sample(range(len(test_hidden_input_texts)), 3):
        print('-' * 50)
        print_bold('Image paths:')
        print(test_hidden_image_paths[i])
        print_bold('Input:')
        print_magenta(test_hidden_input_texts[i])
        print_bold('Gen:')
        print_magenta(test_hidden_gen_texts[i], bold=True)

    # Get modification time in human-readable format for each checkpoint
    modification_time = os.path.getmtime(checkpoint_path)
    modification_time = datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %H:%M:%S')
    
    results_folder_path = get_results_folder_path(checkpoint_folder_path) # Use checkpoint folder
    strings=[
        report_section,
        modification_time,
    ]
    if first_k_classes is not None:
        strings.append(f'first{first_k_classes}')

    # Save generated reports
    _save_gen_reports(
        input_texts=test_hidden_input_texts,
        gen_reports=test_hidden_gen_texts,
        image_paths_per_report=test_hidden_image_paths,
        dataset_name='interpret_cxr_test_hidden',
        results_folder_path=results_folder_path,
        findings_checkpoint_path=checkpoint_path if report_section == 'findings' else None,
        impression_checkpoint_path=checkpoint_path if report_section == 'impression' else None,
        strings=strings,
        save_txt_for_challenge_submission=True,
        save_input_texts_for_challenge_submission=True,
    )

def evaluate(
        checkpoint_folder_path_findings,
        checkpoint_folder_path_impression,
        background_findings_and_impression_per_report_filepath,
        interpret_cxr__label_based_predictions_filepath,
        first_k_classes,
        batch_size,
        num_workers,
        device,
        eval_mode,
        num_beams,
        findings_max_length,
        impression_max_length,
        max_processes_for_chexpert_labeler,
        ):
    
    # if eval_mode == _EvalModes.MIMICCXR_TEST_SET:
    #     assert checkpoint_folder_path_findings is not None
    #     assert checkpoint_folder_path_impression is not None
    #     assert background_findings_and_impression_per_report_filepath is not None
    #     assert interpret_cxr__label_based_predictions_filepath is not None

    #     probs, binary, class_names, image_path_2_idx =\
    #         _load_label_based_predictions(interpret_cxr__label_based_predictions_filepath, first_k_classes)

    #     # Load image paths
    #     imageId2reportId = get_imageId2reportId()
    #     imageId2PartPatientStudy = get_imageId2PartPatientStudy()
    #     test_dicom_ids = get_split2imageIds()['test']
    #     test_report_idxs = [imageId2reportId[x] for x in test_dicom_ids]
    #     test_image_paths = [get_mimiccxr_medium_image_path(*imageId2PartPatientStudy[x], x) for x in test_dicom_ids]
    #     ridx2idxs = { ridx: [] for ridx in test_report_idxs }
    #     for i, ridx in enumerate(test_report_idxs):
    #         ridx2idxs[ridx].append(i)
    #     unique_ridxs = list(ridx2idxs.keys())
    #     print(f'Number of reports: {len(unique_ridxs)}')
    #     print(f'Number of images: {len(test_image_paths)}')

    #     # Generate inputs for models
    #     input_texts = []
    #     for ridx in unique_ridxs:
    #         idxs = ridx2idxs[ridx]
    #         image_idxs = [image_path_2_idx[test_image_paths[i]] for i in idxs]
    #         image_probs = probs[image_idxs]
    #         image_binary = binary[image_idxs]
    #         input_text = _probs_and_preds_to_input_text(class_names, image_probs, image_binary)
    #         input_texts.append(input_text)

    #     # Collect ground-truth reports
    #     gt_reports = _get_ground_truth_reports(background_findings_and_impression_per_report_filepath, unique_ridxs)

    #     # Generate findings using findings model
    #     print_bold('Generating findings...')
    #     public_test_gen_findings, unprocessed_sentences, checkpoint_path_findings = apply_seq2seq_model_to_sentences(
    #         checkpoint_folder_path=checkpoint_folder_path_findings,
    #         sentences=input_texts,
    #         device=device,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         max_length=findings_max_length,
    #         num_beams=num_beams,
    #         postprocess_input_output_func=lambda _, output: output,
    #         save_outputs=False,
    #         return_checkpoint_path=True,
    #     )
    #     assert len(public_test_gen_findings) == len(input_texts)
    #     assert len(unprocessed_sentences) == 0

    #     # Generate impression using impression model
    #     print_bold('Generating impression...')
    #     gen_impression, unprocessed_sentences, checkpoint_path_impression = apply_seq2seq_model_to_sentences(
    #         checkpoint_folder_path=checkpoint_folder_path_impression,
    #         sentences=input_texts,
    #         device=device,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         max_length=impression_max_length,
    #         num_beams=num_beams,
    #         postprocess_input_output_func=lambda _, output: output,
    #         save_outputs=False,
    #         return_checkpoint_path=True,
    #     )
    #     assert len(gen_impression) == len(input_texts)
    #     assert len(unprocessed_sentences) == 0

    #     # Combine findings and impression
    #     gen_reports = []
    #     for findings, impression in zip(public_test_gen_findings, gen_impression):
    #         if findings and impression:
    #             if findings[-1] == '.':
    #                 report = findings + ' ' + impression
    #             else:
    #                 report = findings + '. ' + impression
    #         elif findings:
    #             report = findings
    #         elif impression:
    #             report = impression
    #         else:
    #             report = ''
    #         gen_reports.append(report)
        
    #     # Print some examples
    #     print_bold('Examples:')
    #     for i in random.sample(range(len(unique_ridxs)), 3):
    #         print(f'ridx: {unique_ridxs[i]}')
    #         print_bold('Input:')
    #         print_magenta(input_texts[i])
    #         print_bold('Gen:')
    #         print_magenta(gen_reports[i], bold=True)
    #         print_bold('GT:')
    #         print_orange(gt_reports[i], bold=True)

    #     # Get modification time in human-readable format for each checkpoint
    #     findings_modification_time = os.path.getmtime(checkpoint_path_findings)
    #     findings_modification_time = datetime.fromtimestamp(findings_modification_time).strftime('%Y-%m-%d %H:%M:%S')
    #     impression_modification_time = os.path.getmtime(checkpoint_path_impression)
    #     impression_modification_time = datetime.fromtimestamp(impression_modification_time).strftime('%Y-%m-%d %H:%M:%S')
        
    #     # Compute and save metrics
    #     results_folder_path = get_results_folder_path(checkpoint_folder_path_findings) # Use findings folder
    #     strings=[
    #         'findings+impression',
    #         f'fmt{findings_modification_time}',
    #         f'imt{impression_modification_time}',
    #     ]
    #     if first_k_classes is not None:
    #         strings.append(f'first{first_k_classes}')
    #     _compute_and_save_report_gen_metrics(
    #         gt_reports=gt_reports,
    #         gen_reports=gen_reports,
    #         dataset_name='mimiccxr_test_set',
    #         results_folder_path=results_folder_path,
    #         max_processes=max_processes_for_chexpert_labeler,
    #         strings=strings,
    #     )

    #     # Save generated reports
    #     image_paths_per_report = []
    #     for ridx in unique_ridxs:
    #         idxs = ridx2idxs[ridx]
    #         image_paths_per_report.append([test_image_paths[i] for i in idxs])
    #     _save_gen_reports(
    #         input_texts=input_texts,
    #         gen_reports=gen_reports,
    #         image_paths_per_report=image_paths_per_report,
    #         report_idxs=unique_ridxs,
    #         dataset_name='mimiccxr_test_set',
    #         results_folder_path=results_folder_path,
    #         findings_checkpoint_path=checkpoint_path_findings,
    #         impression_checkpoint_path=checkpoint_path_impression,            
    #         strings=strings,
    #     )
        
    if eval_mode == _EvalModes.INTERPRET_CXR_TEST_PUBLIC__FINDINGS:

        _run_test_public_evaluation(
            checkpoint_folder_path=checkpoint_folder_path_findings,
            interpret_cxr__label_based_predictions_filepath=interpret_cxr__label_based_predictions_filepath,
            first_k_classes=first_k_classes,
            report_section='findings',
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=findings_max_length,
            num_beams=num_beams,
            max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
        )

    elif eval_mode == _EvalModes.INTERPRET_CXR_TEST_PUBLIC__IMPRESSION:

        _run_test_public_evaluation(
            checkpoint_folder_path=checkpoint_folder_path_impression,
            interpret_cxr__label_based_predictions_filepath=interpret_cxr__label_based_predictions_filepath,
            first_k_classes=first_k_classes,
            report_section='impression',
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=impression_max_length,
            num_beams=num_beams,
            max_processes_for_chexpert_labeler=max_processes_for_chexpert_labeler,
        )

    elif eval_mode == _EvalModes.INTERPRET_CXR_TEST_HIDDEN__FINDINGS:

        _run_test_hidden_evaluation(
            checkpoint_folder_path=checkpoint_folder_path_findings,
            interpret_cxr__label_based_predictions_filepath=interpret_cxr__label_based_predictions_filepath,
            first_k_classes=first_k_classes,
            report_section='findings',
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=findings_max_length,
            num_beams=num_beams,
        )

    elif eval_mode == _EvalModes.INTERPRET_CXR_TEST_HIDDEN__IMPRESSION:

        _run_test_hidden_evaluation(
            checkpoint_folder_path=checkpoint_folder_path_impression,
            interpret_cxr__label_based_predictions_filepath=interpret_cxr__label_based_predictions_filepath,
            first_k_classes=first_k_classes,
            report_section='impression',
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=impression_max_length,
            num_beams=num_beams,
        )

    else:
        raise ValueError(f'Unknown eval_mode: {eval_mode}')

class ReportGenerationVisualizer:
    def __init__(self, metrics_filepath, gen_reports_filepath):
        self.metrics = load_pickle(metrics_filepath)
        tmp = load_pickle(gen_reports_filepath)
        self.image_paths_per_report = tmp['image_paths_per_report']
        self.input_texts = tmp['input_texts']
        self.gen_reports = tmp['gen_reports']
        self.gt_reports = tmp['gt_reports']

    def visualize(self, idx):
        print_bold(f'Number of images: {len(self.image_paths_per_report[idx])}')
        for i, image_path in enumerate(self.image_paths_per_report[idx]):
            print(f'{i + 1}: {image_path}')
        # plot a grid of images, with filepaths as captions
        import matplotlib.pyplot as plt
        from PIL import Image
        from matplotlib import gridspec
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2)
        for i, image_path in enumerate(self.image_paths_per_report[idx]):
            ax = fig.add_subplot(gs[i])
            ax.imshow(Image.open(image_path))
            ax.set_title(f'{i + 1}: {os.path.basename(image_path)}')
            ax.axis('off')
        plt.show()
        
        print_bold('Input:')
        print_magenta(self.input_texts[idx])
        
        print_bold('Gen:')
        print_magenta(self.gen_reports[idx], bold=True)
        
        print_bold('GT:')
        print_orange(self.gt_reports[idx], bold=True)

        # Print metrics
        print_bold('Metrics:')
        for k in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'ciderD']:
            print(f'{k}: {self.metrics[k][1][idx]:.3f}')
        for k in ['rougeL', 'meteor']:
            print(f'{k}: {self.metrics[k][idx]:.3f}')
        chexpert_labels_gt = self.metrics['chexpert_labels_gt'][idx]
        chexpert_labels_gen = self.metrics['chexpert_labels_gen'][idx]
        chexbert_labels_gt = self.metrics['chexbert_labels_gt'][idx]
        chexbert_labels_gen = self.metrics['chexbert_labels_gen'][idx]
        radgraph_labels_gt = self.metrics['radgraph_labels_gt'][idx]
        radgraph_labels_gen = self.metrics['radgraph_labels_gen'][idx]
        print_bold('Chexpert labels:')
        print(f'Accuracy: {(chexpert_labels_gt == chexpert_labels_gen).mean():.3f}')
        print(f'GT: {[CHEXPERT_LABELS[i] for i, x in enumerate(chexpert_labels_gt) if x]}')
        print(f'Gen: {[CHEXPERT_LABELS[i] for i, x in enumerate(chexpert_labels_gen) if x]}')
        print_bold('Chexbert labels:')
        print(f'Accuracy: {(chexbert_labels_gt == chexbert_labels_gen).mean():.3f}')
        print(f'GT: {[CHEXBERT_LABELS[i] for i, x in enumerate(chexbert_labels_gt) if x]}')
        print(f'Gen: {[CHEXBERT_LABELS[i] for i, x in enumerate(chexbert_labels_gen) if x]}')
        print_bold('Radgraph F1:')
        print(f1_between_dicts(radgraph_labels_gt, radgraph_labels_gen))

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)