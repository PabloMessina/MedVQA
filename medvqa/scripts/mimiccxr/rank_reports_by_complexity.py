import argparse
import os
import numpy as np
import multiprocessing as mp

from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix,
    load_chest_imagenome_label_names
)
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.utils.constants import CHEXPERT_LABELS
from medvqa.utils.files import load_pickle, load_json, save_pickle
from medvqa.utils.logging import print_bold, print_magenta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports-filename', type=str, required=True)
    parser.add_argument('--chexpert-labels-filename', type=str, required=True)
    parser.add_argument('--chest-imagenome-labels-filename', type=str, required=True)
    parser.add_argument('--chest-imagenome-label-names-filename', type=str, required=True)
    parser.add_argument('--chexpert-weight', type=float, default=1.0)
    parser.add_argument('--chest-imagenome-weight', type=float, default=1.0)
    parser.add_argument('--report-length-weight', type=float, default=1.0)
    parser.add_argument('--n-workers', type=int, default=10)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    # Load reports
    reports_path = os.path.join(MIMICCXR_CACHE_DIR, args.reports_filename)
    print_bold(f'Loading reports from {reports_path}...')
    reports = load_json(reports_path)
    n_reports = len(reports)
    print(f'Number of reports: {n_reports}')

    # Load CheXpert labels
    chexpert_labels_path = os.path.join(MIMICCXR_CACHE_DIR, args.chexpert_labels_filename)
    print_bold(f'Loading CheXpert labels from {chexpert_labels_path}...')
    chexpert_labels = load_pickle(chexpert_labels_path)
    chexpert_labels = np.array(chexpert_labels)
    print(f'CheXpert labels shape: {chexpert_labels.shape}')

    # Load Chest ImaGenome labels
    print_bold(f'Loading Chest ImaGenome labels from {args.chest_imagenome_labels_filename}...')
    _, chest_imagenome_labels = load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix(
        args.chest_imagenome_labels_filename)
    print(f'Chest ImaGenome labels shape: {chest_imagenome_labels.shape}')
    chest_imagenome_label_names = load_chest_imagenome_label_names(args.chest_imagenome_label_names_filename)
    assert len(chest_imagenome_label_names) == chest_imagenome_labels.shape[1], \
        'Number of Chest ImaGenome label names does not match number of labels.'

    # Check that the number of reports is the same as the number of labels
    assert chexpert_labels.shape[0] == chest_imagenome_labels.shape[0] == n_reports, \
        (f'Number of reports ({n_reports}), CheXpert labels ({len(chexpert_labels)}),'
        f' and Chest ImaGenome labels ({len(chest_imagenome_labels)}) do not match.')

    # Concatenate the labels
    print_bold('Concatenating the labels...')
    labels = np.concatenate((chexpert_labels, chest_imagenome_labels), axis=1)
    print(f'Labels shape: {labels.shape}')

    # Count the number of reports with each label
    print_bold('Counting label occurrences...')
    n_labels = labels.shape[1]
    label_count = labels.sum(axis=0)
    
    # Assign reports to a bins based on their labels
    print_bold(f'Finding bins for reports in parallel with {args.n_workers} workers...')
    global_label_indices = [i for i in range(chexpert_labels.shape[1])]
    for i in range(chest_imagenome_labels.shape[1]):
        if len(chest_imagenome_label_names[i]) == 2:
            global_label_indices.append(i + chexpert_labels.shape[1])
    print(f'len(global_label_indices): {len(global_label_indices)}')
    def _collect_label_indices(label_idx):
        if label_idx == -1: # return all indices where there is no label
            return np.where(labels.sum(axis=1) == 0)[0].tolist()
        return np.where(labels[:, label_idx] == 1)[0].tolist()
    with mp.Pool(args.n_workers) as pool:
        label_bins = pool.map(_collect_label_indices, global_label_indices + [-1])
    
    # Sort each bin by complexity
    print_bold('Sorting each bin by complexity...')
    chexp_mean_sum = chexpert_labels[:, 1:].sum(axis=1).mean() # Ignore the first label (No Finding)
    chest_imag_mean_sum = chest_imagenome_labels.sum(axis=1).mean()
    repor_mean_length = np.mean([len(x['findings'] + x['impression']) for x in reports])
    print(f'Mean CheXpert label sum: {chexp_mean_sum}')
    print(f'Mean Chest ImaGenome label sum: {chest_imag_mean_sum}')
    print(f'Mean report length: {repor_mean_length}')
    chexp_weight = args.chexpert_weight
    chest_weight = args.chest_imagenome_weight
    repor_weight = args.report_length_weight
    def _complexity(i):
        chexp_score = chexpert_labels[i][1:].sum() / chexp_mean_sum
        chest_score = chest_imagenome_labels[i][1:].sum() / chest_imag_mean_sum
        repor_score = (len(reports[i]['findings'] + reports[i]['impression'])) / repor_mean_length
        return chexp_weight * chexp_score + chest_weight * chest_score + repor_weight * repor_score
    for label_bin in label_bins:
        label_bin.sort(key=_complexity, reverse=True)

    if args.verbose:
        # Print the number of reports in each bin and the bin label names
        print_bold('Bin sizes and label names:')
        sorted_label_bin_idxs = sorted(range(len(label_bins)), key=lambda x: len(label_bins[x]))
        label_names = CHEXPERT_LABELS + [str(x) for x in chest_imagenome_label_names if len(x) == 2] \
            + ['No Labels']
        assert len(label_names) == len(label_bins)
        for i, idx in enumerate(sorted_label_bin_idxs):
            print(f'{i}: {len(label_bins[idx])} {label_names[idx]}')

    # Create ranked report indices, by taking the last report from each bin, then the second last, etc.
    # until all reports are used. Bin are iterated through in order of increasing label count.
    print_bold('Creating ranked report indices...')
    sorted_label_bins = sorted(label_bins, key=lambda x: len(x))
    ranked_report_indices = [None] * n_reports
    seen = [False] * n_reports
    first_bin_idx = 0
    idx = 0
    for h in range(max(len(x) for x in sorted_label_bins)):
        while len(sorted_label_bins[first_bin_idx]) <= h:
            first_bin_idx += 1
        for i in range(first_bin_idx, len(sorted_label_bins)):
            assert len(sorted_label_bins[i]) > h
            if not seen[sorted_label_bins[i][h]]:
                ranked_report_indices[idx] = sorted_label_bins[i][h]
                seen[sorted_label_bins[i][h]] = True
                idx += 1

    assert all([x is not None for x in ranked_report_indices]), \
        'Not all reports were assigned an index.'
    assert len(set(ranked_report_indices)) == n_reports, \
        'Duplicate report indices were assigned.'

    # Save indices
    save_path = os.path.join(MIMICCXR_CACHE_DIR, f'ranked_report_indices_{args.chexpert_weight}_{args.chest_imagenome_weight}_{args.report_length_weight}.pkl')
    save_pickle(ranked_report_indices, save_path)
    print_magenta(f'Saved ranked report indices to {save_path}', bold=True)