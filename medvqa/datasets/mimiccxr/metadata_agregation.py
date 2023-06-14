import os
import json
from typing import OrderedDict
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_label_names,
    load_chest_imagenome_labels,
    load_scene_graph,
)
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR, get_mimiccxr_large_image_path, load_mimiccxr_reports_detailed_metadata
from medvqa.metrics.medical.chexpert import ChexpertLabeler
from medvqa.utils.constants import CHEXPERT_LABELS
from medvqa.utils.files import get_cached_json_file, get_cached_pickle_file
from medvqa.utils.logging import chest_imagenome_label_array_to_string, chexpert_label_array_to_string, print_bold
import imagesize

import re
w_regex = re.compile(r'\s')

def _clean_intervals(intervals):
    # merge intervals when there is overlap
    intervals = sorted(intervals, key=lambda x: x[0])
    merged_intervals = []
    for i in range(len(intervals)):
        if i == 0:
            merged_intervals.append(intervals[i])
        else:
            if intervals[i][0] <= merged_intervals[-1][1]:
                merged_intervals[-1] = (merged_intervals[-1][0], intervals[i][1])
            else:
                merged_intervals.append(intervals[i])
    return merged_intervals

class MIMICCXR_MetadataAgregator:
    def __init__(self, qa_adapted_reports_filename, chexpert_labels_filename,
                 chest_imagenome_label_names_filename, chest_imagenome_labels_filename):
        
        self.chest_imagenome_labels = load_chest_imagenome_labels(chest_imagenome_labels_filename)
        self.chest_imagenome_label_names = load_chest_imagenome_label_names(chest_imagenome_label_names_filename)
        self.reports_metadata = load_mimiccxr_reports_detailed_metadata(qa_adapted_reports_filename)
        self.qa_adapted_dataset = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, qa_adapted_reports_filename))
        self.chexpert_labels = get_cached_pickle_file(os.path.join(MIMICCXR_CACHE_DIR, chexpert_labels_filename))
        self.chexpert_labeler = ChexpertLabeler(verbose=False)

    def print_metadata(self, rid):
        print_bold("Report ID: ", rid)
        for key in self.reports_metadata:
            print()
            print_bold(key)
            print(self.reports_metadata[key][rid])
        print()
        print_bold("Chexpert Labels:")
        print(chexpert_label_array_to_string(self.chexpert_labels[rid]))
        print(self.chexpert_labels[rid])
        print()
        print_bold("Chest Imagenome Labels:")
        dids = [did for did, _ in self.reports_metadata['dicom_id_view_pos_pairs'][rid] \
                if did in self.chest_imagenome_labels]
        for i in range(1, len(dids)):
            assert all(self.chest_imagenome_labels[dids[0]] == self.chest_imagenome_labels[dids[i]])
        if len(dids) > 0:
            print(chest_imagenome_label_array_to_string(self.chest_imagenome_labels[dids[0]],
                                                        self.chest_imagenome_label_names))
            print(self.chest_imagenome_labels[dids[0]])
        print()
        print_bold('Original report:')
        with open(self.reports_metadata['filepaths'][rid], 'r') as f:
            orig_report = f.read()
        print(orig_report)
        print()
        print_bold('Text grounding for Chexpert Labels:')
        chexp_label2phrases = self._find_text_grounding_for_chexpert_labels(rid, orig_report, include_sentence=True)
        for label in chexp_label2phrases:
            print(label)
            print(chexp_label2phrases[label])
            print()
        if len(dids) > 0:
            chest_label2phrases = self._find_text_grounding_for_chest_imagenome_labels(dids[0], orig_report,
                                                                                       include_sentence=True)
            print()
            print_bold('Text grounding for Chest Imagenome Labels:')
            for label in chest_label2phrases:
                print(label)
                print(chest_label2phrases[label])
                print()

            # find labels in common between the two
            common_labels = set(chexp_label2phrases.keys()) & set(chest_label2phrases.keys())
            if len(common_labels) > 0:
                print()
                print_bold('Common labels:')
                for label in common_labels:
                    print(label)
                    intervals = []
                    for x in chexp_label2phrases[label]:
                        intervals.append((x[1], x[2]))
                    for x in chest_label2phrases[label]:
                        intervals.append((x[1], x[2]))
                    intervals = _clean_intervals(intervals)
                    print(intervals)

    def integrate_metadata(self, rid, save_path=None):
        is_nan = lambda x: x != x
        output = {
            'report_id': rid,
            'report_filepath': self.reports_metadata['filepaths'][rid],
            'split': self.reports_metadata['splits'][rid],
            'part_id': self.reports_metadata['part_ids'][rid],
            'subject_id': self.reports_metadata['subject_ids'][rid],
            'study_id': self.reports_metadata['study_ids'][rid],
            'dicom_id_view_pos_pairs': [ x if not is_nan(x[1]) else (x[0], "UNK") \
                                         for x in self.reports_metadata['dicom_id_view_pos_pairs'][rid] ],
        }
        
        output['original_image_sizes'] = {}
        for did, _ in output['dicom_id_view_pos_pairs']:
            output['original_image_sizes'][did] = imagesize.get(get_mimiccxr_large_image_path(
                output['part_id'], output['subject_id'], output['study_id'], did))
            
        dids = [did for did, _ in self.reports_metadata['dicom_id_view_pos_pairs'][rid] \
                if did in self.chest_imagenome_labels]
        for i in range(1, len(dids)):
            assert all(self.chest_imagenome_labels[dids[0]] == self.chest_imagenome_labels[dids[i]])
        with open(self.reports_metadata['filepaths'][rid], 'r') as f:
            orig_report = f.read()
            output['original_report'] = orig_report
        
        chexp_label2phrases = self._find_text_grounding_for_chexpert_labels(rid, orig_report)
        common_labels = set(chexp_label2phrases.keys())
        output['chexpert_labels'] = {}
        output['chest_imagenome_labels'] = {}
        output['common_labels'] = {}
        if len(dids) > 0:
            chest_label2phrases = self._find_text_grounding_for_chest_imagenome_labels(dids[0], orig_report)
            common_labels &= set(chest_label2phrases.keys())
            if len(common_labels) > 0:
                common_labels = list(common_labels)
                common_labels.sort()
                for label in common_labels:
                    intervals = []
                    for x in chexp_label2phrases[label]:
                        intervals.append(x)
                    for x in chest_label2phrases[label]:
                        intervals.append(x)
                    intervals = _clean_intervals(intervals)
                    output['common_labels'][label] = intervals
            # add chexpert labels
            for label in chexp_label2phrases:
                if label not in common_labels:
                    output['chexpert_labels'][label] = _clean_intervals(chexp_label2phrases[label])
            # add chest imagenome labels
            for label in chest_label2phrases:
                if label not in common_labels:
                    output['chest_imagenome_labels'][label] = _clean_intervals(chest_label2phrases[label])
        else:
            # add chexpert labels
            for label in chexp_label2phrases:
                output['chexpert_labels'][label] = _clean_intervals(chexp_label2phrases[label])
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(output, f, indent=4)
                print(f'Saved to {save_path}')
        return output
    
    def _find_text_grounding_for_chexpert_labels(self, rid, original_report, include_sentence=False):
        report = self.qa_adapted_dataset['reports'][rid]
        matched_sentences = [report['sentences'][idx] for idx in report['matched']]
        fresh_labels = self.chexpert_labeler.get_labels(matched_sentences,
                                                        update_cache_on_disk=False, remove_tmp_files=True)
        assert len(fresh_labels) == len(matched_sentences)
        label2phrases = OrderedDict()
        for i in range(len(fresh_labels)):
            s = matched_sentences[i]
            start, end = self._find_most_likely_matching_interval(s, original_report)
            s_labels = fresh_labels[i]
            for j in range(1, len(s_labels)):
                if s_labels[j] == 1:
                    label_name = CHEXPERT_LABELS[j].lower()
                    if label_name not in label2phrases:
                        label2phrases[label_name] = []
                    if include_sentence:
                        label2phrases[label_name].append((s, start, end))
                    else:
                        label2phrases[label_name].append((start, end))
        return label2phrases

    def _find_most_likely_matching_interval(self, query, text):
        from Levenshtein import distance
        query = query.lower()
        text = text.lower()
        min_dist = 100000
        best_range = None
        for i in range(len(text) - len(query) + 1):
            s = i
            while s > 0 and not w_regex.match(text[s]) and not w_regex.match(text[s-1]):
                s -= 1
            e = i + len(query)
            while e < len(text) and not w_regex.match(text[e-1]) and not w_regex.match(text[e]):
                e += 1
            dist = distance(query, text[s:e])
            if dist < min_dist:
                min_dist = dist
                best_range = (s, e)
        assert best_range is not None
        return best_range

    def _find_text_grounding_for_chest_imagenome_labels(self, dicom_id, original_report, include_sentence=False):
        labels = self.chest_imagenome_labels[dicom_id]
        label_names = [self.chest_imagenome_label_names[i][-1] for i in range(len(labels)) if labels[i] == 1]
        label_names = set(label_names)
        scene_graph = load_scene_graph(dicom_id)
        label2phrases = OrderedDict()
        phrase2indices = {}
        for node in scene_graph['attributes']:
            assert len(node['attributes']) == len(node['phrases'])
            for a, p in zip(node['attributes'], node['phrases']):
                for x in a:
                    category, value, name = x.split('|')
                    if category == 'temporal' or\
                    category == 'severity' or\
                    category == 'laterality':
                        continue
                    value = int(value == 'yes') # convert to 0/1
                    if value == 1 and name in label_names:
                        if p not in phrase2indices:
                            s = original_report.index(p)
                            phrase2indices[p] = (s, s + len(p))
                        if name not in label2phrases:
                            label2phrases[name] = []
                        if include_sentence:
                            label2phrases[name].append((p, phrase2indices[p][0], phrase2indices[p][1]))
                        else:
                            label2phrases[name].append((phrase2indices[p][0], phrase2indices[p][1]))
        for k in label2phrases:
            label2phrases[k] = list(set(label2phrases[k]))
        return label2phrases