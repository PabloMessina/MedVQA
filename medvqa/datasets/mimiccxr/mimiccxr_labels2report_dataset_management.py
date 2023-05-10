import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    get_dicomId2gender,
    load_chest_imagenome_dicom_ids,
    load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix,
    load_chest_imagenome_label_order,
    load_gold_attributes_relations_dicom_ids,
    load_nongold_dicom_ids,
    load_chest_imagenome_label_names,
)
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH
from medvqa.datasets.mimiccxr import (
    MIMICCXR_CACHE_DIR,
    MIMICCXR_ViewModes,
    get_dicom_id_and_orientation_list,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.datasets.mimiccxr.mimiccxr_vision_dataset_management import _create_dataset
from medvqa.utils.files import get_cached_pickle_file, load_json_file, load_pickle
from medvqa.utils.logging import print_bold, print_magenta, print_red

class MIMICCXR_Labels2Report_Dataset(Dataset):
    def __init__(self, indices, reports, report_ids,
                shuffle=False,
                infinite=False,
                # aux task: gender
                use_gender=False,
                genders=None,
                # aux task: chexpert labels
                use_chexpert=False,
                chexpert_labels=None,
                # aux task: chest imagenome labels
                use_chest_imagenome=False,
                chest_imagenome_labels=None,
                # ensemble predictions
                use_ensemble_predictions=False,
                get_ensemble_sigmoid_vector=None,
                dicom_ids=None,
            ):
        self.indices = indices
        self.reports = reports
        self.report_ids = report_ids
        self.use_gender = use_gender
        self.genders = genders
        self.use_chexpert = use_chexpert
        self.chexpert_labels = chexpert_labels
        self.use_chest_imagenome = use_chest_imagenome
        self.chest_imagenome_labels = chest_imagenome_labels
        self.use_ensemble_predictions = use_ensemble_predictions
        self.get_ensemble_sigmoid_vector = get_ensemble_sigmoid_vector
        self.dicom_ids = dicom_ids

        if shuffle:
            random.shuffle(self.indices) # shuffle in place            
        self.infinite = infinite
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self.infinite:
            i = i % len(self.indices)
        idx = self.indices[i]
        rid = self.report_ids[idx]
        output = { 'idx': idx, 'report': self.reports[rid] }
        if self.use_gender:
            output['gender'] = self.genders[idx]
        if self.use_chexpert:
            output['chexpert'] = self.chexpert_labels[rid]
        if self.use_chest_imagenome:
            output['chest_imagenome'] = self.chest_imagenome_labels[rid]
        if self.use_ensemble_predictions:
            output['ensemble_probs'] = self.get_ensemble_sigmoid_vector(self.dicom_ids[idx])
        return output

class MIMICCXR_Labels2ReportTrainer():

    def __init__(self, 
                qa_adapted_reports_filename, tokenizer, batch_size, collate_batch_fn, num_workers,
                train_collate_batch_fn=None,
                eval_collate_batch_fn=None,
                use_test_set=False,
                use_chest_imagenome_label_gold_set=False,
                use_val_set_only=False,
                view_mode=MIMICCXR_ViewModes.ANY_SINGLE,
                use_decent_images_only=False,
                use_gender=False,
                use_chexpert=False,
                chexpert_labels_filename=None,
                use_chest_imagenome=False,
                chest_imagenome_labels_filename=None,
                chest_imagenome_label_names_filename=None,
                balanced_sampling_mode=None,
                balanced_batch_size=None,
                use_ensemble_predictions=False,
                precomputed_sigmoid_paths=None,
                use_hard_predictions=False,
                precomputed_thresholds_paths=None,
                filter_labels=False,
                precomputed_filtered_labels_paths=None,
                reorder_chest_imagenome_labels=False,
            ):
        # Sanity checks
        assert sum([use_test_set, use_val_set_only,
                    use_chest_imagenome_label_gold_set]) <= 1 # at most one of these can be true
        if use_gender:
            assert view_mode == MIMICCXR_ViewModes.CHEST_IMAGENOME

        if use_ensemble_predictions:
            assert precomputed_sigmoid_paths is not None
        if use_hard_predictions:
            assert use_ensemble_predictions
            assert precomputed_thresholds_paths is not None
        if filter_labels:
            assert use_hard_predictions
            assert precomputed_filtered_labels_paths is not None

        if chest_imagenome_label_names_filename is not None:
            self.chest_imagenome_label_names = load_chest_imagenome_label_names(chest_imagenome_label_names_filename)
        elif chest_imagenome_labels_filename is not None:
            chest_imagenome_label_names_filename = chest_imagenome_labels_filename.replace('imageId2labels', 'labels')
            self.chest_imagenome_label_names = load_chest_imagenome_label_names(chest_imagenome_label_names_filename)
        else:
            self.chest_imagenome_label_names = None

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        if collate_batch_fn is None:
            assert train_collate_batch_fn is not None
            assert eval_collate_batch_fn is not None
            self.train_collate_batch_fn = train_collate_batch_fn
            self.eval_collate_batch_fn = eval_collate_batch_fn
        else:
            assert train_collate_batch_fn is None
            assert eval_collate_batch_fn is None
            self.train_collate_batch_fn = collate_batch_fn
            self.eval_collate_batch_fn = collate_batch_fn
        self.num_workers = num_workers
        self.qa_adapted_reports_filename = qa_adapted_reports_filename
        self.use_hard_predictions = use_hard_predictions
        self.filter_labels = filter_labels

        BIG_ENOGUGH = 1000000
        dicom_ids = [None] * BIG_ENOGUGH
        report_ids = [None] * BIG_ENOGUGH
        if use_test_set or use_chest_imagenome_label_gold_set:
            test_indices = []
        else:
            train_indices = []
            val_indices = []
        idx = 0

        if view_mode == MIMICCXR_ViewModes.CHEST_IMAGENOME:
            allowed_train_val_dicom_ids = None
            if use_decent_images_only:
                decent_dicom_ids = set(load_chest_imagenome_dicom_ids(decent_images_only=True))
            if use_test_set:
                if use_decent_images_only:
                    allowed_test_dicom_ids = decent_dicom_ids
                else:
                    allowed_test_dicom_ids = set(load_chest_imagenome_dicom_ids())
            elif use_chest_imagenome_label_gold_set:
                allowed_test_dicom_ids = set(load_gold_attributes_relations_dicom_ids())
                if use_decent_images_only:
                    allowed_test_dicom_ids &= decent_dicom_ids
            else:
                allowed_train_val_dicom_ids = set(load_nongold_dicom_ids())
                if use_decent_images_only:
                    allowed_train_val_dicom_ids &= decent_dicom_ids
        else:
            assert use_decent_images_only is False
            assert use_chest_imagenome_label_gold_set is False
            allowed_train_val_dicom_ids = None
            allowed_test_dicom_ids = None

        allowed_dicom_ids = allowed_train_val_dicom_ids or allowed_test_dicom_ids

        mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata(qa_adapted_reports_filename, exclude_invalid_sentences=True)
        reports = [None] * len(mimiccxr_metadata['reports'])

        max_idx_count = 0

        if use_test_set or use_chest_imagenome_label_gold_set:
            tokenize_func = tokenizer.string2ids
        else:
            tokenize_func = tokenizer.tokenize

        for rid, (report, dicom_id_view_pairs, split) in \
            tqdm(enumerate(zip(
                # mimiccxr_metadata['part_ids'],
                # mimiccxr_metadata['subject_ids'],
                # mimiccxr_metadata['study_ids'],
                mimiccxr_metadata['reports'],
                mimiccxr_metadata['dicom_id_view_pos_pairs'],
                mimiccxr_metadata['splits'])), mininterval=2):

            max_idx_count += len(dicom_id_view_pairs)
            reports[rid] = tokenize_func(report)

            for dicom_id, view in get_dicom_id_and_orientation_list(dicom_id_view_pairs, view_mode, allowed_dicom_ids):
                dicom_ids[idx] = dicom_id
                
                report_ids[idx] = rid
                if use_test_set or use_chest_imagenome_label_gold_set:
                    if use_test_set:
                        if split == 'test':
                            test_indices.append(idx)
                    else:
                        test_indices.append(idx)
                else:
                    if split == 'train':
                        train_indices.append(idx)
                    elif split == 'validate':
                        val_indices.append(idx)
                    elif split == 'test':
                        pass
                    else:
                        raise ValueError(f'Unknown split {split}')
                idx += 1

        print('max_idx_count =', max_idx_count)
        print('actual_idx_count =', idx)
        if idx < max_idx_count:
            print(f'** NOTE: {max_idx_count - idx} images were skipped because they were not in the allowed DICOM IDs')

        # Print a random report to make sure the tokenization is correct
        random_report = random.choice(reports)
        print_bold('Random report:')
        print_magenta(tokenizer.ids2string(random_report), bold=True)
        
        self.reports = reports
        self.dicom_ids = np.array(dicom_ids[:idx])
        self.report_ids = np.array(report_ids[:idx])
        if use_test_set or use_chest_imagenome_label_gold_set:
            self.test_indices = np.array(test_indices)
            print(f'len(self.test_indices) = {len(self.test_indices)}')
        else:
            self.train_indices = np.array(train_indices)
            self.val_indices = np.array(val_indices)
            print(f'len(self.train_indices) = {len(self.train_indices)}')
            print(f'len(self.val_indices) = {len(self.val_indices)}')

        # Optional data to load
        self.use_gender = use_gender
        self.use_chexpert = use_chexpert
        self.use_chest_imagenome = use_chest_imagenome
        self.use_ensemble_predictions = use_ensemble_predictions
        
        if use_chexpert:
            print('Loading CheXpert labels...')
            assert chexpert_labels_filename is not None
            chexpert_labels_path = os.path.join(MIMICCXR_CACHE_DIR, chexpert_labels_filename)
            self.chexpert_labels = get_cached_pickle_file(chexpert_labels_path)
            self.chexpert_labels = np.array(self.chexpert_labels)
        else:
            self.chexpert_labels = None
        
        if use_gender:
            print('Loading Chest Imagenome genders...')
            dicomId2gender = get_dicomId2gender()
            def _get_gender_label(x):
                if x == 'F': return (1, 0)
                if x == 'M': return (0, 1)
                assert np.isnan(x)
                return (0, 0)
            self.genders = np.array([_get_gender_label(dicomId2gender[dicom_id]) for dicom_id in self.dicom_ids])
        else:
            self.genders = None
        
        if use_chest_imagenome:
            print('Loading Chest Imagenome labels...')
            assert chest_imagenome_labels_filename is not None
            assert chest_imagenome_labels_filename != 'gold_imageId2binaryLabels.pkl', \
                'This should be file used for training, not testing'
            if use_chest_imagenome_label_gold_set:
                assert not reorder_chest_imagenome_labels, 'Not supported yet'
                _, self.chest_imagenome_labels = \
                    load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix('gold_imageId2binaryLabels.pkl')
                print('Using gold labels for Chest Imagenome (not the labels used for training)')
                print(f'\tFile used in training: {chest_imagenome_labels_filename}')
                print('\tFile used for testing (right now): gold_imageId2binaryLabels.pkl')
                print(f'self.chest_imagenome_labels.shape = {self.chest_imagenome_labels.shape}')
                # sanity check
                used_rid_set = set()
                _hc, _unhc = 0, 0
                for i in self.test_indices:
                    rid = self.report_ids[i]
                    used_rid_set.add(rid)
                    if self.chest_imagenome_labels[rid].max() == 1:
                        _unhc += 1
                    else:
                        _hc += 1
                print('Sanity check: _hc =', _hc, '_unhc =', _unhc)
                assert _unhc > 0, 'No abnormal reports found in gold labels'
                # choose 100 report_ids not in used_rid_set
                for _ in range(100):
                    while True:
                        rid = random.randint(0, self.chest_imagenome_labels.shape[0] - 1)
                        if rid not in used_rid_set:
                            break
                    assert self.chest_imagenome_labels[rid].max() == 0
            else:
                _, self.chest_imagenome_labels = \
                    load_chest_imagenome_dicom_ids_and_labels_as_numpy_matrix(chest_imagenome_labels_filename)
                if reorder_chest_imagenome_labels:
                    assert chest_imagenome_label_names_filename is not None
                    print_bold('Reordering Chest Imagenome labels...')
                    label_order = load_chest_imagenome_label_order(chest_imagenome_label_names_filename)
                    self.chest_imagenome_labels = self.chest_imagenome_labels[:, label_order]
                    self.chest_imagenome_label_names = [self.chest_imagenome_label_names[i] for i in label_order]
        else:
            self.chest_imagenome_labels = None

        # Load ensemble predictions
        if use_ensemble_predictions:
            assert precomputed_sigmoid_paths is not None
            assert len(precomputed_sigmoid_paths) > 0
            assert use_chexpert or use_chest_imagenome
            print('Loading precomputed sigmoid paths for ensemble predictions...')
            self.precomputed_sigmoid_paths = precomputed_sigmoid_paths
            if use_hard_predictions:
                self.precomputed_thresholds_paths = precomputed_thresholds_paths
            if filter_labels:
                self.precomputed_filtered_labels_paths = precomputed_filtered_labels_paths
            if use_chexpert:
                self.ensemble_chexpert_sigmoids = []
                self.ensemble_chexpert_did2idx_dicts = []
                for i, x in enumerate(precomputed_sigmoid_paths):
                    assert 'chexpert_sigmoids_path' in x
                    print('  Loading', x['chexpert_sigmoids_path'])
                    tmp = load_pickle(x['chexpert_sigmoids_path'])
                    _probs = tmp['pred_chexpert_probs']
                    _dids = tmp['dicom_ids']
                    print(f'    _probs.shape = {_probs.shape}')
                    print(f'    len(_dids) = {len(_dids)}')
                    _did2idx = {did: idx for idx, did in enumerate(_dids)}
                    if use_hard_predictions:
                        _thresholds_filepath = self.precomputed_thresholds_paths[i]['chexpert']
                        _thresholds = load_pickle(_thresholds_filepath)
                        print(f'    _thresholds.shape = {_thresholds.shape}')
                        _probs = (_probs > _thresholds).astype(np.float16)
                        print_red(f'    _probs has been thresholded')
                        if filter_labels:
                            _label_indices_filepath = self.precomputed_filtered_labels_paths[i]['chexpert_indices_path']
                            _label_indices = load_pickle(_label_indices_filepath)
                            print(f'    _label_indices.shape = {_label_indices.shape}')
                            _probs = _probs[:, _label_indices]
                            print_red(f'    _probs has been filtered')
                    # sanity check did2idx
                    assert len(_did2idx) >= len(self.dicom_ids), \
                        f'len(_did2idx) = {len(_did2idx)}, len(self.dicom_ids) = {len(self.dicom_ids)}'
                    for _did in self.dicom_ids:
                        assert _did in _did2idx, f'_did = {_did}'
                    self.ensemble_chexpert_sigmoids.append(_probs)
                    self.ensemble_chexpert_did2idx_dicts.append(_did2idx)
                if use_hard_predictions:
                    self._sanity_check_chexpert_labels()
            if use_chest_imagenome:
                self._load_ensemble_chest_imagenome_label_orders()
                self.ensemble_chest_imagenome_sigmoids = []
                self.ensemble_chest_imagenome_did2idx_dicts = []
                for x in precomputed_sigmoid_paths:
                    assert 'chest_imagenome_sigmoids_path' in x
                    print('  Loading', x['chest_imagenome_sigmoids_path'])
                    tmp = load_pickle(x['chest_imagenome_sigmoids_path'])
                    _probs = tmp['pred_chest_imagenome_probs']
                    _dids = tmp['dicom_ids']
                    print(f'    _probs.shape = {_probs.shape}')
                    print(f'    len(_dids) = {len(_dids)}')
                    _did2idx = {did: idx for idx, did in enumerate(_dids)}
                    if use_hard_predictions:
                        _thresholds_filepath = self.precomputed_thresholds_paths[i]['chest_imagenome']
                        _thresholds = load_pickle(_thresholds_filepath)
                        print(f'    _thresholds.shape = {_thresholds.shape}')
                        _probs = (_probs > _thresholds).astype(np.float16)
                        print_red(f'    _probs has been thresholded')
                        if filter_labels:
                            _label_indices_filepath = self.precomputed_filtered_labels_paths[i]['chest_imagenome_indices_path']
                            _label_indices = load_pickle(_label_indices_filepath)
                            print(f'    _label_indices.shape = {_label_indices.shape}')
                            _probs = _probs[:, _label_indices]
                            print_red(f'    _probs has been filtered')
                    # sanity check did2idx
                    assert len(_did2idx) >= len(self.dicom_ids), \
                        f'len(_did2idx) = {len(_did2idx)}, len(self.dicom_ids) = {len(self.dicom_ids)}'
                    for _did in self.dicom_ids:
                        assert _did in _did2idx, f'_did = {_did}'
                    self.ensemble_chest_imagenome_sigmoids.append(_probs)
                    self.ensemble_chest_imagenome_did2idx_dicts.append(_did2idx)
                if use_hard_predictions:
                    self._sanity_check_chest_imagenome_labels()
            print('Done loading precomputed sigmoid paths for ensemble predictions')
            def _get_ensemble_sigmoid_vector(did):
                # concatenate all sigmoid vectors
                _sigmoid_vector = []
                if use_chexpert:
                    for _probs, _did2idx in zip(self.ensemble_chexpert_sigmoids, self.ensemble_chexpert_did2idx_dicts):
                        _sigmoid_vector.append(_probs[_did2idx[did]])
                if use_chest_imagenome:
                    for _probs, _did2idx in zip(self.ensemble_chest_imagenome_sigmoids, self.ensemble_chest_imagenome_did2idx_dicts):
                        _sigmoid_vector.append(_probs[_did2idx[did]])
                _sigmoid_vector = np.concatenate(_sigmoid_vector)
                return _sigmoid_vector
            def _get_input_labels_breakdown(idxs):
                output = {
                    'is_ensemble': True,
                    'model_folder_paths': [x['model_folder_path'] for x in self.precomputed_sigmoid_paths],
                }
                if use_hard_predictions:
                    output['precomputed_thresholds_paths'] = self.precomputed_thresholds_paths
                if filter_labels:
                    output['precomputed_filtered_labels_paths'] = self.precomputed_filtered_labels_paths
                if use_chexpert:
                    output['chexpert'] = []
                    for _probs, _did2idx in zip(self.ensemble_chexpert_sigmoids, self.ensemble_chexpert_did2idx_dicts):
                        output['chexpert'].append(np.array([_probs[_did2idx[self.dicom_ids[idx]]] for idx in idxs]))
                if use_chest_imagenome:
                    output['chest_imagenome'] = []
                    for _probs, _did2idx in zip(self.ensemble_chest_imagenome_sigmoids, self.ensemble_chest_imagenome_did2idx_dicts):
                        output['chest_imagenome'].append(np.array([_probs[_did2idx[self.dicom_ids[idx]]] for idx in idxs]))
                return output
            self.get_ensemble_sigmoid_vector = _get_ensemble_sigmoid_vector # function pointer
            self.get_input_labels_breakdown = _get_input_labels_breakdown # function pointer
        else:
            def _get_input_labels_breakdown(idxs):
                output = {
                    'is_ensemble': False,
                }
                if self.use_gender:
                    output['gender'] = np.array(self.genders[idxs])
                if self.use_chexpert:
                    output['chexpert'] = np.array([self.chexpert_labels[self.report_ids[idx]] for idx in idxs])
                if self.use_chest_imagenome:
                    output['chest_imagenome'] = np.array([self.chest_imagenome_labels[self.report_ids[idx]] for idx in idxs])
                return output            
            self.precomputed_sigmoid_paths = None
            self.get_ensemble_sigmoid_vector = None
            self.get_input_labels_breakdown = _get_input_labels_breakdown # function pointer

        if use_test_set or use_chest_imagenome_label_gold_set:
            # Create test dataset and dataloader
            self.test_dataset, self.test_dataloader = self._create_dataset_and_dataloader(
                self.test_indices, self.eval_collate_batch_fn)
        else:
            if not use_val_set_only:
                # Create train dataset and dataloader
                self.balanced_batch_size = balanced_batch_size
                self.train_dataset, self.train_dataloader = self._create_dataset_and_dataloader(
                    self.train_indices, self.train_collate_batch_fn, shuffle=True, balanced_sampling_mode=balanced_sampling_mode)

            # Create validation dataset and dataloader
            self.val_dataset, self.val_dataloader = self._create_dataset_and_dataloader(
                self.val_indices, self.eval_collate_batch_fn)

    def _load_ensemble_chest_imagenome_label_orders(self):
        self.ensemble_chest_imagenome_label_orders = []
        self.ensemble_chest_imagenome_label_names_filenames = []
        for x in self.precomputed_sigmoid_paths:
            model_folder_path = x['model_folder_path']
            metadata_path = os.path.join(model_folder_path, 'metadata.json')
            metadata = load_json_file(metadata_path)
            chest_imagenome_label_names_filename = metadata['mimiccxr_trainer_kwargs'].get('chest_imagenome_label_names_filename', None)
            if chest_imagenome_label_names_filename is not None:
                label_order = load_chest_imagenome_label_order(chest_imagenome_label_names_filename)
                self.ensemble_chest_imagenome_label_orders.append(label_order)
                self.ensemble_chest_imagenome_label_names_filenames.append(chest_imagenome_label_names_filename)
                print(f'Label order inferred from {chest_imagenome_label_names_filename}')
            else:
                self.ensemble_chest_imagenome_label_orders.append(None)
    
    def _sanity_check_labels(self, label_name, sigmoids_list, did2idx_dicts, gt_labels, label_order_list=None):
        print_red(f'    Sanity checking {label_name} labels', bold=True)
        if hasattr(self, 'val_indices'):
            print(f'    len(self.val_indices) = {len(self.val_indices)}')            
            indices = self.val_indices
        elif hasattr(self, 'test_indices'):
            print(f'    len(self.test_indices) = {len(self.test_indices)}')            
            indices = self.test_indices
        else:
            raise NotImplementedError
        if label_order_list is not None:
            assert type(label_order_list) == list
            assert len(sigmoids_list) == len(label_order_list)

        from sklearn.metrics import f1_score
        for i, (_probs, _did2idx) in enumerate(zip(sigmoids_list, did2idx_dicts)):
            _pred_labels = np.array([_probs[_did2idx[self.dicom_ids[i]]] for i in indices]).astype(int)
            _gt_labels = np.array([gt_labels[self.report_ids[i]] for i in indices])
            if label_order_list is not None and label_order_list[i] is not None:
                assert len(label_order_list[i]) == _gt_labels.shape[1]
                _gt_labels = _gt_labels[:, label_order_list[i]]
            if self.filter_labels:
                _label_indices_filepath = self.precomputed_filtered_labels_paths[i][f'{label_name}_indices_path']
                _label_indices = load_pickle(_label_indices_filepath)
                _gt_labels = _gt_labels[:, _label_indices]
            print(f'    f1_score(_gt_labels, _pred_labels, average="micro") = {f1_score(_gt_labels, _pred_labels, average="micro")}')
            print(f'    f1_score(_gt_labels, _pred_labels, average="macro") = {f1_score(_gt_labels, _pred_labels, average="macro")}')

    def _sanity_check_chexpert_labels(self):
        self._sanity_check_labels('chexpert', self.ensemble_chexpert_sigmoids,
                                  self.ensemble_chexpert_did2idx_dicts, self.chexpert_labels)
    
    def _sanity_check_chest_imagenome_labels(self):
        self._sanity_check_labels('chest_imagenome', self.ensemble_chest_imagenome_sigmoids,
                                  self.ensemble_chest_imagenome_did2idx_dicts, self.chest_imagenome_labels,
                                  label_order_list=self.ensemble_chest_imagenome_label_orders)

    def _create_dataset(self, indices, shuffle=False, infinite=False):
        return MIMICCXR_Labels2Report_Dataset(
            indices=indices,
            reports=self.reports,
            report_ids=self.report_ids,
            shuffle=shuffle,
            infinite=infinite,
            use_gender=self.use_gender,
            genders=self.genders,
            use_chexpert=self.use_chexpert,
            chexpert_labels=self.chexpert_labels,
            use_chest_imagenome=self.use_chest_imagenome,
            chest_imagenome_labels=self.chest_imagenome_labels,
            use_ensemble_predictions=self.use_ensemble_predictions,
            get_ensemble_sigmoid_vector=self.get_ensemble_sigmoid_vector,
            dicom_ids=self.dicom_ids,
        )
    
    def _create_dataset_and_dataloader(self, indices, collate_batch_fn, shuffle=False, balanced_sampling_mode=None):
        
        dataset = _create_dataset(self, indices, shuffle, balanced_sampling_mode)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and balanced_sampling_mode is None,
            num_workers=self.num_workers,
            collate_fn=collate_batch_fn,
            pin_memory=True,
        )
        return dataset, dataloader