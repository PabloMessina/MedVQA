import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_NUM_BBOX_CLASSES,
    get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices,
)
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_gold_bboxes,
    load_chest_imagenome_silver_bboxes,
    load_gold_bbox_dicom_ids,
    load_nondecent_chest_imagenome_dicom_ids,
)
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH, CompositeInfiniteDataset, SequentialDataLoader
from medvqa.datasets.ms_cxr import get_ms_cxr_dicom_id_2_phrases_and_masks, get_ms_cxr_dicom_ids
from medvqa.datasets.segmentation_utils import compute_mask_from_bounding_box
from medvqa.datasets.mimiccxr import (
    MIMICCXR_LARGE_FAST_CACHE_DIR,
    MIMICCXR_ImageSizeModes,
    MIMICCXR_ViewModes,
    get_dicom_id_and_orientation_list,
    get_image_path_getter,
    load_mimiccxr_reports_detailed_metadata,
)
from medvqa.utils.constants import LABEL_BASED_FACTS
from medvqa.utils.files import get_cached_pickle_file, load_pickle, save_pickle
from medvqa.utils.logging import print_bold

class MIMICCXR_FactGroundingDataset(Dataset):

    def __init__(self, image_paths, image_transform, fact_embeddings, positive_facts, negative_facts, indices, num_facts,
                 infinite=False, shuffle=False, use_weights=False, weights_filepath=None):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.fact_embeddings = fact_embeddings
        self.positive_facts = positive_facts
        self.negative_facts = negative_facts
        self.indices = indices
        self.num_facts = num_facts
        self.infinite = infinite
        if shuffle:
            random.shuffle(self.indices)
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)

        if use_weights:
            assert weights_filepath is not None
            self.use_weights = True
            data = get_cached_pickle_file(weights_filepath)
            self.cluster_assignments = data['cluster_assignments']
            self.cluster_weights = data['cluster_weights']
            self.label_weights = data['label_weights']
            self.num_labels = len(self.label_weights)
            assert self.num_labels == len(LABEL_BASED_FACTS) # sanity check
        else:
            self.use_weights = False

    def __len__(self):
        return self._len

    @staticmethod
    def _adapt_fact_indices(fact_indices, target_num_facts):
        assert len(fact_indices) > 0
        if len(fact_indices) > target_num_facts: # sample a subset of facts
            fact_indices = random.sample(fact_indices, target_num_facts)
        elif len(fact_indices) < target_num_facts: # duplicate facts
            fact_indices_ = []
            x = target_num_facts // len(fact_indices)
            y = target_num_facts % len(fact_indices)
            for _ in range(x):
                fact_indices_.extend(fact_indices)
            if y > 0:
                fact_indices_.extend(random.sample(fact_indices, y))
            fact_indices = fact_indices_
        assert len(fact_indices) == target_num_facts
        return fact_indices
    
    def __getitem__(self, i):
        if self.infinite:
            i = i % len(self.indices)
        idx = self.indices[i]
        image_path = self.image_paths[idx]
        image = self.image_transform(image_path)
        
        positive_facts = self.positive_facts[idx]
        negative_facts = self.negative_facts[idx]
        if len(positive_facts) > 0 and len(negative_facts) > 0:
            if len(positive_facts) < self.num_facts and len(negative_facts) < self.num_facts:
                positive_facts = self._adapt_fact_indices(positive_facts, self.num_facts)
                negative_facts = self._adapt_fact_indices(negative_facts, self.num_facts)
            elif len(positive_facts) < self.num_facts:
                negative_facts = self._adapt_fact_indices(negative_facts, self.num_facts * 2 - len(positive_facts))
            elif len(negative_facts) < self.num_facts:
                positive_facts = self._adapt_fact_indices(positive_facts, self.num_facts * 2 - len(negative_facts))
            else:
                assert len(positive_facts) >= self.num_facts and len(negative_facts) >= self.num_facts
                positive_facts = self._adapt_fact_indices(positive_facts, self.num_facts)
                negative_facts = self._adapt_fact_indices(negative_facts, self.num_facts)
        elif len(positive_facts) > 0:
            positive_facts = self._adapt_fact_indices(positive_facts, self.num_facts * 2)
        elif len(negative_facts) > 0:
            negative_facts = self._adapt_fact_indices(negative_facts, self.num_facts * 2)
        else:
            raise ValueError('No positive or negative facts found!')

        fact_indices = positive_facts + negative_facts
        assert len(fact_indices) == 2 * self.num_facts
        embeddings = self.fact_embeddings[fact_indices]
        labels = np.zeros(len(fact_indices), dtype=np.int64)
        labels[:len(positive_facts)] = 1
        
        if self.use_weights:
            weights = np.empty(len(labels), dtype=np.float32)
            for i, label in enumerate(labels):
                fidx = fact_indices[i]
                if fidx < self.num_labels: # use label-specific weight
                    weights[i] = self.label_weights[fidx, 1 - label] # 0 -> positive fact, 1 -> negative fact
                else: # use cluster-specific weight
                    weights[i] = self.cluster_weights[self.cluster_assignments[fidx], 1 - label] # 0 -> positive fact, 1 -> negative fact
            return {
                'i': image,
                'pe': embeddings, # phrase embeddings
                'pw': weights, # phrase weights
                'l': labels, # labels
            }
        else:
            return {
                'i': image,
                'pe': embeddings, # phrase embeddings
                'l': labels, # labels
            }
    
class MIMICCXR_PhraseGroundingDataset(Dataset):

    def __init__(self, image_paths, image_transform, phrases, phrase_embeddings, phrase_grounding_masks, indices):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrases = phrases
        self.phrase_embeddings = phrase_embeddings
        self.phrase_grounding_masks = phrase_grounding_masks
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        # print(f'MIMICCXR_PhraseGroundingDataset.__getitem__({i})')
        idx = self.indices[i]
        image_path = self.image_paths[idx]
        image = self.image_transform(image_path)
        phrase_embeddings = self.phrase_embeddings[idx]
        phrase_grounding_masks = self.phrase_grounding_masks[idx]
        return {
            'i': image,
            'pe': phrase_embeddings,
            'pgm': phrase_grounding_masks,
        }

class MIMICCXR_BBoxGroundingDataset(Dataset):

    def __init__(self, image_paths, image_transform, phrase_embeddings, phrase_grounding_masks, phrase_classification_labels,
                 bbox_coords, bbox_presence, use_yolov8=False):
        assert use_yolov8 # TODO: add support for non-YOLOv8
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_embeddings = phrase_embeddings
        self.phrase_grounding_masks = phrase_grounding_masks
        self.phrase_classification_labels = phrase_classification_labels
        self.bbox_coords = bbox_coords
        self.bbox_presence = bbox_presence
        self.use_yolov8 = use_yolov8

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        image_path = self.image_paths[i]
        tmp = self.image_transform(image_path, return_image_size=True)
        image, image_size_before, image_size_after = tmp
        phrase_embeddings = self.phrase_embeddings
        phrase_grounding_masks = self.phrase_grounding_masks[i]
        phrase_classification_labels = self.phrase_classification_labels[i]
        bbox_coords = self.bbox_coords[i]
        bbox_presence = self.bbox_presence[i]
        return {
            'i': image,
            'pe': phrase_embeddings,
            'pgm': phrase_grounding_masks,
            'pcl': phrase_classification_labels,
            'bc': bbox_coords,
            'bp': bbox_presence,
            # for YOLOv8
            'im_file': image_path,
            'ori_shape': image_size_before,
            'resized_shape': image_size_after,
        }
    
def _compute_mask_from_bounding_boxes(mask_height, mask_width, bbox_coords, bbox_presence):
    assert len(bbox_coords.shape) == 2
    assert bbox_coords.shape[1] == 4
    assert len(bbox_presence.shape) == 1
    assert bbox_presence.shape[0] == bbox_coords.shape[0]
    mask = np.zeros((len(bbox_coords), mask_height * mask_width), dtype=np.float32)
    for i in range(len(bbox_coords)):
        if bbox_presence[i] == 1:
            x1, y1, x2, y2 = bbox_coords[i]
            mask[i] = compute_mask_from_bounding_box(mask_height, mask_width, x1, y1, x2, y2, flatten=True)
    return mask

def _clean_bbox_coords_and_presence(bbox_coords, bbox_presence):
    bbox_coords = bbox_coords.reshape(-1, 4)
    assert len(bbox_coords) == len(bbox_presence)
    bbox_coords.clip(0, 1, out=bbox_coords)
    for i in range(len(bbox_coords)):
        w = bbox_coords[i, 2] - bbox_coords[i, 0]
        h = bbox_coords[i, 3] - bbox_coords[i, 1]
        if w <= 0 or h <= 0:
            bbox_presence[i] = 0
        else:
            assert bbox_presence[i] == 1
    return bbox_coords, bbox_presence

_shared_mask_height = None
_shared_mask_width = None
_shared_did2bboxes = None
def _clean_bbox_coords_and_presence_and_compute_mask(dicom_id):
    bboxes = _shared_did2bboxes[dicom_id]
    bbox_coords = bboxes['coords']
    bbox_presence = bboxes['presence']
    bbox_coords, bbox_presence = _clean_bbox_coords_and_presence(bbox_coords, bbox_presence)
    mask = _compute_mask_from_bounding_boxes(_shared_mask_height, _shared_mask_width, bbox_coords, bbox_presence)
    return bbox_coords, bbox_presence, mask

def _precompute_bbox_coords_and_presence_and_mask(mask_height, mask_width, did2bboxes, num_workers=None):
    save_path = os.path.join(MIMICCXR_LARGE_FAST_CACHE_DIR, f'bbox_coords_and_presence_and_mask({mask_height},{mask_width},{len(did2bboxes)}).pkl')
    if os.path.exists(save_path):
        print_bold(f'Loading precomputed bbox_coords_and_presence_and_mask from {save_path}...')
        return get_cached_pickle_file(save_path)

    print_bold(f'Precomputing bbox_coords_and_presence_and_mask({mask_height},{mask_width},{len(did2bboxes)})...')
    global _shared_mask_height, _shared_mask_width, _shared_did2bboxes
    _shared_mask_height = mask_height
    _shared_mask_width = mask_width
    _shared_did2bboxes = did2bboxes
    dicom_ids = list(did2bboxes.keys())
    dicom_ids.sort() # sort for reproducibility
    import multiprocessing as mp
    if num_workers is None:
        num_workers = mp.cpu_count()
    with mp.Pool(num_workers) as pool:
        results = pool.map(_clean_bbox_coords_and_presence_and_compute_mask, dicom_ids)
    bbox_coords = np.array([r[0] for r in results])
    bbox_presence = np.array([r[1] for r in results])
    phrase_grounding_masks = np.array([r[2] for r in results])
    print(f'bbox_coords.shape = {bbox_coords.shape}')
    print(f'bbox_presence.shape = {bbox_presence.shape}')
    print(f'phrase_grounding_masks.shape = {phrase_grounding_masks.shape}')
    output = {
        'dicom_ids': dicom_ids, # sorted
        'bbox_coords': bbox_coords,
        'bbox_presence': bbox_presence,
        'phrase_grounding_masks': phrase_grounding_masks,
    }
    save_pickle(output, save_path)
    return output

def visualize_mask(mask, mask_height, mask_width): # for debugging
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    # use gray scale, where white = 1 and black = 0
    plt.imshow(mask.reshape(mask_height, mask_width), cmap='gray', vmin=0, vmax=1)
    plt.show()

_LONG_TAIL = 0
_MIDDLE_TAIL = 1
_SHORT_TAIL = 2

def _assign_distribution_classes_to_reports(report_fact_nli_integrated_data_filepath,
                                            distribution_thresholds=(0.02, 0.05)):
    assert len(distribution_thresholds) == 2
    assert 0 < distribution_thresholds[0] < distribution_thresholds[1] < 1
    print_bold(f'Assigning distribution classes to reports...')
    # Load the data
    print(f'Loading mimiccxr_report_fact_nli_integrated_data from {report_fact_nli_integrated_data_filepath}...')
    data = load_pickle(report_fact_nli_integrated_data_filepath)
    labels = data['label_based_nli_predictions']['nli_predictions']
    print(f'labels.shape = {labels.shape}')
    n_reports = labels.shape[0]
    binary_labels = labels == 0
    count_per_class = binary_labels.sum(0)
    count_no_positives = (binary_labels.sum(1) == 0).sum() # number of reports with no positive class
    distribution_classes = np.zeros(n_reports, dtype=np.int8)    
    t0, t1 = distribution_thresholds
    if count_no_positives < t0 * n_reports:
        no_positive_class = _LONG_TAIL
    elif count_no_positives < t1 * n_reports:
        no_positive_class = _MIDDLE_TAIL
    else:
        no_positive_class = _SHORT_TAIL
    is_long_tail = count_per_class < t0 * n_reports
    is_middle_tail = (t0 * n_reports <= count_per_class) & (count_per_class < t1 * n_reports)
    is_short_tail = count_per_class >= t1 * n_reports
    
    for i in range(n_reports):
        if binary_labels[i].sum() == 0: # no positive class
            distribution_classes[i] = no_positive_class
        elif is_long_tail[binary_labels[i]].any(): # at least one long tail class
            distribution_classes[i] = _LONG_TAIL
        elif is_middle_tail[binary_labels[i]].any(): # at least one middle tail class
            distribution_classes[i] = _MIDDLE_TAIL
        else: # all short tail classes
            assert is_short_tail[binary_labels[i]].all()
            distribution_classes[i] = _SHORT_TAIL

    print(f'count_no_positives = {count_no_positives}')
    print(f'number of long tail classes = {(is_long_tail).sum()}')
    print(f'number of middle tail classes = {(is_middle_tail).sum()}')
    print(f'number of short tail classes = {(is_short_tail).sum()}')
    print(f'number of long tail reports = {(distribution_classes == _LONG_TAIL).sum()}')
    print(f'number of middle tail reports = {(distribution_classes == _MIDDLE_TAIL).sum()}')
    print(f'number of short tail reports = {(distribution_classes == _SHORT_TAIL).sum()}')

    return distribution_classes

class MIMICCXR_PhraseGroundingTrainer:

    def __init__(self, 
                max_images_per_batch, max_phrases_per_batch, max_phrases_per_image,
                num_train_workers=None, num_test_workers=None,
                train_image_transform=None, test_image_transform=None,
                fact_grounding_collate_batch_fn=None,
                phrase_grounding_collate_batch_fn=None,
                bbox_grounding_collate_batch_fn=None,
                test_batch_size_factor=1,
                mask_width=None, mask_height=None,
                use_facts_for_train=False,
                use_facts_for_test=False,
                dicom_id_to_pos_neg_facts_filepath=None,
                use_mscxr_for_train=False,
                use_mscxr_for_test=False,
                mscxr_phrase2embedding_filepath=None,
                use_chest_imagenome_for_train=False,
                use_chest_imagenome_gold_for_test=False,
                chest_imagenome_bbox_phrase_embeddings_filepath=None,
                source_image_size_mode=MIMICCXR_ImageSizeModes.SMALL_256x256,
                exclude_noisy_images=False,
                use_yolov8=False,
                mask_exponent=1,
                balance_long_middle_short_tail=False,
                long_middle_short_tail_thresholds=(0.02, 0.05),
                report_fact_nli_integrated_data_filepath=None,
                use_weighted_phrase_classifier_loss=False,
                cluster_and_label_weights_for_facts_filepath=None,
                **unused_kwargs,
            ):

        if len(unused_kwargs) > 0:
            # Print warning in orange and bold
            print('\033[93m\033[1mWarning: unused kwargs in MIMICCXR_VisualModuleTrainer: {}\033[0m'.format(unused_kwargs))
        # Sanity checks
        assert sum([use_facts_for_train, use_chest_imagenome_for_train, use_mscxr_for_train,
                    use_mscxr_for_test, use_chest_imagenome_gold_for_test]) > 0 # at least one of them must be True
        assert 0 < mask_exponent <= 1
        print(f'mask_exponent = {mask_exponent}')
        
        self.use_facts_for_train = use_facts_for_train
        self.use_chest_imagenome_for_train = use_chest_imagenome_for_train
        self.use_mscxr_for_train = use_mscxr_for_train
        self.use_mscxr_for_test = use_mscxr_for_test
        self.use_chest_imagenome_gold_for_test = use_chest_imagenome_gold_for_test
        
        self.use_yolov8 = use_yolov8

        forbidden_train_dicom_ids = set()

        if exclude_noisy_images:
            noisy_dicom_ids = set(load_nondecent_chest_imagenome_dicom_ids())
            forbidden_train_dicom_ids |= noisy_dicom_ids

        if use_mscxr_for_test:
            ms_cxr_dicom_ids = set(get_ms_cxr_dicom_ids())
            if not use_mscxr_for_train:
                forbidden_train_dicom_ids |= ms_cxr_dicom_ids

        if use_chest_imagenome_gold_for_test:
            gold_dicom_ids = set(load_gold_bbox_dicom_ids())
            forbidden_train_dicom_ids |= gold_dicom_ids

        print(f'len(forbidden_train_dicom_ids) = {len(forbidden_train_dicom_ids)}')

        # Create train mimiccxr facts dataset
        if use_facts_for_train or use_facts_for_test:
            print_bold('Preparing MIMIC-CXR-Facts datasets and dataloaders for training/testing...')
            assert dicom_id_to_pos_neg_facts_filepath is not None
            assert fact_grounding_collate_batch_fn is not None
            assert num_train_workers is not None
            assert train_image_transform is not None

            tmp = load_pickle(dicom_id_to_pos_neg_facts_filepath)
            fact_embeddings = tmp['embeddings']
            dicom_id_to_pos_neg_facts = tmp['dicom_id_to_pos_neg_facts']
            print(f'fact_embeddings.shape = {fact_embeddings.shape}')

            if balance_long_middle_short_tail and use_facts_for_train: # only for training
                assert report_fact_nli_integrated_data_filepath is not None
                distribution_classes = _assign_distribution_classes_to_reports(
                    report_fact_nli_integrated_data_filepath=report_fact_nli_integrated_data_filepath,
                    distribution_thresholds=long_middle_short_tail_thresholds)

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            positive_facts = [None] * BIG_ENOGUGH
            negative_facts = [None] * BIG_ENOGUGH
            report_idxs = [None] * BIG_ENOGUGH
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()

            if use_facts_for_train:
                train_indices = []
            if use_facts_for_test:
                test_indices = []

            idx = 0
            for ridx, (part_id, subject_id, study_id, dicom_id_view_pairs, split) in \
                tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs'],
                    mimiccxr_metadata['splits'])), mininterval=2):
                for dicom_id, view in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id not in dicom_id_to_pos_neg_facts:
                        continue
                    if split == 'validate' or split == 'test':
                        if use_facts_for_test:
                            test_indices.append(idx)
                        else:
                            if dicom_id in forbidden_train_dicom_ids:
                                continue
                            train_indices.append(idx)
                    elif split == 'train':
                        if use_facts_for_train:
                            if dicom_id in forbidden_train_dicom_ids:
                                continue
                            train_indices.append(idx)
                        else:
                            continue
                    else:
                        raise ValueError(f'Invalid split: {split}')
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    pos_neg_facts = dicom_id_to_pos_neg_facts[dicom_id]
                    assert len(pos_neg_facts) == 2
                    positive_facts[idx] = pos_neg_facts[0]
                    negative_facts[idx] = pos_neg_facts[1]
                    report_idxs[idx] = ridx
                    idx += 1

            print(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
            positive_facts = positive_facts[:idx]
            negative_facts = negative_facts[:idx]
            report_idxs = report_idxs[:idx]
            aux = 0
            if use_facts_for_train:
                print(f'len(train_indices) = {len(train_indices)}')
                aux += len(train_indices)
            if use_facts_for_test:
                print(f'len(test_indices) = {len(test_indices)}')
                aux += len(test_indices)
            assert aux == idx # sanity check

            # Calculate the average number of facts per image
            if use_facts_for_train:
                aux = 0
                for i in train_indices:
                    pos_facts = positive_facts[i]
                    neg_facts = negative_facts[i]
                    assert len(pos_facts) + len(neg_facts) > 0 # at least one fact
                    aux += max(len(pos_facts), len(neg_facts))
                avg_facts_per_image = aux / len(train_indices)
                train_num_facts_per_image = min(max_phrases_per_image, int(avg_facts_per_image))
                print(f'avg_facts_per_image = {avg_facts_per_image}')
                print(f'train_num_facts_per_image = {train_num_facts_per_image}')
            if use_facts_for_test:
                aux = 0
                for i in test_indices:
                    pos_facts = positive_facts[i]
                    neg_facts = negative_facts[i]
                    assert len(pos_facts) + len(neg_facts) > 0 # at least one fact
                    aux += max(len(pos_facts), len(neg_facts))
                avg_facts_per_image = aux / len(test_indices)
                test_num_facts_per_image = min(max_phrases_per_image, int(avg_facts_per_image))
                print(f'avg_facts_per_image = {avg_facts_per_image}')
                print(f'test_num_facts_per_image = {test_num_facts_per_image}')

            # Create dataset and dataloader for training
            if use_facts_for_train:
                print_bold('Building train fact dataloader...')
                batch_size = max(min(max_images_per_batch, max_phrases_per_batch // train_num_facts_per_image), 1) # at least 1

                if balance_long_middle_short_tail:
                    print('Balancing long, middle, and short tail classes...')
                    long_tail_indices = [i for i in train_indices if distribution_classes[report_idxs[i]] == _LONG_TAIL]
                    middle_tail_indices = [i for i in train_indices if distribution_classes[report_idxs[i]] == _MIDDLE_TAIL]
                    short_tail_indices = [i for i in train_indices if distribution_classes[report_idxs[i]] == _SHORT_TAIL]
                    assert len(long_tail_indices) > 0
                    assert len(middle_tail_indices) > 0
                    assert len(short_tail_indices) > 0
                    assert len(long_tail_indices) + len(middle_tail_indices) + len(short_tail_indices) == len(train_indices)
                    indices_list = [long_tail_indices, middle_tail_indices, short_tail_indices]
                    indices_list.sort(key=lambda x: len(x), reverse=True)
                    while len(indices_list) >= 2 and len(indices_list[-1]) < 100:
                        indices_list[-2].extend(indices_list[-1])
                        indices_list.pop()
                    assert len(indices_list) >= 1
                    assert len(indices_list[0]) > 0
                    datasets = []
                    for indices_ in indices_list:
                        dataset = MIMICCXR_FactGroundingDataset(
                            image_paths=image_paths, image_transform=train_image_transform,
                            fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                            indices=indices_, num_facts=train_num_facts_per_image, shuffle=True, infinite=True,
                            use_weights=use_weighted_phrase_classifier_loss,
                            weights_filepath=cluster_and_label_weights_for_facts_filepath)
                        datasets.append(dataset)
                    if len(datasets) == 1:
                        train_fact_dataset = datasets[0]
                    else:
                        weights = [1] * len(datasets) # equal weights
                        train_fact_dataset = CompositeInfiniteDataset(datasets, weights)
                    train_fact_dataloader = DataLoader(
                        train_fact_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_train_workers,
                        collate_fn=fact_grounding_collate_batch_fn,
                        pin_memory=True,
                    )
                else:
                    print('Normal (unbalanced) training...')
                    train_fact_dataset = MIMICCXR_FactGroundingDataset(
                        image_paths=image_paths, image_transform=train_image_transform,
                        fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                        indices=indices, num_facts=train_num_facts_per_image,
                        use_weights=use_weighted_phrase_classifier_loss,
                        weights_filepath=cluster_and_label_weights_for_facts_filepath)
                    train_fact_dataloader = DataLoader(
                        train_fact_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_train_workers,
                        collate_fn=fact_grounding_collate_batch_fn,
                        pin_memory=True,
                    )
                self.train_fact_dataset = train_fact_dataset
                self.train_fact_dataloader = train_fact_dataloader
                print(f'len(self.train_fact_dataloader) = {len(self.train_fact_dataloader)}')

            # Create dataset and dataloader for testing
            if use_facts_for_test:
                print_bold('Building test fact dataloaders...')
                test_fact_dataset = MIMICCXR_FactGroundingDataset(
                    image_paths=image_paths, image_transform=test_image_transform,
                    fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                    indices=test_indices, num_facts=test_num_facts_per_image,
                    use_weights=use_weighted_phrase_classifier_loss,
                    weights_filepath=cluster_and_label_weights_for_facts_filepath)
                batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // test_num_facts_per_image), 1) * test_batch_size_factor)
                test_fact_dataloader = DataLoader(
                    test_fact_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_test_workers,
                    collate_fn=fact_grounding_collate_batch_fn,
                    pin_memory=True,
                )
                self.test_fact_dataset = test_fact_dataset
                self.test_fact_dataloader = test_fact_dataloader
                print(f'len(self.test_fact_dataloader) = {len(self.test_fact_dataloader)}')

        # Create mscxr train/test dataset and dataloader
        if use_mscxr_for_train or use_mscxr_for_test:
            print_bold('Preparing MS-CXR dataset and dataloader for training/testing...')
            assert mask_width is not None
            assert mask_height is not None
            assert mscxr_phrase2embedding_filepath is not None
            assert phrase_grounding_collate_batch_fn is not None
            if use_mscxr_for_train:
                assert num_train_workers is not None
                assert train_image_transform is not None
            if use_mscxr_for_test:
                assert num_test_workers is not None
                assert test_image_transform is not None                
            
            dicom_id_2_phrases_and_masks = get_ms_cxr_dicom_id_2_phrases_and_masks(mask_height, mask_width)
            print(f'len(dicom_id_2_phrases_and_masks) = {len(dicom_id_2_phrases_and_masks)}')

            phrase2embedding = load_pickle(mscxr_phrase2embedding_filepath)
            print(f'len(phrase2embedding) = {len(phrase2embedding)}')

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            phrases = [None] * BIG_ENOGUGH
            phrase_embeddings = [None] * BIG_ENOGUGH
            phrase_grounding_masks = [None] * BIG_ENOGUGH
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()

            idx = 0
            for rid, (part_id, subject_id, study_id, dicom_id_view_pairs) in \
                tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs'])), mininterval=2):
                for dicom_id, view in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id not in dicom_id_2_phrases_and_masks:
                        continue
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    phrases_, masks_ = dicom_id_2_phrases_and_masks[dicom_id]
                    phrases[idx] = phrases_
                    phrase_embeddings[idx] = np.array([phrase2embedding[p] for p in phrases_])
                    phrase_grounding_masks[idx] = np.array(masks_) ** mask_exponent # apply mask exponent
                    idx += 1

            print(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
            phrases = phrases[:idx]
            phrase_embeddings = phrase_embeddings[:idx]
            phrase_grounding_masks = phrase_grounding_masks[:idx]

            # Create a mapping from number of phrases to indices
            num_phrases_2_idxs = {}
            for i, pe in enumerate(phrase_embeddings):
                num_phrases = len(pe)
                try:
                    num_phrases_2_idxs[num_phrases].append(i)
                except KeyError:
                    num_phrases_2_idxs[num_phrases] = [i]

            # Create datasets and dataloaders for testing
            if use_mscxr_for_train:
                train_mscxr_dataloaders = []
            if use_mscxr_for_test:
                test_mscxr_dataloaders = []
            for num_phrases, indices in num_phrases_2_idxs.items():
                print(f'Number of phrases: {num_phrases}, # images: {len(indices)}')
                # Sanity checks
                for i in indices:
                    assert phrase_embeddings[i].shape == phrase_embeddings[indices[0]].shape
                    assert phrase_grounding_masks[i].shape == phrase_grounding_masks[indices[0]].shape
                    assert phrase_embeddings[i].shape[0] == phrase_grounding_masks[i].shape[0] == num_phrases
                
                if use_mscxr_for_train:
                    batch_size = max(min(max_images_per_batch, max_phrases_per_batch // num_phrases), 1)
                    dataset = MIMICCXR_PhraseGroundingDataset(
                        image_paths=image_paths, image_transform=train_image_transform, phrases=phrases,
                        phrase_embeddings=phrase_embeddings, phrase_grounding_masks=phrase_grounding_masks,
                        indices=indices)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_train_workers,
                        collate_fn=phrase_grounding_collate_batch_fn,
                        pin_memory=True,
                    )
                    train_mscxr_dataloaders.append(dataloader)
                if use_mscxr_for_test:
                    batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // num_phrases), 1) * test_batch_size_factor)
                    dataset = MIMICCXR_PhraseGroundingDataset(
                        image_paths=image_paths, image_transform=test_image_transform, phrases=phrases,
                        phrase_embeddings=phrase_embeddings, phrase_grounding_masks=phrase_grounding_masks,
                        indices=indices)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_test_workers,
                        collate_fn=phrase_grounding_collate_batch_fn,
                        pin_memory=True,
                    )
                    test_mscxr_dataloaders.append(dataloader)
            if use_mscxr_for_train:
                self.train_mscxr_dataloader = SequentialDataLoader(train_mscxr_dataloaders)
            if use_mscxr_for_test:
                self.test_mscxr_dataloader = SequentialDataLoader(test_mscxr_dataloaders)

        # Create train chest imagenome dataset
        if use_chest_imagenome_for_train:
            print_bold('Preparing Chest Imagenome dataset and dataloader for training...')

            assert chest_imagenome_bbox_phrase_embeddings_filepath is not None
            assert mask_width is not None
            assert mask_height is not None
            assert bbox_grounding_collate_batch_fn is not None
            assert num_train_workers is not None
            assert train_image_transform is not None

            print(f'Loding bbox_phrase_embeddings and bbox_phrases from {chest_imagenome_bbox_phrase_embeddings_filepath}...')
            tmp = get_cached_pickle_file(chest_imagenome_bbox_phrase_embeddings_filepath)
            bbox_phrase_embeddings = tmp['bbox_phrase_embeddings']
            bbox_phrases = tmp['bbox_phrases']
            assert bbox_phrase_embeddings.shape[0] == len(bbox_phrases)
            print(f'bbox_phrase_embeddings.shape = {bbox_phrase_embeddings.shape}')
            print(f'len(bbox_phrases) = {len(bbox_phrases)}')
            for phrase in bbox_phrases:
                print('\t', phrase)

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            chest_imagenome_bbox_coords = [None] * BIG_ENOGUGH
            chest_imagenome_bbox_presence = [None] * BIG_ENOGUGH
            phrase_grounding_masks = [None] * BIG_ENOGUGH
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()
            did2bboxes = load_chest_imagenome_silver_bboxes()
            tmp = _precompute_bbox_coords_and_presence_and_mask(mask_height, mask_width, did2bboxes)
            dicom_ids = tmp['dicom_ids']
            did2idx = {dicom_id: idx for idx, dicom_id in enumerate(dicom_ids)}
            bbox_coords_array = tmp['bbox_coords']
            bbox_presence_array = tmp['bbox_presence']
            phrase_grounding_masks_array = tmp['phrase_grounding_masks']

            idx = 0
            for rid, (part_id, subject_id, study_id, dicom_id_view_pairs) in \
                tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs'])), mininterval=2):
                for dicom_id, view in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id not in did2idx:
                        continue
                    if dicom_id in forbidden_train_dicom_ids:
                        continue
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    i = did2idx[dicom_id]
                    chest_imagenome_bbox_coords[idx] = bbox_coords_array[i]
                    chest_imagenome_bbox_presence[idx] = bbox_presence_array[i]
                    phrase_grounding_masks[idx] = phrase_grounding_masks_array[i] ** mask_exponent # apply mask exponent
                    idx += 1

            print(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
            chest_imagenome_bbox_coords = chest_imagenome_bbox_coords[:idx]
            chest_imagenome_bbox_presence = chest_imagenome_bbox_presence[:idx]
            phrase_grounding_masks = phrase_grounding_masks[:idx]

            # Create dataset and dataloader for training
            self.train_chest_imagenome_dataset = MIMICCXR_BBoxGroundingDataset(
                image_paths=image_paths, image_transform=train_image_transform,
                phrase_embeddings=bbox_phrase_embeddings, phrase_grounding_masks=phrase_grounding_masks,
                phrase_classification_labels=chest_imagenome_bbox_presence, # use bbox_presence as classification labels
                bbox_coords=chest_imagenome_bbox_coords, bbox_presence=chest_imagenome_bbox_presence,
                use_yolov8=use_yolov8)
            batch_size = max(min(max_images_per_batch, max_phrases_per_batch // len(bbox_phrases)), 1) # at least 1 image per batch
            self.train_chest_imagenome_dataloader = DataLoader(
                self.train_chest_imagenome_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_train_workers,
                collate_fn=lambda batch: bbox_grounding_collate_batch_fn(batch, training_mode=True),
                pin_memory=True,
            )
        
        # Create chest imagenome test dataset and dataloader
        if use_chest_imagenome_gold_for_test:
            print_bold('Preparing Chest Imagenome dataset and dataloader for testing...')
            assert mask_width is not None
            assert mask_height is not None
            assert chest_imagenome_bbox_phrase_embeddings_filepath is not None
            assert bbox_grounding_collate_batch_fn is not None
            assert num_test_workers is not None
            assert test_image_transform is not None

            _, gold_pres_indices = get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices()

            print(f'Loding bbox_phrase_embeddings and bbox_phrases from {chest_imagenome_bbox_phrase_embeddings_filepath}...')
            tmp = get_cached_pickle_file(chest_imagenome_bbox_phrase_embeddings_filepath)
            bbox_phrase_embeddings = tmp['bbox_phrase_embeddings']
            bbox_phrases = tmp['bbox_phrases']
            bbox_phrase_embeddings = bbox_phrase_embeddings[gold_pres_indices] # use only gold subset
            bbox_phrases = [bbox_phrases[i] for i in gold_pres_indices] # use only gold subset
            assert bbox_phrase_embeddings.shape[0] == len(bbox_phrases)
            print(f'bbox_phrase_embeddings.shape = {bbox_phrase_embeddings.shape}')
            print(f'len(bbox_phrases) = {len(bbox_phrases)}')
            for phrase in bbox_phrases:
                print('\t', phrase)

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            chest_imagenome_bbox_coords = [None] * BIG_ENOGUGH
            chest_imagenome_bbox_presence = [None] * BIG_ENOGUGH
            phrase_grounding_masks = [None] * BIG_ENOGUGH
            phrase_classification_labels = [None] * BIG_ENOGUGH
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()
            did2bboxes = load_chest_imagenome_gold_bboxes()

            idx = 0
            for rid, (part_id, subject_id, study_id, dicom_id_view_pairs) in \
                tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs'])), mininterval=2):
                for dicom_id, view in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id not in did2bboxes:
                        continue
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    bboxes = did2bboxes[dicom_id]
                    bbox_coords = bboxes['coords']
                    bbox_presence = bboxes['presence']
                    bbox_coords, bbox_presence = _clean_bbox_coords_and_presence(bbox_coords, bbox_presence)
                    chest_imagenome_bbox_coords[idx] = bbox_coords
                    chest_imagenome_bbox_presence[idx] = bbox_presence
                    phrase_grounding_masks[idx] = _compute_mask_from_bounding_boxes(mask_height, mask_width,
                                                                                    bbox_coords[gold_pres_indices], # use only gold subset
                                                                                    bbox_presence[gold_pres_indices]) ** mask_exponent # apply mask exponent
                    phrase_classification_labels[idx] = bbox_presence[gold_pres_indices] # use bbox_presence as classification labels, use only gold subset
                    idx += 1

            print(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
            chest_imagenome_bbox_coords = chest_imagenome_bbox_coords[:idx]
            chest_imagenome_bbox_presence = chest_imagenome_bbox_presence[:idx]
            phrase_grounding_masks = phrase_grounding_masks[:idx]

            # Sanity check
            for i in range(idx):
                assert chest_imagenome_bbox_presence[i].shape[0] == chest_imagenome_bbox_coords[i].shape[0] == CHEST_IMAGENOME_NUM_BBOX_CLASSES,\
                    f'chest_imagenome_bbox_presence[{i}].shape[0] = {chest_imagenome_bbox_presence[i].shape[0]}, chest_imagenome_bbox_coords[{i}].shape[0] = {chest_imagenome_bbox_coords[i].shape[0]}'

            # Create dataset and dataloader for testing
            self.test_chest_imagenome_gold_dataset = MIMICCXR_BBoxGroundingDataset(
                image_paths=image_paths, image_transform=test_image_transform,
                phrase_embeddings=bbox_phrase_embeddings, phrase_grounding_masks=phrase_grounding_masks,
                phrase_classification_labels=phrase_classification_labels,
                bbox_coords=chest_imagenome_bbox_coords, bbox_presence=chest_imagenome_bbox_presence, use_yolov8=use_yolov8)
            batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // len(bbox_phrases)), 1) * test_batch_size_factor)
            self.test_chest_imagenome_gold_dataloader = DataLoader(
                self.test_chest_imagenome_gold_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_test_workers,
                collate_fn=lambda batch: bbox_grounding_collate_batch_fn(batch, training_mode=False),
                pin_memory=True,
            )