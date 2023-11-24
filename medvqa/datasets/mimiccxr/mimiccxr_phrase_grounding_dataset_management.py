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
from medvqa.utils.files import get_cached_pickle_file, load_pickle, save_pickle
from medvqa.utils.logging import print_bold

class MIMICCXR_FactGroundingDataset(Dataset):

    def __init__(self, image_paths, image_transform, fact_embeddings, positive_facts, negative_facts, indices, num_facts):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.fact_embeddings = fact_embeddings
        self.positive_facts = positive_facts
        self.negative_facts = negative_facts
        self.indices = indices
        self.num_facts = num_facts

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        image_path = self.image_paths[idx]
        image = self.image_transform(image_path)
        positive_facts = self.positive_facts[idx]
        if len(positive_facts) > self.num_facts: # sample a subset of positive facts
            positive_facts = random.sample(positive_facts, self.num_facts)
        negative_facts = self.negative_facts[idx]
        if len(negative_facts) > self.num_facts: # sample a subset of negative facts
            negative_facts = random.sample(negative_facts, self.num_facts)
        pos_embeddings = self.fact_embeddings[positive_facts]
        neg_embeddings = self.fact_embeddings[negative_facts]
        return {
            'i': image,
            'pe': pos_embeddings,
            'ne': neg_embeddings,
        }
    
class MIMICCXR_PhraseGroundingDataset(Dataset):

    def __init__(self, image_paths, image_transform, phrase_embeddings, phrase_grounding_masks, indices):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_embeddings = phrase_embeddings
        self.phrase_grounding_masks = phrase_grounding_masks
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        image_path = self.image_paths[idx]
        image = self.image_transform(image_path)
        phrase_embeddings = self.phrase_embeddings
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

class MIMICCXR_PhraseGroundingTrainer:

    def __init__(self, 
                max_images_per_batch, max_phrases_per_batch, max_phrases_per_image,
                num_train_workers, num_test_workers,
                train_image_transform, test_image_transform,
                fact_grounding_collate_batch_fn=None,
                phrase_grounding_collate_batch_fn=None,
                bbox_grounding_collate_batch_fn=None,
                test_batch_size_factor=1,
                mask_width=None, mask_height=None,
                use_facts_for_train=False,
                dicom_id_to_pos_neg_facts_filepath=None,
                use_mscxr_for_test=False,
                mscxr_phrase2embedding_filepath=None,
                use_chest_imagenome_for_train=False,
                use_chest_imagenome_gold_for_test=False,
                chest_imagenome_bbox_phrase_embeddings_filepath=None,
                source_image_size_mode=MIMICCXR_ImageSizeModes.SMALL_256x256,
                exclude_noisy_images=False,
                use_yolov8=False,
                **unused_kwargs,
            ):

        if len(unused_kwargs) > 0:
            # Print warning in orange and bold
            print('\033[93m\033[1mWarning: unused kwargs in MIMICCXR_VisualModuleTrainer: {}\033[0m'.format(unused_kwargs))
        # Sanity checks
        assert sum([use_facts_for_train, use_chest_imagenome_for_train]) > 0
        assert sum([use_mscxr_for_test, use_chest_imagenome_gold_for_test]) > 0
        
        self.use_facts_for_train = use_facts_for_train
        self.use_chest_imagenome_for_train = use_chest_imagenome_for_train
        self.use_mscxr_for_test = use_mscxr_for_test
        self.use_chest_imagenome_gold_for_test = use_chest_imagenome_gold_for_test
        
        self.use_yolov8 = use_yolov8

        forbidden_train_dicom_ids = set()

        if exclude_noisy_images:
            noisy_dicom_ids = set(load_nondecent_chest_imagenome_dicom_ids())
            forbidden_train_dicom_ids |= noisy_dicom_ids

        if use_mscxr_for_test:
            ms_cxr_dicom_ids = set(get_ms_cxr_dicom_ids())
            forbidden_train_dicom_ids |= ms_cxr_dicom_ids

        if use_chest_imagenome_gold_for_test:
            gold_dicom_ids = set(load_gold_bbox_dicom_ids())
            forbidden_train_dicom_ids |= gold_dicom_ids

        print(f'len(forbidden_train_dicom_ids) = {len(forbidden_train_dicom_ids)}')

        # Create train mimiccxr facts dataset
        if use_facts_for_train:
            print_bold('Preparing MIMIC-CXR-Facts datasets and dataloaders for training...')
            assert dicom_id_to_pos_neg_facts_filepath is not None
            assert fact_grounding_collate_batch_fn is not None

            tmp = load_pickle(dicom_id_to_pos_neg_facts_filepath)
            fact_embeddings = tmp['fact_embeddings']
            dicom_id_to_pos_neg_facts = tmp['dicom_id_to_pos_neg_facts']
            print(f'fact_embeddings.shape = {fact_embeddings.shape}')

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
            positive_facts = [None] * BIG_ENOGUGH
            negative_facts = [None] * BIG_ENOGUGH
            image_path_getter = get_image_path_getter(source_image_size_mode, verbose=True)

            mimiccxr_metadata = load_mimiccxr_reports_detailed_metadata()

            idx = 0
            for rid, (part_id, subject_id, study_id, dicom_id_view_pairs) in \
                tqdm(enumerate(zip(mimiccxr_metadata['part_ids'],
                    mimiccxr_metadata['subject_ids'],
                    mimiccxr_metadata['study_ids'],
                    mimiccxr_metadata['dicom_id_view_pos_pairs'])), mininterval=2):
                for dicom_id, view in get_dicom_id_and_orientation_list(dicom_id_view_pairs, MIMICCXR_ViewModes.ALL):
                    if dicom_id in forbidden_train_dicom_ids:
                        continue
                    if dicom_id not in dicom_id_to_pos_neg_facts:
                        continue
                    image_paths[idx] = image_path_getter(part_id, subject_id, study_id, dicom_id)
                    pos_neg_facts = dicom_id_to_pos_neg_facts[dicom_id]
                    assert len(pos_neg_facts) == 2
                    positive_facts[idx] = pos_neg_facts[0]
                    negative_facts[idx] = pos_neg_facts[1]
                    idx += 1

            print(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
            positive_facts = positive_facts[:idx]
            negative_facts = negative_facts[:idx]

            # Create a mapping from number of facts to indices
            num_facts_2_idxs = {}
            for i, (pos_facts, neg_facts) in enumerate(zip(self.train_positive_facts, self.train_negative_facts)):
                assert len(pos_facts) > 0
                assert len(neg_facts) > 0
                num_facts = min(len(pos_facts), len(neg_facts), max_phrases_per_image)
                try:
                    num_facts_2_idxs[num_facts].append(i)
                except KeyError:
                    num_facts_2_idxs[num_facts] = [i]
            
            # Create dataset and dataloader for training
            self.train_fact_datasets = []
            self.train_fact_dataloaders = []
            for num_facts, indices in num_facts_2_idxs.items():
                print(f'Number of facts: {num_facts}, # images: {len(indices)}')
                dataset = MIMICCXR_FactGroundingDataset(
                    image_paths=image_paths, image_transform=train_image_transform,
                    fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                    indices=indices, num_facts=num_facts)
                batch_size = max(min(max_images_per_batch, max_phrases_per_batch // num_facts), 1) # at least 1 image per batch
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_train_workers,
                    collate_fn=fact_grounding_collate_batch_fn,
                    pin_memory=True,
                )
                self.train_fact_datasets.append(dataset)
                self.train_fact_dataloaders.append(dataloader)

        # Create train chest imagenome dataset
        if use_chest_imagenome_for_train:
            print_bold('Preparing Chest Imagenome dataset and dataloader for training...')

            assert chest_imagenome_bbox_phrase_embeddings_filepath is not None
            assert mask_width is not None
            assert mask_height is not None
            assert bbox_grounding_collate_batch_fn is not None

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
                    phrase_grounding_masks[idx] = phrase_grounding_masks_array[i]
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

        # Create mscxr test dataset and dataloader
        if use_mscxr_for_test:
            print_bold('Preparing MS-CXR dataset and dataloader for testing...')
            assert mask_width is not None
            assert mask_height is not None
            assert mscxr_phrase2embedding_filepath is not None
            assert phrase_grounding_collate_batch_fn is not None
            
            dicom_id_2_phrases_and_masks = get_ms_cxr_dicom_id_2_phrases_and_masks(mask_height, mask_width)
            print(f'len(dicom_id_2_phrases_and_masks) = {len(dicom_id_2_phrases_and_masks)}')

            phrase2embedding = load_pickle(mscxr_phrase2embedding_filepath)
            print(f'len(phrase2embedding) = {len(phrase2embedding)}')

            BIG_ENOGUGH = 1000000
            image_paths = [None] * BIG_ENOGUGH
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
                    phrases, masks = dicom_id_2_phrases_and_masks[dicom_id]
                    phrase_embeddings[idx] = np.array([phrase2embedding[phrase] for phrase in phrases])
                    phrase_grounding_masks[idx] = np.array(masks)
                    idx += 1

            print(f'Total number of images: {idx}')
            image_paths = image_paths[:idx]
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
            self.test_mscxr_datasets = []
            self.test_mscxr_dataloaders = []
            for num_phrases, indices in num_phrases_2_idxs.items():
                print(f'Number of phrases: {num_phrases}, # images: {len(indices)}')
                dataset = MIMICCXR_PhraseGroundingDataset(
                    image_paths=image_paths, image_transform=test_image_transform,
                    phrase_embeddings=phrase_embeddings, phrase_grounding_masks=phrase_grounding_masks,
                    indices=indices)
                batch_size = max(min(max_images_per_batch, max_phrases_per_batch // num_phrases), 1) * test_batch_size_factor
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_test_workers,
                    collate_fn=phrase_grounding_collate_batch_fn,
                    pin_memory=True,
                )
                self.test_mscxr_datasets.append(dataset)
                self.test_mscxr_dataloaders.append(dataloader)
        
        # Create chest imagenome test dataset and dataloader
        if use_chest_imagenome_gold_for_test:
            print_bold('Preparing Chest Imagenome dataset and dataloader for testing...')
            assert mask_width is not None
            assert mask_height is not None
            assert chest_imagenome_bbox_phrase_embeddings_filepath is not None
            assert bbox_grounding_collate_batch_fn is not None

            _, gold_pres_indices = get_chest_imagenome_gold_bbox_coords_and_presence_sorted_indices()

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
                                                                                    bbox_presence[gold_pres_indices])
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
            batch_size = max(min(max_images_per_batch, max_phrases_per_batch // len(bbox_phrases)), 1) * test_batch_size_factor
            self.test_chest_imagenome_gold_dataloader = DataLoader(
                self.test_chest_imagenome_gold_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_test_workers,
                collate_fn=lambda batch: bbox_grounding_collate_batch_fn(batch, training_mode=False),
                pin_memory=True,
            )