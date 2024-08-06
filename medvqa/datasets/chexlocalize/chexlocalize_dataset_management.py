from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.chexlocalize import extract_images_segmentation_masks_and_binary_labels
from medvqa.utils.files import get_cached_pickle_file, load_pickle
from medvqa.utils.logging import print_bold
    
class CheXLocalizePhraseGroundingDataset(Dataset):

    def __init__(self, indices, image_paths, image_transform, phrase_embeddings, phrase_grounding_masks, phrase_classification_labels):
        self.indices = indices
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_embeddings = phrase_embeddings
        self.phrase_grounding_masks = phrase_grounding_masks
        self.phrase_classification_labels = phrase_classification_labels

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        i = self.indices[i]
        image_path = self.image_paths[i]
        phrase_embeddings = self.phrase_embeddings
        phrase_grounding_masks = self.phrase_grounding_masks[i]
        phrase_classification_labels = self.phrase_classification_labels[i]
        image, phrase_grounding_masks, phrase_classification_labels = self.image_transform(
            image_path, phrase_grounding_masks, phrase_classification_labels)
        return {
            'i': image,
            'pe': phrase_embeddings,
            'pgm': phrase_grounding_masks,
            'pcl': phrase_classification_labels,
        }

class CheXlocalizePhraseGroundingTrainer:
    def __init__(self, collate_batch_fn, mask_height, mask_width, class_phrase_embeddings_filepath,
                 max_images_per_batch, max_phrases_per_batch, test_batch_size_factor=1,
                 use_training_set=True, use_validation_set=True,
                 train_image_transform=None, val_image_transform=None,
                 num_train_workers=None, num_val_workers=None,
                 use_interpret_cxr_challenge_split=False, interpret_cxr_challenge_split_filepath=None):
        
        assert use_training_set or use_validation_set
        
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform

        print(f'Loding class_phrase_embeddings_filepath and bbox_phrases from {class_phrase_embeddings_filepath}...')
        tmp = get_cached_pickle_file(class_phrase_embeddings_filepath)
        class_phrase_embeddings = tmp['class_phrase_embeddings']
        class_phrases = tmp['class_phrases']
        assert class_phrase_embeddings.shape[0] == len(class_phrases)
        self.class_phrase_embeddings = class_phrase_embeddings
        self.class_phrases = class_phrases
        print(f'class_phrase_embeddings.shape = {class_phrase_embeddings.shape}')
        print(f'len(class_phrases) = {len(class_phrases)}')
        for phrase in class_phrases:
            print('\t', phrase)

        print('Compute phrase grounding masks and labels')
        self.image_paths, self.phrase_grounding_masks, self.phrase_classification_labels = \
            extract_images_segmentation_masks_and_binary_labels(mask_height=mask_height, mask_width=mask_width,
                                                                flatten_masks=True)
        
        if use_interpret_cxr_challenge_split:
            assert interpret_cxr_challenge_split_filepath is not None
            print_bold(f'Using split from {interpret_cxr_challenge_split_filepath}')
            challenge_split = load_pickle(interpret_cxr_challenge_split_filepath)
            all_image_partial_paths = []
            for ip in self.image_paths:
                if 'train/' in ip:
                    all_image_partial_paths.append(ip[ip.index('train/'):])
                elif 'val/' in ip:
                    all_image_partial_paths.append(ip[ip.index('val/'):])
                elif 'test/' in ip:
                    all_image_partial_paths.append(ip[ip.index('test/'):])
                else:
                    raise ValueError(f'Unknown image path: {ip}')
            image_partial_path_2_idx = {p: i for i, p in enumerate(all_image_partial_paths)}

        if use_training_set:
            assert train_image_transform is not None
            assert num_train_workers is not None
            print('Generating train dataset and dataloader')
            if use_interpret_cxr_challenge_split:
                train_indices = [image_partial_path_2_idx[p] for p in challenge_split['train'] if p in image_partial_path_2_idx]
                assert len(train_indices) > 0, 'No training images found in the split'
            else:
                train_indices = list(range(len(self.image_paths)))
            print(f'len(train_indices) = {len(train_indices)}')
            self.train_dataset = CheXLocalizePhraseGroundingDataset(
                indices=train_indices,
                image_paths=self.image_paths,
                image_transform=self.train_image_transform,
                phrase_embeddings=class_phrase_embeddings,
                phrase_grounding_masks=self.phrase_grounding_masks,
                phrase_classification_labels=self.phrase_classification_labels,
            )
            batch_size = max(min(max_images_per_batch, max_phrases_per_batch // len(class_phrases)), 1) # at least 1 image per batch
            print(f'batch_size = {batch_size}')
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_train_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
        
        if use_validation_set:
            assert val_image_transform is not None
            assert num_val_workers is not None
            print('Generating val dataset and dataloader')
            if use_interpret_cxr_challenge_split:
                val_indices = [image_partial_path_2_idx[p] for p in challenge_split['val'] if p in image_partial_path_2_idx]
                assert len(val_indices) > 0, 'No validation images found in the split'
            else:
                val_indices = list(range(len(self.image_paths)))
            print(f'len(val_indices) = {len(val_indices)}')
            self.val_dataset = CheXLocalizePhraseGroundingDataset(
                indices=val_indices,
                image_paths=self.image_paths,
                image_transform=self.val_image_transform,
                phrase_embeddings=class_phrase_embeddings,
                phrase_grounding_masks=self.phrase_grounding_masks,
                phrase_classification_labels=self.phrase_classification_labels,
            )
            batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // len(class_phrases)), 1) * test_batch_size_factor)
            print(f'batch_size = {batch_size}')
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_val_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
