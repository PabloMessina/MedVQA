from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.chexlocalize import extract_images_segmentation_masks_and_binary_labels
from medvqa.utils.files import get_cached_pickle_file
    
class CheXLocalizePhraseGroundingDataset(Dataset):

    def __init__(self, image_paths, image_transform, phrase_embeddings, phrase_grounding_masks, phrase_classification_labels):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_embeddings = phrase_embeddings
        self.phrase_grounding_masks = phrase_grounding_masks
        self.phrase_classification_labels = phrase_classification_labels

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = self.image_transform(image_path)
        phrase_embeddings = self.phrase_embeddings
        phrase_grounding_masks = self.phrase_grounding_masks[i]
        phrase_classification_labels = self.phrase_classification_labels[i]
        return {
            'i': image,
            'pe': phrase_embeddings,
            'pgm': phrase_grounding_masks,
            'pcl': phrase_classification_labels,
        }

class CheXlocalizePhraseGroundingTrainer:
    def __init__(self, train_image_transform, val_image_transform, collate_batch_fn, num_train_workers, num_val_workers,
                 mask_height, mask_width, class_phrase_embeddings_filepath,
                 max_images_per_batch, max_phrases_per_batch, test_batch_size_factor,
                 use_training_set=True, use_validation_set=True):
        
        assert use_training_set or use_validation_set
        
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform

        print(f'Loding class_phrase_embeddings_filepath and bbox_phrases from {class_phrase_embeddings_filepath}...')
        tmp = get_cached_pickle_file(class_phrase_embeddings_filepath)
        class_phrase_embeddings = tmp['class_phrase_embeddings']
        class_phrases = tmp['class_phrases']
        assert class_phrase_embeddings.shape[0] == len(class_phrases)
        print(f'class_phrase_embeddings.shape = {class_phrase_embeddings.shape}')
        print(f'len(class_phrases) = {len(class_phrases)}')
        for phrase in class_phrases:
            print('\t', phrase)

        print('Compute phrase grounding masks and labels')
        self.image_paths, self.phrase_grounding_masks, self.phrase_classification_labels = \
            extract_images_segmentation_masks_and_binary_labels(mask_height=mask_height, mask_width=mask_width,
                                                                flatten_masks=True)

        if use_training_set:
            print('Generating train dataset and dataloader')        
            self.train_dataset = CheXLocalizePhraseGroundingDataset(
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
            print('Generating val dataset and dataloader')
            self.val_dataset = CheXLocalizePhraseGroundingDataset(
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
