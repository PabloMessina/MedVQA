from torch.utils.data import DataLoader
from medvqa.datasets.image_processing import FactVisualGroundingDataset
from medvqa.datasets.iuxray import get_invalid_images, get_iuxray_image_path
from medvqa.utils.files import load_pickle
from medvqa.utils.logging import print_bold

class IUXRayPhraseGroundingTrainer:

    def __init__(self, 
                max_images_per_batch, max_phrases_per_batch, max_phrases_per_image,
                num_train_workers=None, num_test_workers=None,
                train_image_transform=None, test_image_transform=None,
                collate_batch_fn=None,
                test_batch_size_factor=1,
                do_train=False,
                do_test=False,
                image_id_to_pos_neg_facts_filepath=None,
                use_interpret_cxr_challenge_split=False,
                interpret_cxr_challenge_split_filepath=None,
                **unused_kwargs,
            ):

        if len(unused_kwargs) > 0:
            # Print warning in orange and bold
            print('\033[93m\033[1mWarning: unused kwargs in MIMICCXR_VisualModuleTrainer: {}\033[0m'.format(unused_kwargs))
        
        # Sanity checks
        assert sum([do_train, do_test]) > 0 # at least one of them must be True

        # Load the data            
        print_bold('Preparing data for training/testing...')
        assert image_id_to_pos_neg_facts_filepath is not None
        assert num_train_workers is not None
        assert train_image_transform is not None

        tmp = load_pickle(image_id_to_pos_neg_facts_filepath)
        fact_embeddings = tmp['embeddings']
        image_id_to_pos_neg_facts = tmp['image_id_to_pos_neg_facts']
        print(f'fact_embeddings.shape = {fact_embeddings.shape}')

        image_ids = list(image_id_to_pos_neg_facts.keys())
        
        # Remove invalid images
        invalid_image_filenames = get_invalid_images()
        n_bef = len(image_ids)
        image_ids = [image_id for image_id in image_ids if f'{image_id}.png' not in invalid_image_filenames]
        n_aft = len(image_ids)
        assert n_aft < n_bef
        print(f'Number of invalid images removed: {n_bef - n_aft}')

        image_paths = [get_iuxray_image_path(image_id) for image_id in image_ids]
        positive_facts = [image_id_to_pos_neg_facts[image_id][0] for image_id in image_ids]
        negative_facts = [image_id_to_pos_neg_facts[image_id][1] for image_id in image_ids]

        if use_interpret_cxr_challenge_split:
            assert interpret_cxr_challenge_split_filepath is not None
            print_bold(f'Using split from {interpret_cxr_challenge_split_filepath}')
            split = load_pickle(interpret_cxr_challenge_split_filepath)
            image_id_2_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}
            if do_train:
                train_indices = [image_id_2_idx[image_id] for image_id in split['train'] if image_id in image_id_2_idx]
                assert len(train_indices) > 0
                print(f'len(train_indices) = {len(train_indices)}')
            if do_test:
                test_indices = [image_id_2_idx[image_id] for image_id in split['val'] if image_id in image_id_2_idx]
                assert len(test_indices) > 0
                print(f'len(test_indices) = {len(test_indices)}')
                # the actual test set is the validation set in the challenge, and the test set is kept hidden
        else:
            if do_train:
                train_indices = list(range(len(image_ids))) #TODO: eventually implement train/test split
            if do_test:
                test_indices = list(range(len(image_ids))) #TODO: eventually implement train/test split

        # Calculate the average number of facts per image
        if do_train:
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
        if do_test:
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
        if do_train:
            print_bold('Building train dataloader...')
            batch_size = max(min(max_images_per_batch, max_phrases_per_batch // train_num_facts_per_image), 1) # at least 1
            train_dataset = FactVisualGroundingDataset(
                image_paths=image_paths, image_transform=train_image_transform,
                fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                indices=train_indices, num_facts=train_num_facts_per_image)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_train_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
            self.train_dataset = train_dataset
            self.train_dataloader = train_dataloader
            print(f'len(self.train_dataloader) = {len(self.train_dataloader)}')

        # Create dataset and dataloader for testing
        if do_test:
            print_bold('Building test dataloader...')
            test_dataset = FactVisualGroundingDataset(
                image_paths=image_paths, image_transform=test_image_transform,
                fact_embeddings=fact_embeddings, positive_facts=positive_facts, negative_facts=negative_facts,
                indices=test_indices, num_facts=test_num_facts_per_image)
            batch_size = int(max(min(max_images_per_batch, max_phrases_per_batch // test_num_facts_per_image), 1) * test_batch_size_factor)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_test_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
            self.test_dataset = test_dataset
            self.test_dataloader = test_dataloader
            print(f'len(self.test_dataloader) = {len(self.test_dataloader)}')