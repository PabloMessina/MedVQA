import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import os

from medvqa.models.vision import (
    ImageQuestionClassifier,
    ImageFeatureExtractor,
)
from medvqa.models.vision.visual_modules import (
    CLIP_DEFAULT_IMAGE_MEAN_STD,
    CLIP_VERSION_2_IMAGE_MEAN_STD,
)
from medvqa.utils.files import MAX_FILENAME_LENGTH, load_pickle, save_to_pickle
from medvqa.utils.hashing import hash_string
from medvqa.datasets.augmentation import ImageAugmentationTransforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

_AUGMENTATION_MODES = [
    'random-color',
    'random-spatial',
    'random-color-and-spatial',
]

def get_image_transform(
    image_size = (256, 256),
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
    augmentation_mode = None,
    default_prob=0.3,
    use_clip_transform=False,
    clip_version=None,
):

    if type(image_size) is int:
        use_center_crop = True
    elif type(image_size) is list or type(image_size) is tuple:
        use_center_crop = False
        assert len(image_size) == 2
    else: assert False    

    if use_clip_transform:
        tf_resize = T.Resize(image_size, interpolation=BICUBIC)
        mean, std = CLIP_VERSION_2_IMAGE_MEAN_STD.get(clip_version, CLIP_DEFAULT_IMAGE_MEAN_STD)
        tf_normalize = T.Normalize(mean, std)
    else:
        tf_resize = T.Resize(image_size)
        tf_normalize = T.Normalize(mean, std)

    print(f'mean = {mean}, std = {std}')
    
    tf_totensor = T.ToTensor()
    
    if use_center_crop:
        tf_ccrop = T.CenterCrop(image_size)
        default_transform = T.Compose([tf_resize, tf_ccrop, tf_totensor, tf_normalize])
    else:
        default_transform = T.Compose([tf_resize, tf_totensor, tf_normalize])

    if augmentation_mode is None:
        print('Returning default transform')
        return default_transform
    
    assert augmentation_mode in _AUGMENTATION_MODES, f'Unknown augmentation mode {augmentation_mode}'

    image_aug_transforms = ImageAugmentationTransforms(image_size)

    if augmentation_mode == 'random-color':
        aug_transforms = image_aug_transforms.get_color_transforms()
        if use_center_crop:
            final_transforms = [
                T.Compose([tf_resize, tf_ccrop, tf_aug, tf_totensor, tf_normalize])
                for tf_aug in aug_transforms
            ]
        else:
            final_transforms = [
            T.Compose([tf_resize, tf_aug, tf_totensor, tf_normalize])
            for tf_aug in aug_transforms
        ]
    elif augmentation_mode == 'random-spatial':
        aug_transforms = image_aug_transforms.get_spatial_transforms()
        if use_center_crop:
            final_transforms = [
                T.Compose([tf_resize, tf_ccrop, tf_aug, tf_totensor, tf_normalize])
                for tf_aug in aug_transforms
            ]
        else:
            final_transforms = [
            T.Compose([tf_resize, tf_aug, tf_totensor, tf_normalize])
            for tf_aug in aug_transforms
        ]
    elif augmentation_mode == 'random-color-and-spatial':
        spatial_transforms = image_aug_transforms.get_spatial_transforms()
        color_transforms = image_aug_transforms.get_color_transforms()
        final_transforms = []
        for stf in spatial_transforms:
            for ctf in color_transforms:
                if use_center_crop:
                    final_transforms.append(T.Compose([tf_resize, tf_ccrop, stf, ctf, tf_totensor, tf_normalize]))
                else:
                    final_transforms.append(T.Compose([tf_resize, stf, ctf, tf_totensor, tf_normalize]))
    else:
        assert False

    print('len(final_transforms) =', len(final_transforms))
    print('default_prob =', default_prob)

    def transform_fn(img):
        if random.random() < default_prob:
            return default_transform(img)
        return random.choice(final_transforms)(img)

    print(f'Returning augmented transforms with mode {augmentation_mode}')
    return transform_fn

inv_normalize = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

class ImageDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform    
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        return {'i': self.transform(Image.open(self.images[i]).convert('RGB')) }

def classify_and_rank_questions(image_paths, transform, image_local_feat_size, n_questions, pretrained_weights, batch_size,
        top_k, threshold, min_num_q_per_report=5):

    print(f'** Ranking the top {top_k} questions per image according to image classifier:')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = ImageQuestionClassifier(image_local_feat_size, n_questions)
    classifier = classifier.to(device)
    classifier.load_state_dict(pretrained_weights, strict=False)
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    print('len(dataset) =',len(dataset))
    print('len(dataloader) =',len(dataloader))
    print('batch_size =',batch_size)
    questions = [None] * len(image_paths)
    question_ids = list(range(n_questions))
    assert top_k <= n_questions
    i = 0
    K = min(min_num_q_per_report, top_k)
    with torch.set_grad_enabled(False):
        classifier.train(False)        
        for batch in tqdm(dataloader):
            images = batch['i'].to(device)
            logits = classifier(images)                        
            logits = logits.detach()
            probs = torch.sigmoid(logits)
            assert probs.size(1) == n_questions
            for j in range(probs.size(0)):
                question_ids.sort(key=lambda k:probs[j][k], reverse=True)
                # questions[i + j] =  question_ids[:top_k]
                questions[i + j] = [qid for k, qid in enumerate(question_ids) if k  < top_k and probs[j][qid] >= threshold]
                if len(questions[i+j]) < K:
                    questions[i + j] =  question_ids[:K]
                assert 0 < len(questions[i + j]) <= top_k
            i += probs.size(0)
    print('average num of questions per report:', sum(len(q) for q in questions) / len(questions))

    del classifier
    del dataset
    del dataloader
    del logits
    torch.cuda.empty_cache()

    return questions

def get_nearest_neighbors(target_images, reference_images, transform, pretrained_weights, batch_size,
                          cache_dir, pretrained_checkpoint_path, suffix=None):

    strings = [f'pretrained_checkpoint_path={pretrained_checkpoint_path}']
    if suffix is not None: strings.append(suffix)
    file_path = os.path.join(cache_dir, f'nearest_neighbors({";".join(strings)})')    
    if len(file_path) > MAX_FILENAME_LENGTH:
        h = hash_string(file_path)
        file_path = os.path.join(cache_dir, f'nearest_neighbors(hash={h[0]},{h[1]}).pkl')
    nearest_neighbors = load_pickle(file_path)
    if nearest_neighbors is not None:
        print('nearest_neighbors loaded from', file_path)
        return nearest_neighbors    

    print(f'** Finding the nearest neighbors for {len(target_images)} images from among {len(reference_images)} images:')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageFeatureExtractor()
    model = model.to(device)
    model.load_state_dict(pretrained_weights, strict=False)
    dataset = ImageDataset(target_images + reference_images, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    print('len(dataset) =',len(dataset))
    print('len(dataloader) =',len(dataloader))
    print('batch_size =', batch_size)    
    
    n_targ = len(target_images)
    n_ref = len(reference_images)
    n = len(dataset)
    assert n == n_targ + n_ref
    offset = 0
    feat = None
    
    print('\textracting features ...')
    with torch.set_grad_enabled(False):
        model.train(False)
        for batch in tqdm(dataloader):
            images = batch['i'].to(device)
            batch_feat = model(images)
            batch_feat = batch_feat.detach()
            if feat is None:
                feat_size = batch_feat.size(1)
                feat = np.empty((n, feat_size), dtype=float)
                print('feat.shape =', feat.shape)
            actual_batch_size = batch_feat.size(0)
            feat[offset : offset + actual_batch_size] = batch_feat.cpu()
            offset += actual_batch_size
    
    print('\tfinding nearest neighbors (euclidean distance) ...')
    nearest_neighbors = [None] * n_targ
    target_feat = feat[:n_targ]
    ref_feat = feat[n_targ:]
    # TODO: parallelize this
    for i in tqdm(range(n_targ)):
        dist = (ref_feat - target_feat[i])**2
        dist = dist.sum(axis=1)
        assert dist.shape == (n_ref,)
        _, j = min((d,j) for j,d in enumerate(dist))
        nearest_neighbors[i] = j
    
    del model
    del dataset
    del dataloader
    del batch_feat
    torch.cuda.empty_cache()

    save_to_pickle(nearest_neighbors, file_path)
    print('nearest_neighbors saved to', file_path)
    return nearest_neighbors