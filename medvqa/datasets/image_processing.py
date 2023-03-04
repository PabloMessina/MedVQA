import torch
import torchvision.transforms as T
import torchxrayvision as xrv
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import os
import imagesize

from transformers import ViTFeatureExtractor

from medvqa.models.vision import (
    ImageQuestionClassifier,
    ImageFeatureExtractor,
)
from medvqa.models.vision.visual_modules import (
    CLIP_DEFAULT_IMAGE_MEAN_STD,
    CLIP_VERSION_2_IMAGE_MEAN_STD,
)
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.files import MAX_FILENAME_LENGTH, load_pickle, save_to_pickle
from medvqa.utils.hashing import hash_string
from medvqa.datasets.augmentation import (
    ImageAugmentationTransforms,
    ImageBboxAugmentationTransforms,
)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

_AUGMENTATION_MODES = [
    'random-color',
    'random-spatial',
    'random-color-and-spatial',
    'horizontal-flip',
]

def get_image_transform(
    image_size = (256, 256),
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
    augmentation_mode = None,
    default_prob=0.5,
    use_clip_transform=False,
    clip_version=None,
    use_huggingface_vitmodel_transform=False,
    use_torchxrayvision_transform=False,
    huggingface_vitmodel_name=None,
    use_bbox_aware_transform=False,
    horizontal_flip_prob=0,
    use_detectron2_transform=False,
    # detectron2_cfg=None,
):
    print('get_image_transform()')
    assert 0 <= horizontal_flip_prob < 1
    assert 0 <= default_prob <= 1

    # Only one of the following can be true
    assert sum([use_clip_transform, use_huggingface_vitmodel_transform, use_torchxrayvision_transform,
                use_bbox_aware_transform, use_detectron2_transform]) <= 1

    if use_clip_transform:
        assert clip_version is not None
        print(f'Using CLIP transform for version {clip_version}')
        tf_load_image = T.Lambda(lambda x: Image.open(x).convert('RGB'))
        tf_resize = T.Resize(image_size, interpolation=BICUBIC)
        mean, std = CLIP_VERSION_2_IMAGE_MEAN_STD.get(clip_version, CLIP_DEFAULT_IMAGE_MEAN_STD)
        tf_normalize = T.Normalize(mean, std)

    elif use_huggingface_vitmodel_transform:
        assert huggingface_vitmodel_name is not None
        print(f'Using Huggingface ViT model transform for {huggingface_vitmodel_name}')
        feature_extractor = ViTFeatureExtractor.from_pretrained(huggingface_vitmodel_name, use_auth_token=True)
        if type(feature_extractor.size) is int:
            image_size = feature_extractor.size
        elif "shortest_edge" in feature_extractor.size:
            image_size = feature_extractor.size["shortest_edge"]
        else:
            image_size = (feature_extractor.size["height"], feature_extractor.size["width"])
        if type(image_size) is int:
            image_size = (image_size, image_size)
        tf_load_image = T.Lambda(lambda x: Image.open(x).convert('RGB'))
        tf_resize = T.Resize(image_size, interpolation=BICUBIC)
        tf_normalize = T.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    elif use_torchxrayvision_transform:
        assert image_size[0] == image_size[1]
        assert augmentation_mode is None # augmentation not supported. TODO: add support
        print(f'Using torchxrayvision transform (image_size = {image_size})')
        from skimage.io import imread
        tf = T.Compose([
            imread, # read image
            lambda x: xrv.datasets.normalize(x, 255, True), # normalize
            xrv.datasets.XRayResizer(image_size[0], 'cv2'), # resize
            torch.from_numpy, # convert to tensor
        ])
        return tf

    elif use_bbox_aware_transform:
        print(f'  Using bounding box aware transforms')
        tf_load_image = T.Lambda(lambda x: cv2.imread(x))
        # tf_bgr2rgb = T.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        tf_resize = T.Lambda(lambda x: cv2.resize(x, image_size, interpolation=cv2.INTER_CUBIC))
        tf_hflip = T.Lambda(lambda x: cv2.flip(x, 1))
        tf_totensor = T.ToTensor()
        tf_normalize = T.Normalize(mean, std)

        def _default_transform(image_path):
            image = tf_load_image(image_path)
            # image = tf_bgr2rgb(image)
            image = tf_resize(image)
            image = tf_totensor(image)
            image = tf_normalize(image)
            # assert len(image.shape) == 3 # (C, H, W)
            return image

        if augmentation_mode is None: # no augmentation
            print('    Returning default transform (no augmentation)')
            return _default_transform

        if augmentation_mode == 'horizontal-flip':
            assert horizontal_flip_prob > 0
            print('    Returning horizontal flip transform')
            def _transform(image_path, bboxes, presence, flipped_bboxes, flipped_presence, _):
                image = tf_load_image(image_path)
                image = tf_resize(image)
                if random.random() < horizontal_flip_prob:
                    image = tf_hflip(image)
                    bboxes = flipped_bboxes
                    presence = flipped_presence
                image = tf_totensor(image)
                image = tf_normalize(image)
                return image, bboxes, presence
            return _transform
        
        img_bbox_aug_transfoms = ImageBboxAugmentationTransforms(image_size)
        if augmentation_mode == 'random-color':
            aug_transforms = img_bbox_aug_transfoms.get_color_transforms_list()
        elif augmentation_mode == 'random-spatial':
            aug_transforms = img_bbox_aug_transfoms.get_spatial_transforms_list()
        elif augmentation_mode == 'random-color-and-spatial':
            aug_transforms = img_bbox_aug_transfoms.get_merged_spatial_color_transforms_list()
        else:
            raise ValueError(f'Invalid augmentation_mode: {augmentation_mode}')

        flip_image = 'spatial' in augmentation_mode and horizontal_flip_prob > 0        
        # DEBUG = True
        # DEBUG_COUNT = 0
        def _get_transform(tf_img_bbox_aug): # closure (needed to capture tf_img_bbox_aug)
            def _transform(image_path, bboxes, presence, flipped_bboxes, flipped_presence, albumentation_adapter):
                image = tf_load_image(image_path)
                # image = tf_bgr2rgb(image)
                image = tf_resize(image)
                if flip_image:
                    if random.random() < horizontal_flip_prob:
                        # nonlocal DEBUG_COUNT, DEBUG
                        # if DEBUG:
                        #     print('(DEBUG) image_processing.py: flipping image')
                        #     DEBUG_COUNT += 1
                        #     if DEBUG_COUNT > 10:
                        #         DEBUG = False
                        image = tf_hflip(image)
                        bboxes = flipped_bboxes
                        presence = flipped_presence
                bboxes, category_ids = albumentation_adapter.encode(bboxes, presence)
                augmented = tf_img_bbox_aug(image=image, bboxes=bboxes, category_ids=category_ids)
                image = augmented['image']
                image = tf_totensor(image)
                image = tf_normalize(image)
                bboxes = augmented['bboxes']
                category_ids = augmented['category_ids']
                bboxes, presence = albumentation_adapter.decode(bboxes, category_ids)
                assert len(image.shape) == 3 # (C, H, W)
                return image, bboxes, presence
            return _transform

        _augmented_bbox_transforms = [_get_transform(tf_img_bbox_aug) for tf_img_bbox_aug in aug_transforms]
        
        print('    len(_augmented_bbox_transforms) =', len(_augmented_bbox_transforms))
        print('    augmentation_mode =', augmentation_mode)
        print('    default_prob =', default_prob)
        print('    horizontal_flip_prob =', horizontal_flip_prob)
        print('    flip_image =', flip_image)

        def transform_fn(img, bboxes, presence, flipped_bboxes, flipped_presence, albumentation_adapter):
            # randomly choose between default transform and augmented transform
            if random.random() < default_prob:
                img = _default_transform(img)
                return img, bboxes, presence
            return random.choice(_augmented_bbox_transforms)(
                img, bboxes, presence, flipped_bboxes, flipped_presence, albumentation_adapter)

        print(f'    Returning augmented transforms with mode {augmentation_mode}')
        return transform_fn

    elif use_detectron2_transform:
        print(f'  Using detectron2 aware transforms')
        # assert detectron2_cfg is not None
        tf_load_image = lambda x: cv2.imread(x)
        # if detectron2_cfg.INPUT.FORMAT == 'BGR':
        #     pass
        # else:
        #     raise ValueError(f'Unexpected detectron2_cfg.INPUT.FORMAT: {detectron2_cfg.INPUT.FORMAT}')
        # tf_totensor = T.Lambda(lambda x: torch.from_numpy(x).permute(2, 0, 1).float())
        tf_totensor = lambda x: torch.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1)))
        
        def _default_transform(image_path):
            image = tf_load_image(image_path)
            image = tf_totensor(image)
            return image

        img_bbox_aug_transfoms = ImageBboxAugmentationTransforms(image_size)
        
        if augmentation_mode is None: # no augmentation
            print('    Returning default transform (no augmentation)')
            return _default_transform                
        elif augmentation_mode == 'horizontal-flip':
            raise NotImplementedError('horizontal-flip not implemented for detectron2 transforms')        
        elif augmentation_mode == 'random-color':
            raise NotImplementedError('random-color not implemented for detectron2 transforms')        
        elif augmentation_mode == 'random-color-and-spatial':
            raise NotImplementedError('random-color-and-spatial not implemented for detectron2 transforms')
        elif augmentation_mode == 'random-spatial':
            aug_transforms = img_bbox_aug_transfoms.get_merged_spatial_color_transforms_list()
        else:
            raise ValueError(f'Invalid augmentation_mode: {augmentation_mode}')

        def _get_transform(tf_img_bbox_aug): # closure (needed to capture tf_img_bbox_aug)
            def _transform(image_path, bboxes, presence, albumentation_adapter):
                image = tf_load_image(image_path)
                bboxes, category_ids = albumentation_adapter.encode(bboxes, presence)
                augmented = tf_img_bbox_aug(image=image, bboxes=bboxes, category_ids=category_ids)
                image = augmented['image']
                image = tf_totensor(image)                
                bboxes = augmented['bboxes']
                category_ids = augmented['category_ids']
                bboxes, presence = albumentation_adapter.decode(bboxes, category_ids)
                assert len(image.shape) == 3 # (C, H, W)
                return image, bboxes, presence
            return _transform

        _augmented_bbox_transforms = [_get_transform(tf_img_bbox_aug) for tf_img_bbox_aug in aug_transforms]

        def transform_fn(img, bboxes, presence, albumentation_adapter):
            # randomly choose between default transform and augmented transform
            if random.random() < default_prob:
                img = _default_transform(img)
                return img, bboxes, presence
            return random.choice(_augmented_bbox_transforms)(img, bboxes, presence, albumentation_adapter)

        print(f'    Returning augmented transforms with mode {augmentation_mode}')
        return transform_fn

    else:
        print(f'Using standard transform')
        tf_load_image = T.Lambda(lambda x: Image.open(x).convert('RGB'))
        tf_resize = T.Resize(image_size)
        tf_normalize = T.Normalize(mean, std)

    if type(image_size) is int:
        use_center_crop = True
    if type(image_size) is list or type(image_size) is tuple:
        use_center_crop = False
        assert len(image_size) == 2
    else: assert False

    print(f'mean = {mean}, std = {std}, image_size = {image_size}, use_center_crop = {use_center_crop}')
    
    tf_totensor = T.ToTensor()
    
    if use_center_crop:
        tf_ccrop = T.CenterCrop(image_size)
        default_transform = T.Compose([tf_load_image, tf_resize, tf_ccrop, tf_totensor, tf_normalize])
    else:
        default_transform = T.Compose([tf_load_image, tf_resize, tf_totensor, tf_normalize])

    if augmentation_mode is None:
        print('Returning transform without augmentation')
        return default_transform
    
    assert augmentation_mode in _AUGMENTATION_MODES, f'Unknown augmentation mode {augmentation_mode}'

    image_aug_transforms = ImageAugmentationTransforms(image_size)

    if augmentation_mode == 'random-color':
        aug_transforms = image_aug_transforms.get_color_transforms()
        if use_center_crop:
            final_transforms = [
                T.Compose([tf_load_image, tf_resize, tf_ccrop, tf_aug, tf_totensor, tf_normalize])
                for tf_aug in aug_transforms
            ]
        else:
            final_transforms = [
            T.Compose([tf_load_image, tf_resize, tf_aug, tf_totensor, tf_normalize])
            for tf_aug in aug_transforms
        ]
    elif augmentation_mode == 'random-spatial':
        aug_transforms = image_aug_transforms.get_spatial_transforms()
        if use_center_crop:
            final_transforms = [
                T.Compose([tf_load_image, tf_resize, tf_ccrop, tf_aug, tf_totensor, tf_normalize])
                for tf_aug in aug_transforms
            ]
        else:
            final_transforms = [
            T.Compose([tf_load_image, tf_resize, tf_aug, tf_totensor, tf_normalize])
            for tf_aug in aug_transforms
        ]
    elif augmentation_mode == 'random-color-and-spatial':
        spatial_transforms = image_aug_transforms.get_spatial_transforms()
        color_transforms = image_aug_transforms.get_color_transforms()
        final_transforms = []
        for stf in spatial_transforms:
            for ctf in color_transforms:
                if use_center_crop:
                    final_transforms.append(T.Compose([tf_load_image, tf_resize, tf_ccrop, stf, ctf, tf_totensor, tf_normalize]))
                else:
                    final_transforms.append(T.Compose([tf_load_image, tf_resize, stf, ctf, tf_totensor, tf_normalize]))
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

def get_pretrain_vit_mae_image_transform(feature_extractor):
    # Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/run_mae.py#L299
    if type(feature_extractor.size) is int:
        size = feature_extractor.size
    elif "shortest_edge" in feature_extractor.size:
        size = feature_extractor.size["shortest_edge"]
    else:
        size = (feature_extractor.size["height"], feature_extractor.size["width"])
    transforms = T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ]
    )
    return transforms

inv_normalize = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

class ImageDataset(Dataset):
    def __init__(self, image_paths, image_transform):
        self.image_paths = image_paths
        self.image_transform = image_transform    
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, i):
        return {'i': self.image_transform(self.image_paths[i]) }

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


class _ImageSizeCache():
    def __init__(self):
        self._image_size_dict = {}
        self._image_size_cache_path = os.path.join(CACHE_DIR, 'image_size_cache.pkl')
        self._dirty = False
        self._loaded = False

    def get_image_size(self, image_paths, update_cache_on_disk=False):
        assert type(image_paths) == list or type(image_paths) == str

        if not self._loaded:
            self._image_size_dict = load_pickle(self._image_size_cache_path)
            if self._image_size_dict is None:
                self._image_size_dict = {}
            else:
                print('image_size_cache loaded from', self._image_size_cache_path)
            self._loaded = True
        
        if type(image_paths) == str:
            image_path = image_paths
            try:
                image_size = self._image_size_dict[image_path]
            except KeyError:
                image_size = self._image_size_dict[image_path] = imagesize.get(image_path)
                self._dirty = True
            output = image_size
        else:
            assert len(image_paths) > 0
            image_size_list = [None] * len(image_paths)    
            for i, image_path in enumerate(image_paths):
                try:
                    image_size_list[i] = self._image_size_dict[image_path]
                except KeyError:
                    image_size_list[i] = self._image_size_dict[image_path] = imagesize.get(image_path)
                    self._dirty = True
            output = image_size_list
        if update_cache_on_disk:
            self.update_cache_on_disk()
        return output

    def update_cache_on_disk(self):
        if self._dirty:
            save_to_pickle(self._image_size_dict, self._image_size_cache_path)
            print('image size cache saved to', self._image_size_cache_path)
            self._dirty = False

image_size_cache = _ImageSizeCache()