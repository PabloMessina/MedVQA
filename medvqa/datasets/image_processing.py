import math
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
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

from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH
from medvqa.models.vision import (
    ImageQuestionClassifier,
    ImageFeatureExtractor,
)
from medvqa.models.vision.visual_modules import (
    CLIP_DEFAULT_IMAGE_MEAN_STD,
    CLIP_VERSION_2_IMAGE_MEAN_STD,
)
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.files_utils import MAX_FILENAME_LENGTH, load_pickle, save_pickle
from medvqa.utils.hashing_utils import hash_string
from medvqa.datasets.augmentation import (
    ImageAugmentedTransforms,
    ImageBboxAugmentationTransforms,
    ImageSegmentationMaskAugmentationTransforms,
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
]

def get_image_transform(
    image_size=(256, 256),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    mask_height=None,
    mask_width=None,
    augmentation_mode=None,
    default_prob=0.35,
    use_clip_transform=False,
    clip_version=None,
    use_huggingface_vitmodel_transform=False,
    use_torchxrayvision_transform=False,
    huggingface_vitmodel_name=None,
    use_bbox_aware_transform=False,
    use_segmentation_mask_aware_transform=False,
    horizontal_flip_prob=0,
    use_detectron2_transform=False,
    for_yolov8=False,
    for_yolov11=False,
    for_vinbig=False,    
    bbox_format='xyxy',
    # detectron2_cfg=None,
):
    print('get_image_transform()')
    assert 0 <= horizontal_flip_prob < 1
    assert 0 <= default_prob <= 1

    print(f'  image_size = {image_size}')
    print(f'  mean = {mean}')
    print(f'  std = {std}')

    # Only one of the following can be true
    assert sum([use_clip_transform, use_huggingface_vitmodel_transform, use_torchxrayvision_transform,
                use_bbox_aware_transform, use_segmentation_mask_aware_transform, use_detectron2_transform]) <= 1

    if use_clip_transform:
        raise NotImplementedError('CLIP transform not implemented')
        assert clip_version is not None
        print(f'Using CLIP transform for version {clip_version}')
        # tf_load_image = T.Lambda(lambda x: Image.open(x).convert('RGB'))
        # use cv2 instead, to read as RGB
        tf_load_image = T.Lambda(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))
        tf_resize = T.Resize(image_size, interpolation=BICUBIC)
        mean, std = CLIP_VERSION_2_IMAGE_MEAN_STD.get(clip_version, CLIP_DEFAULT_IMAGE_MEAN_STD)
        tf_normalize = T.Normalize(mean, std)

    elif use_huggingface_vitmodel_transform:
        raise NotImplementedError('Huggingface ViT model transform not implemented')
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
        # tf_load_image = T.Lambda(lambda x: Image.open(x).convert('RGB'))
        tf_load_image = T.Lambda(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))
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
        assert bbox_format in ['xyxy', 'cxcywh']
        # Map bbox_format to its equivalent name according to albumentations naming convention
        if bbox_format == 'xyxy':
            bbox_format = 'albumentations'
        elif bbox_format == 'cxcywh':
            bbox_format = 'yolo'
        else: assert False
        image_augmented_transforms = ImageAugmentedTransforms(image_size, mean, std, bbox_format=bbox_format)
        test_transform = image_augmented_transforms.get_test_transform(allow_returning_image_size=for_yolov8 or for_yolov11)

        if augmentation_mode is None: # no augmentation
            print('    Returning default transform (no augmentation)')
            return test_transform
        
        print('    augmentation_mode =', augmentation_mode)
        print('    default_prob =', default_prob)

        if for_vinbig:
            print('    for_vinbig: returning vinbig transform')

            if augmentation_mode == 'random-color':
                train_transform = image_augmented_transforms.get_train_transform('color', bbox_aware=True, for_vinbig=True)
            elif augmentation_mode == 'random-spatial':
                train_transform = image_augmented_transforms.get_train_transform('spatial', bbox_aware=True, for_vinbig=True)
            elif augmentation_mode == 'random-color-and-spatial':
                train_transform = image_augmented_transforms.get_train_transform('both', bbox_aware=True, for_vinbig=True)
            else:
                raise ValueError(f'Invalid augmentation_mode: {augmentation_mode}')

            def transform_fn(image_path, bboxes, classes, albumentation_adapter, return_image_size=False):
                # randomly choose between default transform and augmented transform
                if random.random() < default_prob:
                    if return_image_size:
                        img, size_before, size_after = test_transform(image_path, return_image_size=True)
                        return img, bboxes, classes, size_before, size_after
                    img = test_transform(image_path)
                    return img, bboxes, classes
                return train_transform(
                    image_path=image_path,
                    bboxes=bboxes,
                    classes=classes,
                    albumentation_adapter=albumentation_adapter,
                    return_image_size=return_image_size,
                )

            print(f'    Returning augmented transform with mode {augmentation_mode}')
            return transform_fn
        else:
            print('    returning transform (not for vinbig)')

            if augmentation_mode == 'random-color':
                train_transform = image_augmented_transforms.get_train_transform('color', bbox_aware=True)
                train_transform__vinbig = image_augmented_transforms.get_train_transform('color', bbox_aware=True, for_vinbig=True) # compatible with vinbig-like annotations
                train_transform__no_bbox = image_augmented_transforms.get_train_transform('color', allow_returning_image_size=True) # to be used when no bboxes are present
            elif augmentation_mode == 'random-spatial':
                train_transform = image_augmented_transforms.get_train_transform('spatial', bbox_aware=True)
                train_transform__vinbig = image_augmented_transforms.get_train_transform('spatial', bbox_aware=True, for_vinbig=True) # compatible with vinbig-like annotations
                train_transform__no_bbox = image_augmented_transforms.get_train_transform('spatial', allow_returning_image_size=True) # to be used when no bboxes are present
            elif augmentation_mode == 'random-color-and-spatial':
                train_transform = image_augmented_transforms.get_train_transform('both', bbox_aware=True)
                train_transform__vinbig = image_augmented_transforms.get_train_transform('both', bbox_aware=True, for_vinbig=True) # compatible with vinbig-like annotations
                train_transform__no_bbox = image_augmented_transforms.get_train_transform('both', allow_returning_image_size=True) # to be used when no bboxes are present
            else:
                raise ValueError(f'Invalid augmentation_mode: {augmentation_mode}')

            # def transform_fn(img, **unused):
            #     if random.random() < default_prob:
            #         return test_transform(img) # no augmentation
            #     return train_transform(img) # with augmentation

            def transform_fn(image_path, bboxes=None, classes=None, presence=None, albumentation_adapter=None, return_image_size=False):

                if bboxes is None: # no bboxes -> use different transform
                    assert presence is None
                    assert albumentation_adapter is None
                    # randomly choose between default transform and augmented transform
                    if random.random() < default_prob:
                        if return_image_size:
                            img, size_before, size_after = test_transform(image_path, return_image_size=True)
                            return img, size_before, size_after
                        img = test_transform(image_path)
                        return img
                    return train_transform__no_bbox(
                        image_path=image_path,
                        return_image_size=return_image_size,
                    )

                # randomly choose between default transform and augmented transform
                if random.random() < default_prob:
                    if return_image_size:
                        img, size_before, size_after = test_transform(image_path, return_image_size=True)
                        if presence is not None:
                            return img, bboxes, presence, size_before, size_after
                        else:
                            return img, bboxes, classes, size_before, size_after
                    img = test_transform(image_path)
                    if presence is not None:
                        return img, bboxes, presence
                    else:
                        return img, bboxes, classes
                if presence is not None:
                    return train_transform(
                        image_path=image_path,
                        bboxes=bboxes,
                        presence=presence,
                        albumentation_adapter=albumentation_adapter,
                        return_image_size=return_image_size,
                    )
                else:
                    return train_transform__vinbig( # compatible with vinbig-like annotations
                        image_path=image_path,
                        bboxes=bboxes,
                        classes=classes,
                        albumentation_adapter=albumentation_adapter,
                        return_image_size=return_image_size,
                    )

            print(f'    Returning augmented transform with mode {augmentation_mode}')
            return transform_fn

            # raise NotImplementedError('bbox_aware_transform not implemented')
            # def _get_transform(tf_img_bbox_aug, tf_img_bbox_aug_2): # closure (needed to capture tf_img_bbox_aug)
            #     def _transform(image_path, bboxes=None, albumentation_adapter=None, presence=None,
            #                     flipped_bboxes=None, flipped_presence=None,
            #                     pred_bboxes=None, flipped_pred_bboxes=None, return_image_size=False):
            #         image = tf_load_image(image_path)
            #         if return_image_size:
            #             size_before = image.shape[:2]
            #         # image = tf_bgr2rgb(image)
            #         image = tf_resize(image)
            #         if return_image_size:
            #             size_after = image.shape[:2]
            #         if flip_image:
            #             if bboxes is not None:
            #                 assert flipped_bboxes is not None
            #             if presence is not None:
            #                 assert flipped_presence is not None
            #             if random.random() < horizontal_flip_prob:
            #                 image = tf_hflip(image)
            #                 bboxes = flipped_bboxes
            #                 presence = flipped_presence
            #                 if pred_bboxes is not None:
            #                     assert flipped_pred_bboxes is not None
            #                     pred_bboxes = flipped_pred_bboxes
            #         if bboxes is not None:
            #             bboxes = albumentation_adapter.encode(bboxes, presence)
            #         if pred_bboxes is not None:
            #             pred_bboxes = albumentation_adapter.encode(pred_bboxes)
            #             augmented = tf_img_bbox_aug_2(image=image, bboxes=bboxes, bboxes2=pred_bboxes)
            #         elif bboxes is not None:
            #             augmented = tf_img_bbox_aug(image=image, bboxes=bboxes)
            #         else:
            #             augmented = tf_img_bbox_aug(image=image, bboxes=[]) # no bboxes
            #         image = augmented['image']
            #         image = tf_totensor(image)
            #         image = tf_normalize(image)
            #         # assert len(image.shape) == 3 # (C, H, W)
            #         if bboxes is not None:
            #             bboxes = augmented['bboxes']
            #             bboxes, presence = albumentation_adapter.decode(bboxes)
            #         if pred_bboxes is not None:
            #             pred_bboxes = augmented['bboxes2']
            #             pred_bboxes = albumentation_adapter.decode(pred_bboxes, only_boxes=True)
            #             if return_image_size:
            #                 return image, bboxes, presence, pred_bboxes, size_before, size_after
            #             return image, bboxes, presence, pred_bboxes
            #         elif bboxes is not None:
            #             if return_image_size:
            #                 return image, bboxes, presence, size_before, size_after
            #             return image, bboxes, presence
            #         else:
            #             if return_image_size:
            #                 return image, size_before, size_after
            #             return image
            #     return _transform

            # _augmented_bbox_transforms = [_get_transform(tf, tf2) for tf, tf2 in zip(aug_transforms, aug_transforms_2)]
            
            # print('    len(_augmented_bbox_transforms) =', len(_augmented_bbox_transforms))
            # print('    augmentation_mode =', augmentation_mode)
            # print('    default_prob =', default_prob)
            # print('    horizontal_flip_prob =', horizontal_flip_prob)
            # print('    flip_image =', flip_image)

            # def transform_fn(image_path, bboxes=None, albumentation_adapter=None, presence=None, flipped_bboxes=None, flipped_presence=None,
            #                 pred_bboxes=None, flipped_pred_bboxes=None, return_image_size=False):
            #     # randomly choose between default transform and augmented transform
            #     if random.random() < default_prob:
            #         if return_image_size:
            #             img, size_before, size_after = _default_transform(image_path, return_image_size=True)
            #             if pred_bboxes is not None:
            #                 return img, bboxes, presence, pred_bboxes, size_before, size_after
            #             elif bboxes is not None:
            #                 return img, bboxes, presence, size_before, size_after
            #             else:
            #                 return img, size_before, size_after
            #         img = _default_transform(image_path)
            #         if pred_bboxes is not None:
            #             return img, bboxes, presence, pred_bboxes
            #         elif bboxes is not None:
            #             return img, bboxes, presence
            #         else:
            #             return img
            #     return random.choice(_augmented_bbox_transforms)(
            #         image_path=image_path,
            #         albumentation_adapter=albumentation_adapter,
            #         bboxes=bboxes,
            #         presence=presence,
            #         flipped_bboxes=flipped_bboxes,
            #         flipped_presence=flipped_presence,
            #         pred_bboxes=pred_bboxes,
            #         flipped_pred_bboxes=flipped_pred_bboxes,
            #         return_image_size=return_image_size,
            #     )

            # print(f'    Returning augmented transforms with mode {augmentation_mode}')
            # return transform_fn
        
    elif use_segmentation_mask_aware_transform:

        raise NotImplementedError('segmentation_mask_aware_transform not implemented')

        assert not for_yolov8 # not supported
        assert mask_height is not None
        assert mask_width is not None

        print(f'  Using segmentation mask aware transforms')
        tf_load_image = T.Lambda(lambda x: cv2.imread(x))
        tf_bgr2rgb = T.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        tf_resize = T.Lambda(lambda x: cv2.resize(x, image_size, interpolation=cv2.INTER_CUBIC))
        tf_totensor = T.ToTensor()
        tf_normalize = T.Normalize(mean, std)

        def _default_transform(image_path, *args):
            image = tf_load_image(image_path)
            image = tf_bgr2rgb(image)
            image = tf_resize(image)
            image = tf_totensor(image)
            image = tf_normalize(image)
            if args:
                return image, *args # return image and other arguments
            return image

        if augmentation_mode is None: # no augmentation
            print('    Returning default transform (no augmentation)')
            return _default_transform
        
        img_mask_aug_transfoms = ImageSegmentationMaskAugmentationTransforms(image_size)
        if augmentation_mode == 'random-color':
            aug_transforms = img_mask_aug_transfoms.get_color_transforms_list()
        elif augmentation_mode == 'random-spatial':
            aug_transforms = img_mask_aug_transfoms.get_spatial_transforms_list()
        elif augmentation_mode == 'random-color-and-spatial':
            aug_transforms = img_mask_aug_transfoms.get_merged_spatial_color_transforms_list()
        else:
            raise ValueError(f'Invalid augmentation_mode: {augmentation_mode}')
        
        def _get_transform(tf_img_mask_aug): # closure (needed to capture tf_img_mask_aug)
            def _transform(image_path, masks=None, labels=None):
                # image_path: str, masks: np.ndarray, labels: np.ndarray
                image = tf_load_image(image_path)
                image = tf_bgr2rgb(image)
                image = tf_resize(image)
                if masks is None:
                    assert labels is None
                    augmented = tf_img_mask_aug(image=image) # apply augmentation only to image
                    image = augmented['image']
                    image = tf_totensor(image)
                    image = tf_normalize(image)
                    return image
                assert masks.ndim == 2 # (N, H * W)
                if labels is None:
                    pos_idxs = np.arange(masks.shape[0]) # all masks are positive
                else:
                    assert labels.ndim == 1 # (N,)
                    assert masks.shape[0] <= labels.shape[0], f'masks.shape = {masks.shape}, labels.shape = {labels.shape}'
                    pos_idxs = np.where(labels == 1)[0] # positive masks
                # resize positive masks
                masks_to_augment = [cv2.resize(masks[i].reshape(mask_height, mask_width),
                                               image_size, interpolation=cv2.INTER_NEAREST) for i in pos_idxs]
                augmented = tf_img_mask_aug(image=image, masks=masks_to_augment)
                image = augmented['image']
                image = tf_totensor(image)
                image = tf_normalize(image)
                # resize augmented masks back to original size
                masks_to_augment = augmented['masks']
                final_masks = masks.copy()
                for i, idx in enumerate(pos_idxs): 
                    final_masks[idx] = cv2.resize(masks_to_augment[i], (mask_width, mask_height), interpolation=cv2.INTER_NEAREST).flatten()
                if labels is None:
                    return image, final_masks
                return image, final_masks, labels
            return _transform

        _augmented_mask_transforms = [_get_transform(tf) for tf in aug_transforms]
        
        print('    len(_augmented_mask_transforms) =', len(_augmented_mask_transforms))
        print('    augmentation_mode =', augmentation_mode)
        print('    default_prob =', default_prob)

        def transform_fn(image_path, masks=None, labels=None):
            # randomly choose between default transform and augmented transform
            if random.random() < default_prob:
                img = _default_transform(image_path)
                if masks is None:
                    assert labels is None
                    return img
                if labels is None:
                    return img, masks
                return img, masks, labels
            # randomly choose an augmented transform
            return random.choice(_augmented_mask_transforms)(image_path, masks, labels)

        print(f'    Returning augmented transforms with mode {augmentation_mode}')
        return transform_fn

    elif use_detectron2_transform:
        print(f'  Using detectron2 aware transforms')
        # assert detectr on2_cfg is not None
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

    else: # standard transform
        print(f'Using standard transform (only images, no bounding boxes, no masks)')
        print(f'mean = {mean}, std = {std}, image_size = {image_size}')
        image_augmented_transforms = ImageAugmentedTransforms(image_size=image_size, mean=mean, std=std)
        test_transform = image_augmented_transforms.get_test_transform()
        
        if augmentation_mode is None:
            print('Returning transform without augmentation')
            return lambda img, **unused: test_transform(img)
    
        assert augmentation_mode in _AUGMENTATION_MODES, f'Unknown augmentation mode {augmentation_mode}'

        if augmentation_mode == 'random-color':
            train_transform = image_augmented_transforms.get_train_transform('color')
        elif augmentation_mode == 'random-spatial':
            train_transform = image_augmented_transforms.get_train_transform('spatial')
        elif augmentation_mode == 'random-color-and-spatial':
            train_transform = image_augmented_transforms.get_train_transform('both')
        else:
            raise ValueError(f'Invalid augmentation_mode: {augmentation_mode}')
        
        print('default_prob =', default_prob)

        def transform_fn(img, **unused):
            if random.random() < default_prob:
                return test_transform(img) # no augmentation
            return train_transform(img) # with augmentation

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

# def resize_image(src_image_path, tgt_image_path, new_size, keep_aspect_ratio):
#     image = cv2.imread(src_image_path)
#     if keep_aspect_ratio:
#         # Resize image so that the smallest side is new_size
#         h, w, _ = image.shape
#         if h < w:
#             new_h = new_size
#             new_w = int(w * new_size / h)
#         else:
#             new_w = new_size
#             new_h = int(h * new_size / w)
#         image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
#     else:
#         image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
#     # Save image to new path
#     cv2.imwrite(tgt_image_path, image)

def resize_image(
    src_image_path,
    tgt_image_path,
    new_size,
    keep_aspect_ratio=True,
    jpeg_quality=95,
    png_compression=3,
    interpolation_enlarge=cv2.INTER_CUBIC, # Good default for enlarging
    interpolation_shrink=cv2.INTER_AREA,   # Best default for shrinking
    p_min=1.0, # Percentile min for 16-bit to JPEG windowing
    p_max=99.0, # Percentile max for 16-bit to JPEG windowing
    verbose=False, # Verbose output
):
    """
    Resizes an image using OpenCV with smart interpolation and bit-depth handling.

    Loads various image types (including 16-bit), resizes them while
    optionally preserving aspect ratio (based on the smallest side),
    automatically selects appropriate interpolation (shrinking vs enlarging),
    and saves the output. Performs windowing when converting 16-bit to JPEG.

    Args:
        src_image_path (str): Path to the source image file.
        tgt_image_path (str): Path to save the target resized image file.
                              The format is determined by the extension.
        new_size (int or tuple):
            - If keep_aspect_ratio is True: An integer representing the desired
              size of the *smallest* dimension after resizing.
            - If keep_aspect_ratio is False: A tuple (width, height) representing
              the exact desired output dimensions.
        keep_aspect_ratio (bool, optional): If True, maintains the aspect ratio
                                            by scaling the smallest side to
                                            new_size. If False, resizes to the
                                            exact (width, height) tuple given
                                            in new_size. Defaults to True.
        jpeg_quality (int, optional): Quality setting (0-100) if saving as
                                      JPEG/JPG. Defaults to 95.
        png_compression (int, optional): Compression level (0-9) if saving as
                                         PNG. 0 is no compression, 9 is max.
                                         Defaults to 3.
        interpolation_enlarge (int, optional): OpenCV interpolation flag used
                                               when enlarging the image.
                                               Defaults to cv2.INTER_CUBIC.
        interpolation_shrink (int, optional): OpenCV interpolation flag used
                                              when shrinking the image.
                                              Defaults to cv2.INTER_AREA.
        p_min (float): Lower percentile for windowing when converting 16-bit
                       source to 8-bit JPEG output (0.0-100.0, default 1.0).
        p_max (float): Upper percentile for windowing when converting 16-bit
                       source to 8-bit JPEG output (0.0-100.0, default 99.0).

    Returns:
        None: The function saves the resized image to tgt_image_path.

    Raises:
        FileNotFoundError: If the source image cannot be found.
        ValueError: If new_size parameter is invalid for the chosen mode.
        IOError: If the image cannot be saved.
        Exception: For other potential OpenCV or processing errors.
    """
    # 1. Load image preserving original depth and channels
    image = cv2.imread(src_image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Could not read source image: {src_image_path}")

    # 2. Get original dimensions and data type
    original_dtype = image.dtype
    if image.ndim == 2:
        h, w = image.shape
        is_color = False
    elif image.ndim == 3:
        h, w, _ = image.shape
        is_color = True
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")

    original_pixels = h * w
    if verbose:
        print(f"Loaded '{os.path.basename(src_image_path)}': {w}x{h}, "
                f"Channels: {'Color/Alpha' if is_color else 'Grayscale'}, "
                f"Dtype: {original_dtype}")

    # 3. Calculate target dimensions
    target_w, target_h = 0, 0
    if keep_aspect_ratio:
        if not isinstance(new_size, int) or new_size <= 0:
            raise ValueError("If keep_aspect_ratio is True, new_size must be a positive integer for the smallest side.")
        if h < w:
            target_h = new_size
            target_w = int(w * new_size / h)
        else:
            target_w = new_size
            target_h = int(h * new_size / w)
    else:
        if not isinstance(new_size, tuple) or len(new_size) != 2 or \
            not all(isinstance(d, int) and d > 0 for d in new_size):
            raise ValueError("If keep_aspect_ratio is False, new_size must be a tuple of two positive integers (width, height).")
        target_w, target_h = new_size

    target_pixels = target_h * target_w
    if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Calculated invalid target dimensions: ({target_w}x{target_h})")

    # 4. Choose interpolation method
    if target_pixels < original_pixels:
        interpolation = interpolation_shrink
        if verbose:
            print(f"Shrinking image. Using interpolation: {interpolation} (INTER_AREA)")
    elif target_pixels > original_pixels:
        interpolation = interpolation_enlarge
        if verbose:
            print(f"Enlarging image. Using interpolation: {interpolation} (CUBIC/LANCZOS4)")
    else:
        interpolation = interpolation_enlarge
        if verbose:
            print("Target size is the same as original. No resize needed, but will re-save.")

    # 5. Resize the image (same logic as before)
    if (target_w, target_h) == (w, h):
        resized_image = image
    else:
        if verbose:
            print(f"Resizing to: {target_w}x{target_h}")
        resized_image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        if resized_image.dtype != original_dtype:
                if verbose:
                    print(f"Warning: dtype changed during resize from {original_dtype} to {resized_image.dtype}. Attempting to cast back.")
                resized_image = resized_image.astype(original_dtype)

    # --- Prepare image for saving ---
    image_to_save = resized_image
    ext = os.path.splitext(tgt_image_path)[1].lower()

    # 6. *** Apply windowing/scaling ONLY if saving 16-bit data as JPEG ***
    if ext in ['.jpg', '.jpeg'] and resized_image.dtype == np.uint16:
        if verbose:
            print(f"Input is uint16 and output is JPEG. Applying windowing ({p_min}-{p_max} percentile) and scaling to uint8.")

        # Perform windowing on the potentially resized 16-bit image
        img_float = resized_image.astype(np.float32)
        vmin = np.percentile(img_float, p_min)
        vmax = np.percentile(img_float, p_max)

        if vmax <= vmin: # Handle uniform image case
            vmin = np.min(img_float)
            vmax = np.max(img_float)
            if vmax <= vmin:
                img_norm = np.zeros_like(img_float)
            else:
                img_norm = (img_float - vmin) / (vmax - vmin)
        else:
            img_norm = (img_float - vmin) / (vmax - vmin)

        img_norm = np.clip(img_norm, 0.0, 1.0) # Ensure range [0, 1]

        # Scale to 8-bit [0, 255]
        image_to_save = (img_norm * 255).astype(np.uint8)
        if verbose:
            print(f"Converted to uint8 for JPEG saving. New range approx [0, 255]")

    elif ext in ['.jpg', '.jpeg'] and resized_image.dtype != np.uint8:
        # Handle other non-uint8 types going to JPEG (e.g., float) - scale robustly
        if verbose:
            print(f"Input is {resized_image.dtype} and output is JPEG. Scaling to uint8.")
        img_float = resized_image.astype(np.float32)
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        if max_val > min_val:
            img_norm = (img_float - min_val) / (max_val - min_val)
        else:
            img_norm = np.zeros_like(img_float)
        image_to_save = (np.clip(img_norm, 0.0, 1.0) * 255).astype(np.uint8)

    # 7. Save the potentially converted image
    os.makedirs(os.path.dirname(tgt_image_path), exist_ok=True)
    params = []
    if ext in ['.jpg', '.jpeg']:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        if verbose:
            print(f"Saving as JPEG with quality={jpeg_quality}")
    elif ext == '.png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
        # Check if saving 16-bit PNG is intended
        if verbose:
            if image_to_save.dtype == np.uint16:
                print(f"Saving as 16-bit PNG with compression={png_compression}")
            else:
                print(f"Saving as 8-bit PNG with compression={png_compression}")

    success = cv2.imwrite(tgt_image_path, image_to_save, params)

    if not success:
        raise IOError(f"Failed to save image to {tgt_image_path}")

    if verbose:
        print(f"Successfully saved resized image to: {tgt_image_path}")

inv_normalize = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

# Adapted from: https://www.kaggle.com/code/mrutyunjaybiswal/vbd-chest-x-ray-abnormalities-detection-eda
def dicom_to_jpeg_high_quality(
    dicom_path,
    output_path=None,
    voi_lut=True,
    fix_monochrome=True,
    quality=95,  # Default high-quality JPEG setting
    verbose=False,
):
    """
    Converts a DICOM file to a high-quality JPEG image.

    This function reads a DICOM file, optionally applies Value of Interest (VOI)
    Look-Up Table (LUT) for better visualization, corrects potential inversion
    issues with MONOCHROME1 photometric interpretation, normalizes the pixel
    data to an 8-bit grayscale range [0, 255], and saves the result as a
    JPEG file with specified high-quality settings.

    Args:
        dicom_path (str): Path to the input DICOM file.
        output_path (str, optional): Path to save the output JPEG file.
                                    If None, simply returns the image.
        voi_lut (bool, optional): If True, attempts to apply the VOI LUT
                                  found in the DICOM tags. Defaults to True.
        fix_monochrome (bool, optional): If True, checks for
                                         PhotometricInterpretation == "MONOCHROME1"
                                         and inverts the pixel values if found.
                                         Defaults to True.
        quality (int, optional): The quality setting for the output JPEG image,
                                 ranging from 0 to 100. Higher values mean
                                 better quality and larger file size.
                                 Defaults to 95.
        verbose (bool, optional): If True, prints a confirmation message upon
                                  successful saving. Defaults to False.
    """
    # Read DICOM file
    dcm_data = pydicom.dcmread(dicom_path)

    # Apply VOI LUT if available (improves visualization)
    pixel_array = dcm_data.pixel_array
    if voi_lut:
        try:
            # Use apply_voi_lut which handles windowing etc.
            data = apply_voi_lut(pixel_array, dcm_data)
            if verbose:
                print(f"Applied VOI LUT. Output dtype: {data.dtype}")
        except Exception as e:
            if verbose:
                print(f"Could not apply VOI LUT: {e}. Using raw pixel array.")
            data = pixel_array # Fallback to raw data if VOI LUT fails
    else:
        data = pixel_array

    # Correct MONOCHROME1 images which are often inverted
    if (fix_monochrome and hasattr(dcm_data, 'PhotometricInterpretation') and
        dcm_data.PhotometricInterpretation == "MONOCHROME1"):
        if verbose:
            print("Correcting MONOCHROME1 inversion.")
        # Use the max value of the *current* data for inversion
        max_val = np.max(data)
        data = max_val - data

    # Normalize pixel values to 0-255 range and convert to uint8
    # Convert to float for calculations first
    data = data.astype(np.float32)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        data = (data - min_val) / (max_val - min_val) # Scale to [0, 1]
    else:
        data = np.zeros_like(data) # Handle uniform image
    data_uint8 = (data * 255).astype(np.uint8)

    # Create PIL image explicitly in grayscale mode
    im = Image.fromarray(data_uint8, mode='L')

    if output_path is None:
        # Return the image if no output path is specified
        return im

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save image as JPEG with high-quality parameters
    im.save(
        output_path,
        format='JPEG',
        quality=quality,
        optimize=True, # Improves compression without quality loss
        subsampling=0  # Disable chroma subsampling (best for grayscale)
    )
    if verbose:
        print(f"Saved {dicom_path} as {output_path} with quality={quality}")

class ImageDataset(Dataset):
    def __init__(self, image_paths, image_transform, use_yolov8=False):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.use_yolov8 = use_yolov8
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, i):
        if self.use_yolov8:
            image, _, size_after = self.image_transform(self.image_paths[i], return_image_size=True)
            return {'i': image, 'resized_shape': size_after}
        else:
            return {'i': self.image_transform(self.image_paths[i]) }
        
class ImageFactClassificationDataset(Dataset):
    def __init__(self, image_paths, image_transform, fact_embeddings, positive_facts,
                 indices, num_facts_per_image,
                 use_strong_and_weak_negatives=False, negative_facts=None,
                 weak_negative_facts=None, strong_negative_facts=None,
                 infinite=False, shuffle=False):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.fact_embeddings = fact_embeddings
        self.positive_facts = positive_facts
        self.use_strong_and_weak_negatives = use_strong_and_weak_negatives
        self.negative_facts = negative_facts
        self.weak_negative_facts = weak_negative_facts
        self.strong_negative_facts = strong_negative_facts        
        self.indices = indices
        self.num_facts_per_image = num_facts_per_image
        self.num_neg_facts_per_image = num_facts_per_image // 2
        self.num_pos_facts_per_image = num_facts_per_image - self.num_neg_facts_per_image
        assert self.num_facts_per_image >= 2
        assert self.num_pos_facts_per_image >= 1
        assert self.num_neg_facts_per_image >= 1
        self.infinite = infinite
        if use_strong_and_weak_negatives:
            assert weak_negative_facts is not None
            assert strong_negative_facts is not None
            self.negative_facts = [strong + weak for strong, weak in zip(strong_negative_facts, weak_negative_facts)]
        else:
            assert negative_facts is not None
        if shuffle:
            random.shuffle(self.indices)
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)

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

        if self.use_strong_and_weak_negatives:
            
            positive_facts = self.positive_facts[idx]
            weak_negative_facts = self.weak_negative_facts[idx]
            strong_negative_facts = self.strong_negative_facts[idx]
            negative_facts = self.negative_facts[idx]
            num_pos = len(positive_facts)
            num_weak_neg = len(weak_negative_facts)
            num_strong_neg = len(strong_negative_facts)
            num_neg = num_weak_neg + num_strong_neg
            assert num_neg == len(negative_facts)

            if num_pos > 0 and num_neg > 0:
                if num_pos < self.num_pos_facts_per_image and num_neg < self.num_neg_facts_per_image:
                    positive_facts = self._adapt_fact_indices(positive_facts, self.num_pos_facts_per_image)
                    negative_facts = self._adapt_fact_indices(negative_facts, self.num_neg_facts_per_image)
                elif num_pos < self.num_pos_facts_per_image:
                    if num_strong_neg > 0:
                        if random.random() < 0.5: # emphasize strong negatives
                            if num_strong_neg < self.num_facts_per_image - num_pos:
                                negative_facts = (strong_negative_facts +
                                                  self._adapt_fact_indices(weak_negative_facts, self.num_facts_per_image - num_strong_neg - num_pos))
                            else:
                                negative_facts = self._adapt_fact_indices(strong_negative_facts, self.num_facts_per_image - num_pos)
                        else: # use all negatives
                            negative_facts = self._adapt_fact_indices(negative_facts, self.num_facts_per_image - num_pos)
                    else:
                        negative_facts = self._adapt_fact_indices(negative_facts, self.num_facts_per_image - num_pos)
                elif num_neg < self.num_neg_facts_per_image:
                    positive_facts = self._adapt_fact_indices(positive_facts, self.num_facts_per_image - num_neg)
                else:
                    assert num_pos >= self.num_pos_facts_per_image and num_neg >= self.num_neg_facts_per_image
                    positive_facts = self._adapt_fact_indices(positive_facts, self.num_pos_facts_per_image)
                    if num_strong_neg > 0:
                        if random.random() < 0.5:
                            if num_strong_neg < self.num_neg_facts_per_image:
                                negative_facts = (strong_negative_facts +
                                                  self._adapt_fact_indices(weak_negative_facts, self.num_neg_facts_per_image - num_strong_neg))
                            else:
                                negative_facts = self._adapt_fact_indices(strong_negative_facts, self.num_neg_facts_per_image)
                        else:
                            negative_facts = self._adapt_fact_indices(negative_facts, self.num_neg_facts_per_image)
                    else:
                        negative_facts = self._adapt_fact_indices(negative_facts, self.num_neg_facts_per_image)
            elif num_pos > 0:
                positive_facts = self._adapt_fact_indices(positive_facts, self.num_facts_per_image)
            elif num_neg > 0:
                negative_facts = self._adapt_fact_indices(negative_facts, self.num_facts_per_image)
            else:
                raise ValueError('No positive or negative facts found!')
        
        else:
            
            positive_facts = self.positive_facts[idx]
            negative_facts = self.negative_facts[idx]
            num_pos = len(positive_facts)
            num_neg = len(negative_facts)

            if num_pos > 0 and num_neg > 0:
                if num_pos < self.num_pos_facts_per_image and num_neg < self.num_neg_facts_per_image:
                    positive_facts = self._adapt_fact_indices(positive_facts, self.num_pos_facts_per_image)
                    negative_facts = self._adapt_fact_indices(negative_facts, self.num_neg_facts_per_image)
                elif num_pos < self.num_pos_facts_per_image:
                    negative_facts = self._adapt_fact_indices(negative_facts, self.num_facts_per_image - num_pos)
                elif num_neg < self.num_neg_facts_per_image:
                    positive_facts = self._adapt_fact_indices(positive_facts, self.num_facts_per_image - num_neg)
                else:
                    assert num_pos >= self.num_pos_facts_per_image and num_neg >= self.num_neg_facts_per_image
                    positive_facts = self._adapt_fact_indices(positive_facts, self.num_pos_facts_per_image)
                    negative_facts = self._adapt_fact_indices(negative_facts, self.num_neg_facts_per_image)
            elif num_pos > 0:
                positive_facts = self._adapt_fact_indices(positive_facts, self.num_facts_per_image)
            elif num_neg > 0:
                negative_facts = self._adapt_fact_indices(negative_facts, self.num_facts_per_image)
            else:
                raise ValueError('No positive or negative facts found!')

        fact_indices = positive_facts + negative_facts
        assert len(fact_indices) == self.num_facts_per_image
        embeddings = self.fact_embeddings[fact_indices]
        labels = np.zeros(len(fact_indices), dtype=np.int64) # initialize with zeros
        labels[:len(positive_facts)] = 1 # set positive labels
        
        return {
            'fidxs': fact_indices,
            'i': image,
            'pe': embeddings, # phrase embeddings
            'l': labels, # labels
        }
    
class ImageFactBasedMultilabelClassificationDataset(Dataset):

    def __init__(self, image_paths, image_transform, phrase_embeddings, phrase_classification_labels,
                 indices, infinite=False, shuffle_indices=False):
        self.image_paths = image_paths
        self.image_transform = image_transform
        self.phrase_embeddings = phrase_embeddings
        self.phrase_classification_labels = phrase_classification_labels
        self.indices = indices
        self.infinite = infinite
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(indices)
        if shuffle_indices:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self._len
    
    def __getitem__(self, i):
        if self.infinite:
            i %= len(self.indices)
        i = self.indices[i]
        image_path = self.image_paths[i]
        phrase_embeddings = self.phrase_embeddings
        phrase_classification_labels = self.phrase_classification_labels[i]
        image = self.image_transform(image_path)
        return {
            'i': image,
            'pe': phrase_embeddings,
            'pcl': phrase_classification_labels,
        }

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

    save_pickle(nearest_neighbors, file_path)
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
            save_pickle(self._image_size_dict, self._image_size_cache_path)
            print('image size cache saved to', self._image_size_cache_path)
            self._dirty = False

image_size_cache = _ImageSizeCache()


def convert_bboxes_into_target_tensors(bboxes, classes, num_classes, feature_map_size):
    """
    Creates a target tensor for bounding boxes on a feature map.
    
    Args:
        bboxes: A tensor or array of shape (N, 4) representing bounding box coordinates.
        classes: A tensor or array of shape (N,) representing bounding box classes.
        num_classes: The number of classes.
        feature_map_size: Tuple (H, W) representing the size of the feature map.
    
    Returns:
        A tensor of shape (num_classes, H*W, 4) representing the target tensor with bounding box coordinates (x_min, y_min, x_max, y_max).
        A tensor of shape (num_classes, H*W) representing the target tensor with bounding box presence.
    """
    H, W = feature_map_size
    N = len(bboxes)
    assert len(classes) == N
    target_coords = torch.zeros(num_classes, H, W, 4)
    target_presence = torch.zeros(num_classes, H, W)
    
    for i in range(N):
        x_min, y_min, x_max, y_max = bboxes[i]
        cls = classes[i]
        x_min_scaled = math.floor(x_min * W)
        y_min_scaled = math.floor(y_min * H)
        x_max_scaled = math.ceil(x_max * W)
        y_max_scaled = math.ceil(y_max * H)
        target_coords[cls, y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled, :] = torch.tensor([x_min, y_min, x_max, y_max])
        target_presence[cls, y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled] = 1

    # Reshape target tensors
    target_coords = target_coords.view(num_classes, H*W, 4)
    target_presence = target_presence.view(num_classes, H*W)
    
    return target_coords, target_presence


def cxcywh_to_xyxy(bbox):
    """
    Convert a bounding box from (center_x, center_y, width, height) format 
    to (x_min, y_min, x_max, y_max) format.

    Args:
        bbox: Tuple or list of 4 numbers (cx, cy, w, h).

    Returns:
        A tuple (x_min, y_min, x_max, y_max).
    """
    cx, cy, w, h = bbox
    x_min = cx - w / 2.0
    y_min = cy - h / 2.0
    x_max = cx + w / 2.0
    y_max = cy + h / 2.0
    return x_min, y_min, x_max, y_max

def _scale_bbox(bbox, scale):
    """
    Scales a bounding box (in xyxy format) about its center. A mild limitation is 
    enforced, so that the extension or contraction does not exceed 7% of the original 
    dimensions. This is helpful in cases where ground truth boxes might be noisy.

    Args:
        bbox: Tuple or list of 4 numbers (x_min, y_min, x_max, y_max) in normalized
              coordinates [0,1].
        scale: Scaling factor. For scale > 1, the bbox grows; for scale < 1, the 
               bbox shrinks.

    Returns:
        The scaled bbox as a tuple (new_x_min, new_y_min, new_x_max, new_y_max).
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    # Limit the expansion or contraction to a maximum of 7% of the original width/height.
    if scale > 1:
        dw = min(w * (scale - 1), 0.07)
        dh = min(h * (scale - 1), 0.07)
    else:
        dw = max(w * (scale - 1), -0.07)
        dh = max(h * (scale - 1), -0.07)

    w += dw
    h += dh

    new_x_min = cx - w / 2.0
    new_y_min = cy - h / 2.0
    new_x_max = cx + w / 2.0
    new_y_max = cy + h / 2.0
    return new_x_min, new_y_min, new_x_max, new_y_max

def _bbox_coords_to_grid_cell_indices(x_min, y_min, x_max, y_max, w, h):
    """
    Converts normalized bounding box coordinates to discrete grid cell indices 
    given a feature map of size (h, w).

    Args:
        x_min, y_min, x_max, y_max: Normalized bbox coordinates in the range [0,1].
        w: Width (number of columns) of the feature map.
        h: Height (number of rows) of the feature map.
        
    Returns:
        A tuple of indices (x_min_idx, y_min_idx, x_max_idx, y_max_idx) representing 
        the grid cell boundaries that overlap with the bounding box.
    """
    # Scale normalized coordinates by the number of cells (w, h)
    x_min_idx = math.floor(x_min * w)
    y_min_idx = math.floor(y_min * h)
    x_max_idx = math.ceil(x_max * w)
    y_max_idx = math.ceil(y_max * h)
    
    # Clamp indices within valid range [0, w] or [0, h]
    x_min_idx = max(0, min(w, x_min_idx))
    x_max_idx = max(0, min(w, x_max_idx))
    y_min_idx = max(0, min(h, y_min_idx))
    y_max_idx = max(0, min(h, y_max_idx))
    
    return (x_min_idx, y_min_idx, x_max_idx, y_max_idx)

def convert_bboxes_into_target_tensors(
    bboxes,
    classes,
    num_classes,
    feature_map_size,
    bbox_format="xyxy",
    apply_ignore_band=False,
    small_factor=0.8,
    large_factor=1.2,
):
    """
    Creates target tensors for bounding box regression and classification on a feature map.

    This function maps provided bounding boxes (assumed normalized in [0,1]) into a
    feature map grid. For each cell in the grid that overlaps a bounding box:
      - target_coords: Contains the bounding box coordinates.
      - target_presence: Binary tensor indicating cells with supervision (1)
                         and background (0).
      - loss_mask: If an ignore band is applied, cells within the uncertain region 
                   are marked with 0 so that the loss function can ignore them, while 
                   cells in the supervised region have a value of 1.

    Args:
        bboxes: List or tensor of shape (N, 4) containing bounding box coordinates.
                Coordinates must be normalized to [0,1].
        classes: List or tensor of shape (N,) with integer class labels for each box.
        num_classes: Total number of classes.
        feature_map_size: Tuple (H, W) defining the height and width of the feature map.
        bbox_format: String specifying the format: either "xyxy" 
                     (i.e., (x_min, y_min, x_max, y_max)) or "cxcywh" 
                     (i.e., (x_center, y_center, w, h)).
        apply_ignore_band: Boolean. When True, defines a band between a scaled-down 
                           (B_s) and a scaled-up bounding box (B_l). The cells in B_s 
                           get standard supervision (target_presence = 1), while cells 
                           in the ring between B_l and B_s are ignored in the loss (loss_mask = 0).
        small_factor: Scale factor used to compute the smaller (inner) bbox B_s.
        large_factor: Scale factor used to compute the larger (outer) bbox B_l.
                      
    Returns:
        target_coords: Tensor of shape (num_classes, H*W, 4) with copied bbox coordinates.
        target_presence: Tensor of shape (num_classes, H*W) containing binary supervision labels.
        loss_mask: Tensor of shape (num_classes, H*W) where 1 indicates cells to compute loss 
                   and 0 indicates cells to ignore.
    """
    H, W = feature_map_size  # feature map dimensions (height, width)
    N = len(bboxes)
    assert len(classes) == N

    # Initialize tensors. The bounding box coordinates are stored per cell.
    target_coords = torch.zeros(num_classes, H, W, 4)
    target_presence = torch.zeros(num_classes, H, W)

    if not apply_ignore_band:
        # Standard supervision: assign bbox info directly to all grid cells overlapping the bbox.
        for i in range(N):
            bbox = bboxes[i]
            cls = classes[i]
            # Convert bbox if necessary.
            if bbox_format == "cxcywh":
                x_min, y_min, x_max, y_max = cxcywh_to_xyxy(bbox)
            elif bbox_format == "xyxy":
                x_min, y_min, x_max, y_max = bbox
            else:
                raise ValueError("Unsupported bbox_format: use 'xyxy' or 'cxcywh'.")

            # Map normalized bbox coordinates to grid cell indices.
            x_min_idx, y_min_idx, x_max_idx, y_max_idx = _bbox_coords_to_grid_cell_indices(
                x_min, y_min, x_max, y_max, W, H
            )
            # Fill target tensors for the active region.
            target_coords[cls, y_min_idx:y_max_idx, x_min_idx:x_max_idx, :] = torch.tensor(bbox)
            target_presence[cls, y_min_idx:y_max_idx, x_min_idx:x_max_idx] = 1

    else:
        # When apply_ignore_band is True, consider two regions:
        #   B_s: the inner, reliable region (using small_factor).
        #   B_l: the outer region defining uncertainty (using large_factor).
        # Cells in B_s get standard supervision, while cells in B_l but not in B_s
        # are marked to be ignored during loss computation.
        
        loss_mask_updates = []  # Will store indices to update loss_mask back to 1 for B_s.
        loss_mask = torch.ones(num_classes, H, W)  # By default, all grid cells contribute to loss.

        for i in range(N):
            bbox = bboxes[i]
            cls = classes[i]

            # Convert bbox if provided in cxcywh format.
            if bbox_format == "cxcywh":
                x_min, y_min, x_max, y_max = cxcywh_to_xyxy(bbox)
            elif bbox_format == "xyxy":
                x_min, y_min, x_max, y_max = bbox
            else:
                raise ValueError("Unsupported bbox_format: use 'xyxy' or 'cxcywh'.")

            # Compute the scaled bounding boxes.
            bbox_s = _scale_bbox((x_min, y_min, x_max, y_max), small_factor)
            bbox_l = _scale_bbox((x_min, y_min, x_max, y_max), large_factor)

            # Convert scaled bboxes to grid cell indices.
            x_min_s, y_min_s, x_max_s, y_max_s = _bbox_coords_to_grid_cell_indices(
                *bbox_s, W, H
            )
            x_min_l, y_min_l, x_max_l, y_max_l = _bbox_coords_to_grid_cell_indices(
                *bbox_l, W, H
            )

            # Mark all cells in the larger bbox (B_l) as "ignore" by setting loss_mask to 0.
            loss_mask[cls, y_min_l:y_max_l, x_min_l:x_max_l] = 0

            # For the inner region (B_s), update standard supervision.
            loss_mask_updates.append((cls, y_min_s, y_max_s, x_min_s, x_max_s))
            target_coords[cls, y_min_s:y_max_s, x_min_s:x_max_s, :] = torch.tensor(bbox)
            target_presence[cls, y_min_s:y_max_s, x_min_s:x_max_s] = 1

        # Restore loss_mask to 1 within the supervised (inner) region B_s.
        for cls, y_min_s, y_max_s, x_min_s, x_max_s in loss_mask_updates:
            loss_mask[cls, y_min_s:y_max_s, x_min_s:x_max_s] = 1

    # Reshape the tensors to have shape (num_classes, H*W, ...) for downstream use.
    target_coords = target_coords.view(num_classes, H * W, 4)
    target_presence = target_presence.view(num_classes, H * W)
    if apply_ignore_band:
        loss_mask = loss_mask.view(num_classes, H * W)
        return target_coords, target_presence, loss_mask # with loss mask
    return target_coords, target_presence # no loss mask