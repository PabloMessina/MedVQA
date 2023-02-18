import torchvision.transforms as T
import cv2
import albumentations as A

_SPATIAL_TRANSFORMS = [
    'crop',
    'translate',
    'rotation',
    'shear',
]

_COLOR_TRANSFORMS = [
    'contrast-down',
    'contrast-up',
    'brightness-down',
    'brightness-up',
]

class ImageAugmentationTransforms:

    def __init__(self, image_size,
                crop=0.8, translate=0.1, rotation=15, contrast=0.8, brightness=0.8,
                shear=(10, 10)):
        
        self._transform_fns = dict()

        if crop is not None:
            self._transform_fns['crop'] = T.RandomResizedCrop(
                image_size,
                scale=(crop, 1),
            )

        if translate is not None:
            self._transform_fns['translate'] = T.RandomAffine(
                degrees=0,
                translate=(translate, translate),
            )

        if shear is not None:
            shear_x, shear_y = shear
            self._transform_fns['shear'] = T.RandomAffine(
                degrees=0,
                shear=(-shear_x, shear_x, -shear_y, shear_y),
            )

        if rotation is not None:
            self._transform_fns['rotation'] = T.RandomRotation(rotation)

        if contrast is not None:
            contrast_down_max = 0.9
            self._transform_fns['contrast-down'] = T.ColorJitter(
                contrast=(contrast_down_max - contrast, contrast_down_max),
            )
            contrast_up_min = 1.1
            self._transform_fns['contrast-up'] = T.ColorJitter(
                contrast=(contrast_up_min, contrast_up_min + contrast),
            )

        if brightness is not None:
            brightness_down_max = 0.9
            self._transform_fns['brightness-down'] = T.ColorJitter(
                brightness=(brightness_down_max - brightness, brightness_down_max),
            )
            brightness_up_min = 1.1
            self._transform_fns['brightness-up'] = T.ColorJitter(
                brightness=(brightness_up_min, brightness_up_min + brightness),
            )
        
        self._spatial_transforms = [self._transform_fns[name] for name in _SPATIAL_TRANSFORMS]
        self._color_transforms = [self._transform_fns[name] for name in _COLOR_TRANSFORMS]
    
    def get_spatial_transforms(self):
        return self._spatial_transforms
    
    def get_color_transforms(self):
        return self._color_transforms

class SPATIAL_TRANSFORMS__BBOX:
    CROP = 'crop'
    SHIFT_SCALE_ROTATE = 'shift-scale-rotate'

class COLOR_TRANSFORMS__BBOX:
    RANDOM_BRIGHTNESS_CONTRAST = 'random-brightness-contrast'
    GAUSSIAN_NOISE = 'gaussian-noise'

class ImageBboxAugmentationTransforms:

    def __init__(self, image_size,
                crop=0.8, shift=0.1, scale=0.1, rotate=20, contrast=0.2, brightness=0.2, gaussian_noise_var=(10, 50)):
        
        self._transform_fns = dict()

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            assert len(image_size) == 2
        width, height = image_size

        if crop is not None:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.CROP] = A.RandomResizedCrop(
                width=width, height=height, scale=(crop, 1), always_apply=True,
                interpolation=cv2.INTER_CUBIC,
            )

        if shift is not None or scale is not None or rotate is not None:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.SHIFT_SCALE_ROTATE] = A.ShiftScaleRotate(
                shift_limit=shift if shift is not None else 0,
                scale_limit=scale if scale is not None else 0,
                rotate_limit=rotate if rotate is not None else 0,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )

        if contrast is not None or brightness is not None:
            self._transform_fns[COLOR_TRANSFORMS__BBOX.RANDOM_BRIGHTNESS_CONTRAST] = A.RandomBrightnessContrast(
                brightness_limit=brightness if brightness is not None else 0,
                contrast_limit=contrast if contrast is not None else 0,
                always_apply=True,
            )

        if gaussian_noise_var is not None:
            self._transform_fns['gaussian-noise'] = A.GaussNoise(
                var_limit=gaussian_noise_var,
                always_apply=True,
            )

        self._spatial_transforms = []
        if SPATIAL_TRANSFORMS__BBOX.CROP in self._transform_fns:
            self._spatial_transforms.append(self._transform_fns[SPATIAL_TRANSFORMS__BBOX.CROP])
        if SPATIAL_TRANSFORMS__BBOX.SHIFT_SCALE_ROTATE in self._transform_fns:
            self._spatial_transforms.append(self._transform_fns[SPATIAL_TRANSFORMS__BBOX.SHIFT_SCALE_ROTATE])
        
        self._color_transforms = []
        if COLOR_TRANSFORMS__BBOX.RANDOM_BRIGHTNESS_CONTRAST in self._transform_fns:
            self._color_transforms.append(self._transform_fns[COLOR_TRANSFORMS__BBOX.RANDOM_BRIGHTNESS_CONTRAST])
        if COLOR_TRANSFORMS__BBOX.GAUSSIAN_NOISE in self._transform_fns:
            self._color_transforms.append(self._transform_fns[COLOR_TRANSFORMS__BBOX.GAUSSIAN_NOISE])

        assert len(self._spatial_transforms) > 0 or len(self._color_transforms) > 0,\
            'At least one spatial or color transform must be specified'
    
    def get_transform(self, name):
        return A.Compose(
            [self._transform_fns[name]],
            bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']),
        )

    def get_color_transforms_list(self):
        return [
            A.Compose([transform], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids'])) \
            for transform in self._color_transforms
        ]
    
    def get_spatial_transforms_list(self):
        return [
            A.Compose([transform], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids'])) \
            for transform in self._spatial_transforms
        ]

    def get_merged_spatial_color_transforms_list(self):
        """Returns a list of transforms, each of which is a composition of spatial and color transforms."""
        if len(self._color_transforms) == 0:
            return [A.Compose(self._spatial_transforms, bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']))]
        output = []
        for transform in self._color_transforms:
            tfs = [transform] + self._spatial_transforms
            output.append(A.Compose(tfs, bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids'])))
        return output