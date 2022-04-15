import torchvision.transforms as T

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
        return self._spatial_transforms
        