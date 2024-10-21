import torchvision.transforms as T
import cv2
import albumentations as A
import numpy as np

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
    SHIFT = 'shift'
    SCALE = 'scale'
    ROTATE = 'rotate'
    SHIFT_SCALE_ROTATE = 'shift-scale-rotate'
    @staticmethod
    def get_all():
        return [SPATIAL_TRANSFORMS__BBOX.CROP, SPATIAL_TRANSFORMS__BBOX.SHIFT,
                SPATIAL_TRANSFORMS__BBOX.SCALE, SPATIAL_TRANSFORMS__BBOX.ROTATE,
                SPATIAL_TRANSFORMS__BBOX.SHIFT_SCALE_ROTATE]

class COLOR_TRANSFORMS__BBOX:
    RANDOM_BRIGHTNESS_CONTRAST = 'random-brightness-contrast'
    GAUSSIAN_NOISE = 'gaussian-noise'
    @staticmethod
    def get_all():
        return [COLOR_TRANSFORMS__BBOX.RANDOM_BRIGHTNESS_CONTRAST, COLOR_TRANSFORMS__BBOX.GAUSSIAN_NOISE]

class ImageBboxAugmentationTransforms:

    def __init__(self, image_size=None, crop=0.8, shift=0.1, scale=0.1, rotate=15, contrast=0.2,
                brightness=0.2, gaussian_noise_var=(10, 50)):
        
        self._transform_fns = dict()

        if crop is not None:
            assert image_size is not None
            if isinstance(image_size, int):
                width = image_size
                height = image_size
            else:
                assert len(image_size) == 2
                width, height = image_size
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.CROP] = A.RandomResizedCrop(
                width=width,
                height=height,
                scale=(crop, 1),
                interpolation=cv2.INTER_CUBIC,
                always_apply=True,
            )

        if shift is not None:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.SHIFT] = A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )
        
        if scale is not None:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.SCALE] = A.ShiftScaleRotate(
                shift_limit=0,
                scale_limit=scale,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )

        if rotate is not None:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.ROTATE] = A.ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0,
                rotate_limit=rotate,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )

        if (shift is not None) + (scale is not None) + (rotate is not None) > 1:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.SHIFT_SCALE_ROTATE] = A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=scale,
                rotate_limit=rotate,
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
            self._transform_fns[COLOR_TRANSFORMS__BBOX.GAUSSIAN_NOISE] = A.GaussNoise(
                var_limit=gaussian_noise_var,
                always_apply=True,
            )

        self._spatial_transforms = []
        for key in SPATIAL_TRANSFORMS__BBOX.get_all():
            if key in self._transform_fns:
                self._spatial_transforms.append(self._transform_fns[key])
        
        self._color_transforms = []
        for key in COLOR_TRANSFORMS__BBOX.get_all():
            if key in self._transform_fns:
                self._color_transforms.append(self._transform_fns[key])
    
    def get_transform(self, name, additional_bboxes=None):
        if additional_bboxes is None:
            return A.Compose(
                [self._transform_fns[name]],
                bbox_params=A.BboxParams(format='albumentations'),
            )
        else:
            assert isinstance(additional_bboxes, list)
            additional_targets = { key: 'bboxes' for key in additional_bboxes }
            return A.Compose(
                [self._transform_fns[name]],
                bbox_params=A.BboxParams(format='albumentations'),
                additional_targets=additional_targets,
            )

    def get_color_transforms_list(self, additional_bboxes=None):
        assert len(self._color_transforms) > 0, 'At least one color transform must be specified'
        if additional_bboxes is None:
            return [
                A.Compose([transform], bbox_params=A.BboxParams(format='albumentations')) \
                for transform in self._color_transforms
            ]
        else:
            assert isinstance(additional_bboxes, list)
            additional_targets = { key: 'bboxes' for key in additional_bboxes }
            return [
                A.Compose([transform], bbox_params=A.BboxParams(format='albumentations'),
                          additional_targets=additional_targets) \
                for transform in self._color_transforms
            ]
    
    def get_spatial_transforms_list(self, additional_bboxes=None):
        assert len(self._spatial_transforms) > 0, 'At least one spatial transform must be specified'
        if additional_bboxes is None:
            return [
                A.Compose([transform], bbox_params=A.BboxParams(format='albumentations')) \
                for transform in self._spatial_transforms
            ]
        else:
            assert isinstance(additional_bboxes, list)
            additional_targets = { key: 'bboxes' for key in additional_bboxes }
            return [
                A.Compose([transform], bbox_params=A.BboxParams(format='albumentations'),
                          additional_targets=additional_targets) \
                for transform in self._spatial_transforms
            ]

    def get_merged_spatial_color_transforms_list(self, additional_bboxes=None):
        """Returns a list of transforms, each of which is a composition of spatial and color transforms."""
        assert len(self._spatial_transforms) > 0 and len(self._color_transforms) > 0,\
            'At least one spatial and color transform must be specified'
        output = []
        if additional_bboxes is None:
            for color_tf in self._color_transforms:
                for spatial_tf in self._spatial_transforms:
                    output.append(A.Compose([spatial_tf, color_tf], bbox_params=A.BboxParams(format='albumentations')))
        else:
            assert isinstance(additional_bboxes, list)
            additional_targets = { key: 'bboxes' for key in additional_bboxes }
            for color_tf in self._color_transforms:
                for spatial_tf in self._spatial_transforms:
                    output.append(A.Compose([spatial_tf, color_tf], bbox_params=A.BboxParams(format='albumentations'),
                                            additional_targets=additional_targets))
        return output
    

class ImageSegmentationMaskAugmentationTransforms:

    def __init__(self, image_size=None, crop=0.8, shift=0.1, scale=0.1, rotate=15, contrast=0.2,
                brightness=0.2, gaussian_noise_var=(10, 50)):
        
        self._transform_fns = dict()

        if crop is not None:
            assert image_size is not None
            if isinstance(image_size, int):
                width = image_size
                height = image_size
            else:
                assert len(image_size) == 2
                width, height = image_size
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.CROP] = A.RandomResizedCrop(
                width=width,
                height=height,
                scale=(crop, 1),
                interpolation=cv2.INTER_CUBIC,
                always_apply=True,
            )

        if shift is not None:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.SHIFT] = A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )
        
        if scale is not None:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.SCALE] = A.ShiftScaleRotate(
                shift_limit=0,
                scale_limit=scale,
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )

        if rotate is not None:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.ROTATE] = A.ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0,
                rotate_limit=rotate,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )

        if (shift is not None) + (scale is not None) + (rotate is not None) > 1:
            self._transform_fns[SPATIAL_TRANSFORMS__BBOX.SHIFT_SCALE_ROTATE] = A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=scale,
                rotate_limit=rotate,
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
            self._transform_fns[COLOR_TRANSFORMS__BBOX.GAUSSIAN_NOISE] = A.GaussNoise(
                var_limit=gaussian_noise_var,
                always_apply=True,
            )

        self._spatial_transforms = []
        for key in SPATIAL_TRANSFORMS__BBOX.get_all():
            if key in self._transform_fns:
                self._spatial_transforms.append(self._transform_fns[key])
        
        self._color_transforms = []
        for key in COLOR_TRANSFORMS__BBOX.get_all():
            if key in self._transform_fns:
                self._color_transforms.append(self._transform_fns[key])
    
    def get_transform(self, name):
        return A.Compose([self._transform_fns[name]])

    def get_color_transforms_list(self):
        assert len(self._color_transforms) > 0, 'At least one color transform must be specified'
        return [A.Compose([transform]) for transform in self._color_transforms]
    
    def get_spatial_transforms_list(self):
        assert len(self._spatial_transforms) > 0, 'At least one spatial transform must be specified'
        return [A.Compose([transform]) for transform in self._spatial_transforms]

    def get_merged_spatial_color_transforms_list(self):
        """Returns a list of transforms, each of which is a composition of spatial and color transforms."""
        assert len(self._spatial_transforms) > 0 and len(self._color_transforms) > 0,\
            'At least one spatial and color transform must be specified'
        output = []
        for color_tf in self._color_transforms:
            for spatial_tf in self._spatial_transforms:
                output.append(A.Compose([spatial_tf, color_tf]))
        return output
    
class ChestImagenomeAlbumentationAdapter:

    def __init__(self, num_bbox_classes):
        self.num_bbox_classes = num_bbox_classes
    
    def encode(self, bbox_coords, bbox_presence=None):
        assert len(bbox_coords.shape) == 2
        albumentation_bbox_coords = []
        for i in range(bbox_coords.shape[0]):
            if bbox_presence is None or bbox_presence[i] == 1:
                x_min = bbox_coords[i, 0]
                y_min = bbox_coords[i, 1]
                x_max = bbox_coords[i, 2]
                y_max = bbox_coords[i, 3]
                assert x_min <= x_max
                assert y_min <= y_max
                if x_min < x_max and y_min < y_max: # ignore invalid bboxes
                    albumentation_bbox_coords.append([
                        bbox_coords[i, 0],
                        bbox_coords[i, 1],
                        bbox_coords[i, 2],
                        bbox_coords[i, 3],
                        i, # category id
                    ])
        return albumentation_bbox_coords
    
    def decode(self, albumentation_bbox_coords, only_boxes=False):
        bbox_coords = np.zeros((self.num_bbox_classes, 4), dtype=np.float32)
        # set all bbox coordinates to [0, 0, 1, 1] by default
        bbox_coords[:, 2] = 1
        bbox_coords[:, 3] = 1        
        if not only_boxes:
            bbox_presence = np.zeros(self.num_bbox_classes, dtype=np.float32)
        for bbox in albumentation_bbox_coords:
            cls = bbox[4]
            for i in range(4):
                bbox_coords[cls, i] = bbox[i]
            if not only_boxes:
                bbox_presence[cls] = 1
        if only_boxes:
            return bbox_coords
        else:
            return bbox_coords, bbox_presence
