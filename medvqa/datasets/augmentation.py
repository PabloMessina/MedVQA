import torchvision.transforms as T
import cv2
import albumentations as A
import numpy as np

class CoarseDropoutWithoutBbox(A.CoarseDropout):
    def apply_to_bbox(self, bbox, **params):
        # Skip any transformation to bounding boxes
        return bbox

class ImageAugmentedTransforms:

    def __init__(self,
                image_size,
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                crop=0.8, p_crop=1.0,
                shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p_shift_scale_rotate=0.5,
                contrast=0.2, brightness=0.2, saturation=0.2, hue=0.2, p_color_jitter=0.5,
                p_coarse_dropout=0.5,
                gaussian_noise_limit=(10, 50), p_gaussian_noise=0.2,
                gaussian_blur_limit=(3, 5), p_gaussian_blur=0.2,
                custom_normalization_transform=None):
        
        self._transform_fns = dict()

        # Transformations:

        # Always:
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # 2. Resize
        
        # Spatial (optional):
        # 1. Crop
        # 2. Shift-Scale-Rotate
        # 3. HorizontalFlip

        # Color (optional):
        # 1. CoarseDropout
        # 2. ColorJitter
        # 3. GaussNoise
        # 4. GaussianBlur

        # CLAHE
        self._transform_fns['clahe-test'] = A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True) # apply always
        self._transform_fns['clahe-train'] = A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5) # apply with 50% probability

        # Resize
        if isinstance(image_size, int):
            width = image_size
            height = image_size
        else:
            assert len(image_size) == 2
            width, height = image_size
        self._transform_fns['resize'] = A.Resize(width=width, height=height, always_apply=True, interpolation=cv2.INTER_CUBIC)

        # Spatial transforms

        # 1. Crop
        self._transform_fns['crop'] = A.RandomResizedCrop(
            width=width,
            height=height,
            scale=(crop, 1),
            interpolation=cv2.INTER_CUBIC,
            p=p_crop,
        )
        
        # 2. Shift-Scale-Rotate
        self._transform_fns['shift-scale-rotate'] = A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=cv2.BORDER_CONSTANT,
            p=p_shift_scale_rotate,
        )

        # 3. HorizontalFlip
        self._transform_fns['horizontal-flip'] = A.HorizontalFlip(p=0.5)

        # Color transforms

        # 1. CoarseDropout
        self._transform_fns['coarse-dropout'] = CoarseDropoutWithoutBbox(
            max_holes=8,
            max_height=int(height * 0.05),
            max_width=int(width * 0.05),
            min_holes=4,
            min_height=int(height * 0.01),
            min_width=int(width * 0.01),
            fill_value=0,
            p=p_coarse_dropout,
        )

        # 2. ColorJitter
        self._transform_fns['color-jitter'] = A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=p_color_jitter,
        )

        # 3. GaussNoise
        self._transform_fns['gauss-noise'] = A.GaussNoise(
            var_limit=gaussian_noise_limit,
            p=p_gaussian_noise,
        )

        # 4. GaussianBlur
        self._transform_fns['gaussian-blur'] = A.GaussianBlur(
            blur_limit=gaussian_blur_limit,
            p=p_gaussian_blur,
        )

        # Other transforms
        self._transform_fns['load_image'] = lambda x: cv2.imread(x)
        self._transform_fns['bgr2rgb'] = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        self._transform_fns['totensor'] = T.ToTensor()
        if custom_normalization_transform is not None:
            self._transform_fns['normalize'] = custom_normalization_transform
        else:
            self._transform_fns['normalize'] = T.Normalize(mean=mean, std=std) # default normalization

    def get_test_transform(self, allow_returning_image_size=False):
        tf_load_image = self._transform_fns['load_image']
        tf_bgr2rgb = self._transform_fns['bgr2rgb']
        tf_clahe_test = self._transform_fns['clahe-test']
        tf_resize = self._transform_fns['resize']
        tf_totensor = self._transform_fns['totensor']
        tf_normalize = self._transform_fns['normalize']
        if allow_returning_image_size:
            def test_transform(image_path, return_image_size=False):
                image = tf_load_image(image_path)
                if return_image_size:
                    size_before = image.shape[:2] # (H, W)
                image = tf_bgr2rgb(image)
                image = tf_clahe_test(image=image)['image']
                image = tf_resize(image=image)['image']
                if return_image_size:
                    size_after = image.shape[:2] # (H, W)
                image = tf_totensor(image)
                image = tf_normalize(image)
                if return_image_size:
                    return image, size_before, size_after
                return image
        else:
            def test_transform(image_path):
                image = tf_load_image(image_path)
                image = tf_bgr2rgb(image)
                image = tf_clahe_test(image=image)['image']
                image = tf_resize(image=image)['image']
                image = tf_totensor(image)
                image = tf_normalize(image)
                return image
        return test_transform
    
    def get_train_transform(self, mode, bbox_aware=False, for_vinbig=False):
        tf_load_image = self._transform_fns['load_image']
        tf_bgr2rgb = self._transform_fns['bgr2rgb']
        tf_totensor = self._transform_fns['totensor']
        tf_normalize = self._transform_fns['normalize']

        if mode == 'color':
            tf_alb = A.Compose([
                self._transform_fns['clahe-train'],
                self._transform_fns['resize'],
                self._transform_fns['coarse-dropout'],
                self._transform_fns['color-jitter'],
                self._transform_fns['gauss-noise'],
                self._transform_fns['gaussian-blur'],
            ],
            bbox_params=A.BboxParams(format='albumentations') if bbox_aware else None)
        elif mode == 'spatial':
            tf_alb = A.Compose([
                self._transform_fns['clahe-train'],
                self._transform_fns['crop'],
                self._transform_fns['resize'],
                self._transform_fns['shift-scale-rotate'],
                self._transform_fns['horizontal-flip'],
            ], bbox_params=A.BboxParams(format='albumentations') if bbox_aware else None)
        elif mode == 'both':
            tf_alb = A.Compose([
                self._transform_fns['clahe-train'],
                self._transform_fns['crop'],
                self._transform_fns['resize'],
                self._transform_fns['shift-scale-rotate'],
                self._transform_fns['horizontal-flip'],
                self._transform_fns['coarse-dropout'],
                self._transform_fns['color-jitter'],
                self._transform_fns['gauss-noise'],
                self._transform_fns['gaussian-blur'],
            ], bbox_params=A.BboxParams(format='albumentations') if bbox_aware else None)
        else:
            raise ValueError('Invalid mode: {}'.format(mode))
        
        if bbox_aware:
            if for_vinbig:
                print('get_train_transform(): Using bbox-aware transforms for VinBigData')
                def train_transform(image_path, bboxes, classes, albumentation_adapter, return_image_size=False):
                    image = tf_load_image(image_path)
                    if return_image_size:
                        size_before = image.shape[:2] # (H, W)
                    image = tf_bgr2rgb(image)
                    bboxes = albumentation_adapter.encode(bboxes, classes)
                    augmented = tf_alb(image=image, bboxes=bboxes)
                    image = augmented['image']
                    if return_image_size:
                        size_after = image.shape[:2] # (H, W)
                    image = tf_totensor(image)
                    image = tf_normalize(image)
                    bboxes = augmented['bboxes']
                    bboxes, classes = albumentation_adapter.decode(bboxes)
                    if return_image_size:
                        return image, bboxes, classes, size_before, size_after
                    return image, bboxes, classes
            else:
                raise NotImplementedError('bbox_aware=True is only supported for VinBigData')
        else:
            print('get_train_transform(): Using normal transforms')
            def train_transform(image_path):
                image = tf_load_image(image_path)
                image = tf_bgr2rgb(image)
                image = tf_alb(image=image)['image']
                image = tf_totensor(image)
                image = tf_normalize(image)
                return image
        return train_transform

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
