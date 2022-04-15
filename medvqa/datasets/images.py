import torchvision.transforms as T
import random
from medvqa.datasets.augmentation import ImageAugmentationTransforms

_AUGMENTATION_MODES = [
    'random-color',
    'random-spatial',
]

def get_image_transform(
    image_size = (256, 256),
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
    augmentation_mode = None,
):

    tf_resize = T.Resize(image_size)
    tf_totensor = T.ToTensor()
    tf_normalize = T.Normalize(mean, std)
    default_transform = T.Compose([tf_resize, tf_totensor, tf_normalize])

    if augmentation_mode is None:
        print('Returning default transform')
        return default_transform
    
    assert augmentation_mode in _AUGMENTATION_MODES

    image_aug_transforms = ImageAugmentationTransforms(image_size)

    if augmentation_mode == 'random-color':
        aug_transforms = image_aug_transforms.get_color_transforms()
    else:
        aug_transforms = image_aug_transforms.get_spatial_transforms()

    final_transforms = [
        T.Compose([tf_resize, tf_aug, tf_totensor, tf_normalize])
        for tf_aug in aug_transforms
    ]
    final_transforms.append(default_transform)

    def transform_fn(img):
        return random.choice(final_transforms)(img)

    print(f'Returning augmented transforms with mode {augmentation_mode}')
    return transform_fn

inv_normalize = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)