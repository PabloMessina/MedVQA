import torchvision.transforms as transforms

def get_image_transform(
    image_size = (256, 256),
    mean = (0.485, 0.456, 0.406),
    std= (0.229, 0.224, 0.225),
):
    tfs = []
    tfs.append(transforms.Resize(image_size))
    tfs.append(transforms.ToTensor())
    tfs.append(transforms.Normalize(mean, std))
    return transforms.Compose(tfs)

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)