from medvqa.datasets.padchest import (
    PADCHEST_DATASET_DIR,
    PADCHEST_IMAGES_SMALL_DIR,
    PADCHEST_LABELS_CSV_PATH,
)
import argparse
import os
import pandas as pd
from PIL import Image, UnidentifiedImageError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dirs', nargs='+', type=str, required=True)
    return parser.parse_args()

def load_imagedir2filenames():
    df = pd.read_csv(PADCHEST_LABELS_CSV_PATH)
    image_dir2filenames = {}
    image_dirs = df['ImageDir']
    image_filenames = df['ImageID']
    for image_dir, image_filename in zip(image_dirs, image_filenames):
        try:
            image_dir2filenames[image_dir].append(image_filename)
        except KeyError:
            image_dir2filenames[image_dir] = [image_filename]
    return image_dir2filenames

def resize_image(img_path, images_with_errors, shortest_dim=320):
    try:
        orig_img = Image.open(img_path)
    except UnidentifiedImageError:
        images_with_errors.append(img_path)
        return
    orig_size = orig_img.size
    assert len(orig_size) == 2
    min_dim = min(orig_size)
    new_size = (int(orig_size[0] / min_dim * shortest_dim),
                int(orig_size[1] / min_dim * shortest_dim))
    try:
        resized_img = orig_img.resize(new_size)
        resized_img.save(img_path)
    except OSError:
        images_with_errors.append(img_path)
    except SyntaxError:
        images_with_errors.append(img_path)

def main():
    args = parse_args()
    print('-' * 80)
    print('Loading image_dir2filenames...')
    image_dir2filenames = load_imagedir2filenames()
    image_paths_with_error = []
    for image_dir in args.image_dirs:
        image_dir_path = os.path.join(PADCHEST_DATASET_DIR, image_dir)
        print('-' * 80)
        print(f'Unzipping {image_dir_path}')
        os.system(f'unzip -o -q {image_dir_path}.zip -d {PADCHEST_IMAGES_SMALL_DIR}')
        print(f'Resizing {len(image_dir2filenames[int(image_dir)])} images in {image_dir_path}')
        for image_filename in image_dir2filenames[int(image_dir)]:
            image_path = os.path.join(PADCHEST_IMAGES_SMALL_DIR, image_filename)
            resize_image(image_path, image_paths_with_error)
        print('Done!')
    if len(image_paths_with_error) > 0:
        print('-' * 80)
        print('Images with errors:')
        print(image_paths_with_error)

if __name__ == '__main__':
    main()