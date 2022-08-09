from medvqa.datasets.vinbig import VINBIG_ORIGINAL_IMAGES_FOLDER
from medvqa.datasets.vinbig import VINBIG_ALL_IMAGES_TXT_PATH
from medvqa.utils.files import read_lines_from_txt
from PIL import Image
from tqdm import tqdm
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--save-folder', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    new_size = (args.width, args.height)
    save_folder = args.save_folder
    os.makedirs(save_folder)
    image_names = read_lines_from_txt(VINBIG_ALL_IMAGES_TXT_PATH)
    print(f'Resizing {len(image_names)} images to {args.width} x {args.height} ...')
    for image_name in tqdm(image_names):
        image_path = os.path.join(VINBIG_ORIGINAL_IMAGES_FOLDER, image_name)
        img = Image.open(image_path)
        img = img.resize(new_size)
        save_path = os.path.join(save_folder, image_name)
        img.save(save_path)    
    print('Done!')