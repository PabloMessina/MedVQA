import multiprocessing as mp
import cv2
import os
import argparse
import time
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--keep-aspect-ratio', action='store_true', default=False)
    parser.add_argument('--source-folder', type=str, required=True)
    parser.add_argument('--target-folder', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=6)
    return parser.parse_args()

def resize_image(src_image_path, tgt_image_path, new_size, keep_aspect_ratio):
    image = cv2.imread(src_image_path)
    if keep_aspect_ratio:
        # Resize image so that the smallest side is new_size
        h, w, _ = image.shape
        if h < w:
            new_h = new_size
            new_w = int(w * new_size / h)
        else:
            new_w = new_size
            new_h = int(h * new_size / w)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    # Save image to new path
    cv2.imwrite(tgt_image_path, image)

if __name__ == '__main__':
    args = parse_args()
    if args.keep_aspect_ratio:
        assert args.width == args.height, 'Width and height must be equal if keep-aspect-ratio is True'
        new_size = args.width
    else:
        new_size = (args.width, args.height)
    os.makedirs(args.target_folder, exist_ok=True)
    source_image_filenames = os.listdir(args.source_folder)
    source_image_filepaths = [os.path.join(args.source_folder, filename) for filename in source_image_filenames]
    target_image_filepaths = [os.path.join(args.target_folder, filename) for filename in source_image_filenames]
    print(f'Resizing {len(source_image_filepaths)} images from\n\t{args.source_folder} to\n\t{args.target_folder}')
    print(f'Keep aspect ratio: {args.keep_aspect_ratio}')
    print(f'New size: {new_size}')
    print()
    pprint(source_image_filepaths[:5])
    print()
    pprint(target_image_filepaths[:5])
    print()
    print(f'Resizing images in parallel with {args.num_workers} workers ...')
    start_time = time.time()
    with mp.Pool(args.num_workers) as pool:
        pool.starmap(resize_image, [(src, tgt, new_size, args.keep_aspect_ratio) for src, tgt\
                                     in zip(source_image_filepaths, target_image_filepaths)])
    print(f'Finished in {time.time() - start_time} seconds')