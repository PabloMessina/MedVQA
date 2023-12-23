import multiprocessing as mp
import os
import argparse
import time
from pprint import pprint

from medvqa.datasets.image_processing import resize_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--keep-aspect-ratio', action='store_true', default=False)
    parser.add_argument('--source-folder', type=str, required=True)
    parser.add_argument('--target-folder', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=6)
    return parser.parse_args()

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