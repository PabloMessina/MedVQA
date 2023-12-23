import os
import argparse
import time
import pandas as pd
import random
import multiprocessing as mp
from pprint import pprint

from medvqa.datasets.image_processing import resize_image
from medvqa.datasets.chexlocalize import (
    CHEXLOCALIZE_CHEXPERT_VAL_LABELS_CSV_PATH,
    CHEXLOCALIZE_CHEXPERT_TEST_LABELS_CSV_PATH,
    CHEXLOCALIZE_DATA_DIR,
    CHEXLOCALIZE_WORKSPACE_DIR,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--keep_aspect_ratio', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=6)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.keep_aspect_ratio:
        assert args.width == args.height, 'Width and height must be equal if keep_aspect_ratio is True'
        new_size = args.width
    else:
        new_size = (args.width, args.height)
    target_dir = os.path.join(CHEXLOCALIZE_WORKSPACE_DIR, f'CheXpert_{args.width}x{args.height}'
                                + ('_keep_aspect_ratio' if args.keep_aspect_ratio else ''))
    os.makedirs(target_dir, exist_ok=True)
    val_labels_df = pd.read_csv(CHEXLOCALIZE_CHEXPERT_VAL_LABELS_CSV_PATH)
    test_labels_df = pd.read_csv(CHEXLOCALIZE_CHEXPERT_TEST_LABELS_CSV_PATH)
    source_image_filepaths = []
    target_image_filepaths = []
    for path in val_labels_df['Path'].values:
        path = path.replace('CheXpert-v1.0/valid', 'val')
        source_image_filepaths.append(os.path.join(CHEXLOCALIZE_DATA_DIR, 'CheXpert', path))
        assert os.path.exists(source_image_filepaths[-1]), f'Image {source_image_filepaths[-1]} does not exist'
        target_image_filepaths.append(os.path.join(target_dir, path))
    for path in test_labels_df['Path'].values:
        source_image_filepaths.append(os.path.join(CHEXLOCALIZE_DATA_DIR, 'CheXpert', path))
        assert os.path.exists(source_image_filepaths[-1]), f'Image {source_image_filepaths[-1]} does not exist'
        target_image_filepaths.append(os.path.join(target_dir, path))
    print(f'Number of images: {len(source_image_filepaths)}')
    print(f'Keep aspect ratio: {args.keep_aspect_ratio}')
    print(f'New size: {new_size}')
    print()
    print('Sample images:')
    indices = random.sample(range(len(source_image_filepaths)), 5)
    pprint([source_image_filepaths[i] for i in indices])
    print()
    pprint([target_image_filepaths[i] for i in indices])
    print()
    print('Creating directories ...')
    for filepath in target_image_filepaths: # create directories
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f'Resizing images in parallel with {args.num_workers} workers ...')
    start_time = time.time()
    with mp.Pool(args.num_workers) as pool:
        pool.starmap(resize_image, [(src, tgt, new_size, args.keep_aspect_ratio) for src, tgt\
                                     in zip(source_image_filepaths, target_image_filepaths)])
    print(f'Finished in {time.time() - start_time} seconds')