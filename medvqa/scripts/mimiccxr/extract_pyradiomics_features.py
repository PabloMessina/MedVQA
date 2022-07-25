import os
import argparse
from medvqa.datasets.mimiccxr import MIMICCXR_DATASET_AUX_DIR
from medvqa.utils.common import SOURCE_DIR
from medvqa.datasets.pyradiomics import extract_features_chunk, extract_features_all
from medvqa.datasets.mimiccxr.preprocessing import image_paths_generator

_TMP_FOLDER = os.path.join(MIMICCXR_DATASET_AUX_DIR, '__tmp')
_SCRIPT_PATH = os.path.join(SOURCE_DIR, 'medvqa', 'scripts', 'mimiccxr', 'extract_pyradiomics_features.py')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-mode', dest='chunk_mode', action='store_true')
    parser.set_defaults(chunk_mode=False)
    parser.add_argument('--chunk-id', type=int, default=None)
    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--output-filename', type=str, default=None)
    parser.add_argument('--n-chunks', type=int, default=None)
    return parser.parse_args()

def _load_image_paths():
    image_paths = [None] * 400000
    for i, image_path in enumerate(image_paths_generator()):
        image_paths[i] = image_path
    assert image_paths[i] is not None
    assert image_paths[i+1] is None
    image_paths = image_paths[:i+1]
    return image_paths

if __name__ == '__main__':
    args = parse_args()
    if args.chunk_mode:
        assert args.chunk_id is not None
        assert args.input_path is not None
        assert args.output_path is not None
        extract_features_chunk(
            chunk_id=args.chunk_id,
            input_path=args.input_path,
            output_path=args.output_path,
        )
    else:
        assert args.n_chunks is not None
        assert args.output_filename is not None
        print('-----------------------------------------------------')
        print('Extracting pyradiomics features for MIMIC-CXR images')
        print('-----------------------------------------------------')
        image_paths = _load_image_paths()
        output_path = os.path.join(MIMICCXR_DATASET_AUX_DIR, args.output_filename)
        extract_features_all(
            image_paths=image_paths,
            script_path=_SCRIPT_PATH,
            tmp_folder=_TMP_FOLDER,
            n_chunks=args.n_chunks,
            output_path=output_path,
        )