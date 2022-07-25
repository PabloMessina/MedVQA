import os
import argparse
from medvqa.datasets.iuxray import IUXRAY_REPORTS_MIN_JSON_PATH, IUXRAY_DATASET_AUX_DIR
from medvqa.datasets.iuxray.iuxray_vqa_dataset_management import _get_iuxray_image_path
from medvqa.utils.common import SOURCE_DIR
from medvqa.utils.files import load_json_file
from medvqa.datasets.pyradiomics import extract_features_chunk, extract_features_all

_TMP_FOLDER = os.path.join(IUXRAY_DATASET_AUX_DIR, '__tmp')
_SCRIPT_PATH = os.path.join(SOURCE_DIR, 'medvqa', 'scripts', 'iuxray', 'extract_pyradiomics_features.py')

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
    metadata = load_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)
    image_paths = [None] * 10000
    i = 0
    for report in metadata.values():
        for img in report['images']:
            image_paths[i] = _get_iuxray_image_path(f'{img["id"]}.png')
            i += 1
    image_paths = image_paths[:i]
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
        print('----------------------------------------------------')
        print('Extracting pyradiomics features for IU X-Ray images')
        print('----------------------------------------------------')
        image_paths = _load_image_paths()
        output_path = os.path.join(IUXRAY_DATASET_AUX_DIR, args.output_filename)
        extract_features_all(
            image_paths=image_paths,
            script_path=_SCRIPT_PATH,
            tmp_folder=_TMP_FOLDER,
            n_chunks=args.n_chunks,
            output_path=output_path,
        )