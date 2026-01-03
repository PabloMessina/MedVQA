import multiprocessing as mp
import os
import argparse
import time
from pprint import pprint
from tqdm import tqdm  # install with `pip install tqdm`

from medvqa.datasets.image_processing import dicom_to_jpeg_high_quality


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dicom_folder", type=str, required=True)
    parser.add_argument("--test_dicom_folder", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create folders
    os.makedirs(args.save_folder, exist_ok=True)
    train_save_folder = os.path.join(args.save_folder, "train")
    test_save_folder = os.path.join(args.save_folder, "test")
    os.makedirs(train_save_folder, exist_ok=True)
    os.makedirs(test_save_folder, exist_ok=True)

    # Gather filepaths
    train_dicom_filenames = os.listdir(args.train_dicom_folder)
    test_dicom_filenames = os.listdir(args.test_dicom_folder)
    print(
        f"Found {len(train_dicom_filenames)} train DICOMs and "
        f"{len(test_dicom_filenames)} test DICOMs"
    )

    source_dicom_filepaths = []
    target_jpeg_filepaths = []

    for dicom_filename in train_dicom_filenames:
        source_dicom_filepaths.append(
            os.path.join(args.train_dicom_folder, dicom_filename)
        )
        target_jpeg_filepaths.append(
            os.path.join(train_save_folder, dicom_filename.replace(".dicom", ".jpg"))
        )

    for dicom_filename in test_dicom_filenames:
        source_dicom_filepaths.append(
            os.path.join(args.test_dicom_folder, dicom_filename)
        )
        target_jpeg_filepaths.append(
            os.path.join(test_save_folder, dicom_filename.replace(".dicom", ".jpg"))
        )

    pprint(f"source_dicom_filepaths[:5]: {source_dicom_filepaths[:5]}")
    pprint(f"target_jpeg_filepaths[:5]: {target_jpeg_filepaths[:5]}")
    print()

    # For testing (remove later)
    source_dicom_filepaths = source_dicom_filepaths[:3]
    target_jpeg_filepaths = target_jpeg_filepaths[:3]

    total = len(source_dicom_filepaths)
    print(f"Converting {total} DICOMs to JPEGs...")

    start_time = time.time()

    # --- Multiprocessing with progress bar ---
    with mp.Pool(args.num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                dicom_to_jpeg_high_quality,
                zip(source_dicom_filepaths, target_jpeg_filepaths),
            ),
            total=total,
            desc="Converting",
        ):
            pass

    print(f"Finished in {time.time() - start_time:.2f} seconds")