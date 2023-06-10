import multiprocessing as mp
import os
import argparse
import time
import pydicom
from PIL import Image
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dicom-folder', type=str, required=True)
    parser.add_argument('--test-dicom-folder', type=str, required=True)
    parser.add_argument('--save-folder', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=5)
    return parser.parse_args()

def dicom_to_jpeg(dicom_file, output_path):    
    # Read the DICOM file
    ds = pydicom.dcmread(dicom_file)

    # Extract pixel data
    pixel_array = ds.pixel_array

    # Normalize the pixel data
    image = (pixel_array / pixel_array.max()) * 255.0
    image = image.astype('uint8')

    # Create PIL Image object from pixel array
    img = Image.fromarray(image)

    # Save the image as JPEG
    img.save(output_path)

    # Remove the DICOM file
    os.remove(dicom_file)

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    train_dicom_filenames = os.listdir(args.train_dicom_folder)
    test_dicom_filenames = os.listdir(args.test_dicom_folder)
    print(f'Found {len(train_dicom_filenames)} train DICOMs and {len(test_dicom_filenames)} test DICOMs')
    source_dicom_filepaths = []
    target_jpeg_filepaths = []
    for dicom_filename in train_dicom_filenames:
        source_dicom_filepaths.append(os.path.join(args.train_dicom_folder, dicom_filename))
        target_jpeg_filepaths.append(os.path.join(args.save_folder, dicom_filename.replace('.dicom', '.jpg')))
    for dicom_filename in test_dicom_filenames:
        source_dicom_filepaths.append(os.path.join(args.test_dicom_folder, dicom_filename))
        target_jpeg_filepaths.append(os.path.join(args.save_folder, dicom_filename.replace('.dicom', '.jpg')))
    pprint(f'source_dicom_filepaths[:5]: {source_dicom_filepaths[:5]}')
    pprint(f'target_jpeg_filepaths[:5]: {target_jpeg_filepaths[:5]}')
    print()
    print(f'Converting {len(source_dicom_filepaths)} DICOMs to JPEGs ...')
    start_time = time.time()
    with mp.Pool(args.num_workers) as pool:
        pool.starmap(dicom_to_jpeg, zip(source_dicom_filepaths, target_jpeg_filepaths))
    print(f'Finished in {time.time() - start_time} seconds')