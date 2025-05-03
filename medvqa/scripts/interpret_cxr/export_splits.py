import argparse
import os
import pandas as pd
from medvqa.datasets.chexpert import CHEXPERT_TEST_LABELS_CSV_PATH, CHEXPERT_TRAIN_VAL_CSV_PATH
from medvqa.datasets.iuxray import get_iuxray_all_image_ids
from medvqa.datasets.mimiccxr import (
    get_mimiccxr_test_dicom_ids,
    get_mimiccxr_train_dicom_ids,
    get_mimiccxr_val_dicom_ids,
)
from medvqa.utils.files_utils import load_json, save_pickle
from medvqa.utils.logging_utils import print_blue, print_bold, print_magenta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    # Challenge filepaths
    train_mimic_filepath = os.path.join(data_dir, 'train_mimic.json')
    val_mimic_filepath = os.path.join(data_dir, 'val_mimic.json')
    train_filepath = os.path.join(data_dir, 'train.csv')
    val_filepath = os.path.join(data_dir, 'val.csv')
    assert os.path.exists(train_mimic_filepath)
    assert os.path.exists(val_mimic_filepath)
    assert os.path.exists(train_filepath)
    assert os.path.exists(val_filepath)

    # MIMIC-CXR splits
    mimiccxr_output_filepath = os.path.join(output_dir, 'mimiccxr_splits.pkl')
    if os.path.exists(mimiccxr_output_filepath):
        print_blue(f"MIMIC-CXR splits already exist at {mimiccxr_output_filepath}", bold=True)
    else:
        print_bold("MIMIC-CXR splits")

        all_dicom_ids = set()
        all_dicom_ids.update(get_mimiccxr_train_dicom_ids())
        all_dicom_ids.update(get_mimiccxr_val_dicom_ids())
        all_dicom_ids.update(get_mimiccxr_test_dicom_ids())

        train_mimic = load_json(train_mimic_filepath)
        train_dicom_ids = []
        train_val_dicom_ids_set = set()
        for item in train_mimic:
            for image_path in item['images_path']:
                dicom_id = os.path.basename(image_path).split('.')[0] # remove extension
                assert dicom_id in all_dicom_ids
                train_dicom_ids.append(dicom_id)
                train_val_dicom_ids_set.add(dicom_id)

        val_mimic = load_json(val_mimic_filepath)
        val_dicom_ids = []
        for item in val_mimic:
            for image_path in item['images_path']:
                dicom_id = os.path.basename(image_path).split('.')[0] # remove extension
                assert dicom_id in all_dicom_ids
                val_dicom_ids.append(dicom_id)
                train_val_dicom_ids_set.add(dicom_id)

        test_dicom_ids = [dicom_id for dicom_id in all_dicom_ids if dicom_id not in train_val_dicom_ids_set]
        print(f"MIMIC-CXR splits: train={len(train_dicom_ids)}, val={len(val_dicom_ids)}, test={len(test_dicom_ids)}")

        mimiccxr_output = {
            'train': train_dicom_ids,
            'val': val_dicom_ids,
            'test': test_dicom_ids
        }
        print_blue(f"Saving MIMIC-CXR splits to {mimiccxr_output_filepath}", bold=True)
        save_pickle(mimiccxr_output, mimiccxr_output_filepath)

    # Load challenge dataframes
    
    challenge_train_df = pd.read_csv(train_filepath)
    challenge_val_df = pd.read_csv(val_filepath)

    # CheXpert splits
    chexpert_output_filepath = os.path.join(output_dir, 'chexpert_splits.pkl')
    if os.path.exists(chexpert_output_filepath):
        print_blue(f"CheXpert splits already exist at {chexpert_output_filepath}", bold=True)
    else:
        print_bold("CheXpert splits")

        chexpert_train_val_df = pd.read_csv(CHEXPERT_TRAIN_VAL_CSV_PATH)
        chexpert_test_df = pd.read_csv(CHEXPERT_TEST_LABELS_CSV_PATH)

        all_chexpert_images = set()
        all_chexpert_images.update([path[20:] for path in chexpert_train_val_df['Path']])
        all_chexpert_images.update([os.path.join('test', path[5:]) for path in chexpert_test_df['Path']])
        print(f"CheXpert images: {len(all_chexpert_images)}")

        challenge_train_chexpert_df = challenge_train_df.loc[challenge_train_df.source == 'CheXpert']
        challenge_val_chexpert_df = challenge_val_df.loc[challenge_val_df.source == 'CheXpert']
        assert len(challenge_train_chexpert_df) > 0
        assert len(challenge_val_chexpert_df) > 0

        train_val_chexpert_images_set = set()
        
        chexpert_train_images = []
        for images_path in challenge_train_chexpert_df.images_path_old.values:
            images_path_ = eval(images_path) # convert string to list
            for image_path in images_path_:
                for x in image_path.split('.jpg'):
                    if x:
                        x = x[21:] + '.jpg'
                        assert x in all_chexpert_images, (x, images_path_)
                        chexpert_train_images.append(x)
                        train_val_chexpert_images_set.add(x)

        chexpert_val_images = []
        for images_path in challenge_val_chexpert_df.images_path_old.values:
            images_path = eval(images_path) # convert string to list
            for image_path in images_path:
                for x in image_path.split('.jpg'):
                    if x:
                        x = x[21:] + '.jpg'
                        assert x in all_chexpert_images, (x, images_path)
                        chexpert_val_images.append(x)
                        train_val_chexpert_images_set.add(x)

        chexpert_test_images = [ip for ip in all_chexpert_images if ip not in train_val_chexpert_images_set]

        print(f"CheXpert splits: train={len(chexpert_train_images)}, val={len(chexpert_val_images)}, test={len(chexpert_test_images)}")

        chexpert_output = {
            'train': chexpert_train_images,
            'val': chexpert_val_images,
            'test': chexpert_test_images
        }
        print_blue(f"Saving CheXpert splits to {chexpert_output_filepath}", bold=True)
        save_pickle(chexpert_output, chexpert_output_filepath)

    # OpenI splits
    openi_output_filepath = os.path.join(output_dir, 'openi_splits.pkl')
    if os.path.exists(openi_output_filepath):
        print_blue(f"OpenI splits already exist at {openi_output_filepath}", bold=True)
    else:
        print_bold("OpenI splits")

        challenge_train_openi_df = challenge_train_df.loc[challenge_train_df.source == 'OpenI']
        challenge_val_openi_df = challenge_val_df.loc[challenge_val_df.source == 'OpenI']
        assert len(challenge_train_openi_df) > 0
        assert len(challenge_val_openi_df) > 0

        iuxray_image_ids = set(get_iuxray_all_image_ids())
        print(f"IU X-ray images: {len(iuxray_image_ids)}")

        train_val_openi_images_set = set()

        openi_train_image_ids = []
        for images_path in challenge_train_openi_df.images_path_old.values:
            images_path = eval(images_path) # convert string to list
            for image_path in images_path:
                for x in image_path.split('.png'):
                    if x:
                        image_id = os.path.basename(x)
                        assert image_id in iuxray_image_ids
                        openi_train_image_ids.append(image_id)
                        train_val_openi_images_set.add(image_id)

        openi_val_image_ids = []
        for images_path in challenge_val_openi_df.images_path_old.values:
            images_path = eval(images_path) # convert string to list
            for image_path in images_path:
                for x in image_path.split('.png'):
                    if x:
                        image_id = os.path.basename(x)
                        assert image_id in iuxray_image_ids
                        openi_val_image_ids.append(image_id)
                        train_val_openi_images_set.add(image_id)

        openi_test_image_ids = [image_id for image_id in iuxray_image_ids if image_id not in train_val_openi_images_set]

        print(f"OpenI splits: train={len(openi_train_image_ids)}, val={len(openi_val_image_ids)}, test={len(openi_test_image_ids)}")

        openi_output = {
            'train': openi_train_image_ids,
            'val': openi_val_image_ids,
            'test': openi_test_image_ids
        }
        print_blue(f"Saving OpenI splits to {openi_output_filepath}", bold=True)
        save_pickle(openi_output, openi_output_filepath)

    print_magenta("Done", bold=True)



    
    