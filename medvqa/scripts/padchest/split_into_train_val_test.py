from medvqa.datasets.padchest import (
    PADCHEST_IMAGES_SMALL_DIR,
    PADCHEST_CACHE_DIR,
    PADCHEST_LABELS_CSV_PATH,
)
from medvqa.utils.files import save_to_txt, read_lines_from_txt
from medvqa.utils.common import get_timestamp
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd
import os
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_fraction', type=float, default=0.03)
    parser.add_argument('--test_fraction', type=float, default=0.03)
    return parser.parse_args()

def find_broken_images():
    broken_images = []
    for image_filename in tqdm(os.listdir(PADCHEST_IMAGES_SMALL_DIR)):
        image_path = os.path.join(PADCHEST_IMAGES_SMALL_DIR, image_filename)
        try:
            img = Image.open(image_path)
        except UnidentifiedImageError:
            broken_images.append(image_path)
            continue
        try:
            img.resize((1, 1))
        except OSError:
            broken_images.append(image_path)
        except SyntaxError:
            broken_images.append(image_path)
    return broken_images

if __name__ == '__main__':

    # Parse args
    args = parse_args()

    # Find broken images
    broken_images_path = os.path.join(PADCHEST_CACHE_DIR, 'broken_images.txt')
    if os.path.exists(broken_images_path):
        print(f'Found {broken_images_path}')
        broken_images = read_lines_from_txt(broken_images_path)
        print(f'Found {len(broken_images)} broken images')
    else:
        broken_images = find_broken_images()
        print(f'Found {len(broken_images)} broken images')
        save_to_txt(broken_images, broken_images_path)
        print(f'Broken images written to {broken_images_path}')

    # Broken image filenames
    broken_image_filenames = [os.path.basename(broken_image) for broken_image in broken_images]
    broken_image_filenames = set(broken_image_filenames)
    
    # Load labels
    labels_df = pd.read_csv(PADCHEST_LABELS_CSV_PATH)
    print(f'len(labels_df) = {len(labels_df)}')

    # Remove broken images from labels
    labels_df = labels_df[~labels_df['ImageID'].isin(broken_image_filenames)]
    print(f'len(labels_df) = {len(labels_df)} (after removing broken images)')

    # Drop nan labels
    labels_df = labels_df.dropna(subset=['Labels'])
    print(f'len(labels_df) = {len(labels_df)} (after dropping nan labels)')

    # Split into train, val, test
    study_id2labels = {}
    for study_id, labels in zip(labels_df['StudyID'], labels_df['Labels']):
        try:
            labels = eval(labels)
            labels = [label.strip() for label in labels]
            labels = [label for label in labels if label != '']
        except TypeError:
            print(f'labels = {labels}')
            raise
        if study_id in study_id2labels:
            assert study_id2labels[study_id] == labels
        else:
            study_id2labels[study_id] = labels
    study_id_list = list(study_id2labels.keys())
    study_id2index = {study_id: i for i, study_id in enumerate(study_id_list)}
    
    label2study_ids = {}
    for study_id, labels in study_id2labels.items():
        for label in labels:
            if label not in label2study_ids:
                label2study_ids[label] = []
            label2study_ids[label].append(study_id)
    label_list = list(label2study_ids.keys())
    print(f'len(label_list) = {len(label_list)}')

    used_study_ids = set()
    train_study_ids = []
    val_study_ids = []
    test_study_ids = []
    val_label_counts = {label:0 for label in label_list}
    test_label_counts = {label:0 for label in label_list}

    # Collect val study ids
    i = 0
    max_attempts = int(args.val_fraction * len(study_id_list) * 10)
    while len(val_study_ids) < args.val_fraction * len(study_id_list) and i < max_attempts:
        label = label_list[i % len(label_list)]
        i += 1
        target_count = len(label2study_ids[label]) * args.val_fraction
        if  target_count < 2:
            continue
        target_count = int(target_count)
        if val_label_counts[label] >= target_count:
            continue
        for _ in range(target_count - val_label_counts[label]):
            study_id = random.choice(label2study_ids[label])
            if study_id in used_study_ids:
                continue
            used_study_ids.add(study_id)
            val_study_ids.append(study_id)
            for label in study_id2labels[study_id]:
                val_label_counts[label] += 1

    # Collect test study ids
    i = 0
    max_attempts = int(args.test_fraction * len(study_id_list) * 10)
    while len(test_study_ids) < args.test_fraction * len(study_id_list) and i < max_attempts:
        label = label_list[i % len(label_list)]
        i += 1
        target_count = len(label2study_ids[label]) * args.test_fraction
        if  target_count < 2:
            continue
        target_count = int(target_count)
        if test_label_counts[label] >= target_count:
            continue
        for _ in range(target_count - test_label_counts[label]):
            study_id = random.choice(label2study_ids[label])
            if study_id in used_study_ids:
                continue
            used_study_ids.add(study_id)
            test_study_ids.append(study_id)
            for label in study_id2labels[study_id]:
                test_label_counts[label] += 1
    
    # Collect train study ids
    for study_id in study_id_list:
        if study_id in used_study_ids:
            continue
        train_study_ids.append(study_id)

    # Sanity check
    assert len(train_study_ids) + len(val_study_ids) + len(test_study_ids) == len(study_id_list)
    assert len(set(train_study_ids) & set(val_study_ids)) == 0
    assert len(set(train_study_ids) & set(test_study_ids)) == 0
    assert len(set(val_study_ids) & set(test_study_ids)) == 0

    # Print split stats
    print(f'len(train_study_ids) = {len(train_study_ids)}')
    print(f'len(val_study_ids) = {len(val_study_ids)}')
    print(f'len(test_study_ids) = {len(test_study_ids)}')
    print(f'len(train_study_ids) + len(val_study_ids) + len(test_study_ids) = {len(train_study_ids) + len(val_study_ids) + len(test_study_ids)}')
    print(f'len(study_id_list) = {len(study_id_list)}')

    # Write train, val, test study ids
    timestamp = get_timestamp()
    save_to_txt(train_study_ids, os.path.join(PADCHEST_CACHE_DIR, f'train_study_ids_{timestamp}.txt'))
    save_to_txt(val_study_ids, os.path.join(PADCHEST_CACHE_DIR, f'val_study_ids_{timestamp}.txt'))
    save_to_txt(test_study_ids, os.path.join(PADCHEST_CACHE_DIR, f'test_study_ids_{timestamp}.txt'))
    print(f'Wrote train, val, test study ids to {PADCHEST_CACHE_DIR}')