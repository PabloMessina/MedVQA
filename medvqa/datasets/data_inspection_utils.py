import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imagesize
from tqdm import tqdm
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAMES
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_silver_bboxes,
    load_postprocessed_label_names,
)
from medvqa.metrics.bbox.utils import compute_iou
from medvqa.utils.metrics import (
    chexpert_label_array_to_string,
    chest_imagenome_label_array_to_string,
)
from medvqa.datasets.image_processing import inv_normalize
from medvqa.utils.constants import CHEXPERT_GENDERS, CHEXPERT_ORIENTATIONS
from medvqa.utils.files import get_cached_json_file
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR, MIMICCXR_IMAGE_REGEX, get_mimiccxr_large_image_path

def inspect_chexpert_vision_trainer(chexpert_vision_trainer, i):
    instance = chexpert_vision_trainer.dataset[i]
    idx = instance['idx']
    print('idx:', idx)
    print('chexpert labels:', instance['l'])
    print('chexpert labels (verbose):', chexpert_label_array_to_string(instance['l']))
    print('orientation:', instance['o'], CHEXPERT_ORIENTATIONS[instance['o']])
    print('sex:', instance['g'], CHEXPERT_GENDERS[instance['g']])
    print('image from tensor:')
    img = Image.fromarray((inv_normalize(instance['i']).permute(1,2,0) * 255).numpy().astype(np.uint8))
    plt.imshow(img)
    plt.show()
    print('image from path:')
    print(chexpert_vision_trainer.dataset.images[idx])
    img = Image.open(chexpert_vision_trainer.dataset.images[idx]).convert('RGB')
    plt.imshow(img)
    plt.show()

def inspect_iuxray_vision_trainer(iuxray_vision_trainer, split, i):
    if split == 'train':
        dataset = iuxray_vision_trainer.train_dataset
    else:
        assert split == 'validation'
        dataset = iuxray_vision_trainer.val_dataset
    instance = dataset[i]
    idx = instance['idx']
    print('idx:', idx)
    print('chexpert labels:', instance['chexpert'])
    print('chexpert labels (verbose):', chexpert_label_array_to_string(instance['chexpert']))
    print('orientation:', instance['orientation'])
    print('question labels:', instance['qlabels'])
    print('image from tensor:')
    img = Image.fromarray((inv_normalize(instance['i']).permute(1,2,0) * 255).numpy().astype(np.uint8))
    plt.imshow(img)
    plt.show()
    print('image from path:')
    print(iuxray_vision_trainer.images[idx])
    img = Image.open(iuxray_vision_trainer.images[idx]).convert('RGB')
    plt.imshow(img)
    plt.show()    

def inspect_mimiccxr_vision_trainer(mimiccxr_vision_trainer, split, i):
    if split == 'train':
        dataset = mimiccxr_vision_trainer.train_dataset
    else:
        assert split == 'validation'
        dataset = mimiccxr_vision_trainer.val_dataset
    instance = dataset[i]
    idx = instance['idx']
    print('idx:', idx)
    print('chexpert labels:', instance['chexpert'])
    print('chexpert labels (verbose):', chexpert_label_array_to_string(instance['chexpert']))
    print('orientation:', instance['orientation'])
    print('question labels:', instance['qlabels'])
    print('image from tensor:')
    img = Image.fromarray((inv_normalize(instance['i']).permute(1,2,0) * 255).numpy().astype(np.uint8))
    plt.imshow(img)
    plt.show()
    print('image from path:')
    print(mimiccxr_vision_trainer.images[idx])
    img = Image.open(mimiccxr_vision_trainer.images[idx]).convert('RGB')
    plt.imshow(img)
    plt.show()

def _inspect_vqa_trainer(vqa_trainer, cache_dir, dataset_name, i, figsize=(10, 10)):
    assert hasattr(vqa_trainer, dataset_name)
    dataset = getattr(vqa_trainer, dataset_name)

    instance = dataset[i]
    idx = instance['idx']    
    
    # Idx
    print('idx:', idx)

    # Orientation
    if vqa_trainer.classify_orientation:
        print('orientation:', instance['orientation'])
    
    # Question labels
    if vqa_trainer.classify_questions:
        print('question labels:', instance['qlabels'])
    
    # Question
    print('question:', instance['q'])
    # qa_reports = get_cached_json_file(os.path.join(cache_dir, vqa_trainer.qa_adapted_reports_filename))
    # print('question:', instance['q'], qa_reports['questions'][instance['q']])
    if vqa_trainer.classify_questions:
        assert instance['qlabels'][instance['q']] == 1
    #     
    # Answer
    print('answer:', vqa_trainer.tokenizer.ids2string(instance['a']))

    # Image from tensor
    print('image from tensor:')
    img = Image.fromarray((inv_normalize(instance['i']).permute(1,2,0) * 255).numpy().astype(np.uint8))
    plt.imshow(img)
    plt.show()

    # Image from path
    print('image from path:')
    print(vqa_trainer.images[idx])
    img = Image.open(vqa_trainer.images[idx]).convert('RGB')
    plt.imshow(img)
    plt.show()

    # Report
    print('post-processed report:')
    qa_reports = get_cached_json_file(os.path.join(cache_dir, vqa_trainer.qa_adapted_reports_filename))
    rid = vqa_trainer.report_ids[idx]
    print(qa_reports['reports'][rid])

    # Print original report
    print()
    print('original report:')
    report_path = qa_reports['reports'][rid]['filepath']
    with open(report_path, 'r') as f:
        print(f.read())

    # Chexpert labels
    if vqa_trainer.classify_chexpert:
        print('chexpert labels:', instance['chexpert'])
        print('chexpert labels (verbose):', chexpert_label_array_to_string(instance['chexpert']))

    # Chest ImaGenome labels
    if hasattr(vqa_trainer, 'classify_chest_imagenome'):
        if vqa_trainer.classify_chest_imagenome:
            if hasattr(vqa_trainer, 'chest_imagenome_label_names'):
                chest_imagenome_label_names = vqa_trainer.chest_imagenome_label_names
            else:
                chest_imagenome_label_names = load_postprocessed_label_names(
                    vqa_trainer.chest_imagenome_label_names_filename)
            print('chest imagenome labels:', instance['chest_imagenome'])
            print('chest imagenome labels (verbose):',
                chest_imagenome_label_array_to_string(instance['chest_imagenome'],
                chest_imagenome_label_names))

    # Chest ImaGenome bounding boxes
    if hasattr(vqa_trainer, 'predict_bboxes_chest_imagenome'):
        if vqa_trainer.predict_bboxes_chest_imagenome:
            bbox_coords = instance['chest_imagenome_bbox_coords']
            bbox_presence = instance['chest_imagenome_bbox_presence']
            assert len(bbox_coords) == len(bbox_presence) * 4
            print('chest imagenome bbox coords:', bbox_coords)
            print('chest imagenome bbox presence:', bbox_presence)
            print('dicom id:', vqa_trainer.dicom_ids[idx])
            bboxes_dict = load_chest_imagenome_silver_bboxes()
            bbox = bboxes_dict[vqa_trainer.dicom_ids[idx]]
            print('bbox:', bbox)
            # Create figure and axes
            fig, ax = plt.subplots(1, figsize=figsize)            
            # Get the current image size
            img_large_path = get_mimiccxr_large_image_path(*MIMICCXR_IMAGE_REGEX.findall(vqa_trainer.images[idx])[0])
            width, height = imagesize.get(img_large_path)
            # Load large image
            img_large = Image.open(img_large_path).convert('RGB')
            assert img_large.width == width
            assert img_large.height == height
            # Display the image
            ax.imshow(img_large)
            # Create a Rectangle patch for each bbox            
            for i in range(len(bbox_presence)):
                if bbox_presence[i] == 0:
                    continue
                x1 = bbox_coords[i*4] * img_large.width
                y1 = bbox_coords[i*4+1] * img_large.height
                x2 = bbox_coords[i*4+2] * img_large.width
                y2 = bbox_coords[i*4+3] * img_large.height
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor=plt.cm.tab20(i), facecolor='none')
                ax.add_patch(rect)
                # add a label (make it transparent if it is not in labels)
                ax.text(x1, y1-3, CHEST_IMAGENOME_BBOX_NAMES[i], fontsize=16, color=plt.cm.tab20(i))                
                print(f'Object: {CHEST_IMAGENOME_BBOX_NAMES[i]} ({x1:.1f}, {y1:.1f}, {x2-x1:.1f}, {y2-y1:.1f})')


def inspect_iuxray_vqa_trainer(iuxray_vqa_trainer, dataset_name, i):
    _inspect_vqa_trainer(iuxray_vqa_trainer, IUXRAY_CACHE_DIR, dataset_name, i)

def inspect_mimiccxr_vqa_trainer(mimiccxr_vqa_trainer, dataset_name, i):
    _inspect_vqa_trainer(mimiccxr_vqa_trainer, MIMICCXR_CACHE_DIR, dataset_name, i)

_shared_image_id_to_binary_labels = None
def _count_labels(idx):
        count = 0
        for binary_labels in _shared_image_id_to_binary_labels.values():
            if binary_labels[idx] == 1:
                count += 1
        return count, idx
def _count_labels_in_parallel(label_idxs, label_alias, image_id_to_binary_labels, num_workers=10):
    from multiprocessing import Pool
    import time
    global _shared_image_id_to_binary_labels
    _shared_image_id_to_binary_labels = image_id_to_binary_labels
    print(f'Num {label_alias} labels:', len(label_idxs))
    print(f'Counting labels in parallel ({num_workers} workers)...')
    start = time.time()
    with Pool(num_workers) as p:
        label_counts = p.map(_count_labels, label_idxs)
    print(f'Done counting labels ({time.time() - start:.1f}s)')
    return label_counts

def _plot_chest_imagenome_label_distribution(label_idxs, label_alias, label_names, image_id_to_binary_labels,
        num_workers=10, figsize=(8, 6), n_splits=3):
    label_counts = _count_labels_in_parallel(label_idxs, label_alias, image_id_to_binary_labels, num_workers)
    label_counts = sorted(label_counts, key=lambda x: x[0])
    label_names = [label_names[i] for _, i in label_counts]
    label_counts = [count for count, _ in label_counts]
    total = len(image_id_to_binary_labels)    
    xlim = None
    # Generate n_splits plots
    for i in range(n_splits):
        i = n_splits - i - 1
        a = i * len(label_counts) // n_splits
        b = (i+1) * len(label_counts) // n_splits
        # Vertical bar chart
        plt.figure(figsize=figsize)
        if xlim is not None:
            plt.xlim(xlim)
        plt.barh(range(b-a), label_counts[a:b])
        yticks = [f'{str(label_name)} ({count}) ({count/total:.3%})' for\
            label_name, count in zip(label_names[a:b], label_counts[a:b])]
        plt.yticks(range(b-a), yticks)
        if xlim is None:
            xlim = plt.gca().get_xlim()
        plt.show()

def plot_chest_imagenome_localized_label_distribution(label_names, image_id_to_binary_labels,
        num_workers=10, figsize=(8, 6), n_splits=6):
    localized_idxs = [i for i, label_name in enumerate(label_names) if len(label_name) == 3]
    _plot_chest_imagenome_label_distribution(localized_idxs, 'localized', label_names, image_id_to_binary_labels,
        num_workers=num_workers, figsize=figsize, n_splits=n_splits) 

def plot_chest_imagenome_nonlocalized_label_distribution(label_names, image_id_to_binary_labels,
        num_workers=10, figsize=(8, 6), n_splits=3):
    nonlocalized_idxs = [i for i, label_name in enumerate(label_names) if len(label_name) == 2]
    _plot_chest_imagenome_label_distribution(nonlocalized_idxs, 'nonlocalized', label_names, image_id_to_binary_labels,
        num_workers=num_workers, figsize=figsize, n_splits=n_splits)

def _plot_chest_imagenome_label_frequency_histogram(label_idxs, label_alias, label_names, image_id_to_binary_labels,
        num_workers=10, figsize=(8, 6)):
    label_counts = _count_labels_in_parallel(label_idxs, label_alias, image_id_to_binary_labels, num_workers)    
    label_counts = [count for count, _ in label_counts]
    plt.figure(figsize=figsize)
    plt.title(f'Label frequency histogram ({label_alias} labels)')
    plt.hist(label_counts, bins=100)
    plt.xlabel('Number of images')
    plt.ylabel('Number of labels')
    plt.show()

def plot_chest_imagenome_localized_label_frequency_histogram(label_names, image_id_to_binary_labels,
        num_workers=10, figsize=(8, 6)):
    localized_idxs = [i for i, label_name in enumerate(label_names) if len(label_name) == 3]
    _plot_chest_imagenome_label_frequency_histogram(localized_idxs, 'localized', label_names, image_id_to_binary_labels,
        num_workers=num_workers, figsize=figsize)

def plot_chest_imagenome_nonlocalized_label_frequency_histogram(label_names, image_id_to_binary_labels,
        num_workers=10, figsize=(8, 6)):
    nonlocalized_idxs = [i for i, label_name in enumerate(label_names) if len(label_name) == 2]
    _plot_chest_imagenome_label_frequency_histogram(nonlocalized_idxs, 'nonlocalized', label_names, image_id_to_binary_labels,
        num_workers=num_workers, figsize=figsize)

def plot_chest_imagenome_iou_histogram(average_bbox_coords, dicom_ids, figsize=(8, 6), return_iou=False):
    bbox_dict = load_chest_imagenome_silver_bboxes()
    ious = [None] * len(dicom_ids)
    for i, dicom_id in enumerate(dicom_ids):
        bboxes = bbox_dict[dicom_id]
        coords = bboxes['coords']
        presence = bboxes['presence']
        mean_iou = 0
        for j in range(len(presence)):
            if presence[j] == 1:
                mean_iou += compute_iou(average_bbox_coords[4*j:4*j+4], coords[4*j:4*j+4])
        mean_iou /= sum(presence)
        ious[i] = mean_iou
    plt.figure(figsize=figsize)
    plt.title('IoU histogram')
    plt.hist(ious, bins=100)
    plt.xlabel('IoU')
    plt.ylabel('Number of images')
    plt.show()
    if return_iou:
        idxs = np.argsort(ious)
        return {
            'ious': [ious[i] for i in idxs],
            'dicom_ids': [dicom_ids[i] for i in idxs]
        }

