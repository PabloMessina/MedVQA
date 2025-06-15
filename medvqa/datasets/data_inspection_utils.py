import math
import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imagesize
from time import time
from multiprocessing import Pool

import torch
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAMES, CHEST_IMAGENOME_GOLD_BBOX_NAMES, CHEST_IMAGENOME_NUM_BBOX_CLASSES, CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES
from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
    load_chest_imagenome_gold_bboxes,
    load_chest_imagenome_silver_bboxes,
    load_chest_imagenome_label_names,
)
from medvqa.metrics.bbox.utils import compute_iou
from medvqa.datasets.image_processing import inv_normalize
from medvqa.utils.constants import CHEXPERT_GENDERS, CHEXPERT_ORIENTATIONS
from medvqa.utils.files_utils import get_cached_json_file
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR, MIMICCXR_IMAGE_REGEX, get_mimiccxr_large_image_path
from medvqa.utils.logging_utils import chest_imagenome_label_array_to_string, chexpert_label_array_to_string, print_bold

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
                chest_imagenome_label_names = load_chest_imagenome_label_names(
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


def inspect_labels2report_trainer(trainer, dataset_name, i):

    assert hasattr(trainer, dataset_name)
    dataset = getattr(trainer, dataset_name)

    instance = dataset[i]
    idx = instance['idx']
    
    # Idx
    print('idx:', idx)
    
    # Report (from encoding)
    print()
    print('report (from encoding):', trainer.tokenizer.ids2string(instance['report']))

    # Report (from file)
    print()
    print('report (qa adapted):')
    qa_reports = get_cached_json_file(os.path.join(MIMICCXR_CACHE_DIR, trainer.qa_adapted_reports_filename))
    rid = trainer.report_ids[idx]
    print(qa_reports['reports'][rid])

    # Print original report
    print()
    print('original report:')
    report_path = qa_reports['reports'][rid]['filepath']
    with open(report_path, 'r') as f:
        print(f.read())

    # Chexpert labels
    if trainer.use_chexpert:
        print('chexpert labels:', instance['chexpert'])
        print('chexpert labels (verbose):', chexpert_label_array_to_string(instance['chexpert']))

    # Chest ImaGenome labels
    if trainer.use_chest_imagenome:
        chest_imagenome_label_names = trainer.chest_imagenome_label_names
        print('chest imagenome labels:', instance['chest_imagenome'])
        print('chest imagenome labels (verbose):',
            chest_imagenome_label_array_to_string(instance['chest_imagenome'],
            chest_imagenome_label_names))
        

def inspect_mscxr_dataset(mimiccxr_trainer, i, train=True, grounding_only_mode=False,
                          image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)):
    from medvqa.datasets.mimiccxr import get_detailed_metadata_for_dicom_id
    from medvqa.utils.files_utils import read_txt
    if train:
        dataset = mimiccxr_trainer.mscxr_train_dataset
        idx = dataset.indices[i]
        if grounding_only_mode:
            phrase_bboxes = dataset.phrase_bboxes[idx]
            phrase_idx = dataset.phrase_idxs[idx]
        else:
            phrase_bboxes, phrase_classes = dataset.phrase_bboxes_and_classes[idx]
            phrase_idxs = dataset.phrase_idxs[idx]
        if not grounding_only_mode:
            pos_fact_idxs = dataset.positive_fact_idxs[idx]
            strong_neg_fact_idxs = dataset.strong_neg_fact_idxs[idx]
            weak_neg_fact_idxs = dataset.weak_neg_fact_idxs[idx]
        image_path = dataset.image_paths[idx]
        dicom_id = os.path.basename(image_path).split('.')[0]
        metadata = get_detailed_metadata_for_dicom_id(dicom_id)[0]
        report = read_txt(metadata['filepath'])
        instance = dataset[i]
        image_tensor = instance['i'] # (3, H, W)
        if grounding_only_mode:
            phrase_embedding = instance['pe'] # (128,)
            target_bbox_coords = instance['tbc'] # (num_regions, 4)
            target_bbox_presence = instance['tbp'] # (num_regions,)
            target_prob_mask = instance['tpm'] # (num_regions,)
        else:
            phrase_embeddings = instance['pe'] # (num_phrases, 128)
            phrase_classifier_labels = instance['pcl'] # (num_phrases, num_classes)
            target_bbox_coords = instance['tbc'] # (num_phrases, num_regions, 4)
            target_bbox_presence = instance['tbp'] # (num_phrases, num_regions)
            gidxs = instance['gidxs'] # (num_phrases_with_bbox,)

        num_regions = target_bbox_coords.shape[-2]
        feat_W = feat_H = math.isqrt(num_regions)
        assert feat_W * feat_H == num_regions

        print_bold('idx:', end=' '); print(idx)
        print_bold('dicom_id:', end=' '); print(dicom_id)
        print_bold('image_path:', end=' '); print(image_path)
        print_bold('report:'); print(report)
        print_bold('metadata:'); print(metadata)
        print_bold('image_tensor:', end=' '); print(image_tensor.shape)
        print_bold(f'num_regions: {num_regions}, feat_W: {feat_W}, feat_H: {feat_H}')
        if grounding_only_mode:
            print_bold('phrase_embedding:', end=' '); print(phrase_embedding.shape)
            print_bold('target_bbox_coords:', end=' '); print(target_bbox_coords.shape)
            print_bold('target_bbox_presence:', end=' '); print(target_bbox_presence.shape)
            print_bold('target_prob_mask:', end=' '); print(target_prob_mask.shape)
        else:
            print_bold('phrase_embeddings:', end=' '); print(phrase_embeddings.shape)
            print_bold('phrase_classifier_labels:', end=' '); print(phrase_classifier_labels.shape)
            print_bold('target_bbox_coords:', end=' '); print(target_bbox_coords.shape)
            print_bold('target_bbox_presence:', end=' '); print(target_bbox_presence.shape)
            print_bold('gidxs:', end=' '); print(gidxs.shape)

        # Obtain bbox coordinates from target
        target_bbox_coords_ = target_bbox_coords.view(-1, 4)
        target_bbox_presence_ = target_bbox_presence.view(-1)
        bbox_coords = set()
        for i in range(len(target_bbox_presence_)):
            if target_bbox_presence_[i] == 1:
                bbox_coords.add(tuple(target_bbox_coords_[i].tolist()))
        bbox_coords = list(bbox_coords)
        bbox_format = dataset.bbox_format
        print_bold('bbox_coords:', end=' '); print(bbox_coords)
        print_bold('bbox_format:', end=' '); print(bbox_format)

        # Recover image from tensor
        image_tensor.mul_(torch.tensor(image_std).view(3, 1, 1))
        image_tensor.add_(torch.tensor(image_mean).view(3, 1, 1))
        image_tensor = torch.clamp(image_tensor, 0, 1)
        img = Image.fromarray((image_tensor.permute(1,2,0) * 255).numpy().astype(np.uint8))
        W_img = img.width
        H_img = img.height

        # Display transformed image + bboxes
        print_bold('Transformed image:')
        
        if grounding_only_mode:
            # Start visualization.
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Compute grid lines positions based on the image size and feature map size.
            cell_width = W_img / feat_W
            cell_height = H_img / feat_H
            # Grid lines (excluding image borders):
            vlines = [cell_width * i for i in range(1, feat_W)]
            hlines = [cell_height * i for i in range(1, feat_H)]

            # Plot 1: Original image with bounding boxes.
            axes[0].imshow(img)
            axes[0].set_title("Image with Bounding Boxes")
            for bbox_coord in bbox_coords:
                if bbox_format == 'xyxy':
                    x1, y1, x2, y2 = bbox_coord
                elif bbox_format == 'cxcywh':
                    cx, cy, w, h = bbox_coord
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                else: raise ValueError(f'Invalid bbox_format: {bbox_format}')
                x1 *= img.width
                y1 *= img.height
                x2 *= img.width
                y2 *= img.height
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                axes[0].add_patch(rect)
            for x in vlines:
                axes[0].axvline(x, color="white", linestyle="--", linewidth=1)
            for y in hlines:
                axes[0].axhline(y, color="white", linestyle="--", linewidth=1)

            # Plot 2: Upsampled target presence heatmap with grid.
            target_bbox_presence = target_bbox_presence.view(feat_H, feat_W)
            up_presence = np.array(
                Image.fromarray((target_bbox_presence.numpy() * 255).astype(np.uint8)).resize(
                    (W_img, H_img), resample=Image.NEAREST
                )
            ) / 255.0
            axes[1].imshow(up_presence, cmap="viridis")
            axes[1].set_title("Target Presence Heatmap")
            for x in vlines:
                axes[1].axvline(x, color="white", linestyle="--", linewidth=1)
            for y in hlines:
                axes[1].axhline(y, color="white", linestyle="--", linewidth=1)

            # Plot 3: Upsampled prob mask with grid.
            target_prob_mask = target_prob_mask.view(feat_H, feat_W)
            up_prob_mask = cv2.resize(
                target_prob_mask.numpy(),
                (W_img, H_img), interpolation=cv2.INTER_NEAREST,
            )
            axes[2].imshow(up_prob_mask, cmap="viridis")
            axes[2].set_title("Target Prob Mask")
            for x in vlines:
                axes[2].axvline(x, color="white", linestyle="--", linewidth=1)
            for y in hlines:
                axes[2].axhline(y, color="white", linestyle="--", linewidth=1)

            plt.show()
        else:
            plt.imshow(img)
            ax = plt.gca()
            for bbox_coord in bbox_coords:
                if bbox_format == 'xyxy':
                    x1, y1, x2, y2 = bbox_coord
                elif bbox_format == 'cxcywh':
                    cx, cy, w, h = bbox_coord
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                else: raise ValueError(f'Invalid bbox_format: {bbox_format}')
                x1 *= img.width
                y1 *= img.height
                x2 *= img.width
                y2 *= img.height
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()

        # Display original image
        print_bold('Original image:')
        img = Image.open(image_path).convert('RGB')
        plt.imshow(img)
        plt.show()

        print_bold('Phrase bboxes:', end=' '); print(phrase_bboxes)
        if grounding_only_mode:
            print_bold('Phrase idx:', end=' '); print(phrase_idx)
        else:
            print_bold('Phrase classes:', end=' '); print(phrase_classes)
            print_bold('Phrase idxs:', end=' '); print(phrase_idxs)
            print_bold('Positive fact idxs:', end=' '); print(pos_fact_idxs)
            print_bold('Strong negative fact idxs:', end=' '); print(strong_neg_fact_idxs)
            print_bold('Weak negative fact idxs:', end=' '); print(weak_neg_fact_idxs)

        if grounding_only_mode:
            print_bold('Phrase:', end=' '); print(mimiccxr_trainer.mscxr_phrases[phrase_idx])
        else:
            print_bold('Phrases:')
            for i in phrase_idxs:
                print(mimiccxr_trainer.mscxr_phrases[i])

            print_bold('Positive facts:')
            for i in pos_fact_idxs:
                print(mimiccxr_trainer.mscxr_facts[i])

            print_bold('Strong negative facts:')
            for i in strong_neg_fact_idxs:
                print(mimiccxr_trainer.mscxr_facts[i])

            print_bold('Weak negative facts:')
            for i in weak_neg_fact_idxs[:20]:
                print(mimiccxr_trainer.mscxr_facts[i])
    
    else:

        dataset = mimiccxr_trainer.mscxr_val_dataset
        idx = dataset.indices[i]
        image_path = dataset.image_paths[idx]
        dicom_id = os.path.basename(image_path).split('.')[0]
        metadata = get_detailed_metadata_for_dicom_id(dicom_id)[0]
        report = read_txt(metadata['filepath'])
        instance = dataset[i]
        if grounding_only_mode:
            phrase_idx = dataset.phrase_idxs[idx]
        else:
            phrase_idxs = dataset.phrase_idxs[idx]
            pos_fact_idxs = dataset.positive_fact_idxs[idx]
            strong_neg_fact_idxs = dataset.strong_neg_fact_idxs[idx]
            weak_neg_fact_idxs = dataset.weak_neg_fact_idxs[idx]
        image_tensor = instance['i'] # (3, H, W)
        if grounding_only_mode:
            phrase_embedding = instance['pe'] # (128,)
            phrase_bboxes = instance['bboxes'] # (list of 4-tuples)
        else:
            phrase_embeddings = instance['pe'] # (num_phrases, 128)
            phrase_classifier_labels = instance['pcl'] # (num_phrases, num_classes)
            phrase_bboxes = instance['bboxes'] # (list of 4-tuples)
            phrase_classes = instance['classes'] # (list of ints)
        bbox_format = dataset.bbox_format

        print_bold('idx:', end=' '); print(idx)
        print_bold('dicom_id:', end=' '); print(dicom_id)
        print_bold('image_path:', end=' '); print(image_path)
        print_bold('report:'); print(report)
        print_bold('metadata:'); print(metadata)
        print_bold('image_tensor:', end=' '); print(image_tensor.shape)
        if grounding_only_mode:
            print_bold('phrase_embedding:', end=' '); print(phrase_embedding.shape)
            print_bold('phrase_bboxes:', end=' '); print(phrase_bboxes)
        else:
            print_bold('phrase_embeddings:', end=' '); print(phrase_embeddings.shape)
            print_bold('phrase_classifier_labels:', end=' '); print(phrase_classifier_labels.shape)
            print_bold('phrase_bboxes:', end=' '); print(phrase_bboxes)
            print_bold('phrase_classes:', end=' '); print(phrase_classes)
        print_bold('bbox_format:', end=' '); print(bbox_format)

        # Recover image from tensor
        image_tensor.mul_(torch.tensor(image_std).view(3, 1, 1))
        image_tensor.add_(torch.tensor(image_mean).view(3, 1, 1))
        image_tensor = torch.clamp(image_tensor, 0, 1)
        img = Image.fromarray((image_tensor.permute(1,2,0) * 255).numpy().astype(np.uint8))

        # Display transformed image + bboxes
        print_bold('Transformed image:')
        plt.imshow(img)
        ax = plt.gca()
        for bbox_coord in phrase_bboxes:
            if bbox_format == 'xyxy':
                x1, y1, x2, y2 = bbox_coord
            elif bbox_format == 'cxcywh':
                cx, cy, w, h = bbox_coord
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
            else: raise ValueError(f'Invalid bbox_format: {bbox_format}')
            x1 *= img.width
            y1 *= img.height
            x2 *= img.width
            y2 *= img.height
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

        # Display original image
        print_bold('Original image:')
        img = Image.open(image_path).convert('RGB')
        plt.imshow(img)
        plt.show()

        if grounding_only_mode:
            print_bold('Phrase idx:', end=' '); print(phrase_idx)
            print_bold('Phrase:', end=' '); print(mimiccxr_trainer.mscxr_phrases[phrase_idx])
        else:
            print_bold('Phrase idxs:', end=' '); print(phrase_idxs)
            print_bold('Positive fact idxs:', end=' '); print(pos_fact_idxs)
            print_bold('Strong negative fact idxs:', end=' '); print(strong_neg_fact_idxs)
            print_bold('Weak negative fact idxs:', end=' '); print(weak_neg_fact_idxs)

            print_bold('Phrases:')
            for i in phrase_idxs:
                print(mimiccxr_trainer.mscxr_phrases[i])

            print_bold('Positive facts:')
            for i in pos_fact_idxs:
                print(mimiccxr_trainer.mscxr_facts[i])

            print_bold('Strong negative facts:')
            for i in strong_neg_fact_idxs:
                print(mimiccxr_trainer.mscxr_facts[i])

            print_bold('Weak negative facts:')
            for i in weak_neg_fact_idxs[:20]:
                print(mimiccxr_trainer.mscxr_facts[i])


def inspect_vinbig_phrase_grounding_dataset(vinbig_trainer, i, train=True,
                                            image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)):
    if train:
        dataset = vinbig_trainer.train_dataset
        idx = dataset.indices[i]
        phrase_bboxes = dataset.phrase_bboxes[idx]
        phrase_idx = dataset.phrase_idxs[idx]
        image_path = dataset.image_paths[idx]
        dicom_id = os.path.basename(image_path).split('.')[0]
        instance = dataset[i]
        image_tensor = instance['i'] # (3, H, W)
        phrase_embedding = instance['pe'] # (128,)
        target_bbox_coords = instance['tbc'] # (num_regions, 4)
        target_bbox_presence = instance['tbp'] # (num_regions,)
        target_prob_mask = instance['tpm'] # (num_regions,)

        num_regions = target_bbox_coords.shape[-2]
        feat_W = feat_H = math.isqrt(num_regions)
        assert feat_W * feat_H == num_regions

        print_bold('i:', end=' '); print(i)
        print_bold('dicom_id:', end=' '); print(dicom_id)
        print_bold('image_path:', end=' '); print(image_path)
        print_bold('image_tensor:', end=' '); print(image_tensor.shape)
        print_bold(f'num_regions: {num_regions}, feat_W: {feat_W}, feat_H: {feat_H}')
        print_bold('phrase_embedding:', end=' '); print(phrase_embedding.shape)
        print_bold('target_bbox_coords:', end=' '); print(target_bbox_coords.shape)
        print_bold('target_bbox_presence:', end=' '); print(target_bbox_presence.shape)
        print_bold('target_prob_mask:', end=' '); print(target_prob_mask.shape)

        # Obtain bbox coordinates from target
        target_bbox_coords_ = target_bbox_coords.view(-1, 4)
        target_bbox_presence_ = target_bbox_presence.view(-1)
        bbox_coords = set()
        for i in range(len(target_bbox_presence_)):
            if target_bbox_presence_[i] == 1:
                bbox_coords.add(tuple(target_bbox_coords_[i].tolist()))
        bbox_coords = list(bbox_coords)
        bbox_format = vinbig_trainer.bbox_format
        print_bold('bbox_coords:', end=' '); print(bbox_coords)
        print_bold('bbox_format:', end=' '); print(bbox_format)

        # Recover image from tensor
        image_tensor.mul_(torch.tensor(image_std).view(3, 1, 1))
        image_tensor.add_(torch.tensor(image_mean).view(3, 1, 1))
        image_tensor = torch.clamp(image_tensor, 0, 1)
        img = Image.fromarray((image_tensor.permute(1,2,0) * 255).numpy().astype(np.uint8))
        W_img = img.width
        H_img = img.height

        # Display transformed image + bboxes
        print_bold('Transformed image:')
            
        # Start visualization.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Compute grid lines positions based on the image size and feature map size.
        cell_width = W_img / feat_W
        cell_height = H_img / feat_H
        # Grid lines (excluding image borders):
        vlines = [cell_width * i for i in range(1, feat_W)]
        hlines = [cell_height * i for i in range(1, feat_H)]

        # Plot 1: Original image with bounding boxes.
        axes[0].imshow(img)
        axes[0].set_title("Image with Bounding Boxes")
        for bbox_coord in bbox_coords:
            if bbox_format == 'xyxy':
                x1, y1, x2, y2 = bbox_coord
            elif bbox_format == 'cxcywh':
                cx, cy, w, h = bbox_coord
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
            else: raise ValueError(f'Invalid bbox_format: {bbox_format}')
            x1 *= img.width
            y1 *= img.height
            x2 *= img.width
            y2 *= img.height
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)
        for x in vlines:
            axes[0].axvline(x, color="white", linestyle="--", linewidth=1)
        for y in hlines:
            axes[0].axhline(y, color="white", linestyle="--", linewidth=1)

        # Plot 2: Upsampled target presence heatmap with grid.
        target_bbox_presence = target_bbox_presence.view(feat_H, feat_W)
        up_presence = np.array(
            Image.fromarray((target_bbox_presence.numpy() * 255).astype(np.uint8)).resize(
                (W_img, H_img), resample=Image.NEAREST
            )
        ) / 255.0
        axes[1].imshow(up_presence, cmap="viridis")
        axes[1].set_title("Target Presence Heatmap")
        for x in vlines:
            axes[1].axvline(x, color="white", linestyle="--", linewidth=1)
        for y in hlines:
            axes[1].axhline(y, color="white", linestyle="--", linewidth=1)

        # Plot 3: Upsampled prob mask with grid.
        target_prob_mask = target_prob_mask.view(feat_H, feat_W)
        up_prob_mask = cv2.resize(
            target_prob_mask.numpy(),
            (W_img, H_img), interpolation=cv2.INTER_NEAREST,
        )
        axes[2].imshow(up_prob_mask, cmap="viridis")
        axes[2].set_title("Target Prob Mask")
        for x in vlines:
            axes[2].axvline(x, color="white", linestyle="--", linewidth=1)
        for y in hlines:
            axes[2].axhline(y, color="white", linestyle="--", linewidth=1)

        plt.show()

        # Display original image
        print_bold('Original image:')
        img = Image.open(image_path).convert('RGB')
        plt.imshow(img)
        plt.show()

        print_bold('Phrase bboxes:', end=' '); print(phrase_bboxes)
        print_bold('Phrase idx:', end=' '); print(phrase_idx)
        print_bold('Phrase:', end=' '); print(vinbig_trainer.phrases[phrase_idx])


def inspect_padchest_phrase_grounding_dataset(
        padchestgr_trainer, i, train=True, image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)):
    import os
    from medvqa.utils.logging_utils import print_bold
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import math
    if train:
        dataset = padchestgr_trainer.train_dataset
        phrase = dataset.phrase_texts[i]
        phrase_bboxes = dataset.phrase_bboxes[i]
        image_path = dataset.image_paths[i]
        image_id = os.path.basename(image_path).split('.')[0]
        instance = dataset[i]
        image_tensor = instance['i'] # (3, H, W)
        phrase_embedding = instance['pe'] # (128,)
        target_bbox_coords = instance['tbc'] # (num_regions, 4)
        target_bbox_presence = instance['tbp'] # (num_regions,)
        target_prob_mask = instance['tpm'] # (num_regions,)

        num_regions = target_bbox_coords.shape[-2]
        feat_W = feat_H = math.isqrt(num_regions)
        assert feat_W * feat_H == num_regions

        print_bold('i:', end=' '); print(i)
        print_bold('image_id:', end=' '); print(image_id)
        print_bold('image_path:', end=' '); print(image_path)
        print_bold('image_tensor:', end=' '); print(image_tensor.shape)
        print_bold(f'num_regions: {num_regions}, feat_W: {feat_W}, feat_H: {feat_H}')
        print_bold('phrase_embedding:', end=' '); print(phrase_embedding.shape)
        print_bold('target_bbox_coords:', end=' '); print(target_bbox_coords.shape)
        print_bold('target_bbox_presence:', end=' '); print(target_bbox_presence.shape)
        print_bold('target_prob_mask:', end=' '); print(target_prob_mask.shape)

        print_bold('Phrase bboxes:', end=' '); print(phrase_bboxes)
        print_bold('Phrase:', end=' '); print(phrase)

        # Obtain bbox coordinates from target
        target_bbox_coords_ = target_bbox_coords.view(-1, 4)
        target_bbox_presence_ = target_bbox_presence.view(-1)
        bbox_coords = set()
        for i in range(len(target_bbox_presence_)):
            if target_bbox_presence_[i] == 1:
                bbox_coords.add(tuple(target_bbox_coords_[i].tolist()))
        bbox_coords = list(bbox_coords)
        bbox_format = dataset.bbox_format
        print_bold('bbox_coords:', end=' '); print(bbox_coords)
        print_bold('bbox_format:', end=' '); print(bbox_format)

        # Recover image from tensor
        image_tensor.mul_(torch.tensor(image_std).view(3, 1, 1))
        image_tensor.add_(torch.tensor(image_mean).view(3, 1, 1))
        image_tensor = torch.clamp(image_tensor, 0, 1)
        img = Image.fromarray((image_tensor.permute(1,2,0) * 255).numpy().astype(np.uint8))
        W_img = img.width
        H_img = img.height

        # Display transformed image + bboxes
        print_bold('Transformed image:')
            
        # Start visualization.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Compute grid lines positions based on the image size and feature map size.
        cell_width = W_img / feat_W
        cell_height = H_img / feat_H
        # Grid lines (excluding image borders):
        vlines = [cell_width * i for i in range(1, feat_W)]
        hlines = [cell_height * i for i in range(1, feat_H)]

        # Plot 1: Original image with bounding boxes.
        axes[0].imshow(img)
        axes[0].set_title("Image with Bounding Boxes")
        for bbox_coord in bbox_coords:
            if bbox_format == 'xyxy':
                x1, y1, x2, y2 = bbox_coord
            elif bbox_format == 'cxcywh':
                cx, cy, w, h = bbox_coord
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
            else: raise ValueError(f'Invalid bbox_format: {bbox_format}')
            x1 *= img.width
            y1 *= img.height
            x2 *= img.width
            y2 *= img.height
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)
        for x in vlines:
            axes[0].axvline(x, color="white", linestyle="--", linewidth=1)
        for y in hlines:
            axes[0].axhline(y, color="white", linestyle="--", linewidth=1)

        # Plot 2: Upsampled target presence heatmap with grid.
        target_bbox_presence = target_bbox_presence.view(feat_H, feat_W)
        up_presence = np.array(
            Image.fromarray((target_bbox_presence.numpy() * 255).astype(np.uint8)).resize(
                (W_img, H_img), resample=Image.NEAREST
            )
        ) / 255.0
        axes[1].imshow(up_presence, cmap="viridis")
        axes[1].set_title("Target Presence Heatmap")
        for x in vlines:
            axes[1].axvline(x, color="white", linestyle="--", linewidth=1)
        for y in hlines:
            axes[1].axhline(y, color="white", linestyle="--", linewidth=1)

        # Plot 3: Upsampled prob mask with grid.
        target_prob_mask = target_prob_mask.view(feat_H, feat_W)
        up_prob_mask = cv2.resize(
            target_prob_mask.numpy(),
            (W_img, H_img), interpolation=cv2.INTER_NEAREST,
        )
        axes[2].imshow(up_prob_mask, cmap="viridis")
        axes[2].set_title("Target Prob Mask")
        for x in vlines:
            axes[2].axvline(x, color="white", linestyle="--", linewidth=1)
        for y in hlines:
            axes[2].axhline(y, color="white", linestyle="--", linewidth=1)

        plt.show()

        # Display original image
        print_bold('Original image:')
        img = Image.open(image_path).convert('RGB')
        plt.imshow(img)
        plt.show()


def inspect_chest_imagenome_alg_item(
    item: dict,
    dataset,
    image_mean: tuple = (0.485, 0.456, 0.406),
    image_std: tuple = (0.229, 0.224, 0.225),
):
    """
    Visualizes a single item from a ChestImaGenome_AnatomicalLocationGroundingDataset.

    - If the item is for training, it plots a grid where each row corresponds to a
      grounded anatomical location and shows:
      1. The image with the specific bounding box(es).
      2. The upsampled target presence heatmap.
      3. The upsampled target probabilistic mask.
    - If the item is for validation/inference, it plots a single image with all
      ground-truth bounding boxes and their labels.

    Args:
        item: A dictionary returned by the dataset's __getitem__ method.
        dataset: The dataset instance, used to access metadata like bbox_format.
        image_mean: The mean used for image normalization.
        image_std: The standard deviation used for image normalization.
    """
    # --- 1. Common Setup: Recover Image from Tensor ---
    image_tensor = item["i"].clone()
    std = torch.tensor(image_std).view(3, 1, 1)
    mean = torch.tensor(image_mean).view(3, 1, 1)
    image_tensor.mul_(std).add_(mean)
    image_tensor = torch.clamp(image_tensor, 0, 1)
    img = Image.fromarray(
        (image_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
    )
    H_img, W_img = img.height, img.width

    # --- 2. Check if it's a training or validation item ---
    is_training_item = "tbc" in item

    if is_training_item:
        print_bold("Visualizing Training Item")
        # --- 3a. Logic for Training Items ---
        target_coords = item["tbc"]
        target_presence = item["tbp"]
        target_prob_masks = item["tpm"]
        gt_indices = item["gidxs"]

        bbox_format = dataset.bbox_format
        feat_H, feat_W = dataset.feature_map_size
        num_grounded_locs = len(gt_indices)

        print(f"Found {num_grounded_locs} grounded locations in this sample.")

        if num_grounded_locs == 0:
            print("No grounded locations to visualize. Displaying image only.")
            plt.imshow(img)
            plt.show()
            return

        fig, axes = plt.subplots(
            num_grounded_locs, 3, figsize=(18, 6 * num_grounded_locs)
        )
        if num_grounded_locs == 1:
            axes = axes.reshape(1, -1)

        cell_width = W_img / feat_W
        cell_height = H_img / feat_H
        vlines = [cell_width * i for i in range(1, feat_W)]
        hlines = [cell_height * i for i in range(1, feat_H)]

        for i, loc_idx in enumerate(gt_indices):
            loc_name = CHEST_IMAGENOME_BBOX_NAMES[loc_idx]
            coords_i = target_coords[i]
            presence_i = target_presence[i]
            prob_mask_i = target_prob_masks[i]

            # --- MODIFICATION 1: Skeptical BBox Extraction ---
            bbox_coords = set()
            for j in range(len(presence_i)):
                if presence_i[j] == 1:
                    bbox_coords.add(tuple(coords_i[j].tolist()))
            bbox_coords = list(bbox_coords)
            if len(bbox_coords) > 1:
                print(
                    f"\033[93mWARNING: Found {len(bbox_coords)} unique bboxes for location '{loc_name}'. This might be unexpected.\033[0m"
                )

            # --- Plot 1: Image with Bounding Box(es) ---
            ax1 = axes[i, 0]
            ax1.imshow(img)
            ax1.set_title(
                f"Location: {loc_name} (idx: {loc_idx})\nImage with BBox(es)"
            )
            for bbox_coord in bbox_coords:
                if bbox_format == "xyxy":
                    x1, y1, x2, y2 = bbox_coord
                elif bbox_format == "cxcywh":
                    cx, cy, w, h = bbox_coord
                    x1, y1 = cx - w / 2, cy - h / 2
                    x2, y2 = cx + w / 2, cy + h / 2
                else:
                    raise ValueError(f"Invalid bbox_format: {bbox_format}")

                x1, x2 = x1 * W_img, x2 * W_img
                y1, y2 = y1 * H_img, y2 * H_img
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                )
                ax1.add_patch(rect)

            # --- MODIFICATION 2: Upsampling Heatmaps ---
            # Plot 2: Upsampled Target Presence Heatmap
            ax2 = axes[i, 1]
            presence_map = presence_i.view(feat_H, feat_W).numpy()
            up_presence = (
                np.array(
                    Image.fromarray((presence_map * 255).astype(np.uint8)).resize(
                        (W_img, H_img), resample=Image.NEAREST
                    )
                )
                / 255.0
            )
            ax2.imshow(up_presence, cmap="viridis", vmin=0, vmax=1)
            ax2.set_title("Upsampled Target Presence")

            # Plot 3: Upsampled Probabilistic Mask
            ax3 = axes[i, 2]
            prob_map = prob_mask_i.view(feat_H, feat_W).numpy()
            up_prob_mask = np.array(
                Image.fromarray((prob_map * 255).astype(np.uint8)).resize(
                    (W_img, H_img), resample=Image.NEAREST
                )
            ) / 255.0
            ax3.imshow(up_prob_mask, cmap="magma", vmin=0, vmax=1)
            ax3.set_title("Upsampled Probabilistic Mask")

            # Add grid lines to all plots
            for ax in [ax1, ax2, ax3]:
                for x in vlines:
                    ax.axvline(x, color="white", linestyle="--", linewidth=0.8)
                for y in hlines:
                    ax.axhline(y, color="white", linestyle="--", linewidth=0.8)
                ax.axis("off")

        plt.tight_layout(pad=3.0)
        plt.show()

    else:
        print_bold("Visualizing Validation/Inference Item")
        # --- 3b. Logic for Validation/Inference Items (Unchanged) ---
        gt_bboxes = item["bboxes"]
        gt_indices = item["classes"]
        bbox_format = dataset.bbox_format

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        ax.set_title("Image with all Ground-Truth BBoxes")
        ax.axis("off")

        for bbox_coord, loc_idx in zip(gt_bboxes, gt_indices):
            loc_name = CHEST_IMAGENOME_BBOX_NAMES[loc_idx]

            if bbox_format == "xyxy":
                x1, y1, x2, y2 = bbox_coord
            elif bbox_format == "cxcywh":
                cx, cy, w, h = bbox_coord
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
            else:
                raise ValueError(f"Invalid bbox_format: {bbox_format}")

            x1, x2 = x1 * W_img, x2 * W_img
            y1, y2 = y1 * H_img, y2 * H_img

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                loc_name,
                color="black",
                fontsize=8,
                bbox=dict(facecolor="cyan", alpha=0.7, pad=1),
            )

        plt.show()


def inspect_chest_imagenome_pg_item(
    i: int,
    dataset,  # Type hint would be ChestImaGenome_PhraseGroundingDataset
    image_mean: tuple = (0.485, 0.456, 0.406),
    image_std: tuple = (0.229, 0.224, 0.225),
):
    """
    Visualizes a single item from a ChestImaGenome_PhraseGroundingDataset.

    It displays a 1x3 grid:
    1. The augmented image as seen by the model.
    2. The original, un-augmented image with color-coded bounding boxes for
       'original' and 'similar' groundings.
    3. The upsampled target presence map.

    Args:
        i: The index of the item to visualize.
        dataset: The dataset instance.
        image_mean: The mean used for image normalization.
        image_std: The standard deviation used for image normalization.
    """

    def _draw_bboxes(ax, bboxes, edgecolor, label, W_img, H_img):
        """Helper function to draw a list of bounding boxes on an axis."""
        for i, bbox_coord in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox_coord
            x1, y1 = x_min * W_img, y_min * H_img
            w, h = (x_max - x_min) * W_img, (y_max - y_min) * H_img

            # Add label only to the first box to avoid duplicate legend entries
            rect_label = label if i == 0 else None
            rect = patches.Rectangle(
                (x1, y1),
                w,
                h,
                linewidth=1.5,
                edgecolor=edgecolor,
                facecolor="none",
                alpha=0.9,
                label=rect_label,
            )
            ax.add_patch(rect)

    # --- 1. Data Retrieval ---
    item = dataset[i]
    image_path = dataset.image_paths[i]
    phrase_text = dataset.phrases[i]
    original_bboxes = dataset.phrase_original_bboxes[i]
    similar_bboxes = dataset.phrase_similar_bboxes[i]
    target_area_ratio = item["tar"]

    print_bold("Visualizing Phrase Grounding Item")
    print("-" * 40)
    print_bold("Index:", i)
    print_bold("Image Path:", image_path)
    print_bold("Phrase Text:", phrase_text)
    print_bold("Num Original BBoxes:", len(original_bboxes))
    print_bold("Num Similar BBoxes:", len(similar_bboxes))
    print_bold("Target Area Ratios:", target_area_ratio)

    print("-" * 40)

    # --- 2. Visualization Setup ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    feat_H, feat_W = dataset.feature_map_size

    # --- Plot 1: Original Image with BBoxes ---
    ax1 = axes[0]
    orig_img = Image.open(image_path).convert("RGB")
    H_img, W_img = orig_img.height, orig_img.width
    ax1.imshow(orig_img)
    ax1.set_title("2. Original Image with Ground-Truth BBoxes")
    _draw_bboxes(ax1, original_bboxes, "lime", "Original BBox", W_img, H_img)
    _draw_bboxes(ax1, similar_bboxes, "cyan", "Similar BBox", W_img, H_img)
    ax1.legend()

    # --- Plot 2: Augmented Image (from tensor) ---
    ax2 = axes[1]
    image_tensor = item["i"].clone()
    std = torch.tensor(image_std).view(3, 1, 1)
    mean = torch.tensor(image_mean).view(3, 1, 1)
    image_tensor.mul_(std).add_(mean)
    image_tensor = torch.clamp(image_tensor, 0, 1)
    aug_img = Image.fromarray(
        (image_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
    )
    H_aug_img, W_aug_img = aug_img.height, aug_img.width
    ax2.imshow(aug_img)
    ax2.set_title("1. Augmented Image (Input to Model)")

    # --- Plot 3: Upsampled Target Presence Map ---
    ax3 = axes[2]
    target_presence = item["tbp"]
    presence_map = target_presence.view(feat_H, feat_W).numpy()
    up_presence = (
        np.array(
            Image.fromarray((presence_map * 255).astype(np.uint8)).resize(
                (W_aug_img, H_aug_img), resample=Image.NEAREST
            )
        )
        / 255.0
    )
    ax3.imshow(up_presence, cmap="viridis", vmin=0, vmax=1)
    ax3.set_title("3. Target Presence Map (Input to Loss)")

    # --- Final Touches ---
    cell_width = W_aug_img / feat_W
    cell_height = H_aug_img / feat_H
    vlines = [cell_width * i for i in range(1, feat_W)]
    hlines = [cell_height * i for i in range(1, feat_H)]

    for ax in [ax2, ax3]:  # Add grid lines to augmented image and heatmap
        for x in vlines:
            ax.axvline(x, color="white", linestyle="--", linewidth=0.8)
        for y in hlines:
            ax.axhline(y, color="white", linestyle="--", linewidth=0.8)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


_shared_image_id_to_binary_labels = None
def _count_labels(idx):
        count = 0
        for binary_labels in _shared_image_id_to_binary_labels.values():
            if binary_labels[idx] == 1:
                count += 1
        return count, idx
def _count_labels_in_parallel(label_idxs, label_alias, image_id_to_binary_labels, num_workers=10):
    global _shared_image_id_to_binary_labels
    _shared_image_id_to_binary_labels = image_id_to_binary_labels
    print(f'Num {label_alias} labels:', len(label_idxs))
    print(f'Counting labels in parallel ({num_workers} workers)...')
    start = time()
    with Pool(num_workers) as p:
        label_counts = p.map(_count_labels, label_idxs)
    print(f'Done counting labels ({time() - start:.1f}s)')
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
        mean_iou /= len(presence)
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

_shared_dicom_ids = None
_shared_average_bbox_coords = None
_shared_bbox_dict = None
def _compute_iou_mean_std(j):
    ious = np.zeros(len(_shared_dicom_ids), dtype=np.float32)
    ref_bbox = _shared_average_bbox_coords[4*j:4*j+4]
    for i, dicom_id in enumerate(_shared_dicom_ids):
        bboxes = _shared_bbox_dict[dicom_id]
        coords = bboxes['coords']
        presence = bboxes['presence']
        if presence[j] == 1:
            ious[i] = compute_iou(ref_bbox, coords[4*j:4*j+4])
    return j, np.mean(ious), np.std(ious)    
def plot_chest_imagenome_iou_mean_std_per_bbox_class(average_bbox_coords, dicom_ids, figsize=(6, 9), num_workers=8):
    bbox_dict = load_chest_imagenome_silver_bboxes()
    global _shared_dicom_ids, _shared_average_bbox_coords, _shared_bbox_dict
    _shared_dicom_ids = dicom_ids
    _shared_average_bbox_coords = average_bbox_coords
    _shared_bbox_dict = bbox_dict
    task_args = [j for j in range(CHEST_IMAGENOME_NUM_BBOX_CLASSES)]
    start = time()
    print(f'Computing IoU mean and std per bounding box class in parallel with {num_workers} workers...')
    with Pool(num_workers) as p:
        results = p.map(_compute_iou_mean_std, task_args)
    mean_ious = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES, dtype=np.float32)
    std_ious = np.zeros(CHEST_IMAGENOME_NUM_BBOX_CLASSES, dtype=np.float32)
    for j, mean_iou, std_iou in results:
        mean_ious[j] = mean_iou
        std_ious[j] = std_iou
    print('Done in {:.2f} seconds'.format(time() - start))
    # Sort by mean IoU
    idxs = np.argsort(mean_ious)
    bbox_class_names = [f'{CHEST_IMAGENOME_BBOX_NAMES[i]} (mean IoU={mean_ious[i]:.3f}, std IoU={std_ious[i]:.3f})' for i in idxs]
    mean_ious = mean_ious[idxs]
    std_ious = std_ious[idxs]
    # Plot horizontal bar chart
    plt.figure(figsize=figsize)
    plt.title(f'Mean and std IoU per bounding box class (n={len(dicom_ids)}, mean IoU={np.mean(mean_ious):.3f}, std IoU={np.mean(std_ious):.3f})')
    plt.barh(bbox_class_names, mean_ious, xerr=std_ious)
    plt.xlabel('Mean IoU')
    plt.ylabel('Bounding box class')
    plt.show()

def plot_chest_imagenome_silver_vs_gold_iou_per_bbox_class():
    # The difference between silver and gold is that the silver bounding boxes are
    # automatically generated by a model, while the gold bounding boxes are manually
    # annotated by a radiologist.
    silver_bbox_dict = load_chest_imagenome_silver_bboxes()
    gold_bbox_dict = load_chest_imagenome_gold_bboxes()
    assert set(gold_bbox_dict.keys()).issubset(set(silver_bbox_dict.keys()))
    
    gold_idxs = [i for i in range(CHEST_IMAGENOME_NUM_BBOX_CLASSES) \
        if CHEST_IMAGENOME_BBOX_NAMES[i] in CHEST_IMAGENOME_GOLD_BBOX_NAMES]
    mean_ious = np.zeros(CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES, dtype=np.float32)
    counts = np.zeros(CHEST_IMAGENOME_NUM_GOLD_BBOX_CLASSES, dtype=np.int32)
    
    for dicom_id in gold_bbox_dict.keys():
        silver_bboxes = silver_bbox_dict[dicom_id]
        gold_bboxes = gold_bbox_dict[dicom_id]
        for i, idx in enumerate(gold_idxs):
            if gold_bboxes['presence'][idx] == 0:
                continue
            counts[i] += 1
            if silver_bboxes['presence'][idx] == 0:
                continue
            silver_bbox = silver_bboxes['coords'][4*idx:4*idx+4]
            gold_bbox = gold_bboxes['coords'][4*idx:4*idx+4]
            mean_ious[i] += compute_iou(silver_bbox, gold_bbox)
    mean_ious /= counts
    
    # Sort by mean IoU
    idxs = np.argsort(mean_ious)
    bbox_class_names = [f'{CHEST_IMAGENOME_BBOX_NAMES[i]} (mean IoU={mean_ious[i]:.3f}, count={counts[i]})' for i in idxs]
    mean_ious = mean_ious[idxs]
    
    # Plot horizontal bar chart
    plt.figure(figsize=(6, 9))
    plt.title(f'Mean IoU per bounding box class (n={len(gold_bbox_dict)}, mean IoU={np.mean(mean_ious):.3f})')    
    plt.barh(bbox_class_names, mean_ious, align='center')
    plt.xlabel('Mean IoU')
    plt.ylabel('Bounding box class')
    plt.show()

def print_number_of_contradictions_per_class(label_names, image_id_to_contradictions):
    num_contradictions_per_class = np.zeros(len(label_names), dtype=np.int32)
    for contradictions in image_id_to_contradictions.values():
        for i in range(len(contradictions)):
            if contradictions[i] == 1:
                num_contradictions_per_class[i] += 1
    idxs = np.argsort(num_contradictions_per_class)[::-1]
    for i in idxs:
        if num_contradictions_per_class[i] > 0:
            print(f'{label_names[i]}: {num_contradictions_per_class[i]}')

def collect_dicom_ids_with_contradictions(label_names, image_id_to_contradictions):
    dicom_ids = []
    for dicom_id, contradictions in image_id_to_contradictions.items():
        nc = np.sum(contradictions)
        if nc > 0:
            problematic_labels = [label_names[i] for i in range(len(contradictions)) if contradictions[i] == 1]
            dicom_ids.append((dicom_id, nc, problematic_labels))
    dicom_ids.sort(key=lambda x: x[1], reverse=True)
    return dicom_ids

_shared_image_id_to_binary_labels = None
_shared_image_id_to_mask = None

def _count_pos_neg_unmasked(label_idx):
    global _shared_image_id_to_binary_labels
    global _shared_image_id_to_mask
    num_positive_masked = 0
    num_negative_masked = 0
    num_unmasked = 0
    for image_id, binary_labels in _shared_image_id_to_binary_labels.items():
        mask = _shared_image_id_to_mask[image_id]
        if binary_labels[label_idx] == 1:
            assert mask[label_idx] == 1
            num_positive_masked += 1
        else:
            if mask[label_idx] == 1:
                num_negative_masked += 1
            else:
                num_unmasked += 1
    return num_positive_masked, num_negative_masked, num_unmasked

def plot_positive_negative_masked_distribution_per_label(label_names, image_id_to_binary_labels, image_id_to_mask, figsize=(10, 10), num_workers=6):
    """
    For each label, plot a single horizontal bar divided into three parts (each of a different color):
    - the left part shows the number of images with a positive label and mask == 1
    - the middle part shows the number of images with a negative label and mask == 1
    - the right part shows the number of images with mask == 0 (i.e. the rest)
    """
    global _shared_image_id_to_binary_labels
    global _shared_image_id_to_mask
    _shared_image_id_to_binary_labels = image_id_to_binary_labels
    _shared_image_id_to_mask = image_id_to_mask
    with Pool(num_workers) as pool:
        results = pool.map(_count_pos_neg_unmasked, range(len(label_names)))
    num_positive_masked = np.array([r[0] for r in results], dtype=np.int32)
    num_negative_masked = np.array([r[1] for r in results], dtype=np.int32)
    num_unmasked = np.array([r[2] for r in results], dtype=np.int32)
    idxs = np.argsort(num_unmasked)[::-1]
    label_names = [f'{label_names[i]} (p={num_positive_masked[i]}, n={num_negative_masked[i]}, u={num_unmasked[i]})' for i in idxs]
    num_positive_masked = num_positive_masked[idxs]
    num_negative_masked = num_negative_masked[idxs]
    num_unmasked = num_unmasked[idxs]
    plt.figure(figsize=figsize)
    plt.title('Number of images per label')
    plt.barh(label_names, num_positive_masked, color='green', label='Positive')
    plt.barh(label_names, num_negative_masked, left=num_positive_masked, color='red', label='Negative')
    plt.barh(label_names, num_unmasked, left=num_positive_masked+num_negative_masked, color='gray', label='Not annotated')
    plt.xlabel('Number of images')
    plt.ylabel('Label')
    plt.legend()
    plt.tight_layout()
    plt.show()