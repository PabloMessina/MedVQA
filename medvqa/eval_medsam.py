import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.patches as patches
from tqdm import tqdm
from PIL import Image
from transformers import SamModel, SamProcessor
from medvqa.datasets.vinbig import visualize_image_with_bounding_boxes
from medvqa.utils.common import RESULTS_DIR, get_timestamp, parsed_args_to_dict
from medvqa.utils.files_utils import save_pickle
from medvqa.utils.logging_utils import print_magenta, print_orange

class EvalDatasets:
    VINBIG_TEST_SET = 'vinbig_test_set'
    @staticmethod
    def get_choices():
        return [
            EvalDatasets.VINBIG_TEST_SET,
        ]

MEDSAM_MASK_SIZE = (256, 256)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dataset', type=str, default=EvalDatasets.VINBIG_TEST_SET,
                        choices=EvalDatasets.get_choices(), help='Evaluation dataset')
    parser.add_argument('--huggingface_model_name', type=str, default='wanglab/medsam-vit-base', help='Huggingface model name')
    parser.add_argument('--num_bounding_boxes', type=int, default=4, help='Number of bounding boxes to evaluate')
    return parser.parse_args()

def _show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def _show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
        _show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def _visualize_segmentation(image, masks, colors=None, alpha=0.4):
    """
    Visualize segmentation masks overlaid on a PIL image.

    Parameters:
        image (PIL.Image.Image): The original image.
        masks (numpy.ndarray): Binary numpy array of shape (N, H, W) containing segmentation masks.
        colors (list of tuples): List of RGB tuples for mask colors. Defaults to random colors.
        alpha (float): Transparency level for the masks. Defaults to 0.5.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input 'image' must be a PIL.Image.Image.")
    
    if masks.ndim != 3:
        raise ValueError("Input 'masks' must have shape (N, H, W).")
    
    # Convert PIL image to RGB mode if not already
    image = image.convert("RGB")
    image_width, image_height = image.size
    
    # Resize masks to the image's dimensions
    resized_masks = np.array([
        np.array(Image.fromarray(mask).resize((image_width, image_height), Image.NEAREST))
        for mask in masks
    ])
    print(resized_masks.shape)
    
    
    # Debug: Check if resized masks contain non-zero values
    if not np.any(resized_masks):
        raise ValueError("All resized masks are empty (contain only zeros). Check your input masks.")
    
    # Generate random colors if not provided
    if colors is None:
        np.random.seed(42)  # For reproducibility
        colors = [tuple(np.random.randint(0, 256, 3)) for _ in range(masks.shape[0])]
    
    # Ensure the number of colors matches the number of masks
    assert len(colors) >= masks.shape[0], "Not enough colors for all masks."
    
    # Create an overlay layer with all masks
    overlay = np.zeros((image_height, image_width, 3), dtype=np.float32)
    for mask, color in zip(resized_masks, colors):
        for i in range(3):  # Apply the color to each channel
            overlay[..., i] += mask * color[i]
    
    # Normalize overlay to prevent over-saturation (values > 255)
    overlay = np.clip(overlay, 0, 255)
    
    # Blend the overlay with the original image
    original = np.array(image, dtype=np.float32)
    blended = (original * (1 - alpha) + overlay * alpha).astype(np.uint8)
    
    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
#     plt.axis("off")
    plt.show()

import numpy as np

def convert_boxes_to_mask(boxes, mask_size):
    """
    Convert bounding boxes to a binary mask.
    
    Args:
        boxes (list of tuple): List of bounding boxes, each defined as (xmin, ymin, xmax, ymax) 
                               with normalized coordinates [0, 1].
        mask_size (tuple): The size of the binary mask (height, width).
        
    Returns:
        np.ndarray: A binary mask of the bounding boxes.
    """
    # Initialize an empty mask
    mask = np.zeros(mask_size, dtype=np.uint8)
    
    # Denormalize and create the bounding box mask
    height, width = mask_size
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
        mask[y1:y2, x1:x2] = 1
    
    return mask

def compute_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    Args:
        mask1 (np.ndarray): The first binary mask.
        mask2 (np.ndarray): The second binary mask.

    Returns:
        float: The IoU score between the two masks.
    """

    assert mask1.shape == mask2.shape, "The two masks must have the same shape."
    assert mask1.dtype == np.uint8 and mask2.dtype == np.uint8, "The masks must be binary (0 or 1)."

    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Compute IoU
    iou = intersection / union if union > 0 else 0.0
    return iou

def run_medsam_and_visualize_predictions_on_example(
    image_path, input_boxes,
    boxes_are_normalized=True,
    huggingface_model_name="wanglab/medsam-vit-base",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SamProcessor.from_pretrained(huggingface_model_name)
    model = SamModel.from_pretrained(huggingface_model_name).to(device)
    model.eval()

    image = Image.open(image_path)
    image = image.convert('RGB').convert()
    image = image.resize((1024, 1024))
    w = image.width
    h = image.height
    if boxes_are_normalized:
        input_boxes = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in input_boxes] # denormalize

    # prepare image + box prompt for the model
    inputs = processor(image, input_boxes=[input_boxes], return_tensors="pt").to(device)
    for k,v in inputs.items():
        print(k,v.shape)

    # forward pass
    # note that the authors use `multimask_output=False` when performing inference
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False, output_hidden_states=True)

    print(f'outputs.pred_masks.shape: {outputs.pred_masks.shape}')

    # apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    print(f'medsam_seg_prob.shape: {medsam_seg_prob.shape}')
    
    # convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    medsam_seg = medsam_seg.squeeze(0)
    print(f'medsam_seg.shape: {medsam_seg.shape}')

    _show_boxes_on_image(image, input_boxes)

    _visualize_segmentation(image, medsam_seg)


def overlay_mask(ax, mask, alpha=0.5, cmap='jet'):
    """
    Overlay a segmentation mask on an image with the background fully transparent.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
        mask (numpy.ndarray): Segmentation mask.
        alpha (float): Transparency level for the mask foreground.
        cmap (str): Colormap for the mask.
    """
    # Create an RGBA version of the colormap
    colormap = plt.colormaps[cmap]
    rgba_mask = colormap(mask / mask.max())  # Normalize mask to [0, 1]
    rgba_mask[..., -1] = (mask > 0) * alpha  # Set alpha only for foreground pixels

    # Overlay the mask
    ax.imshow(rgba_mask)

def visualize_bboxes_and_segmentations(data, vinbig_image_id_2_bboxes=None):
    """
    Visualize the image with original bounding boxes and overlay the first input bbox and segmentation mask.
    
    Args:
        data (dict): A dictionary containing the image path, bounding boxes, segmentation masks, and other info.
    """
    # Load the image
    image = Image.open(data['image_path'])
    image = image.convert('RGB')
    image_array = np.array(image)
    
    # Extract relevant information
    gt_bboxes = data['gt_bboxes']
    normalized_input_boxes = data['normalized_input_boxes']
    segmentation_masks = data['medsam_seg']  # Assume it's a 3D array with masks along the first dimension

    # Function to denormalize bounding boxes
    def denormalize_bbox(bbox, image_width, image_height):
        xmin, ymin, xmax, ymax = bbox
        return [
            xmin * image_width,
            ymin * image_height,
            (xmax - xmin) * image_width,
            (ymax - ymin) * image_height,
        ]

    # Visualize original image with ground truth bounding boxes
    if vinbig_image_id_2_bboxes is not None:
        image_id = os.path.basename(data['image_path']).split('.')[0]
        visualize_image_with_bounding_boxes(
            image_id=image_id,
            bbox_dict=vinbig_image_id_2_bboxes[image_id],
            verbose=False,
            denormalize=True,
            figsize=(8, 8)
        )
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_array)
    for bbox in gt_bboxes:
        x, y, width, height = denormalize_bbox(bbox, image.width, image.height)
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    ax.set_title("Original Image with Ground Truth Bounding Boxes")
    plt.show()

    # Visualize input bounding boxes and segmentation masks one by one
    for i, bbox in enumerate(normalized_input_boxes):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image_array)

        # Draw the input bounding box
        x, y, width, height = denormalize_bbox(bbox, image.width, image.height)
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        
        # Overlay the segmentation mask
        if i < segmentation_masks.shape[0]:  # Ensure there's a corresponding mask
            mask = segmentation_masks[i]
            resized_mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
            overlay_mask(ax, resized_mask, cmap='jet', alpha=0.5)  # Overlay mask with transparency

        ax.set_title(f"Image with Input BBox {i+1} and Segmentation {i+1}")
        plt.show()

def eval_medsam(
    eval_dataset,
    huggingface_model_name="wanglab/medsam-vit-base",
    num_bounding_boxes=4,
):
    assert  num_bounding_boxes >= 2, f'num_bounding_boxes must be >= 2, got: {num_bounding_boxes}'

    if eval_dataset == EvalDatasets.VINBIG_TEST_SET:
        # vinbigdata test set
        from medvqa.datasets.vinbig import load_test_image_id_2_bboxes, get_original_image_path
        test_image_id_2_bboxes = load_test_image_id_2_bboxes(normalize=True)
        image_ids = list(test_image_id_2_bboxes.keys())
        image_paths = [get_original_image_path(image_id) for image_id in image_ids]
        bboxes_list = []
        for image_id in image_ids:
            bboxes = test_image_id_2_bboxes[image_id]
            flattened_bboxes = [bbox for bboxes in bboxes.values() for bbox in bboxes]
            assert all(0 <= x <= 1 for bbox in flattened_bboxes for x in bbox), f'Invalid bbox: {flattened_bboxes}'
            assert all(len(bbox) == 4 for bbox in flattened_bboxes), f'Invalid bbox: {flattened_bboxes}'
            assert len(flattened_bboxes) > 0, f'No bboxes found for image_id: {image_id}'
            bboxes_list.append(flattened_bboxes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = SamProcessor.from_pretrained(huggingface_model_name)
        model = SamModel.from_pretrained(huggingface_model_name).to(device)
        model.eval()

        results = []

        for image_path, gt_bboxes in tqdm(zip(image_paths, bboxes_list), total=len(image_paths), mininterval=5):

            image = Image.open(image_path)
            image = image.convert('RGB').convert()
            image = image.resize((1024, 1024), Image.BICUBIC)
            w = image.width
            h = image.height

            # Generate bounding boxes containing the ground-truth annotations
            min_x = min([bbox[0] for bbox in gt_bboxes])
            min_y = min([bbox[1] for bbox in gt_bboxes])
            max_x = max([bbox[2] for bbox in gt_bboxes])
            max_y = max([bbox[3] for bbox in gt_bboxes])
            normalized_input_boxes = []
            denormalized_input_boxes = []
            for i in range(num_bounding_boxes):
                x0 = (min_x / (num_bounding_boxes-1)) * i
                y0 = (min_y / (num_bounding_boxes-1)) * i
                x1 = 1.0 - ((1.0 - max_x) / (num_bounding_boxes-1)) * i
                y1 = 1.0 - ((1.0 - max_y) / (num_bounding_boxes-1)) * i
                assert (0 <= x0 <= 1 and 0 <= y0 <= 1 and 0 <= x1 <= 1 and 0 <= y1 <= 1
                        and x0 < x1 and y0 < y1), f'Invalid bounding box: {x0, y0, x1, y1}'
                normalized_input_boxes.append([x0, y0, x1, y1])
                denormalized_input_boxes.append([x0*w, y0*h, x1*w, y1*h])

            # Prepare image + box prompt for the model
            inputs = processor(image, input_boxes=[denormalized_input_boxes], return_tensors="pt").to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False, output_hidden_states=True)

            # Apply sigmoid
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))

            # Convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze() # (num_bounding_boxes, 256, 256)
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            assert medsam_seg.shape == (num_bounding_boxes, *MEDSAM_MASK_SIZE), f'Invalid medsam_seg shape: {medsam_seg.shape}'

            # Convert gt bbox to mask
            gt_mask = convert_boxes_to_mask(gt_bboxes, MEDSAM_MASK_SIZE)

            # Compute IoU with respect gt_mask
            ious_with_gt = [compute_iou(gt_mask, medsam_seg[i]) for i in range(num_bounding_boxes)]
            mean_iou_with_gt = np.mean(ious_with_gt)

            # Compute IoU between pairs of medsam_seg masks
            ious_between_pairs = []
            for i in range(num_bounding_boxes):
                for j in range(i+1, num_bounding_boxes):
                    iou = compute_iou(medsam_seg[i], medsam_seg[j])
                    ious_between_pairs.append(iou)
            mean_iou_between_pairs = np.mean(ious_between_pairs)

            # Collect results
            results.append({
                'image_path': image_path,
                'gt_bboxes': gt_bboxes,
                'normalized_input_boxes': normalized_input_boxes,
                'denormalized_input_boxes': denormalized_input_boxes,
                'medsam_seg': medsam_seg,
                'mean_iou_with_gt': mean_iou_with_gt,
                'mean_iou_between_pairs': mean_iou_between_pairs,
            })

        # Print results
        mean_iou_with_gt = np.mean([r['mean_iou_with_gt'] for r in results])
        mean_iou_between_pairs = np.mean([r['mean_iou_between_pairs'] for r in results])
        print_magenta(f'mean_iou_with_gt: {mean_iou_with_gt}', bold=True)
        print_magenta(f'mean_iou_between_pairs: {mean_iou_between_pairs}', bold=True)

        # Save results
        clean_model_name = huggingface_model_name.replace('/', '_')
        timestamp = get_timestamp()
        save_path = os.path.join(RESULTS_DIR, clean_model_name,
                                 f'{eval_dataset}_num_bboxes={num_bounding_boxes}_{timestamp}_segmentation_results.pkl')
        save_pickle(results, save_path)
        print_orange(f'Saved results to: {save_path}', bold=True)
        
    else:
        raise ValueError(f'Invalid eval_dataset: {eval_dataset}')

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    eval_medsam(**args)