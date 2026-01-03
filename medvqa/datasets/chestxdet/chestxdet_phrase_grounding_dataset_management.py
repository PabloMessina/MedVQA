import os
import torch
from typing import Tuple, Callable, List, Dict
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from medvqa.utils.files_utils import load_json, load_pickle

def get_image_size(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # (width, height)

def normalize_bbox(bbox, img_w, img_h, bbox_format='xyxy'):
    x1, y1, x2, y2 = bbox
    x1, x2 = x1 / img_w, x2 / img_w
    y1, y2 = y1 / img_h, y2 / img_h
    if bbox_format == 'xyxy':
        return [x1, y1, x2, y2]
    elif bbox_format == 'cxcywh':
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return [cx, cy, w, h]
    else:
        raise ValueError(f"Unknown bbox_format: {bbox_format}")

def normalize_polygon(polygon, img_w, img_h):
    return [[x / img_w, y / img_h] for x, y in polygon]

def polygons_to_mask(norm_polygons: List[List[List[float]]], H: int, W: int):
    import cv2
    mask = np.zeros((H, W), dtype=np.uint8)
    for norm_polygon in norm_polygons:
        pts = np.array(norm_polygon, dtype=np.float32)
        pts_scaled = np.stack([pts[:, 0] * W, pts[:, 1] * H], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts_scaled.reshape((-1, 1, 2))], color=1)
    return mask

class ChestXDetInferenceDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        image_transform: Callable,
        label2embedding_path: str,
        mask_res: Tuple[int, int] = (16, 16),
        bbox_format: str = 'xyxy',
    ):
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.mask_res = mask_res
        self.bbox_format = bbox_format

        # Load label2embedding
        self.label2embedding = load_pickle(label2embedding_path)
        # Convert all embeddings to torch.Tensor for easier batching
        for k in self.label2embedding:
            emb = self.label2embedding[k]
            if not torch.is_tensor(emb):
                self.label2embedding[k] = torch.tensor(emb, dtype=torch.float32)

        data = load_json(json_path)
        # Group by (image, label)
        group_dict: Dict[Tuple[str, str], Dict] = {}
        for item in data:
            file_name = item['file_name']
            img_path = os.path.join(image_dir, file_name)
            img_w, img_h = get_image_size(img_path)
            for sym, bbox, polygon in zip(item['syms'], item['boxes'], item['polygons']):
                key = (img_path, sym)
                norm_bbox = normalize_bbox(bbox, img_w, img_h, bbox_format)
                norm_polygon = normalize_polygon(polygon, img_w, img_h)
                if key not in group_dict:
                    group_dict[key] = {
                        'image_path': img_path,
                        'label': sym,
                        'bboxes': [],
                        'polygons': []
                    }
                group_dict[key]['bboxes'].append(norm_bbox)
                group_dict[key]['polygons'].append(norm_polygon)
        # Precompute masks
        self.samples = []
        for v in group_dict.values():
            mask = polygons_to_mask(v['polygons'], *mask_res)
            self.samples.append({
                'image_path': v['image_path'],
                'label': v['label'],
                'bboxes': v['bboxes'],
                'polygons': v['polygons'],
                'mask': mask
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        pixel_values = self.image_transform(image_path)['pixel_values']
        label = sample['label']
        embedding = self.label2embedding[label]
        return {
            'image_path': image_path,
            'pixel_values': pixel_values,
            'bboxes': sample['bboxes'],      # list of normalized bboxes
            'polygons': sample['polygons'],  # list of normalized polygons
            'mask': sample['mask'],          # union mask
            'label': label,
            'embedding': embedding           # embedding for the label
        }
    
    def collate_fn(self, batch):
        image_paths = [item['image_path'] for item in batch]
        pixel_values = torch.stack([item['pixel_values'] for item in batch]) # (N, C, H, W)
        bboxes = [item['bboxes'] for item in batch]
        polygons = [item['polygons'] for item in batch]
        masks = [item['mask'] for item in batch]
        labels = [item['label'] for item in batch]
        embeddings = torch.stack([item['embedding'] for item in batch])
        return {
            'image_paths': image_paths,
            'pixel_values': pixel_values,
            'bboxes': bboxes,
            'polygons': polygons,
            'masks': masks,
            'labels': labels,
            'embeddings': embeddings
        }