import numpy as np
import math

def _area_of_intersection(x1, y1, w1, h1, x2, y2, w2, h2):
    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0
    
def compute_mask_from_bounding_box(mask_height, mask_width, x1, y1, x2, y2, flatten=False):
    assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, f'Invalid bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}'
    mask = np.zeros((mask_height, mask_width), dtype=np.float32)
    cw = 1.0 / mask_width
    ch = 1.0 / mask_height
    ca = cw * ch
    w = x2 - x1
    h = y2 - y1
    x_min = math.floor(x1 / cw)
    y_min = math.floor(y1 / ch)
    x_max = math.ceil(x2 / cw)
    y_max = math.ceil(y2 / ch)
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if 0 <= x < mask_width and 0 <= y < mask_height:
                mask[y, x] = _area_of_intersection(x1, y1, w, h, x * cw, y * ch, cw, ch) / ca
    if flatten:
        mask = mask.reshape(-1)
    return mask

def compute_mask_from_bounding_boxes(mask_height, mask_width, bboxes, flatten=False):
    assert type(bboxes) == list
    mask = np.zeros((mask_height, mask_width), dtype=np.float32)
    cw = 1.0 / mask_width
    ch = 1.0 / mask_height
    ca = cw * ch
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        assert 0 <= x1 < x2 <= 1
        assert 0 <= y1 < y2 <= 1
        w = x2 - x1
        h = y2 - y1
        x_min = math.floor(x / cw)
        y_min = math.floor(y / ch)
        x_max = math.ceil(x / cw)
        y_max = math.ceil(y / ch)
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if 0 <= x < mask_width and 0 <= y < mask_height:
                    mask[y, x] = max(mask[y, x], _area_of_intersection(x1, y1, w, h, x_min * cw, y_min * ch, cw, ch) / ca)
    if flatten:
        mask = mask.reshape(-1)
    return mask