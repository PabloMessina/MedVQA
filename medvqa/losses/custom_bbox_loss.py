import torch

def compute_bbox_loss(pred_bbox_logits, gt_bbox_coords, gt_bbox_presence):
    # pred_bbox_logits: (batch_size, K, num_regions, 4)
    # gt_bbox_coords: (batch_size, K, num_regions, 4)
    # gt_bbox_presence: (batch_size, K, num_regions)
    assert pred_bbox_logits.shape == gt_bbox_coords.shape
    assert pred_bbox_logits.shape[:-1] == gt_bbox_presence.shape
    assert pred_bbox_logits.shape[-1] == 4  
    assert pred_bbox_logits.ndim == 4
    assert gt_bbox_presence.ndim == 3

    total_presence = torch.sum(gt_bbox_presence)
    if total_presence == 0:
        return 0 # no bbox present in any image

    # L1 loss for bbox coordinates
    bbox_loss = torch.nn.functional.l1_loss(pred_bbox_logits, gt_bbox_coords, reduction='none')
    bbox_loss = torch.sum(bbox_loss, dim=-1) # (batch_size, K, num_regions)
    bbox_loss = bbox_loss * gt_bbox_presence # (batch_size, K, num_regions)
    bbox_loss = torch.sum(bbox_loss) / total_presence
    return bbox_loss