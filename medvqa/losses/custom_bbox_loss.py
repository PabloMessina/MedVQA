import torch

def compute_bbox_loss(pred_bbox_logits, gt_bbox_coords, gt_bbox_presence, beta=1.0):
    # pred_bbox_logits: (batch_size, K, num_regions, 4) or (batch_size, num_regions, 4)
    # gt_bbox_coords: (batch_size, K, num_regions, 4) or (batch_size, num_regions, 4)
    # gt_bbox_presence: (batch_size, K, num_regions) or (batch_size, num_regions)
    assert pred_bbox_logits.shape == gt_bbox_coords.shape
    assert pred_bbox_logits.shape[:-1] == gt_bbox_presence.shape
    assert pred_bbox_logits.shape[-1] == 4

    total_presence = torch.sum(gt_bbox_presence)
    if total_presence == 0:
        return torch.tensor(0.0, device=pred_bbox_logits.device)  # No bbox present in any image

    # Smooth L1 loss for bbox coordinates
    bbox_loss = torch.nn.functional.smooth_l1_loss(pred_bbox_logits, gt_bbox_coords, beta=beta, reduction='none')
    bbox_loss = torch.sum(bbox_loss, dim=-1)  # (batch_size, K, num_regions) or (batch_size, num_regions)
    bbox_loss = bbox_loss * gt_bbox_presence  # Mask loss where no bbox is present
    bbox_loss = torch.sum(bbox_loss) / total_presence  # Normalize by total presence count

    return bbox_loss
