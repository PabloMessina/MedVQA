import torch
import torch.nn.functional as F


def compute_bbox_loss(pred_bbox_logits: torch.Tensor, gt_bbox_coords: torch.Tensor,
                      weights: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Compute the bounding box regression loss using Smooth L1 loss, weighted by
    per-region weights. The loss is normalized by the total weight sum.

    Args:
        pred_bbox_logits (Tensor): Predicted bounding box coordinates (logits)
            with shape (batch_size, K, num_regions, 4) or (batch_size, num_regions, 4).
        gt_bbox_coords (Tensor): Ground truth bounding box coordinates, having
            the same shape as pred_bbox_logits.
        weights (Tensor): A tensor indicating the weight of each region's loss
            contribution, with shape (batch_size, K, num_regions) or
            (batch_size, num_regions). These could be probabilistic masks or
            other importance scores.
        beta (float): The beta parameter for Smooth L1 loss (default is 1.0).

    Returns:
        Tensor: A scalar tensor representing the normalized, weighted Smooth L1
        loss for the bounding boxes.

    Notes:
        - If the total sum of weights is zero (or very close to it), the
          function returns a loss of zero.
        - The loss for each predicted bounding box is computed element-wise,
          then summed over the 4 coordinate dimensions.
        - The weighting ensures that the contribution of each region's loss
          is scaled by its corresponding weight before normalization.
    """
    # Verify the consistency of shapes:
    # 1. pred_bbox_logits and gt_bbox_coords must have identical shapes.
    # 2. The shape of weights must match that of pred_bbox_logits, except the last dimension.
    # 3. The last dimension of pred_bbox_logits (and gt_bbox_coords) must be 4 (e.g., xyxy or cxcywh format).
    assert pred_bbox_logits.shape == gt_bbox_coords.shape
    assert pred_bbox_logits.shape[:-1] == weights.shape
    assert pred_bbox_logits.shape[-1] == 4

    # Calculate the sum of all weights across all regions and batches.
    total_weight = torch.sum(weights) # Scalar tensor
    if total_weight < 1e-9: # Use a small epsilon for float comparison
        # If total weight is effectively zero, return zero loss.
        return torch.tensor(0.0, device=pred_bbox_logits.device, dtype=pred_bbox_logits.dtype)
    # Clamp to avoid potential division by very small numbers if not exactly zero
    total_weight = total_weight.clamp(min=1e-6)

    # Compute Smooth L1 loss element-wise for each coordinate dimension.
    # The 'none' reduction preserves the loss per coordinate for manual aggregation.
    bbox_loss = F.smooth_l1_loss(pred_bbox_logits, gt_bbox_coords, beta=beta, reduction='none')

    # Sum the loss over the 4 coordinate dimensions to obtain the loss per region.
    bbox_loss = torch.sum(bbox_loss, dim=-1)  # Resulting shape: (batch_size, K, num_regions) or (batch_size, num_regions)

    # Apply the weights to scale the loss for each region.
    bbox_loss = bbox_loss * weights  # Shape: (batch_size, K, num_regions) or (batch_size, num_regions)

    # Sum the weighted losses and normalize by the total weight.
    bbox_loss = torch.sum(bbox_loss) / total_weight

    return bbox_loss