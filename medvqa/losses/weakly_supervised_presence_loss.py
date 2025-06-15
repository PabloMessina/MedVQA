import torch
import torch.nn as nn
import torch.nn.functional as F


class WeaklySupervisedPresenceLoss(nn.Module):
    """
    A custom loss for weakly supervised visual grounding confidence.

    This loss combines two distinct supervision strategies:
    1.  For cells where the target presence is 0 (background), it applies a
        standard Binary Cross-Entropy (BCE) loss.
    2.  For cells where the target presence is 1 (foreground), it applies a
        weak supervision loss based on area ratio. It computes the average
        predicted probability across all foreground cells and regresses this
        value towards a given `target_area_ratio` using Mean Squared Error (MSE).

    This approach allows for precise supervision of the background while using
    a statistical, region-level constraint for the foreground, which is useful
    when exact foreground masks are noisy or unavailable.
    """

    def __init__(self, area_loss_weight: float = 1.0):
        """
        Initializes the WeaklySupervisedPresenceLoss.

        Args:
            area_loss_weight: A float to scale the contribution of the
                              area-based weak supervision loss. Defaults to 1.0.
        """
        super().__init__()
        self.area_loss_weight = area_loss_weight
        # Use reduction='none' to get per-element loss for manual masking
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        visual_grounding_confidence_logits: torch.Tensor,
        visual_grounding_confidence_probs: torch.Tensor,
        target_presence: torch.Tensor,
        target_area_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the combined loss.

        Args:
            visual_grounding_confidence_logits: The raw model output logits.
                                                Shape: (B, H*W).
            visual_grounding_confidence_probs: Sigmoid of the logits.
                                               Shape: (B, H*W).
            target_presence: The ground truth binary mask indicating
                                  foreground (1) and background (0) cells.
                                  Shape: (B, H*W).
            target_area_ratio: The target average probability for the
                               foreground region for each batch item.
                               Shape: (B,).

        Returns:
            A single scalar tensor representing the total loss.
        """
        # Ensure target is float for BCEWithLogitsLoss
        target_presence = target_presence.float()

        # --- 1. BCE Loss for Negative Cells (target_presence == 0) ---
        per_element_bce_loss = self.bce_with_logits(
            visual_grounding_confidence_logits, target_presence
        )

        negative_mask = target_presence == 0
        negative_loss = per_element_bce_loss * negative_mask

        # Normalize by the number of negative cells to get a stable mean loss
        num_negative_cells = negative_mask.sum(dim=1)
        # Avoid division by zero if a sample has no negative cells
        safe_num_negative_cells = torch.clamp(num_negative_cells, min=1)
        bce_loss_per_sample = negative_loss.sum(dim=1) / safe_num_negative_cells
        bce_component_loss = bce_loss_per_sample.mean()

        # --- 2. Weak Supervision Loss for Positive Cells (target_presence == 1) ---
        positive_mask = target_presence == 1

        # Sum of probabilities in the predicted positive region
        predicted_area = (
            visual_grounding_confidence_probs * positive_mask
        ).sum(dim=1)

        # Total number of cells in the ground truth positive region
        true_area = positive_mask.sum(dim=1)

        # Mask for samples that actually have positive cells to supervise
        samples_with_positives = true_area > 0
        area_component_loss = torch.tensor(
            0.0, device=visual_grounding_confidence_logits.device
        )

        if samples_with_positives.any():
            # Calculate average probability only for valid samples
            valid_predicted_area = predicted_area[samples_with_positives]
            valid_true_area = true_area[samples_with_positives]
            valid_target_ratio = target_area_ratio[samples_with_positives]

            predicted_avg_prob = valid_predicted_area / valid_true_area

            # Use MSE to enforce the average probability to be close to the target ratio
            area_loss = F.mse_loss(predicted_avg_prob, valid_target_ratio)
            area_component_loss = area_loss

        # --- 3. Combine the two loss components ---
        total_loss = (
            bce_component_loss
            + self.area_loss_weight * area_component_loss
        )

        return total_loss