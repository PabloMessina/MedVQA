import torch
import torch.nn as nn
    
class WeightedByClassBCELoss(nn.Module):
    """
    Computes Binary Cross Entropy loss with logits, applying balancing weights
    between positive and negative samples independently for each class.

    Assumes input tensors (`output`, `target`) have shape (batch_size, num_classes).

    For each class `j`:
    - Calculates the number of positive samples (`num_pos_j`) and negative
      samples (`num_neg_j`) in the batch.
    - Computes a weight for positive samples: `pos_weight_j = num_neg_j / num_pos_j`
    - Computes a weight for negative samples: `neg_weight_j = num_pos_j / num_neg_j`
    - These weights are clamped between 1.0 and `batch_size` to limit
      extreme values.
    - The element-wise BCE loss is then multiplied by `pos_weight_j` if the
      target is 1, or `neg_weight_j` if the target is 0.

    An optional `classes_mask` can be provided to compute the loss only over a
    subset of classes. The final loss is the mean of the weighted element-wise
    losses over all active elements (considering the mask if provided).
    """
    def __init__(self, classes_mask: torch.Tensor = None):
        """
        Initializes the WeightedByClassBCELoss module.

        Args:
            classes_mask (torch.Tensor, optional): A 1D tensor of shape
                (num_classes,) containing 0s and 1s (or booleans).
                If provided, the loss is calculated only for classes where the
                mask is 1. Defaults to None (all classes are used).
        """
        super().__init__()
        # Use reduction='none' to get per-element loss values
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.classes_mask = classes_mask
        self.n_classes = None
        if self.classes_mask is not None:
            # Ensure mask is on the same device potentially later used by inputs
            # (though it might be moved explicitly in the training loop)
            # Also ensure it's boolean or float for calculations
            self.classes_mask = self.classes_mask.bool() # Use bool for masking
            self.n_classes = self.classes_mask.sum().item() # Number of active classes

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted and balanced BCE loss per class.

        Args:
            output (torch.Tensor): Raw model outputs (logits).
                Shape: (batch_size, num_classes).
            target (torch.Tensor): Binary target labels (0 or 1).
                Shape: (batch_size, num_classes).

        Returns:
            torch.Tensor: A scalar tensor containing the final computed loss.
        """
        # Input validation
        assert output.ndim == 2 and target.ndim == 2, \
            f"Expected 2D tensors, got output.ndim={output.ndim}, target.ndim={target.ndim}"
        assert output.shape == target.shape, \
            f"Output shape {output.shape} must match target shape {target.shape}"

        batch_size = target.size(0)

        # Ensure target is float for calculations
        target = target.float()
        not_target = 1.0 - target

        # Calculate positive/negative counts per class (shape: [num_classes])
        num_pos_per_class = target.sum(dim=0)
        num_neg_per_class = not_target.sum(dim=0) # or batch_size - num_pos_per_class

        # Calculate balancing weights per class (shape: [num_classes])
        # Clamp denominator to avoid division by zero if a class has 0 pos/neg samples
        pos_weight = num_neg_per_class / num_pos_per_class.clamp(min=1.0)
        neg_weight = num_pos_per_class / num_neg_per_class.clamp(min=1.0)

        # Clamp the weights themselves to limit the maximum balancing effect
        # Lower bound 1 means no sample is down-weighted below its original loss
        # Upper bound batch_size limits up-weighting for classes with few samples
        pos_weight.clamp_(min=1.0, max=batch_size)
        neg_weight.clamp_(min=1.0, max=batch_size)

        # Compute element-wise BCE loss (shape: [batch_size, num_classes])
        loss_elem = self.bce(output, target)

        # Apply per-class weights (weights broadcast across batch dim)
        # Shape: [batch_size, num_classes]
        weighted_loss = (target * pos_weight + not_target * neg_weight) * loss_elem

        # Apply mask and aggregate loss
        if self.classes_mask is not None:
            # Ensure mask is on the correct device
            mask = self.classes_mask.to(weighted_loss.device)
            # Apply mask element-wise (zeros out inactive classes)
            masked_loss = weighted_loss * mask
            # Average loss over active elements only
            # Avoid division by zero if n_classes or batch_size is 0
            if self.n_classes > 0 and batch_size > 0:
                loss = masked_loss.sum() / (self.n_classes * batch_size)
            else:
                loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        else:
            # No mask: average loss over all elements
            if batch_size > 0:
                loss = weighted_loss.mean()
            else:
                loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)

        return loss
    

class WeightedBCELoss(nn.Module):
    """
    Weighted BCE Loss
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target, weights):
        loss = self.bce(output, target)
        weighted_loss = loss * weights
        return weighted_loss.sum() / weights.sum() # mean weighted loss
    
  
class NegativePositiveBalancedBCELoss(nn.Module):
    """
    Computes Binary Cross Entropy loss with logits, balancing the loss
    contribution from positive and negative samples by averaging their
    respective mean losses.

    Specifically, it calculates:
    1. The mean BCE loss for all positive samples.
    2. The mean BCE loss for all negative samples.
    3. The final loss is the average of these two means: (mean_pos_loss + mean_neg_loss) / 2.

    This ensures that the average loss value for the positive class and the
    average loss value for the negative class contribute equally to the final
    loss, regardless of the number of samples in each class.

    If only positive or only negative samples are present, the loss is simply
    the mean loss of the present class. If no samples are present, the loss is 0.
    """
    def __init__(self):
        """
        Initializes the NegativePositiveBalancedBCELoss module.
        """
        super().__init__()
        # Use reduction='none' to get per-element loss values
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the balanced BCE loss.

        Args:
            output: The raw model outputs (logits). Tensor of arbitrary shape.
            target: The corresponding binary targets (0 or 1). Tensor of the
                    same shape as output.

        Returns:
            A scalar tensor containing the computed balanced loss. Returns 0.0
            if both num_pos and num_neg are zero.
        """
        # Ensure target is float and flatten tensors for easier processing.
        target = target.float()
        output = output.view(-1)
        target = target.view(-1)

        # Calculate per-element BCE loss.
        loss_elem = self.bce(output, target)

        # Identify positive and negative samples.
        is_pos = target == 1
        is_neg = target == 0

        # Count positive and negative samples.
        num_pos = is_pos.sum()
        num_neg = is_neg.sum()

        # Calculate loss based on the presence of positive and negative samples.
        if num_pos > 0 and num_neg > 0:
            # Both positive and negative samples exist.
            # Calculate mean loss for positives.
            pos_loss_mean = loss_elem[is_pos].mean()
            # Calculate mean loss for negatives.
            neg_loss_mean = loss_elem[is_neg].mean()
            # Final loss is the average of the two mean losses.
            loss = (pos_loss_mean + neg_loss_mean) / 2.0
        elif num_pos > 0:
            # Only positive samples exist.
            # Loss is the mean loss of positive samples.
            loss = loss_elem[is_pos].mean()
        elif num_neg > 0:
            # Only negative samples exist.
            # Loss is the mean loss of negative samples.
            loss = loss_elem[is_neg].mean()
        else:
            # No positive or negative samples found (e.g., empty input).
            # Return 0.0 to avoid errors in training loops.
            # Ensure the tensor is on the same device and dtype as output.
            loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)

        return loss
    

class WeightedNegativePositiveBalancedBCELoss(nn.Module):
    """
    Computes Binary Cross Entropy loss with logits, incorporating per-sample
    weights and balancing the influence of positive and negative groups.

    This loss calculates the weighted average BCE loss separately for positive
    and negative samples, using the provided per-sample `weights`. The final
    loss is then computed as the simple average of these two weighted means:
    `(weighted_mean_pos_loss + weighted_mean_neg_loss) / 2`.

    This ensures that the weighted average loss for the positive class and the
    weighted average loss for the negative class contribute equally to the final
    loss, regardless of the number of samples or the total weight in each class.

    If only positive or only negative samples are present, the loss is simply
    the weighted mean loss of the present class. If no samples are present,
    the loss is 0.0.
    """
    def __init__(self, epsilon: float = 1e-8):
        """
        Initializes the WeightedNegativePositiveBalancedBCELoss module.

        Args:
            epsilon: A small value added to the denominator during weight
                     normalization to prevent division by zero. Defaults to 1e-8.
        """
        super().__init__()
        # Use reduction='none' to get per-element loss values
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.epsilon = epsilon

    def forward(self, output: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted and balanced BCE loss.

        Args:
            output: Raw model outputs (logits). Tensor of arbitrary shape.
            target: Binary targets (0 or 1). Tensor of the same shape as output.
            weights: Per-sample importance weights. Tensor of the same shape
                     as output. Weights should be non-negative.

        Returns:
            A scalar tensor containing the computed weighted and balanced loss.
            Returns 0.0 if no positive or negative samples exist.
        """
        # Ensure correct types and flatten tensors for easier processing.
        target = target.float()
        weights = weights.float() # Ensure weights are float
        output = output.view(-1)
        target = target.view(-1)
        weights = weights.view(-1)
        weights = weights.clamp(min=self.epsilon) # Ensure weights are non-zero for stability

        # Compute BCE loss per element.
        loss_elem = self.bce(output, target)

        # Identify positive and negative samples.
        is_pos = target == 1
        is_neg = target == 0

        # Count positive and negative samples.
        num_pos = is_pos.sum()
        num_neg = is_neg.sum()

        # --- Calculate Weighted Average Loss for Each Group ---
        pos_loss_mean = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        neg_loss_mean = torch.tensor(0.0, device=output.device, dtype=output.dtype)

        if num_pos > 0:
            # Select weights and losses for positive samples
            pos_weights_raw = weights[is_pos]
            pos_losses = loss_elem[is_pos]
            # Calculate sum of weights for normalization, clamp for stability
            sum_pos_weights = pos_weights_raw.sum().clamp(min=self.epsilon)
            # Calculate weighted average loss for positives
            pos_loss_mean = (pos_losses * pos_weights_raw).sum() / sum_pos_weights

        if num_neg > 0:
            # Select weights and losses for negative samples
            neg_weights_raw = weights[is_neg]
            neg_losses = loss_elem[is_neg]
            # Calculate sum of weights for normalization, clamp for stability
            sum_neg_weights = neg_weights_raw.sum().clamp(min=self.epsilon)
            # Calculate weighted average loss for negatives
            neg_loss_mean = (neg_losses * neg_weights_raw).sum() / sum_neg_weights

        # --- Combine Group Losses ---
        if num_pos > 0 and num_neg > 0:
            # Both classes present: average the weighted means
            loss = (pos_loss_mean + neg_loss_mean) / 2.0
        elif num_pos > 0:
            # Only positives present: loss is the weighted mean of positives
            loss = pos_loss_mean
        elif num_neg > 0:
            # Only negatives present: loss is the weighted mean of negatives
            loss = neg_loss_mean
        else:
            # No samples present: loss is 0
            loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)

        return loss