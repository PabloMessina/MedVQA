from torch.optim import Adam, AdamW, SGD
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def create_optimizer(name, params, lr):
    logger.info(f'create_optimizer(): name = {name}, lr = {lr}')
    if name == 'adam':
        optimizer = Adam(params, lr=lr)
    elif name == 'adamw':
        optimizer = AdamW(params, lr=lr)
    elif name == 'sgd':
        optimizer = SGD(params, lr=lr)
    else:
        assert False, f'Unknown optimizer {name}'
    return optimizer


class GradientAccumulator:
    def __init__(
        self,
        optimizer: Optimizer,
        scaler: GradScaler = None,
        num_accumulation_steps: int = 1,
        max_grad_norm: float = None,
    ):
        """
        Args:
            optimizer: The optimizer (e.g., AdamW).
            scaler: The GradScaler for AMP (optional). If None, standard precision is assumed.
            num_accumulation_steps: Number of steps to accumulate gradients before updating.
            max_grad_norm: Maximum norm for gradient clipping (optional).
        """
        logger.info(
            f'GradientAccumulator initialized: accumulation_steps={num_accumulation_steps}, '
            f'max_grad_norm={max_grad_norm}, use_amp={scaler is not None}'
        )
        self.optimizer = optimizer
        self.scaler = scaler
        self.num_accumulation_steps = num_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

    def step(self, batch_loss: Tensor, model: nn.Module):
        """
        Performs the backward pass and conditionally steps the optimizer.
        """
        if batch_loss is None:
            raise ValueError("batch_loss cannot be None")

        # 1. Normalize loss for gradient accumulation
        loss = batch_loss / self.num_accumulation_steps

        # 2. Backward Pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.step_count += 1

        # 3. Optimizer Step (only when accumulation is complete)
        if self.step_count % self.num_accumulation_steps == 0:
            if self.scaler is not None:
                # AMP Logic: Unscale BEFORE clipping
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard Logic
                if self.max_grad_norm is not None:
                    clip_grad_norm_(model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()

            # 4. Zero Gradients
            self.optimizer.zero_grad()
            return True # Indicates an optimizer step occurred
            
        return False # Indicates we are still accumulating