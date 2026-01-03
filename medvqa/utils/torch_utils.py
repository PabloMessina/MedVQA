import re
import logging
from typing import Dict, List, Optional, Union
import torch.nn as nn


logger = logging.getLogger(__name__)


def set_inplace_flag(module, inplace_value):
    for submodule in module.modules():
        if hasattr(submodule, 'inplace'):
            submodule.inplace = inplace_value


def freeze_model_parts(
    model: nn.Module,
    freeze_config: Optional[Dict[str, Union[bool, List[str], str]]] = None
) -> int:
    """
    Freezes specified parts of a model based on configuration.

    Sets `requires_grad = False` for parameters matching the criteria.

    Args:
        model: The model to modify.
        freeze_config: A dictionary specifying what to freeze. Examples:
            {
                "freeze_all": False, # If True, freezes the entire model initially
                "unfreeze_patterns": ["^classifier\\.", "^projection\\."], # Regex patterns to unfreeze
                "freeze_patterns": ["^vision_encoder\\."], # Regex patterns to freeze (overrides unfreeze)
                "unfreeze_modules": ["encoder.layer.0"], # Specific module names to unfreeze
                "freeze_modules": ["encoder.layer.11"], # Specific module names to freeze
                # Add more specific flags like "freeze_vision_encoder": True if needed
            }

    Returns:
        The number of parameters that remain trainable.
    """
    if not freeze_config:
        logger.info("No freeze configuration provided. All parameters are trainable by default.")
        # Ensure all params are trainable initially if no config
        for param in model.parameters():
            param.requires_grad = True
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = 0

    # Initial state based on freeze_all
    initial_freeze_state = freeze_config.get("freeze_all", False)
    for name, param in model.named_parameters():
        param.requires_grad = not initial_freeze_state
        total_params += param.numel()

    logger.info(f"Initial freeze state (freeze_all={initial_freeze_state}): All params require_grad = {not initial_freeze_state}")

    # Unfreeze based on patterns
    unfreeze_patterns = freeze_config.get("unfreeze_patterns", [])
    if initial_freeze_state and unfreeze_patterns: # Only unfreeze if initially frozen
        compiled_unfreeze_patterns = [re.compile(p) for p in unfreeze_patterns]
        for name, param in model.named_parameters():
            if any(pattern.match(name) for pattern in compiled_unfreeze_patterns):
                if not param.requires_grad:
                    logger.debug(f"Unfreezing parameter: {name}")
                    param.requires_grad = True

    # Freeze based on patterns (takes precedence over unfreeze)
    freeze_patterns = freeze_config.get("freeze_patterns", [])
    if freeze_patterns:
        compiled_freeze_patterns = [re.compile(p) for p in freeze_patterns]
        for name, param in model.named_parameters():
             if any(pattern.match(name) for pattern in compiled_freeze_patterns):
                if param.requires_grad:
                    logger.debug(f"Freezing parameter: {name}")
                    param.requires_grad = False

    # Unfreeze specific modules by name
    unfreeze_modules = freeze_config.get("unfreeze_modules", [])
    if unfreeze_modules:
        for module_name_to_unfreeze in unfreeze_modules:
            try:
                module = dict(model.named_modules())[module_name_to_unfreeze]
                for name, param in module.named_parameters():
                    full_name = f"{module_name_to_unfreeze}.{name}"
                    if not param.requires_grad:
                        logger.debug(f"Unfreezing parameter within module '{module_name_to_unfreeze}': {full_name}")
                        param.requires_grad = True
            except KeyError:
                logger.warning(f"Module name '{module_name_to_unfreeze}' not found for unfreezing.")

    # Freeze specific modules by name
    freeze_modules = freeze_config.get("freeze_modules", [])
    if freeze_modules:
        for module_name_to_freeze in freeze_modules:
            try:
                module = dict(model.named_modules())[module_name_to_freeze]
                for name, param in module.named_parameters():
                    full_name = f"{module_name_to_freeze}.{name}"
                    if param.requires_grad:
                        logger.debug(f"Freezing parameter within module '{module_name_to_freeze}': {full_name}")
                        param.requires_grad = False
            except KeyError:
                logger.warning(f"Module name '{module_name_to_freeze}' not found for freezing.")

    # Log final state and count trainable parameters
    frozen_params_list = []
    trainable_params_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_count += param.numel()
        else:
            frozen_params_list.append(name)

    logger.info(f"Parameter freezing applied. {trainable_params_count}/{total_params} parameters are trainable.")
    if frozen_params_list:
        logger.info(f"Frozen parameters/prefixes: {', '.join(list(set(p.split('.')[0] for p in frozen_params_list)))}...") # Log top-level modules frozen

    return trainable_params_count


def _check_module_trainability_recursive(module: nn.Module, base_name: str = ""):
    """
    Recursive helper function to check and log fully frozen or fully trainable modules.

    Args:
        module: The current nn.Module being inspected.
        base_name: The full name path to this module from the root model.
    """
    has_parameters = False
    all_frozen = True
    all_trainable = True

    # Iterate through all parameters *within* this module and its descendants
    # module.parameters(recurse=True) gets all params in module + submodules
    for param in module.parameters(recurse=True):
        has_parameters = True
        if param.requires_grad:
            all_frozen = False # Found a trainable param, so not fully frozen
        else:
            all_trainable = False # Found a frozen param, so not fully trainable

        # Optimization: If we've already determined it's neither fully frozen nor fully trainable,
        # we can potentially break early, though the loop is still needed to set has_parameters
        # correctly based on *any* parameter existence.

    # --- Decision Logic ---

    # If the module has parameters:
    if has_parameters:
        # 1. Check if the module's subtree is fully frozen
        if all_frozen:
            logger.info(f"Module fully frozen ❄️: {base_name if base_name else 'model (root)'}")
            # Interrupt recursion for this branch as it's fully characterized
            return

        # 2. Check if the module's subtree is fully trainable
        if all_trainable:
            logger.info(f"Module fully trainable 💪: {base_name if base_name else 'model (root)'}")
            # Interrupt recursion for this branch as it's fully characterized
            return

    # If the module has parameters and is neither fully frozen nor fully trainable,
    # OR if the module has no parameters, recursively explore its immediate children.
    for child_name, child_module in module.named_children():
        # Construct the full name for the child module
        full_child_name = f"{base_name}.{child_name}" if base_name else child_name
        # Recursively call the function for the child
        _check_module_trainability_recursive(child_module, full_child_name)


def log_module_trainability(model: nn.Module):
    """
    Recursively traverses a PyTorch model and logs the names of modules
    that are either fully frozen or fully trainable.

    A module is considered "fully frozen" if it contains parameters, and all
    parameters within that module AND all of its submodules have
    `requires_grad=False`.

    A module is considered "fully trainable" if it contains parameters, and all
    parameters within that module AND all of its submodules have
    `requires_grad=True`.

    If a module's subtree is neither fully frozen nor fully trainable (i.e.,
    it contains a mix of trainable and frozen parameters), this function will
    recursively check its immediate children.

    Modules with no parameters (like activation functions) are effectively
    skipped in the output, but their children (if any) are still traversed.

    Args:
        model: The PyTorch model (nn.Module) to inspect.
    """
    logger.info("--- Checking for fully frozen and fully trainable modules ---")
    # Start recursion with the root model and an empty base name
    _check_module_trainability_recursive(model, "")
    logger.info("--- Finished checking ---")