import random
from shapely.geometry import box as shapely_box
import itertools
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import List, Tuple, Union, Optional

from medvqa.utils.logging_utils import rgba_to_ansi

# Define type aliases for clarity
BoundingBox = Union[Tuple[float, float, float, float], List[float]]
FeatureMapSize = Tuple[int, int]


def cxcywh_to_xyxy(bbox: BoundingBox) -> Tuple[float, float, float, float]:
    """
    Convert a bounding box from (center_x, center_y, width, height) format 
    to (x_min, y_min, x_max, y_max) format.

    Args:
        bbox: Tuple or list of 4 numbers (cx, cy, w, h), typically normalized 
              coordinates.

    Returns:
        A tuple (x_min, y_min, x_max, y_max).
    """
    cx, cy, w, h = bbox
    x_min = cx - w / 2.0
    y_min = cy - h / 2.0
    x_max = cx + w / 2.0
    y_max = cy + h / 2.0
    return x_min, y_min, x_max, y_max

def xyxy_to_cxcywh(bbox: BoundingBox) -> Tuple[float, float, float, float]:
    """
    Convert a bounding box from (x_min, y_min, x_max, y_max) format to 
    (center_x, center_y, width, height) format.

    Args:
        bbox: Tuple or list of 4 numbers (x_min, y_min, x_max, y_max), typically 
              normalized coordinates.

    Returns:
        A tuple (cx, cy, w, h).
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def _bbox_coords_to_grid_cell_indices(
    x_min: float, 
    y_min: float, 
    x_max: float, 
    y_max: float, 
    w: int, 
    h: int
) -> Tuple[int, int, int, int]:
    """
    Converts normalized bounding box coordinates to discrete grid cell indices 
    given a feature map of size (h, w).

    Calculates the range of grid cells (inclusive start, exclusive end) that 
    overlap with the bounding box.

    Args:
        x_min: Normalized minimum x-coordinate (range [0, 1]).
        y_min: Normalized minimum y-coordinate (range [0, 1]).
        x_max: Normalized maximum x-coordinate (range [0, 1]).
        y_max: Normalized maximum y-coordinate (range [0, 1]).
        w: Width (number of columns) of the feature map grid.
        h: Height (number of rows) of the feature map grid.
        
    Returns:
        A tuple of indices (x_min_idx, y_min_idx, x_max_idx, y_max_idx) 
        representing the grid cell boundaries suitable for slicing, e.g., 
        `grid[y_min_idx:y_max_idx, x_min_idx:x_max_idx]`.
    """
    # Scale normalized coordinates by the number of cells (w, h)
    # Use floor for min and ceil for max to capture all overlapping cells
    x_min_idx = math.floor(x_min * w)
    y_min_idx = math.floor(y_min * h)
    x_max_idx = math.ceil(x_max * w)
    y_max_idx = math.ceil(y_max * h)

    # Ensure the slice range covers at least one cell if the bbox falls 
    # entirely within a single cell's boundaries after floor/ceil.
    if x_min_idx == x_max_idx: 
        x_max_idx += 1
    if y_min_idx == y_max_idx: 
        y_max_idx += 1
    
    # Clamp indices to valid range for slicing: [0, w] or [0, h].
    # Max index can be w or h because Python slicing is exclusive of the end.
    x_min_idx = max(0, min(w, x_min_idx))
    x_max_idx = max(0, min(w, x_max_idx))
    y_min_idx = max(0, min(h, y_min_idx))
    y_max_idx = max(0, min(h, y_max_idx))
    
    return (x_min_idx, y_min_idx, x_max_idx, y_max_idx)


# --- Helper for Single BBox Mask Calculation ---
def _calculate_single_bbox_probabilistic_mask(
    bbox: List[float], # [x_min, y_min, x_max, y_max] in normalized coordinates (0, 1)
    grid_shape: Tuple[int, int], # Target grid dimensions (Hg, Wg)
) -> np.ndarray:
    """
    Calculates a probabilistic mask for a single bounding box on a grid.

    The mask assigns higher probabilities to grid cells closer to the center
    of the bounding box.

    Args:
        bbox: A list containing [x_min, y_min, x_max, y_max] in normalized
              coordinates (ranging from 0 to 1).
        grid_shape: A tuple representing the target mask resolution (Hg, Wg).

    Returns:
        A numpy array of shape (Hg, Wg) representing the probabilistic mask,
        with values typically between 0.4 and 1.0 within the box region,
        and 0.0 outside.
    """ 

    Hg, Wg = grid_shape
    prob_mask = np.zeros((Hg, Wg), dtype=np.float32)

    # --- 1. Map normalized bbox coordinates to grid coordinates ---
    # Note: (0,0) is top-left corner. x corresponds to Wg, y to Hg.
    x_min_g, y_min_g = bbox[0] * Wg, bbox[1] * Hg
    x_max_g, y_max_g = bbox[2] * Wg, bbox[3] * Hg

    # --- 2. Handle zero-sized or invalid boxes gracefully ---
    if x_max_g <= x_min_g or y_max_g <= y_min_g:
        return prob_mask # Return an empty (all zeros) mask

    # --- 3. Calculate Bbox center and dimensions in grid coordinates ---
    center_x_g = (x_min_g + x_max_g) / 2.0
    center_y_g = (y_min_g + y_max_g) / 2.0
    bbox_w_g = x_max_g - x_min_g
    bbox_h_g = y_max_g - y_min_g

    # --- 4. Define inner rectangle scale (for high confidence area) ---
    inner_scale = 0.70 # Central 70% of the bbox width/height

    # --- 5. Create coordinate grid for vectorized calculations ---
    # `indexing='ij'` ensures yy corresponds to rows (Hg) and xx to columns (Wg)
    yy, xx = np.meshgrid(np.arange(Hg), np.arange(Wg), indexing='ij')
    # Shift coordinates to represent the center of each grid cell
    yy = yy + 0.5
    xx = xx + 0.5

    # --- 6. Calculate distances and define regions ---
    # Absolute distances of each grid cell center from the bbox center
    dist_x = np.abs(xx - center_x_g)
    dist_y = np.abs(yy - center_y_g)

    # Define the outer bounding box region on the grid
    # Add ~half-pixel buffer (0.95 / 2.0) to include cells partially overlapping the box edge
    bbox_region_mask = (
        (dist_x <= (bbox_w_g + 0.95) / 2.0) & \
        (dist_y <= (bbox_h_g + 0.95) / 2.0)
    )

    # Check if the bbox is too narrow to apply decreasing probability.
    # If the bbox is too narrow, return a uniform mask
    if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < 0.05:
        prob_mask[bbox_region_mask] = 1.0
        return prob_mask

    # Define the inner, high-confidence region (scaled bbox)
    # Also includes the half-pixel buffer for partial overlap
    inner_mask = (
        (dist_x <= (bbox_w_g * inner_scale + 0.95) / 2.0) & \
        (dist_y <= (bbox_h_g * inner_scale + 0.95) / 2.0)
    )

    # Define the interpolation region (between inner and outer boundaries)
    interp_mask = bbox_region_mask & ~inner_mask

    # --- 7. Assign probabilities based on regions ---
    # Assign maximum probability (1.0) to the inner region
    prob_mask[inner_mask] = 1.0

    # Calculate probabilities for the interpolation region
    if np.any(interp_mask): # Avoid division by zero if bbox_w_g or bbox_h_g is zero (already handled, but safe)
        # Calculate normalized distance (max of x/y relative to half-width/height)
        # This is like a Chebyshev distance scaled by the bbox dimensions
        norm_dist = np.maximum(
            dist_x[interp_mask] / (bbox_w_g / 2.0 + 1e-6), # Add epsilon for stability
            dist_y[interp_mask] / (bbox_h_g / 2.0 + 1e-6)
        )

        # Interpolate linearly from 1.0 (at norm_dist=inner_scale) down to 0.4 (at norm_dist=1.0)
        # Formula: P = P_outer + (1 - norm_dist) / (1 - inner_scale) * (P_inner - P_outer)
        # Here: P_inner = 1.0, P_outer = 0.4
        interp_values = 0.4 + (1.0 - norm_dist) / (1.0 - inner_scale + 1e-6) * 0.6
        # Clip values to ensure they stay within the [0.4, 1.0] range, handling potential float inaccuracies
        prob_mask[interp_mask] = np.clip(interp_values, 0.4, 1.0)

    return prob_mask

def calculate_probabilistic_mask_from_bboxes(
    bboxes: List[List[float]],
    mask_resolution: Tuple[int, int],
    bbox_format: str = "xyxy",
    image_resolution: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Precomputes one probabilistic segmentation mask for a list of bounding boxes.

    Args:
        bboxes: List of bounding boxes. Each bbox is a list or tuple of 4 floats,
                typically normalized coordinates in the range [0, 1].
        mask_resolution: Tuple (H, W) representing the target mask resolution.
        bbox_format: The format of the input `bboxes`. Must be either 'xyxy'
                        (x_min, y_min, x_max, y_max) or 'cxcywh'
                        (center_x, center_y, width, height). Defaults to "xyxy".
        image_resolution: Optional tuple (H, W) representing the original image's
                            resolution. If provided, bboxes will be normalized with
                            respect to this resolution. If None, bboxes are assumed
                            to be already normalized to [0, 1] range.

    Returns:
        A numpy array of shape (H, W) representing the probabilistic mask.
    """
    H, W = mask_resolution

    # Ensure bboxes are in the correct format
    if bbox_format == "cxcywh":
        bboxes = [cxcywh_to_xyxy(bbox) for bbox in bboxes]
    elif bbox_format != "xyxy":
        raise ValueError("Unsupported bbox_format: use 'xyxy' or 'cxcywh'.")

    # If image resolution is provided, normalize bboxes
    if image_resolution is not None:
        image_H, image_W = image_resolution
        # Normalize bboxes to [0, 1] range
        bboxes = [
            [bbox[0] / image_W, bbox[1] / image_H, bbox[2] / image_W, bbox[3] / image_H]
            for bbox in bboxes
        ]
    assert all(0 <= coord <= 1 for bbox in bboxes for coord in bbox), \
        "Bbox coordinates must be in [0, 1] range."

    # Initialize mask with zeros
    mask = np.zeros((H, W), dtype=np.float32)

    # Iterate through each bbox
    for bbox in bboxes:
        # Calculate the mask contribution from this single bbox
        single_bbox_mask = _calculate_single_bbox_probabilistic_mask(bbox=bbox, grid_shape=mask_resolution)
        # Aggregate using maximum value
        mask = np.maximum(mask, single_bbox_mask)
        
    return mask


def convert_bboxes_into_target_tensors(
    bboxes: List[BoundingBox],
    probabilistic_mask: np.ndarray,
    feature_map_size: FeatureMapSize,
    bbox_format: str = "xyxy",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates target tensors for grid-based bounding box prediction models.

    Creates two tensors based on input bounding boxes and a segmentation mask:
    1. `target_coords`: A tensor where each grid cell overlapping with a 
       bounding box is assigned that bounding box's coordinates (in the 
       original `bbox_format`).
    2. `target_presence`: A boolean tensor indicating whether a grid cell 
       contains any object, based on the `probabilistic_mask`.
    3. `target_prob_mask`: A tensor representing the probabilistic mask,
         converted to a PyTorch tensor.

    Overlap Handling:
    - Initially, all grid cells covered by a bounding box `B` are assigned the 
      coordinates of `B`.
    - If two bounding boxes `B1` and `B2` overlap, the grid cells within their 
      *intersection* are reassigned the coordinates of the bounding box of the 
      *union* of `B1` and `B2`. This serves as a heuristic to handle ambiguous
      regions where multiple objects overlap.

    Note: Requires the `shapely` library for geometric operations (intersection, 
    union).

    Args:
        bboxes: A list of bounding boxes. Each bbox is a list or tuple of 4 
                floats, typically normalized coordinates in the range [0, 1].
                The format is specified by `bbox_format`.
        probabilistic_mask: A NumPy array of shape (H, W) 
                                         containing object presence probabilities 
                                         (e.g., from a segmentation model) for 
                                         each grid cell.
        feature_map_size: A tuple (H, W) representing the height and width of 
                          the target feature map grid. Must match the shape of 
                          `probabilistic_mask`.
        bbox_format: The format of the input `bboxes`. Must be either 'xyxy' 
                     (x_min, y_min, x_max, y_max) or 'cxcywh' 
                     (center_x, center_y, width, height). Defaults to "xyxy".
                     The output `target_coords` tensor will store coordinates 
                     in this same format.
        eps: A small threshold value. Grid cells in the 
             `probabilistic_mask` with values greater than `eps` 
             are marked as present in the `target_presence` tensor. 
             Defaults to 1e-6.

    Returns:
        A tuple containing:
        - target_coords (torch.Tensor): Tensor of shape (H, W, 4) where each 
          `target_coords[y, x, :]` contains the coordinates of the assigned 
          bounding box for grid cell (y, x), in the original `bbox_format`. 
          Cells not covered by any box (or overwritten by background if mask 
          is used differently) will contain zeros.
        - target_presence (torch.Tensor): Binary tensor of shape (H, W) 
          indicating object presence in each grid cell based on the thresholded 
          segmentation mask. `1.0` if `probabilistic_mask[y, x] > eps`.
        - target_prob_mask (torch.Tensor): A tensor of shape (H, W)
          representing the probabilistic mask, converted to a PyTorch tensor.

    Raises:
        ValueError: If `bbox_format` is not 'xyxy' or 'cxcywh'.
        AssertionError: If `probabilistic_mask.shape` does not 
                        match `feature_map_size`.
    """
    H, W = feature_map_size  # feature map dimensions (height, width)
    assert probabilistic_mask.shape == (H, W), \
        f"Mask shape {probabilistic_mask.shape} != feature map size {(H, W)}"

    # Initialize target tensors
    # target_coords stores the bbox assigned to each grid cell
    target_coords = torch.zeros(H, W, 4) 
    # target_presence indicates if a cell corresponds to an object based on seg mask
    target_presence = torch.from_numpy(probabilistic_mask > eps).float() # Convert to float tensor

    # Store original bboxes to assign them later in the original format
    original_bboxes = bboxes 
    processed_bboxes_xyxy = [] # Store bboxes in xyxy format for geometry processing

    # Ensure bboxes are in 'xyxy' format for internal calculations
    if bbox_format == "cxcywh":
        processed_bboxes_xyxy = [cxcywh_to_xyxy(bbox) for bbox in bboxes]
    elif bbox_format == "xyxy":
        processed_bboxes_xyxy = list(bboxes) # Keep a mutable copy
    else:
        raise ValueError("Unsupported bbox_format: use 'xyxy' or 'cxcywh'.")

    # --- Initial Assignment ---
    # Assign each original bounding box to the grid cells it covers
    for i, bbox_xyxy in enumerate(processed_bboxes_xyxy):
        x_min, y_min, x_max, y_max = bbox_xyxy
        # Find the grid cell indices covered by this bounding box
        x_min_idx, y_min_idx, x_max_idx, y_max_idx = _bbox_coords_to_grid_cell_indices(
            x_min, y_min, x_max, y_max, W, H
        )
        # Assign the *original* bbox coordinates to these cells
        target_coords[y_min_idx:y_max_idx, x_min_idx:x_max_idx] = torch.tensor(
            original_bboxes[i]
        )

    # --- Overlap Handling ---
    # If there are multiple boxes, handle overlaps using Shapely
    if len(bboxes) > 1:
        # Convert xyxy bboxes to Shapely box objects for geometric operations
        shapely_bboxes = [shapely_box(*bbox_xyxy) for bbox_xyxy in processed_bboxes_xyxy]
        
        # Iterate through all unique pairs of bounding boxes
        for bbox1, bbox2 in itertools.combinations(shapely_bboxes, 2):
            # Calculate the intersection area
            intersection = bbox1.intersection(bbox2)
            
            # If the boxes intersect significantly (not just touching)
            if not intersection.is_empty and intersection.area > 1e-8: # Use area threshold
                # Get the coordinates of the intersection area
                inter_x_min, inter_y_min, inter_x_max, inter_y_max = intersection.bounds
                # Find grid cell indices corresponding to the intersection
                inter_x_min_idx, inter_y_min_idx, inter_x_max_idx, inter_y_max_idx = \
                    _bbox_coords_to_grid_cell_indices(
                        inter_x_min, inter_y_min, inter_x_max, inter_y_max, W, H
                    )
                
                # Calculate the union of the two original boxes
                union = bbox1.union(bbox2)
                # Get the bounding box coordinates of the union area (always xyxy)
                union_bbox_xyxy = union.bounds
                
                # Prepare the union bbox coordinates in the required output format
                target_union_bbox = union_bbox_xyxy
                if bbox_format == "cxcywh":
                    target_union_bbox = xyxy_to_cxcywh(union_bbox_xyxy)
                                    
                # Assign the union bounding box to the grid cells in the intersection area
                target_coords[
                    inter_y_min_idx:inter_y_max_idx, inter_x_min_idx:inter_x_max_idx
                ] = torch.tensor(target_union_bbox)

    target_prob_mask = torch.from_numpy(probabilistic_mask)
    
    # Set the presence of cells with no assigned bbox to False
    no_coords_mask = target_coords.sum(dim=-1) == 0
    target_presence[no_coords_mask] = 0.0
    target_prob_mask[no_coords_mask] = 0.0

    # Flatten spatial dimensions for easier processing
    target_coords = target_coords.view(-1, 4) # (Hg * Wg, 4)
    target_presence = target_presence.view(-1) # (Hg * Wg,)
    target_prob_mask = target_prob_mask.view(-1) # (Hg * Wg,)

    return target_coords, target_presence, target_prob_mask


def convert_bboxes_into_presence_map(
    bboxes: List[BoundingBox], feature_map_size: Tuple[int, int]
) -> np.ndarray:
    """
    Generates a binary presence map based on a list of bounding boxes.

    This function creates a 2D numpy array where grid cells are marked as `1.0` if they
    fall within any of the provided bounding boxes and `0.0` otherwise. 

    Args:
        bboxes: A list of bounding boxes in 'xyxy' format (x_min, y_min,
                x_max, y_max), with coordinates normalized to [0, 1].
        feature_map_size: A tuple (H, W) representing the height and width of
                          the target feature map grid.

    Returns:
        np.ndarray: A 2D numpy array of shape (H, W) representing the presence map.
    """
    H, W = feature_map_size
    presence_map = np.zeros((H, W), dtype=np.float32)

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min_idx, y_min_idx, x_max_idx, y_max_idx = _bbox_coords_to_grid_cell_indices(
            x_min, y_min, x_max, y_max, W, H
        )
        presence_map[y_min_idx:y_max_idx, x_min_idx:x_max_idx] = 1.0

    return presence_map


def visualize_bbox_and_prob_mask_targets_tensors(
    target_coords: torch.Tensor,
    target_prob_mask: torch.Tensor,
    bbox_format: str,
    background_image: Optional[np.ndarray] = None,
    original_bboxes: Optional[List[BoundingBox]] = None,
    eps: float = 1e-6,
    title: str = "Target Visualization",
    verbose: bool = False,
):
    """
    Visualizes target grid tensors overlaid on an optional background image.

    Handles cases where the target grid (Hg, Wg) has different dimensions
    than the background image (Hi, Wi). Displays the target_prob_mask grid
    using a colormap, visually scaled to cover the image area. Overlays unique
    bounding boxes from target_coords (normalized to image dimensions), each
    with a distinct color. For each grid cell with probability > eps, draws a
    point at the center of the corresponding image region, colored according
    to the bounding box assigned to that cell. Optionally displays original
    input bounding boxes. The origin (0,0) is at the top-left corner.

    Args:
        target_coords: Target coordinates tensor (Hg, Wg, 4). Coordinates are
                       expected to be normalized (0, 1) relative to the full
                       image dimensions, with (0,0) at the top-left.
        target_prob_mask: Target probability mask tensor (Hg, Wg) with values
                          typically between 0 and 1.
        bbox_format: The format ('xyxy' or 'cxcywh') used in target_coords
                     and original_bboxes. Assumes (0,0) is top-left.
        background_image: Optional background image as a NumPy array (Hi, Wi, 3)
                          in RGB format (values 0-255 or 0-1). If provided,
                          its dimensions define the pixel coordinate space.
        original_bboxes: Optional list of the original input bounding boxes
                         (List[List[float]]) in normalized coordinates (0,0 top-left)
                         relative to the full image dimensions, and in the
                         specified bbox_format.
        eps: Threshold for target_prob_mask to consider a cell active for
             drawing its center point. Also used as a threshold to consider
             bounding box coordinates in target_coords as non-zero.
        title: Title for the plot.
        verbose: If True, prints additional information about the drawing
    """
    # --- Input Validation and Setup ---
    Hg, Wg = target_prob_mask.shape
    if target_coords.shape[:2] != (Hg, Wg):
        raise ValueError(
            f"Shape mismatch: target_coords {target_coords.shape[:2]} vs "
            f"target_prob_mask {(Hg, Wg)}"
        )
    if bbox_format not in ["xyxy", "cxcywh"]:
        raise ValueError("bbox_format must be 'xyxy' or 'cxcywh'")

    # --- Determine Image Dimensions (Pixel Coordinate Space) ---
    if background_image is not None:
        Hi, Wi = background_image.shape[:2]
        if background_image.ndim != 3 or background_image.shape[2] != 3:
            print(f"Warning: background_image shape {background_image.shape} might not be standard HWC RGB.")
    else:
        # If no background image, assume the "image" space matches the grid space
        Hi, Wi = Hg, Wg
        print("Warning: No background image provided. Assuming image dimensions match grid dimensions.")

    # Calculate the size of each grid cell in image pixel coordinates
    # Add epsilon to prevent division by zero if Hg or Wg is 0 (unlikely)
    cell_h = Hi / Hg
    cell_w = Wi / Wg

    # Ensure tensors are on CPU for numpy conversion and plotting
    target_prob_mask_np = target_prob_mask.detach().cpu().numpy()
    target_coords_np = target_coords.detach().cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10 * Hi / Wi)) # Aspect ratio based on image dims

    # --- Optionally Display Background Image ---
    if background_image is not None:
        ax.imshow(background_image)

    # --- Draw Probability Grid Cells as Rectangles ---
    cmap = cm.get_cmap('viridis') # Get the colormap instance
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0) # Normalization for probabilities

    for y_grid in range(Hg):
        for x_grid in range(Wg):
            prob = target_prob_mask_np[y_grid, x_grid]
            if prob < eps:
                continue # Skip empty cells

            # Calculate pixel coordinates for the cell rectangle
            rect_x_pix = x_grid * cell_w
            rect_y_pix = y_grid * cell_h # Top edge y-coordinate
            rect_w_pix = cell_w
            rect_h_pix = cell_h

            # Get the RGBA color corresponding to the probability
            cell_color = cmap(norm(prob))

            # Create rectangle patch for the grid cell
            cell_rect = patches.Rectangle(
                (rect_x_pix, rect_y_pix), rect_w_pix, rect_h_pix,
                linewidth=0,          # No border between cells
                edgecolor='none',
                facecolor=cell_color, # Use the calculated color
                zorder=1,             # Draw above background, below points/boxes
                alpha=0.7
            )
            ax.add_patch(cell_rect)

    # --- Add Manual Colorbar for Probability ---
    # Create a ScalarMappable to represent the colormap and normalization
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # Required for ScalarMappable
    fig.colorbar(sm, ax=ax, label='Target Probability', shrink=0.8)

    # --- Color Management for Target Boxes ---
    bbox_to_color = {}
    colors = plt.cm.tab20.colors
    num_colors = len(colors)
    color_idx = 0
    drawn_target_boxes = set()
    print(f"Number of unique colors available: {num_colors}")

    # --- Iterate Through Grid Cells and Draw ---
    for y_grid in range(Hg): # y is grid row index (0=top)
        for x_grid in range(Wg): # x is grid column index (0=left)
            bbox_cell = target_coords_np[y_grid, x_grid]
            if np.any(np.abs(bbox_cell) > eps):
                if bbox_format == "cxcywh":
                    bbox_xyxy = cxcywh_to_xyxy(bbox_cell)
                else:
                    bbox_xyxy = tuple(bbox_cell)

                if len(bbox_xyxy) != 4 or bbox_xyxy[0] >= bbox_xyxy[2] or bbox_xyxy[1] >= bbox_xyxy[3]:
                    raise ValueError(
                        f"Invalid bbox coordinates: {bbox_xyxy}. "
                        "Ensure they are in the correct format and range."
                    )
                
                bbox_str = f"{bbox_xyxy[0]:.3f}, {bbox_xyxy[1]:.3f}, {bbox_xyxy[2]:.3f}, {bbox_xyxy[3]:.3f}"

                if bbox_str not in bbox_to_color:
                    assigned_color = colors[color_idx % num_colors]
                    bbox_to_color[bbox_str] = assigned_color
                    color_idx += 1
                else:
                    assigned_color = bbox_to_color[bbox_str]

                # --- Draw Center Point for this Active Cell ---
                # Calculate center of the *image region* corresponding to this grid cell
                center_x_pix = (x_grid + 0.5) * cell_w
                center_y_pix = (y_grid + 0.5) * cell_h

                ax.scatter(
                    center_x_pix, center_y_pix, # Use pixel coordinates
                    color=assigned_color,
                    s=40, marker='o', zorder=3, alpha=0.7
                )

                # --- Draw Unique Target Bounding Box Rectangle ---
                # Bboxes are normalized relative to image dims (Hi, Wi)
                if bbox_str not in drawn_target_boxes:
                    x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox_xyxy

                    # Scale normalized coords to image pixel coords
                    rect_x_pix = x_min_norm * Wi
                    rect_y_pix = y_min_norm * Hi # y_min is the top edge coordinate
                    rect_w_pix = (x_max_norm - x_min_norm) * Wi
                    rect_h_pix = (y_max_norm - y_min_norm) * Hi
                    
                    rect = patches.Rectangle(
                        (rect_x_pix, rect_y_pix), rect_w_pix, rect_h_pix,
                        linewidth=2, edgecolor=assigned_color,
                        facecolor='none', zorder=2, alpha=0.9
                    )

                    if verbose:
                        bbox_xyxy_grid = (
                            x_min_norm * Wg, y_min_norm * Hg,
                            x_max_norm * Wg, y_max_norm * Hg
                        )
                        ansi_color_code = rgba_to_ansi(assigned_color)                    
                        print(f"{ansi_color_code}Drawing bbox:\n"
                                f"    bbox_xyxy_grid={bbox_xyxy_grid}\n"
                                f"    rect_x_pix={rect_x_pix}, rect_y_pix={rect_y_pix}, "
                                f"    rect_w_pix={rect_w_pix}, rect_h_pix={rect_h_pix}\n"
                                f"    bbox_str={bbox_str}\n"
                                f"    bbox_xyxy={bbox_xyxy}\n"
                                f"    assigned_color={assigned_color}\033[0m")

                    ax.add_patch(rect)
                    drawn_target_boxes.add(bbox_str)

    # --- Optionally Draw Original Bounding Boxes ---
    # These are also normalized relative to image dims (Hi, Wi)
    if original_bboxes:
        drawn_original_label = False
        for bbox_orig_raw in original_bboxes:
            if bbox_format == "cxcywh":
                bbox_xyxy_orig = cxcywh_to_xyxy(np.array(bbox_orig_raw))
            else:
                bbox_xyxy_orig = tuple(bbox_orig_raw)

            if len(bbox_xyxy_orig) != 4:
                raise ValueError(
                    f"Invalid original bbox coordinates: {bbox_xyxy_orig}. "
                    "Ensure they are in the correct format and range."
                )

            x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox_xyxy_orig

            if x_min_norm >= x_max_norm or y_min_norm >= y_max_norm:
                raise ValueError(
                    f"Invalid original bbox coordinates: {bbox_xyxy_orig}. "
                    "Ensure they are in the correct format and range."
                )

            # Scale normalized coords to image pixel coords
            rect_x_pix = x_min_norm * Wi
            rect_y_pix = y_min_norm * Hi
            rect_w_pix = (x_max_norm - x_min_norm) * Wi
            rect_h_pix = (y_max_norm - y_min_norm) * Hi

            if verbose:
                bbox_xyxy_grid = (
                    x_min_norm * Wg, y_min_norm * Hg,
                    x_max_norm * Wg, y_max_norm * Hg
                )
                print(f"Drawing original bbox: {bbox_xyxy_grid}")

            rect = patches.Rectangle(
                (rect_x_pix, rect_y_pix), rect_w_pix, rect_h_pix,
                linewidth=1.5, edgecolor='black', linestyle=':', facecolor='none',
                label='Original Input Box' if not drawn_original_label else "",
                zorder=4
            )
            ax.add_patch(rect)
            drawn_original_label = True

    # --- Plot Formatting ---
    # Set limits and ticks based on image pixel dimensions
    ax.set_xlim(0, Wi)
    ax.set_ylim(Hi, 0) # Y-axis inverted (0 at top)

    # Set ticks to show grid cell boundaries in pixel coordinates
    ax.set_xticks(ticks=np.arange(0, Wi + eps, cell_w), labels=np.arange(0, Wg + 1))
    ax.set_yticks(ticks=np.arange(0, Hi + eps, cell_h), labels=np.arange(0, Hg + 1))
    ax.grid(True, which='both', color='gray', linewidth=0.5, linestyle='-', alpha=0.3)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box') # Keep aspect ratio, adjust box size

    if original_bboxes and drawn_original_label:
        ax.legend()

    plt.tight_layout()
    plt.show()