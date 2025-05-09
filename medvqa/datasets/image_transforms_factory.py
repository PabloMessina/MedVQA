import random
import torch
from torchvision.transforms import v2
from torchvision import transforms as T # Use T alias for clarity
from typing import Callable, Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import cv2
import numpy as np
import logging

import albumentations as A

logger = logging.getLogger(__name__)

class LoadImageFromPath:
    """
    A callable class transform that loads an image from a path string,
    allowing selection between PIL and OpenCV (cv2) loaders and ensuring
    conversion to a specified target mode.

    Handles various input formats, including 16-bit grayscale, converting
    them appropriately for common computer vision tasks (typically to 8-bit).

    Returns the loaded and converted image object:
    - 'pil': Returns a PIL.Image.Image object in the target mode.
    - 'cv2': Returns a NumPy ndarray (HWC for RGB, HW for L) in uint8 dtype.
             OpenCV output is typically RGB if mode='RGB', Grayscale if mode='L'.
    """

    def __init__(
        self,
        loader_library: str = "cv2",
        mode: Optional[str] = "RGB",
    ):
        """
        Initializes the image loader transform.

        Args:
            loader_library (str): The library to use for loading.
                                  Options: 'pil', 'cv2' (default).
            mode (Optional[str]): The target image mode.
                                  Options: 'RGB', 'L' (grayscale).
                                  If None, the image is loaded with minimal
                                  conversion (PIL default, cv2 default BGR/Gray
                                  converted to uint8). Defaults to 'RGB'.

        Raises:
            ValueError: If an unsupported loader_library is specified.
        """
        self.loader_library = loader_library.lower()
        self.mode = mode.upper() if mode else None

        if self.loader_library not in ["pil", "cv2"]:
            raise ValueError(
                f"Unsupported loader_library: '{loader_library}'. "
                "Choose 'pil' or 'cv2'."
            )
        
        if self.mode not in ["RGB", "L"]:
            raise ValueError(
                f"Unsupported mode: '{mode}'. "
                "Choose 'RGB' or 'L'."
            )

    def _load_pil(self, image_path: str) -> Image.Image:
        """Loads an image using PIL and converts to the target mode."""
        try:
            img = Image.open(image_path)
            # Ensure image data is loaded
            img.load()
            
            # Check if conversion is necessary
            current_mode = img.mode
            if current_mode != self.mode:
                try:
                    # Handle common conversions, especially palette ('P')
                    if img.mode == "P" and self.mode == "RGB":
                        img = img.convert("RGB")
                    elif img.mode == "I;16":
                        # Convert 16-bit int -> 8-bit L
                        img_np = np.array(img)
                        img_np = cv2.normalize(
                            img_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                        )
                        img = Image.fromarray(img_np, mode="L")
                        if self.mode == "RGB":
                            img = img.convert("RGB")
                        else:
                            pass # Keep as grayscale
                    else:
                        img = img.convert(self.mode)

                except ValueError as e:
                    logger.warning(
                        f"PIL could not convert image {image_path} from "
                        f"mode '{current_mode}' to target mode "
                        f"'{self.mode}': {e}. Returning image in mode "
                        f"'{img.mode}'."
                    )
                except OSError as e:
                        logger.warning(
                        f"PIL encountered OS error during conversion "
                        f"for {image_path} from '{current_mode}' to "
                        f"'{self.mode}': {e}. Returning image in mode "
                        f"'{img.mode}'."
                    )

            return img
        except FileNotFoundError:
            logger.error(f"Image file not found at: {image_path}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to load/process image {image_path} using PIL: {e}"
            )
            raise

    def _load_cv2(self, image_path: str) -> np.ndarray:
        """
        Loads an image using OpenCV, converts to target mode, ensuring uint8.
        """
        try:
            # Load image with unchanged flag to preserve channels and depth
            img_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if img_np is None:
                raise IOError(f"cv2.imread failed to load image: {image_path}")

            # --- Handle Bit Depth (Convert to uint8) ---
            if img_np.dtype == np.uint16:
                # Normalize 16-bit (0-65535) to 8-bit (0-255)
                img_np = cv2.normalize(
                    img_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
            elif img_np.dtype != np.uint8:
                # Handle other potential dtypes if necessary, e.g., float
                # For simplicity, we'll warn and attempt conversion if not uint8/16
                logger.warning(
                    f"Image {image_path} has unexpected dtype {img_np.dtype}. "
                    "Attempting conversion to uint8. Results may vary."
                )
                if np.issubdtype(img_np.dtype, np.floating):
                    # Assuming float is in [0, 1], scale to [0, 255]
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                else:
                    # Attempt direct conversion for other integer types
                    img_np = img_np.astype(np.uint8)

            # --- Handle Channels and Target Mode ---

            # Determine current channels
            if img_np.ndim == 2:
                current_channels = 1  # Grayscale
            elif img_np.ndim == 3:
                current_channels = img_np.shape[2]  # 3 (BGR) or 4 (BGRA)
            else:
                raise ValueError(
                    f"Unexpected image dimensions: {img_np.ndim} "
                    f"for {image_path}"
                )

            # Convert to target mode
            target_mode = self.mode
            if target_mode == "L":
                if current_channels == 1:
                    # Already grayscale
                    pass
                elif current_channels == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                elif current_channels == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
            elif target_mode == "RGB":
                if current_channels == 1:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                elif current_channels == 3:
                    # Assume BGR input from imread
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                elif current_channels == 4:
                    # Assume BGRA input from imread
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
            else:
                assert False, f"Unsupported target mode: {target_mode}" # This should not happen

            return img_np

        except FileNotFoundError:
            logger.error(f"Image file not found at: {image_path}")
            raise
        except (IOError, cv2.error) as e:  # Catch cv2 specific errors too
            logger.error(
                f"Failed to load/process image {image_path} using cv2: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error processing {image_path} using cv2: {e}"
            )
            raise

    def __call__(self, image_path: str) -> Union[Image.Image, np.ndarray]:
        """
        Loads the image using the configured library and converts to target mode.

        Args:
            image_path (str): The path to the image file.

        Returns:
            Union[PIL.Image.Image, np.ndarray]:
                - PIL.Image.Image if loader_library is 'pil'.
                - NumPy ndarray (uint8) if loader_library is 'cv2'.
        """
        if self.loader_library == "pil":
            return self._load_pil(image_path)
        elif self.loader_library == "cv2":
            return self._load_cv2(image_path)
        else:
            # Should be caught by __init__, but as a safeguard
            raise ValueError(f"Invalid loader library: {self.loader_library}")
        
    def __repr__(self) -> str:
        return f"LoadImageFromPath(loader_library={self.loader_library}, mode={self.mode})"


# --- Helper Class ---
class CoarseDropoutWithoutBbox(A.CoarseDropout):
    """Custom CoarseDropout that doesn't affect bounding boxes."""
    def apply_to_bbox(self, bbox, **params):
        return bbox

# --- Augmentation Class based on Albumentations ---
class ConfigurableAlbumentations:
    """
    Configurable Albumentations pipeline generator.
    Handles creation of train/test augmentation pipelines.
    Can optionally use the test pipeline during training with a given probability.
    Assumes input is NumPy array, outputs augmented NumPy array.
    Compatible with recent Albumentations versions (addressing API changes).
    """

    def __init__(self,
                 image_size: Union[int, Tuple[int, int]],
                 # --- Global Flags ---
                 apply_clahe: bool = True,
                 apply_spatial_train: bool = True,
                 apply_color_train: bool = True,
                 p_test_in_train: float = 0.35, # i.e. 35% of train images will use test pipeline
                 # --- CLAHE Params ---
                 clahe_clip_limit: float = 4.0,
                 clahe_tile_grid_size: Tuple[int, int] = (8, 8),
                 clahe_p_train: float = 0.5,
                 # --- Spatial Params (for training) ---
                 crop_scale: Tuple[float, float] = (0.8, 1.0),
                 crop_p: float = 1.0,
                 ssr_shift_limit: float = 0.1,
                 ssr_scale_limit: float = 0.1,
                 ssr_rotate_limit: int = 20,
                 ssr_p: float = 0.5,
                 hflip_p: float = 0.5,
                 # --- Color Params (for training) ---
                 # Coarse Dropout - Use ranges directly now
                 dropout_num_holes_range: Tuple[int, int] = (4, 8),
                 dropout_hole_height_range_frac: Tuple[float, float] = (0.01, 0.05),
                 dropout_hole_width_range_frac: Tuple[float, float] = (0.01, 0.05),
                 dropout_fill_value: Union[int, float, str] = 0,
                 dropout_p: float = 0.5,
                 # Color Jitter
                 jitter_brightness: float = 0.15,
                 jitter_contrast: float = 0.15,
                 jitter_saturation: float = 0.15,
                 jitter_hue: float = 0.15,
                 jitter_p: float = 0.5,
                 # Gauss Noise - Store absolute std limit, convert later
                 noise_std_limit_abs: Tuple[int, int] = (3, 8),
                 noise_mean_abs: float = 0.0, # Keep mean at 0
                 noise_p: float = 0.2,
                 # Gaussian Blur
                 blur_limit: Tuple[int, int] = (3, 7),
                 blur_p: float = 0.2):

        if isinstance(image_size, int):
            self.height, self.width = image_size, image_size
        else:
            self.height, self.width = image_size

        if not 0.0 <= p_test_in_train <= 1.0:
             raise ValueError("p_test_in_train must be between 0.0 and 1.0")

        # Store config
        self.apply_clahe = apply_clahe
        self.apply_spatial_train = apply_spatial_train
        self.apply_color_train = apply_color_train
        self.p_test_in_train = p_test_in_train
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.clahe_p_train = clahe_p_train
        self.crop_scale = crop_scale
        self.crop_p = crop_p
        self.ssr_shift_limit = ssr_shift_limit
        self.ssr_scale_limit = ssr_scale_limit
        self.ssr_rotate_limit = ssr_rotate_limit
        self.ssr_p = ssr_p
        self.hflip_p = hflip_p

        # CoarseDropout params
        self.dropout_num_holes_range = dropout_num_holes_range
        self.dropout_hole_height_range = dropout_hole_height_range_frac 
        self.dropout_hole_width_range = dropout_hole_width_range_frac
        self.dropout_fill = dropout_fill_value # Renamed param
        self.dropout_p = dropout_p

        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast
        self.jitter_saturation = jitter_saturation
        self.jitter_hue = jitter_hue
        self.jitter_p = jitter_p

        # Store absolute noise limits, will convert to fractional std_range later
        self.noise_std_limit_abs = noise_std_limit_abs
        self.noise_mean_abs = noise_mean_abs
        self.noise_p = noise_p

        self.blur_limit = (max(3, blur_limit[0] // 2 * 2 + 1),
                           max(3, blur_limit[1] // 2 * 2 + 1))
        self.blur_p = blur_p


    def _build_single_pipeline(self, is_train: bool, bbox_format: Optional[str] = None) -> A.Compose:
        """Internal helper to build either the train or test A.Compose object."""
        pipeline_steps: List[A.BasicTransform] = []

        # 1. CLAHE (Conditional) - Uses p=1.0 for always_apply
        if self.apply_clahe:
            clahe_p = self.clahe_p_train if is_train else 1.0
            pipeline_steps.append(A.CLAHE(clip_limit=self.clahe_clip_limit,
                                          tile_grid_size=self.clahe_tile_grid_size,
                                          p=clahe_p))

        # --- Training Specific Augmentations ---
        if is_train:
            if self.apply_spatial_train:
                # Uses size=(h, w)
                pipeline_steps.append(A.RandomResizedCrop(height=self.height,
                                                          width=self.width,
                                                          scale=self.crop_scale,
                                                          interpolation=cv2.INTER_CUBIC,
                                                          p=self.crop_p))
                affine_transform = A.Affine(
                    scale=(1.0 - self.ssr_scale_limit, 1.0 + self.ssr_scale_limit),
                    translate_percent=self.ssr_shift_limit,
                    rotate=self.ssr_rotate_limit,
                    shear=0,
                    interpolation=cv2.INTER_CUBIC,
                    mode=cv2.BORDER_CONSTANT,
                    p=self.ssr_p
                )
                pipeline_steps.append(affine_transform)

                pipeline_steps.append(A.HorizontalFlip(p=self.hflip_p))

            if self.apply_color_train:
                pipeline_steps.append(CoarseDropoutWithoutBbox(
                    min_holes=self.dropout_num_holes_range[0],
                    max_holes=self.dropout_num_holes_range[1],
                    min_height=self.dropout_hole_height_range[0],
                    max_height=self.dropout_hole_height_range[1],
                    min_width=self.dropout_hole_width_range[0],
                    max_width=self.dropout_hole_width_range[1],
                    fill_value=self.dropout_fill,
                    p=self.dropout_p
                ))
                pipeline_steps.append(A.ColorJitter(brightness=self.jitter_brightness,
                                                    contrast=self.jitter_contrast,
                                                    saturation=self.jitter_saturation,
                                                    hue=self.jitter_hue,
                                                    p=self.jitter_p))

                # Convert the standard deviation limits (absolute) to variance limits (absolute) by squaring.
                # Assumes self.noise_std_limit_abs contains the desired std dev range in absolute terms.
                var_limit_abs = (self.noise_std_limit_abs[0]**2, self.noise_std_limit_abs[1]**2)

                # The mean for the old API is a single value.
                # Assumes self.noise_mean_abs contains the desired mean in absolute terms.
                mean_abs = self.noise_mean_abs

                pipeline_steps.append(A.GaussNoise(
                    var_limit=var_limit_abs,
                    mean=mean_abs,
                    # per_channel=True, # Default is True, keep it?
                    p=self.noise_p
                ))
                pipeline_steps.append(A.GaussianBlur(blur_limit=self.blur_limit, p=self.blur_p))

        # 4. Resize (Always apply at the end for consistency) - Uses p=1.0
        pipeline_steps.append(A.Resize(height=self.height, width=self.width,
                                       interpolation=cv2.INTER_CUBIC,
                                       p=1.0))

        # --- BBox Params ---
        bbox_params = None
        if bbox_format:
            bbox_params = A.BboxParams(format=bbox_format,
                                       label_fields=['bbox_labels'])

        return A.Compose(pipeline_steps, bbox_params=bbox_params)


    def build_pipeline(self, is_train: bool, bbox_format: Optional[str] = None) -> Callable[..., Dict[str, Any]]:
        """
        Builds the augmentation pipeline. Handles stochastic train/test switching.
        """
        # --- Build the Test Pipeline ---
        test_pipeline = self._build_single_pipeline(is_train=False, bbox_format=bbox_format)

        if not is_train:
            return test_pipeline
        else:
            if self.p_test_in_train <= 0.0:
                # --- Build the Full Train Pipeline ---
                return self._build_single_pipeline(is_train=True, bbox_format=bbox_format)
            else:
                # --- Build Full Train Pipeline for Stochastic Wrapper ---
                full_train_pipeline = self._build_single_pipeline(is_train=True, bbox_format=bbox_format)

                # --- Define Stochastic Wrapper ---
                def stochastic_pipeline_wrapper(**kwargs: Any) -> Dict[str, Any]:
                    if random.random() < self.p_test_in_train:
                        return test_pipeline(**kwargs)
                    else:
                        return full_train_pipeline(**kwargs)

                return stochastic_pipeline_wrapper


def get_cxrmate_rrg24_transforms(
    model: torch.nn.Module,
    train: bool = False,
) -> Callable:
    """
    Returns the CXR-Mate-RRG24 transforms. Based on https://huggingface.co/aehrc/cxrmate-rrg24
    Args:
        model (torch.nn.Module): The model for which the transforms are defined.
        train (bool): If True, returns training transforms. If False, returns evaluation transforms.
    Returns:
        Callable: A callable that applies the defined transforms.
    """

    # Access model config parameters
    try:
        image_size = model.config.encoder.image_size
        image_mean = model.config.encoder.image_mean
        image_std = model.config.encoder.image_std
    except AttributeError as e:
        raise ValueError("Model config does not contain expected encoder attributes.") from e
    
    if train:
        raise NotImplementedError("Training transforms not implemented for CXR-Mate-RRG24 yet.")
    else:
        transforms_list = [
            # Load image from path        
            LoadImageFromPath(loader_library='pil', mode='RGB'),
            # Copy of the original CXR-Mate-RRG24 transforms
            v2.PILToTensor(),
            v2.Grayscale(num_output_channels=3), # Operates on Tensor
            v2.Resize(size=image_size, antialias=True),
            v2.CenterCrop(size=[image_size]*2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_mean, std=image_std),
        ]
        return v2.Compose(transforms_list)


_BBOX_FORMAT_TO_ALBUMENTATIONS = {
    'pascal_voc': 'pascal_voc',
    'coco': 'coco',
    'yolo': 'yolo',
    'albumentations': 'albumentations',
    'xyxy': 'albumentations',
    'cxcywh': 'yolo',
}

def create_image_transforms(
    use_model_specific_transforms: bool,
    # --- Model Specific Args ---
    model_name: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    # --- General Pipeline Args ---
    image_size: Optional[Union[int, Tuple[int, int]]] = None, # Required if not model-specific
    loader_library: str = 'cv2',
    loader_mode: Optional[str] = 'RGB',
    augmenter_override_params: Optional[Dict[str, Any]] = None,
    bbox_format: Optional[str] = None,
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    # --- Common Args ---
    is_train: bool = False,
) -> Callable:
    """
    Creates a callable image transformation pipeline.

    Handles model-specific hardcoded pipelines or general pipelines using
    LoadImageFromPath -> ConfigurableAlbumentations -> ToTensor -> Normalize.

    Args:
        use_model_specific_transforms: If True, uses model-specific transforms.
        model_name: Identifier for model-specific transforms. Required if use_model_specific_transforms is True.
        model: Model object, potentially needed for model-specific transforms.
        image_size: Target output image size (int or H, W tuple). MANDATORY if
                    use_model_specific_transforms is False.
        loader_library: Library for LoadImageFromPath ('pil' or 'cv2').
        loader_mode: Mode for LoadImageFromPath (e.g., 'RGB').
        augmenter_override_params: Dictionary of parameters to override defaults in
                                   ConfigurableAlbumentations constructor.
        bbox_format: Albumentations bbox format string (e.g., 'pascal_voc'). If set,
                     the returned callable expects 'bboxes' and 'bbox_labels' inputs.
        image_mean: Mean values for normalization.
        image_std: Standard deviation values for normalization.
        is_train: If True, creates training transforms (enables augmentations).

    Returns:
        Callable: A function that accepts keyword arguments (like image_path=...,
                  bboxes=..., bbox_labels=...) and returns a dictionary containing
                  at least 'pixel_values' (the processed tensor) and potentially
                  augmented 'bboxes' and 'bbox_labels'.

    Raises:
        ValueError: If configuration is invalid (e.g., missing model name/object
                    for specific transforms, missing image_size for general transforms,
                    or unimplemented model).
    """
    
    # --- Validate bounding box format ---
    if bbox_format is not None:
        if bbox_format not in _BBOX_FORMAT_TO_ALBUMENTATIONS:
            raise ValueError(f"Unsupported bbox_format '{bbox_format}'. Supported formats are: {list(_BBOX_FORMAT_TO_ALBUMENTATIONS.keys())}")
        else:
            bbox_format = _BBOX_FORMAT_TO_ALBUMENTATIONS[bbox_format] # Convert to Albumentations format

    if use_model_specific_transforms:
        # --- Model-Specific Pipeline ---
        if not model_name:
            raise ValueError("model_name must be provided when use_model_specific_transforms is True.")

        logger.info(f"Using model-specific transforms for: {model_name}")
        if model_name == "aehrc/cxrmate-rrg24":
            if model is None:
                raise ValueError("Model object must be provided for cxrmate-rrg24 transforms.")
            return get_cxrmate_rrg24_transforms(model, train=is_train)
        # Add elif blocks for other known models here
        else:
            raise ValueError(f"Model-specific transforms not implemented for '{model_name}'.")
    else:
        # --- Build General Pipeline ---
        if image_size is None:
            raise ValueError("image_size must be provided when use_model_specific_transforms is False.")

        logger.info(f"Building general image transform pipeline (is_train={is_train}, bbox_format={bbox_format})")

        # 1. Image Loader
        image_loader = LoadImageFromPath(loader_library=loader_library, mode=loader_mode)
        if loader_library == 'pil':
            def pil_loader_wrapper(path):
                pil_img = image_loader(path)
                return np.array(pil_img) # Convert to NumPy array
            image_loader_fn = pil_loader_wrapper
        else:
            image_loader_fn = image_loader

        # 2. Albumentations Augmenter
        # Start with override params, ensure image_size is included
        override_params = augmenter_override_params if augmenter_override_params is not None else {}
        # image_size is now a mandatory top-level arg for this branch
        augmenter_init_args = {'image_size': image_size, **override_params}

        try:
            augmenter = ConfigurableAlbumentations(**augmenter_init_args)
        except TypeError as e:
            logger.error(f"Error initializing ConfigurableAlbumentations. Check parameters in augmenter_override_params: {e}")
            raise ValueError(f"Invalid parameter provided for ConfigurableAlbumentations: {e}") from e

        tf_augs = augmenter.build_pipeline(is_train=is_train, bbox_format=bbox_format)

        # 3. ToTensor and Normalize
        tf_totensor = T.ToTensor()
        tf_normalize = T.Normalize(mean=image_mean, std=image_std)

        # 4. Define the final wrapper function
        def general_transform_fn(
            image_path: str,
            bboxes: Optional[List[List[float]]] = None,
            bbox_labels: Optional[List[Any]] = None,
            masks: Optional[List[Any]] = None,
        ) -> Dict[str, Any]:
            try:
                # Step 1: Load image -> NumPy HWC RGB
                image_np = image_loader_fn(image_path)

                # Step 2: Apply Albumentations
                alb_input = {'image': image_np}
                if bbox_format is not None:
                    if bboxes is not None and bbox_labels is not None:
                        alb_input['bboxes'] = bboxes
                        alb_input['bbox_labels'] = bbox_labels
                    else:
                        raise ValueError(f"bbox_format '{bbox_format}' requires both bboxes and bbox_labels for {image_path}. Augmenting image only.")
                if masks is not None:
                    # If masks do not match the image size, resize them before passing
                    image_height, image_width = image_np.shape[:2]
                    masks_resized = False
                    original_mask_shapes = [mask.shape for mask in masks]
                    for i, mask in enumerate(masks):
                        if mask.shape[0] != image_height or mask.shape[1] != image_width:
                            masks[i] = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
                            masks_resized = True
                    alb_input['masks'] = masks

                augmented_data = tf_augs(**alb_input)
                augmented_image_np = augmented_data['image']

                # --- Resize masks back to their original size if resized ---
                if masks is not None and masks_resized:
                    for i, mask in enumerate(augmented_data['masks']):
                        mask = cv2.resize(mask, original_mask_shapes[i][1::-1], interpolation=cv2.INTER_LINEAR)
                        augmented_data['masks'][i] = mask

                # Step 3: Convert augmented image to Tensor
                image_tensor = tf_totensor(augmented_image_np)

                # Step 4: Normalize Tensor
                normalized_tensor = tf_normalize(image_tensor)

                # --- Prepare Output Dictionary ---
                output_dict = {'pixel_values': normalized_tensor}
                if bbox_format is not None:
                    output_dict['bboxes'] = augmented_data['bboxes']
                    output_dict['bbox_labels'] = augmented_data['bbox_labels']
                if masks is not None:
                    output_dict['masks'] = augmented_data['masks']

                return output_dict

            except FileNotFoundError as e:
                logger.error(f"Transform error: Image not found - {e}")
                raise
            except Exception as e:
                logger.error(f"Error during transform for {image_path}: {e}", exc_info=True)
                raise

        # Return the callable wrapper function
        return general_transform_fn