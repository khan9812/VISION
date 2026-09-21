"""
Preprocessing Module for Particle Analysis
===========================================
Sequential denoising pipeline with BM3D and Noise2SR.

This module implements a two-stage denoising approach:
1. BM3D: Traditional block-matching 3D filtering (Dabov et al., 2007)
2. Noise2SR: Zero-shot deep learning denoising (Tian et al., 2024)

Functions:
- apply_clahe: Contrast Limited Adaptive Histogram Equalization
- auto_preprocess: Sequential BM3D + Noise2SR preprocessing
- preprocess_with_config: Custom preprocessing with specific method combinations
- load_parameters: Load parameters from YAML configuration files

References:
    - BM3D: Dabov et al. (2007) "Image Denoising by Sparse 3-D Transform-Domain
      Collaborative Filtering"
    - Noise2SR: Tian et al. (2024) "Zero-Shot Image Denoising for High-Resolution
      Electron Microscopy"
    - CLAHE: Zuiderveld (1994) "Contrast Limited Adaptive Histogram Equalization"
"""

import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
from modules.runtime_config import resolve_noise2sr_config
from modules.enhancement import (
    apply_bm3d_denoising,
    apply_noise2sr_network,
    check_bm3d_available,
    check_noise2sr_available
)

# Global variables for callback functions
ref_point = []
cropping = False
temp_image = None
img_for_draw = None
center = None
radius = 0
drawing = False
scale_bar_points = []

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, temp_image, img_for_draw

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            temp_image = img_for_draw.copy()
            cv2.rectangle(temp_image, ref_point[0], (x, y), (73, 64, 225), 2)
            cv2.imshow("image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv2.rectangle(img_for_draw, ref_point[0], ref_point[1], (73, 64, 225), 2)
        cv2.imshow("image", img_for_draw)

def click_and_draw_circle(event, x, y, flags, param):
    global center, radius, drawing, temp_image, img_for_draw

    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        drawing = True
        temp_image = img_for_draw.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            radius = int(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2))
            temp_image = img_for_draw.copy()
            # 드래그할 때 나타나는 원의 색상을 #7200da(BGR: 218, 0, 114)로 설정
            cv2.circle(temp_image, center, radius, (73, 64, 225), 2)
            cv2.imshow("image", temp_image)
            cv2.waitKey(1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        radius = int(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2))
        # 선택 완료 후에도 동일한 색상으로 원 표시
        cv2.circle(img_for_draw, center, radius, (73, 64, 225), 2)
        cv2.imshow("image", img_for_draw)

def click_and_select_scale_bar(event, x, y, flags, param):
    global scale_bar_points, img_for_draw

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(scale_bar_points) < 2:
            scale_bar_points.append((x, y))
            cv2.circle(img_for_draw, (x, y), 5, (73, 64, 225), -1)
            cv2.imshow("image", img_for_draw)

def set_global_variables(img_for_draw_param, ref_point_param=None, center_param=None, radius_param=0, scale_bar_points_param=None):
    """Set global variables used by callback functions"""
    global img_for_draw, ref_point, center, radius, scale_bar_points, temp_image, cropping, drawing
    
    img_for_draw = img_for_draw_param
    temp_image = img_for_draw_param.copy()
    
    if ref_point_param is not None:
        ref_point = ref_point_param
    else:
        ref_point = []
    
    if center_param is not None:
        center = center_param
    else:
        center = None
        
    radius = radius_param
    
    if scale_bar_points_param is not None:
        scale_bar_points = scale_bar_points_param
    else:
        scale_bar_points = []
    
    cropping = False
    drawing = False

def get_global_variables():
    """Get current global variables"""
    return {
        'ref_point': ref_point,
        'center': center,
        'radius': radius,
        'scale_bar_points': scale_bar_points
    }


def extract_boundary_pixels_dict(filtered_masks):
    """
    Extract boundary pixels from filtered masks and store in dictionary.

    This function extracts boundary pixels for each mask using cv2.findContours
    and stores them in a dictionary with centroid as key.

    Args:
        filtered_masks: List of mask dictionaries from SAM
            Each mask dict must have 'segmentation' key with binary mask array

    Returns:
        dict: Dictionary mapping centroid tuple (x, y) to list of boundary pixels
            - key: tuple(centroid_x, centroid_y) - centroid coordinates
            - value: list of tuples [(x1, y1), (x2, y2), ...] - boundary pixel coordinates

    Example:
        >>> boundary_pixels = extract_boundary_pixels_dict(filtered_masks)
        >>> # Access boundary pixels for specific centroid
        >>> centroid = (100.5, 200.3)
        >>> pixels = boundary_pixels[centroid]
        >>> print(f"Centroid {centroid} has {len(pixels)} boundary pixels")

    Notes:
        - Uses cv2.RETR_EXTERNAL to get only outer contours
        - Uses cv2.CHAIN_APPROX_SIMPLE for efficient storage
        - Automatically calculates centroids from mask moments
        - Skips masks with no contours or invalid moments

    Reference:
        Based on Distribution.py lines 307-333
    """
    import cv2

    # Dictionary to store boundary pixels with centroid as key
    boundary_pixels = {}

    for idx, mask_dict in enumerate(filtered_masks):
        mask = mask_dict['segmentation']  # Extract segmentation numpy array

        # Convert to binary mask
        binary_mask = (mask > 0).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"Warning: No contours found for mask {idx}")
            continue

        # Calculate centroid from largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] == 0:
            print(f"Warning: Invalid moment (m00=0) for mask {idx}")
            continue

        # Calculate centroid coordinates
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        key = (centroid_x, centroid_y)

        print(f"Processing centroid {key} for mask {idx}")

        # Store boundary pixels for this centroid
        boundary_pixels[key] = []

        for contour in contours:
            for point in contour:
                boundary_pixels[key].append(tuple(point[0]))

        # Debug output
        if not boundary_pixels[key]:
            print(f"Warning: No boundary pixels stored for centroid {key}")
        else:
            print(f"Centroid {key} has {len(boundary_pixels[key])} boundary pixels")

    return boundary_pixels

def load_parameters(
    default_config: str = 'configs/sam2.1/default_parameters.yaml',
    dataset_config: Optional[str] = 'configs/sam2.1/dataset_parameters.yaml'
) -> Dict:
    """
    Load preprocessing parameters from configuration files.

    Loads literature-based default parameters and dataset-specific
    estimated parameters.

    Args:
        default_config: Path to default parameters YAML
        dataset_config: Path to dataset-specific parameters YAML (None to skip)

    Returns:
        dict: Combined parameters with dataset values taking precedence

    Raises:
        FileNotFoundError: If configuration files are missing
    """
    params = {}

    project_root = Path(__file__).resolve().parent.parent

    # Load the repository-local default parameters regardless of launch cwd.
    default_path = Path(default_config)
    if not default_path.is_absolute():
        default_path = project_root / default_path
    if default_path.exists():
        with open(default_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Required VISION configuration not found: {default_path}")

    # Load dataset-specific parameters (if available)
    dataset_path = Path(dataset_config) if dataset_config else None
    if dataset_path is not None and not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    if dataset_path is not None and dataset_path.exists():
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset_params = yaml.safe_load(f)

        # Support both new and old config formats
        # New format: direct keys (bm3d, noise2sr, etc.)
        # Old format: recommended_parameters wrapper

        if 'bm3d' in dataset_params:
            # New format: merge bm3d section directly
            if params.get('bm3d') is None:
                params['bm3d'] = {}
            params['bm3d'].update(dataset_params['bm3d'])

        elif 'recommended_parameters' in dataset_params:
            # Old format: extract from recommended_parameters
            if params.get('bm3d') is None:
                params['bm3d'] = {}
            params['bm3d']['sigma_psd'] = dataset_params['recommended_parameters']['bm3d_sigma_psd']
            params['bm3d']['estimation_method'] = dataset_params['recommended_parameters']['estimation_method']
    elif dataset_path is not None:
        print(f"Warning: {dataset_config} not found")
        print(f"   Run 'python setup/estimate_parameters.py' to generate it")

    return params


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Applies CLAHE on the L channel in LAB color space to preserve color
    information while enhancing contrast.

    Args:
        image: Input image in BGR format (OpenCV standard)
        clip_limit: Threshold for contrast limiting (range: 2.0-4.0)
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Enhanced image in BGR format

    Reference:
        Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram
        Equalization. Graphics Gems IV, pp. 474-485.

    Example:
        >>> image = cv2.imread('particle_image.png')
        >>> enhanced = apply_clahe(image, clip_limit=2.0)
    """
    if image is None:
        raise ValueError("Input image is None")

    if len(image.shape) != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape: {image.shape}")

    print(f"  CLAHE (clip_limit={clip_limit}, tile_grid={tile_grid_size})...")

    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
    l_enhanced = clahe.apply(l)

    # Merge channels back
    lab_enhanced = cv2.merge([l_enhanced, a, b])

    # Convert back to BGR
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    print("  CLAHE applied")

    return bgr_enhanced


def auto_preprocess(
    image,
    config_path: Optional[str] = None,
    sigma_psd: Optional[float] = None,
    use_dataset_params: bool = True,
    force_noise2sr: Optional[bool] = None,
    noise2sr_epochs: Optional[int] = None,
    noise2sr_settings: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Automatic preprocessing with sequential BM3D + Noise2SR denoising.

    Applies denoising methods sequentially:
    1. BM3D (traditional method) - if enabled and sigma_psd available
    2. Noise2SR (zero-shot deep learning) - if enabled

    Args:
        image: Input image in BGR format
        config_path: Path to default parameters config (default: config/default_parameters.yaml)
        sigma_psd: BM3D noise level (default: use dataset config value)
        use_dataset_params: Use dataset-specific parameters if available
        force_noise2sr: Optional runtime override for Noise2SR. When False,
            Noise2SR is skipped even if enabled in the config file.
        noise2sr_epochs: Optional Noise2SR epoch override

    Returns:
        tuple: (preprocessed_image, info_dict)
            - preprocessed_image: Denoised image
            - info_dict: Dictionary with applied parameters and metadata

    References:
        - BM3D: Dabov et al. (2007)
        - Noise2SR: Tian et al. (2024)

    Example:
        >>> image = cv2.imread('tem_image.png')
        >>> enhanced, info = auto_preprocess(image)
        >>> print(f"Method: {info['denoising_method']}")
    """
    if image is None:
        raise ValueError("Input image is None")

    print("\n" + "="*80)
    print("PREPROCESSING - Sequential BM3D + Noise2SR")
    print("="*80)

    # Load parameters
    if config_path is None:
        config_path = 'configs/sam2.1/default_parameters.yaml'

    dataset_config = 'configs/sam2.1/dataset_parameters.yaml' if use_dataset_params else None
    params = load_parameters(config_path, dataset_config)

    # Determine sigma_psd for BM3D
    estimation_method = None
    if sigma_psd is None:
        # Try to get from dataset parameters
        if params.get('bm3d', {}).get('sigma_psd') is not None:
            sigma_psd = params['bm3d']['sigma_psd']
            estimation_method = params['bm3d'].get('estimation_method', 'dataset_config')
            print(f"\nBM3D sigma_psd from dataset config: {sigma_psd:.2f}")
            print(f"   Method: {estimation_method}")
        else:
            # No sigma_psd available - BM3D will be skipped
            print(f"\nWarning: No sigma_psd available - BM3D will be skipped")
            print(f"   Set sigma_psd in dataset config or pass as parameter")
    else:
        estimation_method = 'user_override'
        print(f"\nUsing user-specified sigma_psd: {sigma_psd:.2f}")

    print()
    print("="*80)
    print("SEQUENTIAL DENOISING PIPELINE")
    print("="*80)

    # Apply preprocessing
    preprocessed = image.copy()

    # Track which methods are applied
    bm3d_applied = False
    noise2sr_applied = False
    bm3d_info = {}
    noise2sr_info = {}

    # STEP 1: Apply BM3D (if enabled and sigma_psd available)
    bm3d_config = params.get('bm3d', {})
    bm3d_enabled = bm3d_config.get('enabled', True)

    if bm3d_enabled and check_bm3d_available() and sigma_psd is not None:
        print(f"\nSTEP 1: Applying BM3D denoising...")
        print(f"   σ_psd = {sigma_psd:.2f}")
        print(f"   Reference: Dabov et al. (2007)")

        preprocessed = apply_bm3d_denoising(preprocessed, sigma_psd=sigma_psd)
        bm3d_applied = True
        bm3d_info = {
            'sigma_psd': sigma_psd,
            'estimation_method': estimation_method
        }
        print(f"   BM3D complete")

    elif bm3d_enabled and not check_bm3d_available():
        print(f"\nWarning: BM3D enabled but not installed (pip install bm3d)")

    elif bm3d_enabled and sigma_psd is None:
        print(f"\nWarning: BM3D enabled but no sigma_psd available - skipped")

    # STEP 2: Apply Noise2SR (if enabled)
    noise2sr_config = params.get('noise2sr', {})
    noise2sr_enabled = bool(noise2sr_config.get('enabled', False))

    # ✨ Override with force_noise2sr if provided
    if force_noise2sr is not None:
        noise2sr_enabled = bool(force_noise2sr)
        state = "enabled" if noise2sr_enabled else "disabled"
        print(f"\n[INFO] Noise2SR {state} by runtime setting")

    if noise2sr_enabled and check_noise2sr_available():
        print(f"\nSTEP 2: Applying Noise2SR zero-shot denoising")
        print(f"   Reference: Tian et al. (2024)")

        # Prepare Noise2SR config
        n2sr_config = resolve_noise2sr_config(
            noise2sr_settings, parameters=params, epochs=noise2sr_epochs,
            image_shape=preprocessed.shape,
        )

        preprocessed, noise2sr_info = apply_noise2sr_network(
            preprocessed,
            config=n2sr_config,
            verbose=True
        )
        noise2sr_applied = True
        print(f"   Noise2SR complete")

    elif noise2sr_enabled and not check_noise2sr_available():
        print(f"\nWarning: Noise2SR enabled but not available")
    else:
        print(f"\nSTEP 2: Noise2SR skipped")

    # Determine overall method
    if bm3d_applied and noise2sr_applied:
        denoising_method = 'bm3d+noise2sr'
    elif bm3d_applied:
        denoising_method = 'bm3d'
    elif noise2sr_applied:
        denoising_method = 'noise2sr'
    else:
        denoising_method = 'none'
        print(f"\nWarning: No denoising applied")

    # Optional CLAHE (if enabled in config)
    clahe_applied = False
    if params.get('clahe', {}).get('enabled', False):
        clip_limit = params['clahe']['clip_limit']
        tile_grid_size = tuple(params['clahe']['tile_grid_size'])

        print(f"\nApplying CLAHE...")
        print(f"   clip_limit = {clip_limit}")
        print(f"   tile_grid = {tile_grid_size}")
        print(f"   Reference: Zuiderveld (1994)")

        preprocessed = apply_clahe(preprocessed, clip_limit, tile_grid_size)
        clahe_applied = True

    print("\n" + "="*80)
    print("Preprocessing Complete")
    print("="*80)

    # Prepare info dictionary
    info = {
        'denoising_method': denoising_method,
        'bm3d_applied': bm3d_applied,
        'noise2sr_applied': noise2sr_applied,
        'clahe_applied': clahe_applied,
        'parameters_source': {
            'default_config': config_path,
            'dataset_config': dataset_config if use_dataset_params else None
        }
    }

    # Add method-specific info
    if bm3d_applied:
        info['bm3d_info'] = bm3d_info
        info['sigma_psd'] = bm3d_info.get('sigma_psd')
        info['estimation_method'] = bm3d_info.get('estimation_method')

    if noise2sr_applied:
        info['noise2sr_info'] = noise2sr_info

    if clahe_applied:
        info['clahe_params'] = {
            'clip_limit': params['clahe']['clip_limit'],
            'tile_grid_size': params['clahe']['tile_grid_size']
        }

    return preprocessed, info


def preprocess_with_config(
    image: np.ndarray,
    sigma_psd: Optional[float] = None,
    bm3d_enabled: bool = True,
    noise2sr_enabled: bool = True,
    clahe_enabled: bool = False,
    verbose: bool = True,
    noise2sr_epochs: Optional[int] = None,
    noise2sr_settings: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess image with specific combination of methods.

    This function allows explicit control over which preprocessing methods to apply,
    useful for comparing different preprocessing strategies.

    Parameters:
    -----------
    image : numpy.ndarray
        Input TEM image in BGR format
    sigma_psd : float, optional
        Noise standard deviation for BM3D (0-255 scale)
        If None, will try to load from dataset_parameters.yaml
    bm3d_enabled : bool, default=True
        Whether to apply BM3D denoising
    noise2sr_enabled : bool, default=True
        Whether to apply Noise2SR denoising
    clahe_enabled : bool, default=False
        Whether to apply CLAHE contrast enhancement
    verbose : bool, default=True
        Whether to print progress messages
    noise2sr_epochs : int, optional
        Runtime override for per-image Noise2SR training epochs

    Note:
        Configuration is always loaded from config/default_parameters.yaml
        and config/dataset_parameters.yaml

    Returns:
    --------
    tuple(numpy.ndarray, dict)
        - Preprocessed image in BGR format
        - Dictionary containing preprocessing information

    Examples:
    ---------
    # BM3D only
    img, info = preprocess_with_config(img, bm3d_enabled=True,
                                       noise2sr_enabled=False, clahe_enabled=False)

    # BM3D + Noise2SR
    img, info = preprocess_with_config(img, bm3d_enabled=True,
                                       noise2sr_enabled=True, clahe_enabled=False)

    # BM3D + Noise2SR + CLAHE
    img, info = preprocess_with_config(img, bm3d_enabled=True,
                                       noise2sr_enabled=True, clahe_enabled=True)

    # BM3D + CLAHE
    img, info = preprocess_with_config(img, bm3d_enabled=True,
                                       noise2sr_enabled=False, clahe_enabled=True)
    """

    # Load parameters from config folder
    params = load_parameters()

    # Validate image
    if image is None:
        raise ValueError("Input image is None")

    # Get sigma_psd
    estimation_method = None
    if sigma_psd is None:
        if params.get('bm3d', {}).get('sigma_psd') is not None:
            sigma_psd = params['bm3d']['sigma_psd']
            estimation_method = params['bm3d'].get('estimation_method', 'dataset_config')
        else:
            if verbose and bm3d_enabled:
                print(f"\nWarning: No sigma_psd available - BM3D will be skipped")
    else:
        estimation_method = 'user_override'

    # Apply preprocessing
    preprocessed = image.copy()
    bm3d_applied = False
    noise2sr_applied = False
    clahe_applied_flag = False
    bm3d_info = {}
    noise2sr_info = {}
    if verbose:
        print()
        print("="*80)
        print("CUSTOM PREPROCESSING PIPELINE")
        print("="*80)

    # STEP 1: BM3D (if enabled)
    if bm3d_enabled and check_bm3d_available() and sigma_psd is not None:
        if verbose:
            print(f"\nSTEP: Applying BM3D denoising...")
            print(f"   σ_psd = {sigma_psd:.2f}")
            print(f"   Profile: Normal (np)")
        preprocessed = apply_bm3d_denoising(preprocessed, sigma_psd=sigma_psd)
        bm3d_applied = True
        bm3d_info = {
            'sigma_psd': sigma_psd,
            'estimation_method': estimation_method,
            'profile': 'np'
        }
        if verbose:
            print(f"   BM3D complete")

    # STEP 2: Noise2SR (if enabled)
    if noise2sr_enabled and check_noise2sr_available():
        if verbose:
            print(f"\nSTEP: Applying Noise2SR zero-shot denoising")
        noise2sr_config = params.get('noise2sr', {})

        # Get desired patch_size from config
        resolved_noise2sr = resolve_noise2sr_config(
            noise2sr_settings, parameters=params, epochs=noise2sr_epochs,
            image_shape=preprocessed.shape,
        )
        desired_patch_size = resolved_noise2sr['patch_size']

        # Adjust patch_size to fit image dimensions
        img_height, img_width = preprocessed.shape[:2]
        min_dimension = min(img_height, img_width)

        # Ensure patch_size doesn't exceed image dimensions
        actual_patch_size = min(desired_patch_size, min_dimension)
        # Noise2SR uses pixel_unshuffle(scale=2); keep its training patches even.
        actual_patch_size = max(2, actual_patch_size - (actual_patch_size % 2))

        if actual_patch_size < desired_patch_size and verbose:
            print(f"   [ICON]️  Adjusted patch_size: {desired_patch_size} → {actual_patch_size} (image too small)")

        # CRITICAL: Ensure image dimensions are EVEN for pixel_unshuffle
        # Noise2SR uses pixel_unshuffle which requires height and width divisible by scale_factor (2)
        original_height, original_width = preprocessed.shape[:2]
        adjusted_height = original_height if original_height % 2 == 0 else original_height - 1
        adjusted_width = original_width if original_width % 2 == 0 else original_width - 1

        if adjusted_height != original_height or adjusted_width != original_width:
            if verbose:
                print(f"   [WARN] Adjusting image size for pixel_unshuffle:")
                print(f"          {original_height}x{original_width} → {adjusted_height}x{adjusted_width}")
            preprocessed = preprocessed[:adjusted_height, :adjusted_width]

        n2sr_config = dict(resolved_noise2sr, patch_size=actual_patch_size)
        preprocessed, noise2sr_info = apply_noise2sr_network(preprocessed, config=n2sr_config, verbose=verbose)

        # Noise2SR requires even dimensions. Restore a cropped final row/column
        # so downstream masks remain aligned pixel-for-pixel with the ROI and
        # original image used for overlays and exports.
        restored_rows = original_height - preprocessed.shape[0]
        restored_cols = original_width - preprocessed.shape[1]
        if restored_rows not in (0, 1) or restored_cols not in (0, 1):
            raise RuntimeError(
                "Unexpected Noise2SR output shape: "
                f"{preprocessed.shape[:2]} for input {(original_height, original_width)}"
            )
        if restored_rows or restored_cols:
            preprocessed = cv2.copyMakeBorder(
                preprocessed,
                0,
                restored_rows,
                0,
                restored_cols,
                cv2.BORDER_REPLICATE,
            )
            noise2sr_info['shape_restoration'] = {
                'cropped_for_even_dimensions': [adjusted_height, adjusted_width],
                'restored_output_dimensions': [original_height, original_width],
                'method': 'replicate_last_row_or_column',
            }
        noise2sr_applied = True
        if verbose:
            print(f"   Noise2SR complete")

    # STEP 3: CLAHE (if enabled)
    if clahe_enabled:
        clahe_config = params.get('clahe', {})
        clip_limit = clahe_config.get('clip_limit', 4.0)
        tile_grid_size = tuple(clahe_config.get('tile_grid_size', [64, 64]))
        if verbose:
            print(f"\nSTEP: Applying CLAHE...")
            print(f"   Clip limit: {clip_limit}")
            print(f"   Tile grid: {tile_grid_size}")
        preprocessed = apply_clahe(preprocessed, clip_limit, tile_grid_size)
        clahe_applied_flag = True
        if verbose:
            print(f"   CLAHE complete")

    # Determine method name
    methods = []
    if bm3d_applied:
        methods.append('bm3d')
    if noise2sr_applied:
        methods.append('noise2sr')
    if clahe_applied_flag:
        methods.append('clahe')

    denoising_method = '+'.join(methods) if methods else 'none'

    if verbose:
        print("\n" + "="*80)
        print(f"Preprocessing Complete: {denoising_method.upper()}")
        print("="*80)

    # Prepare info
    info = {
        'denoising_method': denoising_method,
        'bm3d_applied': bm3d_applied,
        'noise2sr_applied': noise2sr_applied,
        'clahe_applied': clahe_applied_flag,
        'bm3d_info': bm3d_info if bm3d_applied else {},
        'noise2sr_info': noise2sr_info if noise2sr_applied else {}
    }

    if 'sigma_psd' in bm3d_info:
        info['sigma_psd'] = bm3d_info['sigma_psd']

    return preprocessed, info


def get_preprocessing_summary(data_quality):
    """
    Get human-readable summary of preprocessing strategy.

    Args:
        data_quality: str
            Quality assessment: "Basic" | "Good" | "Excellent"

    Returns:
        dict: Summary information with keys:
            - strategy: str (preprocessing approach)
            - methods: list[str] (applied methods)
            - description: str (detailed explanation)
    """
    strategies = {
        "Basic": {
            "strategy": "Aggressive Enhancement",
            "methods": ["CLAHE (Contrast)", "Noise2SR (SR + Denoising)"],
            "description": "Contrast enhancement followed by super-resolution upscaling with denoising for very low-quality images"
        },
        "Good": {
            "strategy": "Moderate Denoising",
            "methods": ["BM3D Denoising"],
            "description": "Noise reduction while preserving edge details for typical EM images"
        },
        "Excellent": {
            "strategy": "Light Preprocessing",
            "methods": ["CLAHE Enhancement"],
            "description": "Contrast enhancement only for high-quality images"
        }
    }

    return strategies.get(data_quality, {
        "strategy": "Unknown",
        "methods": [],
        "description": "Invalid data quality specification"
    })


if __name__ == "__main__":
    # Test module
    print("Preprocessing Module")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - apply_clahe(image, clip_limit, tile_grid_size)")
    print("  - auto_preprocess(image, data_quality, sigma_psd, clip_limit)")
    print("  - get_preprocessing_summary(data_quality)")

    print("\nPreprocessing Strategies:")
    for quality in ["Basic", "Good", "Excellent"]:
        summary = get_preprocessing_summary(quality)
        print(f"\n{quality} Quality:")
        print(f"  Strategy: {summary['strategy']}")
        print(f"  Methods: {', '.join(summary['methods'])}")
        print(f"  Description: {summary['description']}")
