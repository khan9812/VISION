"""
Image Enhancement Module for Low-Quality EM Images

This module provides advanced image enhancement techniques including:
- BM3D denoising for noise reduction while preserving edges
- Noise2SR zero-shot denoising using deep learning (Tian et al., 2024)
- Multiscale NLM denoising (legacy)
- CLAHE for contrast enhancement

References:
    - BM3D: Dabov et al. (2007)
    - Noise2SR: Tian et al. (2024), IEEE TCI
    - NLM: Buades et al. (2005)

Author: Enhanced by Claude
Date: 2025
"""

import numpy as np
import cv2
import os
import gc
from typing import Dict, Tuple, Optional

# Set threading environment variables to avoid BM3D threading issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

try:
    import bm3d
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False
    print("Warning: bm3d package not installed. Install with: pip install bm3d")

# Noise2SR zero-shot denoising
try:
    from modules.noise2sr import denoise_image_noise2sr
    NOISE2SR_AVAILABLE = True
except ImportError as e:
    NOISE2SR_AVAILABLE = False
    print(f"Warning: Noise2SR not available: {e}")
    print("   Make sure PyTorch, scikit-image, and tifffile are installed")

# Check for opencv-contrib super-resolution models (legacy)
try:
    from cv2 import dnn_superres
    SR_AVAILABLE = True
except (ImportError, AttributeError):
    SR_AVAILABLE = False
    # Silently fallback - this is legacy code


def apply_bm3d_denoising(image, sigma_psd=40, color_mode='auto'):
    """
    Apply BM3D (Block-Matching and 3D filtering) denoising to an image.

    BM3D is a state-of-the-art denoising algorithm that:
    - Removes Gaussian noise effectively
    - Preserves edges and fine structures
    - Works well for EM microscopy images

    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format (OpenCV standard)
    sigma_psd : float, optional (default=30)
        Noise standard deviation. Higher values = stronger denoising
        Typical range: 10-50
        - 10-20: Light denoising (high quality images)
        - 20-40: Medium denoising (typical EM images)
        - 40-60: Heavy denoising (very noisy images)
    color_mode : str, optional (default='auto')
        How to handle color:
        - 'auto': Automatically detect if grayscale or color
        - 'grayscale': Convert to grayscale, denoise, keep grayscale
        - 'color': Denoise each channel separately (slower)

    Returns:
    --------
    numpy.ndarray
        Denoised image in same format as input (BGR)

    Raises:
    -------
    ImportError
        If bm3d package is not installed

    Examples:
    ---------
    >>> import cv2
    >>> from utils.enhancement import apply_bm3d_denoising
    >>>
    >>> # Load noisy EM image
    >>> image = cv2.imread('noisy_em_image.png')
    >>>
    >>> # Apply light denoising
    >>> denoised = apply_bm3d_denoising(image, sigma_psd=20)
    >>>
    >>> # Apply heavy denoising for very noisy images
    >>> denoised_heavy = apply_bm3d_denoising(image, sigma_psd=50)
    """

    if not BM3D_AVAILABLE:
        raise ImportError(
            "bm3d package is not installed. "
            "Please install it using: pip install bm3d"
        )

    if image is None:
        raise ValueError("Input image is None")

    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D array.")

    print(f"[ICON] Applying BM3D denoising (sigma_psd={sigma_psd})...")

    # Determine if image is grayscale or color
    is_grayscale = len(image.shape) == 2 or image.shape[2] == 1

    if color_mode == 'auto':
        color_mode = 'grayscale' if is_grayscale else 'color'

    # Case 1: Grayscale image
    if color_mode == 'grayscale':
        # Convert BGR to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Normalize to [0, 1] for BM3D
        gray_normalized = gray.astype(np.float32) / 255.0

        # Apply BM3D with Normal profile
        # Profile 'np' uses literature-based parameters for high-quality denoising
        denoised_normalized = bm3d.bm3d(gray_normalized, sigma_psd=sigma_psd/255.0, profile='np')

        # Convert back to [0, 255]
        denoised = (np.clip(denoised_normalized, 0, 1) * 255).astype(np.uint8)

        # If original was BGR, convert back to BGR
        if len(image.shape) == 3:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        # Force garbage collection to clean up BM3D resources
        gc.collect()
        
        print("[ICON] BM3D denoising completed (grayscale mode)")
        return denoised

    # Case 2: Color image (denoise each channel separately)
    elif color_mode == 'color':
        # Split into BGR channels
        b, g, r = cv2.split(image)

        # Normalize each channel
        b_norm = b.astype(np.float32) / 255.0
        g_norm = g.astype(np.float32) / 255.0
        r_norm = r.astype(np.float32) / 255.0

        # Apply BM3D to each channel with Normal profile
        print("  Denoising Blue channel...")
        b_denoised = bm3d.bm3d(b_norm, sigma_psd=sigma_psd/255.0, profile='np')

        print("  Denoising Green channel...")
        g_denoised = bm3d.bm3d(g_norm, sigma_psd=sigma_psd/255.0, profile='np')

        print("  Denoising Red channel...")
        r_denoised = bm3d.bm3d(r_norm, sigma_psd=sigma_psd/255.0, profile='np')

        # Convert back to [0, 255]
        b_final = (np.clip(b_denoised, 0, 1) * 255).astype(np.uint8)
        g_final = (np.clip(g_denoised, 0, 1) * 255).astype(np.uint8)
        r_final = (np.clip(r_denoised, 0, 1) * 255).astype(np.uint8)

        # Merge channels back
        denoised = cv2.merge([b_final, g_final, r_final])

        print("[ICON] BM3D denoising completed (color mode)")
        return denoised

    else:
        raise ValueError(f"Invalid color_mode: {color_mode}. Use 'auto', 'grayscale', or 'color'")


def apply_noise2sr(image, scale=2, denoise_strength=20):
    """
    Apply Noise2SR-style super-resolution and denoising for low-quality images.

    This function combines super-resolution with denoising to enhance low-quality
    TEM images. It uses:
    1. Super-resolution (EDSR/ESPCN if available, else bicubic)
    2. Non-local means denoising
    3. Sharpening to restore details

    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    scale : int, optional (default=2)
        Upscaling factor (1, 2, 3, or 4)
        Note: scale=1 means no upscaling, only denoising
    denoise_strength : int, optional (default=10)
        Denoising strength (0-30)
        - 0-10: Light denoising
        - 10-20: Medium denoising (recommended)
        - 20-30: Heavy denoising

    Returns:
    --------
    numpy.ndarray
        Enhanced image with same color channels as input

    Examples:
    ---------
    >>> from modules.enhancement import apply_noise2sr
    >>> enhanced = apply_noise2sr(image, scale=2, denoise_strength=15)
    """

    print(f"[ICON] Applying Noise2SR enhancement (scale={scale}, denoise={denoise_strength})...")

    if image is None:
        raise ValueError("Input image is None")

    # Store original size for potential downscaling
    original_h, original_w = image.shape[:2]

    # Step 1: Super-resolution (if scale > 1)
    if scale > 1:
        if SR_AVAILABLE:
            try:
                # Try using DNN super-resolution
                sr = dnn_superres.DnnSuperResImpl_create()

                # Try to load EDSR model (best quality)
                model_path = f"models/EDSR_x{scale}.pb"
                if os.path.exists(model_path):
                    print(f"  Using EDSR model: {model_path}")
                    sr.readModel(model_path)
                    sr.setModel("edsr", scale)
                    upscaled = sr.upsample(image)
                else:
                    # Fallback to ESPCN (faster, smaller)
                    model_path = f"models/ESPCN_x{scale}.pb"
                    if os.path.exists(model_path):
                        print(f"  Using ESPCN model: {model_path}")
                        sr.readModel(model_path)
                        sr.setModel("espcn", scale)
                        upscaled = sr.upsample(image)
                    else:
                        print(f"  SR models not found, using bicubic interpolation")
                        upscaled = cv2.resize(image, None, fx=scale, fy=scale,
                                            interpolation=cv2.INTER_CUBIC)
            except Exception as e:
                print(f"  SR model error: {e}, using bicubic interpolation")
                upscaled = cv2.resize(image, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_CUBIC)
        else:
            # Fallback: bicubic interpolation
            print("  Using bicubic interpolation for upscaling")
            upscaled = cv2.resize(image, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = image.copy()

    # Step 2: Non-local means denoising
    if denoise_strength > 0:
        print(f"  Applying non-local means denoising (h={denoise_strength})...")
        if len(upscaled.shape) == 3:
            # Color image
            denoised = cv2.fastNlMeansDenoisingColored(
                upscaled,
                None,
                h=denoise_strength,
                hColor=denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            # Grayscale image
            denoised = cv2.fastNlMeansDenoising(
                upscaled,
                None,
                h=denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
    else:
        denoised = upscaled.copy()

    # Step 3: Gentle sharpening to restore details
    print("  Applying sharpening...")
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Blend sharpened with denoised (30% sharp, 70% denoised)
    enhanced = cv2.addWeighted(sharpened, 0.3, denoised, 0.7, 0)

    # Step 4: Downscale back to original size if needed
    if scale > 1:
        enhanced = cv2.resize(enhanced, (original_w, original_h),
                            interpolation=cv2.INTER_AREA)
        print(f"  Downscaled back to original size: {original_w}x{original_h}")

    print("[ICON] Noise2SR enhancement completed")
    return enhanced


def check_bm3d_available():
    """
    Check if BM3D package is available.

    Returns:
    --------
    bool
        True if bm3d is installed, False otherwise
    """
    return BM3D_AVAILABLE


def check_sr_available():
    """
    Check if super-resolution (opencv-contrib dnn_superres) is available.

    Returns:
    --------
    bool
        True if dnn_superres is available, False otherwise
    """
    return SR_AVAILABLE


def apply_noise2sr_network(
    image: np.ndarray,
    config: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Apply Noise2SR zero-shot denoising using neural network.

    This function uses the actual Noise2SR method from Tian et al. (2024),
    which trains a neural network specifically for each input image.

    Args:
        image: Input image in BGR format or grayscale
        config: Configuration dictionary (see denoise_image_noise2sr for details)
        verbose: Print training progress

    Returns:
        tuple: (denoised_image, info_dict)

    Reference:
        Tian, X., et al. (2024). Zero-Shot Image Denoising for High-Resolution
        Electron Microscopy. IEEE Transactions on Computational Imaging, 10, 1462-1475.

    Note:
        This method trains a neural network per image, which takes time:
        - GPU: 2-5 minutes per image
        - CPU: 10-30 minutes per image

    Example:
        >>> image = cv2.imread('tem_image.png')
        >>> denoised, info = apply_noise2sr_network(image)
        >>> print(f"Training loss: {info['final_loss']:.6f}")
    """
    if not NOISE2SR_AVAILABLE:
        raise RuntimeError(
            "Noise2SR is not available. Please ensure:\n"
            "  1. PyTorch is installed: pip install torch\n"
            "  2. scikit-image is installed: pip install scikit-image\n"
            "  3. tifffile is installed: pip install tifffile\n"
            "  4. modules/noise2sr/ directory exists with required files"
        )

    return denoise_image_noise2sr(image, config, verbose)


def check_noise2sr_available():
    """
    Check if Noise2SR zero-shot denoising is available.

    Returns:
        bool: True if Noise2SR can be used, False otherwise
    """
    return NOISE2SR_AVAILABLE


if __name__ == "__main__":
    # Simple test
    print("Image Enhancement Module")
    print("="*60)
    print(f"BM3D Available: {BM3D_AVAILABLE}")
    print(f"Noise2SR Available: {NOISE2SR_AVAILABLE}")
    print(f"Super-Resolution (legacy) Available: {SR_AVAILABLE}")

    if BM3D_AVAILABLE:
        print("\n[ICON] BM3D is ready to use!")
        print("  from modules.enhancement import apply_bm3d_denoising")
        print("  denoised = apply_bm3d_denoising(image, sigma_psd=30)")
    else:
        print("\n[ICON] BM3D is not installed.")
        print("  Install with: pip install bm3d")

    if NOISE2SR_AVAILABLE:
        print("\n[ICON] Noise2SR (Tian et al., 2024) is ready to use!")
        print("  from modules.enhancement import apply_noise2sr_network")
        print("  denoised, info = apply_noise2sr_network(image)")
    else:
        print("\n[ICON] Noise2SR is not available.")
        print("  Install dependencies:")
        print("    pip install torch scikit-image tifffile")

    print("\n[ICON] Multiscale NLM (legacy) is always available!")
    print("  from modules.enhancement import apply_noise2sr")
    print("  enhanced = apply_noise2sr(image, scale=2, denoise_strength=15)")

    if not SR_AVAILABLE:
        print("\n[ICON]️  For best SR quality, install opencv-contrib-python:")
        print("  pip install opencv-contrib-python")
