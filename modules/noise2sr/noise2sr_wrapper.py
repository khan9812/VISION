"""
Noise2SR Wrapper for TEM Image Preprocessing

Integrates Noise2SR zero-shot denoising into the TEM analysis pipeline.

Reference:
    Tian, X., et al. (2024). Zero-Shot Image Denoising for High-Resolution
    Electron Microscopy. IEEE Transactions on Computational Imaging, 10, 1462-1475.
"""

import torch
import numpy as np
import cv2
import random
from typing import Dict, Tuple, Optional
import sys
from pathlib import Path

from modules.noise2sr.netarch import Noise2SR
from modules.runtime_config import resolve_noise2sr_config
# Use explicit import from local utils module to avoid conflict with VISION/utils
from modules.noise2sr import utils as noise2sr_utils
PD_sampler = noise2sr_utils.PD_sampler
from modules.noise2sr import dataset as noise2sr_dataset
from tqdm import tqdm


def denoise_image_noise2sr(
    image: np.ndarray,
    config: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Apply Noise2SR zero-shot denoising to a TEM image.

    This function implements the zero-shot self-supervised learning approach
    from Tian et al. (2024) for electron microscopy image denoising.

    Args:
        image: Input image in BGR format (OpenCV standard) or grayscale
        config: Configuration dictionary with parameters:
            - lr (float): Learning rate (default: 1e-4)
            - epoch (int): Training epochs (default: 1500)
            - batch_size (int): Batch size (default: 12)
            - patch_size (int): Training patch size (default: 128)
            - s (int): Scale factor for sub-sampling (default: 2)
            - M (int): Number of inference averages (default: 50)
            - gpu (int): GPU device id (default: 0, -1 for CPU)
        verbose: Print training progress

    Returns:
        tuple: (denoised_image, info_dict)
            - denoised_image: Denoised image in same format as input
            - info_dict: Dictionary with training information

    Reference:
        Tian, X., et al. (2024). Zero-Shot Image Denoising for High-Resolution
        Electron Microscopy. IEEE Transactions on Computational Imaging, 10, 1462-1475.

    Note:
        This is a zero-shot method, meaning it trains a neural network
        specifically for each input image. Training time varies:
        - GPU (recommended): 2-5 minutes per image
        - CPU: 10-30 minutes per image

    Example:
        >>> image = cv2.imread('tem_image.png')
        >>> denoised, info = denoise_image_noise2sr(image)
        >>> print(f"Training epochs: {info['epochs']}")
    """
    if image is None:
        raise ValueError("Input image is None")

    # Default configuration
    config = resolve_noise2sr_config(config, image_shape=image.shape)

    # Extract parameters
    lr = config['lr']
    epoch = config['epoch']
    batch_size = config['batch_size']
    patch_size = config['patch_size']
    if patch_size < 64:
        raise ValueError('Noise2SR requires a training patch and both image dimensions of at least 64 pixels; enlarge the ROI or disable Noise2SR.')
    s = config['s']
    M = config['M']
    gpu = config['gpu']
    seed = int(config.get('seed', 42))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if verbose:
        print("\n" + "="*80)
        print("NOISE2SR - Zero-Shot Image Denoising")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"   Learning rate: {lr}")
        print(f"   Training epochs: {epoch}")
        print(f"   Batch size: {batch_size}")
        print(f"   Patch size: {patch_size}x{patch_size}")
        print(f"   Scale factor: {s}")
        print(f"   Inference averages: {M}")
        print(f"   Random seed: {seed}")
        print(f"   Device: {'GPU ' + str(gpu) if gpu >= 0 and torch.cuda.is_available() else 'CPU'}")

    # Prepare image data
    # Convert to grayscale float32 if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = image.copy()
        is_color = False

    # Normalize to float32 [0, 1] range (Noise2SR expects this)
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0
    else:
        gray = gray.astype(np.float32)
        # Normalize if not already in [0, 1]
        if gray.max() > 1.0:
            span = float(gray.max() - gray.min())
            gray = (gray - gray.min()) / span if span > 0 else np.zeros_like(gray)

    noisy_data = gray
    original_shape = noisy_data.shape

    # Pad image to make dimensions divisible by 32 (for UNet with 5 MaxPool layers)
    # UNet has 5 MaxPool2d(2) layers: 2^5 = 32x downsampling
    # Note: UNet now uses adaptive interpolation for skip connections, so minimal padding is sufficient
    h, w = noisy_data.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    if verbose:
        print(f"\nPadding calculation:")
        print(f"   Original: {h}x{w}")
        print(f"   h % 32 = {h % 32}, w % 32 = {w % 32}")
        print(f"   Padding needed: +{pad_h} (h), +{pad_w} (w)")
        print(f"   Target: {h + pad_h}x{w + pad_w}")

    if pad_h > 0 or pad_w > 0:
        # Pad with reflection to avoid edge artifacts
        noisy_data = np.pad(noisy_data,
                           ((0, pad_h), (0, pad_w)),
                           mode='reflect')
        if verbose:
            print(f"   Padded successfully: {original_shape} -> {noisy_data.shape}")
            print(f"   (UNet has adaptive skip connections for any size)")
    else:
        if verbose:
            print(f"   No padding needed (already 32x multiple)")

    if verbose:
        print(f"\nImage info:")
        print(f"   Shape: {noisy_data.shape}")
        print(f"   Data range: [{noisy_data.min():.4f}, {noisy_data.max():.4f}]")
        print(f"   Format: {'Color (BGR)' if is_color else 'Grayscale'}")

    # Setup device
    if gpu >= 0 and torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{gpu}')
        if verbose:
            print(f"\nUsing GPU: cuda:{gpu}")
    else:
        DEVICE = torch.device('cpu')
        if verbose:
            print(f"\nWarning: Using CPU (training will be slow)")
            print(f"   Consider using GPU for faster training")

    # Initialize components
    Subsampler = PD_sampler(s, DEVICE)

    # Prepare data loaders
    train_data = noisy_data[None]  # Add batch dimension

    if verbose:
        print(f"\nData loader setup:")
        print(f"   Training data shape: {train_data.shape}")
        print(f"   Training patches: {patch_size}x{patch_size} (repeated 40x)")
        print(f"   Validation: Full image {train_data.shape[1]}x{train_data.shape[2]}")

    train_loader = noise2sr_dataset.loader_train(
        train_data.repeat(40, axis=0),
        patch_size,
        batch_size,
        num_workers=config['num_workers'],
        persistent_workers=config['persistent_workers'],
        generator_seed=config['loader_generator_seed'],
    )
    val_loader = noise2sr_dataset.loader_val(train_data, batch_size=1)

    # Initialize model
    model = Noise2SR(
        in_c=1,
        out_c=1,
        feature_dim=128,
        scale_factor=s
    ).to(DEVICE)

    loss_fun = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    if verbose:
        print(f"\n🔧 Training neural network...")
        print(f"   This may take several minutes...")

    # Training loop
    train_loop = tqdm(
        range(epoch),
        desc="Training",
        colour='green',
        leave=False,
        ncols=100,
        disable=not verbose
    )

    for e in train_loop:
        model.train()
        loss_train = 0

        for i, x1 in enumerate(train_loader):
            x1 = x1.to(DEVICE).float()
            x1_in, mask = Subsampler.sample_img(x1, 1)
            mask = mask.to(DEVICE)

            img_pred = model(x1_in)
            loss = loss_fun(img_pred[mask == 1], x1[mask == 1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        avg_loss = loss_train / len(train_loader)
        train_loop.set_description(f'Epoch [{e+1}/{epoch}] loss:{avg_loss:.6f}')

    if verbose:
        print(f"Training complete!")

    # Inference
    if verbose:
        print(f"\nRunning inference (averaging {M} predictions)...")

    denoised_res = np.zeros_like(noisy_data)
    model.eval()

    with torch.no_grad():
        for i, x1 in enumerate(val_loader):
            x1 = x1.to(DEVICE).float().unsqueeze(1)

            for _ in range(M):
                x1_in, mask = Subsampler.sample_img(x1, 1)
                denoised_res += model(x1_in).cpu().numpy().squeeze()

    denoised_res = denoised_res / M
    denoised_res = np.array(denoised_res, dtype=np.float32)

    # Remove padding if it was added
    if pad_h > 0 or pad_w > 0:
        orig_h, orig_w = original_shape
        denoised_res = denoised_res[:orig_h, :orig_w]
        if verbose:
            print(f"   Removed padding: {denoised_res.shape}")

    # Denormalize back to original range
    if image.dtype == np.uint8:
        denoised_res = (denoised_res * 255).clip(0, 255).astype(np.uint8)
    else:
        # Scale back to original image range
        denoised_res = denoised_res * (gray.max() - gray.min()) + gray.min()

    # Convert back to color if input was color
    if is_color:
        denoised_bgr = cv2.cvtColor(denoised_res, cv2.COLOR_GRAY2BGR)
    else:
        denoised_bgr = denoised_res

    # Prepare info dictionary
    info = {
        'method': 'Noise2SR',
        'epochs_trained': epoch,
        'final_loss': avg_loss,
        'device': str(DEVICE),
        'scale_factor': s,
        'inference_averages': M,
        'config': config
    }

    if verbose:
        print(f"\n" + "="*80)
        print(f"Noise2SR Denoising Complete")
        print(f"="*80)
        print(f"   Final training loss: {avg_loss:.6f}")
        print(f"   Output shape: {denoised_bgr.shape}")
        print(f"   Output range: [{denoised_bgr.min():.4f}, {denoised_bgr.max():.4f}]")
        print("="*80 + "\n")

    return denoised_bgr, info


def estimate_noise_parameters(image: np.ndarray) -> Tuple[float, float]:
    """
    Estimate Poisson-Gaussian noise parameters (a, b) from image.

    For real TEM images without ground truth, this provides rough estimates.

    Args:
        image: Input TEM image (grayscale)

    Returns:
        tuple: (a, b) noise parameters
            - a: Poisson noise coefficient
            - b: Gaussian noise standard deviation

    Note:
        These are rough estimates. For more accurate values, consider:
        - Using noise calibration images
        - Analyzing flat field regions
        - Manufacturer specifications
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Normalize to [0, 1]
    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray = (gray - gray.min()) / (gray.max() - gray.min())

    # Estimate noise from local variance
    # This is a simplified estimation
    # More sophisticated methods could be used

    # Divide into patches
    patch_size = 64
    h, w = gray.shape
    variances = []

    for y in range(0, h - patch_size, patch_size // 2):
        for x in range(0, w - patch_size, patch_size // 2):
            patch = gray[y:y+patch_size, x:x+patch_size]
            variances.append(np.var(patch))

    variances = np.array(variances)

    # Poisson noise: variance proportional to signal
    # Gaussian noise: constant variance
    # Simple estimation: use median variance
    median_var = np.median(variances)

    # Rough estimates (these are heuristic)
    a = 0.05  # Typical for TEM
    b = np.sqrt(median_var) * 0.1  # Rough estimate

    return a, b
