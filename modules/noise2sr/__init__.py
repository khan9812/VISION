"""
Noise2SR Module - Zero-Shot Image Denoising for TEM

Wrapper for Noise2SR (Tian et al., 2024) zero-shot denoising method.

Reference:
    Tian, X., et al. (2024). Zero-Shot Image Denoising for High-Resolution
    Electron Microscopy. IEEE Transactions on Computational Imaging, 10, 1462-1475.

GitHub: https://github.com/MeijiTian/ZS-Denoiser-HREM
"""

from .noise2sr_wrapper import denoise_image_noise2sr

__all__ = ['denoise_image_noise2sr']
