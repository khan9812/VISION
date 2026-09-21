"""
Preprocessing Options Component
===============================
Configuration UI for image preprocessing (BM3D, Noise2SR, CLAHE).
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import sys
import hashlib

parent_dir = Path(__file__).parent.parent.parent
ui_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(ui_dir))

from utils.session_state import get_current_image, update_step, initialize_session_state


def _get_image_hash(image: np.ndarray) -> str:
    """Get hash of image for caching."""
    return hashlib.md5(image.tobytes()).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def _apply_bm3d_cached(image_bytes: bytes, shape: tuple, sigma: float = 40.0) -> np.ndarray:
    """Apply BM3D denoising with caching to avoid repeated computation."""
    import gc
    import os
    
    # Set thread count to 1 to avoid threading issues with BM3D
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Reconstruct image from bytes
    gray = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
    
    try:
        import bm3d
        # Normalize to [0,1] for bm3d
        gray_normalized = gray.astype(np.float32) / 255.0
        sigma_normalized = sigma / 255.0
        
        # Apply BM3D (single-threaded to avoid threading issues)
        denoised = bm3d.bm3d(gray_normalized, sigma_psd=sigma_normalized, 
                            stage_arg=bm3d.BM3DStages.ALL_STAGES)
        result = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
        
        # Force garbage collection to clean up BM3D resources
        gc.collect()
        
        return result
    except Exception as e:
        print(f"BM3D error: {e}")
        gc.collect()
        return gray  # Return original if BM3D fails


def render_preprocessing_options():
    """Render preprocessing configuration options."""

    # Ensure session state is initialized
    if 'preprocessing_config' not in st.session_state:
        initialize_session_state()

    st.markdown("### ⚙️ Preprocessing Configuration")

    # Check if image is loaded
    name, image = get_current_image()

    if image is None:
        st.warning("⚠️ Please upload an image first.")
        return

    st.markdown(f"**Current Image:** {name}")

    # Get preprocessing config
    config = st.session_state.get('preprocessing_config', {})

    # Preprocessing options in one row
    st.markdown("#### 🔧 Preprocessing Methods")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        enable_bm3d = st.checkbox(
            "🔇 BM3D Denoising",
            value=config.get('enable_bm3d', True),
            help="Block-matching 3D denoising (σ=40, recommended for TEM)"
        )
        config['enable_bm3d'] = enable_bm3d
        config['bm3d_sigma'] = 40.0  # Experimental setting

    with col2:
        enable_noise2sr = st.checkbox(
            "🧠 Noise2SR",
            value=config.get('enable_noise2sr', True),
            help="Deep learning denoising (slow, for severe noise)"
        )
        config['enable_noise2sr'] = enable_noise2sr
        if enable_noise2sr:
            noise2sr_epochs = st.number_input(
                "Noise2SR Epochs",
                min_value=100,
                max_value=5000,
                value=int(config.get('noise2sr_epochs', 1500)),
                step=100,
                help="Training epochs for per-image Noise2SR optimization."
            )
            config['noise2sr_epochs'] = int(noise2sr_epochs)
        else:
            config.setdefault('noise2sr_epochs', 1500)

    with col3:
        enable_clahe = st.checkbox(
            "🔆 CLAHE",
            value=config.get('enable_clahe', False),
            help="Contrast enhancement"
        )
        config['enable_clahe'] = enable_clahe
        config['clahe_clip_limit'] = 4.0  # Experimental setting
        config['clahe_tile_size'] = 64  # Experimental setting

    if enable_noise2sr:
        st.warning("⚠️ Noise2SR requires GPU and takes 2-5 minutes per image")

    # Store config
    st.session_state['preprocessing_config'] = config

    st.divider()

    # Show preprocessing preview with before/after
    st.markdown("#### 📊 Preprocessing Preview")

    # Get ROI image
    roi_config = st.session_state.get('roi_config', {})
    particle_roi = roi_config.get('particle_roi')

    if particle_roi:
        x1, y1, x2, y2 = particle_roi
        roi_image = image[y1:y2, x1:x2].copy()
    else:
        roi_image = image.copy()

    # Convert to grayscale for display
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_image

    # Keep the quick preview responsive. Noise2SR is applied during analysis,
    # where its per-image training output is persisted by the pipeline cache.
    preprocessed = gray.copy()
    if enable_bm3d:
        # Use cached BM3D to avoid repeated computation and threading issues
        preprocessed = _apply_bm3d_cached(
            gray.tobytes(), gray.shape, sigma=float(config['bm3d_sigma'])
        )

    if enable_clahe:
        tile_size = int(config['clahe_tile_size'])
        clahe = cv2.createCLAHE(
            clipLimit=float(config['clahe_clip_limit']),
            tileGridSize=(tile_size, tile_size),
        )
        preprocessed = clahe.apply(preprocessed)

    if enable_noise2sr:
        st.info(
            "The quick preview omits Noise2SR. The analysis run applies the full "
            "BM3D + Noise2SR pipeline with the configured epoch count."
        )

    # Show before/after comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original ROI:**")
        st.image(gray, use_column_width=True, clamp=True)
        st.caption(f"Mean: {gray.mean():.1f}, Std: {gray.std():.1f}")

    with col2:
        st.markdown("**Preprocessed Preview:**")
        st.image(preprocessed, use_column_width=True, clamp=True)
        st.caption(f"Mean: {preprocessed.mean():.1f}, Std: {preprocessed.std():.1f}")

    st.divider()

    # SAM Configuration
    st.markdown("#### 🤖 SAM 2.1 Configuration")

    sam_config = st.session_state.get('sam_config', {})

    col1, col2 = st.columns(2)

    with col1:
        small_particles = st.checkbox(
            "Small Particles Mode",
            value=sam_config.get('small_particles', False),
            help="Enable for particles < 10nm. Uses finer grid (slower but more accurate)."
        )
        sam_config['small_particles'] = small_particles

        if small_particles:
            sam_config['points_per_side'] = 64
            st.info("Points per side: 64 (fine grid)")
        else:
            sam_config['points_per_side'] = 32
            st.info("Points per side: 32 (standard grid)")

    with col2:
        st.info("Points per batch: 256 (fixed)")
        sam_config['points_per_batch'] = 256

    # Fixed SAM parameters (except thresholds and small-particle grid density)
    sam_config['model_type'] = 'hiera_l'
    sam_config.setdefault('pred_iou_thresh', 0.95)
    sam_config.setdefault('stability_score_thresh', 0.80)
    sam_config['crop_n_layers'] = 1
    sam_config['crop_n_points_downscale_factor'] = 2
    sam_config['crop_nms_thresh'] = 0.7
    sam_config['box_nms_thresh'] = 0.7
    sam_config['use_m2m'] = True

    # Advanced SAM settings
    with st.expander("🔧 Advanced SAM Settings", expanded=False):
        st.markdown("`model_type`: `hiera_l` (fixed)")
        st.markdown("`points_per_batch`: `256` (fixed)")
        pred_iou_thresh = st.number_input(
            "Predicted IoU threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(sam_config.get('pred_iou_thresh', 0.95)),
            step=0.05,
            format="%.2f",
            help="Empirical default: 0.95",
        )
        stability_score_thresh = st.number_input(
            "Stability score threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(sam_config.get('stability_score_thresh', 0.80)),
            step=0.05,
            format="%.2f",
            help="Empirical default: 0.80",
        )
        sam_config['pred_iou_thresh'] = float(pred_iou_thresh)
        sam_config['stability_score_thresh'] = float(stability_score_thresh)
        st.markdown("`crop_n_layers`: `1` (fixed)")
        st.markdown("`use_m2m`: `True` (fixed)")

    st.session_state['sam_config'] = sam_config

    # Navigation
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Back to ROI", use_container_width=True):
            update_step(1)
            st.rerun()

    with col2:
        if st.button("Continue to Analysis Config ➡️", type="primary", use_container_width=True):
            update_step(3)
            st.rerun()
