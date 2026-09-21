"""
Session State Management
========================
Manages Streamlit session state for the TEM Analysis UI.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
import numpy as np
from modules.runtime_config import resolve_noise2sr_config


CONFIG_SCHEMA_VERSION = 2


def initialize_session_state():
    """Initialize all session state variables with default values."""

    # Current workflow step
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = 0

    # Image data (supports multiple images)
    if 'uploaded_images' not in st.session_state:
        st.session_state['uploaded_images'] = []  # List of (name, image_array) tuples

    if 'current_image_index' not in st.session_state:
        st.session_state['current_image_index'] = 0

    # ROI configuration - per image
    # roi_configs stores ROI for each image: {image_name: (x1, y1, x2, y2)}
    if 'roi_configs' not in st.session_state:
        st.session_state['roi_configs'] = {}  # Dict: image_name -> (x1, y1, x2, y2)

    # Scale configuration - shared across all images (same experiment conditions)
    if 'scale_config' not in st.session_state:
        st.session_state['scale_config'] = {
            'scale_bar_points': None,  # ((x1, y1), (x2, y2))
            'scale_bar_length_px': 100,
            'scale_value': 100.0,
            'scale_unit': 'nm',
            'pixel_to_real': None
        }

    # Legacy roi_config for backward compatibility (will be migrated)
    if 'roi_config' not in st.session_state:
        st.session_state['roi_config'] = {
            'particle_roi': None,  # (x1, y1, x2, y2)
            'scale_text_roi': None,  # (x1, y1, x2, y2)
            'scale_bar_points': None,  # ((x1, y1), (x2, y2))
            'scale_value': 100.0,
            'scale_unit': 'nm'
        }

    # Preprocessing options
    if 'preprocessing_config' not in st.session_state:
        st.session_state['preprocessing_config'] = {
            'enable_bm3d': True,
            'bm3d_sigma': 40.0,
            'enable_noise2sr': True,
            'noise2sr_epochs': 1500,
            'enable_clahe': False,
            'clahe_clip_limit': 4.0,
            'clahe_tile_size': 64
        }

    # SAM configuration - Optimized for TEM nanoparticle detection
    if 'sam_config' not in st.session_state:
        st.session_state['sam_config'] = {
            'model_type': 'hiera_l',
            'points_per_side': 32,
            'points_per_batch': 256,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.80,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'crop_nms_thresh': 0.7,
            'box_nms_thresh': 0.7,
            'use_m2m': True,
            'small_particles': False,
            'gpu_quality': 'high'
        }

    # Analysis configuration
    if 'analysis_config' not in st.session_state:
        st.session_state['analysis_config'] = {
            'enable_size': True,
            'enable_distribution': True,
            'enable_spatial_uniformity': True,
            'enable_shape': True,
            # Size analysis options
            'enable_particle_area_filter': False,
            'min_particle_area': 10,
            'max_particle_area': 100000,
            # Shape options - using 2D preset (same as vision.py)
            'shape_preset': '2D',
            'shape_labels': ['Circle', 'Triangle', 'Quadrilateral', 'Hexagon', 'Irregular'],
            'clip_model': 'ViT-L/14@336px',
            'clip_batch_size': 64,
            'clip_temperature': None,
            'clip_confidence_threshold': None  # None means no threshold, always return highest probability
        }

    # Output configuration. These choices control which download actions are
    # shown; the complete ZIP always contains the full reproducibility bundle.
    if 'output_config' not in st.session_state:
        st.session_state['output_config'] = {
            'summary_csv': True,
            'particle_csv': True,
            'xlsx': True,
            'png': True,
            'zip': True,
        }

    # Analysis results (per image)
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = {}  # Dict: image_name -> results

    # Processing state
    if 'is_processing' not in st.session_state:
        st.session_state['is_processing'] = False

    if 'processing_progress' not in st.session_state:
        st.session_state['processing_progress'] = 0.0

    if 'processing_status' not in st.session_state:
        st.session_state['processing_status'] = ""


def get_current_image() -> tuple:
    """Get the currently selected image."""
    images = st.session_state.get('uploaded_images', [])
    idx = st.session_state.get('current_image_index', 0)

    if images and 0 <= idx < len(images):
        return images[idx]
    return None, None


def set_current_image(index: int):
    """Set the current image index."""
    images = st.session_state.get('uploaded_images', [])
    if 0 <= index < len(images):
        st.session_state['current_image_index'] = index


def add_image(name: str, image: np.ndarray):
    """Add an image to the uploaded images list."""
    if 'uploaded_images' not in st.session_state:
        st.session_state['uploaded_images'] = []
    st.session_state['uploaded_images'].append((name, image))


def remove_image(index: int):
    """Remove an image from the list."""
    images = st.session_state.get('uploaded_images', [])
    if 0 <= index < len(images):
        images.pop(index)
        # Adjust current index if needed
        if st.session_state['current_image_index'] >= len(images):
            st.session_state['current_image_index'] = max(0, len(images) - 1)


def clear_images():
    """Clear all uploaded images."""
    st.session_state['uploaded_images'] = []
    st.session_state['current_image_index'] = 0
    st.session_state['analysis_results'] = {}


def get_results_for_current_image() -> Dict[str, Any]:
    """Get analysis results for the current image."""
    name, _ = get_current_image()
    if name:
        return st.session_state.get('analysis_results', {}).get(name, {})
    return {}


def set_results_for_image(name: str, results: Dict[str, Any]):
    """Set analysis results for a specific image."""
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = {}
    st.session_state['analysis_results'][name] = results


def update_step(step: int):
    """Update the current workflow step."""
    st.session_state['current_step'] = step


def get_all_config() -> Dict[str, Any]:
    """Return a JSON-compatible snapshot of reproducibility settings."""
    config = {
        'schema_version': CONFIG_SCHEMA_VERSION,
        'roi': st.session_state.get('roi_config', {}),
        'roi_by_image': st.session_state.get('roi_configs', {}),
        'scale': st.session_state.get('scale_config', {}),
        'preprocessing': {
            **st.session_state.get('preprocessing_config', {}),
            'noise2sr': resolve_noise2sr_config(
                st.session_state.get('preprocessing_config', {}).get('noise2sr'),
                epochs=st.session_state.get('preprocessing_config', {}).get('noise2sr_epochs', 1500),
            ),
        },
        'sam': st.session_state.get('sam_config', {}),
        'analysis': st.session_state.get('analysis_config', {}),
        'export': st.session_state.get('output_config', {}),
    }
    return _to_json_compatible(deepcopy(config))


def _to_json_compatible(value):
    """Convert NumPy scalars and tuples recursively for JSON export."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    return value


def apply_all_config(config: Dict[str, Any]):
    """Validate and apply a configuration snapshot while retaining uploaded images."""
    if not isinstance(config, dict):
        raise ValueError("Configuration JSON must contain an object at the top level.")

    schema_version = config.get('schema_version', CONFIG_SCHEMA_VERSION)
    if schema_version not in (1, CONFIG_SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported configuration schema version: {schema_version}. "
            f"Expected 1 or {CONFIG_SCHEMA_VERSION}."
        )

    section_names = ('roi', 'roi_by_image', 'scale', 'preprocessing', 'sam', 'analysis', 'export')
    for section_name in section_names:
        if section_name in config and not isinstance(config[section_name], dict):
            raise ValueError(f"Configuration section '{section_name}' must be an object.")

    sam = config.get('sam', {})
    for key in ('pred_iou_thresh', 'stability_score_thresh'):
        if key in sam and not 0.0 <= float(sam[key]) <= 1.0:
            raise ValueError(f"sam.{key} must be between 0 and 1.")

    analysis = config.get('analysis', {})
    preprocessing = config.get('preprocessing', {})
    resolve_noise2sr_config(preprocessing.get('noise2sr'), epochs=preprocessing.get('noise2sr_epochs'))
    if float(analysis.get('min_particle_area', 1)) < 0:
        raise ValueError("analysis.min_particle_area must be non-negative.")
    if (
        'max_particle_area' in analysis
        and float(analysis['max_particle_area']) < float(analysis.get('min_particle_area', 0))
    ):
        raise ValueError("analysis.max_particle_area must not be smaller than min_particle_area.")
    if 'shape_labels' in analysis:
        labels = analysis['shape_labels']
        if not isinstance(labels, list) or len(labels) < 2 or not all(str(label).strip() for label in labels):
            raise ValueError("analysis.shape_labels must contain at least two non-empty labels.")

    runtime_keys = (
        'uploaded_images',
        'current_image_index',
        'current_step',
        'is_processing',
        'processing_progress',
        'processing_status',
    )
    runtime_state = {
        key: st.session_state[key]
        for key in runtime_keys
        if key in st.session_state
    }

    # Clearing widget state ensures imported values become the widget defaults on rerun.
    st.session_state.clear()
    initialize_session_state()
    st.session_state.update(runtime_state)
    st.session_state['analysis_results'] = {}

    state_sections = {
        'roi': 'roi_config',
        'roi_by_image': 'roi_configs',
        'scale': 'scale_config',
        'preprocessing': 'preprocessing_config',
        'sam': 'sam_config',
        'analysis': 'analysis_config',
        'export': 'output_config',
    }
    for config_key, state_key in state_sections.items():
        if config_key in config:
            merged = deepcopy(st.session_state.get(state_key, {}))
            merged.update(deepcopy(config[config_key]))
            st.session_state[state_key] = merged

    st.session_state['config_load_message'] = "Configuration loaded. Existing results were cleared."


# =====================================================
# Multiple Image ROI Management
# =====================================================

def get_roi_for_image(image_name: str):
    """Get ROI for a specific image."""
    return st.session_state.get('roi_configs', {}).get(image_name)


def set_roi_for_image(image_name: str, roi: tuple):
    """Set ROI for a specific image.

    Args:
        image_name: Name of the image
        roi: Tuple of (x1, y1, x2, y2)
    """
    if 'roi_configs' not in st.session_state:
        st.session_state['roi_configs'] = {}
    st.session_state['roi_configs'][image_name] = roi

    # Also update legacy roi_config for current image compatibility
    st.session_state['roi_config']['particle_roi'] = roi


def get_roi_for_current_image():
    """Get ROI for the currently selected image."""
    name, _ = get_current_image()
    if name:
        return get_roi_for_image(name)
    return None


def get_images_without_roi() -> List[str]:
    """Get list of image names that don't have ROI configured."""
    images = st.session_state.get('uploaded_images', [])
    roi_configs = st.session_state.get('roi_configs', {})

    missing = []
    for name, _ in images:
        if name not in roi_configs or roi_configs[name] is None:
            missing.append(name)
    return missing


def get_images_with_roi() -> List[str]:
    """Get list of image names that have ROI configured."""
    images = st.session_state.get('uploaded_images', [])
    roi_configs = st.session_state.get('roi_configs', {})

    configured = []
    for name, _ in images:
        if name in roi_configs and roi_configs[name] is not None:
            configured.append(name)
    return configured


def all_images_have_roi() -> bool:
    """Check if all uploaded images have ROI configured."""
    return len(get_images_without_roi()) == 0


def get_scale_config() -> Dict[str, Any]:
    """Get the shared scale configuration."""
    return st.session_state.get('scale_config', {})


def set_scale_config(scale_value: float, scale_unit: str, scale_bar_length_px: int,
                     scale_bar_points: tuple = None):
    """Set the shared scale configuration.

    Args:
        scale_value: Physical value (e.g., 100 for "100 nm")
        scale_unit: Unit string (e.g., "nm")
        scale_bar_length_px: Length of scale bar in pixels
        scale_bar_points: Optional tuple of ((x1,y1), (x2,y2)) endpoints
    """
    if 'scale_config' not in st.session_state:
        st.session_state['scale_config'] = {}

    pixel_to_real = scale_value / scale_bar_length_px if scale_bar_length_px > 0 else 1.0

    st.session_state['scale_config'].update({
        'scale_value': scale_value,
        'scale_unit': scale_unit,
        'scale_bar_length_px': scale_bar_length_px,
        'scale_bar_points': scale_bar_points,
        'pixel_to_real': pixel_to_real
    })

    # Also update legacy roi_config for compatibility
    st.session_state['roi_config']['scale_value'] = scale_value
    st.session_state['roi_config']['scale_unit'] = scale_unit
    st.session_state['roi_config']['scale_bar_length_px'] = scale_bar_length_px
    st.session_state['roi_config']['pixel_to_real'] = pixel_to_real
    st.session_state['roi_config']['scale_bar_points'] = scale_bar_points
