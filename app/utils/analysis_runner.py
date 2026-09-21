"""
Analysis Runner Utility
=======================
Runs the TEM particle analysis pipeline using existing modules.
"""

import numpy as np
import cv2
import streamlit as st
from pathlib import Path
import sys
import io
import matplotlib.pyplot as plt
import os
import hashlib

# Add parent directory for module imports
parent_dir = Path(__file__).parent.parent.parent
ui_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(ui_dir))

# Import session state helpers for per-image ROI and shared scale config
from utils.session_state import get_roi_for_image, get_scale_config
from modules.pipeline_cache import load_pipeline_cache, save_pipeline_cache
from modules.sam2_utils import SAM_POSTPROCESSING_VERSION
from modules.runtime_config import resolve_noise2sr_config
from modules.scientific_plotting import (
    create_size_distribution_figure,
    create_spatial_distribution_figure,
    plot_shape_composition,
    plot_size_distribution,
    plot_spatial_distribution,
    summarize_spatial_distribution,
)
from utils.provenance import collect_execution_provenance

# Import analysis modules
try:
    from modules.preprocessing import (
        apply_clahe,
        extract_boundary_pixels_dict,
        preprocess_with_config,
    )
    from modules.size_analysis import calculate_size_metrics, create_size_histogram
    from modules.distribution_analysis import (
        compute_unified_voronoi_areas,
        create_voronoi_overlay,
        calculate_distribution_metrics,
        create_distribution_violin_plot,
    )
    from modules.visualization import (
        find_optimal_alpha,
        create_integrated_dashboard
    )
    from modules.shape_analysis import (
        classify_shapes_with_clip,
        calculate_shape_metrics,
        create_shape_pie_chart,
        create_shape_overlay,
        generate_shape_colors
    )
    MODULES_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)


def get_missing_modules_message():
    """Generate user-friendly message about missing modules."""
    messages = []
    
    # Check individual modules
    try:
        import torch
    except ImportError:
        messages.append("• **PyTorch**: `pip install torch torchvision`")
    
    try:
        import clip
    except ImportError:
        messages.append("• **CLIP**: `pip install git+https://github.com/openai/CLIP.git`")
    
    try:
        import bm3d
    except ImportError:
        messages.append("• **BM3D**: `pip install bm3d`")
    
    if messages:
        return "**Missing packages:**\n" + "\n".join(messages)
    return None

# Reproducibility
SEED = 42
CACHE_ROOT_DIR = str(parent_dir / "cache")


def _fig_to_image(fig):
    """Convert matplotlib figure to numpy image array."""
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def _image_hash(image: np.ndarray) -> str:
    """Create a stable content hash for cache keys."""
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def filter_masks_by_area(masks, min_particle_area=None, max_particle_area=None):
    """Filter SAM mask dictionaries by segmentation pixel count."""
    filtered = []
    for mask_dict in masks:
        area = int(np.count_nonzero(mask_dict['segmentation']))
        if min_particle_area is not None and area < float(min_particle_area):
            continue
        if max_particle_area is not None and area > float(max_particle_area):
            continue
        mask_dict['area'] = area
        filtered.append(mask_dict)
    return filtered


def run_analysis(image: np.ndarray, image_name: str) -> dict:
    """
    Run the complete analysis pipeline on an image.

    Args:
        image: Input BGR image
        image_name: Name of the image file

    Returns:
        Dictionary containing all analysis results
    """

    results = {
        'image_name': image_name,
        'error': None,
        'error_details': None
    }

    if not MODULES_AVAILABLE:
        missing_msg = get_missing_modules_message()
        results['error'] = (
            "Required VISION analysis modules are unavailable. "
            "No surrogate OpenCV measurements were generated."
        )
        results['error_details'] = IMPORT_ERROR or missing_msg
        results['execution_provenance'] = collect_execution_provenance(
            preprocessing_config=st.session_state.get('preprocessing_config', {}),
            sam_config=st.session_state.get('sam_config', {}),
            analysis_config=st.session_state.get('analysis_config', {}),
            pipeline_cache_status="not_checked_missing_modules",
        )
        return results

    try:
        # Get configurations from session state
        preprocess_config = st.session_state.get('preprocessing_config', {})
        sam_config = st.session_state.get('sam_config', {})
        analysis_config = st.session_state.get('analysis_config', {})

        # Extract ROI for this specific image (per-image ROI)
        particle_roi = get_roi_for_image(image_name)
        if particle_roi:
            x1, y1, x2, y2 = particle_roi
            roi_image = image[y1:y2, x1:x2].copy()
        else:
            roi_image = image.copy()

        # Get scale conversion (shared across all images)
        scale_cfg = get_scale_config()
        pixel_to_real = scale_cfg.get('pixel_to_real') or 1.0
        scale_unit = scale_cfg.get('scale_unit', 'px') if scale_cfg.get('pixel_to_real') else 'px'
        resolved_noise2sr = resolve_noise2sr_config(
            preprocess_config.get('noise2sr'),
            epochs=preprocess_config.get('noise2sr_epochs', 1500),
            image_shape=roi_image.shape,
        )

        # Persistent cache (per-code, per-image-content, per-config)
        force_cache_refresh = (
            bool(st.session_state.get("force_cache_refresh", False))
            or os.environ.get("FORCE_CACHE_REFRESH", "0") == "1"
        )
        cache_key = f"{image_name}_{_image_hash(roi_image)}"
        cache_options = {
            "preprocess_config": preprocess_config,
            "resolved_noise2sr": resolved_noise2sr,
            "sam_config": sam_config,
            "analysis_config": analysis_config,
            "particle_roi": particle_roi,
            "scale_cfg": scale_cfg,
            "sam_postprocessing_version": SAM_POSTPROCESSING_VERSION,
            "cache_version": 8,
        }
        cache_status = "forced_refresh" if force_cache_refresh else "miss"
        if not force_cache_refresh:
            cached_payload = load_pipeline_cache(
                code_name=Path(__file__).stem,
                image_path=cache_key,
                options=cache_options,
                root_dir=CACHE_ROOT_DIR,
            )
            if isinstance(cached_payload, dict) and "results" in cached_payload:
                st.text("Loaded analysis from cache")
                cached_results = cached_payload["results"]
                cached_results['execution_provenance'] = collect_execution_provenance(
                    preprocessing_config=preprocess_config,
                    sam_config=sam_config,
                    analysis_config=analysis_config,
                    preprocessing_info=cached_results.get('preprocessing_info', {}),
                    pipeline_cache_status="hit",
                    pipeline_cache_key=cache_key,
                )
                return cached_results

        # Step 1: Preprocessing
        st.text("Preprocessing...")

        processed, preprocessing_info = preprocess_with_config(
            roi_image,
            sigma_psd=preprocess_config.get('bm3d_sigma', 40.0),
            bm3d_enabled=preprocess_config.get('enable_bm3d', True),
            noise2sr_enabled=preprocess_config.get('enable_noise2sr', True),
            clahe_enabled=False,
            verbose=True,
            noise2sr_epochs=preprocess_config.get('noise2sr_epochs', 1500),
            noise2sr_settings=resolved_noise2sr,
        )

        if preprocess_config.get('enable_clahe', False):
            clip_limit = preprocess_config.get('clahe_clip_limit', 4.0)
            tile_size = preprocess_config.get('clahe_tile_size', 64)
            processed = apply_clahe(processed, clip_limit, (tile_size, tile_size))
            preprocessing_info['clahe_applied'] = True
            preprocessing_info['clahe_params'] = {
                'clip_limit': float(clip_limit),
                'tile_grid_size': [int(tile_size), int(tile_size)],
            }

        requested_steps = {
            'BM3D': (
                bool(preprocess_config.get('enable_bm3d', True)),
                bool(preprocessing_info.get('bm3d_applied', False)),
            ),
            'Noise2SR': (
                bool(preprocess_config.get('enable_noise2sr', True)),
                bool(preprocessing_info.get('noise2sr_applied', False)),
            ),
            'CLAHE': (
                bool(preprocess_config.get('enable_clahe', False)),
                bool(preprocessing_info.get('clahe_applied', False)),
            ),
        }
        unavailable_steps = [
            name for name, (requested, applied) in requested_steps.items()
            if requested and not applied
        ]
        if unavailable_steps:
            raise RuntimeError(
                "Requested preprocessing step(s) were not applied: "
                + ", ".join(unavailable_steps)
            )
        results['preprocessing_info'] = preprocessing_info
        results['execution_provenance'] = collect_execution_provenance(
            preprocessing_config=preprocess_config,
            sam_config=sam_config,
            analysis_config=analysis_config,
            preprocessing_info=preprocessing_info,
            pipeline_cache_status=cache_status,
            pipeline_cache_key=cache_key,
        )

        # Step 2: Segmentation (SAM)
        st.text("Running segmentation...")

        masks, centroids, boundary_pixels = run_sam_segmentation(
            processed,
            sam_config,
            enable_area_filter=analysis_config.get('enable_particle_area_filter', False),
            min_particle_area=analysis_config.get('min_particle_area'),
            max_particle_area=analysis_config.get('max_particle_area'),
        )

        if len(masks) == 0:
            results['error'] = "No particles detected"
            return results

        results['particle_count'] = len(masks)
        results['sam_config'] = dict(sam_config)
        results['sam_postprocessing_version'] = SAM_POSTPROCESSING_VERSION
        results['centroids'] = centroids
        results['filtered_masks'] = masks  # Store filtered_masks (list of dicts)
        results['boundary_pixels'] = boundary_pixels

        # Create segmentation overlay (masks are dicts with 'segmentation' key)
        overlay = create_segmentation_overlay(roi_image, masks)
        results['segmentation_overlay'] = overlay

        # Step 3: Projected particle-area analysis
        if analysis_config.get('enable_size', True):
            st.text("Running projected particle-area analysis...")
            results['size'] = run_size_analysis(
                masks, boundary_pixels, pixel_to_real, scale_unit,
                analysis_config
            )

        # Step 4: PF-SUI analysis using the shared centroid convex-hull domain.
        spatial_uniformity_enabled = analysis_config.get(
            'enable_distribution',
            analysis_config.get('enable_spatial_uniformity', True)
        )
        if spatial_uniformity_enabled:
            st.text("Running PF-SUI analysis...")
            dist_result = run_distribution_analysis(
                centroids, boundary_pixels, roi_image.shape[:2],
                pixel_to_real, scale_unit, analysis_config
            )
            results['distribution'] = dist_result
            if dist_result.get('error'):
                raise RuntimeError(f"PF-SUI analysis failed: {dist_result['error']}")
            
            # Create PF-SUI visualization images.
            if dist_result.get('status') == 'complete':
                try:
                    # Raincloud plot: half violin, observed values, IQR, and median 95% CI.
                    if dist_result.get('voronoi_areas_real'):
                        spatial_fig = create_distribution_violin_plot(
                            dist_result['voronoi_areas_real'],
                            scale=scale_unit,
                            figsize=(8, 5),
                        )
                        results['distribution']['spatial_distribution_image'] = _fig_to_image(spatial_fig)
                        plt.close(spatial_fig)

                    # Voronoi overlay
                    if dist_result.get('voronoi_areas_real_dict'):
                        voronoi_fig, voronoi_img = create_voronoi_overlay(
                            roi_image,
                            boundary_pixels,
                            dist_result['voronoi_areas_real_dict'],
                            scale=scale_unit,
                            concave_vertex=dist_result.get('concave_vertex'),
                            inside_centroids=dist_result.get('inside_centroids'),
                        )
                        results['distribution']['voronoi_image'] = _fig_to_image(voronoi_fig)
                        plt.close(voronoi_fig)
                    
                except Exception as e:
                    import traceback
                    print(f"PF-SUI visualization error: {e}")
                    traceback.print_exc()

        # Step 5: Projected-morphology analysis using CLIP.
        if analysis_config.get('enable_shape', True):
            st.text("Running projected-morphology analysis...")
            shape_result = run_shape_analysis(
                processed, masks, centroids, boundary_pixels,
                analysis_config
            )
            results['shape'] = shape_result
            if shape_result.get('error'):
                raise RuntimeError(
                    f"Projected-morphology analysis failed: {shape_result['error']}"
                )
            
            # Create projected-morphology visualization images.
            if 'error' not in shape_result and shape_result.get('shape_counts'):
                try:
                    # Pie chart
                    if shape_result.get('color_map'):
                        pie_fig = create_shape_pie_chart(
                            shape_result['shape_counts'],
                            shape_result['color_map']
                        )
                        results['shape']['pie_chart_image'] = _fig_to_image(pie_fig)
                        plt.close(pie_fig)
                    
                    # Shape overlay - masks are already dicts, pass directly
                    if shape_result.get('shapes'):
                        overlay_fig = create_shape_overlay(
                            roi_image, masks, shape_result['shapes'],
                            shape_result.get('color_map', {})
                        )
                        results['shape']['overlay_image'] = _fig_to_image(overlay_fig)
                        plt.close(overlay_fig)
                except Exception as e:
                    import traceback
                    print(f"Projected-morphology visualization error: {e}")
                    traceback.print_exc()

        # Create integrated dashboard
        st.text("Creating dashboard...")
        results['dashboard_image'] = create_dashboard_image(results, roi_image)

        # Save successful analysis cache
        try:
            save_pipeline_cache(
                code_name=Path(__file__).stem,
                image_path=cache_key,
                payload={"results": results},
                options=cache_options,
                root_dir=CACHE_ROOT_DIR,
            )
        except Exception:
            pass

    except Exception as e:
        error_msg = str(e)
        results['error'] = error_msg
        import traceback
        results['traceback'] = traceback.format_exc()
        if 'execution_provenance' not in results:
            results['execution_provenance'] = collect_execution_provenance(
                preprocessing_config=st.session_state.get('preprocessing_config', {}),
                sam_config=st.session_state.get('sam_config', {}),
                analysis_config=st.session_state.get('analysis_config', {}),
                preprocessing_info=results.get('preprocessing_info', {}),
                pipeline_cache_status="failed_before_cache_save",
            )

        # Check for CUDA memory error and provide helpful message
        if "CUDA" in error_msg or "out of memory" in error_msg.lower():
            results['error_type'] = 'cuda_memory'
            results['error_suggestion'] = "Lower the GPU Quality setting (high → medium → low) in Preprocessing tab."

    return results


def run_simplified_analysis(image: np.ndarray, image_name: str) -> dict:
    """
    Run simplified analysis when full modules are not available.
    Uses basic OpenCV operations for demonstration.
    """

    results = {
        'image_name': image_name,
        'error': None,
        'note': 'Running simplified analysis (full modules not loaded)'
    }

    try:
        # Get ROI for this specific image (per-image ROI)
        particle_roi = get_roi_for_image(image_name)

        if particle_roi:
            x1, y1, x2, y2 = particle_roi
            roi_image = image[y1:y2, x1:x2].copy()
        else:
            roi_image = image.copy()

        # Use the same physical scale configuration as the full pipeline.
        scale_cfg = get_scale_config()
        pixel_to_real = scale_cfg.get('pixel_to_real', 1.0)
        scale_unit = scale_cfg.get('scale_unit', 'px')

        # Convert to grayscale
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()

        # Simple preprocessing
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        min_area = st.session_state.get('analysis_config', {}).get('min_particle_area', 10)
        max_area = st.session_state.get('analysis_config', {}).get('max_particle_area', 100000)

        valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

        results['particle_count'] = len(valid_contours)

        # Calculate centroids
        centroids = []
        areas = []

        for contour in valid_contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
                areas.append(cv2.contourArea(contour))

        results['centroids'] = centroids

        # Create overlay
        overlay = roi_image.copy()
        cv2.drawContours(overlay, valid_contours, -1, (0, 255, 0), 2)
        for cx, cy in centroids:
            cv2.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)

        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        results['segmentation_overlay'] = overlay_rgb

        # Size analysis
        if areas:
            areas_real = [a * (pixel_to_real ** 2) for a in areas]

            results['size'] = {
                'count': len(areas),
                'areas': areas,
                'areas_real': areas_real,
                'scale_unit': scale_unit,
                'unit': f"{scale_unit}^2",
                'mean_area': np.mean(areas_real),
                'median_area': np.median(areas_real),
                'std_area': np.std(areas_real),
                'cv': (np.std(areas_real) / np.mean(areas_real) * 100) if np.mean(areas_real) > 0 else 0,
                'iqr': np.percentile(areas_real, 75) - np.percentile(areas_real, 25),
                'skewness': 0,  # Simplified
                'kurtosis': 0   # Simplified
            }

            # Create the same histogram/KDE/median chart used by the full pipeline.
            results['size']['histogram_image'] = create_histogram_image(
                areas_real,
                scale=scale_unit,
            )

        # Spatial uniformity fallback: nearest-neighbor distances have the same
        # raincloud presentation, but are explicitly labeled as a fallback.
        if len(centroids) > 3:
            centroid_array = np.asarray(centroids, dtype=float)
            deltas = centroid_array[:, None, :] - centroid_array[None, :, :]
            distances = np.sqrt(np.sum(deltas**2, axis=2))
            np.fill_diagonal(distances, np.inf)
            nearest_neighbor = np.min(distances, axis=1) * pixel_to_real
            summary = summarize_spatial_distribution(nearest_neighbor)

            spatial_fig = create_spatial_distribution_figure(
                nearest_neighbor,
                scale=scale_unit,
                figsize=(8, 5),
                title='Nearest-neighbor distribution (fallback)',
                value_label='Nearest-neighbor distance',
                unit_suffix='',
            )
            results['distribution'] = {
                'n_clusters': 1,
                'noise_points': 0,
                'scale_unit': scale_unit,
                'spatial_measure_label': 'Nearest-neighbor distance (fallback)',
                'spatial_metric_label': 'Nearest-neighbor Distance',
                'spatial_measure_unit': scale_unit,
                'spatial_values': nearest_neighbor.tolist(),
                'mean_spatial_value': float(np.mean(nearest_neighbor)),
                'median_spatial_value': summary['median'],
                'median_spatial_ci_low': summary['median_ci_low'],
                'median_spatial_ci_high': summary['median_ci_high'],
                'spatial_cv': float(np.std(nearest_neighbor, ddof=1) / np.mean(nearest_neighbor) * 100)
                if len(nearest_neighbor) > 1 and np.mean(nearest_neighbor) > 0 else 0.0,
                'max_l': 0,
                'max_l_radius': 0,
                'spatial_pattern': 'Unknown',
                'cluster_fraction': 1.0,
                'spatial_distribution_image': _fig_to_image(spatial_fig),
            }
            plt.close(spatial_fig)

        # Shape (simplified - not available without CLIP)
        results['shape'] = {
            'dominant_shape': 'Unknown',
            'dominant_percentage': 100.0,
            'entropy': 0,
            'avg_confidence': 0,
            'shape_counts': {'Unknown': len(valid_contours)}
        }

    except Exception as e:
        results['error'] = str(e)

    return results


def run_sam_segmentation(
    image: np.ndarray,
    sam_config: dict,
    enable_area_filter=False,
    min_particle_area=None,
    max_particle_area=None,
):
    """
    Run SAM 2.1 with the post-processing shared by experiments and validation.
    
    Returns:
        filtered_masks: List of mask dicts (with 'segmentation', 'bbox', etc.)
        filtered_centroids: List of (x, y) tuples
        boundary_pixels: Dict mapping (cx, cy) -> list of boundary points
    """
    try:
        import torch
        from modules.sam2_utils import (
            filter_background_masks,
            filter_overlapping_masks_by_centroid,
            generate_masks,
            get_largest_contour_centroid,
            load_sam_model,
        )

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_sam_model(model_type="hiera_l", device=device)

        # Generate raw masks; shared post-processing is applied exactly once below.
        masks = generate_masks(
            model, image,
            points_per_side=sam_config.get('points_per_side', 32),
            points_per_batch=sam_config.get('points_per_batch', 256),
            pred_iou_thresh=sam_config.get('pred_iou_thresh', 0.95),
            stability_score_thresh=sam_config.get('stability_score_thresh', 0.80),
            crop_n_layers=sam_config.get('crop_n_layers', 1),
            crop_n_points_downscale_factor=sam_config.get('crop_n_points_downscale_factor', 2),
            crop_nms_thresh=sam_config.get('crop_nms_thresh', 0.7),
            box_nms_thresh=sam_config.get('box_nms_thresh', 0.7),
            use_m2m=sam_config.get('use_m2m', True),
            filter_background=False,
        )
        
        print(f"[SAM] Generated {len(masks)} raw masks")

        # Step 1: Remove full-image background masks.
        filtered_masks = filter_background_masks(masks, image.shape[:2])
        
        print(f"[SAM] After background removal: {len(filtered_masks)} masks")

        # Optional operational adaptation. Disabled by default to match validation.
        if enable_area_filter:
            filtered_masks = filter_masks_by_area(
                filtered_masks,
                min_particle_area=min_particle_area,
                max_particle_area=max_particle_area,
            )
            print(
                "[SAM] After optional area filtering "
                f"({min_particle_area or 0} to {max_particle_area or 'unbounded'} px2): "
                f"{len(filtered_masks)} masks"
            )

        # Step 2: Remove larger centroid-contained duplicate masks.
        filtered_masks = filter_overlapping_masks_by_centroid(filtered_masks)
        valid_pairs = [
            (mask_dict, get_largest_contour_centroid(mask_dict['segmentation']))
            for mask_dict in filtered_masks
        ]
        filtered_masks = [mask_dict for mask_dict, centroid in valid_pairs if centroid is not None]
        filtered_centroids = [centroid for _, centroid in valid_pairs if centroid is not None]
        print(f"[SAM] After containment filtering: {len(filtered_centroids)} particles")

        # Step 3: Extract boundary pixels.
        boundary_pixels = extract_boundary_pixels_dict(filtered_masks)
        print(f"[SAM] Extracted boundary pixels for {len(boundary_pixels)} particles")

        return filtered_masks, filtered_centroids, boundary_pixels

    except Exception as e:
        import traceback
        print(f"SAM error: {e}")
        traceback.print_exc()
        raise RuntimeError(f"SAM 2.1 segmentation failed: {e}") from e


def run_size_analysis(masks, boundary_pixels, pixel_to_real, scale_unit, config):
    """Run projected particle-area analysis on detected particles.
    
    Args:
        masks: List of mask dicts with 'segmentation' key
    """
    areas = []
    for mask_dict in masks:
        # Extract segmentation from dict
        if isinstance(mask_dict, dict):
            mask = mask_dict['segmentation']
        else:
            mask = mask_dict
        area_px = np.sum(mask)
        areas.append(area_px)

    areas_real = [a * (pixel_to_real ** 2) for a in areas]

    if areas_real:
        metrics = calculate_size_metrics(areas_real, scale=scale_unit)
    else:
        metrics = {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'cv': 0.0,
            'iqr': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
        }

    area_map_real = {
        centroid: areas_real[index]
        for index, centroid in enumerate(boundary_pixels)
        if index < len(areas_real)
    }
    result = {
        'count': len(areas),
        'areas': areas,
        'areas_real': areas_real,
        'area_map_real': area_map_real,
        'boundary_pixels': boundary_pixels,
        'metrics': metrics,
        'scale_unit': scale_unit,
        'unit': f"{scale_unit}²",
        'mean_area': metrics.get('mean', 0.0),
        'median_area': metrics.get('median', 0.0),
        'std_area': metrics.get('std', 0.0),
        'cv': metrics.get('cv', 0.0) * 100.0,
        'iqr': metrics.get('iqr', 0.0),
        'skewness': metrics.get('skewness', 0.0),
        'kurtosis': metrics.get('kurtosis', 0.0),
    }

    if areas_real:
        histogram_fig = create_size_histogram(areas_real, scale=scale_unit, figsize=(8, 5))
        result['histogram_image'] = _fig_to_image(histogram_fig)
        plt.close(histogram_fig)

    return result


def run_distribution_analysis(centroids, boundary_pixels, image_shape, pixel_to_real, scale_unit, config):
    """Report PF-SUI only for a valid hull and at least two interior regions."""
    result = {
        'status': 'unavailable', 'reason': None,
        'boundary_pixels': boundary_pixels, 'scale_unit': scale_unit,
        'unit': f"{scale_unit}²",
        'boundary_method': 'convex_hull_with_5px_inward_centroid_filter',
        'spatial_measure_label': 'Particle-boundary-based Voronoi cell area',
        'spatial_metric_label': 'Particle-boundary-based Voronoi area',
        'spatial_measure_unit': f"{scale_unit}²",
        'inside_centroids': [], 'concave_vertex': [],
        'voronoi_areas': {}, 'voronoi_areas_dict': {}, 'voronoi_areas_list': [],
        'voronoi_areas_real_dict': {}, 'voronoi_areas_real': [], 'spatial_values': [],
        'n_pf': 0,
    }
    for key in ('mean_voronoi_area', 'std_voronoi_area', 'median_voronoi_area',
                'median_voronoi_ci_low', 'median_voronoi_ci_high', 'voronoi_cv',
                'spatial_uniformity_index', 'sui', 'mean_spatial_value',
                'median_spatial_value', 'median_spatial_ci_low', 'median_spatial_ci_high', 'spatial_cv'):
        result[key] = None
    if len(centroids) < 3:
        result['reason'] = 'A convex analysis hull requires at least three particles.'
        return result
    try:
        inside, _, hull, alpha = find_optimal_alpha(centroids)
        result.update(inside_centroids=inside, concave_vertex=hull, concave_alpha=alpha)
        if len(hull) < 3:
            result['reason'] = 'Particle centroids do not define a nonzero-area convex hull.'
            return result
        if not inside:
            result['reason'] = 'No particle centroids remain inside the 5-pixel inward hull.'
            return result
        areas, vor, regions = compute_unified_voronoi_areas(boundary_pixels, hull, inside)
        calibrated = {key: area * pixel_to_real ** 2 for key, area in areas.items()}
        values = list(calibrated.values())
        result.update(voronoi_areas=areas, voronoi_areas_dict=areas,
                      voronoi_areas_list=list(areas.values()), vor_obj=vor,
                      unified_regions=regions, voronoi_areas_real_dict=calibrated,
                      voronoi_areas_real=values, spatial_values=values, n_pf=len(values))
        if len(values) < 2:
            result['reason'] = 'PF-SUI requires at least two finite interior particle-associated regions.'
            return result
        metrics = calculate_distribution_metrics(values, scale=scale_unit)
        summary = summarize_spatial_distribution(values)
        result.update(status='complete', reason=None, distribution_metrics=metrics,
                      mean_voronoi_area=metrics['mean'], std_voronoi_area=metrics['std'],
                      median_voronoi_area=summary['median'],
                      median_voronoi_ci_low=summary['median_ci_low'],
                      median_voronoi_ci_high=summary['median_ci_high'],
                      voronoi_cv=metrics['cv'], spatial_uniformity_index=metrics['sui'],
                      sui=metrics['sui'], mean_spatial_value=metrics['mean'],
                      median_spatial_value=summary['median'],
                      median_spatial_ci_low=summary['median_ci_low'],
                      median_spatial_ci_high=summary['median_ci_high'], spatial_cv=metrics['cv'])
    except Exception as exc:
        result.update(status='failed', error=str(exc), reason=str(exc))
    return result


# Shape preset configurations (same as vision.py)
SHAPE_PRESETS = {
    '2D': {
        'labels': ['Circle', 'Triangle', 'Quadrilateral', 'Hexagon', 'Irregular'],
        'descriptions': {
            'Circle': 'circle',
            'Triangle': ['equilateral triangle', 'isosceles triangle', 'scalene triangle'],
            'Quadrilateral': ['square', 'rhombus', 'rectangle', 'rhomboid', 'isosceles trapezium', 'trapezium'],
            'Hexagon': 'hexagon',
            'Irregular': 'irregular',
        }
    },
    '3D': {
        'labels': ['Spheroid', 'Pyramid', 'Hexahedron', 'Cylinder', 'Irregular'],
        'descriptions': {
            'Spheroid': ['oblate spheroid', 'prolate spheroid', 'sphere'],
            'Pyramid': 'tetrahedron',
            'Hexahedron': ['cube', 'parallelpiped'],
            'Cylinder': ['disc', 'tube'],
            'Irregular': 'irregular',
        }
    }
}


def generate_shape_descriptions(labels, preset=None):
    """
    Generate CLIP text descriptions for shape labels.
    
    Args:
        labels: List of shape labels
        preset: '2D' or '3D' or None for custom labels
    
    Returns:
        List of text descriptions for CLIP
    """
    descriptions = []
    for label in labels:
        # Check if using preset and label exists in preset
        if preset and preset in SHAPE_PRESETS:
            if label in SHAPE_PRESETS[preset]['descriptions']:
                desc_info = SHAPE_PRESETS[preset]['descriptions'][label]
                # If multiple descriptions (list), join them
                if isinstance(desc_info, list):
                    desc_text = ', '.join(desc_info)
                else:
                    desc_text = desc_info
                descriptions.append(
                    f"This is a {label} nanoparticle on electron microscope image. "
                    f"Shape description: {desc_text}"
                )
            else:
                # Custom label not in preset
                descriptions.append(
                    f"This is a {label} nanoparticle on electron microscope image. "
                    f"Shape description: {label}"
                )
        else:
            # No preset - use simple description
            descriptions.append(
                f"This is a {label} nanoparticle on electron microscope image. "
                f"Shape description: {label}"
            )
    return descriptions


def run_shape_analysis(image, masks, centroids, boundary_pixels, config):
    """Run CLIP-based projected-morphology classification."""
    import torch
    import clip
    
    result = {}

    try:
        # Get shape preset (default to 2D)
        shape_preset = config.get('shape_preset', '2D')
        
        # Get labels from config or preset
        if 'shape_labels' in config and config['shape_labels']:
            labels = config['shape_labels']
        else:
            labels = SHAPE_PRESETS.get(shape_preset, SHAPE_PRESETS['2D'])['labels']
        
        batch_size = config.get('clip_batch_size', 64)
        temperature = config.get('clip_temperature', None)
        # Always use highest probability (no threshold)
        threshold = None
        
        # Load CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model_name = config.get('clip_model', 'ViT-L/14@336px')
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
        clip_model.eval()
        
        # Generate descriptions using preset
        shape_descriptions = generate_shape_descriptions(labels, preset=shape_preset)
        
        # Tokenize descriptions
        text_tokens = clip.tokenize(shape_descriptions).to(device)
        
        # Convert masks to format expected by classify_shapes_with_clip
        # The function expects masks with 'segmentation' key
        mask_dicts = []
        for mask in masks:
            if isinstance(mask, dict):
                mask_dicts.append(mask)
            else:
                # mask is a numpy array
                mask_dicts.append({'segmentation': mask})
        
        # Run CLIP classification
        shapes, confidences, shape_counts = classify_shapes_with_clip(
            mask_dicts, image, labels, shape_descriptions,
            clip_model, clip_preprocess, text_tokens, device,
            batch_size=batch_size,
            temperature=temperature,
            confidence_threshold=threshold,
        )

        result['shapes'] = shapes
        result['confidences'] = confidences
        result['shape_counts'] = dict(shape_counts)
        result['avg_confidence'] = np.mean(confidences) if confidences else 0
        result['masks'] = mask_dicts
        result['clip_config'] = {
            'model': clip_model_name,
            'batch_size': int(batch_size),
            'temperature': temperature,
            'confidence_threshold': threshold,
            'prompts': shape_descriptions,
        }

        # Dominant shape (always highest count, never Unknown)
        if shape_counts:
            dominant = max(shape_counts, key=shape_counts.get)
            result['dominant_shape'] = dominant
            result['dominant_percentage'] = shape_counts[dominant] / len(shapes) * 100
        else:
            # Fallback to first label if somehow empty
            result['dominant_shape'] = labels[0]
            result['dominant_percentage'] = 0

        # Shape metrics (entropy, diversity)
        metrics = calculate_shape_metrics(shapes, confidences)
        result['entropy'] = metrics.get('shannon_entropy', 0)
        result['simpson_diversity'] = metrics.get('simpson_diversity', 0)
        
        # Generate color map for visualization
        result['color_map'] = generate_shape_colors(list(shape_counts.keys()))

    except Exception as e:
        import traceback
        print(f"Projected-morphology analysis error: {e}")
        traceback.print_exc()
        result['error'] = str(e)
        result['dominant_shape'] = None
        result['entropy'] = None
        result['simpson_diversity'] = None
        result['avg_confidence'] = None
        result['shape_counts'] = {}
        result['shapes'] = []
        result['confidences'] = []
        result['color_map'] = {}

    return result

def create_segmentation_overlay(image, masks):
    """Create segmentation overlay image.
    
    Args:
        image: BGR image
        masks: List of mask dicts with 'segmentation' key (like vision.py)
    """
    overlay = image.copy()

    for i, mask_dict in enumerate(masks):
        # Extract segmentation from dict
        if isinstance(mask_dict, dict):
            mask = mask_dict['segmentation']
        else:
            mask = mask_dict  # Fallback for numpy array
            
        color = plt.cm.tab10(i % 10)[:3]
        color = tuple(int(c * 255) for c in color)

        mask_rgb = np.zeros_like(overlay)
        mask_rgb[mask > 0] = color

        overlay = cv2.addWeighted(overlay, 1.0, mask_rgb, 0.4, 0)

        # Draw contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 1)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def create_histogram_image(areas, scale='px'):
    """Render the production size histogram for the simplified fallback too."""
    fig = create_size_distribution_figure(areas, scale=scale, figsize=(6, 4))
    image = _fig_to_image(fig)
    plt.close(fig)
    return image


def create_dashboard_image(results, image):
    """Create integrated dashboard image."""

    try:
        # Normalize the result dictionaries to the dashboard's public contract.
        size_data = dict(results.get('size') or {})
        distribution_data = dict(results.get('distribution') or {})
        shape_data = dict(results.get('shape') or {})
        boundary_pixels = results.get('boundary_pixels', {})

        if size_data:
            size_data.setdefault('boundary_pixels', boundary_pixels)
        if distribution_data:
            distribution_data.setdefault('boundary_pixels', boundary_pixels)
        if shape_data:
            shape_data.setdefault('masks', results.get('filtered_masks', []))

        scale = (
            size_data.get('scale_unit')
            or distribution_data.get('scale_unit')
            or get_scale_config().get('scale_unit', 'px')
        )
        dashboard = create_integrated_dashboard(
            image,
            size_data=size_data or None,
            distribution_data=distribution_data or None,
            shape_data=shape_data or None,
            scale=scale,
        )
        dashboard_image = _fig_to_image(dashboard)
        plt.close(dashboard)
        return dashboard_image
    except Exception:
        # Create simple dashboard
        return create_simple_dashboard(results, image)


def create_simple_dashboard(results, image):
    """Create simple dashboard when visualization module not available."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    if len(image.shape) == 3:
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Segmentation
    if 'segmentation_overlay' in results:
        axes[0, 1].imshow(results['segmentation_overlay'])
    axes[0, 1].set_title(f"Segmentation ({results.get('particle_count', 0)} particles)")
    axes[0, 1].axis('off')

    scale_unit = get_scale_config().get('scale_unit', 'px')

    # Size histogram
    if 'size' in results and results['size'] and 'areas_real' in results['size']:
        size_data = results['size']
        plot_size_distribution(
            axes[0, 2],
            size_data['areas_real'],
            scale=size_data.get('scale_unit', scale_unit),
            title='Projected-area distribution',
        )
    else:
        axes[0, 2].text(0.5, 0.5, 'Projected particle area\nNot performed', ha='center', va='center')
        axes[0, 2].axis('off')

    # PF-SUI: show the same raincloud chart whenever raw values exist.
    if 'distribution' in results and results['distribution']:
        dist = results['distribution']
        spatial_values = dist.get('spatial_values', dist.get('voronoi_areas_real', []))
        if len(spatial_values) > 0:
            fallback_measure = dist.get('spatial_measure_label', '')
            is_fallback = 'Nearest-neighbor' in fallback_measure
            plot_spatial_distribution(
                axes[1, 0],
                spatial_values,
                scale=dist.get('scale_unit', scale_unit),
                title='PF-SUI',
                value_label='Nearest-neighbor distance' if is_fallback else 'Particle-boundary-based Voronoi cell area',
                unit_suffix='' if is_fallback else '^2',
            )
        else:
            text = "No included finite particle-associated regions\nPF-SUI unavailable"
            axes[1, 0].text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('PF-SUI')
            axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'PF-SUI\nNot performed', ha='center', va='center')
        axes[1, 0].set_title('PF-SUI')
        axes[1, 0].axis('off')

    # Shape pie
    if 'shape' in results and results['shape'] and 'shape_counts' in results['shape']:
        shape_data = results['shape']
        counts = shape_data['shape_counts']
        if counts:
            plot_shape_composition(
                axes[1, 1],
                counts,
                color_map=shape_data.get('color_map', {}),
                title='Morphology composition',
            )
        else:
            axes[1, 1].text(0.5, 0.5, 'Projected morphology\nNot performed', ha='center', va='center')
            axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'Projected morphology\nNot performed', ha='center', va='center')
        axes[1, 1].axis('off')

    # Summary
    summary_text = f"Total Particles: {results.get('particle_count', 0)}\n"
    if 'size' in results and results['size']:
        summary_text += f"Mean projected area: {results['size'].get('mean_area', 0):.2f}\n"
        summary_text += f"CV: {results['size'].get('cv', 0):.1f}%\n"
    if 'shape' in results and results['shape']:
        summary_text += f"Dominant morphology: {results['shape'].get('dominant_shape', 'N/A')}"

    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11)
    axes[1, 2].set_title('Summary')
    axes[1, 2].axis('off')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
