"""
Results Display Component
=========================
Displays analysis results, visualizations, and export options.
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import json
from pathlib import Path
import sys
import io
import zipfile
from datetime import datetime
from copy import deepcopy
from PIL import Image

parent_dir = Path(__file__).parent.parent.parent
ui_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(ui_dir))

from utils.session_state import (
    get_current_image, get_results_for_current_image,
    get_all_config, set_results_for_image, update_step, initialize_session_state
)
from utils.analysis_runner import run_analysis, run_distribution_analysis, run_size_analysis
from utils.provenance import collect_execution_provenance


def render_results_display():
    """Render analysis results and visualizations."""

    # Ensure session state is initialized
    if 'analysis_results' not in st.session_state:
        initialize_session_state()

    st.markdown("### 📈 Analysis Results")

    # Check if analysis should run
    if st.session_state.get('run_analysis', False):
        st.session_state['run_analysis'] = False
        run_analysis_with_progress()

    # Check for results
    name, image = get_current_image()

    if image is None:
        st.warning("⚠️ Please upload an image first.")
        return

    results = get_results_for_current_image()

    if not results:
        st.info("📊 No results yet. Configure and run analysis from the 'Analysis Config' tab.")

        # Show run button
        if st.button("🚀 Run Analysis Now", type="primary"):
            run_analysis_with_progress()
        return

    st.markdown(f"**Results for:** {name}")

    if results.get('error'):
        st.error(f"Analysis failed: {results['error']}")
        details = results.get('error_details') or results.get('traceback')
        if details:
            with st.expander("Failure details", expanded=False):
                st.code(str(details))
        st.info(
            "VISION does not substitute OpenCV segmentation or fabricated module "
            "values when a required scientific step fails."
        )
        return

    # Results tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "📏 Projected particle area",
        "PF-SUI",
        "🔷 Projected morphology",
        "✏️ Edit",
        "💾 Export"
    ])

    with tab1:
        render_dashboard(results, image)

    with tab2:
        render_size_results(results)

    with tab3:
        render_distribution_results(results)

    with tab4:
        render_shape_results(results)

    with tab5:
        render_edit_segmentation(results, image, name)

    with tab6:
        render_export_options(results, name)


def run_analysis_with_progress():
    """Run analysis with progress bar."""

    images = st.session_state.get('uploaded_images', [])
    if not images:
        st.error("❌ No images to analyze. Please upload images first.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_images = len(images)
    success_count = 0
    error_count = 0

    for i, (name, image) in enumerate(images):
        status_text.text(f"🔬 Analyzing {name} ({i+1}/{total_images})...")

        try:
            results = run_analysis(image, name)
            set_results_for_image(name, results)
            
            if results.get('error'):
                error_count += 1
                st.warning(f"⚠️ {name}: {results['error']}")
            else:
                success_count += 1
                
        except Exception as e:
            error_count += 1
            error_msg = str(e)
            st.error(f"❌ Error analyzing {name}: {error_msg}")

            # Provide helpful suggestions based on error type
            if "CUDA" in error_msg or "out of memory" in error_msg.lower():
                st.warning("⚠️ **GPU Memory Error**: Please go back to 'Preprocessing' tab and lower the **GPU Quality** setting (high → medium → low).")
            elif "memory" in error_msg.lower():
                st.info("💡 Try closing other applications or processing fewer images")
            elif "module" in error_msg.lower() or "import" in error_msg.lower():
                st.info("💡 Some packages may be missing. Run INSTALL_AND_RUN.bat to install all dependencies.")
            
            results = {'error': error_msg}
            set_results_for_image(name, results)

        progress_bar.progress((i + 1) / total_images)

    status_text.text("✅ Analysis complete!")
    
    if error_count == 0:
        st.success(f"✅ Successfully analyzed {success_count} image(s)")
    else:
        st.warning(f"⚠️ Completed: {success_count} success, {error_count} errors")
    
    st.rerun()


def render_dashboard(results, image):
    """Render the integrated dashboard view."""

    st.markdown("#### 📊 Integrated Dashboard")

    # Check if dashboard image exists
    if 'dashboard_image' in results:
        st.image(results['dashboard_image'], use_column_width=True)
    else:
        # Create a simple summary dashboard
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📏 Projected particle area**")
            if 'size' in results and results['size']:
                size_data = results['size']
                st.metric("Particle Count", size_data.get('count', 'N/A'))
                st.metric("Mean projected area", f"{size_data.get('mean_area', 0):.2f}")
                st.metric("CV (%)", f"{size_data.get('cv', 0):.1f}")
            else:
                st.info("Not analyzed")

        with col2:
            st.markdown("**PF-SUI**")
            if 'distribution' in results and results['distribution']:
                dist_data = results['distribution']
                st.metric(
                    "Included particle-associated regions",
                    len(dist_data.get('voronoi_areas_real', [])),
                )
                cv = dist_data.get('voronoi_cv')
                st.metric("Voronoi-area CV (%)", f"{cv:.1f}" if cv is not None else 'N/A')
                sui = dist_data.get('spatial_uniformity_index', dist_data.get('sui', 0))
                st.metric("PF-SUI", f"{sui:.3f}" if sui is not None else 'N/A')
                if dist_data.get('reason'):
                    st.info(dist_data['reason'])
            else:
                st.info("Not analyzed")

        with col3:
            st.markdown("**🔷 Projected morphology**")
            if 'shape' in results and results['shape']:
                shape_data = results['shape']
                st.metric("Dominant morphology", shape_data.get('dominant_shape', 'N/A'))
                st.metric("Morphology entropy", f"{shape_data.get('entropy', 0):.2f}")
                st.metric("Avg Confidence", f"{shape_data.get('avg_confidence', 0):.2f}")
            else:
                st.info("Not analyzed")

    # Show segmentation overlay
    if 'segmentation_overlay' in results:
        st.divider()
        st.markdown("**🎯 Segmentation Result:**")
        st.image(results['segmentation_overlay'], use_column_width=True)


def render_size_results(results):
    """Render projected particle-area results."""

    st.markdown("#### 📏 Projected particle area results")

    if 'size' not in results or not results['size']:
        st.info("Projected particle area was not analyzed.")
        return

    size_data = results['size']
    area_unit = size_data.get('unit', 'px²')

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Particle Count", size_data.get('count', 0))
        st.metric(
            f"Mean scale-calibrated projected area ({area_unit})",
            f"{size_data.get('mean_area', 0):.2f}",
        )

    with col2:
        st.metric(f"Median projected area ({area_unit})", f"{size_data.get('median_area', 0):.2f}")
        st.metric(f"Sample Std Dev ({area_unit})", f"{size_data.get('std_area', 0):.2f}")

    with col3:
        st.metric("CV (%)", f"{size_data.get('cv', 0):.1f}")
        st.metric(f"IQR ({area_unit})", f"{size_data.get('iqr', 0):.2f}")

    with col4:
        st.metric("Skewness", f"{size_data.get('skewness', 0):.2f}")
        st.metric("Excess kurtosis", f"{size_data.get('kurtosis', 0):.2f}")

    # Metric explanations
    with st.expander("📖 Metric Definitions", expanded=False):
        st.markdown("""
**CV (Coefficient of Variation)**: Ratio of std to mean (std/mean × 100%). Measures relative dispersion. Lower CV indicates a more uniform projected-area distribution.

**Skewness**: Measures asymmetry of distribution.
- Skewness > 0: Right-tailed (more small particles)
- Skewness ≈ 0: Symmetric distribution
- Skewness < 0: Left-tailed (more large particles)

**Excess kurtosis**: Measures tail weight relative to a normal distribution.
- Excess kurtosis > 0: Heavier tails
- Excess kurtosis ≈ 0: Similar tail weight to a normal distribution
- Excess kurtosis < 0: Lighter tails
""")

    # Histogram
    if 'histogram_image' in size_data:
        st.divider()
        st.markdown("**Projected-area distribution:**")
        st.image(size_data['histogram_image'], use_column_width=True)

    # Area table
    if 'areas' in size_data:
        st.divider()
        with st.expander("📋 Projected particle areas", expanded=False):
            calibrated_unit = size_data.get('unit', 'px²')
            df = pd.DataFrame({
                'Particle ID': range(1, len(size_data['areas']) + 1),
                'Projected area (px²)': size_data['areas'],
                f'Scale-calibrated projected area ({calibrated_unit})': size_data.get(
                    'areas_real', size_data['areas']
                )
            })
            st.dataframe(df, use_container_width=True)


def render_distribution_results(results):
    """Render PF-SUI results."""

    st.markdown("#### PF-SUI results")

    if 'distribution' not in results or not results['distribution']:
        st.info("PF-SUI was not analyzed.")
        return

    dist_data = results['distribution']
    if dist_data.get('status') in ('unavailable', 'failed'):
        st.metric('PF-SUI', 'N/A')
        st.metric('Included particle-associated regions', dist_data.get('n_pf', 0))
        st.info(dist_data.get('reason', 'PF-SUI could not be calculated.'))
        return
    measure_label = dist_data.get('spatial_measure_label', 'Particle-boundary-based Voronoi cell area')
    metric_label = dist_data.get('spatial_metric_label', 'Particle-boundary-based Voronoi area')
    measure_unit = dist_data.get('spatial_measure_unit', dist_data.get('unit', 'area units'))

    st.markdown("**Particle-boundary-based Voronoi:**")
    col1, col2 = st.columns(2)

    with col1:
        if 'spatial_distribution_image' in dist_data:
            st.image(
                dist_data['spatial_distribution_image'],
                caption=f"Observed {measure_label.lower()} values within the centroid convex hull, IQR, and median 95% CI",
                use_column_width=True,
            )

    with col2:
        mean_value = dist_data.get('mean_spatial_value', dist_data.get('mean_voronoi_area', 0))
        median_value = dist_data.get('median_spatial_value', dist_data.get('median_voronoi_area', 0))
        ci_low = dist_data.get('median_spatial_ci_low', dist_data.get('median_voronoi_ci_low', 0))
        ci_high = dist_data.get('median_spatial_ci_high', dist_data.get('median_voronoi_ci_high', 0))
        cv = dist_data.get('spatial_cv', dist_data.get('voronoi_cv', 0))
        st.metric(f"Mean {metric_label}", f"{mean_value:.2f}")
        st.metric(f"Median {metric_label}", f"{median_value:.2f}")
        st.caption(f"Median 95% CI: [{ci_low:.2f}, {ci_high:.2f}] {measure_unit}")
        st.metric(f"{metric_label} CV (%)", f"{cv:.1f}")
        sui = dist_data.get('spatial_uniformity_index', dist_data.get('sui', 0))
        st.metric("PF-SUI", f"{sui:.3f}")

    if 'voronoi_image' in dist_data:
        st.divider()
        st.markdown("**Particle-boundary-based Voronoi:**")
        st.image(
            dist_data['voronoi_image'],
            caption="Particle-boundary-based Voronoi within the centroid convex hull",
            use_column_width=True,
        )

def render_shape_results(results):
    """Render projected-morphology results."""

    st.markdown("#### 🔷 Projected morphology results")

    if 'shape' not in results or not results['shape']:
        st.info("Projected morphology was not analyzed.")
        return

    shape_data = results['shape']

    # Pie chart
    col1, col2 = st.columns(2)

    with col1:
        if 'pie_chart_image' in shape_data:
            st.image(shape_data['pie_chart_image'], caption="Morphology composition", use_column_width=True)

    with col2:
        st.metric("Dominant morphology", shape_data.get('dominant_shape', 'N/A'))
        st.metric("Dominant %", f"{shape_data.get('dominant_percentage', 0):.1f}%")
        st.metric("Shannon Entropy", f"{shape_data.get('entropy', 0):.2f}")
        st.metric("Avg Confidence", f"{shape_data.get('avg_confidence', 0):.2f}")

    # Morphology counts
    if 'shape_counts' in shape_data:
        st.divider()
        st.markdown("**Morphology composition:**")

        df = pd.DataFrame({
            'Morphology': list(shape_data['shape_counts'].keys()),
            'Count': list(shape_data['shape_counts'].values())
        })
        df['Percentage'] = df['Count'] / df['Count'].sum() * 100

        st.dataframe(df, use_container_width=True)

    # Projected-morphology overlay
    if 'overlay_image' in shape_data:
        st.divider()
        st.markdown("**Projected-morphology overlay:**")
        st.image(shape_data['overlay_image'], use_column_width=True)


def create_numbered_overlay(image, masks, centroids, deleted_indices=None):
    """Create an overlay image with numbered particles.

    Args:
        image: BGR image
        masks: List of mask dicts with 'segmentation' key
        centroids: List of (x, y) centroid tuples
        deleted_indices: Set of indices to mark as deleted (shown in red)

    Returns:
        RGB overlay image with numbered particles
    """
    import matplotlib.pyplot as plt

    overlay = image.copy()
    deleted_indices = deleted_indices or set()

    # Define colors - use tab20 for variety
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for i, (mask_dict, centroid) in enumerate(zip(masks, centroids)):
        if isinstance(mask_dict, dict):
            mask = mask_dict['segmentation']
        else:
            mask = mask_dict

        # Choose color - red for deleted, normal color otherwise
        if i in deleted_indices:
            color_bgr = (0, 0, 200)  # Red for deleted
            alpha = 0.5
        else:
            color = colors[i % 20][:3]
            color_bgr = tuple(int(c * 255) for c in color[::-1])
            alpha = 0.4

        # Create colored mask overlay
        mask_colored = np.zeros_like(overlay)
        mask_colored[mask > 0] = color_bgr

        # Blend with original
        mask_region = mask > 0
        overlay[mask_region] = cv2.addWeighted(
            overlay[mask_region], 1 - alpha,
            mask_colored[mask_region], alpha, 0
        )

        # Draw contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_color = (0, 0, 255) if i in deleted_indices else color_bgr
        cv2.drawContours(overlay, contours, -1, contour_color, 2)

        # Draw number at centroid
        cx, cy = int(centroid[0]), int(centroid[1])
        label = str(i + 1)

        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw background rectangle
        bg_color = (0, 0, 180) if i in deleted_indices else (0, 0, 0)
        cv2.rectangle(overlay,
                      (cx - text_w//2 - 2, cy - text_h//2 - 2),
                      (cx + text_w//2 + 2, cy + text_h//2 + 2),
                      bg_color, -1)

        # Draw text
        text_color = (255, 255, 255)
        cv2.putText(overlay, label,
                    (cx - text_w//2, cy + text_h//2),
                    font, font_scale, text_color, thickness)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def render_edit_segmentation(results, image, image_name):
    """Render the segmentation editing interface for noise removal."""

    st.markdown("#### ✏️ Edit Segmentation")
    st.markdown("Remove noise or false-positive particles from segmentation results.")

    # Check if we have segmentation results
    if 'filtered_masks' not in results or 'centroids' not in results:
        st.warning("⚠️ No segmentation data available. Please run analysis first.")
        return

    masks = results.get('filtered_masks', [])
    centroids = results.get('centroids', [])

    if len(masks) == 0:
        st.info("No particles detected in this image.")
        return

    # Initialize deleted indices in session state for this image
    deleted_key = f'deleted_indices_{image_name}'
    if deleted_key not in st.session_state:
        st.session_state[deleted_key] = set()

    deleted_indices = st.session_state[deleted_key]

    # Store original results if not already stored
    original_key = f'original_results_{image_name}'
    if original_key not in st.session_state:
        st.session_state[original_key] = results.copy()

    # Create numbered overlay
    # Get ROI image if available
    from utils.session_state import get_roi_for_image, get_scale_config
    particle_roi = get_roi_for_image(image_name)
    if particle_roi:
        x1, y1, x2, y2 = particle_roi
        roi_image = image[y1:y2, x1:x2].copy()
    else:
        roi_image = image.copy()

    numbered_overlay = create_numbered_overlay(roi_image, masks, centroids, deleted_indices)

    # Display info
    total_particles = len(masks)
    deleted_count = len(deleted_indices)
    active_count = total_particles - deleted_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detected", total_particles)
    with col2:
        st.metric("Marked for Deletion", deleted_count, delta=-deleted_count if deleted_count > 0 else None)
    with col3:
        st.metric("Active Particles", active_count)

    st.divider()

    # Display numbered overlay
    st.markdown("**Numbered Particle Overlay:**")
    st.markdown("_Particles marked for deletion are shown in red._")
    st.image(numbered_overlay, use_column_width=True)

    st.divider()

    # Selection interface
    st.markdown("**Select Particles to Delete:**")

    # Create options list with current status
    particle_options = []
    for i in range(len(masks)):
        status = "🔴" if i in deleted_indices else "✅"
        particle_options.append(f"{status} Particle {i+1}")

    # Multiselect for deletion
    selected = st.multiselect(
        "Select particles to mark for deletion",
        options=particle_options,
        default=[particle_options[i] for i in deleted_indices if i < len(particle_options)],
        help="Select noise or false-positive particles to remove from analysis"
    )

    # Parse selected indices
    new_deleted_indices = set()
    for s in selected:
        # Extract particle number from "✅ Particle X" or "🔴 Particle X"
        try:
            num = int(s.split("Particle")[1].strip()) - 1
            new_deleted_indices.add(num)
        except:
            pass

    # Update deleted indices if changed
    if new_deleted_indices != deleted_indices:
        st.session_state[deleted_key] = new_deleted_indices
        st.rerun()

    st.divider()

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Reset to Original", use_container_width=True,
                     help="Restore all particles and original analysis results"):
            # Clear deleted indices
            st.session_state[deleted_key] = set()
            # Restore original results
            if original_key in st.session_state:
                set_results_for_image(image_name, st.session_state[original_key])
            st.success("✅ Reset to original results!")
            st.rerun()

    with col2:
        recalc_disabled = len(new_deleted_indices) == 0
        if st.button("🗑️ Delete & Recalculate", type="primary", use_container_width=True,
                     disabled=recalc_disabled,
                     help="Remove selected particles and recalculate all metrics"):
            if new_deleted_indices:
                with st.spinner("Recalculating metrics..."):
                    recalculate_after_deletion(
                        image_name, roi_image, results,
                        new_deleted_indices, masks, centroids
                    )
                st.success(f"✅ Removed {len(new_deleted_indices)} particle(s) and recalculated metrics!")
                st.rerun()

    with col3:
        if st.button("📊 View Updated Results", use_container_width=True):
            # Switch to Dashboard tab
            st.info("Switch to the 'Dashboard' tab to see updated results.")


def recalculate_after_deletion(image_name, roi_image, results, deleted_indices, masks, centroids):
    """Rebuild derived results and exports from exactly the surviving masks."""
    from collections import Counter
    from modules.preprocessing import extract_boundary_pixels_dict
    from modules.shape_analysis import calculate_shape_metrics, generate_shape_colors
    from utils.session_state import get_scale_config
    from utils.analysis_runner import create_dashboard_image, create_shape_pie_chart, create_shape_overlay, _fig_to_image
    import matplotlib.pyplot as plt

    kept = [i for i in range(len(masks)) if i not in deleted_indices]
    retained_masks = [masks[i] for i in kept]
    retained_centroids = [centroids[i] for i in kept]
    cfg = st.session_state.get('analysis_config', {})
    scale = get_scale_config()
    factor = scale.get('pixel_to_real') or 1.0
    unit = scale.get('scale_unit', 'px') if scale.get('pixel_to_real') else 'px'
    boundary = extract_boundary_pixels_dict(retained_masks)
    new = {key: value for key, value in results.items() if key not in (
        'size', 'distribution', 'shape', 'dashboard_image', 'segmentation_overlay',
        'error', 'error_details', 'traceback', 'error_type', 'error_suggestion')}
    new.update(image_name=image_name, error=None, particle_count=len(kept),
               filtered_masks=retained_masks, centroids=retained_centroids, boundary_pixels=boundary)
    provenance = deepcopy(new.get('execution_provenance') or {})
    original_ids = results.get('particle_source_ids', list(range(1, len(masks)+1)))
    previous_deleted = provenance.get('manual_postprocessing', {}).get('deleted_particle_ids', [])
    removed = [original_ids[i] for i in sorted(deleted_indices)]
    new['particle_source_ids'] = [original_ids[i] for i in kept]
    provenance['manual_postprocessing'] = {
        'applied': True, 'deleted_particle_ids': sorted(set(previous_deleted+removed)),
        'remaining_particle_count': len(kept),
        'policy': 'User-reviewed deletion; derived tables and figures regenerated from surviving masks.',
    }
    new['execution_provenance'] = provenance
    new['segmentation_overlay'] = create_simple_overlay(roi_image, retained_masks)
    if cfg.get('enable_size', True):
        new['size'] = run_size_analysis(retained_masks, boundary, factor, unit, cfg)
    if cfg.get('enable_distribution', cfg.get('enable_spatial_uniformity', True)):
        new['distribution'] = run_distribution_analysis(retained_centroids, boundary, roi_image.shape[:2], factor, unit, cfg)
        dist = new['distribution']
        if dist.get('status') == 'complete':
            from utils.analysis_runner import create_distribution_violin_plot, create_voronoi_overlay
            fig = create_distribution_violin_plot(dist['voronoi_areas_real'], scale=unit, figsize=(8, 5))
            dist['spatial_distribution_image'] = _fig_to_image(fig)
            plt.close(fig)
            fig, _ = create_voronoi_overlay(
                roi_image, boundary, dist['voronoi_areas_real_dict'], scale=unit,
                concave_vertex=dist['concave_vertex'], inside_centroids=dist['inside_centroids'])
            dist['voronoi_image'] = _fig_to_image(fig)
            plt.close(fig)
    if cfg.get('enable_shape', True) and 'shape' in results:
        old = results['shape']
        shapes = [old['shapes'][i] for i in kept]
        confidences = [old['confidences'][i] for i in kept]
        counts = dict(Counter(shapes))
        dominant = max(counts, key=counts.get) if counts else None
        metrics = calculate_shape_metrics(shapes, confidences) if shapes else {}
        shape = {
            'shapes': shapes, 'confidences': confidences, 'shape_counts': counts,
            'masks': retained_masks, 'color_map': generate_shape_colors(list(counts)),
            'dominant_shape': dominant, 'dominant_percentage': 100*counts[dominant]/len(shapes) if dominant else 0,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0,
            'entropy': metrics.get('shannon_entropy', 0),
            'simpson_diversity': metrics.get('simpson_diversity', 0),
            'clip_config': old.get('clip_config', {}),
        }
        if counts:
            for key, fig in (
                ('pie_chart_image', create_shape_pie_chart(counts, shape['color_map'])),
                ('overlay_image', create_shape_overlay(roi_image, retained_masks, shapes, shape['color_map'])),
            ):
                shape[key] = _fig_to_image(fig)
                plt.close(fig)
        new['shape'] = shape
    new['dashboard_image'] = create_dashboard_image(new, roi_image)
    set_results_for_image(image_name, new)


def create_simple_overlay(image, masks):
    """Create a simple segmentation overlay without numbers."""
    import matplotlib.pyplot as plt

    overlay = image.copy()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, mask_dict in enumerate(masks):
        if isinstance(mask_dict, dict):
            mask = mask_dict['segmentation']
        else:
            mask = mask_dict

        color = colors[i % 10][:3]
        color_bgr = tuple(int(c * 255) for c in color[::-1])

        mask_colored = np.zeros_like(overlay)
        mask_colored[mask > 0] = color_bgr

        overlay = cv2.addWeighted(overlay, 1.0, mask_colored, 0.4, 0)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color_bgr, 1)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def render_export_options(results, image_name):
    """Render export/download options."""

    st.markdown("#### 💾 Export Results")

    output_config = st.session_state.get('output_config', {})
    col1, col2 = st.columns(2)
    image_stem = Path(image_name).stem

    with col1:
        st.markdown("**Machine-readable tables:**")
        if output_config.get('summary_csv', True):
            st.download_button(
                label="Download Image-summary CSV",
                data=create_summary_csv(results),
                file_name=f"{image_stem}_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if output_config.get('particle_csv', True):
            st.download_button(
                label="Download Particle-level CSV",
                data=create_particle_csv(results),
                file_name=f"{image_stem}_particles.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if output_config.get('xlsx', True):
            st.download_button(
                label="Download XLSX Report",
                data=create_excel_report(results),
                file_name=f"{image_stem}_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with col2:
        st.markdown("**PNG and complete package:**")
        if output_config.get('png', True):
            st.download_button(
                label="Download PNG Masks and Visualizations",
                data=create_visualizations_zip(results),
                file_name=f"{image_stem}_png_outputs.zip",
                mime="application/zip",
                use_container_width=True,
            )
        if output_config.get('zip', True):
            st.download_button(
                label="Download Complete ZIP Package",
                data=create_results_zip(),
                file_name=f"vision_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True,
            )

    if not any(output_config.get(key, True) for key in (
        'summary_csv', 'particle_csv', 'xlsx', 'png', 'zip'
    )):
        st.info("No output format is enabled. Change Output configuration in Analysis Config.")


def iter_visualization_images(results):
    """Yield stable PNG filenames and result image arrays."""
    size_results = results.get('size') or {}
    distribution_results = results.get('distribution') or {}
    shape_results = results.get('shape') or {}
    candidates = [
        ('dashboard.png', results.get('dashboard_image')),
        ('segmentation_overlay.png', results.get('segmentation_overlay')),
        ('projected_area_distribution.png', size_results.get('histogram_image')),
        (
            'pf_sui_particle_associated_area_distribution.png',
            distribution_results.get('spatial_distribution_image'),
        ),
        ('particle_boundary_voronoi.png', distribution_results.get('voronoi_image')),
        ('morphology_composition.png', shape_results.get('pie_chart_image')),
        ('projected_morphology_overlay.png', shape_results.get('overlay_image')),
    ]
    for filename, image_array in candidates:
        if image_array is not None:
            yield filename, image_array


def image_array_to_png(image_array):
    """Encode a Streamlit result image array as PNG bytes."""
    array = np.asarray(image_array)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if array.size and np.nanmax(array) <= 1.0 else 1.0
        array = np.nan_to_num(array * scale, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ValueError(f"Unsupported visualization array shape: {array.shape}")

    output = io.BytesIO()
    Image.fromarray(array).save(output, format='PNG')
    return output.getvalue()


def mask_array_to_png(mask_array):
    """Encode a binary or integer instance mask without losing label values."""
    array = np.asarray(mask_array)
    output = io.BytesIO()
    if array.dtype == np.bool_:
        Image.fromarray(array.astype(np.uint8) * 255, mode='L').save(output, format='PNG')
    else:
        labels = np.asarray(array, dtype=np.uint16)
        Image.fromarray(labels, mode='I;16').save(output, format='PNG')
    return output.getvalue()


def iter_mask_png_files(results):
    """Yield a lossless instance label map and each particle mask as PNG."""
    masks = results.get('filtered_masks') or []
    if not masks:
        return
    first_mask = np.asarray(
        masks[0]['segmentation'] if isinstance(masks[0], dict) else masks[0],
        dtype=bool,
    )
    if len(masks) > np.iinfo(np.uint16).max:
        raise ValueError("Too many particles for a 16-bit instance label map")
    label_map = np.zeros(first_mask.shape, dtype=np.uint16)
    for particle_id, mask_item in enumerate(masks, start=1):
        mask = np.asarray(
            mask_item['segmentation'] if isinstance(mask_item, dict) else mask_item,
            dtype=bool,
        )
        if mask.shape != first_mask.shape:
            raise ValueError("Particle masks do not share a common image shape")
        label_map[mask] = particle_id
        yield (
            f"particle_{particle_id:04d}_binary_mask.png",
            mask_array_to_png(mask),
        )
    yield "instance_label_map_uint16.png", mask_array_to_png(label_map)


def create_visualizations_zip(results):
    """Create a ZIP containing generated visualizations and masks as PNG."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename, image_array in iter_visualization_images(results):
            zf.writestr(f"visualizations/{filename}", image_array_to_png(image_array))
        for filename, mask_png in iter_mask_png_files(results):
            zf.writestr(f"masks/{filename}", mask_png)
        zf.writestr(
            "masks/README.txt",
            "Binary masks store background=0 and particle=255.\n"
            "instance_label_map_uint16.png stores background=0 and Particle_ID as the pixel value.\n"
            "When masks overlap, the label map stores the later Particle_ID; individual masks preserve exact overlap.\n",
        )
    output.seek(0)
    return output.getvalue()


def create_summary_dataframe(results):
    """Return one image-level summary row using stable, documented fields."""
    data = {
        'image_name': results.get('image_name', ''),
        'particle_count': results.get('particle_count', 0),
        'status': 'failed' if results.get('error') else 'complete',
        'error': results.get('error', ''),
    }

    size = results.get('size') or {}
    if size:
        data.update({
            'projected_area_unit': size.get('unit', ''),
            'mean_projected_area': size.get('mean_area', 0),
            'median_projected_area': size.get('median_area', 0),
            'projected_area_sample_std': size.get('std_area', 0),
            'projected_area_cv_percent': size.get('cv', 0),
            'projected_area_iqr': size.get('iqr', 0),
            'projected_area_skewness': size.get('skewness', 0),
            'projected_area_excess_kurtosis': size.get('kurtosis', 0),
        })

    dist = results.get('distribution') or {}
    if dist:
        data.update({
            'pf_sui_status': dist.get('status', 'complete'),
            'pf_sui_reason': dist.get('reason', ''),
            'particle_boundary_voronoi_unit': dist.get('unit', ''),
            'particle_boundary_voronoi_count': len(dist.get('voronoi_areas_real', [])),
            'particle_boundary_voronoi_mean_area': dist.get('mean_voronoi_area', 0),
            'particle_boundary_voronoi_sample_std': dist.get('std_voronoi_area', 0),
            'particle_boundary_voronoi_median_area': dist.get('median_voronoi_area', 0),
            'particle_boundary_voronoi_cv_percent': dist.get('voronoi_cv', 0),
            'pf_sui': dist.get('spatial_uniformity_index', dist.get('sui', 0)),
        })

    shape = results.get('shape') or {}
    if shape:
        data.update({
            'dominant_morphology': shape.get('dominant_shape', ''),
            'dominant_morphology_percent': shape.get('dominant_percentage', 0),
            'morphology_entropy_bits': shape.get('entropy', 0),
            'mean_morphology_confidence': shape.get('avg_confidence', 0),
            'morphology_composition_json': json.dumps(
                shape.get('shape_counts', {}), ensure_ascii=False, sort_keys=True
            ),
        })
    return pd.DataFrame([data])


def create_summary_csv(results):
    """Create an image-level summary CSV."""
    return create_summary_dataframe(results).to_csv(index=False)


def _mapping_value_for_centroid(mapping, centroid):
    if not mapping or centroid is None:
        return np.nan
    key = tuple(centroid)
    if key in mapping:
        return mapping[key]
    return np.nan


def create_particle_dataframe(results):
    """Combine all selected modules into one row per SAM particle."""
    masks = results.get('filtered_masks') or []
    centroids = list(results.get('centroids') or [])
    size = results.get('size') or {}
    shape = results.get('shape') or {}
    dist = results.get('distribution') or {}

    areas_px = list(size.get('areas') or [])
    if not areas_px and masks:
        areas_px = [
            int(np.count_nonzero(
                item['segmentation'] if isinstance(item, dict) else item
            ))
            for item in masks
        ]
    areas_real = list(size.get('areas_real') or [])
    morphologies = list(shape.get('shapes') or [])
    confidences = list(shape.get('confidences') or [])
    voronoi_map = dist.get('voronoi_areas_real_dict') or {}

    particle_count = max(
        len(masks), len(centroids), len(areas_px), len(areas_real),
        len(morphologies), len(confidences),
    )
    rows = []
    for index in range(particle_count):
        centroid = centroids[index] if index < len(centroids) else None
        voronoi_area = _mapping_value_for_centroid(voronoi_map, centroid)
        rows.append({
            'Particle_ID': index + 1,
            'Centroid_x_px': centroid[0] if centroid is not None else np.nan,
            'Centroid_y_px': centroid[1] if centroid is not None else np.nan,
            'Projected_area_px2': areas_px[index] if index < len(areas_px) else np.nan,
            'Scale_calibrated_projected_area': (
                areas_real[index] if index < len(areas_real) else np.nan
            ),
            'Projected_area_unit': size.get('unit', ''),
            'Projected_morphology': (
                morphologies[index] if index < len(morphologies) else ''
            ),
            'Morphology_confidence': (
                confidences[index] if index < len(confidences) else np.nan
            ),
            'PF_SUI_included': bool(np.isfinite(voronoi_area)),
            'Particle_boundary_Voronoi_area': voronoi_area,
            'Particle_boundary_Voronoi_area_unit': dist.get('unit', ''),
        })
    columns = ['Particle_ID', 'Centroid_x_px', 'Centroid_y_px', 'Projected_area_px2',
               'Scale_calibrated_projected_area', 'Projected_area_unit',
               'Projected_morphology', 'Morphology_confidence', 'PF_SUI_included',
               'Particle_boundary_Voronoi_area', 'Particle_boundary_Voronoi_area_unit']
    return pd.DataFrame(rows, columns=columns)


def create_particle_csv(results):
    """Create a particle-level CSV for one image."""
    return create_particle_dataframe(results).to_csv(index=False)


def _flatten_json_rows(data, prefix=''):
    """Flatten nested provenance/configuration into Excel-safe key/value rows."""
    rows = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_json_rows(value, child_prefix))
    elif isinstance(data, (list, tuple)):
        rows.append({'Field': prefix, 'Value': json.dumps(data, ensure_ascii=False)})
    else:
        rows.append({'Field': prefix, 'Value': data})
    return rows


def create_excel_report(results, settings=None):
    """Create an XLSX report with summary, particle, settings, and provenance sheets."""
    output = io.BytesIO()
    if settings is None:
        settings = get_all_config()
    provenance = results.get('execution_provenance') or collect_execution_provenance()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary = create_summary_dataframe(results).T.reset_index()
        summary.columns = ['Metric', 'Value']
        summary.to_excel(writer, sheet_name='Image Summary', index=False)
        create_particle_dataframe(results).to_excel(
            writer, sheet_name='Particles', index=False
        )
        pd.DataFrame(_flatten_json_rows(settings)).to_excel(
            writer, sheet_name='Settings', index=False
        )
        pd.DataFrame(_flatten_json_rows(provenance)).to_excel(
            writer, sheet_name='Provenance', index=False
        )

    output.seek(0)
    return output.getvalue()


def create_results_zip():
    """Create the complete multi-image result and reproducibility package."""
    output = io.BytesIO()
    all_results = st.session_state.get('analysis_results', {})
    settings = get_all_config()
    failure_rows = []

    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img_name, results in all_results.items():
            image_stem = Path(img_name).stem
            failure_rows.append({
                'image_name': img_name,
                'status': 'failed' if results.get('error') else 'complete',
                'error': results.get('error', ''),
            })
            zf.writestr(
                f"{image_stem}/{image_stem}_summary.csv",
                create_summary_csv(results),
            )
            zf.writestr(
                f"{image_stem}/{image_stem}_particles.csv",
                create_particle_csv(results),
            )
            zf.writestr(
                f"{image_stem}/{image_stem}_report.xlsx",
                create_excel_report(results, settings=settings),
            )

            for filename, image_array in iter_visualization_images(results):
                zf.writestr(
                    f"{image_stem}/visualizations/{filename}",
                    image_array_to_png(image_array),
                )
            for filename, mask_png in iter_mask_png_files(results):
                zf.writestr(f"{image_stem}/masks/{filename}", mask_png)

            zf.writestr(
                f"{image_stem}/{image_stem}_execution_provenance.json",
                json.dumps(
                    results.get('execution_provenance', {}),
                    indent=2,
                    ensure_ascii=False,
                ),
            )

        zf.writestr(
            "vision_settings.json",
            json.dumps(settings, indent=2, ensure_ascii=False),
        )
        zf.writestr(
            "failure_manifest.csv",
            pd.DataFrame(
                failure_rows,
                columns=['image_name', 'status', 'error'],
            ).to_csv(index=False),
        )
        zf.writestr(
            "README.txt",
            "VISION result package\n"
            "- Each image folder contains image-summary CSV, particle-level CSV, XLSX, PNG outputs, and provenance.\n"
            "- Binary masks use 0/255; the uint16 instance label map uses Particle_ID values.\n"
            "- vision_settings.json contains the reusable GUI configuration.\n"
            "- failure_manifest.csv records complete and failed images without silent substitution.\n",
        )

    output.seek(0)
    return output.getvalue()
