"""
Analysis Configuration Component
================================
Configuration UI for Projected particle area, Projected morphology, and PF-SUI.
"""

import html

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import sys
from PIL import Image

parent_dir = Path(__file__).parent.parent.parent
ui_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(ui_dir))

from utils.session_state import get_current_image, update_step, initialize_session_state


ANALYSIS_DEFAULTS = {
    'enable_size': True,
    'enable_distribution': True,
    'enable_spatial_uniformity': True,
    'enable_shape': True,
    'enable_particle_area_filter': False,
    'min_particle_area': 10,
    'max_particle_area': 100000,
    'shape_preset': '2D',
    'shape_labels': ['Circle', 'Triangle', 'Quadrilateral', 'Hexagon', 'Irregular'],
    'clip_model': 'ViT-L/14@336px',
    'clip_batch_size': 64,
    'clip_temperature': None,
    'clip_confidence_threshold': None,
}

OUTPUT_OPTIONS = {
    'summary_csv': 'Image-summary CSV',
    'particle_csv': 'Particle-level CSV',
    'xlsx': 'XLSX report',
    'png': 'PNG masks and visualizations',
    'zip': 'Complete ZIP package',
}


def normalize_analysis_config(config):
    """Fill defaults and keep old/new spatial-uniformity keys in sync."""
    config = config or {}
    normalized = {**ANALYSIS_DEFAULTS, **config}
    if 'enable_distribution' in config:
        spatial_enabled = bool(config['enable_distribution'])
    else:
        spatial_enabled = bool(normalized.get('enable_spatial_uniformity', True))
    normalized['enable_distribution'] = spatial_enabled
    normalized['enable_spatial_uniformity'] = spatial_enabled
    return normalized


OVERVIEW_EXAMPLES_DIR = ui_dir / "assets" / "overview"

OVERVIEW_MODULES = {
    "size": {
        "title": "Projected particle area",
        "config_key": "enable_size",
        "widget_key": "overview_enable_size",
        "images": [
            ("projected_area_overlay.png", "Scale-calibrated projected area"),
            ("projected_area_distribution.png", "Projected-area distribution"),
        ],
        "details": [
            "Scale-calibrated projected area",
            "Projected-area distribution",
            "Projected-area summary statistics",
        ],
    },
    "shape": {
        "title": "Projected morphology",
        "config_key": "enable_shape",
        "widget_key": "overview_enable_shape",
        "images": [
            ("projected_morphology_overlay.png", "Projected-morphology overlay"),
            ("morphology_composition.png", "Morphology composition"),
        ],
        "details": [
            "Projected-morphology overlay",
            "Projected-morphology classification",
            "Morphology composition",
        ],
    },
    "distribution": {
        "title": "PF-SUI",
        "config_key": "enable_distribution",
        "widget_key": "overview_enable_distribution",
        "images": [
            ("particle_boundary_voronoi.png", "Particle-boundary-based Voronoi"),
            ("pf_sui_overview.png", "Centroid convex hull"),
        ],
        "details": [
            "Particle-boundary-based Voronoi",
            "Centroid convex hull",
            "PF-SUI",
        ],
    },
}


def get_overview_analysis_config():
    """Return the current analysis configuration with legacy keys normalized."""
    if "analysis_config" not in st.session_state:
        initialize_session_state()
    return normalize_analysis_config(st.session_state.get("analysis_config", {}))


def sync_overview_analysis_selection(config):
    """Keep overview checkboxes aligned after changes on the detailed config step."""
    normalized = normalize_analysis_config(config)
    for module in OVERVIEW_MODULES.values():
        st.session_state[module["widget_key"]] = bool(
            normalized.get(module["config_key"], True)
        )


def render_analysis_overview_card(module_name, card_height=None):
    """Render one selectable analysis card with representative output images."""
    if module_name not in OVERVIEW_MODULES:
        raise ValueError(f"Unknown overview module: {module_name}")

    module = OVERVIEW_MODULES[module_name]
    config = get_overview_analysis_config()
    widget_key = module["widget_key"]
    config_key = module["config_key"]

    if widget_key not in st.session_state:
        st.session_state[widget_key] = bool(config.get(config_key, True))

    with st.container(border=True, height=card_height):
        st.markdown(
            f'<div class="overview-card-title">{module["title"]}</div>',
            unsafe_allow_html=True,
        )
        enabled = st.checkbox(
            f"Include {module['title']}",
            key=widget_key,
        )

        image_columns = st.columns(len(module["images"]))
        for image_column, (filename, caption) in zip(image_columns, module["images"]):
            image_path = OVERVIEW_EXAMPLES_DIR / filename
            if image_path.exists():
                with image_column:
                    st.image(str(image_path), caption=caption, use_column_width=True)

        for detail in module["details"]:
            st.markdown(
                f'<div class="overview-detail">{detail}</div>',
                unsafe_allow_html=True,
            )

    config[config_key] = enabled
    if module_name == "distribution":
        config["enable_spatial_uniformity"] = enabled
    st.session_state["analysis_config"] = normalize_analysis_config(config)


def _preprocessing_summary_label():
    """Describe the active preprocessing defaults without duplicating pipeline logic."""
    config = st.session_state.get("preprocessing_config", {})
    bm3d_enabled = bool(config.get("enable_bm3d", True))
    noise2sr_enabled = bool(config.get("enable_noise2sr", True))

    if bm3d_enabled and noise2sr_enabled:
        return "BM3D → Noise2SR"
    if bm3d_enabled:
        return "BM3D"
    if noise2sr_enabled:
        return "Noise2SR"
    return "None"


def _output_summary_label():
    """Describe the output formats selected in the GUI."""
    config = st.session_state.get('output_config', {})
    selected = [
        label for key, label in OUTPUT_OPTIONS.items()
        if config.get(key, True)
    ]
    return ", ".join(selected) if selected else "None"


def render_analysis_overview_summary(card_height=None):
    """Render the compact summary panel for the upload/setup step."""
    config = get_overview_analysis_config()
    name, _ = get_current_image()

    selected_modules = []
    if config.get("enable_size", True):
        selected_modules.append("Projected particle area")
    if config.get("enable_shape", True):
        selected_modules.append("Projected morphology")
    if config.get("enable_distribution", True):
        selected_modules.append("PF-SUI")

    roi_config = st.session_state.get("roi_config", {})
    scale_config = st.session_state.get("scale_config", {})
    roi_status = "Defined" if roi_config.get("particle_roi") else "Set in next step"
    calibration_status = "Defined" if scale_config.get("pixel_to_real") else "Set in next step"
    is_ready = bool(name and selected_modules)

    summary_items = [
        ("Input image", name or "Not selected"),
        ("ROI", roi_status),
        ("Calibration", calibration_status),
        ("Preprocessing", _preprocessing_summary_label()),
        ("Selected modules", ", ".join(selected_modules) if selected_modules else "None selected"),
        ("Outputs", _output_summary_label()),
    ]

    with st.container(border=True, height=card_height):
        st.markdown('<div class="overview-card-title">Analysis Summary</div>', unsafe_allow_html=True)
        for label, value in summary_items:
            st.markdown(
                f'<div class="overview-summary-label">{label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="overview-summary-value">{html.escape(value)}</div>',
                unsafe_allow_html=True,
            )

        if is_ready:
            st.success("Ready for ROI & Scale")
        elif not name:
            st.info("Image required")
        else:
            st.info("Select an analysis module")

# Try to import streamlit-drawable-canvas for interactive drawing
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


def render_analysis_config():
    """Render analysis configuration options."""

    # Ensure session state is initialized
    if 'analysis_config' not in st.session_state:
        initialize_session_state()

    st.markdown("### 📊 Analysis Configuration")

    # Check if image is loaded
    name, image = get_current_image()

    if image is None:
        st.warning("⚠️ Please upload an image first.")
        return

    st.markdown(f"**Current Image:** {name}")

    # Get analysis config
    config = normalize_analysis_config(st.session_state.get('analysis_config', {}))

    # Analysis type selection
    st.markdown("#### 🎯 Analysis Types")
    st.markdown("Select which analyses to perform:")

    col1, col2, col3 = st.columns(3)

    with col1:
        enable_size = st.checkbox(
            "📏 Projected particle area",
            value=config.get('enable_size', True),
            help="Calculate scale-calibrated projected particle areas and their statistics"
        )
        config['enable_size'] = enable_size

        if enable_size:
            st.markdown("**Includes:**")
            st.markdown("- Scale-calibrated projected area")
            st.markdown("- Projected-area distribution")
            st.markdown("- Projected-area summary statistics")

    with col2:
        enable_distribution = st.checkbox(
            "PF-SUI",
            value=config.get('enable_distribution', True),
            help="PF-SUI analysis using particle-boundary-based Voronoi cells within a centroid convex-hull observation domain",
            key="analysis_enable_spatial_uniformity",
        )
        config['enable_distribution'] = enable_distribution
        config['enable_spatial_uniformity'] = enable_distribution

        if enable_distribution:
            st.markdown("**Includes:**")
            st.markdown("- Particle-boundary-based Voronoi")
            st.markdown("- Centroid convex hull")
            st.markdown("- PF-SUI")

    with col3:
        enable_shape = st.checkbox(
            "🔷 Projected morphology",
            value=config.get('enable_shape', True),
            help="CLIP-based projected-morphology classification"
        )
        config['enable_shape'] = enable_shape

        if enable_shape:
            st.markdown("**Includes:**")
            st.markdown("- Projected-morphology overlay")
            st.markdown("- Projected-morphology classification")
            st.markdown("- Morphology composition")
            st.markdown("- Confidence scores")

    # Validate at least one is selected
    if not (enable_size or enable_distribution or enable_shape):
        st.error("⚠️ Please select at least one analysis type")

    st.divider()

    # Shared particle-mask options. The filter precedes all selected analysis
    # modules, so it must remain available even when projected area is disabled.
    st.markdown("#### Shared particle-mask options")
    enable_area_filter = st.checkbox(
        "Apply minimum projected-area filter",
        value=config.get('enable_particle_area_filter', False),
        help=(
            "Optionally exclude SAM masks below a user-defined projected area. "
            "Leave off to reproduce the validation post-processing."
        ),
    )
    config['enable_particle_area_filter'] = enable_area_filter
    if not enable_area_filter:
        st.caption("Area filtering is disabled; SAM post-processing matches the experiment pipeline.")

    if enable_area_filter:
        # Interactive circle-based minimum area selection
        if CANVAS_AVAILABLE:
            st.info("🖱️ **Draw a circle** on the image to set the minimum particle area. Particles smaller than this will be filtered out.")

            # Get ROI image for display
            roi_config = st.session_state.get('roi_config', {})
            particle_roi = roi_config.get('particle_roi')

            if particle_roi and image is not None:
                x1, y1, x2, y2 = particle_roi
                roi_image = image[y1:y2, x1:x2].copy()
            elif image is not None:
                roi_image = image.copy()
            else:
                roi_image = None

            if roi_image is not None:
                # Center the canvas
                col_left, col_center, col_right = st.columns([1, 3, 1])
                
                with col_center:
                    # Clear canvas button
                    if st.button("🗑️ Clear Circle", key="clear_min_area_canvas"):
                        if 'min_area_canvas_key' not in st.session_state:
                            st.session_state['min_area_canvas_key'] = 0
                        st.session_state['min_area_canvas_key'] += 1
                        st.rerun()

                    # Get unique key for canvas
                    min_area_key = f"min_area_canvas_{st.session_state.get('min_area_canvas_key', 0)}"

                    # Convert image for canvas display
                    display_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

                    h, w = roi_image.shape[:2]

                    # Calculate canvas dimensions (centered, reasonable size)
                    canvas_width = min(600, w)
                    scale_factor = canvas_width / w
                    canvas_height = int(h * scale_factor)
                    
                    # Resize image to match canvas size for proper background display
                    resized_image = cv2.resize(display_image, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)
                    pil_image = Image.fromarray(resized_image)

                    # Create canvas for circle drawing
                    min_area_canvas = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.2)",
                        stroke_width=2,
                        stroke_color="#FF0000",
                        background_image=pil_image,
                        update_streamlit=True,
                        height=canvas_height,
                        width=canvas_width,
                        drawing_mode="circle",
                        key=min_area_key,
                    )

                    # Process circle to get minimum area
                    if min_area_canvas.json_data is not None:
                        objects = min_area_canvas.json_data.get("objects", [])
                        if objects:
                            last_circle = objects[-1]
                            if last_circle.get("type") == "circle":
                                # Get circle radius
                                radius_canvas = last_circle.get("radius", 10)
                                # Convert to image coordinates
                                radius_img = radius_canvas / scale_factor
                                # Calculate area
                                min_area = int(np.pi * radius_img ** 2)
                                config['min_particle_area'] = min_area
                                st.success(f"✅ Minimum particle area: **{min_area} px²** (radius ≈ {radius_img:.1f} px)")

                    # Show current value and manual input option
                    st.markdown(f"**Current Min Area:** {config.get('min_particle_area', 10)} px²")
                    
                    with st.expander("📝 Manual Entry", expanded=False):
                        manual_min = st.number_input(
                            "Enter min area (px²)",
                            min_value=1,
                            max_value=10000,
                            value=config.get('min_particle_area', 10),
                            key="manual_min_area"
                        )
                        if st.button("Apply", key="apply_manual_min"):
                            config['min_particle_area'] = manual_min
                            st.rerun()

        else:
            # Fallback to number input (centered)
            st.warning("⚠️ For interactive circle selection, install: `pip install streamlit-drawable-canvas`")

            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                min_area = st.number_input(
                    "Minimum Particle Area (px²)",
                    min_value=1,
                    max_value=10000,
                    value=config.get('min_particle_area', 10),
                    help="Filter out particles smaller than this"
                )
                config['min_particle_area'] = min_area

        st.divider()

    # PF-SUI method summary
    if enable_distribution:
        st.markdown("#### PF-SUI options")
        st.caption(
            "Fixed method: particle-boundary-based Voronoi allocation, centroid "
            "convex-hull observation domain, and a 5-pixel inward inclusion buffer."
        )
        st.divider()

    # Projected-morphology options
    if enable_shape:
        st.markdown("#### 🔷 Projected morphology options")

        # Morphology preset selection
        st.markdown("**Morphology preset:**")
        shape_preset = st.selectbox(
            "Select morphology preset",
            ["2D", "3D", "Custom"],
            index=["2D", "3D", "Custom"].index(config.get('shape_preset', '2D')),
            help="2D: Circle, Triangle, Quadrilateral, Hexagon, Irregular\n"
                 "3D: Spheroid, Pyramid, Hexahedron, Cylinder, Irregular"
        )
        config['shape_preset'] = shape_preset

        # Show preset labels
        preset_labels = {
            '2D': ['Circle', 'Triangle', 'Quadrilateral', 'Hexagon', 'Irregular'],
            '3D': ['Spheroid', 'Pyramid', 'Hexahedron', 'Cylinder', 'Irregular']
        }

        if shape_preset in preset_labels:
            st.info(f"**{shape_preset} morphology labels:** {', '.join(preset_labels[shape_preset])}")
            config['shape_labels'] = preset_labels[shape_preset]
        else:
            # Custom labels
            st.markdown("**Custom morphology labels:**")
            st.markdown("Define the morphological categories for classification:")

            default_custom_labels = ['Circle', 'Triangle', 'Quadrilateral', 'Hexagon', 'Irregular']
            current_labels = config.get('shape_labels', default_custom_labels)

            # Text area for editing labels
            labels_text = st.text_area(
                "Morphology labels (one per line)",
                value="\n".join(current_labels),
                height=150,
                help="Enter shape categories, one per line"
            )

            # Parse labels
            new_labels = [l.strip() for l in labels_text.split("\n") if l.strip()]
            if len(new_labels) >= 2:
                config['shape_labels'] = new_labels
                st.success(f"✅ {len(new_labels)} shape categories defined")
            else:
                st.error("Please define at least 2 shape categories")

        # Keep inference settings aligned with the frozen shape-validation protocol.
        config['clip_model'] = 'ViT-L/14@336px'
        config['clip_batch_size'] = 64
        config['clip_temperature'] = None
        config['clip_confidence_threshold'] = None

    st.divider()

    # Output configuration
    st.markdown("#### 💾 Output configuration")
    current_output_config = st.session_state.get(
        'output_config',
        {key: True for key in OUTPUT_OPTIONS},
    )
    selected_output_labels = st.multiselect(
        "Files available from the Results page",
        options=list(OUTPUT_OPTIONS.values()),
        default=[
            label for key, label in OUTPUT_OPTIONS.items()
            if current_output_config.get(key, True)
        ],
        help=(
            "The complete ZIP contains the full particle table, image summary, "
            "XLSX report, PNG masks/overlays, settings, provenance, and failure manifest."
        ),
    )
    st.session_state['output_config'] = {
        key: label in selected_output_labels
        for key, label in OUTPUT_OPTIONS.items()
    }
    if not selected_output_labels:
        st.warning("No download format is selected; analysis can still run and be viewed in the GUI.")

    # Store config
    normalized_config = normalize_analysis_config(config)
    st.session_state['analysis_config'] = normalized_config
    sync_overview_analysis_selection(normalized_config)

    st.divider()

    # Summary
    st.markdown("#### 📋 Configuration Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown("**Selected Analyses:**")
        if enable_size:
            st.markdown("- ✅ Projected particle area")
        if enable_distribution:
            st.markdown("- ✅ PF-SUI")
        if enable_shape:
            st.markdown("- ✅ Projected morphology")

    with summary_col2:
        # Estimate processing time
        images = st.session_state.get('uploaded_images', [])
        num_images = len(images)

        st.markdown("**Estimated Processing:**")
        st.markdown(f"- Images: {num_images}")

    st.divider()

    # Run Analysis Button
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            update_step(2)
            st.rerun()

    with col2:
        run_disabled = not (enable_size or enable_distribution or enable_shape)

        if st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=run_disabled
        ):
            # Run analysis directly here
            run_analysis_directly()

    with col3:
        pass  # Spacer

    if run_disabled:
        st.warning("Select at least one analysis type to continue")


def run_analysis_directly():
    """Run analysis directly from the config page."""
    from utils.analysis_runner import run_analysis
    from utils.session_state import set_results_for_image

    images = st.session_state.get('uploaded_images', [])
    if not images:
        st.error("❌ No images to analyze. Please upload images first.")
        return

    st.markdown("---")
    st.markdown("### 🔬 Running Analysis...")

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

    status_text.empty()
    progress_bar.empty()

    if error_count == 0:
        st.success(f"✅ Successfully analyzed {success_count} image(s)!")
    else:
        st.warning(f"⚠️ Completed: {success_count} success, {error_count} errors")

    # Update step to results and auto-navigate
    update_step(4)
    st.rerun()
