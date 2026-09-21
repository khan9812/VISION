"""
ROI Selection Component
=======================
Handles Region of Interest selection and scale bar configuration.
Supports interactive drag-based selection for ROI and scale bar.
"""

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

from utils.session_state import (
    get_current_image, update_step, initialize_session_state,
    set_current_image, get_roi_for_image, set_roi_for_image,
    get_images_without_roi, get_images_with_roi, all_images_have_roi,
    get_scale_config, set_scale_config
)
import hashlib

# Try to import streamlit-drawable-canvas for interactive drawing
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


def _get_image_hash(image: np.ndarray) -> str:
    """Get short hash of image for canvas key to handle image changes."""
    return hashlib.md5(image.tobytes()).hexdigest()[:8]


def render_roi_selection():
    """Render the ROI selection and scale bar configuration component."""

    # Ensure session state is initialized
    if 'roi_config' not in st.session_state:
        initialize_session_state()

    st.markdown("### 🎯 ROI & Scale Configuration")

    # Check if image is loaded
    images = st.session_state.get('uploaded_images', [])

    if not images:
        st.warning("⚠️ Please upload an image first in the 'Image Upload' tab.")
        return

    # =========================================================================
    # MULTIPLE IMAGE ROI STATUS
    # =========================================================================
    num_images = len(images)

    if num_images > 1:
        st.info(f"📷 **Multiple Images Mode**: {num_images} images uploaded. "
                "ROI is set per image, Scale Bar is shared across all images.")

        # Show ROI configuration progress
        images_with_roi = get_images_with_roi()
        images_without_roi = get_images_without_roi()

        col_progress1, col_progress2 = st.columns(2)
        with col_progress1:
            st.metric("✅ ROI Configured", len(images_with_roi))
        with col_progress2:
            st.metric("⏳ ROI Pending", len(images_without_roi))

        # Progress bar
        progress = len(images_with_roi) / num_images
        st.progress(progress, text=f"ROI Progress: {len(images_with_roi)}/{num_images}")

        st.divider()

        # Image selector with ROI status
        st.markdown("#### 📷 Select Image to Configure ROI")

        image_options = []
        for img_name, _ in images:
            roi = get_roi_for_image(img_name)
            status = "✅" if roi else "⏳"
            image_options.append(f"{status} {img_name}")

        current_idx = st.session_state.get('current_image_index', 0)

        selected_option = st.selectbox(
            "Choose image",
            image_options,
            index=current_idx,
            help="✅ = ROI configured, ⏳ = ROI pending"
        )

        # Extract index from selection
        new_idx = image_options.index(selected_option)
        if new_idx != current_idx:
            set_current_image(new_idx)
            st.rerun()

    # Get current image
    name, image = get_current_image()

    if image is None:
        st.warning("⚠️ Please upload an image first in the 'Image Upload' tab.")
        return

    st.markdown(f"**Current Image:** {name}")

    h, w = image.shape[:2]

    # Display image dimensions
    st.markdown(f"**Image Dimensions:** {w} x {h} px")

    # Show current ROI status for this image
    current_roi = get_roi_for_image(name)
    if current_roi:
        st.success(f"✅ ROI configured: {current_roi}")
    else:
        st.warning("⏳ ROI not yet configured for this image")

    st.divider()

    # =========================================================================
    # ROI SELECTION - Interactive or Slider-based
    # =========================================================================
    st.markdown("#### 📐 Particle Region of Interest (ROI)")

    if CANVAS_AVAILABLE:
        st.info("🖱️ **Drag a rectangle** on the image below to select the ROI for particle analysis. Click and drag from one corner to the opposite corner.")

        # Button row for canvas controls
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            if st.button("🗑️ Clear Drawing", key="clear_roi_canvas"):
                if 'roi_canvas_key' not in st.session_state:
                    st.session_state['roi_canvas_key'] = 0
                st.session_state['roi_canvas_key'] += 1
                st.session_state['roi_config']['particle_roi'] = None
                st.rerun()
        
        with col_btn2:
            if st.button("🔄 Refresh Canvas", key="refresh_roi_canvas"):
                # Force canvas refresh by incrementing key
                if 'roi_canvas_key' not in st.session_state:
                    st.session_state['roi_canvas_key'] = 0
                st.session_state['roi_canvas_key'] += 1
                st.rerun()

        # Get unique key for canvas (includes image hash for proper refresh on image change)
        image_hash = _get_image_hash(image)
        canvas_key = f"roi_canvas_{st.session_state.get('roi_canvas_key', 0)}_{image_hash}"

        # Convert image for canvas display
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate canvas dimensions (max 700px width, maintain aspect ratio)
        canvas_width = min(700, w)
        scale_factor = canvas_width / w
        canvas_height = int(h * scale_factor)
        
        # Resize image to match canvas size for proper background display
        resized_image = cv2.resize(display_image, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(resized_image)

        # Create canvas for ROI selection with explicit image mode
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=3,
            stroke_color="#00FF00",
            background_image=pil_image,
            background_color="#EEEEEE",  # Fallback background color if image fails
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            display_toolbar=True,
            key=canvas_key,
        )
        
        # Show a small hint if image might not be displaying
        st.caption("💡 If the image is not visible, click the 'Refresh Canvas' button.")

        # Process canvas result
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                # Get the last drawn rectangle
                last_rect = objects[-1]
                if last_rect.get("type") == "rect":
                    # Convert canvas coordinates back to image coordinates
                    left = int(last_rect["left"] / scale_factor)
                    top = int(last_rect["top"] / scale_factor)
                    rect_w = int(last_rect["width"] * last_rect.get("scaleX", 1) / scale_factor)
                    rect_h = int(last_rect["height"] * last_rect.get("scaleY", 1) / scale_factor)

                    roi_x1 = max(0, left)
                    roi_y1 = max(0, top)
                    roi_x2 = min(w, left + rect_w)
                    roi_y2 = min(h, top + rect_h)

                    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                        # Store ROI for this specific image
                        set_roi_for_image(name, (roi_x1, roi_y1, roi_x2, roi_y2))
                        roi_w = roi_x2 - roi_x1
                        roi_h = roi_y2 - roi_y1
                        st.success(f"✅ ROI set for '{name}': ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2}) = {roi_w} x {roi_h} pixels")

        # Fallback: Use full image if no ROI drawn for this image
        if get_roi_for_image(name) is None:
            set_roi_for_image(name, (0, 0, w, h))
            st.info("ℹ️ No ROI drawn. Using full image as ROI.")

    else:
        # Fallback to slider-based ROI selection
        st.warning("⚠️ For interactive drag selection, install: `pip install streamlit-drawable-canvas`")
        st.info("Using slider-based ROI selection instead.")

        col1, col2 = st.columns(2)

        with col1:
            roi_x1 = st.slider("X Start", 0, w-1, 0, key="roi_x1")
            roi_y1 = st.slider("Y Start", 0, h-1, 0, key="roi_y1")

        with col2:
            roi_x2 = st.slider("X End", 0, w, w, key="roi_x2")
            roi_y2 = st.slider("Y End", 0, h, h, key="roi_y2")

        # Validate ROI
        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
            st.error("Invalid ROI: End coordinates must be greater than start coordinates")
        else:
            # Store ROI for this specific image
            set_roi_for_image(name, (roi_x1, roi_y1, roi_x2, roi_y2))

    # Show ROI preview for current image
    particle_roi = get_roi_for_image(name)
    if particle_roi:
        roi_x1, roi_y1, roi_x2, roi_y2 = particle_roi
        roi_image = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()

        if roi_image.size > 0:
            roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

            # Draw ROI on original image
            preview_image = image.copy()
            cv2.rectangle(preview_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 3)
            preview_rgb = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)

            with st.expander("👁️ ROI Preview", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Full Image with ROI:**")
                    st.image(preview_rgb, use_column_width=True)
                with col2:
                    st.markdown("**Cropped ROI:**")
                    st.image(roi_rgb, use_column_width=True)

            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            st.success(f"✅ ROI: {roi_w} x {roi_h} pixels ({roi_w * roi_h:,} total pixels)")

    st.divider()

    # =========================================================================
    # SCALE BAR CONFIGURATION - Shared across all images
    # =========================================================================
    st.markdown("#### 📏 Scale Bar Configuration")

    # Get current scale config (shared)
    scale_cfg = get_scale_config()

    if num_images > 1:
        st.info("📐 **Shared Setting**: Scale bar applies to all images (same experimental conditions).")

    col1, col2 = st.columns(2)

    with col1:
        scale_value = st.number_input(
            "Scale Value",
            min_value=0.1,
            max_value=10000.0,
            value=scale_cfg.get('scale_value', 100.0),
            step=1.0,
            help="The physical length that the scale bar represents (e.g., 100 for '100 nm')"
        )

    with col2:
        scale_unit = st.selectbox(
            "Scale Unit",
            ["nm", "μm", "mm"],
            index=["nm", "μm", "mm"].index(scale_cfg.get('scale_unit', 'nm')),
            help="Physical unit of measurement"
        )

    st.divider()

    # Scale bar length selection - Interactive or Manual
    st.markdown("##### 📐 Scale Bar Length (in pixels)")

    if CANVAS_AVAILABLE:
        st.info("🖱️ **Draw a line** on the image below to measure the scale bar length. Click at one end and drag to the other end of the scale bar.")

        # Button row for canvas controls
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            if st.button("🗑️ Clear Line", key="clear_scale_canvas"):
                if 'scale_canvas_key' not in st.session_state:
                    st.session_state['scale_canvas_key'] = 0
                st.session_state['scale_canvas_key'] += 1
                # Reset scale bar length in shared config
                set_scale_config(scale_value, scale_unit, 100, None)
                st.rerun()
        
        with col_btn2:
            if st.button("🔄 Refresh Canvas", key="refresh_scale_canvas"):
                if 'scale_canvas_key' not in st.session_state:
                    st.session_state['scale_canvas_key'] = 0
                st.session_state['scale_canvas_key'] += 1
                st.rerun()

        # Get unique key for canvas (includes image hash for proper refresh on image change)
        image_hash = _get_image_hash(image)
        scale_canvas_key = f"scale_canvas_{st.session_state.get('scale_canvas_key', 0)}_{image_hash}"

        # Convert image for canvas display
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate canvas dimensions
        canvas_width = min(700, w)
        scale_factor = canvas_width / w
        canvas_height = int(h * scale_factor)
        
        # Resize image to match canvas size for proper background display
        resized_image = cv2.resize(display_image, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(resized_image)

        # Create canvas for scale bar line drawing with explicit background color fallback
        scale_canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=4,
            stroke_color="#FF0000",
            background_image=pil_image,
            background_color="#EEEEEE",
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="line",
            display_toolbar=True,
            key=scale_canvas_key,
        )
        
        st.caption("💡 If the image is not visible, click the 'Refresh Canvas' button.")

        # Process scale bar line - use shared scale config
        scale_bar_length_px = scale_cfg.get('scale_bar_length_px', 100)
        scale_bar_points = scale_cfg.get('scale_bar_points', None)

        if scale_canvas_result.json_data is not None:
            objects = scale_canvas_result.json_data.get("objects", [])
            if objects:
                # Get the last drawn line
                last_line = objects[-1]
                if last_line.get("type") == "line":
                    # Get line endpoints
                    x1 = last_line.get("x1", 0) + last_line.get("left", 0)
                    y1 = last_line.get("y1", 0) + last_line.get("top", 0)
                    x2 = last_line.get("x2", 0) + last_line.get("left", 0)
                    y2 = last_line.get("y2", 0) + last_line.get("top", 0)

                    # Convert to image coordinates
                    x1_img = int(x1 / scale_factor)
                    y1_img = int(y1 / scale_factor)
                    x2_img = int(x2 / scale_factor)
                    y2_img = int(y2 / scale_factor)

                    # Calculate length
                    scale_bar_length_px = int(np.sqrt((x2_img - x1_img)**2 + (y2_img - y1_img)**2))
                    scale_bar_points = ((x1_img, y1_img), (x2_img, y2_img))

                    # Store in shared scale config
                    set_scale_config(scale_value, scale_unit, scale_bar_length_px, scale_bar_points)
                    st.success(f"✅ Scale bar length measured: **{scale_bar_length_px} pixels**")

        # Show current value
        st.markdown(f"**Current scale bar length:** {scale_bar_length_px} pixels")

        # Manual override option
        with st.expander("📝 Manual Entry (Optional)", expanded=False):
            manual_length = st.number_input(
                "Scale Bar Length (pixels)",
                min_value=1,
                max_value=w,
                value=scale_bar_length_px,
                step=1,
                key="manual_scale_length",
                help="Manually enter scale bar length if drag measurement is not accurate"
            )
            if st.button("Use Manual Value"):
                set_scale_config(scale_value, scale_unit, manual_length, scale_bar_points)
                st.rerun()

    else:
        # Fallback to number input
        st.warning("⚠️ For drag-based scale bar measurement, install: `pip install streamlit-drawable-canvas`")

        scale_bar_length_px = st.number_input(
            "Scale Bar Length (pixels)",
            min_value=1,
            max_value=w,
            value=scale_cfg.get('scale_bar_length_px', 100),
            step=1,
            help="Length of the scale bar in pixels - measure this manually from your image"
        )
        set_scale_config(scale_value, scale_unit, scale_bar_length_px, None)

    # Calculate and display pixel-to-real conversion (from shared config)
    scale_cfg = get_scale_config()  # Refresh to get latest values
    scale_bar_length_px = scale_cfg.get('scale_bar_length_px', 100)
    if scale_bar_length_px > 0:
        pixel_to_real = scale_value / scale_bar_length_px
        st.info(f"**Conversion factor:** 1 pixel = {pixel_to_real:.4f} {scale_unit}")
        # Update shared config with final values
        set_scale_config(scale_value, scale_unit, scale_bar_length_px, scale_cfg.get('scale_bar_points'))

    # Navigation
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Back to Upload", use_container_width=True):
            update_step(0)
            st.rerun()

    with col2:
        if st.button("Continue to Preprocessing ➡️", type="primary", use_container_width=True):
            # Validate configuration for all images
            images_without_roi = get_images_without_roi()
            scale_cfg = get_scale_config()

            if images_without_roi:
                if len(images_without_roi) == 1:
                    st.error(f"⚠️ Please set the ROI for: {images_without_roi[0]}")
                else:
                    st.error(f"⚠️ Please set the ROI for {len(images_without_roi)} images: {', '.join(images_without_roi[:3])}{'...' if len(images_without_roi) > 3 else ''}")
            elif scale_cfg.get('pixel_to_real') is None:
                st.error("⚠️ Please configure the scale bar before continuing")
            else:
                update_step(2)
                st.success("✅ ROI and scale configured for all images!")
                st.rerun()
