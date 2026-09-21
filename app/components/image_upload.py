"""
Image Upload Component
======================
Handles single and multiple image uploads for EM analysis.
"""

import streamlit as st
import numpy as np
import cv2
import hashlib
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
ui_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(ui_dir))

from utils.session_state import (
    add_image, remove_image, clear_images,
    get_current_image, set_current_image, update_step,
    initialize_session_state
)


def _upload_signature(uploaded_files):
    """Create a stable signature so widget reruns do not reload the same files."""
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8", errors="replace"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def _store_overview_uploads(uploaded_files):
    """Decode newly selected files for the combined upload/setup screen."""
    signature = _upload_signature(uploaded_files)
    if signature == st.session_state.get("overview_upload_signature"):
        return False, 0

    decoded_images = []
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            decoded_images.append((uploaded_file.name, image))

    if not decoded_images:
        return False, 0

    clear_images()
    st.session_state.pop("overview_current_image", None)
    for name, image in decoded_images:
        add_image(name, image)

    st.session_state["overview_upload_signature"] = signature
    return True, len(decoded_images)


def render_compact_upload_panel(card_height=None):
    """Render the upload panel used beside the analysis module cards.

    Unlike the legacy upload screen, this panel does not advance the workflow
    when a file is selected. The user can choose analysis modules first.
    """
    if "uploaded_images" not in st.session_state:
        initialize_session_state()

    upload_nonce = st.session_state.get("overview_upload_nonce", 0)

    with st.container(border=True, height=card_height):
        st.markdown('<div class="overview-card-title">Image Upload</div>', unsafe_allow_html=True)
        upload_mode = st.radio(
            "Upload mode",
            ["Single image", "Multiple images"],
            horizontal=True,
            key="overview_upload_mode",
            label_visibility="collapsed",
        )

        if upload_mode == "Single image":
            uploaded_file = st.file_uploader(
                "Choose an EM image",
                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                key=f"overview_single_upload_{upload_nonce}",
            )
            uploaded_files = [uploaded_file] if uploaded_file is not None else []
        else:
            uploaded_files = st.file_uploader(
                "Choose EM images",
                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                accept_multiple_files=True,
                key=f"overview_multi_upload_{upload_nonce}",
            )

        if uploaded_files:
            loaded, count = _store_overview_uploads(uploaded_files)
            if loaded:
                suffix = "image" if count == 1 else "images"
                st.success(f"Loaded {count} {suffix}")
            elif st.session_state.get("overview_upload_signature") != _upload_signature(uploaded_files):
                st.error("Unable to decode the selected image files.")

        images = st.session_state.get("uploaded_images", [])
        if not images:
            st.markdown('<div class="overview-summary-value">No image selected</div>', unsafe_allow_html=True)
            return

        if len(images) > 1:
            image_names = [name for name, _ in images]
            current_index = min(st.session_state.get("current_image_index", 0), len(image_names) - 1)
            selected_name = st.selectbox(
                "Current image",
                image_names,
                index=current_index,
                key="overview_current_image",
            )
            set_current_image(image_names.index(selected_name))

        name, image = get_current_image()
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption=name, use_column_width=True)
            st.caption(f"{image.shape[1]} x {image.shape[0]} px")

        if st.button("Clear uploaded images", key="overview_clear_images", use_container_width=True):
            clear_images()
            st.session_state.pop("overview_upload_signature", None)
            st.session_state.pop("overview_current_image", None)
            st.session_state["overview_upload_nonce"] = upload_nonce + 1
            st.rerun()


def render_image_upload():
    """Render the image upload component."""

    # Ensure session state is initialized
    if 'uploaded_images' not in st.session_state:
        initialize_session_state()

    st.markdown("### 📁 Image Upload")
    st.markdown("Upload one or more EM images for analysis. Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF")

    # Upload mode selection
    upload_mode = st.radio(
        "Upload Mode",
        ["Single Image", "Multiple Images"],
        horizontal=True,
        help="Choose whether to upload one image or multiple images for batch processing"
    )

    # File uploader
    if upload_mode == "Single Image":
        uploaded_file = st.file_uploader(
            "Choose an EM image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
            accept_multiple_files=False,
            key="single_upload"
        )

        if uploaded_file is not None:
            # Read and process single image (use getvalue() instead of read() to avoid 403 error)
            file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is not None:
                # Clear existing and add new
                clear_images()
                add_image(uploaded_file.name, image)
                st.success(f"✅ Loaded: {uploaded_file.name} ({image.shape[1]}x{image.shape[0]} pixels)")
                update_step(1)
            else:
                st.error("Failed to decode image. Please try another file.")

    else:  # Multiple Images
        uploaded_files = st.file_uploader(
            "Choose EM images",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
            accept_multiple_files=True,
            key="multi_upload"
        )

        if uploaded_files:
            # Clear existing images
            clear_images()

            success_count = 0
            for uploaded_file in uploaded_files:
                file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if image is not None:
                    add_image(uploaded_file.name, image)
                    success_count += 1

            if success_count > 0:
                st.success(f"✅ Loaded {success_count} image(s)")
                update_step(1)

    # Display uploaded images
    images = st.session_state.get('uploaded_images', [])

    if images:
        st.divider()
        st.markdown("### 🖼️ Uploaded Images")

        # Image selector for multiple images
        if len(images) > 1:
            image_names = [name for name, _ in images]
            current_idx = st.session_state.get('current_image_index', 0)

            col1, col2 = st.columns([3, 1])
            with col1:
                selected_name = st.selectbox(
                    "Select image to view/configure",
                    image_names,
                    index=current_idx
                )
                new_idx = image_names.index(selected_name)
                if new_idx != current_idx:
                    set_current_image(new_idx)

            with col2:
                st.metric("Total Images", len(images))

        # Display current image
        name, image = get_current_image()
        if image is not None:
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(image_rgb, caption=name, use_column_width=True)

            with col2:
                st.markdown("**Image Info:**")
                st.write(f"- **Name:** {name}")
                st.write(f"- **Size:** {image.shape[1]} x {image.shape[0]}")
                st.write(f"- **Channels:** {image.shape[2] if len(image.shape) > 2 else 1}")
                st.write(f"- **Dtype:** {image.dtype}")

                # Image statistics
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
                st.write(f"- **Mean intensity:** {gray.mean():.1f}")
                st.write(f"- **Std intensity:** {gray.std():.1f}")

        # Thumbnail grid for multiple images
        if len(images) > 1:
            st.divider()
            st.markdown("**All Images:**")

            cols_per_row = 4
            rows = (len(images) + cols_per_row - 1) // cols_per_row

            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    img_idx = row * cols_per_row + col_idx
                    if img_idx < len(images):
                        img_name, img = images[img_idx]
                        with cols[col_idx]:
                            # Create thumbnail
                            thumb_size = 150
                            h, w = img.shape[:2]
                            scale = thumb_size / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            thumb = cv2.resize(img, (new_w, new_h))
                            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

                            # Highlight current image
                            current_idx = st.session_state.get('current_image_index', 0)
                            border_color = "🟢" if img_idx == current_idx else "⬜"

                            st.image(thumb_rgb, caption=f"{border_color} {img_name[:20]}...")

                            if st.button(f"Select", key=f"select_{img_idx}"):
                                set_current_image(img_idx)
                                st.rerun()

        # Navigation and Clear buttons
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear All Images", type="secondary", use_container_width=True):
                clear_images()
                st.rerun()

        with col2:
            if st.button("Continue to ROI ➡️", type="primary", use_container_width=True):
                update_step(1)
                st.rerun()

    else:
        # Show placeholder
        st.info("👆 Upload EM image(s) to begin analysis")

        # Show example images if available
        example_dir = parent_dir / "upload"
        if example_dir.exists():
            example_images = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
            if example_images:
                st.markdown("---")
                st.markdown("**Or load from examples:**")

                cols = st.columns(min(4, len(example_images)))
                for i, img_path in enumerate(example_images[:4]):
                    with cols[i]:
                        if st.button(f"📷 {img_path.name[:15]}...", key=f"example_{i}"):
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                clear_images()
                                add_image(img_path.name, img)
                                update_step(1)
                                st.rerun()
