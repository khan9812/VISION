"""Combined image-upload and analysis-selection screen for VISION."""

import streamlit as st

from components.image_upload import render_compact_upload_panel
from components.analysis_config import (
    get_overview_analysis_config,
    render_analysis_overview_card,
    render_analysis_overview_summary,
)
from utils.session_state import get_current_image, initialize_session_state, update_step

OVERVIEW_CARD_HEIGHT = 520

def render_upload_analysis_setup():
    """Show image upload, module choices, examples, and status in one step."""
    if "uploaded_images" not in st.session_state:
        initialize_session_state()

    st.markdown(
        '<div class="overview-section-title">Image Upload & Analysis Setup</div>',
        unsafe_allow_html=True,
    )

    upload_col, size_col, shape_col, distribution_col, summary_col = st.columns(
        [1, 1, 1, 1, 1]
    )

    with upload_col:
        render_compact_upload_panel(card_height=OVERVIEW_CARD_HEIGHT)
    with size_col:
        render_analysis_overview_card("size", card_height=OVERVIEW_CARD_HEIGHT)
    with shape_col:
        render_analysis_overview_card("shape", card_height=OVERVIEW_CARD_HEIGHT)
    with distribution_col:
        render_analysis_overview_card("distribution", card_height=OVERVIEW_CARD_HEIGHT)
    with summary_col:
        render_analysis_overview_summary(card_height=OVERVIEW_CARD_HEIGHT)

    config = get_overview_analysis_config()
    _, image = get_current_image()
    has_selected_module = any(
        config.get(key, True)
        for key in ("enable_size", "enable_shape", "enable_distribution")
    )
    can_continue = image is not None and has_selected_module

    left, action_col = st.columns([4, 1])
    with action_col:
        if st.button(
            "Continue to ROI & Scale",
            type="primary",
            use_container_width=True,
            disabled=not can_continue,
        ):
            update_step(1)
            st.rerun()
