"""
VISION - Versatile Intelligent Segmentation for Image-based Observation of Nanoparticles
===========================================================================================
A comprehensive Streamlit-based user interface for TEM nanoparticle characterization.

Features:
- Image upload (single or multiple images)
- ROI selection and scale bar configuration
- Preprocessing (BM3D → Noise2SR; optional CLAHE)
- SAM 2.1 segmentation
- Projected particle area, Projected morphology, and PF-SUI analysis
- Interactive results visualization
"""

import streamlit as st
import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path

# Add UI and project directories to path for module imports.
ui_dir = Path(__file__).parent
parent_dir = ui_dir.parent
sys.path.insert(0, str(ui_dir))
sys.path.insert(0, str(parent_dir))


# ============================================================================
# SYSTEM STATUS CHECK
# ============================================================================
@st.cache_resource
def check_system_status():
    """Check GPU and module availability."""
    status = {
        'gpu_available': False,
        'gpu_name': None,
        'gpu_memory': None,
        'torch_available': False,
        'clip_available': False,
        'sam2_available': False,
        'sam2_checkpoint_available': False,
        'bm3d_available': False,
        'modules_available': False,
        'errors': []
    }
    
    # Check PyTorch and GPU
    try:
        import torch
        status['torch_available'] = True
        if torch.cuda.is_available():
            status['gpu_available'] = True
            status['gpu_name'] = torch.cuda.get_device_name(0)
            status['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    except ImportError:
        status['errors'].append("PyTorch not installed")
    
    # Check CLIP
    try:
        import clip
        status['clip_available'] = True
    except ImportError:
        status['errors'].append("CLIP not installed (run: pip install git+https://github.com/openai/CLIP.git)")
    
    # Check SAM2
    try:
        from sam2.build_sam import build_sam2
        status['sam2_available'] = True
        checkpoint_path = parent_dir / 'checkpoints' / 'sam2.1_hiera_large.pt'
        status['sam2_checkpoint_available'] = checkpoint_path.is_file()
        if not status['sam2_checkpoint_available']:
            status['errors'].append(
                "SAM2 checkpoint missing: checkpoints/sam2.1_hiera_large.pt"
            )
    except ImportError:
        status['errors'].append("SAM2 not found in path")
    
    # Check BM3D
    try:
        import bm3d
        status['bm3d_available'] = True
    except ImportError:
        status['errors'].append("BM3D not installed (run: pip install bm3d)")
    
    # Check analysis modules
    try:
        from modules.preprocessing import auto_preprocess
        from modules.size_analysis import calculate_size_metrics
        from modules.shape_analysis import classify_shapes_with_clip
        status['modules_available'] = True
    except ImportError as e:
        status['errors'].append(f"Analysis modules: {str(e)}")
    
    return status


def render_system_status(status):
    """Render system status in sidebar."""
    st.subheader("🖥️ System Status")
    
    # GPU Status
    if status['gpu_available']:
        st.success(f"✅ GPU: {status['gpu_name']}")
        st.caption(f"   Memory: {status['gpu_memory']}")
    else:
        st.warning("⚠️ GPU: Not available (CPU mode)")
        st.caption("   Analysis will be slower")
    
    # Module Status
    col1, col2 = st.columns(2)
    with col1:
        if status['torch_available']:
            st.markdown("✅ PyTorch")
        else:
            st.markdown("❌ PyTorch")
        
        if status['clip_available']:
            st.markdown("✅ CLIP")
        else:
            st.markdown("❌ CLIP")
    
    with col2:
        if status['sam2_available'] and status['sam2_checkpoint_available']:
            st.markdown("✅ SAM2")
        elif status['sam2_available']:
            st.markdown("⚠️ SAM2 checkpoint")
        else:
            st.markdown("❌ SAM2")
        
        if status['bm3d_available']:
            st.markdown("✅ BM3D")
        else:
            st.markdown("❌ BM3D")
    
    # Show errors if any
    if status['errors']:
        with st.expander("⚠️ Missing Components", expanded=False):
            for error in status['errors']:
                st.caption(f"• {error}")


# Import UI components
from components.upload_analysis_setup import render_upload_analysis_setup
from components.roi_selection import render_roi_selection
from components.preprocessing_options import render_preprocessing_options
from components.analysis_config import render_analysis_config
from components.results_display import render_results_display
from utils.session_state import apply_all_config, get_all_config, initialize_session_state
from utils.analysis_runner import run_analysis

# Application version
__version__ = "1.1.0"
__app_name__ = "VISION"
__app_full_name__ = "Versatile Intelligent Segmentation for Image-based Observation of Nanoparticles"

# Page configuration
st.set_page_config(
    page_title="VISION - Nanoparticle Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .app-name {
        font-size: 1.85rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.25rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
    }
    .stButton>button {
        width: 100%;
    }
    div[data-testid="stMain"] .stButton > button {
        min-height: 3.15rem;
        padding-left: 0.35rem;
        padding-right: 0.35rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    div[data-testid="stMain"] [data-testid="stCheckbox"] label p,
    div[data-testid="stMain"] [data-testid="stRadio"] label p {
        font-size: 1.22rem !important;
    }
    div[data-testid="stMain"] [data-testid="stImage"] figcaption {
        font-size: 1.05rem !important;
        line-height: 1.25;
    }
    .overview-section-title {
        font-size: 2.15rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0.1rem 0 1rem;
    }
    .overview-card-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #172554;
        margin-bottom: 0.5rem;
    }
    .overview-detail {
        font-size: 1.35rem;
        line-height: 1.45;
        color: #1f2937;
        margin: 0.35rem 0;
    }
    .overview-summary-label {
        font-size: 1.15rem;
        font-weight: 700;
        color: #334155;
        margin-top: 0.35rem;
    }
    .overview-summary-value {
        font-size: 1.2rem;
        line-height: 1.35;
        color: #475569;
        margin-top: 0.1rem;
    }
    .sidebar-workflow-step {
        font-size: 1.1rem;
        line-height: 1.3;
        color: #334155;
        margin: 0.2rem 0;
        padding: 0.35rem 0.45rem;
        border-radius: 4px;
    }
    .sidebar-workflow-step.complete {
        color: #166534;
    }
    .sidebar-workflow-step.current {
        color: #1d4ed8;
        font-weight: 700;
        background-color: #e8f1ff;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        font-size: 1.12rem !important;
        font-weight: 600;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        display: block;
        font-size: 0 !important;
        margin-top: 0.25rem;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] small::after {
        content: "Limit 1GB per file\A PNG, JPG, JPEG, BMP, TIF, TIFF";
        display: block;
        white-space: pre-line;
        font-size: 1.02rem;
        font-weight: 400;
        line-height: 1.35;
    }
    .vision-logo {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Header with VISION branding
    st.markdown('<div class="vision-logo">🔬</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">VISION</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-name">Versatile Intelligent Segmentation for Image-based Observation of Nanoparticles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated nanoparticle characterization with SAM 2.1, CLIP, and PF-SUI analysis</div>', unsafe_allow_html=True)

    # Check system status
    system_status = check_system_status()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔬 VISION")
        st.caption(f"v{__version__}")
        st.divider()
        
        # System status
        render_system_status(system_status)
        st.divider()

        st.title("📋 Analysis Workflow")

        # Progress indicator
        steps = [
            "1. Image Upload & Setup",
            "2. ROI & Scale",
            "3. Preprocessing",
            "4. Analysis Config",
            "5. Results",
        ]
        current_step = st.session_state.get('current_step', 0)

        for i, step in enumerate(steps):
            step_state = "complete" if i < current_step else "current" if i == current_step else "pending"
            st.markdown(
                f'<div class="sidebar-workflow-step {step_state}">{step}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Quick actions
        st.subheader("Quick Actions")
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.rerun()

        st.download_button(
            "Save Settings JSON",
            data=json.dumps(get_all_config(), indent=2, ensure_ascii=True),
            file_name="vision_settings.json",
            mime="application/json",
            use_container_width=True,
        )
        uploaded_config = st.file_uploader(
            "Load Settings JSON",
            type=["json"],
            accept_multiple_files=False,
            key="settings_json_uploader",
        )
        if st.button(
            "Apply Loaded Settings",
            disabled=uploaded_config is None,
            use_container_width=True,
        ):
            try:
                loaded_config = json.loads(uploaded_config.getvalue().decode("utf-8"))
                apply_all_config(loaded_config)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                st.error(f"Could not load settings: {exc}")
            else:
                st.rerun()

        config_load_message = st.session_state.pop('config_load_message', None)
        if config_load_message:
            st.success(config_load_message)

        st.divider()

        # Info
        st.subheader("ℹ️ About VISION")
        st.markdown("""
        **Analysis Types:**
        - **Projected particle area**: Scale-calibrated projected-area statistics
        - **Projected morphology**: CLIP-based projected-morphology classification
        - **PF-SUI**: Particle-boundary-based Voronoi within a centroid convex hull

        **Supports:** Single or batch image processing

        **Powered by:**
        - SAM 2.1 (Segmentation)
        - CLIP (Projected-morphology classification)
        - BM3D → Noise2SR (Denoising)
        """)

    # Main content - Step-based navigation (not tabs, for programmatic control)
    current_step = st.session_state.get('current_step', 0)

    step_names = [
        "Image Upload & Setup",
        "ROI & Scale",
        "Preprocessing",
        "Analysis Config",
        "Results",
    ]

    # Step navigation buttons at the top
    st.markdown("---")
    cols = st.columns(5)
    for i, (col, name) in enumerate(zip(cols, step_names)):
        with col:
            # Highlight current step
            if i == current_step:
                if st.button(name, key=f"nav_{i}", use_container_width=True, type="primary"):
                    pass  # Already on this step
            else:
                if st.button(name, key=f"nav_{i}", use_container_width=True):
                    st.session_state['current_step'] = i
                    st.rerun()
    st.markdown("---")

    # Render content based on current step
    if current_step == 0:
        render_upload_analysis_setup()
    elif current_step == 1:
        render_roi_selection()
    elif current_step == 2:
        render_preprocessing_options()
    elif current_step == 3:
        render_analysis_config()
    elif current_step == 4:
        render_results_display()

    # Footer
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <strong>VISION</strong> v{__version__} - Versatile Intelligent Segmentation for Image-based Observation of Nanoparticles<br>
        Powered by SAM 2.1, CLIP, and Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
