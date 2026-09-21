"""Execution-provenance records for VISION exports."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Optional


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "opencv": "opencv-contrib-python",
    "pillow": "Pillow",
    "matplotlib": "matplotlib",
    "scikit-learn": "scikit-learn",
    "scikit-image": "scikit-image",
    "shapely": "shapely",
    "hdbscan": "hdbscan",
    "bm3d": "bm3d",
    "streamlit": "streamlit",
    "sam2": "sam-2",
    "openai-clip": "openai-clip",
    "torch": "torch",
    "torchvision": "torchvision",
}


def _distribution_version(distribution_name: str) -> Optional[str]:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str:
    """Return the checked-out commit without making Git a runtime requirement."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _runtime_details() -> Dict[str, Any]:
    runtime: Dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "packages": {
            name: _distribution_version(distribution)
            for name, distribution in PACKAGE_DISTRIBUTIONS.items()
        },
        "torch": {
            "available": False,
            "version": _distribution_version("torch"),
            "cuda_available": False,
            "cuda_version": None,
            "cudnn_version": None,
            "gpu": None,
        },
    }
    try:
        import torch

        runtime["torch"].update(
            {
                "available": True,
                "version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "cudnn_version": (
                    torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available()
                    else None
                ),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError:
        pass
    return runtime


def collect_execution_provenance(
    *,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    sam_config: Optional[Dict[str, Any]] = None,
    analysis_config: Optional[Dict[str, Any]] = None,
    preprocessing_info: Optional[Dict[str, Any]] = None,
    pipeline_cache_status: str = "not_checked",
    pipeline_cache_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Create the machine-readable provenance record promised by the GUI."""
    preprocessing_config = dict(preprocessing_config or {})
    sam_config = dict(sam_config or {})
    analysis_config = dict(analysis_config or {})
    preprocessing_info = dict(preprocessing_info or {})

    return {
        "provenance_schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": {
            "application": "VISION",
            "application_version": "1.1.0",
            "git_commit": _git_commit(),
        },
        "models": {
            "segmentation": {
                "framework": "SAM 2.1",
                "model": "Hiera Large",
                "model_type": sam_config.get("model_type", "hiera_l"),
                "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
                "config": "sam2 package resource: configs/sam2.1/sam2.1_hiera_l.yaml",
            },
            "projected_morphology": {
                "framework": "OpenAI CLIP",
                "model": analysis_config.get("clip_model", "ViT-L/14@336px"),
                "batch_size": int(analysis_config.get("clip_batch_size", 64)),
                "pretrained_logit_scale": analysis_config.get("clip_temperature") is None,
                "confidence_threshold": analysis_config.get("clip_confidence_threshold"),
            },
        },
        "random_seed": SEED,
        "runtime": _runtime_details(),
        "preprocessing": {
            "requested": preprocessing_config,
            "executed": preprocessing_info,
            "cache_provenance": {
                "full_pipeline_cache_status": pipeline_cache_status,
                "cache_key": pipeline_cache_key,
                "policy": (
                    "Preprocessing is recomputed on a signed full-pipeline cache miss "
                    "and reused only when image content, ROI, scale, preprocessing, SAM, "
                    "analysis, and post-processing signatures match."
                ),
            },
        },
        "failure_handling": {
            "missing_dependency": "Fail the selected analysis; do not substitute surrogate measurements.",
            "requested_preprocessing_unavailable": "Fail before segmentation.",
            "sam_exception": "Fail the image; do not substitute OpenCV segmentation.",
            "zero_sam_masks": "Report no particles detected; do not compute downstream metrics.",
            "selected_module_exception": "Mark the image failed; do not emit fabricated module values.",
        },
    }

