#%%
# ============================================================================
# FULL PREPROCESSING COMPARISON WITH SAM + CLIP
# ============================================================================
# Compare raw input and eight preprocessing strategies with the full pipeline.
# Existing BM3D-first cache entries are reused method by method, while the
# Noise2SR-only stage is shared by both Noise2SR-first child pipelines.
#
# For each method, measure:
# - SAM performance (masks, IoU, stability)
# - CLIP performance (confidence, diversity)
# - Image quality (SNR, contrast)
# - Final analysis quality (particle count, filtering efficiency)
# This script runs full end-to-end comparison (not preprocessing-only).
# ============================================================================


def get_sam2_config_path(config_name):
    """Get absolute path to SAM2 config from installed package"""
    import sam2, os
    sam2_path = os.path.dirname(sam2.__file__)
    return os.path.join(sam2_path, 'configs', 'sam2.1', config_name)

print("="*80)
print("FULL PREPROCESSING COMPARISON: SAM + CLIP ANALYSIS")
print("="*80)

# Core libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import sys
import argparse
import gc
import copy
import torch
from pathlib import Path
from scipy import stats
from scipy.stats import entropy
import pandas as pd
import json
import pickle
import base64
import zlib
import csv
from collections import Counter
from scipy.optimize import linear_sum_assignment

# Add parent directory (Entire_Framework) to path for module imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Entire_Framework/
sys.path.insert(0, PROJECT_ROOT)

# SAM imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# CLIP imports
import clip

# Import from modules (now accessible via PROJECT_ROOT in sys.path)
from modules.preprocessing import apply_clahe, load_parameters, preprocess_with_config
from modules.shape_analysis import classify_shapes_with_clip, generate_shape_colors
from modules.color_palette import PALETTE_NEUTRAL, palette_hex
from modules.project_paths import get_optimization_dir
from modules.sam2_utils import (
    filter_background_masks as filter_sam_background_masks,
    filter_overlapping_masks_by_centroid,
)

print("\n[OK] All imports successful!")
print(f"[INFO] Working directory: {os.getcwd()}")

DEFAULT_PREPROCESS_OUTPUT_DIR = str(get_optimization_dir("preprocessing_comparison_full"))
DEFAULT_OPTIMIZER_BM3D_CACHE_DIR = str(
    Path(get_optimization_dir("sam_optimization_v2"))
    / "bm3d_only_gt_disjoint"
    / "preprocessed_bm3d_sigma40"
)
DEFAULT_OPTIMIZER_SPLIT_MANIFEST = str(
    Path(get_optimization_dir("sam_optimization_v2"))
    / "bm3d_only_gt_disjoint"
    / "optimization_split_manifest.json"
)
DEFAULT_REFERENCE_ANN_DIR = os.path.join(PROJECT_ROOT, "emps-DatasetNinja", "ds", "ann")
DEFAULT_REFERENCE_IMAGE_DIR = os.path.join(PROJECT_ROOT, "emps-DatasetNinja", "ds", "img")
REFERENCE_MATCH_IOU_THRESHOLD = 0.5
REFERENCE_BORDER_MARGIN = 5
ORIGIN_FIGURE_WORKBOOK_NAME = "origin_2026_figure_A_B_D.xlsx"
ORIGIN_CONFIDENCE_LEVEL = 0.95
ORIGIN_TIE_TOLERANCE = 1e-12
REFERENCE_ERROR_COLORS = {
    # Keep correct regions subtle so FP/FN stand out clearly on dark TEM backgrounds.
    "correct": (0.32, 0.82, 0.36, 0.22),
    "false_positive": (1.00, 0.18, 0.10, 0.88),
    "false_negative": (0.08, 0.78, 1.00, 0.94),
}

# Keep the original method keys stable so existing cache entries remain valid.
# The insertion order also guarantees that reusable parent stages are available
# before their dependent methods are processed.
PREPROCESSING_METHODS = {
    "0_raw": {
        "name": "Raw image",
        "bm3d": False,
        "noise2sr": False,
        "clahe": False,
        "reuse_from": None,
        "reuse_stage": None,
    },
    "1_bm3d": {
        "name": "BM3D only",
        "bm3d": True,
        "noise2sr": False,
        "clahe": False,
        "reuse_from": None,
        "reuse_stage": None,
    },
    "2_bm3d_noise2sr": {
        "name": "BM3D + Noise2SR",
        "bm3d": True,
        "noise2sr": True,
        "clahe": False,
        "reuse_from": "1_bm3d",
        "reuse_stage": "noise2sr",
    },
    "3_bm3d_noise2sr_clahe": {
        "name": "BM3D + Noise2SR + CLAHE",
        "bm3d": True,
        "noise2sr": True,
        "clahe": True,
        "reuse_from": "2_bm3d_noise2sr",
        "reuse_stage": "clahe",
    },
    "4_bm3d_clahe": {
        "name": "BM3D + CLAHE",
        "bm3d": True,
        "noise2sr": False,
        "clahe": True,
        "reuse_from": "1_bm3d",
        "reuse_stage": "clahe",
    },
    "5_noise2sr": {
        "name": "Noise2SR Only",
        "bm3d": False,
        "noise2sr": True,
        "clahe": False,
        "reuse_from": None,
        "reuse_stage": None,
    },
    "6_clahe": {
        "name": "CLAHE only",
        "bm3d": False,
        "noise2sr": False,
        "clahe": True,
        "reuse_from": None,
        "reuse_stage": None,
    },
    "7_noise2sr_bm3d": {
        "name": "Noise2SR > BM3D",
        "bm3d": True,
        "noise2sr": True,
        "clahe": False,
        "reuse_from": "5_noise2sr",
        "reuse_stage": "bm3d",
    },
    "8_noise2sr_clahe": {
        "name": "Noise2SR > CLAHE",
        "bm3d": False,
        "noise2sr": True,
        "clahe": True,
        "reuse_from": "5_noise2sr",
        "reuse_stage": "clahe",
    },
}

METHOD_KEYS = tuple(PREPROCESSING_METHODS)
LEGACY_METHOD_KEYS = (
    "1_bm3d",
    "2_bm3d_noise2sr",
    "3_bm3d_noise2sr_clahe",
    "4_bm3d_clahe",
)
ADDITIONAL_METHOD_KEYS = tuple(key for key in METHOD_KEYS if key not in LEGACY_METHOD_KEYS)
# The primary comparison includes the unprocessed image as the baseline plus
# the four BM3D-first methods. Noise2SR-first variants remain opt-in through
# --method-keys.
PUBLICATION_METHOD_KEYS = ("0_raw", *LEGACY_METHOD_KEYS)
METHOD_LABELS = {key: config["name"] for key, config in PREPROCESSING_METHODS.items()}
METHOD_STAGE_ORDER = {
    "0_raw": (),
    "1_bm3d": ("bm3d",),
    "2_bm3d_noise2sr": ("bm3d", "noise2sr"),
    "3_bm3d_noise2sr_clahe": ("bm3d", "noise2sr", "clahe"),
    "4_bm3d_clahe": ("bm3d", "clahe"),
    "5_noise2sr": ("noise2sr",),
    "6_clahe": ("clahe",),
    "7_noise2sr_bm3d": ("noise2sr", "bm3d"),
    "8_noise2sr_clahe": ("noise2sr", "clahe"),
}
METHOD_COLORS_BY_KEY = {
    "0_raw": "#7F7F7F",
    "1_bm3d": "#BF4E58",
    "2_bm3d_noise2sr": "#4971A6",
    "3_bm3d_noise2sr_clahe": "#56A662",
    "4_bm3d_clahe": "#D97652",
    "5_noise2sr": "#17BECF",
    "6_clahe": "#9467BD",
    "7_noise2sr_bm3d": "#8C564B",
    "8_noise2sr_clahe": "#E377C2",
}
METHOD_COLORS = [METHOD_COLORS_BY_KEY[key] for key in METHOD_KEYS]


def _parse_runtime_args():
    """Parse runtime args while keeping notebook/interactive compatibility."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Process only one image (filename stem/name or absolute path).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Dataset"),
        help="Dataset directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: preprocessing_comparison_full).",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only build aggregate summary graph/statistics from existing cache files, then exit.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Show which method keys are missing per image without loading SAM/CLIP models.",
    )
    parser.add_argument(
        "--reuse-preprocessed-only",
        action="store_true",
        help=(
            "Re-evaluate SAM2/metrics from existing *_preprocessed.png stages only. "
            "The cohort is taken from existing *_cache.json files in --output-dir; "
            "BM3D, Noise2SR, and CLAHE are never executed. Raw uses the source image."
        ),
    )
    parser.add_argument(
        "--origin-workbook-only",
        action="store_true",
        help="Only build the Origin 2026 Figure A/B/D workbook from existing cache files, then exit.",
    )
    parser.add_argument(
        "--save-individual-avg-plots",
        action="store_true",
        help="Also save the legacy one-file-per-metric avg_*.png plots.",
    )
    parser.add_argument(
        "--reference-ann-dir",
        type=str,
        default=DEFAULT_REFERENCE_ANN_DIR,
        help="Reference annotation directory (DatasetNinja bitmap JSON files).",
    )
    parser.add_argument(
        "--reference-image-dir",
        type=str,
        default=DEFAULT_REFERENCE_IMAGE_DIR,
        help="Reference full-image directory used for crop alignment.",
    )
    parser.add_argument(
        "--method-keys",
        nargs="+",
        choices=tuple(PREPROCESSING_METHODS),
        default=None,
        help=(
            "Methods to process and summarize. The default is Raw plus the "
            "four BM3D-first publication methods."
        ),
    )
    parser.add_argument(
        "--dataset-scope",
        choices=("optimizer-disjoint", "all"),
        default="optimizer-disjoint",
        help=(
            "Use the frozen optimizer-development split (default, 256 images) or "
            "every image in --dataset-dir. Single-image mode is unaffected."
        ),
    )
    parser.add_argument(
        "--optimizer-split-manifest",
        type=str,
        default=DEFAULT_OPTIMIZER_SPLIT_MANIFEST,
        help="optimization_split_manifest.json generated by sam_param_optimizer.py.",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.95,
        help="SAM2 predicted-IoU threshold selected by the optimizer.",
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.80,
        help="SAM2 stability-score threshold selected by the optimizer.",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=None,
        help=(
            "Optional exact post-scope image-count guard. For optimizer-disjoint "
            "scope the manifest optimization_count is always enforced."
        ),
    )
    parser.add_argument(
        "--optimizer-bm3d-cache-dir",
        type=str,
        default=DEFAULT_OPTIMIZER_BM3D_CACHE_DIR,
        help=(
            "BM3D-only images prepared by sam_param_optimizer.py. These are used "
            "as the 1_bm3d parent stage when the comparison output is missing."
        ),
    )
    args, _ = parser.parse_known_args()
    for name, value in (
        ("--pred-iou-thresh", args.pred_iou_thresh),
        ("--stability-score-thresh", args.stability_score_thresh),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")
    if args.expected_images is not None and args.expected_images < 1:
        parser.error("--expected-images must be at least one")
    return args


ARGS = _parse_runtime_args()

if ARGS.method_keys:
    # Preserve CLI order while rejecting accidental duplicates.
    METHOD_KEYS = tuple(dict.fromkeys(ARGS.method_keys))
else:
    METHOD_KEYS = PUBLICATION_METHOD_KEYS
METHOD_COLORS = [METHOD_COLORS_BY_KEY[key] for key in METHOD_KEYS]


def _load_optimizer_split_manifest(path):
    """Load and validate the frozen development-set image IDs."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Optimizer split manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid optimizer split manifest: {manifest_path}") from exc
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Optimizer split manifest has no rows list: {manifest_path}")
    included_ids = {
        str(row.get("Image_ID"))
        for row in rows
        if bool(row.get("Optimization_Included"))
    }
    included_ids.discard("None")
    expected_count = int(payload.get("optimization_count", len(included_ids)))
    if len(included_ids) != expected_count:
        raise ValueError(
            f"Optimizer split manifest count mismatch: {len(included_ids)} IDs vs "
            f"optimization_count={expected_count}"
        )
    return {
        "path": str(manifest_path),
        "split_hash": payload.get("split_hash"),
        "dataset_count": payload.get("dataset_count"),
        "optimization_count": expected_count,
        "included_ids": included_ids,
    }


ACTIVE_SPLIT = (
    _load_optimizer_split_manifest(ARGS.optimizer_split_manifest)
    if ARGS.dataset_scope == "optimizer-disjoint" and not ARGS.image
    else None
)


def _filter_dataframe_to_active_split(dataframe, image_column="image_name"):
    """Keep only frozen development-set rows for publication aggregation."""
    if ACTIVE_SPLIT is None or dataframe.empty:
        return dataframe.copy()
    if image_column not in dataframe.columns:
        raise ValueError(f"Missing image ID column for split filtering: {image_column}")
    out = dataframe.copy()
    out[image_column] = out[image_column].map(str)
    return out.loc[out[image_column].isin(ACTIVE_SPLIT["included_ids"])].copy()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ============================================================================
# UNIFIED SAM2 PARAMETERS
# ============================================================================
# Implementation: SAM 2.1 (Ravi et al., "SAM 2", 2024), not SAM1.
# Unified across all validation scripts for reproducibility
SAM_CONFIG = {
    "points_per_side": 32,                    # Standard grid density
    "points_per_batch": 256,                   # Batch size for point processing
    "pred_iou_thresh": float(ARGS.pred_iou_thresh),
    "stability_score_thresh": float(ARGS.stability_score_thresh),
    "crop_n_layers": 1,                       # Multi-scale cropping
    "crop_n_points_downscale_factor": 2,      # Point density in crops
    "crop_nms_thresh": 0.7,                   # NMS threshold for crops
    "box_nms_thresh": 0.7,                    # NMS threshold for boxes
    "use_m2m": True                           # Mask-to-mask refinement
}


def ensure_even_dimensions(image):
    """
    Ensure image dimensions are even (required for Noise2SR pixel_unshuffle).

    Args:
        image: Input image (numpy array)

    Returns:
        Image with even dimensions (may be slightly cropped)
    """
    h, w = image.shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return image[:new_h, :new_w]


def _complete_preprocessing_info(info, method_key):
    """Fill stable stage-order metadata without discarding measured parameters."""
    completed = copy.deepcopy(info) if info else {}
    stage_order = list(METHOD_STAGE_ORDER[method_key])
    completed["preprocessing_order"] = stage_order
    completed["denoising_method"] = ">".join(stage_order) if stage_order else "raw"
    completed.setdefault("bm3d_applied", "bm3d" in stage_order)
    completed.setdefault("noise2sr_applied", "noise2sr" in stage_order)
    completed.setdefault("clahe_applied", "clahe" in stage_order)
    completed.setdefault("bm3d_info", {})
    completed.setdefault("noise2sr_info", {})
    return completed


def apply_clahe_to_preprocessed(base_image, base_info, reused_from, method_key, verbose=True):
    """Apply CLAHE to a cached parent stage and retain its metadata."""
    params = load_parameters()
    clahe_config = params.get('clahe', {})
    clip_limit = clahe_config.get('clip_limit', 4.0)
    tile_grid_size = tuple(clahe_config.get('tile_grid_size', [64, 64]))

    if verbose:
        print(f"\nSTEP: Reusing {METHOD_LABELS[reused_from]} output, applying CLAHE only...")
        print(f"   Clip limit: {clip_limit}")
        print(f"   Tile grid: {tile_grid_size}")

    preprocessed = apply_clahe(base_image.copy(), clip_limit, tile_grid_size)

    info = copy.deepcopy(base_info)
    info['clahe_applied'] = True
    info['clahe_params'] = {
        'clip_limit': clip_limit,
        'tile_grid_size': list(tile_grid_size),
    }
    info['reused_preprocessing_from'] = reused_from
    info = _complete_preprocessing_info(info, method_key)

    if verbose:
        print("   CLAHE complete")

    return preprocessed, info


def apply_bm3d_to_preprocessed(base_image, base_info, reused_from, method_key, verbose=True):
    """Apply BM3D after a cached parent stage without rerunning that parent."""
    if verbose:
        print(f"\nSTEP: Reusing {METHOD_LABELS[reused_from]} output, applying BM3D only...")

    preprocessed, bm3d_stage_info = preprocess_with_config(
        base_image.copy(),
        bm3d_enabled=True,
        noise2sr_enabled=False,
        clahe_enabled=False,
        verbose=verbose,
    )
    if not bm3d_stage_info.get("bm3d_applied"):
        raise RuntimeError("BM3D was requested after Noise2SR but was not applied")

    info = copy.deepcopy(base_info)
    info["bm3d_applied"] = True
    info["bm3d_info"] = copy.deepcopy(bm3d_stage_info.get("bm3d_info", {}))
    if "sigma_psd" in bm3d_stage_info:
        info["sigma_psd"] = bm3d_stage_info["sigma_psd"]
    info["reused_preprocessing_from"] = reused_from
    info = _complete_preprocessing_info(info, method_key)
    return preprocessed, info


def apply_noise2sr_to_preprocessed(base_image, base_info, reused_from, method_key, verbose=True):
    """Apply Noise2SR to an existing BM3D stage without repeating BM3D."""
    if verbose:
        print(f"\nSTEP: Reusing {METHOD_LABELS[reused_from]} output, applying Noise2SR only...")

    preprocessed, noise2sr_stage_info = preprocess_with_config(
        base_image.copy(),
        bm3d_enabled=False,
        noise2sr_enabled=True,
        clahe_enabled=False,
        verbose=verbose,
    )
    if not noise2sr_stage_info.get("noise2sr_applied"):
        raise RuntimeError("Noise2SR was requested after BM3D but was not applied")

    info = copy.deepcopy(base_info)
    info["noise2sr_applied"] = True
    info["noise2sr_info"] = copy.deepcopy(
        noise2sr_stage_info.get("noise2sr_info", {})
    )
    info["reused_preprocessing_from"] = reused_from
    info = _complete_preprocessing_info(info, method_key)
    return preprocessed, info

# ============================================================================
# REPRODUCIBILITY - SEED FIXING
# ============================================================================
print("\n" + "="*80)
print("REPRODUCIBILITY SETUP")
print("="*80)

SEED = 42

# Python built-in random
import random
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# OpenCV
cv2.setRNGSeed(SEED)

# CuDNN deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"\n[OK] All random seeds fixed to: {SEED}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_preprocessed_path(image_name, method_key, output_dir=None):
    """Return the persistent preprocessing-stage image path for one method."""
    if output_dir is None:
        output_dir = DEFAULT_PREPROCESS_OUTPUT_DIR
    return os.path.join(output_dir, f"{image_name}_{method_key}_preprocessed.png")


def get_optimizer_bm3d_stage_path(image_name):
    """Return the optimizer's reusable BM3D-only stage for one image."""
    return os.path.join(
        str(Path(ARGS.optimizer_bm3d_cache_dir).expanduser().resolve()),
        f"{image_name}_1_bm3d_preprocessed.png",
    )


def has_reusable_preprocessing_stage(image_name, method_key, output_dir):
    """Report whether preprocessing can be loaded without recomputing the stage."""
    if os.path.exists(get_preprocessed_path(image_name, method_key, output_dir)):
        return True
    return method_key == "1_bm3d" and os.path.exists(
        get_optimizer_bm3d_stage_path(image_name)
    )


def _read_cache_data(image_name, output_dir=None, warn=False):
    """Read cache without emitting normal progress output."""
    if output_dir is None:
        output_dir = DEFAULT_PREPROCESS_OUTPUT_DIR
    cache_path = os.path.join(output_dir, f"{image_name}_cache.json")
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        if warn:
            print(f"[WARN] Failed to load cache for {image_name}: {exc}")
        return {}


def _cache_entry_matches_sam_config(entry):
    """Return True only for cache rows created with the active SAM settings."""
    if not isinstance(entry, dict):
        return False
    cached_config = entry.get("sam_config")
    if not isinstance(cached_config, dict):
        return False

    for key, expected in SAM_CONFIG.items():
        if key not in cached_config:
            return False
        actual = cached_config[key]
        if isinstance(expected, bool):
            if bool(actual) is not expected:
                return False
        elif isinstance(expected, (int, float)):
            if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
                return False
        elif actual != expected:
            return False
    return True


def get_missing_method_keys(image_name, output_dir=None, method_keys=None, cache_data=None):
    """Return methods lacking metrics, stage image, or matching SAM provenance."""
    if output_dir is None:
        output_dir = DEFAULT_PREPROCESS_OUTPUT_DIR
    selected_keys = tuple(method_keys or METHOD_KEYS)
    cache = cache_data if cache_data is not None else _read_cache_data(image_name, output_dir)
    return [
        method_key
        for method_key in selected_keys
        if method_key not in cache
        or not _cache_entry_matches_sam_config(cache.get(method_key))
        or not os.path.exists(get_preprocessed_path(image_name, method_key, output_dir))
    ]


def check_results_exist(image_name, output_dir=None, method_keys=None):
    """Return True when every requested method has cache metrics and a stage image."""
    return not get_missing_method_keys(image_name, output_dir, method_keys=method_keys)


def is_cuda_runtime_error(exc):
    """Return True when an exception looks like a CUDA runtime failure."""
    if not isinstance(exc, RuntimeError):
        return False

    message = str(exc).lower()
    cuda_markers = (
        "cuda",
        "cudnn",
        "cublas",
        "curand",
        "device-side assert",
        "out of memory",
    )
    return any(marker in message for marker in cuda_markers)


def cleanup_after_cuda_error():
    """Best-effort CUDA cleanup so the next image can continue."""
    plt.close('all')
    gc.collect()

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def get_cache_path(image_name, output_dir=None):
    """Get the path for the cache file."""
    if output_dir is None:
        output_dir = DEFAULT_PREPROCESS_OUTPUT_DIR
    return os.path.join(output_dir, f'{image_name}_cache.json')


def _json_compatible(value):
    """Convert NumPy/path/container values into JSON-safe native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def save_results_cache(image_name, all_results, output_dir=None,
                       include_mask_data=False, reference_data=None):
    """
    Save key metrics to a JSON cache file for quick reloading.

    Args:
        image_name: Image name without extension
        all_results: Dictionary with all processing results
        output_dir: Output directory path
        include_mask_data: When True, include per-mask bitmap payloads (single-image mode)
        reference_data: Reference mask bundle to persist alongside per-method results
    """
    if output_dir is None:
        output_dir = DEFAULT_PREPROCESS_OUTPUT_DIR

    # Merge into the existing cache so a partial/resumed run never erases
    # completed methods. This is what preserves the original four results.
    cache_data = _read_cache_data(image_name, output_dir, warn=True)

    for method_key, result in all_results.items():
        reference_eval = result.get('reference_eval', {})
        cache_data[method_key] = {
            'method_name': result['method_name'],
            'method_key': result['method_key'],
            'sam_config': _json_compatible(SAM_CONFIG),
            # SAM metrics
            'sam_total_masks': result['sam_total_masks'],
            'sam_pred_iou_mean': result['sam_pred_iou_mean'],
            'sam_pred_iou_std': result['sam_pred_iou_std'],
            'sam_stability_mean': result['sam_stability_mean'],
            'sam_stability_std': result['sam_stability_std'],
            'sam_area_mean': result['sam_area_mean'],
            'sam_area_std': result['sam_area_std'],
            'sam_area_cv': result['sam_area_cv'],
            'sam_crop_n_layers_used': result.get('sam_crop_n_layers_used'),
            'sam_fallback_used': result.get('sam_fallback_used', False),
            # Filtering metrics
            'bg_removed_masks': result['bg_removed_masks'],
            'bg_filter_ratio': result['bg_filter_ratio'],
            'overlap_removed_masks': result.get('overlap_removed_masks', 0),
            'overlap_filter_ratio': result.get('overlap_filter_ratio', 0),
            'size_filtered_masks': result['size_filtered_masks'],
            'total_filter_ratio': result['total_filter_ratio'],
            # CLIP metrics
            'clip_total_classified': result['clip_total_classified'],
            'clip_confidence_mean': result['clip_confidence_mean'],
            'clip_confidence_std': result['clip_confidence_std'],
            'clip_high_conf_ratio': result['clip_high_conf_ratio'],
            'clip_shape_diversity': result['clip_shape_diversity'],
            'clip_dominant_shape': result['clip_dominant_shape'],
            'clip_dominant_ratio': result['clip_dominant_ratio'],
            'clip_shape_counts': result['clip_shape_counts'],
            # Reference segmentation metrics
            'reference_available': bool(reference_eval),
            'reference_gt_masks': reference_eval.get('n_gt', 0),
            'reference_pred_masks': reference_eval.get('n_pred', 0),
            'reference_matched_masks': reference_eval.get('n_matched', 0),
            'reference_false_positive_masks': reference_eval.get('n_false_positive', 0),
            'reference_false_negative_masks': reference_eval.get('n_false_negative', 0),
            'reference_precision': reference_eval.get('precision', 0),
            'reference_recall': reference_eval.get('recall', 0),
            'reference_f1': reference_eval.get('f1', 0),
            'reference_mean_iou': reference_eval.get('mean_iou', 0),
            'reference_median_iou': reference_eval.get('median_iou', 0),
            'reference_match_confidence': reference_eval.get('match_confidence', 0),
            'preprocessing_info': _json_compatible(result.get('preprocessing_info', {})),
        }

        if include_mask_data:
            cache_data[method_key]['filtered_masks_data'] = serialize_filtered_masks(
                result.get('filtered_masks', [])
            )

    if include_mask_data and reference_data:
        cache_data['_reference'] = serialize_reference_masks(reference_data)

    cache_path = get_cache_path(image_name, output_dir)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(_json_compatible(cache_data), f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved cache: {image_name}_cache.json")


def load_results_cache(image_name, output_dir=None):
    """
    Load results from cache file.

    Args:
        image_name: Image name without extension
        output_dir: Output directory path

    Returns:
        Dictionary with cached results, or None if cache doesn't exist
    """
    if output_dir is None:
        output_dir = DEFAULT_PREPROCESS_OUTPUT_DIR
    cache_data = _read_cache_data(image_name, output_dir, warn=True)
    if cache_data:
        print(f"[OK] Loaded cache: {image_name}_cache.json")
        return cache_data
    return None


def check_cache_exists(image_name, output_dir=None):
    """Check if cache file exists for an image."""
    if output_dir is None:
        output_dir = DEFAULT_PREPROCESS_OUTPUT_DIR
    cache_path = get_cache_path(image_name, output_dir)
    return os.path.exists(cache_path)


def _mean_student_t_ci(values, confidence_level=ORIGIN_CONFIDENCE_LEVEL):
    """Return the arithmetic mean and two-sided Student t confidence interval."""
    numeric_values = np.asarray(values, dtype=np.float64)
    if numeric_values.ndim != 1 or len(numeric_values) < 2:
        raise ValueError("At least two observations are required for a Student t CI")
    if not np.isfinite(numeric_values).all():
        raise ValueError("Confidence interval input contains a missing or non-finite value")

    mean_value = float(np.mean(numeric_values))
    standard_error = float(stats.sem(numeric_values, ddof=1))
    if standard_error == 0.0:
        return mean_value, mean_value, mean_value

    ci_low, ci_high = stats.t.interval(
        confidence_level,
        len(numeric_values) - 1,
        loc=mean_value,
        scale=standard_error,
    )
    return mean_value, float(ci_low), float(ci_high)


def save_origin_figure_workbook(per_image_df, output_dir):
    """Build the publication workbook for Origin Figures A, B, and D."""
    method_labels = {
        "1_bm3d": "BM3D only",
        "2_bm3d_noise2sr": "BM3D + Noise2SR",
        "3_bm3d_noise2sr_clahe": "BM3D + Noise2SR + CLAHE",
        "4_bm3d_clahe": "BM3D + CLAHE",
    }
    summary_method_order = [
        "2_bm3d_noise2sr",
        "1_bm3d",
        "3_bm3d_noise2sr_clahe",
        "4_bm3d_clahe",
    ]
    comparator_order = [
        "1_bm3d",
        "3_bm3d_noise2sr_clahe",
        "4_bm3d_clahe",
    ]
    baseline_key = "2_bm3d_noise2sr"
    required_metrics = [
        "reference_false_positive_masks",
        "reference_precision",
        "reference_recall",
        "reference_f1",
    ]
    required_columns = ["image_name", "method_key", *required_metrics]
    missing_columns = [column for column in required_columns if column not in per_image_df.columns]
    if missing_columns:
        raise ValueError(
            "Cannot build Origin workbook; missing per-image columns: "
            + ", ".join(missing_columns)
        )

    source_columns = required_columns.copy()
    if "reference_available" in per_image_df.columns:
        source_columns.append("reference_available")
    source = per_image_df.loc[
        per_image_df["method_key"].isin(summary_method_order), source_columns
    ].copy()
    if source["image_name"].isna().any():
        raise ValueError("Image IDs contain missing values")
    source["image_name"] = source["image_name"].map(str)

    duplicate_rows = source.duplicated(["image_name", "method_key"], keep=False)
    if duplicate_rows.any():
        duplicate_keys = source.loc[
            duplicate_rows, ["image_name", "method_key"]
        ].drop_duplicates()
        raise ValueError(
            "Duplicate image/method rows found: "
            + duplicate_keys.to_dict(orient="records").__repr__()
        )

    for metric in required_metrics:
        source[metric] = pd.to_numeric(source[metric], errors="coerce")

    valid_rows = np.ones(len(source), dtype=bool)
    for metric in required_metrics:
        valid_rows &= np.isfinite(source[metric].to_numpy(dtype=np.float64))

    if "reference_available" in source.columns:
        def _is_reference_available(value):
            if pd.isna(value):
                return False
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, float, np.integer, np.floating)):
                return bool(value)
            return str(value).strip().lower() in {"true", "1", "yes"}

        valid_rows &= source["reference_available"].map(
            _is_reference_available
        ).to_numpy(dtype=bool)

    valid_source = source.loc[valid_rows].copy()
    valid_image_sets = {
        method_key: set(
            valid_source.loc[
                valid_source["method_key"] == method_key, "image_name"
            ].tolist()
        )
        for method_key in summary_method_order
    }
    if any(len(image_set) == 0 for image_set in valid_image_sets.values()):
        counts = {key: len(value) for key, value in valid_image_sets.items()}
        raise ValueError(f"A method has no valid reference metrics: {counts}")

    common_images = set.intersection(*valid_image_sets.values())
    if len(common_images) < 2:
        raise ValueError("Fewer than two complete images are shared by all four methods")
    common_image_order = sorted(common_images)
    analysis_source = valid_source.loc[
        valid_source["image_name"].isin(common_images)
    ].copy()

    compared_image_sets = {
        method_key: set(
            analysis_source.loc[
                analysis_source["method_key"] == method_key, "image_name"
            ].tolist()
        )
        for method_key in summary_method_order
    }
    same_image_set = all(
        image_set == common_images for image_set in compared_image_sets.values()
    )
    if not same_image_set:
        raise AssertionError("The four methods do not use the same complete image set")

    metric_wide = {}
    for metric in required_metrics:
        wide = analysis_source.pivot(
            index="image_name",
            columns="method_key",
            values=metric,
        ).reindex(index=common_image_order, columns=summary_method_order)
        if wide.isna().any().any():
            raise AssertionError(f"Metric {metric} is incomplete after common-set filtering")
        metric_wide[metric] = wide

    n_images = len(common_image_order)

    a_columns = [
        "Method_Key",
        "Method_Label",
        "N",
        "Mean_F1",
        "CI95_Low",
        "CI95_High",
    ]
    a_rows = []
    for method_key in summary_method_order:
        mean_f1, ci_low, ci_high = _mean_student_t_ci(
            metric_wide["reference_f1"][method_key].to_numpy()
        )
        a_rows.append({
            "Method_Key": method_key,
            "Method_Label": method_labels[method_key],
            "N": n_images,
            "Mean_F1": mean_f1,
            "CI95_Low": ci_low,
            "CI95_High": ci_high,
        })
    a_df = pd.DataFrame(a_rows, columns=a_columns)

    b_columns = [
        "Method_Key",
        "Method_Label",
        "N",
        "Mean_FP_Masks_Per_Image",
        "FP_CI95_Low",
        "FP_CI95_High",
        "Mean_Precision",
        "Precision_CI95_Low",
        "Precision_CI95_High",
        "Mean_Recall",
        "Recall_CI95_Low",
        "Recall_CI95_High",
    ]
    b_rows = []
    for method_key in summary_method_order:
        fp_mean, fp_low, fp_high = _mean_student_t_ci(
            metric_wide["reference_false_positive_masks"][method_key].to_numpy()
        )
        precision_mean, precision_low, precision_high = _mean_student_t_ci(
            metric_wide["reference_precision"][method_key].to_numpy()
        )
        recall_mean, recall_low, recall_high = _mean_student_t_ci(
            metric_wide["reference_recall"][method_key].to_numpy()
        )
        b_rows.append({
            "Method_Key": method_key,
            "Method_Label": method_labels[method_key],
            "N": n_images,
            "Mean_FP_Masks_Per_Image": fp_mean,
            "FP_CI95_Low": fp_low,
            "FP_CI95_High": fp_high,
            "Mean_Precision": precision_mean,
            "Precision_CI95_Low": precision_low,
            "Precision_CI95_High": precision_high,
            "Mean_Recall": recall_mean,
            "Recall_CI95_Low": recall_low,
            "Recall_CI95_High": recall_high,
        })
    b_df = pd.DataFrame(b_rows, columns=b_columns)

    paired_columns = [
        "Comparator_Key",
        "Comparator_Label",
        "N",
        "Mean_Delta_FP",
        "Delta_FP_CI95_Low",
        "Delta_FP_CI95_High",
        "Relative_FP_Change_Percent",
        "Mean_Delta_Precision",
        "Delta_Precision_CI95_Low",
        "Delta_Precision_CI95_High",
        "Mean_Delta_Recall",
        "Delta_Recall_CI95_Low",
        "Delta_Recall_CI95_High",
    ]
    paired_rows = []
    for comparator_key in comparator_order:
        delta_fp = (
            metric_wide["reference_false_positive_masks"][baseline_key]
            - metric_wide["reference_false_positive_masks"][comparator_key]
        ).to_numpy(dtype=np.float64)
        delta_precision = (
            metric_wide["reference_precision"][baseline_key]
            - metric_wide["reference_precision"][comparator_key]
        ).to_numpy(dtype=np.float64)
        delta_recall = (
            metric_wide["reference_recall"][baseline_key]
            - metric_wide["reference_recall"][comparator_key]
        ).to_numpy(dtype=np.float64)

        fp_mean, fp_low, fp_high = _mean_student_t_ci(delta_fp)
        precision_mean, precision_low, precision_high = _mean_student_t_ci(
            delta_precision
        )
        recall_mean, recall_low, recall_high = _mean_student_t_ci(delta_recall)
        comparator_fp_mean = float(
            metric_wide["reference_false_positive_masks"][comparator_key].mean()
        )
        if comparator_fp_mean == 0.0:
            raise ZeroDivisionError(
                f"Comparator {comparator_key} has zero mean FP; relative change is undefined"
            )

        paired_rows.append({
            "Comparator_Key": comparator_key,
            "Comparator_Label": f"vs {method_labels[comparator_key]}",
            "N": n_images,
            "Mean_Delta_FP": fp_mean,
            "Delta_FP_CI95_Low": fp_low,
            "Delta_FP_CI95_High": fp_high,
            "Relative_FP_Change_Percent": fp_mean / comparator_fp_mean * 100.0,
            "Mean_Delta_Precision": precision_mean,
            "Delta_Precision_CI95_Low": precision_low,
            "Delta_Precision_CI95_High": precision_high,
            "Mean_Delta_Recall": recall_mean,
            "Delta_Recall_CI95_Low": recall_low,
            "Delta_Recall_CI95_High": recall_high,
        })
    paired_df = pd.DataFrame(paired_rows, columns=paired_columns)

    d_columns = [
        "Comparator_Key",
        "Comparator_Label",
        "N",
        "Win_Count",
        "Tie_Count",
        "Loss_Count",
        "Win_Percent",
        "Tie_Percent",
        "Loss_Percent",
        "Mean_Delta_F1",
        "Delta_F1_CI95_Low",
        "Delta_F1_CI95_High",
    ]
    d_rows = []
    for comparator_key in comparator_order:
        delta_f1 = (
            metric_wide["reference_f1"][baseline_key]
            - metric_wide["reference_f1"][comparator_key]
        ).to_numpy(dtype=np.float64)
        ties = np.isclose(
            delta_f1,
            0.0,
            rtol=0.0,
            atol=ORIGIN_TIE_TOLERANCE,
        )
        wins = (~ties) & (delta_f1 > 0.0)
        losses = (~ties) & (delta_f1 < 0.0)
        win_count = int(np.count_nonzero(wins))
        tie_count = int(np.count_nonzero(ties))
        loss_count = int(np.count_nonzero(losses))
        mean_delta, ci_low, ci_high = _mean_student_t_ci(delta_f1)

        d_rows.append({
            "Comparator_Key": comparator_key,
            "Comparator_Label": f"vs {method_labels[comparator_key]}",
            "N": n_images,
            "Win_Count": win_count,
            "Tie_Count": tie_count,
            "Loss_Count": loss_count,
            "Win_Percent": win_count / n_images * 100.0,
            "Tie_Percent": tie_count / n_images * 100.0,
            "Loss_Percent": loss_count / n_images * 100.0,
            "Mean_Delta_F1": mean_delta,
            "Delta_F1_CI95_Low": ci_low,
            "Delta_F1_CI95_High": ci_high,
        })
    d_df = pd.DataFrame(d_rows, columns=d_columns)

    metadata_df = pd.DataFrame(
        [
            {"Item": "Number_of_images", "Value": n_images},
            {"Item": "Method_keys", "Value": ";".join(summary_method_order)},
            {"Item": "Tie_tolerance", "Value": ORIGIN_TIE_TOLERANCE},
            {
                "Item": "CI_method",
                "Value": (
                    "Two-sided Student t interval for the arithmetic mean; "
                    "paired on per-image deltas for paired summaries"
                ),
            },
            {"Item": "Confidence_level", "Value": ORIGIN_CONFIDENCE_LEVEL},
            {
                "Item": "Precision_averaging_method",
                "Value": "Macro arithmetic mean of per-image object-level precision",
            },
            {
                "Item": "Recall_averaging_method",
                "Value": "Macro arithmetic mean of per-image object-level recall",
            },
            {
                "Item": "Missing_value_policy",
                "Value": (
                    "Complete-case intersection across all four methods and all "
                    "required reference metrics"
                ),
            },
            {
                "Item": "Object_matching_criterion",
                "Value": (
                    "One-to-one Hungarian assignment on mask IoU; "
                    f"matched when IoU >= {REFERENCE_MATCH_IOU_THRESHOLD}"
                ),
            },
            {
                "Item": "Code_version_or_execution_date",
                "Value": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            },
        ],
        columns=["Item", "Value"],
    )

    if not (a_df["N"] == n_images).all() or not (b_df["N"] == n_images).all():
        raise AssertionError("A/B method N does not equal the common image count")
    count_sums = d_df[["Win_Count", "Tie_Count", "Loss_Count"]].sum(axis=1)
    if not (count_sums == d_df["N"]).all():
        raise AssertionError("D win/tie/loss counts do not sum to N")
    percent_sums = d_df[["Win_Percent", "Tie_Percent", "Loss_Percent"]].sum(axis=1)
    if not np.allclose(percent_sums.to_numpy(), 100.0, rtol=0.0, atol=1e-12):
        raise AssertionError("D win/tie/loss percentages do not sum to 100")

    sheets = {
        "A_F1_summary": a_df,
        "B_metric_summary": b_df,
        "B_paired_difference_summary": paired_df,
        "D_pairwise_summary": d_df,
        "metadata": metadata_df,
    }
    workbook_path = Path(output_dir) / ORIGIN_FIGURE_WORKBOOK_NAME
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

    from openpyxl import load_workbook

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        if workbook.sheetnames != list(sheets):
            raise AssertionError(
                f"Unexpected workbook sheets: {workbook.sheetnames}"
            )
        for sheet_name, dataframe in sheets.items():
            worksheet = workbook[sheet_name]
            stored_headers = [
                cell.value for cell in next(worksheet.iter_rows(min_row=1, max_row=1))
            ]
            if stored_headers != list(dataframe.columns):
                raise AssertionError(
                    f"Unexpected columns in {sheet_name}: {stored_headers}"
                )
            if worksheet.max_row - 1 != len(dataframe):
                raise AssertionError(
                    f"Unexpected row count in {sheet_name}: {worksheet.max_row - 1}"
                )
    finally:
        workbook.close()

    print("\n--- Origin 2026 Workbook Validation ---")
    for sheet_name, dataframe in sheets.items():
        print(f"  {sheet_name}: {len(dataframe)} data rows")
    print(f"  A N by method: {dict(zip(a_df['Method_Key'], a_df['N']))}")
    print(f"  B N by method: {dict(zip(b_df['Method_Key'], b_df['N']))}")
    for row_index, row in d_df.iterrows():
        print(
            f"  D {row['Comparator_Key']}: "
            f"count sum={int(count_sums.iloc[row_index])}/{int(row['N'])}, "
            f"percent sum={percent_sums.iloc[row_index]:.15g}%"
        )
    print(f"  Same complete image set across all methods: {same_image_set} (n={n_images})")
    print(f"  Valid source images by method: "
          f"{dict((key, len(value)) for key, value in valid_image_sets.items())}")
    print(f"[OK] Saved Origin workbook: {workbook_path}")
    return workbook_path


def save_preprocessing_manuscript_workbook(per_image_df, output_dir):
    """Save every preprocessing value referenced by the manuscript as raw numbers."""
    method_order = list(METHOD_KEYS)
    required_metrics = [
        "reference_false_positive_masks",
        "reference_precision",
        "reference_recall",
        "reference_f1",
    ]
    required_columns = ["image_name", "method_key", *required_metrics]
    missing_columns = [column for column in required_columns if column not in per_image_df]
    if missing_columns:
        raise ValueError(
            "Cannot build manuscript preprocessing workbook; missing columns: "
            + ", ".join(missing_columns)
        )

    source_columns = list(required_columns)
    if "reference_available" in per_image_df:
        source_columns.append("reference_available")
    source = per_image_df.loc[
        per_image_df["method_key"].isin(method_order), source_columns
    ].copy()
    source = _filter_dataframe_to_active_split(source)
    source["image_name"] = source["image_name"].map(str)
    if source.duplicated(["image_name", "method_key"]).any():
        raise ValueError("Duplicate image/method rows in preprocessing manuscript source")
    for metric in required_metrics:
        source[metric] = pd.to_numeric(source[metric], errors="coerce")
    finite = np.ones(len(source), dtype=bool)
    for metric in required_metrics:
        finite &= np.isfinite(source[metric].to_numpy(dtype=np.float64))
    if "reference_available" in source:
        finite &= source["reference_available"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
            if not isinstance(value, (bool, np.bool_))
            else bool(value)
        ).to_numpy(dtype=bool)
    source = source.loc[finite].copy()

    image_sets = {
        method_key: set(
            source.loc[source["method_key"] == method_key, "image_name"]
        )
        for method_key in method_order
    }
    if any(not image_set for image_set in image_sets.values()):
        counts = {key: len(value) for key, value in image_sets.items()}
        raise ValueError(f"A publication preprocessing method has no valid rows: {counts}")
    common_ids = set.intersection(*image_sets.values())
    if len(common_ids) < 2:
        raise ValueError("Fewer than two complete images are shared by all configured methods")
    common_order = sorted(common_ids)
    expected_ids = (
        set(ACTIVE_SPLIT["included_ids"])
        if ACTIVE_SPLIT is not None
        else set(common_ids)
    )
    publication_ready = (
        common_ids == expected_ids
        and all(image_set == expected_ids for image_set in image_sets.values())
    )

    metric_wide = {
        metric: source.pivot(
            index="image_name", columns="method_key", values=metric
        ).reindex(index=common_order, columns=method_order)
        for metric in required_metrics
    }
    if any(wide.isna().any().any() for wide in metric_wide.values()):
        raise AssertionError("Complete-case preprocessing pivot contains missing values")

    summary_rows = []
    for method_key in method_order:
        f1_mean, f1_low, f1_high = _mean_student_t_ci(
            metric_wide["reference_f1"][method_key].to_numpy()
        )
        fp_mean, fp_low, fp_high = _mean_student_t_ci(
            metric_wide["reference_false_positive_masks"][method_key].to_numpy()
        )
        precision_mean, precision_low, precision_high = _mean_student_t_ci(
            metric_wide["reference_precision"][method_key].to_numpy()
        )
        recall_mean, recall_low, recall_high = _mean_student_t_ci(
            metric_wide["reference_recall"][method_key].to_numpy()
        )
        summary_rows.append(
            {
                "Method_Key": method_key,
                "Method_Label": METHOD_LABELS[method_key],
                "N": len(common_order),
                "Mean_F1": f1_mean,
                "F1_CI95_Low": f1_low,
                "F1_CI95_High": f1_high,
                "Mean_FP_Masks_Per_Image": fp_mean,
                "FP_CI95_Low": fp_low,
                "FP_CI95_High": fp_high,
                "Mean_Precision": precision_mean,
                "Precision_CI95_Low": precision_low,
                "Precision_CI95_High": precision_high,
                "Mean_Recall": recall_mean,
                "Recall_CI95_Low": recall_low,
                "Recall_CI95_High": recall_high,
            }
        )
    method_summary = pd.DataFrame(summary_rows)
    selected_key = min(
        method_order,
        key=lambda key: (
            -float(
                method_summary.loc[
                    method_summary["Method_Key"] == key, "Mean_F1"
                ].iloc[0]
            ),
            method_order.index(key),
        ),
    )

    paired_rows = []
    for comparator_key in method_order:
        if comparator_key == selected_key:
            continue
        deltas = {
            "F1": (
                metric_wide["reference_f1"][selected_key]
                - metric_wide["reference_f1"][comparator_key]
            ).to_numpy(dtype=np.float64),
            "FP": (
                metric_wide["reference_false_positive_masks"][selected_key]
                - metric_wide["reference_false_positive_masks"][comparator_key]
            ).to_numpy(dtype=np.float64),
            "Precision": (
                metric_wide["reference_precision"][selected_key]
                - metric_wide["reference_precision"][comparator_key]
            ).to_numpy(dtype=np.float64),
            "Recall": (
                metric_wide["reference_recall"][selected_key]
                - metric_wide["reference_recall"][comparator_key]
            ).to_numpy(dtype=np.float64),
        }
        f1_mean, f1_low, f1_high = _mean_student_t_ci(deltas["F1"])
        fp_mean, fp_low, fp_high = _mean_student_t_ci(deltas["FP"])
        precision_mean, precision_low, precision_high = _mean_student_t_ci(
            deltas["Precision"]
        )
        recall_mean, recall_low, recall_high = _mean_student_t_ci(deltas["Recall"])
        ties = np.isclose(
            deltas["F1"], 0.0, rtol=0.0, atol=ORIGIN_TIE_TOLERANCE
        )
        wins = (~ties) & (deltas["F1"] > 0.0)
        losses = (~ties) & (deltas["F1"] < 0.0)
        n_pairs = len(deltas["F1"])
        paired_rows.append(
            {
                "Selected_Method_Key": selected_key,
                "Selected_Method_Label": METHOD_LABELS[selected_key],
                "Comparator_Key": comparator_key,
                "Comparator_Label": METHOD_LABELS[comparator_key],
                "N": n_pairs,
                "Mean_Delta_F1": f1_mean,
                "Delta_F1_CI95_Low": f1_low,
                "Delta_F1_CI95_High": f1_high,
                "Win_Count": int(wins.sum()),
                "Tie_Count": int(ties.sum()),
                "Loss_Count": int(losses.sum()),
                "Win_Percent": float(wins.sum() / n_pairs * 100.0),
                "Tie_Percent": float(ties.sum() / n_pairs * 100.0),
                "Loss_Percent": float(losses.sum() / n_pairs * 100.0),
                "Mean_Delta_FP": fp_mean,
                "Delta_FP_CI95_Low": fp_low,
                "Delta_FP_CI95_High": fp_high,
                "Mean_Delta_Precision": precision_mean,
                "Delta_Precision_CI95_Low": precision_low,
                "Delta_Precision_CI95_High": precision_high,
                "Mean_Delta_Recall": recall_mean,
                "Delta_Recall_CI95_Low": recall_low,
                "Delta_Recall_CI95_High": recall_high,
            }
        )
    paired_summary = pd.DataFrame(paired_rows)

    selected_summary = method_summary.loc[
        method_summary["Method_Key"] == selected_key
    ].iloc[0]
    manuscript_rows = [
        {
            "Item": "Publication_Ready",
            "Value": bool(publication_ready),
            "Unit": "boolean",
            "Definition": "All configured methods have complete rows for the evaluation cohort",
        },
        {
            "Item": "Expected_Image_Count",
            "Value": len(expected_ids),
            "Unit": "images",
            "Definition": "Frozen optimizer-development image set",
        },
        {
            "Item": "Analyzed_Common_Image_Count",
            "Value": len(common_order),
            "Unit": "images",
            "Definition": "Complete-case intersection across all configured methods",
        },
        {
            "Item": "Preprocessing_Candidate_Count",
            "Value": len(method_order),
            "Unit": "methods",
            "Definition": "Configured preprocessing candidates; Raw image is excluded",
        },
        {
            "Item": "Selected_Method_Key",
            "Value": selected_key,
            "Unit": "method key",
            "Definition": "Highest arithmetic mean image-level F1; method order breaks exact ties",
        },
        {
            "Item": "Selected_Method_Label",
            "Value": METHOD_LABELS[selected_key],
            "Unit": "label",
            "Definition": "Highest arithmetic mean image-level F1",
        },
    ]
    selected_fields = {
        "Selected_Mean_F1": ("Mean_F1", "proportion"),
        "Selected_F1_CI95_Low": ("F1_CI95_Low", "proportion"),
        "Selected_F1_CI95_High": ("F1_CI95_High", "proportion"),
        "Selected_Mean_FP_Masks_Per_Image": ("Mean_FP_Masks_Per_Image", "masks/image"),
        "Selected_Mean_Precision": ("Mean_Precision", "proportion"),
        "Selected_Mean_Recall": ("Mean_Recall", "proportion"),
    }
    for item, (field, unit) in selected_fields.items():
        manuscript_rows.append(
            {
                "Item": item,
                "Value": selected_summary[field],
                "Unit": unit,
                "Definition": "Selected preprocessing method on the common image set",
            }
        )
    for _, row in paired_summary.iterrows():
        prefix = f"Selected_vs_{row['Comparator_Key']}"
        for field, unit in (
            ("Mean_Delta_F1", "proportion"),
            ("Delta_F1_CI95_Low", "proportion"),
            ("Delta_F1_CI95_High", "proportion"),
            ("Win_Percent", "percent"),
            ("Tie_Percent", "percent"),
            ("Loss_Percent", "percent"),
        ):
            manuscript_rows.append(
                {
                    "Item": f"{prefix}_{field}",
                    "Value": row[field],
                    "Unit": unit,
                    "Definition": (
                        f"{METHOD_LABELS[selected_key]} minus "
                        f"{row['Comparator_Label']} on paired images"
                    ),
                }
            )
    manuscript_values = pd.DataFrame(
        manuscript_rows, columns=["Item", "Value", "Unit", "Definition"]
    )
    metadata = pd.DataFrame(
        [
            {"Item": "Dataset_Scope", "Value": ARGS.dataset_scope},
            {
                "Item": "Existing_Preprocessed_Only",
                "Value": bool(ARGS.reuse_preprocessed_only),
            },
            {
                "Item": "Optimization_Split_Manifest",
                "Value": ACTIVE_SPLIT["path"] if ACTIVE_SPLIT else "",
            },
            {
                "Item": "Optimization_Split_Hash",
                "Value": ACTIVE_SPLIT["split_hash"] if ACTIVE_SPLIT else "",
            },
            {"Item": "Method_Keys", "Value": ";".join(method_order)},
            {"Item": "Segmentation_Model", "Value": "SAM 2.1 Hiera Large"},
            {"Item": "SAM_Implementation", "Value": "sam2.SAM2AutomaticMaskGenerator"},
            {"Item": "SAM_Checkpoint", "Value": "checkpoints/sam2.1_hiera_large.pt"},
            {"Item": "SAM_Config", "Value": "sam2.1_hiera_l.yaml"},
            {"Item": "SAM_Predicted_IoU_Threshold", "Value": SAM_CONFIG["pred_iou_thresh"]},
            {"Item": "SAM_Stability_Score_Threshold", "Value": SAM_CONFIG["stability_score_thresh"]},
            {"Item": "Tie_Tolerance", "Value": ORIGIN_TIE_TOLERANCE},
            {
                "Item": "CI_Method",
                "Value": "Two-sided Student t interval; paired for method differences",
            },
            {
                "Item": "Object_Matching",
                "Value": f"Hungarian mask IoU >= {REFERENCE_MATCH_IOU_THRESHOLD}",
            },
        ]
    )
    workbook_path = Path(output_dir) / "manuscript_preprocessing_values.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        manuscript_values.to_excel(writer, sheet_name="Manuscript Values", index=False)
        method_summary.to_excel(writer, sheet_name="Method Summary", index=False)
        paired_summary.to_excel(writer, sheet_name="Paired Summary", index=False)
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
    print(
        f"[OK] Saved manuscript preprocessing workbook: {workbook_path} "
        f"(ready={publication_ready}, n={len(common_order)})"
    )
    return workbook_path


def generate_origin_figure_workbook_from_cache(output_dir):
    """Load reference metrics from existing cache files and save the Origin workbook."""
    output_path = Path(output_dir)
    cache_files = sorted(output_path.glob("*_cache.json"))
    if not cache_files:
        raise FileNotFoundError(f"No *_cache.json files found in {output_path}")

    method_keys = [
        "1_bm3d",
        "2_bm3d_noise2sr",
        "3_bm3d_noise2sr_clahe",
        "4_bm3d_clahe",
    ]
    metric_names = [
        "reference_false_positive_masks",
        "reference_precision",
        "reference_recall",
        "reference_f1",
    ]
    rows = []
    for cache_file in cache_files:
        image_name = cache_file.name[:-len("_cache.json")]
        with open(cache_file, "r", encoding="utf-8") as file_handle:
            cache = json.load(file_handle)
        for method_key in method_keys:
            metrics = cache.get(method_key)
            if metrics is None:
                continue
            row = {
                "image_name": str(image_name),
                "method_key": method_key,
                "reference_available": metrics.get("reference_available", False),
            }
            for metric_name in metric_names:
                row[metric_name] = metrics.get(metric_name, np.nan)
            rows.append(row)

    print(
        f"[INFO] Origin workbook source: {len(cache_files)} cache files, "
        f"{len(rows)} method rows"
    )
    source_df = _filter_dataframe_to_active_split(pd.DataFrame(rows))
    return save_origin_figure_workbook(source_df, output_path)


def generate_avg_plots(output_dir, save_individual_plots=False):
    """
    Generate one aggregate average graph and one aggregate statistics file.

    Primary source:
      - *_cache.json files

    Fallback source when no cache exists:
      - *_comparison_table.csv files

    Produces by default:
      - avg_preprocessing_comparison_summary.png
      - avg_preprocessing_comparison_statistics.xlsx
      - origin_2026_figure_A_B_D.xlsx

    Set save_individual_plots=True to also emit the old per-metric avg_*.png files.
    """
    from pathlib import Path as _Path
    output_path = _Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _METHOD_KEYS = list(METHOD_KEYS)
    _METHOD_LABELS = {
        "0_raw": "Raw image",
        "1_bm3d": "BM3D",
        "2_bm3d_noise2sr": "BM3D\n+ Noise2SR",
        "3_bm3d_noise2sr_clahe": "BM3D\n+ Noise2SR\n+ CLAHE",
        "4_bm3d_clahe": "BM3D\n+ CLAHE",
        "5_noise2sr": "Noise2SR Only",
        "6_clahe": "CLAHE only",
        "7_noise2sr_bm3d": "Noise2SR\n> BM3D",
        "8_noise2sr_clahe": "Noise2SR\n> CLAHE",
    }
    _METHOD_NAMES = dict(METHOD_LABELS)
    _TABLE_METHOD_TO_KEY = {name: key for key, name in _METHOD_NAMES.items()}
    _TABLE_METHOD_TO_KEY["BM3D"] = "1_bm3d"

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def _stats(values):
        series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if len(series) == 0:
            return {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        return {
            "count": int(len(series)),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    def _metric_values(df, method_key, metric, positive_only=False):
        if metric not in df.columns:
            return pd.Series(dtype=float)
        values = pd.to_numeric(
            df.loc[df["method_key"] == method_key, metric],
            errors="coerce",
        ).dropna()
        if positive_only:
            values = values[values > 0]
        return values

    per_image_rows = []
    shape_counter = Counter()
    cache_files = sorted(output_path.glob("*_cache.json"))

    if cache_files:
        source_type = "cache"
        for cache_file in cache_files:
            image_name = cache_file.name.replace("_cache.json", "")
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)

            for method_key in _METHOD_KEYS:
                if method_key not in cache:
                    continue
                metrics = cache[method_key]
                row = {
                    "image_name": image_name,
                    "method_key": method_key,
                    "method_name": metrics.get("method_name", _METHOD_NAMES[method_key]),
                    "source_file": str(cache_file),
                }

                for metric_name, value in metrics.items():
                    if isinstance(value, dict):
                        if metric_name == "clip_shape_counts":
                            for shape_name, shape_count in value.items():
                                shape_counter[(method_key, shape_name)] += int(shape_count)
                        continue
                    row[metric_name] = value

                per_image_rows.append(row)
    else:
        table_files = sorted(output_path.glob("*_comparison_table.csv"))
        if len(table_files) == 0:
            print("[WARN] No cache or comparison table files found, skipping aggregate summary")
            return

        source_type = "comparison_table"
        print("[WARN] No cache files found; using *_comparison_table.csv as fallback")
        for table_file in table_files:
            image_name = table_file.name.replace("_comparison_table.csv", "")
            table = pd.read_csv(table_file)
            for _, table_row in table.iterrows():
                method_name = str(table_row.get("Method", "")).strip()
                method_key = _TABLE_METHOD_TO_KEY.get(method_name)
                if method_key is None:
                    continue
                per_image_rows.append(
                    {
                        "image_name": image_name,
                        "method_key": method_key,
                        "method_name": method_name,
                        "source_file": str(table_file),
                        "sam_pred_iou_mean": _to_float(table_row.get("Pred IoU")),
                        "sam_stability_mean": _to_float(table_row.get("Stability")),
                        "size_filtered_masks": _to_float(table_row.get("Final Particles")),
                        "clip_confidence_mean": _to_float(table_row.get("Mean Confidence")),
                        "reference_false_positive_masks": _to_float(table_row.get("FP Masks")),
                        "reference_precision": _to_float(table_row.get("Precision")),
                        "reference_recall": _to_float(table_row.get("Recall")),
                        "reference_f1": _to_float(table_row.get("F1 Score")),
                    }
                )

    if len(per_image_rows) == 0:
        print("[WARN] Aggregate summary has no usable rows")
        return

    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df = _filter_dataframe_to_active_split(per_image_df)
    if per_image_df.empty:
        raise ValueError("No cached preprocessing rows belong to the active dataset scope")
    method_order = {method_key: i for i, method_key in enumerate(_METHOD_KEYS)}
    per_image_df["method_order"] = per_image_df["method_key"].map(method_order)
    per_image_df = per_image_df.sort_values(["image_name", "method_order"]).drop(columns=["method_order"])

    # A compact, analysis-ready table for comparing reference segmentation
    # quality across images and preprocessing methods.
    reference_metric_columns = {
        "image_name": "Image",
        "method_key": "Method Key",
        "method_name": "Method",
        "reference_available": "Reference Available",
        "reference_false_positive_masks": "FP Masks / Image",
        "reference_precision": "Precision",
        "reference_recall": "Recall",
        "reference_f1": "F1 Score",
    }
    reference_per_image_df = pd.DataFrame(index=per_image_df.index)
    for source_column, output_column in reference_metric_columns.items():
        if source_column in per_image_df.columns:
            reference_per_image_df[output_column] = per_image_df[source_column]
        else:
            reference_per_image_df[output_column] = np.nan

    if "reference_available" not in per_image_df.columns:
        reference_per_image_df["Reference Available"] = reference_per_image_df[
            ["FP Masks / Image", "Precision", "Recall", "F1 Score"]
        ].notna().any(axis=1)
    reference_per_image_path = output_path / "per_image_reference_segmentation_metrics.csv"
    reference_per_image_df.to_csv(reference_per_image_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {reference_per_image_path.name}")

    origin_workbook_path = save_origin_figure_workbook(per_image_df, output_path)
    manuscript_workbook_path = save_preprocessing_manuscript_workbook(
        per_image_df, output_path
    )

    # Image-level F1 win/tie/loss comparison requested for the two denoising methods.
    f1_method_keys = ["1_bm3d", "2_bm3d_noise2sr"]
    f1_method_names = {
        "1_bm3d": "BM3D only",
        "2_bm3d_noise2sr": "BM3D + Noise2SR",
    }
    f1_pair_source = per_image_df[
        per_image_df["method_key"].isin(f1_method_keys)
    ][["image_name", "method_key", "reference_f1"]].copy()
    f1_pair_source["reference_f1"] = pd.to_numeric(
        f1_pair_source["reference_f1"], errors="coerce"
    )
    f1_pair_wide = f1_pair_source.pivot_table(
        index="image_name",
        columns="method_key",
        values="reference_f1",
        aggfunc="first",
    ).reindex(columns=f1_method_keys).dropna(subset=f1_method_keys)

    f1_difference = f1_pair_wide["1_bm3d"] - f1_pair_wide["2_bm3d_noise2sr"]
    f1_tie = np.isclose(f1_difference, 0.0, rtol=0.0, atol=1e-12)
    bm3d_win = (~f1_tie) & (f1_difference > 0)
    noise2sr_win = (~f1_tie) & (f1_difference < 0)

    f1_wtl_detail_df = pd.DataFrame({
        "Image": f1_pair_wide.index,
        "BM3D only F1": f1_pair_wide["1_bm3d"].to_numpy(),
        "BM3D + Noise2SR F1": f1_pair_wide["2_bm3d_noise2sr"].to_numpy(),
        "F1 Difference (BM3D only - BM3D + Noise2SR)": f1_difference.to_numpy(),
        "BM3D only Result": np.where(f1_tie, "Tie", np.where(bm3d_win, "Win", "Lose")),
        "BM3D + Noise2SR Result": np.where(
            f1_tie, "Tie", np.where(noise2sr_win, "Win", "Lose")
        ),
        "Winner": np.where(
            f1_tie,
            "Tie",
            np.where(bm3d_win, f1_method_names["1_bm3d"], f1_method_names["2_bm3d_noise2sr"]),
        ),
    })

    n_f1_pairs = len(f1_wtl_detail_df)
    f1_wtl_summary_rows = []
    for method_name, result_column in (
        (f1_method_names["1_bm3d"], "BM3D only Result"),
        (f1_method_names["2_bm3d_noise2sr"], "BM3D + Noise2SR Result"),
    ):
        result_counts = f1_wtl_detail_df[result_column].value_counts()
        wins = int(result_counts.get("Win", 0))
        ties = int(result_counts.get("Tie", 0))
        losses = int(result_counts.get("Lose", 0))
        denominator = n_f1_pairs if n_f1_pairs > 0 else 1
        f1_wtl_summary_rows.append({
            "Method": method_name,
            "Wins": wins,
            "Ties": ties,
            "Losses": losses,
            "Compared Images": n_f1_pairs,
            "Win Rate (%)": wins / denominator * 100,
            "Tie Rate (%)": ties / denominator * 100,
            "Loss Rate (%)": losses / denominator * 100,
        })
    f1_wtl_summary_df = pd.DataFrame(f1_wtl_summary_rows)

    n_images = int(per_image_df["image_name"].nunique())
    n_method_rows = int(len(per_image_df))

    plot_configs = [
        {
            "metric": "sam_pred_iou_mean",
            "title": "SAM Predicted IoU",
            "ylabel": "Mean Predicted IoU",
            "positive_only": False,
            "ymax_cap": 1.05,
        },
        {
            "metric": "sam_stability_mean",
            "title": "SAM Stability Score",
            "ylabel": "Mean Stability Score",
            "positive_only": False,
            "ymax_cap": 1.05,
        },
        {
            "metric": "size_filtered_masks",
            "title": "Final Particle Count",
            "ylabel": "Mean Particle Count",
            "positive_only": False,
            "ymax_cap": None,
        },
        {
            "metric": "clip_confidence_mean",
            "title": "CLIP Mean Confidence",
            "ylabel": "Mean CLIP Confidence",
            "positive_only": True,
            "ymax_cap": 1.05,
        },
    ]

    avg_metrics = {}
    for method_key in _METHOD_KEYS:
        avg_metrics[method_key] = {}
        for cfg in plot_configs:
            metric_stats = _stats(
                _metric_values(
                    per_image_df,
                    method_key,
                    cfg["metric"],
                    positive_only=cfg["positive_only"],
                )
            )
            avg_metrics[method_key][cfg["metric"]] = metric_stats

    print(f"\n--- Aggregate Average Metrics ({source_type}, n={n_images} images) ---")
    for method_key in _METHOD_KEYS:
        label = _METHOD_NAMES[method_key]
        parts = []
        for cfg in plot_configs:
            metric_stats = avg_metrics[method_key][cfg["metric"]]
            parts.append(
                f"{cfg['metric']}={metric_stats['mean']:.4f} +/- {metric_stats['std']:.4f}"
            )
        print(f"  {label}: " + ", ".join(parts))

    # One combined graph file with the four primary aggregate metrics.
    x = np.arange(len(_METHOD_KEYS))
    bar_width = 0.62
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    axes = axes.ravel()

    for ax, cfg in zip(axes, plot_configs):
        means = [avg_metrics[k][cfg["metric"]]["mean"] for k in _METHOD_KEYS]
        stds = [avg_metrics[k][cfg["metric"]]["std"] for k in _METHOD_KEYS]
        bars = ax.bar(
            x,
            means,
            bar_width,
            yerr=stds,
            color=METHOD_COLORS,
            edgecolor=palette_hex(0),
            linewidth=0.8,
            capsize=5,
            error_kw={"linewidth": 1.4},
        )

        finite_means = [m for m in means if np.isfinite(m)]
        finite_stds = [s for s in stds if np.isfinite(s)]
        max_mean = max(finite_means) if finite_means else 1.0
        max_std = max(finite_stds) if finite_stds else 0.0

        for bar, mean in zip(bars, means):
            if not np.isfinite(mean):
                continue
            y_pos = bar.get_height() + max(max_mean * 0.025, 0.01)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(cfg["title"], fontsize=13, fontweight="bold")
        ax.set_ylabel(cfg["ylabel"], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([_METHOD_LABELS[k] for k in _METHOD_KEYS], fontsize=8)
        ax.grid(axis="y", alpha=0.28, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if cfg["ymax_cap"] is not None:
            ax.set_ylim(0, cfg["ymax_cap"])
        else:
            ymax = max((m if np.isfinite(m) else 0) + (s if np.isfinite(s) else 0)
                       for m, s in zip(means, stds))
            ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)

    fig.suptitle(
        f"Preprocessing Method Comparison Summary (n={n_images} images)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combined_plot_path = output_path / "avg_preprocessing_comparison_summary.png"
    fig.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {combined_plot_path.name}")

    # One statistics workbook with raw rows, long summaries, wide summaries, and CLIP shape totals.
    identity_cols = {"image_name", "method_key", "method_name", "method_name", "source_file"}
    metric_cols = []
    for col in per_image_df.columns:
        if col in identity_cols:
            continue
        converted = pd.to_numeric(per_image_df[col], errors="coerce")
        if converted.notna().any():
            metric_cols.append(col)

    summary_rows = []
    for method_key in _METHOD_KEYS:
        method_df = per_image_df[per_image_df["method_key"] == method_key]
        if method_df.empty:
            continue
        method_name = method_df["method_name"].iloc[0]
        for metric in metric_cols:
            metric_stats = _stats(method_df[metric])
            summary_rows.append(
                {
                    "method_key": method_key,
                    "method_name": method_name,
                    "metric": metric,
                    **metric_stats,
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    wide_rows = []
    for method_key in _METHOD_KEYS:
        method_df = per_image_df[per_image_df["method_key"] == method_key]
        if method_df.empty:
            continue
        row = {
            "method_key": method_key,
            "method_name": method_df["method_name"].iloc[0],
            "n_images": int(method_df["image_name"].nunique()),
            "n_rows": int(len(method_df)),
        }
        for cfg in plot_configs:
            metric = cfg["metric"]
            metric_stats = avg_metrics[method_key][metric]
            row[f"{metric}_mean"] = metric_stats["mean"]
            row[f"{metric}_std"] = metric_stats["std"]
            row[f"{metric}_count"] = metric_stats["count"]
        wide_rows.append(row)

    wide_summary_df = pd.DataFrame(wide_rows)

    shape_rows = []
    for (method_key, shape_name), count in sorted(shape_counter.items()):
        shape_rows.append(
            {
                "method_key": method_key,
                "method_name": _METHOD_NAMES.get(method_key, method_key),
                "shape": shape_name,
                "count": int(count),
            }
        )
    shape_counts_df = pd.DataFrame(shape_rows)

    metadata_df = pd.DataFrame(
        [
            {"key": "source_type", "value": source_type},
            {"key": "output_dir", "value": str(output_path)},
            {"key": "n_images", "value": n_images},
            {"key": "n_method_rows", "value": n_method_rows},
            {"key": "existing_preprocessed_only", "value": bool(ARGS.reuse_preprocessed_only)},
            {"key": "segmentation_model", "value": "SAM 2.1 Hiera Large"},
            {"key": "sam_implementation", "value": "sam2.SAM2AutomaticMaskGenerator"},
            {"key": "sam_checkpoint", "value": "checkpoints/sam2.1_hiera_large.pt"},
            {"key": "sam_config", "value": "sam2.1_hiera_l.yaml"},
            {"key": "sam_pred_iou_threshold", "value": SAM_CONFIG["pred_iou_thresh"]},
            {"key": "sam_stability_score_threshold", "value": SAM_CONFIG["stability_score_thresh"]},
            {"key": "created_at", "value": pd.Timestamp.now().isoformat()},
        ]
    )

    stats_path = output_path / "avg_preprocessing_comparison_statistics.xlsx"
    with pd.ExcelWriter(stats_path) as writer:
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)
        wide_summary_df.to_excel(writer, sheet_name="summary_wide", index=False)
        summary_df.to_excel(writer, sheet_name="summary_long", index=False)
        per_image_df.to_excel(writer, sheet_name="per_image_metrics", index=False)
        reference_per_image_df.to_excel(
            writer,
            sheet_name="per_image_reference_metrics",
            index=False,
        )
        f1_wtl_summary_df.to_excel(
            writer,
            sheet_name="f1_bm3d_vs_noise2sr",
            index=False,
        )
        f1_wtl_detail_df.to_excel(
            writer,
            sheet_name="f1_bm3d_vs_noise2sr",
            startrow=len(f1_wtl_summary_df) + 3,
            index=False,
        )
        if not shape_counts_df.empty:
            shape_counts_df.to_excel(writer, sheet_name="clip_shape_counts", index=False)

    print(f"  Saved: {stats_path.name}")

    if not save_individual_plots:
        print(f"[OK] Aggregate graph/statistics saved to {output_dir}")
        return {
            "summary_plot": combined_plot_path,
            "statistics": stats_path,
            "origin_workbook": origin_workbook_path,
            "manuscript_workbook": manuscript_workbook_path,
            "per_image": per_image_df,
            "summary": summary_df,
        }

    # Optional legacy one-file-per-metric plots.
    plot_configs = [
        {
            "title": "SAM Predicted IoU",
            "means": [avg_metrics[k]["sam_pred_iou_mean"]["mean"] for k in _METHOD_KEYS],
            "stds": [avg_metrics[k]["sam_pred_iou_mean"]["std"] for k in _METHOD_KEYS],
            "ylabel": "Mean Predicted IoU",
            "filename": "avg_sam_predicted_iou.png",
            "ylim_pad": True,
            "ymax_cap": 1.05,
        },
        {
            "title": "SAM Stability Score",
            "means": [avg_metrics[k]["sam_stability_mean"]["mean"] for k in _METHOD_KEYS],
            "stds": [avg_metrics[k]["sam_stability_mean"]["std"] for k in _METHOD_KEYS],
            "ylabel": "Mean Stability Score",
            "filename": "avg_sam_stability_score.png",
            "ylim_pad": True,
            "ymax_cap": 1.05,
        },
        {
            "title": "Final Particle Count",
            "means": [avg_metrics[k]["size_filtered_masks"]["mean"] for k in _METHOD_KEYS],
            "stds": [avg_metrics[k]["size_filtered_masks"]["std"] for k in _METHOD_KEYS],
            "ylabel": "Mean Particle Count",
            "filename": "avg_final_particle_count.png",
            "ylim_pad": False,
        },
        {
            "title": "CLIP Mean Confidence",
            "means": [avg_metrics[k]["clip_confidence_mean"]["mean"] for k in _METHOD_KEYS],
            "stds": [avg_metrics[k]["clip_confidence_mean"]["std"] for k in _METHOD_KEYS],
            "ylabel": "Mean CLIP Confidence",
            "filename": "avg_clip_mean_confidence.png",
            "ylim_pad": True,
        },
    ]

    # --- Text versions ---
    for cfg in plot_configs:
        fig, ax = plt.subplots(figsize=(15, 7))
        bars = ax.bar(x, cfg["means"], bar_width, yerr=cfg["stds"],
                      color=METHOD_COLORS, edgecolor=palette_hex(0), linewidth=0.8,
                      capsize=5, error_kw={'linewidth': 1.5})
        for bar, mean, std in zip(bars, cfg["means"], cfg["stds"]):
            y_pos = bar.get_height() - (max(cfg["means"]) * 0.03)
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{mean:.3f}', ha='center', va='top', fontsize=12,
                    fontweight='bold', color=PALETTE_NEUTRAL,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=palette_hex(0), alpha=0.6))
        ax.set_xlabel('Preprocessing Method', fontsize=13)
        ax.set_ylabel(cfg["ylabel"], fontsize=13)
        ax.set_title(f'{cfg["title"]} (n={n_images})', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([_METHOD_LABELS[k] for k in _METHOD_KEYS], fontsize=9)
        ax.tick_params(axis='y', labelsize=11)
        if cfg.get("ymax_cap"):
            ymin = min(cfg["means"]) - max(cfg["stds"]) * 1.5
            ax.set_ylim(max(0, ymin * 0.95), cfg["ymax_cap"])
        elif cfg["ylim_pad"]:
            ymin = min(cfg["means"]) - max(cfg["stds"]) * 1.5
            ymax = max(cfg["means"]) + max(cfg["stds"]) * 1.8
            if ymin > 0:
                ax.set_ylim(ymin * 0.95, ymax)
        else:
            ax.set_ylim(0, max(m + s for m, s in zip(cfg["means"], cfg["stds"])) * 1.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, cfg["filename"]), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {cfg['filename']}")

    # --- No-text versions ---
    for cfg in plot_configs:
        fig, ax = plt.subplots(figsize=(15, 7))
        bars = ax.bar(x, cfg["means"], bar_width, yerr=cfg["stds"],
                      color=METHOD_COLORS, edgecolor=palette_hex(0), linewidth=0.8,
                      capsize=5, error_kw={'linewidth': 1.5})
        if cfg.get("ymax_cap"):
            ymin = min(cfg["means"]) - max(cfg["stds"]) * 1.5
            ax.set_ylim(max(0, ymin * 0.95), cfg["ymax_cap"])
        elif cfg["ylim_pad"]:
            ymin = min(cfg["means"]) - max(cfg["stds"]) * 1.5
            ymax = max(cfg["means"]) + max(cfg["stds"]) * 1.8
            if ymin > 0:
                ax.set_ylim(ymin * 0.95, ymax)
        else:
            ax.set_ylim(0, max(m + s for m, s in zip(cfg["means"], cfg["stds"])) * 1.3)
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Keep tick marks/error bars/grid; hide only tick label text.
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        notext_name = cfg["filename"].replace('.png', '_notext.png')
        fig.savefig(os.path.join(output_dir, notext_name), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {notext_name}")

    print(f"[OK] All avg plots (text + notext) saved to {output_dir}")


if ARGS.origin_workbook_only:
    origin_output_dir = (
        Path(ARGS.output_dir).resolve()
        if ARGS.output_dir
        else Path(DEFAULT_PREPROCESS_OUTPUT_DIR)
    )
    print("\n" + "="*80)
    print("ORIGIN 2026 WORKBOOK-ONLY MODE")
    print("="*80)
    print(f"[INFO] Reading existing cache files from: {origin_output_dir}")
    generate_origin_figure_workbook_from_cache(origin_output_dir)
    sys.exit(0)


if ARGS.aggregate_only:
    aggregate_output_dir = Path(ARGS.output_dir).resolve() if ARGS.output_dir else Path(DEFAULT_PREPROCESS_OUTPUT_DIR)
    print("\n" + "="*80)
    print("AGGREGATE-ONLY MODE")
    print("="*80)
    print(f"[INFO] Reading existing cache/stat files from: {aggregate_output_dir}")
    generate_avg_plots(
        str(aggregate_output_dir),
        save_individual_plots=ARGS.save_individual_avg_plots,
    )
    sys.exit(0)


# ============================================================================
# GPU SETUP
# ============================================================================
print("\n" + "="*80)
print("GPU SETUP")
print("="*80)

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n[OK] GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("\n[WARN] No GPU detected, using CPU (slower)")

print(f"   Device: {device}")

# ============================================================================
# LOAD IMAGES FROM FOLDER
# ============================================================================
print("\n" + "="*80)
print("LOAD IMAGES FROM FOLDER")
print("="*80)

# Specify folder path containing images (relative to PROJECT_ROOT)
folder_path = str(Path(ARGS.dataset_dir).resolve())
print(f"\n[INFO] Folder path: {folder_path}")
print(f"   Exists: {os.path.exists(folder_path)}")

# Get all image files from folder (using set to avoid duplicates)
image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
all_image_files = set()  # Use set to automatically remove duplicates

for ext in image_extensions:
    all_image_files.update(Path(folder_path).glob(f'*{ext}'))
    all_image_files.update(Path(folder_path).glob(f'*{ext.upper()}'))

# Convert to sorted path objects
all_image_files_sorted = sorted(all_image_files, key=lambda p: p.name.lower())
if not all_image_files_sorted:
    raise ValueError(f"No images found in folder: {folder_path}")

target_image_stem = None

# Optional single-image mode via --image
if ARGS.image:
    target_raw = ARGS.image.strip()
    target_path = Path(target_raw)

    if target_path.is_file():
        selected = target_path.resolve()
        target_image_stem = selected.stem
        image_paths = [str(selected)]
    else:
        target_name = target_path.name.lower()
        target_stem = (target_path.stem if target_path.suffix else target_raw).lower()
        matches = [
            p for p in all_image_files_sorted
            if (p.name.lower() == target_name) or (p.stem.lower() == target_stem)
        ]
        if not matches:
            raise ValueError(
                f"Target image not found: {target_raw}\n"
                f"Searched in: {folder_path}"
            )
        selected = matches[0]
        if len(matches) > 1:
            print(f"[WARN] Multiple matches found for '{target_raw}'. Using: {selected.name}")
        target_image_stem = selected.stem
        image_paths = [str(selected)]

    print(f"\n[INFO] Target image mode: {Path(image_paths[0]).name}")
else:
    if ARGS.reuse_preprocessed_only:
        existing_output_dir = (
            Path(ARGS.output_dir).expanduser().resolve()
            if ARGS.output_dir
            else Path(DEFAULT_PREPROCESS_OUTPUT_DIR).resolve()
        )
        existing_ids = {
            path.name[:-len("_cache.json")]
            for path in existing_output_dir.glob("*_cache.json")
        }
        if not existing_ids:
            raise FileNotFoundError(
                "--reuse-preprocessed-only found no *_cache.json cohort in "
                f"{existing_output_dir}"
            )
        dataset_by_id = {path.stem: path for path in all_image_files_sorted}
        missing_existing_ids = sorted(existing_ids - set(dataset_by_id))
        if missing_existing_ids:
            raise ValueError(
                "Dataset is missing images from the existing preprocessing cohort: "
                + ", ".join(missing_existing_ids[:20])
            )
        image_paths = [str(dataset_by_id[image_id]) for image_id in sorted(existing_ids)]
        print(
            "[INFO] Dataset scope: existing preprocessing-cache cohort "
            f"({len(image_paths)} images from {existing_output_dir})"
        )
    elif ACTIVE_SPLIT is not None:
        dataset_by_id = {path.stem: path for path in all_image_files_sorted}
        missing_split_ids = sorted(ACTIVE_SPLIT["included_ids"] - set(dataset_by_id))
        if missing_split_ids:
            raise ValueError(
                "Dataset is missing optimizer-development images: "
                + ", ".join(missing_split_ids)
            )
        image_paths = [
            str(dataset_by_id[image_id])
            for image_id in sorted(ACTIVE_SPLIT["included_ids"])
        ]
        if len(image_paths) != ACTIVE_SPLIT["optimization_count"]:
            raise AssertionError("Filtered optimizer-development image count is inconsistent")
        print(
            "[INFO] Dataset scope: optimizer-disjoint "
            f"({ACTIVE_SPLIT['optimization_count']} images, "
            f"split={ACTIVE_SPLIT['split_hash']})"
        )
    else:
        image_paths = [str(p) for p in all_image_files_sorted]

print(f"\n[OK] Found {len(image_paths)} image(s) to process")
if (
    ARGS.expected_images is not None
    and not ARGS.image
    and len(image_paths) != ARGS.expected_images
):
    raise ValueError(
        f"Expected exactly {ARGS.expected_images} Dataset images, found {len(image_paths)} "
        f"in {folder_path}."
    )
if (
    ACTIVE_SPLIT is not None
    and ARGS.expected_images is not None
    and ARGS.expected_images != ACTIVE_SPLIT["optimization_count"]
):
    raise ValueError(
        f"--expected-images={ARGS.expected_images} conflicts with frozen optimizer "
        f"split count {ACTIVE_SPLIT['optimization_count']}"
    )
print("[INFO] Active method keys: " + ", ".join(METHOD_KEYS))
print(
    "[INFO] Active SAM thresholds: "
    f"pred_iou={SAM_CONFIG['pred_iou_thresh']:.2f}, "
    f"stability={SAM_CONFIG['stability_score_thresh']:.2f}"
)

# Optional random subsampling (set to None to process all images)
NUM_RANDOM_IMAGES = None
if (not ARGS.image) and NUM_RANDOM_IMAGES is not None and len(image_paths) > NUM_RANDOM_IMAGES:
    image_paths = random.sample(image_paths, NUM_RANDOM_IMAGES)
    print(f"[INFO] Randomly selected {NUM_RANDOM_IMAGES} images for analysis:")
else:
    print(f"[INFO] Using all {len(image_paths)} images:")

SINGLE_IMAGE_MODE = len(image_paths) == 1
print(f"[INFO] Single-image mode: {SINGLE_IMAGE_MODE}")

if not ARGS.plan_only or len(image_paths) <= 10:
    for i, img_path in enumerate(image_paths, 1):
        print(f"   {i}. {os.path.basename(img_path)}")

# ============================================================================
# CHECK FOR ALREADY-PROCESSED IMAGES
# ============================================================================
print("\n" + "="*80)
print("CHECKING FOR EXISTING RESULTS")
print("="*80)

# Define output directory
base_output_dir = Path(ARGS.output_dir).resolve() if ARGS.output_dir else Path(DEFAULT_PREPROCESS_OUTPUT_DIR)
if ARGS.image and target_image_stem:
    output_dir_path = base_output_dir / f"{target_image_stem}_preprocessed"
else:
    output_dir_path = base_output_dir

OUTPUT_DIR = str(output_dir_path)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Output directory: {OUTPUT_DIR}")

PROCESSING_LOG_PATH = os.path.join(OUTPUT_DIR, "processing_log.csv")
SKIPPED_IMAGES_LOG_PATH = os.path.join(OUTPUT_DIR, "skipped_images.csv")


def append_processing_log(event, image_name, image_path="", status="", method="", reason=""):
    """Append one processing event immediately so interrupted runs still leave breadcrumbs."""
    fieldnames = ["timestamp", "event", "image_name", "image_path", "status", "method", "reason"]
    row = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "event": event,
        "image_name": image_name,
        "image_path": image_path,
        "status": status,
        "method": method,
        "reason": reason,
    }
    file_exists = os.path.exists(PROCESSING_LOG_PATH)
    with open(PROCESSING_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_skipped_image_log(image_name, image_path, failed_method, reason):
    """Persist skipped-image details as soon as a skip happens."""
    fieldnames = ["timestamp", "image_name", "image_path", "failed_method", "reason"]
    row = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "image_name": image_name,
        "image_path": image_path,
        "failed_method": failed_method,
        "reason": reason,
    }
    file_exists = os.path.exists(SKIPPED_IMAGES_LOG_PATH)
    with open(SKIPPED_IMAGES_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    append_processing_log("skipped", image_name, image_path, "skipped", failed_method, reason)

# Filter images by method-level completion. Existing methods are never rerun
# merely because a newly added method is absent.
images_to_process = []
images_already_done = []
images_with_cache = []
image_missing_methods = {}
image_missing_metrics = {}
image_missing_preprocessing = {}

for image_path in image_paths:
    image_name = Path(image_path).stem
    cache_data = _read_cache_data(image_name, OUTPUT_DIR, warn=True)
    missing_method_keys = get_missing_method_keys(
        image_name,
        OUTPUT_DIR,
        cache_data=cache_data,
    )
    image_missing_methods[image_name] = missing_method_keys
    image_missing_metrics[image_name] = [
        method_key
        for method_key in METHOD_KEYS
        if method_key not in cache_data
        or not _cache_entry_matches_sam_config(cache_data.get(method_key))
    ]
    image_missing_preprocessing[image_name] = [
        method_key
        for method_key in METHOD_KEYS
        if not (
            ARGS.reuse_preprocessed_only
            and method_key == "0_raw"
        )
        and not has_reusable_preprocessing_stage(image_name, method_key, OUTPUT_DIR)
    ]

    if cache_data:
        images_with_cache.append(image_path)

    if not missing_method_keys:
        images_already_done.append(image_path)
        print(f"  [OK] Skipping {os.path.basename(image_path)} (all {len(METHOD_KEYS)} methods exist)")
    else:
        images_to_process.append(image_path)
        if not ARGS.plan_only:
            missing_labels = ", ".join(METHOD_LABELS[key] for key in missing_method_keys)
            print(f"  [TODO] {os.path.basename(image_path)}: {missing_labels}")

# Summary
print(f"\n[INFO] Summary:")
print(f"   Total images: {len(image_paths)}")
print(f"   Complete for all {len(METHOD_KEYS)} methods: {len(images_already_done)}")
print(f"   With any reusable cache: {len(images_with_cache)}")
print(f"   Images with missing methods: {len(images_to_process)}")

# Show cached results summary if any exist
if images_with_cache and not ARGS.plan_only:
    print("\n" + "="*80)
    print("[INFO] CACHED RESULTS SUMMARY")
    print("="*80)
    cached_method_counts = Counter()
    for image_path in images_with_cache:
        image_name = Path(image_path).stem
        cached_data = _read_cache_data(image_name, OUTPUT_DIR)
        for method_key in METHOD_KEYS:
            if method_key in cached_data:
                cached_method_counts[method_key] += 1
    for method_key in METHOD_KEYS:
        print(
            f"   {method_key:<28} "
            f"{cached_method_counts.get(method_key, 0):>4}/{len(image_paths)} cached"
        )

if ARGS.plan_only:
    missing_counts = Counter(
        method_key
        for missing_keys in image_missing_methods.values()
        for method_key in missing_keys
    )
    print("\n" + "="*80)
    print("METHOD EXECUTION PLAN")
    print("="*80)
    for method_key in METHOD_KEYS:
        metric_count = sum(
            method_key in missing for missing in image_missing_metrics.values()
        )
        preprocessing_count = sum(
            method_key in missing for missing in image_missing_preprocessing.values()
        )
        print(
            f"  {method_key:<28} total={missing_counts.get(method_key, 0):>4}, "
            f"SAM/metrics={metric_count:>4}, preprocessing={preprocessing_count:>4}"
        )
    print("[INFO] Plan-only mode: SAM, CLIP, and preprocessing models were not loaded.")
    sys.exit(0)

# Early exit if all images are already processed
if len(images_to_process) == 0:
    print("\n[OK] All images already processed!")
    print(f"   Found {len(images_already_done)} completed images in {OUTPUT_DIR}")
    print(f"   Each image contains all {len(METHOD_KEYS)} preprocessing methods.")
    generate_avg_plots(
        OUTPUT_DIR,
        save_individual_plots=ARGS.save_individual_avg_plots,
    )
    sys.exit(0)

print(f"\n[OK] Proceeding with {len(images_to_process)} new image(s)")
print("\n[OK]  Starting automated processing...")

#%%
# ============================================================================
# LOAD MODELS (SAM + CLIP)
# ============================================================================
print("\n" + "="*80)
print("LOAD MODELS")
print("="*80)

# Load SAM2 (fixed large model)
model_type = "hiera_l"
sam_checkpoint = os.path.join(PROJECT_ROOT, "checkpoints", "sam2.1_hiera_large.pt")
sam_config = get_sam2_config_path("sam2.1_hiera_l.yaml")

if not os.path.exists(sam_checkpoint):
    raise FileNotFoundError(f"SAM2 checkpoint not found: {sam_checkpoint}")

print(f"\n[INFO] Loading SAM2: {model_type}")
print(f"   Checkpoint: {sam_checkpoint}")
print(f"   Config: {sam_config}")
sam = build_sam2(sam_config, sam_checkpoint, device=device)
if not type(sam).__module__.startswith("sam2."):
    raise RuntimeError(
        "Expected a SAM2 model, but loaded "
        f"{type(sam).__module__}.{type(sam).__name__}"
    )
print("[OK] SAM2 model loaded successfully")

# Load CLIP (fixed model)
print(f"\n[INFO] Loading CLIP (ViT-L/14@336px)...")
clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
clip_model.eval()
print("[OK] CLIP model loaded successfully (ViT-L/14@336px)")

# Define the two morphology classes and their CLIP prompt ensembles.
shape_labels = ['Quasi-sphere', 'Cluster']
CLIP_PROMPT_TEMPLATE = 'This is a {} nanoparticle on electron microscope image'
shape_description_phrases = [
    [
        'microscopy image of a single isolated compact particle',
        'single round or slightly elliptical particle with a smooth boundary',
        'convex compact shape with one rounded lobe',
        'solitary quasi-spherical particle with uniform curvature',
        'single particle with a smooth outline and no lobes',
    ],
    [
        'connected cluster of multiple touching round particles',
        'multi-lobed particle made of fused spherical components',
        'irregular non-convex shape with concave indentations',
        'particle aggregate with several rounded lobes',
        'fused doublet or triplet of particles',
        'single connected object composed of multiple round blobs',
    ],
]
shape_descriptions = [
    [CLIP_PROMPT_TEMPLATE.format(phrase) for phrase in class_phrases]
    for class_phrases in shape_description_phrases
]

texts = [prompt for class_prompts in shape_descriptions for prompt in class_prompts]
text_tokens = clip.tokenize(texts).to(device)

print(f"\n[INFO] Shape categories (prompt ensemble): {', '.join(shape_labels)}")

#%%
# ============================================================================
# HELPER FUNCTIONS FOR METRICS
# ============================================================================

def calculate_shape_diversity(shapes):
    """Calculate Shannon entropy of shape distribution."""
    if not shapes:
        return 0
    counts = Counter(shapes)
    total = len(shapes)
    probs = [count / total for count in counts.values()]
    return entropy(probs)


def prepare_display_image(image):
    """Convert OpenCV image arrays to matplotlib-friendly RGB/RGBA."""
    if image is None:
        return None
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def encode_binary_mask(mask):
    """Encode a binary mask as compressed PNG base64."""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    ok, buffer = cv2.imencode('.png', mask_uint8)
    if not ok:
        raise ValueError("Failed to encode mask to PNG")
    compressed = zlib.compress(buffer.tobytes(), level=9)
    return base64.b64encode(compressed).decode('utf-8')


def serialize_filtered_masks(filtered_masks):
    """Serialize predicted masks for lightweight cache storage."""
    serialized = []
    for mask_dict in filtered_masks:
        segmentation = mask_dict.get('segmentation')
        if segmentation is None:
            continue
        serialized.append({
            'bbox': [int(v) for v in mask_dict.get('bbox', [0, 0, 0, 0])],
            'area': int(mask_dict.get('area', int(np.sum(segmentation)))),
            'predicted_iou': float(mask_dict.get('predicted_iou', 0.0)),
            'stability_score': float(mask_dict.get('stability_score', 0.0)),
            'mask_shape': list(segmentation.shape),
            'mask_data': encode_binary_mask(segmentation),
        })
    return serialized


def serialize_reference_masks(reference_data):
    """Serialize reference GT masks for single-image cache files."""
    if not reference_data:
        return None

    return {
        'annotation_path': reference_data.get('annotation_path'),
        'original_image_path': reference_data.get('original_image_path'),
        'offset': list(reference_data.get('offset', (0, 0))),
        'match_confidence': float(reference_data.get('match_confidence', 0.0)),
        'gt_masks': [
            {
                'mask_shape': list(mask.shape),
                'mask_data': encode_binary_mask(mask),
                'touches_boundary': bool(boundary_flag),
            }
            for mask, boundary_flag in zip(
                reference_data.get('gt_masks', []),
                reference_data.get('boundary_flags', []),
            )
        ],
    }


def find_reference_annotation_path(image_stem, reference_ann_dir):
    """Find DatasetNinja-style annotation JSON for an image stem."""
    ann_dir = Path(reference_ann_dir)
    if not ann_dir.exists():
        return None

    candidates = sorted(ann_dir.glob(f"{image_stem}.*.json"))
    if candidates:
        return candidates[0]

    direct_path = ann_dir / f"{image_stem}.json"
    if direct_path.exists():
        return direct_path

    return None


def find_reference_image_path(image_stem, reference_image_dir):
    """Find the full-size reference image for an image stem."""
    image_dir = Path(reference_image_dir)
    if not image_dir.exists():
        return None

    allowed_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
    candidates = sorted(
        p for p in image_dir.glob(f"{image_stem}.*")
        if p.is_file() and p.suffix.lower() in allowed_exts
    )
    return candidates[0] if candidates else None


def decode_reference_bitmap(bitmap_data):
    """Decode DatasetNinja bitmap payload or cached mask payload into a binary mask."""
    if "base64," in bitmap_data:
        bitmap_data = bitmap_data.split("base64,")[1]

    bitmap_data = bitmap_data.replace('\n', '').replace('\r', '').replace(' ', '').strip()
    decoded = base64.b64decode(bitmap_data)

    if len(decoded) > 2 and decoded[0] == 0x78:
        decoded = zlib.decompress(decoded)

    nparr = np.frombuffer(decoded, np.uint8)
    mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Failed to decode bitmap")
    return mask > 0


def find_crop_offset(original_img, cropped_img):
    """Find where the cropped dataset image sits inside the original image."""
    result = cv2.matchTemplate(original_img, cropped_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc[0], max_loc[1], max_val


def load_reference_masks(image_path, reference_ann_dir, reference_image_dir,
                         border_margin=REFERENCE_BORDER_MARGIN):
    """
    Load and crop reference masks so they align with the dataset image.

    Returns:
        Dict with gt_masks and crop metadata, or None when reference files are missing.
    """
    image_stem = Path(image_path).stem
    ann_path = find_reference_annotation_path(image_stem, reference_ann_dir)
    original_image_path = find_reference_image_path(image_stem, reference_image_dir)

    if ann_path is None or original_image_path is None:
        return None

    cropped_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(str(original_image_path), cv2.IMREAD_GRAYSCALE)
    if cropped_img is None or original_img is None:
        return None

    offset_x, offset_y, match_confidence = find_crop_offset(original_img, cropped_img)
    crop_h, crop_w = cropped_img.shape[:2]

    with open(ann_path, 'r', encoding='utf-8') as f:
        ann_data = json.load(f)

    ann_h = int(ann_data.get('size', {}).get('height', original_img.shape[0]))
    ann_w = int(ann_data.get('size', {}).get('width', original_img.shape[1]))

    gt_masks = []
    boundary_flags = []

    for obj in ann_data.get('objects', []):
        bitmap = obj.get('bitmap', {})
        if 'data' not in bitmap or 'origin' not in bitmap:
            continue

        bitmap_mask = decode_reference_bitmap(bitmap['data'])
        orig_x, orig_y = bitmap['origin']

        full_mask = np.zeros((ann_h, ann_w), dtype=bool)
        mask_h, mask_w = bitmap_mask.shape[:2]
        x_end = min(orig_x + mask_w, ann_w)
        y_end = min(orig_y + mask_h, ann_h)
        if x_end <= orig_x or y_end <= orig_y:
            continue

        full_mask[orig_y:y_end, orig_x:x_end] = bitmap_mask[:y_end - orig_y, :x_end - orig_x]

        cropped_mask = np.zeros((crop_h, crop_w), dtype=bool)
        crop_y_end = min(offset_y + crop_h, ann_h)
        crop_x_end = min(offset_x + crop_w, ann_w)
        cropped_region = full_mask[offset_y:crop_y_end, offset_x:crop_x_end]
        cropped_mask[:cropped_region.shape[0], :cropped_region.shape[1]] = cropped_region

        if not np.any(cropped_mask):
            continue

        coords = np.argwhere(cropped_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        touches_boundary = (
            x_min <= border_margin or
            y_min <= border_margin or
            x_max >= (crop_w - border_margin) or
            y_max >= (crop_h - border_margin)
        )

        gt_masks.append(cropped_mask)
        boundary_flags.append(touches_boundary)

    return {
        'gt_masks': gt_masks,
        'boundary_flags': boundary_flags,
        'offset': (offset_x, offset_y),
        'match_confidence': match_confidence,
        'annotation_path': str(ann_path),
        'original_image_path': str(original_image_path),
    }


def calculate_mask_iou(mask1, mask2):
    """Calculate IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def hungarian_match_masks(gt_masks, pred_masks, iou_threshold=REFERENCE_MATCH_IOU_THRESHOLD):
    """Match GT and predicted masks using Hungarian assignment on IoU."""
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=float)

    if n_gt == 0 or n_pred == 0:
        return [], iou_matrix

    for i, gt_mask in enumerate(gt_masks):
        for j, pred_mask in enumerate(pred_masks):
            iou_matrix[i, j] = calculate_mask_iou(gt_mask, pred_mask)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches = []
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            matches.append((int(i), int(j), float(iou_matrix[i, j])))

    return matches, iou_matrix


def evaluate_masks_against_reference(gt_masks, filtered_masks, match_confidence,
                                     iou_threshold=REFERENCE_MATCH_IOU_THRESHOLD):
    """Compute object-level segmentation metrics against reference masks."""
    pred_masks = [mask['segmentation'].astype(bool) for mask in filtered_masks]
    matches, iou_matrix = hungarian_match_masks(gt_masks, pred_masks, iou_threshold=iou_threshold)

    matched_gt = {m[0] for m in matches}
    matched_pred = {m[1] for m in matches}
    matched_ious = [m[2] for m in matches]
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    n_matched = len(matches)

    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'pred_masks': pred_masks,
        'matches': matches,
        'matched_gt_indices': sorted(matched_gt),
        'matched_pred_indices': sorted(matched_pred),
        'n_gt': n_gt,
        'n_pred': n_pred,
        'n_matched': n_matched,
        'n_false_positive': n_pred - n_matched,
        'n_false_negative': n_gt - n_matched,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': float(np.mean(matched_ious)) if matched_ious else 0.0,
        'median_iou': float(np.median(matched_ious)) if matched_ious else 0.0,
        'iou_matrix': iou_matrix,
        'match_confidence': float(match_confidence),
    }


def build_reference_error_overlay(gt_masks, pred_masks,
                                  matched_gt_indices=None, matched_pred_indices=None):
    """
    Build an object-level confusion overlay.

    Green = matched objects, Red = unmatched predicted masks, Blue = unmatched GT masks.
    """
    if not gt_masks and not pred_masks:
        return None, {'correct_pixels': 0, 'false_positive_pixels': 0, 'false_negative_pixels': 0}

    shape = gt_masks[0].shape if gt_masks else pred_masks[0].shape
    matched_gt_indices = set(matched_gt_indices or [])
    matched_pred_indices = set(matched_pred_indices or [])

    correct = np.zeros(shape, dtype=bool)
    false_positive = np.zeros(shape, dtype=bool)
    false_negative = np.zeros(shape, dtype=bool)

    if matched_gt_indices or matched_pred_indices:
        for idx, mask in enumerate(gt_masks):
            mask_bool = mask.astype(bool)
            if idx in matched_gt_indices:
                correct |= mask_bool
            else:
                false_negative |= mask_bool

        for idx, mask in enumerate(pred_masks):
            mask_bool = mask.astype(bool)
            if idx in matched_pred_indices:
                correct |= mask_bool
            else:
                false_positive |= mask_bool
    else:
        gt_union = np.zeros(shape, dtype=bool)
        pred_union = np.zeros(shape, dtype=bool)

        for mask in gt_masks:
            gt_union |= mask
        for mask in pred_masks:
            pred_union |= mask

        correct = gt_union & pred_union
        false_positive = pred_union & (~gt_union)
        false_negative = gt_union & (~pred_union)

    overlay = np.zeros((*shape, 4), dtype=float)
    overlay[correct] = REFERENCE_ERROR_COLORS['correct']
    overlay[false_positive] = REFERENCE_ERROR_COLORS['false_positive']
    overlay[false_negative] = REFERENCE_ERROR_COLORS['false_negative']

    return overlay, {
        'correct_pixels': int(correct.sum()),
        'false_positive_pixels': int(false_positive.sum()),
        'false_negative_pixels': int(false_negative.sum()),
    }


def normalize_image_for_sam(image):
    """
    Normalize preprocessing output to SAM-expected format (HWC, 3-channel, uint8).
    """
    if image is None:
        raise ValueError("Input image for SAM is None")

    out = image

    # Channel normalization
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.ndim == 3 and out.shape[2] == 1:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.ndim == 3 and out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    elif out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(f"Unsupported image shape for SAM: {out.shape}")

    # Dtype/range normalization
    if out.dtype != np.uint8:
        if np.issubdtype(out.dtype, np.floating):
            finite = np.isfinite(out)
            if finite.any():
                vmin = float(np.nanmin(out[finite]))
                vmax = float(np.nanmax(out[finite]))
            else:
                vmin, vmax = 0.0, 0.0

            if 0.0 <= vmin and vmax <= 1.0:
                out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            else:
                out = np.clip(out, 0, 255).astype(np.uint8)
        else:
            out = np.clip(out, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(out)


def build_sam_mask_generator(model, sam_config):
    """Build SAM2AutomaticMaskGenerator from unified config."""
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=sam_config["points_per_side"],
        points_per_batch=sam_config["points_per_batch"],
        pred_iou_thresh=sam_config["pred_iou_thresh"],
        stability_score_thresh=sam_config["stability_score_thresh"],
        crop_n_layers=sam_config["crop_n_layers"],
        crop_n_points_downscale_factor=sam_config["crop_n_points_downscale_factor"],
        crop_nms_thresh=sam_config["crop_nms_thresh"],
        box_nms_thresh=sam_config["box_nms_thresh"],
        use_m2m=sam_config["use_m2m"],
    )


def generate_masks_with_fallback(model, image, sam_config):
    """
    Generate SAM masks with fallback for known SAM2 crop-layer edge case.

    In some SAM2 builds, when crop_n_layers > 0 and no masks are produced in
    all crops, internal crop_boxes may become 1D empty tensor and trigger:
    IndexError: too many indices for tensor of dimension 1
    """
    mask_generator = build_sam_mask_generator(model, sam_config)
    try:
        masks = mask_generator.generate(image)
        return masks, sam_config["crop_n_layers"], False
    except IndexError as e:
        # Known SAM2 edge-case fallback
        if "too many indices for tensor of dimension 1" not in str(e):
            raise

        print("   [WARN] SAM2 crop-layer edge case detected (empty crop_boxes).")
        print("   [WARN] Retrying SAM with crop_n_layers=0 for this method/image.")

        fallback_config = dict(sam_config)
        fallback_config["crop_n_layers"] = 0
        fallback_generator = build_sam_mask_generator(model, fallback_config)

        try:
            masks = fallback_generator.generate(image)
            return masks, 0, True
        except Exception as fallback_error:
            print(f"   [WARN] Fallback SAM generation failed: {fallback_error}")
            print("   [WARN] Continuing with empty mask list.")
            return [], 0, True


def filter_overlapping_masks(masks):
    """Compatibility wrapper around the canonical SAM overlap filter."""
    return filter_overlapping_masks_by_centroid(masks)


def build_sam_overlay(filtered_masks, alpha=0.35):
    """
    Build a transparent RGBA overlay from SAM masks.

    Args:
        filtered_masks: List of SAM mask dicts with 'segmentation'
        alpha: Overlay alpha

    Returns:
        RGBA overlay array or None when no masks exist
    """
    if len(filtered_masks) == 0:
        return None

    sorted_masks = sorted(
        filtered_masks,
        key=(lambda x: x.get('area', int(np.sum(x['segmentation'])))),
        reverse=True
    )
    h, w = sorted_masks[0]['segmentation'].shape
    overlay = np.zeros((h, w, 4), dtype=float)

    # Keep SAM mask color consistent across preprocessing methods.
    fixed_hex = palette_hex(1)
    fixed_rgb = tuple(int(fixed_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))

    for mask in sorted_masks:
        m = mask['segmentation']
        overlay[m] = np.array([fixed_rgb[0], fixed_rgb[1], fixed_rgb[2], alpha], dtype=float)

    return overlay


def create_method_figure(method_count, panel_width=6.0, panel_height=5.0):
    """Create a compact dynamic grid and hide any unused axes."""
    if method_count < 1:
        raise ValueError("At least one method is required for a visualization")
    ncols = min(3, method_count)
    nrows = int(np.ceil(method_count / ncols))
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False,
    )
    axes = list(axes_grid.ravel())
    for unused_ax in axes[method_count:]:
        unused_ax.axis("off")
    return fig, axes[:method_count]

#%%
# ============================================================================
# PREPROCESSING METHODS
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING ALL METHODS")
print("="*80)

preprocessing_methods = PREPROCESSING_METHODS

skipped_images = []

#%%
# ============================================================================
# PROCESS ALL IMAGES
# ============================================================================

# Loop over images that still need processing
for img_idx, image_path in enumerate(images_to_process, 1):
    print("\n" + "="*80)
    print(f"PROCESSING IMAGE {img_idx}/{len(images_to_process)}: {os.path.basename(image_path)}")
    print("="*80)

    # Re-evaluate completion in case another process finished this image.
    image_name = Path(image_path).stem
    append_processing_log("started", image_name, image_path, "running")
    existing_cache = _read_cache_data(image_name, OUTPUT_DIR, warn=True)
    missing_method_keys = get_missing_method_keys(
        image_name,
        OUTPUT_DIR,
        cache_data=existing_cache,
    )
    if not missing_method_keys:
        print(f"[OK] All method results already exist for {os.path.basename(image_path)}, skipping...")
        append_processing_log("skipped", image_name, image_path, "already_complete", reason="Results already exist")
        continue
    print("[INFO] Methods to run: " + ", ".join(METHOD_LABELS[key] for key in missing_method_keys))

    # Load image directly (Dataset images are already cropped)
    current_image = cv2.imread(image_path)
    if current_image is None:
        reason = "Failed to load image with cv2.imread"
        print(f"[OK] Failed to load {image_path}, skipping...")
        skipped_images.append({
            'image_name': image_name,
            'image_path': image_path,
            'failed_method': 'image_load',
            'error': reason,
        })
        append_skipped_image_log(image_name, image_path, "image_load", reason)
        continue

    # Ensure even dimensions for Noise2SR compatibility
    particle_part = ensure_even_dimensions(current_image)

    print(f"[OK] Loaded image:")
    print(f"   Image: {os.path.basename(image_path)}")
    print(f"   Original shape: {current_image.shape}")
    print(f"   Processing shape: {particle_part.shape} (even dimensions)")

    # Store original for comparison
    particle_part_original = particle_part.copy()

    reference_data = load_reference_masks(
        image_path,
        reference_ann_dir=ARGS.reference_ann_dir,
        reference_image_dir=ARGS.reference_image_dir,
    )
    if reference_data is not None:
        proc_h, proc_w = particle_part_original.shape[:2]
        reference_data['gt_masks'] = [
            mask[:proc_h, :proc_w] for mask in reference_data.get('gt_masks', [])
        ]
        print(f"[OK] Loaded reference masks: {len(reference_data['gt_masks'])}")
        print(f"   Annotation: {os.path.basename(reference_data['annotation_path'])}")
        print(f"   Crop match confidence: {reference_data['match_confidence']:.3f}")
    else:
        print("[WARN] Reference masks not found - skipping reference error visualization")

    # Storage for this image's results
    all_results = {}
    preprocessing_cache = {}
    skip_current_image = False
    skip_reason = None
    failed_method_name = None

    # ========================================================================
    # PROCESS EACH METHOD FOR THIS IMAGE
    # ========================================================================

    for method_key in METHOD_KEYS:
        config = preprocessing_methods[method_key]
        if method_key not in missing_method_keys:
            print(f"\n[CACHE] Keeping existing result: {config['name']}")
            continue

        print(f"\n{'='*80}")
        print(f"PROCESSING: {config['name']}")
        print(f"{'='*80}")

        result = {
            'method_name': config['name'],
            'method_key': method_key
        }

        # ========================================================================
        # STEP 1: PREPROCESSING
        # ========================================================================
        print(f"\n[1/4] Preprocessing...")

        try:
            own_stage_path = get_preprocessed_path(image_name, method_key, OUTPUT_DIR)
            saved_stage = None
            saved_stage_source = own_stage_path
            if os.path.exists(own_stage_path):
                saved_stage = cv2.imread(own_stage_path, cv2.IMREAD_COLOR)
            elif method_key == "1_bm3d":
                optimizer_stage_path = get_optimizer_bm3d_stage_path(image_name)
                if os.path.exists(optimizer_stage_path):
                    saved_stage = cv2.imread(optimizer_stage_path, cv2.IMREAD_COLOR)
                    saved_stage_source = optimizer_stage_path

            if (
                saved_stage is not None
                and saved_stage.shape[:2] != particle_part_original.shape[:2]
            ):
                print(
                    f"[WARN] Ignoring incompatible preprocessing stage: "
                    f"{saved_stage_source} ({saved_stage.shape[:2]} != "
                    f"{particle_part_original.shape[:2]})"
                )
                saved_stage = None

            if saved_stage is not None:
                preprocessed = saved_stage
                cached_preproc_info = existing_cache.get(method_key, {}).get(
                    "preprocessing_info"
                )
                preproc_info = _complete_preprocessing_info(
                    cached_preproc_info,
                    method_key,
                )
                print(
                    f"[CACHE] Reusing completed preprocessing stage: "
                    f"{os.path.basename(saved_stage_source)}"
                )
            elif ARGS.reuse_preprocessed_only:
                if method_key != "0_raw":
                    raise FileNotFoundError(
                        "Preprocessed-only re-evaluation forbids regenerating a "
                        f"missing stage: {own_stage_path}"
                    )
                preprocessed = particle_part_original.copy()
                preproc_info = _complete_preprocessing_info(None, method_key)
                preproc_info["reused_source_image_as_identity_stage"] = True
                print("[CACHE] Raw-image identity stage reused from Dataset input")
            elif config.get('reuse_from'):
                reuse_from = config['reuse_from']
                if reuse_from not in preprocessing_cache:
                    dependency_path = get_preprocessed_path(image_name, reuse_from, OUTPUT_DIR)
                    dependency_image = cv2.imread(dependency_path, cv2.IMREAD_COLOR)
                    if dependency_image is None:
                        raise RuntimeError(
                            f"Required preprocessing stage is missing: {dependency_path}"
                        )
                    dependency_metrics = existing_cache.get(reuse_from, {})
                    dependency_info = dependency_metrics.get("preprocessing_info")
                    dependency_info = _complete_preprocessing_info(
                        dependency_info,
                        reuse_from,
                    )
                    preprocessing_cache[reuse_from] = {
                        "image": dependency_image,
                        "info": dependency_info,
                    }
                    print(f"[CACHE] Loaded reusable stage: {os.path.basename(dependency_path)}")
                cached_preprocessing = preprocessing_cache[reuse_from]
                print(f"\n[INFO] Reusing preprocessing from: {preprocessing_methods[reuse_from]['name']}")
                reuse_stage = config.get("reuse_stage")
                if reuse_stage == "clahe":
                    preprocessed, preproc_info = apply_clahe_to_preprocessed(
                        cached_preprocessing['image'],
                        cached_preprocessing['info'],
                        reused_from=reuse_from,
                        method_key=method_key,
                        verbose=True,
                    )
                elif reuse_stage == "bm3d":
                    preprocessed, preproc_info = apply_bm3d_to_preprocessed(
                        cached_preprocessing['image'],
                        cached_preprocessing['info'],
                        reused_from=reuse_from,
                        method_key=method_key,
                        verbose=True,
                    )
                elif reuse_stage == "noise2sr":
                    preprocessed, preproc_info = apply_noise2sr_to_preprocessed(
                        cached_preprocessing['image'],
                        cached_preprocessing['info'],
                        reused_from=reuse_from,
                        method_key=method_key,
                        verbose=True,
                    )
                else:
                    raise ValueError(f"Unsupported reusable stage: {reuse_stage}")
            else:
                preprocessed, preproc_info = preprocess_with_config(
                    particle_part_original.copy(),
                    bm3d_enabled=config['bm3d'],
                    noise2sr_enabled=config['noise2sr'],
                    clahe_enabled=config['clahe'],
                    verbose=True
                )
                preproc_info = _complete_preprocessing_info(preproc_info, method_key)
        except RuntimeError as exc:
            if not is_cuda_runtime_error(exc):
                raise
            skip_current_image = True
            skip_reason = str(exc)
            failed_method_name = config['name']
            print(f"\n[WARN] CUDA runtime error during preprocessing for {config['name']}.")
            print(f"       Skipping image {os.path.basename(image_path)} and moving to next one.")
            cleanup_after_cuda_error()
            break

        result['preprocessing_info'] = preproc_info
        preprocessing_cache[method_key] = {
            'image': preprocessed.copy(),
            'info': copy.deepcopy(preproc_info),
        }
        result['preprocessed_image_raw'] = preprocessed.copy()

        # Persist the reusable preprocessing stage immediately. The two
        # Noise2SR-first children use the in-memory image during this run; the
        # PNG also avoids repeating a completed stage after an interruption.
        preprocessed_path = get_preprocessed_path(image_name, method_key, OUTPUT_DIR)
        image_to_save = normalize_image_for_sam(result['preprocessed_image_raw'])
        if not cv2.imwrite(preprocessed_path, image_to_save):
            raise IOError(f"Failed to save preprocessed image: {preprocessed_path}")
        print(f"[OK] Saved preprocessing stage: {os.path.basename(preprocessed_path)}")

        print(f"\n[OK] Preprocessing complete")
        print(f"   Method: {preproc_info['denoising_method']}")

        # ========================================================================
        # STEP 2: SAM SEGMENTATION
        # ========================================================================
        print(f"\n[2/4] SAM Segmentation...")

        preprocessed = normalize_image_for_sam(preprocessed)
        try:
            masks, crop_layers_used, used_sam_fallback = generate_masks_with_fallback(
                sam, preprocessed, SAM_CONFIG
            )
        except RuntimeError as exc:
            if not is_cuda_runtime_error(exc):
                raise
            skip_current_image = True
            skip_reason = str(exc)
            failed_method_name = config['name']
            print(f"\n[WARN] CUDA runtime error during SAM segmentation for {config['name']}.")
            print(f"       Skipping image {os.path.basename(image_path)} and moving to next one.")
            cleanup_after_cuda_error()
            break
        result['sam_crop_n_layers_used'] = crop_layers_used
        result['sam_fallback_used'] = used_sam_fallback

        result['sam_total_masks'] = len(masks)

        # Calculate SAM metrics
        pred_ious = [mask['predicted_iou'] for mask in masks]
        stability_scores = [mask['stability_score'] for mask in masks]
        mask_areas = [np.sum(mask['segmentation']) for mask in masks]

        result['sam_pred_iou_mean'] = np.mean(pred_ious) if pred_ious else 0
        result['sam_pred_iou_std'] = np.std(pred_ious) if pred_ious else 0
        result['sam_stability_mean'] = np.mean(stability_scores) if stability_scores else 0
        result['sam_stability_std'] = np.std(stability_scores) if stability_scores else 0
        result['sam_area_mean'] = np.mean(mask_areas) if mask_areas else 0
        result['sam_area_std'] = np.std(mask_areas) if mask_areas else 0
        result['sam_area_cv'] = (result['sam_area_std'] / result['sam_area_mean']) if result['sam_area_mean'] > 0 else 0

        print(f"\n[OK] SAM Segmentation complete")
        print(f"   Total masks: {result['sam_total_masks']}")
        print(f"   Pred IoU: {result['sam_pred_iou_mean']:.3f} +/- {result['sam_pred_iou_std']:.3f}")
        print(f"   Stability: {result['sam_stability_mean']:.3f} +/- {result['sam_stability_std']:.3f}")
        print(f"   Mask area: {result['sam_area_mean']:.1f} +/- {result['sam_area_std']:.1f} (CV={result['sam_area_cv']:.3f})")

        # ========================================================================
        # STEP 3: MASK FILTERING (Background + Overlap Removal)
        # ========================================================================
        print(f"\n[3/4] Mask Filtering...")

        # Handle empty masks case
        if len(masks) == 0:
            print(f"[INFO] No masks detected - skipping mask filtering")
            filtered_masks = []
            result['bg_removed_masks'] = 0
            result['bg_filter_ratio'] = 0
            result['overlap_removed_masks'] = 0
            result['overlap_filter_ratio'] = 0
            result['size_filtered_masks'] = 0
            result['total_filter_ratio'] = 0
        else:
            masks_before_bg = len(masks)
            filtered_masks = filter_sam_background_masks(
                masks,
                preprocessed.shape[:2],
            )

            result['bg_removed_masks'] = len(filtered_masks)
            result['bg_filter_ratio'] = len(filtered_masks) / masks_before_bg if masks_before_bg > 0 else 0
            masks_before_overlap = len(filtered_masks)
            filtered_masks = filter_overlapping_masks(filtered_masks)
            result['overlap_removed_masks'] = masks_before_overlap - len(filtered_masks)
            result['overlap_filter_ratio'] = (
                len(filtered_masks) / masks_before_overlap if masks_before_overlap > 0 else 0
            )
            result['size_filtered_masks'] = len(filtered_masks)
            result['total_filter_ratio'] = len(filtered_masks) / masks_before_bg if masks_before_bg > 0 else 0

        print(f"\n[OK] Mask filtering complete")
        print(f"   After BG removal: {result['bg_removed_masks']} ({result['bg_filter_ratio']*100:.1f}%)")
        print(
            f"   After overlap removal: {result['size_filtered_masks']} "
            f"({result['overlap_filter_ratio']*100:.1f}% of BG-filtered masks)"
        )
        print(f"   Final masks: {result['size_filtered_masks']} ({result['total_filter_ratio']*100:.1f}%)")

        if reference_data is not None:
            reference_eval = evaluate_masks_against_reference(
                reference_data.get('gt_masks', []),
                filtered_masks,
                reference_data.get('match_confidence', 0.0),
            )
            error_overlay, pixel_stats = build_reference_error_overlay(
                reference_data.get('gt_masks', []),
                reference_eval.get('pred_masks', []),
                matched_gt_indices=reference_eval.get('matched_gt_indices', []),
                matched_pred_indices=reference_eval.get('matched_pred_indices', []),
            )
            reference_eval.update(pixel_stats)
            result['reference_eval'] = reference_eval
            result['reference_error_overlay'] = error_overlay

            print("\n[OK] Reference comparison complete")
            print(f"   Matched: {reference_eval['n_matched']}/{reference_eval['n_gt']}")
            print(f"   FP: {reference_eval['n_false_positive']}, FN: {reference_eval['n_false_negative']}")
            print(f"   Precision/Recall/F1: {reference_eval['precision']:.3f} / "
                  f"{reference_eval['recall']:.3f} / {reference_eval['f1']:.3f}")
            print(f"   Mean matched IoU: {reference_eval['mean_iou']:.3f}")
        else:
            result['reference_eval'] = {}
            result['reference_error_overlay'] = None

        # ========================================================================
        # STEP 4: CLIP CLASSIFICATION
        # ========================================================================
        print(f"\n[4/4] CLIP Classification...")

        # Handle case with no filtered masks
        if len(filtered_masks) == 0:
            print("   [INFO] No filtered masks - skipping CLIP classification")
            shapes, confidences, shape_counts = [], [], Counter()
        else:
            # CLIP unified parameters
            # Reference: Radford et al., "Learning Transferable Visual Models" (2021)
            try:
                shapes, confidences, shape_counts = classify_shapes_with_clip(
                    filtered_masks, preprocessed, shape_labels, shape_descriptions,
                    clip_model, clip_preprocess, text_tokens, device, batch_size=64,
                    confidence_threshold=None  # Uses model's learned logit_scale
                )
            except RuntimeError as exc:
                if not is_cuda_runtime_error(exc):
                    raise
                skip_current_image = True
                skip_reason = str(exc)
                failed_method_name = config['name']
                print(f"\n[WARN] CUDA runtime error during CLIP classification for {config['name']}.")
                print(f"       Skipping image {os.path.basename(image_path)} and moving to next one.")
                cleanup_after_cuda_error()
                break

        result['clip_total_classified'] = len(shapes)
        result['clip_confidence_mean'] = np.mean(confidences) if confidences else 0
        result['clip_confidence_std'] = np.std(confidences) if confidences else 0
        result['clip_high_conf_ratio'] = np.sum(np.array(confidences) > 0.9) / len(confidences) if confidences else 0
        result['clip_shape_diversity'] = calculate_shape_diversity(shapes) if shapes else 0

        # Dominant shape
        if shape_counts:
            dominant_shape = max(shape_counts.items(), key=lambda x: x[1])
            result['clip_dominant_shape'] = dominant_shape[0]
            result['clip_dominant_ratio'] = dominant_shape[1] / len(shapes) if shapes else 0
        else:
            result['clip_dominant_shape'] = 'None'
            result['clip_dominant_ratio'] = 0

        result['clip_shape_counts'] = shape_counts

        print(f"\n[OK] CLIP Classification complete")
        print(f"   Classified: {result['clip_total_classified']}")
        print(f"   Confidence: {result['clip_confidence_mean']:.3f} +/- {result['clip_confidence_std']:.3f}")
        print(f"   High conf (>0.9): {result['clip_high_conf_ratio']*100:.1f}%")
        print(f"   Shape diversity: {result['clip_shape_diversity']:.3f}")
        print(f"   Dominant shape: {result['clip_dominant_shape']} ({result['clip_dominant_ratio']*100:.1f}%)")

        # Store results
        result['preprocessed_image'] = preprocessed
        result['masks'] = masks
        result['filtered_masks'] = filtered_masks
        result['shapes'] = shapes
        result['confidences'] = confidences

        all_results[method_key] = result
        save_results_cache(
            image_name,
            {method_key: result},
            OUTPUT_DIR,
            include_mask_data=SINGLE_IMAGE_MODE,
            reference_data=reference_data,
        )
        existing_cache = _read_cache_data(image_name, OUTPUT_DIR, warn=True)
        append_processing_log(
            "method_complete",
            image_name,
            image_path,
            "complete",
            method=config['name'],
            reason="Method metrics merged into cache",
        )

        print(f"\n{'='*80}")
        print(f"[OK] {config['name']} COMPLETE")
        print(f"{'='*80}")

    if skip_current_image:
        failed_method = failed_method_name or 'unknown'
        error_reason = skip_reason or 'CUDA runtime error'
        skipped_images.append({
            'image_name': image_name,
            'image_path': image_path,
            'failed_method': failed_method,
            'error': error_reason,
        })
        append_skipped_image_log(image_name, image_path, failed_method, error_reason)
        print("\n" + "="*80)
        print(f"[WARN] IMAGE PAUSED: {os.path.basename(image_path)}")
        print(f"   Failed method: {failed_method}")
        print(f"   Reason: {error_reason}")
        if all_results:
            print("   Completed methods were kept in the cache for the next run.")
        print("="*80)
        continue

    # ========================================================================
    # CREATE COMPARISON DATAFRAME FOR THIS IMAGE
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING COMPARISON TABLE")
    print("="*80)

    complete_cache = _read_cache_data(image_name, OUTPUT_DIR, warn=True)
    still_missing = get_missing_method_keys(
        image_name,
        OUTPUT_DIR,
        cache_data=complete_cache,
    )
    if still_missing:
        raise RuntimeError(
            "Image finished without all method results: "
            + ", ".join(still_missing)
        )

    comparison_data = []

    for method_key in METHOD_KEYS:
        metrics = complete_cache[method_key]

        # Core pipeline metrics plus reference segmentation metrics when available.
        row = {
            'Method Key': method_key,
            'Method': metrics.get('method_name', METHOD_LABELS[method_key]),
            'Pred IoU': metrics.get('sam_pred_iou_mean', np.nan),
            'Stability': metrics.get('sam_stability_mean', np.nan),
            'Final Particles': metrics.get('size_filtered_masks', np.nan),
            'Mean Confidence': metrics.get('clip_confidence_mean', np.nan),
        }

        if metrics.get('reference_available', False):
            row.update({
                'FP Masks': metrics.get('reference_false_positive_masks', 0),
                'Precision': metrics.get('reference_precision', np.nan),
                'Recall': metrics.get('reference_recall', np.nan),
                'F1 Score': metrics.get('reference_f1', np.nan),
            })
        else:
            row.update({
                'FP Masks': '',
                'Precision': '',
                'Recall': '',
                'F1 Score': '',
            })

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(comparison_df.to_string(index=False))

    # ========================================================================
    # VISUALIZATIONS FOR THIS IMAGE
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Extract image name for file naming
    # (OUTPUT_DIR already defined and created earlier)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    method_keys_ordered = [key for key in METHOD_KEYS if key in all_results]
    visual_suffix = "" if len(method_keys_ordered) == len(METHOD_KEYS) else "_additional_methods"

    print(
        f"\n[INFO] Building mask visualizations for {len(method_keys_ordered)} "
        "newly processed method(s); existing legacy visualizations are preserved."
    )

    # ========================================================================
    # VIS 1: CLIP Shape Overlay Grid
    # ========================================================================
    print("\nCreating CLIP shape overlay grid...")

    fig, axes = create_method_figure(len(method_keys_ordered))

    shape_colors = generate_shape_colors(shape_labels)

    for ax, method_key in zip(axes, method_keys_ordered):
        result = all_results[method_key]

        # Display base image
        display_image = prepare_display_image(result['preprocessed_image'])
        if display_image.ndim == 2:
            ax.imshow(display_image, cmap='gray')
        else:
            ax.imshow(display_image)

        # Overlay colored masks for each shape
        for mask, shape in zip(result['filtered_masks'], result['shapes']):
            segmentation = mask['segmentation']
            color = shape_colors.get(shape, (0.5, 0.5, 0.5))  # Gray for unknown

            # Create RGBA overlay with transparency
            overlay = np.zeros((*segmentation.shape, 4))
            overlay[segmentation] = [*color, 0.5]  # RGBA with alpha=0.5

            ax.imshow(overlay)

        title = f"{result['method_name']}\n"
        title += f"Masks={result['size_filtered_masks']}, CLIP Conf={result['clip_confidence_mean']:.3f}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    fig.suptitle('CLIP Shape Classification Results Comparison', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        os.path.join(OUTPUT_DIR, f'{image_name}{visual_suffix}_CLIP.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)

    # ========================================================================
    # VIS 2: SAM Masks Grid
    # ========================================================================
    print("Creating SAM masks grid...")

    fig, axes = create_method_figure(len(method_keys_ordered))
    sam_overlays = {}

    for ax, method_key in zip(axes, method_keys_ordered):
        result = all_results[method_key]

        # Display base image
        display_image = prepare_display_image(result['preprocessed_image'])
        if display_image.ndim == 2:
            ax.imshow(display_image, cmap='gray')
        else:
            ax.imshow(display_image)

        overlay = build_sam_overlay(result['filtered_masks'], alpha=0.35)
        sam_overlays[method_key] = overlay
        if overlay is not None:
            ax.imshow(overlay)

        title = f"{result['method_name']}\n"
        title += f"Masks: {result['size_filtered_masks']}, IoU={result['sam_pred_iou_mean']:.3f}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    fig.suptitle('SAM Segmentation Results Comparison', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        os.path.join(OUTPUT_DIR, f'{image_name}{visual_suffix}_SAM.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)

    # Save no-text SAM grid (image + SAM masks only)
    fig_nt, axes_nt = create_method_figure(len(method_keys_ordered))
    for ax_nt, method_key in zip(axes_nt, method_keys_ordered):
        result = all_results[method_key]
        display_image = prepare_display_image(result['preprocessed_image'])
        if display_image.ndim == 2:
            ax_nt.imshow(display_image, cmap='gray')
        else:
            ax_nt.imshow(display_image)
        overlay = sam_overlays.get(method_key)
        if overlay is not None:
            ax_nt.imshow(overlay)
        ax_nt.axis('off')
        ax_nt.set_title('')
    fig_nt.tight_layout(pad=0.05)
    fig_nt.savefig(
        os.path.join(OUTPUT_DIR, f'{image_name}{visual_suffix}_SAM_notext.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig_nt)

    # Single-image mode: save per-method SAM notext files.
    if SINGLE_IMAGE_MODE:
        for method_key in method_keys_ordered:
            result = all_results[method_key]
            fig_single, ax_single = plt.subplots(figsize=(6, 6))
            display_image = prepare_display_image(result['preprocessed_image'])
            if display_image.ndim == 2:
                ax_single.imshow(display_image, cmap='gray')
            else:
                ax_single.imshow(display_image)
            overlay = sam_overlays.get(method_key)
            if overlay is not None:
                ax_single.imshow(overlay)
            ax_single.axis('off')
            ax_single.set_title('')
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f'{image_name}_{method_key}_SAM_notext.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig_single)

    if reference_data is not None:
        print("Creating reference error grid...")

        fig, axes = create_method_figure(len(method_keys_ordered))

        for ax, method_key in zip(axes, method_keys_ordered):
            result = all_results[method_key]
            reference_eval = result.get('reference_eval', {})
            display_image = prepare_display_image(result['preprocessed_image'])
            if display_image.ndim == 2:
                ax.imshow(display_image, cmap='gray')
            else:
                ax.imshow(display_image)

            error_overlay = result.get('reference_error_overlay')
            if error_overlay is not None:
                ax.imshow(error_overlay)

            title = f"{result['method_name']}\n"
            title += (
                f"TP={reference_eval.get('n_matched', 0)}, "
                f"FP={reference_eval.get('n_false_positive', 0)}, "
                f"FN={reference_eval.get('n_false_negative', 0)}, "
                f"F1={reference_eval.get('f1', 0):.3f}"
            )
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

        legend_handles = [
            Patch(
                facecolor=REFERENCE_ERROR_COLORS['correct'][:3],
                alpha=REFERENCE_ERROR_COLORS['correct'][3],
                label='Matched'
            ),
            Patch(
                facecolor=REFERENCE_ERROR_COLORS['false_positive'][:3],
                alpha=REFERENCE_ERROR_COLORS['false_positive'][3],
                label='False Positive'
            ),
            Patch(
                facecolor=REFERENCE_ERROR_COLORS['false_negative'][:3],
                alpha=REFERENCE_ERROR_COLORS['false_negative'][3],
                label='False Negative'
            ),
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.01), fontsize=11)
        fig.suptitle('Reference Error Comparison (Green=Matched, Red=FP, Blue=FN)',
                     fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        fig.savefig(
            os.path.join(OUTPUT_DIR, f'{image_name}{visual_suffix}_reference_error.png'),
            dpi=300,
            bbox_inches='tight',
        )
        plt.close(fig)

    # ========================================================================
    # VIS 4: Metrics Comparison Bar Charts
    # ========================================================================
    print("Creating metrics comparison charts...")

    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    fig.suptitle('Quantitative Metrics Comparison', fontsize=16, fontweight='bold')
    methods = [METHOD_LABELS[key].replace(" + ", "\n+ ").replace(" > ", "\n> ") for key in METHOD_KEYS]
    x_positions = np.arange(len(METHOD_KEYS))
    metric_plot_configs = (
        ("sam_pred_iou_mean", "sam_pred_iou_std", "SAM: Predicted IoU (mean)", "Score", 1.05, False),
        ("sam_stability_mean", "sam_stability_std", "SAM: Stability Score (mean)", "Score", 1.05, False),
        ("size_filtered_masks", None, "Final Particle Count (after filtering)", "Count", None, True),
        ("clip_confidence_mean", "clip_confidence_std", "CLIP: Mean Confidence", "Confidence", 1.05, False),
    )

    for ax, (value_key, error_key, title, ylabel, y_cap, is_count) in zip(
        axes.ravel(), metric_plot_configs
    ):
        values = [float(complete_cache[key].get(value_key, 0)) for key in METHOD_KEYS]
        if is_count:
            errors = [float(np.sqrt(max(value, 0))) for value in values]
        else:
            errors = [float(complete_cache[key].get(error_key, 0)) for key in METHOD_KEYS]
        bars = ax.bar(
            x_positions,
            values,
            yerr=errors,
            color=[METHOD_COLORS_BY_KEY[key] for key in METHOD_KEYS],
            capsize=4,
            error_kw={'linewidth': 1.2, 'ecolor': palette_hex(0)},
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        if y_cap is not None:
            ax.set_ylim(0, y_cap)
        else:
            ymax = max((value + error for value, error in zip(values, errors)), default=1)
            ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        for bar, value in zip(bars, values):
            value_text = f"{int(value)}" if is_count else f"{value:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_text,
                ha='center',
                va='bottom',
                fontsize=7,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT_DIR, f'{image_name}_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ============================================================================
    # SAVE DETAILED RESULTS
    # ============================================================================
    print("\nSaving detailed results...")
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, f'{image_name}_comparison_table.csv'), index=False, encoding='utf-8-sig')
    print(f"[OK] Saved: {image_name}_comparison_table.csv")
    
    # Save detailed report as Excel with multiple sheets
    report_path = os.path.join(OUTPUT_DIR, f'{image_name}_report.xlsx')
    
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        # Sheet 1: Preprocessing comparison with GENERAL INFORMATION
        general_info_data = []
        general_info_data.append(['GENERAL INFORMATION', ''])
        general_info_data.append(['Image', os.path.basename(image_path)])
        general_info_data.append(['Particle region shape', str(particle_part.shape)])
        general_info_data.append(['', ''])
        general_info_data.append(['', ''])
    
        # Add comparison table header row
        general_info_data.append(list(comparison_df.columns))
    
        # Add comparison data
        for _, row in comparison_df.iterrows():
            general_info_data.append(list(row))
    
        # Create DataFrame and save
        general_df = pd.DataFrame(general_info_data)
        general_df.to_excel(writer, sheet_name='Preprocessing comparison', index=False, header=False)
    
        # One detail sheet per method. Values come directly from the merged
        # cache; legacy fields that were never cached remain explicitly N/A.
        for method_key in METHOD_KEYS:
            metrics = complete_cache[method_key]
            preproc_info = _complete_preprocessing_info(
                metrics.get('preprocessing_info'),
                method_key,
            )
            detail_data = [
                ['PREPROCESSING', ''],
                ['Method key', method_key],
                ['Method label', metrics.get('method_name', METHOD_LABELS[method_key])],
                ['Order', preproc_info['denoising_method']],
            ]
            if preproc_info.get('bm3d_applied'):
                bm3d_info = preproc_info.get('bm3d_info', {})
                detail_data.append(['BM3D sigma_psd', bm3d_info.get('sigma_psd', 'N/A')])
            if preproc_info.get('noise2sr_applied'):
                n2sr_info = preproc_info.get('noise2sr_info', {})
                detail_data.append(['Noise2SR epochs', n2sr_info.get('epochs_trained', 'N/A')])

            detail_data.extend([
                ['', ''],
                ['SAM PERFORMANCE', ''],
                ['Total masks', metrics.get('sam_total_masks', np.nan)],
                ['Pred IoU (mean)', metrics.get('sam_pred_iou_mean', np.nan)],
                ['Pred IoU (std)', metrics.get('sam_pred_iou_std', np.nan)],
                ['Stability (mean)', metrics.get('sam_stability_mean', np.nan)],
                ['Stability (std)', metrics.get('sam_stability_std', np.nan)],
                ['Mask area (mean, px^2)', metrics.get('sam_area_mean', np.nan)],
                ['Mask area (std, px^2)', metrics.get('sam_area_std', np.nan)],
                ['Mask area CV', metrics.get('sam_area_cv', np.nan)],
                ['', ''],
                ['FILTERING', ''],
                ['After BG removal', metrics.get('bg_removed_masks', np.nan)],
                ['BG removal ratio', metrics.get('bg_filter_ratio', np.nan)],
                ['Overlap masks removed', metrics.get('overlap_removed_masks', 'N/A')],
                ['Overlap retention ratio', metrics.get('overlap_filter_ratio', 'N/A')],
                ['Final masks', metrics.get('size_filtered_masks', np.nan)],
                ['Total filter ratio', metrics.get('total_filter_ratio', np.nan)],
            ])

            if metrics.get('reference_available', False):
                detail_data.extend([
                    ['', ''],
                    ['REFERENCE SEGMENTATION', ''],
                    ['GT masks', metrics.get('reference_gt_masks', 0)],
                    ['Matched masks', metrics.get('reference_matched_masks', 0)],
                    ['False positives', metrics.get('reference_false_positive_masks', 0)],
                    ['False negatives', metrics.get('reference_false_negative_masks', 0)],
                    ['Precision', metrics.get('reference_precision', np.nan)],
                    ['Recall', metrics.get('reference_recall', np.nan)],
                    ['F1 score', metrics.get('reference_f1', np.nan)],
                    ['Mean matched IoU', metrics.get('reference_mean_iou', np.nan)],
                ])

            detail_data.extend([
                ['', ''],
                ['CLIP PERFORMANCE', ''],
                ['Classified particles', metrics.get('clip_total_classified', np.nan)],
                ['Mean confidence', metrics.get('clip_confidence_mean', np.nan)],
                ['Confidence std', metrics.get('clip_confidence_std', np.nan)],
                ['High confidence ratio (>0.9)', metrics.get('clip_high_conf_ratio', np.nan)],
                ['Shape diversity', metrics.get('clip_shape_diversity', np.nan)],
                ['Dominant shape', metrics.get('clip_dominant_shape', 'None')],
                ['Dominant ratio', metrics.get('clip_dominant_ratio', np.nan)],
                ['', ''],
                ['SHAPE DISTRIBUTION', ''],
            ])
            for shape, count in metrics.get('clip_shape_counts', {}).items():
                detail_data.append([shape, count])

            detail_df = pd.DataFrame(detail_data, columns=['Parameter', 'Value'])
            sheet_name = METHOD_LABELS[method_key][:31]
            detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"[OK] Saved: {image_name}_report.xlsx")

    # Save cache for quick reloading
    save_results_cache(
        image_name,
        all_results,
        OUTPUT_DIR,
        include_mask_data=SINGLE_IMAGE_MODE,
        reference_data=reference_data,
    )
    append_processing_log("success", image_name, image_path, "complete", reason="Cache and result files saved")

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n[INFO] All results saved to: {OUTPUT_DIR}")
    print(f"\n[INFO] Files created:")
    print(f"   - {image_name}_<method_key>_preprocessed.png ({len(METHOD_KEYS)} files total)")
    print(f"   - {image_name}{visual_suffix}_CLIP.png")
    print(f"   - {image_name}{visual_suffix}_SAM.png")
    print(f"   - {image_name}{visual_suffix}_SAM_notext.png")
    if SINGLE_IMAGE_MODE:
        print(f"   - {image_name}_<method_key>_SAM_notext.png")
    if reference_data is not None:
        print(f"   - {image_name}{visual_suffix}_reference_error.png")
    print(f"   - {image_name}_metrics.png")
    print(f"   - {image_name}_comparison_table.csv")
    print(f"   - {image_name}_report.xlsx")
    print(f"   - {image_name}_cache.json (for quick reload)")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    complete_method_results = {key: complete_cache[key] for key in METHOD_KEYS}
    best_iou = max(complete_method_results.items(), key=lambda x: x[1]['sam_pred_iou_mean'])
    best_stability = max(complete_method_results.items(), key=lambda x: x[1]['sam_stability_mean'])
    best_particles = max(complete_method_results.items(), key=lambda x: x[1]['size_filtered_masks'])
    best_clip_conf = max(complete_method_results.items(), key=lambda x: x[1]['clip_confidence_mean'])

    print(f"\nBest performance by metric:")
    print(f"  Highest Pred IoU: {best_iou[1]['method_name']} ({best_iou[1]['sam_pred_iou_mean']:.3f})")
    print(f"  Highest Stability: {best_stability[1]['method_name']} ({best_stability[1]['sam_stability_mean']:.3f})")
    print(f"  Most final particles: {best_particles[1]['method_name']} ({best_particles[1]['size_filtered_masks']})")
    print(f"  Highest Mean Confidence: {best_clip_conf[1]['method_name']} ({best_clip_conf[1]['clip_confidence_mean']:.3f})")
    if reference_data is not None:
        best_ref_f1 = max(
            complete_method_results.items(),
            key=lambda x: x[1].get('reference_f1', 0),
        )
        best_ref_eval = best_ref_f1[1]
        print(f"  Best Reference F1: {best_ref_eval['method_name']} "
              f"({best_ref_eval.get('reference_f1', 0):.3f}, "
              f"FP={best_ref_eval.get('reference_false_positive_masks', 0)}, "
              f"FN={best_ref_eval.get('reference_false_negative_masks', 0)})")

    print("\n" + "="*80)

# ============================================================================
# GENERATE AVERAGE PLOTS (text + notext) AFTER ALL IMAGES PROCESSED
# ============================================================================
generate_avg_plots(
    OUTPUT_DIR,
    save_individual_plots=ARGS.save_individual_avg_plots,
)

if skipped_images:
    print("\n" + "="*80)
    print("SKIPPED IMAGES SUMMARY")
    print("="*80)
    for entry in skipped_images:
        print(
            f" - {os.path.basename(entry['image_path'])}: "
            f"{entry['failed_method']} ({entry['error']})"
        )
    print("="*80)

print(f"\n[INFO] Processing log saved to: {PROCESSING_LOG_PATH}")
if skipped_images:
    print(f"[INFO] Skipped-image log saved to: {SKIPPED_IMAGES_LOG_PATH}")

#%%
