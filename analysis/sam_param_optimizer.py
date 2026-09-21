#%%
# -*- coding: utf-8 -*-
"""
SAM Parameter Optimizer V2 - Per-Image Optimization
====================================================
Optimizes SAM parameters individually for each image using:
- 13-stage, 19-pair sequential testing
- DatasetNinja GT with Hungarian IoU matching (TP=valid, FP=noise)
- Lossless compressed post-processed mask cache for all 19 pairs
- Size/shape validation datasets excluded from optimizer fitting
- Growth rate = noise_growth / valid_growth
- Configurable engineering threshold: growth_rate >= 0.5 by default
  * Ensures noise increases at most 50% of valid particle increase rate
  * More stringent than explosion detection (growth_rate > 1.0)
  * Provides robust segmentation with publication-quality defaults
- Per-image optimal parameters
- Statistical aggregation across all images (mode for paper defaults)

Author: Claude
Version: 2.1
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import json
import hashlib
import traceback
from collections import Counter
import seaborn as sns
from torchvision.ops.boxes import batched_nms, box_area

# Import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.amg import MaskData, generate_crop_boxes
from modules.enhancement import check_bm3d_available
from modules.preprocessing import preprocess_with_config
from modules.project_paths import get_optimization_dir
from modules.sam2_utils import (
    SAM_POSTPROCESSING_VERSION,
    filter_background_masks as filter_sam_background_masks,
    filter_overlapping_masks_by_centroid,
)
from modules.parameter_mask_cache import (
    PARAMETER_MASK_CACHE_VERSION,
    load_all_parameter_masks,
    load_parameter_mask_metadata,
    save_parameter_mask_cache,
)
from modules.reference_segmentation import (
    REFERENCE_EVALUATION_VERSION,
    crop_reference_masks,
    find_reference_annotation_path,
    find_reference_image_path,
    hungarian_assignment_scores,
    load_datasetninja_reference_masks,
    metrics_from_assignment_scores,
)


class EmptyMaskSafeSAM2AutomaticMaskGenerator(SAM2AutomaticMaskGenerator):
    """SAM2 AMG with a shape-safe empty-mask path for multi-crop inference.

    sam-2==1.0 creates ``crop_boxes`` as a one-dimensional empty tensor when a
    crop has no accepted masks.  If every crop is empty, its cross-crop NMS
    then indexes that tensor as ``boxes[:, ...]`` and raises ``IndexError``.
    An empty prediction is a valid result at strict thresholds, so preserve it
    as zero masks and skip cross-crop NMS when there is nothing to compare.
    """

    def _process_crop(self, image, crop_box, crop_layer_idx, orig_size):
        data = super()._process_crop(image, crop_box, crop_layer_idx, orig_size)
        if len(data["rles"]) == 0:
            # Keep empty per-crop tensors concatenable with later non-empty
            # crops and with one another.  This changes no non-empty result.
            data["boxes"] = data["boxes"].reshape(0, 4)
            data["points"] = data["points"].reshape(0, 2)
            data["crop_boxes"] = data["crop_boxes"].reshape(0, 4)
        return data

    def _generate_masks(self, image):
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # The upstream sam-2==1.0 implementation enters box_area/NMS even
        # when every crop produced zero masks.  Zero masks need no deduping.
        if len(crop_boxes) > 1 and len(data["rles"]) > 0:
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        data.to_numpy()
        return data


def _parse_runtime_args():
    """Parse runtime arguments without breaking notebook/interactive execution."""
    parser = argparse.ArgumentParser(add_help=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--figure-only",
        action="store_true",
        help="Generate only the 2-panel summary figure from an existing optimization Excel file.",
    )
    mode_group.add_argument(
        "--cache-summary",
        action="store_true",
        help=(
            "Rebuild optimization_results.xlsx and summary figures from saved "
            "per-image cache JSON files without loading SAM or rerunning preprocessing."
        ),
    )
    mode_group.add_argument(
        "--reevaluate-gt-cache",
        action="store_true",
        help=(
            "Recalculate GT metrics and parameter selection from cached Hungarian "
            "assignments or compressed masks without running SAM2."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Cache directory used with --cache-summary. Relative paths are checked "
            "from the current directory and the optimizer output directory."
        ),
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default="optimization_results.xlsx",
        help="Path to optimization_results.xlsx (used with --figure-only).",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=str,
        default=None,
        help="Output directory for summary figure files (default: current optimizer OUTPUT_DIR).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(parent_dir, "Dataset"),
        help="Cropped input image directory (default: Dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optimizer output directory. The default is "
            "sam_optimization_v2/bm3d_only_gt_disjoint."
        ),
    )
    parser.add_argument(
        "--gt-iou-threshold",
        "--tau",
        dest="gt_iou_threshold",
        type=float,
        default=0.5,
        help="Hungarian GT-match IoU threshold tau (default: 0.5).",
    )
    parser.add_argument(
        "--growth-rate-threshold",
        type=float,
        default=0.5,
        help=(
            "Engineering stopping threshold for R(p)=delta_FP/delta_TP. "
            "Use 0.5 for FP growth >= half of TP growth, or 1.0 for "
            "FP growth >= TP growth (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--reference-ann-dir",
        type=str,
        default=os.path.join(parent_dir, "emps-DatasetNinja", "ds", "ann"),
        help="DatasetNinja annotation directory.",
    )
    parser.add_argument(
        "--reference-image-dir",
        type=str,
        default=os.path.join(parent_dir, "emps-DatasetNinja", "ds", "img"),
        help="DatasetNinja full-size source image directory.",
    )
    parser.add_argument(
        "--size-dataset-dir",
        type=str,
        default=os.path.join(parent_dir, "Dataset_size"),
        help="Images excluded because they belong to size validation.",
    )
    parser.add_argument(
        "--shape-dataset-dir",
        type=str,
        default=os.path.join(parent_dir, "Dataset_shape"),
        help="Images excluded because they belong to shape validation.",
    )
    parser.add_argument(
        "--bm3d-sigma",
        type=float,
        default=40.0,
        help="BM3D sigma_psd on the 0-255 scale (default: 40).",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=465,
        help="Required image count for a full optimizer run (default: 465).",
    )
    parser.add_argument(
        "--allow-image-count-mismatch",
        action="store_true",
        help="Allow a full run when --dataset-dir does not contain --expected-images images.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Debug-only limit applied after validation-image exclusion and sorting.",
    )
    parser.add_argument(
        "--single-image",
        type=str,
        default=None,
        help="Debug-only single image path; supersedes --dataset-dir.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore optimizer result JSON cache and recompute selected images.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Audit image counts and BM3D/result cache reuse without loading SAM2.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Prepare/reuse BM3D-only images for every input and exit before SAM2.",
    )
    args, _ = parser.parse_known_args()
    if args.bm3d_sigma <= 0:
        parser.error("--bm3d-sigma must be greater than zero")
    if args.expected_images < 1:
        parser.error("--expected-images must be at least one")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be at least one")
    if not 0.0 <= args.gt_iou_threshold <= 1.0:
        parser.error("--gt-iou-threshold/--tau must be between 0 and 1")
    if args.growth_rate_threshold <= 0:
        parser.error("--growth-rate-threshold must be greater than zero")
    return args


ARGS = _parse_runtime_args()

def get_sam2_config_path(config_name):
    """Get absolute path to SAM2 config from installed package"""
    import sam2, os
    sam2_path = os.path.dirname(sam2.__file__)
    return os.path.join(sam2_path, 'configs', 'sam2.1', config_name)

print("="*80)
print("SAM PARAMETER OPTIMIZER V2 - PER-IMAGE OPTIMIZATION")
print("="*80)

if ARGS.figure_only or ARGS.cache_summary or ARGS.plan_only or ARGS.reevaluate_gt_cache:
    if ARGS.figure_only:
        mode_label = "Figure-only summary generation"
    elif ARGS.cache_summary:
        mode_label = "Cache-only summary rebuild"
    elif ARGS.reevaluate_gt_cache:
        mode_label = "GT cache-only reevaluation"
    else:
        mode_label = "Dataset/cache audit"
    print(f"[MODE] {mode_label}")
else:
    # Verify the only preprocessing component used by this optimizer run.
    if not check_bm3d_available():
        raise ImportError("[FAIL] BM3D not available")
    print("[OK] BM3D available")

#%%
# ============================================================================
# CONFIGURATION
# ============================================================================

# 13-stage, 19-pair parameter schedule (ordered from strict to loose)
PARAM_STAGES = [
    [(0.95, 0.95)],                     # Stage 1: 최상위 (0.95)
    [(0.95, 0.80), (0.80, 0.95)],       # Stage 2
    [(0.80, 0.80)],                     # Stage 3
    [(0.80, 0.65), (0.65, 0.80)],       # Stage 4
    [(0.65, 0.65)],                     # Stage 5
    [(0.65, 0.50), (0.50, 0.65)],       # Stage 6
    [(0.50, 0.50)],                     # Stage 7
    [(0.50, 0.35), (0.35, 0.50)],       # Stage 8
    [(0.35, 0.35)],                     # Stage 9
    [(0.35, 0.20), (0.20, 0.35)],       # Stage 10
    [(0.20, 0.20)],                     # Stage 11
    [(0.20, 0.05), (0.05, 0.20)],       # Stage 12
    [(0.05, 0.05)],                     # Stage 13: 가장 느슨한 기준 (0.05)
]

# GT-based valid/noise classification.
GT_CLASSIFICATION_METHOD = "datasetninja_gt_hungarian"

# Engineering stopping threshold for parameter selection. It is deliberately
# independent of the GT matching IoU threshold and can be changed while reusing
# the threshold-independent mask/assignment caches.
CONSERVATIVE_GROWTH_THRESHOLD = float(ARGS.growth_rate_threshold)

# Image settings
IMAGE_FOLDER = str(Path(ARGS.dataset_dir).expanduser().resolve())
SAMPLE_SIZE = ARGS.sample_size  # None = all images
SINGLE_IMAGE_PATH = ARGS.single_image or ""

# Output. Keep this experiment separate from the former BM3D+Noise2SR run.
_default_output_dir = (
    Path(get_optimization_dir('sam_optimization_v2')) / 'bm3d_only_gt_disjoint'
)
OUTPUT_DIR = str(
    Path(ARGS.output_dir).expanduser().resolve()
    if ARGS.output_dir
    else _default_output_dir.resolve()
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BM3D_SIGMA = float(ARGS.bm3d_sigma)
_bm3d_sigma_tag = f"{BM3D_SIGMA:g}".replace('.', 'p')
PREPROCESSING_MODE = f'bm3d_sigma{_bm3d_sigma_tag}'
GT_IOU_THRESHOLD = float(ARGS.gt_iou_threshold)
_tau_tag = f"{GT_IOU_THRESHOLD:g}".replace('.', 'p')
CACHE_DIR = os.path.join(
    OUTPUT_DIR, f'cache_{PREPROCESSING_MODE}_gt_tau{_tau_tag}'
)
os.makedirs(CACHE_DIR, exist_ok=True)
PREPROCESSED_CACHE_DIR = os.path.join(OUTPUT_DIR, f'preprocessed_{PREPROCESSING_MODE}')
os.makedirs(PREPROCESSED_CACHE_DIR, exist_ok=True)
PARAMETER_MASK_CACHE_DIR = os.path.join(
    OUTPUT_DIR, f'parameter_masks_{PREPROCESSING_MODE}'
)
os.makedirs(PARAMETER_MASK_CACHE_DIR, exist_ok=True)
GT_ASSIGNMENT_CACHE_DIR = os.path.join(
    OUTPUT_DIR, f'gt_assignments_{PREPROCESSING_MODE}'
)
os.makedirs(GT_ASSIGNMENT_CACHE_DIR, exist_ok=True)
COMPARE_PREPROCESS_DIR = Path(get_optimization_dir('preprocessing_comparison_full'))
LEGACY_BM3D_PREPROCESS_DIR = (
    Path(get_optimization_dir('sam_optimization_v2'))
    / 'bm3d_only_465'
    / f'preprocessed_{PREPROCESSING_MODE}'
)
REFERENCE_ANNOTATION_DIR = str(Path(ARGS.reference_ann_dir).expanduser().resolve())
REFERENCE_IMAGE_DIR = str(Path(ARGS.reference_image_dir).expanduser().resolve())
SIZE_DATASET_DIR = str(Path(ARGS.size_dataset_dir).expanduser().resolve())
SHAPE_DATASET_DIR = str(Path(ARGS.shape_dataset_dir).expanduser().resolve())
EXPECTED_IMAGES = int(ARGS.expected_images)
FORCE_REPROCESS = ARGS.force_reprocess or os.environ.get("FORCE_REPROCESS", "0") == "1"
SELECTION_RULE_VERSION = "growth_r_ge_configured_threshold_else_global_min_total_fp_over_tp_v2"

# Visualization palette (requested)
COLOR_RED = "#BF5065"
COLOR_BLUE = "#4B7BA6"
COLOR_GREEN = "#58A65D"
COLOR_ORANGE = "#D96D55"
COLOR_OFFWHITE = "#F2F2F2"

def _blend_with_offwhite(color, mix=0.42, base=COLOR_OFFWHITE):
    """Lighten a color toward off-white for softer pastel heatmaps."""
    fg = np.array(to_rgba(color), dtype=float)
    bg = np.array(to_rgba(base), dtype=float)
    blended = (1.0 - mix) * fg + mix * bg
    blended[3] = 1.0
    return tuple(blended)

PASTEL_BLUE = _blend_with_offwhite(COLOR_BLUE, mix=0.42)
PASTEL_GREEN = _blend_with_offwhite(COLOR_GREEN, mix=0.42)
PASTEL_ORANGE = _blend_with_offwhite(COLOR_ORANGE, mix=0.42)
PASTEL_RED = _blend_with_offwhite(COLOR_RED, mix=0.42)
FREQ_CMAP = LinearSegmentedColormap.from_list(
    "freq_unified",
    [COLOR_OFFWHITE, PASTEL_BLUE, PASTEL_GREEN, PASTEL_ORANGE, PASTEL_RED],
    N=256,
)

print(f"\n[ICON] Configuration:")
print(f"   Stages: {len(PARAM_STAGES)}")
print(f"   Total combinations: {sum(len(stage) for stage in PARAM_STAGES)}")
print(f"   GT matching: Hungarian IoU >= {GT_IOU_THRESHOLD:g}")
print(f"   Growth-rate stopping threshold: R(p) >= {CONSERVATIVE_GROWTH_THRESHOLD:g}")
print(f"   Reference annotations: {REFERENCE_ANNOTATION_DIR}")
print(f"   Validation exclusions: {SIZE_DATASET_DIR}, {SHAPE_DATASET_DIR}")
print(f"   Image folder: {IMAGE_FOLDER}")
if SINGLE_IMAGE_PATH:
    print(f"   Single image override: {SINGLE_IMAGE_PATH}")
print(f"   Output: {OUTPUT_DIR}")
if ARGS.cache_summary:
    print(f"   Summary cache input: {ARGS.cache_dir or CACHE_DIR}")
    print("   Model/preprocessing execution: disabled")
else:
    print(f"   Preprocessing: BM3D only (sigma_psd={BM3D_SIGMA:g})")
    print(f"   Reusable preprocessing cache: {PREPROCESSED_CACHE_DIR}")
    print(f"   Force reprocess: {FORCE_REPROCESS}")

#%%
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_filename(filename):
    """Convert filename to safe version (remove special characters)"""
    name, ext = os.path.splitext(filename)
    # Replace special characters with underscore
    safe_name = name.replace('/', '_').replace('\\', '_').replace('%', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    return safe_name + ext


def calculate_growth_rate(valid_growth, noise_growth):
    """
    Calculate growth rate with robust edge-case handling.

    Rules:
    - valid_growth > 0: growth_rate = noise_growth / valid_growth
    - valid_growth <= 0 and noise_growth > 0: explosion (inf)
    - valid_growth <= 0 and noise_growth <= 0: no growth (0.0)
    """
    if valid_growth > 0:
        return noise_growth / valid_growth
    if noise_growth > 0:
        return float('inf')
    return 0.0


def cap_growth_rate_for_plot(growth_rate, cap_value=1.0):
    """
    Cap growth rate for visualization.

    Returns:
        plot_value: value to plot on z-axis
        is_capped: True when original value is >= cap_value (or inf)
    """
    if not np.isfinite(growth_rate):
        return cap_value, True
    if growth_rate >= cap_value:
        return cap_value, True
    if growth_rate <= 0:
        return 0.0, False
    return growth_rate, False


def build_threshold_barrier_facecolors(xx, yy, alpha=0.28):
    """
    Build colorful stripe facecolors for threshold surface (barrier 느낌).
    """
    barrier_palette = [COLOR_RED, COLOR_ORANGE, COLOR_GREEN, COLOR_BLUE]
    stripe_index = (np.floor((xx + yy) * 12.0).astype(int)) % len(barrier_palette)
    facecolors = np.empty(xx.shape + (4,), dtype=float)

    for idx, hex_color in enumerate(barrier_palette):
        facecolors[stripe_index == idx] = to_rgba(hex_color, alpha=alpha)

    return facecolors


def save_notext_figure(fig, path, dpi=200, remove_tick_labels=False):
    """Remove textual elements and optionally hide tick labels (keep tick bars)."""
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text('')

    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title('')
        if hasattr(ax, 'set_xlabel'):
            ax.set_xlabel('')
        if hasattr(ax, 'set_ylabel'):
            ax.set_ylabel('')
        if hasattr(ax, 'set_zlabel'):
            ax.set_zlabel('')
        if remove_tick_labels:
            # Keep tick marks/bars but remove numeric labels.
            if hasattr(ax, 'set_xticklabels'):
                ax.set_xticklabels([])
            if hasattr(ax, 'set_yticklabels'):
                ax.set_yticklabels([])
            if hasattr(ax, 'set_zticklabels'):
                ax.set_zticklabels([])
        for txt in ax.texts[:]:
            txt.remove()

    fig.savefig(path, dpi=dpi, bbox_inches='tight')


def _coerce_exception_mask(series):
    """Coerce Exception column values to boolean mask (True means exception)."""
    if series is None:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    lower = series.astype(str).str.strip().str.lower()
    true_tokens = {"true", "1", "yes", "y", "t"}
    return lower.isin(true_tokens)


def _find_column(df, candidates):
    """Find first matching column by normalized name."""
    norm_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        norm_map[key] = col
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in norm_map:
            return norm_map[key]
    return None


def _resolve_excel_path(excel_path, output_dir):
    """Resolve Excel path with fallback to OUTPUT_DIR and optional .xlsx suffix."""
    raw = Path(excel_path)
    candidates = []

    def _add_candidate(path_obj):
        if path_obj is None:
            return
        p = Path(path_obj)
        candidates.append(p)
        if p.suffix.lower() != ".xlsx":
            candidates.append(Path(f"{p}.xlsx"))

    _add_candidate(raw)
    if not raw.is_absolute():
        _add_candidate(Path.cwd() / raw)
        _add_candidate(Path(output_dir) / raw)

    seen = set()
    ordered_unique = []
    for cand in candidates:
        key = str(cand.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        ordered_unique.append(cand)

    for cand in ordered_unique:
        if cand.exists() and cand.is_file():
            return cand

    searched = "\n".join([f" - {str(c)}" for c in ordered_unique])
    raise FileNotFoundError(f"Excel file not found. Searched:\n{searched}")


def generate_sam_opt_ab_figure(excel_path, output_dir, dpi=600):
    """
    Generate publication-quality 2-panel summary figure:
      (a) Optimal parameter frequency heatmap ratio
      (b) 3D growth-rate plot with the recorded stopping-threshold plane
    Also saves a no-text version.
    """
    level_candidates = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excel_path = Path(excel_path)
    sheet_map = pd.read_excel(excel_path, sheet_name=None)
    if not sheet_map:
        raise ValueError(f"No sheets found in Excel: {excel_path}")

    growth_rate_threshold = float(CONSERVATIVE_GROWTH_THRESHOLD)
    metadata_sheet = sheet_map.get("Metadata")
    if metadata_sheet is not None:
        item_col = _find_column(metadata_sheet, ["Item"])
        value_col = _find_column(metadata_sheet, ["Value"])
        if item_col is not None and value_col is not None:
            threshold_rows = metadata_sheet.loc[
                metadata_sheet[item_col].astype(str).str.strip()
                == "Growth_Rate_Threshold",
                value_col,
            ]
            if not threshold_rows.empty:
                recorded_threshold = pd.to_numeric(
                    threshold_rows.iloc[0], errors="coerce"
                )
                if np.isfinite(recorded_threshold):
                    growth_rate_threshold = float(recorded_threshold)

    df_all = sheet_map.get("All Images")
    if df_all is None:
        first_sheet_name = next(iter(sheet_map.keys()))
        df_all = sheet_map[first_sheet_name]

    pred_col = _find_column(df_all, ["pred_iou_thresh"])
    stab_col = _find_column(df_all, ["stability_score_thresh"])
    exc_col = _find_column(df_all, ["Exception"])
    if pred_col is None or stab_col is None:
        raise KeyError("Required columns pred_iou_thresh and stability_score_thresh were not found in Excel.")

    total_images = int(len(df_all))
    exception_mask = _coerce_exception_mask(df_all[exc_col]) if exc_col is not None else pd.Series(False, index=df_all.index)
    valid_df = df_all.loc[~exception_mask].copy()
    valid_images = int(len(valid_df))

    valid_df[pred_col] = pd.to_numeric(valid_df[pred_col], errors="coerce")
    valid_df[stab_col] = pd.to_numeric(valid_df[stab_col], errors="coerce")
    valid_df = valid_df.dropna(subset=[pred_col, stab_col]).copy()
    if valid_df.empty:
        raise ValueError("No valid rows with pred_iou_thresh/stability_score_thresh after filtering Exception=False.")

    valid_df["pred_key"] = valid_df[pred_col].round(2)
    valid_df["stab_key"] = valid_df[stab_col].round(2)

    pair_counts = Counter(zip(valid_df["pred_key"], valid_df["stab_key"]))
    default_frequency_sheet = sheet_map.get("Default Pair Frequency")
    mode_pair = None
    if default_frequency_sheet is not None:
        default_pred_col = _find_column(
            default_frequency_sheet, ["pred_iou_thresh"]
        )
        default_stab_col = _find_column(
            default_frequency_sheet, ["stability_score_thresh"]
        )
        default_flag_col = _find_column(
            default_frequency_sheet, ["Is_Default_Nonterminal_Pair"]
        )
        if None not in (default_pred_col, default_stab_col, default_flag_col):
            default_flags = (
                default_frequency_sheet[default_flag_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes"})
            )
            default_rows = default_frequency_sheet.loc[default_flags]
            if len(default_rows) == 1:
                default_pred = pd.to_numeric(
                    default_rows.iloc[0][default_pred_col], errors="coerce"
                )
                default_stab = pd.to_numeric(
                    default_rows.iloc[0][default_stab_col], errors="coerce"
                )
                if np.isfinite(default_pred) and np.isfinite(default_stab):
                    mode_pair = (
                        round(float(default_pred), 2),
                        round(float(default_stab), 2),
                    )
    if mode_pair is None:
        mode_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
    mode_count = int(pair_counts.get(mode_pair, 0))

    present_pred = {float(v) for v in valid_df["pred_key"].unique()}
    present_stab = {float(v) for v in valid_df["stab_key"].unique()}
    pred_levels = [lv for lv in level_candidates if lv in present_pred] or sorted(present_pred)
    stab_levels = [lv for lv in level_candidates if lv in present_stab] or sorted(present_stab)

    pred_to_idx = {v: i for i, v in enumerate(pred_levels)}
    stab_to_idx = {v: i for i, v in enumerate(stab_levels)}
    heatmap_ratio = np.zeros((len(stab_levels), len(pred_levels)), dtype=float)
    denom = max(total_images, 1)
    for (pred_val, stab_val), count in pair_counts.items():
        if pred_val not in pred_to_idx or stab_val not in stab_to_idx:
            continue
        yi = stab_to_idx[stab_val]
        xi = pred_to_idx[pred_val]
        heatmap_ratio[yi, xi] = count / denom

    # Growth-rate values must come from the optimizer output. Never synthesize
    # publication data when the source workbook is incomplete.
    growth_map = {}
    growth_found = False
    for _, df_sheet in sheet_map.items():
        p_col = _find_column(df_sheet, ["pred_iou_thresh"])
        s_col = _find_column(df_sheet, ["stability_score_thresh"])
        g_col = _find_column(df_sheet, ["growth_rate", "growth_rate_raw", "growthrate"])
        if p_col is None or s_col is None or g_col is None:
            continue

        temp = df_sheet.copy()
        e_col = _find_column(temp, ["Exception"])
        if e_col is not None:
            temp = temp.loc[~_coerce_exception_mask(temp[e_col])].copy()

        temp[p_col] = pd.to_numeric(temp[p_col], errors="coerce")
        temp[s_col] = pd.to_numeric(temp[s_col], errors="coerce")
        temp[g_col] = pd.to_numeric(temp[g_col], errors="coerce")
        temp = temp.dropna(subset=[p_col, s_col, g_col]).copy()
        if temp.empty:
            continue

        temp["pred_key"] = temp[p_col].round(2)
        temp["stab_key"] = temp[s_col].round(2)
        grouped = temp.groupby(["pred_key", "stab_key"])[g_col].mean()
        for (pred_key, stab_key), gr in grouped.items():
            growth_map[(float(pred_key), float(stab_key))] = float(gr)
        growth_found = True
        break

    if not growth_found:
        raise ValueError(
            "No recorded growth-rate rows were found in the optimizer workbook"
        )

    growth_pred_levels = {k[0] for k in growth_map.keys()}
    growth_stab_levels = {k[1] for k in growth_map.keys()}
    pred_levels_3d_set = present_pred | growth_pred_levels
    stab_levels_3d_set = present_stab | growth_stab_levels
    pred_levels_3d = [lv for lv in level_candidates if lv in pred_levels_3d_set] or sorted(pred_levels_3d_set)
    stab_levels_3d = [lv for lv in level_candidates if lv in stab_levels_3d_set] or sorted(stab_levels_3d_set)

    tested_pairs = {(float(p), float(s)) for (p, s) in growth_map.keys() if p in pred_levels_3d and s in stab_levels_3d}
    all_grid_pairs = {(float(p), float(s)) for p in pred_levels_3d for s in stab_levels_3d}
    not_tested_pairs = sorted(all_grid_pairs - tested_pairs)

    def _draw_one(with_text):
        rc = {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
        with plt.rc_context(rc):
            sns.set_style("white")
            fig = plt.figure(figsize=(14, 5))
            gs = fig.add_gridspec(1, 2, wspace=0.20, width_ratios=[1.0, 1.15])

            # Panel (a): heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            vmax = float(heatmap_ratio.max()) if heatmap_ratio.size > 0 else 1.0
            if vmax <= 0:
                vmax = 1.0
            im = ax1.imshow(heatmap_ratio, cmap=FREQ_CMAP, origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
            ax1.set_xticks(np.arange(len(pred_levels)))
            ax1.set_xticklabels([f"{v:.2f}" for v in pred_levels], fontsize=8)
            ax1.set_yticks(np.arange(len(stab_levels)))
            ax1.set_yticklabels([f"{v:.2f}" for v in stab_levels], fontsize=8)

            mode_x = pred_to_idx.get(mode_pair[0], 0)
            mode_y = stab_to_idx.get(mode_pair[1], 0)
            ax1.add_patch(
                Rectangle(
                    (mode_x - 0.5, mode_y - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.6,
                    zorder=4,
                )
            )
            ax1.scatter([mode_x], [mode_y], marker="*", s=190, c="yellow", edgecolors="black", linewidths=1.0, zorder=5)

            cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            if with_text:
                ax1.set_title(f"Optimal Parameter Frequency Ratio (n={total_images} images)")
                ax1.set_xlabel("pred_iou_thresh")
                ax1.set_ylabel("stability_score_thresh")
                cbar.set_label("Frequency ratio")
                cbar.ax.tick_params(labelsize=8)
                for yi in range(len(stab_levels)):
                    for xi in range(len(pred_levels)):
                        value = heatmap_ratio[yi, xi]
                        if value <= 0:
                            continue
                        txt_color = "white" if value >= (vmax * 0.45) else "black"
                        ax1.text(xi, yi, f"{value:.3f}", ha="center", va="center", color=txt_color, fontsize=8, fontweight="bold")
                ax1.text(0.01, 0.99, "(a)", transform=ax1.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
            else:
                ax1.set_title("")
                ax1.set_xlabel("")
                ax1.set_ylabel("")
                cbar.set_label("")
            ax1.grid(False)

            # Panel (b): 3D growth-rate plot
            ax2 = fig.add_subplot(gs[0, 1], projection="3d")
            x_min, x_max = min(pred_levels_3d), max(pred_levels_3d)
            y_min, y_max = min(stab_levels_3d), max(stab_levels_3d)
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 25), np.linspace(y_min, y_max, 25))
            z_cap = max(1.0, growth_rate_threshold)
            zz = np.full_like(xx, growth_rate_threshold)
            ax2.plot_surface(xx, yy, zz, color=COLOR_BLUE, alpha=0.15, linewidth=0, antialiased=True)

            acceptable, aggressive, capped = [], [], []
            for pair in sorted(tested_pairs):
                gr_raw = growth_map[pair]
                if not np.isfinite(gr_raw) or gr_raw >= z_cap:
                    capped.append((pair[0], pair[1], z_cap))
                elif gr_raw >= growth_rate_threshold:
                    aggressive.append((pair[0], pair[1], max(0.0, float(gr_raw))))
                else:
                    acceptable.append((pair[0], pair[1], max(0.0, float(gr_raw))))

            if acceptable:
                ax2.scatter(
                    [p[0] for p in acceptable], [p[1] for p in acceptable], [p[2] for p in acceptable],
                    c=COLOR_GREEN, s=56, marker="o", edgecolors="black", linewidths=0.6, depthshade=False
                )
            if aggressive:
                ax2.scatter(
                    [p[0] for p in aggressive], [p[1] for p in aggressive], [p[2] for p in aggressive],
                    c=COLOR_ORANGE, s=56, marker="o", edgecolors="black", linewidths=0.6, depthshade=False
                )
            if capped:
                ax2.scatter(
                    [p[0] for p in capped], [p[1] for p in capped], [p[2] for p in capped],
                    c=COLOR_RED, s=56, marker="o", edgecolors="black", linewidths=0.6, depthshade=False
                )

            if not_tested_pairs:
                ax2.scatter(
                    [p[0] for p in not_tested_pairs], [p[1] for p in not_tested_pairs], [0.0] * len(not_tested_pairs),
                    c="none", s=42, marker="s", edgecolors="gray", linewidths=0.9, depthshade=False
                )

            opt_raw = growth_map.get(mode_pair, 0.0)
            opt_z = z_cap if (not np.isfinite(opt_raw) or opt_raw >= z_cap) else max(0.0, float(opt_raw))
            ax2.scatter(
                [mode_pair[0]], [mode_pair[1]], [opt_z],
                c="yellow", s=220, marker="*", edgecolors="black", linewidths=1.2, depthshade=False
            )

            ax2.set_xlim(x_min - 0.02, x_max + 0.02)
            ax2.set_ylim(y_min - 0.02, y_max + 0.02)
            ax2.set_zlim(0.0, z_cap * 1.05)
            ax2.view_init(elev=20, azim=240)
            ax2.grid(False)
            ax2.set_xticks(pred_levels_3d)
            ax2.set_yticks(stab_levels_3d)
            ax2.set_zticks(sorted({0.0, growth_rate_threshold, z_cap}))
            ax2.tick_params(axis="x", labelsize=10)
            ax2.tick_params(axis="y", labelsize=10)

            if with_text:
                ax2.set_title(f"Growth-Rate Criterion (valid={valid_images}, total={total_images})")
                ax2.set_xlabel("pred_iou_thresh")
                ax2.set_ylabel("stability_score_thresh")
                ax2.set_zlabel("growth_rate")
                ax2.text2D(0.02, 0.99, "(b)", transform=ax2.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
                ax2.text2D(
                    0.02,
                    0.92,
                    f"Growth rate threshold ({growth_rate_threshold:g})",
                    transform=ax2.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )

                legend_handles = [
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_GREEN, markeredgecolor="black",
                           markeredgewidth=0.6, markersize=7, label=f"Below threshold (<{growth_rate_threshold:g})"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_ORANGE, markeredgecolor="black",
                           markeredgewidth=0.6, markersize=7, label=f"Threshold reached ({growth_rate_threshold:g}<=g<{z_cap:g})"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_RED, markeredgecolor="black",
                           markeredgewidth=0.6, markersize=7, label="Capped points"),
                    Line2D([0], [0], marker="*", color="w", markerfacecolor="yellow", markeredgecolor="black",
                           markeredgewidth=1.0, markersize=11, label="Optimal (mode pair)"),
                    Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor="gray",
                           markeredgewidth=0.9, markersize=7, label="Not tested"),
                    Patch(facecolor=COLOR_BLUE, alpha=0.15, edgecolor="none", label=f"Growth rate threshold ({growth_rate_threshold:g})"),
                ]
                ax2.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.95)
            else:
                ax2.set_title("")
                ax2.set_xlabel("")
                ax2.set_ylabel("")
                ax2.set_zlabel("")
                legend = ax2.get_legend()
                if legend is not None:
                    legend.remove()

            fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.10, wspace=0.22)
            return fig

    fig_main = _draw_one(with_text=True)
    fig_main.savefig(output_dir / "sam_opt_ab.pdf", bbox_inches="tight")
    fig_main.savefig(output_dir / "sam_opt_ab.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig_main)

    fig_notext = _draw_one(with_text=False)
    fig_notext.savefig(output_dir / "sam_opt_ab_notext.pdf", bbox_inches="tight")
    fig_notext.savefig(output_dir / "sam_opt_ab_notext.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig_notext)

    return {
        "total_images": total_images,
        "valid_images": valid_images,
        "mode_pair": mode_pair,
        "mode_count": mode_count,
        "growth_from_excel": growth_found,
    }


def generate_optimal_param_frequency_from_excel(excel_path, output_dir, dpi=300):
    """
    Generate optimal parameter frequency heatmap files from existing optimization Excel.
    Saves:
      - optimal_param_frequency.png
      - optimal_param_frequency_notext.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excel_path = Path(excel_path)
    sheet_map = pd.read_excel(excel_path, sheet_name=None)
    if not sheet_map:
        raise ValueError(f"No sheets found in Excel: {excel_path}")

    df_all = sheet_map.get("All Images")
    if df_all is None:
        first_sheet_name = next(iter(sheet_map.keys()))
        df_all = sheet_map[first_sheet_name]

    pred_col = _find_column(df_all, ["pred_iou_thresh"])
    stab_col = _find_column(df_all, ["stability_score_thresh"])
    exc_col = _find_column(df_all, ["Exception"])
    if pred_col is None or stab_col is None:
        raise KeyError("Required columns pred_iou_thresh and stability_score_thresh were not found in Excel.")

    total_images = int(len(df_all))
    exception_mask = _coerce_exception_mask(df_all[exc_col]) if exc_col is not None else pd.Series(False, index=df_all.index)
    valid_df = df_all.loc[~exception_mask].copy()
    valid_df[pred_col] = pd.to_numeric(valid_df[pred_col], errors="coerce")
    valid_df[stab_col] = pd.to_numeric(valid_df[stab_col], errors="coerce")
    valid_df = valid_df.dropna(subset=[pred_col, stab_col]).copy()
    if valid_df.empty:
        raise ValueError("No valid rows with pred_iou_thresh/stability_score_thresh after filtering Exception=False.")

    valid_df["pred_key"] = valid_df[pred_col].round(2)
    valid_df["stab_key"] = valid_df[stab_col].round(2)

    pair_counts = Counter(zip(valid_df["pred_key"], valid_df["stab_key"]))
    pred_iou_levels = sorted({p[0] for p in pair_counts.keys()})
    stability_levels = sorted({p[1] for p in pair_counts.keys()})
    pred_iou_to_idx = {v: i for i, v in enumerate(pred_iou_levels)}
    stability_to_idx = {v: i for i, v in enumerate(stability_levels)}

    heatmap_ratio = np.zeros((len(stability_levels), len(pred_iou_levels)), dtype=float)
    denom = max(total_images, 1)
    for (pred_iou, stability), count in pair_counts.items():
        yi = stability_to_idx[stability]
        xi = pred_iou_to_idx[pred_iou]
        heatmap_ratio[yi, xi] = count / denom

    vmax = float(heatmap_ratio.max()) if heatmap_ratio.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0

    # Heatmap (with text)
    fig, ax = plt.subplots(figsize=(max(7, len(pred_iou_levels) * 0.9), max(6, len(stability_levels) * 0.75)))
    im = ax.imshow(heatmap_ratio, cmap=FREQ_CMAP, origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xlabel("pred_iou_thresh", fontsize=12, fontweight="bold")
    ax.set_ylabel("stability_score_thresh", fontsize=12, fontweight="bold")
    ax.set_title(f"Optimal Parameter Frequency Ratio (n={total_images} images)", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(len(pred_iou_levels)))
    ax.set_xticklabels([f"{v:.2f}" for v in pred_iou_levels], fontsize=10)
    ax.set_yticks(np.arange(len(stability_levels)))
    ax.set_yticklabels([f"{v:.2f}" for v in stability_levels], fontsize=10)

    text_threshold = vmax * 0.5
    for yi in range(len(stability_levels)):
        for xi in range(len(pred_iou_levels)):
            value = heatmap_ratio[yi, xi]
            if value <= 0:
                continue
            txt_color = "white" if value >= text_threshold else "black"
            ax.text(xi, yi, f"{value:.3f}", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Frequency / Total Images", fontsize=11)
    plt.tight_layout()

    freq_path = output_dir / "optimal_param_frequency.png"
    fig.savefig(freq_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Heatmap (no-text version)
    fig_nt, ax_nt = plt.subplots(figsize=(max(7, len(pred_iou_levels) * 0.9), max(6, len(stability_levels) * 0.75)))
    ax_nt.imshow(heatmap_ratio, cmap=FREQ_CMAP, origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
    ax_nt.set_xticks(np.arange(len(pred_iou_levels)))
    ax_nt.set_xticklabels([f"{v:.2f}" for v in pred_iou_levels], fontsize=10)
    ax_nt.set_yticks(np.arange(len(stability_levels)))
    ax_nt.set_yticklabels([f"{v:.2f}" for v in stability_levels], fontsize=10)
    ax_nt.set_xlabel("")
    ax_nt.set_ylabel("")
    ax_nt.set_title("")
    for spine in ax_nt.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    freq_notext_path = output_dir / "optimal_param_frequency_notext.png"
    fig_nt.savefig(freq_notext_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_nt)

    return {
        "total_images": total_images,
        "valid_images": int(len(valid_df)),
        "freq_path": freq_path,
        "freq_notext_path": freq_notext_path,
    }


def _resolve_cache_directory(cache_dir, output_dir):
    """Resolve a cache directory from an absolute path or common relative roots."""
    raw = Path(cache_dir).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([Path.cwd() / raw, Path(output_dir) / raw])

    seen = set()
    searched = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        searched.append(resolved)
        if resolved.is_dir():
            return resolved

    locations = "\n".join(f" - {path}" for path in searched)
    raise FileNotFoundError(f"Cache directory not found. Searched:\n{locations}")


def _build_cache_summary_tables(cache_dir):
    """Load cache JSON files and reproduce the optimizer's summary tables."""
    cache_dir = Path(cache_dir).resolve()
    cache_paths = sorted(cache_dir.glob("*_cache.json"), key=lambda path: path.name.lower())
    if not cache_paths:
        raise ValueError(f"No *_cache.json files found in {cache_dir}")

    all_results = []
    image_names = set()
    for cache_path in cache_paths:
        result = load_image_cache(cache_path)
        if not isinstance(result, dict):
            raise ValueError(f"Cache payload is not an object: {cache_path}")
        image_name = str(result.get("image_name") or "").strip()
        if not image_name:
            raise ValueError(f"Cache is missing image_name: {cache_path}")
        image_key = image_name.lower()
        if image_key in image_names:
            raise ValueError(f"Duplicate image_name in cache directory: {image_name}")
        image_names.add(image_key)
        all_results.append(result)

    normal_results = [result for result in all_results if not result.get("exception", False)]
    exception_results = [result for result in all_results if result.get("exception", False)]
    if not normal_results:
        raise ValueError(f"No normal optimization results found in {cache_dir}")

    split_manifest_path = cache_dir.parent / "optimization_split_manifest.json"
    split_manifest = {}
    if split_manifest_path.is_file():
        try:
            split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid optimization split manifest: {split_manifest_path}") from exc
    expected_optimization_count = split_manifest.get("optimization_count")
    expected_optimization_ids = {
        str(row.get("Image_ID")).strip().lower()
        for row in split_manifest.get("rows", [])
        if bool(row.get("Optimization_Included"))
    }
    expected_optimization_ids.discard("")
    expected_optimization_ids.discard("none")
    actual_optimization_ids = {
        Path(str(result["image_name"])).stem.strip().lower()
        for result in all_results
    }
    run_complete = (
        expected_optimization_count is not None
        and len(all_results) == int(expected_optimization_count)
        and bool(expected_optimization_ids)
        and actual_optimization_ids == expected_optimization_ids
    )

    required_normal_keys = {
        "optimal_stage",
        "optimal_pred_iou",
        "optimal_stability",
        "optimal_valid",
        "optimal_noise",
        "optimal_ratio",
    }
    for result in normal_results:
        missing = sorted(required_normal_keys - set(result))
        if missing:
            raise ValueError(
                f"Normal cache {result['image_name']} is missing fields: {', '.join(missing)}"
            )

    all_columns = [
        "Image",
        "Exception",
        "Optimal_Stage",
        "pred_iou_thresh",
        "stability_score_thresh",
        "Valid_Masks",
        "Noise_Masks",
        "Noise_Ratio",
        "GT_Masks",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1",
        "Mean_Matched_IoU",
        "GT_IoU_Threshold",
        "FP_per_TP",
        "Cache_Source",
        "Reason",
    ]
    all_rows = [
        {
            "Image": str(result["image_name"]),
            "Exception": False,
            "Optimal_Stage": int(result["optimal_stage"]) + 1,
            "pred_iou_thresh": float(result["optimal_pred_iou"]),
            "stability_score_thresh": float(result["optimal_stability"]),
            "Valid_Masks": int(result["optimal_valid"]),
            "Noise_Masks": int(result["optimal_noise"]),
            "Noise_Ratio": float(result["optimal_ratio"]),
            "GT_Masks": result.get("optimal_gt_masks"),
            "TP": result.get("optimal_tp", result.get("optimal_valid")),
            "FP": result.get("optimal_fp", result.get("optimal_noise")),
            "FN": result.get("optimal_fn"),
            "Precision": result.get("optimal_precision"),
            "Recall": result.get("optimal_recall"),
            "F1": result.get("optimal_f1"),
            "Mean_Matched_IoU": result.get("optimal_mean_matched_iou"),
            "GT_IoU_Threshold": result.get("gt_iou_threshold"),
            "FP_per_TP": result.get("optimal_noise_valid_ratio"),
            "Cache_Source": result.get("cache_source"),
            "Reason": str(result.get("reason", "")),
        }
        for result in normal_results
    ]
    all_rows.extend(
        {
            "Image": str(result["image_name"]),
            "Exception": True,
            "Optimal_Stage": None,
            "pred_iou_thresh": None,
            "stability_score_thresh": None,
            "Valid_Masks": None,
            "Noise_Masks": None,
            "Noise_Ratio": None,
            "GT_Masks": None,
            "TP": None,
            "FP": None,
            "FN": None,
            "Precision": None,
            "Recall": None,
            "F1": None,
            "Mean_Matched_IoU": None,
            "GT_IoU_Threshold": None,
            "FP_per_TP": None,
            "Cache_Source": result.get("cache_source"),
            "Reason": str(result.get("reason", "")),
        }
        for result in exception_results
    )
    df_all_results = pd.DataFrame(all_rows, columns=all_columns)

    normal_columns = [column for column in all_columns if column != "Exception"]
    normal_rows = [
        {column: row[column] for column in normal_columns}
        for row in all_rows
        if not row["Exception"]
    ]
    df_normal_results = pd.DataFrame(normal_rows, columns=normal_columns)

    pred_iou_values = df_normal_results["pred_iou_thresh"].to_numpy(dtype=float)
    stability_values = df_normal_results["stability_score_thresh"].to_numpy(dtype=float)
    df_stats = pd.DataFrame(
        {
            "Metric": ["pred_iou_thresh", "stability_score_thresh"],
            "Mode": [
                pd.Series(pred_iou_values).mode().iloc[0],
                pd.Series(stability_values).mode().iloc[0],
            ],
            "Mean": [pred_iou_values.mean(), stability_values.mean()],
            "Median": [np.median(pred_iou_values), np.median(stability_values)],
            "Std": [pred_iou_values.std(), stability_values.std()],
            "Min": [pred_iou_values.min(), stability_values.min()],
            "Max": [pred_iou_values.max(), stability_values.max()],
        }
    )

    pair_rows = []
    growth_rows = []
    for result in normal_results:
        stage_results = result.get("stage_results", [])
        if not isinstance(stage_results, list):
            raise ValueError(f"stage_results is not a list for {result['image_name']}")
        for stage_index in range(1, len(stage_results)):
            previous_stage = stage_results[stage_index - 1]
            current_stage = stage_results[stage_index]
            if not previous_stage or not current_stage:
                continue
            previous_best = min(previous_stage, key=_combo_quality_key)
            for combo_index, combo in enumerate(current_stage):
                valid_growth = int(combo["valid_masks"]) - int(previous_best["valid_masks"])
                noise_growth = int(combo["noise_masks"]) - int(previous_best["noise_masks"])
                growth_rate = calculate_growth_rate(valid_growth, noise_growth)
                growth_rate_plot, growth_rate_capped = cap_growth_rate_for_plot(growth_rate)
                growth_rows.append(
                    {
                        "Image": str(result["image_name"]),
                        "Stage": int(combo.get("stage", stage_index)) + 1,
                        "Combo": int(combo.get("combo_idx", combo_index)),
                        "pred_iou_thresh": float(combo["pred_iou"]),
                        "stability_score_thresh": float(combo["stability"]),
                        "Previous_Valid_Masks": int(previous_best["valid_masks"]),
                        "Previous_Noise_Masks": int(previous_best["noise_masks"]),
                        "Valid_Masks": int(combo["valid_masks"]),
                        "Noise_Masks": int(combo["noise_masks"]),
                        "Valid_Growth": valid_growth,
                        "Noise_Growth": noise_growth,
                        "Growth_Rate": growth_rate,
                        "Growth_Rate_Plot": growth_rate_plot,
                        "Growth_Rate_Capped": bool(growth_rate_capped),
                    }
                )
        for stage_index, stage in enumerate(stage_results):
            for combo_index, combo in enumerate(stage):
                pair_rows.append(
                    {
                        "Image": str(result["image_name"]),
                        "Stage": int(combo.get("stage", stage_index)) + 1,
                        "Combo": int(combo.get("combo_idx", combo_index)),
                        "pred_iou_thresh": float(combo["pred_iou"]),
                        "stability_score_thresh": float(combo["stability"]),
                        "GT_Masks": combo.get("gt_masks"),
                        "Predicted_Masks": int(combo["total_masks"]),
                        "TP": int(combo["valid_masks"]),
                        "FP": int(combo["noise_masks"]),
                        "FN": combo.get("false_negative_masks"),
                        "Precision": combo.get("precision"),
                        "Recall": combo.get("recall"),
                        "F1": combo.get("f1"),
                        "Mean_Matched_IoU": combo.get("mean_matched_iou"),
                        "GT_IoU_Threshold": result.get("gt_iou_threshold"),
                        "FP_per_TP": combo.get("noise_valid_ratio"),
                        "Selected": (
                            stage_index == int(result["optimal_stage"])
                            and combo_index == int(result.get("optimal_combo", 0))
                        ),
                    }
                )
    if not growth_rows:
        raise ValueError(
            f"No stage-transition growth rates are available in cache directory {cache_dir}"
        )
    df_growth = pd.DataFrame(growth_rows)
    df_pair_metrics = pd.DataFrame(pair_rows)

    schedule_order = {
        (round(float(pred_iou), 12), round(float(stability), 12)): {
            "stage": stage_index + 1,
            "combo": combo_index,
            "schedule_index": schedule_index,
            "is_terminal": stage_index == len(PARAM_STAGES) - 1,
        }
        for schedule_index, (stage_index, combo_index, pred_iou, stability) in enumerate(
            (
                (stage_index, combo_index, pred_iou, stability)
                for stage_index, stage in enumerate(PARAM_STAGES)
                for combo_index, (pred_iou, stability) in enumerate(stage)
            )
        )
    }
    selected_pair_counts = Counter(
        (
            round(float(result["optimal_pred_iou"]), 12),
            round(float(result["optimal_stability"]), 12),
        )
        for result in normal_results
    )
    unknown_selected_pairs = set(selected_pair_counts) - set(schedule_order)
    if unknown_selected_pairs:
        raise ValueError(
            f"Selected parameter pairs are outside the 19-pair schedule: "
            f"{sorted(unknown_selected_pairs)}"
        )
    default_candidates = [
        pair
        for pair in selected_pair_counts
        if pair in schedule_order and not schedule_order[pair]["is_terminal"]
    ]
    if not default_candidates:
        raise ValueError("No selected non-terminal parameter pair is available")
    default_pair = min(
        default_candidates,
        key=lambda pair: (
            -selected_pair_counts[pair],
            schedule_order[pair]["schedule_index"],
        ),
    )
    default_count = int(selected_pair_counts[default_pair])
    default_stage = int(schedule_order[default_pair]["stage"])
    terminal_count = int(
        sum(
            count
            for pair, count in selected_pair_counts.items()
            if schedule_order.get(pair, {}).get("is_terminal", False)
        )
    )
    neighborhood_counts = {
        radius: int(
            sum(
                count
                for pair, count in selected_pair_counts.items()
                if pair in schedule_order
                and abs(int(schedule_order[pair]["stage"]) - default_stage)
                <= radius
            )
        )
        for radius in (1, 2)
    }
    no_crossing_fallback_count = int(
        sum(
            str(result.get("reason", "")).startswith(
                "No stage transition had R(p)"
            )
            for result in normal_results
        )
    )
    frequency_rows = []
    for pair, schedule_info in sorted(
        schedule_order.items(), key=lambda item: item[1]["schedule_index"]
    ):
        count = int(selected_pair_counts.get(pair, 0))
        frequency_rows.append(
            {
                "pred_iou_thresh": pair[0],
                "stability_score_thresh": pair[1],
                "Stage": int(schedule_info["stage"]),
                "Combo": int(schedule_info["combo"]),
                "Is_Terminal": bool(schedule_info["is_terminal"]),
                "Selected_Count": count,
                "Percent_of_All_Images": count / len(all_results) * 100.0,
                "Percent_of_Normal_Images": count / len(normal_results) * 100.0,
                "Is_Default_Nonterminal_Pair": pair == default_pair,
                "Within_One_Stage_Of_Default": (
                    abs(int(schedule_info["stage"]) - default_stage) <= 1
                ),
                "Within_Two_Stages_Of_Default": (
                    abs(int(schedule_info["stage"]) - default_stage) <= 2
                ),
            }
        )
    df_default_frequency = pd.DataFrame(frequency_rows)

    cache_dir_name = cache_dir.name.lower()
    recorded_configs = [
        result.get("optimizer_config")
        for result in all_results
        if isinstance(result.get("optimizer_config"), dict)
    ]
    summary_config = (
        recorded_configs[0]
        if recorded_configs and all(config == recorded_configs[0] for config in recorded_configs)
        else {}
    )
    recorded_growth_threshold = pd.to_numeric(
        summary_config.get("growth_rate_threshold"), errors="coerce"
    )
    recorded_gt_iou_threshold = pd.to_numeric(
        summary_config.get("gt_iou_threshold"), errors="coerce"
    )
    run_complete = bool(
        run_complete
        and len(exception_results) == 0
        and summary_config.get("classification_method") == GT_CLASSIFICATION_METHOD
        and summary_config.get("selection_rule_version") == SELECTION_RULE_VERSION
        and np.isclose(
            recorded_growth_threshold,
            CONSERVATIVE_GROWTH_THRESHOLD,
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            recorded_gt_iou_threshold,
            GT_IOU_THRESHOLD,
            rtol=0.0,
            atol=1e-12,
        )
    )
    if recorded_configs and all(config == recorded_configs[0] for config in recorded_configs):
        active_config = summary_config
        if active_config.get("preprocessing") == "bm3d_only":
            preprocessing_mode = (
                "BM3D only, sigma_psd="
                f"{active_config.get('bm3d_sigma_psd')}"
            )
        else:
            preprocessing_mode = str(active_config.get("preprocessing", "recorded"))
        provenance_note = "Read from optimizer_config in every cache JSON"
        metadata_recorded = "Yes"
    elif cache_dir_name == "cache_bm3d_noise2sr_e1500_verified":
        preprocessing_mode = "BM3D sigma_psd=40 followed by Noise2SR epochs=1500"
        provenance_note = "Verified cache directory generated by the fixed optimizer pipeline"
        metadata_recorded = "No (inferred from the legacy cache directory name)"
    else:
        preprocessing_mode = "Not established from cache JSON"
        provenance_note = "Cache JSON does not record preprocessing metadata"
        metadata_recorded = "No"

    df_metadata = pd.DataFrame(
        {
            "Item": [
                "Source_Cache_Directory",
                "Preprocessing_Mode",
                "Preprocessing_Provenance",
                "Cache_File_Count",
                "Normal_Result_Count",
                "Exception_Result_Count",
                "Candidate_Stage_Count",
                "Candidate_Pair_Count",
                "Growth_Rate_Threshold",
                "Growth_Rate_Row_Count",
                "Growth_Rate_Method",
                "Classification_Method",
                "GT_IoU_Threshold",
                "Selection_Fallback",
                "Parameter_Mask_Cache_Version",
                "Optimization_Split_Hash",
                "Validation_Exclusion_Rule",
                "Generated_At",
                "Preprocessing_Metadata_In_Cache_JSON",
            ],
            "Value": [
                str(cache_dir),
                preprocessing_mode,
                provenance_note,
                len(cache_paths),
                len(normal_results),
                len(exception_results),
                len(PARAM_STAGES),
                sum(len(stage) for stage in PARAM_STAGES),
                CONSERVATIVE_GROWTH_THRESHOLD,
                len(df_growth),
                "noise_growth / valid_growth; inf when valid_growth <= 0 and noise_growth > 0",
                summary_config.get("classification_method", "legacy_area_based"),
                summary_config.get("gt_iou_threshold"),
                (
                    "Global minimum total FP/TP across all pairs when no R(p) >= "
                    f"{CONSERVATIVE_GROWTH_THRESHOLD:g}"
                    if summary_config.get("selection_rule_version") == SELECTION_RULE_VERSION
                    else "Legacy cache selection rule"
                ),
                summary_config.get("parameter_mask_cache_version"),
                summary_config.get("optimization_split_hash"),
                "Exclude the union of Dataset_size and Dataset_shape image IDs",
                datetime.now().astimezone().isoformat(timespec="seconds"),
                metadata_recorded,
            ],
        }
    )

    df_manuscript_values = pd.DataFrame(
        [
            {
                "Item": "Publication_Ready",
                "Value": bool(run_complete),
                "Unit": "boolean",
                "Definition": (
                    "True only when cache IDs exactly match the frozen split, no "
                    "image failed, and all caches use the active GT, selection, "
                    "and threshold settings"
                ),
            },
            {
                "Item": "Expected_Optimization_Image_Count",
                "Value": expected_optimization_count,
                "Unit": "images",
                "Definition": "optimization_count recorded in optimization_split_manifest.json",
            },
            {
                "Item": "Optimization_Image_Count",
                "Value": len(all_results),
                "Unit": "images",
                "Definition": "All attempted optimizer images, including exceptions",
            },
            {
                "Item": "Optimization_Normal_Image_Count",
                "Value": len(normal_results),
                "Unit": "images",
                "Definition": "Images with a selected parameter pair",
            },
            {
                "Item": "Optimization_Exception_Image_Count",
                "Value": len(exception_results),
                "Unit": "images",
                "Definition": "Images without a selected parameter pair",
            },
            {
                "Item": "Candidate_Stage_Count",
                "Value": len(PARAM_STAGES),
                "Unit": "stages",
                "Definition": "Ordered strict-to-loose parameter stages",
            },
            {
                "Item": "Candidate_Pair_Count",
                "Value": sum(len(stage) for stage in PARAM_STAGES),
                "Unit": "pairs",
                "Definition": "All predicted-IoU/stability candidate pairs",
            },
            {
                "Item": "Growth_Rate_Threshold",
                "Value": CONSERVATIVE_GROWTH_THRESHOLD,
                "Unit": "ratio",
                "Definition": "Stopping threshold for delta_FP / delta_TP",
            },
            {
                "Item": "GT_Matching_IoU_Threshold",
                "Value": GT_IOU_THRESHOLD,
                "Unit": "IoU",
                "Definition": "Hungarian TP matching threshold",
            },
            {
                "Item": "Default_Predicted_IoU_Threshold",
                "Value": default_pair[0],
                "Unit": "proportion",
                "Definition": "Joint pair with the highest selection count after excluding terminal",
            },
            {
                "Item": "Default_Stability_Score_Threshold",
                "Value": default_pair[1],
                "Unit": "proportion",
                "Definition": "Joint pair with the highest selection count after excluding terminal",
            },
            {
                "Item": "Default_Pair",
                "Value": f"({default_pair[0]:.2f}, {default_pair[1]:.2f})",
                "Unit": "predicted-IoU, stability",
                "Definition": "Highest-frequency selected non-terminal joint pair",
            },
            {
                "Item": "Default_Pair_Selected_Count",
                "Value": default_count,
                "Unit": "images",
                "Definition": "Numerator for the reported default-pair frequency",
            },
            {
                "Item": "Default_Pair_Percent_of_All_Images",
                "Value": default_count / len(all_results) * 100.0,
                "Unit": "percent",
                "Definition": "Default count divided by all attempted optimizer images",
            },
            {
                "Item": "Default_Pair_Percent_of_Normal_Images",
                "Value": default_count / len(normal_results) * 100.0,
                "Unit": "percent",
                "Definition": "Default count divided by images with a selected pair",
            },
            {
                "Item": "Default_Neighborhood_Within_One_Stage_Count",
                "Value": neighborhood_counts[1],
                "Unit": "images",
                "Definition": "Selected pairs in the default pair's stage or one adjacent stage",
            },
            {
                "Item": "Default_Neighborhood_Within_One_Stage_Percent_of_All_Images",
                "Value": neighborhood_counts[1] / len(all_results) * 100.0,
                "Unit": "percent",
                "Definition": "One-stage neighborhood count divided by all attempted images",
            },
            {
                "Item": "Default_Neighborhood_Within_Two_Stages_Count",
                "Value": neighborhood_counts[2],
                "Unit": "images",
                "Definition": "Selected pairs in the default pair's stage or within two stages",
            },
            {
                "Item": "Default_Neighborhood_Within_Two_Stages_Percent_of_All_Images",
                "Value": neighborhood_counts[2] / len(all_results) * 100.0,
                "Unit": "percent",
                "Definition": "Two-stage neighborhood count divided by all attempted images",
            },
            {
                "Item": "Terminal_Pair_Selected_Count",
                "Value": terminal_count,
                "Unit": "images",
                "Definition": "Images for which the terminal pair was selected",
            },
            {
                "Item": "Terminal_Pair_Percent_of_All_Images",
                "Value": terminal_count / len(all_results) * 100.0,
                "Unit": "percent",
                "Definition": "Terminal count divided by all attempted optimizer images",
            },
            {
                "Item": "No_Crossing_Global_Minimum_Fallback_Count",
                "Value": no_crossing_fallback_count,
                "Unit": "images",
                "Definition": "Images with no R(p) threshold crossing; global minimum total FP/TP selected",
            },
            {
                "Item": "No_Crossing_Global_Minimum_Fallback_Percent_of_All_Images",
                "Value": no_crossing_fallback_count / len(all_results) * 100.0,
                "Unit": "percent",
                "Definition": "No-crossing fallback count divided by all attempted images",
            },
            {
                "Item": "Optimization_Split_Hash",
                "Value": summary_config.get("optimization_split_hash"),
                "Unit": "SHA256",
                "Definition": "Hash of included/excluded image IDs",
            },
            {
                "Item": "Optimization_Split_IDs_Match",
                "Value": actual_optimization_ids == expected_optimization_ids,
                "Unit": "boolean",
                "Definition": "Cached image stems exactly equal manifest-included IDs",
            },
        ],
        columns=["Item", "Value", "Unit", "Definition"],
    )

    return {
        "all_results": df_all_results,
        "normal_results": df_normal_results,
        "statistics": df_stats,
        "pair_metrics": df_pair_metrics,
        "growth_rates": df_growth,
        "default_pair_frequency": df_default_frequency,
        "manuscript_values": df_manuscript_values,
        "metadata": df_metadata,
        "cache_file_count": len(cache_paths),
        "normal_count": len(normal_results),
        "exception_count": len(exception_results),
    }


def generate_cache_summary(cache_dir, output_dir):
    """Rebuild Excel and aggregate figures from saved optimizer cache JSON files."""
    cache_dir = Path(cache_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = _build_cache_summary_tables(cache_dir)

    excel_path = output_dir / "optimization_results.xlsx"
    temp_excel_path = output_dir / ".optimization_results.cache_summary.tmp.xlsx"
    try:
        with pd.ExcelWriter(temp_excel_path, engine="openpyxl") as writer:
            tables["all_results"].to_excel(writer, sheet_name="All Images", index=False)
            tables["normal_results"].to_excel(writer, sheet_name="Normal Results", index=False)
            tables["statistics"].to_excel(writer, sheet_name="Statistics", index=False)
            tables["pair_metrics"].to_excel(writer, sheet_name="Pair Metrics", index=False)
            tables["growth_rates"].to_excel(writer, sheet_name="Growth Rates", index=False)
            tables["default_pair_frequency"].to_excel(
                writer, sheet_name="Default Pair Frequency", index=False
            )
            tables["manuscript_values"].to_excel(
                writer, sheet_name="Manuscript Values", index=False
            )
            tables["metadata"].to_excel(writer, sheet_name="Metadata", index=False)
        os.replace(temp_excel_path, excel_path)
    finally:
        if temp_excel_path.exists():
            temp_excel_path.unlink()

    summary_info = generate_sam_opt_ab_figure(excel_path, output_dir, dpi=600)
    frequency_info = generate_optimal_param_frequency_from_excel(
        excel_path, output_dir, dpi=300
    )
    return {
        **tables,
        "excel_path": excel_path,
        "summary_info": summary_info,
        "frequency_info": frequency_info,
    }


def total_noise_valid_ratio(combo):
    """Return total FP/TP; a pair with no true positives is not ratio-optimal."""
    valid_masks = int(combo.get('valid_masks', 0))
    noise_masks = int(combo.get('noise_masks', 0))
    if valid_masks <= 0:
        return float('inf')
    return noise_masks / valid_masks


def _combo_quality_key(combo):
    """Deterministic tie-breaking for a total FP/TP comparison."""
    return (
        total_noise_valid_ratio(combo),
        -int(combo.get('valid_masks', 0)),
        int(combo.get('noise_masks', 0)),
        int(combo.get('stage', 0)),
        int(combo.get('combo_idx', 0)),
    )


def _select_from_stage(stage_results, stage_idx, reason):
    """
    Select best combo from a given stage.

    For 1-combo stages: return the only combo.
    For 2-combo stages: calculate growth rate for each combo from the previous stage,
                        select the one with smaller growth rate.
    """
    combos = stage_results[stage_idx]

    if len(combos) == 1:
        return stage_idx, 0, reason

    # 2-combo stage: select by growth rate from previous stage
    if stage_idx > 0:
        prev_best = min(stage_results[stage_idx - 1], key=_combo_quality_key)

        combo_growth_rates = []
        for combo in combos:
            vg = combo['valid_masks'] - prev_best['valid_masks']
            ng = combo['noise_masks'] - prev_best['noise_masks']
            gr = calculate_growth_rate(vg, ng)
            combo_growth_rates.append(gr)

        best_idx = min(
            range(len(combos)),
            key=lambda index: (
                combo_growth_rates[index],
                _combo_quality_key(combos[index]),
            ),
        )
        gr_str = ', '.join([f'{gr:.3f}' if gr != float('inf') else 'inf' for gr in combo_growth_rates])
        reason += f" | 2-combo growth_rates=[{gr_str}], selected combo {best_idx}"
        return stage_idx, best_idx, reason
    else:
        # First stage with 2 combos, no previous stage: use lowest noise ratio
        best_idx = min(range(len(combos)), key=lambda j: _combo_quality_key(combos[j]))
        return stage_idx, best_idx, reason + f" | First stage, selected lowest noise/valid ratio"


def find_optimal_stage(stage_results):
    """
    Find optimal stage using growth rate analysis.

    Growth rate = noise_growth / valid_growth
    When growth_rate >= CONSERVATIVE_GROWTH_THRESHOLD: previous stage is optimal
    For 2-combo stages: select combo with smaller growth rate from prev stage

    Returns:
        optimal_stage_idx: Index of optimal stage
        optimal_combo_idx: Index of best combination within that stage
        reason: Explanation string
    """
    n_stages = len(stage_results)

    if n_stages < 2:
        return 0, 0, "Only one stage tested"

    # Calculate growth rates between stages
    for i in range(1, n_stages):
        prev_stage = stage_results[i-1]
        curr_stage = stage_results[i]

        # Get best result from each stage (lowest noise/valid ratio)
        prev_best = min(prev_stage, key=_combo_quality_key)
        curr_best = min(curr_stage, key=_combo_quality_key)

        valid_growth = curr_best['valid_masks'] - prev_best['valid_masks']
        noise_growth = curr_best['noise_masks'] - prev_best['noise_masks']

        # Check explosion
        if valid_growth <= 0:
            if noise_growth <= 0:
                continue
            else:
                reason = f"Stage {i}: valid_growth <= 0, but noise_growth > 0 (explosion)"
                return _select_from_stage(stage_results, i-1, reason)

        growth_rate = noise_growth / valid_growth

        if growth_rate >= CONSERVATIVE_GROWTH_THRESHOLD:
            reason = f"Stage {i}: growth_rate={growth_rate:.3f} >= {CONSERVATIVE_GROWTH_THRESHOLD} (conservative threshold)"
            return _select_from_stage(stage_results, i-1, reason)

    # No threshold crossing: select the global minimum total FP/TP pair.
    indexed_combos = [
        (stage_idx, combo_idx, combo)
        for stage_idx, stage in enumerate(stage_results)
        for combo_idx, combo in enumerate(stage)
    ]
    stage_idx, combo_idx, best_combo = min(
        indexed_combos,
        key=lambda item: _combo_quality_key(item[2]),
    )
    ratio = total_noise_valid_ratio(best_combo)
    ratio_text = f"{ratio:.6g}" if np.isfinite(ratio) else "inf"
    reason = (
        f"No stage transition had R(p) >= {CONSERVATIVE_GROWTH_THRESHOLD}; "
        f"selected global minimum total FP/TP={ratio_text} across all parameter pairs"
    )
    return stage_idx, combo_idx, reason


def save_image_cache(result, cache_dir):
    """Save GT metrics plus threshold-independent assignments (never mask arrays)."""
    cache = {
        'result_cache_version': 'gt_optimizer_result_v1',
        'image_name': result['image_name'],
        'image_path': result['image_path'],
        'exception': result.get('exception', False),
        'reason': result.get('reason', ''),
        'optimizer_config': {
            'preprocessing': 'bm3d_only',
            'bm3d_sigma_psd': BM3D_SIGMA,
            'points_per_side': 32,
            'points_per_batch': 256,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'crop_nms_thresh': 0.7,
            'box_nms_thresh': 0.7,
            'use_m2m': True,
            'sam_postprocessing_version': SAM_POSTPROCESSING_VERSION,
            'central_region_fraction': 0.8,
            'parameter_stage_count': len(PARAM_STAGES),
            'parameter_pair_count': sum(len(stage) for stage in PARAM_STAGES),
            'parameter_schedule': _parameter_schedule_records(),
            'growth_rate_threshold': CONSERVATIVE_GROWTH_THRESHOLD,
            'classification_method': GT_CLASSIFICATION_METHOD,
            'reference_evaluation_version': REFERENCE_EVALUATION_VERSION,
            'gt_iou_threshold': GT_IOU_THRESHOLD,
            'selection_rule_version': SELECTION_RULE_VERSION,
            'parameter_mask_cache_version': PARAMETER_MASK_CACHE_VERSION,
            'optimization_split_hash': globals().get('OPTIMIZATION_SPLIT_HASH'),
        },
        'source_size_bytes': result.get('source_size_bytes'),
        'source_mtime_ns': result.get('source_mtime_ns'),
        'cache_source': result.get('cache_source'),
        'crop_match_confidence': result.get('crop_match_confidence'),
        'boundary_gt_masks_in_region': result.get('boundary_gt_masks_in_region'),
        'mask_cache_path': result.get('mask_cache_path'),
        'assignment_cache_path': result.get('assignment_cache_path'),
    }

    if not result.get('exception', False):
        cache.update({
            'optimal_stage': int(result['optimal_stage']),
            'optimal_combo': int(result['optimal_combo']),
            'optimal_pred_iou': float(result['optimal_pred_iou']),
            'optimal_stability': float(result['optimal_stability']),
            'optimal_valid': int(result['optimal_valid']),
            'optimal_noise': int(result['optimal_noise']),
            'optimal_ratio': float(result['optimal_ratio']),
            'optimal_noise_valid_ratio': result.get('optimal_noise_valid_ratio'),
            'optimal_gt_masks': int(result['optimal_gt_masks']),
            'optimal_tp': int(result['optimal_tp']),
            'optimal_fp': int(result['optimal_fp']),
            'optimal_fn': int(result['optimal_fn']),
            'optimal_precision': float(result['optimal_precision']),
            'optimal_recall': float(result['optimal_recall']),
            'optimal_f1': float(result['optimal_f1']),
            'optimal_mean_matched_iou': float(result['optimal_mean_matched_iou']),
            'gt_iou_threshold': float(result['gt_iou_threshold']),
        })

    # Stage metrics and IoU assignments only; binary masks live in compressed NPZ.
    cache['stage_results'] = []
    for stage_data in result.get('stage_results', []):
        stage_cache = []
        for combo in stage_data:
            stage_cache.append({
                'stage': int(combo['stage']),
                'combo_idx': int(combo['combo_idx']),
                'pred_iou': float(combo['pred_iou']),
                'stability': float(combo['stability']),
                'total_masks': int(combo['total_masks']),
                'valid_masks': int(combo['valid_masks']),
                'noise_masks': int(combo['noise_masks']),
                'noise_ratio': float(combo['noise_ratio']),
                'gt_masks': int(combo['gt_masks']),
                'false_negative_masks': int(combo['false_negative_masks']),
                'precision': float(combo['precision']),
                'recall': float(combo['recall']),
                'f1': float(combo['f1']),
                'mean_matched_iou': float(combo['mean_matched_iou']),
                'noise_valid_ratio': combo.get('noise_valid_ratio'),
                'hungarian_assignments': combo.get('hungarian_assignments', []),
            })
        cache['stage_results'].append(stage_cache)

    name_stem = os.path.splitext(result['image_name'])[0]
    cache_path = os.path.join(cache_dir, f'{name_stem}_cache.json')
    temp_path = f"{cache_path}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
    os.replace(temp_path, cache_path)
    return cache_path


def load_image_cache(cache_path):
    """Load lightweight per-image cache."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _result_cache_matches_active_config(cache, image_path):
    """Reject optimizer caches produced from another preprocessing setup or file."""
    if not isinstance(cache, dict):
        return False
    if cache.get('result_cache_version') != 'gt_optimizer_result_v1':
        return False
    config = cache.get('optimizer_config')
    if not isinstance(config, dict):
        return False
    if config.get('preprocessing') != 'bm3d_only':
        return False
    try:
        if not np.isclose(
            float(config.get('bm3d_sigma_psd')),
            BM3D_SIGMA,
            rtol=0.0,
            atol=1e-12,
        ):
            return False
    except (TypeError, ValueError):
        return False
    expected_sam_config = {
        'points_per_side': 32,
        'points_per_batch': 256,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'crop_nms_thresh': 0.7,
        'box_nms_thresh': 0.7,
        'use_m2m': True,
        'sam_postprocessing_version': SAM_POSTPROCESSING_VERSION,
        'classification_method': GT_CLASSIFICATION_METHOD,
        'reference_evaluation_version': REFERENCE_EVALUATION_VERSION,
        'selection_rule_version': SELECTION_RULE_VERSION,
        'parameter_mask_cache_version': PARAMETER_MASK_CACHE_VERSION,
        'parameter_stage_count': len(PARAM_STAGES),
        'parameter_pair_count': sum(len(stage) for stage in PARAM_STAGES),
        'parameter_schedule': _parameter_schedule_records(),
        'growth_rate_threshold': CONSERVATIVE_GROWTH_THRESHOLD,
        'gt_iou_threshold': GT_IOU_THRESHOLD,
        'optimization_split_hash': globals().get('OPTIMIZATION_SPLIT_HASH'),
    }
    for key, expected_value in expected_sam_config.items():
        actual_value = config.get(key)
        if isinstance(expected_value, float):
            try:
                if not np.isclose(float(actual_value), expected_value, rtol=0.0, atol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
        elif actual_value != expected_value:
            return False
    path = Path(image_path)
    stat = path.stat()
    source_matches = (
        cache.get('source_size_bytes') == stat.st_size
        and cache.get('source_mtime_ns') == stat.st_mtime_ns
    )
    if not source_matches:
        return False
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    optimizer_input = ensure_even_dimensions(image)
    signature = _common_cache_signature(
        path, optimizer_input, _central_region_box(optimizer_input.shape)
    )
    return _load_valid_assignment_cache(path.name, signature) is not None


def _preprocessed_cache_path(image_name):
    """Return this optimizer's BM3D-only cache path."""
    return Path(PREPROCESSED_CACHE_DIR) / f"{Path(image_name).stem}_1_bm3d_preprocessed.png"


def _reusable_bm3d_candidates(image_name):
    """Locate compatible BM3D-only outputs from prior experiments."""
    stem = Path(image_name).stem
    filename = f"{stem}_1_bm3d_preprocessed.png"
    direct_candidates = [
        LEGACY_BM3D_PREPROCESS_DIR / filename,
        COMPARE_PREPROCESS_DIR / filename,
        COMPARE_PREPROCESS_DIR / f"{stem}_preprocessed" / filename,
    ]
    if any(path.exists() for path in direct_candidates):
        return direct_candidates
    if COMPARE_PREPROCESS_DIR.exists():
        return list(COMPARE_PREPROCESS_DIR.rglob(filename))
    return []


def _normalize_preprocessed_image(image):
    """Normalize reusable preprocessing outputs to BGR uint8."""
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported preprocessed image shape: {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _array_sha256(image):
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode('ascii'))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _preprocessed_cache_matches_source(cache_path, source_image):
    metadata_path = cache_path.with_suffix('.json')
    if not metadata_path.exists():
        return False
    try:
        with metadata_path.open('r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        return (
            metadata.get('method') == 'bm3d_only'
            and np.isclose(
                float(metadata.get('bm3d_sigma_psd')),
                BM3D_SIGMA,
                rtol=0.0,
                atol=1e-12,
            )
            and metadata.get('input_shape') == list(source_image.shape)
            and metadata.get('input_sha256') == _array_sha256(source_image)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _is_compatible_preprocessed_shape(source, preprocessed):
    """BM3D-only outputs must retain the optimizer input dimensions."""
    return source.shape[:2] == preprocessed.shape[:2]


def _save_preprocessed_cache(image_name, image, source, source_image):
    """Save an auditable, reusable BM3D-only image for this optimizer."""
    preprocessed = _normalize_preprocessed_image(image)
    cache_path = _preprocessed_cache_path(image_name)
    if not cv2.imwrite(str(cache_path), preprocessed):
        raise IOError(f"Failed to save preprocessed image: {cache_path}")

    metadata = {
        'image_name': Path(image_name).stem,
        'method': 'bm3d_only',
        'bm3d_sigma_psd': BM3D_SIGMA,
        'clahe_applied': False,
        'source': source,
        'input_shape': list(source_image.shape),
        'input_dtype': str(source_image.dtype),
        'input_sha256': _array_sha256(source_image),
        'shape': list(preprocessed.shape),
        'saved_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(cache_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return preprocessed


def preprocess_bm3d_only(image, image_name):
    """
    Return a verified BM3D-only image without Noise2SR or CLAHE.

    Reuse order: optimizer cache, preprocessing-comparison output, fresh run.
    """
    cache_path = _preprocessed_cache_path(image_name)
    if cache_path.exists():
        cached = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
        if cached is not None and _preprocessed_cache_matches_source(cache_path, image):
            cached = _normalize_preprocessed_image(cached)
            if _is_compatible_preprocessed_shape(image, cached):
                print(f"   [CACHE] BM3D only: {cache_path.name}")
                return cached
        print(f"   [WARN] Ignoring incompatible preprocessing cache: {cache_path}")

    for candidate in _reusable_bm3d_candidates(image_name):
        if not candidate.exists():
            continue
        reused = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if reused is None:
            continue
        reused = _normalize_preprocessed_image(reused)
        if not _is_compatible_preprocessed_shape(image, reused):
            print(f"   [WARN] Ignoring incompatible comparison output: {candidate}")
            continue
        print(f"   [REUSE] BM3D only: {candidate.name}")
        return _save_preprocessed_cache(image_name, reused, str(candidate), image)

    preprocessed, info = preprocess_with_config(
        image.copy(),
        sigma_psd=BM3D_SIGMA,
        bm3d_enabled=True,
        noise2sr_enabled=False,
        clahe_enabled=False,
        verbose=True,
    )
    if not info.get('bm3d_applied'):
        raise RuntimeError(
            f"Expected BM3D-only preprocessing, got '{info.get('denoising_method', 'unknown')}'."
        )
    if info.get('noise2sr_applied') or info.get('clahe_applied'):
        raise RuntimeError('Optimizer preprocessing must be BM3D only.')
    return _save_preprocessed_cache(
        image_name,
        preprocessed,
        'generated_by_sam_param_optimizer',
        image,
    )


def ensure_even_dimensions(image):
    """Use the same trailing-pixel rule as the preprocessing comparison."""
    height, width = image.shape[:2]
    even_height = height if height % 2 == 0 else height - 1
    even_width = width if width % 2 == 0 else width - 1
    if even_height < 2 or even_width < 2:
        raise ValueError(f"Image is too small after even-dimension adjustment: {image.shape}")
    return image[:even_height, :even_width]


def _parameter_schedule_records():
    return [
        {
            'stage': int(stage_idx),
            'combo_idx': int(combo_idx),
            'pred_iou': float(pred_iou),
            'stability': float(stability),
        }
        for stage_idx, stage in enumerate(PARAM_STAGES)
        for combo_idx, (pred_iou, stability) in enumerate(stage)
    ]


def _fixed_sam_generator_config():
    return {
        'points_per_side': 32,
        'points_per_batch': 256,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'crop_nms_thresh': 0.7,
        'box_nms_thresh': 0.7,
        'use_m2m': True,
    }


def _parameter_mask_cache_path(image_name):
    return Path(PARAMETER_MASK_CACHE_DIR) / f"{Path(image_name).stem}_parameter_masks.npz"


def _assignment_cache_path(image_name):
    return Path(GT_ASSIGNMENT_CACHE_DIR) / f"{Path(image_name).stem}_assignments.json"


def _central_region_box(image_shape):
    height, width = image_shape[:2]
    margin_h = int(height * 0.1)
    margin_w = int(width * 0.1)
    return margin_w, margin_h, width - margin_w, height - margin_h


def _common_cache_signature(image_path, optimizer_input, crop_box):
    source_path = Path(image_path)
    source_stat = source_path.stat()
    x1, y1, x2, y2 = crop_box
    return {
        'image_name': source_path.name,
        'source_size_bytes': int(source_stat.st_size),
        'source_mtime_ns': int(source_stat.st_mtime_ns),
        'optimizer_input_shape': list(optimizer_input.shape),
        'optimizer_input_sha256': _array_sha256(optimizer_input),
        'preprocessing': 'bm3d_only',
        'bm3d_sigma_psd': BM3D_SIGMA,
        'central_crop_box_xyxy': [int(x1), int(y1), int(x2), int(y2)],
        'mask_shape': [int(y2 - y1), int(x2 - x1)],
        'sam_generator_config': _fixed_sam_generator_config(),
        'sam_postprocessing_version': SAM_POSTPROCESSING_VERSION,
        'parameter_schedule': _parameter_schedule_records(),
    }


def _metadata_contains_signature(metadata, signature):
    if not isinstance(metadata, dict):
        return False
    return all(metadata.get(key) == value for key, value in signature.items())


def _reference_file_signature(image_stem):
    annotation_path = find_reference_annotation_path(
        image_stem, REFERENCE_ANNOTATION_DIR
    )
    reference_image_path = find_reference_image_path(
        image_stem, REFERENCE_IMAGE_DIR
    )
    if annotation_path is None or reference_image_path is None:
        return None
    annotation_stat = annotation_path.stat()
    reference_image_stat = reference_image_path.stat()
    return {
        'annotation_path': str(annotation_path.resolve()),
        'annotation_size_bytes': int(annotation_stat.st_size),
        'annotation_mtime_ns': int(annotation_stat.st_mtime_ns),
        'reference_image_path': str(reference_image_path.resolve()),
        'reference_image_size_bytes': int(reference_image_stat.st_size),
        'reference_image_mtime_ns': int(reference_image_stat.st_mtime_ns),
    }


def _load_valid_mask_cache(image_name, common_signature, verify_checksum=False):
    archive_path = _parameter_mask_cache_path(image_name)
    try:
        metadata = load_parameter_mask_metadata(
            archive_path, verify_checksum=verify_checksum
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None, None
    if not _metadata_contains_signature(metadata, common_signature):
        return None, None
    if metadata.get('pair_count') != len(_parameter_schedule_records()):
        return None, None
    return archive_path, metadata


def _load_valid_assignment_cache(image_name, common_signature):
    cache_path = _assignment_cache_path(image_name)
    if not cache_path.is_file():
        return None
    try:
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if cache.get('cache_version') != 'gt_hungarian_assignments_v1':
        return None
    if cache.get('reference_evaluation_version') != REFERENCE_EVALUATION_VERSION:
        return None
    if not _metadata_contains_signature(cache, common_signature):
        return None
    reference_signature = _reference_file_signature(Path(image_name).stem)
    if reference_signature is None or not _metadata_contains_signature(
        cache, reference_signature
    ):
        return None
    archive_path, mask_metadata = _load_valid_mask_cache(
        image_name, common_signature, verify_checksum=False
    )
    if archive_path is None:
        return None
    if cache.get('mask_archive_sha256') != mask_metadata.get('archive_sha256'):
        return None
    if cache.get('pair_count') != len(_parameter_schedule_records()):
        return None
    return cache


def _load_gt_region(image_path, crop_box):
    reference = load_datasetninja_reference_masks(
        image_path,
        REFERENCE_ANNOTATION_DIR,
        REFERENCE_IMAGE_DIR,
        target_class='particle',
    )
    gt_masks, boundary_flags = crop_reference_masks(
        reference['gt_masks'], crop_box, border_margin=5
    )
    reference = dict(reference)
    reference.pop('gt_masks', None)
    reference['gt_masks_in_region'] = int(len(gt_masks))
    reference['boundary_gt_masks_in_region'] = int(sum(boundary_flags))
    return gt_masks, reference


def _save_assignment_cache(
    image_name,
    common_signature,
    reference,
    pair_entries,
    mask_metadata,
):
    pairs = []
    n_gt = int(reference['gt_masks_in_region'])
    for entry in pair_entries:
        assignments, _ = hungarian_assignment_scores(
            entry['gt_masks'], entry['masks']
        )
        pairs.append(
            {
                'stage': int(entry['stage']),
                'combo_idx': int(entry['combo_idx']),
                'pred_iou': float(entry['pred_iou']),
                'stability': float(entry['stability']),
                'n_pred': int(len(entry['masks'])),
                'assignments': assignments,
            }
        )
    cache = {
        **common_signature,
        'cache_version': 'gt_hungarian_assignments_v1',
        'reference_evaluation_version': REFERENCE_EVALUATION_VERSION,
        'classification_method': GT_CLASSIFICATION_METHOD,
        'n_gt': n_gt,
        'pair_count': len(pairs),
        'pairs': pairs,
        'mask_cache_version': PARAMETER_MASK_CACHE_VERSION,
        'mask_archive_sha256': mask_metadata['archive_sha256'],
        'mask_archive_path': str(_parameter_mask_cache_path(image_name).resolve()),
        'crop_offset': list(reference['crop_offset']),
        'crop_match_confidence': float(reference['crop_match_confidence']),
        'boundary_gt_masks_in_region': int(reference['boundary_gt_masks_in_region']),
        'skipped_non_target_objects': int(reference['skipped_non_target_objects']),
        'annotation_path': reference['annotation_path'],
        'annotation_size_bytes': int(reference['annotation_size_bytes']),
        'annotation_mtime_ns': int(reference['annotation_mtime_ns']),
        'reference_image_path': reference['reference_image_path'],
        'reference_image_size_bytes': int(reference['reference_image_size_bytes']),
        'reference_image_mtime_ns': int(reference['reference_image_mtime_ns']),
    }
    cache_path = _assignment_cache_path(image_name)
    temp_path = cache_path.with_name(f".{cache_path.name}.tmp")
    try:
        temp_path.write_text(json.dumps(cache, indent=2), encoding='utf-8')
        os.replace(temp_path, cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return cache


def _stage_results_from_assignments(assignment_cache, tau):
    schedule = _parameter_schedule_records()
    cached_pairs = assignment_cache.get('pairs', [])
    if len(cached_pairs) != len(schedule):
        raise ValueError(
            f"Expected {len(schedule)} cached parameter pairs, found {len(cached_pairs)}"
        )
    stage_results = [[] for _ in PARAM_STAGES]
    n_gt = int(assignment_cache['n_gt'])
    for expected, pair in zip(schedule, cached_pairs):
        for key in ('stage', 'combo_idx', 'pred_iou', 'stability'):
            if pair.get(key) != expected[key]:
                raise ValueError(
                    f"Assignment cache schedule mismatch at {expected}: {pair}"
                )
        metrics = metrics_from_assignment_scores(
            n_gt,
            int(pair['n_pred']),
            pair.get('assignments', []),
            tau,
        )
        noise_valid_ratio = (
            metrics['false_positive'] / metrics['true_positive']
            if metrics['true_positive'] > 0
            else None
        )
        combo = {
            'stage': int(pair['stage']),
            'combo_idx': int(pair['combo_idx']),
            'pred_iou': float(pair['pred_iou']),
            'stability': float(pair['stability']),
            'total_masks': int(metrics['n_pred']),
            'gt_masks': int(metrics['n_gt']),
            'valid_masks': int(metrics['true_positive']),
            'noise_masks': int(metrics['false_positive']),
            'false_negative_masks': int(metrics['false_negative']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'mean_matched_iou': float(metrics['mean_matched_iou']),
            'noise_ratio': (
                metrics['false_positive'] / metrics['n_pred']
                if metrics['n_pred'] > 0
                else 0.0
            ),
            'noise_valid_ratio': noise_valid_ratio,
            'hungarian_assignments': pair.get('assignments', []),
        }
        stage_results[combo['stage']].append(combo)
    return stage_results


def _generate_and_cache_parameter_masks(
    image_name,
    particle_region,
    sam,
    common_signature,
):
    if sam is None:
        raise RuntimeError(
            "Compressed masks are missing or stale; rerun without "
            "--reevaluate-gt-cache once to generate them with SAM2."
        )
    pair_entries = []
    fixed_config = _fixed_sam_generator_config()
    schedule = _parameter_schedule_records()
    print(f"\n[INFO] Generating and caching all {len(schedule)} parameter pairs...")
    for pair_index, pair in enumerate(schedule, start=1):
        print(
            f"   [{pair_index:02d}/{len(schedule)}] Stage {pair['stage'] + 1} "
            f"({pair['pred_iou']:.2f}, {pair['stability']:.2f})...",
            end='',
        )
        mask_generator = EmptyMaskSafeSAM2AutomaticMaskGenerator(
            model=sam,
            pred_iou_thresh=pair['pred_iou'],
            stability_score_thresh=pair['stability'],
            **fixed_config,
        )
        raw_masks = mask_generator.generate(particle_region)
        masks_after_background = filter_sam_background_masks(
            raw_masks, particle_region.shape[:2]
        )
        postprocessed_masks = filter_overlapping_masks_by_centroid(
            masks_after_background
        )
        print(
            f" raw={len(raw_masks)}, background={len(masks_after_background)}, "
            f"postprocessed={len(postprocessed_masks)}"
        )
        pair_entries.append({**pair, 'masks': postprocessed_masks})

    archive_path = _parameter_mask_cache_path(image_name)
    mask_metadata = save_parameter_mask_cache(
        archive_path,
        pair_entries,
        {
            **common_signature,
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        },
    )
    print(
        f"   [CACHE] Lossless postprocessed masks: {archive_path.name} "
        f"({mask_metadata['archive_size_bytes']} bytes)"
    )
    return pair_entries, mask_metadata


def _pair_entries_from_mask_cache(archive_path, mask_metadata, gt_masks):
    cached_pairs = load_all_parameter_masks(archive_path, mask_metadata)
    return [
        {
            'stage': int(pair['stage']),
            'combo_idx': int(pair['combo_idx']),
            'pred_iou': float(pair['pred_iou']),
            'stability': float(pair['stability']),
            'masks': pair['masks'],
            'gt_masks': gt_masks,
        }
        for pair in cached_pairs
    ]


def _result_from_assignment_cache(image_path, assignment_cache, cache_source):
    image_name = Path(image_path).name
    stage_results = _stage_results_from_assignments(
        assignment_cache, GT_IOU_THRESHOLD
    )
    optimal_stage_idx, optimal_combo_idx, reason = find_optimal_stage(stage_results)
    optimal = stage_results[optimal_stage_idx][optimal_combo_idx]
    ratio = total_noise_valid_ratio(optimal)
    print(
        f"   [OK] Optimal: Stage {optimal_stage_idx + 1}, "
        f"({optimal['pred_iou']:.2f}, {optimal['stability']:.2f})"
    )
    print(
        f"   GT={optimal['gt_masks']}, TP={optimal['valid_masks']}, "
        f"FP={optimal['noise_masks']}, FN={optimal['false_negative_masks']}, "
        f"FP/TP={'inf' if not np.isfinite(ratio) else f'{ratio:.6g}'}"
    )
    print(f"   Reason: {reason}")
    source_stat = Path(image_path).stat()
    return {
        'image_name': image_name,
        'image_path': str(Path(image_path).resolve()),
        'exception': False,
        'optimal_stage': optimal_stage_idx,
        'optimal_combo': optimal_combo_idx,
        'optimal_pred_iou': optimal['pred_iou'],
        'optimal_stability': optimal['stability'],
        'optimal_valid': optimal['valid_masks'],
        'optimal_noise': optimal['noise_masks'],
        'optimal_ratio': optimal['noise_ratio'],
        'optimal_noise_valid_ratio': None if not np.isfinite(ratio) else ratio,
        'optimal_gt_masks': optimal['gt_masks'],
        'optimal_tp': optimal['valid_masks'],
        'optimal_fp': optimal['noise_masks'],
        'optimal_fn': optimal['false_negative_masks'],
        'optimal_precision': optimal['precision'],
        'optimal_recall': optimal['recall'],
        'optimal_f1': optimal['f1'],
        'optimal_mean_matched_iou': optimal['mean_matched_iou'],
        'gt_iou_threshold': GT_IOU_THRESHOLD,
        'reason': reason,
        'stage_results': stage_results,
        'particle_region': None,
        'source_size_bytes': int(source_stat.st_size),
        'source_mtime_ns': int(source_stat.st_mtime_ns),
        'cache_source': cache_source,
        'crop_match_confidence': assignment_cache.get('crop_match_confidence'),
        'boundary_gt_masks_in_region': assignment_cache.get(
            'boundary_gt_masks_in_region'
        ),
        'mask_cache_path': assignment_cache.get('mask_archive_path'),
        'assignment_cache_path': str(_assignment_cache_path(image_name).resolve()),
    }


def process_single_image_gt(image_path, sam, device):
    """Evaluate all parameter pairs against DatasetNinja GT for one image."""
    del device
    image_path = str(Path(image_path).resolve())
    image_name = Path(image_path).name
    print(f"\n{'=' * 80}")
    print(f"Processing: {image_name}")
    print(f"{'=' * 80}")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read optimizer image: {image_path}")
    optimizer_input = ensure_even_dimensions(image)
    crop_box = _central_region_box(optimizer_input.shape)
    common_signature = _common_cache_signature(
        image_path, optimizer_input, crop_box
    )

    if not FORCE_REPROCESS:
        assignment_cache = _load_valid_assignment_cache(
            image_name, common_signature
        )
        if assignment_cache is not None:
            print(
                "   [CACHE] Reusing threshold-independent Hungarian IoU "
                f"assignments; applying tau={GT_IOU_THRESHOLD:g}"
            )
            return _result_from_assignment_cache(
                image_path, assignment_cache, 'hungarian_assignment_cache'
            )

    allow_mask_reuse = not FORCE_REPROCESS or ARGS.reevaluate_gt_cache
    archive_path = None
    mask_metadata = None
    if allow_mask_reuse:
        archive_path, mask_metadata = _load_valid_mask_cache(
            image_name, common_signature, verify_checksum=True
        )

    gt_masks, reference = _load_gt_region(image_path, crop_box)
    print(
        f"   [GT] particles={len(gt_masks)}, "
        f"boundary-touching={reference['boundary_gt_masks_in_region']}, "
        f"crop match={reference['crop_match_confidence']:.6f}"
    )

    if archive_path is not None:
        print(f"   [CACHE] Re-evaluating compressed masks: {archive_path.name}")
        pair_entries = _pair_entries_from_mask_cache(
            archive_path, mask_metadata, gt_masks
        )
        cache_source = 'compressed_parameter_mask_cache'
    else:
        if ARGS.reevaluate_gt_cache:
            raise RuntimeError(
                f"No compatible compressed 19-pair mask cache for {image_name}. "
                "Run the optimizer once without --reevaluate-gt-cache."
            )
        print(f"   [PREPROCESS] BM3D only (sigma_psd={BM3D_SIGMA:g})")
        preprocessed = preprocess_bm3d_only(optimizer_input, image_name)
        x1, y1, x2, y2 = crop_box
        particle_region = np.ascontiguousarray(preprocessed[y1:y2, x1:x2])
        if tuple(particle_region.shape[:2]) != tuple(common_signature['mask_shape']):
            raise ValueError(
                f"Central-region shape {particle_region.shape[:2]} does not match "
                f"cache signature {common_signature['mask_shape']}"
            )
        pair_entries, mask_metadata = _generate_and_cache_parameter_masks(
            image_name,
            particle_region,
            sam,
            common_signature,
        )
        for entry in pair_entries:
            entry['gt_masks'] = gt_masks
        cache_source = 'sam2_generated_and_compressed'

    assignment_cache = _save_assignment_cache(
        image_name,
        common_signature,
        reference,
        pair_entries,
        mask_metadata,
    )
    print(
        f"   [CACHE] Hungarian assignments: "
        f"{_assignment_cache_path(image_name).name}"
    )
    return _result_from_assignment_cache(
        image_path, assignment_cache, cache_source
    )


# Cache-summary mode: rebuild aggregate outputs from saved JSON only and exit.
if ARGS.cache_summary:
    print(f"\n{'='*80}")
    print("CACHE-ONLY SUMMARY REBUILD")
    print(f"{'='*80}")
    resolved_cache_dir = _resolve_cache_directory(
        ARGS.cache_dir or CACHE_DIR, OUTPUT_DIR
    )
    summary_out_dir = ARGS.summary_output_dir if ARGS.summary_output_dir else OUTPUT_DIR
    print(f"[INFO] Cache: {resolved_cache_dir}")
    print(f"[INFO] Output: {Path(summary_out_dir).resolve()}")
    cache_summary = generate_cache_summary(resolved_cache_dir, summary_out_dir)
    summary_info = cache_summary["summary_info"]
    print(f"[OK] Saved: {cache_summary['excel_path']}")
    print(f"[OK] Saved: {Path(summary_out_dir).resolve() / 'optimal_param_frequency.png'}")
    print(f"[OK] Saved: {Path(summary_out_dir).resolve() / 'optimal_param_frequency_notext.png'}")
    print(f"[OK] Saved: {Path(summary_out_dir).resolve() / 'sam_opt_ab.pdf'}")
    print(f"[OK] Saved: {Path(summary_out_dir).resolve() / 'sam_opt_ab.png'}")
    print(f"[OK] Saved: {Path(summary_out_dir).resolve() / 'sam_opt_ab_notext.pdf'}")
    print(f"[OK] Saved: {Path(summary_out_dir).resolve() / 'sam_opt_ab_notext.png'}")
    print(
        "[INFO] Rebuilt from cache: "
        f"total={cache_summary['cache_file_count']}, "
        f"normal={cache_summary['normal_count']}, "
        f"exceptions={cache_summary['exception_count']}, "
        f"growth_rows={len(cache_summary['growth_rates'])}, "
        f"mode_pair={summary_info['mode_pair']}, "
        f"growth_from_excel={summary_info['growth_from_excel']}"
    )
    sys.exit(0)


# Figure-only mode: build paper figure from existing Excel and exit.
if ARGS.figure_only:
    print(f"\n{'='*80}")
    print("FIGURE-ONLY MODE")
    print(f"{'='*80}")
    resolved_excel = _resolve_excel_path(ARGS.excel_path, OUTPUT_DIR)
    summary_out_dir = ARGS.summary_output_dir if ARGS.summary_output_dir else OUTPUT_DIR
    print(f"[INFO] Excel: {resolved_excel}")
    print(f"[INFO] Output: {summary_out_dir}")
    summary_info = generate_sam_opt_ab_figure(resolved_excel, summary_out_dir, dpi=600)
    freq_info = generate_optimal_param_frequency_from_excel(resolved_excel, summary_out_dir, dpi=300)
    print(f"[OK] Saved: {Path(summary_out_dir) / 'sam_opt_ab.pdf'}")
    print(f"[OK] Saved: {Path(summary_out_dir) / 'sam_opt_ab.png'}")
    print(f"[OK] Saved: {Path(summary_out_dir) / 'sam_opt_ab_notext.pdf'}")
    print(f"[OK] Saved: {Path(summary_out_dir) / 'sam_opt_ab_notext.png'}")
    print(f"[OK] Saved: {freq_info['freq_path']}")
    print(f"[OK] Saved: {freq_info['freq_notext_path']}")
    print(
        "[INFO] Summary: "
        f"total={summary_info['total_images']}, "
        f"valid={summary_info['valid_images']}, "
        f"mode_pair={summary_info['mode_pair']}, "
        f"growth_from_excel={summary_info['growth_from_excel']}"
    )
    sys.exit(0)


def collect_optimizer_images():
    """Use Dataset images not present in either validation dataset."""
    global OPTIMIZATION_SPLIT_HASH

    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    def image_paths_in(folder):
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {folder}")
        return sorted(
            (
                path.resolve()
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            ),
            key=lambda path: path.name.lower(),
        )

    dataset_paths = image_paths_in(IMAGE_FOLDER)
    if not dataset_paths:
        raise ValueError(f"No images found in {IMAGE_FOLDER}")
    if (
        not SINGLE_IMAGE_PATH
        and SAMPLE_SIZE is None
        and len(dataset_paths) != EXPECTED_IMAGES
        and not ARGS.allow_image_count_mismatch
    ):
        raise ValueError(
            f"Full Dataset must contain {EXPECTED_IMAGES} images; found "
            f"{len(dataset_paths)} in {IMAGE_FOLDER}."
        )

    stems = [path.stem for path in dataset_paths]
    duplicate_stems = sorted(
        stem for stem, count in Counter(stems).items() if count > 1
    )
    if duplicate_stems:
        raise ValueError(f"Duplicate Dataset image IDs: {duplicate_stems}")

    size_ids = {path.stem for path in image_paths_in(SIZE_DATASET_DIR)}
    shape_ids = {path.stem for path in image_paths_in(SHAPE_DATASET_DIR)}
    excluded_ids = size_ids | shape_ids
    dataset_ids = set(stems)
    eligible_paths = [path for path in dataset_paths if path.stem not in excluded_ids]

    manifest_rows = []
    for path in dataset_paths:
        in_size = path.stem in size_ids
        in_shape = path.stem in shape_ids
        annotation_available = find_reference_annotation_path(
            path.stem, REFERENCE_ANNOTATION_DIR
        ) is not None
        reference_image_available = find_reference_image_path(
            path.stem, REFERENCE_IMAGE_DIR
        ) is not None
        reasons = []
        if in_size:
            reasons.append('size_validation')
        if in_shape:
            reasons.append('shape_validation')
        manifest_rows.append(
            {
                'Image_ID': path.stem,
                'Dataset_Path': str(path),
                'In_Size_Dataset': in_size,
                'In_Shape_Dataset': in_shape,
                'Optimization_Included': not reasons,
                'Exclusion_Reason': ';'.join(reasons),
                'Reference_Annotation_Available': annotation_available,
                'Reference_Image_Available': reference_image_available,
            }
        )

    hash_payload = [
        {
            'image_id': row['Image_ID'],
            'included': row['Optimization_Included'],
            'reason': row['Exclusion_Reason'],
        }
        for row in manifest_rows
    ]
    OPTIMIZATION_SPLIT_HASH = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    manifest_csv = Path(OUTPUT_DIR) / 'optimization_split_manifest.csv'
    manifest_json = Path(OUTPUT_DIR) / 'optimization_split_manifest.json'
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False, encoding='utf-8-sig')
    manifest_json.write_text(
        json.dumps(
            {
                'split_hash': OPTIMIZATION_SPLIT_HASH,
                'dataset_count': len(dataset_paths),
                'size_dataset_count': len(size_ids & dataset_ids),
                'shape_dataset_count': len(shape_ids & dataset_ids),
                'validation_overlap_count': len(size_ids & shape_ids & dataset_ids),
                'excluded_union_count': len(excluded_ids & dataset_ids),
                'optimization_count': len(eligible_paths),
                'rows': manifest_rows,
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    print("\n[INFO] Optimization split:")
    print(f"   Dataset: {len(dataset_paths)}")
    print(f"   Size validation IDs: {len(size_ids & dataset_ids)}")
    print(f"   Shape validation IDs: {len(shape_ids & dataset_ids)}")
    print(f"   Size/shape overlap: {len(size_ids & shape_ids & dataset_ids)}")
    print(f"   Excluded union: {len(excluded_ids & dataset_ids)}")
    print(f"   Optimizer images: {len(eligible_paths)}")
    print(f"   Split hash: {OPTIMIZATION_SPLIT_HASH}")
    print(f"   Manifest: {manifest_csv}")

    missing_reference_rows = [
        row
        for row in manifest_rows
        if row['Optimization_Included']
        and (
            not row['Reference_Annotation_Available']
            or not row['Reference_Image_Available']
        )
    ]
    if missing_reference_rows:
        missing_ids = ', '.join(row['Image_ID'] for row in missing_reference_rows[:10])
        raise RuntimeError(
            f"GT reference files are missing for {len(missing_reference_rows)} "
            f"optimizer images: {missing_ids}"
        )

    if SINGLE_IMAGE_PATH:
        single_path = Path(SINGLE_IMAGE_PATH).expanduser()
        if not single_path.is_absolute():
            single_path = (Path(parent_dir) / single_path).resolve()
        if not single_path.is_file():
            raise ValueError(f"Single image not found: {single_path}")
        if single_path.stem in excluded_ids:
            raise ValueError(
                f"{single_path.stem} belongs to a size/shape validation dataset "
                "and is excluded from optimizer fitting."
            )
        selected_paths = [single_path]
    else:
        selected_paths = eligible_paths

    if SAMPLE_SIZE is not None:
        selected_paths = selected_paths[:SAMPLE_SIZE]
    return [str(path.resolve()) for path in selected_paths]


image_paths = collect_optimizer_images()
compatible_result_cache_count = 0
optimizer_bm3d_cache_count = 0
comparison_bm3d_reuse_count = 0
compressed_mask_cache_count = 0
hungarian_assignment_cache_count = 0
missing_compressed_mask_cache_images = []
for audit_image_path in image_paths:
    audit_path = Path(audit_image_path)
    result_cache_path = Path(CACHE_DIR) / f"{audit_path.stem}_cache.json"
    if result_cache_path.exists():
        try:
            cached_result = load_image_cache(result_cache_path)
            if _result_cache_matches_active_config(cached_result, audit_path):
                compatible_result_cache_count += 1
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if _preprocessed_cache_path(audit_path.name).exists():
        optimizer_bm3d_cache_count += 1
    elif any(path.exists() for path in _reusable_bm3d_candidates(audit_path.name)):
        comparison_bm3d_reuse_count += 1
    audit_image = cv2.imread(str(audit_path), cv2.IMREAD_COLOR)
    if audit_image is not None:
        audit_input = ensure_even_dimensions(audit_image)
        audit_signature = _common_cache_signature(
            audit_path,
            audit_input,
            _central_region_box(audit_input.shape),
        )
        mask_path, _ = _load_valid_mask_cache(
            audit_path.name, audit_signature, verify_checksum=False
        )
        if mask_path is not None:
            compressed_mask_cache_count += 1
        else:
            missing_compressed_mask_cache_images.append(audit_path.name)
        if _load_valid_assignment_cache(audit_path.name, audit_signature) is not None:
            hungarian_assignment_cache_count += 1
    else:
        missing_compressed_mask_cache_images.append(audit_path.name)

print(f"\n[INFO] Optimizer input audit:")
print(f"   Input images: {len(image_paths)}")
print(f"   Compatible optimizer results: {compatible_result_cache_count}")
print(f"   Compatible compressed 19-pair mask caches: {compressed_mask_cache_count}")
print(f"   Compatible Hungarian assignment caches: {hungarian_assignment_cache_count}")
print(f"   Optimizer BM3D cache: {optimizer_bm3d_cache_count}")
print(f"   Reusable prior BM3D images: {comparison_bm3d_reuse_count}")
print(
    f"   BM3D images requiring generation: "
    f"{len(image_paths) - optimizer_bm3d_cache_count - comparison_bm3d_reuse_count}"
)

if ARGS.reevaluate_gt_cache and missing_compressed_mask_cache_images:
    preview = ', '.join(missing_compressed_mask_cache_images[:10])
    remainder = len(missing_compressed_mask_cache_images) - 10
    if remainder > 0:
        preview += f", ... (+{remainder} more)"
    raise RuntimeError(
        "--reevaluate-gt-cache cannot run because compatible compressed masks "
        f"are missing for {len(missing_compressed_mask_cache_images)} images: {preview}. "
        "Run the optimizer once without --reevaluate-gt-cache."
    )

if ARGS.plan_only:
    print("[INFO] Plan-only mode: SAM2 and BM3D execution were skipped.")
    sys.exit(0)

if ARGS.preprocess_only:
    print(f"\n{'='*80}")
    print("PREPARE BM3D-ONLY OPTIMIZER INPUTS")
    print(f"{'='*80}")
    preprocessing_failures = []
    for index, image_path in enumerate(image_paths, start=1):
        image_name = Path(image_path).name
        print(f"\n[{index}/{len(image_paths)}] {image_name}")
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cv2.imread returned None")
            preprocess_bm3d_only(ensure_even_dimensions(image), image_name)
        except Exception as exc:
            preprocessing_failures.append(
                (image_name, f"{type(exc).__name__}: {exc}")
            )
            print(f"[ERROR] {preprocessing_failures[-1][1]}")
    if preprocessing_failures:
        details = "\n".join(
            f"  - {name}: {reason}" for name, reason in preprocessing_failures
        )
        raise RuntimeError(
            f"BM3D preparation failed for {len(preprocessing_failures)} images:\n{details}"
        )
    print(f"\n[OK] BM3D-only inputs are ready for all {len(image_paths)} images.")
    sys.exit(0)


#%%
# ============================================================================
# LOAD SAM MODEL (only when a missing compressed mask cache may need inference)
# ============================================================================

all_masks_cached = compressed_mask_cache_count == len(image_paths)
if ARGS.reevaluate_gt_cache or (all_masks_cached and not FORCE_REPROCESS):
    sam = None
    device = "cache-only"
    print("\n[MODE] Compatible masks are cached: SAM2 model loading is disabled.")
else:
    print(f"\n{'='*80}")
    print("LOAD SAM MODEL")
    print(f"{'='*80}")

    model_type = "hiera_l"
    sam_checkpoint = os.path.join(parent_dir, "checkpoints", "sam2.1_hiera_large.pt")
    sam_config = get_sam2_config_path("sam2.1_hiera_l.yaml")

    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam2(sam_config, sam_checkpoint, device=device)

    print(f"[OK] SAM loaded: {model_type} on {device}")

#%%
# ============================================================================
# PROCESS ALL IMAGES
# ============================================================================

print(f"\n{'='*80}")
print("PROCESS ALL IMAGES")
print(f"{'='*80}")

print(f"\nFound {len(image_paths)} images")

# Process each image (with cache support)
all_results = []
cached_count = 0
failed_images = []

for idx, image_path in enumerate(image_paths, 1):
    image_name = os.path.basename(image_path)
    name_stem = os.path.splitext(image_name)[0]
    cache_path = os.path.join(CACHE_DIR, f'{name_stem}_cache.json')

    # Check cache first
    if os.path.exists(cache_path) and not FORCE_REPROCESS:
        cached_result = load_image_cache(cache_path)
        if _result_cache_matches_active_config(cached_result, image_path):
            print(f"\n[{idx}/{len(image_paths)}] [CACHE HIT] {image_name}")
            cached_result['particle_region'] = None  # Not available from cache
            cached_result['from_cache'] = True
            all_results.append(cached_result)
            cached_count += 1
            continue
        print(f"\n[{idx}/{len(image_paths)}] [STALE CACHE] {image_name}; recomputing")

    print(f"\n[{idx}/{len(image_paths)}]")
    try:
        result = process_single_image_gt(image_path, sam, device)
    except Exception as exc:
        failed_images.append((image_name, f"{type(exc).__name__}: {exc}"))
        print(f"   [ERROR] Failed: {failed_images[-1][1]}")
        traceback.print_exc()
        continue

    if result and result.get('exception', False):
        failed_images.append((image_name, result.get('reason', 'Unknown processing failure')))
        print(f"   [ERROR] Failed: {failed_images[-1][1]}")
        continue

    if result:
        # Save only verified successful results. Failed images remain retryable.
        cp = save_image_cache(result, CACHE_DIR)
        print(f"   [CACHE] Saved: {os.path.basename(cp)}")
        all_results.append(result)

print(f"\n{'='*80}")
print(f"[OK] Processed {len(all_results)} images ({cached_count} from cache)")
print(f"{'='*80}")

if failed_images:
    print(f"\n[FAIL] {len(failed_images)} images were not processed successfully:")
    for image_name, reason in failed_images:
        print(f"   - {image_name}: {reason}")
    raise RuntimeError(
        "SAM parameter optimization is incomplete. Successful images were cached; "
        "rerun the command to retry only the failed images."
    )

#%%
# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("STATISTICAL ANALYSIS")
print(f"{'='*80}")

# Separate normal and exception cases
normal_results = [r for r in all_results if not r.get('exception', False)]
exception_results = [r for r in all_results if r.get('exception', False)]

print(f"\nNormal cases: {len(normal_results)}")
print(f"Exception cases: {len(exception_results)}")

if len(normal_results) == 0:
    print("\n[WARN] No normal cases found! Cannot generate statistics.")
    sys.exit(1)

# Prepare data - Include ALL images (both normal and exception)
data_rows_all = []

# Add normal results
for r in normal_results:
    data_rows_all.append({
        'Image': r['image_name'],
        'Exception': False,
        'Optimal_Stage': r['optimal_stage'] + 1,
        'pred_iou_thresh': r['optimal_pred_iou'],
        'stability_score_thresh': r['optimal_stability'],
        'Valid_Masks': r['optimal_valid'],
        'Noise_Masks': r['optimal_noise'],
        'Noise_Ratio': r['optimal_ratio'],
        'Reason': r['reason'],
    })

# Add exception results
for r in exception_results:
    data_rows_all.append({
        'Image': r['image_name'],
        'Exception': True,
        'Optimal_Stage': None,
        'pred_iou_thresh': None,
        'stability_score_thresh': None,
        'Valid_Masks': None,
        'Noise_Masks': None,
        'Noise_Ratio': None,
        'Reason': r['reason'],
    })

df_all_results = pd.DataFrame(data_rows_all)

# Prepare normal results only for statistics
data_rows_normal = []
for r in normal_results:
    data_rows_normal.append({
        'Image': r['image_name'],
        'Optimal_Stage': r['optimal_stage'] + 1,
        'pred_iou_thresh': r['optimal_pred_iou'],
        'stability_score_thresh': r['optimal_stability'],
        'Valid_Masks': r['optimal_valid'],
        'Noise_Masks': r['optimal_noise'],
        'Noise_Ratio': r['optimal_ratio'],
        'Reason': r['reason'],
    })

df_normal_results = pd.DataFrame(data_rows_normal)

# Statistics (from normal results only)
pred_iou_values = df_normal_results['pred_iou_thresh'].values
stability_values = df_normal_results['stability_score_thresh'].values

stats_data = {
    'Metric': ['pred_iou_thresh', 'stability_score_thresh'],
    'Mode': [
        pd.Series(pred_iou_values).mode()[0] if len(pd.Series(pred_iou_values).mode()) > 0 else np.nan,
        pd.Series(stability_values).mode()[0] if len(pd.Series(stability_values).mode()) > 0 else np.nan,
    ],
    'Mean': [pred_iou_values.mean(), stability_values.mean()],
    'Median': [np.median(pred_iou_values), np.median(stability_values)],
    'Std': [pred_iou_values.std(), stability_values.std()],
    'Min': [pred_iou_values.min(), stability_values.min()],
    'Max': [pred_iou_values.max(), stability_values.max()],
}

df_stats = pd.DataFrame(stats_data)

print(f"\n[ICON] Statistics (normal cases only):")
print(df_stats.to_string(index=False))

# Rebuild the workbook from persisted JSON so cache-only and fresh-run outputs match.
summary_tables = _build_cache_summary_tables(CACHE_DIR)
df_all_results = summary_tables['all_results']
df_normal_results = summary_tables['normal_results']
df_stats = summary_tables['statistics']
df_pair_metrics = summary_tables['pair_metrics']
df_growth_rates = summary_tables['growth_rates']
df_default_pair_frequency = summary_tables['default_pair_frequency']
df_manuscript_values = summary_tables['manuscript_values']
df_metadata = summary_tables['metadata']

excel_path = os.path.join(OUTPUT_DIR, 'optimization_results.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_all_results.to_excel(writer, sheet_name='All Images', index=False)
    df_normal_results.to_excel(writer, sheet_name='Normal Results', index=False)
    df_stats.to_excel(writer, sheet_name='Statistics', index=False)
    df_pair_metrics.to_excel(writer, sheet_name='Pair Metrics', index=False)
    df_growth_rates.to_excel(writer, sheet_name='Growth Rates', index=False)
    df_default_pair_frequency.to_excel(
        writer, sheet_name='Default Pair Frequency', index=False
    )
    df_manuscript_values.to_excel(writer, sheet_name='Manuscript Values', index=False)
    df_metadata.to_excel(writer, sheet_name='Metadata', index=False)

print(f"\n[OK] Saved: {excel_path}")
print(f"   - Sheet 1: All Images ({len(df_all_results)} total)")
print(f"   - Sheet 2: Normal Results ({len(df_normal_results)} images)")
print(f"   - Sheet 3: Statistics (from normal results)")
print(f"   - Sheet 4: Pair Metrics ({len(df_pair_metrics)} rows)")
print(f"   - Sheet 5: Growth Rates ({len(df_growth_rates)} rows)")
print(
    f"   - Sheet 6: Default Pair Frequency "
    f"({len(df_default_pair_frequency)} rows)"
)
print(f"   - Sheet 7: Manuscript Values ({len(df_manuscript_values)} rows)")
print(f"   - Sheet 8: Metadata ({len(df_metadata)} rows)")

print(f"\n[ICON] Generating publication 2-panel summary figure...")
try:
    summary_info = generate_sam_opt_ab_figure(excel_path, OUTPUT_DIR, dpi=600)
    print(f"[OK] Saved: {os.path.join(OUTPUT_DIR, 'sam_opt_ab.pdf')}")
    print(f"[OK] Saved: {os.path.join(OUTPUT_DIR, 'sam_opt_ab.png')}")
    print(f"[OK] Saved: {os.path.join(OUTPUT_DIR, 'sam_opt_ab_notext.pdf')}")
    print(f"[OK] Saved: {os.path.join(OUTPUT_DIR, 'sam_opt_ab_notext.png')}")
    print(
        f"[INFO] Summary figure: total={summary_info['total_images']}, "
        f"valid={summary_info['valid_images']}, "
        f"mode_pair={summary_info['mode_pair']}, "
        f"growth_from_excel={summary_info['growth_from_excel']}"
    )
except Exception as fig_err:
    print(f"[WARN] Failed to generate sam_opt_ab figure: {fig_err}")

#%%
# ============================================================================
# 3D GROWTH RATE VISUALIZATION
# ============================================================================

print(f"\n{'='*80}")
print("3D GROWTH RATE VISUALIZATION")
print(f"{'='*80}")

# Visualize all results that have stage data (normal cases + 'no explosion' exceptions)
results_to_visualize = (
    []
    if ARGS.reevaluate_gt_cache
    else [r for r in all_results if r.get('stage_results')]
)

if ARGS.reevaluate_gt_cache:
    print("\n[INFO] Cache-only tau reevaluation skips per-image 3D figures.")

if len(results_to_visualize) > 0:
    from matplotlib.lines import Line2D

    print(f"\n[ICON] Generating 3D growth rate plots for {len(results_to_visualize)} images...")

    for img_idx, sample_result in enumerate(results_to_visualize, 1):
        print(f"\n[{img_idx}/{len(results_to_visualize)}] Processing: {sample_result['image_name']}")

        stage_results = sample_result['stage_results']
        n_stages = len(stage_results)
        opt_pred = sample_result['optimal_pred_iou']
        opt_stab = sample_result['optimal_stability']

        # Build per-combo growth records for all tested stages
        point_records = []
        growth_lookup = {}

        # Stage 1 baseline (growth undefined -> 0 for plotting)
        for combo in stage_results[0]:
            rec = {
                'pred_iou': combo['pred_iou'],
                'stability': combo['stability'],
                'growth_rate_raw': 0.0,
                'growth_rate_plot': 0.0,
                'is_capped': False,
                'is_optimal': False,
                'stage_idx': 0,
                'combo_idx': combo['combo_idx'],
                'combo': combo,
            }
            point_records.append(rec)
            growth_lookup[(0, combo['combo_idx'])] = rec

        for i in range(1, n_stages):
            prev_stage = stage_results[i-1]
            curr_stage = stage_results[i]
            prev_best = min(prev_stage, key=_combo_quality_key)

            for combo in curr_stage:
                valid_growth = combo['valid_masks'] - prev_best['valid_masks']
                noise_growth = combo['noise_masks'] - prev_best['noise_masks']
                gr_raw = calculate_growth_rate(valid_growth, noise_growth)
                gr_plot, gr_capped = cap_growth_rate_for_plot(gr_raw, cap_value=1.0)

                rec = {
                    'pred_iou': combo['pred_iou'],
                    'stability': combo['stability'],
                    'growth_rate_raw': gr_raw,
                    'growth_rate_plot': gr_plot,
                    'is_capped': gr_capped,
                    'is_optimal': False,
                    'stage_idx': i,
                    'combo_idx': combo['combo_idx'],
                    'combo': combo,
                }
                point_records.append(rec)
                growth_lookup[(i, combo['combo_idx'])] = rec

        if len(point_records) == 0:
            print(f"   [WARN] No tested points, skipping 3D plot")
            continue

        # Build deterministic stage-indexed records for split/merge transitions.
        # IMPORTANT: do not create averaged (anchor) points to avoid misinterpretation.
        stage_point_records = {stage_idx: [] for stage_idx in range(n_stages)}
        for rec in point_records:
            stage_point_records[rec['stage_idx']].append(rec)
        for stage_idx in stage_point_records:
            stage_point_records[stage_idx] = sorted(
                stage_point_records[stage_idx],
                key=lambda r: (-r['pred_iou'], r['stability'])
            )

        # Stage transitions: single->double (split), double->single (merge), generic fallback otherwise
        transition_segments = []
        for stage_idx in range(n_stages - 1):
            src_records = stage_point_records.get(stage_idx, [])
            dst_records = stage_point_records.get(stage_idx + 1, [])
            if len(src_records) == 0 or len(dst_records) == 0:
                continue

            if len(src_records) == 1:
                for dst_rec in dst_records:
                    transition_segments.append((src_records[0], dst_rec))
            elif len(dst_records) == 1:
                for src_rec in src_records:
                    transition_segments.append((src_rec, dst_records[0]))
            elif len(src_records) == len(dst_records):
                for src_rec, dst_rec in zip(src_records, dst_records):
                    transition_segments.append((src_rec, dst_rec))
            else:
                for src_rec in src_records:
                    for dst_rec in dst_records:
                        transition_segments.append((src_rec, dst_rec))

        # Stage-best records (actual tested points) for stage-level diagnostics/markers
        stage_best_records = []
        for stage_idx, stage_combos in enumerate(stage_results):
            best_combo = min(stage_combos, key=_combo_quality_key)
            best_rec = growth_lookup.get((stage_idx, best_combo['combo_idx']))
            if best_rec is not None:
                stage_best_records.append(best_rec)

        # First crossing: first stage-best point at or above the threshold.
        first_crossing_rec = None
        for rec in stage_best_records[1:]:
            raw = rec['growth_rate_raw']
            if (not np.isfinite(raw)) or (raw >= CONSERVATIVE_GROWTH_THRESHOLD):
                first_crossing_rec = rec
                break

        # Optional diagnostic: if crossing happens at a single-point stage, report
        # how the previous two candidates are compared by noise/valid ratio.
        crossing_prev_choice_rec = None
        crossing_prev_candidates = []
        crossing_prev_choice_text = ""
        if first_crossing_rec is not None:
            crossing_stage_idx = int(first_crossing_rec['stage_idx'])
            prev_stage_idx = crossing_stage_idx - 1
            if (
                crossing_stage_idx > 0
                and len(stage_results[crossing_stage_idx]) == 1
                and len(stage_results[prev_stage_idx]) >= 2
            ):
                prev_candidates = []
                for combo in stage_results[prev_stage_idx]:
                    ratio = total_noise_valid_ratio(combo)
                    rec = growth_lookup.get((prev_stage_idx, combo['combo_idx']))
                    if rec is not None:
                        prev_candidates.append((ratio, rec))

                if len(prev_candidates) >= 2:
                    prev_candidates = sorted(prev_candidates, key=lambda x: x[0])[:2]
                    (best_ratio, best_rec), (second_ratio, second_rec) = prev_candidates[0], prev_candidates[1]
                    crossing_prev_choice_rec = best_rec
                    ratio_min = min(best_ratio, second_ratio)
                    ratio_max = max(best_ratio, second_ratio)
                    ratio_span = max(ratio_max - ratio_min, 1e-12)
                    for ratio, rec in prev_candidates:
                        # Lower noise/valid -> more transparent, Higher noise/valid -> less transparent
                        ratio_norm = (ratio - ratio_min) / ratio_span if ratio_span > 0 else 0.5
                        alpha_val = 0.28 + 0.62 * ratio_norm
                        alpha_val = min(0.94, max(0.25, alpha_val))
                        crossing_prev_candidates.append({
                            'ratio': ratio,
                            'rec': rec,
                            'alpha_val': alpha_val,
                            'is_selected': (rec is best_rec),
                        })
                    crossing_prev_choice_text = (
                        f"Crossing fallback: compare S{prev_stage_idx + 1} pair "
                        f"({best_rec['pred_iou']:.2f}, {best_rec['stability']:.2f})={best_ratio:.3f} vs "
                        f"({second_rec['pred_iou']:.2f}, {second_rec['stability']:.2f})={second_ratio:.3f}; "
                        f"select lower ratio."
                    )

        # Selected optimal: actual algorithm-selected tested point
        optimal_stage_idx = int(sample_result.get('optimal_stage', 0))
        optimal_combo_idx = int(sample_result.get('optimal_combo', 0))
        selected_optimal_stage_rec = None
        if 0 <= optimal_stage_idx < n_stages:
            stage_combos = stage_results[optimal_stage_idx]
            if 0 <= optimal_combo_idx < len(stage_combos):
                opt_combo = stage_combos[optimal_combo_idx]
                selected_optimal_stage_rec = growth_lookup.get(
                    (optimal_stage_idx, opt_combo['combo_idx']),
                    None
                )
        if selected_optimal_stage_rec is None and len(point_records) > 0:
            selected_optimal_stage_rec = min(
                point_records,
                key=lambda rec: (
                    abs(rec['pred_iou'] - opt_pred)
                    + abs(rec['stability'] - opt_stab)
                    + abs(rec['stage_idx'] - optimal_stage_idx) * 0.01
                )
            )

        if first_crossing_rec is not None:
            print(
                f"   [INFO] First crossing: Stage {first_crossing_rec['stage_idx'] + 1} "
                f"({first_crossing_rec['pred_iou']:.2f}, {first_crossing_rec['stability']:.2f})"
            )
            if crossing_prev_choice_text:
                print(f"   [INFO] {crossing_prev_choice_text}")
        else:
            print(
                "   [INFO] First crossing: none "
                f"(growth rate never reached {CONSERVATIVE_GROWTH_THRESHOLD:g})"
            )

        if selected_optimal_stage_rec is not None:
            print(
                f"   [INFO] Selected optimal marker: Stage {selected_optimal_stage_rec['stage_idx'] + 1} "
                f"({selected_optimal_stage_rec['pred_iou']:.2f}, {selected_optimal_stage_rec['stability']:.2f})"
            )

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.computed_zorder = False

        # Palette requested by user (high visibility)
        color_red = "#BF4E58"
        color_blue = "#4971A6"
        color_green = "#56A662"
        color_orange = "#D97652"
        color_offwhite = "#F2F2F2"
        color_tested = color_offwhite
        color_path = color_green
        color_threshold = color_orange
        color_first_cross = color_blue
        color_fallback_tag = color_green
        color_selected_opt_tag = color_red
        color_text_main = "#3D4A5E"
        color_text_minor = "#5A6A82"

        # 1) Threshold plane (muted, low-alpha)
        xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 20), np.linspace(0.0, 1.0, 20))
        zz = np.ones_like(xx) * CONSERVATIVE_GROWTH_THRESHOLD
        ax.plot_surface(
            xx, yy, zz,
            color=color_threshold,
            alpha=0.14,
            linewidth=0,
            antialiased=True,
            zorder=1
        )

        # 2) Show all test candidates (not circles).
        # Every candidate is available because all 19 pairs are cached.
        tested_lookup = {
            (round(rec['pred_iou'], 2), round(rec['stability'], 2)): rec
            for rec in point_records
        }
        candidate_points = []
        for stage_combos in PARAM_STAGES:
            for pred_iou, stability in stage_combos:
                key = (round(pred_iou, 2), round(stability, 2))
                rec = tested_lookup.get(key)
                candidate_points.append({
                    'pred_iou': pred_iou,
                    'stability': stability,
                    'growth_rate_plot': rec['growth_rate_plot'] if rec is not None else 0.0,
                    'tested': rec is not None,
                })

        # Draw all 19 candidates as subtle hollow circles (candidate-only cue).
        # Keep exact stage-grid intersections (no z lift).
        all_candidate_x = [p['pred_iou'] for p in candidate_points]
        all_candidate_y = [p['stability'] for p in candidate_points]
        all_candidate_z = [
            (p['growth_rate_plot'] if p['tested'] else 0.0)
            for p in candidate_points
        ]
        ax.scatter(
            all_candidate_x,
            all_candidate_y,
            all_candidate_z,
            facecolors='none',
            edgecolors=to_rgba(color_blue, 0.45),
            s=52,
            marker='o',
            linewidths=0.95,
            depthshade=False,
            zorder=10
        )

        # 3) Stage transitions on tested coordinates (split/merge arrows)
        path_color = "#000000"
        for src_rec, dst_rec in transition_segments:
            sx, sy, sz = src_rec['pred_iou'], src_rec['stability'], src_rec['growth_rate_plot']
            ex, ey, ez = dst_rec['pred_iou'], dst_rec['stability'], dst_rec['growth_rate_plot']
            dx, dy, dz = ex - sx, ey - sy, ez - sz

            ax.plot([sx, ex], [sy, ey], [sz, ez], color=path_color, linewidth=2.1, alpha=0.95, zorder=8)
            ax.quiver(
                sx, sy, sz,
                dx, dy, dz,
                color=path_color,
                linewidth=1.2,
                arrow_length_ratio=0.14,
                normalize=False,
                alpha=0.95
            )

        # 4) Decision markers
        if len(crossing_prev_candidates) >= 2:
            for cand in crossing_prev_candidates:
                rec = cand['rec']
                ax.scatter(
                    [rec['pred_iou']],
                    [rec['stability']],
                    [rec['growth_rate_plot']],
                    c=[to_rgba(color_blue, cand['alpha_val'])],
                    s=250,
                    marker='o',
                    edgecolors=to_rgba(color_blue, 0.95),
                    linewidth=1.3,
                    depthshade=False,
                    zorder=11
                )

        if first_crossing_rec is not None:
            ax.scatter(
                [first_crossing_rec['pred_iou']],
                [first_crossing_rec['stability']],
                [first_crossing_rec['growth_rate_plot']],
                c=color_first_cross, s=170, marker='D',
                edgecolors=to_rgba(color_green, 0.98), linewidth=1.0, depthshade=False, zorder=12
            )
            ax.text(
                first_crossing_rec['pred_iou'],
                first_crossing_rec['stability'],
                min(1.03, first_crossing_rec['growth_rate_plot'] + 0.06),
                "First crossing",
                fontsize=8, color=color_text_main
            )

        if selected_optimal_stage_rec is not None:
            ox = selected_optimal_stage_rec['pred_iou']
            oy = selected_optimal_stage_rec['stability']
            oz = selected_optimal_stage_rec['growth_rate_plot']
            opt_tag_x = min(0.98, ox + 0.10)
            opt_tag_y = max(0.02, oy - 0.04)
            opt_tag_z = min(1.03, oz + 0.06)

            # Offset selected-optimal marker to avoid overlap with candidate circles.
            ax.plot(
                [ox, opt_tag_x], [oy, opt_tag_y], [oz, opt_tag_z],
                color=color_selected_opt_tag, linewidth=1.3, alpha=0.98, zorder=14
            )
            ax.scatter(
                [opt_tag_x], [opt_tag_y], [opt_tag_z],
                c=color_selected_opt_tag, s=240, marker='*',
                edgecolors=to_rgba(color_blue, 0.98), linewidth=1.1, depthshade=False, zorder=15
            )
            ax.text(
                opt_tag_x,
                opt_tag_y,
                min(1.03, opt_tag_z + 0.03),
                "Selected optimal",
                fontsize=8, color=color_text_main
            )

        # Axis ticks at 0.15 intervals
        tick_values = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0, 1.05)
        ax.tick_params(axis='x', labelsize=14, colors=color_text_main)
        ax.tick_params(axis='y', labelsize=14, colors=color_text_main)
        ax.tick_params(axis='z', labelsize=12, colors=color_text_main)

        ax.set_xlabel('pred_iou_thresh', fontsize=12, fontweight='bold', color=color_text_main)
        ax.set_ylabel('stability_score_thresh', fontsize=12, fontweight='bold', color=color_text_main)
        ax.set_zlabel('Growth Rate (noise/valid)', fontsize=12, fontweight='bold', color=color_text_main)
        ax.set_title(f"{sample_result['image_name']}", fontsize=13, fontweight='bold', color=color_text_main)
        ax.text2D(
            0.03, 0.92,
            f"Stopping threshold (growth rate = {CONSERVATIVE_GROWTH_THRESHOLD:g})",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8, color=color_text_main
        )
        ax.text2D(
            0.03, 0.86,
            "Stage progression (strict \u2192 relaxed)",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8, color=color_text_minor
        )
        ax.text2D(
            0.03, 0.80,
            "Split/merge transitions use only tested stage points",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8, color=color_text_minor
        )
        if crossing_prev_choice_text:
            ax.text2D(
                0.03, 0.74,
                crossing_prev_choice_text,
                transform=ax.transAxes, ha='left', va='top',
                fontsize=7.5, color=color_text_minor
            )
        ax.text(0.95, 0.95, 0.02, "S1 (0.95, 0.95)", fontsize=8, color=color_text_main)
        ax.text(0.05, 0.05, 0.02, "S13 (0.05, 0.05)", fontsize=8, color=color_text_main)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color=to_rgba(color_blue, 0.45), markerfacecolor='none', markersize=7,
                   markeredgewidth=1.0, label='Test candidates (all)'),
            Line2D([0], [0], color="#000000", linewidth=2.4, label='Stage transition arrows (split/merge)'),
            Patch(facecolor=color_threshold, alpha=0.14, edgecolor='none', label='Stopping threshold plane'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_blue, markersize=8,
                   markeredgecolor=color_blue, markeredgewidth=0.9, label='Fallback pair (higher ratio = denser fill)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor=color_first_cross, markersize=7,
                   markeredgecolor=color_green, markeredgewidth=0.8, label='First crossing'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=color_selected_opt_tag, markersize=11,
                   markeredgecolor=color_blue, markeredgewidth=1.0, label='Selected optimal'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.92)

        plt.tight_layout()
        safe_name = safe_filename(sample_result['image_name'])
        safe_base, safe_ext = os.path.splitext(safe_name)
        if safe_ext == '':
            safe_ext = '.png'

        fig_path = os.path.join(OUTPUT_DIR, f'3d_growth_rate_{img_idx:02d}_{safe_base}{safe_ext}')
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"   [OK] Saved: {os.path.basename(fig_path)}")

        fig_notext_path = os.path.join(OUTPUT_DIR, f'3d_growth_rate_{img_idx:02d}_{safe_base}_notext{safe_ext}')
        save_notext_figure(fig, fig_notext_path, dpi=200, remove_tick_labels=True)
        print(f"   [OK] Saved: {os.path.basename(fig_notext_path)}")
        plt.close()

    print(f"\n[OK] All 3D plots saved to: {OUTPUT_DIR}")
else:
    print("\n[WARN] No results to visualize.")

#%%
# ============================================================================
# COMPREHENSIVE IMAGE VISUALIZATION
# ============================================================================

print(f"\n{'='*80}")
print("COMPREHENSIVE IMAGE VISUALIZATION")
print(f"{'='*80}")

# Only visualize results with full mask data (not from cache)
results_to_visualize_full = [r for r in all_results
                             if r.get('stage_results')
                             and r.get('particle_region') is not None
                             and not r.get('from_cache', False)]

if len(results_to_visualize_full) > 0:
    print(f"\n[ICON] Generating comprehensive visualizations for {len(results_to_visualize_full)} images...")
    print(f"   (Skipping {len(results_to_visualize) - len(results_to_visualize_full)} cached images - already have saved PNGs)")

    for img_idx, sample_result in enumerate(results_to_visualize_full, 1):
        print(f"\n[{img_idx}/{len(results_to_visualize_full)}] Processing: {sample_result['image_name']}")

        stage_results = sample_result['stage_results']
        n_stages = len(stage_results)
        particle_region = sample_result['particle_region']

        # Calculate total number of combinations tested
        total_combos = sum(len(stage_data) for stage_data in stage_results)

        # Create large figure - each combo gets 1 row with 4 columns
        fig = plt.figure(figsize=(24, 6 * total_combos))

        optimal_stage = sample_result['optimal_stage']
        optimal_combo = sample_result['optimal_combo']

        combo_counter = 0  # Global counter for subplot positioning

        for stage_idx, stage_data in enumerate(stage_results):
            for combo_idx, combo_result in enumerate(stage_data):
                is_optimal = (stage_idx == optimal_stage and combo_idx == optimal_combo)

                # Base column position for this combo
                base_col = combo_counter * 4

                # 1. All masks
                ax1 = plt.subplot(total_combos, 4, base_col + 1)
                ax1.imshow(cv2.cvtColor(particle_region, cv2.COLOR_BGR2RGB))

                # Draw all masks with valid/noise color separation
                if len(combo_result['all_masks']) > 0:
                    img_overlay = np.ones((*particle_region.shape[:2], 4))
                    img_overlay[:, :, 3] = 0
                    for mask in combo_result.get('valid_only', []):
                        m = mask['segmentation']
                        img_overlay[m] = [0.15, 0.85, 0.25, 0.42]  # Green (valid)
                    for mask in combo_result.get('noise_only', []):
                        m = mask['segmentation']
                        img_overlay[m] = [0.90, 0.20, 0.20, 0.42]  # Red (noise)
                    ax1.imshow(img_overlay)

                title = f"Stage {stage_idx+1} | All ({combo_result['total_masks']})\n"
                title += f"({combo_result['pred_iou']:.2f}, {combo_result['stability']:.2f})"
                if is_optimal:
                    title = "[STAR] " + title + " [STAR]"
                    ax1.patch.set_edgecolor('gold')
                    ax1.patch.set_linewidth(5)
                ax1.set_title(title, fontsize=10, fontweight='bold' if is_optimal else 'normal')
                ax1.axis('off')

                # 2. Valid masks
                ax2 = plt.subplot(total_combos, 4, base_col + 2)
                ax2.imshow(cv2.cvtColor(particle_region, cv2.COLOR_BGR2RGB))
                if len(combo_result['valid_only']) > 0:
                    img_overlay = np.ones((*particle_region.shape[:2], 4))
                    img_overlay[:, :, 3] = 0
                    for mask in combo_result['valid_only']:
                        m = mask['segmentation']
                        img_overlay[m] = [0, 1, 0, 0.4]  # Green
                    ax2.imshow(img_overlay)
                ax2.set_title(f"Valid ({combo_result['valid_masks']})", fontsize=10)
                ax2.axis('off')

                # 3. Noise masks
                ax3 = plt.subplot(total_combos, 4, base_col + 3)
                ax3.imshow(cv2.cvtColor(particle_region, cv2.COLOR_BGR2RGB))
                if len(combo_result['noise_only']) > 0:
                    img_overlay = np.ones((*particle_region.shape[:2], 4))
                    img_overlay[:, :, 3] = 0
                    for mask in combo_result['noise_only']:
                        m = mask['segmentation']
                        img_overlay[m] = [1, 0, 0, 0.4]  # Red
                    ax3.imshow(img_overlay)
                ax3.set_title(f"Noise ({combo_result['noise_masks']})", fontsize=10)
                ax3.axis('off')

                # 4. Bar chart
                ax4 = plt.subplot(total_combos, 4, base_col + 4)
                bars = ax4.bar(['Valid', 'Noise'],
                              [combo_result['valid_masks'], combo_result['noise_masks']],
                              color=['green', 'red'], alpha=0.7)
                ax4.set_ylabel('Count')
                ax4.set_title('Statistics', fontsize=10)
                ax4.grid(axis='y', alpha=0.3)

                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

                combo_counter += 1  # Increment counter after processing this combo

        plt.suptitle(f'SAM Parameter Optimization - Per Stage Results\n{sample_result["image_name"]}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save with safe filename
        safe_name = safe_filename(sample_result['image_name'])
        safe_base, safe_ext = os.path.splitext(safe_name)
        if safe_ext == '':
            safe_ext = '.png'

        fig_path = os.path.join(OUTPUT_DIR, f'comprehensive_{img_idx:02d}_{safe_base}{safe_ext}')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {os.path.basename(fig_path)}")

        fig_notext_path = os.path.join(OUTPUT_DIR, f'comprehensive_{img_idx:02d}_{safe_base}_notext{safe_ext}')
        save_notext_figure(fig, fig_notext_path, dpi=150)
        print(f"   [OK] Saved: {os.path.basename(fig_notext_path)}")
        plt.close()

    print(f"\n[OK] All comprehensive plots saved to: {OUTPUT_DIR}")
else:
    print("\n[INFO] No new comprehensive plots needed (all from cache or no results).")

#%%
# ============================================================================
# OPTIMAL PARAMETER FREQUENCY VISUALIZATION
# ============================================================================

print(f"\n{'='*80}")
print("OPTIMAL PARAMETER FREQUENCY")
print(f"{'='*80}")

if len(normal_results) > 0:
    # Count frequency of each (pred_iou, stability) pair
    param_pairs = []
    for r in normal_results:
        pair = (round(r['optimal_pred_iou'], 2), round(r['optimal_stability'], 2))
        param_pairs.append(pair)

    pair_counts = Counter(param_pairs)
    total_images = len(all_results) if len(all_results) > 0 else len(normal_results)

    # Build heatmap grid
    pred_iou_levels = sorted({p[0] for p in param_pairs})
    stability_levels = sorted({p[1] for p in param_pairs})
    pred_iou_to_idx = {v: i for i, v in enumerate(pred_iou_levels)}
    stability_to_idx = {v: i for i, v in enumerate(stability_levels)}

    heatmap_ratio = np.zeros((len(stability_levels), len(pred_iou_levels)), dtype=float)
    for (pred_iou, stability), count in pair_counts.items():
        yi = stability_to_idx[stability]
        xi = pred_iou_to_idx[pred_iou]
        heatmap_ratio[yi, xi] = count / total_images

    # Report sorted by frequency
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\nOptimal Parameter Frequency Ratio:")
    for (pred_iou, stability), count in sorted_pairs:
        ratio = count / total_images
        print(f"   ({pred_iou:.2f}, {stability:.2f}): {ratio:.4f} ({count}/{total_images})")

    vmax = float(heatmap_ratio.max()) if heatmap_ratio.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0

    # Heatmap (with text)
    fig, ax = plt.subplots(figsize=(max(7, len(pred_iou_levels) * 0.9), max(6, len(stability_levels) * 0.75)))
    im = ax.imshow(heatmap_ratio, cmap=FREQ_CMAP, origin='lower', aspect='auto', vmin=0.0, vmax=vmax)

    ax.set_xlabel('pred_iou_thresh', fontsize=12, fontweight='bold')
    ax.set_ylabel('stability_score_thresh', fontsize=12, fontweight='bold')
    ax.set_title(f'Optimal Parameter Frequency Ratio (n={total_images} images)', fontsize=14, fontweight='bold')

    ax.set_xticks(np.arange(len(pred_iou_levels)))
    ax.set_xticklabels([f'{v:.2f}' for v in pred_iou_levels], fontsize=10)
    ax.set_yticks(np.arange(len(stability_levels)))
    ax.set_yticklabels([f'{v:.2f}' for v in stability_levels], fontsize=10)

    # Cell annotations (ratio)
    text_threshold = vmax * 0.5
    for yi in range(len(stability_levels)):
        for xi in range(len(pred_iou_levels)):
            value = heatmap_ratio[yi, xi]
            if value <= 0:
                continue
            txt_color = 'white' if value >= text_threshold else 'black'
            ax.text(xi, yi, f'{value:.3f}', ha='center', va='center',
                    color=txt_color, fontsize=9, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Frequency / Total Images', fontsize=11)

    plt.tight_layout()
    freq_path = os.path.join(OUTPUT_DIR, 'optimal_param_frequency.png')
    fig.savefig(freq_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] Saved: {freq_path}")

    # Heatmap (no-text version)
    fig_nt, ax_nt = plt.subplots(figsize=(max(7, len(pred_iou_levels) * 0.9), max(6, len(stability_levels) * 0.75)))
    ax_nt.imshow(heatmap_ratio, cmap=FREQ_CMAP, origin='lower', aspect='auto', vmin=0.0, vmax=vmax)
    ax_nt.set_xticks(np.arange(len(pred_iou_levels)))
    ax_nt.set_xticklabels([f'{v:.2f}' for v in pred_iou_levels], fontsize=10)
    ax_nt.set_yticks(np.arange(len(stability_levels)))
    ax_nt.set_yticklabels([f'{v:.2f}' for v in stability_levels], fontsize=10)
    ax_nt.set_xlabel('')
    ax_nt.set_ylabel('')
    ax_nt.set_title('')
    for spine in ax_nt.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    freq_notext_path = os.path.join(OUTPUT_DIR, 'optimal_param_frequency_notext.png')
    fig_nt.savefig(freq_notext_path, dpi=300, bbox_inches='tight')
    plt.close(fig_nt)
    print(f"[OK] Saved: {freq_notext_path}")
else:
    print("\n[WARN] No normal results for frequency heatmap.")

#%%
# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*80}")

print(f"\n[ICON] Summary:")
print(f"   Total images: {len(all_results)}")
print(f"   Normal cases: {len(normal_results)}")
print(f"   Exception cases: {len(exception_results)}")

if len(normal_results) > 0:
    mode_pred = df_stats[df_stats['Metric'] == 'pred_iou_thresh']['Mode'].values[0]
    mode_stab = df_stats[df_stats['Metric'] == 'stability_score_thresh']['Mode'].values[0]

    print(f"\n[ICON] Recommended Parameters (Mode):")
    print(f"   pred_iou_thresh: {mode_pred:.2f}")
    print(f"   stability_score_thresh: {mode_stab:.2f}")

print(f"\n[ICON] Output files:")
print(f"   {excel_path}")
print(f"   {OUTPUT_DIR}/optimal_param_frequency.png")
print(f"   {OUTPUT_DIR}/optimal_param_frequency_notext.png")
print(f"   {OUTPUT_DIR}/sam_opt_ab.pdf")
print(f"   {OUTPUT_DIR}/sam_opt_ab.png")
print(f"   {OUTPUT_DIR}/sam_opt_ab_notext.pdf")
print(f"   {OUTPUT_DIR}/sam_opt_ab_notext.png")
print(f"   {OUTPUT_DIR}/3d_growth_rate_*.png")
print(f"   {OUTPUT_DIR}/3d_growth_rate_*_notext.png")
print(f"   {OUTPUT_DIR}/comprehensive_*.png")
print(f"   {OUTPUT_DIR}/comprehensive_*_notext.png")
print(f"   {CACHE_DIR}/ ({len(all_results)} cache files)")
print(f"   {PARAMETER_MASK_CACHE_DIR}/ (lossless 19-pair mask archives)")
print(f"   {GT_ASSIGNMENT_CACHE_DIR}/ (threshold-independent IoU assignments)")
print(f"   {OUTPUT_DIR}/optimization_split_manifest.csv")

print(f"\n{'='*80}")

# %%
