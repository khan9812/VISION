#%%
# -*- coding: utf-8 -*-
"""
SAM Parameter Optimizer - Single/Multi Image
============================================
Runs the SAM stage search for one or more images and saves stage-by-stage outputs.

Per stage/combo outputs:
  - *_all.png          : valid + noise overlay + compact chart
  - *_all_notext.png   : same as above, no text
  - *_valid.png : valid-only overlay
  - *_noise.png : noise-only overlay
  - *_valid_notext.png : valid-only, no text
  - *_noise_notext.png : noise-only, no text

Per image outputs:
  - 3d_growth_rate.png
  - 3d_growth_rate_notext.png
  - 3d_growth_rate_spiky.png
  - 3d_growth_rate_spiky_notext.png

Usage examples:
  python code/sam_param_single_image.py --image a8851978de.png
  python code/sam_param_single_image.py --image C:\\data\\sample.png
  python code/sam_param_single_image.py --images a.png b.png c.png
  python code/sam_param_single_image.py --all-images --dataset-dir Dataset_size
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from modules.enhancement import apply_bm3d_denoising, check_bm3d_available
from modules.project_paths import get_optimization_dir


PARAM_STAGES = [
    [(0.95, 0.95)],
    [(0.95, 0.80), (0.80, 0.95)],
    [(0.80, 0.80)],
    [(0.80, 0.65), (0.65, 0.80)],
    [(0.65, 0.65)],
    [(0.65, 0.50), (0.50, 0.65)],
    [(0.50, 0.50)],
    [(0.50, 0.35), (0.35, 0.50)],
    [(0.35, 0.35)],
    [(0.35, 0.20), (0.20, 0.35)],
    [(0.20, 0.20)],
    [(0.20, 0.05), (0.05, 0.20)],
    [(0.05, 0.05)],
]

THRESHOLD_RATIO = 0.5
CONSERVATIVE_GROWTH_THRESHOLD = 0.5

# Palette requested earlier
COLOR_RED = "#BF5065"
COLOR_BLUE = "#4B7BA6"
COLOR_GREEN = "#58A65D"
COLOR_ORANGE = "#D96D55"


def get_sam2_config_path(config_name):
    """Get absolute path to SAM2 config from installed package."""
    import sam2

    sam2_path = os.path.dirname(sam2.__file__)
    return os.path.join(sam2_path, "configs", "sam2.1", config_name)


def safe_name(name):
    """Keep filenames stable across platforms."""
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return f"{stem}{ext}"


def resolve_image_path(image_arg, dataset_dir):
    """Resolve image path from absolute/relative path or dataset filename."""
    as_path = Path(image_arg)
    if as_path.exists():
        return str(as_path.resolve())

    candidate = Path(dataset_dir) / image_arg
    if candidate.exists():
        return str(candidate.resolve())

    if candidate.suffix == "":
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            ext_candidate = candidate.with_suffix(ext)
            if ext_candidate.exists():
                return str(ext_candidate.resolve())

    raise FileNotFoundError(
        f"Image not found: {image_arg}\n"
        f"Checked direct path and dataset dir: {dataset_dir}"
    )


def list_dataset_images(dataset_dir):
    """List all supported image files under dataset_dir (non-recursive)."""
    ds = Path(dataset_dir)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(ds.glob(ext))
        paths.extend(ds.glob(ext.upper()))
    # stable ordering + remove duplicates
    uniq = []
    seen = set()
    for p in sorted(paths):
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def resolve_target_images(image, images, all_images, dataset_dir, max_images=None):
    """Resolve one or more target image paths from CLI args."""
    targets = []
    if image:
        targets.append(resolve_image_path(image, dataset_dir))
    if images:
        for item in images:
            targets.append(resolve_image_path(item, dataset_dir))
    if all_images:
        targets.extend(list_dataset_images(dataset_dir))

    # de-duplicate while preserving order
    deduped = []
    seen = set()
    for p in targets:
        rp = str(Path(p).resolve())
        if rp not in seen:
            seen.add(rp)
            deduped.append(rp)

    if max_images is not None:
        deduped = deduped[: max(0, int(max_images))]

    if len(deduped) == 0:
        raise ValueError("No target images found. Use --image, --images, or --all-images.")

    return deduped


def hex_to_rgba(hex_color, alpha):
    """Convert #RRGGBB to matplotlib RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)


def remove_overlapped_masks(masks, iou_threshold=0.5):
    """Remove highly-overlapped masks, keeping higher predicted_iou masks."""
    if len(masks) == 0:
        return masks

    sorted_masks = sorted(masks, key=lambda x: x.get("predicted_iou", 0), reverse=True)
    keep_masks = []

    for current_mask in sorted_masks:
        current_seg = current_mask["segmentation"]
        should_keep = True
        for kept_mask in keep_masks:
            kept_seg = kept_mask["segmentation"]
            intersection = np.logical_and(current_seg, kept_seg).sum()
            union = np.logical_or(current_seg, kept_seg).sum()
            if union > 0:
                iou = intersection / union
                if iou > iou_threshold:
                    should_keep = False
                    break
        if should_keep:
            keep_masks.append(current_mask)

    return keep_masks


def remove_background_masks(masks, image_shape, border_tolerance=5, max_coverage=0.90):
    """Remove background-like masks."""
    filtered = []
    img_h, img_w = image_shape[:2]
    total_pixels = img_h * img_w

    for mask in masks:
        x, y, w, h = mask["bbox"]
        coverage = (w * h) / total_pixels
        if coverage > max_coverage:
            continue

        touches_left = x <= border_tolerance
        touches_top = y <= border_tolerance
        touches_right = (x + w) >= (img_w - border_tolerance)
        touches_bottom = (y + h) >= (img_h - border_tolerance)
        if touches_left and touches_top and touches_right and touches_bottom:
            continue

        filtered.append(mask)

    return filtered


def calculate_growth_rate(valid_growth, noise_growth):
    """Growth rate with explicit edge-case handling."""
    if valid_growth > 0:
        return noise_growth / valid_growth
    if noise_growth > 0:
        return float("inf")
    return 0.0


def cap_growth_rate_for_plot(growth_rate, cap_value=1.0):
    """Cap growth value for visualization while marking capped points."""
    if not np.isfinite(growth_rate):
        return cap_value, True
    if growth_rate >= cap_value:
        return cap_value, True
    if growth_rate <= 0:
        return 0.0, False
    return growth_rate, False


def build_threshold_barrier_facecolors(xx, yy, alpha=0.30):
    """Build smooth gradient facecolors for threshold surface."""
    x_span = max(float(np.ptp(xx)), 1e-9)
    y_span = max(float(np.ptp(yy)), 1e-9)
    x_norm = (xx - float(np.min(xx))) / x_span
    y_norm = (yy - float(np.min(yy))) / y_span

    hue = (
        0.58 * x_norm
        + 0.42 * (1.0 - y_norm)
        + 0.08 * np.sin(2.0 * np.pi * (x_norm + y_norm))
    ) % 1.0
    sat = 0.28 + 0.42 * (1.0 - np.abs(x_norm - y_norm))
    val = 0.96 - 0.10 * (0.5 * x_norm + 0.5 * y_norm)

    hsv = np.stack([hue, np.clip(sat, 0.0, 1.0), np.clip(val, 0.0, 1.0)], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return np.dstack([rgb, np.full_like(xx, alpha, dtype=float)])


def _triangle_wave(values, teeth=24):
    """Triangle wave in [-1, 1] for stylized jagged lines."""
    frac = np.mod(values * float(teeth), 1.0)
    return 2.0 * np.abs(2.0 * frac - 1.0) - 1.0


def add_spiky_plane_outline(ax, z_level, spike_amp=0.028, teeth=26, color="#2F2F2F", lw=1.2):
    """Draw jagged perimeter lines around threshold plane (spiky style)."""
    t = np.linspace(0.0, 1.0, 260)
    ax.plot(
        t,
        np.zeros_like(t),
        z_level + spike_amp * _triangle_wave(t + 0.00, teeth),
        color=color,
        linewidth=lw,
        alpha=0.92,
        zorder=9,
    )
    ax.plot(
        t,
        np.ones_like(t),
        z_level + spike_amp * _triangle_wave(t + 0.19, teeth),
        color=color,
        linewidth=lw,
        alpha=0.92,
        zorder=9,
    )
    ax.plot(
        np.zeros_like(t),
        t,
        z_level + spike_amp * _triangle_wave(t + 0.37, teeth),
        color=color,
        linewidth=lw,
        alpha=0.92,
        zorder=9,
    )
    ax.plot(
        np.ones_like(t),
        t,
        z_level + spike_amp * _triangle_wave(t + 0.53, teeth),
        color=color,
        linewidth=lw,
        alpha=0.92,
        zorder=9,
    )


def save_notext_figure(fig, path, dpi=200):
    """Remove textual elements only (keep ticks/scale) and save."""
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")

    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title("")
        if hasattr(ax, "set_xlabel"):
            ax.set_xlabel("")
        if hasattr(ax, "set_ylabel"):
            ax.set_ylabel("")
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel("")
        for txt in ax.texts[:]:
            txt.remove()

    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def save_growth_rate_3d(stage_results, output_dir, image_name, opt_pred, opt_stab):
    """Save 3D growth-rate plots (normal + spiky, each text/notext) for one image."""
    if len(stage_results) < 2:
        print("3D graph   : skipped (only 1 stage tested)")
        return None, None, None, None

    point_records = []
    n_stages = len(stage_results)

    for i in range(1, n_stages):
        prev_stage = stage_results[i - 1]
        curr_stage = stage_results[i]
        prev_best = min(prev_stage, key=lambda x: x["noise_masks"] / max(x["valid_masks"], 1))

        for combo in curr_stage:
            valid_growth = combo["valid_masks"] - prev_best["valid_masks"]
            noise_growth = combo["noise_masks"] - prev_best["noise_masks"]
            gr_raw = calculate_growth_rate(valid_growth, noise_growth)
            gr_plot, gr_capped = cap_growth_rate_for_plot(gr_raw, cap_value=1.0)
            point_records.append(
                {
                    "pred_iou": combo["pred_iou"],
                    "stability": combo["stability"],
                    "growth_rate_raw": gr_raw,
                    "growth_rate_plot": gr_plot,
                    "is_capped": gr_capped,
                    "is_optimal": (
                        np.isclose(combo["pred_iou"], opt_pred) and np.isclose(combo["stability"], opt_stab)
                    ),
                }
            )

    if len(point_records) == 0:
        print("3D graph   : skipped (no growth data)")
        return None, None, None, None

    pred_ious = [p["pred_iou"] for p in point_records]
    stabilities = [p["stability"] for p in point_records]
    growth_rates_plot = [p["growth_rate_plot"] for p in point_records]

    # Untested combinations from early stop
    tested_set = set(zip(pred_ious, stabilities))
    for combo in stage_results[0]:
        tested_set.add((combo["pred_iou"], combo["stability"]))
    untested_ious = []
    untested_stabs = []
    for stage_combos in PARAM_STAGES[n_stages:]:
        for iou_val, stab_val in stage_combos:
            if (iou_val, stab_val) not in tested_set:
                untested_ious.append(iou_val)
                untested_stabs.append(stab_val)
    def build_figure(spiky=False):
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection="3d")
        ax.computed_zorder = False

        xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100))
        zz = np.ones_like(xx) * CONSERVATIVE_GROWTH_THRESHOLD
        barrier_facecolors = build_threshold_barrier_facecolors(xx, yy, alpha=0.34)
        ax.plot_surface(
            xx,
            yy,
            zz,
            facecolors=barrier_facecolors,
            shade=False,
            linewidth=0.0,
            edgecolor="none",
            zorder=1,
            antialiased=True,
        )
        if spiky:
            add_spiky_plane_outline(ax, CONSERVATIVE_GROWTH_THRESHOLD)

        if untested_ious:
            ax.scatter(
                untested_ious,
                untested_stabs,
                [0] * len(untested_ious),
                c="none",
                s=300,
                alpha=0.6,
                marker="s",
                edgecolors="gray",
                linewidth=2,
                zorder=4,
            )

        for i in range(len(point_records)):
            ax.plot(
                [pred_ious[i], pred_ious[i]],
                [stabilities[i], stabilities[i]],
                [0, growth_rates_plot[i]],
                "k--",
                alpha=0.6,
                linewidth=1.2,
                zorder=5,
            )
        ax.scatter(
            pred_ious,
            stabilities,
            [0] * len(pred_ious),
            c="gray",
            s=150,
            alpha=0.7,
            marker="x",
            linewidth=3,
            zorder=6,
        )

        for rec in point_records:
            if rec["is_optimal"]:
                continue
            if rec["is_capped"]:
                point_color = COLOR_RED
            else:
                point_color = (
                    COLOR_GREEN
                    if rec["growth_rate_raw"] <= CONSERVATIVE_GROWTH_THRESHOLD
                    else COLOR_ORANGE
                )
            ax.scatter(
                [rec["pred_iou"]],
                [rec["stability"]],
                [rec["growth_rate_plot"]],
                c=point_color,
                marker="o",
                s=400,
                alpha=0.85,
                edgecolors="black",
                linewidth=2,
                zorder=7,
            )

        for rec in point_records:
            if rec["is_optimal"]:
                ax.scatter(
                    [rec["pred_iou"]],
                    [rec["stability"]],
                    [rec["growth_rate_plot"]],
                    c="yellow",
                    s=1000,
                    marker="*",
                    edgecolors="black",
                    linewidth=3,
                    zorder=8,
                )
                break

        tick_values = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0, 1.05)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

        ax.set_xlabel("pred_iou_thresh", fontsize=12, fontweight="bold")
        ax.set_ylabel("stability_score_thresh", fontsize=12, fontweight="bold")
        ax.set_zlabel("Growth Rate (noise/valid)", fontsize=12, fontweight="bold")
        ax.set_title(f"{image_name}", fontsize=13, fontweight="bold")

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLOR_GREEN,
                markersize=15,
                label=f"Acceptable (\u2264{CONSERVATIVE_GROWTH_THRESHOLD})",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLOR_ORANGE,
                markersize=15,
                label=f"Aggressive ({CONSERVATIVE_GROWTH_THRESHOLD}<g<1.0)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLOR_RED,
                markersize=15,
                markeredgecolor="black",
                markeredgewidth=1.2,
                label="Capped points",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="yellow",
                markersize=22,
                markeredgecolor="black",
                markeredgewidth=2,
                label="Optimal",
            ),
        ]
        if untested_ious:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="none",
                    markersize=12,
                    markeredgecolor=COLOR_BLUE,
                    markeredgewidth=2,
                    label="Not Tested",
                )
            )
        ax.legend(handles=legend_elements, loc="upper left", fontsize=12)
        plt.tight_layout()
        return fig

    fig_path = os.path.join(output_dir, "3d_growth_rate.png")
    fig_notext_path = os.path.join(output_dir, "3d_growth_rate_notext.png")
    fig = build_figure(spiky=False)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    save_notext_figure(fig, fig_notext_path, dpi=200)
    plt.close(fig)

    fig_spiky_path = os.path.join(output_dir, "3d_growth_rate_spiky.png")
    fig_spiky_notext_path = os.path.join(output_dir, "3d_growth_rate_spiky_notext.png")
    fig_spiky = build_figure(spiky=True)
    fig_spiky.savefig(fig_spiky_path, dpi=200, bbox_inches="tight")
    save_notext_figure(fig_spiky, fig_spiky_notext_path, dpi=200)
    plt.close(fig_spiky)

    return fig_path, fig_notext_path, fig_spiky_path, fig_spiky_notext_path


def _select_from_stage(stage_results, stage_idx, reason):
    """Pick best combo from target stage."""
    combos = stage_results[stage_idx]
    if len(combos) == 1:
        return stage_idx, 0, reason

    if stage_idx > 0:
        prev_best = min(
            stage_results[stage_idx - 1],
            key=lambda x: x["noise_masks"] / max(x["valid_masks"], 1),
        )
        combo_growth_rates = []
        for combo in combos:
            valid_growth = combo["valid_masks"] - prev_best["valid_masks"]
            noise_growth = combo["noise_masks"] - prev_best["noise_masks"]
            combo_growth_rates.append(calculate_growth_rate(valid_growth, noise_growth))
        best_idx = int(np.argmin(combo_growth_rates))
        return stage_idx, best_idx, reason

    best_idx = min(
        range(len(combos)),
        key=lambda j: combos[j]["noise_masks"] / max(combos[j]["valid_masks"], 1),
    )
    return stage_idx, best_idx, reason


def find_optimal_stage(stage_results):
    """Select optimal stage using growth-rate threshold."""
    n_stages = len(stage_results)
    if n_stages < 2:
        return 0, 0, "Only one stage tested"

    for i in range(1, n_stages):
        prev_stage = stage_results[i - 1]
        curr_stage = stage_results[i]
        prev_best = min(prev_stage, key=lambda x: x["noise_masks"] / max(x["valid_masks"], 1))
        curr_best = min(curr_stage, key=lambda x: x["noise_masks"] / max(x["valid_masks"], 1))

        valid_growth = curr_best["valid_masks"] - prev_best["valid_masks"]
        noise_growth = curr_best["noise_masks"] - prev_best["noise_masks"]
        growth_rate = calculate_growth_rate(valid_growth, noise_growth)

        if growth_rate >= CONSERVATIVE_GROWTH_THRESHOLD:
            reason = (
                f"Stage {i + 1}: growth_rate={growth_rate:.3f} "
                f">= {CONSERVATIVE_GROWTH_THRESHOLD}"
            )
            return _select_from_stage(stage_results, i - 1, reason)

    last_stage_idx = n_stages - 1
    reason = f"Conservative threshold never exceeded; using stage {last_stage_idx + 1}"
    return _select_from_stage(stage_results, last_stage_idx, reason)


def build_overlay(image_shape, valid_masks, noise_masks, valid_alpha=0.45, noise_alpha=0.55):
    """RGBA overlay for valid/noise separation."""
    overlay = np.zeros((image_shape[0], image_shape[1], 4), dtype=float)
    valid_rgba = hex_to_rgba(COLOR_BLUE, valid_alpha)
    noise_rgba = hex_to_rgba(COLOR_RED, noise_alpha)

    for mask in valid_masks:
        overlay[mask["segmentation"]] = valid_rgba
    for mask in noise_masks:
        overlay[mask["segmentation"]] = noise_rgba

    return overlay


def save_stage_images(region_rgb, combo_result, output_dir, stage_idx, combo_idx):
    """Save stage/combo images (text + notext variants)."""
    prefix = f"stage_{stage_idx + 1:02d}_combo_{combo_idx + 1:02d}"
    pred_iou = combo_result["pred_iou"]
    stability = combo_result["stability"]
    n_total = combo_result["total_masks"]
    n_valid = combo_result["valid_masks"]
    n_noise = combo_result["noise_masks"]

    def save_all_panel(notext=False):
        suffix = "_notext" if notext else ""
        fig, (ax_img, ax_chart) = plt.subplots(
            1,
            2,
            figsize=(11, 8),
            gridspec_kw={"width_ratios": [5.6, 1.0], "wspace": 0.05},
        )
        fig.patch.set_facecolor("white")

        ax_img.imshow(region_rgb)
        ax_img.imshow(
            build_overlay(
                region_rgb.shape,
                combo_result["valid_only"],
                combo_result["noise_only"],
                valid_alpha=0.45,
                noise_alpha=0.60,
            )
        )
        if not notext:
            ax_img.set_title(
                f"Stage {stage_idx + 1} ({pred_iou:.2f}, {stability:.2f}) - All ({n_total})",
                fontsize=14,
                fontweight="bold",
            )
        ax_img.axis("off")

        bars = ax_chart.bar(
            ["V", "N"],
            [n_valid, n_noise],
            color=[COLOR_BLUE, COLOR_RED],
            edgecolor=[COLOR_BLUE, COLOR_RED],
            linewidth=1.8,
            width=0.55,
            alpha=0.95,
        )
        ax_chart.grid(axis="y", alpha=0.25, linestyle="--")
        ax_chart.spines["top"].set_visible(False)
        ax_chart.spines["right"].set_visible(False)
        if not notext:
            ax_chart.set_ylabel("Count")
            ax_chart.tick_params(axis="x", labelsize=10)
            ax_chart.tick_params(axis="y", labelsize=10)
            for bar in bars:
                height = bar.get_height()
                ax_chart.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
        else:
            ax_chart.set_ylabel("")
            ax_chart.set_xticks([])
            ax_chart.set_yticks([])

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_all{suffix}.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)

    def save_mask_panel(mask_overlay, panel_name, title, title_color, notext=False):
        suffix = "_notext" if notext else ""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(region_rgb)
        ax.imshow(mask_overlay)
        if not notext:
            ax.set_title(title, fontsize=14, fontweight="bold", color=title_color)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_{panel_name}{suffix}.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)

    valid_overlay = build_overlay(region_rgb.shape, combo_result["valid_only"], [], valid_alpha=0.50)
    noise_overlay = build_overlay(region_rgb.shape, [], combo_result["noise_only"], noise_alpha=0.65)

    save_all_panel(notext=False)
    save_all_panel(notext=True)
    save_mask_panel(
        valid_overlay,
        "valid",
        f"Stage {stage_idx + 1} ({pred_iou:.2f}, {stability:.2f}) - Valid ({n_valid})",
        COLOR_BLUE,
        notext=False,
    )
    save_mask_panel(
        valid_overlay,
        "valid",
        "",
        COLOR_BLUE,
        notext=True,
    )
    save_mask_panel(
        noise_overlay,
        "noise",
        f"Stage {stage_idx + 1} ({pred_iou:.2f}, {stability:.2f}) - Noise ({n_noise})",
        COLOR_RED,
        notext=False,
    )
    save_mask_panel(
        noise_overlay,
        "noise",
        "",
        COLOR_RED,
        notext=True,
    )

def write_stage_csv(stage_results, output_dir):
    """Write compact stage summary CSV."""
    out_path = os.path.join(output_dir, "stage_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage",
                "combo",
                "pred_iou_thresh",
                "stability_score_thresh",
                "total_masks",
                "valid_masks",
                "noise_masks",
                "noise_ratio",
            ]
        )
        for stage_idx, stage_data in enumerate(stage_results):
            for combo_idx, combo in enumerate(stage_data):
                writer.writerow(
                    [
                        stage_idx + 1,
                        combo_idx + 1,
                        f"{combo['pred_iou']:.2f}",
                        f"{combo['stability']:.2f}",
                        combo["total_masks"],
                        combo["valid_masks"],
                        combo["noise_masks"],
                        f"{combo['noise_ratio']:.4f}",
                    ]
                )
    return out_path


def build_sam_model(base_dir):
    """Load SAM2 model with the same fixed setting as sam_param_optimizer.py."""
    model_type = "hiera_l"
    checkpoint = os.path.join(base_dir, "checkpoints", "sam2.1_hiera_large.pt")
    config_path = get_sam2_config_path("sam2.1_hiera_l.yaml")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam2(config_path, checkpoint, device=device)
    return sam, device, model_type, checkpoint


def process_one_image(image_path, output_root, sam, device, model_type, checkpoint):
    """Run stage optimization for one image path."""
    image_name = os.path.basename(image_path)
    image_stem = safe_name(Path(image_name).stem)
    output_dir = os.path.join(output_root, image_stem)
    os.makedirs(output_dir, exist_ok=True)

    print("-" * 80)
    print(f"Image      : {image_path}")
    print(f"Output dir : {output_dir}")
    print(f"Stages     : {len(PARAM_STAGES)}")
    print("Early stop : True")

    # Load image and crop central 80%
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    h, w = image.shape[:2]
    margin_h = int(h * 0.1)
    margin_w = int(w * 0.1)
    particle_region = image[margin_h : h - margin_h, margin_w : w - margin_w]

    print("Applying BM3D preprocessing (sigma_psd=40)...")
    particle_region = apply_bm3d_denoising(particle_region, sigma_psd=40, color_mode="auto")
    region_rgb = cv2.cvtColor(particle_region, cv2.COLOR_BGR2RGB)
    print(f"Region size : {particle_region.shape}")

    print(f"SAM model   : {model_type} ({device})")
    print(f"Checkpoint  : {checkpoint}")

    # Reference masks -> fixed threshold
    print("\nCalculating fixed threshold from reference masks...")
    ref_params = [
        (0.90, 0.90, "strict"),
        (0.80, 0.80, "medium"),
        (0.70, 0.70, "loose"),
    ]
    masks_ref = []
    ref_used = None

    for pred_iou, stability, level in ref_params:
        print(f"  Trying ({pred_iou:.2f}, {stability:.2f}) [{level}]...", end="")
        mask_generator_ref = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            points_per_batch=256,
            pred_iou_thresh=pred_iou,
            stability_score_thresh=stability,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
        )
        masks_ref = mask_generator_ref.generate(particle_region)
        if len(masks_ref) > 0:
            ref_used = (pred_iou, stability, level)
            print(f" OK ({len(masks_ref)} masks)")
            break
        print(" fail")

    if len(masks_ref) == 0:
        raise RuntimeError("Failed to generate reference masks with fallback params.")

    masks_ref = remove_overlapped_masks(masks_ref, iou_threshold=0.5)
    masks_ref = remove_background_masks(masks_ref, particle_region.shape)
    if len(masks_ref) == 0:
        raise RuntimeError("No masks remain after overlap/background filtering for reference.")

    areas = np.array([m["area"] for m in masks_ref], dtype=np.float64)
    q1 = float(np.percentile(areas, 25))
    q3 = float(np.percentile(areas, 75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    masks_ref_iqr = [m for m in masks_ref if lower_bound <= m["area"] <= upper_bound]
    if len(masks_ref_iqr) == 0:
        masks_ref_iqr = masks_ref

    median_area = float(np.median([m["area"] for m in masks_ref_iqr]))
    fixed_threshold = median_area * THRESHOLD_RATIO

    print(f"  Reference params : ({ref_used[0]:.2f}, {ref_used[1]:.2f}) [{ref_used[2]}]")
    print(f"  IQR bounds       : [{lower_bound:.1f}, {upper_bound:.1f}]")
    print(f"  Median area      : {median_area:.1f}")
    print(f"  Fixed threshold  : {fixed_threshold:.1f}")

    # Stage search
    print("\nTesting stage combinations...")
    stage_results = []

    for stage_idx, stage_combos in enumerate(PARAM_STAGES):
        print(f"\n--- Stage {stage_idx + 1}/{len(PARAM_STAGES)} ---")
        stage_data = []

        for combo_idx, (pred_iou, stability) in enumerate(stage_combos):
            print(f"  Combo {combo_idx + 1}: ({pred_iou:.2f}, {stability:.2f})...", end="")
            mask_generator = SAM2AutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                points_per_batch=256,
                pred_iou_thresh=pred_iou,
                stability_score_thresh=stability,
                crop_n_layers=1,
                crop_n_points_downscale_factor=1,
            )

            masks = mask_generator.generate(particle_region)
            masks = remove_overlapped_masks(masks, iou_threshold=0.5)
            masks = remove_background_masks(masks, particle_region.shape)
            masks_filtered = [m for m in masks if lower_bound <= m["area"] <= upper_bound]

            valid_masks = [m for m in masks_filtered if m["area"] >= fixed_threshold]
            noise_masks = [m for m in masks_filtered if m["area"] < fixed_threshold]

            combo_result = {
                "stage": stage_idx,
                "combo_idx": combo_idx,
                "pred_iou": pred_iou,
                "stability": stability,
                "total_masks": len(masks_filtered),
                "valid_masks": len(valid_masks),
                "noise_masks": len(noise_masks),
                "noise_ratio": len(noise_masks) / max(len(masks_filtered), 1),
                "all_masks": masks_filtered,
                "valid_only": valid_masks,
                "noise_only": noise_masks,
            }
            stage_data.append(combo_result)

            save_stage_images(region_rgb, combo_result, output_dir, stage_idx, combo_idx)
            print(
                f" total={combo_result['total_masks']}, "
                f"valid={combo_result['valid_masks']}, "
                f"noise={combo_result['noise_masks']}"
            )

        stage_results.append(stage_data)

        if stage_idx >= 1:
            prev_best = min(
                stage_results[stage_idx - 1],
                key=lambda x: x["noise_masks"] / max(x["valid_masks"], 1),
            )
            curr_best = min(
                stage_results[stage_idx],
                key=lambda x: x["noise_masks"] / max(x["valid_masks"], 1),
            )
            valid_growth = curr_best["valid_masks"] - prev_best["valid_masks"]
            noise_growth = curr_best["noise_masks"] - prev_best["noise_masks"]
            should_stop = False
            if valid_growth <= 0 and noise_growth > 0:
                print(f"  [EARLY STOP] explosion: valid_growth <= 0, noise_growth={noise_growth}")
                should_stop = True
            elif valid_growth > 0:
                growth_rate = noise_growth / valid_growth
                if growth_rate >= CONSERVATIVE_GROWTH_THRESHOLD:
                    print(
                        f"  [EARLY STOP] growth_rate={growth_rate:.3f} "
                        f">= {CONSERVATIVE_GROWTH_THRESHOLD}"
                    )
                    should_stop = True
            if should_stop:
                break

    # Final selection + summary files
    optimal_stage, optimal_combo, reason = find_optimal_stage(stage_results)
    optimal = stage_results[optimal_stage][optimal_combo]
    stage_csv = write_stage_csv(stage_results, output_dir)
    (
        graph_path,
        graph_notext_path,
        graph_spiky_path,
        graph_spiky_notext_path,
    ) = save_growth_rate_3d(
        stage_results=stage_results,
        output_dir=output_dir,
        image_name=image_name,
        opt_pred=optimal["pred_iou"],
        opt_stab=optimal["stability"],
    )

    summary_path = os.path.join(output_dir, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Single Image SAM Optimization Summary\n")
        f.write(f"image={image_path}\n")
        f.write(f"model={model_type}\n")
        f.write(f"device={device}\n")
        f.write(f"threshold_ratio={THRESHOLD_RATIO}\n")
        f.write(f"fixed_threshold={fixed_threshold:.4f}\n")
        f.write(f"optimal_stage={optimal_stage + 1}\n")
        f.write(f"optimal_combo={optimal_combo + 1}\n")
        f.write(f"optimal_pred_iou={optimal['pred_iou']:.2f}\n")
        f.write(f"optimal_stability={optimal['stability']:.2f}\n")
        f.write(f"optimal_valid={optimal['valid_masks']}\n")
        f.write(f"optimal_noise={optimal['noise_masks']}\n")
        f.write(f"reason={reason}\n")
        if graph_path is not None:
            f.write(f"graph_3d={graph_path}\n")
            f.write(f"graph_3d_notext={graph_notext_path}\n")
            f.write(f"graph_3d_spiky={graph_spiky_path}\n")
            f.write(f"graph_3d_spiky_notext={graph_spiky_notext_path}\n")

    print(f"Optimal: Stage {optimal_stage + 1}, Combo {optimal_combo + 1}")
    print(
        f"  Params: ({optimal['pred_iou']:.2f}, {optimal['stability']:.2f}) | "
        f"valid={optimal['valid_masks']}, noise={optimal['noise_masks']}"
    )
    print(f"Reason       : {reason}")
    print(f"Stage CSV    : {stage_csv}")
    if graph_path is not None:
        print(f"3D graph     : {graph_path}")
        print(f"3D graph NT  : {graph_notext_path}")
        print(f"3D spiky     : {graph_spiky_path}")
        print(f"3D spiky NT  : {graph_spiky_notext_path}")
    print(f"Run summary  : {summary_path}")
    print(f"Output dir   : {output_dir}")

    return {
        "image_path": image_path,
        "output_dir": output_dir,
        "stage_csv": stage_csv,
        "summary_path": summary_path,
        "optimal_stage": optimal_stage + 1,
        "optimal_combo": optimal_combo + 1,
        "optimal_pred_iou": optimal["pred_iou"],
        "optimal_stability": optimal["stability"],
        "optimal_valid": optimal["valid_masks"],
        "optimal_noise": optimal["noise_masks"],
        "reason": reason,
        "graph_3d": graph_path,
        "graph_3d_notext": graph_notext_path,
        "graph_3d_spiky": graph_spiky_path,
        "graph_3d_spiky_notext": graph_spiky_notext_path,
    }


def main():
    parser = argparse.ArgumentParser(description="SAM stage visualizer (single or multiple images)")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--image",
        help="Single image filename (in dataset-dir) or absolute/relative path",
    )
    target_group.add_argument(
        "--images",
        nargs="+",
        help="Multiple image filenames/paths",
    )
    target_group.add_argument(
        "--all-images",
        action="store_true",
        help="Process all images in dataset-dir",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(parent_dir, "Dataset_size"),
        help="Dataset directory used for filename resolution and --all-images",
    )
    parser.add_argument(
        "--output-root",
        default=str(get_optimization_dir("sam_optimization_v2", "single_image")),
        help="Root output directory",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of target images (useful with --all-images)",
    )
    args = parser.parse_args()

    if not check_bm3d_available():
        raise ImportError("BM3D is not available.")

    os.makedirs(args.output_root, exist_ok=True)
    image_paths = resolve_target_images(
        image=args.image,
        images=args.images,
        all_images=args.all_images,
        dataset_dir=args.dataset_dir,
        max_images=args.max_images,
    )

    print("=" * 80)
    print("SAM PARAMETER OPTIMIZATION (SINGLE/MULTI)")
    print("=" * 80)
    print(f"Targets    : {len(image_paths)} image(s)")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Output root: {args.output_root}")
    print(f"Stages     : {len(PARAM_STAGES)}")
    print("Early stop : True")
    if len(image_paths) <= 8:
        for idx, p in enumerate(image_paths, 1):
            print(f"  {idx:>2}. {os.path.basename(p)}")
    else:
        for idx, p in enumerate(image_paths[:8], 1):
            print(f"  {idx:>2}. {os.path.basename(p)}")
        print(f"  ... ({len(image_paths) - 8} more)")

    # Build SAM model once for all images
    print("Loading SAM2 model...")
    sam, device, model_type, checkpoint = build_sam_model(parent_dir)
    print(f"SAM model   : {model_type} ({device})")
    print(f"Checkpoint  : {checkpoint}")

    results = []
    failures = []

    for idx, image_path in enumerate(image_paths, 1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(image_paths)}] PROCESSING")
        print("=" * 80)
        try:
            result = process_one_image(
                image_path=image_path,
                output_root=args.output_root,
                sam=sam,
                device=device,
                model_type=model_type,
                checkpoint=checkpoint,
            )
            results.append(result)
        except Exception as e:
            failures.append((image_path, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {os.path.basename(image_path)}")
            print(f"       {type(e).__name__}: {e}")

    # Batch summary
    batch_summary_path = os.path.join(args.output_root, "batch_summary.csv")
    with open(batch_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "optimal_stage",
                "optimal_combo",
                "optimal_pred_iou",
                "optimal_stability",
                "optimal_valid",
                "optimal_noise",
                "reason",
                "output_dir",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["image_path"],
                    r["optimal_stage"],
                    r["optimal_combo"],
                    f"{r['optimal_pred_iou']:.2f}",
                    f"{r['optimal_stability']:.2f}",
                    r["optimal_valid"],
                    r["optimal_noise"],
                    r["reason"],
                    r["output_dir"],
                ]
            )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Succeeded    : {len(results)}/{len(image_paths)}")
    print(f"Batch CSV    : {batch_summary_path}")
    if failures:
        print(f"Failed       : {len(failures)}")
        for path, err in failures:
            print(f"  - {os.path.basename(path)} -> {err}")


if __name__ == "__main__":
    main()

# %%
