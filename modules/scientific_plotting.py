"""Shared scientific figures for VISION and the standalone analysis workflow.

Every production chart and the upload-screen example chart uses these helpers so
the GUI never presents a visualization style that the analysis pipeline cannot
produce from real measurements.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np


CHART_COLORS = {
    "ink": "#334155",
    "grid": "#d9e3ef",
    "axis": "#9fb0c4",
    "size_bar": "#4f7cac",
    "size_bar_alt": "#6d9dc5",
    "size_trend": "#d96d3b",
    "size_fill": "#f4a261",
    "spatial_fill": "#7b61a8",
    "spatial_line": "#5b3f8c",
    "spatial_points": "#2a9d8f",
    "spatial_iqr": "#264653",
    "spatial_ci": "#f4a261",
}

SHAPE_COLORS = (
    "#4f7cac",
    "#2a9d8f",
    "#d96d3b",
    "#7b61a8",
    "#76b7b2",
    "#e9c46a",
)

SPATIAL_KDE_X_PADDING_FRACTION = 0.12
SPATIAL_KDE_GRID_POINTS = 360
SPATIAL_KDE_HEIGHT = 0.76
SPATIAL_KDE_BASE_Y = 0.08
SPATIAL_BOOTSTRAP_RESAMPLES = 10_000
SPATIAL_BOOTSTRAP_SEED = 42


def shape_color_sequence(count: int) -> list[Tuple[float, float, float]]:
    """Return the shared presentation palette as Matplotlib RGB tuples."""
    return [to_rgb(SHAPE_COLORS[index % len(SHAPE_COLORS)]) for index in range(count)]


def _as_numeric_values(values: Iterable[Any] | Mapping[Any, Any]) -> np.ndarray:
    """Convert data to a finite one-dimensional float array."""
    if isinstance(values, Mapping):
        values = values.values()
    array = np.asarray(list(values), dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _data_limits(values: np.ndarray) -> Tuple[float, float]:
    """Provide padded plotting bounds, including for a single repeated value."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span <= np.finfo(float).eps:
        span = max(abs(lower) * 0.15, 1.0)
    padding = span * SPATIAL_KDE_X_PADDING_FRACTION
    return lower - padding, upper + padding


def _kde(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Small dependency-free Gaussian kernel density estimate."""
    if len(values) == 1:
        bandwidth = max(abs(float(values[0])) * 0.1, 1.0)
    else:
        std = float(np.std(values, ddof=1))
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        robust_std = iqr / 1.349 if iqr > 0 else std
        scale = min(std, robust_std) if robust_std > 0 else std
        bandwidth = 0.9 * max(scale, np.finfo(float).eps) * len(values) ** (-0.2)
        bandwidth = max(bandwidth, max(np.ptp(values) * 0.03, 1e-6))

    scaled = (x[:, None] - values[None, :]) / bandwidth
    kernels = np.exp(-0.5 * scaled**2) / np.sqrt(2 * np.pi)
    return kernels.mean(axis=1) / bandwidth


def bootstrap_median_ci(
    values: Iterable[Any] | Mapping[Any, Any],
    confidence: float = 0.95,
    n_bootstrap: int = SPATIAL_BOOTSTRAP_RESAMPLES,
    seed: int = SPATIAL_BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Calculate a deterministic percentile bootstrap CI for the median."""
    array = _as_numeric_values(values)
    if len(array) == 0:
        return float("nan"), float("nan")
    if len(array) == 1:
        value = float(array[0])
        return value, value

    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, len(array)), replace=True)
    medians = np.median(samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(medians, alpha)), float(np.quantile(medians, 1.0 - alpha))


def summarize_spatial_distribution(values: Iterable[Any] | Mapping[Any, Any]) -> Dict[str, float]:
    """Summarize values represented by the spatial raincloud plot."""
    array = _as_numeric_values(values)
    if len(array) == 0:
        return {}

    ci_low, ci_high = bootstrap_median_ci(array)
    return {
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "median_ci_low": ci_low,
        "median_ci_high": ci_high,
        "count": int(len(array)),
    }


def spatial_distribution_plot_data(
    values: Iterable[Any] | Mapping[Any, Any],
) -> Dict[str, Any]:
    """Return the deterministic coordinates used by the spatial raincloud plot."""
    array = _as_numeric_values(values)
    if len(array) == 0:
        return {}

    lower, upper = _data_limits(array)
    kde_x = np.linspace(lower, upper, SPATIAL_KDE_GRID_POINTS)
    density = _kde(array, kde_x)
    density_max = float(np.max(density))
    kde_y_normalized = density / density_max if density_max > 0 else density

    rng = np.random.default_rng(SPATIAL_BOOTSTRAP_SEED)
    observed_y = rng.uniform(-0.14, -0.035, size=len(array))

    return {
        "observed_x": array,
        "observed_y": observed_y,
        "kde_x": kde_x,
        "kde_y_normalized": kde_y_normalized,
        "kde_y_plot": SPATIAL_KDE_BASE_Y + SPATIAL_KDE_HEIGHT * kde_y_normalized,
        "kde_base_y": np.full(kde_x.shape, SPATIAL_KDE_BASE_Y, dtype=float),
        "summary": summarize_spatial_distribution(array),
    }


def _style_axes(ax, grid_axis: str = "y") -> None:
    """Apply the common clean scientific chart style to an axes."""
    ax.set_facecolor("#fbfcfe")
    ax.grid(axis=grid_axis, color=CHART_COLORS["grid"], linewidth=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHART_COLORS["axis"])
    ax.spines["bottom"].set_color(CHART_COLORS["axis"])
    ax.tick_params(colors="#4b5563", labelsize=11)


def plot_size_distribution(
    ax,
    values: Iterable[Any] | Mapping[Any, Any],
    scale: str = "nm",
    title: str = "Projected-area distribution",
) -> Dict[str, float]:
    """Plot a histogram with a KDE trend line and median reference."""
    array = _as_numeric_values(values)
    if len(array) == 0:
        ax.text(0.5, 0.5, "No projected-area measurements", ha="center", va="center", color=CHART_COLORS["ink"])
        ax.axis("off")
        return {}

    _style_axes(ax, grid_axis="y")
    lower, upper = _data_limits(array)
    bin_count = int(np.clip(np.sqrt(len(array)) * 1.7, 8, 24))
    counts, edges, bars = ax.hist(
        array,
        bins=bin_count,
        range=(lower, upper),
        edgecolor="white",
        linewidth=1.25,
        alpha=0.94,
    )
    for index, bar in enumerate(bars):
        bar.set_facecolor(
            CHART_COLORS["size_bar"] if index % 2 == 0 else CHART_COLORS["size_bar_alt"]
        )

    x = np.linspace(lower, upper, 320)
    density = _kde(array, x) * len(array) * (edges[1] - edges[0])
    ax.fill_between(x, density, color=CHART_COLORS["size_fill"], alpha=0.20)
    ax.plot(x, density, color=CHART_COLORS["size_trend"], linewidth=2.7, label="KDE trend")

    median = float(np.median(array))
    ax.axvline(median, color=CHART_COLORS["size_bar"], linewidth=1.9, linestyle=(0, (4, 3)), label="Median")
    ax.set_xlim(lower, upper)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Projected particle area ({scale}^2)", fontsize=13, fontweight="semibold", color=CHART_COLORS["ink"])
    ax.set_ylabel("Count", fontsize=13, fontweight="semibold", color=CHART_COLORS["ink"])
    ax.set_title(title, fontsize=14, fontweight="bold", color=CHART_COLORS["ink"], pad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    return {
        "median": median,
        "mean": float(np.mean(array)),
        "count": int(len(array)),
    }


def create_size_distribution_figure(
    values: Iterable[Any] | Mapping[Any, Any],
    scale: str = "nm",
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Create a standalone figure using the production size plot style."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    plot_size_distribution(ax, values, scale=scale)
    fig.tight_layout(pad=1.0)
    return fig


def plot_spatial_distribution(
    ax,
    values: Iterable[Any] | Mapping[Any, Any],
    scale: str = "nm",
    title: str = "PF-SUI",
    value_label: str = "Particle-boundary-based Voronoi cell area",
    unit_suffix: str = "^2",
) -> Dict[str, float]:
    """Plot a half-violin raincloud with raw values, IQR, and median 95% CI."""
    plot_data = spatial_distribution_plot_data(values)
    if not plot_data:
        ax.text(0.5, 0.5, "No PF-SUI measurements", ha="center", va="center", color=CHART_COLORS["ink"])
        ax.axis("off")
        return {}

    _style_axes(ax, grid_axis="x")
    ax.grid(axis="y", visible=False)
    observed_x = plot_data["observed_x"]
    observed_y = plot_data["observed_y"]
    kde_x = plot_data["kde_x"]
    kde_y_plot = plot_data["kde_y_plot"]
    kde_base_y = plot_data["kde_base_y"]
    ax.fill_between(kde_x, kde_base_y, kde_y_plot, color=CHART_COLORS["spatial_fill"], alpha=0.34)
    ax.plot(kde_x, kde_y_plot, color=CHART_COLORS["spatial_line"], linewidth=2.7)

    ax.scatter(observed_x, observed_y, s=22, color=CHART_COLORS["spatial_points"], alpha=0.66, edgecolors="none", label="Observed values")

    summary = plot_data["summary"]
    q1 = summary["q1"]
    q3 = summary["q3"]
    median = summary["median"]
    ci_low = summary["median_ci_low"]
    ci_high = summary["median_ci_high"]

    ax.hlines(-0.23, q1, q3, color=CHART_COLORS["spatial_iqr"], linewidth=7, capstyle="round", label="IQR")
    ax.hlines(-0.33, ci_low, ci_high, color=CHART_COLORS["spatial_ci"], linewidth=5, capstyle="round", label="Median 95% CI")
    ax.scatter([median], [-0.33], s=48, color=CHART_COLORS["spatial_ci"], zorder=4)

    ax.set_xlim(float(kde_x[0]), float(kde_x[-1]))
    ax.set_ylim(-0.43, 1.02)
    ax.set_yticks([])
    ax.set_xlabel(
        f"{value_label} ({scale}{unit_suffix})",
        fontsize=13,
        fontweight="semibold",
        color=CHART_COLORS["ink"],
    )
    ax.set_title(title, fontsize=14, fontweight="bold", color=CHART_COLORS["ink"], pad=10)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CHART_COLORS["spatial_points"], markersize=7, label="Observed values"),
        Line2D([0], [0], color=CHART_COLORS["spatial_iqr"], linewidth=5, label="IQR"),
        Line2D([0], [0], color=CHART_COLORS["spatial_ci"], linewidth=4, label="Median 95% CI"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9)
    return summary


def create_spatial_distribution_figure(
    values: Iterable[Any] | Mapping[Any, Any],
    scale: str = "nm",
    figsize: Tuple[float, float] = (10, 6),
    title: str = "PF-SUI",
    value_label: str = "Particle-boundary-based Voronoi cell area",
    unit_suffix: str = "^2",
) -> plt.Figure:
    """Create a standalone figure using the production spatial plot style."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    plot_spatial_distribution(
        ax,
        values,
        scale=scale,
        title=title,
        value_label=value_label,
        unit_suffix=unit_suffix,
    )
    fig.tight_layout(pad=1.0)
    return fig


def plot_shape_composition(
    ax,
    shape_counts: Mapping[str, int],
    color_map: Mapping[str, Any] | None = None,
    title: str = "Morphology composition",
) -> None:
    """Plot a clean donut chart using the same palette as the shape overlay."""
    counts = {label: int(count) for label, count in shape_counts.items() if count > 0}
    if not counts:
        ax.text(0.5, 0.5, "No projected-morphology classifications", ha="center", va="center", color=CHART_COLORS["ink"])
        ax.axis("off")
        return

    labels = list(counts)
    values = list(counts.values())
    fallback_colors = shape_color_sequence(len(labels))
    colors = [color_map.get(label, fallback_colors[index]) if color_map else fallback_colors[index] for index, label in enumerate(labels)]

    def _percentage(pct: float) -> str:
        return f"{pct:.0f}%" if pct >= 6 else ""

    wedges, _, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=_percentage,
        pctdistance=0.76,
        wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 2.0},
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    for text in autotexts:
        text.set_color("white")

    ax.text(
        0,
        0,
        f"n = {sum(values)}\nparticles",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="semibold",
        color=CHART_COLORS["ink"],
    )
    ax.set_title(title, fontsize=14, fontweight="bold", color=CHART_COLORS["ink"], pad=10)
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=9)
    ax.set_aspect("equal")


def create_shape_composition_figure(
    shape_counts: Mapping[str, int],
    color_map: Mapping[str, Any] | None = None,
    figsize: Tuple[float, float] = (8, 7),
) -> plt.Figure:
    """Create a standalone figure using the production shape chart style."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    plot_shape_composition(ax, shape_counts, color_map=color_map)
    fig.tight_layout(pad=1.0)
    return fig
