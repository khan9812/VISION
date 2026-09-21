"""
Size Analysis Module for Particle Analysis
==========================================
Particle size distribution analysis and visualization.

Functions:
- calculate_mask_areas: Calculate particle areas from masks
- calculate_size_metrics: Compute statistical metrics
- create_size_histogram: Histogram visualization
- create_size_overlay: Heatmap overlay visualization
- create_size_statistics_box: Statistics box figure
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import skew, kurtosis
from shapely.geometry import Polygon
from modules.color_palette import PALETTE_NEUTRAL, make_value_colormap, palette_hex
from modules.scientific_plotting import create_size_distribution_figure


def calculate_mask_areas(boundary_pixels, pixel_to_real):
    """
    Calculate real-world areas of particles from boundary pixels.

    Args:
        boundary_pixels: dict
            Dictionary mapping centroid tuples to lists of boundary pixel coordinates
            Format: {(x, y): [(x1, y1), (x2, y2), ...], ...}
        pixel_to_real: float
            Conversion factor from pixels to real units (nm/pixel)

    Returns:
        dict: Mapping from centroid to real area
            Format: {(x, y): area_nm2, ...}

    Examples:
        >>> areas = calculate_mask_areas(boundary_pixels, pixel_to_real=0.5)
        >>> print(f"Particle at (100, 200): {areas[(100, 200)]:.2f} nm^2")
    """
    mask_areas = {}

    for key, points in boundary_pixels.items():
        try:
            # Create polygon from boundary points
            polygon = Polygon(points)

            if polygon.is_valid:
                # Calculate pixel area
                pixel_area = polygon.area

                # Convert to real units
                real_area = pixel_area * (pixel_to_real ** 2)

                mask_areas[key] = real_area
            else:
                print(f"[WARN] Invalid polygon for particle at {key}, skipping")

        except Exception as e:
            print(f"[WARN] Error calculating area for particle at {key}: {e}")

    return mask_areas


def calculate_size_metrics(areas, scale="nm"):
    """
    Calculate comprehensive statistical metrics for particle sizes.

    Metrics computed:
    - Mean: Average particle size
    - Std Dev: Standard deviation (variability)
    - CV (Coefficient of Variation): Std/Mean (relative variability)
    - Median: Middle value (robust to outliers)
    - IQR (Interquartile Range): Q3 - Q1 (spread of middle 50%)
    - Skewness: Distribution asymmetry (<0: left-skewed, >0: right-skewed)
    - Excess kurtosis: Tail heaviness relative to a normal distribution (normal = 0)

    Args:
        areas: list or np.ndarray
            Particle areas in real units
        scale: str, optional (default="nm")
            Unit of measurement for display

    Returns:
        dict: Statistical metrics with keys:
            - total_particles: int
            - mean, std, cv, median, iqr, skewness, kurtosis: float
            - min, max: float
            - scale: str
    """
    areas_array = np.array(areas)

    if len(areas_array) == 0:
        raise ValueError("No particle areas provided")

    # Basic statistics
    mean_area = np.mean(areas_array)
    std_area = np.std(areas_array, ddof=1) if len(areas_array) > 1 else 0.0
    median_area = np.median(areas_array)
    min_area = np.min(areas_array)
    max_area = np.max(areas_array)

    # Derived statistics
    cv = std_area / mean_area if mean_area > 0 else 0
    q1 = np.percentile(areas_array, 25)
    q3 = np.percentile(areas_array, 75)
    iqr = q3 - q1

    # Distribution shape
    if len(areas_array) < 3 or np.allclose(areas_array, areas_array[0]):
        skewness = 0.0
        kurt = 0.0
    else:
        skewness = skew(areas_array)
        kurt = kurtosis(areas_array, fisher=True)  # Excess kurtosis (normal = 0)

    metrics = {
        'total_particles': len(areas_array),
        'mean': mean_area,
        'std': std_area,
        'cv': cv,
        'median': median_area,
        'iqr': iqr,
        'skewness': skewness,
        'kurtosis': kurt,
        'min': min_area,
        'max': max_area,
        'q1': q1,
        'q3': q3,
        'scale': scale
    }

    return metrics


def create_size_histogram(areas, scale="nm", figsize=(10, 6)):
    """Create the shared histogram, KDE trend, and median reference figure."""
    return create_size_distribution_figure(areas, scale=scale, figsize=figsize)


def create_size_overlay(
    image,
    boundary_pixels,
    mask_areas,
    scale="nm",
    figsize=(10, 10),
    show_area_labels=True,
):
    """
    Create particle size overlay with heatmap coloring and area labels.

    Args:
        image: np.ndarray
            Original image (BGR or RGB)
        boundary_pixels: dict
            Centroid to boundary-pixels mapping
        mask_areas: dict
            Centroid to area mapping
        scale: str
            Unit of measurement
        figsize: tuple
            Figure size
        show_area_labels: Whether to draw each particle's numeric area value.

    Returns:
        matplotlib.figure.Figure: Overlay figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Display base image
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume BGR, convert to RGB for display
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap='gray')

    # Normalize areas for colormap
    areas = np.array(list(mask_areas.values()))
    norm = Normalize(vmin=areas.min(), vmax=areas.max())
    cmap_colors = make_value_colormap()

    # Draw particles with heatmap colors and labels
    for key, points in boundary_pixels.items():
        if key not in mask_areas:
            continue

        area = mask_areas[key]
        polygon = Polygon(points)

        if polygon.is_valid:
            # Get color based on area
            color = cmap_colors(norm(area))

            # Draw filled polygon
            x, y = polygon.exterior.xy
            ax.fill(x, y, color=color, alpha=0.5)
            ax.plot(x, y, color=palette_hex(0), linewidth=1)

            if show_area_labels:
                # Add area label (positioned at top of particle)
                min_y = min([point[1] for point in points])
                ax.text(
                    polygon.centroid.x,
                    min_y - 5,
                    f'{area:.1f}',
                    color=palette_hex(0),
                    fontsize=10,
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE_NEUTRAL, alpha=0.9)
                )

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap_colors, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.04, pad=0.04)
    cbar.ax.tick_params(labelsize=0, length=0)  # Hide ticks

    ax.set_title(f'Projected-area distribution ({scale}^2)', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    return fig


def create_size_statistics_box(metrics, figsize=(8, 6)):
    """
    Create a statistics summary box figure.

    Args:
        metrics: dict
            Output from calculate_size_metrics()
        figsize: tuple
            Figure size

    Returns:
        matplotlib.figure.Figure: Statistics box figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Create statistics text
    stats_text = f"""
    SIZE ANALYSIS STATISTICS
    ------------------------

    Total Particles: {metrics['total_particles']}

    Central Tendency:
      Mean: {metrics['mean']:.2f} {metrics['scale']}^2
      Median: {metrics['median']:.2f} {metrics['scale']}^2

    Variability:
      Std Dev: {metrics['std']:.2f} {metrics['scale']}^2
      CV (Coefficient of Variation): {metrics['cv']:.3f}
      IQR (Interquartile Range): {metrics['iqr']:.2f} {metrics['scale']}^2

    Range:
      Min: {metrics['min']:.2f} {metrics['scale']}^2
      Max: {metrics['max']:.2f} {metrics['scale']}^2
      Q1 (25th percentile): {metrics['q1']:.2f} {metrics['scale']}^2
      Q3 (75th percentile): {metrics['q3']:.2f} {metrics['scale']}^2

    Distribution Shape:
      Skewness: {metrics['skewness']:.3f}
        {get_skewness_interpretation(metrics['skewness'])}
      Kurtosis: {metrics['kurtosis']:.3f}
        {get_kurtosis_interpretation(metrics['kurtosis'])}
    """

    # Add text box
    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=PALETTE_NEUTRAL, edgecolor=palette_hex(1), alpha=0.9),
            family='monospace')

    plt.tight_layout()
    return fig


def get_skewness_interpretation(skewness):
    """Get human-readable interpretation of skewness value."""
    if skewness > 0.5:
        return "Right-skewed (many small particles, few large ones)"
    if skewness < -0.5:
        return "Left-skewed (many large particles, few small ones)"
    return "Approximately symmetric"


def get_kurtosis_interpretation(kurtosis):
    """Get human-readable interpretation of kurtosis value."""
    if kurtosis > 1:
        return "Heavy tails (more extreme sizes than normal)"
    if kurtosis < -1:
        return "Light tails (fewer extreme sizes than normal)"
    return "Normal-like distribution"


if __name__ == "__main__":
    # Test module
    print("Size Analysis Module")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - calculate_mask_areas(boundary_pixels, pixel_to_real)")
    print("  - calculate_size_metrics(areas, scale)")
    print("  - create_size_histogram(areas, scale, figsize)")
    print("  - create_size_overlay(image, boundary_pixels, mask_areas, scale, figsize)")
    print("  - create_size_statistics_box(metrics, figsize)")

    # Example usage
    print("\nExample usage:")
    print("  areas = calculate_mask_areas(boundary_pixels, pixel_to_real=0.5)")
    print("  metrics = calculate_size_metrics(list(areas.values()), scale='nm')")
    print("  fig_hist = create_size_histogram(list(areas.values()), scale='nm')")
    print("  fig_overlay = create_size_overlay(image, boundary_pixels, areas, scale='nm')")
    print("  fig_stats = create_size_statistics_box(metrics)")
