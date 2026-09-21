"""
Spatial Uniformity Analysis Module for Particle Layout

This module provides comprehensive spatial uniformity analysis including:
- Voronoi diagram generation and area calculation (unified regions per particle)
- HDBSCAN clustering for spatial organization
- Statistical metrics and visualizations

Key metrics:
- Voronoi Area: Mean, Std, Skewness, Kurtosis
- SUI (Spatial Uniformity Index): 1 / (1 + CV), where CV = Std / Mean
- Clustering: HDBSCAN labels and statistics
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import skew, kurtosis
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import alphashape
from modules.color_palette import (
    PALETTE_NEUTRAL,
    make_value_colormap,
    palette_bgr255,
    palette_hex,
    palette_sequence,
)
from modules.scientific_plotting import (
    bootstrap_median_ci,
    create_spatial_distribution_figure,
)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan package not installed. Install with: pip install hdbscan")
from typing import Dict, List, Tuple, Any, Optional


def compute_unified_voronoi_areas(
    boundary_pixels: Dict[int, List[Tuple[int, int]]],
    concave_vertex: Optional[List[Tuple[float, float]]] = None,
    inside_centroids: Optional[List[Tuple[float, float]]] = None
) -> Tuple[Dict[int, float], Voronoi, Dict[int, Polygon]]:
    """
    Compute unified Voronoi areas using Distribution.py's proven algorithm.

    This algorithm (from Distribution.py lines 730-746, 910-932):
    1. Creates Voronoi diagram from ALL boundary pixels
    2. For each particle key, collects all Voronoi cells belonging to that particle
    3. Intersects each cell with the convex analysis hull
    4. Unifies all intersected cells for each particle key

    Args:
        boundary_pixels: Dictionary mapping particle key (tuple) to list of (x, y) boundary points
        concave_vertex: Legacy parameter name for convex analysis-hull vertices
        inside_centroids: Centroids retained by the inward-buffer hull rule

    Returns:
        Tuple of:
        - region_areas: Dict mapping particle key to unified Voronoi area
        - vor: Voronoi object for visualization
        - unified_regions: Dict mapping particle key to unified Polygon
    """
    # Create Voronoi diagram from ALL boundary pixels (Distribution.py line 731-732)
    all_points = [point for points in boundary_pixels.values() for point in points]
    vor = Voronoi(all_points)

    # Create the analysis-domain polygon (legacy parameter name retained).
    if concave_vertex is not None:
        concave_polygon = Polygon(concave_vertex)
    else:
        concave_polygon = None

    # Determine which keys to include (robust mapping due to float rounding)
    # Map inside_centroids to nearest boundary_pixels keys within tolerance
    if inside_centroids is not None:
        bp_keys = list(boundary_pixels.keys())
        if len(bp_keys) > 0 and len(inside_centroids) > 0:
            try:
                from scipy.spatial import cKDTree
                key_array = np.array(bp_keys, dtype=float)
                tree = cKDTree(key_array)
                inside_array = np.array(inside_centroids, dtype=float)
                # tolerance in pixels for centroid-key association
                tol = 2.0
                included_keys = set()
                for p in inside_array:
                    dist, idx = tree.query(p, k=1)
                    if np.isfinite(dist) and dist <= tol:
                        included_keys.add(tuple(key_array[idx]))
                excluded_keys = set(bp_keys) - included_keys
            except Exception:
                # Fallback to exact match if KDTree unavailable
                excluded_keys = set(bp_keys) - set(tuple(c) for c in inside_centroids)
        else:
            excluded_keys = set(bp_keys)
    else:
        excluded_keys = set()

    # For each key, collect and unify Voronoi cells (Distribution.py lines 911-931)
    key_polygons_map = {}
    region_areas = {}

    total_keys = len([k for k in boundary_pixels.keys() if k not in excluded_keys])
    processed = 0
    
    # Pre-build point-to-key mapping for O(1) lookup (major optimization!)
    point_to_key_map = {}
    for key, points in boundary_pixels.items():
        for p in points:
            point_to_key_map[tuple(p)] = key
    
    for key, points in boundary_pixels.items():
        if key in excluded_keys:
            continue

        processed += 1
        if processed % 10 == 0 or processed == total_keys:
            print(f"    Voronoi progress: {processed}/{total_keys} particles...")

        selected_polygons = []
        
        # Convert points to set of tuples for O(1) lookup
        points_set = set(tuple(p) for p in points)

        # Find all Voronoi regions for this particle's boundary pixels (Distribution.py line 914)
        for region_index, point in enumerate(vor.points):
            # Check if this point belongs to current particle (O(1) lookup instead of O(n))
            point_tuple = tuple(point)
            if point_tuple in points_set:
                region = vor.regions[vor.point_region[region_index]]

                # Skip infinite regions (Distribution.py line 917)
                if -1 in region:
                    continue

                # Create polygon from Voronoi region (Distribution.py line 920)
                polygon = Polygon([vor.vertices[i] for i in region])

                if polygon.is_valid:
                    # Clip each finite Voronoi cell to the analysis domain.
                    if concave_polygon is not None:
                        intersection = polygon.intersection(concave_polygon)
                        if not intersection.is_empty and isinstance(intersection, (Polygon, MultiPolygon)):
                            selected_polygons.append(intersection)
                    else:
                        selected_polygons.append(polygon)

        # Unify all polygons for this particle key (Distribution.py lines 928-931)
        if selected_polygons:
            unified_polygon = unary_union(selected_polygons)
            key_polygons_map[key] = unified_polygon

            # Calculate area
            if isinstance(unified_polygon, MultiPolygon):
                area = sum(poly.area for poly in unified_polygon.geoms)
            else:
                area = unified_polygon.area
            region_areas[key] = area

    return region_areas, vor, key_polygons_map


def calculate_distribution_metrics(
    voronoi_areas: List[float],
    scale: str = "nm"
) -> Dict[str, float]:
    """
    Calculate statistical metrics for spatial uniformity from Voronoi areas.

    Args:
        voronoi_areas: List of unified Voronoi area values
        scale: Unit scale ("nm" or "um")

    Returns:
        Dictionary containing statistical metrics
    """
    areas_array = np.array(voronoi_areas, dtype=float)
    if len(areas_array) < 2:
        raise ValueError("PF-SUI requires at least two finite interior particle-associated regions")
    if not np.isfinite(areas_array).all() or (areas_array <= 0).any():
        raise ValueError("PF-SUI areas must be finite and positive")

    mean_area = float(np.mean(areas_array))
    std_area = float(np.std(areas_array, ddof=1)) if len(areas_array) > 1 else 0.0
    cv_fraction = std_area / mean_area if mean_area > 0 else 0.0
    sui = 1.0 / (1.0 + cv_fraction) if mean_area > 0 else 0.0
    if len(areas_array) < 3 or np.allclose(areas_array, areas_array[0]):
        skewness_value = 0.0
        kurtosis_value = 0.0
    else:
        skewness_value = float(skew(areas_array))
        kurtosis_value = float(kurtosis(areas_array))

    metrics = {
        'mean': mean_area,
        'std': std_area,
        'median': float(np.median(areas_array)),
        'skewness': skewness_value,
        'kurtosis': kurtosis_value,
        'min': float(np.min(areas_array)),
        'max': float(np.max(areas_array)),
        'q1': float(np.percentile(areas_array, 25)),
        'q3': float(np.percentile(areas_array, 75)),
        'iqr': float(np.percentile(areas_array, 75) - np.percentile(areas_array, 25)),
        'cv': float(cv_fraction * 100.0),
        'cv_fraction': float(cv_fraction),
        'sui': float(sui),
        'spatial_uniformity_index': float(sui),
        'count': len(areas_array),
        'scale': scale
    }

    ci_low, ci_high = bootstrap_median_ci(areas_array)
    metrics['median_ci_low'] = ci_low
    metrics['median_ci_high'] = ci_high

    return metrics


def create_distribution_violin_plot(
    voronoi_areas: List[float],
    scale: str = "nm",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Create the shared half-violin raincloud with raw values and median CI."""
    return create_spatial_distribution_figure(voronoi_areas, scale=scale, figsize=figsize)


def create_voronoi_overlay(
    image: np.ndarray,
    boundary_pixels: Dict[int, List[Tuple[int, int]]],
    voronoi_areas: Dict[int, float],
    scale: Optional[str] = "nm",
    figsize: Tuple[int, int] = (12, 12),
    concave_vertex: Optional[List[Tuple[float, float]]] = None,
    inside_centroids: Optional[List[Tuple[float, float]]] = None,
    show_area_labels: bool = True,
    show_title: bool = True,
    colorbar_label: Optional[str] = None,
    relative_colorbar: bool = False,
    draw_region_edges: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create spatial-uniformity Voronoi overlay with area-based heatmap.

    Args:
        image: Original input image
        boundary_pixels: Boundary pixels for each particle
        voronoi_areas: Unified Voronoi area for each particle
        scale: Unit scale. Set to None when physical calibration is unavailable.
        figsize: Figure size
        concave_vertex: Concave hull vertices for clipping Voronoi cells
        inside_centroids: Centroids inside concave hull
        show_area_labels: Draw numeric area values at particle centroids.
        show_title: Draw the plot title.
        colorbar_label: Optional colorbar label. Uses a relative-area label when
            scale is None.
        relative_colorbar: Replace numeric colorbar ticks with Smaller/Larger
            when calibrated physical area is not available.
        draw_region_edges: Draw outlined unified Voronoi cell boundaries over
            the area colors.

    Returns:
        Tuple of (figure, overlay_image)
    """
    # Recompute Voronoi for visualization (with concave hull clipping)
    _, vor, unified_regions = compute_unified_voronoi_areas(
        boundary_pixels, concave_vertex, inside_centroids
    )

    # Create overlay image
    overlay = image.copy()

    # Get area range for colormap (red=large, blue=small)
    areas = list(voronoi_areas.values())
    min_area = min(areas)
    max_area = max(areas)
    cmap = make_value_colormap()

    # Color each particle's Voronoi region by area
    for key, polygon in unified_regions.items():
        if key in voronoi_areas:
            area = voronoi_areas[key]
            normalized = (area - min_area) / (max_area - min_area) if max_area > min_area else 0.5
            color = cmap(normalized)[:3]  # RGB: red (large) to blue (small)
            color_bgr = tuple(int(c * 255) for c in reversed(color))

            # Draw filled polygon
            if isinstance(polygon, MultiPolygon):
                for poly in polygon.geoms:
                    coords = np.array(poly.exterior.coords, dtype=np.int32)
                    cv2.fillPoly(overlay, [coords], color_bgr)
            else:
                coords = np.array(polygon.exterior.coords, dtype=np.int32)
                cv2.fillPoly(overlay, [coords], color_bgr)

    # Blend with original image
    overlay = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    if draw_region_edges:
        # A dark edge plus a thin light center keeps adjacent similar colors distinct.
        for polygon in unified_regions.values():
            polygons = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]
            for poly in polygons:
                coords = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.polylines(
                    overlay, [coords], True, (35, 35, 35), 2, lineType=cv2.LINE_AA
                )
                cv2.polylines(
                    overlay, [coords], True, (245, 245, 245), 1, lineType=cv2.LINE_AA
                )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    if show_area_labels:
        # Add text labels showing Voronoi area values when physical units are meaningful.
        for key in boundary_pixels.keys():
            if key in voronoi_areas:
                points = boundary_pixels[key]
                centroid_x = np.mean([p[0] for p in points])
                centroid_y = np.mean([p[1] for p in points])
                area = voronoi_areas[key]
                ax.text(centroid_x, centroid_y, f'{area:.1f}',
                       color=palette_hex(0), fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE_NEUTRAL, alpha=0.9))

    if show_title:
        ax.set_title('PF-SUI: particle-boundary-based Voronoi', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap,
                          norm=plt.Normalize(vmin=min_area, vmax=max_area))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label is None:
        colorbar_label = (
            f'Voronoi Cell Area ({scale}^2)'
            if scale is not None
            else 'Relative Voronoi Cell Area'
        )
    cbar.set_label(colorbar_label, fontsize=12, fontweight='bold')
    if relative_colorbar:
        cbar.set_ticks([min_area, max_area])
        cbar.set_ticklabels(['Smaller', 'Larger'])

    return fig, overlay


def find_optimal_alpha_hull(
    centroids: List[Tuple[float, float]]
) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[Polygon]],
           List[Tuple[float, float]], float]:
    """
    Find optimal alpha value for concave hull generation.

    Args:
        centroids: List of particle centroids

    Returns:
        Tuple of (inside_centroids, polygons, concave_vertices, alpha_value)
    """
    points = np.array(centroids)
    best_alpha = None
    best_shape = None

    # Search for optimal alpha from 0.0 to 0.03
    for alpha in np.arange(0.0, 0.03, 0.001):
        try:
            alpha_shape = alphashape.alphashape(points, alpha)
            if isinstance(alpha_shape, Polygon):
                best_alpha = alpha
                best_shape = alpha_shape
                break
            elif isinstance(alpha_shape, MultiPolygon) and len(alpha_shape.geoms) == 1:
                best_alpha = alpha
                best_shape = alpha_shape.geoms[0]
                break
        except (TypeError, ValueError):
            continue

    if best_alpha is not None and best_shape is not None:
        polygons = [best_shape]

        # Filter centroids inside concave hull
        inside_centroids = [
            tuple(point) for point in centroids
            if any(poly.contains(Point(point)) for poly in polygons)
        ]
        inside_centroids = list(set(inside_centroids))

        # Extract concave hull vertices
        concave_vertices = []
        for poly in polygons:
            concave_vertices.extend(poly.exterior.coords)

        return inside_centroids, polygons, concave_vertices, best_alpha
    else:
        return None, None, [], 0.0

def perform_hdbscan_clustering(
    centroids: List[Tuple[float, float]],
    min_cluster_size: int = 5,
    min_samples: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform HDBSCAN clustering on particle centroids.

    Args:
        centroids: List of particle centroids
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples parameter
        random_state: Random seed for reproducibility (Note: HDBSCAN is deterministic,
                     so this parameter is accepted for API consistency but not used)

    Returns:
        Tuple of (labels, statistics_dict)

    Note:
        HDBSCAN is a deterministic algorithm based on minimum spanning tree construction,
        so results are reproducible by default without requiring random_state.
        This parameter is included for API consistency with sklearn's clustering interface.
    """
    if not HDBSCAN_AVAILABLE:
        print("Warning: HDBSCAN clustering skipped - hdbscan not installed")
        labels = np.zeros(len(centroids), dtype=int)  # All in one cluster
        stats = {
            'n_clusters': 1,
            'n_noise': 0,
            'cluster_sizes': [len(centroids)],
            'avg_cluster_size': float(len(centroids)),
            'std_cluster_size': 0.0,
            'total_points': len(centroids),
            'clustered_points': len(centroids),
            'clustered_percentage': 100.0
        }
        return labels, stats

    centroids_array = np.array(centroids)

    # Perform HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(centroids_array)

    # Calculate statistics
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)

    cluster_sizes = []
    for label in unique_labels:
        if label != -1:
            cluster_sizes.append(np.sum(labels == label))

    stats = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cluster_sizes': cluster_sizes,
        'avg_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        'std_cluster_size': float(np.std(cluster_sizes)) if cluster_sizes else 0.0,
        'total_points': len(labels),
        'clustered_points': int(len(labels) - n_noise),
        'clustered_percentage': float((len(labels) - n_noise) / len(labels) * 100) if len(labels) > 0 else 0.0
    }

    return labels, stats




def create_clustering_map(
    image: np.ndarray,
    centroids: List[Tuple[float, float]],
    labels: np.ndarray,
    figsize: Tuple[int, int] = (12, 12)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create HDBSCAN clustering visualization overlay.

    Args:
        image: Original input image
        centroids: List of particle centroids
        labels: Cluster labels from HDBSCAN
        figsize: Figure size

    Returns:
        Tuple of (figure, overlay_image)
    """
    overlay = image.copy()

    # Generate unique colors for each cluster
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = palette_sequence(max(n_clusters, 1))

    # Create color map
    color_map = {}
    color_idx = 0
    for label in unique_labels:
        if label == -1:
            color_map[label] = palette_bgr255(4, include_neutral=True)
        else:
            rgb = colors[color_idx % len(colors)]
            color_map[label] = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            color_idx += 1

    # Draw centroids with cluster colors
    for centroid, label in zip(centroids, labels):
        color = color_map[label]
        cv2.circle(overlay, (int(centroid[0]), int(centroid[1])),
                  8, color, -1)
        cv2.circle(overlay, (int(centroid[0]), int(centroid[1])),
                  8, palette_bgr255(0), 2)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.set_title(f'HDBSCAN Clustering Map ({n_clusters} clusters)',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    return fig, overlay
